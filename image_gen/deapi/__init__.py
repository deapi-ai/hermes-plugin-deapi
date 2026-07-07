"""deAPI image generation provider for Hermes Agent (prototype v0.1).

Uses the native deAPI v2 REST API (async jobs + polling) because the
OpenAI-compatible gateway rejects FLUX.2 Klein on /v1/images/generations
(misclassified as edit-only there). Native v2 serves Klein for txt2img.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

API_BASE = os.environ.get("DEAPI_BASE_URL", "https://api.deapi.ai").rstrip("/")
POLL_INTERVAL = 2.0
POLL_TIMEOUT = 180.0
HTTP_TIMEOUT = 60
USER_AGENT = "hermes-plugin-deapi/0.1 (+https://github.com/deapi-ai/hermes-plugin-deapi)"

# Aspect-ratio -> (width, height); multiples of 16, within Klein 256-1536.
_RATIO_SIZES = {
    "landscape": (1024, 768),
    "square": (1024, 1024),
    "portrait": (768, 1024),
}

_MODEL_PREFERENCE = [r"klein", r"flux", r""]


def _api_key() -> str:
    key = (os.environ.get("DEAPI_API_KEY") or "").strip()
    # Native REST authenticates with the raw token; the dpn-sk- prefix is
    # only for the OpenAI gateway. Accept either form.
    if key.startswith("dpn-sk-"):
        key = key[len("dpn-sk-"):]
    return key


def _request(method: str, path: str, payload: Optional[dict] = None) -> dict:
    url = API_BASE + path
    body = None
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", "Bearer " + _api_key())
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", USER_AGENT)  # default python UA is CF-banned
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        req.add_header("Content-Type", "application/json")
        req.data = body
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


class DeapiImageGenProvider(ImageGenProvider):
    _models_cache: Optional[List[Dict[str, Any]]] = None

    @property
    def name(self) -> str:
        return "deapi"

    @property
    def display_name(self) -> str:
        return "deAPI"

    def is_available(self) -> bool:
        return bool(_api_key())

    def _fetch_models(self) -> List[Dict[str, Any]]:
        if self._models_cache is None:
            data = _request("GET", "/api/v2/models?per_page=100")
            models = data.get("data") or []
            self._models_cache = [
                m for m in models
                if "txt2img" in (m.get("inference_types") or [])
            ]
        return self._models_cache

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            models = self._fetch_models()
        except Exception:
            return []
        return [
            {
                "id": m["slug"],
                "display": m.get("name") or m["slug"],
                "speed": "fast",
                "strengths": ", ".join(m.get("inference_types") or []),
                "price": "pay-per-use",
            }
            for m in models
        ]

    def default_model(self) -> Optional[str]:
        try:
            models = self._fetch_models()
        except Exception:
            return None
        for pattern in _MODEL_PREFERENCE:
            for m in models:
                if re.search(pattern, m["slug"], re.IGNORECASE):
                    return m["slug"]
        return None

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "deAPI",
            "badge": "deapi",
            "tag": "open-source models, pay-per-use",
            "env_vars": [
                {
                    "key": "DEAPI_API_KEY",
                    "prompt": "deAPI API key (free $5 credit at app.deapi.ai)",
                    "url": "https://app.deapi.ai",
                }
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = "landscape",
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return error_response(
                error="DEAPI_API_KEY is not set (get one at https://app.deapi.ai)",
                error_type="missing_api_key",
                provider=self.name,
            )
        ratio = resolve_aspect_ratio(aspect_ratio)
        width, height = _RATIO_SIZES.get(ratio, _RATIO_SIZES["landscape"])

        model = kwargs.get("model") or os.environ.get("DEAPI_IMAGE_MODEL")
        model_info: Dict[str, Any] = {}
        try:
            models = self._fetch_models()
            if not model:
                model = self.default_model()
            model_info = next((m for m in models if m["slug"] == model), {})
        except Exception:
            pass
        if not model:
            return error_response(
                error="Could not resolve a deAPI image model (GET /api/v2/models failed)",
                error_type="provider_error",
                provider=self.name,
            )

        info = model_info.get("info") or {}
        defaults = info.get("defaults") or {}
        features = info.get("features") or {}
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "steps": int(defaults.get("steps", 4)),
            "seed": random.randint(0, 2**31 - 1),
        }
        if features.get("supports_guidance", False):
            payload["guidance"] = float(defaults.get("guidance", 3.5))

        try:
            data = _request("POST", "/api/v2/images/generations", payload)
            request_id = (data.get("data") or {}).get("request_id")
            if not request_id:
                return error_response(
                    error="deAPI returned no request_id: %s" % json.dumps(data)[:200],
                    error_type="provider_error",
                    provider=self.name,
                )
            deadline = time.time() + POLL_TIMEOUT
            while time.time() < deadline:
                job = (_request("GET", "/api/v2/jobs/%s" % request_id).get("data") or {})
                status = str(job.get("status", "")).lower()
                if status == "done":
                    result_url = job.get("result_url")
                    if not result_url:
                        return error_response(
                            error="job done but no result_url",
                            error_type="provider_error",
                            provider=self.name,
                        )
                    # result URLs expire (~24h) and are not platform-friendly:
                    # persist locally like the xai plugin does.
                    path = save_url_image(result_url, prefix="deapi")
                    return success_response(
                        image=str(path),
                        model=model,
                        prompt=prompt,
                        aspect_ratio=ratio,
                        provider=self.name,
                    )
                if status in ("error", "failed"):
                    return error_response(
                        error="deAPI job failed: %s" % json.dumps(job)[:300],
                        error_type="provider_error",
                        provider=self.name,
                    )
                time.sleep(POLL_INTERVAL)
            return error_response(
                error="deAPI job timed out after %ds" % int(POLL_TIMEOUT),
                error_type="provider_timeout",
                provider=self.name,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            return error_response(
                error="deAPI HTTP %d: %s" % (exc.code, detail),
                error_type="provider_error",
                provider=self.name,
            )
        except Exception as exc:  # network etc. — never raise per contract
            return error_response(
                error="deAPI request failed: %s" % exc,
                error_type="provider_error",
                provider=self.name,
            )


def register(ctx) -> None:
    ctx.register_image_gen_provider(DeapiImageGenProvider())
