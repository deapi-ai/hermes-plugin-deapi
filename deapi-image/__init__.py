"""deAPI image generation provider for Hermes Agent.

Uses the native deAPI v2 REST API (async jobs + polling) because the
OpenAI-compatible gateway rejects FLUX.2 Klein on /v1/images/generations
(misclassified as edit-only there). Native v2 serves Klein for txt2img.
Models are discovered live from GET /api/v2/models - nothing hardcoded.

https://github.com/deapi-ai/hermes-plugin-deapi
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

DEFAULT_API_BASE = "https://api.deapi.ai"
POLL_INTERVAL = 2.0
POLL_TIMEOUT = 180.0
HTTP_TIMEOUT = 60
USER_AGENT = "hermes-plugin-deapi/0.2 (+https://github.com/deapi-ai/hermes-plugin-deapi)"

# Aspect-ratio -> (width, height) base sizes; clamped to model limits.
_RATIO_SIZES = {
    "landscape": (1024, 768),
    "square": (1024, 1024),
    "portrait": (768, 1024),
}

_MODEL_PREFERENCE = [r"klein", r"flux", r""]


def _api_base() -> str:
    """Base URL, locked to api.deapi.ai unless the operator explicitly opts
    into a custom host (prevents an injected env var from redirecting the
    Authorization header to an attacker)."""
    base = (os.environ.get("DEAPI_BASE_URL") or DEFAULT_API_BASE).rstrip("/")
    if base == DEFAULT_API_BASE:
        return base
    if os.environ.get("DEAPI_ALLOW_CUSTOM_BASE_URL") == "1":
        parsed = urllib.parse.urlparse(base)
        if parsed.scheme == "https" and parsed.hostname:
            return base
    return DEFAULT_API_BASE


def _api_key() -> str:
    key = (os.environ.get("DEAPI_API_KEY") or "").strip()
    # Native REST authenticates with the raw token; the dpn-sk- prefix is
    # only for the OpenAI gateway. Accept either form.
    if key.startswith("dpn-sk-"):
        key = key[len("dpn-sk-"):]
    return key


def _request(method: str, path: str, payload: Optional[dict] = None) -> dict:
    req = urllib.request.Request(_api_base() + path, method=method)
    req.add_header("Authorization", "Bearer " + _api_key())
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", USER_AGENT)  # default python UA is CF-banned
    if payload is not None:
        req.data = json.dumps(payload).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _clamp(value: int, limits: Dict[str, Any], low: str, high: str) -> int:
    if limits.get(low) is not None:
        value = max(int(limits[low]), value)
    if limits.get(high) is not None:
        value = min(int(limits[high]), value)
    return value


def _short(value: Any, limit: int = 160) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


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
            self._models_cache = [
                m for m in (data.get("data") or [])
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
        return models[0]["slug"] if models else None

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

    def _resolve_model(
        self, requested: Optional[str]
    ) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
        try:
            models = self._fetch_models()
        except Exception as exc:
            return None, {}, "could not fetch deAPI models: %s" % _short(exc)
        requested = requested or os.environ.get("DEAPI_IMAGE_MODEL")
        if requested:
            info = next((m for m in models if m["slug"] == requested), None)
            if info is None:
                return None, {}, "model %r not found in the live deAPI model list" % requested
            return requested, info, None
        slug = self.default_model()
        if not slug:
            return None, {}, "no deAPI model supports text-to-image"
        info = next((m for m in models if m["slug"] == slug), {})
        return slug, info, None

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

        slug, model_info, err = self._resolve_model(kwargs.get("model"))
        if err:
            return error_response(error=err, error_type="provider_error",
                                  provider=self.name)

        info = model_info.get("info") or {}
        defaults = info.get("defaults") or {}
        limits = info.get("limits") or {}
        features = info.get("features") or {}

        ratio = resolve_aspect_ratio(aspect_ratio)
        width, height = _RATIO_SIZES.get(ratio, _RATIO_SIZES["landscape"])
        width = _clamp(width, limits, "min_width", "max_width")
        height = _clamp(height, limits, "min_height", "max_height")

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": slug,
            "width": width,
            "height": height,
            "steps": _clamp(int(defaults.get("steps", 4)), limits, "min_steps", "max_steps"),
            "seed": random.randint(0, 2**31 - 1),
        }
        if features.get("supports_guidance", False):
            payload["guidance"] = float(defaults.get("guidance", 3.5))

        try:
            data = _request("POST", "/api/v2/images/generations", payload)
            request_id = (data.get("data") or {}).get("request_id")
            if not request_id:
                return error_response(
                    error="deAPI accepted the request but returned no job id",
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
                            error="deAPI job finished without a result",
                            error_type="provider_error",
                            provider=self.name,
                        )
                    # result URLs expire (~24h): persist locally (save_url_image
                    # enforces its own size cap).
                    path = save_url_image(result_url, prefix="deapi")
                    return success_response(
                        image=str(path),
                        model=slug,
                        prompt=prompt,
                        aspect_ratio=ratio,
                        provider=self.name,
                    )
                if status in ("error", "failed"):
                    return error_response(
                        error="deAPI job failed: %s" % _short(job.get("error") or status),
                        error_type="provider_error",
                        provider=self.name,
                    )
                if status and status not in ("pending", "processing", "queued", "running"):
                    return error_response(
                        error="deAPI job returned unexpected status %r" % status,
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
            return error_response(
                error="deAPI returned HTTP %d" % exc.code,
                error_type="provider_error",
                provider=self.name,
            )
        except Exception as exc:  # never raise per contract
            return error_response(
                error="deAPI request failed: %s" % _short(exc),
                error_type="provider_error",
                provider=self.name,
            )


def register(ctx) -> None:
    ctx.register_image_gen_provider(DeapiImageGenProvider())
