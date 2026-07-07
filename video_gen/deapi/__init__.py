"""deAPI video generation provider for Hermes Agent.

Text-to-video and image-to-video via the native deAPI v2 REST API
(async jobs + polling). Models (e.g. LTX family) are discovered live from
GET /api/v2/models - nothing hardcoded.

https://github.com/deapi-ai/hermes-plugin-deapi
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    save_bytes_video,
    success_response,
)

API_BASE = os.environ.get("DEAPI_BASE_URL", "https://api.deapi.ai").rstrip("/")
POLL_INTERVAL = 5.0
POLL_TIMEOUT = 600.0
HTTP_TIMEOUT = 120
USER_AGENT = "hermes-plugin-deapi/0.1 (+https://github.com/deapi-ai/hermes-plugin-deapi)"

_MODEL_PREFERENCE = [r"ltx", r""]


def _api_key() -> str:
    key = (os.environ.get("DEAPI_API_KEY") or "").strip()
    # Native REST authenticates with the raw token; the dpn-sk- prefix is
    # only for the OpenAI gateway. Accept either form.
    if key.startswith("dpn-sk-"):
        key = key[len("dpn-sk-"):]
    return key


def _request(method: str, path: str, payload: Optional[dict] = None,
             fields: Optional[dict] = None, files: Optional[list] = None) -> dict:
    url = API_BASE + path
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", "Bearer " + _api_key())
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", USER_AGENT)  # default python UA is CF-banned
    if payload is not None:
        req.data = json.dumps(payload).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    elif fields is not None:
        boundary = "----deapi-" + uuid.uuid4().hex
        buf = io.BytesIO()

        def part(header: str, body: bytes) -> None:
            buf.write(("--%s\r\n%s\r\n\r\n" % (boundary, header)).encode("utf-8"))
            buf.write(body)
            buf.write(b"\r\n")

        for name, value in fields.items():
            if value is None:
                continue
            part('Content-Disposition: form-data; name="%s"' % name,
                 str(value).encode("utf-8"))
        for name, file_path in (files or []):
            ctype = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            with open(file_path, "rb") as handle:
                data = handle.read()
            part('Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
                 'Content-Type: %s' % (name, os.path.basename(file_path), ctype),
                 data)
        buf.write(("--%s--\r\n" % boundary).encode("utf-8"))
        req.data = buf.getvalue()
        req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return resp.read()


def _materialize_image(image_url: str) -> str:
    """Return a local file path for image_url (download if it is http(s))."""
    if image_url.startswith(("http://", "https://")):
        raw = _download(image_url)
        path = "/tmp/deapi-vidgen-%s.png" % uuid.uuid4().hex[:8]
        with open(path, "wb") as handle:
            handle.write(raw)
        return path
    return image_url


def _clamp(value: int, limits: Dict[str, Any], low: str, high: str) -> int:
    if limits.get(low) is not None:
        value = max(int(limits[low]), value)
    if limits.get(high) is not None:
        value = min(int(limits[high]), value)
    return value


# Aspect ratio -> (width, height) base sizes; clamped to model limits and
# rounded to multiples of 16 at request time.
_RATIO_SIZES = {
    "16:9": (768, 432),
    "9:16": (432, 768),
    "1:1": (512, 512),
    "4:3": (640, 480),
    "3:4": (480, 640),
}


class DeapiVideoGenProvider(VideoGenProvider):
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
                if {"txt2video", "img2video"} & set(m.get("inference_types") or [])
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
                "speed": "medium",
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
            "tag": "open-source video models, pay-per-use",
            "env_vars": [
                {
                    "key": "DEAPI_API_KEY",
                    "prompt": "deAPI API key (free $5 credit at app.deapi.ai)",
                    "url": "https://app.deapi.ai",
                }
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": list(_RATIO_SIZES.keys()),
            "resolutions": ["480p"],
            "max_duration": 4,
            "min_duration": 1,
            "supports_audio": False,
            "supports_negative_prompt": True,
            "max_reference_images": 0,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return error_response(
                error="DEAPI_API_KEY is not set (get one at https://app.deapi.ai)",
                error_type="missing_api_key",
                provider=self.name,
            )

        model = model or os.environ.get("DEAPI_VIDEO_MODEL")
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
                error="Could not resolve a deAPI video model (GET /api/v2/models failed)",
                error_type="provider_error",
                provider=self.name,
            )

        info = model_info.get("info") or {}
        defaults = info.get("defaults") or {}
        limits = info.get("limits") or {}

        width, height = _RATIO_SIZES.get(aspect_ratio, _RATIO_SIZES[DEFAULT_ASPECT_RATIO])
        width = _clamp(width - width % 16, limits, "min_width", "max_width")
        height = _clamp(height - height % 16, limits, "min_height", "max_height")

        fps = _clamp(int(defaults.get("fps", 24)), limits, "min_fps", "max_fps")
        if duration:
            frames = _clamp(int(duration) * fps, limits, "min_frames", "max_frames")
        else:
            frames = _clamp(int(defaults.get("frames", 97)), limits, "min_frames", "max_frames")

        common: Dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "steps": _clamp(int(defaults.get("steps", 8)), limits, "min_steps", "max_steps"),
            "guidance": float(defaults.get("guidance", 3.0)),
            "seed": seed if seed is not None else random.randint(0, 2**31 - 1),
            "frames": frames,
            "fps": fps,  # required by the live API even where the spec says optional
        }
        if negative_prompt:
            common["negative_prompt"] = negative_prompt

        try:
            if image_url:
                local = _materialize_image(image_url)
                data = _request("POST", "/api/v2/videos/animations",
                                fields=common, files=[("first_frame_image", local)])
                modality = "image"
            else:
                data = _request("POST", "/api/v2/videos/generations", payload=common)
                modality = "text"

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
                    # result URLs expire (~24h): persist to the local cache.
                    path = save_bytes_video(_download(result_url), prefix="deapi")
                    return success_response(
                        video=str(path),
                        model=model,
                        prompt=prompt,
                        modality=modality,
                        aspect_ratio=aspect_ratio,
                        duration=int(round(frames / float(fps))) if fps else 0,
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
        except Exception as exc:  # never raise per contract
            return error_response(
                error="deAPI request failed: %s" % exc,
                error_type="provider_error",
                provider=self.name,
            )


def register(ctx) -> None:
    ctx.register_video_gen_provider(DeapiVideoGenProvider())
