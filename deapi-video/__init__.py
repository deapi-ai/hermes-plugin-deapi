"""deAPI video generation provider for Hermes Agent.

Text-to-video and image-to-video via the native deAPI v2 REST API
(async jobs + polling). Models (e.g. LTX family) are discovered live from
GET /api/v2/models - nothing hardcoded.

https://github.com/deapi-ai/hermes-plugin-deapi
"""

from __future__ import annotations

import io
import ipaddress
import json
import mimetypes
import os
import random
import re
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    save_bytes_video,
    success_response,
)

DEFAULT_API_BASE = "https://api.deapi.ai"
POLL_INTERVAL = 5.0
POLL_TIMEOUT = 600.0
HTTP_TIMEOUT = 120
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # cap for downloaded first-frame images
USER_AGENT = "hermes-plugin-deapi/0.2 (+https://github.com/deapi-ai/hermes-plugin-deapi)"

_MODEL_PREFERENCE = [r"ltx", r""]

# Aspect ratio -> (width, height) base sizes; clamped to model limits and
# rounded to multiples of 16 at request time.
_RATIO_SIZES = {
    "16:9": (768, 432),
    "9:16": (432, 768),
    "1:1": (512, 512),
    "4:3": (640, 480),
    "3:4": (480, 640),
}


def _api_base() -> str:
    """Base URL. Locked to api.deapi.ai unless the operator explicitly opts
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


def _request(method: str, path: str, payload: Optional[dict] = None,
             fields: Optional[dict] = None, files: Optional[list] = None) -> dict:
    url = _api_base() + path
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


# --------------------------------------------------------------------------
# Input-image handling (image-to-video). Both paths are locked down: local
# paths must live under the Hermes cache, remote URLs are SSRF-filtered,
# size-capped and content-type-checked.
# --------------------------------------------------------------------------

def _hermes_cache_root() -> Optional[str]:
    try:
        from hermes_constants import get_hermes_home
        return str((get_hermes_home() / "cache").resolve())
    except Exception:
        return None


def _host_is_public(hostname: str) -> bool:
    """True only if every resolved address for hostname is a global unicast
    address (blocks loopback, private, link-local, multicast, and cloud
    metadata ranges)."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_multicast or addr.is_reserved or addr.is_unspecified):
            return False
        # AWS/GCP/Azure metadata endpoint
        if str(addr) == "169.254.169.254":
            return False
    return True


class _ImageInputError(Exception):
    pass


def _local_image_path(image_url: str) -> str:
    """Return a validated local path if image_url points inside the Hermes
    cache, else raise. Only Hermes-produced artifacts may be uploaded."""
    root = _hermes_cache_root()
    if not root:
        raise _ImageInputError("cannot resolve Hermes cache root for local image")
    resolved = os.path.realpath(image_url)
    if not (resolved == root or resolved.startswith(root + os.sep)):
        raise _ImageInputError(
            "local image path is outside the Hermes cache; only cached "
            "artifacts may be used as a first frame")
    if not os.path.isfile(resolved):
        raise _ImageInputError("local image file not found")
    ctype = mimetypes.guess_type(resolved)[0] or ""
    if not ctype.startswith("image/"):
        raise _ImageInputError("first-frame file is not an image")
    return resolved


def _download_remote_image(image_url: str) -> str:
    """SSRF-safe download of a remote first-frame image into a private temp
    file. Returns the temp path (caller cleans up)."""
    parsed = urllib.parse.urlparse(image_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise _ImageInputError("unsupported image URL scheme")
    if not _host_is_public(parsed.hostname):
        raise _ImageInputError("image URL resolves to a non-public address")
    req = urllib.request.Request(image_url)
    req.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            # urllib follows redirects; re-check the final host.
            final_host = urllib.parse.urlparse(resp.geturl()).hostname
            if not final_host or not _host_is_public(final_host):
                raise _ImageInputError("image URL redirected to a non-public address")
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
            if not ctype.startswith("image/"):
                raise _ImageInputError("image URL did not return an image")
            data = resp.read(MAX_IMAGE_BYTES + 1)
    except urllib.error.URLError as exc:
        raise _ImageInputError("could not fetch image URL: %s" % exc.reason)
    if len(data) > MAX_IMAGE_BYTES:
        raise _ImageInputError("image exceeds %d MB limit" % (MAX_IMAGE_BYTES // (1024 * 1024)))
    ext = mimetypes.guess_extension(ctype) or ".img"
    fd, path = tempfile.mkstemp(prefix="deapi-vidgen-", suffix=ext)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except Exception:
        os.unlink(path)
        raise
    return path


def _clamp(value: int, limits: Dict[str, Any], low: str, high: str) -> int:
    if limits.get(low) is not None:
        value = max(int(limits[low]), value)
    if limits.get(high) is not None:
        value = min(int(limits[high]), value)
    return value


def _clamp_to_multiple(value: int, limits: Dict[str, Any], low: str, high: str,
                       step: int = 16) -> int:
    """Clamp to [low, high] first, then snap down to a multiple of step while
    staying >= low."""
    value = _clamp(value, limits, low, high)
    snapped = value - (value % step)
    lo = limits.get(low)
    if lo is not None and snapped < int(lo):
        snapped = _clamp(int(lo) + (step - int(lo) % step) % step, limits, low, high)
    return max(snapped, step)


def _short(value: Any, limit: int = 160) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _download_result(url: str) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", USER_AGENT)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return resp.read()


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

    def default_model(self, need: str = "txt2video") -> Optional[str]:
        try:
            models = [m for m in self._fetch_models()
                      if need in (m.get("inference_types") or [])]
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

    def _resolve_model(self, requested: Optional[str],
                       need: str) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
        """Return (slug, model_info, error). Enforces that the chosen model
        supports the requested modality (txt2video / img2video)."""
        try:
            models = self._fetch_models()
        except Exception as exc:
            return None, {}, "could not fetch deAPI models: %s" % _short(exc)
        requested = requested or os.environ.get("DEAPI_VIDEO_MODEL")
        if requested:
            info = next((m for m in models if m["slug"] == requested), None)
            if info is None:
                return None, {}, "model %r not found in the live deAPI model list" % requested
            if need not in (info.get("inference_types") or []):
                return None, {}, (
                    "model %r does not support %s (supports: %s)"
                    % (requested, need, ", ".join(info.get("inference_types") or [])))
            return requested, info, None
        slug = self.default_model(need)
        if not slug:
            return None, {}, "no deAPI model supports %s" % need
        info = next((m for m in models if m["slug"] == slug), {})
        return slug, info, None

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

        need = "img2video" if image_url else "txt2video"
        slug, model_info, err = self._resolve_model(model, need)
        if err:
            return error_response(error=err, error_type="provider_error",
                                  provider=self.name)

        info = model_info.get("info") or {}
        defaults = info.get("defaults") or {}
        limits = info.get("limits") or {}
        features = info.get("features") or {}

        width, height = _RATIO_SIZES.get(aspect_ratio, _RATIO_SIZES[DEFAULT_ASPECT_RATIO])
        width = _clamp_to_multiple(width, limits, "min_width", "max_width")
        height = _clamp_to_multiple(height, limits, "min_height", "max_height")

        fps = _clamp(int(defaults.get("fps", 24)), limits, "min_fps", "max_fps")
        if duration:
            frames = _clamp(int(duration) * fps, limits, "min_frames", "max_frames")
        else:
            frames = _clamp(int(defaults.get("frames", 97)), limits, "min_frames", "max_frames")

        common: Dict[str, Any] = {
            "prompt": prompt,
            "model": slug,
            "width": width,
            "height": height,
            "steps": _clamp(int(defaults.get("steps", 8)), limits, "min_steps", "max_steps"),
            "seed": seed if seed is not None else random.randint(0, 2**31 - 1),
            "frames": frames,
            "fps": fps,  # required by the live API even where the spec says optional
        }
        if features.get("supports_guidance", True):
            common["guidance"] = float(defaults.get("guidance", 3.0))
        if negative_prompt and features.get("supports_negative_prompt", True):
            common["negative_prompt"] = negative_prompt

        tmp_image: Optional[str] = None
        try:
            if image_url:
                try:
                    if image_url.startswith(("http://", "https://")):
                        tmp_image = _download_remote_image(image_url)
                        local = tmp_image
                    else:
                        local = _local_image_path(image_url)
                except _ImageInputError as exc:
                    return error_response(
                        error="invalid first-frame image: %s" % exc,
                        error_type="invalid_input",
                        provider=self.name,
                    )
                data = _request("POST", "/api/v2/videos/animations",
                                fields=common, files=[("first_frame_image", local)])
                modality = "image"
            else:
                data = _request("POST", "/api/v2/videos/generations", payload=common)
                modality = "text"

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
                    path = save_bytes_video(_download_result(result_url), prefix="deapi")
                    return success_response(
                        video=str(path),
                        model=slug,
                        prompt=prompt,
                        modality=modality,
                        aspect_ratio=aspect_ratio,
                        duration=int(round(frames / float(fps))) if fps else 0,
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
        finally:
            if tmp_image and os.path.exists(tmp_image):
                try:
                    os.unlink(tmp_image)
                except OSError:
                    pass


def register(ctx) -> None:
    ctx.register_video_gen_provider(DeapiVideoGenProvider())
