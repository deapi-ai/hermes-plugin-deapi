"""deAPI HTTP layer: model catalog, job submission and job polling on the native API v2.

Every generation on deAPI is an asynchronous job: ``POST`` the task → ``request_id`` → poll
``GET /jobs/{request_id}`` until ``done`` (``result_url``) or ``error``. Nothing here logs or
returns the API key; error messages are built from response bodies, which never echo it.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import logging
import re
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.deapi.ai/api/v2"
KEY_ENV = "DEAPI_API_KEY"
KEY_URL = "https://app.deapi.ai/settings/api-keys"
BILLING_URL = "https://app.deapi.ai/billing"
DASHBOARD_URL = "https://app.deapi.ai/dashboard"
USER_AGENT = "hermes-plugin-deapi/0.3.1"

HTTP_TIMEOUT = (10, 60)  # (connect, read) for submits; deAPI answers a submit in well under a second
POLL_HTTP_TIMEOUT = (10, 30)
CATALOG_TTL_S = 600.0
MAX_SOURCE_BYTES = 10 * 1024 * 1024  # deAPI's per-file upload cap
SUBMIT_429_RETRIES = 2
MAX_RETRY_AFTER_S = 10
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_REQUEST_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")
_AUTH_CODES = {"missing_key", "invalid_key", "key_expired", "key_revoked"}

# Indirection points for tests (fake HTTP, no real sleeping).
_http: Any = requests
_sleep = time.sleep
_clock = time.monotonic


class DeapiError(Exception):
    """A failed deAPI call, carrying the ``error_type`` reported to the agent."""

    def __init__(self, message: str, error_type: str = "api_error", *, status: int = 0,
                 request_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status = status
        self.request_id = request_id


def api_key() -> str:
    """``DEAPI_API_KEY`` for the active profile (never a bare ``os.environ`` read); ``""`` when unset
    or when no profile scope is bound (multiplexing fails closed)."""
    from agent.secret_scope import get_secret

    try:
        return (get_secret(KEY_ENV, "") or "").strip()
    except Exception as exc:
        logger.debug("deAPI key lookup failed: %s", type(exc).__name__)
        return ""


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Accept": "application/json", "User-Agent": USER_AGENT}


def _json(resp: Any) -> Any:
    try:
        return resp.json()
    except Exception:
        return None


def _header(resp: Any, name: str) -> str:
    headers = getattr(resp, "headers", None) or {}
    try:
        return str(headers.get(name) or "")
    except Exception:
        return ""


def error_from_response(resp: Any, request_id: Optional[str] = None) -> DeapiError:
    """Map an HTTP error from deAPI to a :class:`DeapiError` with an actionable message."""
    status = int(getattr(resp, "status_code", 0) or 0)
    body = _json(resp)
    message, code, details = "", "", ""
    if isinstance(body, dict):
        message = str(body.get("message") or "")
        code = str(body.get("code") or "")
        errors = body.get("errors")
        if isinstance(errors, dict):
            details = "; ".join(
                f"{field}: {' '.join(map(str, value)) if isinstance(value, list) else value}"
                for field, value in errors.items())
    text = message or (getattr(resp, "text", "") or "")[:300] or f"HTTP {status}"
    if details and details not in text:
        text = f"{text} ({details})"

    def err(msg: str, error_type: str) -> DeapiError:
        return DeapiError(msg, error_type, status=status, request_id=request_id)

    if status == 401 or code in _AUTH_CODES:
        return err(f"deAPI rejected the API key ({code or status}). Check {KEY_ENV}; keys are "
                   f"managed at {KEY_URL}.", "auth_error")
    if status == 402 or code == "key_limit_exceeded":
        return err(f"This deAPI key has reached its own spend limit: {text}", "key_limit_exceeded")
    if code == "insufficient_balance":
        return err(f"The deAPI account balance is too low for this request. Top up at {BILLING_URL}.",
                   "insufficient_balance")
    if status == 429:
        kind = _header(resp, "X-RateLimit-Type")
        return err(f"deAPI rate limit reached{f' ({kind})' if kind else ''}: {text}", "rate_limited")
    if status == 403:
        return err(f"deAPI refused the request: {text}", "forbidden")
    if status == 422:
        return err(f"deAPI rejected the request: {text}", "invalid_request")
    if status == 404:
        return err(f"deAPI returned 404: {text}", "not_found")
    return err(f"deAPI request failed ({status}): {text}", "api_error")


def _never_connected(exc: BaseException) -> bool:
    """True when no connection was made (DNS failure, refused): nothing reached deAPI."""
    try:
        from urllib3.exceptions import NameResolutionError, NewConnectionError
    except Exception:  # pragma: no cover - urllib3 always ships with requests
        return False
    seen: set[int] = set()
    stack: list[Any] = [exc]
    while stack:
        item = stack.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, (NewConnectionError, NameResolutionError)):
            return True
        stack.extend([getattr(item, "reason", None), item.__cause__, item.__context__])
        stack.extend(arg for arg in getattr(item, "args", ()) if isinstance(arg, BaseException))
    return False


def _get(key: str, path: str, *, params: Optional[dict[str, Any]] = None, retries: int = 2) -> Any:
    """GET with retries on transient failures (reads are free and idempotent)."""
    url = f"{API_BASE}{path}"
    for attempt in range(retries + 1):
        try:
            resp = _http.request("GET", url, headers=_headers(key), params=params,
                                 timeout=POLL_HTTP_TIMEOUT, allow_redirects=False)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt < retries:
                _sleep(2 ** attempt)
                continue
            raise DeapiError(f"Could not reach deAPI ({type(exc).__name__}).", "connection_error") from None
        if resp.status_code in RETRYABLE_STATUSES and attempt < retries:
            _sleep(_retry_after(resp, 2 ** attempt))
            continue
        if resp.status_code >= 400:
            raise error_from_response(resp)
        body = _json(resp)
        if body is None:
            raise DeapiError(f"deAPI returned invalid JSON for {path}.", "invalid_response")
        return body
    raise DeapiError(f"deAPI GET {path} failed.", "api_error")  # pragma: no cover


def _retry_after(resp: Any, fallback: float) -> float:
    try:
        value = float(_header(resp, "Retry-After"))
    except ValueError:
        return fallback
    return max(1.0, min(value, float(MAX_RETRY_AFTER_S)))


# --------------------------------------------------------------------------- model catalog

_catalog_lock = threading.Lock()
_catalog_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _fingerprint(key: str) -> str:
    # Keyed per credential: one process can serve several profiles with different plans.
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def clear_cache() -> None:
    with _catalog_lock:
        _catalog_cache.clear()


def cached_models(key: str) -> Optional[list[dict[str, Any]]]:
    """Last fetched catalog for this key (any age), or None. Never touches the network."""
    if not key:
        return None
    with _catalog_lock:
        hit = _catalog_cache.get(_fingerprint(key))
    return hit[1] if hit else None


def fetch_models(key: str, *, force: bool = False) -> list[dict[str, Any]]:
    """``GET /models`` (all pages) for this key's plan; cached for :data:`CATALOG_TTL_S`."""
    fp = _fingerprint(key)
    with _catalog_lock:
        hit = _catalog_cache.get(fp)
    if hit and not force and _clock() - hit[0] < CATALOG_TTL_S:
        return hit[1]
    models: list[dict[str, Any]] = []
    page = 1
    while page <= 20:
        payload = _get(key, "/models", params={"per_page": 100, "page": page})
        batch = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(batch, list) or not batch:
            break
        models.extend(m for m in batch if isinstance(m, dict) and isinstance(m.get("slug"), str))
        meta = payload.get("meta") if isinstance(payload, dict) else None
        try:
            last = int((meta or {}).get("last_page"))
        except (TypeError, ValueError):
            break
        if page >= last:
            break
        page += 1
    with _catalog_lock:
        _catalog_cache[fp] = (_clock(), models)
    return models


# --------------------------------------------------------------------------- jobs

def submit(key: str, path: str, *, json_body: Optional[dict[str, Any]] = None,
           data: Optional[dict[str, str]] = None, files: Optional[list[Any]] = None) -> str:
    """POST a paid job and return its ``request_id``.

    Only a 429 is retried: deAPI sends it before accepting a job. A 5xx or a connection
    dropped after sending may mean the job was accepted, so it is reported, never resent.
    """
    url = f"{API_BASE}{path}"
    for attempt in range(SUBMIT_429_RETRIES + 1):
        try:
            resp = _http.request("POST", url, headers=_headers(key), json=json_body, data=data,
                                 files=files, timeout=HTTP_TIMEOUT, allow_redirects=False)
        except requests.ConnectionError as exc:
            if _never_connected(exc):
                raise DeapiError("Could not connect to deAPI; nothing was submitted.",
                                 "connection_error") from None
            raise DeapiError(
                "The connection to deAPI dropped while submitting; the job may have been accepted. "
                f"Check the dashboard at {DASHBOARD_URL} before retrying so it is not paid twice.",
                "unknown_state") from None
        except requests.Timeout:
            raise DeapiError(
                "deAPI did not answer the submit in time; the job may have been accepted. "
                f"Check the dashboard at {DASHBOARD_URL} before retrying so it is not paid twice.",
                "unknown_state") from None
        if resp.status_code == 429 and attempt < SUBMIT_429_RETRIES:
            kind = _header(resp, "X-RateLimit-Type").lower()
            wait = _retry_after(resp, 2.0 ** (attempt + 1))
            if kind not in ("daily", "ip-not-allowed") and wait <= MAX_RETRY_AFTER_S:
                _sleep(wait)
                continue
        if resp.status_code >= 500:
            raise DeapiError(
                f"deAPI returned {resp.status_code} while submitting; the job may have been accepted. "
                f"Check the dashboard at {DASHBOARD_URL} before retrying.", "unknown_state",
                status=resp.status_code)
        if resp.status_code >= 400:
            raise error_from_response(resp)
        body = _json(resp)
        data_block = body.get("data") if isinstance(body, dict) else None
        request_id = data_block.get("request_id") if isinstance(data_block, dict) else None
        if not isinstance(request_id, str) or not _REQUEST_ID.match(request_id):
            raise DeapiError("deAPI accepted the request but returned no request_id.", "invalid_response")
        return request_id
    raise DeapiError("deAPI kept rate limiting the request.", "rate_limited")  # pragma: no cover


def wait_for_job(key: str, request_id: str, *, timeout_s: float, initial_interval: float = 2.0,
                 max_interval: float = 10.0, backoff: float = 1.5) -> dict[str, Any]:
    """Poll ``GET /jobs/{request_id}`` until ``done`` (returns the job) or a terminal failure."""
    url = f"{API_BASE}/jobs/{urllib.parse.quote(request_id, safe='')}"
    deadline = _clock() + timeout_s
    interval = initial_interval
    last_status = "submitted"

    def timed_out() -> DeapiError:
        return DeapiError(
            f"deAPI job {request_id} did not finish within {int(timeout_s)}s (last status: "
            f"{last_status}). It may still finish and is billed; do not resubmit it.",
            "timeout", request_id=request_id)

    while True:
        _sleep(interval)
        interval = min(interval * backoff, max_interval)
        try:
            resp = _http.request("GET", url, headers=_headers(key), timeout=POLL_HTTP_TIMEOUT,
                                 allow_redirects=False)
        except (requests.Timeout, requests.ConnectionError) as exc:
            logger.debug("deAPI poll transient error for %s: %s", request_id, type(exc).__name__)
            if _clock() >= deadline:
                raise timed_out() from None
            continue
        if resp.status_code in RETRYABLE_STATUSES:
            if _clock() >= deadline:
                raise timed_out()
            continue
        if resp.status_code >= 400:
            raise error_from_response(resp, request_id)
        body = _json(resp)
        job = body.get("data") if isinstance(body, dict) else None
        if not isinstance(job, dict):
            if _clock() >= deadline:
                raise timed_out()
            continue
        last_status = str(job.get("status") or "unknown").lower()
        if last_status == "done":
            return job
        if last_status in ("error", "failed"):
            raise job_failure(request_id, job)
        if _clock() >= deadline:
            raise timed_out()


def job_failure(request_id: str, job: dict[str, Any]) -> DeapiError:
    reason = job.get("error_reason") or job.get("error_code") or "unknown reason"
    parts = [f"deAPI job {request_id} failed ({reason})"]
    retryable = job.get("retryable")
    if retryable is True:
        parts.append("resubmitting the same request may succeed")
    elif retryable is False:
        parts.append("resubmitting the same request will not help")
    if job.get("refunded") is True:
        parts.append("the charge was refunded")
    return DeapiError("; ".join(parts) + ".", "job_failed", request_id=request_id)


def job_cost(job: dict[str, Any]) -> dict[str, Any]:
    """``{"cost_usd": float, "cost_is_estimate": bool}`` from a terminal job, or ``{}``."""
    price = job.get("price")
    if not isinstance(price, dict):
        return {}
    try:
        amount = float(price.get("amount"))
    except (TypeError, ValueError):
        return {}
    return {"cost_usd": amount, "cost_is_estimate": bool(price.get("is_estimated"))}


# --------------------------------------------------------------------------- source images

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
    (b"\xff\xd8\xff", "image/jpeg", "jpg"),
    (b"GIF87a", "image/gif", "gif"),
    (b"GIF89a", "image/gif", "gif"),
    (b"BM", "image/bmp", "bmp"),
)


def sniff_image(raw: bytes) -> Optional[tuple[str, str]]:
    """``(mime, extension)`` for the formats deAPI accepts, by magic bytes."""
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp", "webp"
    for magic, mime, ext in _MAGIC:
        if raw.startswith(magic):
            return mime, ext
    return None


def _check_image(raw: bytes, label: str) -> tuple[bytes, str, str]:
    if not raw:
        raise DeapiError(f"Source image {label} is empty.", "invalid_image_url")
    if len(raw) > MAX_SOURCE_BYTES:
        raise DeapiError(f"Source image {label} is larger than 10 MB (deAPI's upload limit).",
                         "source_too_large")
    kind = sniff_image(raw)
    if kind is None:
        raise DeapiError(f"Source image {label} is not a PNG, JPEG, GIF, BMP or WebP image.",
                         "invalid_image_url")
    return raw, kind[0], kind[1]


def _run_coroutine(factory: Any) -> Any:
    """Run ``factory()`` to completion from sync code, also when called inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


async def _download(url: str) -> bytes:
    from tools.url_safety import create_ssrf_safe_async_client

    chunks, total = [], 0
    # The client validates every connection it opens (redirect hops included) against Hermes's
    # SSRF policy; the size cap is enforced while streaming.
    async with create_ssrf_safe_async_client(timeout=60.0, follow_redirects=True) as http_client:
        async with http_client.stream("GET", url) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > MAX_SOURCE_BYTES:
                    raise DeapiError("Source image URL is larger than 10 MB (deAPI's upload limit).",
                                     "source_too_large")
                chunks.append(chunk)
    return b"".join(chunks)


def _fetch_remote(url: str) -> bytes:
    try:
        from tools.url_safety import is_safe_url
    except Exception:
        raise DeapiError("This Hermes version has no SSRF-safe downloader; pass a local file instead "
                         "of a URL.", "invalid_image_url") from None
    if not is_safe_url(url):
        raise DeapiError("Source image URL failed the SSRF safety check.", "invalid_image_url")
    try:
        return _run_coroutine(lambda: _download(url))
    except DeapiError:
        raise
    except Exception as exc:
        raise DeapiError(f"Could not download the source image ({type(exc).__name__}).",
                         "invalid_image_url") from None


def _read_local(value: str) -> bytes:
    """Read a local image. Only image files are read in full: the header is checked against the
    image formats deAPI accepts before anything else is read, so no other kind of file can be
    uploaded through this path."""
    path = Path(value).expanduser()
    if not path.is_file():
        raise DeapiError(f"Source image not found: {value}", "invalid_image_url")
    try:
        with path.open("rb") as fh:
            head = fh.read(16)
            if sniff_image(head) is None:
                raise DeapiError(f"Source image {value} is not a PNG, JPEG, GIF, BMP or WebP image.",
                                 "invalid_image_url")
            return head + fh.read(MAX_SOURCE_BYTES + 1 - len(head))
    except OSError as exc:
        raise DeapiError(f"Source image unreadable: {value} ({exc.strerror})", "invalid_image_url") from None


def load_source_image(value: str, *, index: int = 0) -> tuple[str, bytes, str]:
    """``(filename, bytes, mime)`` for an http(s) URL, a ``data:image/...;base64`` URL or a local path.

    deAPI takes source images as multipart uploads only, so URLs are downloaded here through
    Hermes's SSRF-guarded client, and a local path is read only when its header is an image's.
    """
    candidate = (value or "").strip()
    if not candidate:
        raise DeapiError("Blank source image.", "invalid_image_url")
    lowered = candidate.lower()
    if lowered.startswith(("http://", "https://")):
        raw, label = _fetch_remote(candidate), "URL"
    elif lowered.startswith("data:"):
        header, _, payload = candidate.partition(",")
        if not header.lower().startswith("data:image/") or ";base64" not in header.lower():
            raise DeapiError("Only base64 data:image/... URLs are accepted as source images.",
                             "invalid_image_url")
        try:
            raw = base64.b64decode(payload, validate=True)
        except Exception:
            raise DeapiError("The data: URL is not valid base64.", "invalid_image_url") from None
        label = "data URL"
    else:
        raw, label = _read_local(candidate), candidate
    raw, mime, ext = _check_image(raw, label)
    return f"source_{index}.{ext}", raw, mime
