"""deAPI speech-to-text backend: an audio file → ``POST /api/v2/audio/transcriptions`` → text.

Hermes calls :meth:`DeapiTranscriptionProvider.transcribe` for voice messages and voice mode
when ``stt.provider: deapi``, with ``stt.deapi.model`` and the language hint it knows
(``stt.deapi.language`` → ``stt.language``). The default model is Whisper Large V3 CT2, which
honors the language hint (Whisper Large V3 ignores it). The recording is uploaded once; the
transcript comes back inline in the job (``return_result_in_response``).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from agent.transcription_provider import TranscriptionProvider

from . import cards, client, settings

logger = logging.getLogger(__name__)

PROVIDER = "deapi"
KIND = "stt"
INFERENCE_TYPE = "audio_file2text"
DEFAULT_TIMEOUT_S = 300.0
MAX_AUDIO_BYTES = 20 * 1024 * 1024  # deAPI's upload cap for audio files
MAX_VIDEO_BYTES = 50 * 1024 * 1024  # and for video files
MAX_TRANSCRIPT_BYTES = 10 * 1024 * 1024
# Content types by extension for the formats Hermes hands to STT providers.
CONTENT_TYPES = {
    ".mp3": "audio/mpeg", ".mpga": "audio/mpeg", ".mpeg": "audio/mpeg", ".wav": "audio/wav",
    ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg", ".webm": "audio/webm",
    ".flac": "audio/flac", ".aac": "audio/aac", ".m4a": "audio/mp4", ".mp4": "video/mp4",
}
_VIDEO_EXTENSIONS = {".mp4"}
_LANGUAGE = re.compile(r"^[a-z]{2,3}$")


def normalize_language(value: Optional[str]) -> Optional[str]:
    """deAPI takes a primary language subtag (``en``, ``pl``); ``en-US`` → ``en``, ``auto`` → None."""
    if not isinstance(value, str):
        return None
    primary = re.split(r"[-_]", value.strip().lower(), maxsplit=1)[0]
    return primary if _LANGUAGE.match(primary) else None


def transcript_text(raw: Any) -> str:
    """The text of a transcription result: a JSON document with ``text`` (Whisper Large V3 CT2)
    or plain text (Whisper Large V3)."""
    if isinstance(raw, dict):
        return str(raw.get("text") or "").strip()
    if not isinstance(raw, str):
        return ""
    if raw.lstrip().startswith("{"):
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = None
        if isinstance(doc, dict) and isinstance(doc.get("text"), str):
            return doc["text"].strip()
    return raw.strip()


def _result(job: dict[str, Any]) -> str:
    inline = job.get("result")
    if isinstance(inline, (str, dict)) and inline:
        return transcript_text(inline)
    result_url = job.get("result_url")
    if isinstance(result_url, str) and result_url.strip():
        raw = client.download_result(result_url, max_bytes=MAX_TRANSCRIPT_BYTES, label="transcript")
        return transcript_text(raw.decode("utf-8", errors="replace"))
    return ""


class DeapiTranscriptionProvider(TranscriptionProvider):
    """deAPI speech-to-text for recordings."""

    @property
    def name(self) -> str:
        return PROVIDER

    @property
    def display_name(self) -> str:
        return "deAPI"

    def is_available(self) -> bool:
        return bool(client.api_key())

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "deAPI", "badge": "paid",
            "tag": "Whisper Large V3 — live catalog from api.deapi.ai; speech-to-text",
            "env_vars": [{"key": client.KEY_ENV, "prompt": "deAPI API key", "url": client.KEY_URL}],
        }

    # ------------------------------------------------------------------ catalog

    def _models(self) -> list[dict[str, Any]]:
        key = client.api_key()
        if not key:
            return []
        try:
            return client.fetch_models(key)
        except client.DeapiError as exc:
            logger.debug("deAPI catalog unavailable: %s", exc)
            return []

    def list_models(self) -> list[dict[str, Any]]:
        rows = []
        for model in cards.sort_rows([m for m in self._models() if cards.serves(m, [INFERENCE_TYPE])]):
            strengths = []
            if re.search(r"ct2", model["slug"], re.IGNORECASE):
                strengths.append("honors the language hint")
            if cards.flag(model, "supports_diarization"):
                strengths.append("speaker labels available")
            if cards.is_partner(model):
                strengths.append("partner-served (priced by the partner)")
            rows.append({"id": model["slug"], "display": cards.display_name(model),
                         "strengths": " · ".join(strengths)})
        return rows

    def default_model(self) -> Optional[str]:
        model, _ = cards.choose(self._models(), INFERENCE_TYPE, [settings.configured_model(KIND)])
        return model["slug"] if model else None

    # ------------------------------------------------------------------ transcription

    def transcribe(
        self, file_path: str, *, model: Optional[str] = None, language: Optional[str] = None, **extra: Any,
    ) -> dict[str, Any]:
        model_slug = ""

        def fail(error: str, error_type: str, **fields: Any) -> dict[str, Any]:
            return {"success": False, "transcript": "", "provider": PROVIDER, "error": error,
                    "error_type": error_type, "model": model_slug, **fields}

        try:
            key = client.api_key()
            if not key:
                return fail(f"{client.KEY_ENV} not set. Add your deAPI API key to Hermes (keys at "
                            f"{client.KEY_URL}).", "auth_required")
            path = Path(file_path)
            suffix = path.suffix.lower()
            if suffix not in CONTENT_TYPES:
                return fail(f"deAPI does not take {suffix or 'extensionless'} files for transcription.",
                            "unsupported_format")
            cap = MAX_VIDEO_BYTES if suffix in _VIDEO_EXTENSIONS else MAX_AUDIO_BYTES
            try:
                size = path.stat().st_size
            except OSError as exc:
                return fail(f"Audio file unreadable: {exc.strerror}", "invalid_file")
            if size > cap:
                return fail(f"The recording is larger than {cap // (1024 * 1024)} MB (deAPI's upload limit).",
                            "source_too_large")
            try:
                models = client.fetch_models(key)
            except client.DeapiError as exc:
                return fail(f"Could not load the deAPI model catalog: {exc}", exc.error_type)
            card, note = cards.choose(models, INFERENCE_TYPE, [model, settings.configured_model(KIND)])
            if card is None:
                return fail("No deAPI transcription model is available on this key's plan.", "no_model_available")
            model_slug = card["slug"]

            fields: dict[str, Any] = {"model": model_slug, "include_ts": False, "return_result_in_response": True}
            lang = normalize_language(language)
            allowed = [str(x).lower() for x in (cards.limit(card, "languages") or [])]
            if lang and allowed and lang not in allowed:
                note = "; ".join(filter(None, [note, f"language {lang!r} is not offered by {model_slug}; "
                                                     "auto-detected instead"]))
                lang = None
            if lang:
                fields["lang"] = lang
            # Hermes may pass a vocabulary ``prompt``; deAPI's transcription has no such field.
            with path.open("rb") as fh:
                audio = fh.read(cap + 1)
            files = [("source_file", (f"audio{suffix}", audio, CONTENT_TYPES[suffix]))]
            request_id = client.submit(key, "/audio/transcriptions", data=cards.as_form(fields), files=files)
            job = client.wait_for_job(key, request_id,
                                      timeout_s=settings.timeout_seconds(KIND, DEFAULT_TIMEOUT_S),
                                      initial_interval=1.5, max_interval=5.0)
            text = _result(job)
        except client.DeapiError as exc:
            extra_fields = {"request_id": exc.request_id} if exc.request_id else {}
            return fail(str(exc), exc.error_type, **extra_fields)
        except OSError as exc:
            return fail(f"Audio file unreadable: {exc.strerror or type(exc).__name__}", "invalid_file")
        except Exception as exc:  # the ABC forbids raising; report anything unexpected
            logger.debug("deAPI transcription failed", exc_info=True)
            return fail(f"deAPI transcription failed ({type(exc).__name__}).", "api_error")

        result: dict[str, Any] = {"success": True, "transcript": text, "provider": PROVIDER,
                                  "model": model_slug, "request_id": request_id, **client.job_cost(job)}
        if lang:
            result["language"] = lang
        if note:
            result["note"] = note
        return result
