"""deAPI ``text_to_speech`` backend: preset voices on ``POST /api/v2/audio/speech``.

Hermes calls :meth:`DeapiTTSProvider.synthesize` once per text chunk with ``tts.voice``,
``tts.model``, ``tts.speed`` and ``tts.output_format``. Models, voices, languages, the speed
range, text length limits and output formats all come from the key's live model list, so the
request always fits the chosen model. Only models with preset voices are offered: voice cloning
and voice design need a reference recording or a voice description, which Hermes does not pass.

Model choice: ``tts.model`` when it is a deAPI model with preset voices → ``tts.deapi.model`` →
Kokoro (else the first own-fleet speech model). Voice choice: ``tts.voice`` when the model has it
(by id, e.g. ``af_nova``, or by name, e.g. ``nova``) → ``tts.deapi.voice`` → the model's default.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from agent.tts_provider import DEFAULT_OUTPUT_FORMAT, TTSProvider

from . import cards, client, settings

logger = logging.getLogger(__name__)

PROVIDER = "deapi"
KIND = "tts"
INFERENCE_TYPE = "txt2audio"
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_SAMPLE_RATE = 24000
# File extension for a format name, where the two differ.
_EXTENSIONS = {"opus": "ogg", "mpeg": "mp3"}


def _lower(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def speaks(model: dict[str, Any]) -> bool:
    """A text-to-speech model with preset voices (deAPI's ``custom_voice`` mode)."""
    if not cards.serves(model, [INFERENCE_TYPE]) or cards.flag(model, "supports_custom_voice") is False:
        return False
    return any(cards.voices(language) for language in cards.languages(model))


def pick_model(models: list[dict[str, Any]],
               candidates: list[Optional[str]]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """First candidate that is a speech model with preset voices, else Kokoro / the first such model.

    Returns ``(model, note)``; ``note`` explains a skipped candidate.
    """
    notes = []
    for slug in candidates:
        if not isinstance(slug, str) or not slug.strip():
            continue
        model = cards.find(models, slug)
        if model is None:
            notes.append(f"model {slug.strip()!r} is not in this deAPI key's catalog")
        elif speaks(model):
            return model, None
        elif cards.serves(model, [INFERENCE_TYPE]):
            notes.append(f"model {slug.strip()!r} has no preset voices (voice cloning and voice design "
                         "are not available through text_to_speech)")
        else:
            notes.append(f"model {slug.strip()!r} is not a text-to-speech model")
    picked = cards.auto_pick([m for m in models if speaks(m)], INFERENCE_TYPE)
    note = "; ".join(notes) + f"; used {picked['slug']}" if notes and picked else None
    return picked, note


def _find_language(model: dict[str, Any], value: Optional[str]) -> Optional[dict[str, Any]]:
    wanted = _lower(value)
    if not wanted:
        return None
    return next((lang for lang in cards.languages(model)
                 if wanted in (_lower(lang.get("slug")), _lower(lang.get("name")))), None)


def _default_language(model: dict[str, Any]) -> Optional[dict[str, Any]]:
    langs = [lang for lang in cards.languages(model) if cards.voices(lang)]
    default = _find_language(model, cards.default(model, "lang"))
    if default is not None and cards.voices(default):
        return default
    return langs[0] if langs else None


def _match_voice(language: dict[str, Any], value: str) -> Optional[dict[str, Any]]:
    wanted = _lower(value)
    return next((v for v in cards.voices(language)
                 if wanted in (_lower(v.get("slug")), _lower(v.get("name")))), None)


def resolve_voice(model: dict[str, Any], candidates: list[Optional[str]],
                  language: Optional[str] = None) -> tuple[str, str, Optional[str]]:
    """``(voice slug, language slug, note)`` for a request to ``model``.

    ``language`` (``tts.deapi.language``: a slug such as ``en-gb`` or a name) narrows the voices
    when the model has it. A candidate voice may be a voice id or a voice name; the first one the
    model has wins. Otherwise the model's default voice, or the first voice of the language.
    """
    pinned = _find_language(model, language)
    if pinned is not None and not cards.voices(pinned):
        pinned = None
    notes = []
    if language and pinned is None:
        notes.append(f"language {language!r} is not offered by {model['slug']}")
    default_lang = _default_language(model)
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        pool = [pinned] if pinned is not None else (
            [default_lang] if default_lang is not None else []) + [
            lang for lang in cards.languages(model) if lang is not default_lang]
        for lang in pool:
            voice = _match_voice(lang, candidate)
            if voice is not None:
                return voice["slug"], lang["slug"], "; ".join(notes) or None
        notes.append(f"voice {candidate.strip()!r} is not a voice of {model['slug']}"
                     + (f" in {pinned['slug']}" if pinned is not None else ""))
    lang = pinned or default_lang
    if lang is None:
        raise client.DeapiError(f"deAPI model {model['slug']} lists no voices.", "invalid_response")
    voice = _match_voice(lang, str(cards.default(model, "voice", ""))) or cards.voices(lang)[0]
    return voice["slug"], lang["slug"], "; ".join(notes) or None


def output_format(model: dict[str, Any], requested: Optional[str]) -> str:
    """The requested format when the model offers it, else the model's default / first format."""
    offered = [_lower(f) for f in (cards.limit(model, "output_formats") or []) if isinstance(f, str)]
    wanted = _lower(requested)
    if wanted and wanted in offered:
        return wanted
    default = _lower(cards.default(model, "format"))
    if default and (not offered or default in offered):
        return default
    return offered[0] if offered else DEFAULT_OUTPUT_FORMAT


def sample_rate(model: dict[str, Any]) -> int:
    rate = cards.default(model, "sample_rate")
    if not rate:
        rates = cards.limit(model, "available_ratios") or []
        rate = rates[0] if rates else DEFAULT_SAMPLE_RATE
    try:
        return int(rate)
    except (TypeError, ValueError):
        return DEFAULT_SAMPLE_RATE


def sniff_audio(raw: bytes) -> Optional[str]:
    """File extension from the audio's magic bytes, or None when unrecognised."""
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "wav"
    if raw.startswith(b"OggS"):
        return "ogg"
    if raw.startswith(b"fLaC"):
        return "flac"
    if raw.startswith(b"ID3") or (len(raw) > 1 and raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0):
        return "mp3"
    return None


def _target_path(output_path: str, ext: str) -> Path:
    """``output_path`` with the real extension. A renamed path never replaces an existing file."""
    path = Path(output_path)
    if path.suffix.lower() == f".{ext}":
        return path
    renamed = path.with_suffix(f".{ext}")
    if os.path.lexists(renamed):
        renamed = path.with_name(f"{path.stem}.{uuid.uuid4().hex[:8]}.{ext}")
    return renamed


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class DeapiTTSProvider(TTSProvider):
    """deAPI text-to-speech with preset voices."""

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
            "tag": "Kokoro, Chatterbox, Qwen3-TTS, … — live catalog from api.deapi.ai; preset voices",
            "env_vars": [{"key": client.KEY_ENV, "prompt": "deAPI API key", "url": client.KEY_URL}],
        }

    @property
    def voice_compatible(self) -> bool:
        """Voice-bubble delivery (Hermes converts the MP3 to Opus with ffmpeg). On by default;
        ``tts.deapi.voice_compatible: false`` sends replies as regular audio files instead."""
        value = settings.deapi_section(KIND).get("voice_compatible")
        return value if isinstance(value, bool) else True

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

    def _default_card(self) -> Optional[dict[str, Any]]:
        model, _ = pick_model(self._models(), [settings.configured_model(KIND)])
        return model

    def list_models(self) -> list[dict[str, Any]]:
        rows = []
        for model in cards.sort_rows([m for m in self._models() if speaks(m)]):
            langs = [lang for lang in cards.languages(model) if cards.voices(lang)]
            voice_ids = {v["slug"] for lang in langs for v in cards.voices(lang)}
            strengths = [f"{len(voice_ids)} voice{'s' if len(voice_ids) != 1 else ''}",
                         f"{len(langs)} language{'s' if len(langs) != 1 else ''}"]
            max_text = cards.limit(model, "max_text")
            if max_text:
                strengths.append(f"up to {int(max_text)} characters per request")
            if cards.is_partner(model):
                strengths.append("partner-served (priced by the partner)")
            rows.append({"id": model["slug"], "display": cards.display_name(model),
                         "strengths": " · ".join(strengths)})
        return rows

    def default_model(self) -> Optional[str]:
        model = self._default_card()
        return model["slug"] if model else None

    def list_voices(self) -> list[dict[str, Any]]:
        model = self._default_card()
        if model is None:
            return []
        rows: dict[str, dict[str, Any]] = {}
        for lang in cards.languages(model):
            for voice in cards.voices(lang):
                row = rows.setdefault(voice["slug"], {
                    "id": voice["slug"], "display": str(voice.get("name") or voice["slug"]),
                    "language": [], "gender": voice.get("gender")})
                row["language"].append(str(lang.get("name") or lang["slug"]))
        return [{**row, "language": ", ".join(row["language"])} for row in rows.values()]

    def default_voice(self) -> Optional[str]:
        model = self._default_card()
        if model is None:
            return None
        try:
            voice, _, _ = resolve_voice(model, [settings.configured_text(KIND, "voice")],
                                        settings.configured_text(KIND, "language"))
        except client.DeapiError:
            return None
        return voice

    # ------------------------------------------------------------------ synthesis

    def synthesize(
        self, text: str, output_path: str, *, voice: Optional[str] = None, model: Optional[str] = None,
        speed: Optional[float] = None, format: str = DEFAULT_OUTPUT_FORMAT, **extra: Any,
    ) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Text is required")
        key = client.api_key()
        if not key:
            raise ValueError(f"{client.KEY_ENV} not set. Run `hermes tools` → Text-to-Speech → deAPI to "
                             f"configure it (keys at {client.KEY_URL}).")
        card, note = pick_model(client.fetch_models(key), [model, settings.configured_model(KIND)])
        if card is None:
            raise ValueError("No deAPI text-to-speech model with preset voices is available on this key's plan.")
        slug = card["slug"]
        if note:
            logger.info("deAPI TTS: %s", note)

        max_text, min_text = cards.limit(card, "max_text"), cards.limit(card, "min_text")
        if max_text and len(text) > int(max_text):
            raise ValueError(f"deAPI model {slug} takes at most {int(max_text)} characters per request "
                             f"(got {len(text)}). Set tts.deapi.max_text_length: {int(max_text)} in "
                             "config.yaml so Hermes splits longer text.")
        if min_text and len(text) < int(min_text):
            raise ValueError(f"deAPI model {slug} needs at least {int(min_text)} characters of text "
                             f"(got {len(text)}).")

        voice_slug, lang, voice_note = resolve_voice(
            card, [voice, settings.configured_text(KIND, "voice")], settings.configured_text(KIND, "language"))
        if voice_note:
            logger.info("deAPI TTS: %s; used voice %s (%s)", voice_note, voice_slug, lang)
        rate = float(speed) if isinstance(speed, (int, float)) and not isinstance(speed, bool) else None
        rate = float(cards.clamp(card, "speed", rate if rate is not None else float(cards.default(card, "speed", 1))))
        fmt = output_format(card, format)
        body = {"text": text, "model": slug, "voice": voice_slug, "lang": lang, "speed": rate,
                "format": fmt, "sample_rate": sample_rate(card)}
        # The endpoint is multipart/form-data; (None, value) parts send plain form fields.
        fields = [(name, (None, value)) for name, value in cards.as_form(body).items()]
        request_id = client.submit(key, "/audio/speech", files=fields)
        job = client.wait_for_job(key, request_id, timeout_s=settings.timeout_seconds(KIND, DEFAULT_TIMEOUT_S),
                                  initial_interval=1.0, max_interval=3.0)
        result_url = job.get("result_url")
        if not isinstance(result_url, str) or not result_url.strip():
            raise client.DeapiError(f"deAPI job {request_id} finished without a result URL.", "empty_response",
                                    request_id=request_id)
        audio = client.download_result(result_url, label="audio")
        target = _target_path(output_path, sniff_audio(audio) or _EXTENSIONS.get(fmt, fmt))
        _write(target, audio)
        cost = client.job_cost(job)
        logger.info("deAPI TTS: %s, voice %s (%s), request %s%s", slug, voice_slug, lang, request_id,
                    f", ${cost['cost_usd']:.6f}" if cost else "")
        return str(target)
