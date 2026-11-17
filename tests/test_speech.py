"""text_to_speech backend: model/voice/language choice, request shape, output file, errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import CATALOG, REQUEST_ID, FakeResponse, catalog_page, client, job, speech_provider, submitted

MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" * 16
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
AUDIO_URL = "https://results.example/speech.mp3"


@pytest.fixture
def provider():
    return speech_provider.DeapiTTSProvider()


def ok_routes(http, downloads, audio=MP3):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/audio/speech", submitted())
    http.add("GET", f"/jobs/{REQUEST_ID}", job("processing"), job("done", result_url=AUDIO_URL))
    downloads.served[AUDIO_URL] = audio


def fields(post) -> dict:
    """The multipart form fields of a submit (sent as ``(name, (None, value))`` parts)."""
    assert post["json"] is None and post["data"] is None
    assert all(part[1][0] is None for part in post["files"])
    return {name: value for name, (_, value) in post["files"]}


def test_default_request_uses_kokoro_and_its_default_voice(http, downloads, provider, tmp_path):
    ok_routes(http, downloads)
    out = tmp_path / "audio" / "tts_1.mp3"
    out.parent.mkdir()
    written = provider.synthesize("Hello from deAPI.", str(out))
    assert written == str(out) and out.read_bytes() == MP3
    post = http.posts()[0]
    assert post["path"] == "/audio/speech" and post["allow_redirects"] is False
    assert fields(post) == {"text": "Hello from deAPI.", "model": "Kokoro", "voice": "af_alloy",
                            "lang": "en-us", "speed": "1.0", "format": "mp3", "sample_rate": "24000"}
    assert downloads.calls == [(AUDIO_URL, client.MAX_RESULT_BYTES)]
    assert post["headers"]["Authorization"] == "Bearer test-key-not-real"
    assert not list(out.parent.glob("*.part"))


@pytest.mark.parametrize("voice, expected", [
    ("af_nova", ("af_nova", "en-us")),
    ("nova", ("af_nova", "en-us")),          # OpenAI-style name matches the voice name
    ("BF_EMMA", ("bf_emma", "en-gb")),       # language follows the voice
    ("shimmer", ("af_alloy", "en-us")),      # unknown voice: model default
])
def test_voice_by_id_or_name(http, downloads, provider, tmp_path, voice, expected):
    ok_routes(http, downloads)
    provider.synthesize("Hello there.", str(tmp_path / "a.mp3"), voice=voice)
    sent = fields(http.posts()[0])
    assert (sent["voice"], sent["lang"]) == expected


def test_config_voice_and_language(http, downloads, provider, config, tmp_path):
    config({"tts": {"deapi": {"voice": "bm_fable", "language": "pl"}}})
    ok_routes(http, downloads)
    provider.synthesize("Dzień dobry.", str(tmp_path / "a.mp3"))
    sent = fields(http.posts()[0])
    # bm_fable is not a Polish voice: the Polish language wins, with its first voice.
    assert (sent["voice"], sent["lang"]) == ("pf_zofia", "pl")
    http.calls.clear()
    config({"tts": {"deapi": {"voice": "bm_fable"}}})
    provider.synthesize("Good morning.", str(tmp_path / "b.mp3"), voice="not-a-voice")
    assert (fields(http.posts()[0])["voice"], fields(http.posts()[0])["lang"]) == ("bm_fable", "en-gb")


def test_model_from_tts_model_and_config(http, downloads, provider, config, tmp_path):
    ok_routes(http, downloads)
    provider.synthesize("Hello there, Chatterbox.", str(tmp_path / "a.mp3"), model="Chatterbox", speed=1.5)
    sent = fields(http.posts()[0])
    assert (sent["model"], sent["voice"], sent["lang"], sent["speed"]) == ("Chatterbox", "default", "en", "1.0")
    http.calls.clear()
    config({"tts": {"deapi": {"model": "Qwen3_TTS_12Hz_1_7B_CustomVoice"}}})
    provider.synthesize("Hello there, Qwen.", str(tmp_path / "b.mp3"), voice="Serena")
    sent = fields(http.posts()[0])
    assert (sent["model"], sent["voice"], sent["lang"]) == ("Qwen3_TTS_12Hz_1_7B_CustomVoice", "Serena", "Chinese")
    http.calls.clear()
    provider.synthesize("Hello there, Qwen.", str(tmp_path / "c.mp3"), voice="Vivian")
    assert fields(http.posts()[0])["lang"] == "English"  # the card's default language first


def test_models_without_preset_voices_fall_back(http, downloads, provider, tmp_path):
    ok_routes(http, downloads)
    provider.synthesize("Hello there.", str(tmp_path / "a.mp3"), model="Qwen3_TTS_12Hz_1_7B_VoiceDesign")
    assert fields(http.posts()[0])["model"] == "Kokoro"
    model, note = speech_provider.pick_model(CATALOG, ["Qwen3_TTS_12Hz_1_7B_VoiceDesign"])
    assert model["slug"] == "Kokoro" and "no preset voices" in note
    _, note = speech_provider.pick_model(CATALOG, ["Flux1schnell", "nope"])
    assert "not a text-to-speech model" in note and "'nope' is not in" in note


def test_speed_is_clamped_to_the_card(http, downloads, provider, tmp_path):
    ok_routes(http, downloads)
    provider.synthesize("Hello there.", str(tmp_path / "a.mp3"), speed=3.0)
    assert fields(http.posts()[0])["speed"] == "2.0"
    http.calls.clear()
    provider.synthesize("Hello there.", str(tmp_path / "b.mp3"), speed=0.25)
    assert fields(http.posts()[0])["speed"] == "0.5"


def test_unsupported_format_writes_the_real_extension(http, downloads, provider, tmp_path):
    ok_routes(http, downloads)
    out = tmp_path / "speech.wav"
    written = Path(provider.synthesize("Hello there.", str(out), format="wav"))
    assert fields(http.posts()[0])["format"] == "mp3"
    assert written == tmp_path / "speech.mp3" and written.read_bytes() == MP3 and not out.exists()
    # A renamed path never replaces an existing file.
    keep = tmp_path / "speech.mp3"
    keep.write_bytes(b"user file")
    written = Path(provider.synthesize("Hello there.", str(out), format="wav"))
    assert keep.read_bytes() == b"user file"
    assert written.suffix == ".mp3" and written != keep and written.read_bytes() == MP3


def test_extension_follows_the_audio_bytes(http, downloads, provider, tmp_path):
    ok_routes(http, downloads, audio=WAV)
    written = provider.synthesize("Hello there.", str(tmp_path / "a.mp3"))
    assert written.endswith("a.wav")


@pytest.mark.parametrize("text, model, needle", [
    ("x" * 2001, "Chatterbox", "max_text_length: 2000"),
    ("Hi!", "Chatterbox", "at least 10 characters"),
])
def test_text_length_limits_are_checked_before_submitting(http, provider, tmp_path, text, model, needle):
    http.add("GET", "/models", catalog_page())
    with pytest.raises(ValueError, match=needle):
        provider.synthesize(text, str(tmp_path / "a.mp3"), model=model)
    assert http.posts() == []


def test_missing_key_and_text(http, provider, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Text is required"):
        provider.synthesize("  ", str(tmp_path / "a.mp3"))
    monkeypatch.delenv("DEAPI_API_KEY")
    with pytest.raises(ValueError, match="DEAPI_API_KEY not set"):
        provider.synthesize("Hello there.", str(tmp_path / "a.mp3"))
    assert http.calls == []


def test_failed_job_raises_with_reason(http, provider, tmp_path):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/audio/speech", submitted())
    http.add("GET", f"/jobs/{REQUEST_ID}", job("error", error_reason="PROCESSING_FAILED", refunded=True))
    with pytest.raises(client.DeapiError, match="PROCESSING_FAILED.*refunded") as exc:
        provider.synthesize("Hello there.", str(tmp_path / "a.mp3"))
    assert exc.value.error_type == "job_failed"


def test_submit_5xx_is_not_resent(http, provider, tmp_path):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/audio/speech", FakeResponse(502, {"message": "bad gateway"}))
    with pytest.raises(client.DeapiError) as exc:
        provider.synthesize("Hello there.", str(tmp_path / "a.mp3"))
    assert exc.value.error_type == "unknown_state" and len(http.posts()) == 1


def test_result_download_failure_is_reported(http, downloads, provider, tmp_path):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/audio/speech", submitted())
    http.add("GET", f"/jobs/{REQUEST_ID}", job("done", result_url="https://results.example/missing.mp3"))
    with pytest.raises(client.DeapiError, match="download"):
        provider.synthesize("Hello there.", str(tmp_path / "a.mp3"))
    assert not (tmp_path / "a.mp3").exists()


def test_picker_lists_models_with_preset_voices(http, provider):
    http.add("GET", "/models", catalog_page())
    rows = provider.list_models()
    assert [r["id"] for r in rows] == ["Kokoro", "Chatterbox", "Qwen3_TTS_12Hz_1_7B_CustomVoice"]
    assert rows[0]["strengths"].startswith("6 voices · 3 languages · up to 10001 characters")
    assert provider.default_model() == "Kokoro"
    assert provider.default_voice() == "af_alloy"


def test_voice_list_and_config_defaults(http, provider, config):
    http.add("GET", "/models", catalog_page())
    config({"tts": {"deapi": {"model": "Qwen3_TTS_12Hz_1_7B_CustomVoice", "voice": "ryan"}}})
    assert provider.default_model() == "Qwen3_TTS_12Hz_1_7B_CustomVoice"
    assert provider.default_voice() == "Ryan"
    voices = {v["id"]: v for v in provider.list_voices()}
    assert set(voices) == {"Vivian", "Serena", "Ryan"}
    assert voices["Vivian"]["language"] == "Chinese, English"


def test_voice_compatible_is_on_unless_disabled(provider, config):
    assert provider.voice_compatible is True
    config({"tts": {"deapi": {"voice_compatible": False}}})
    assert provider.voice_compatible is False
    assert provider.streams_pcm is False


def test_sniff_audio():
    assert speech_provider.sniff_audio(MP3) == "mp3"
    assert speech_provider.sniff_audio(b"\xff\xf3\x00\x00") == "mp3"
    assert speech_provider.sniff_audio(WAV) == "wav"
    assert speech_provider.sniff_audio(b"OggS\x00") == "ogg"
    assert speech_provider.sniff_audio(b"fLaC\x00") == "flac"
    assert speech_provider.sniff_audio(b"<html>") is None
