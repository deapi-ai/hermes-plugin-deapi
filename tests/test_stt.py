"""Speech-to-text backend: upload, language hint, result parsing, error envelope (never raises)."""

from __future__ import annotations

import json

import pytest

from conftest import REQUEST_ID, FakeResponse, catalog_page, job, stt_provider, submitted

OGG = b"OggS" + b"\x00" * 60
CT2_RESULT = json.dumps({"text": " Hello from a voice note. ", "language": "en", "ts_level": None,
                         "diarization_available": True, "segments": []})


@pytest.fixture
def provider():
    return stt_provider.DeapiTranscriptionProvider()


@pytest.fixture
def voice_note(tmp_path):
    path = tmp_path / "voice.ogg"
    path.write_bytes(OGG)
    return str(path)


def ok_routes(http, final=None):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/audio/transcriptions", submitted())
    http.add("GET", f"/jobs/{REQUEST_ID}", job("processing"),
             final or job("done", result=CT2_RESULT, result_url=None))


def test_upload_and_inline_json_result(http, provider, voice_note):
    ok_routes(http)
    result = provider.transcribe(voice_note, language="en-US")
    assert result == {"success": True, "transcript": "Hello from a voice note.", "provider": "deapi",
                      "model": "WhisperLargeV3Ct2", "request_id": REQUEST_ID, "cost_usd": 0.0021,
                      "cost_is_estimate": False, "language": "en"}
    post = http.posts()[0]
    assert post["path"] == "/audio/transcriptions" and post["json"] is None
    assert post["data"] == {"model": "WhisperLargeV3Ct2", "include_ts": "false",
                            "return_result_in_response": "true", "lang": "en"}
    assert post["files"] == [("source_file", ("audio.ogg", OGG, "audio/ogg"))]


def test_no_language_means_auto_detect(http, provider, voice_note):
    ok_routes(http)
    for language in (None, "auto", ""):
        http.calls.clear()
        result = provider.transcribe(voice_note, language=language)
        assert result["success"] and "lang" not in http.posts()[0]["data"] and "language" not in result


def test_plain_text_result_from_whisper_v3(http, provider, voice_note):
    ok_routes(http, final=job("done", result="Plain transcript.\n", result_url=None))
    result = provider.transcribe(voice_note, model="WhisperLargeV3")
    assert result["transcript"] == "Plain transcript." and result["model"] == "WhisperLargeV3"


def test_result_url_is_downloaded_when_not_inline(http, downloads, provider, voice_note):
    url = "https://results.example/transcript.json"
    downloads.served[url] = CT2_RESULT.encode()
    ok_routes(http, final=job("done", result=None, result_url=url))
    result = provider.transcribe(voice_note)
    assert result["transcript"] == "Hello from a voice note."
    assert downloads.calls == [(url, stt_provider.MAX_TRANSCRIPT_BYTES)]


def test_silence_is_an_empty_success(http, provider, voice_note):
    ok_routes(http, final=job("done", result=json.dumps({"text": "", "segments": []}), result_url=None))
    result = provider.transcribe(voice_note)
    assert result["success"] is True and result["transcript"] == ""


def test_config_model_and_foreign_model_note(http, provider, voice_note, config):
    ok_routes(http)
    config({"stt": {"deapi": {"model": "WhisperLargeV3"}}})
    assert provider.default_model() == "WhisperLargeV3"
    assert provider.transcribe(voice_note)["model"] == "WhisperLargeV3"
    http.calls.clear()
    # Hermes passes stt.deapi.model as ``model``; a model id from another provider falls back.
    config({"stt": {"deapi": {"model": "whisper-1"}}})
    result = provider.transcribe(voice_note, model="whisper-1")
    assert result["model"] == "WhisperLargeV3Ct2" and "'whisper-1' is not in" in result["note"]


def test_language_outside_the_model_list_is_dropped(http, provider, voice_note):
    limited = [dict(m) for m in catalog_page().json()["data"]]
    for model in limited:
        if model["slug"] == "WhisperLargeV3Ct2":
            model["info"] = {**model["info"], "limits": {**model["info"]["limits"], "languages": ["en", "de"]}}
    http.add("GET", "/models", catalog_page(limited))
    http.add("POST", "/audio/transcriptions", submitted())
    http.add("GET", f"/jobs/{REQUEST_ID}", job("done", result=CT2_RESULT, result_url=None))
    result = provider.transcribe(voice_note, language="pl")
    assert result["success"] and "lang" not in http.posts()[0]["data"]
    assert "auto-detected" in result["note"]


@pytest.mark.parametrize("value, expected", [
    ("en", "en"), ("en-US", "en"), ("pt_BR", "pt"), ("PL", "pl"), ("auto", None), ("english", None),
    (None, None), ("", None)])
def test_normalize_language(value, expected):
    assert stt_provider.normalize_language(value) == expected


def test_transcript_text_shapes():
    assert stt_provider.transcript_text(CT2_RESULT) == "Hello from a voice note."
    assert stt_provider.transcript_text("{not json") == "{not json"
    assert stt_provider.transcript_text({"text": "x"}) == "x"
    assert stt_provider.transcript_text(None) == ""


def test_errors_come_back_as_envelopes(http, provider, voice_note, tmp_path, monkeypatch):
    weird = tmp_path / "voice.xyz"
    weird.write_bytes(OGG)
    result = provider.transcribe(str(weird))
    assert result["success"] is False and result["error_type"] == "unsupported_format"
    big = tmp_path / "big.wav"
    big.write_bytes(b"\x00" * (stt_provider.MAX_AUDIO_BYTES + 1))
    assert provider.transcribe(str(big))["error_type"] == "source_too_large"
    assert provider.transcribe(str(tmp_path / "missing.ogg"))["error_type"] == "invalid_file"
    assert http.calls == []
    monkeypatch.delenv("DEAPI_API_KEY")
    result = provider.transcribe(voice_note)
    assert result == {"success": False, "transcript": "", "provider": "deapi", "error_type": "auth_required",
                      "model": "", "error": result["error"]}
    assert "DEAPI_API_KEY" in result["error"]


def test_api_errors_and_failed_jobs(http, provider, voice_note):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/audio/transcriptions",
             FakeResponse(422, {"message": "Insufficient balance.", "code": "insufficient_balance"}))
    result = provider.transcribe(voice_note)
    assert result["success"] is False and result["error_type"] == "insufficient_balance"
    assert result["transcript"] == ""

    http.routes[("POST", "/audio/transcriptions")] = [submitted()]
    http.add("GET", f"/jobs/{REQUEST_ID}", job("error", error_reason="SOURCE_UNPROCESSABLE", retryable=False))
    result = provider.transcribe(voice_note)
    assert result["error_type"] == "job_failed" and "SOURCE_UNPROCESSABLE" in result["error"]
    assert result["request_id"] == REQUEST_ID


def test_unexpected_exceptions_never_escape(http, provider, voice_note, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(stt_provider.client, "fetch_models", boom)
    result = provider.transcribe(voice_note)
    assert result["success"] is False and result["error_type"] == "api_error"


def test_picker_rows(http, provider):
    http.add("GET", "/models", catalog_page())
    rows = provider.list_models()
    assert [r["id"] for r in rows] == ["WhisperLargeV3", "WhisperLargeV3Ct2"]
    assert "language hint" in rows[1]["strengths"] and "speaker labels" in rows[1]["strengths"]
    assert provider.default_model() == "WhisperLargeV3Ct2"
    assert provider.streaming_capable is False
