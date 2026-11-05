"""image_generate backend: request shape, polling, errors. Fake HTTP only."""

from __future__ import annotations

import base64
import json

import pytest
import requests

from conftest import (
    FAKE_KEY, REQUEST_ID, FakeResponse, catalog_page, client, image_provider, job, submitted)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()


@pytest.fixture
def provider():
    return image_provider.DeapiImageGenProvider()


def ok_routes(http, *, path="/images/generations", final=None):
    http.add("GET", "/models", catalog_page())
    http.add("POST", path, submitted())
    http.add("GET", f"/jobs/{REQUEST_ID}", job("processing", progress=40), final or job("done"))
    return http


def test_text_to_image_happy_path(http, saved, provider):
    ok_routes(http)
    result = provider.generate("a lighthouse at dusk", "landscape", seed=11)
    assert result["success"] is True, result
    assert result["provider"] == "deapi" and result["modality"] == "text"
    assert result["model"] == "Flux_2_Klein_4B_BF16"
    assert result["image"].endswith("deapi_Flux_2_Klein_4B_BF16.bin")
    assert result["request_id"] == REQUEST_ID
    assert result["cost_usd"] == pytest.approx(0.0021) and result["cost_is_estimate"] is False
    post = http.posts()[0]
    assert post["path"] == "/images/generations"
    assert post["json"] == {"prompt": "a lighthouse at dusk", "model": "Flux_2_Klein_4B_BF16",
                            "width": 1024, "height": 672, "seed": 11, "steps": 4}
    assert post["files"] is None and post["allow_redirects"] is False
    assert saved == [("https://results.example/out.png", "deapi_Flux_2_Klein_4B_BF16")]


def test_dispatcher_model_is_used_when_it_is_a_deapi_slug(http, saved, provider):
    ok_routes(http)
    result = provider.generate("x", "portrait", model="Flux1schnell")
    assert result["model"] == "Flux1schnell" and "note" not in result
    assert http.posts()[0]["json"]["width"] == 512 and http.posts()[0]["json"]["height"] == 768


def test_foreign_model_id_falls_back_with_a_note(http, saved, provider):
    ok_routes(http)
    result = provider.generate("x", "square", model="gpt-image-2-medium")
    assert result["success"] and result["model"] == "Flux_2_Klein_4B_BF16"
    assert "gpt-image-2-medium" in result["note"]


def test_config_model_is_used(http, saved, provider, config):
    config({"image_gen": {"deapi": {"model": "gpt-image-2", "timeout": 60}}})
    ok_routes(http)
    result = provider.generate("x", "square")
    assert result["model"] == "gpt-image-2"
    assert http.posts()[0]["json"]["width"] == 1024


def test_edit_uploads_the_source_image(http, saved, provider):
    ok_routes(http, path="/images/edits")
    result = provider.generate("make it night", "square", image_url=PNG_DATA_URL, seed=3)
    assert result["success"] and result["modality"] == "image"
    post = http.posts()[0]
    assert post["path"] == "/images/edits" and post["json"] is None
    assert post["data"] == {"prompt": "make it night", "model": "Flux_2_Klein_4B_BF16", "seed": "3", "steps": "4"}
    assert post["files"] == [("image", ("source_0.png", PNG, "image/png"))]


def test_edit_with_text_only_model_switches_to_an_editor(http, saved, provider, config):
    config({"image_gen": {"deapi": {"model": "Flux1schnell"}}})
    ok_routes(http, path="/images/edits")
    result = provider.generate("edit", image_url=PNG_DATA_URL)
    assert result["model"] == "Flux_2_Klein_4B_BF16"
    assert "does not serve img2img" in result["note"]


def test_edit_clamps_references_to_the_model_limit(http, saved, provider):
    ok_routes(http, path="/images/edits")
    result = provider.generate("combine", image_url=PNG_DATA_URL, reference_image_urls=[PNG_DATA_URL] * 4)
    assert result["dropped_reference_images"] == 2
    files = http.posts()[0]["files"]
    assert [f[0] for f in files] == ["images[]"] * 3
    assert [f[1][0] for f in files] == ["source_0.png", "source_1.png", "source_2.png"]


def test_edit_from_local_file(http, saved, provider, tmp_path):
    path = tmp_path / "in.png"
    path.write_bytes(PNG)
    ok_routes(http, path="/images/edits")
    result = provider.generate("edit", image_url=str(path))
    assert result["success"], result
    assert http.posts()[0]["files"][0][1][1] == PNG


@pytest.mark.parametrize("source, error_type", [
    ("data:image/png;base64," + base64.b64encode(b"not an image").decode(), "invalid_image_url"),
    ("data:text/plain;base64,aGk=", "invalid_image_url"),
    ("/definitely/missing/file.png", "invalid_image_url"),
])
def test_bad_sources_fail_before_any_paid_call(http, provider, source, error_type):
    http.add("GET", "/models", catalog_page())
    result = provider.generate("edit", image_url=source)
    assert result["success"] is False and result["error_type"] == error_type
    assert http.posts() == []


def test_oversized_source_is_refused(http, provider, monkeypatch):
    monkeypatch.setattr(client, "MAX_SOURCE_BYTES", 32)
    http.add("GET", "/models", catalog_page())
    result = provider.generate("edit", image_url=PNG_DATA_URL)
    assert result["error_type"] == "source_too_large" and http.posts() == []


def test_missing_key_and_prompt(http, provider, monkeypatch):
    assert provider.generate("  ")["error_type"] == "invalid_argument"
    monkeypatch.delenv("DEAPI_API_KEY")
    result = provider.generate("x")
    assert result["error_type"] == "auth_required" and "DEAPI_API_KEY" in result["error"]
    assert http.calls == []


@pytest.mark.parametrize("response, error_type", [
    (FakeResponse(401, {"message": "Unauthenticated.", "code": "invalid_key"}), "auth_error"),
    (FakeResponse(402, {"message": "spend limit", "code": "key_limit_exceeded"}), "key_limit_exceeded"),
    (FakeResponse(422, {"message": "The balance field is invalid.", "code": "insufficient_balance",
                        "errors": {"balance": ["Insufficient balance."]}}), "insufficient_balance"),
    (FakeResponse(422, {"message": "The given data was invalid.",
                        "errors": {"width": ["The width must be at most 1536."]}}), "invalid_request"),
    (FakeResponse(429, {"message": "Too Many Requests"}, {"X-RateLimit-Type": "daily"}), "rate_limited"),
    (FakeResponse(403, {"message": "Account suspended"}), "forbidden"),
])
def test_submit_errors_are_mapped(http, provider, response, error_type):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/images/generations", response)
    result = provider.generate("x")
    assert result["success"] is False and result["error_type"] == error_type
    assert FAKE_KEY not in json.dumps(result)


def test_validation_details_reach_the_agent(http, provider):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/images/generations", FakeResponse(
        422, {"message": "The given data was invalid.", "errors": {"width": ["The width must be at most 1536."]}}))
    assert "width: The width must be at most 1536." in provider.generate("x")["error"]


def test_submit_5xx_is_never_resent(http, provider):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/images/generations", FakeResponse(502, None, text="bad gateway"))
    result = provider.generate("x")
    assert result["error_type"] == "unknown_state" and len(http.posts()) == 1


def test_dropped_connection_after_send_is_unknown_state(http, provider):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/images/generations", requests.ConnectionError("reset by peer"))
    result = provider.generate("x")
    assert result["error_type"] == "unknown_state" and len(http.posts()) == 1


def test_submit_429_with_short_retry_after_is_retried(http, saved, provider):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/images/generations",
             FakeResponse(429, {"message": "slow down"}, {"Retry-After": "1"}), submitted())
    http.add("GET", f"/jobs/{REQUEST_ID}", job("done"))
    assert provider.generate("x")["success"] is True
    assert len(http.posts()) == 2


def test_poll_survives_transient_errors(http, saved, provider):
    ok_routes(http)
    http.routes[("GET", f"/jobs/{REQUEST_ID}")] = [
        FakeResponse(503, None, text="unavailable"), requests.Timeout("slow"), job("pending"), job("done")]
    assert provider.generate("x")["success"] is True


def test_failed_job_reports_reason_and_refund(http, provider):
    ok_routes(http, final=job("error", error_code="WORKER_ERROR", error_reason="CAPACITY_UNAVAILABLE",
                              retryable=True, refunded=True))
    result = provider.generate("x")
    assert result["error_type"] == "job_failed"
    assert "CAPACITY_UNAVAILABLE" in result["error"] and "refunded" in result["error"]
    assert "may succeed" in result["error"]


def test_poll_deadline_reports_request_id_and_does_not_resubmit(http, provider, monkeypatch, config):
    config({"image_gen": {"deapi": {"timeout": 30}}})
    ticks = iter(range(0, 10_000, 20))
    monkeypatch.setattr(client, "_clock", lambda: next(ticks))
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/images/generations", submitted())
    http.add("GET", f"/jobs/{REQUEST_ID}", job("processing"))
    result = provider.generate("x")
    assert result["error_type"] == "timeout"
    assert REQUEST_ID in result["error"] and "do not resubmit" in result["error"]
    assert len(http.posts()) == 1


def test_done_without_result_url(http, provider):
    ok_routes(http, final=FakeResponse(200, {"data": {"status": "done", "result_url": None}}))
    assert provider.generate("x")["error_type"] == "empty_response"


def test_cache_failure_returns_the_bare_url(http, provider, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(image_provider, "save_url_image", boom)
    ok_routes(http)
    result = provider.generate("x")
    assert result["success"] and result["image"] == "https://results.example/out.png"


def test_catalog_failure_is_reported(http, provider):
    http.add("GET", "/models", FakeResponse(401, {"message": "Unauthenticated.", "code": "invalid_key"}))
    result = provider.generate("x")
    assert result["error_type"] == "auth_error" and http.posts() == []


class _FakeStreamResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeAsyncClient:
    """Stand-in for Hermes's SSRF-safe httpx.AsyncClient."""

    requested: list = []

    def __init__(self, chunks, **kwargs):
        self._chunks, self.kwargs = chunks, kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url):
        _FakeAsyncClient.requested.append((method, url, self.kwargs.get("follow_redirects")))
        chunks = self._chunks

        class _Ctx:
            async def __aenter__(self_inner):
                return _FakeStreamResponse(chunks)

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


@pytest.fixture
def remote(monkeypatch):
    import tools.url_safety as url_safety

    _FakeAsyncClient.requested = []
    state = {"safe": True, "chunks": [PNG[:10], PNG[10:]]}
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url: state["safe"])
    monkeypatch.setattr(url_safety, "create_ssrf_safe_async_client",
                        lambda **kw: _FakeAsyncClient(state["chunks"], **kw))
    return state


def test_edit_downloads_url_sources_through_the_ssrf_safe_client(http, saved, provider, remote):
    ok_routes(http, path="/images/edits")
    result = provider.generate("edit", image_url="https://images.example/cat.png")
    assert result["success"], result
    assert _FakeAsyncClient.requested == [("GET", "https://images.example/cat.png", True)]
    assert http.posts()[0]["files"] == [("image", ("source_0.png", PNG, "image/png"))]


def test_unsafe_source_url_is_refused(http, provider, remote):
    remote["safe"] = False
    http.add("GET", "/models", catalog_page())
    result = provider.generate("edit", image_url="http://169.254.169.254/latest")
    assert result["error_type"] == "invalid_image_url" and "SSRF" in result["error"]
    assert _FakeAsyncClient.requested == [] and http.posts() == []


def test_oversized_url_source_stops_streaming(http, provider, remote, monkeypatch):
    monkeypatch.setattr(client, "MAX_SOURCE_BYTES", 32)
    remote["chunks"] = [b"\x89PNG\r\n\x1a\n" + b"\x00" * 30, b"\x00" * 30]
    http.add("GET", "/models", catalog_page())
    result = provider.generate("edit", image_url="https://images.example/big.png")
    assert result["error_type"] == "source_too_large" and http.posts() == []


def test_local_non_image_file_is_never_uploaded(http, provider, tmp_path):
    secret = tmp_path / "credentials.txt"
    secret.write_text("token=abc\n")
    http.add("GET", "/models", catalog_page())
    result = provider.generate("edit", image_url=str(secret))
    assert result["error_type"] == "invalid_image_url" and http.posts() == []
    assert "token" not in json.dumps(result)
