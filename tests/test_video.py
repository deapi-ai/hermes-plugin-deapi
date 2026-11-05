"""video_generate backend: frames/fps/size mapping, image-to-video upload, errors."""

from __future__ import annotations

import base64

import pytest

from conftest import REQUEST_ID, catalog_page, job, submitted, video_provider

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()


@pytest.fixture
def provider():
    return video_provider.DeapiVideoGenProvider()


def ok_routes(http, path="/videos/generations"):
    http.add("GET", "/models", catalog_page())
    http.add("POST", path, submitted())
    http.add("GET", f"/jobs/{REQUEST_ID}", job("processing"), job("done", result_url="https://results.example/v.mp4"))


def test_text_to_video_maps_duration_and_resolution(http, saved, provider):
    ok_routes(http)
    # The tool passes video_gen.model or default_model() as ``model``.
    result = provider.generate("a fox in snow", model=provider.default_model(), duration=5,
                               aspect_ratio="16:9", resolution="720p", seed=5)
    assert result["success"] is True, result
    assert result["model"] == "Ltx2_5_22B_Dist_INT8" and result["modality"] == "text"
    assert result["duration"] == 5 and result["aspect_ratio"] == "16:9"
    assert result["video"].endswith("deapi_Ltx2_5_22B_Dist_INT8.bin")
    assert result["request_id"] == REQUEST_ID and "cost_usd" in result
    assert http.posts()[0]["json"] == {"prompt": "a fox in snow", "model": "Ltx2_5_22B_Dist_INT8",
                                       "width": 1280, "height": 704, "frames": 120, "fps": 24, "seed": 5}
    assert saved == [("https://results.example/v.mp4", "deapi_Ltx2_5_22B_Dist_INT8")]


def test_defaults_without_duration(http, saved, provider):
    ok_routes(http)
    result = provider.generate("x")
    body = http.posts()[0]["json"]
    assert body["frames"] == 120 and body["fps"] == 24  # card defaults
    assert result["duration"] == 5


def test_duration_is_clamped_to_the_card(http, saved, provider):
    ok_routes(http)
    provider.generate("x", duration=60)
    assert http.posts()[0]["json"]["frames"] == 241
    http.calls.clear()
    provider.generate("x", duration=1)
    assert http.posts()[0]["json"]["frames"] == 49


def test_image_to_video_uploads_the_first_frame(http, saved, provider):
    ok_routes(http, "/videos/animations")
    result = provider.generate("animate it", image_url=PNG_DATA_URL, aspect_ratio="9:16",
                               resolution="480p", seed=9)
    assert result["success"] and result["modality"] == "image"
    post = http.posts()[0]
    assert post["path"] == "/videos/animations"
    assert post["files"] == [("first_frame_image", ("source_0.png", PNG, "image/png"))]
    assert post["data"] == {"prompt": "animate it", "model": "Ltx2_5_22B_Dist_INT8", "width": "512",
                            "height": "896", "frames": "120", "fps": "24", "seed": "9"}


def test_negative_prompt_only_where_supported(http, saved, provider):
    ok_routes(http)
    result = provider.generate("x", negative_prompt="blur")
    assert "negative_prompt" not in http.posts()[0]["json"] and result["negative_prompt_ignored"] is True
    http.calls.clear()
    result = provider.generate("x", model="Ltxv_13B_0_9_8_Distilled_FP8", negative_prompt="blur")
    body = http.posts()[0]["json"]
    assert body["negative_prompt"] == "blur" and body["fps"] == 30 and body["steps"] == 1
    assert body["width"] <= 768 and body["height"] <= 768


def test_foreign_video_model_falls_back(http, saved, provider):
    ok_routes(http)
    result = provider.generate("x", model="veo-3.1")
    assert result["model"] == "Ltx2_5_22B_Dist_INT8" and "veo-3.1" in result["note"]


def test_unparseable_ratio_and_resolution_use_defaults(http, saved, provider):
    ok_routes(http)
    result = provider.generate("x", aspect_ratio="wide", resolution="hd")
    body = http.posts()[0]["json"]
    assert result["aspect_ratio"] == "16:9" and (body["width"], body["height"]) == (1280, 704)


def test_missing_key_and_prompt(http, provider, monkeypatch):
    assert provider.generate("")["error_type"] == "invalid_request"
    monkeypatch.delenv("DEAPI_API_KEY")
    assert provider.generate("x")["error_type"] == "missing_credentials"
    assert http.calls == []


def test_failed_video_job(http, provider):
    http.add("GET", "/models", catalog_page())
    http.add("POST", "/videos/generations", submitted())
    http.add("GET", f"/jobs/{REQUEST_ID}", job("error", error_reason="INPUT_TOO_LARGE", retryable=False))
    result = provider.generate("x")
    assert result["error_type"] == "job_failed" and "will not help" in result["error"]
