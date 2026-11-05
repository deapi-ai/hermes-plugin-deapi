"""Registration, picker surface and capabilities."""

from __future__ import annotations

from conftest import (
    CATALOG, FakeResponse, catalog_page, client, image_provider, plugin, video_provider)


class FakeCtx:
    def __init__(self) -> None:
        self.images, self.videos = [], []

    def register_image_gen_provider(self, provider) -> None:
        self.images.append(provider)

    def register_video_gen_provider(self, provider) -> None:
        self.videos.append(provider)


def test_register_registers_both_backends():
    ctx = FakeCtx()
    plugin.register(ctx)
    assert [type(p).__name__ for p in ctx.images] == ["DeapiImageGenProvider"]
    assert [type(p).__name__ for p in ctx.videos] == ["DeapiVideoGenProvider"]
    assert ctx.images[0].name == ctx.videos[0].name == "deapi"
    assert ctx.images[0].display_name == "deAPI"


def test_is_available_follows_the_key_and_never_hits_the_network(http, monkeypatch):
    image, video = image_provider.DeapiImageGenProvider(), video_provider.DeapiVideoGenProvider()
    assert image.is_available() and video.is_available()
    monkeypatch.delenv("DEAPI_API_KEY")
    assert not image.is_available() and not video.is_available()
    assert http.calls == []


def test_setup_schema_asks_for_one_key():
    for provider in (image_provider.DeapiImageGenProvider(), video_provider.DeapiVideoGenProvider()):
        schema = provider.get_setup_schema()
        assert schema["badge"] == "paid"
        assert [e["key"] for e in schema["env_vars"]] == ["DEAPI_API_KEY"]
        assert schema["env_vars"][0]["url"].startswith("https://")


def test_image_picker_lists_image_models_own_fleet_first(http):
    http.add("GET", "/models", catalog_page())
    rows = image_provider.DeapiImageGenProvider().list_models()
    ids = [r["id"] for r in rows]
    assert set(ids) == {"Flux1schnell", "Flux_2_Klein_4B_BF16", "QwenImageEdit_Plus_NF4", "gpt-image-2"}
    assert ids[-1] == "gpt-image-2"  # partner-served model last
    assert "partner" in rows[-1]["strengths"]
    klein = next(r for r in rows if r["id"] == "Flux_2_Klein_4B_BF16")
    assert klein["modalities"] == ["text", "image"]
    # Catalog is cached per key: a second paint does not refetch.
    image_provider.DeapiImageGenProvider().list_models()
    assert len(http.calls) == 1
    assert http.calls[0]["headers"]["Authorization"] == "Bearer test-key-not-real"
    assert http.calls[0]["headers"]["User-Agent"].startswith("hermes-plugin-deapi/")


def test_video_picker_lists_video_models(http):
    http.add("GET", "/models", catalog_page())
    rows = video_provider.DeapiVideoGenProvider().list_models()
    ids = [r["id"] for r in rows]
    assert ids == ["Ltx2_5_22B_Dist_INT8", "Ltxv_13B_0_9_8_Distilled_FP8", "seedance-2.5"]
    assert rows[0]["modalities"] == ["text", "image"]
    assert "up to 10s at 24 fps" in rows[0]["strengths"]


def test_default_models_prefer_own_fleet_and_honor_config(http, config):
    http.add("GET", "/models", catalog_page())
    assert image_provider.DeapiImageGenProvider().default_model() == "Flux_2_Klein_4B_BF16"
    assert video_provider.DeapiVideoGenProvider().default_model() == "Ltx2_5_22B_Dist_INT8"
    config({"image_gen": {"deapi": {"model": "Flux1schnell"}},
            "video_gen": {"deapi": {"model": "Ltxv_13B_0_9_8_Distilled_FP8"}}})
    assert image_provider.DeapiImageGenProvider().default_model() == "Flux1schnell"
    assert video_provider.DeapiVideoGenProvider().default_model() == "Ltxv_13B_0_9_8_Distilled_FP8"


def test_picker_is_empty_without_key_or_catalog(http, monkeypatch):
    http.add("GET", "/models", FakeResponse(503, {"message": "down"}))
    assert image_provider.DeapiImageGenProvider().list_models() == []
    assert video_provider.DeapiVideoGenProvider().default_model() is None
    monkeypatch.delenv("DEAPI_API_KEY")
    assert image_provider.DeapiImageGenProvider().list_models() == []


def test_capabilities_use_only_the_cache(http):
    image, video = image_provider.DeapiImageGenProvider(), video_provider.DeapiVideoGenProvider()
    # Before any fetch: static fallback, no network.
    assert image.capabilities() == {"modalities": ["text", "image"], "max_reference_images": 2}
    assert video.capabilities()["modalities"] == ["text", "image"]
    assert http.calls == []
    http.add("GET", "/models", catalog_page())
    client.fetch_models(client.api_key())
    assert image.capabilities() == {"modalities": ["text", "image"], "max_reference_images": 2}
    caps = video.capabilities()
    assert caps["max_duration"] == 10 and caps["min_duration"] == 3
    assert caps["supports_negative_prompt"] is False and caps["supports_seed"] is True


def test_capabilities_text_only_when_plan_has_no_editing_or_animation(http):
    text_only = [m for m in CATALOG if m["slug"] == "Flux1schnell"]
    http.add("GET", "/models", catalog_page(text_only))
    client.fetch_models(client.api_key())
    assert image_provider.DeapiImageGenProvider().capabilities() == {
        "modalities": ["text"], "max_reference_images": 0}


def test_catalog_pagination(http):
    first = [m for m in CATALOG if m["slug"] == "Flux1schnell"]
    second = [m for m in CATALOG if m["slug"] == "Kokoro"]
    http.add("GET", "/models",
             FakeResponse(200, {"data": first, "meta": {"last_page": 2}}),
             FakeResponse(200, {"data": second, "meta": {"last_page": 2}}))
    models = client.fetch_models(client.api_key())
    assert [m["slug"] for m in models] == ["Flux1schnell", "Kokoro"]
    assert [c["params"]["page"] for c in http.calls] == [1, 2]
