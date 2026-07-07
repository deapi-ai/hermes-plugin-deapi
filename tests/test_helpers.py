"""Unit tests for deapi plugin helpers (no network, no Hermes runtime).

Loads each provider module in isolation with a stubbed agent.* dependency so
the pure helper functions can be exercised.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _stub_agent_modules():
    """Provide minimal agent.image_gen_provider / agent.video_gen_provider so
    the plugin __init__ modules import without the full Hermes tree."""
    agent = types.ModuleType("agent")
    sys.modules.setdefault("agent", agent)

    def _mk(name, extra):
        mod = types.ModuleType(name)
        class _Base:  # stand-in ABC
            pass
        mod.ImageGenProvider = _Base
        mod.VideoGenProvider = _Base
        mod.error_response = lambda **kw: {"success": False, **kw}
        mod.success_response = lambda **kw: {"success": True, **kw}
        mod.resolve_aspect_ratio = lambda r: r if r in ("landscape", "square", "portrait") else "landscape"
        mod.save_url_image = lambda url, **kw: Path("/tmp/x.png")
        mod.save_bytes_video = lambda raw, **kw: Path("/tmp/x.mp4")
        mod.DEFAULT_ASPECT_RATIO = "16:9"
        mod.DEFAULT_RESOLUTION = "720p"
        for k, v in extra.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    _mk("agent.image_gen_provider", {})
    _mk("agent.video_gen_provider", {})


def _load(dirname):
    _stub_agent_modules()
    path = ROOT / dirname / "__init__.py"
    spec = importlib.util.spec_from_file_location("deapi_%s" % dirname.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def img():
    return _load("deapi-image")


@pytest.fixture()
def vid():
    return _load("deapi-video")


# --- key normalization -----------------------------------------------------

def test_api_key_strips_prefix(img, monkeypatch):
    monkeypatch.setenv("DEAPI_API_KEY", "dpn-sk-123|abc")
    assert img._api_key() == "123|abc"


def test_api_key_raw_untouched(img, monkeypatch):
    monkeypatch.setenv("DEAPI_API_KEY", "123|abc")
    assert img._api_key() == "123|abc"


def test_api_key_empty(img, monkeypatch):
    monkeypatch.delenv("DEAPI_API_KEY", raising=False)
    assert img._api_key() == ""


# --- base URL lock ---------------------------------------------------------

def test_base_url_default(img, monkeypatch):
    monkeypatch.delenv("DEAPI_BASE_URL", raising=False)
    assert img._api_base() == "https://api.deapi.ai"


def test_base_url_custom_ignored_without_optin(img, monkeypatch):
    monkeypatch.setenv("DEAPI_BASE_URL", "https://evil.example.com")
    monkeypatch.delenv("DEAPI_ALLOW_CUSTOM_BASE_URL", raising=False)
    assert img._api_base() == "https://api.deapi.ai"


def test_base_url_custom_allowed_with_optin_and_https(img, monkeypatch):
    monkeypatch.setenv("DEAPI_BASE_URL", "https://staging.deapi.ai")
    monkeypatch.setenv("DEAPI_ALLOW_CUSTOM_BASE_URL", "1")
    assert img._api_base() == "https://staging.deapi.ai"


def test_base_url_custom_http_rejected_even_with_optin(img, monkeypatch):
    monkeypatch.setenv("DEAPI_BASE_URL", "http://staging.deapi.ai")
    monkeypatch.setenv("DEAPI_ALLOW_CUSTOM_BASE_URL", "1")
    assert img._api_base() == "https://api.deapi.ai"


# --- clamp -----------------------------------------------------------------

def test_clamp(img):
    lim = {"min_width": 256, "max_width": 1024}
    assert img._clamp(2000, lim, "min_width", "max_width") == 1024
    assert img._clamp(64, lim, "min_width", "max_width") == 256
    assert img._clamp(512, lim, "min_width", "max_width") == 512


def test_clamp_to_multiple(vid):
    lim = {"min_width": 256, "max_width": 768}
    v = vid._clamp_to_multiple(800, lim, "min_width", "max_width", step=16)
    assert v <= 768 and v % 16 == 0


# --- SSRF guard ------------------------------------------------------------

def test_host_public_blocks_loopback(vid):
    assert vid._host_is_public("localhost") is False


def test_host_public_blocks_metadata_ip(vid, monkeypatch):
    monkeypatch.setattr(vid.socket, "getaddrinfo",
                        lambda *a, **k: [(0, 0, 0, "", ("169.254.169.254", 0))])
    assert vid._host_is_public("metadata.local") is False


def test_host_public_allows_global(vid, monkeypatch):
    monkeypatch.setattr(vid.socket, "getaddrinfo",
                        lambda *a, **k: [(0, 0, 0, "", ("93.184.216.34", 0))])
    assert vid._host_is_public("example.com") is True


def test_local_image_outside_cache_rejected(vid, monkeypatch):
    monkeypatch.setattr(vid, "_hermes_cache_root", lambda: "/home/u/.hermes/cache")
    with pytest.raises(vid._ImageInputError):
        vid._local_image_path("/etc/passwd")


# --- modality-aware default model -----------------------------------------

def test_video_default_model_filters_by_modality(vid, monkeypatch):
    vid.DeapiVideoGenProvider._models_cache = [
        {"slug": "TextOnly", "inference_types": ["txt2video"]},
        {"slug": "ImgOnly", "inference_types": ["img2video"]},
    ]
    p = vid.DeapiVideoGenProvider()
    assert p.default_model("img2video") == "ImgOnly"
    assert p.default_model("txt2video") == "TextOnly"
    vid.DeapiVideoGenProvider._models_cache = None
