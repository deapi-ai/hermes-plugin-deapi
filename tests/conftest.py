"""Test setup: a Hermes source tree on ``sys.path`` (the providers subclass its ABCs), the plugin
loaded as a package the way Hermes loads it, and a fake HTTP layer. No test reaches the network."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERMES_DIR = Path(os.environ.get("HERMES_AGENT_DIR", "/hermes"))
if HERMES_DIR.is_dir() and str(HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(HERMES_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PACKAGE = "deapi_plugin_under_test"


def _load_plugin():
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    spec = importlib.util.spec_from_file_location(
        PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


plugin = _load_plugin()
client = importlib.import_module(f"{PACKAGE}.client")
cards = importlib.import_module(f"{PACKAGE}.cards")
settings = importlib.import_module(f"{PACKAGE}.settings")
image_provider = importlib.import_module(f"{PACKAGE}.image_provider")
video_provider = importlib.import_module(f"{PACKAGE}.video_provider")

from samples import CATALOG  # noqa: E402

FAKE_KEY = "test-key-not-real"
REQUEST_ID = "c08a339c-73e5-4d67-a4d5-231302fbff9a"


class FakeResponse:
    def __init__(self, status: int = 200, body: Any = None, headers: Optional[dict] = None,
                 text: Optional[str] = None) -> None:
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


class FakeHTTP:
    """Stand-in for the ``requests`` module: routes ``(METHOD, path)`` to queued responses."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.routes: dict[tuple[str, str], list[Any]] = {}

    def add(self, method: str, path: str, *responses: Any) -> "FakeHTTP":
        self.routes.setdefault((method, path), []).extend(responses)
        return self

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        assert url.startswith(client.API_BASE), url
        path = url[len(client.API_BASE):]
        self.calls.append({"method": method, "path": path, **kwargs})
        queue = self.routes.get((method, path))
        if not queue:
            raise AssertionError(f"unexpected {method} {path}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(**kwargs)
        return item

    def posts(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["method"] == "POST"]


def catalog_page(models: Optional[list] = None) -> FakeResponse:
    return FakeResponse(200, {"data": copy.deepcopy(models if models is not None else CATALOG),
                              "meta": {"current_page": 1, "last_page": 1}})


def submitted(request_id: str = REQUEST_ID) -> FakeResponse:
    return FakeResponse(200, {"data": {"request_id": request_id}})


def job(status: str = "done", **fields: Any) -> FakeResponse:
    data = {"status": status, **fields}
    if status == "done":
        data.setdefault("result_url", "https://results.example/out.png")
        data.setdefault("price", {"amount": 0.0021, "is_estimated": False})
    return FakeResponse(200, {"data": data})


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Own HERMES_HOME, a fake key, empty catalog cache, no sleeping, no config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setenv("DEAPI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(client, "_sleep", lambda seconds: None)
    monkeypatch.setattr(settings, "load_section", lambda kind: {})
    client.clear_cache()
    yield
    client.clear_cache()


@pytest.fixture
def http(monkeypatch) -> FakeHTTP:
    fake = FakeHTTP()
    monkeypatch.setattr(client, "_http", fake)
    return fake


@pytest.fixture
def config(monkeypatch) -> Callable[[dict], None]:
    """Set the ``image_gen`` / ``video_gen`` sections of config.yaml for one test."""
    def apply(sections: dict) -> None:
        monkeypatch.setattr(settings, "load_section", lambda kind: sections.get(kind, {}))
    return apply


@pytest.fixture
def saved(monkeypatch, tmp_path):
    """Make result caching succeed without a download; returns the recorded calls."""
    calls: list[tuple[str, str]] = []

    def fake_save(url: str, *, prefix: str, **_: Any) -> Path:
        calls.append((url, prefix))
        return tmp_path / f"{prefix}.bin"

    monkeypatch.setattr(image_provider, "save_url_image", fake_save)
    monkeypatch.setattr(video_provider, "save_url_video", fake_save)
    return calls
