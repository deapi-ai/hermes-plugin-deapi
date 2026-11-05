"""Non-secret settings from config.yaml: ``image_gen.deapi.*`` and ``video_gen.deapi.*``.

Only the API key lives in the environment (``DEAPI_API_KEY``); model and timeout are config keys.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def load_section(kind: str) -> dict[str, Any]:
    """``<kind>`` (``image_gen`` / ``video_gen``) top-level section, ``{}`` on any failure."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:
        logger.debug("Could not load %s config: %s", kind, exc)
        return {}
    section = cfg.get(kind) if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def deapi_section(kind: str) -> dict[str, Any]:
    sub = load_section(kind).get("deapi")
    return sub if isinstance(sub, dict) else {}


def configured_model(kind: str) -> Optional[str]:
    value = deapi_section(kind).get("model")
    return value.strip() if isinstance(value, str) and value.strip() else None


def timeout_seconds(kind: str, fallback: float) -> float:
    """``<kind>.deapi.timeout`` in seconds, bounded to 30..3600; ``fallback`` when unset/invalid."""
    value = deapi_section(kind).get("timeout")
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(max(seconds, 30.0), 3600.0)
