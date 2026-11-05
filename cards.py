"""Model cards from ``GET /api/v2/models``: limits, defaults, picker rows and model choice.

deAPI publishes per-model limits (sizes, frames, fps, steps) and defaults. Generation requests
must fit them, so sizes are derived from the card instead of a hardcoded table, and model ids
always come from the live catalog of the user's plan.
"""

from __future__ import annotations

import logging
import math
import random
import re
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Auto-pick order among the plan's own-fleet models when the user has not chosen one. Patterns,
# not ids: a renamed or retired model just falls through to the next match.
PREFERENCES: dict[str, tuple[str, ...]] = {
    "txt2img": (r"klein", r"flux"),
    "img2img": (r"klein", r"qwen"),
    "txt2video": (r"ltx2_5", r"ltx2", r"ltx"),
    "img2video": (r"ltx2_5", r"ltx2", r"ltx"),
}


def _section(model: Optional[dict[str, Any]], name: str) -> dict[str, Any]:
    info = (model or {}).get("info")
    value = info.get(name) if isinstance(info, dict) else None
    return value if isinstance(value, dict) else {}


def limit(model: Optional[dict[str, Any]], name: str, fallback: Any = None) -> Any:
    value = _section(model, "limits").get(name)
    return fallback if value is None else value


def default(model: Optional[dict[str, Any]], name: str, fallback: Any = None) -> Any:
    value = _section(model, "defaults").get(name)
    return fallback if value is None else value


def feature(model: Optional[dict[str, Any]], name: str) -> bool:
    """True only when the card says the feature is supported."""
    return _section(model, "features").get(name) is True


def pinned(model: Optional[dict[str, Any]], name: str) -> Any:
    """The value a model fixes a parameter at (``min_<name> == max_<name>``), else None."""
    low, high = limit(model, f"min_{name}"), limit(model, f"max_{name}")
    if low is not None and high is not None and float(low) == float(high):
        return low
    return None


def clamp(model: Optional[dict[str, Any]], name: str, value: float) -> float:
    low, high = limit(model, f"min_{name}"), limit(model, f"max_{name}")
    if low is not None and value < low:
        value = low
    if high is not None and value > high:
        value = high
    return value


def is_partner(model: dict[str, Any]) -> bool:
    """Partner-served (external) models: billed by the partner's price, chosen only explicitly."""
    tags = [str(t).lower() for t in (model.get("tags") or [])]
    return model.get("status") == "test_model" or "external" in tags


def serves(model: dict[str, Any], types: Iterable[str]) -> bool:
    return bool(set(types) & set(model.get("inference_types") or []))


def find(models: list[dict[str, Any]], slug: Optional[str]) -> Optional[dict[str, Any]]:
    if not isinstance(slug, str) or not slug.strip():
        return None
    return next((m for m in models if m.get("slug") == slug.strip()), None)


def auto_pick(models: list[dict[str, Any]], inference_type: str) -> Optional[dict[str, Any]]:
    """Preferred own-fleet model for a type; any serving model when the plan has no own-fleet one."""
    pool = [m for m in models if serves(m, [inference_type])]
    standard = [m for m in pool if not is_partner(m)] or pool
    for pattern in PREFERENCES.get(inference_type, ()):
        for model in standard:
            haystack = f"{model.get('slug', '')} {model.get('name', '')}"
            if re.search(pattern, haystack, re.IGNORECASE):
                return model
    return standard[0] if standard else None


def choose(models: list[dict[str, Any]], inference_type: str,
           candidates: Iterable[Optional[str]]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """First candidate slug that serves ``inference_type``, else the auto pick.

    Returns ``(model, note)``; ``note`` explains a candidate that was skipped so the agent can
    tell the user (an unknown slug, or a model that does not serve this kind of request).
    """
    notes = []
    for slug in candidates:
        if not isinstance(slug, str) or not slug.strip():
            continue
        model = find(models, slug)
        if model is None:
            notes.append(f"model {slug.strip()!r} is not in this deAPI key's catalog")
            continue
        if serves(model, [inference_type]):
            return model, None
        notes.append(f"model {slug.strip()!r} does not serve {inference_type}")
    picked = auto_pick(models, inference_type)
    note = "; ".join(notes) + f"; used {picked['slug']}" if notes and picked else None
    return picked, note


# --------------------------------------------------------------------------- sizes

def _snap(value: float, step: int, minimum: Optional[float]) -> int:
    snapped = int(value) - int(value) % step
    floor = max(step, int(minimum or 0))
    if snapped < floor:
        snapped = int(math.ceil(floor / step) * step)
    return snapped


def fit_dimensions(model: Optional[dict[str, Any]], width: float, height: float) -> tuple[int, int]:
    """Scale (keeping the ratio where the card allows it) and snap a size to the model's limits."""
    w, h = float(width), float(height)
    max_ratio = limit(model, "max_ratio")
    if max_ratio:
        if w / h > max_ratio:
            w = h * max_ratio
        elif h / w > max_ratio:
            h = w * max_ratio
    min_w = limit(model, "min_width", limit(model, "min_side"))
    max_w = limit(model, "max_width", limit(model, "max_side"))
    min_h = limit(model, "min_height", limit(model, "min_side"))
    max_h = limit(model, "max_height", limit(model, "max_side"))
    scale = 1.0
    max_pixels = limit(model, "max_pixels")
    if max_pixels and w * h > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / (w * h)))
    if max_w and w * scale > max_w:
        scale = min(scale, max_w / w)
    if max_h and h * scale > max_h:
        scale = min(scale, max_h / h)
    w, h = w * scale, h * scale
    grow = 1.0
    if min_w and w < min_w:
        grow = max(grow, min_w / w)
    if min_h and h < min_h:
        grow = max(grow, min_h / h)
    w, h = w * grow, h * grow
    # A ratio the card cannot hold at all is clamped per side as a last resort.
    if max_w:
        w = min(w, max_w)
    if max_h:
        h = min(h, max_h)
    step = limit(model, "dimension_multiple") or limit(model, "resolution_step")
    if step:
        return _snap(w, int(step), min_w), _snap(h, int(step), min_h)
    return int(round(w)), int(round(h))


def size_for_ratio(model: Optional[dict[str, Any]], ratio_w: float, ratio_h: float, *,
                   short_side: Optional[int] = None) -> tuple[int, int]:
    """A size with the given ratio: ``short_side`` px on the short side, else the card's default
    long side (``max(defaults.width, defaults.height)``, 1024 when absent), fitted to the card."""
    if short_side:
        if ratio_w >= ratio_h:
            h = float(short_side)
            w = h * ratio_w / ratio_h
        else:
            w = float(short_side)
            h = w * ratio_h / ratio_w
    else:
        long_side = float(max(default(model, "width", 0) or 0, default(model, "height", 0) or 0) or 1024)
        if ratio_w >= ratio_h:
            w, h = long_side, long_side * ratio_h / ratio_w
        else:
            h, w = long_side, long_side * ratio_w / ratio_h
    return fit_dimensions(model, w, h)


# --------------------------------------------------------------------------- common params

def apply_sampling(model: dict[str, Any], body: dict[str, Any], *, seed: Any = None) -> dict[str, Any]:
    """Fill ``seed`` (required everywhere) plus ``steps`` / ``guidance`` where the card asks for them."""
    if isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2 ** 32:
        body["seed"] = seed
    else:
        body["seed"] = random.randint(0, 2 ** 31 - 1)
    if feature(model, "supports_steps"):
        steps = default(model, "steps", pinned(model, "steps"))
        if steps is not None:
            body["steps"] = int(clamp(model, "steps", int(steps)))
    if feature(model, "supports_guidance"):
        guidance = default(model, "guidance", pinned(model, "guidance"))
        if guidance is not None:
            body["guidance"] = float(clamp(model, "guidance", float(guidance)))
    return body


def as_form(body: dict[str, Any]) -> dict[str, str]:
    """Multipart form fields (deAPI's edit and animation endpoints are multipart)."""
    return {k: (("true" if v else "false") if isinstance(v, bool) else str(v))
            for k, v in body.items() if v is not None}


def display_name(model: dict[str, Any]) -> str:
    name = model.get("name")
    return name if isinstance(name, str) and name.strip() else str(model.get("slug"))


def sort_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Own-fleet models first (they are the cheap ones), partner-served after; stable otherwise."""
    return sorted(models, key=lambda m: is_partner(m))
