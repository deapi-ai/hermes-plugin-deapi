"""deAPI ``video_generate`` backend: text-to-video (``/videos/generations``) and image-to-video
(``/videos/animations``, the image becomes the first frame) on the native API v2.

Hermes passes duration (s), aspect ratio and resolution; deAPI takes width/height, frames and
fps, bounded per model. Both are derived from the model card: fps is the card's (deAPI models
fix it), frames = duration × fps clamped to the card, the size keeps the ratio and the
resolution's short side, scaled down to the model's pixel budget.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO, DEFAULT_RESOLUTION, VideoGenProvider, error_response, save_url_video,
    success_response)

from . import cards, client, settings

logger = logging.getLogger(__name__)

PROVIDER = "deapi"
KIND = "video_gen"
DEFAULT_TIMEOUT_S = 900.0
ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"]
RESOLUTIONS = ["480p", "540p", "720p", "768p"]
_FALLBACK_FPS = 24
_FALLBACK_CAPABILITIES: dict[str, Any] = {
    "modalities": ["text", "image"], "aspect_ratios": ASPECT_RATIOS, "resolutions": RESOLUTIONS,
    "max_duration": 10, "min_duration": 2, "supports_audio": False, "supports_negative_prompt": False,
    "supports_seed": True, "supports_upscale": False, "max_reference_images": 0,
}


def parse_ratio(value: Optional[str]) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d{1,2})\s*:\s*(\d{1,2})\s*", value or "")
    if match and int(match.group(1)) > 0 and int(match.group(2)) > 0:
        return int(match.group(1)), int(match.group(2))
    return 16, 9


def parse_resolution(value: Optional[str]) -> int:
    match = re.fullmatch(r"\s*(\d{3,4})\s*p?\s*", (value or "").lower())
    return int(match.group(1)) if match else int(DEFAULT_RESOLUTION.rstrip("p"))


def model_fps(model: dict[str, Any]) -> int:
    fps = cards.default(model, "fps", cards.pinned(model, "fps")) or cards.limit(model, "min_fps") or _FALLBACK_FPS
    return int(fps)


def frames_for(model: dict[str, Any], duration: Optional[int], fps: int) -> int:
    if duration and duration > 0:
        frames = int(round(duration * fps))
    else:
        frames = int(cards.default(model, "frames", 5 * fps))
    return int(cards.clamp(model, "frames", frames))


def _cache_prefix(slug: str) -> str:
    return "deapi_" + re.sub(r"[^A-Za-z0-9_.-]", "_", slug)


class DeapiVideoGenProvider(VideoGenProvider):
    """deAPI text-to-video and image-to-video."""

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
            "tag": "LTX-2.5, MiniMax, … — live catalog from api.deapi.ai; "
                   "text-to-video & image-to-video",
            "env_vars": [{"key": client.KEY_ENV, "prompt": "deAPI API key", "url": client.KEY_URL}],
        }

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

    def list_models(self) -> list[dict[str, Any]]:
        rows = []
        for model in cards.sort_rows([m for m in self._models() if cards.serves(m, ("txt2video", "img2video"))]):
            types = model.get("inference_types") or []
            fps = model_fps(model)
            max_frames = cards.limit(model, "max_frames")
            strengths = []
            if max_frames:
                strengths.append(f"up to {int(max_frames) // fps}s at {fps} fps")
            if cards.limit(model, "max_width") and cards.limit(model, "max_height"):
                strengths.append(f"max {cards.limit(model, 'max_width')}x{cards.limit(model, 'max_height')}")
            if cards.is_partner(model):
                strengths.append("partner-served (priced by the partner)")
            rows.append({
                "id": model["slug"], "display": cards.display_name(model), "strengths": " · ".join(strengths),
                "modalities": [m for t, m in (("txt2video", "text"), ("img2video", "image")) if t in types],
            })
        return rows

    def default_model(self) -> Optional[str]:
        # The video tool passes ``video_gen.model`` or this value as ``model``, so the
        # provider-scoped ``video_gen.deapi.model`` has to be honored here.
        model, _ = cards.choose(self._models(), "txt2video", [settings.configured_model(KIND)])
        return model["slug"] if model else None

    def capabilities(self) -> dict[str, Any]:
        """From the cached catalog only; no network on the schema path."""
        models = client.cached_models(client.api_key())
        if not models:
            return dict(_FALLBACK_CAPABILITIES)
        model, _ = cards.choose(models, "txt2video", [settings.configured_model(KIND)])
        if model is None:
            return dict(_FALLBACK_CAPABILITIES)
        fps = model_fps(model)
        caps = dict(_FALLBACK_CAPABILITIES)
        caps["modalities"] = ["text", "image"] if any(cards.serves(m, ["img2video"]) for m in models) else ["text"]
        if cards.limit(model, "max_frames"):
            caps["max_duration"] = max(1, int(cards.limit(model, "max_frames")) // fps)
        if cards.limit(model, "min_frames"):
            caps["min_duration"] = max(1, math.ceil(int(cards.limit(model, "min_frames")) / fps))
        caps["supports_negative_prompt"] = cards.feature(model, "supports_negative_prompt")
        return caps

    # ------------------------------------------------------------------ generation

    def generate(
        self, prompt: str, *, model: Optional[str] = None, image_url: Optional[str] = None,
        reference_image_urls: Optional[list[str]] = None, duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO, resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None, audio: Optional[bool] = None,
        seed: Optional[int] = None, **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        ratio_w, ratio_h = parse_ratio(aspect_ratio)
        aspect = f"{ratio_w}:{ratio_h}"
        model_slug = ""

        def fail(error: str, error_type: str) -> dict[str, Any]:
            return error_response(error=error, error_type=error_type, provider=PROVIDER,
                                  model=model_slug, prompt=prompt, aspect_ratio=aspect)

        if not prompt:
            return fail("prompt is required", "invalid_request")
        key = client.api_key()
        if not key:
            return fail(f"{client.KEY_ENV} not set. Run `hermes tools` → Video Generation → deAPI "
                        f"to configure it (keys at {client.KEY_URL}).", "missing_credentials")
        image_url = image_url.strip() if isinstance(image_url, str) and image_url.strip() else None
        modality = "image" if image_url else "text"
        inference_type = "img2video" if image_url else "txt2video"

        try:
            models = client.fetch_models(key)
        except client.DeapiError as exc:
            return fail(f"Could not load the deAPI model catalog: {exc}", exc.error_type)
        card, note = cards.choose(models, inference_type, [model, settings.configured_model(KIND)])
        if card is None:
            return fail(f"No deAPI model for {inference_type} is available on this key's plan.", "no_model")
        model_slug = card["slug"]

        fps = model_fps(card)
        frames = frames_for(card, duration, fps)
        width, height = cards.size_for_ratio(card, ratio_w, ratio_h, short_side=parse_resolution(resolution))
        body: dict[str, Any] = {"prompt": prompt, "model": model_slug, "width": width, "height": height,
                                "frames": frames, "fps": fps}
        cards.apply_sampling(card, body, seed=seed)
        extra: dict[str, Any] = {"width": width, "height": height, "frames": frames, "fps": fps,
                                 "seed": body["seed"]}
        if note:
            extra["note"] = note
        if negative_prompt and negative_prompt.strip():
            if cards.feature(card, "supports_negative_prompt"):
                body["negative_prompt"] = negative_prompt.strip()
            else:
                extra["negative_prompt_ignored"] = True

        try:
            if image_url:
                files = [("first_frame_image", client.load_source_image(image_url))]
                request_id = client.submit(key, "/videos/animations", data=cards.as_form(body), files=files)
            else:
                request_id = client.submit(key, "/videos/generations", json_body=body)
            extra["request_id"] = request_id
            job = client.wait_for_job(key, request_id,
                                      timeout_s=settings.timeout_seconds(KIND, DEFAULT_TIMEOUT_S),
                                      initial_interval=3.0, max_interval=10.0)
        except client.DeapiError as exc:
            return fail(str(exc), exc.error_type)

        result_url = job.get("result_url")
        if not isinstance(result_url, str) or not result_url.strip():
            return fail(f"deAPI job {request_id} finished without a result URL.", "empty_response")
        extra.update(client.job_cost(job))
        try:
            video_ref = str(save_url_video(result_url, prefix=_cache_prefix(model_slug)))
        except Exception as exc:
            logger.debug("deAPI: caching the video failed (%s); returning the URL", type(exc).__name__)
            video_ref = result_url
        return success_response(video=video_ref, model=model_slug, prompt=prompt, modality=modality,
                                aspect_ratio=aspect, duration=max(1, round(frames / fps)),
                                provider=PROVIDER, extra=extra)
