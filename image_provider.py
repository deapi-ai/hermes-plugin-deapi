"""deAPI ``image_generate`` backend: text-to-image (``/images/generations``) and image editing
(``/images/edits``) on the native API v2, with models from the key's live catalog.

Model choice: the ``model`` the dispatcher forwards (``image_gen.model``) when it is a deAPI
slug → ``image_gen.deapi.model`` → an own-fleet model picked from the catalog. An edit request
with a text-only model falls back to a catalog model that edits, and says so in ``note``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO, ImageGenProvider, error_response, normalize_reference_images,
    resolve_aspect_ratio, save_url_image, success_response)

from . import cards, client, settings

logger = logging.getLogger(__name__)

PROVIDER = "deapi"
KIND = "image_gen"
DEFAULT_TIMEOUT_S = 300.0
# Hermes' three semantic aspects → width:height.
ASPECTS = {"landscape": (3, 2), "square": (1, 1), "portrait": (2, 3)}
# Picker/schema fallback before the catalog has been fetched once.
_FALLBACK_CAPABILITIES = {"modalities": ["text", "image"], "max_reference_images": 2}


def _cache_prefix(slug: str) -> str:
    return "deapi_" + re.sub(r"[^A-Za-z0-9_.-]", "_", slug)


class DeapiImageGenProvider(ImageGenProvider):
    """deAPI image generation and editing."""

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
            "tag": "FLUX.2 Klein, Z-Image, Qwen Image Edit, … — live catalog from api.deapi.ai; "
                   "text-to-image & image editing",
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
        for model in cards.sort_rows([m for m in self._models() if cards.serves(m, ("txt2img", "img2img"))]):
            types = model.get("inference_types") or []
            uses = [label for t, label in (("txt2img", "text-to-image"), ("img2img", "editing")) if t in types]
            strengths = ", ".join(uses)
            if cards.is_partner(model):
                strengths += " · partner-served (priced by the partner)"
            rows.append({
                "id": model["slug"], "display": cards.display_name(model), "strengths": strengths,
                "modalities": ["text", "image"] if "img2img" in types else ["text"],
            })
        return rows

    def default_model(self) -> Optional[str]:
        model, _ = cards.choose(self._models(), "txt2img", [settings.configured_model(KIND)])
        return model["slug"] if model else None

    def capabilities(self) -> dict[str, Any]:
        """From the cached catalog only (the schema builder calls this often; no network here)."""
        models = client.cached_models(client.api_key())
        if not models:
            return dict(_FALLBACK_CAPABILITIES)
        editor, _ = cards.choose(models, "img2img", [settings.configured_model(KIND)])
        if editor is None:
            return {"modalities": ["text"], "max_reference_images": 0}
        max_inputs = int(cards.limit(editor, "max_input_images", 1) or 1)
        return {"modalities": ["text", "image"], "max_reference_images": max(0, max_inputs - 1)}

    # ------------------------------------------------------------------ generation

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model_slug = ""

        def fail(error: str, error_type: str) -> dict[str, Any]:
            return error_response(error=error, error_type=error_type, provider=PROVIDER,
                                  model=model_slug, prompt=prompt, aspect_ratio=aspect)

        if not prompt:
            return fail("Prompt is required and must be a non-empty string", "invalid_argument")
        key = client.api_key()
        if not key:
            return fail(f"{client.KEY_ENV} not set. Run `hermes tools` → Image Generation → deAPI "
                        f"to configure it (keys at {client.KEY_URL}).", "auth_required")

        sources: list[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        sources.extend(normalize_reference_images(reference_image_urls) or [])
        modality = "image" if sources else "text"
        inference_type = "img2img" if sources else "txt2img"

        try:
            models = client.fetch_models(key)
        except client.DeapiError as exc:
            return fail(f"Could not load the deAPI model catalog: {exc}", exc.error_type)
        model, note = cards.choose(models, inference_type, [kwargs.get("model"), settings.configured_model(KIND)])
        if model is None:
            return fail(f"No deAPI model for {inference_type} is available on this key's plan.",
                        "no_model_available")
        model_slug = model["slug"]
        extra: dict[str, Any] = {}
        if note:
            extra["note"] = note

        try:
            if sources:
                max_inputs = int(cards.limit(model, "max_input_images", 1) or 1)
                if len(sources) > max_inputs:
                    extra["dropped_reference_images"] = len(sources) - max_inputs
                    sources = sources[:max_inputs]
                files = []
                field = "image" if len(sources) == 1 else "images[]"
                for index, source in enumerate(sources):
                    files.append((field, client.load_source_image(source, index=index)))
                body = cards.apply_sampling(model, {"prompt": prompt, "model": model_slug}, seed=kwargs.get("seed"))
                request_id = client.submit(key, "/images/edits", data=cards.as_form(body), files=files)
            else:
                ratio = ASPECTS.get(aspect, ASPECTS["square"])
                width, height = cards.size_for_ratio(model, *ratio)
                body = cards.apply_sampling(
                    model, {"prompt": prompt, "model": model_slug, "width": width, "height": height},
                    seed=kwargs.get("seed"))
                extra.update(width=width, height=height)
                request_id = client.submit(key, "/images/generations", json_body=body)
            extra["request_id"] = request_id
            job = client.wait_for_job(key, request_id,
                                      timeout_s=settings.timeout_seconds(KIND, DEFAULT_TIMEOUT_S),
                                      initial_interval=1.5, max_interval=5.0)
        except client.DeapiError as exc:
            return fail(str(exc), exc.error_type)

        result_url = job.get("result_url")
        if not isinstance(result_url, str) or not result_url.strip():
            return fail(f"deAPI job {request_id} finished without a result URL.", "empty_response")
        extra.update(client.job_cost(job))
        extra["seed"] = body.get("seed")
        # Result URLs expire; keep a local copy when possible, else hand back the URL.
        try:
            image_ref = str(save_url_image(result_url, prefix=_cache_prefix(model_slug)))
        except Exception as exc:
            logger.debug("deAPI: caching the result failed (%s); returning the URL", type(exc).__name__)
            image_ref = result_url
        return success_response(image=image_ref, model=model_slug, prompt=prompt, aspect_ratio=aspect,
                                provider=PROVIDER, modality=modality, extra=extra)
