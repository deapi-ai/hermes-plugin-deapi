"""deAPI image and video generation backends for Hermes Agent.

One plugin, two providers named ``deapi``: an ``image_generate`` backend and a
``video_generate`` backend, both on deAPI's native API v2 (https://api.deapi.ai/api/v2).
"""

from __future__ import annotations


def register(ctx) -> None:
    """Plugin entry point: register both backends."""
    from .image_provider import DeapiImageGenProvider
    from .video_provider import DeapiVideoGenProvider

    ctx.register_image_gen_provider(DeapiImageGenProvider())
    ctx.register_video_gen_provider(DeapiVideoGenProvider())
