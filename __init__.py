"""deAPI image, video, speech and transcription backends for Hermes Agent.

One plugin, four providers named ``deapi``: an ``image_generate`` backend, a ``video_generate``
backend, a ``text_to_speech`` backend and a speech-to-text backend, all on deAPI's native API v2
(https://api.deapi.ai/api/v2).
"""

from __future__ import annotations


def register(ctx) -> None:
    """Plugin entry point: register every backend."""
    from .image_provider import DeapiImageGenProvider
    from .speech_provider import DeapiTTSProvider
    from .stt_provider import DeapiTranscriptionProvider
    from .video_provider import DeapiVideoGenProvider

    ctx.register_image_gen_provider(DeapiImageGenProvider())
    ctx.register_video_gen_provider(DeapiVideoGenProvider())
    ctx.register_tts_provider(DeapiTTSProvider())
    ctx.register_transcription_provider(DeapiTranscriptionProvider())
