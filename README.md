# hermes-plugin-deapi

[deAPI](https://deapi.ai) image & video generation backends for
[Hermes Agent](https://github.com/NousResearch/hermes-agent) — cheap, hosted
open-source models as a drop-in alternative to FAL.

- **Image**: FLUX.2 Klein from **~$0.002/image** (512px), FLUX.1 schnell,
  Z-Image Turbo and more — auto-discovered live, nothing hardcoded
- **Video**: LTX family text-to-video **and** image-to-video from ~$0.002/clip
- Free **$5 signup credit** at [app.deapi.ai](https://app.deapi.ai), no card required
- Pure-Python stdlib, zero dependencies, zero Hermes core changes

Ships as a standalone plugin repo per the Hermes
[contributing policy](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
for third-party product integrations.

## Install

```bash
hermes plugins install deapi-ai/hermes-plugin-deapi --enable
```

Or manually:

```bash
git clone https://github.com/deapi-ai/hermes-plugin-deapi.git /tmp/hpd
mkdir -p ~/.hermes/plugins/image_gen ~/.hermes/plugins/video_gen
cp -r /tmp/hpd/image_gen/deapi ~/.hermes/plugins/image_gen/deapi
cp -r /tmp/hpd/video_gen/deapi ~/.hermes/plugins/video_gen/deapi
hermes plugins enable image_gen/deapi
hermes plugins enable video_gen/deapi
```

## Configure

1. Get an API key at [app.deapi.ai](https://app.deapi.ai) and put it in
   `~/.hermes/.env`:

   ```
   DEAPI_API_KEY=your-key-here
   ```

   (Both the raw panel format `12345|...` and the `dpn-sk-`-prefixed form work.)

2. Select deAPI as the backend — either via `hermes tools` → Image/Video
   Generation (deAPI appears in the picker automatically), or directly in
   `~/.hermes/config.yaml`:

   ```yaml
   image_gen:
     provider: deapi
     # model: Flux_2_Klein_4B_BF16   # optional; default = best live Klein/FLUX
   video_gen:
     provider: deapi
     # model: Ltxv_13B_0_9_8_Distilled_FP8
   ```

3. Start a new Hermes session and ask: *"generate an image of a red fox in
   snow"* or *"animate this photo"*.

Model overrides: `image_gen.model` / `video_gen.model` in config, or
`DEAPI_IMAGE_MODEL` / `DEAPI_VIDEO_MODEL` env vars. Available models are
fetched live from `GET /api/v2/models` — run `hermes tools` to browse them.

## Agent-pasteable install

Paste this to your Hermes (or any agent with a terminal):

> Install the deAPI image/video plugin for Hermes: clone
> https://github.com/deapi-ai/hermes-plugin-deapi.git, copy image_gen/deapi to
> ~/.hermes/plugins/image_gen/deapi and video_gen/deapi to
> ~/.hermes/plugins/video_gen/deapi, run `hermes plugins enable image_gen/deapi`
> and `hermes plugins enable video_gen/deapi`, add `image_gen.provider: deapi`
> and `video_gen.provider: deapi` to ~/.hermes/config.yaml, and make sure
> DEAPI_API_KEY is set in ~/.hermes/.env.

## How it works

Both providers call the native deAPI v2 REST API (`api.deapi.ai`): submit a
job, poll `GET /api/v2/jobs/{id}`, download the result into Hermes' local
cache (`~/.hermes/cache/images|videos/`) — result URLs expire after ~24 h, so
files are persisted immediately. Model parameters (steps, guidance, fps,
frame limits) come from each model's live metadata, never hardcoded.

## Pricing (approx., pay-per-use)

| Task | Model | Cost |
|---|---|---|
| Image 512×512 | FLUX.2 Klein | ~$0.0019 |
| Image 1024×1024 | FLUX.2 Klein | ~$0.0074 |
| Video 2 s / 480p | LTX | ~$0.002+ |

Exact prices: `POST /api/v2/<endpoint>/price` or [deapi.ai/pricing](https://deapi.ai/pricing).

## Related

- deAPI Agent Skill (portable, agentskills.io): https://github.com/deapi-ai/skills
- deAPI MCP server: https://github.com/deapi-ai/mcp-server-deapi
- deAPI docs: https://docs.deapi.ai

## License

MIT
