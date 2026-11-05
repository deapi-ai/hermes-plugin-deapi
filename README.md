# deAPI for Hermes Agent

Image and video generation backends for [Hermes Agent](https://github.com/NousResearch/hermes-agent)
on [deAPI](https://deapi.ai), a GPU inference API for open models (FLUX.2 Klein, Z-Image,
Qwen Image Edit, LTX-2.5, …) plus partner-served ones.

One plugin, two backends, both named `deapi`:

| Hermes tool | What it does | deAPI endpoint |
|---|---|---|
| `image_generate` | text-to-image | `POST /api/v2/images/generations` |
| `image_generate` with `image_url` | image editing (the image and any references are uploaded) | `POST /api/v2/images/edits` |
| `video_generate` | text-to-video | `POST /api/v2/videos/generations` |
| `video_generate` with `image_url` | image-to-video (the image is the first frame) | `POST /api/v2/videos/animations` |

Every deAPI generation is an asynchronous job: the plugin submits it, polls
`GET /api/v2/jobs/{request_id}` until it is done, then saves the result under
`$HERMES_HOME/cache/generated/` with Hermes's own helpers (result URLs expire).

## Install

> Upgrading from 0.2? The old `deapi-image` and `deapi-video` plugins are now one plugin,
> `deapi`. Remove the old ones first: `hermes plugins remove deapi-image` and
> `hermes plugins remove deapi-video`.

```bash
hermes plugins install deapi        # from the plugin catalog
# or: hermes plugins install deapi-ai/hermes-plugin-deapi
hermes plugins enable deapi
```

Create an API key at [deapi.ai/settings/api-keys](https://deapi.ai/settings/api-keys) and give it to Hermes as
`DEAPI_API_KEY` (the install prompt asks for it, or add it to the `.env` of your Hermes home).
Then pick the backend:

```bash
hermes tools    # Image Generation → deAPI, Video Generation → deAPI
```

or in `config.yaml`:

```yaml
image_gen:
  provider: deapi
video_gen:
  provider: deapi
```

## Models

Models come from `GET /api/v2/models` for your key, so the list always matches your plan and
new models show up without an update. The picker in `hermes tools` shows them; own-fleet models
are listed first, partner-served models (billed at the partner's price) after.

Which model a call uses:

- **Image**: `image_gen.model` when it is a deAPI model id → `image_gen.deapi.model` → the
  first own-fleet text-to-image model matching FLUX.2 Klein, then FLUX. An edit request with a
  text-only model uses an editing model from the catalog instead and says so in the result's
  `note`.
- **Video**: `video_gen.model` → `video_gen.deapi.model` → the first own-fleet LTX model.

A model id that is not in your catalog is never sent: the plugin falls back to the automatic
choice and explains it in `note`.

```yaml
image_gen:
  provider: deapi
  deapi:
    model: Flux1schnell   # optional; any id from `hermes tools` or GET /api/v2/models
    timeout: 300          # optional, seconds to wait for a job (default 300)
video_gen:
  provider: deapi
  deapi:
    model: Ltx2_5_22B_Dist_INT8
    timeout: 900          # default 900
```

## How Hermes's options map to deAPI

deAPI publishes per-model limits (sizes, size grid, pixel budget, frames, fps, steps) and
defaults. The plugin derives every request from the selected model's card:

- **Image size**: `landscape` 3:2, `square` 1:1, `portrait` 2:3 at the model's default long
  side, scaled into its limits and snapped to its size grid.
- **Video size**: the requested aspect ratio with the requested resolution as the short side
  (`720p` 16:9 → 1280×720), scaled down to the model's pixel budget and snapped to its grid.
- **Duration**: frames = seconds × the model's fps, clamped to the model's frame range.
  Without a duration the model's default clip length is used.
- **Steps**: the model's default where the model takes steps. A seed is always sent (random
  unless you give one) and is returned in the result.
- **Negative prompt** (video): sent only to models that support it.

Successful results include `request_id`, the size used, and `cost_usd` (with
`cost_is_estimate`) as reported by deAPI for that job.

## Errors

Errors come back as Hermes tool errors with an `error_type`: `auth_error`, `insufficient_balance`,
`key_limit_exceeded`, `rate_limited`, `invalid_request` (with deAPI's field messages),
`job_failed` (with deAPI's failure reason, whether a retry may help and whether the charge was
refunded), `timeout` and `unknown_state`.

A submit is never sent twice. Only a `429`, which deAPI returns before it accepts a job, is
retried. If the connection drops after a submit, or a job outlives the timeout, the error says
so and gives the `request_id`: the job may still run and be billed, so check the dashboard
before trying again.

## What this plugin does on your machine

- **Network**: prompts, source images and the API key go to `api.deapi.ai` only. Image URLs
  you pass as sources are downloaded by the plugin through Hermes's SSRF-guarded client, because
  deAPI takes source images as uploads. Result files are downloaded from deAPI's result URLs.
- **Reads**: only local image files you pass as sources. A file is read in full only when its
  header is a PNG, JPEG, GIF, BMP or WebP image header, so no other kind of file can be uploaded.
- **Writes**: generated media under `$HERMES_HOME/cache/generated/` through Hermes's helpers.
  The model catalog is cached in memory only (10 minutes, keyed by a hash of the key).
- **Credentials**: `DEAPI_API_KEY`, read with `agent.secret_scope.get_secret`; never logged or
  put in error messages.
- **Cost**: every generation is billed to your deAPI account.
- No telemetry, shell commands, background processes, extra Python dependencies or
  self-updating code.

## Why the native API

deAPI also has an OpenAI-compatible API, but the native API v2 fits Hermes's backends better:
its model list carries each model's limits and defaults (so sizes and clip lengths can be fitted
per model), it lists every model on the plan including partner-served ones, it reports the
price and failure reason of each job, and image-to-video takes the first frame directly.

## Development

The tests need a Hermes Agent checkout (the providers subclass its ABCs) and use a fake HTTP
layer; nothing touches the network.

```bash
HERMES_AGENT_DIR=/path/to/hermes-agent python -m pytest
ruff check .
hermes plugins validate . --install-deps
```

CI (`.github/workflows/tests.yml`) runs the same three steps against the current Hermes Agent
`main` on every push and pull request, and weekly. Security reports: see `SECURITY.md`.

## License

MIT
