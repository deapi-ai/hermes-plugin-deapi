# Changelog

## 0.3.1

- Fixed the links in setup prompts and error messages: API keys at
  https://app.deapi.ai/settings/api-keys, top-ups at https://app.deapi.ai/billing, job history at
  https://app.deapi.ai/dashboard (0.3.0 pointed to a page that does not exist).

## 0.3.0

Rewritten for the current Hermes plugin API and the plugin catalog. Replaces the two 0.2
plugins (`deapi-image`, `deapi-video`) with one plugin, `deapi`, at the repository root.
To upgrade, remove the old ones (`hermes plugins remove deapi-image`, `hermes plugins remove
deapi-video`) and install this one. `DEAPI_API_KEY` stays the same.


- `image_generate` backend: text-to-image and image editing on deAPI's native API v2.
- `video_generate` backend: text-to-video and image-to-video.
- Models, limits and defaults from the key's live `GET /api/v2/models`.
