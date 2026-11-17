# Security

Report vulnerabilities in this plugin privately to **support@deapi.ai** (subject: "Hermes plugin
security"), not in a public issue. Include the plugin version or commit, the Hermes Agent version
and steps to reproduce.

The plugin handles one secret, `DEAPI_API_KEY`. It is read through Hermes's profile-scoped secret
store, sent only to `api.deapi.ai`, and never logged or included in error messages. If you think a
key has leaked, revoke it at https://app.deapi.ai/settings/api-keys.

What goes to deAPI: prompts and source images for image and video generation, the text Hermes
asks the speech backend to read aloud, and the recordings Hermes asks the speech-to-text backend
to transcribe (voice messages, voice-mode recordings). All of it is sent to `api.deapi.ai` only,
and every request is billed to the deAPI account. Nothing else from the machine is read or sent.
