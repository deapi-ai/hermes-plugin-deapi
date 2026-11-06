# Security

Report vulnerabilities in this plugin privately to **support@deapi.ai** (subject: "Hermes plugin
security"), not in a public issue. Include the plugin version or commit, the Hermes Agent version
and steps to reproduce.

The plugin handles one secret, `DEAPI_API_KEY`. It is read through Hermes's profile-scoped secret
store, sent only to `api.deapi.ai`, and never logged or included in error messages. If you think a
key has leaked, revoke it at https://app.deapi.ai/settings/api-keys.
