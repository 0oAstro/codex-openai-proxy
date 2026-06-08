# Security

Security work in this repository centers on local credential storage, proxy
API-key enforcement, safe upstream authorization, and careful logging.

## Secrets

Treat the following as secrets:

- `~/auth.json`
- `PROXY_API_KEY`
- access tokens
- refresh tokens
- upstream authorization headers
- account identifiers when combined with auth context

Never commit real credentials or logs that contain them. Never print tokens in
CLI output, HTTP responses, debug logs, tests, fixtures, or documentation.

## Authentication Boundaries

- `/health` may remain unauthenticated.
- All other HTTP routes must honor `PROXY_API_KEY` when it is configured.
- The proxy's client-facing API key is separate from upstream ChatGPT/Codex
  credentials. Do not mix the two auth domains.
- Token refresh must use the stored refresh token only for upstream
  authentication recovery, using the Codex-compatible JSON refresh-token
  request shape against `https://auth.openai.com/oauth/token`. Container deployments
  must mount the auth file read-write so refreshed tokens can be persisted.

## Upstream Identity Headers

Outbound ChatGPT/Codex requests use `Authorization: Bearer <access_token>`,
`chatgpt-account-id` from `chatgpt_account_id` in the ID token when present,
Codex `originator`, `version`, and `User-Agent` headers. Client-supplied
OpenAI organization/project headers may be forwarded, but bearer credentials
must always come from the local auth file.

## Logging

Use `tracing` for operational visibility. Logs may include route names,
statuses, retry attempts, and non-sensitive error summaries. Logs must not
include bearer tokens, refresh tokens, full request headers, or raw auth files.

## Dependency And Protocol Changes

- Prefer stable, well-maintained Rust crates already used by the project.
- Review new dependencies for transitive risk and whether they handle secrets
  or network traffic.
- When changing CORS, middleware ordering, auth headers, or credential
  persistence, update this document and run targeted verification.

## Review Checklist

- Does the change expose any new route without considering `PROXY_API_KEY`?
- Could logs or error bodies reveal credentials?
- Does the change preserve token refresh and logout behavior?
- Are new config values documented without including real secrets?
