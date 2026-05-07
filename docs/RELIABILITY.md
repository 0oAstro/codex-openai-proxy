# Reliability

This proxy sits between OpenAI-compatible clients and upstream ChatGPT/Codex
APIs. Reliability work should protect client compatibility, streaming behavior,
and authentication recovery.

## Invariants

- `/health` returns a local health response and does not require upstream
  connectivity.
- When `PROXY_API_KEY` is configured, all non-health routes require a matching
  bearer token.
- Token refresh should happen before expiry when possible and retry once after
  an upstream `401`.
- Streaming endpoints must preserve SSE semantics and should avoid buffering
  unless the client explicitly requested a non-streaming response.
- The service remains stateless across requests unless the README and
  architecture docs are updated in the same change.

## Verification

Use the smallest check that proves the changed behavior, then broaden when
protocol translation, auth, or streaming is touched.

Recommended baseline:

```bash
cargo fmt --check
cargo check
cargo test
```

For route changes, run the service locally and exercise the affected route with
`curl` or an SDK client. For streaming changes, validate both streaming and
non-streaming code paths when applicable.

## Failure Handling

- Return clear JSON error bodies for client-visible failures where practical.
- Use `tracing` for diagnostics, but never log secrets or full auth headers.
- Preserve upstream status codes when forwarding responses unless translation
  is required for OpenAI compatibility.
- Avoid infinite retry loops. A single refresh-and-retry on auth failure is the
  expected behavior.

## Agent Feedback Loops

When reliability issues repeat, encode the lesson mechanically:

- Add a focused unit test for parser or translator behavior.
- Add an integration-style route test if the issue crosses modules.
- Add documentation only when it captures a durable operational rule.
- Add scripts or CI checks if humans would otherwise need to remember the rule.
