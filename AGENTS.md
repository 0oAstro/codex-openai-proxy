# Agent Operating Guide

This repository is a Rust/Axum proxy that exposes OpenAI-compatible endpoints
backed by a ChatGPT/Codex subscription. Treat this file as the short table of
contents for agent work in the project: keep direct instructions here concise,
and move durable design, architecture, reliability, or product notes into the
versioned docs linked below.

## Harness Engineering Protocol

- Humans steer; agents execute. Convert requests into concrete acceptance
  criteria, implement the change, verify it locally, and report the result.
- Prefer depth-first execution. Break large work into small capabilities,
  land the missing capability, then use it to unlock the next step.
- When a task fails, diagnose the missing tool, invariant, documentation, or
  feedback loop. Do not paper over failures with retries alone.
- Keep repository knowledge agent-legible. If a decision matters after this
  task, encode it in code, tests, CI, or docs rather than relying on chat
  context.
- Enforce invariants mechanically when practical. Prefer tests, type checks,
  compiler errors, or lints over prose-only rules.
- Optimize for maintainable agent throughput: small scoped changes, clear
  commands, reproducible verification, and minimal hidden state.

## Project Map

- `docs/INDEX.md`: documentation map and system-of-record entrypoint.
- `docs/ARCHITECTURE.md`: service structure, request flow, and module
  ownership.
- `docs/RELIABILITY.md`: operational invariants, verification, and failure
  handling.
- `docs/SECURITY.md`: auth, secret handling, and boundary expectations.
- `docs/PLANS.md`: planning protocol for larger or multi-step work.
- `src/main.rs`: CLI entrypoint, Axum router, middleware, and command dispatch.
- `src/auth.rs`: OAuth/device login, token persistence, refresh, and revocation.
- `src/config.rs`: shared application state, upstream constants, environment
  configuration, and Codex client version handling.
- `src/models.rs`: model listing and upstream authentication header helpers.
- `src/proxy.rs`: `/v1/responses` passthrough and SSE collection/streaming.
- `src/chat.rs`: OpenAI chat completions translation, streaming, tool calls,
  and reasoning suffix parsing.
- `src/images.rs`: OpenAI-compatible image generation/edit endpoints.
- `src/usage.rs`: usage endpoint handling when present in the working tree.
- `.github/workflows/`: release and image build automation.

If you add substantial new behavior, add or update docs that explain the new
route, data boundary, or operational assumption. Keep `AGENTS.md` short; use
linked docs for deeper design records.

## Rust Standards

- Preserve the existing Rust 2021 style and module layout.
- Use `anyhow::Result` for top-level fallible flows and explicit HTTP error
  responses at request boundaries.
- Parse and validate external request shapes with `serde` types at the
  boundary. Avoid building logic on guessed `serde_json::Value` shapes unless
  the upstream format is genuinely dynamic.
- Keep async code cancellation-safe where possible. Avoid blocking work inside
  request handlers.
- Keep logs structured and useful with `tracing`; never log access tokens,
  refresh tokens, API keys, account secrets, or full auth headers.
- Preserve OpenAI-compatible response shapes and SSE behavior when touching
  `/v1/responses`, `/v1/chat/completions`, or image endpoints.

## Security And Reliability

- Treat `~/auth.json`, `PROXY_API_KEY`, bearer tokens, refresh tokens, and
  upstream authorization headers as secrets.
- `/health` may remain unauthenticated. Other endpoints must continue to honor
  `PROXY_API_KEY` when it is configured.
- Keep token refresh behavior deterministic: refresh before expiry when
  possible, retry once on upstream `401`, and surface clear errors after that.
- Be careful with CORS and auth middleware ordering. Validate behavior after
  changing either one.
- Avoid introducing persistent server-side conversation state unless the
  stateless contract in `README.md` is intentionally revised.

## Verification

Run the smallest useful verification for the change, then broaden when the
blast radius warrants it.

Recommended commands:

```bash
cargo fmt --check
cargo check
cargo test
```

For endpoint or streaming changes, also run the server locally and exercise the
affected route with `curl` or an SDK client. Do not require live upstream calls
for purely local parser or translator tests when unit coverage can prove the
behavior.

## Change Discipline

- Before editing, inspect the relevant files and current git status.
- Do not overwrite user changes. If files are already modified, read them and
  work with the existing state.
- Keep changes tightly scoped to the task. Avoid drive-by refactors,
  dependency churn, or unrelated formatting.
- Prefer adding focused tests near the changed behavior. If test coverage is
  missing and the change affects protocol translation, auth, or streaming,
  add coverage unless there is a concrete blocker.
- After edits, summarize changed files, verification performed, and any
  residual risk or commands that could not be run.

## Documentation Practice

- `README.md` is the user-facing contract for setup, commands, endpoints, and
  limitations. Update it when behavior visible to users changes.
- Use `docs/` as the system of record for durable decisions:
  `docs/ARCHITECTURE.md` for structure, `docs/RELIABILITY.md` for operational
  behavior, `docs/SECURITY.md` for auth/secrets boundaries, and
  `docs/PLANS.md` for multi-step work.
- Prefer a map plus links over a monolithic manual. Keep guidance fresh by
  deleting stale instructions when behavior changes.
