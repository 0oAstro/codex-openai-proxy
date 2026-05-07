# Architecture

`codex-openai-proxy` is a Rust 2021 service built with Axum and Tokio. It
exposes an OpenAI-compatible API surface and forwards requests to upstream
ChatGPT/Codex APIs using locally stored subscription credentials.

## Runtime Shape

The binary has two primary modes:

- CLI auth commands for browser login, device-code login, logout, and auth
  status.
- HTTP server mode for OpenAI-compatible API routes.

The server is intentionally stateless across requests. Clients must send the
full conversation history each turn unless the README's stateless contract is
explicitly revised.

## Request Flow

1. `src/main.rs` parses CLI arguments and builds the Axum router.
2. Middleware applies proxy API-key auth when `PROXY_API_KEY` is set. `/health`
   remains unauthenticated.
3. Route handlers load or refresh ChatGPT/Codex credentials as needed.
4. Handlers translate OpenAI-compatible inputs where required, call upstream
   APIs with Codex-compatible headers, and stream or collect responses back to
   the client.

## Module Ownership

- `src/main.rs`: CLI entrypoint, middleware, route registration, and server
  startup.
- `src/auth.rs`: OAuth PKCE login, device-code login, token persistence,
  refresh, revocation, and expiry checks.
- `src/config.rs`: application state, upstream constants, environment-derived
  configuration, HTTP client setup, and Codex client version handling.
- `src/models.rs`: `/v1/models` handling and shared upstream auth header
  construction.
- `src/proxy.rs`: `/v1/responses` passthrough, request body sanitization, SSE
  forwarding, non-stream collection, and upstream 401 retry.
- `src/chat.rs`: `/v1/chat/completions` translation, reasoning suffix parsing,
  tool/function call conversion, and streaming chunk generation.
- `src/images.rs`: OpenAI-compatible image generation and edit endpoint
  handling.
- `src/usage.rs`: usage endpoint handling when present in the working tree.

## Boundary Rules

- Parse external request bodies into typed `serde` structures at route
  boundaries whenever the shape is known.
- Keep upstream API details behind handler or helper functions; do not leak
  upstream-only fields into OpenAI-compatible responses unless intentionally
  documented.
- Keep auth, token refresh, and header construction centralized enough that
  secret handling rules are easy to audit.
- Add tests around protocol translation code before broadening the supported
  OpenAI-compatible surface.
