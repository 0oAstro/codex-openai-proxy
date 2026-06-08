# codex-openai-proxy

Proxy your ChatGPT/Codex subscription as an OpenAI-compatible API.

## Features

- **OpenAI-compatible endpoints**: `/v1/models`, `/v1/responses`, `/v1/chat/completions`, `/v1/images/generations`, `/v1/images/edits`
- **Usage metrics**: `/usage` returns Codex rate-limit and credit snapshots
- **OAuth PKCE + device-code login**: browser-based or headless ChatGPT authentication with automatic token refresh
- **Streaming support**: SSE streaming for both responses and chat completions
- **Chat completions translation**: translates OpenAI chat format to/from Codex Responses API
- **Reasoning effort**: parse model name suffixes (e.g. `gpt-5.5-xhigh`) into reasoning parameters
- **Tool/function calling**: full support for function calling in chat completions
- **Token auto-refresh**: refreshes tokens 5 minutes before expiry
- **401 retry**: automatically refreshes and retries on auth failures

## Quick Start

```bash
# Build
cargo build --release

# Log in (opens browser)
./target/release/codex-openai-proxy login

# Or log in from SSH/headless hosts
./target/release/codex-openai-proxy login-device

# Start the proxy
./target/release/codex-openai-proxy serve

# Check auth status
./target/release/codex-openai-proxy auth status

# Log out
./target/release/codex-openai-proxy logout
```

## Usage

### Start the server

```bash
# Default: 0.0.0.0:8080
codex-openai-proxy serve

# Custom port
codex-openai-proxy serve --port 3000

# Or via environment variable
PORT=3000 codex-openai-proxy serve
```

### Authentication

```bash
# Browser login (OAuth PKCE)
codex-openai-proxy login

# Headless login (ChatGPT device code flow)
codex-openai-proxy login-device

# Check status
codex-openai-proxy auth status

# Logout
codex-openai-proxy logout
```

Credentials are stored in `~/auth.json` by default and are compatible with Codex `auth.json` files. Set `CODEX_AUTH_FILE` to use a different path. The OAuth requests mirror Codex: `originator=codex_cli_rs`, connector scopes, organization-bearing ID tokens, JSON refresh requests, and `chatgpt_account_id` extraction for upstream `chatgpt-account-id`.

## API Endpoints

### `GET /health`
Returns `{"status": "ok"}`.

### `GET /v1/models`
Returns available models in OpenAI format. Cached for 5 minutes.

### `GET /usage`
Returns Codex usage metrics parsed from the upstream rate-limit headers/events.

### `POST /v1/responses`
Passthrough to the Codex Responses API. Streams SSE response back verbatim.

### `POST /v1/chat/completions`
Translates OpenAI chat completions format to Codex Responses API and back.
Supports both streaming (`"stream": true`) and non-streaming modes.

### `POST /v1/images/generations` and `POST /v1/images/edits`
Translates OpenAI-compatible image generation/edit requests to Codex Responses API image-generation tools. The proxy sends the same backend family used by Codex (`https://chatgpt.com/backend-api/codex/responses`) and converts `image_generation_call` SSE output into OpenAI image response data.

#### Reasoning effort
Append a reasoning suffix to the model name:
```
gpt-5.5-none     -> reasoning: none
gpt-5.5-minimal  -> reasoning: minimal
gpt-5.5-low      -> reasoning: low
gpt-5.5-medium   -> reasoning: medium
gpt-5.5-high     -> reasoning: high
gpt-5.5-xhigh    -> reasoning: xhigh
```

## Docker Compose (preferred)

```bash
docker network create main-network
docker compose up -d
```

The container mounts `/opt/codex-openai-proxy/auth.json` read-write and sets `CODEX_AUTH_FILE=/opt/codex-openai-proxy/auth.json` so automatic token refresh can persist refreshed tokens. Run `codex-openai-proxy login-device` on the host or copy an existing Codex-compatible `auth.json` there before starting the service.

## Example: Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # Auth is handled by the proxy
)

response = client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Limitations

- **No WebSocket transport**: The Codex CLI supports a WebSocket mode at `wss://chatgpt.com/backend-api/codex/responses` for persistent connections with `previous_response_id` chaining (~40% faster for 20+ tool-call chains). This proxy only supports HTTP/SSE. Adding WebSocket would require `tokio-tungstenite`, in-memory response state, and the `response.create` event protocol. Note that HTTP gateway proxies (e.g. Bifrost) cannot route WebSocket connections anyway, so this would only benefit direct-to-proxy clients.
- **Stateless**: Each request is independent. `previous_response_id` and `item_reference` are not supported -- clients must send the full conversation history each turn.
- **No image endpoint streaming via Bifrost**: Image generation/edit streaming uses custom SSE event types that may not be forwarded by all HTTP gateways.

## License

MIT
