# codex-openai-proxy

Proxy your ChatGPT/Codex subscription as an OpenAI-compatible API.

## Features

- **OpenAI-compatible endpoints**: `/v1/models`, `/v1/responses`, `/v1/chat/completions`
- **OAuth PKCE login**: browser-based authentication with automatic token refresh
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

# Check status
codex-openai-proxy auth status

# Logout
codex-openai-proxy logout
```

Credentials are stored in `~/auth.json`.

## API Endpoints

### `GET /health`
Returns `{"status": "ok"}`.

### `GET /v1/models`
Returns available models in OpenAI format. Cached for 5 minutes.

### `POST /v1/responses`
Passthrough to the Codex Responses API. Streams SSE response back verbatim.

### `POST /v1/chat/completions`
Translates OpenAI chat completions format to Codex Responses API and back.
Supports both streaming (`"stream": true`) and non-streaming modes.

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

The container mounts `/opt/codex-openai-proxy/auth.json` from the host to `/root/auth.json` in the container.

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

## License

MIT
