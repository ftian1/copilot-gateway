# GitHub Copilot Gateway

A minimal, standalone gateway that provides **OpenAI-compatible** and **Anthropic-compatible** API endpoints to access GitHub Copilot's LLM services. On first launch it walks you through GitHub OAuth device-code authentication right in the terminal. The token is persisted to disk so subsequent starts are instant.

## Architecture

```
┌───── Upper App ─────┐                      ┌── Copilot Gateway ──┐                      ┌ GitHub Copilot API ─┐
│  OpenAI SDK         │──── OpenAI request ─▶│  • same API         │──── forward (same) ─▶│  /chat/completions  │
│                     │                      │    → forward        │                      │  /responses         │
│  Anthropic SDK      │─ Anthropic request ─▶│  • different API    │── convert (An↔OAI) ─▶│  /v1/messages       │
│                     │                      │    → convert        │                      │  (Claude only)      │
│                     │◀── SSE stream ───────│                     │◀── SSE stream ───────│                     │
└─────────────────────┘                      └─────────────────────┘                      └─────────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the gateway

```bash
python main.py
```

The gateway starts on `http://localhost:9992`. If no token is found, it auto-initiates the GitHub device-code flow and prints instructions to the terminal.

To skip the interactive prompt and auth via curl instead:

```bash
python main.py --no-auth-prompt
```

### 3. Authenticate (interactive — default)

The gateway prints a banner with a URL and a code. Open the URL, enter the code, approve the device. The token is automatically saved to `~/.copilot-gateway/token.json`. Next time you start the gateway, auth is skipped.

### 4. Authenticate (via curl — alternative)

```bash
# Step 1: initiate device code
curl -X POST http://localhost:9992/auth/device

# Step 2: open the verification_uri in a browser, enter the user_code

# Step 3: poll for token
curl -X POST http://localhost:9992/auth/token \
  -H "Content-Type: application/json" \
  -d '{"device_code": "..."}'
```

### 5. Make LLM requests

**OpenAI:**
```bash
curl -X POST http://localhost:9992/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Anthropic:**
```bash
curl -X POST http://localhost:9992/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

**Streaming:** Add `"stream": true` to any request body.

## CLI Flags

```
python main.py [options]

  -v, --verbose        Dump raw GitHub Copilot /models response at startup
  -p, --port PORT      Listen port (default: 9992, env: GATEWAY_PORT)
  --enterprise DOMAIN  GitHub Enterprise domain
  --no-auth-prompt     Skip interactive auth; use /auth endpoints instead
  -h, --help           Show help
```

## Using with SDKs

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9992/v1",
    api_key="not-needed"  # gateway handles auth
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Anthropic Python SDK

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:9992/v1",
    api_key="not-needed"  # gateway handles auth
)

response = client.messages.create(
    model="claude-sonnet-4-6",
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.content[0].text)
```

### Claude Code

Start [Claude Code](https://docs.anthropic.com/en/docs/claude-code) against the gateway by setting these environment variables, then launch `claude`:

```bash
export ANTHROPIC_BASE_URL=http://localhost:9992
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_MODEL=claude-opus-4.8
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-4.8
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-4.6
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-4.5
export CLAUDE_CODE_SUBAGENT_MODEL=claude-haiku-4.5
export DISABLE_NON_ESSENTIAL_MODEL_CALLS=1
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export IS_SANDBOX=1
export ENABLE_TOOL_SEARCH=true
export CLAUDE_CODE_ATTRIBUTION_HEADER=0

# Then start Claude Code
claude --dangerously-skip-permissions
```

> **Note:** Model IDs must match what the Copilot API exposes. Run `curl http://localhost:9992/v1/models | python3 -m json.tool` to see available models and pick the ones you want.

## Configuration

| Env Variable                 | Default                         | Description                                       |
|------------------------------|---------------------------------|---------------------------------------------------|
| `GATEWAY_PORT`               | `9992`                          | Listen port                                       |
| `GATEWAY_HOST`               | `0.0.0.0`                       | Listen host                                       |
| `GATEWAY_ENTERPRISE_DOMAIN`  | _(empty)_                       | GitHub Enterprise domain (e.g. `company.ghe.com`) |
| `GATEWAY_TOKEN_FILE`         | `~/.copilot-gateway/token.json` | OAuth token persistence path                      |
| `GATEWAY_MODEL_REFRESH_SECS` | `300`                           | Model list refresh interval (seconds)             |
| `GATEWAY_NO_AUTH_PROMPT`     | _(empty)_                       | Set to `1` to skip interactive auth at startup    |

## API Endpoints

### LLM Endpoints

| Method | Path                   | Description                                                                     |
|--------|------------------------|---------------------------------------------------------------------------------|
| `GET`  | `/v1/models`           | List models (OpenAI format, includes `supported_endpoints`, `anthropic_native`) |
| `GET`  | `/v1/models/debug`     | Full parsed model metadata (pricing, limits, capabilities)                      |
| `GET`  | `/v1/models/raw`       | **Untouched** upstream Copilot `/models` JSON — for inspecting raw Copilot data |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions                                                         |
| `POST` | `/v1/responses`        | OpenAI Responses API                                                            |
| `POST` | `/v1/messages`         | Anthropic Messages API                                                          |

### Auth Endpoints

| Method | Path           | Description                     |
|--------|----------------|---------------------------------|
| `POST` | `/auth/device` | Initiate OAuth device code flow |
| `POST` | `/auth/token`  | Poll for access token           |
| `GET`  | `/auth/status` | Check authentication status     |

### Health

| Method | Path      | Description                                                            |
|--------|-----------|------------------------------------------------------------------------|
| `GET`  | `/health` | Health check (`{"status":"ok","authenticated":true,"models_count":N}`) |

## Routing Logic

### GPT-5 Model Routing

GPT-5+ models (except `gpt-5-mini`) are automatically routed to the **Responses API** instead of Chat Completions, matching GitHub Copilot's own routing logic.

### Anthropic Protocol Support

- **Models with `/v1/messages` in `supported_endpoints`** (e.g., Claude models): Anthropic requests are forwarded directly to Copilot's native `/v1/messages` endpoint.
- **Models without `/v1/messages`** (e.g., GPT models): Anthropic requests are converted to OpenAI Chat Completions format, proxied, and the response is converted back to Anthropic format. This includes full SSE streaming conversion.

Check which protocol each model supports:

```bash
curl http://localhost:9992/v1/models | python3 -m json.tool
```

Look at the `anthropic_native` and `supported_endpoints` fields per model.

## Docker

```bash
# Build
docker build -t copilot-gateway .

# Run (mount a volume for token persistence)
docker run -p 9992:9992 \
  -e GATEWAY_PORT=9992 \
  -v ~/.copilot-gateway:/home/gateway/.copilot-gateway \
  copilot-gateway
```

## Enterprise GitHub

```bash
python main.py --enterprise company.ghe.com
# or
GATEWAY_ENTERPRISE_DOMAIN=company.ghe.com python main.py
```
