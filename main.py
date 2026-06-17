"""
GitHub Copilot Gateway — Minimal LLM Access Gateway.

Provides both OpenAI-compatible and Anthropic-compatible endpoints
that proxy to GitHub Copilot's LLM services with automatic protocol
conversion when needed.

On startup, if no token is found, the gateway automatically initiates
the GitHub OAuth device-code flow and guides you through authentication
right in the terminal. You can also use the /auth endpoints from another
terminal or via curl.

Endpoints:
  GET  /v1/models              — List models (OpenAI format)
  POST /v1/chat/completions    — OpenAI Chat Completions
  POST /v1/responses           — OpenAI Responses API
  POST /v1/messages            — Anthropic Messages API
  GET  /health                 — Health check

Auth:
  POST /auth/device            — Initiate OAuth device code flow
  POST /auth/token             — Poll for access token
  GET  /auth/status            — Check authentication status

Usage:
  python main.py
  GATEWAY_PORT=29381 python main.py
  GATEWAY_NO_AUTH_PROMPT=1 python main.py   # skip interactive auth, use /auth endpoints

Environment variables:
  GATEWAY_PORT                 — Listen port (default: 18742)
  GATEWAY_HOST                 — Listen host (default: 0.0.0.0)
  GATEWAY_ENTERPRISE_DOMAIN     — GitHub Enterprise domain (optional)
  GATEWAY_TOKEN_FILE            — Path to persist OAuth token (default: ~/.copilot-gateway/token.json)
  GATEWAY_MODEL_REFRESH_SECS    — Model refresh interval (default: 300)
  GATEWAY_NO_AUTH_PROMPT        — Set to 1 to skip interactive auth at startup
"""

import argparse
import asyncio
import json as jsonlib
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import load_config, CLIENT_ID
from auth import AuthManager
from models import ModelStore, print_model_table
from proxy import ProxyHandler

# ── stdout buffering ───────────────────────────────────────────

# Force line-buffered stdout so terminal banners appear immediately
# even when output is piped (e.g. docker logs, tee, systemd).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


# ── CLI parsing (runs first so --help exits cleanly) ───────────

def _parse_args() -> None:
    """Parse CLI arguments and apply them to the global config.

    Called at module level so --help / --version exit before we
    initialise auth, models, or the server.
    """
    parser = argparse.ArgumentParser(
        description="GitHub Copilot Gateway — OpenAI & Anthropic LLM proxy",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Dump raw GitHub Copilot /models response at startup",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=None,
        help="Listen port (default: 9992, env: GATEWAY_PORT)",
    )
    parser.add_argument(
        "--enterprise",
        type=str,
        default=None,
        help="GitHub Enterprise domain (env: GATEWAY_ENTERPRISE_DOMAIN)",
    )
    parser.add_argument(
        "--no-auth-prompt",
        action="store_true",
        default=None,
        help="Skip interactive auth at startup; use /auth endpoints instead",
    )
    args = parser.parse_args()

    if args.verbose:
        config.verbose = True
    if args.port is not None:
        config.port = args.port
    if args.enterprise is not None:
        config.enterprise_domain = args.enterprise
    if args.no_auth_prompt:
        os.environ["GATEWAY_NO_AUTH_PROMPT"] = "1"


# ── logging ────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("gateway")

# Reduce noise from httpx and uvicorn
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# ── globals ────────────────────────────────────────────────────

config = load_config()
_parse_args()  # apply CLI overrides (exits early on --help)
auth = AuthManager(config)
models = ModelStore(config, auth)
proxy = ProxyHandler(config, auth, models)

# Track the auth prompt task so we can cancel it on shutdown
_auth_prompt_task: asyncio.Task | None = None


# ── terminal helpers ───────────────────────────────────────────

def _print_banner(text: str) -> None:
    """Print a highlighted banner to the terminal (always flushed)."""
    width = 64
    bar = "═" * width
    lines = [f"\n  ╔{bar}╗"]
    for line in text.strip().split("\n"):
        lines.append(f"  ║ {line:<{width-2}} ║")
    lines.append(f"  ╚{bar}╝\n")
    msg = "\n".join(lines)
    print(msg, flush=True)


def _print_auth_instructions(verification_uri: str, user_code: str) -> None:
    """Print OAuth device-code instructions prominently."""
    _print_banner(f"""
🔐 GitHub Copilot authentication required

1. Open this URL in your browser:

   {verification_uri}

2. Enter this code when prompted:

   █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
   █   >>> {user_code} <<<   █
   █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

3. Approve the device activation request

Waiting for you to complete authorization...
(press Ctrl+C to cancel, then restart and use /auth endpoints instead)
""")


# ── verbose dump helper ────────────────────────────────────────

def _dump_raw_models() -> None:
    """Print the untouched upstream Copilot /models response to stdout."""
    raw = models.get_raw_response()
    if not raw:
        print("(no raw model data available yet)", flush=True)
        return
    print("\n── Raw GitHub Copilot /models response ──", flush=True)
    print(jsonlib.dumps(raw, indent=2, ensure_ascii=False), flush=True)
    print("── End of raw response ──\n", flush=True)


# ── startup auth flow ──────────────────────────────────────────

async def _run_startup_auth() -> None:
    """Auto-initiate device-code auth if no token exists.

    Runs in background during startup.  Prints instructions to the
    terminal and polls until the user completes authorization, then
    refreshes models.  The HTTP /auth endpoints remain available in
    parallel so the user can also drive auth via curl or code.
    """
    if auth.has_token:
        logger.info("Token loaded from disk — skipping auth prompt")
        # Models may not have been refreshed yet (the background loop
        # skips the first refresh when no token was available).  Force
        # a refresh now, then dump if verbose.
        if not models.get_models():
            await models.refresh()
        print_model_table(models)
        if config.verbose:
            _dump_raw_models()
        return

    # Check if interactive prompt is suppressed
    if os.getenv("GATEWAY_NO_AUTH_PROMPT", "") == "1":
        _print_banner(
            "No token found and GATEWAY_NO_AUTH_PROMPT=1 is set.\n\n"
            "Authenticate via the HTTP endpoints:\n\n"
            f"  curl -X POST http://localhost:{config.port}/auth/device\n"
            f"  curl -X POST http://localhost:{config.port}/auth/token \\\n"
            "       -H 'Content-Type: application/json' \\\n"
            "       -d '{\"device_code\":\"...\"}'\n\n"
            f"Or check status:  curl http://localhost:{config.port}/auth/status"
        )
        return

    try:
        # Step 1 — initiate device code
        logger.info("No token found — initiating device-code auth flow")
        result = await auth.initiate_device_code("github.com", "")

        _print_auth_instructions(
            result["verification_uri"],
            result["user_code"],
        )

        # Step 2 — poll for token (blocking, but server is already up)
        poll_result = await auth.poll_token(result["device_code"])

        if poll_result["status"] == "success":
            _print_banner(
                "✅ Authenticated successfully!\n\n"
                f"Token saved to: {config.token_file}\n"
                "Gateway is ready to serve requests."
            )
            # Immediately refresh models now that we have a token
            await models.refresh()
            print_model_table(models)

            # In verbose mode, dump the raw upstream response
            if config.verbose:
                _dump_raw_models()
        else:
            error = poll_result.get("error", "unknown")
            _print_banner(
                f"❌ Authentication failed: {error}\n\n"
                "You can try again:\n\n"
                f"  curl -X POST http://localhost:{config.port}/auth/device\n\n"
                "Or restart the gateway."
            )

    except asyncio.CancelledError:
        logger.info("Auth prompt cancelled")
        raise
    except Exception as e:
        logger.error(f"Startup auth flow failed: {e}")
        _print_banner(
            f"❌ Authentication error: {e}\n\n"
            "Retry via the HTTP endpoints:\n\n"
            f"  curl -X POST http://localhost:{config.port}/auth/device"
        )


# ── lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: start background tasks, run auth prompt if needed."""
    global _auth_prompt_task

    logger.info(f"Starting GitHub Copilot Gateway on {config.host}:{config.port}")
    logger.info(f"API base: {config.api_base_url}")
    logger.info(f"Token file: {config.token_file}")
    if config.enterprise_domain:
        logger.info(f"Enterprise: {config.enterprise_domain}")

    await proxy.startup()

    # Start model refresh loop (will retry once we have a token)
    await models.start_refresh_loop()

    # Start auth prompt as a background task — server is already
    # accepting requests, so /auth endpoints work in parallel
    _auth_prompt_task = asyncio.create_task(
        _run_startup_auth(), name="startup-auth"
    )

    yield

    # Shutdown
    if _auth_prompt_task and not _auth_prompt_task.done():
        _auth_prompt_task.cancel()
        try:
            await _auth_prompt_task
        except asyncio.CancelledError:
            pass

    await models.stop_refresh_loop()
    await proxy.shutdown()
    logger.info("Gateway shut down")


# ── app ────────────────────────────────────────────────────────

app = FastAPI(
    title="GitHub Copilot Gateway",
    version="1.0.0",
    lifespan=lifespan,
)


# ── health ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "authenticated": auth.has_token,
        "models_count": len(models.get_models()),
    }


# ── model endpoints ────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    """List available models in OpenAI-compatible format."""
    return await proxy.list_models()


@app.get("/v1/models/debug")
async def list_models_debug():
    """List models with full debug metadata.

    Shows supported_endpoints, anthropic_native, uses_responses_api,
    pricing, limits, and all capabilities — exactly what the gateway
    uses internally for routing decisions.
    """
    return await proxy.list_debug_models()


@app.get("/v1/models/raw")
async def list_models_raw():
    """Return the untouched upstream GitHub Copilot /models response.

    This is the raw JSON from api.githubcopilot.com/models — useful
    for inspecting the full model metadata that Copilot exposes.
    """
    return await proxy.list_raw_response()


# ── OpenAI-compatible endpoints ────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI Chat Completions API.

    Proxies to GitHub Copilot's /chat/completions (or /responses for GPT-5+).
    """
    return await proxy.chat_completions(request)


@app.post("/v1/responses")
async def responses(request: Request):
    """OpenAI Responses API.

    Proxies to GitHub Copilot's /responses endpoint.
    """
    return await proxy.responses(request)


# ── Anthropic-compatible endpoints ─────────────────────────────

@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic Messages API.

    If the model supports /v1/messages natively, forwards directly.
    Otherwise, converts Anthropic → OpenAI, proxies, and converts back.
    """
    return await proxy.messages(request)


# ── auth endpoints ─────────────────────────────────────────────

@app.post("/auth/device")
async def auth_device(request: Request):
    """Initiate OAuth device code flow.

    Request body (optional):
      {
        "deployment_type": "github.com" | "enterprise",
        "enterprise_url": "company.ghe.com"   // required if enterprise
      }

    Returns:
      {
        "verification_uri": "https://github.com/login/device",
        "user_code": "ABCD-1234",
        "device_code": "...",
        "interval": 5
      }
    """
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    deployment_type = body.get("deployment_type", "github.com")
    enterprise_url = body.get("enterprise_url", "")

    return await auth.initiate_device_code(deployment_type, enterprise_url)


@app.post("/auth/token")
async def auth_token(request: Request):
    """Poll for OAuth access token.

    Request body:
      {
        "device_code": "..."
      }

    Returns:
      {"status": "success", "access_token": "..."}
      or
      {"status": "failed", "error": "..."}
    """
    body = await request.json()
    device_code = body.get("device_code", "")
    if not device_code:
        return JSONResponse(
            {"status": "failed", "error": "device_code is required"},
            status_code=400,
        )
    return await auth.poll_token(device_code)


@app.get("/auth/status")
async def auth_status():
    """Check authentication status."""
    return await auth.get_status()


# ── main ───────────────────────────────────────────────────────

def main():
    """Entry point."""
    import uvicorn

    # Use __main__:app to avoid double-import when running as a script.
    # (Otherwise uvicorn re-imports the module as "main", re-running all
    # module-level code and creating duplicate AuthManager / ModelStore.)
    uvicorn.run(
        "__main__:app" if __name__ == "__main__" else "main:app",
        host=config.host,
        port=config.port,
        log_level="warning",
        reload=False,
    )


if __name__ == "__main__":
    main()
