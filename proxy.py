"""
LLM request proxy with model-aware routing.

Routes:
  POST /v1/chat/completions  → OpenAI Chat Completions API
  POST /v1/responses         → OpenAI Responses API
  POST /v1/messages           → Anthropic Messages API (native or converted)

Based on the OpenCode reference:
  packages/llm/src/providers/github-copilot.ts
  packages/opencode/src/plugin/github-copilot/copilot.ts
"""

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse, JSONResponse

from config import API_VERSION, USER_AGENT, Config
from auth import AuthManager
from models import ModelStore, CopilotModel
from usage import (
    UsageTracker,
    extract_openai_usage,
    extract_responses_usage,
    extract_anthropic_usage,
)
from convert import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
    openai_sse_to_anthropic_sse,
    finalize_anthropic_stream,
    openai_to_anthropic_request,
    anthropic_to_openai_response,
    anthropic_sse_to_openai_sse,
)

logger = logging.getLogger(__name__)

# Upstream rate-limit retry policy
RETRYABLE_STATUS = {429, 503}
MAX_RETRIES = 3            # attempts after the first try
MAX_RETRY_DELAY = 30.0    # cap any single backoff wait (seconds)
DEFAULT_RETRY_DELAY = 1.0  # base for exponential backoff when no Retry-After


def _parse_retry_after(value: str) -> Optional[float]:
    """Parse a Retry-After header value (delta-seconds form) into seconds.

    Copilot sends delta-seconds (e.g. "5"). HTTP-date form is not used by the
    upstream, so we only handle the numeric case and ignore anything else.
    """
    value = (value or "").strip()
    if not value:
        return None
    try:
        secs = float(value)
    except ValueError:
        return None
    if secs < 0:
        return None
    return secs


def _retry_delay(resp: "httpx.Response", attempt: int) -> float:
    """Compute how long to wait before the next retry.

    Honors the upstream Retry-After header when present; otherwise falls back
    to exponential backoff (1s, 2s, 4s, …). Always capped at MAX_RETRY_DELAY.
    """
    retry_after = _parse_retry_after(resp.headers.get("retry-after", ""))
    if retry_after is not None:
        return min(retry_after, MAX_RETRY_DELAY)
    return min(DEFAULT_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)


# ── error logging ────────────────────────────────────────────────

def _log_http_error(
    url: str,
    req_headers: dict[str, str],
    req_body: dict | None,
    resp_status: int | None,
    resp_headers: dict | None,
    resp_body: str | None,
    elapsed_ms: float,
    model_id: str = "",
    error: str | None = None,
) -> None:
    """Log full HTTP request and response details for errored requests.

    Redacts the Authorization token and truncates bodies > 2 KiB
    so the log remains compact without losing critical information.
    """
    # --- sanitise request headers ---
    safe_rh = dict(req_headers)
    if "Authorization" in safe_rh:
        auth = safe_rh["Authorization"]
        if auth.startswith("Bearer "):
            safe_rh["Authorization"] = "Bearer …" + auth[-8:]

    # --- request body ---
    req_str = json.dumps(req_body, ensure_ascii=False) if req_body is not None else "(none)"
    if len(req_str) > 2048:
        req_str = req_str[:2048] + " …(truncated)"

    # --- response body ---
    resp_str = resp_body or "(none)"
    if len(resp_str) > 2048:
        resp_str = resp_str[:2048] + " …(truncated)"

    # --- response headers ---
    safe_resh = dict(resp_headers) if resp_headers else {}

    parts = [
        f"HTTP error  model={model_id}  elapsed={elapsed_ms:.0f}ms",
        f"  >> REQUEST  POST {url}",
        f"  >> headers  {json.dumps(safe_rh, ensure_ascii=False)}",
        f"  >> body     {req_str}",
    ]
    if error:
        parts.append(f"  << ERROR    {error}")
    if resp_status is not None:
        parts.append(f"  << RESPONSE {resp_status}")
        parts.append(f"  << headers  {json.dumps(safe_resh, ensure_ascii=False)}")
        parts.append(f"  << body     {resp_str}")

    logger.error("\n".join(parts))


# Regex for detecting GPT-5+ models that should use Responses API
GPT5_PATTERN = re.compile(r"^gpt-(\d+)")


def should_use_responses_api(model_id: str) -> bool:
    """Check if model should use the OpenAI Responses API instead of Chat Completions.

    From github-copilot.ts:shouldUseResponsesApi:
      GPT-5+ models (except gpt-5-mini) use the Responses API.
    """
    m = GPT5_PATTERN.match(model_id)
    if not m:
        return False
    major = int(m.group(1))
    return major >= 5 and not model_id.startswith("gpt-5-mini")


def detect_vision_request(body: dict) -> bool:
    """Detect if a request contains images (for Copilot-Vision-Request header)."""
    messages = body.get("messages", [])
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in ("image_url", "image", "input_image"):
                        return True
                    # Check nested content in tool results
                    if part.get("type") == "tool_result":
                        nested = part.get("content", [])
                        if isinstance(nested, list):
                            for n in nested:
                                if isinstance(n, dict) and n.get("type") == "image":
                                    return True
        elif isinstance(content, str):
            # Check for image URLs in string content
            if content.startswith("data:image/") or "data:image/" in content:
                return True

    # Also check Anthropic-style content blocks
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    return True

    return False


# Anthropic fields that Copilot's /v1/messages endpoint rejects
_ANTHROPIC_UNSUPPORTED_FIELDS = [
    "context_management",
]

# Map thinking budget_tokens to Copilot effort levels
_THINKING_BUDGET_TO_EFFORT = [
    (2000, "low"),
    (8000, "medium"),
    (16000, "high"),
    (32000, "xhigh"),
]


def _effort_from_budget(budget_tokens: int) -> str:
    """Map a thinking budget to a Copilot effort level."""
    for threshold, effort in _THINKING_BUDGET_TO_EFFORT:
        if budget_tokens <= threshold:
            return effort
    return "xhigh"


def _sanitize_anthropic_body(body: dict, model: "CopilotModel | None" = None) -> dict:
    """Remove / convert Anthropic-API fields that Copilot's /v1/messages doesn't support.

    Returns a (possibly modified) copy of the body.
    """
    dropped = [k for k in _ANTHROPIC_UNSUPPORTED_FIELDS if k in body]
    if dropped:
        body = {k: v for k, v in body.items() if k not in _ANTHROPIC_UNSUPPORTED_FIELDS}
        logger.debug(f"Stripped unsupported Anthropic fields: {dropped}")

    # ── normalise system: accept "system" role inside messages ────
    # Many clients (especially OpenAI SDK users) send system prompts as
    # messages with role="system". The Anthropic Messages API requires a
    # top-level `system` parameter instead, so we extract and promote them.
    messages = body.get("messages")
    if isinstance(messages, list):
        system_contents: list[dict] = []
        cleaned_messages: list[dict] = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, str):
                    system_contents.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    system_contents.extend(content)
            else:
                cleaned_messages.append(msg)
        if system_contents:
            # Merge with existing top-level system parameter if present
            existing = body.get("system")
            if existing is None:
                body["system"] = system_contents
            elif isinstance(existing, str):
                body["system"] = [{"type": "text", "text": existing}] + system_contents
            elif isinstance(existing, list):
                body["system"] = existing + system_contents
            body["messages"] = cleaned_messages
            logger.debug(
                f"Promoted {len(system_contents)} system content block(s) "
                f"from messages[] to top-level system parameter"
            )

    # Strip all reasoning fields for models that don't support reasoning at all
    if model and not model.supports_reasoning:
        stripped = [k for k in ("thinking", "output_config") if k in body]
        if stripped:
            body = {k: v for k, v in body.items() if k not in ("thinking", "output_config")}
            logger.debug(f"Stripped reasoning fields for {model.id}: {stripped}")
        return body

    # Models that support reasoning but not effort (e.g. haiku-4.5):
    # convert thinking, but always strip output_config (not supported)
    if model and not model.reasoning_efforts:
        if "output_config" in body:
            body = {k: v for k, v in body.items() if k != "output_config"}
            logger.debug(f"Stripped output_config for {model.id} (no effort levels)")
        thinking = body.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            body = dict(body)
            body["thinking"] = {"type": "adaptive"}
            logger.debug(
                f"Converted thinking: enabled → adaptive for {model.id} (no effort)"
            )
        return body

    # Full effort support: convert thinking.type=enabled → adaptive + effort
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = thinking.get("budget_tokens", 16000)
        body = dict(body)
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": _effort_from_budget(budget)}
        logger.debug(
            f"Converted thinking: enabled(budget={budget}) → "
            f"adaptive(effort={_effort_from_budget(budget)})"
        )

    return body


class ProxyHandler:
    """Handles proxying LLM requests to GitHub Copilot API."""

    def __init__(self, config: Config, auth: AuthManager, models: ModelStore, usage: UsageTracker):
        self._config = config
        self._auth = auth
        self._models = models
        self._usage = usage
        self._client: Optional[httpx.AsyncClient] = None

    async def startup(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()

    # ── upstream send with rate-limit retry ────────────────────

    async def _post_with_retry(
        self, url: str, body: dict, headers: dict[str, str], model_id: str = ""
    ) -> "httpx.Response":
        """POST to upstream, retrying on 429/503 and transient network errors.

        Returns the final httpx.Response (success or the last error response).
        ReadTimeout is retried once; other RequestErrors propagate to callers.
        """
        assert self._client is not None
        try:
            resp = await self._client.post(url, json=body, headers=headers)
        except httpx.ReadTimeout as e:
            logger.warning(
                f"Upstream read timeout (model={model_id}), retrying once"
            )
            resp = await self._client.post(url, json=body, headers=headers)
        attempt = 0
        while resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
            delay = _retry_delay(resp, attempt)
            logger.warning(
                f"Upstream {resp.status_code} (model={model_id}), "
                f"retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s"
            )
            await resp.aclose()
            await asyncio.sleep(delay)
            resp = await self._client.post(url, json=body, headers=headers)
            attempt += 1
        return resp

    @asynccontextmanager
    async def _stream_with_retry(
        self, url: str, body: dict, headers: dict[str, str], model_id: str = ""
    ):
        """Open an upstream streaming POST, retrying on 429/503 before any body.

        Yields an open httpx.Response inside its stream context. Retry only
        happens before the first byte is read, so callers can still surface a
        clean error if every attempt fails. The caller must check is_success.
        """
        assert self._client is not None
        attempt = 0
        while True:
            cm = self._client.stream("POST", url, json=body, headers=headers)
            try:
                resp = await cm.__aenter__()
            except httpx.ReadTimeout:
                logger.warning(
                    f"Upstream read timeout (model={model_id}, stream), "
                    f"retrying once"
                )
                await cm.__aexit__(None, None, None)
                cm = self._client.stream("POST", url, json=body, headers=headers)
                resp = await cm.__aenter__()
            if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                delay = _retry_delay(resp, attempt)
                logger.warning(
                    f"Upstream {resp.status_code} (model={model_id}, stream), "
                    f"retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s"
                )
                await cm.__aexit__(None, None, None)
                await asyncio.sleep(delay)
                attempt += 1
                continue
            try:
                yield resp
            finally:
                await cm.__aexit__(None, None, None)
            return

    # ── public handlers ────────────────────────────────────────

    async def list_models(self) -> JSONResponse:
        """GET /v1/models — list available models in OpenAI format."""
        models_list = self._models.list_openai_models()
        return JSONResponse({
            "object": "list",
            "data": models_list,
        })

    async def list_anthropic_models(self) -> JSONResponse:
        """GET /v1/models (Anthropic format) — list available models."""
        models_list = self._models.list_anthropic_models()
        return JSONResponse({
            "data": models_list,
        })

    async def list_debug_models(self) -> JSONResponse:
        """GET /v1/models/debug — full model metadata for inspection."""
        models_list = self._models.list_debug_models()
        return JSONResponse({
            "object": "list",
            "data": models_list,
            "refreshed_at": self._models._last_refresh,
        })

    async def list_raw_response(self) -> JSONResponse:
        """GET /v1/models/raw — untouched upstream Copilot /models response."""
        raw = self._models.get_raw_response()
        if not raw:
            return JSONResponse(
                {"error": "No raw response available — models not yet refreshed"},
                status_code=503,
            )
        return JSONResponse(raw)

    async def chat_completions(self, request: Request) -> Response:
        """POST /v1/chat/completions — proxy to GitHub Copilot."""
        return await self._proxy_openai_request(request, "chat")

    async def responses(self, request: Request) -> Response:
        """POST /v1/responses — proxy to GitHub Copilot Responses API."""
        return await self._proxy_openai_request(request, "responses")

    async def messages(self, request: Request) -> Response:
        """POST /v1/messages — proxy Anthropic Messages API request.

        Routing logic:
          - If model supports /v1/messages natively → forward directly
          - Otherwise → convert to OpenAI Chat Completions, proxy, convert response back
        """
        body = await request.json()
        model_id = body.get("model", "")
        model = self._models.get_model(model_id)

        if model and model.supports_anthropic_api:
            # Native Anthropic API support → forward directly
            return await self._proxy_anthropic_native(request, body, model)
        else:
            # No native support → convert Anthropic → OpenAI → Anthropic
            return await self._proxy_anthropic_converted(request, body, model_id)

    # ── internal proxy methods ─────────────────────────────────

    async def _proxy_openai_request(self, request: Request, api_type: str) -> Response:
        """Proxy an OpenAI-format request to GitHub Copilot.

        Args:
            request: The incoming FastAPI request.
            api_type: "chat" for /chat/completions, "responses" for /responses.
        """
        body = await request.json()
        model_id = body.get("model", "")
        model = self._models.get_model(model_id)
        is_stream = body.get("stream", False)

        # Determine the upstream endpoint
        base_url = self._config.api_base_url

        if api_type == "chat":
            # Ensure streaming responses include usage (OpenAI requires this flag)
            if is_stream and "stream_options" not in body:
                body["stream_options"] = {"include_usage": True}
            # If model doesn't support /chat/completions natively but speaks
            # Anthropic, convert OpenAI → Anthropic → proxy → convert back
            if model and "/v1/chat/completions" not in model.supported_endpoints:
                if model.supports_anthropic_api:
                    return await self._proxy_openai_via_anthropic(request, body, model)
            # Check if this model should use Responses API instead
            if model and model.is_gpt5:
                upstream_url = f"{base_url}/responses"
            else:
                upstream_url = f"{base_url}/chat/completions"
        elif api_type == "responses":
            upstream_url = f"{base_url}/responses"
        else:
            upstream_url = f"{base_url}/chat/completions"

        # Build upstream headers
        headers = self._build_upstream_headers(body, request.headers)

        logger.debug(f"Proxying OpenAI {api_type} request to {upstream_url} (model={model_id}, stream={is_stream})")

        return await self._send_upstream(
            upstream_url, headers, body, is_stream,
            model_id=model_id, api_type=api_type,
        )

    async def _proxy_anthropic_native(
        self, request: Request, body: dict, model: CopilotModel
    ) -> Response:
        """Forward an Anthropic request directly to Copilot's /v1/messages endpoint."""
        base_url = self._config.api_base_url
        upstream_url = f"{base_url}/v1/messages"

        headers = self._build_upstream_headers(body, request.headers)
        headers["anthropic-version"] = "2023-06-01"
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        # Strip Anthropic fields that Copilot's /v1/messages doesn't support
        body = _sanitize_anthropic_body(body, model)

        is_stream = body.get("stream", False)
        model_id = body.get("model", "")

        logger.debug(f"Proxying Anthropic native request to {upstream_url} (model={model_id}, stream={is_stream})")

        return await self._send_upstream(
            upstream_url, headers, body, is_stream,
            model_id=model_id, api_type="anthropic",
        )

    async def _proxy_anthropic_converted(
        self, request: Request, body: dict, model_id: str
    ) -> Response:
        """Convert Anthropic request to OpenAI format, proxy, convert response back."""
        # Convert request
        openai_body = anthropic_to_openai_request(body)
        is_stream = body.get("stream", False)

        # Determine endpoint
        base_url = self._config.api_base_url
        model = self._models.get_model(model_id)
        if model and model.is_gpt5:
            upstream_url = f"{base_url}/responses"
        else:
            upstream_url = f"{base_url}/chat/completions"

        headers = self._build_upstream_headers(openai_body, request.headers)

        logger.debug(
            f"Proxying Anthropic→OpenAI converted request to {upstream_url} "
            f"(model={model_id}, stream={is_stream})"
        )

        if is_stream:
            # Streaming: convert SSE events on the fly
            return await self._send_upstream_with_anthropic_ssE_conversion(
                upstream_url, headers, openai_body, model_id
            )
        else:
            # Non-streaming: convert response
            return await self._send_upstream_and_convert_response(
                upstream_url, headers, openai_body, model_id
            )

    async def _proxy_openai_via_anthropic(
        self, request: Request, body: dict, model: CopilotModel
    ) -> Response:
        """Convert OpenAI request to Anthropic, proxy, convert response back to OpenAI.

        Used when a model only supports /v1/messages natively (e.g. Claude on Copilot)
        but the client is speaking OpenAI Chat Completions.
        """
        model_id = body.get("model", "")
        is_stream = body.get("stream", False)

        # Convert OpenAI → Anthropic request
        anth_body = openai_to_anthropic_request(body)
        anth_body["stream"] = is_stream
        anth_body = _sanitize_anthropic_body(anth_body, model)

        base_url = self._config.api_base_url
        upstream_url = f"{base_url}/v1/messages"

        headers = self._build_upstream_headers(anth_body, request.headers)
        headers["anthropic-version"] = "2023-06-01"
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        logger.debug(
            f"Proxying OpenAI→Anthropic converted request to {upstream_url} "
            f"(model={model_id}, stream={is_stream})"
        )

        if is_stream:
            return await self._send_upstream_with_openai_sse_conversion(
                upstream_url, headers, anth_body, model_id
            )
        else:
            return await self._send_anthropic_and_convert_to_openai(
                upstream_url, headers, anth_body, model_id
            )

    async def _send_anthropic_and_convert_to_openai(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
        model_id: str,
    ) -> Response:
        """Send Anthropic request and convert response to OpenAI format."""
        if not self._client:
            raise RuntimeError("ProxyHandler not started")

        t0 = time.monotonic()
        try:
            resp = await self._post_with_retry(url, body, headers, model_id)
            elapsed_ms = (time.monotonic() - t0) * 1000
            if resp.is_success:
                anth_response = resp.json()
                self._record_usage(model_id, anth_response, "anthropic")
                openai_response = anthropic_to_openai_response(anth_response, model_id)
                return JSONResponse(openai_response)
            else:
                _log_http_error(
                    url, headers, body,
                    resp_status=resp.status_code,
                    resp_headers=dict(resp.headers),
                    resp_body=resp.text[:2048],
                    elapsed_ms=elapsed_ms,
                    model_id=model_id,
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type="application/json",
                )
        except httpx.RequestError as e:
            _log_http_error(
                url, headers, body,
                resp_status=None, resp_headers=None, resp_body=None,
                elapsed_ms=(time.monotonic() - t0) * 1000,
                model_id=model_id, error=str(e),
            )
            return JSONResponse(
                {"error": {"type": "upstream_error", "message": str(e)}},
                status_code=502,
            )

    async def _send_upstream_with_openai_sse_conversion(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
        model_id: str,
    ) -> StreamingResponse:
        """Send Anthropic streaming request and convert SSE to OpenAI format."""
        async def event_generator():
            if not self._client:
                yield f"data: {{\"error\":\"Client not started\"}}\n\n"
                return

            t0 = time.monotonic()
            try:
                async with self._stream_with_retry(url, body, headers, model_id) as resp:
                    if not resp.is_success:
                        error_text = await resp.aread()
                        err_body = error_text.decode()[:2048]
                        _log_http_error(
                            url, headers, body,
                            resp_status=resp.status_code,
                            resp_headers=dict(resp.headers),
                            resp_body=err_body,
                            elapsed_ms=(time.monotonic() - t0) * 1000,
                            model_id=model_id,
                        )
                        yield f"data: {{\"error\":{json.dumps(err_body[:500])}}}\n\n"
                        return

                    buffer = ""
                    async for chunk in resp.aiter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue

                            if line.startswith("data: "):
                                self._check_sse_line(line, model_id, "anthropic")

                            converted = anthropic_sse_to_openai_sse(line, model_id)
                            if converted:
                                yield converted

            except httpx.RequestError as e:
                _log_http_error(
                    url, headers, body,
                    resp_status=None, resp_headers=None, resp_body=None,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    model_id=model_id, error=str(e),
                )
                yield f"data: {{\"error\":{json.dumps(str(e))}}}\n\n"
            except Exception as e:
                _log_http_error(
                    url, headers, body,
                    resp_status=None, resp_headers=None, resp_body=None,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    model_id=model_id, error=f"unexpected: {e}",
                )
                yield f"data: {{\"error\":{json.dumps(str(e))}}}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _send_upstream(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
        is_stream: bool,
        model_id: str = "",
        api_type: str = "chat",
    ) -> Response:
        """Send request to upstream and return the response.

        For streaming requests, returns a StreamingResponse that proxies SSE.
        For non-streaming, returns a JSONResponse and records token usage.
        """
        if not self._client:
            raise RuntimeError("ProxyHandler not started")

        if is_stream:
            req_headers = {**headers}
            return StreamingResponse(
                self._stream_upstream(url, req_headers, body, model_id, api_type),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            t0 = time.monotonic()
            try:
                resp = await self._post_with_retry(url, body, headers, model_id)
            except httpx.RequestError as e:
                _log_http_error(
                    url, headers, body,
                    resp_status=None, resp_headers=None, resp_body=None,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    model_id=model_id, error=str(e),
                )
                return JSONResponse(
                    {"error": {"type": "upstream_error", "message": str(e)}},
                    status_code=502,
                )
            elapsed_ms = (time.monotonic() - t0) * 1000
            if resp.is_success:
                try:
                    self._record_usage(model_id, resp.json(), api_type)
                except Exception:
                    pass
            else:
                _log_http_error(
                    url, headers, body,
                    resp_status=resp.status_code,
                    resp_headers=dict(resp.headers),
                    resp_body=resp.text[:2048],
                    elapsed_ms=elapsed_ms,
                    model_id=model_id,
                )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers={
                    "Content-Type": "application/json",
                    **dict(resp.headers),
                },
            )

    async def _send_upstream_and_convert_response(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
        model_id: str,
    ) -> Response:
        """Send non-streaming request and convert OpenAI response to Anthropic format."""
        if not self._client:
            raise RuntimeError("ProxyHandler not started")

        t0 = time.monotonic()
        try:
            resp = await self._post_with_retry(url, body, headers, model_id)
            elapsed_ms = (time.monotonic() - t0) * 1000
            if resp.is_success:
                openai_response = resp.json()
                # Record usage from the OpenAI response
                self._record_usage(model_id, openai_response, "chat")
                # Convert to Anthropic format
                anthropic_response = openai_to_anthropic_response(openai_response, model_id)
                return JSONResponse(anthropic_response)
            else:
                # Pass through errors
                _log_http_error(
                    url, headers, body,
                    resp_status=resp.status_code,
                    resp_headers=dict(resp.headers),
                    resp_body=resp.text[:2048],
                    elapsed_ms=elapsed_ms,
                    model_id=model_id,
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type="application/json",
                )
        except httpx.RequestError as e:
            _log_http_error(
                url, headers, body,
                resp_status=None, resp_headers=None, resp_body=None,
                elapsed_ms=(time.monotonic() - t0) * 1000,
                model_id=model_id, error=str(e),
            )
            return JSONResponse(
                {"error": {"type": "upstream_error", "message": str(e)}},
                status_code=502,
            )

    async def _send_upstream_with_anthropic_ssE_conversion(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
        model_id: str,
    ) -> StreamingResponse:
        """Send streaming request and convert OpenAI SSE to Anthropic SSE."""
        request_id = str(uuid.uuid4())[:8]

        async def event_generator():
            if not self._client:
                yield f"event: error\ndata: {{\"type\":\"error\",\"error\":{{\"message\":\"Client not started\"}}}}\n\n"
                return

            t0 = time.monotonic()
            try:
                async with self._stream_with_retry(url, body, headers, model_id) as resp:
                    if not resp.is_success:
                        # Non-streaming error
                        error_text = await resp.aread()
                        err_body = error_text.decode()[:2048]
                        _log_http_error(
                            url, headers, body,
                            resp_status=resp.status_code,
                            resp_headers=dict(resp.headers),
                            resp_body=err_body,
                            elapsed_ms=(time.monotonic() - t0) * 1000,
                            model_id=model_id,
                        )
                        yield f"event: error\ndata: {{\"type\":\"error\",\"error\":{{\"message\":{json.dumps(err_body[:500])}}}}}\n\n"
                        return

                    buffer = ""
                    async for chunk in resp.aiter_text():
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                # Empty line between events in Anthropic format — skip for OpenAI SSE
                                continue

                            if line.startswith("data: "):
                                self._check_sse_line(line, model_id, "chat")
                                converted = openai_sse_to_anthropic_sse(line, model_id)
                                if converted:
                                    yield converted
                            elif line:
                                # Pass through non-data lines
                                yield line + "\n"

                    # Emit final message_stop if stream ended without [DONE]
                    yield finalize_anthropic_stream(request_id)

            except httpx.RequestError as e:
                _log_http_error(
                    url, headers, body,
                    resp_status=None, resp_headers=None, resp_body=None,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    model_id=model_id, error=str(e),
                )
                yield f"event: error\ndata: {{\"type\":\"error\",\"error\":{{\"message\":{json.dumps(str(e))}}}}}\n\n"
            except Exception as e:
                _log_http_error(
                    url, headers, body,
                    resp_status=None, resp_headers=None, resp_body=None,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    model_id=model_id, error=f"unexpected: {e}",
                )
                yield f"event: error\ndata: {{\"type\":\"error\",\"error\":{{\"message\":{json.dumps(str(e))}}}}}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _stream_upstream(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
        model_id: str = "",
        api_type: str = "chat",
    ):
        """Stream upstream SSE response, tracking usage from the final event.

        OpenAI SSE: usage in the last data: chunk before [DONE].
        Anthropic SSE: usage in the message_stop event.
        """
        if not self._client:
            yield f"data: {{\"error\":\"Client not started\"}}\n\n"
            return

        t0 = time.monotonic()
        try:
            async with self._stream_with_retry(url, body, headers, model_id) as resp:
                if not resp.is_success:
                    error_text = await resp.aread()
                    err_body = error_text.decode()[:2048]
                    _log_http_error(
                        url, headers, body,
                        resp_status=resp.status_code,
                        resp_headers=dict(resp.headers),
                        resp_body=err_body,
                        elapsed_ms=(time.monotonic() - t0) * 1000,
                        model_id=model_id,
                    )
                    yield f"data: {{\"error\":{json.dumps(err_body[:500])}}}\n\n"
                    return

                text_buf = ""
                async for chunk in resp.aiter_bytes():
                    yield chunk  # pass through unchanged
                    # Decode in parallel to scan for usage
                    text_buf += chunk.decode(errors="ignore")
                    while "\n" in text_buf:
                        line, text_buf = text_buf.split("\n", 1)
                        self._check_sse_line(line.strip(), model_id, api_type)
                # Flush remaining
                if text_buf.strip():
                    self._check_sse_line(text_buf.strip(), model_id, api_type)

        except httpx.RequestError as e:
            _log_http_error(
                url, headers, body,
                resp_status=None, resp_headers=None, resp_body=None,
                elapsed_ms=(time.monotonic() - t0) * 1000,
                model_id=model_id, error=str(e),
            )
            yield f"data: {{\"error\":{json.dumps(str(e))}}}\n\n"

    # ── usage tracking ─────────────────────────────────────────

    def _record_usage(self, model_id: str, response_body: dict, api_type: str) -> None:
        """Record token usage from an upstream response body.

        api_type: "chat", "responses", or "anthropic"

        Pricing differs by provider:
          - Anthropic: cache reads at model.cache_read_price (deep discount, ~10%)
          - OpenAI:    cache reads at 50% of input price (no separate cache_price API)
        """
        if not model_id:
            return

        model = self._models.get_model(model_id)
        input_price = model.input_price if model else 0.0
        output_price = model.output_price if model else 0.0

        if api_type == "anthropic":
            # Anthropic: use the model's explicit cache_read_price from Copilot
            cache_read_price = model.cache_read_price if model else 0.0
            in_tok, out_tok, cache_r, cache_w, reasoning = extract_anthropic_usage(response_body)
            asyncio.ensure_future(self._usage.record(
                model_id, input_tokens=in_tok, output_tokens=out_tok,
                cache_read_tokens=cache_r, cache_write_tokens=cache_w,
                reasoning_tokens=reasoning,
                input_price_per_m=input_price, output_price_per_m=output_price,
                cache_read_price_per_m=cache_read_price,
            ))
        else:
            # OpenAI: cache hits are 50% of input price
            # If Copilot provides a cache_price, use it; otherwise default to input * 0.5
            cache_price = model.cache_read_price if model else 0.0
            if cache_price <= 0:
                cache_price = input_price * 0.5

            if api_type == "responses":
                in_tok, out_tok, cache_r, cache_w, reasoning = extract_responses_usage(response_body)
            else:
                in_tok, out_tok, cache_r, cache_w, reasoning = extract_openai_usage(response_body)

            asyncio.ensure_future(self._usage.record(
                model_id, input_tokens=in_tok, output_tokens=out_tok,
                cache_read_tokens=cache_r, cache_write_tokens=cache_w,
                reasoning_tokens=reasoning,
                input_price_per_m=input_price, output_price_per_m=output_price,
                cache_read_price_per_m=cache_price,
            ))

    def _check_sse_line(self, line: str, model_id: str, api_type: str) -> None:
        """Check an SSE line for usage data and record it.

        OpenAI SSE:  data: {..."usage":{"prompt_tokens":...}}
        Anthropic SSE: event: message_stop → next data: line has usage
        """
        if not model_id or not line:
            return

        # Anthropic: track event type for message_stop
        if api_type == "anthropic":
            if line.startswith("event: ") and "message_stop" in line:
                self._anth_sse_expecting_usage = True
                return
            if line.startswith("data: ") and getattr(self, "_anth_sse_expecting_usage", False):
                self._anth_sse_expecting_usage = False
                try:
                    data = json.loads(line[6:])
                    self._record_usage(model_id, data, "anthropic")
                except (json.JSONDecodeError, KeyError):
                    pass
                return

        # OpenAI: look for usage in data: lines
        if line.startswith("data: ") and "[DONE]" not in line:
            try:
                data = json.loads(line[6:])
                if "usage" in data:
                    self._record_usage(model_id, data, api_type)
            except (json.JSONDecodeError, KeyError):
                pass

    # ── header helpers ─────────────────────────────────────────

    def _build_upstream_headers(
        self, body: dict, incoming_headers: dict
    ) -> dict[str, str]:
        """Build headers for upstream GitHub Copilot API requests.

        Based on copilot.ts auth loader and chat.headers hook.
        """
        headers: dict[str, str] = {}

        # Auth
        token = self._auth.get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # Required GitHub Copilot headers
        headers["X-GitHub-Api-Version"] = API_VERSION
        headers["User-Agent"] = USER_AGENT
        headers["Openai-Intent"] = "conversation-edits"

        # x-initiator: detect if request is from an agent
        # Default to "user"; check for agent markers in the request
        x_initiator = incoming_headers.get("x-initiator", "user")
        headers["x-initiator"] = x_initiator

        # Vision request detection
        if detect_vision_request(body):
            headers["Copilot-Vision-Request"] = "true"

        # Content-Type
        headers["Content-Type"] = "application/json"

        return headers
