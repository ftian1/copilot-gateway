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
import uuid
from typing import Optional

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse, JSONResponse

from config import API_VERSION, USER_AGENT, Config
from auth import AuthManager
from models import ModelStore, CopilotModel
from convert import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
    openai_sse_to_anthropic_sse,
    finalize_anthropic_stream,
)

logger = logging.getLogger(__name__)

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


class ProxyHandler:
    """Handles proxying LLM requests to GitHub Copilot API."""

    def __init__(self, config: Config, auth: AuthManager, models: ModelStore):
        self._config = config
        self._auth = auth
        self._models = models
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
            upstream_url, headers, body, is_stream
        )

    async def _proxy_anthropic_native(
        self, request: Request, body: dict, model: CopilotModel
    ) -> Response:
        """Forward an Anthropic request directly to Copilot's /v1/messages endpoint."""
        base_url = self._config.api_base_url
        # Anthropic Messages API is at /v1 on the Copilot API
        upstream_url = f"{base_url}/v1/messages"

        headers = self._build_upstream_headers(body, request.headers)
        # Add Anthropic-specific headers
        headers["anthropic-version"] = "2023-06-01"
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        is_stream = body.get("stream", False)
        model_id = body.get("model", "")

        logger.debug(f"Proxying Anthropic native request to {upstream_url} (model={model_id}, stream={is_stream})")

        return await self._send_upstream(upstream_url, headers, body, is_stream)

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

    async def _send_upstream(
        self,
        url: str,
        headers: dict[str, str],
        body: dict,
        is_stream: bool,
    ) -> Response:
        """Send request to upstream and return the response.

        For streaming requests, returns a StreamingResponse that proxies SSE.
        For non-streaming, returns a JSONResponse.
        """
        if not self._client:
            raise RuntimeError("ProxyHandler not started")

        if is_stream:
            # For streaming, we need to stream the response
            req_headers = {**headers}
            # Build the upstream request
            return StreamingResponse(
                self._stream_upstream(url, req_headers, body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            resp = await self._client.post(url, json=body, headers=headers)
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

        try:
            resp = await self._client.post(url, json=body, headers=headers)
            if resp.is_success:
                openai_response = resp.json()
                anthropic_response = openai_to_anthropic_response(openai_response, model_id)
                return JSONResponse(anthropic_response)
            else:
                # Pass through errors
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type="application/json",
                )
        except httpx.RequestError as e:
            logger.error(f"Upstream request failed: {e}")
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

            try:
                async with self._client.stream("POST", url, json=body, headers=headers) as resp:
                    if not resp.is_success:
                        # Non-streaming error
                        error_text = await resp.aread()
                        yield f"event: error\ndata: {{\"type\":\"error\",\"error\":{{\"message\":{json.dumps(error_text.decode()[:500])}}}}}\n\n"
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
                                converted = openai_sse_to_anthropic_sse(line, model_id)
                                if converted:
                                    yield converted
                            elif line:
                                # Pass through non-data lines
                                yield line + "\n"

                    # Emit final message_stop if stream ended without [DONE]
                    yield finalize_anthropic_stream(request_id)

            except httpx.RequestError as e:
                logger.error(f"Upstream streaming request failed: {e}")
                yield f"event: error\ndata: {{\"type\":\"error\",\"error\":{{\"message\":{json.dumps(str(e))}}}}}\n\n"
            except Exception as e:
                logger.exception(f"Unexpected error in SSE stream: {e}")
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
    ):
        """Stream upstream SSE response directly to the client (OpenAI→OpenAI pass-through)."""
        if not self._client:
            yield f"data: {{\"error\":\"Client not started\"}}\n\n"
            return

        try:
            async with self._client.stream("POST", url, json=body, headers=headers) as resp:
                if not resp.is_success:
                    error_text = await resp.aread()
                    yield f"data: {{\"error\":{json.dumps(error_text.decode()[:500])}}}\n\n"
                    return

                async for chunk in resp.aiter_bytes():
                    yield chunk

        except httpx.RequestError as e:
            logger.error(f"Upstream stream failed: {e}")
            yield f"data: {{\"error\":{json.dumps(str(e))}}}\n\n"

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
