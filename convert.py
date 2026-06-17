"""
Anthropic Messages API ↔ OpenAI Chat Completions API format conversion.

Handles:
  - Request conversion: Anthropic → OpenAI (for models without native /v1/messages support)
  - Response conversion: OpenAI → Anthropic (both streaming SSE and non-streaming)
  - Streaming SSE event transformation

Based on:
  - Anthropic Messages API: https://docs.anthropic.com/en/api/messages
  - OpenAI Chat Completions: https://platform.openai.com/docs/api-reference/chat
"""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Anthropic → OpenAI (Request Conversion)
# ═══════════════════════════════════════════════════════════════


def anthropic_to_openai_request(anthropic_body: dict) -> dict:
    """Convert an Anthropic Messages API request to OpenAI Chat Completions format.

    Args:
        anthropic_body: The parsed JSON body of an Anthropic /v1/messages request.

    Returns:
        A dict suitable for POST to /v1/chat/completions.
    """
    openai_body: dict[str, Any] = {}

    # ── model ──
    openai_body["model"] = anthropic_body.get("model", "")

    # ── messages ──
    messages: list[dict] = []

    # System prompt: in Anthropic it's a top-level field (string or list)
    system = anthropic_body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    messages.append({"role": "system", "content": block.get("text", "")})
                # Other system block types (e.g. cache_control) → best-effort text extraction
                elif isinstance(block, dict):
                    text = block.get("text", "") or json.dumps(block)
                    messages.append({"role": "system", "content": text})

    # Anthropic messages → OpenAI messages
    for msg in anthropic_body.get("messages", []):
        role = msg.get("role", "user")

        # Map Anthropic roles to OpenAI roles
        if role == "assistant":
            openai_role = "assistant"
        elif role == "user":
            openai_role = "user"
        else:
            openai_role = "user"  # fallback

        content = msg.get("content")

        if isinstance(content, str):
            # Simple string content
            messages.append({"role": openai_role, "content": content})
        elif isinstance(content, list):
            # Content blocks → OpenAI format
            openai_msg = _convert_content_blocks_to_openai(openai_role, content)
            messages.append(openai_msg)
        else:
            # Fallback
            messages.append({"role": openai_role, "content": str(content)})

    openai_body["messages"] = messages

    # ── max_tokens → max_completion_tokens (preferred) or max_tokens ──
    max_tokens = anthropic_body.get("max_tokens")
    if max_tokens is not None:
        openai_body["max_completion_tokens"] = max_tokens

    # ── temperature ──
    if "temperature" in anthropic_body:
        openai_body["temperature"] = anthropic_body["temperature"]

    # ── top_p ──
    if "top_p" in anthropic_body:
        openai_body["top_p"] = anthropic_body["top_p"]

    # ── top_k (Anthropic-specific, no OpenAI equivalent; drop) ──

    # ── stop_sequences → stop ──
    stop = anthropic_body.get("stop_sequences")
    if stop:
        # OpenAI stop can be string or array of up to 4
        if isinstance(stop, list) and len(stop) == 1:
            openai_body["stop"] = stop[0]
        elif isinstance(stop, list):
            openai_body["stop"] = stop[:4]
        else:
            openai_body["stop"] = str(stop)

    # ── tools ──
    tools = anthropic_body.get("tools")
    if tools:
        openai_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                openai_tool = _convert_anthropic_tool_to_openai(tool)
                if openai_tool:
                    openai_tools.append(openai_tool)
        if openai_tools:
            openai_body["tools"] = openai_tools
            # Anthropic's tool_choice → OpenAI tool_choice
            tool_choice = anthropic_body.get("tool_choice")
            if tool_choice:
                openai_body["tool_choice"] = _convert_tool_choice_to_openai(tool_choice)

    # ── stream ──
    if anthropic_body.get("stream"):
        openai_body["stream"] = True
        openai_body["stream_options"] = {"include_usage": True}

    # ── metadata (user_id) ──
    metadata = anthropic_body.get("metadata")
    if isinstance(metadata, dict) and metadata.get("user_id"):
        openai_body["user"] = metadata["user_id"]

    return openai_body


def _convert_content_blocks_to_openai(role: str, blocks: list[dict]) -> dict:
    """Convert Anthropic content blocks to an OpenAI message dict."""
    openai_content: list[dict] = []
    tool_calls: list[dict] = []

    for block in blocks:
        block_type = block.get("type", "")

        if block_type == "text":
            openai_content.append({"type": "text", "text": block.get("text", "")})

        elif block_type == "image":
            source = block.get("source", {})
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            url = f"data:{media_type};base64,{data}"
            openai_content.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "auto"},
            })

        elif block_type == "tool_use":
            # Assistant tool use → OpenAI tool_calls
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

        elif block_type == "tool_result":
            # Tool result → OpenAI tool message
            # This should be handled at the message level, not content level
            tc_content = block.get("content", "")
            if isinstance(tc_content, list):
                # Extract text from content blocks
                texts = []
                for c in tc_content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        texts.append(c.get("text", ""))
                tc_content = "\n".join(texts) if texts else json.dumps(tc_content)
            elif not isinstance(tc_content, str):
                tc_content = json.dumps(tc_content)

            openai_content.append({
                "type": "text",
                "text": f"[Tool result for {block.get('tool_use_id', 'unknown')}]: {tc_content}",
            })

        elif block_type == "thinking":
            # Thinking blocks → skip or convert to text
            thinking_text = block.get("thinking", "")
            if thinking_text:
                openai_content.append({
                    "type": "text",
                    "text": f"[Thinking]: {thinking_text}",
                })

        else:
            # Unknown block type → best effort
            logger.debug(f"Unknown Anthropic content block type: {block_type}")
            openai_content.append({
                "type": "text",
                "text": json.dumps(block),
            })

    # Build the message
    if tool_calls and role == "assistant":
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": openai_content[0]["text"] if openai_content and len(openai_content) == 1 and openai_content[0]["type"] == "text" else None,
        }
        if msg["content"] is None and openai_content:
            msg["content"] = json.dumps([c for c in openai_content if c["type"] == "text"])
        if not msg["content"]:
            msg["content"] = None
        msg["tool_calls"] = tool_calls
        return msg
    elif role == "assistant" and openai_content:
        # Assistant with text content
        if len(openai_content) == 1 and openai_content[0]["type"] == "text":
            return {"role": "assistant", "content": openai_content[0]["text"]}
        else:
            return {"role": "assistant", "content": json.dumps(openai_content)}

    # User message
    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": role, "content": openai_content[0]["text"]}
    else:
        return {"role": role, "content": openai_content}


def _convert_anthropic_tool_to_openai(tool: dict) -> Optional[dict]:
    """Convert an Anthropic tool definition to OpenAI format."""
    tool_type = tool.get("type", "")

    if tool_type == "custom":
        # Anthropic tool: {name, description, input_schema}
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }
    elif tool_type == "computer_20250124" or tool_type.startswith("computer_"):
        # Computer use → skip (no OpenAI equivalent)
        return None
    elif tool_type == "bash_20250124" or tool_type.startswith("bash_"):
        # Bash tool → skip (no OpenAI equivalent in standard API)
        return None
    elif tool_type == "text_editor_20250124" or tool_type.startswith("text_editor_"):
        # Text editor → skip
        return None

    # Unknown tool type — try to convert generically
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", tool_type),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


def _convert_tool_choice_to_openai(tool_choice: Any) -> Any:
    """Convert Anthropic tool_choice to OpenAI format."""
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type", "")
        if choice_type == "auto":
            return "auto"
        elif choice_type == "any":
            return "required"
        elif choice_type == "tool" and tool_choice.get("name"):
            return {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
        return "auto"
    elif isinstance(tool_choice, str):
        if tool_choice == "auto":
            return "auto"
        elif tool_choice == "any":
            return "required"
        return "auto"
    return "auto"


# ═══════════════════════════════════════════════════════════════
# OpenAI → Anthropic (Response Conversion)
# ═══════════════════════════════════════════════════════════════


def openai_to_anthropic_response(openai_body: dict, model_id: str = "") -> dict:
    """Convert an OpenAI Chat Completions response to Anthropic Messages format.

    Args:
        openai_body: Parsed JSON response from OpenAI /v1/chat/completions.
        model_id: The model ID for the response.

    Returns:
        An Anthropic Messages API response dict.
    """
    choice = (openai_body.get("choices") or [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")
    usage = openai_body.get("usage", {})

    # Build Anthropic content blocks
    content: list[dict] = []

    # Text content
    text_content = message.get("content")
    if text_content:
        if isinstance(text_content, str):
            content.append({"type": "text", "text": text_content})
        elif isinstance(text_content, list):
            for part in text_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    content.append({"type": "text", "text": part.get("text", "")})

    # Tool calls → tool_use blocks
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            tool_input = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            tool_input = {"_raw": func.get("arguments", "")}

        content.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": tool_input,
        })

    # Map finish reason
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",  # best approximation
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    # Build the Anthropic response
    response: dict[str, Any] = {
        "id": f"msg_{openai_body.get('id', '')}",
        "type": "message",
        "role": "assistant",
        "model": model_id or openai_body.get("model", ""),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) if isinstance(usage.get("prompt_tokens_details"), dict) else 0,
            "cache_read_input_tokens": 0,
        },
    }

    return response


# ═══════════════════════════════════════════════════════════════
# Streaming SSE Conversion
# ═══════════════════════════════════════════════════════════════


def openai_sse_to_anthropic_sse(line: str, model_id: str = "") -> Optional[str]:
    """Convert a single OpenAI SSE line to an Anthropic SSE event.

    OpenAI streaming format:
      data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"..."},"index":0}]}
      data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"tool_calls":[...]},"index":0}]}
      data: {"id":"...","object":"chat.completion.chunk","choices":[{"finish_reason":"stop","index":0}]}
      data: [DONE]

    Anthropic streaming format:
      event: message_start
      data: {"type":"message_start","message":{...}}

      event: content_block_start
      data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"..."}}

      event: content_block_stop
      data: {"type":"content_block_stop","index":0}

      event: message_delta
      data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{...}}

      event: message_stop
      data: {"type":"message_stop"}
    """
    if not line.startswith("data: "):
        return line  # Pass through non-data lines

    data_str = line[6:]  # Strip "data: " prefix

    if data_str.strip() == "[DONE]":
        # Emit message_stop event
        return (
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n'
        )

    try:
        chunk = json.loads(data_str)
    except json.JSONDecodeError:
        return line  # Pass through unparseable lines

    # OpenAI chunk → Anthropic SSE events
    events = _convert_openai_chunk_to_anthropic_events(chunk, model_id)
    return events


# Track state for SSE conversion
_SSE_STATE: dict[str, Any] = {}


def _get_sse_state(key: str) -> dict:
    """Get per-request SSE conversion state."""
    return _SSE_STATE.setdefault(key, {
        "message_started": False,
        "content_block_started": False,
        "content_block_index": 0,
        "tool_call_blocks": {},  # index → {id, name, started, input}
        "message_id": "",
    })


def _clear_sse_state(key: str) -> None:
    """Clear SSE state for a request."""
    _SSE_STATE.pop(key, None)


def _convert_openai_chunk_to_anthropic_events(chunk: dict, model_id: str) -> str:
    """Convert a single OpenAI streaming chunk to Anthropic SSE events.

    Returns a string of one or more SSE event blocks.
    """
    # We use a simple counter-based key for state. In production, use request ID.
    request_key = chunk.get("id", "default")
    state = _get_sse_state(request_key)

    events: list[str] = []
    choices = chunk.get("choices", [])
    usage = chunk.get("usage")

    # ── message_start (first chunk only) ──
    if not state["message_started"]:
        state["message_started"] = True
        state["message_id"] = f"msg_{chunk.get('id', '')}"
        events.append(
            f"event: message_start\n"
            f'data: {{"type":"message_start","message":{{"id":"{state["message_id"]}","type":"message","role":"assistant","model":"{model_id or chunk.get("model", "")}","content":[],"stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":0,"output_tokens":0}}}}}}\n'
        )

    for choice in choices:
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        index = choice.get("index", 0)

        # ── Text content delta ──
        text_delta = delta.get("content")
        if text_delta:
            if not state["content_block_started"]:
                state["content_block_started"] = True
                events.append(
                    f"event: content_block_start\n"
                    f'data: {{"type":"content_block_start","index":{state["content_block_index"]},"content_block":{{"type":"text","text":""}}}}\n'
                )
            # Escape the text for JSON
            escaped_text = json.dumps(text_delta)
            events.append(
                f"event: content_block_delta\n"
                f'data: {{"type":"content_block_delta","index":{state["content_block_index"]},"delta":{{"type":"text_delta","text":{escaped_text}}}}}\n'
            )

        # ── Tool call delta ──
        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            tc_index = tc.get("index", 0)
            tc_id = tc.get("id", "")
            func = tc.get("function", {})

            if tc_index not in state["tool_call_blocks"]:
                # New tool call block starting
                state["tool_call_blocks"][tc_index] = {
                    "id": tc_id,
                    "name": func.get("name", ""),
                    "started": False,
                    "arguments": "",
                }

            tc_state = state["tool_call_blocks"][tc_index]

            if tc_id:
                tc_state["id"] = tc_id
            if func.get("name"):
                tc_state["name"] = func.get("name")

            if not tc_state["started"] and tc_state["name"]:
                # Start a new content block for this tool use
                tc_state["started"] = True
                content_block_index = state["content_block_index"] + len(state["tool_call_blocks"])
                state.setdefault("tc_block_idx", {})[tc_index] = content_block_index

                events.append(
                    f"event: content_block_start\n"
                    f'data: {{"type":"content_block_start","index":{content_block_index},"content_block":{{"type":"tool_use","id":"{tc_state["id"]}","name":"{tc_state["name"]}","input":{{}}}}}}\n'
                )

            if func.get("arguments"):
                tc_state["arguments"] += func["arguments"]
                # Try to parse as incremental JSON; if valid, emit delta
                block_idx = state.get("tc_block_idx", {}).get(tc_index, 0)
                try:
                    parsed = json.loads(tc_state["arguments"])
                    escaped_args = json.dumps(parsed)
                    events.append(
                        f"event: content_block_delta\n"
                        f'data: {{"type":"content_block_delta","index":{block_idx},"delta":{{"type":"input_json_delta","partial_json":{escaped_args}}}}}\n'
                    )
                except json.JSONDecodeError:
                    # Partial JSON — emit as partial
                    events.append(
                        f"event: content_block_delta\n"
                        f'data: {{"type":"content_block_delta","index":{block_idx},"delta":{{"type":"input_json_delta","partial_json":{json.dumps(tc_state["arguments"])}}}}}\n'
                    )

        # ── Finish reason ──
        if finish_reason:
            # Stop all active content blocks
            if state["content_block_started"]:
                events.append(
                    f"event: content_block_stop\n"
                    f'data: {{"type":"content_block_stop","index":{state["content_block_index"]}}}\n'
                )
                state["content_block_started"] = False

            for tc_idx in state.get("tool_call_blocks", {}):
                block_idx = state.get("tc_block_idx", {}).get(tc_idx, 0)
                events.append(
                    f"event: content_block_stop\n"
                    f'data: {{"type":"content_block_stop","index":{block_idx}}}\n'
                )

            # Message delta
            stop_reason_map = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
                "content_filter": "stop_sequence",
            }
            stop_reason = stop_reason_map.get(finish_reason, "end_turn")

            usage_fields = ""
            if usage:
                usage_fields = (
                    f',"usage":{{'
                    f'"input_tokens":{usage.get("prompt_tokens", 0)},'
                    f'"output_tokens":{usage.get("completion_tokens", 0)}'
                    f'}}'
                )

            events.append(
                f"event: message_delta\n"
                f'data: {{"type":"message_delta","delta":{{"stop_reason":"{stop_reason}","stop_sequence":null}}{usage_fields}}}\n'
            )

    # Check if this is the final chunk (has usage but no choices)
    if usage and not choices:
        events.append(
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n'
        )
        _clear_sse_state(request_key)

    return "\n".join(events) + "\n" if events else ""


def finalize_anthropic_stream(request_key: str) -> str:
    """Emit the final message_stop event and clean up state."""
    _clear_sse_state(request_key)
    return (
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n'
    )
