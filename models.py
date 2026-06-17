"""
Model discovery and store for GitHub Copilot.

Based on the OpenCode reference:
  packages/opencode/src/plugin/github-copilot/models.ts

Fetches available models from GitHub Copilot's /models endpoint,
filters disabled models, and exposes them in both OpenAI-compatible
and Anthropic-compatible formats.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from config import API_VERSION, USER_AGENT, Config
from auth import AuthManager

logger = logging.getLogger(__name__)


@dataclass
class CopilotModel:
    """Internal representation of a GitHub Copilot model."""

    id: str
    name: str
    family: str
    version: str
    supported_endpoints: list[str] = field(default_factory=list)
    model_picker_enabled: bool = False

    # Capabilities
    supports_tool_calls: bool = False
    supports_vision: bool = False
    supports_streaming: bool = False

    # Vision details (populated when supports_vision is True)
    vision_max_images: Optional[int] = None
    vision_max_size_bytes: Optional[int] = None
    vision_media_types: list[str] = field(default_factory=list)
    supports_structured_outputs: bool = False
    supports_reasoning: bool = False
    reasoning_efforts: list[str] = field(default_factory=list)
    max_thinking_budget: Optional[int] = None

    # Limits
    max_context_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_prompt_tokens: Optional[int] = None

    # Pricing (USD per million tokens)
    input_price: float = 0.0
    output_price: float = 0.0
    cache_read_price: float = 0.0

    @property
    def supports_anthropic_api(self) -> bool:
        """Check if this model supports the Anthropic Messages API natively."""
        return "/v1/messages" in self.supported_endpoints

    @property
    def is_gpt5(self) -> bool:
        """Check if this is a GPT-5+ model that should use the Responses API."""
        m = re.match(r"^gpt-(\d+)", self.id)
        if not m:
            return False
        return int(m.group(1)) >= 5 and not self.id.startswith("gpt-5-mini")

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible model object."""
        return {
            "id": self.id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "github-copilot",
            # Expose which protocol endpoints this model supports so
            # clients know whether to use /v1/chat/completions, /v1/responses,
            # or /v1/messages (Anthropic).
            "supported_endpoints": self.supported_endpoints,
            "anthropic_native": self.supports_anthropic_api,
            "uses_responses_api": self.is_gpt5,
            "capabilities": {
                "tool_calls": self.supports_tool_calls,
                "vision": self.supports_vision,
                "streaming": self.supports_streaming,
                "reasoning": self.supports_reasoning,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic-compatible model object."""
        return {
            "id": self.id,
            "display_name": self.name,
            "type": "model",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "supported_endpoints": self.supported_endpoints,
            "anthropic_native": self.supports_anthropic_api,
        }

    def to_debug_format(self) -> dict[str, Any]:
        """Full debug dump of all model metadata."""
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "version": self.version,
            "supported_endpoints": self.supported_endpoints,
            "anthropic_native": self.supports_anthropic_api,
            "uses_responses_api": self.is_gpt5,
            "model_picker_enabled": self.model_picker_enabled,
            "capabilities": {
                "tool_calls": self.supports_tool_calls,
                "vision": {
                    "supported": self.supports_vision,
                    "max_images": self.vision_max_images,
                    "max_size_bytes": self.vision_max_size_bytes,
                    "media_types": self.vision_media_types,
                },
                "streaming": self.supports_streaming,
                "structured_outputs": self.supports_structured_outputs,
                "reasoning": self.supports_reasoning,
                "reasoning_efforts": self.reasoning_efforts,
                "max_thinking_budget": self.max_thinking_budget,
            },
            "limits": {
                "max_context_tokens": self.max_context_tokens,
                "max_output_tokens": self.max_output_tokens,
                "max_prompt_tokens": self.max_prompt_tokens,
            },
            "pricing": {
                "input": self.input_price,
                "output": self.output_price,
                "cache_read": self.cache_read_price,
            },
        }


class ModelStore:
    """Thread-safe store of GitHub Copilot models.

    Auto-refreshes from the API on a background interval.
    """

    def __init__(self, config: Config, auth: AuthManager):
        self._config = config
        self._auth = auth
        self._models: dict[str, CopilotModel] = {}
        self._raw_response: dict[str, Any] = {}  # untouched upstream /models response
        self._lock = asyncio.Lock()
        self._last_refresh: float = 0
        self._refresh_task: Optional[asyncio.Task] = None

    # ── public API ──────────────────────────────────────────────

    async def start_refresh_loop(self) -> None:
        """Start background model refresh loop."""
        # Do an initial fetch
        await self.refresh()
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def stop_refresh_loop(self) -> None:
        """Stop background refresh loop."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

    async def refresh(self) -> None:
        """Fetch models from GitHub Copilot API."""
        token = self._auth.get_token()
        if not token:
            logger.warning("Cannot refresh models: no token available")
            return

        base_url = self._config.api_base_url
        url = f"{base_url}/models"

        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": API_VERSION,
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers=headers)

                if not resp.is_success:
                    logger.error(f"Failed to fetch models: {resp.status_code} {resp.text[:200]}")
                    return

                data = resp.json()
        except httpx.RequestError as e:
            logger.error(f"Failed to fetch models: {e}")
            return
        except ValueError as e:
            logger.error(f"Failed to parse models response: {e}")
            return

        models = self._parse_models(data.get("data", []))
        async with self._lock:
            self._models = models
            self._raw_response = data  # keep untouched response for /v1/models/raw
            self._last_refresh = time.time()

        logger.info(f"Refreshed models: {len(models)} available")

    def get_model(self, model_id: str) -> Optional[CopilotModel]:
        """Get a single model by ID."""
        return self._models.get(model_id)

    def get_models(self) -> dict[str, CopilotModel]:
        """Get all models (returns a copy of the dict for safety)."""
        return dict(self._models)

    def list_openai_models(self) -> list[dict[str, Any]]:
        """List models in OpenAI-compatible format."""
        return [m.to_openai_format() for m in self._models.values()]

    def list_anthropic_models(self) -> list[dict[str, Any]]:
        """List models in Anthropic-compatible format."""
        return [m.to_anthropic_format() for m in self._models.values()]

    def list_debug_models(self) -> list[dict[str, Any]]:
        """List models with full debug metadata including pricing, limits, etc."""
        return [m.to_debug_format() for m in self._models.values()]

    def get_raw_response(self) -> dict[str, Any]:
        """Return the untouched upstream /models API response."""
        return dict(self._raw_response)

    # ── internal ────────────────────────────────────────────────

    async def _refresh_loop(self) -> None:
        """Background loop that refreshes models periodically."""
        while True:
            await asyncio.sleep(self._config.model_refresh_secs)
            await self.refresh()

    def _parse_models(self, data: list[dict]) -> dict[str, CopilotModel]:
        """Parse raw model data from the Copilot API response."""
        models: dict[str, CopilotModel] = {}

        for raw in data:
            if not isinstance(raw, dict):
                continue

            # Skip disabled models
            policy = raw.get("policy", {})
            if isinstance(policy, dict) and policy.get("state") == "disabled":
                continue

            model_id = raw.get("id", "")
            if not model_id:
                continue

            caps = raw.get("capabilities", {})
            family = caps.get("family", model_id)
            model_type = caps.get("type", "")

            # Skip embedding models — they aren't chat models
            if "embedding" in family.lower() or model_type == "embeddings":
                continue

            limits = caps.get("limits", {}) or {}
            supports = caps.get("supports", {}) or {}
            billing = raw.get("billing", {})
            prices = billing.get("token_prices", {}) or {}

            # Convert Copilot AIC pricing to USD per million tokens
            batch_size = prices.get("batch_size", 1)
            usd_per_million = 10_000 / batch_size if batch_size > 0 else 0

            default_price = prices.get("default", {}) or {}

            # Determine if model supports vision
            vision_limits = limits.get("vision", {}) or {}
            vision_media_types: list[str] = []
            vision_max_images: Optional[int] = None
            vision_max_size: Optional[int] = None
            has_vision_images = False
            if isinstance(vision_limits, dict):
                vision_media_types = vision_limits.get("supported_media_types", []) or []
                vision_max_images = vision_limits.get("max_prompt_images")
                vision_max_size = vision_limits.get("max_prompt_image_size")
                has_vision_images = any(
                    t.startswith("image/") for t in vision_media_types
                )
            vision_support = supports.get("vision", False) or has_vision_images

            # Determine reasoning support
            reasoning_efforts = supports.get("reasoning_effort", []) or []
            has_adaptive_thinking = supports.get("adaptive_thinking", False)
            has_reasoning = (
                has_adaptive_thinking
                or len(reasoning_efforts) > 0
                or supports.get("max_thinking_budget") is not None
            )

            # Infer supported_endpoints if the API returned an empty list
            eps = raw.get("supported_endpoints", []) or []
            if not eps:
                eps = _infer_endpoints(model_id, family, supports)

            model = CopilotModel(
                id=model_id,
                name=raw.get("name", model_id),
                family=family,
                version=raw.get("version", ""),
                supported_endpoints=eps,
                model_picker_enabled=raw.get("model_picker_enabled", False),
                supports_tool_calls=supports.get("tool_calls", False),
                supports_vision=vision_support,
                supports_streaming=supports.get("streaming", False),
                supports_structured_outputs=supports.get("structured_outputs", False),
                supports_reasoning=has_reasoning,
                reasoning_efforts=reasoning_efforts,
                max_thinking_budget=supports.get("max_thinking_budget"),
                vision_max_images=vision_max_images,
                vision_max_size_bytes=vision_max_size,
                vision_media_types=vision_media_types,
                max_context_tokens=limits.get("max_context_window_tokens") or limits.get("max_prompt_tokens"),
                max_output_tokens=limits.get("max_output_tokens"),
                max_prompt_tokens=limits.get("max_prompt_tokens"),
                input_price=(default_price.get("input_price", 0) or 0) * usd_per_million,
                output_price=(default_price.get("output_price", 0) or 0) * usd_per_million,
                cache_read_price=(default_price.get("cache_price", 0) or 0) * usd_per_million,
            )
            models[model_id] = model

        return models


def _infer_endpoints(model_id: str, family: str, supports: dict) -> list[str]:
    """Infer supported endpoints when the API doesn't provide them.

    Even when the upstream /models response omits or returns an empty
    supported_endpoints list, every chat model should show its endpoints
    so the table is never blank.
    """
    eps: list[str] = []

    # GPT-5+ models use the Responses API
    m = re.match(r"^gpt-(\d+)", model_id)
    gpt_ver = int(m.group(1)) if m else 0
    if gpt_ver >= 5 and not model_id.startswith("gpt-5-mini"):
        eps.append("/v1/responses")

    # Models with tool_calls or streaming support are chat models
    has_chat = (
        supports.get("tool_calls", False)
        or supports.get("streaming", False)
        or supports.get("structured_outputs", False)
        or supports.get("vision", False)
        or "gpt" in model_id.lower()
        or "claude" in model_id.lower()
        or "gemini" in model_id.lower()
    )

    if has_chat and "/v1/responses" not in eps:
        eps.append("/v1/chat/completions")

    # Anthropic models support the Messages API
    if "claude" in model_id.lower():
        eps.append("/v1/messages")

    # Fallback: if nothing matched, assume basic chat completions
    if not eps:
        eps.append("/v1/chat/completions")

    return eps


# ═══════════════════════════════════════════════════════════════
# Terminal table rendering
# ═══════════════════════════════════════════════════════════════

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_DIM = "\033[2m"
_RST = "\033[0m"
_BOLD = "\033[1m"

# Number of columns in the model table
_TABLE_COLS = 5


def _vlen(s: str) -> int:
    """Visual length — strip ANSI codes before counting."""
    return len(_ANSI_RE.sub("", str(s)))


def _dim(s: str) -> str:
    return f"{_DIM}{s}{_RST}"


def _bold(s: str) -> str:
    return f"{_BOLD}{s}{_RST}"


def _pad(text: str, width: int) -> str:
    """Left-pad with visual-width-aware spacing."""
    t = str(text)
    need = width - _vlen(t)
    return t + " " * max(need, 0)


def _abbrev(n: int) -> str:
    """Abbreviate token counts: 200000 → '200K', 1000000 → '1000K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M" if n % 1_000_000 != 0 else f"{n // 1000}K"
    return f"{n // 1000}K"


def _fmt_price(pin: float, pout: float) -> str:
    return f"${pin:.2f} / ${pout:.2f}"


def _sep_line(l: str, m: str, r: str, widths: list[int]) -> str:
    """Draw a horizontal separator line with given column widths."""
    return l + m.join("─" * w for w in widths) + r


def _row_to_lines(cells: list, widths: list[int]) -> list[str]:
    """Convert a row spec to one or more visual │...│ lines.

    Each cell is either a single string or a list of strings
    (for multi-line content like endpoints).
    """
    # Normalise: every cell becomes a list of strings
    normalized: list[tuple[list[str], int]] = []
    for content, w in zip(cells, widths):
        if isinstance(content, list):
            normalized.append(([str(c) for c in content], w))
        elif content is not None and content != "":
            normalized.append(([str(content)], w))
        else:
            normalized.append(([], w))

    max_lines = max((len(c[0]) for c in normalized), default=1)
    lines: list[str] = []
    for li in range(max_lines):
        parts: list[str] = []
        for content_list, cw in normalized:
            txt = content_list[li] if li < len(content_list) else ""
            # Leading space + text padded to fill the column
            parts.append(" " + _pad(txt, cw - 1))
        lines.append("│" + "│".join(parts) + "│")
    return lines


def _fmt_vision(m: "CopilotModel") -> list[str]:
    """Format vision capabilities as two lines: count/size then types."""
    if not m.supports_vision:
        return ["—"]

    # Line 1: <=N imgs / <=SIZE
    parts: list[str] = []
    if m.vision_max_images is not None:
        parts.append(f"<={m.vision_max_images} imgs")
    else:
        parts.append("<=? imgs")
    if m.vision_max_size_bytes is not None:
        parts.append(f"<={_fmt_bytes(m.vision_max_size_bytes)}")
    else:
        parts.append("<=?")
    line1 = " / ".join(parts)

    # Line 2: media types (abbreviated)
    if m.vision_media_types:
        short = [_abbrev_mime(t) for t in m.vision_media_types]
        line2 = "/".join(short)
    else:
        line2 = "?"

    return [line1, line2]


def _abbrev_mime(mime: str) -> str:
    """Abbreviate MIME type: image/png → PNG, image/jpeg → JPEG."""
    if "/" in mime:
        return mime.split("/", 1)[1].upper()
    return mime.upper()


def _fmt_bytes(b: int) -> str:
    """Format byte count human-readably: 20971520 → 20MB."""
    if b >= 1_073_741_824:
        return f"{b / 1_073_741_824:.1f}GB"
    if b >= 1_048_576:
        return f"{b / 1_048_576:.0f}MB"
    if b >= 1024:
        return f"{b / 1024:.0f}KB"
    return f"{b}B"


def _fmt_limits(m: "CopilotModel") -> str:
    """Format token limits as context/prompt/output, e.g. 200K/128K/16K."""
    parts = []
    for val in (m.max_context_tokens, m.max_prompt_tokens, m.max_output_tokens):
        parts.append(_abbrev(val) if val else "—")
    return "/".join(parts)


def _max_vlen(cells: list) -> int:
    """Maximum visual length across all lines in a cell or cell list."""
    if isinstance(cells, list) and all(isinstance(c, str) for c in cells):
        return max((_vlen(c) for c in cells), default=0)
    return _vlen(str(cells)) if cells else 0


def _compute_widths(header: list[list[str]], rows: list[list]) -> list[int]:
    """Compute column widths so every cell fits, with 1 char left-padding."""
    num_cols = len(header)
    max_w = [0] * num_cols

    # Header cells (each is a list of 1-2 strings)
    for ci, hdr_cell in enumerate(header):
        for line in hdr_cell:
            max_w[ci] = max(max_w[ci], _vlen(line))

    # Body cells
    for row in rows:
        for ci, cell in enumerate(row):
            max_w[ci] = max(max_w[ci], _max_vlen(cell))

    # Add 1 for the leading space in each cell, minimum width 3 (space + content + pad)
    return [max(w + 1, 3) for w in max_w]


def print_model_table(store: "ModelStore") -> None:
    """Print a formatted table of all models to stdout.

    Columns: Model ID, Supported Endpoints, Price, Vision, Limits.

    Column widths are computed dynamically so the │ separators align
    on the widest cell in each column with minimal whitespace.
    """
    raw_models = store.get_models()
    if not raw_models:
        print("(no models loaded)", flush=True)
        return

    # Sort for stable output
    sorted_ids = sorted(raw_models.keys())
    models_list = [raw_models[mid] for mid in sorted_ids]

    # ═══ Collect all row data ═══

    # Header: two lines of labels
    header = [
        ["Model ID", ""],
        ["Supported", "Endpoints"],
        ["Price in/out", "$/1M tokens"],
        ["Vision", "count/size/type"],
        ["Limits", "ctx/prompt/output"],
    ]

    # Body: one row per model
    rows: list[list] = []
    for m in models_list:
        eps = sorted(m.supported_endpoints) if m.supported_endpoints else []
        rows.append([
            m.id,                              # col 0: single string
            eps if eps else "",                # col 1: list or empty
            _fmt_price(m.input_price, m.output_price),  # col 2
            _fmt_vision(m),                    # col 3
            _fmt_limits(m),                    # col 4
        ])

    # ═══ Compute dynamic column widths ═══
    widths = _compute_widths(header, rows)

    # ═══ Render ═══
    # Top border
    print(_sep_line("┌", "┬", "┐", widths), flush=True)

    # Header (2-line)
    for li in range(2):
        hdr_cells = [hdr_col[li] for hdr_col in header]
        parts = []
        for txt, w in zip(hdr_cells, widths):
            parts.append(" " + _pad(_bold(txt) if txt else "", w - 1))
        print("│" + "│".join(parts) + "│", flush=True)

    print(_sep_line("├", "┼", "┤", widths), flush=True)

    # ═══ Body ═══
    prev_key = None

    for mi, (m, row_cells) in enumerate(zip(models_list, rows)):
        eps = row_cells[1]
        vis = row_cells[3]
        price = row_cells[2]
        limits = row_cells[4]
        cur_key = (tuple(eps) if isinstance(eps, list) else eps,
                   tuple(vis) if isinstance(vis, list) else vis,
                   price, limits)

        # Separator only between groups
        if mi > 0 and cur_key != prev_key:
            print(_sep_line("├", "┼", "┤", widths), flush=True)

        m_lines = _row_to_lines(row_cells, widths)
        for line in m_lines:
            print(line, flush=True)

        prev_key = cur_key

    # Bottom border
    print(_sep_line("└", "┴", "┘", widths), flush=True)
