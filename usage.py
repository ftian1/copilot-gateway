"""
Per-request token usage tracker with cost estimation.

GitHub Copilot moved to usage-based billing (AI Credits) on 2026-06-01.
One AI Credit = $0.01 USD. Models are priced per 1M tokens.

This module records token counts from API responses and multiplies by
the per-model pricing from the /models endpoint to estimate spending.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelUsage:
    """Cumulative usage for a single model."""

    requests: int = 0
    input_tokens: int = 0        # total input (includes cache reads + writes)
    output_tokens: int = 0
    cache_read_tokens: int = 0   # tokens served from cache (billed at discount)
    cache_write_tokens: int = 0  # tokens written to cache (billed at input price)
    reasoning_tokens: int = 0
    cost_usd: float = 0.0


class UsageTracker:
    """Thread-safe tracker for token usage and cost estimation."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._usage: dict[str, ModelUsage] = {}
        self._started_at: float = time.time()

    # ── public API ──────────────────────────────────────────────

    async def record(
        self,
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        input_price_per_m: float = 0.0,
        output_price_per_m: float = 0.0,
        cache_read_price_per_m: float = 0.0,
    ) -> None:
        """Record token usage for a request.

        Prices are in USD per 1M tokens.

        Cost formula:
          - Non-cache input: charged at input_price_per_m
          - Cache writes: charged at input_price_per_m (same rate)
          - Cache reads:  charged at cache_read_price_per_m (discounted)
          - Output:       charged at output_price_per_m

        input_tokens includes cache_read_tokens + cache_write_tokens,
        so we charge everything at input price, then apply a discount
        for the cache-read portion.
        """
        if not model_id:
            return

        # Charge all input at full price, then apply cache-read discount
        input_cost = input_tokens / 1_000_000 * input_price_per_m
        cache_discount = cache_read_tokens / 1_000_000 * (input_price_per_m - cache_read_price_per_m)
        output_cost = output_tokens / 1_000_000 * output_price_per_m
        cost = input_cost - cache_discount + output_cost

        async with self._lock:
            u = self._usage.get(model_id)
            if u is None:
                u = ModelUsage()
                self._usage[model_id] = u

            u.requests += 1
            u.input_tokens += input_tokens
            u.output_tokens += output_tokens
            u.cache_read_tokens += cache_read_tokens
            u.cache_write_tokens += cache_write_tokens
            u.reasoning_tokens += reasoning_tokens
            u.cost_usd += cost

        logger.debug(
            f"Usage: model={model_id} in={input_tokens} out={output_tokens} "
            f"cache_r={cache_read_tokens} cache_w={cache_write_tokens} cost=${cost:.6f}"
        )

    def snapshot(self) -> dict:
        """Return a snapshot of all usage data."""
        result: dict[str, dict] = {}
        total_cost = 0.0
        total_requests = 0
        total_input = 0
        total_output = 0

        for model_id, u in sorted(self._usage.items()):
            result[model_id] = {
                "requests": u.requests,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_read_tokens": u.cache_read_tokens,
                "cache_write_tokens": u.cache_write_tokens,
                "reasoning_tokens": u.reasoning_tokens,
                "cost_usd": round(u.cost_usd, 6),
            }
            total_cost += u.cost_usd
            total_requests += u.requests
            total_input += u.input_tokens
            total_output += u.output_tokens

        return {
            "models": result,
            "totals": {
                "requests": total_requests,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cost_usd": round(total_cost, 6),
            },
            "uptime_seconds": round(time.time() - self._started_at, 0),
        }

    def reset(self) -> None:
        """Reset all usage counters."""
        self._usage.clear()
        self._started_at = time.time()


def extract_openai_usage(body: dict) -> tuple[int, int, int, int, int]:
    """Extract (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)."""
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        return 0, 0, 0, 0, 0
    prompt_details = usage.get("prompt_tokens_details", {}) or {}
    completion_details = usage.get("completion_tokens_details", {}) or {}
    return (
        usage.get("prompt_tokens", 0) or 0,                                  # input_tokens
        usage.get("completion_tokens", 0) or 0,                              # output_tokens
        prompt_details.get("cached_tokens", 0) or 0,                         # cache_read_tokens
        0,                                                                   # OpenAI doesn't report cache writes separately
        completion_details.get("reasoning_tokens", 0) or 0,                  # reasoning_tokens
    )


def extract_responses_usage(body: dict) -> tuple[int, int, int, int, int]:
    """Extract (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)."""
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        return 0, 0, 0, 0, 0
    input_details = usage.get("input_tokens_details", {}) or {}
    output_details = usage.get("output_tokens_details", {}) or {}
    return (
        usage.get("input_tokens", 0) or 0,                                   # input_tokens
        usage.get("output_tokens", 0) or 0,                                  # output_tokens
        input_details.get("cached_tokens", 0) or 0,                          # cache_read_tokens
        0,                                                                   # OpenAI doesn't report cache writes
        output_details.get("reasoning_tokens", 0) or 0,                      # reasoning_tokens
    )


def extract_anthropic_usage(body: dict) -> tuple[int, int, int, int, int]:
    """Extract (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)."""
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        return 0, 0, 0, 0, 0
    return (
        usage.get("input_tokens", 0) or 0,                                   # input_tokens
        usage.get("output_tokens", 0) or 0,                                  # output_tokens
        usage.get("cache_read_input_tokens", 0) or 0,                        # cache_read_tokens
        usage.get("cache_creation_input_tokens", 0) or 0,                    # cache_write_tokens
        0,                                                                   # Anthropic doesn't have reasoning_tokens
    )


def _k(n: int) -> str:
    """Abbreviate token count: 200000 → 200K."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    return f"{n//1000}K"


# Last printed snapshot hash — only reprint when usage changes
_last_hash = ""


def print_usage_table(tracker: "UsageTracker") -> None:
    """Print a compact running cost summary to stderr.

    Only prints when token counts change — no repeated output when idle.
    """
    import sys

    global _last_hash

    snap = tracker.snapshot()
    models = snap.get("models", {})
    totals = snap.get("totals", {})

    # Skip if nothing changed
    h = f"{len(models)}:{totals.get('input_tokens',0)}:{totals.get('output_tokens',0)}"
    if h == _last_hash and _last_hash != "":
        return
    _last_hash = h

    if not models:
        sys.stderr.write("── Usage ── (no requests yet, waiting...)\n")
        sys.stderr.flush()
        return

    # Build all lines
    lines: list[str] = []
    lines.append("── Usage ──")
    for mid, u in sorted(models.items()):
        parts = [
            f"  {mid:<22s}  req:{u['requests']:>5d}  "
            f"in:{_k(u['input_tokens']):>6s}  out:{_k(u['output_tokens']):>6s}",
        ]
        if u["cache_read_tokens"] > 0 or u["cache_write_tokens"] > 0:
            parts.append(
                f"  cache_r:{_k(u['cache_read_tokens']):>5s}  cache_w:{_k(u['cache_write_tokens']):>5s}"
            )
        parts.append(f"  ${u['cost_usd']:.4f}")
        lines.append("".join(parts))

    # TOTAL line
    total_parts = [
        f"  {'TOTAL':<22s}  req:{totals['requests']:>5d}  "
        f"in:{_k(totals['input_tokens']):>6s}  out:{_k(totals['output_tokens']):>6s}",
    ]
    total_cache_r = sum(u["cache_read_tokens"] for u in models.values())
    total_cache_w = sum(u["cache_write_tokens"] for u in models.values())
    if total_cache_r > 0 or total_cache_w > 0:
        total_parts.append(
            f"  cache_r:{_k(total_cache_r):>5s}  cache_w:{_k(total_cache_w):>5s}"
        )
    total_parts.append(f"  ${totals['cost_usd']:.4f}")
    lines.append("".join(total_parts))

    max_w = max(len(line) for line in lines)
    sep = "─" * max_w

    out: list[str] = []
    out.append("── Usage ──")
    out.append(sep)
    out.extend(lines[1:])
    out.append(sep)

    sys.stderr.write("\n".join(out) + "\n")
    sys.stderr.flush()
