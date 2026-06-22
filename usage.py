"""
Per-request token usage tracker with cost estimation and cumulative billing.

GitHub Copilot moved to usage-based billing (AI Credits) on 2026-06-01.
One AI Credit = $0.01 USD. Models are priced per 1M tokens.

Usage data is persisted to disk as daily aggregates so costs survive
gateway restarts and accumulate across the current month, week, and day.
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional, TextIO

logger = logging.getLogger(__name__)


# ── date helpers ──────────────────────────────────────────────────

def _today() -> str:
    """ISO date string for today, e.g. '2026-06-22'."""
    return date.today().isoformat()


def _current_month_key() -> str:
    """Key for the current month, e.g. '2026-06'."""
    return date.today().strftime("%Y-%m")


def _current_week_key() -> str:
    """Key for the current ISO week, e.g. '2026-W25'."""
    return date.today().strftime("%Y-W%W")


def _current_day_key() -> str:
    """Key for today, e.g. '2026-06-22' — alias for _today()."""
    return _today()


def _month_from_date(d: str) -> str:
    """Extract 'YYYY-MM' from a 'YYYY-MM-DD' date string."""
    return d[:7]


def _week_from_date(d: str) -> str:
    """Extract 'YYYY-Www' from a 'YYYY-MM-DD' date string."""
    return date.fromisoformat(d).strftime("%Y-W%W")


# ── token abbreviation ────────────────────────────────────────────

def _k(n: int) -> str:
    """Abbreviate token count: 200000 → '200K', 1500000 → '1.5M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n // 1000}K"
    return str(n)


# ── data class ────────────────────────────────────────────────────

@dataclass
class ModelUsage:
    """Cumulative usage for a single model (in-memory, current session)."""

    requests: int = 0
    input_tokens: int = 0          # total input (includes cache reads + writes)
    output_tokens: int = 0
    cache_read_tokens: int = 0     # tokens served from cache (billed at discount)
    cache_write_tokens: int = 0    # tokens written to cache (billed at input price)
    reasoning_tokens: int = 0
    cost_usd: float = 0.0


# ── tracker ───────────────────────────────────────────────────────

class UsageTracker:
    """Thread-safe tracker for token usage with cumulative billing.

    Persists daily aggregates to disk so costs accumulate across
    gateway restarts within the same billing period (month / week / day).
    """

    def __init__(self, persist_path: str | None = None):
        self._lock = asyncio.Lock()
        self._usage: dict[str, ModelUsage] = {}
        self._started_at: float = time.time()

        # Persisted daily aggregates:
        #   { "2026-06-22": { model_id: {input_tokens, output_tokens,
        #     cache_read_tokens, cache_write_tokens, cost_usd} } }
        self._persist_path = persist_path
        self._daily: dict[str, dict[str, dict]] = {}
        self._dirty = False
        self._last_flush = 0.0
        if persist_path:
            self._load_persisted()

    # ── persistence ────────────────────────────────────────────

    def _load_persisted(self) -> None:
        """Load daily usage aggregates from disk."""
        try:
            path = Path(self._persist_path)
            if not path.exists():
                logger.info("No persisted usage file — starting fresh")
                return
            data = json.loads(path.read_text())
            self._daily = data.get("daily", {})
            if self._daily:
                months = len({_month_from_date(d) for d in self._daily})
                logger.info(
                    f"Loaded persisted usage: {len(self._daily)} day(s) "
                    f"across {months} month(s)"
                )
        except Exception as e:
            logger.warning(f"Failed to load persisted usage: {e}")
            self._daily = {}

    def _flush(self) -> None:
        """Write daily aggregates to disk (only if dirty)."""
        if not self._persist_path or not self._dirty:
            return

        now = time.time()
        # Debounce: flush at most once per 2 seconds
        if now - self._last_flush < 2.0:
            return
        self._last_flush = now
        self._dirty = False

        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "daily": self._daily,
                "updated_at": datetime.now().isoformat(),
            }
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to persist usage: {e}")

    def flush(self) -> None:
        """Public: force a sync of persisted usage (call on shutdown)."""
        self._dirty = True
        self._last_flush = 0.0  # bypass debounce
        self._flush()

    def _update_daily_aggregate(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        cost_usd: float,
    ) -> None:
        """Merge a single request's usage into today's daily bucket."""
        today = _today()
        if today not in self._daily:
            self._daily[today] = {}
        if model_id not in self._daily[today]:
            self._daily[today][model_id] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost_usd": 0.0,
            }
        d = self._daily[today][model_id]
        d["input_tokens"] += input_tokens
        d["output_tokens"] += output_tokens
        d["cache_read_tokens"] += cache_read_tokens
        d["cache_write_tokens"] += cache_write_tokens
        d["cost_usd"] += cost_usd

    # ── period cost helpers ────────────────────────────────────

    def _period_costs(self) -> dict[str, dict[str, float]]:
        """Compute costs by model for the current month, week, day.

        Uses _daily as the single source of truth — it is always
        up-to-date (updated on every record() call) and includes
        both live session data and pre-restart persisted data.

        Returns:
            { model_id: {"month": float, "week": float, "day": float} }
        """
        month_key = _current_month_key()
        week_key = _current_week_key()
        day_key = _current_day_key()

        costs: dict[str, dict[str, float]] = {}

        for d, models in self._daily.items():
            in_month = _month_from_date(d) == month_key
            in_week = _week_from_date(d) == week_key
            in_day = d == day_key

            if not (in_month or in_week or in_day):
                continue

            for mid, m in models.items():
                if mid not in costs:
                    costs[mid] = {"month": 0.0, "week": 0.0, "day": 0.0}
                c = m.get("cost_usd", 0.0)
                if in_month:
                    costs[mid]["month"] += c
                if in_week:
                    costs[mid]["week"] += c
                if in_day:
                    costs[mid]["day"] += c

        return costs

    def _today_tokens(self) -> dict[str, dict[str, int]]:
        """Compute today's token totals by model from _daily only.

        _daily is always current (updated on every record call), so
        no need to merge with _usage separately.
        """
        today = _today()
        totals: dict[str, dict[str, int]] = {}

        for mid, m in self._daily.get(today, {}).items():
            totals[mid] = {
                "input_tokens": m.get("input_tokens", 0),
                "output_tokens": m.get("output_tokens", 0),
                "cache_read_tokens": m.get("cache_read_tokens", 0),
                "cache_write_tokens": m.get("cache_write_tokens", 0),
            }

        return totals

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
        cache_discount = (
            cache_read_tokens / 1_000_000 * (input_price_per_m - cache_read_price_per_m)
        )
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

            # Persist to daily aggregate
            self._update_daily_aggregate(
                model_id,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
                cost,
            )
            self._dirty = True

        # Flush outside the lock (debounced internally)
        self._flush()

        logger.debug(
            f"Usage: model={model_id} in={input_tokens} out={output_tokens} "
            f"cache_r={cache_read_tokens} cache_w={cache_write_tokens} cost=${cost:.6f}"
        )

    def snapshot(self) -> dict:
        """Return a snapshot of all usage data with period costs."""
        period = self._period_costs()
        today = self._today_tokens()

        result: dict[str, dict] = {}
        total_cost_month = 0.0
        total_cost_week = 0.0
        total_cost_day = 0.0
        total_requests = 0
        total_input = 0
        total_output = 0
        total_cache_r = 0
        total_cache_w = 0

        # Collect all model IDs from both session and period data
        all_ids = set(self._usage.keys()) | set(period.keys()) | set(today.keys())

        for mid in sorted(all_ids):
            u = self._usage.get(mid)
            t = today.get(mid, {})
            p = period.get(mid, {"month": 0.0, "week": 0.0, "day": 0.0})

            result[mid] = {
                "requests": u.requests if u else 0,
                "input_tokens": t.get("input_tokens", 0),
                "output_tokens": t.get("output_tokens", 0),
                "cache_read_tokens": t.get("cache_read_tokens", 0),
                "cache_write_tokens": t.get("cache_write_tokens", 0),
                "reasoning_tokens": u.reasoning_tokens if u else 0,
                "cost_month": round(p["month"], 6),
                "cost_week": round(p["week"], 6),
                "cost_day": round(p["day"], 6),
            }
            total_cost_month += p["month"]
            total_cost_week += p["week"]
            total_cost_day += p["day"]
            if u:
                total_requests += u.requests
            total_input += t.get("input_tokens", 0)
            total_output += t.get("output_tokens", 0)
            total_cache_r += t.get("cache_read_tokens", 0)
            total_cache_w += t.get("cache_write_tokens", 0)

        return {
            "models": result,
            "totals": {
                "requests": total_requests,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cache_read_tokens": total_cache_r,
                "cache_write_tokens": total_cache_w,
                "cost_month": round(total_cost_month, 6),
                "cost_week": round(total_cost_week, 6),
                "cost_day": round(total_cost_day, 6),
            },
            "uptime_seconds": round(time.time() - self._started_at, 0),
        }

    def reset(self) -> None:
        """Reset all usage counters (session only; persisted data is untouched)."""
        self._usage.clear()
        self._started_at = time.time()


# ── usage extraction (unchanged) ──────────────────────────────────


def extract_openai_usage(body: dict) -> tuple[int, int, int, int, int]:
    """Extract (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)."""
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        return 0, 0, 0, 0, 0
    prompt_details = usage.get("prompt_tokens_details", {}) or {}
    completion_details = usage.get("completion_tokens_details", {}) or {}
    return (
        usage.get("prompt_tokens", 0) or 0,
        usage.get("completion_tokens", 0) or 0,
        prompt_details.get("cached_tokens", 0) or 0,
        0,  # OpenAI doesn't report cache writes separately
        completion_details.get("reasoning_tokens", 0) or 0,
    )


def extract_responses_usage(body: dict) -> tuple[int, int, int, int, int]:
    """Extract (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)."""
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        return 0, 0, 0, 0, 0
    input_details = usage.get("input_tokens_details", {}) or {}
    output_details = usage.get("output_tokens_details", {}) or {}
    return (
        usage.get("input_tokens", 0) or 0,
        usage.get("output_tokens", 0) or 0,
        input_details.get("cached_tokens", 0) or 0,
        0,  # OpenAI doesn't report cache writes
        output_details.get("reasoning_tokens", 0) or 0,
    )


def extract_anthropic_usage(body: dict) -> tuple[int, int, int, int, int]:
    """Extract (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens).

    Anthropic reports `input_tokens` exclusive of cache reads/writes (they are
    separate, additive fields), unlike OpenAI's `prompt_tokens` which already
    includes cached tokens. The cost formula in record() assumes input_tokens
    is inclusive, so we fold the cache counts in here — otherwise the cache-read
    discount over-subtracts and the cost can go negative.
    """
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        return 0, 0, 0, 0, 0
    base_input = usage.get("input_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_write = usage.get("cache_creation_input_tokens", 0) or 0
    return (
        base_input + cache_read + cache_write,
        usage.get("output_tokens", 0) or 0,
        cache_read,
        cache_write,
        0,  # Anthropic doesn't have reasoning_tokens
    )


# ── ANSI table printer ────────────────────────────────────────────

# Column headers (model left-aligned, others right-aligned)
_COL_H = ["Model", "In", "Out", "Hit", "Wrt", "$/Mo", "$/Wk", "$/Day"]

# Track how many lines the last table occupied (for cursor-positioned refresh)
_last_lines = 0


def _build_table(tracker: "UsageTracker") -> list[str]:
    """Build a compact usage table — column widths auto-sized to data."""
    period = tracker._period_costs()
    today = tracker._today_tokens()

    all_ids = (
        set(tracker._usage.keys())
        | set(period.keys())
        | set(today.keys())
    )

    if not all_ids:
        return ["── Usage ── (no requests yet, waiting...)"]

    # ── collect data rows as string tuples ─────────────────────
    _MAX_MODEL = 22
    rows: list[tuple] = []
    totals = [0, 0, 0, 0, 0.0, 0.0, 0.0]

    for mid in sorted(all_ids):
        t = today.get(mid, {})
        p = period.get(mid, {"month": 0.0, "week": 0.0, "day": 0.0})

        in_tok  = t.get("input_tokens", 0)
        out_tok = t.get("output_tokens", 0)
        hit     = t.get("cache_read_tokens", 0)
        wrt     = t.get("cache_write_tokens", 0)
        cm, cw, cd = p["month"], p["week"], p["day"]

        if in_tok == 0 and out_tok == 0 and cm == 0 and cw == 0 and cd == 0:
            continue

        totals[0] += in_tok
        totals[1] += out_tok
        totals[2] += hit
        totals[3] += wrt
        totals[4] += cm
        totals[5] += cw
        totals[6] += cd

        name = mid
        if len(name) > _MAX_MODEL:
            name = name[:_MAX_MODEL - 1] + "…"

        rows.append((
            name,
            _k(in_tok), _k(out_tok), _k(hit), _k(wrt),
            f"${cm:.2f}", f"${cw:.2f}", f"${cd:.2f}",
        ))

    total_tup = (
        "TOTAL", _k(totals[0]), _k(totals[1]), _k(totals[2]), _k(totals[3]),
        f"${totals[4]:.2f}", f"${totals[5]:.2f}", f"${totals[6]:.2f}",
    )

    # ── auto-size column widths ────────────────────────────────
    widths = [len(h) for h in _COL_H]
    for row in rows + [total_tup]:
        for i, cell in enumerate(row):
            w = len(str(cell))
            if w > widths[i]:
                widths[i] = w

    # ── build format string ────────────────────────────────────
    # i=0 left-aligned, rest right-aligned; " │ " column separator
    fmts = [f"%-{widths[0]}s"] + [f"%{w}s" for w in widths[1:]]
    row_fmt = " │ ".join(fmts)
    sep = "─" * (sum(widths) + 3 * (len(widths) - 1))

    # ── render ─────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(row_fmt % tuple(_COL_H))
    lines.append(sep)

    for row in rows:
        lines.append(row_fmt % row)

    lines.append(sep)
    lines.append(row_fmt % total_tup)

    return lines


def print_usage_table(tracker: "UsageTracker", output: TextIO | None = None) -> None:
    """Print a compact usage table, refreshing in-place via ANSI escapes.

    First call prints normally. Subsequent calls restore the saved cursor
    position, clear below, and reprint — values appear to update in-place.

    Args:
        tracker: The UsageTracker to query.
        output: File-like to write to (default: sys.stderr).
    """
    global _last_lines

    if output is None:
        output = sys.stderr

    lines = _build_table(tracker)

    if _last_lines > 0:
        # Move cursor back up to where the last table started
        output.write(f"\033[{_last_lines}A")
        # Clear from cursor to end of screen
        output.write("\033[J")
    else:
        # First print: ensure we start on a fresh line
        output.write("\n")

    for line in lines:
        output.write(line + "\n")

    output.flush()
    _last_lines = len(lines)
