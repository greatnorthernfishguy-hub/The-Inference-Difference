# ---- Changelog ----
# [2026-06-15] Claude Code (DudeMan CC, Opus 4.8) — #330/#331 1b Phase 1: raw deposit signal
# What: raw_outcome() — the OBJECTIVE, unjudged outcome TID deposits to its local NG-Lite.
# Why: success was `model_response.success AND quality.is_success` — a surface+latency verdict
#   that recorded usable-but-slow responses as FAILURES (LAW 7 violation: classify-at-deposit).
#   Deposit raw; classify at extraction. Raw facts in metadata enable future bucket-tunes.
# How: pure function over the raw fields; success = call returned content or tool_calls.
# -------------------
from __future__ import annotations
from typing import Any, Optional


def raw_outcome(*, call_ok: bool, content: Optional[str], tool_calls: Any,
                tools_requested: bool, latency_ms: Optional[float],
                finish_reason: Optional[str] = None) -> tuple[bool, float, dict]:
    """Objective deposit signal (no quality verdict).

    success = the call returned a USABLE response (content or tool_calls).
    strength = uniform 1.0 (no quality modulation).
    raw_meta = the raw facts, deposited for future bucket-tunes.
    """
    had_content = bool((content or "").strip())
    had_tool_calls = bool(tool_calls)
    usable = bool(call_ok) and (had_content or had_tool_calls)
    raw_meta = {
        "had_content": had_content,
        "had_tool_calls": had_tool_calls,
        "tools_requested": bool(tools_requested),
        "call_ok": bool(call_ok),
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
    }
    return usable, 1.0, raw_meta
