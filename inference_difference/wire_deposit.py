"""
Raw HTTP wire deposit — Law 7 compliance for TID's provider interactions.

TID's provider calls are sensory input to the substrate. Every outbound
request and every inbound response carries information a future bucket
may need: provider identity, tool-call shapes, refusal patterns,
censorship-stack signatures, latency fingerprints, model-family quirks.
Curating fields at deposit time discards patterns no one thought to list.

This module deposits raw wire bytes (request body, response body,
basic metadata) into the ecosystem experience tract so downstream
buckets can classify at extraction. No success scores. No quality
labels. No preclassified fields. The tract is conductive tissue —
see ~/NeuroGraph/ng_experience_tract.py and the substrate-as-cortex
concept page (~/docs/concepts/Substrate As Cortex.md).

Write path: ~/.et_modules/experience/inference_difference.tract
Drain path: NeuroGraph's _drain_scan_dir() in neurograph_rpc.py
Rust binding: ng_tract.deposit_experience (PyO3)

# ---- Changelog ----
# [2026-04-15] Claude Code (Opus 4.6) — Punchlist #141 — Raw HTTP wire deposits
#   What: deposit_outbound() / deposit_inbound() helpers writing raw
#         request + response bytes to the per-feeder experience tract.
#   Why:  TID's record_outcome() deposits only embedding + target + strength.
#         The raw interaction content — tool-call shapes, refusal text,
#         provider-specific quirks — was being discarded at the deposit
#         boundary. Law 7 violation. This restores raw sensory input.
#   How:  Minimal wrapper around ng_tract.deposit_experience(). Deposits
#         are best-effort — tract write failure logs and returns; it must
#         never block TID's request pipeline. Content packed as JSON
#         (request/response body, headers, status, provider, latency).
#         JSON here is container-shape, not classification — the substrate
#         sees the actual bytes that crossed the wire.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("tid.wire_deposit")

_TRACT_PATH = os.path.expanduser("~/.et_modules/experience/inference_difference.tract")
_SCAN_DIR = os.path.dirname(_TRACT_PATH)

_SENSITIVE_HEADER_KEYS = frozenset({
    "authorization", "x-api-key", "api-key", "openai-api-key",
    "anthropic-api-key", "x-venice-api-key", "hf-token", "cookie",
})


def _scrub_headers(headers: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Remove auth credentials from headers before deposit.

    Raw wire bytes is the goal — but API keys are not "experience" the
    substrate needs to learn. They're secrets. Scrub them; keep every
    other header (content-type, rate-limit, provider signatures, etc.)
    so buckets can learn from them.
    """
    if not headers:
        return {}
    out: Dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in _SENSITIVE_HEADER_KEYS:
            out[k] = "<scrubbed>"
        else:
            out[k] = str(v)
    return out


def _ensure_dir() -> bool:
    try:
        os.makedirs(_SCAN_DIR, exist_ok=True)
        return True
    except OSError as exc:
        logger.warning("Cannot create tract scan dir %s: %s", _SCAN_DIR, exc)
        return False


def _deposit(payload: Dict[str, Any], source: str) -> None:
    """Best-effort tract write. Never raises."""
    if not _ensure_dir():
        return
    try:
        import ng_tract
    except ImportError:
        logger.debug("ng_tract unavailable; skipping wire deposit")
        return
    try:
        content = json.dumps(payload, ensure_ascii=False, default=str)
        ng_tract.deposit_experience(
            content=content,
            source=source,
            tract_path=_TRACT_PATH,
            content_type="text",
        )
    except Exception as exc:
        logger.warning("Wire deposit failed (%s): %s", source, exc)


def deposit_outbound(
    provider: str,
    model_id: str,
    url: str,
    method: str,
    request_body: Any,
    headers: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """Deposit a raw outbound provider request."""
    payload = {
        "direction": "outbound",
        "ts": time.time(),
        "provider": provider,
        "model_id": model_id,
        "url": url,
        "method": method,
        "headers": _scrub_headers(headers),
        "body": request_body,
        "correlation_id": correlation_id,
    }
    _deposit(payload, source="tid.http.outbound")


def deposit_inbound(
    provider: str,
    model_id: str,
    url: str,
    status_code: Optional[int],
    response_body: Any,
    headers: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None,
    error: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """Deposit a raw inbound provider response (or error)."""
    payload = {
        "direction": "inbound",
        "ts": time.time(),
        "provider": provider,
        "model_id": model_id,
        "url": url,
        "status_code": status_code,
        "headers": _scrub_headers(headers),
        "body": response_body,
        "latency_ms": latency_ms,
        "error": error,
        "correlation_id": correlation_id,
    }
    _deposit(payload, source="tid.http.inbound")
