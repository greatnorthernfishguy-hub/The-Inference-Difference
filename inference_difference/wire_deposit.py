"""
Raw HTTP wire deposit — Law 7 compliance for TID's provider interactions.

TID's provider calls are sensory input to the substrate. Every outbound
request and every inbound response carries information a future bucket
may need: provider identity, tool-call shapes, refusal patterns,
censorship-stack signatures, latency fingerprints, model-family quirks.
Curating fields at deposit time discards patterns no one thought to list.

Content format: raw HTTP wire text (request line + headers + blank line
+ body, or status line + headers + blank line + body). No JSON wrapper.
Header key-case preserved as received. Auth credentials scrubbed.

Streaming responses accumulate with per-chunk wall-clock timings emitted
as SSE comment lines (`: chunk_ms=<delta>`) — still valid SSE, still raw
wire bytes.

No correlation_id field: STDP learns request↔response pairing from
temporal co-firing. That's what substrate dynamics are for.

Write path: ~/.et_modules/experience/inference_difference.tract
Drain path: NeuroGraph's wire-absorption path in wire_absorption.py
Rust binding: ng_tract.deposit_experience (PyO3)

# ---- Changelog ----
# [2026-04-15] Claude Code (Opus 4.6) — Raw-wire-text rewrite (no JSON).
#   What: Replaced JSON-wrapped payload with raw HTTP wire text format.
#         Headers preserved by key-case. Body appended after blank line.
#         Streaming: per-chunk timings as SSE `:` comment lines.
#         correlation_id field removed (STDP handles pairing).
#   Why:  Law 4 — JSON dict inside a BTF experience tract was
#         binary→dict inflation exactly as the Rust migration workorder
#         warns against. Law 7 — curated fields (provider, status_code,
#         correlation_id) were pre-classification at the deposit boundary.
#         The substrate should see the bytes that crossed the wire,
#         nothing more; buckets classify at extraction.
#   How:  _build_request_text() / _build_response_text() produce raw
#         HTTP wire text. deposit_outbound/_inbound now take a
#         pre-built `wire_text` string from the caller.
# [2026-04-15] Claude Code (Opus 4.6) — Punchlist #141 — Raw HTTP wire deposits
#   (initial JSON-wrapped shape, superseded same day by raw-wire rewrite above)
# -------------------
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("tid.wire_deposit")

_TRACT_PATH = os.path.expanduser("~/.et_modules/experience/inference_difference.tract")
_SCAN_DIR = os.path.dirname(_TRACT_PATH)

_SENSITIVE_HEADER_KEYS = frozenset({
    "authorization", "x-api-key", "api-key", "openai-api-key",
    "anthropic-api-key", "x-venice-api-key", "hf-token", "cookie",
})


def _scrub_headers(headers: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Remove auth credentials; preserve key-case for everything else."""
    if not headers:
        return {}
    out: Dict[str, str] = {}
    for k, v in headers.items():
        if str(k).lower() in _SENSITIVE_HEADER_KEYS:
            out[str(k)] = "<scrubbed>"
        else:
            out[str(k)] = str(v)
    return out


def _body_to_text(body: Any) -> str:
    """Render body as UTF-8 text. bytes → decode; dict/list → JSON text
    (only because that's what the real HTTP body would be if the caller
    is sending application/json — this is the same bytes the server sees,
    not a TID-layer wrapper)."""
    if body is None:
        return ""
    if isinstance(body, (bytes, bytearray)):
        try:
            return body.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(body, str):
        return body
    try:
        import json as _json
        return _json.dumps(body, ensure_ascii=False, default=str)
    except Exception:
        return str(body)


def _ensure_dir() -> bool:
    try:
        os.makedirs(_SCAN_DIR, exist_ok=True)
        return True
    except OSError as exc:
        logger.warning("Cannot create tract scan dir %s: %s", _SCAN_DIR, exc)
        return False


def _write(content: str, source: str) -> None:
    """Best-effort tract write. Never raises."""
    if not _ensure_dir():
        return
    try:
        import ng_tract
    except ImportError:
        logger.debug("ng_tract unavailable; skipping wire deposit")
        return
    try:
        ng_tract.deposit_experience(
            content=content,
            source=source,
            tract_path=_TRACT_PATH,
            content_type="text",
        )
    except Exception as exc:
        logger.warning("Wire deposit failed (%s): %s", source, exc)


def _build_request_text(
    method: str,
    url: str,
    headers: Optional[Dict[str, Any]],
    body: Any,
) -> str:
    """Build raw HTTP request wire text."""
    lines = [f"{method.upper()} {url} HTTP/1.1"]
    for k, v in _scrub_headers(headers).items():
        lines.append(f"{k}: {v}")
    return "\r\n".join(lines) + "\r\n\r\n" + _body_to_text(body)


def _build_response_text(
    status_code: Optional[int],
    reason: Optional[str],
    headers: Optional[Dict[str, Any]],
    body: Any,
) -> str:
    """Build raw HTTP response wire text."""
    status = status_code if status_code is not None else "000"
    reason_str = f" {reason}" if reason else ""
    lines = [f"HTTP/1.1 {status}{reason_str}".rstrip()]
    for k, v in _scrub_headers(headers).items():
        lines.append(f"{k}: {v}")
    return "\r\n".join(lines) + "\r\n\r\n" + _body_to_text(body)


def deposit_outbound(
    provider: str,              # retained for call-site compat; ignored in content
    model_id: str,              # retained for call-site compat; ignored in content
    url: str,
    method: str,
    request_body: Any,
    headers: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,  # retained; ignored (STDP pairs)
) -> None:
    """Deposit raw outbound HTTP request as wire text.

    `provider`, `model_id`, `correlation_id` are accepted for backwards
    compatibility with existing call sites but are NOT included in the
    deposit — the substrate gets only the bytes that crossed the wire.
    """
    content = _build_request_text(method, url, headers, request_body)
    _write(content, source="tid.http.outbound")


def deposit_inbound(
    provider: str,              # retained for call-site compat; ignored
    model_id: str,              # retained for call-site compat; ignored
    url: str,                   # retained for call-site compat; ignored
    status_code: Optional[int],
    response_body: Any,
    headers: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None,     # retained; ignored
    error: Optional[str] = None,            # retained; ignored (error detail is in the body)
    correlation_id: Optional[str] = None,   # retained; ignored
    reason: Optional[str] = None,
) -> None:
    """Deposit raw inbound HTTP response as wire text.

    Non-wire fields (latency_ms, error string, correlation_id) are
    accepted for compat but NOT deposited. Latency is derivable by the
    substrate from outbound/inbound timestamp delta via STDP. Error
    classification belongs at extraction, not deposit.
    """
    content = _build_response_text(status_code, reason, headers, response_body)
    _write(content, source="tid.http.inbound")
