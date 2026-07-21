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

Write path (#80, 2026-07-21): TID Substrate Peninsula (tid_peninsula_body.deposit_wire)
  -> Commons-side (tid_peninsula_commons._forward_wire_experience) -> Commons deposit,
  windowed by recency (commons.py._evict_old_wire). No longer written to
  ~/.et_modules/experience/inference_difference.tract / drained via wire_absorption.py —
  that path force-wrote every deposit into Syl's cognitive _memory graph as a per-call
  orphan node (60,142 observed), which is what this flip fixes.

# ---- Changelog ----
# [2026-07-21] Claude Code (Sonnet 5) — #80 wire → Commons: flip _write() to the peninsula (Task 4)
#   What: _write() no longer calls ng_tract.deposit_experience against the
#         inference_difference.tract scan-dir file. It now sends a wire_experience frame
#         through tid_peninsula_body.get_peninsula().deposit_wire() — content-derived
#         target_id (sha256(content)[:16], LAW 7) computed here, embedding happens
#         Commons-side (Task 2/3). deposit_outbound/deposit_inbound signatures unchanged;
#         only the internal write path moved.
#   Why:  Design v4 / implementation plan Task 4, the "flip." TID's raw wire deposits were
#         landing in ~/.et_modules/experience/inference_difference.tract, drained by
#         NeuroGraph's neurograph_rpc.py:_drain_scan_dir() into wire_absorption.py, which
#         force-wrote them into Syl's cognitive _memory graph as per-call orphan nodes — a
#         churn firehose (60,142 observed orphans). Wire traffic belongs in the shared
#         Commons instead (recency-windowed via commons.py._evict_old_wire, #80 Task 1),
#         not Syl's primary substrate. Task 1-3 landed the receiving/eviction/send
#         infrastructure inert; this task is the actual behavior change.
#   How:  Fail-soft, fire-and-forget, matching the peninsula's existing posture: if
#         get_peninsula() returns None (not started) or the socket isn't connected
#         (NeuroGraph down, or TID running standalone), the deposit is silently dropped —
#         no fallback to the old tract write, no blocking, no raise. ng_tract import
#         removed from this module (no remaining caller); _TRACT_PATH/_SCAN_DIR/_ensure_dir
#         retained as dead-but-harmless in case a future consumer needs the constants, but
#         are no longer written to by _write().
# [2026-05-25] Claude Code (Opus 4.7) — Sync to new ng_tract.deposit_experience signature
#   What: _write() now calls ng_tract.deposit_experience(raw_bytes, source, [tract_path])
#         positionally, matching the new Rust API. content is UTF-8 encoded;
#         tract_path is wrapped in a list.
#   Why:  NeuroGraph commit 59c0703 (Sonnet 4.6, 2026-05-25 07:19 UTC) implemented
#         the deposit_experience/deposit_topology Rust API on the producer side
#         after AttributeError had been silently swallowed since 2026-04-28.
#         Old kwargs (content=, tract_path=, content_type=) hit TypeError every
#         call, severing TID's River feed. LAW 4: consumer side fix.
#   How:  Three-arg positional call: content.encode("utf-8"), source, [_TRACT_PATH].
#         No behavior change for callers; tract_path environment unchanged.
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

import hashlib
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
    """Best-effort peninsula deposit. Never raises.

    #80 Task 4 flip: routes through the Commons Peninsula (tid_peninsula_body.deposit_wire)
    instead of the retired ng_tract/inference_difference.tract write. target_id is
    content-derived (LAW 7) — wire:{source}:{sha256(content)[:16]} — matching the shape
    commons.py._evict_old_wire windows on (Commons-side, #80 Task 1). Fail-soft: no
    peninsula, no connection, or any send error is silently swallowed; there is no
    fallback write path.
    """
    if not content:
        return
    try:
        from inference_difference.tid_peninsula_body import get_peninsula
    except ImportError:
        logger.debug("tid_peninsula_body unavailable; skipping wire deposit")
        return
    try:
        peninsula = get_peninsula()
        if peninsula is None:
            return
        target_id = f"wire:{source}:{hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]}"
        peninsula.deposit_wire(content, target_id)
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
