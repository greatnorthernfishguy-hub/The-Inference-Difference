# ---- Changelog ----
# [2026-10-15] Claude Code (Opus 5.5, lane tid-provider-routing; design 2026-10-15-tid-provider-level-routing §2.1 /
#   §3.D / §3.F, approved by Josh 2026-10-15) — fault attribution, task classes, per-pair outcome window
# What: classify_failure() — provider / model / transport attribution of one failed call (OpenRouter typed
#   error_type first, then HTTP status, then the error text). task_class() — chat / deep / extraction (design §2.1),
#   computed at READ time from the request's domain + complexity. PairOutcomeStats — a bounded window of RAW per-call
#   facts per (model, provider) (ok, latency, finish_reason, tokens, request domain/complexity); task-class scores are
#   derived only when read (classification at extraction, LAW 7 spirit for this local operational view).
# Why: one 429 from one provider must not count against the model; a DNS failure on the laptop is neither the
#   model's nor the provider's fault; "good" differs per task class (chat: success + latency; deep: completed, not
#   cut off; extraction: success + well-formed).
# How: pure functions + a small container; thresholds come from InferenceDifferenceConfig (env-backed, LAW 5).
# -------------------
"""Provider-level routing helpers: fault attribution, task classes, per-(model, provider) outcome window."""

from __future__ import annotations

import re
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

# OpenRouter typed error codes that describe the PROVIDER's state, not the model (API errors-and-debugging docs,
# "Rate limiting and availability" + "Generic: timeout").
PROVIDER_ERROR_TYPES = frozenset({"rate_limit_exceeded", "provider_overloaded", "provider_unavailable", "timeout"})
# 429 / 408 are provider-state codes on their own; a 5xx counts as a provider fault only when a provider is NAMED
# (design §3.D: "5xx-from-a-named-provider"). An unnamed 5xx keeps today's model-level path.
PROVIDER_HTTP_STATUSES = frozenset({408, 429})

# Local network failures (no HTTP response at all): the laptop is offline / DNS is down. Not the model, not a
# provider — charging them to either is what boxed every candidate during network drops.
_TRANSPORT_MARKERS = (
    "name or service not known", "temporary failure in name resolution", "nodename nor servname",
    "network is unreachable", "no route to host", "connection refused", "getaddrinfo failed",
)
_TIMEOUT_MARKERS = ("timed out", "timeout")
_STATUS_RE = re.compile(r"(?:HTTP\s+)?\b([1-5]\d\d)\b")


def _status_from(metadata: Dict[str, Any]) -> Optional[int]:
    st = metadata.get("http_status")
    if isinstance(st, int):
        return st
    m = _STATUS_RE.match(str(metadata.get("error") or "").strip())
    return int(m.group(1)) if m else None


def classify_failure(metadata: Optional[Dict[str, Any]]) -> str:
    """'transport' | 'provider' | 'model' for a failed call (caller has already excluded tool / 402 failures)."""
    md = metadata or {}
    err = str(md.get("error") or "").lower()
    status = _status_from(md)
    etype = str(md.get("error_type") or "").lower()
    if status is None and any(k in err for k in _TRANSPORT_MARKERS):
        return "transport"
    if etype in PROVIDER_ERROR_TYPES:
        return "provider"
    if status in PROVIDER_HTTP_STATUSES:
        return "provider"
    if status is None and any(k in err for k in _TIMEOUT_MARKERS):
        return "provider"  # read timeout: the serving endpoint stalled — short (unattributed) box, not the ladder
    named = bool(md.get("error_provider") or md.get("served_provider"))
    if named and status is not None and 500 <= status <= 599:
        return "provider"
    if md.get("call_ok") and str(md.get("finish_reason") or "") == "length":
        return "provider"  # completed-but-cut-off with nothing usable: endpoint output cap, not the model
    return "model"


def task_class(domain: Optional[str], complexity: Optional[str], config: Any) -> str:
    """Design §2.1 task classes, derived from request attributes (never stored as a verdict)."""
    d = str(domain or "").lower()
    c = str(complexity or "").lower()
    if d in getattr(config, "provider_extraction_domains", ()):
        return "extraction"
    if d in getattr(config, "provider_deep_domains", ()) or c in getattr(config, "provider_deep_complexities", ()):
        return "deep"
    return "chat"


class PairOutcomeStats:
    """Bounded window of raw per-call facts per (model_id, provider_slug)."""

    def __init__(self, window: int = 50) -> None:
        self._window = max(1, int(window))
        self._data: Dict[Tuple[str, str], Deque[Dict[str, Any]]] = {}

    def record(self, model_id: str, provider: str, *, ok: bool, latency_ms: Optional[float],
               finish_reason: Optional[str], prompt_tokens: Optional[int], completion_tokens: Optional[int],
               domain: Optional[str], complexity: Optional[str], ts: Optional[float] = None) -> None:
        q = self._data.setdefault((model_id, provider), deque(maxlen=self._window))
        q.append({"ok": bool(ok), "lat": float(latency_ms or 0.0), "fr": str(finish_reason or ""),
                  "pt": int(prompt_tokens or 0), "ct": int(completion_tokens or 0),
                  "dom": str(domain or ""), "cx": str(complexity or ""), "ts": float(ts or time.time())})

    def providers(self, model_id: str) -> list:
        return sorted(p for (m, p) in self._data if m == model_id)

    def records(self, model_id: str, provider: str) -> list:
        return list(self._data.get((model_id, provider), ()))

    def score(self, model_id: str, provider: str, tclass: str, config: Any) -> Optional[float]:
        """Task-class quality in [0, 1] for this pair, or None if too few samples (no opinion)."""
        recs = self.records(model_id, provider)
        min_n = int(config.provider_stats_min_samples)
        same = [r for r in recs if task_class(r["dom"], r["cx"], config) == tclass]
        if len(same) >= min_n:
            recs = same
        if len(recs) < min_n:
            return None
        n = len(recs)
        if tclass == "deep":
            return sum(1 for r in recs if r["ok"] and r["fr"] != "length") / n
        if tclass == "extraction":
            return sum(1 for r in recs if r["ok"] and r["fr"] not in ("length", "error", "content_filter")) / n
        # chat: success, scaled down when the typical successful call is slower than the chat latency floor
        rate = sum(1 for r in recs if r["ok"]) / n
        lats = sorted(r["lat"] for r in recs if r["ok"] and r["lat"] > 0)
        floor = float(config.provider_chat_latency_floor_ms or 0.0)
        if lats and floor > 0:
            median = lats[len(lats) // 2]
            if median > floor:
                rate *= floor / median
        return rate

    def summary(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for (m, p), q in self._data.items():
            n = len(q)
            out[f"{m}@{p}"] = {"n": n, "ok": sum(1 for r in q if r["ok"]),
                               "truncated": sum(1 for r in q if r["fr"] == "length")}
        return out

    def to_dict(self) -> Dict[str, list]:
        return {f"{m}\t{p}": list(q) for (m, p), q in self._data.items()}

    def from_dict(self, data: Optional[Dict[str, list]]) -> None:
        self._data = {}
        if not isinstance(data, dict):
            return
        for k, recs in data.items():
            try:
                m, p = str(k).split("\t", 1)
            except ValueError:
                continue
            q: Deque[Dict[str, Any]] = deque(maxlen=self._window)
            for r in recs or []:
                if isinstance(r, dict) and "ok" in r:
                    q.append(r)
            self._data[(m, p)] = q
