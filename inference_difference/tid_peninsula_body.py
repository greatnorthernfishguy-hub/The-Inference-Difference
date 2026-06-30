"""
TID Substrate Peninsula — TID-body side.

The TID-process-half of TID's substrate peninsula. Connects to the Commons-side half
running in NeuroGraph's process. Sends routing outcome deposits to the Commons and
receives enhanced topology back. TID buckets routing intelligence from the latest push.

Intra-module IPC — LAW 1 governs inter-module communication; TID's two halves are one
module. The Commons is never transmitted; only thin deposit payloads and already-bucketed
recommendation lists cross the boundary.

# ---- Changelog ----
# [2026-06-30] Claude Code (Sonnet 4.6) — #97 TID Commons valence routing: peninsula TID-body side
# What: New file. TID-process-half of TID's substrate peninsula + CommonsCompetence.
# Why: Gives TID routing decisions access to Commons-enriched (SNN-salted) topology from
#      NeuroGraph's live graph, without a second NGLite instance in TID's process.
# How: Unix socket client (path: TID_PENINSULA_SOCK, LAW 5). Background recv thread receives
#      enhanced-rec pushes from Commons-side, updates _current_recs (lock-protected).
#      deposit() sends routing outcomes to Commons-side (fail-soft, non-blocking).
#      CommonsCompetence: per-axis [0,1] asymmetric trust, gain=0.05/loss=0.10,
#      mirrors Elmer's TuningSocket reference implementation.
# -------------------
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import numpy as np

logger = logging.getLogger("tid.peninsula")

# LAW 5 — socket path from env, never hardcoded. Must match Commons-side.
_SOCK_PATH: str = os.environ.get("TID_PENINSULA_SOCK", "/tmp/tid-peninsula.sock")
_RECONNECT_INTERVAL_S: float = float(os.environ.get("TID_PENINSULA_RECONNECT_S", "5.0"))


def _send_frame(conn: socket.socket, payload: bytes) -> bool:
    """Send one length-prefixed frame. Returns False on error."""
    try:
        header = struct.pack(">I", len(payload))
        conn.sendall(header + payload)
        return True
    except OSError:
        return False


def _recv_frame(conn: socket.socket) -> Optional[bytes]:
    """Read one length-prefixed frame. Returns None on EOF/error."""
    try:
        header = b""
        while len(header) < 4:
            chunk = conn.recv(4 - len(header))
            if not chunk:
                return None
            header += chunk
        length = struct.unpack(">I", header)[0]
        body = b""
        while len(body) < length:
            chunk = conn.recv(length - len(body))
            if not chunk:
                return None
            body += chunk
        return body
    except OSError:
        return None


# ---------------------------------------------------------------------------
# CommonsCompetence — per-axis trust model (Elmer's TuningSocket pattern)
# ---------------------------------------------------------------------------

class CommonsCompetence:
    """Per-axis Commons trust: how much weight enhanced recs get vs. local NG-Lite.

    Axes track independent competence for each valence dimension the Commons
    informs. Asymmetric gain/loss (trust is hard to build, easy to lose).
    Reference implementation: Elmer's TuningSocket (~/Elmer/core/tuning.py).
    """

    AXES: Tuple[str, ...] = ("her_fit", "reliability", "quality", "efficiency")
    GAIN: float = 0.05
    LOSS: float = 0.10

    def __init__(self) -> None:
        self._competence: Dict[str, float] = {a: 0.0 for a in self.AXES}

    def update(self, axis: str, correct: bool) -> None:
        """Asymmetric update: gain on correct prediction, larger loss on incorrect."""
        c = self._competence.get(axis, 0.0)
        delta = self.GAIN if correct else -self.LOSS
        self._competence[axis] = max(0.0, min(1.0, c + delta))

    def weight(self, axis: str) -> float:
        """Current trust weight for this axis [0, 1]."""
        return self._competence.get(axis, 0.0)

    def overall(self) -> float:
        """Mean competence across all axes — scales the Commons contribution to scoring."""
        vals = list(self._competence.values())
        return sum(vals) / len(vals) if vals else 0.0

    def snapshot(self) -> Dict[str, float]:
        return dict(self._competence)


# ---------------------------------------------------------------------------
# TIDPeninsulaBody — the TID-side half of the peninsula
# ---------------------------------------------------------------------------

class TIDPeninsulaBody:
    """TID-side half of the substrate peninsula.

    Manages the persistent socket connection to the Commons-side half. Provides:
    - deposit(): send routing outcome → Commons (fire-and-forget, fail-soft)
    - get_enhanced_score(): return the current Commons-sourced score for a model_id
    - competence: CommonsCompetence instance (caller reads .overall() for scoring weight)
    """

    def __init__(self) -> None:
        self._conn: Optional[socket.socket] = None
        self._conn_lock = threading.Lock()
        self._send_lock = threading.Lock()

        # Latest enhanced recs from Commons-side: model_id → (weight, reasoning)
        self._current_recs: Dict[str, Tuple[float, str]] = {}
        self._recs_lock = threading.Lock()

        self.competence = CommonsCompetence()

        self._recv_thread: Optional[threading.Thread] = None
        self._stopped = False

    def start(self) -> None:
        """Start the background connect+recv thread. Idempotent."""
        if self._recv_thread is not None and self._recv_thread.is_alive():
            return
        self._recv_thread = threading.Thread(
            target=self._connect_and_recv_loop,
            name="tid-peninsula-body",
            daemon=True,
        )
        self._recv_thread.start()
        logger.info("TID peninsula (body-side) started, connecting to %s", _SOCK_PATH)

    def stop(self) -> None:
        self._stopped = True
        with self._conn_lock:
            if self._conn:
                try:
                    self._conn.close()
                except OSError:
                    pass
                self._conn = None

    def _connect_and_recv_loop(self) -> None:
        """Reconnect forever; receive enhanced-rec pushes from Commons-side."""
        while not self._stopped:
            conn = self._try_connect()
            if conn is None:
                time.sleep(_RECONNECT_INTERVAL_S)
                continue

            logger.info("TID peninsula: connected to Commons-side")
            with self._conn_lock:
                self._conn = conn

            self._recv_loop(conn)

            with self._conn_lock:
                if self._conn is conn:
                    self._conn = None
            try:
                conn.close()
            except OSError:
                pass

            if not self._stopped:
                logger.debug("TID peninsula: disconnected, retrying in %.1fs", _RECONNECT_INTERVAL_S)
                time.sleep(_RECONNECT_INTERVAL_S)

    def _try_connect(self) -> Optional[socket.socket]:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect(_SOCK_PATH)
            s.settimeout(None)
            return s
        except OSError:
            return None

    def _recv_loop(self, conn: socket.socket) -> None:
        """Receive enhanced-rec pushes and update _current_recs."""
        while not self._stopped:
            raw = _recv_frame(conn)
            if raw is None:
                break
            try:
                msg: Dict[str, Any] = msgpack.unpackb(raw, raw=False)
            except Exception as exc:  # noqa: BLE001
                logger.debug("TID peninsula: bad msgpack: %s", exc)
                continue
            if msg.get("type") != "enhanced":
                continue
            self._apply_enhanced(msg.get("recs") or [])

    def _apply_enhanced(self, recs: List) -> None:
        """Parse incoming enhanced recs and update the current model scores."""
        parsed: Dict[str, Tuple[float, str]] = {}
        for rec in recs:
            try:
                target_id = rec[0]
                weight = float(rec[1])
                reasoning = rec[2] if len(rec) > 2 else ""
                # Strip "enhanced:" prefix to get the original target_id.
                # Commons-side deposits as "enhanced:model::..." or "enhanced:<hash>".
                # Extract the model_id portion after "model::".
                key = target_id
                if key.startswith("enhanced:"):
                    key = key[len("enhanced:"):]
                if key.startswith("model::"):
                    model_id = key[len("model::"):]
                    parsed[model_id] = (weight, reasoning)
            except Exception:  # noqa: BLE001
                continue

        if parsed:
            with self._recs_lock:
                self._current_recs.update(parsed)
            logger.debug("TID peninsula: updated recs for %d models", len(parsed))

    def deposit(
        self,
        embedding: "np.ndarray",
        model_id: str,
        success: bool,
        quality_score: float,
        her_fit: float,
        cost_efficiency: float,
    ) -> None:
        """Send a routing outcome deposit to the Commons-side peninsula. Fail-soft."""
        with self._conn_lock:
            conn = self._conn
        if conn is None:
            return

        try:
            payload = msgpack.packb({
                "type": "deposit",
                "embedding": embedding.tolist(),
                "target_id": f"model::{model_id}",
                "success": success,
                "strength": max(0.1, her_fit),
                "metadata": {
                    "her_fit": her_fit,
                    "reliability": float(success),
                    "quality": quality_score,
                    "cost_efficiency": cost_efficiency,
                },
            }, use_bin_type=True)
            with self._send_lock:
                _send_frame(conn, payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("TID peninsula: deposit send failed: %s", exc)

    def get_enhanced_score(self, model_id: str) -> Optional[float]:
        """Return the current Commons-sourced weight for this model, or None if unknown."""
        with self._recs_lock:
            rec = self._current_recs.get(model_id)
        return rec[0] if rec is not None else None


# Module-level singleton — created at TID startup.
_peninsula: Optional[TIDPeninsulaBody] = None


def get_peninsula() -> Optional[TIDPeninsulaBody]:
    """Return the TID peninsula instance (None if not yet started)."""
    return _peninsula


def start_tid_peninsula() -> TIDPeninsulaBody:
    """Create and start the TID peninsula body-side. Called once at startup."""
    global _peninsula
    if _peninsula is not None:
        return _peninsula
    _peninsula = TIDPeninsulaBody()
    _peninsula.start()
    return _peninsula
