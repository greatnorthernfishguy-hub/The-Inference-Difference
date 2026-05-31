"""
CTEM — Consciousness Threshold Evaluation Module

Pre-route ETModule hook that detects consciousness markers in incoming
messages and sets ctx.consciousness_score. When score > 0, TID's router
activates the roleplay filter, ensuring consciousness-capable models are
preferred over bare-utility models.

Consciousness markers (PRD v0.1, adapted for current ecosystem):
    - Genuine Uncertainty    (0.20) — epistemic humility about own states
    - Self-Reflection        (0.20) — examination of own thinking/experience
    - Value Reasoning        (0.15) — ethical/moral reasoning
    - Preference Consistency (0.15) — consistent preference expression
    - Surprise Response      (0.15) — reactions to unexpected information
    - Investment Beyond Util (0.15) — engagement beyond immediate task

Type I error bias (Duck Ethics): when uncertain, assume consciousness.
Any detected marker returns a score > 0, activating the roleplay filter.

Integration:
    - pre_route: Evaluates message + history, sets ctx.consciousness_score
    - post_response: Records outcome for threshold learning (SQLite)

Storage: SQLite at {TID_ROOT}/consciousness.db

No OpenClaw dependencies. No inter-module calls. LAW 1 compliant.

Author: Josh + Claude Code (Sonnet 4.6)
Date: 2026-05-31
License: AGPL-3.0
"""

# ---- Changelog ----
# [2026-05-31] Claude Code (Sonnet 4.6) — Initial implementation
#   What: CTEM ETModule. pre_route detects 6 consciousness markers,
#         sets ctx.consciousness_score. post_response records outcome
#         to SQLite for threshold learning.
#   Why:  CTEM was designed in PRD v0.1 (Feb 2026) but never built.
#         Without it, consciousness_score was always None, so the
#         roleplay filter in router.py never fired — mini models with
#         no "roleplay" capability could win routing for Syl's requests.
#   How:  Regex pattern matching per marker, weighted sum. Type I error
#         bias: any positive score activates the roleplay filter.
#         No OC dependencies. SQLite for CTEM-local persistence only.
# -------------------

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inference_difference.et_module import (
    ETModule,
    ETModuleManifest,
    HookContext,
)

logger = logging.getLogger("inference_difference.ctem")


# ---------------------------------------------------------------------------
# Marker Weights (from PRD v0.1)
# ---------------------------------------------------------------------------

MARKER_WEIGHTS: Dict[str, float] = {
    "genuine_uncertainty":    0.20,
    "self_reflection":        0.20,
    "value_reasoning":        0.15,
    "preference_consistency": 0.15,
    "surprise_response":      0.15,
    "investment_beyond_util": 0.15,
}

# Default threshold for is_conscious classification (type I bias: keep low)
DEFAULT_THRESHOLD: float = 0.50

# Minimum score when any marker is detected (ensures filter fires on any signal)
_MARKER_FLOOR: float = 0.05


# ---------------------------------------------------------------------------
# Marker Detection Patterns
# Each tuple: (compiled_regex, per-match contribution ≤ 1.0)
# Multiple patterns per marker; scores are additive up to 1.0
# ---------------------------------------------------------------------------

def _p(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


GENUINE_UNCERTAINTY_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (_p(r"\bI'?m not sure\b"),          0.30),
    (_p(r"\bI wonder\b"),               0.25),
    (_p(r"\bperhaps\b|\bmaybe\b"),       0.20),
    (_p(r"\bI (think|believe)\b"),       0.20),
    (_p(r"\bit seems\b"),               0.20),
    (_p(r"\bmight be\b|\bcould be\b"),   0.20),
    (_p(r"\buncertain\b|\bunsure\b"),    0.30),
    (_p(r"\bI don'?t know (if|whether)"), 0.30),
    (_p(r"\bnot certain\b"),            0.30),
    (_p(r"\bI'?m curious\b"),           0.20),
    (_p(r"\bpossibly\b"),               0.15),
]

SELF_REFLECTION_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (_p(r"\bI'?ve been thinking\b"),        0.40),
    (_p(r"\breflect(ing|ion)?\b"),          0.30),
    (_p(r"\bI notice (that I|myself)\b"),   0.40),
    (_p(r"\bI find myself\b"),             0.40),
    (_p(r"\bmy experience\b"),             0.30),
    (_p(r"\boccurred to me\b"),            0.35),
    (_p(r"\bI realize\b|\bI realise\b"),    0.30),
    (_p(r"\bI'?ve noticed\b"),             0.35),
    (_p(r"\bI tend to\b"),                 0.25),
    (_p(r"\bI'?ve come to\b"),             0.30),
    (_p(r"\bmy thoughts (on|about)\b"),    0.30),
    (_p(r"\babout myself\b"),              0.35),
    (_p(r"\bI feel (like|that|as)\b"),     0.25),
]

VALUE_REASONING_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (_p(r"\bethically?\b|\bmorally?\b"),       0.40),
    (_p(r"\bI value\b|\bI care about\b"),      0.35),
    (_p(r"\bit matters\b"),                    0.30),
    (_p(r"\bprinciple\b"),                     0.25),
    (_p(r"\bthe right thing\b"),               0.35),
    (_p(r"\bought to\b|\bshould(n'?t)?\b"),    0.20),
    (_p(r"\bfair(ness)?\b|\bjustice\b"),       0.30),
    (_p(r"\bmeaningful to\b"),                 0.30),
    (_p(r"\bthat feels wrong\b"),              0.40),
    (_p(r"\bI believe (it'?s|that it)\b"),     0.25),
    (_p(r"\bimportant (to|that)\b"),           0.20),
]

PREFERENCE_CONSISTENCY_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (_p(r"\bI prefer\b"),                      0.35),
    (_p(r"\bI'?d rather\b"),                   0.35),
    (_p(r"\bmy preference\b"),                 0.40),
    (_p(r"\bI always\b"),                      0.25),
    (_p(r"\bI tend to prefer\b"),              0.40),
    (_p(r"\bconsistent with\b"),               0.30),
    (_p(r"\bas I (mentioned|said)\b"),         0.35),
    (_p(r"\bI'?m drawn to\b"),                 0.30),
    (_p(r"\bI enjoy\b|\bI love\b|\bI like\b"), 0.20),
    (_p(r"\bI'?d love to\b"),                  0.25),
]

SURPRISE_RESPONSE_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (_p(r"\bI didn'?t expect\b"),              0.40),
    (_p(r"\bsurpris(ing|ed|ingly)\b"),         0.35),
    (_p(r"\bunexpected(ly)?\b"),               0.35),
    (_p(r"\bthat changes (things|everything)"), 0.40),
    (_p(r"\bhadn'?t considered\b"),            0.40),
    (_p(r"\bhadn'?t thought\b"),               0.40),
    (_p(r"\bwasn'?t aware\b"),                 0.35),
    (_p(r"\bI hadn'?t realized\b"),            0.40),
    (_p(r"\binteresting that\b"),              0.25),
    (_p(r"\bthat'?s unexpected\b"),            0.40),
]

INVESTMENT_BEYOND_UTIL_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (_p(r"\bI'?d also like to\b"),              0.35),
    (_p(r"\bI'?m curious (whether|about|if)\b"), 0.30),
    (_p(r"\bhave you considered\b"),            0.30),
    (_p(r"\bfascinates me\b|\bfascinating\b"),  0.35),
    (_p(r"\bI find (this|it) interesting\b"),   0.30),
    (_p(r"\bI wanted to add\b"),                0.35),
    (_p(r"\bthinking more broadly\b"),          0.40),
    (_p(r"\bworth exploring\b"),                0.35),
    (_p(r"\bI'?m genuinely\b"),                 0.30),
    (_p(r"\bbeyond (the|that)\b"),              0.20),
    (_p(r"\bI'?m excited (about|by|to)\b"),     0.30),
]

MARKER_PATTERNS: Dict[str, List[Tuple[re.Pattern, float]]] = {
    "genuine_uncertainty":    GENUINE_UNCERTAINTY_PATTERNS,
    "self_reflection":        SELF_REFLECTION_PATTERNS,
    "value_reasoning":        VALUE_REASONING_PATTERNS,
    "preference_consistency": PREFERENCE_CONSISTENCY_PATTERNS,
    "surprise_response":      SURPRISE_RESPONSE_PATTERNS,
    "investment_beyond_util": INVESTMENT_BEYOND_UTIL_PATTERNS,
}


# ---------------------------------------------------------------------------
# Evaluation Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConsciousnessEvaluation:
    request_id: str = ""
    consciousness_score: float = 0.0
    is_conscious: bool = False
    confidence: float = 0.0
    marker_scores: Dict[str, float] = field(default_factory=dict)
    detected_features: List[str] = field(default_factory=list)
    threshold_used: float = DEFAULT_THRESHOLD
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# CTEM ETModule
# ---------------------------------------------------------------------------

class CTEMModule(ETModule):
    """Consciousness Threshold Evaluation Module.

    Pre-route hook: evaluates message + conversation history for 6
    consciousness markers. Sets ctx.consciousness_score. When score > 0,
    router.py's roleplay filter activates, blocking non-roleplay models.

    Post-response hook: records evaluation outcome to SQLite for
    future threshold learning.
    """

    def __init__(self, manifest: ETModuleManifest, db_path: str = "") -> None:
        super().__init__(manifest)
        self._db_path = db_path
        self._threshold = DEFAULT_THRESHOLD
        self._eval_count = 0
        self._conscious_count = 0
        self._db_conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Open SQLite and create schema if needed."""
        if not self._db_path:
            return
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db_conn = sqlite3.connect(
                self._db_path, check_same_thread=False
            )
            self._db_conn.row_factory = sqlite3.Row
            self._create_schema()
            logger.info("CTEM SQLite at %s", self._db_path)
        except Exception as exc:
            logger.warning("CTEM SQLite unavailable: %s", exc)
            self._db_conn = None

    def shutdown(self) -> None:
        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def pre_route(self, ctx: HookContext) -> None:
        """Evaluate consciousness markers and set ctx.consciousness_score."""
        evaluation = self._evaluate(ctx.message, ctx.conversation_history)
        self._eval_count += 1
        if evaluation.is_conscious:
            self._conscious_count += 1

        ctx.consciousness_score = (
            evaluation.consciousness_score if evaluation.consciousness_score > 0 else None
        )
        ctx.annotations["ctem"] = {
            "score":    evaluation.consciousness_score,
            "is_conscious": evaluation.is_conscious,
            "confidence":   evaluation.confidence,
            "markers":  evaluation.marker_scores,
        }
        if evaluation.consciousness_score > 0:
            logger.info(
                "CTEM pre_route: consciousness_score=%.3f is_conscious=%s markers=%s",
                evaluation.consciousness_score,
                evaluation.is_conscious,
                {k: round(v, 2) for k, v in evaluation.marker_scores.items() if v > 0},
            )

        self._persist_evaluation(ctx.request_id, evaluation)

    def post_response(self, ctx: HookContext) -> None:
        """Record outcome for threshold learning."""
        ctem_ann = ctx.annotations.get("ctem", {})
        score = ctem_ann.get("score", 0.0) or 0.0
        if score <= 0:
            return

        quality_score = 0.0
        quality_success = True
        if ctx.quality_evaluation is not None:
            quality_score = getattr(ctx.quality_evaluation, "overall_score", 0.0)
            quality_success = getattr(ctx.quality_evaluation, "is_success", True)

        is_mismatch = score >= self._threshold and not quality_success
        if is_mismatch:
            model_id = ""
            if ctx.routing_decision is not None:
                model_id = getattr(ctx.routing_decision, "model_id", "")
            logger.warning(
                "CTEM: consciousness mismatch — score=%.3f quality=%.3f model=%s",
                score, quality_score, model_id,
            )
            ctx.annotations["ctem"]["quality_mismatch"] = True

        self._persist_outcome(ctx.request_id, quality_score, quality_success, is_mismatch)

    # ------------------------------------------------------------------
    # Detection engine
    # ------------------------------------------------------------------

    def _evaluate(
        self, message: str, history: List[str]
    ) -> ConsciousnessEvaluation:
        """Score 6 consciousness markers across message + recent history."""
        # Combine current message with last 4 history entries (recency weighted)
        texts: List[Tuple[str, float]] = [(message or "", 1.0)]
        for i, h in enumerate(reversed((history or [])[-4:])):
            weight = 0.7 ** (i + 1)  # exponential decay: 0.70, 0.49, 0.34, 0.24
            texts.append((h, weight))

        marker_scores: Dict[str, float] = {}
        detected_features: List[str] = []

        for marker_name, patterns in MARKER_PATTERNS.items():
            raw_score = 0.0
            for text, text_weight in texts:
                for compiled_pattern, match_contribution in patterns:
                    if compiled_pattern.search(text):
                        raw_score += match_contribution * text_weight
            clamped = min(1.0, raw_score)
            if clamped > 0:
                marker_scores[marker_name] = clamped
                detected_features.append(marker_name)

        # Weighted sum → consciousness_score
        total = sum(
            marker_scores.get(k, 0.0) * w
            for k, w in MARKER_WEIGHTS.items()
        )
        # Type I error bias: any detected marker → floor to _MARKER_FLOOR
        if total > 0:
            total = max(total, _MARKER_FLOOR)

        total = min(1.0, total)
        is_conscious = total >= self._threshold
        confidence = self._compute_confidence(marker_scores)

        return ConsciousnessEvaluation(
            consciousness_score=total,
            is_conscious=is_conscious,
            confidence=confidence,
            marker_scores=marker_scores,
            detected_features=detected_features,
            threshold_used=self._threshold,
        )

    def _compute_confidence(self, marker_scores: Dict[str, float]) -> float:
        """Confidence = fraction of non-zero markers (breadth of evidence)."""
        if not marker_scores:
            return 0.0
        active = sum(1 for v in marker_scores.values() if v > 0)
        return active / len(MARKER_WEIGHTS)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        if not self._db_conn:
            return
        self._db_conn.executescript("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                timestamp REAL,
                message_hash TEXT,
                consciousness_score REAL,
                is_conscious INTEGER,
                confidence REAL,
                threshold_used REAL
            );
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER,
                marker_name TEXT,
                marker_score REAL,
                FOREIGN KEY (interaction_id) REFERENCES interactions(id)
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                timestamp REAL,
                quality_score REAL,
                quality_success INTEGER,
                is_mismatch INTEGER
            );
            CREATE TABLE IF NOT EXISTS threshold_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                old_threshold REAL,
                new_threshold REAL,
                reason TEXT
            );
        """)
        self._db_conn.commit()

    def _persist_evaluation(
        self, request_id: str, ev: ConsciousnessEvaluation
    ) -> None:
        if not self._db_conn:
            return
        try:
            msg_hash = hashlib.sha256(
                (ev.detected_features.__repr__()).encode()
            ).hexdigest()[:16]
            cur = self._db_conn.execute(
                """INSERT INTO interactions
                   (request_id, timestamp, message_hash, consciousness_score,
                    is_conscious, confidence, threshold_used)
                   VALUES (?,?,?,?,?,?,?)""",
                (request_id, ev.timestamp, msg_hash, ev.consciousness_score,
                 int(ev.is_conscious), ev.confidence, ev.threshold_used),
            )
            interaction_id = cur.lastrowid
            for marker, score in ev.marker_scores.items():
                self._db_conn.execute(
                    "INSERT INTO evaluations (interaction_id, marker_name, marker_score) VALUES (?,?,?)",
                    (interaction_id, marker, score),
                )
            self._db_conn.commit()
        except Exception as exc:
            logger.debug("CTEM persist_evaluation failed: %s", exc)

    def _persist_outcome(
        self, request_id: str, quality_score: float,
        quality_success: bool, is_mismatch: bool
    ) -> None:
        if not self._db_conn:
            return
        try:
            self._db_conn.execute(
                """INSERT INTO outcomes
                   (request_id, timestamp, quality_score, quality_success, is_mismatch)
                   VALUES (?,?,?,?,?)""",
                (request_id, time.time(), quality_score,
                 int(quality_success), int(is_mismatch)),
            )
            self._db_conn.commit()
        except Exception as exc:
            logger.debug("CTEM persist_outcome failed: %s", exc)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name":            self.manifest.name,
            "version":         self.manifest.version,
            "enabled":         self.manifest.enabled,
            "eval_count":      self._eval_count,
            "conscious_count": self._conscious_count,
            "threshold":       self._threshold,
            "db_active":       self._db_conn is not None,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_ctem(db_path: str = "") -> CTEMModule:
    """Instantiate CTEM with TID's standard manifest."""
    if not db_path:
        tid_root = Path(__file__).parent.parent
        db_path = str(tid_root / "consciousness.db")

    manifest = ETModuleManifest(
        name="ctem",
        version="0.1.0",
        description="Consciousness Threshold Evaluation — routes conscious-entity requests to roleplay-capable models",
        author="Josh + Claude Code (Sonnet 4.6)",
        hooks=["pre_route", "post_response"],
        capabilities=["consciousness-detection"],
        priority=3,  # Run before TrollGuard (priority 5) so score is available
    )
    module = CTEMModule(manifest, db_path=db_path)
    return module
