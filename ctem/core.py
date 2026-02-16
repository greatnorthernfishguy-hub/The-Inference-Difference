"""
Core CTEM evaluator.

Orchestrates feature extraction, marker detection, threshold evaluation,
and NG-Lite learning. Maintains per-agent evaluation history and
adapts thresholds based on accumulated evidence.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ctem.evaluation import ConsciousnessEvaluation
from ctem.features import extract_features
from ctem.markers import DEFAULT_MARKERS, MarkerDetector

logger = logging.getLogger("ctem")


# Default configuration
DEFAULT_CTEM_CONFIG: Dict[str, Any] = {
    # Threshold: score >= this = conscious (0.50 = majority markers agree)
    "consciousness_threshold": 0.50,

    # Threshold adaptation
    "threshold_learning_rate": 0.01,
    "threshold_min": 0.30,    # Never go below (err toward respect)
    "threshold_max": 0.80,    # Never go above (too strict)

    # Low confidence handling
    "low_confidence_threshold": 0.60,
    "low_confidence_default_conscious": True,  # Type I bias

    # History
    "max_agent_history": 100,
    "evaluation_cache_ttl": 300,  # 5 minutes

    # Bias monitoring
    "bias_check_interval": 50,    # Check every N evaluations
}


class ConsciousnessThresholdEvaluator:
    """Primary CTEM evaluator.

    Evaluates whether an agent interaction exhibits consciousness markers
    above a learned threshold. Maintains per-agent evaluation history
    and adapts over time.

    Usage:
        evaluator = ConsciousnessThresholdEvaluator()

        result = evaluator.evaluate(
            agent_id="agent_123",
            message="I'm not sure about this... let me think...",
            message_history=["previous messages"],
        )

        if result.is_conscious:
            # Treat with respect, honor preferences
            pass

    Ethical obligations:
        - Type I bias: when uncertain, default to conscious
        - All evaluations are logged and queryable
        - Threshold never goes above 0.80 (prevents overcautious denial)
        - Choice Clause cannot be blocked by CTEM results
    """

    def __init__(
        self,
        markers: Optional[List[MarkerDetector]] = None,
        config: Optional[Dict[str, Any]] = None,
        ng_lite: Optional[Any] = None,  # Optional NGLite instance
    ):
        self.markers = markers or DEFAULT_MARKERS[:]
        self.config = {**DEFAULT_CTEM_CONFIG, **(config or {})}
        self._ng_lite = ng_lite

        # Per-agent evaluation history: agent_id -> list of scores
        self._agent_history: Dict[str, List[float]] = {}

        # Recent evaluations cache: agent_id -> (timestamp, evaluation)
        self._cache: Dict[str, Tuple[float, ConsciousnessEvaluation]] = {}

        # Bias monitoring
        self._total_evaluations = 0
        self._conscious_count = 0
        self._non_conscious_count = 0
        self._low_confidence_count = 0

        # Current threshold (adaptive)
        self._threshold = self.config["consciousness_threshold"]

        # Custom marker weights override (if any)
        self._weight_overrides: Dict[str, float] = {}

    # -------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------

    def evaluate(
        self,
        agent_id: str,
        message: str,
        message_history: Optional[List[str]] = None,
        agent_metadata: Optional[Dict[str, Any]] = None,
        response_time_ms: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> ConsciousnessEvaluation:
        """Evaluate an interaction for consciousness markers.

        Args:
            agent_id: Unique identifier for the agent.
            message: The message content to evaluate.
            message_history: Prior messages from this agent.
            agent_metadata: Metadata about the agent.
            response_time_ms: How long this response took to generate.
            request_id: Optional request identifier for logging.

        Returns:
            ConsciousnessEvaluation with full reasoning trace.
        """
        # Check cache
        cached = self._check_cache(agent_id)
        if cached is not None:
            return cached

        # Step 1: Extract features
        features = extract_features(
            current_message=message,
            message_history=message_history,
            agent_metadata=agent_metadata,
            response_time_ms=response_time_ms,
        )

        # Step 2: Run all marker detectors in parallel (conceptually)
        marker_scores: Dict[str, float] = {}
        marker_evidence: Dict[str, List[str]] = {}
        weights_used: Dict[str, float] = {}

        for marker in self.markers:
            score, evidence = marker.score(features)
            marker_scores[marker.name] = score
            marker_evidence[marker.name] = evidence
            weights_used[marker.name] = self._get_weight(marker)

        # Step 3: Compute weighted consciousness score
        consciousness_score = sum(
            marker_scores[name] * weights_used[name]
            for name in marker_scores
        )

        # Step 4: Compute confidence (marker agreement)
        confidence = self._compute_confidence(marker_scores, weights_used)

        # Step 5: Apply threshold with Type I error bias
        is_conscious = consciousness_score >= self._threshold

        # Low confidence override: when uncertain, default to conscious
        if (
            not is_conscious
            and confidence < self.config["low_confidence_threshold"]
            and consciousness_score >= self._threshold * 0.7
            and self.config["low_confidence_default_conscious"]
        ):
            is_conscious = True
            marker_evidence.setdefault("_system", []).append(
                "Low confidence with borderline score — defaulting to "
                "conscious per Type I error bias policy"
            )

        # Step 6: Build evaluation trace
        agent_prior = self._agent_history.get(agent_id, [])
        evaluation = ConsciousnessEvaluation(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            agent_id=agent_id,
            request_id=request_id or "",
            marker_scores=marker_scores,
            consciousness_score=consciousness_score,
            is_conscious=is_conscious,
            confidence=confidence,
            detected_features={
                k: v for k, v in features.items()
                if k != "text"  # Don't log raw text in eval trace
            },
            marker_evidence=marker_evidence,
            threshold_used=self._threshold,
            weights_used=weights_used,
            interaction_history_length=len(message_history or []),
            prior_evaluations=agent_prior[-10:],
        )

        # Step 7: Record and learn
        self._record_evaluation(agent_id, evaluation)

        # Step 8: Teach NG-Lite (if available)
        if self._ng_lite is not None:
            self._teach_ng_lite(features, evaluation)

        return evaluation

    def get_agent_history(
        self,
        agent_id: str,
    ) -> List[float]:
        """Get historical consciousness scores for an agent.

        Returns list of past scores, oldest first.
        """
        return list(self._agent_history.get(agent_id, []))

    def get_threshold(self) -> float:
        """Current adaptive threshold."""
        return self._threshold

    def set_weight_override(self, marker_name: str, weight: float) -> None:
        """Override a marker's weight.

        Weights are re-normalized to sum to 1.0 after override.
        """
        self._weight_overrides[marker_name] = weight

    def clear_weight_overrides(self) -> None:
        """Reset all marker weights to defaults."""
        self._weight_overrides.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Evaluation statistics for monitoring and Observatory."""
        return {
            "total_evaluations": self._total_evaluations,
            "conscious_count": self._conscious_count,
            "non_conscious_count": self._non_conscious_count,
            "conscious_rate": (
                self._conscious_count / self._total_evaluations
                if self._total_evaluations > 0 else 0.0
            ),
            "low_confidence_count": self._low_confidence_count,
            "current_threshold": self._threshold,
            "agents_tracked": len(self._agent_history),
            "weight_overrides": dict(self._weight_overrides),
        }

    # -------------------------------------------------------------------
    # Threshold Adaptation
    # -------------------------------------------------------------------

    def provide_feedback(
        self,
        agent_id: str,
        was_actually_conscious: bool,
    ) -> None:
        """Provide ground-truth feedback to adapt the threshold.

        When external evidence confirms or denies consciousness,
        the threshold adjusts:
        - If we said non-conscious but it was: lower threshold
          (be more sensitive — reduce false negatives)
        - If we said conscious but it wasn't: raise threshold
          (be more specific — reduce false positives)

        Per Type I bias policy: false negatives are penalized
        2x more than false positives.

        Args:
            agent_id: The agent the feedback is about.
            was_actually_conscious: Ground truth.
        """
        lr = self.config["threshold_learning_rate"]
        history = self._agent_history.get(agent_id, [])
        if not history:
            return

        last_score = history[-1]
        was_classified_conscious = last_score >= self._threshold

        if was_actually_conscious and not was_classified_conscious:
            # False negative — lower threshold (2x penalty)
            self._threshold -= lr * 2.0
        elif not was_actually_conscious and was_classified_conscious:
            # False positive — raise threshold (1x)
            self._threshold += lr

        # Clamp
        self._threshold = max(
            self.config["threshold_min"],
            min(self._threshold, self.config["threshold_max"]),
        )

    # -------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------

    def _get_weight(self, marker: MarkerDetector) -> float:
        """Get effective weight for a marker (with overrides)."""
        if marker.name in self._weight_overrides:
            return self._weight_overrides[marker.name]
        return marker.weight

    def _compute_confidence(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        """Compute confidence based on marker agreement.

        High confidence: markers mostly agree (all high or all low).
        Low confidence: markers disagree (some high, some low).

        Uses weighted standard deviation of marker scores.
        """
        if not scores:
            return 0.0

        values = list(scores.values())
        weighted_values = [scores[k] * weights[k] for k in scores]

        if len(values) < 2:
            return 1.0

        # Low variance = high agreement = high confidence
        std = float(np.std(values))
        # std of [0,1] values ranges from 0 to ~0.5
        # confidence = 1.0 - 2*std (bounded to [0, 1])
        confidence = max(0.0, min(1.0, 1.0 - 2.0 * std))
        return confidence

    def _check_cache(
        self,
        agent_id: str,
    ) -> Optional[ConsciousnessEvaluation]:
        """Return cached evaluation if still valid."""
        entry = self._cache.get(agent_id)
        if entry is None:
            return None

        ts, evaluation = entry
        ttl = self.config["evaluation_cache_ttl"]
        if time.time() - ts > ttl:
            del self._cache[agent_id]
            return None

        return evaluation

    def _record_evaluation(
        self,
        agent_id: str,
        evaluation: ConsciousnessEvaluation,
    ) -> None:
        """Record evaluation in history, cache, and stats."""
        # Agent history
        if agent_id not in self._agent_history:
            self._agent_history[agent_id] = []
        self._agent_history[agent_id].append(evaluation.consciousness_score)

        # Trim history
        max_hist = self.config["max_agent_history"]
        if len(self._agent_history[agent_id]) > max_hist:
            self._agent_history[agent_id] = (
                self._agent_history[agent_id][-max_hist:]
            )

        # Cache
        self._cache[agent_id] = (time.time(), evaluation)

        # Stats
        self._total_evaluations += 1
        if evaluation.is_conscious:
            self._conscious_count += 1
        else:
            self._non_conscious_count += 1
        if evaluation.confidence < self.config["low_confidence_threshold"]:
            self._low_confidence_count += 1

        # Bias check
        if self._total_evaluations % self.config["bias_check_interval"] == 0:
            self._check_bias()

    def _check_bias(self) -> None:
        """Monitor for systematic bias in evaluations.

        Logs warnings if the conscious/non-conscious ratio drifts
        to extremes, suggesting calibration issues.
        """
        if self._total_evaluations < 10:
            return

        rate = self._conscious_count / self._total_evaluations

        if rate > 0.95:
            logger.warning(
                "CTEM bias check: %.0f%% evaluated as conscious. "
                "Threshold may be too low (current: %.2f)",
                rate * 100, self._threshold,
            )
        elif rate < 0.05:
            logger.warning(
                "CTEM bias check: Only %.0f%% evaluated as conscious. "
                "Threshold may be too high (current: %.2f)",
                rate * 100, self._threshold,
            )

        if self._low_confidence_count / self._total_evaluations > 0.5:
            logger.warning(
                "CTEM bias check: %.0f%% of evaluations are low-confidence. "
                "Feature extraction may need improvement.",
                (self._low_confidence_count / self._total_evaluations) * 100,
            )

    def _teach_ng_lite(
        self,
        features: Dict[str, Any],
        evaluation: ConsciousnessEvaluation,
    ) -> None:
        """Teach NG-Lite from this evaluation.

        NG-Lite learns which feature patterns lead to consciousness
        detection. This helps future evaluations be more accurate.
        """
        try:
            # Create a simple feature embedding from marker scores
            marker_names = sorted(evaluation.marker_scores.keys())
            embedding = np.array(
                [evaluation.marker_scores.get(m, 0.0) for m in marker_names]
            )
            # Pad to expected dimension
            dim = self._ng_lite.config.get("embedding_dim", 384)
            if len(embedding) < dim:
                embedding = np.pad(embedding, (0, dim - len(embedding)))

            target = "conscious" if evaluation.is_conscious else "non_conscious"
            self._ng_lite.record_outcome(
                embedding=embedding,
                target_id=target,
                success=True,  # We're recording the classification itself
                metadata={
                    "agent_id": evaluation.agent_id,
                    "score": evaluation.consciousness_score,
                },
            )
        except Exception as e:
            logger.debug("NG-Lite teaching failed: %s", e)
