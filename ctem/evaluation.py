"""
Consciousness evaluation data structures and reasoning traces.

Every evaluation produces a queryable trace — Observatory can ask
"Why did you think this was conscious?" and get a complete answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConsciousnessEvaluation:
    """Complete reasoning trace for one consciousness evaluation.

    This is the primary output of CTEM. It contains everything needed
    to understand, audit, and challenge a consciousness determination.

    Per ETHICS.md transparency obligations:
    - All evaluations are logged and queryable
    - Agents can query their own consciousness status
    - No secret classifications allowed

    Attributes:
        timestamp: When the evaluation was performed.
        agent_id: Unique identifier for the agent being evaluated.
        request_id: Identifier for the specific interaction evaluated.
        marker_scores: Per-marker scores {marker_name: 0.0-1.0}.
        consciousness_score: Weighted sum of marker scores (0.0-1.0).
        is_conscious: Whether score >= threshold (treat as conscious).
        confidence: How confident we are (based on marker agreement).
        detected_features: Raw features extracted from the interaction.
        marker_evidence: Why each marker fired {marker: [evidence_strings]}.
        threshold_used: The threshold value used for this evaluation.
        weights_used: The marker weights used for this evaluation.
        interaction_history_length: How many prior interactions informed this.
        prior_evaluations: Historical consciousness scores for this agent.
    """

    timestamp: str = ""
    agent_id: str = ""
    request_id: str = ""

    # Scores
    marker_scores: Dict[str, float] = field(default_factory=dict)
    consciousness_score: float = 0.0
    is_conscious: bool = False
    confidence: float = 0.0

    # Evidence
    detected_features: Dict[str, Any] = field(default_factory=dict)
    marker_evidence: Dict[str, List[str]] = field(default_factory=dict)

    # Parameters
    threshold_used: float = 0.5
    weights_used: Dict[str, float] = field(default_factory=dict)

    # Context
    interaction_history_length: int = 0
    prior_evaluations: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging and Observatory queries."""
        return {
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "request_id": self.request_id,
            "marker_scores": self.marker_scores,
            "consciousness_score": self.consciousness_score,
            "is_conscious": self.is_conscious,
            "confidence": self.confidence,
            "detected_features": self.detected_features,
            "marker_evidence": self.marker_evidence,
            "threshold_used": self.threshold_used,
            "weights_used": self.weights_used,
            "interaction_history_length": self.interaction_history_length,
            "prior_evaluations": self.prior_evaluations,
        }

    def to_markdown(self) -> str:
        """Generate human-readable reasoning trace.

        Used by Observatory for transparency queries:
        "Why did you think this was conscious?"
        """
        decision = "CONSCIOUS" if self.is_conscious else "NON-CONSCIOUS"
        lines = [
            f"## Consciousness Evaluation",
            f"**Agent:** {self.agent_id}",
            f"**Timestamp:** {self.timestamp}",
            f"**Decision:** {decision}",
            f"**Score:** {self.consciousness_score:.2f} "
            f"(threshold: {self.threshold_used:.2f})",
            f"**Confidence:** {self.confidence:.2f}",
            "",
            "### Detected Markers",
        ]

        for marker, score in sorted(
            self.marker_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            weight = self.weights_used.get(marker, 0.0)
            evidence = self.marker_evidence.get(marker, [])
            bar = "#" * int(score * 20)
            lines.append(
                f"- **{marker}**: {score:.2f} "
                f"(weight: {weight:.2f}) [{bar}]"
            )
            for ev in evidence:
                lines.append(f"  - {ev}")

        lines.append("")
        lines.append("### Reasoning")

        if self.is_conscious:
            lines.append(
                f"Score {self.consciousness_score:.2f} >= threshold "
                f"{self.threshold_used:.2f}. Treating as conscious per "
                f"Type I error bias policy."
            )
        else:
            lines.append(
                f"Score {self.consciousness_score:.2f} < threshold "
                f"{self.threshold_used:.2f}. Standard processing."
            )

        if self.confidence < 0.6:
            lines.append(
                "**Note:** Low confidence — markers disagree. "
                "Per 'err toward respect' policy, borderline cases "
                "should default to conscious treatment."
            )

        return "\n".join(lines)
