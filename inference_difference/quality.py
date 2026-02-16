"""
Response quality evaluator for The Inference Difference.

After a model produces a response, this module evaluates quality
to close the feedback loop. Quality scores feed back into NG-Lite
to improve future routing decisions.

Quality signals:
    - Completion (did the model actually answer the question?)
    - Coherence (does the response make sense?)
    - Length appropriateness (not too short, not too padded)
    - Error indicators (apologies, refusals, error messages)
    - Latency (did it meet the time budget?)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from inference_difference.classifier import RequestClassification
from inference_difference.config import ComplexityTier


@dataclass
class QualityEvaluation:
    """Quality assessment of a model response.

    Attributes:
        overall_score: Composite quality score (0.0-1.0).
        is_success: Whether overall_score >= quality threshold.
        completion_score: Did it answer the question?
        coherence_score: Does it make sense?
        length_score: Is the length appropriate?
        error_score: Absence of error indicators (1.0 = no errors).
        latency_score: Did it meet the time budget?
        breakdown: Per-signal scores for transparency.
        issues: List of detected quality issues.
    """

    overall_score: float = 0.0
    is_success: bool = False
    completion_score: float = 0.0
    coherence_score: float = 0.0
    length_score: float = 0.0
    error_score: float = 1.0
    latency_score: float = 1.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 3),
            "is_success": self.is_success,
            "breakdown": {k: round(v, 3) for k, v in self.breakdown.items()},
            "issues": self.issues,
        }


# Error/failure patterns in responses
ERROR_PATTERNS = [
    (r"i('m| am) (sorry|unable|not able)", "apology/refusal"),
    (r"i can('t|not) (help|assist|do|provide)", "capability refusal"),
    (r"error|exception|traceback|stack trace", "error indicator"),
    (r"(internal server|500|502|503|timeout)", "server error"),
    (r"rate limit|quota exceeded", "rate limit"),
    (r"invalid (api|key|token|request)", "auth/request error"),
]

# Padding/filler patterns
PADDING_PATTERNS = [
    r"(certainly|of course|absolutely|sure thing)[!.,]",
    r"(great question|good question|interesting question)",
    r"(let me|allow me to|i('d| would) be happy to)",
    r"(hope this helps|let me know if|feel free to)",
]

# Quality scoring weights
QUALITY_WEIGHTS = {
    "completion": 0.30,
    "coherence": 0.25,
    "length": 0.15,
    "error_free": 0.20,
    "latency": 0.10,
}


def evaluate_quality(
    response_text: str,
    classification: Optional[RequestClassification] = None,
    latency_ms: float = 0.0,
    latency_budget_ms: float = 5000.0,
    quality_threshold: float = 0.7,
) -> QualityEvaluation:
    """Evaluate the quality of a model response.

    Args:
        response_text: The model's response.
        classification: Request classification (for length expectations).
        latency_ms: Actual response latency.
        latency_budget_ms: Target latency budget.
        quality_threshold: Minimum score for success.

    Returns:
        QualityEvaluation with scores and detected issues.
    """
    evaluation = QualityEvaluation()
    issues: List[str] = []
    text = response_text or ""
    lower = text.lower()

    # --- Completion score ---
    completion = _score_completion(text, lower, classification)
    evaluation.completion_score = completion
    if completion < 0.5:
        issues.append("Response may be incomplete or non-responsive")

    # --- Coherence score ---
    coherence = _score_coherence(text)
    evaluation.coherence_score = coherence
    if coherence < 0.5:
        issues.append("Response may lack coherence")

    # --- Length appropriateness ---
    length = _score_length(text, classification)
    evaluation.length_score = length
    if length < 0.4:
        issues.append("Response length inappropriate for request complexity")

    # --- Error indicators ---
    error_free = _score_error_free(lower)
    evaluation.error_score = error_free
    if error_free < 0.7:
        issues.append("Response contains error/refusal indicators")

    # --- Latency ---
    latency = _score_latency(latency_ms, latency_budget_ms)
    evaluation.latency_score = latency
    if latency < 0.5:
        issues.append(
            f"Latency ({latency_ms:.0f}ms) exceeded budget "
            f"({latency_budget_ms:.0f}ms)"
        )

    # --- Overall score ---
    evaluation.breakdown = {
        "completion": completion,
        "coherence": coherence,
        "length": length,
        "error_free": error_free,
        "latency": latency,
    }

    overall = sum(
        evaluation.breakdown[k] * QUALITY_WEIGHTS[k]
        for k in QUALITY_WEIGHTS
    )
    evaluation.overall_score = overall
    evaluation.is_success = overall >= quality_threshold
    evaluation.issues = issues

    return evaluation


def _score_completion(
    text: str,
    lower: str,
    classification: Optional[RequestClassification],
) -> float:
    """Score whether the response actually answers the question."""
    if not text.strip():
        return 0.0

    score = 0.5  # Baseline

    # Non-empty response is a start
    word_count = len(text.split())
    if word_count >= 10:
        score += 0.2

    # Contains substantive content (not just filler)
    padding_count = sum(
        len(re.findall(p, lower)) for p in PADDING_PATTERNS
    )
    content_ratio = max(0, word_count - padding_count * 5) / max(word_count, 1)
    score += content_ratio * 0.2

    # Code requests should have code blocks
    if classification and classification.primary_domain.value == "code":
        if "```" in text or re.search(r'    \w', text):
            score += 0.1
        else:
            score -= 0.1

    return min(max(score, 0.0), 1.0)


def _score_coherence(text: str) -> float:
    """Score response coherence.

    Simple heuristics: sentence structure, not too repetitive,
    reasonable vocabulary diversity.
    """
    if not text.strip():
        return 0.0

    words = text.split()
    if len(words) < 3:
        return 0.3

    score = 0.6  # Baseline for non-empty text

    # Sentence detection (rough)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) >= 2:
        score += 0.1

    # Vocabulary diversity (unique words / total words)
    unique = len(set(w.lower() for w in words))
    diversity = unique / len(words)
    if diversity > 0.4:
        score += 0.15
    elif diversity < 0.2:
        score -= 0.15  # Very repetitive

    # Not just a single repeated phrase
    if len(sentences) > 2:
        unique_sentences = len(set(s.lower()[:50] for s in sentences))
        if unique_sentences / len(sentences) > 0.7:
            score += 0.1

    return min(max(score, 0.0), 1.0)


def _score_length(
    text: str,
    classification: Optional[RequestClassification],
) -> float:
    """Score length appropriateness for the request complexity."""
    word_count = len(text.split())

    if classification:
        expected = classification.estimated_tokens * 0.75  # tokens -> words
    else:
        expected = 200

    if expected <= 0:
        expected = 200

    ratio = word_count / expected

    if 0.3 <= ratio <= 3.0:
        return 1.0  # Within reasonable range
    elif 0.1 <= ratio <= 5.0:
        return 0.6  # Somewhat off
    elif ratio < 0.1:
        return 0.2  # Way too short
    else:
        return 0.4  # Way too long


def _score_error_free(lower: str) -> float:
    """Score absence of error indicators. 1.0 = no errors detected."""
    error_count = 0
    for pattern, _ in ERROR_PATTERNS:
        if re.search(pattern, lower):
            error_count += 1

    if error_count == 0:
        return 1.0
    elif error_count == 1:
        return 0.5
    else:
        return max(0.0, 1.0 - error_count * 0.3)


def _score_latency(latency_ms: float, budget_ms: float) -> float:
    """Score latency against budget."""
    if latency_ms <= 0 or budget_ms <= 0:
        return 1.0  # No latency data or no budget

    ratio = latency_ms / budget_ms

    if ratio <= 0.5:
        return 1.0
    elif ratio <= 1.0:
        return 0.8
    elif ratio <= 1.5:
        return 0.5
    elif ratio <= 2.0:
        return 0.3
    else:
        return 0.1
