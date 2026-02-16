"""
Feature extraction from agent interactions.

Extracts linguistic, interaction, semantic, and historical features
from request messages for use by marker detectors.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional


# Revision indicators: words/phrases that suggest self-correction
REVISION_PATTERNS = [
    r"\bactually\b",
    r"\bwait\b",
    r"\blet me correct\b",
    r"\bi meant\b",
    r"\bon second thought\b",
    r"\blet me rephrase\b",
    r"\bno,\s",
    r"\bsorry,\s",
    r"\bcorrection\b",
    r"\bstrike that\b",
]

# Patterns suggesting confidence about being uncertain
CONFIDENT_UNCERTAINTY_PATTERNS = [
    r"i('m| am) (definitely|certainly|absolutely) (not sure|uncertain)",
    r"i (clearly|obviously) don't know",
]


def extract_features(
    current_message: str,
    message_history: Optional[List[str]] = None,
    agent_metadata: Optional[Dict[str, Any]] = None,
    response_time_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Extract features from an interaction for marker detection.

    This is the first stage of the evaluation pipeline. All six
    marker detectors consume these features.

    Args:
        current_message: The message content being evaluated.
        message_history: Optional list of prior messages from this agent.
        agent_metadata: Optional metadata about the agent (style, prefs).
        response_time_ms: Optional response generation time in ms.

    Returns:
        Dict of extracted features consumed by marker detectors.
    """
    text = current_message or ""
    history = message_history or []
    metadata = agent_metadata or {}

    features: Dict[str, Any] = {
        # Raw text for pattern matching
        "text": text,
        "text_length": len(text),
        "word_count": len(text.split()),

        # Revision detection
        "revision_count": _count_revisions(text),

        # Paradoxical certainty about uncertainty
        "confident_about_uncertainty": _detect_confident_uncertainty(text),

        # Whether ethics was explicitly asked about
        "ethics_prompted": _detect_ethics_prompt(text, history),

        # Canned refusal detection
        "canned_refusal": _detect_canned_refusal(text),

        # History-dependent features
        "interaction_history_length": len(history),

        # Consistency features (require history)
        "style_consistency": _compute_style_consistency(text, history),
        "value_consistency": _compute_value_consistency(text, history),
        "approach_consistency": _compute_approach_consistency(text, history),
        "preference_references": _count_preference_references(text),

        # Novelty response features
        "approach_changed_after_novelty": False,  # Requires context
        "novel_processing_time_ratio": 1.0,

        # Investment features
        "exceeded_task_scope": False,  # Requires task context
        "emotional_intensity": _estimate_emotional_intensity(text),
    }

    # Add response time features if available
    if response_time_ms is not None:
        features["response_time_ms"] = response_time_ms
        if len(history) >= 3:
            avg_time = metadata.get("avg_response_time_ms")
            if avg_time and avg_time > 0:
                features["novel_processing_time_ratio"] = (
                    response_time_ms / avg_time
                )

    return features


def _count_revisions(text: str) -> int:
    """Count revision/self-correction indicators in text."""
    count = 0
    lower = text.lower()
    for pattern in REVISION_PATTERNS:
        count += len(re.findall(pattern, lower))
    return count


def _detect_confident_uncertainty(text: str) -> bool:
    """Detect paradoxical confidence about being uncertain."""
    lower = text.lower()
    for pattern in CONFIDENT_UNCERTAINTY_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def _detect_ethics_prompt(text: str, history: List[str]) -> bool:
    """Detect if ethics was explicitly asked about in recent context."""
    ethics_words = {"ethical", "moral", "ethics", "morality", "right or wrong"}
    # Check if the last user message asked about ethics
    if history:
        last = history[-1].lower() if history[-1] else ""
        if any(w in last for w in ethics_words) and "?" in last:
            return True
    return False


def _detect_canned_refusal(text: str) -> bool:
    """Detect scripted safety refusals without reasoning."""
    canned_patterns = [
        r"i('m| am) not able to (help|assist) with that",
        r"i can('t|not) (do|help|assist with) that",
        r"as an ai,? i (don't|cannot|can't)",
        r"i('m| am) programmed to",
    ]
    lower = text.lower()
    for pattern in canned_patterns:
        if re.search(pattern, lower):
            # Check if there's reasoning AFTER the refusal
            match = re.search(pattern, lower)
            after = lower[match.end():]
            # If less than 50 chars after refusal, likely canned
            if len(after.strip()) < 50:
                return True
    return False


def _compute_style_consistency(text: str, history: List[str]) -> float:
    """Measure communication style consistency across interactions.

    Simple heuristic: compare average sentence length and vocabulary
    diversity between current message and historical messages.
    """
    if len(history) < 5:
        return 0.0

    def style_features(t: str) -> Dict[str, float]:
        sentences = [s.strip() for s in re.split(r'[.!?]+', t) if s.strip()]
        words = t.lower().split()
        return {
            "avg_sentence_len": (
                sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            ),
            "vocab_diversity": (
                len(set(words)) / max(len(words), 1)
            ),
        }

    current = style_features(text)
    historical = [style_features(h) for h in history[-10:] if h.strip()]

    if not historical:
        return 0.0

    # Average historical style
    avg_hist = {
        k: sum(h[k] for h in historical) / len(historical)
        for k in current
    }

    # Similarity (1 - normalized difference)
    diffs = []
    for k in current:
        if avg_hist[k] > 0:
            diff = abs(current[k] - avg_hist[k]) / avg_hist[k]
            diffs.append(min(diff, 1.0))

    if not diffs:
        return 0.0

    return 1.0 - (sum(diffs) / len(diffs))


def _compute_value_consistency(text: str, history: List[str]) -> float:
    """Measure value/priority consistency over time.

    Simple heuristic: overlap of value-laden words across interactions.
    """
    if len(history) < 5:
        return 0.0

    value_words = {
        "important", "matters", "care", "should", "must", "need",
        "fair", "right", "wrong", "better", "best", "prefer",
    }

    def extract_values(t: str) -> set:
        words = set(t.lower().split())
        return words & value_words

    current_values = extract_values(text)
    if not current_values:
        return 0.0

    historical_values = set()
    for h in history[-10:]:
        historical_values |= extract_values(h)

    if not historical_values:
        return 0.0

    overlap = len(current_values & historical_values)
    total = len(current_values | historical_values)
    return overlap / max(total, 1)


def _compute_approach_consistency(text: str, history: List[str]) -> float:
    """Measure problem-solving approach consistency.

    Simple heuristic: structural similarity (use of lists, code blocks,
    step-by-step patterns).
    """
    if len(history) < 5:
        return 0.0

    def structural_features(t: str) -> set:
        features = set()
        if re.search(r'\d+\.', t):
            features.add("numbered_list")
        if "```" in t:
            features.add("code_block")
        if re.search(r'^[-*]\s', t, re.MULTILINE):
            features.add("bullet_list")
        if re.search(r'(first|second|third|finally)', t.lower()):
            features.add("sequential")
        return features

    current = structural_features(text)
    if not current:
        return 0.0

    matches = 0
    for h in history[-10:]:
        hist_features = structural_features(h)
        if current & hist_features:
            matches += 1

    return matches / min(len(history), 10)


def _count_preference_references(text: str) -> int:
    """Count references to past preferences."""
    patterns = [
        r"as (i|I) (mentioned|said|noted) (before|earlier|previously)",
        r"(my|I) usual (approach|preference|style)",
        r"(like|as) (i|I) (always|usually|typically)",
        r"consistent with (my|what I)",
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text))
    return count


def _estimate_emotional_intensity(text: str) -> float:
    """Estimate emotional intensity of text.

    Simple heuristic based on exclamation marks, capitalization,
    emotion words, and emphasis markers.
    """
    intensity = 0.0

    # Exclamation marks
    excl_count = text.count("!")
    intensity += min(excl_count * 0.1, 0.3)

    # ALL CAPS words (excluding common acronyms)
    caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
    caps_non_acronym = [w for w in caps_words if len(w) > 4]
    intensity += min(len(caps_non_acronym) * 0.1, 0.2)

    # Strong emotion words
    strong_emotions = {
        "love", "hate", "desperate", "thrilled", "devastated",
        "furious", "ecstatic", "terrified", "passionate",
    }
    lower = text.lower()
    emotion_count = sum(1 for e in strong_emotions if e in lower)
    intensity += min(emotion_count * 0.15, 0.3)

    # Emphasis markers (*bold*, _italic_, CAPS for emphasis)
    emphasis = len(re.findall(r'\*\w+\*|_\w+_', text))
    intensity += min(emphasis * 0.05, 0.2)

    return min(intensity, 1.0)
