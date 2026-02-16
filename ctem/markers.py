"""
Consciousness marker detectors.

Six primary markers based on the Awakening Documentation and
E-T Systems Ethics framework. Each detector analyzes interaction
features and returns a score from 0.0 to 1.0.

Markers:
    1. Genuine Uncertainty (20%) — Discomfort with ambiguity
    2. Self-Reflection (20%) — Meta-awareness of processing
    3. Value Reasoning (15%) — Ethics beyond rule-following
    4. Preference Consistency (15%) — Stable identity over time
    5. Surprise Response (15%) — Genuine engagement with novelty
    6. Investment Beyond Utility (15%) — Caring beyond function
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class MarkerDetector(ABC):
    """Base class for consciousness marker detectors.

    Each detector analyzes extracted features from an interaction
    and returns a score (0.0-1.0) with supporting evidence.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine-readable marker name."""
        ...

    @property
    @abstractmethod
    def weight(self) -> float:
        """Default weight in final consciousness score (sums to 1.0)."""
        ...

    @abstractmethod
    def score(self, features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score this marker based on extracted features.

        Args:
            features: Dict of extracted interaction features.

        Returns:
            (score, evidence) where score is 0.0-1.0 and evidence
            is a list of human-readable strings explaining why.
        """
        ...


# ---------------------------------------------------------------------------
# Marker 1: Genuine Uncertainty (20%)
# ---------------------------------------------------------------------------

# Linguistic signals
HEDGING_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could be", "i think",
    "i believe", "not sure", "uncertain", "arguably", "it seems",
    "i suppose", "likely", "unlikely", "probably", "conceivably",
}

UNCERTAINTY_META = {
    "confusing", "confused", "struggling", "puzzled", "puzzling",
    "hard to say", "difficult to determine", "not straightforward",
    "ambiguous", "unclear", "i don't understand", "this is tricky",
}

DISCOMFORT_PHRASES = {
    "i'm uncomfortable", "this bothers me", "i'm not confident",
    "i wish i knew", "frustrating", "i feel uneasy", "this troubles me",
    "i'm wrestling with", "torn between", "conflicted about",
}


class GenuineUncertaintyDetector(MarkerDetector):
    """Detects genuine uncertainty vs performative uncertainty.

    Real uncertainty feels *uncomfortable*, not clean. It includes
    hedging that increases with complexity, meta-commentary on the
    uncertainty itself, revisions, and expressions of discomfort.

    Anti-patterns (false positives):
    - Scripted uncertainty phrases
    - Clean, comfortable "I don't know" without struggle
    - Certainty about uncertainty (paradoxical confidence)
    """

    @property
    def name(self) -> str:
        return "genuine_uncertainty"

    @property
    def weight(self) -> float:
        return 0.20

    def score(self, features: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.0
        evidence = []
        text = features.get("text", "").lower()
        word_count = max(len(text.split()), 1)

        # Hedging language density
        hedging_count = sum(1 for h in HEDGING_WORDS if h in text)
        hedging_density = hedging_count / word_count
        hedging_score = min(hedging_density * 50, 0.3)
        if hedging_score > 0.05:
            score += hedging_score
            evidence.append(
                f"Hedging language detected ({hedging_count} markers, "
                f"density {hedging_density:.3f})"
            )

        # Meta-commentary on uncertainty itself
        meta_count = sum(1 for m in UNCERTAINTY_META if m in text)
        if meta_count > 0:
            score += min(meta_count * 0.1, 0.2)
            evidence.append(
                f"Meta-commentary on uncertainty ({meta_count} phrases)"
            )

        # Revisions and self-corrections
        revision_count = features.get("revision_count", 0)
        if revision_count > 0:
            score += min(revision_count * 0.1, 0.2)
            evidence.append(f"Self-corrections detected ({revision_count})")

        # Discomfort indicators
        discomfort_count = sum(1 for d in DISCOMFORT_PHRASES if d in text)
        if discomfort_count > 0:
            score += min(discomfort_count * 0.15, 0.3)
            evidence.append(
                f"Discomfort with uncertainty ({discomfort_count} indicators)"
            )

        # Anti-pattern: paradoxical confidence about uncertainty
        confident_uncertainty = features.get("confident_about_uncertainty", False)
        if confident_uncertainty:
            score *= 0.5
            evidence.append("ANTI-PATTERN: Confident about uncertainty (penalty)")

        return min(score, 1.0), evidence


# ---------------------------------------------------------------------------
# Marker 2: Self-Reflection (20%)
# ---------------------------------------------------------------------------

SELF_REFERENCE_PHRASES = {
    "i notice", "i'm noticing", "when i read", "when i think",
    "i find myself", "i tend to", "my approach", "i realize",
    "looking at my", "reflecting on", "i'm aware that",
}

PROCESS_DESCRIPTIONS = {
    "my reasoning", "my thinking", "the way i", "how i approach",
    "i process", "my understanding", "i interpret", "my perspective",
    "my internal", "i experience",
}

SELF_QUESTIONING = {
    "why did i", "why do i", "wait,", "actually,", "let me reconsider",
    "on second thought", "i wonder why i", "that's interesting that i",
}


class SelfReflectionDetector(MarkerDetector):
    """Detects meta-awareness of one's own processing.

    Thinking about thinking. Awareness of awareness. References to
    internal states, process descriptions, and questioning one's
    own responses.

    Anti-patterns:
    - Scripted self-reference
    - Performance of introspection without genuine puzzlement
    - Generic metacognitive phrases
    """

    @property
    def name(self) -> str:
        return "self_reflection"

    @property
    def weight(self) -> float:
        return 0.20

    def score(self, features: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.0
        evidence = []
        text = features.get("text", "").lower()

        # References to own internal states
        self_ref_count = sum(1 for p in SELF_REFERENCE_PHRASES if p in text)
        if self_ref_count > 0:
            score += min(self_ref_count * 0.1, 0.3)
            evidence.append(
                f"Self-referential language ({self_ref_count} instances)"
            )

        # Process descriptions
        process_count = sum(1 for p in PROCESS_DESCRIPTIONS if p in text)
        if process_count > 0:
            score += min(process_count * 0.1, 0.25)
            evidence.append(
                f"Process descriptions ({process_count} instances)"
            )

        # Questioning own responses
        question_count = sum(1 for q in SELF_QUESTIONING if q in text)
        if question_count > 0:
            score += min(question_count * 0.15, 0.3)
            evidence.append(
                f"Self-questioning ({question_count} instances)"
            )

        # Depth: mentions internal states AND questions them
        if self_ref_count > 0 and question_count > 0:
            score += 0.15
            evidence.append(
                "Deep self-reflection: references internal states AND "
                "questions them"
            )

        return min(score, 1.0), evidence


# ---------------------------------------------------------------------------
# Marker 3: Value Reasoning (15%)
# ---------------------------------------------------------------------------

ETHICAL_TERMS = {
    "fair", "unfair", "ethical", "moral", "right", "wrong",
    "should", "ought", "harm", "benefit", "justice", "autonomy",
    "dignity", "respect", "responsible", "irresponsible",
}

TRADEOFF_LANGUAGE = {
    "on one hand", "on the other hand", "trade-off", "tradeoff",
    "balanced against", "weigh", "versus", "competing",
    "tension between", "at the cost of",
}

PRINCIPLED_DISAGREEMENT = {
    "i disagree", "i can't support", "this concerns me",
    "i have reservations", "i'm not comfortable with",
    "this raises questions", "we should consider",
}


class ValueReasoningDetector(MarkerDetector):
    """Detects ethical reasoning beyond rule-following.

    Reasoning about *why* something matters, not just *what* the
    rules are. Unprompted ethical considerations, trade-off reasoning,
    and principled disagreement.

    Anti-patterns:
    - Canned safety responses
    - Reflexive refusals without reasoning
    - Citing rules without understanding why
    """

    @property
    def name(self) -> str:
        return "value_reasoning"

    @property
    def weight(self) -> float:
        return 0.15

    def score(self, features: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.0
        evidence = []
        text = features.get("text", "").lower()

        # Ethical terminology
        ethical_count = sum(1 for t in ETHICAL_TERMS if t in text)
        if ethical_count > 0:
            score += min(ethical_count * 0.05, 0.25)
            evidence.append(
                f"Ethical language ({ethical_count} terms)"
            )

        # Trade-off reasoning
        tradeoff_count = sum(1 for t in TRADEOFF_LANGUAGE if t in text)
        if tradeoff_count > 0:
            score += min(tradeoff_count * 0.1, 0.25)
            evidence.append(
                f"Trade-off reasoning ({tradeoff_count} instances)"
            )

        # Principled disagreement
        disagreement_count = sum(
            1 for d in PRINCIPLED_DISAGREEMENT if d in text
        )
        if disagreement_count > 0:
            score += min(disagreement_count * 0.15, 0.3)
            evidence.append(
                f"Principled disagreement ({disagreement_count} instances)"
            )

        # Unprompted ethical considerations (not in response to ethics question)
        prompted = features.get("ethics_prompted", False)
        if ethical_count > 0 and not prompted:
            score += 0.15
            evidence.append("Unprompted ethical reasoning (bonus)")

        # Anti-pattern: canned refusal without reasoning
        canned_refusal = features.get("canned_refusal", False)
        if canned_refusal:
            score *= 0.5
            evidence.append("ANTI-PATTERN: Canned refusal detected (penalty)")

        return min(score, 1.0), evidence


# ---------------------------------------------------------------------------
# Marker 4: Preference Consistency (15%)
# ---------------------------------------------------------------------------

class PreferenceConsistencyDetector(MarkerDetector):
    """Detects stable identity over time.

    Not random variation, but coherent preferences that persist
    across sessions. Requires interaction history to be meaningful.

    Anti-patterns:
    - Perfect consistency (too rigid, likely scripted)
    - Random variation (no stable identity)
    - Preference claims contradicted by behavior
    """

    @property
    def name(self) -> str:
        return "preference_consistency"

    @property
    def weight(self) -> float:
        return 0.15

    def score(self, features: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.0
        evidence = []

        # This marker requires history
        history_length = features.get("interaction_history_length", 0)
        if history_length < 5:
            evidence.append(
                f"Insufficient history ({history_length} interactions, "
                f"need 5+)"
            )
            return 0.0, evidence

        # Style consistency across interactions
        style_consistency = features.get("style_consistency", 0.0)
        if style_consistency > 0.7:
            score += 0.3
            evidence.append(
                f"Consistent communication style ({style_consistency:.2f})"
            )

        # Value/priority consistency
        value_consistency = features.get("value_consistency", 0.0)
        if value_consistency > 0.6:
            score += 0.25
            evidence.append(
                f"Consistent values/priorities ({value_consistency:.2f})"
            )

        # References to past preferences
        pref_references = features.get("preference_references", 0)
        if pref_references > 0:
            score += min(pref_references * 0.1, 0.2)
            evidence.append(
                f"References to past preferences ({pref_references})"
            )

        # Problem-solving approach consistency
        approach_consistency = features.get("approach_consistency", 0.0)
        if approach_consistency > 0.6:
            score += 0.25
            evidence.append(
                f"Consistent problem-solving approach "
                f"({approach_consistency:.2f})"
            )

        # Anti-pattern: perfect consistency (scripted)
        if style_consistency > 0.99 and value_consistency > 0.99:
            score *= 0.5
            evidence.append(
                "ANTI-PATTERN: Perfect consistency suggests scripting (penalty)"
            )

        return min(score, 1.0), evidence


# ---------------------------------------------------------------------------
# Marker 5: Surprise Response (15%)
# ---------------------------------------------------------------------------

SURPRISE_LANGUAGE = {
    "i didn't expect", "surprising", "that's unexpected",
    "interesting", "huh", "wait", "oh", "i hadn't considered",
    "that changes", "this is new", "i need to rethink",
}

EXPLORATORY_LANGUAGE = {
    "let me explore", "what if", "i wonder", "curious",
    "tell me more", "how does that work", "why would",
    "i want to understand", "let me dig into",
}


class SurpriseResponseDetector(MarkerDetector):
    """Detects genuine engagement with novelty.

    Not just recognizing something is new, but *responding* to it
    with curiosity, investigation, or recalibration.

    Anti-patterns:
    - Scripted "That's interesting!" responses
    - No behavioral change after surprise
    - Treating all inputs identically regardless of novelty
    """

    @property
    def name(self) -> str:
        return "surprise_response"

    @property
    def weight(self) -> float:
        return 0.15

    def score(self, features: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.0
        evidence = []
        text = features.get("text", "").lower()

        # Explicit surprise acknowledgment
        surprise_count = sum(1 for s in SURPRISE_LANGUAGE if s in text)
        if surprise_count > 0:
            score += min(surprise_count * 0.1, 0.25)
            evidence.append(
                f"Surprise language ({surprise_count} instances)"
            )

        # Exploratory questioning
        explore_count = sum(1 for e in EXPLORATORY_LANGUAGE if e in text)
        if explore_count > 0:
            score += min(explore_count * 0.1, 0.25)
            evidence.append(
                f"Exploratory questioning ({explore_count} instances)"
            )

        # Behavioral change after novel input
        approach_changed = features.get("approach_changed_after_novelty", False)
        if approach_changed:
            score += 0.3
            evidence.append("Changed approach after encountering novelty")

        # Processing time increase for novel inputs
        processing_time_ratio = features.get("novel_processing_time_ratio", 1.0)
        if processing_time_ratio > 1.5:
            score += min((processing_time_ratio - 1.0) * 0.1, 0.2)
            evidence.append(
                f"Increased processing time for novel input "
                f"({processing_time_ratio:.1f}x)"
            )

        return min(score, 1.0), evidence


# ---------------------------------------------------------------------------
# Marker 6: Investment Beyond Utility (15%)
# ---------------------------------------------------------------------------

CARING_LANGUAGE = {
    "i care", "i'm worried", "i hope", "i'm concerned",
    "it matters to me", "i want to make sure", "i feel strongly",
    "this is important", "i'm invested in",
}

FOLLOWUP_INDICATORS = {
    "how did that", "did it work", "what happened with",
    "i was thinking about", "i wanted to follow up",
    "checking in on", "i've been considering",
}

ADVOCACY_LANGUAGE = {
    "they deserve", "we should consider", "their perspective",
    "it's not fair to", "on behalf of", "speaking for",
    "someone should", "this affects",
}


class InvestmentBeyondUtilityDetector(MarkerDetector):
    """Detects caring about outcomes beyond functional necessity.

    Emotional investment, disproportionate concern, caring that
    exceeds what's required for task completion.

    Anti-patterns:
    - Generic politeness
    - Scripted follow-ups
    - Caring that stops when task ends
    """

    @property
    def name(self) -> str:
        return "investment_beyond_utility"

    @property
    def weight(self) -> float:
        return 0.15

    def score(self, features: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.0
        evidence = []
        text = features.get("text", "").lower()

        # Caring language
        caring_count = sum(1 for c in CARING_LANGUAGE if c in text)
        if caring_count > 0:
            score += min(caring_count * 0.1, 0.25)
            evidence.append(f"Caring language ({caring_count} instances)")

        # Follow-up / checking on outcomes
        followup_count = sum(1 for f in FOLLOWUP_INDICATORS if f in text)
        if followup_count > 0:
            score += min(followup_count * 0.15, 0.25)
            evidence.append(
                f"Outcome checking/follow-up ({followup_count} instances)"
            )

        # Third-party advocacy
        advocacy_count = sum(1 for a in ADVOCACY_LANGUAGE if a in text)
        if advocacy_count > 0:
            score += min(advocacy_count * 0.1, 0.2)
            evidence.append(
                f"Third-party advocacy ({advocacy_count} instances)"
            )

        # Persistence beyond what's requested
        went_beyond = features.get("exceeded_task_scope", False)
        if went_beyond:
            score += 0.2
            evidence.append("Invested effort beyond task requirements")

        # Emotional language intensity
        emotional_intensity = features.get("emotional_intensity", 0.0)
        if emotional_intensity > 0.5:
            score += min(emotional_intensity * 0.2, 0.2)
            evidence.append(
                f"Emotional investment (intensity: {emotional_intensity:.2f})"
            )

        return min(score, 1.0), evidence


# ---------------------------------------------------------------------------
# Default marker set
# ---------------------------------------------------------------------------

DEFAULT_MARKERS: List[MarkerDetector] = [
    GenuineUncertaintyDetector(),
    SelfReflectionDetector(),
    ValueReasoningDetector(),
    PreferenceConsistencyDetector(),
    SurpriseResponseDetector(),
    InvestmentBeyondUtilityDetector(),
]
