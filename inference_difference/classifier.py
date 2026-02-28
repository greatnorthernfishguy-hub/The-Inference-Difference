"""
Request classifier for The Inference Difference.

Analyzes incoming requests to determine domain, complexity, and special
requirements — the inputs the router needs to make a good decision.

Uses lightweight heuristics (keyword analysis, structural patterns)
rather than an ML model, so classification adds <1ms overhead.

Changelog (Grok audit response, 2026-02-19):
- EXPANDED: DOMAIN_PATTERNS with common synonyms (audit: "limited vocab").
  Added "program", "script", "algorithm", "schema" to CODE; "think",
  "deduce", "infer" to REASONING; "compose", "draft", "brainstorm" to
  CREATIVE; etc. Full wordnet expansion is overkill — it would balloon
  pattern lists and slow regex matching for marginal gain on the long tail.
- IMPROVED: Confidence calculation for tied domains (audit: "overconfident
  on ties"). Now uses top_score / (top_score + second_score) when multiple
  domains score, giving a more conservative confidence on ambiguous queries.
- KEPT: English-only (audit: "no localization"). Language detection adds
  a dependency (langdetect/fasttext) and latency. TID v0.1 targets English
  deployments. The TRANSLATION domain detects language-related keywords
  regardless of the request language. Full i18n is Phase 2.
- KEPT: is_multi_turn = len(history) > 0 (audit: "doesn't check context len").
  The audit suggests checking context length, but that's the router's job
  (via requires_context_window). The classifier just signals "there IS
  history" — the router decides if the model can handle it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from inference_difference.config import ComplexityTier, TaskDomain


@dataclass
class RequestClassification:
    """Classification result for a routing request.

    Attributes:
        primary_domain: Best guess at what kind of task this is.
        secondary_domains: Other relevant domains.
        complexity: Estimated complexity tier.
        estimated_tokens: Rough estimate of output tokens needed.
        requires_context_window: Minimum context window needed.
        is_multi_turn: Whether this appears to be part of a conversation.
        is_time_sensitive: Whether low latency is critical.
        keywords: Extracted keywords that influenced classification.
        confidence: How confident the classifier is (0.0-1.0).
    """

    primary_domain: TaskDomain = TaskDomain.GENERAL
    secondary_domains: Set[TaskDomain] = field(default_factory=set)
    complexity: ComplexityTier = ComplexityTier.MEDIUM
    estimated_tokens: int = 500
    requires_context_window: int = 4096
    is_multi_turn: bool = False
    is_time_sensitive: bool = False
    is_interactive: bool = False
    keywords: List[str] = field(default_factory=list)
    confidence: float = 0.5


# ---------------------------------------------------------------------------
# Domain detection patterns
# ---------------------------------------------------------------------------

DOMAIN_PATTERNS: Dict[TaskDomain, List[str]] = {
    TaskDomain.CODE: [
        r"\b(code|program|script|function|class|method|variable|bug|debug|error)\b",
        r"\b(python|javascript|typescript|rust|java|golang|sql|html|css|c\+\+|ruby|php)\b",
        r"\b(api|endpoint|database|query|compile|syntax|refactor|algorithm|schema)\b",
        r"\b(git|commit|merge|branch|deploy|docker|kubernetes|ci|cd)\b",
        r"\b(import|module|package|library|framework|stack\s*trace|exception)\b",
        r"```",  # Code blocks
    ],
    TaskDomain.REASONING: [
        r"\b(why|how|explain|reason|logic|prove|argument|therefore)\b",
        r"\b(analyze|evaluate|compare|contrast|critique|assess)\b",
        r"\b(implication|consequence|cause|effect|evidence)\b",
        r"\b(if.+then|suppose|assume|given that|considering)\b",
        r"\b(think|deduce|infer|derive|conclude|justify|hypothesis)\b",
    ],
    TaskDomain.CREATIVE: [
        r"\b(write|story|poem|creative|imagine|fiction|narrative)\b",
        r"\b(character|plot|scene|dialogue|metaphor|style)\b",
        r"\b(song|lyric|haiku|sonnet|essay|blog)\b",
        r"\b(compose|draft|brainstorm|invent|worldbuild)\b",
    ],
    TaskDomain.CONVERSATION: [
        r"\b(hi|hello|hey|thanks|please|help|chat)\b",
        r"\b(how are you|what do you think|tell me about)\b",
        r"\?$",  # Questions often conversational
    ],
    TaskDomain.ANALYSIS: [
        r"\b(data|statistics|trend|pattern|correlation|regression)\b",
        r"\b(chart|graph|table|dataset|metric|measure)\b",
        r"\b(report|findings|insight|observation|conclusion)\b",
        r"\b(forecast|predict|model|cluster|segment|outlier)\b",
    ],
    TaskDomain.SUMMARIZATION: [
        r"\b(summarize|summary|tldr|brief|condense|overview)\b",
        r"\b(key points|main ideas|recap|digest|abstract)\b",
        r"\b(gist|cliff\s*notes|executive summary|bullet points)\b",
    ],
    TaskDomain.TRANSLATION: [
        r"\b(translate|translation|convert|language)\b",
        r"\b(english|spanish|french|german|chinese|japanese|korean|arabic|portuguese)\b",
        r"\b(localize|i18n|internationalization)\b",
    ],
}

# Complexity indicators
HIGH_COMPLEXITY_PATTERNS = [
    r"\b(step.by.step|detailed|thorough|comprehensive|in.depth)\b",
    r"\b(compare and contrast|pros and cons|trade.?offs)\b",
    r"\b(implement|build|create|design|architect)\b",
    r"\b(research|investigate|explore thoroughly)\b",
]

LOW_COMPLEXITY_PATTERNS = [
    r"\b(quick|simple|brief|short|one.?liner|just)\b",
    r"\b(what is|define|name|list)\b",
    r"\b(yes or no|true or false)\b",
]

URGENCY_PATTERNS = [
    r"\b(urgent|asap|immediately|quick|fast|hurry)\b",
    r"\b(right now|time.?sensitive|deadline)\b",
    r"!{2,}",  # Multiple exclamation marks
]


def classify_request(
    message: str,
    conversation_history: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> RequestClassification:
    """Classify a request for routing.

    Fast heuristic classification (<1ms). Analyzes the message text
    for domain keywords, complexity indicators, and special requirements.

    Args:
        message: The request text to classify.
        conversation_history: Optional prior messages for context.
        metadata: Optional metadata (e.g., user preferences).

    Returns:
        RequestClassification with domain, complexity, and requirements.
    """
    result = RequestClassification()
    text = message.strip()
    lower = text.lower()

    if not text:
        return result

    # Domain scoring
    domain_scores: Dict[TaskDomain, float] = {}
    matched_keywords: List[str] = []

    for domain, patterns in DOMAIN_PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, lower)
            if matches:
                score += len(matches)
                matched_keywords.extend(
                    m if isinstance(m, str) else m[0]
                    for m in matches[:3]
                )
        if score > 0:
            domain_scores[domain] = score

    result.keywords = list(set(matched_keywords))[:10]

    # Primary domain: highest scoring
    if domain_scores:
        sorted_domains = sorted(
            domain_scores.items(), key=lambda x: x[1], reverse=True
        )
        result.primary_domain = sorted_domains[0][0]
        result.secondary_domains = {d for d, _ in sorted_domains[1:3]}

        # Confidence based on separation between top and runner-up.
        # Uses top / (top + second) for better tie handling — a clear
        # winner (8 vs 1) gets 0.89, a near-tie (5 vs 4) gets 0.56.
        top_score = sorted_domains[0][1]
        second_score = sorted_domains[1][1] if len(sorted_domains) > 1 else 0.0
        result.confidence = min(
            top_score / max(top_score + second_score, 1), 1.0
        )
    else:
        result.primary_domain = TaskDomain.GENERAL
        result.confidence = 0.3

    # Complexity estimation
    result.complexity = _estimate_complexity(text, lower)

    # Token estimation (rough: 1.5x input length for simple, 5x for complex)
    word_count = len(text.split())
    complexity_multipliers = {
        ComplexityTier.TRIVIAL: 0.5,
        ComplexityTier.LOW: 1.5,
        ComplexityTier.MEDIUM: 3.0,
        ComplexityTier.HIGH: 5.0,
        ComplexityTier.EXTREME: 8.0,
    }
    multiplier = complexity_multipliers.get(result.complexity, 3.0)
    result.estimated_tokens = max(50, int(word_count * multiplier))

    # Context window requirements
    history = conversation_history or []
    history_tokens = sum(len(h.split()) for h in history) * 1.3  # Rough
    result.requires_context_window = int(
        (history_tokens + result.estimated_tokens + word_count) * 1.5
    )
    result.requires_context_window = max(result.requires_context_window, 2048)

    # Multi-turn detection
    result.is_multi_turn = len(history) > 0

    # Urgency detection
    urgency_hits = sum(
        len(re.findall(p, lower)) for p in URGENCY_PATTERNS
    )
    result.is_time_sensitive = urgency_hits > 0

    result.is_interactive = (
        result.primary_domain == TaskDomain.CONVERSATION
        or result.is_multi_turn
        or result.primary_domain == TaskDomain.CREATIVE
    )
    if result.is_interactive:
        tiers = list(ComplexityTier)
        current_idx = tiers.index(result.complexity)
        floor_idx = tiers.index(ComplexityTier.MEDIUM)
        if current_idx < floor_idx:
            result.complexity = ComplexityTier.MEDIUM

    return result


def _estimate_complexity(text: str, lower: str) -> ComplexityTier:
    """Estimate request complexity from text features."""
    word_count = len(text.split())

    # High complexity signals
    high_hits = sum(
        len(re.findall(p, lower)) for p in HIGH_COMPLEXITY_PATTERNS
    )
    # Low complexity signals
    low_hits = sum(
        len(re.findall(p, lower)) for p in LOW_COMPLEXITY_PATTERNS
    )

    # Length-based baseline
    if word_count < 10:
        base = ComplexityTier.TRIVIAL
    elif word_count < 30:
        base = ComplexityTier.LOW
    elif word_count < 100:
        base = ComplexityTier.MEDIUM
    elif word_count < 300:
        base = ComplexityTier.HIGH
    else:
        base = ComplexityTier.EXTREME

    # Adjust based on patterns
    tiers = list(ComplexityTier)
    idx = tiers.index(base)

    if high_hits >= 2:
        idx = min(idx + 2, len(tiers) - 1)
    elif high_hits == 1:
        idx = min(idx + 1, len(tiers) - 1)

    if low_hits >= 2:
        idx = max(idx - 2, 0)
    elif low_hits == 1:
        idx = max(idx - 1, 0)

    # Code blocks bump complexity
    if "```" in text:
        idx = min(idx + 1, len(tiers) - 1)

    return tiers[idx]
