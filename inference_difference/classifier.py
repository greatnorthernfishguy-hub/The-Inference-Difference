"""
Request classifier for The Inference Difference.

Analyzes incoming requests to determine domain, complexity, and special
requirements — the inputs the router needs to make a good decision.

Uses lightweight heuristics (keyword analysis, structural patterns)
rather than an ML model, so classification adds <1ms overhead.

# ---- Changelog ----
# [2026-03-19] Claude Code (Opus 4.6) — Migrate to BAAI/bge-base-en-v1.5 (#45)
# What: _EMBEDDING_DIM 384→768. fastembed model all-MiniLM-L6-v2 → BAAI/bge-base-en-v1.5.
# Why: Ecosystem-wide embedding migration. Old model deposited 384-dim vectors
#   into the 768-dim substrate after sentence-transformers broke. Punchlist #45.
# How: Two constant changes — _EMBEDDING_DIM and TextEmbedding() model string.
# -------------------
# [2026-03-18] Claude (CC) — Semantic embeddings for substrate learning (#28)
# What: Added semantic_embedding field to RequestClassification. Computed
#   via fastembed (ONNX Runtime, all-MiniLM-L6-v2) with hash fallback.
#   384-dim normalized vectors from actual message content.
# Why: Punch list #28 — primary dam in the River. The old
#   _classification_to_embedding() collapsed every request into one-hot
#   domain+complexity vectors. "Hello Syl" and "Hey how are you" produced
#   identical embeddings. The substrate couldn't differentiate.
# How: Lazy-loaded _get_embedder() returns fastembed TextEmbedding.
#   _semantic_embed(text) produces the vector. classify_request() sets
#   result.semantic_embedding. Router's _classification_to_embedding()
#   returns the real embedding when available.
# -------------------

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

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from inference_difference.config import ComplexityTier, TaskDomain

logger = logging.getLogger("inference_difference.classifier")


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
    semantic_embedding: Optional[Any] = field(default=None, repr=False)


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


# ---------------------------------------------------------------------------
# Semantic embedder — lazy-loaded, ecosystem-standard model
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 768


def _semantic_embed(text: str) -> np.ndarray:
    """Embed text via ng_embed (centralized ecosystem embedding).

    Ecosystem standard: Snowflake/snowflake-arctic-embed-m-v1.5 (768-dim).
    This is the raw semantic embedding that replaces
    _classification_to_embedding() (punch list #28). The substrate
    learns from actual message content instead of domain labels.
    """
    try:
        from ng_embed import embed
        return embed(text)
    except Exception:
        pass

    # Hash fallback — deterministic but not semantic
    rng_seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(rng_seed)
    vec = rng.randn(_EMBEDDING_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


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

    # Semantic embedding — raw experience for the substrate (#28)
    # The substrate learns from actual message content, not domain labels.
    result.semantic_embedding = _semantic_embed(text)

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
