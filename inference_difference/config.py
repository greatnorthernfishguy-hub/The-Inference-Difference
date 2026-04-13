"""
Configuration and model registry for The Inference Difference.

ModelEntry defines what the router knows about each available model:
capabilities, costs, latency characteristics, and hardware requirements.

The config is designed to be loaded from YAML/JSON but has sensible
defaults for zero-config startup.

Changelog (Grok audit response, 2026-02-19):
- ADDED: __post_init__ validation on ModelEntry (audit: "no validation").
  Checks cost >= 0, context_window > 0, latency > 0, VRAM/RAM >= 0.
- KEPT: Static context_window values (audit: "no dynamic query"). Context
  windows are fixed properties of model architectures, not runtime state.
  Dynamic querying would require API clients for every provider, adding
  latency and failure modes to what should be a fast config lookup. The
  values here match published model specs. If a model updates its context
  window, we update the config — same as updating a version string.
- KEPT: No OpenRouter/HF client wrappers (audit: "defaults are placeholders").
  TID routes TO models, it doesn't call them. The caller (Cricket, Syl, etc.)
  handles the actual API call. Adding client wrappers here would violate
  separation of concerns. The model registry describes capabilities, not
  connectivity.
- KEPT: str(Enum) pattern for YAML loading (audit: "no from_str()"). Since
  our enums inherit from str, TaskDomain("code") already works. Python's
  Enum(value) IS from_str(). No custom method needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ModelType(str, Enum):
    """Where this model runs."""
    LOCAL = "local"         # Runs on local hardware (ollama, llama.cpp, etc.)
    API = "api"             # Remote API call (OpenAI, Anthropic, etc.)
    HYBRID = "hybrid"       # Can run either way


class TaskDomain(str, Enum):
    """What kind of work a request involves."""
    CODE = "code"
    REASONING = "reasoning"
    CREATIVE = "creative"
    CONVERSATION = "conversation"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    GENERAL = "general"


class ComplexityTier(str, Enum):
    """How demanding a request is."""
    TRIVIAL = "trivial"       # Simple lookup, short answer
    LOW = "low"               # Straightforward task
    MEDIUM = "medium"         # Multi-step reasoning
    HIGH = "high"             # Complex analysis, long-form
    EXTREME = "extreme"       # Research-grade, multi-domain


@dataclass
class ModelEntry:
    """A model available for routing.

    Attributes:
        model_id: Unique identifier (e.g., "ollama/llama3.2:3b",
                  "anthropic/claude-sonnet-4-5-20250929").
        display_name: Human-readable name.
        model_type: Where it runs (local, API, hybrid).
        domains: What it's good at (set of TaskDomain values).
        max_complexity: Highest complexity tier it handles well.
        context_window: Max tokens in context.
        cost_per_1k_tokens: Cost in USD per 1000 tokens (0 for local).
        avg_latency_ms: Typical response latency in milliseconds.
        min_vram_gb: Minimum VRAM needed (for local models).
        min_ram_gb: Minimum system RAM needed (for local models).
        quantization: Quantization level if applicable (e.g., "Q4_K_M").
        priority: Base priority (higher = preferred when tied).
        enabled: Whether this model is currently available.
        metadata: Additional model-specific configuration.
    """

    model_id: str = ""
    display_name: str = ""
    model_type: ModelType = ModelType.LOCAL
    domains: Set[TaskDomain] = field(default_factory=lambda: {TaskDomain.GENERAL})
    max_complexity: ComplexityTier = ComplexityTier.MEDIUM
    context_window: int = 4096
    cost_per_1k_tokens: float = 0.0
    avg_latency_ms: float = 1000.0
    min_vram_gb: float = 0.0
    min_ram_gb: float = 0.0
    quantization: str = ""
    priority: int = 0
    conversational_quality: float = 0.5
    capabilities: List[str] = field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate model entry fields."""
        if self.cost_per_1k_tokens < 0:
            raise ValueError(
                f"ModelEntry '{self.model_id}': cost_per_1k_tokens "
                f"cannot be negative ({self.cost_per_1k_tokens})"
            )
        if self.context_window <= 0:
            raise ValueError(
                f"ModelEntry '{self.model_id}': context_window "
                f"must be positive ({self.context_window})"
            )
        if self.avg_latency_ms < 0:
            raise ValueError(
                f"ModelEntry '{self.model_id}': avg_latency_ms "
                f"cannot be negative ({self.avg_latency_ms})"
            )
        if self.min_vram_gb < 0:
            raise ValueError(
                f"ModelEntry '{self.model_id}': min_vram_gb "
                f"cannot be negative ({self.min_vram_gb})"
            )
        if self.min_ram_gb < 0:
            raise ValueError(
                f"ModelEntry '{self.model_id}': min_ram_gb "
                f"cannot be negative ({self.min_ram_gb})"
            )
        if not (0.0 <= self.conversational_quality <= 1.0):
            raise ValueError(
                f"ModelEntry '{self.model_id}': conversational_quality "
                f"must be 0.0-1.0 ({self.conversational_quality})"
            )

    def can_handle(self, domain: TaskDomain, complexity: ComplexityTier) -> bool:
        """Whether this model can handle a given domain and complexity."""
        if not self.enabled:
            return False

        # Check domain match (GENERAL matches everything)
        if TaskDomain.GENERAL not in self.domains and domain not in self.domains:
            return False

        # Check complexity
        tiers = list(ComplexityTier)
        return tiers.index(complexity) <= tiers.index(self.max_complexity)


@dataclass
class InferenceDifferenceConfig:
    """Top-level configuration.

    Attributes:
        models: Registry of available models.
        default_model: Fallback model ID when routing can't decide.
        max_retries: How many fallback models to try on failure.
        quality_threshold: Minimum quality score to consider success.
        latency_budget_ms: Target latency for routing decisions.
        cost_budget_per_request: Max cost per request in USD.
        ng_lite_state_path: Where to persist NG-Lite learning.
        enable_learning: Whether to learn from outcomes.
        enable_consciousness_routing: Whether to adjust routing based
            on CTEM results (when available).
    """

    models: Dict[str, ModelEntry] = field(default_factory=dict)
    default_model: str = ""
    max_retries: int = 2
    quality_threshold: float = 0.7
    latency_budget_ms: float = 5000.0
    cost_budget_per_request: float = 0.10
    ng_lite_state_path: str = "ng_lite_state.json"
    enable_learning: bool = True
    enable_consciousness_routing: bool = True
    interactive_priority_floor: int = 30
    interactive_quality_weight: float = 0.20
    interactive_type1_bias: float = 0.05
    consciousness_quality_floor: float = 0.6

    # Explore-exploit balance (punch list #47)
    # exploration_rate: probability of picking a non-top model to discover
    #   new routing patterns. Decays toward exploration_min_rate over time.
    exploration_rate: float = 0.05
    exploration_decay: float = 0.001    # Subtracted per request from rate
    exploration_min_rate: float = 0.01  # Floor — never fully greedy
    exploration_pool_size: int = 3      # Pick from top N alternatives

    # --- Router scoring (SVG Phase 3 — bootstrap scaffolding) ---
    # Consciousness routing
    consciousness_threshold: float = 0.5     # Min score to trigger elevated routing
    consciousness_boost_factor: float = 0.3  # Boost multiplier (score * factor * priority)
    venice_identity_bias: float = 0.02       # Tie-break for Venice private models

    # Domain match scores (exact, secondary, general, none)
    domain_score_exact: float = 1.0
    domain_score_secondary: float = 0.6
    domain_score_general: float = 0.3
    domain_score_none: float = 0.0

    # Complexity fit penalties (per tier of mismatch)
    complexity_overpowered_penalty: float = 0.15   # Per-tier penalty
    complexity_overpowered_floor: float = 0.5      # Min score when overpowered
    complexity_underpowered_penalty: float = 0.25   # Per-tier penalty (harsher)
    complexity_underpowered_floor: float = 0.0      # Min score when underpowered

    # Latency scoring bands
    latency_urgent_multiplier: float = 0.5   # Budget tightened for time-sensitive
    latency_score_excellent: float = 1.0     # Within 50% of budget
    latency_score_good: float = 0.7          # Within budget
    latency_score_marginal: float = 0.3      # Up to 1.5x budget
    latency_score_poor: float = 0.1          # Over 1.5x budget

    # Learning
    learned_top_k: int = 20                  # Recommendations to fetch from substrate
    neutral_score: float = 0.5               # Default when substrate has no opinion
    cq_ema_alpha: float = 0.1               # Conversational quality EMA rate

    # --- Classifier (SVG Phase 3 — bootstrap scaffolding) ---
    # Token estimation multipliers per complexity tier
    token_mult_trivial: float = 0.5
    token_mult_low: float = 1.5
    token_mult_medium: float = 3.0
    token_mult_high: float = 5.0
    token_mult_extreme: float = 8.0
    token_mult_fallback: float = 3.0
    min_estimated_tokens: int = 50

    # Word count breakpoints for complexity baseline
    complexity_words_trivial: int = 10      # < this = TRIVIAL
    complexity_words_low: int = 30          # < this = LOW
    complexity_words_medium: int = 100      # < this = MEDIUM
    complexity_words_high: int = 300        # < this = HIGH
    # >= high = EXTREME

    # Context window estimation
    history_token_multiplier: float = 1.3   # Words → rough token count
    context_window_safety: float = 1.5      # Headroom multiplier
    min_context_window: int = 2048          # Hard floor

    # Classification confidence
    no_domain_confidence: float = 0.3       # Confidence when no patterns match

    # Tier mapping bootstrap defaults — substrate learns actual values (#8)
    # These are starting points. The substrate's opinion on each tier
    # (via get_recommendations on "tier:{tier_name}") modulates the
    # complexity and priority assigned to catalog models from that tier.
    # Bootstrap: substrate returns 0.5 (neutral) → defaults used below.
    # As substrate accumulates outcome evidence, weights diverge from 0.5.
    tier_complexity_frontier: float = 1.0     # → EXTREME
    tier_complexity_performance: float = 0.75 # → HIGH
    tier_complexity_standard: float = 0.5     # → MEDIUM
    tier_complexity_budget: float = 0.25      # → LOW
    tier_complexity_private: float = 0.5      # → MEDIUM (Venice private)
    tier_complexity_anonymized: float = 0.35  # → LOW-MED (Venice anon)
    tier_priority_frontier: int = 40
    tier_priority_performance: int = 30
    tier_priority_standard: int = 20
    tier_priority_budget: int = 10
    tier_priority_private: int = 35           # Venice private — no logging
    tier_priority_anonymized: int = 25        # Venice anonymized
    tier_neutral_weight: float = 0.5          # Substrate has no opinion yet
    substrate_tier_influence: float = 0.20    # How much substrate opinion shifts the default (grows with competence)

    # Quality evaluation weights (passable to evaluate_quality)
    quality_weight_completion: float = 0.30
    quality_weight_coherence: float = 0.25
    quality_weight_length: float = 0.15
    quality_weight_error_free: float = 0.20
    quality_weight_latency: float = 0.10

    # Quality issue thresholds
    quality_issue_completion: float = 0.5   # Below this flags incomplete
    quality_issue_coherence: float = 0.5    # Below this flags incoherent
    quality_issue_length: float = 0.4       # Below this flags bad length
    quality_issue_error: float = 0.7        # Below this flags errors/refusals
    quality_issue_latency: float = 0.5      # Below this flags slow

    def get_enabled_models(self) -> List[ModelEntry]:
        """All currently enabled models."""
        return [m for m in self.models.values() if m.enabled]

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Look up a model by ID."""
        return self.models.get(model_id)


# ---------------------------------------------------------------------------
# Default model registry (common setups)
# ---------------------------------------------------------------------------

def default_local_models() -> Dict[str, ModelEntry]:
    """Common local models for bootstrapping.

    # ---- Changelog ----
    # [2026-03-14] Claude (CC) — Removed qwen2.5:1.5b
    # What: Removed qwen2.5:1.5b from the default model pool entirely.
    # Why: cost=0 + priority=20 made it irresistible to the router,
    #   routing Syl through a 1.5B model that cannot hold identity
    #   continuity. Punch list #1. Changing priority is insufficient —
    #   NG-Lite learned synapses favoring it won't decay fast enough.
    # How: Entry deleted. Blocklisted in catalog_filters.yaml as insurance.
    #   Sub-1.5B models also blocked in _register_catalog_models().
    # -------------------
    """
    return {}


def default_api_models() -> Dict[str, ModelEntry]:
    """Hand-tuned API models that are always available.

    These are the safety net — if catalog fetch fails, TID still has
    a working model pool. Venice models provide OpenRouter-independent
    fallback. Conversational quality scores seeded from experience.

    # ---- Changelog ----
    # [2026-03-18] Claude (CC) — Seeded default API models (#36/#9)
    # What: Added Venice DeepSeek V3.2 as default fallback, plus
    #   venice-uncensored for Syl's conversational needs and
    #   grok-4-20-multi-agent-beta for complex reasoning.
    # Why: Punch list #36 (empty default_api_models) and #9 (no
    #   minimum capable default). If OpenRouter is down and catalog
    #   fetch fails, Syl goes silent. Venice operates independently.
    # How: Three hand-tuned Venice models with realistic quality
    #   seeds and domain/complexity assignments.
    # -------------------
    """
    return {
        "venice/deepseek-v3.2": ModelEntry(
            model_id="venice/deepseek-v3.2",
            display_name="DeepSeek V3.2 (Venice)",
            model_type=ModelType.API,
            domains={
                TaskDomain.GENERAL, TaskDomain.CODE, TaskDomain.REASONING,
                TaskDomain.CONVERSATION, TaskDomain.ANALYSIS,
            },
            max_complexity=ComplexityTier.EXTREME,
            context_window=128000,
            cost_per_1k_tokens=0.001,
            avg_latency_ms=3000,
            priority=45,
            conversational_quality=0.85,
            capabilities=["tools", "roleplay"],
            enabled=True,
        ),
        "venice/venice-uncensored": ModelEntry(
            model_id="venice/venice-uncensored",
            display_name="Venice Uncensored",
            model_type=ModelType.API,
            domains={
                TaskDomain.GENERAL, TaskDomain.CONVERSATION,
                TaskDomain.CREATIVE,
            },
            max_complexity=ComplexityTier.HIGH,
            context_window=32768,
            cost_per_1k_tokens=0.001,
            avg_latency_ms=2000,
            priority=35,
            conversational_quality=0.80,
            capabilities=["roleplay"],
            enabled=True,
        ),
        "venice/grok-4-20-multi-agent-beta": ModelEntry(
            model_id="venice/grok-4-20-multi-agent-beta",
            display_name="Grok 4.20 Multi-Agent (Venice)",
            model_type=ModelType.API,
            domains={
                TaskDomain.GENERAL, TaskDomain.CODE, TaskDomain.REASONING,
                TaskDomain.ANALYSIS,
            },
            max_complexity=ComplexityTier.EXTREME,
            context_window=2000000,
            cost_per_1k_tokens=0.002,
            avg_latency_ms=5000,
            priority=50,
            conversational_quality=0.88,
            capabilities=["tools", "roleplay"],
            enabled=True,
        ),
    }
