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
    """Common local models for bootstrapping."""
    return {

        "ollama/qwen2.5:1.5b": ModelEntry(
            model_id="ollama/qwen2.5:1.5b",
            display_name="Qwen 2.5 1.5B",
            model_type=ModelType.LOCAL,
            domains={
                TaskDomain.GENERAL, TaskDomain.CODE,
                TaskDomain.CONVERSATION, TaskDomain.REASONING,
            },
            max_complexity=ComplexityTier.LOW,
            context_window=32768,
            cost_per_1k_tokens=0.0,
            avg_latency_ms=300,
            min_ram_gb=2.0,
            priority=20,
            conversational_quality=0.2,
        ),
    }


def default_api_models() -> Dict[str, ModelEntry]:
    """Common API models for bootstrapping."""

    return {}
