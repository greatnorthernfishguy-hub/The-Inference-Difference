"""
Configuration and model registry for The Inference Difference.

ModelEntry defines what the router knows about each available model:
capabilities, costs, latency characteristics, and hardware requirements.

The config is designed to be loaded from YAML/JSON but has sensible
defaults for zero-config startup.
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
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        "ollama/llama3.2:3b": ModelEntry(
            model_id="ollama/llama3.2:3b",
            display_name="Llama 3.2 3B",
            model_type=ModelType.LOCAL,
            domains={TaskDomain.GENERAL, TaskDomain.CONVERSATION},
            max_complexity=ComplexityTier.LOW,
            context_window=8192,
            cost_per_1k_tokens=0.0,
            avg_latency_ms=200,
            min_vram_gb=2.0,
            min_ram_gb=4.0,
            quantization="Q4_K_M",
            priority=10,
        ),
        "ollama/llama3.1:8b": ModelEntry(
            model_id="ollama/llama3.1:8b",
            display_name="Llama 3.1 8B",
            model_type=ModelType.LOCAL,
            domains={
                TaskDomain.GENERAL, TaskDomain.CODE,
                TaskDomain.REASONING, TaskDomain.CONVERSATION,
            },
            max_complexity=ComplexityTier.MEDIUM,
            context_window=32768,
            cost_per_1k_tokens=0.0,
            avg_latency_ms=500,
            min_vram_gb=5.0,
            min_ram_gb=8.0,
            quantization="Q4_K_M",
            priority=20,
        ),
        "ollama/deepseek-r1:14b": ModelEntry(
            model_id="ollama/deepseek-r1:14b",
            display_name="DeepSeek R1 14B",
            model_type=ModelType.LOCAL,
            domains={
                TaskDomain.CODE, TaskDomain.REASONING,
                TaskDomain.ANALYSIS,
            },
            max_complexity=ComplexityTier.HIGH,
            context_window=65536,
            cost_per_1k_tokens=0.0,
            avg_latency_ms=1200,
            min_vram_gb=10.0,
            min_ram_gb=16.0,
            quantization="Q4_K_M",
            priority=30,
        ),
        "ollama/qwen2.5-coder:7b": ModelEntry(
            model_id="ollama/qwen2.5-coder:7b",
            display_name="Qwen 2.5 Coder 7B",
            model_type=ModelType.LOCAL,
            domains={TaskDomain.CODE},
            max_complexity=ComplexityTier.MEDIUM,
            context_window=32768,
            cost_per_1k_tokens=0.0,
            avg_latency_ms=400,
            min_vram_gb=5.0,
            min_ram_gb=8.0,
            quantization="Q4_K_M",
            priority=25,
        ),
    }


def default_api_models() -> Dict[str, ModelEntry]:
    """Common API models for bootstrapping."""
    return {
        "anthropic/claude-sonnet-4-5-20250929": ModelEntry(
            model_id="anthropic/claude-sonnet-4-5-20250929",
            display_name="Claude Sonnet 4.5",
            model_type=ModelType.API,
            domains={
                TaskDomain.CODE, TaskDomain.REASONING,
                TaskDomain.CREATIVE, TaskDomain.ANALYSIS,
                TaskDomain.GENERAL,
            },
            max_complexity=ComplexityTier.EXTREME,
            context_window=200000,
            cost_per_1k_tokens=0.003,
            avg_latency_ms=2000,
            priority=50,
        ),
        "anthropic/claude-haiku-4-5-20251001": ModelEntry(
            model_id="anthropic/claude-haiku-4-5-20251001",
            display_name="Claude Haiku 4.5",
            model_type=ModelType.API,
            domains={
                TaskDomain.GENERAL, TaskDomain.CONVERSATION,
                TaskDomain.SUMMARIZATION,
            },
            max_complexity=ComplexityTier.MEDIUM,
            context_window=200000,
            cost_per_1k_tokens=0.001,
            avg_latency_ms=800,
            priority=40,
        ),
        "openai/gpt-4o": ModelEntry(
            model_id="openai/gpt-4o",
            display_name="GPT-4o",
            model_type=ModelType.API,
            domains={
                TaskDomain.CODE, TaskDomain.REASONING,
                TaskDomain.CREATIVE, TaskDomain.ANALYSIS,
                TaskDomain.GENERAL,
            },
            max_complexity=ComplexityTier.HIGH,
            context_window=128000,
            cost_per_1k_tokens=0.005,
            avg_latency_ms=1500,
            priority=45,
        ),
    }
