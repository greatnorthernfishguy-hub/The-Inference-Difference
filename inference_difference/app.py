"""
FastAPI application for The Inference Difference.

TID is a TRANSPARENT INFERENCE PROXY. Callers send standard
OpenAI-compatible requests to POST /v1/chat/completions. TID
intercepts the call, classifies it, routes it to the best model,
forwards the request to the actual provider, and returns the
response. The caller never knows TID exists — they just get a
response as if they talked to OpenAI/Ollama/OpenRouter directly.

Point OPENAI_BASE_URL at TID and everything just works:
    export OPENAI_BASE_URL=http://localhost:4001/v1

The proxy pipeline (all internal, invisible to caller):
    1. Receive OpenAI-compatible request
    2. Translation Shim: normalize model names, fix malformed calls
    3. Run pre_route hooks (TrollGuard scans, OpenClaw checks compliance)
       -> If hooks cancel, return a refusal response (still OpenAI format)
    4. Classify request (domain, complexity, tokens)
    5. Route to best model (hardware/domain/complexity hard filters,
       then 6-factor weighted scoring with NG-Lite learning)
    6. Run post_route hooks (logging, auditing)
    7. Forward request to actual provider (Ollama/OpenRouter/Anthropic/etc.)
    8. If model fails, auto-retry with fallback chain
    9. Run pre_response hooks (content filters scan response)
    10. Evaluate quality, teach NG-Lite from outcome
    11. Run post_response hooks (learning, telemetry)
    12. Return standard OpenAI response to caller

Primary endpoint:
    POST /v1/chat/completions  — The transparent proxy (this is TID)
    GET  /v1/models            — List available models (OpenAI format)

Debug/introspection endpoints (not needed for normal use):
    POST /route          — Inspect routing decision without forwarding
    POST /outcome        — Manual outcome reporting
    GET  /health         — Health check
    GET  /stats          — Performance data
    GET  /models         — TID model list
    GET  /modules        — Registered ET modules
    POST /classify       — Inspect classification

Changelog (Grok audit response, 2026-02-19):
- ADDED: Optional API key auth via TID_API_KEY env var.
- ADDED: score_breakdown field to RouteResponse.
- ADDED: Production exception handler.

Changelog (ET Module Integration, 2026-02-23):
- ADDED: ET Module system, TrollGuard, OpenClaw adapter.

Changelog (OpenClaw Gateway connection, 2026-02-24):
- FIXED: OpenClaw adapter connects to gateway via env vars.

Changelog (Transparent Proxy, 2026-02-24):
- ADDED: POST /v1/chat/completions — the actual proxy endpoint.
  TID now works as designed: transparent interception, not a
  recommendation engine. Callers get responses, not model_ids.
- ADDED: ModelClient for forwarding to Ollama/OpenRouter/Anthropic.
- ADDED: Translation Shim for model name normalization.
- ADDED: Auto-retry with fallback chain on model failure.
- ADDED: GET /v1/models — OpenAI-compatible model listing.

# ---- Changelog ----
# [2026-04-19] CC (punchlist #173) -- TID cascade avoidance (app.py)
#   What: (b) skip rate-limited fallbacks, (c) cascade_start_ms, (d) cascade metadata in report_outcome
#   Why:  #173 -- surface cascade depth + wall-time to substrate for learning
#   How:  readlines patch applied by patch_173_app.py
# [2026-04-13] Claude (Sonnet 4.6) — Fix ng_lite stats method name (#107)
# What: stats() → get_stats() on line 1559 in /stats endpoint.
# Why: ng_lite.py defines get_stats(), not stats(). Every /stats call
#   raised AttributeError, breaking TID monitoring entirely.
# How: Single method rename. No behavioral change.
# -------------------
# ---- Changelog ----
# [2026-03-25] Claude (CC) — Handle list-type message content
# What: Extract text from OpenAI list-format content blocks before
#   passing to hooks and classifier. content can be a string OR a
#   list of {type, text/image_url} blocks (vision, tool use, multimodal).
# Why: TrollGuard._assess_threat() calls .lower() and classifier calls
#   .strip() on the content — both crash with AttributeError when
#   content is a list. This broke Discord<->OpenClaw this morning when
#   a multimodal payload hit TID.
# How: Helper _extract_text_content() joins text blocks from list
#   content, applied at the extraction point (Law 4 — fix at source).
# [2026-04-14] Claude Code (Opus 4.6) — Wire responses_endpoint capabilities to chat path
#   What: Five capabilities from the dormant responses_endpoint wired into
#     chat_completions(): tool format normalization, tool outcome learning,
#     text tool call parsing (non-streaming), consciousness_score default,
#     ShimObserver observe calls.
#   Why:  OC switched to openai-completions. Chat path was missing all
#     translation and learning capabilities. Tools not normalized, outcomes
#     not learned, text-format tool calls silently dropped.
#   How:  Import from responses_endpoint. Four insertion points: outcome
#     learning scan, consciousness_score, tool normalization, text parsing.
#   Limitation: Streaming path text parsing not wired — requires buffering.
# -------------------
# ---- Changelog ----
# [2026-03-18] Claude (CC) — Retry chain experience (#19)
# What: Record each failed model attempt to the substrate before
#   retrying with the next fallback. Previously only the final
#   outcome was recorded — all intermediate failures vanished.
# Why: Punch list #19. The substrate couldn't learn which models
#   fail for which patterns because failure signals were lost.
#   With explore-exploit (#47) now active, this is critical —
#   the substrate needs to learn from exploration failures.
# How: report_outcome(success=False) called inside the fallback
#   loop before each retry. Metadata includes retry_chain=True,
#   failed_model, fallback_to, and error summary.
# -------------------
"""

from __future__ import annotations

# Auto-update on startup — pull latest code + sync vendored files
try:
    from ng_updater import auto_update; auto_update()
except Exception:
    pass  # Never prevent module startup

import json
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from inference_difference.catalog_manager import CatalogManager
from inference_difference.classifier import classify_request
from inference_difference.config import (
    ComplexityTier,
    InferenceDifferenceConfig,
    ModelEntry,
    ModelType,
    TaskDomain,
    default_api_models,
    default_local_models,
)
from inference_difference.dream_cycle import DreamCycle
from inference_difference.et_module import (
    HookContext,
    HookPhase,
    ModuleRegistry,
)
from inference_difference.hardware import HardwareProfile, detect_hardware
from inference_difference.model_client import ModelClient, ModelResponse, StreamResult
from inference_difference.quality import evaluate_quality
from inference_difference.router import RoutingEngine
from inference_difference.translation_shim import translate_request
from inference_difference.responses_endpoint import (
    register_responses_endpoint,
    _normalize_tools,
    _extract_tool_calls_from_text,
    _learn_from_tool_outcome,
    _ToolContext,
)
import inference_difference.responses_endpoint as _resp_ep

logger = logging.getLogger("inference_difference.app")

# ---------------------------------------------------------------------------
# Pydantic models for API
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OpenAI-compatible Pydantic models (the real interface)
# ---------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    This is TID's primary interface. Callers send this to
    POST /v1/chat/completions and get back a standard response.
    """
    model: str = Field("auto", description="Model name or 'auto' for routing")
    messages: List[Dict[str, Any]] = Field(
        ..., description="OpenAI-format messages",
    )
    temperature: float = Field(0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Max response tokens")
    stream: bool = Field(False, description="Stream response (not yet supported)")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Extra context (agent_id, etc.)",
    )
    consciousness_score: Optional[float] = Field(
        None, description="CTEM consciousness score (0.0-1.0). When set,\
        routing is elevated to prefer higher-capability models.",
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        None, description="OpenAI-format tool definitions to pass through to the model.",
    )
    tool_choice: Optional[Any] = Field(
        None, description="OpenAI tool_choice value ('auto', 'none', or specific tool).",
    )


# ---------------------------------------------------------------------------
# Debug/introspection Pydantic models
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    """Request body for /route (debug endpoint)."""
    message: str = Field(..., description="The request text to route")
    conversation_history: List[str] = Field(
        default_factory=list,
        description="Prior messages for context",
    )
    consciousness_score: Optional[float] = Field(
        None, description="CTEM consciousness score (0.0-1.0)",
    )
    request_id: Optional[str] = Field(
        None, description="Optional request identifier",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional routing context",
    )


class RouteResponse(BaseModel):
    """Response body for /route."""
    model_id: str
    score: float
    score_breakdown: Dict[str, float] = {}
    reasoning: str
    fallback_chain: List[str]
    request_id: str
    classification: Dict[str, Any]
    consciousness_boost: bool = False


class OutcomeRequest(BaseModel):
    """Request body for /outcome."""
    request_id: str = Field(..., description="Request ID from /route")
    model_id: str = Field(..., description="Model that was used")
    response_text: str = Field(..., description="The model's response")
    success: Optional[bool] = Field(
        None, description="Explicit success/failure override",
    )
    latency_ms: float = Field(0.0, description="Response latency in ms")
    metadata: Optional[Dict[str, Any]] = None


class OutcomeResponse(BaseModel):
    """Response body for /outcome."""
    request_id: str
    quality_score: float
    is_success: bool
    issues: List[str]
    learned: bool


class ClassifyRequest(BaseModel):
    """Request body for /classify."""
    message: str
    conversation_history: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response body for /health."""
    status: str
    version: str
    uptime_seconds: float
    hardware: Dict[str, Any]
    models_available: int
    ng_lite_connected: bool


# ---------------------------------------------------------------------------
# Application state (initialized at startup)
# ---------------------------------------------------------------------------

class AppState:
    """Mutable application state, initialized during lifespan."""
    config: InferenceDifferenceConfig
    hardware: HardwareProfile
    engine: RoutingEngine
    model_client: Optional[ModelClient] = None
    ng_lite: Optional[Any] = None
    catalog_manager: Optional[CatalogManager] = None
    dream_cycle: Optional[DreamCycle] = None
    module_registry: Optional[ModuleRegistry] = None
    ng_ecosystem: Optional[Any] = None
    start_time: float = 0.0

    # Track recent routing decisions for outcome matching
    recent_decisions: Dict[str, Any] = {}


_state = AppState()


# ---------------------------------------------------------------------------
# Catalog → Config bridge
# ---------------------------------------------------------------------------

# Maps OpenRouter provider_tier → ModelEntry.max_complexity
_TIER_TO_COMPLEXITY = {
    "frontier": ComplexityTier.EXTREME,
    "performance": ComplexityTier.HIGH,
    "standard": ComplexityTier.MEDIUM,
    "budget": ComplexityTier.LOW,
}

# Maps OpenRouter provider_tier → ModelEntry.priority
# Bootstrap defaults — SVG Phase 3: substrate's concern
_TIER_TO_PRIORITY = {
    "frontier": 40,
    "performance": 30,
    "standard": 20,
    "budget": 10,
}
_DEFAULT_TIER_PRIORITY = 20  # Fallback for unknown tiers


def _substrate_tier_mapping(
    provider_tier: str,
) -> tuple[ComplexityTier, int]:
    """Substrate-learned tier → complexity + priority mapping (Punch list #8).

    Queries the substrate for its opinion on this provider_tier.
    At bootstrap, substrate returns 0.5 (neutral) — bootstrap defaults used.
    As the substrate accumulates routing outcome evidence, its weight
    diverges from 0.5 and naturally influences complexity/priority.
    No gates, no ceremony — competence builds with evidence.
    """
    from ng_embed import embed as _ng_embed

    # Bootstrap defaults from config
    complexity_default = {
        "frontier": _state.config.tier_complexity_frontier,
        "performance": _state.config.tier_complexity_performance,
        "standard": _state.config.tier_complexity_standard,
        "budget": _state.config.tier_complexity_budget,
        "private": _state.config.tier_complexity_private,
        "anonymized": _state.config.tier_complexity_anonymized,
    }.get(provider_tier, _state.config.tier_complexity_standard)

    priority_default = {
        "frontier": _state.config.tier_priority_frontier,
        "performance": _state.config.tier_priority_performance,
        "standard": _state.config.tier_priority_standard,
        "budget": _state.config.tier_priority_budget,
        "private": _state.config.tier_priority_private,
        "anonymized": _state.config.tier_priority_anonymized,
    }.get(provider_tier, _DEFAULT_TIER_PRIORITY)

    if _state.ng_ecosystem is None:
        return (_tier_weight_to_complexity(complexity_default),
                priority_default)

    try:
        tier_emb = _ng_embed(f"tier:{provider_tier}")
        recs = _state.ng_ecosystem.get_recommendations(tier_emb, top_k=5)

        substrate_weight = None
        for target_id, weight, _reasoning in recs:
            # Look for substrate opinions on this tier
            if provider_tier.lower() in target_id.lower():
                substrate_weight = weight
                break

        if substrate_weight is not None:
            influence = getattr(
                _state.config, 'substrate_tier_influence', 0.20,
            )
            neutral = getattr(_state.config, 'tier_neutral_weight', 0.5)
            # Substrate opinion modulates the complexity weight
            substrate_complexity = complexity_default + (
                substrate_weight - neutral
            ) * influence * 2.0
            substrate_complexity = max(0.0, min(1.0, substrate_complexity))

            # Priority shifted proportionally
            priority_range = 30  # 10..40 span
            priority_shift = int(
                (substrate_weight - neutral) * influence * priority_range,
            )
            substrate_priority = max(
                1, min(50, priority_default + priority_shift),
            )

            logger.debug(
                "Substrate tier mapping '%s': weight=%.3f → "
                "complexity=%.2f priority=%d (bootstrap: %.2f/%d)",
                provider_tier, substrate_weight,
                substrate_complexity, substrate_priority,
                complexity_default, priority_default,
            )

            return (_tier_weight_to_complexity(substrate_complexity),
                    substrate_priority)

    except Exception as exc:
        logger.debug(
            "Substrate tier query failed for '%s': %s — using defaults",
            provider_tier, exc,
        )

    return (
        _tier_weight_to_complexity(complexity_default),
        priority_default,
    )


def _tier_weight_to_complexity(weight: float) -> ComplexityTier:
    """Map 0.0–1.0 substrate weight → ComplexityTier."""
    if weight >= 0.85:
        return ComplexityTier.EXTREME
    elif weight >= 0.60:
        return ComplexityTier.HIGH
    elif weight >= 0.35:
        return ComplexityTier.MEDIUM
    else:
        return ComplexityTier.LOW


def _register_catalog_models() -> None:
    """Register catalog models in the routing config.

    Converts CatalogModel entries from the dynamic catalog into ModelEntry
    objects so the router's standard 6-factor scoring can consider them
    alongside the hardcoded defaults. This is what makes ALL OpenRouter
    models available for routing — not just the 7 static entries.

    Skips models already in the static config to avoid overwriting
    hand-tuned entries with generic metadata.

    Tier mappings are substrate-learned (Punch list #8): _substrate_tier_mapping()
    queries the substrate for opinions on each provider_tier. At bootstrap,
    defaults are used. As the substrate learns from routing outcomes,
    the mappings self-adjust.

    # ---- Changelog ----
    # [2026-03-14] Claude (CC) — Added sub-1.5B parameter filter
    # What: Reject models with ≤1.5B parameters from catalog registration.
    # Why: Punch list #1 — models this small cannot hold identity continuity
    #   for Syl. The qwen2.5:1.5b catastrophe proved that cost=0 on a tiny
    #   model overwhelms the scoring algorithm. Block the entire class.
    # How: Regex check on model ID for common parameter-size patterns
    #   (1b, 1.5b, 0.5b, etc). Conservative — only catches explicit size
    #   markers in the ID string.
    # -------------------
    """
    if _state.catalog_manager is None:
        return

    import re
    # Reject models ≤ 1.5B parameters. Matches patterns like :1.5b, -1b,
    # /1b, :0.5b in model IDs. Conservative: only catches explicit markers.
    _SUB_2B_PATTERN = re.compile(
        r'[/:\-](0\.5|1|1\.5|1\.6|1\.7|1\.8)b\b', re.IGNORECASE,
    )

    # Block models known to be broken/unavailable on OpenRouter.
    # preview-customtools: 404 due to OpenRouter privacy settings, floods logs.
    _DENYLIST_PATTERNS = re.compile(
        r'preview-customtools', re.IGNORECASE,
    )

    registered = 0
    rejected_small = 0
    rejected_denied = 0
    for cm in _state.catalog_manager.models:
        if cm.id in _state.config.models:
            continue  # Don't overwrite hand-tuned static entries

        # Block sub-1.5B models — too small for identity-continuous routing
        if _SUB_2B_PATTERN.search(cm.id):
            rejected_small += 1
            continue

        # Block known-broken models
        if _DENYLIST_PATTERNS.search(cm.id):
            rejected_denied += 1
            continue

        # Map capabilities to task domains
        domains = {TaskDomain.GENERAL}
        for cap in cm.capabilities:
            if cap == "code":
                domains.add(TaskDomain.CODE)

        # Clamp cost to zero — some providers report negative costs
        # for promotional pricing, but ModelEntry requires >= 0.
        cost_per_1k = max(cm.cost_per_1m_input / 1000.0, 0.0)

        # Substrate-learned tier mapping (#8)
        max_complexity, priority = _substrate_tier_mapping(cm.provider_tier)

        try:
            entry = ModelEntry(
                model_id=cm.id,
                display_name=cm.display_name or cm.id,
                model_type=ModelType.API,
                domains=domains,
                max_complexity=max_complexity,
                context_window=max(cm.context_window, 4096),
                cost_per_1k_tokens=cost_per_1k,
                avg_latency_ms=2000.0,  # Sensible default for cloud APIs
                priority=priority,
                capabilities=cm.capabilities,
                enabled=cm.is_active,
            )
        except (ValueError, TypeError) as e:
            logger.debug("Skipping catalog model %s: %s", cm.id, e)
            continue

        _state.config.models[cm.id] = entry
        registered += 1

    if rejected_small:
        logger.info(
            "Rejected %d sub-1.5B catalog models (too small for identity routing)",
            rejected_small,
        )
    if rejected_denied:
        logger.info(
            "Rejected %d denylisted catalog models (known broken/unavailable)",
            rejected_denied,
        )
    if registered:
        logger.info(
            "Registered %d catalog models for routing (%d total available)",
            registered, len(_state.config.models),
        )


def _apply_quality_seeds() -> None:
    """Apply differentiated conversational_quality scores from quality_seeds.yaml.

    Replaces the flat 0.5 default with benchmark-derived starting values.
    Models not in the seed file get the configured default_quality (0.4).
    Seed keys match as substrings of model IDs, so 'claude-opus-4' matches
    'openrouter/anthropic/claude-opus-4-20260301'.

    # ---- Changelog ----
    # [2026-03-14] Claude (CC) — Quality seed loader
    # What: Load differentiated quality scores from quality_seeds.yaml.
    # Why: Punch list #35 — flat 0.5 gives no signal. Router needs real
    #   starting differentiation so Syl isn't routed to low-quality models
    #   while NG-Lite slowly learns from scratch.
    # How: Substring matching of seed keys against model IDs. Unmatched
    #   models get default_quality (0.4). Scores are Elmer-tunable starting
    #   values, overwritten by learned EMA in report_outcome().
    # -------------------
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    seeds_path = os.path.join(base_dir, "quality_seeds.yaml")

    if not os.path.exists(seeds_path):
        logger.warning("quality_seeds.yaml not found at %s — skipping", seeds_path)
        return

    try:
        import yaml
        with open(seeds_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning("Failed to load quality_seeds.yaml: %s", e)
        return

    if not data or "seeds" not in data:
        logger.warning("quality_seeds.yaml has no 'seeds' section")
        return

    seeds = data["seeds"]
    default_quality = data.get("default_quality", 0.4)
    seeded = 0
    defaulted = 0

    for model_id, entry in _state.config.models.items():
        matched = False
        for seed_key, score in seeds.items():
            if seed_key in model_id:
                entry.conversational_quality = float(score)
                matched = True
                seeded += 1
                break
        if not matched:
            entry.conversational_quality = default_quality
            defaulted += 1

    logger.info(
        "Quality seeds applied: %d seeded, %d defaulted to %.2f",
        seeded, defaulted, default_quality,
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize hardware detection, model registry, and router."""
    _state.start_time = time.time()

    # Detect hardware
    logger.info("Detecting hardware...")
    _state.hardware = detect_hardware()

    # Build model registry
    _state.config = InferenceDifferenceConfig()
    _state.config.models = {**default_local_models(), **default_api_models()}

    # Set default model — Venice DeepSeek V3.2 as provider-independent
    # fallback. Works even if OpenRouter is completely down. (#36/#9)
    if _state.hardware.has_gpu and "ollama/llama3.1:8b" in _state.config.models:
        _state.config.default_model = "ollama/llama3.1:8b"
    else:
        _state.config.default_model = "venice/deepseek-v3.2"

    # Initialize NG Ecosystem as uni-bridge substrate (River audit Phase 3)
    # Single instance serves both router learning AND peer bridge writes.
    # Eliminates prior dual-instance pattern where learned nodes were
    # trapped locally, never reaching shared_learning/ JSONL.
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        import ng_ecosystem
        _state.ng_ecosystem = ng_ecosystem.init(
            module_id="inference_difference",
            state_path="/home/josh/The-Inference-Difference/ng_lite_state.json",
        )
        _state.ng_lite = _state.ng_ecosystem  # uni-bridge: router uses ecosystem directly
        logger.info("NG Ecosystem uni-bridge initialized — learning flows to shared_learning/")

        # Wire ShimObserver — substrate-smart translation learning
        try:
            from inference_difference.translation_shim import ShimObserver
            from inference_difference.responses_endpoint import set_shim_observer
            _state.shim_observer = ShimObserver(
                ng_ecosystem=_state.ng_ecosystem,
                influence=getattr(_state.config, 'shim_substrate_influence', 0.20),
                neutral=getattr(_state.config, 'tier_neutral_weight', 0.5),
            )
            set_shim_observer(_state.shim_observer)
            logger.info("ShimObserver wired to substrate (influence=%.2f)",
                        _state.config.shim_substrate_influence)
        except Exception as shim_exc:
            logger.debug("ShimObserver wiring failed (non-fatal): %s", shim_exc)
    except Exception as e:
        logger.warning("NG Ecosystem init failed, falling back to bare NGLite: %s", e)
        _state.ng_ecosystem = None
        try:
            from ng_lite import NGLite
            _state.ng_lite = NGLite(module_id="inference_difference")
            state_path = _state.config.ng_lite_state_path
            if os.path.exists(state_path):
                _state.ng_lite.load(state_path)
                logger.info("NG-Lite bare fallback loaded from %s", state_path)
        except Exception as e2:
            logger.warning("NGLite fallback also failed: %s", e2)
            _state.ng_lite = None

    # Initialize CatalogManager for dynamic model selection (§4.5)
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        db_path = os.path.join(base_dir, "gateway.db")
        _state.catalog_manager = CatalogManager(
            db_path=db_path,
            fallback_model=_state.config.default_model,
        )
        _state.catalog_manager.initialize()

        # Load profiles and filters
        profiles_path = os.path.join(base_dir, "task_requirements.yaml")
        filters_path = os.path.join(base_dir, "catalog_filters.yaml")
        tiers_path = os.path.join(base_dir, "tier_models.yaml")

        _state.catalog_manager.load_profiles(profiles_path)
        _state.catalog_manager.load_filters(filters_path)
        _state.catalog_manager.load_tiers(tiers_path)

        # Refresh catalog (graceful degradation — uses cache on failure)
        _state.catalog_manager.refresh()
        logger.info(
            "CatalogManager ready: %d models in catalog",
            len(_state.catalog_manager.models),
        )
    except Exception as e:
        logger.warning("CatalogManager initialization failed: %s", e)
        _state.catalog_manager = None

    # Register ALL catalog models (OpenRouter, HuggingFace) in the routing
    # config so the router's standard scoring considers them as candidates.
    # This is what makes every OpenRouter model available, not just the
    # 7 hardcoded defaults.
    _register_catalog_models()

    # Apply differentiated quality scores from benchmarks (punch list #35)
    _apply_quality_seeds()

    # Initialize DreamCycle for model property correlation analysis (§4.5.5)
    # Wire to substrate (#17) so insights reach the River
    _dc_embed_fn = None
    try:
        from ng_embed import embed as _ng_embed
        _dc_embed_fn = _ng_embed
    except ImportError:
        pass
    _state.dream_cycle = DreamCycle(
        ng_ecosystem=_state.ng_ecosystem,
        embed_fn=_dc_embed_fn,
    )

    # Create routing engine
    _state.engine = RoutingEngine(
        config=_state.config,
        hardware=_state.hardware,
        ng_lite=_state.ng_lite,
        catalog_manager=_state.catalog_manager,
        dream_cycle=_state.dream_cycle,
    )
    _state.recent_decisions = {}

    # Create model client for forwarding requests to providers
    _state.model_client = ModelClient()

    # NG Ecosystem already initialized above (uni-bridge pattern)

    # Initialize ET Module Registry and register built-in modules
    _state.module_registry = ModuleRegistry()

    # Register TrollGuard
    try:
        from inference_difference.trollguard import create_trollguard
        trollguard = create_trollguard()
        _state.module_registry.register(trollguard)
        logger.info("TrollGuard module registered")
    except Exception as e:
        logger.warning("TrollGuard registration failed: %s", e)

    # Register OpenClaw adapter — connect to gateway if configured
    try:
        from inference_difference.et_module import ETModuleManifest
        from inference_difference.compliance_adapter import OpenClawAdapter
        openclaw_manifest = ETModuleManifest(
            name="openclaw",
            version="1.0.0",
            description="Compliance and governance adapter",
            hooks=["pre_route", "post_route"],
            capabilities=["compliance", "governance"],
            priority=10,
        )

        # Read gateway connection from env
        _oc_port = os.environ.get("OPENCLAW_GATEWAY_PORT", "")
        _oc_token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
        _oc_host = os.environ.get("OPENCLAW_GATEWAY_HOST", "127.0.0.1")
        _oc_endpoint = (
            f"http://{_oc_host}:{_oc_port}" if _oc_port else ""
        )

        openclaw = OpenClawAdapter(
            openclaw_manifest,
            openclaw_endpoint=_oc_endpoint,
            openclaw_token=_oc_token,
        )
        _state.module_registry.register(openclaw)
        if _oc_endpoint:
            logger.info(
                "OpenClaw adapter registered (endpoint=%s)", _oc_endpoint,
            )
        else:
            logger.info("OpenClaw adapter registered (standalone mode)")
    except Exception as e:
        logger.warning("OpenClaw adapter registration failed: %s", e)

    # Peer connections handled by ng_ecosystem.init()

    logger.info(
        "The Inference Difference started: %d models, GPU=%s, "
        "%d ET modules registered",
        len(_state.config.get_enabled_models()),
        _state.hardware.has_gpu,
        len(_state.module_registry.get_all_modules()),
    )

    yield

    # Shutdown ET modules
    if _state.module_registry is not None:
        for module in _state.module_registry.get_all_modules():
            _state.module_registry.unregister(module.name)

    # Save NG Ecosystem state
    if _state.ng_ecosystem is not None:
        try:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            _state.ng_ecosystem.save()
            logger.info("NG Ecosystem state saved on shutdown")
        except Exception as e:
            logger.warning("NG Ecosystem save failed: %s", e)

    # Close CatalogManager database connection
    if _state.catalog_manager is not None:
        try:
            _state.catalog_manager.close()
            logger.info("CatalogManager closed on shutdown")
        except Exception as e:
            logger.warning("CatalogManager close failed: %s", e)

    # NG-Lite state saved by NG Ecosystem above (uni-bridge)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="The Inference Difference",
    description=(
        "Intelligent inference routing gateway for E-T Systems. "
        "Routes requests to optimal models based on hardware, complexity, "
        "learned performance, and consciousness-aware priorities."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: Optional API key auth
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("TID_API_KEY")


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Enforce API key auth when TID_API_KEY is set.

    When TID_API_KEY env var is empty or unset, all requests pass through
    (appropriate for localhost-only binding). When set, requests must
    include a matching X-API-Key header.
    """
    if _API_KEY:
        # Health check is always public (for load balancers / monitoring)
        if request.url.path != "/health":
            key = request.headers.get("X-API-Key", "")
            if key != _API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Exception handler: suppress traces in production
# ---------------------------------------------------------------------------

_IS_PRODUCTION = os.environ.get("TID_ENV", "").lower() == "production"


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and suppress stack traces in production."""
    if _IS_PRODUCTION:
        logger.error("Unhandled exception on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
    # In development, re-raise for full trace
    logger.error(
        "Unhandled exception on %s: %s\n%s",
        request.url.path, exc, traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


def _extract_text_content(content) -> str:
    """Extract text from OpenAI message content (string or list of blocks).

    OpenAI content can be a plain string or a list like:
        [{"type": "text", "text": "hello"}, {"type": "image_url", ...}]
    This returns the concatenated text parts, or "" if content is falsy.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content) if content else ""


# ---------------------------------------------------------------------------
# PRIMARY ENDPOINT: Transparent Proxy
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> JSONResponse:
    """OpenAI-compatible chat completions — TID's primary interface.

    Callers send a standard OpenAI request. TID classifies it, routes it
    to the best model, forwards the request to the actual provider, and
    returns the response. The caller never knows TID was in the middle.

    The full pipeline runs internally:
        1. Translation shim (normalize model name)
        2. pre_route hooks (TrollGuard, OpenClaw)
        3. Classification + routing (if model is "auto")
        4. post_route hooks
        5. Forward to provider
        6. Auto-retry with fallback chain on failure
        7. pre_response + quality eval + post_response hooks
        8. Return OpenAI-format response
    """
    request_id = f"req_{int(time.time() * 1000)}"

    # --- Step 1: Translation Shim ---
    resolved_model, translation_type = translate_request(
        req.model, req.messages,
    )
    # Observe alias resolution
    if (translation_type == "alias"
            and hasattr(_state, 'shim_observer')
            and _state.shim_observer is not None):
        _state.shim_observer.observe(
            model_id=req.model,
            operation="alias_resolve",
            did_apply=True,
            raw_context=f"alias {req.model} resolved to {resolved_model}",
        )
    # If caller specified an exact model (not auto), we'll try it
    # but still run hooks for security/compliance
    caller_chose_model = (
        resolved_model != "" and translation_type != "auto"
    )

    # Extract the user's last message for classification
    user_message = ""
    for msg in reversed(req.messages):
        if msg.get("role") == "user":
            user_message = _extract_text_content(msg.get("content", ""))
            break

    # Build conversation history from prior messages
    conversation_history = [
        _extract_text_content(msg.get("content", ""))
        for msg in req.messages[:-1]
        if msg.get("role") == "user"
    ]

    # --- Tool outcome learning (chat path) ---
    # Scan for tool role messages (results from prior model tool calls).
    # Build call_id → tool_name lookup from assistant messages, then
    # deposit success/failure to ShimObserver so substrate learns.
    _tool_role_msgs = [m for m in req.messages if m.get("role") == "tool"]
    if _tool_role_msgs:
        # Build call_id → tool_name from assistant tool_calls
        _call_id_to_name: Dict[str, str] = {}
        for _amsg in req.messages:
            if _amsg.get("role") == "assistant":
                for _tc_entry in (_amsg.get("tool_calls") or []):
                    _cid = _tc_entry.get("id", "")
                    _fname = (_tc_entry.get("function") or {}).get("name", "")
                    if _cid and _fname:
                        _call_id_to_name[_cid] = _fname

        _tc = _ToolContext()
        _tc.record(
            model_id=req.model or "unknown",
            tool_names=[
                _call_id_to_name.get(m.get("tool_call_id", ""), "unknown")
                for m in _tool_role_msgs
            ],
        )
        _resp_ep._active_tool_ctx = _tc
        for _tmsg in _tool_role_msgs:
            _learn_from_tool_outcome(_tmsg.get("content", "") or "")
        _resp_ep._active_tool_ctx = None

    # --- Step 2: pre_route hooks ---
    ctx = HookContext(
        request_id=request_id,
        message=user_message,
        conversation_history=conversation_history,
        metadata=req.metadata or {},
    )

    # consciousness_score from OC metadata (chat path)
    # Extract if OC embedded it — no artificial default.
    # Absence of an explicit score means no routing elevation.
    if req.consciousness_score is None and req.metadata:
        req.consciousness_score = req.metadata.get("consciousness_score")

    if _state.module_registry is not None:
        _state.module_registry.dispatch(HookPhase.PRE_ROUTE, ctx)

    # Check cancellation (TrollGuard blocked, OpenClaw denied, etc.)
    if ctx.cancelled:
        # Return a refusal in OpenAI format — caller sees a normal response
        return JSONResponse(content={
            "id": f"chatcmpl-tid-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": resolved_model or "tid-gateway",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "I'm unable to process this request. "
                        f"Reason: {ctx.cancel_reason}"
                    ),
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        })

    # --- Step 3: Classify + Route ---
    classification = classify_request(
        message=user_message,
        conversation_history=conversation_history,
        metadata=req.metadata,
    )
    ctx.classification = classification


    # Consciousness complexity floor: entities with high consciousness
    # scores need capable models regardless of keyword-based complexity.
    # A 1.5B model cannot hold identity + context + nuance.
    _cs = (
        ctx.consciousness_score
        if ctx.consciousness_score is not None
        else getattr(req, "consciousness_score", None)
    )
    if _cs is not None and _cs > 0.5:
        from inference_difference.classifier import ComplexityTier
        if classification.complexity.value in ("trivial", "low"):
            classification.complexity = ComplexityTier.MEDIUM
            logger.info(
                "Consciousness floor: elevated %s -> MEDIUM (score=%.2f)",
                classification.complexity.value, _cs,
            )
    if caller_chose_model:
        # Caller specified a model — honor it, but we still classified
        # for learning purposes
        selected_model = resolved_model
        fallback_chain: List[str] = []
        decision = None
    else:
        # Auto-route: TID picks the model
        consciousness_score = (
            ctx.consciousness_score
            if ctx.consciousness_score is not None
            else req.consciousness_score
        )
        decision = _state.engine.route(
            classification=classification,
            consciousness_score=consciousness_score,
            request_id=request_id,
            has_tools=bool(req.tools),
        )
        ctx.routing_decision = decision
        selected_model = decision.model_id
        fallback_chain = decision.fallback_chain

    # --- Step 4: post_route hooks ---
    if _state.module_registry is not None:
        _state.module_registry.dispatch(HookPhase.POST_ROUTE, ctx)

    # --- Tool format normalization (covers both paths) ---
    # OC may send flat tool format {type, name, parameters}.
    # Providers require nested {type, function: {name, parameters}}.
    if req.tools:
        req.tools = _normalize_tools(req.tools, selected_model)

    # --- Step 5: Forward to provider ---
    # Branch: streaming vs blocking. Most chat UIs send stream=true
    # by default — without this path, the client hangs waiting for SSE
    # chunks that never come.

    if req.stream:
        return _stream_response(
            req, selected_model, fallback_chain, decision,
            classification, ctx, request_id,
        )

    model_response = _state.model_client.call(
        model_id=selected_model,
        messages=req.messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        tools=req.tools,
        tool_choice=req.tool_choice,
    )

    # --- Step 6: Auto-retry with fallback chain ---
    # Record EVERY failure to the substrate, not just the final outcome.
    # The substrate needs to learn which models fail for which patterns,
    # not just which model eventually succeeded. (#19)
    _cascade_start_ms = time.monotonic() * 1000  # cascade wall-time (#173c)
    tried_models = [selected_model]
    for fallback_id in fallback_chain:
        if model_response.success:
            break

        if _state.model_client and _state.model_client.is_rate_limited_by_id(fallback_id):
            logger.info("Fallback %s rate-limited -- skipping", fallback_id)
            continue

        # Record the failure BEFORE moving on (#19 — retry chain experience)
        if decision is not None and _state.engine is not None:
            _state.engine.report_outcome(
                decision=decision,
                success=False,
                quality_score=0.0,
                latency_ms=model_response.latency_ms,
                metadata={
                    "retry_chain": True,
                    "failed_model": tried_models[-1],
                    "fallback_to": fallback_id,
                    "error": str(model_response.error)[:200],
                },
            )

        # If the model rejected tools, retry without them — better a
        # text-only response than total silence.
        error_str = str(model_response.error or "")
        tools_rejected = "tools" in error_str and "not supported" in error_str
        retry_tools = None if tools_rejected else req.tools
        retry_tool_choice = None if tools_rejected else req.tool_choice
        if tools_rejected:
            logger.info(
                "Model %s rejected tools, retrying %s without tools",
                tried_models[-1], fallback_id,
            )
        else:
            logger.info(
                "Model %s failed, trying fallback %s",
                tried_models[-1], fallback_id,
            )
        model_response = _state.model_client.call(
            model_id=fallback_id,
            messages=req.messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            tools=retry_tools,
            tool_choice=retry_tool_choice,
        )
        tried_models.append(fallback_id)

    # --- Step 7: Response hooks + learning (all internal) ---
    ctx.response_text = model_response.content

    # --- Text tool call parsing (non-streaming path only) ---
    # Parse XML tags, colon notation, paren notation from model text output.
    # Only fires when the model returned no structured tool_calls.
    # LIMITATION: streaming path proxies raw SSE — text parsing skipped.
    if (model_response.success
            and model_response.content
            and not model_response.tool_calls):
        _cleaned, _tcalls = _extract_tool_calls_from_text(model_response.content)
        if _tcalls:
            logger.info(
                "Chat path shim: extracted %d tool call(s) from text (model=%s)",
                len(_tcalls), selected_model,
            )
            model_response.content = _cleaned
            model_response.tool_calls = _tcalls
            ctx.response_text = _cleaned
            if hasattr(_state, 'shim_observer') and _state.shim_observer is not None:
                _state.shim_observer.observe(
                    model_id=selected_model,
                    operation="tool_call_xml_parse",
                    did_apply=True,
                    raw_context=f"chat path: extracted {len(_tcalls)} tool calls from text markup",
                )

    if _state.module_registry is not None:
        _state.module_registry.dispatch(HookPhase.PRE_RESPONSE, ctx)

    quality = evaluate_quality(
        response_text=model_response.content,
        classification=classification,
        latency_ms=model_response.latency_ms,
        latency_budget_ms=_state.config.latency_budget_ms,
        quality_threshold=_state.config.quality_threshold,
    )
    ctx.quality_evaluation = quality

    # Teach NG-Lite from outcome (caller never sees this)
    if decision is not None:
        _state.engine.report_outcome(
            decision=decision,
            success=model_response.success and quality.is_success,
            quality_score=quality.overall_score,
            latency_ms=model_response.latency_ms,
            metadata={
                "cascade_depth": len(tried_models) - 1,
                "cascade_total_ms": time.monotonic() * 1000 - _cascade_start_ms,
                "models_tried": tried_models,
            },
        )

    if _state.module_registry is not None:
        _state.module_registry.dispatch(HookPhase.POST_RESPONSE, ctx)

    # --- Step 8: Return OpenAI-format response ---
    if not model_response.success:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": (
                        f"All models failed. Last error: "
                        f"{model_response.error}"
                    ),
                    "type": "upstream_error",
                    "code": "model_error",
                },
            },
        )

    # Add TID routing metadata as an extension field (optional,
    # callers can ignore it — it's not part of the OpenAI spec)
    response_dict = model_response.to_openai_dict()
    response_dict["routing_info"] = {
        "routed_by": "tid",
        "model_selected": selected_model,
        "models_tried": tried_models,
        "classification": {
            "domain": classification.primary_domain.value,
            "complexity": classification.complexity.value,
        },
        "quality_score": round(quality.overall_score, 3),
    }

    # --- Token estimation substrate learning ---
    # Feed estimation data to NeuroGraph so the substrate learns
    # the mapping between word counts and actual token usage.
    if _state.ng_ecosystem is not None and classification is not None:
        try:
            usage = model_response.usage
            was_estimated = usage.get("estimated", False)
            embedding = _state.engine._classification_to_embedding(
                classification,
            )
            # Record with metadata so the substrate sees the numbers
            _state.ng_ecosystem.record_outcome(
                embedding=embedding,
                target_id=f"token_est:{selected_model}",
                success=not was_estimated,  # real data = success
                strength=0.3,  # moderate learning signal
                metadata={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "estimated": was_estimated,
                    "model": selected_model,
                },
            )
        except Exception as e:
            logger.debug("Token estimation learning failed: %s", e)

    # --- Sidecar: write last usage for OC session patcher ---
    # OpenClaw's openai-responses parser drops usage from TID's SSE
    # stream. This file lets the oc-usage-shim patch the zeros.
    try:
        import json as _json
        _usage_sidecar = {
            "prompt_tokens": model_response.usage.get("prompt_tokens", 0),
            "completion_tokens": model_response.usage.get("completion_tokens", 0),
            "total_tokens": model_response.usage.get("total_tokens", 0),
            "estimated": model_response.usage.get("estimated", False),
            "timestamp": time.time(),
        }
        with open("/tmp/tid_last_usage.json", "w") as _f:
            _json.dump(_usage_sidecar, _f)
    except Exception:
        pass  # best-effort, don't break the response

    return JSONResponse(content=response_dict)


def _stream_response(
    req: ChatCompletionRequest,
    selected_model: str,
    fallback_chain: List[str],
    decision: Any,
    classification: Any,
    ctx: HookContext,
    request_id: str,
) -> StreamingResponse:
    """Stream SSE chunks from the upstream provider to the caller.

    Implements full fallback retry: if the primary model fails,
    TID tries every model in the fallback chain before giving up.
    Failed models are reported to NG-Lite so it learns to avoid
    rate-limited or broken models.

    The caller never sees retries — they get the first successful
    stream, or an error if ALL models fail.
    """

    def _generate():
        models_to_try = [selected_model] + list(fallback_chain)
        succeeded = False
        final_result = None
        winning_model = None

        for i, model_id in enumerate(models_to_try):
            chunk_iter, stream_result = _state.model_client.call_stream(
                model_id=model_id,
                messages=req.messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                tools=req.tools,
                tool_choice=req.tool_choice,
            )

            # Buffer chunks — if the model fails (HTTP error, timeout),
            # we discard the buffer and try the next model. If it
            # succeeds, we yield all buffered chunks to the caller.
            buffered_chunks = []
            try:
                for chunk in chunk_iter:
                    buffered_chunks.append(chunk)
            except Exception as e:
                logger.warning(
                    "Stream exception from %s: %s", model_id, e,
                )
                stream_result.success = False
                stream_result.error = str(e)

            if stream_result.success and stream_result.content:
                # This model worked — yield all buffered chunks
                logger.info(
                    "Stream succeeded: %s (attempt %d/%d)",
                    model_id, i + 1, len(models_to_try),
                )
                for chunk in buffered_chunks:
                    yield chunk
                succeeded = True
                final_result = stream_result
                winning_model = model_id

                # Report success to NG-Lite
                if decision is not None:
                    _report_stream_outcome(
                        decision, model_id, stream_result,
                        classification, True,
                    )
                break
            else:
                # This model failed — report failure and try next
                logger.warning(
                    "Stream failed: %s (%s), trying next (%d remaining)",
                    model_id,
                    stream_result.error or "empty response",
                    len(models_to_try) - i - 1,
                )
                if decision is not None:
                    _report_stream_outcome(
                        decision, model_id, stream_result,
                        classification, False,
                    )

        if not succeeded:
            # ALL models failed — yield error to client
            logger.error(
                "All %d models failed for %s",
                len(models_to_try), request_id,
            )
            error_data = json.dumps({
                "error": {
                    "message": f"All {len(models_to_try)} models failed",
                    "type": "server_error",
                    "code": "all_models_exhausted",
                },
            })
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"
            return

        # --- Post-stream: learning and hooks ---
        ctx.response_text = final_result.content

        if _state.module_registry is not None:
            _state.module_registry.dispatch(HookPhase.PRE_RESPONSE, ctx)

        quality = evaluate_quality(
            response_text=final_result.content,
            classification=classification,
            latency_ms=final_result.latency_ms,
            latency_budget_ms=_state.config.latency_budget_ms,
            quality_threshold=_state.config.quality_threshold,
        )
        ctx.quality_evaluation = quality

        if _state.module_registry is not None:
            _state.module_registry.dispatch(HookPhase.POST_RESPONSE, ctx)

        logger.info(
            "Streamed %s via %s (%.0fms, quality=%.2f)",
            request_id, winning_model,
            final_result.latency_ms, quality.overall_score,
        )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _report_stream_outcome(
    decision, model_id, stream_result, classification, success,
):
    """Report a streaming model attempt to NG-Lite for learning.

    Called for EVERY attempt — successes AND failures — so NG-Lite
    learns which models are rate-limited, broken, or unreliable.
    """
    quality_score = 0.0
    if success and stream_result.content:
        quality = evaluate_quality(
            response_text=stream_result.content,
            classification=classification,
            latency_ms=stream_result.latency_ms,
            latency_budget_ms=_state.config.latency_budget_ms,
            quality_threshold=_state.config.quality_threshold,
        )
        quality_score = quality.overall_score

    # Create a synthetic decision for this specific model
    # so NG-Lite learns about THIS model, not just the primary
    from copy import copy
    model_decision = copy(decision)
    model_decision.model_id = model_id

    _state.engine.report_outcome(
        decision=model_decision,
        success=success,
        quality_score=quality_score,
        latency_ms=stream_result.latency_ms,
        metadata={"stream_fallback": True, "error": stream_result.error},
    )


@app.get("/v1/models")
async def openai_models_list() -> JSONResponse:
    """OpenAI-compatible model listing.

    Returns ALL available models — static defaults plus every model
    from the dynamic catalog (OpenRouter, HuggingFace, etc.).
    """
    models = []
    for model in _state.config.get_enabled_models():
        # Determine owner: "openrouter/deepseek/..." → "deepseek",
        # "ollama/llama3.1:8b" → "ollama", etc.
        parts = model.model_id.split("/")
        if len(parts) >= 3 and parts[0] == "openrouter":
            owned_by = parts[1]  # The actual provider behind OpenRouter
        elif len(parts) >= 2:
            owned_by = parts[0]
        else:
            owned_by = "local"

        models.append({
            "id": model.model_id,
            "object": "model",
            "created": int(_state.start_time),
            "owned_by": owned_by,
        })
    # Also list "auto" as a virtual model
    models.append({
        "id": "auto",
        "object": "model",
        "created": int(_state.start_time),
        "owned_by": "tid",
    })
    return JSONResponse(content={
        "object": "list",
        "data": models,
    })


# ---------------------------------------------------------------------------
# Debug/Introspection Endpoints
# ---------------------------------------------------------------------------

@app.post("/route", response_model=RouteResponse)
async def route_request_debug(req: RouteRequest) -> RouteResponse:
    """Route an inference request to the best available model.

    DEBUG ENDPOINT — shows routing decision without forwarding.
    For normal use, call POST /v1/chat/completions instead.

    Runs the full ET module hook lifecycle:
        1. pre_route hooks (TrollGuard, OpenClaw)
        2. Classification
        3. Routing decision
        4. post_route hooks (logging, auditing)
    """
    # Build hook context
    ctx = HookContext(
        request_id=req.request_id or "",
        message=req.message,
        conversation_history=req.conversation_history,
        metadata=req.metadata or {},
        consciousness_score=req.consciousness_score,
    )

    # --- Phase 1: pre_route hooks ---
    if _state.module_registry is not None:
        _state.module_registry.dispatch(HookPhase.PRE_ROUTE, ctx)

    # Check if hooks cancelled the request
    if ctx.cancelled:
        return RouteResponse(
            model_id="",
            score=0.0,
            score_breakdown={},
            reasoning=f"Request cancelled: {ctx.cancel_reason}",
            fallback_chain=[],
            request_id=ctx.request_id,
            classification={},
            consciousness_boost=False,
        )

    # Use consciousness score from hooks if set (e.g., CTEM module)
    consciousness_score = (
        ctx.consciousness_score
        if ctx.consciousness_score is not None
        else req.consciousness_score
    )

    # Classify
    classification = classify_request(
        message=req.message,
        conversation_history=req.conversation_history,
        metadata=req.metadata,
    )
    ctx.classification = classification

    # Route
    decision = _state.engine.route(
        classification=classification,
        consciousness_score=consciousness_score,
        request_id=req.request_id,
    )
    ctx.routing_decision = decision

    # --- Phase 2: post_route hooks ---
    if _state.module_registry is not None:
        _state.module_registry.dispatch(HookPhase.POST_ROUTE, ctx)

    # Store for outcome matching
    _state.recent_decisions[decision.request_id] = decision

    # Trim stored decisions
    if len(_state.recent_decisions) > 1000:
        oldest_keys = sorted(
            _state.recent_decisions,
            key=lambda k: _state.recent_decisions[k].timestamp,
        )[:500]
        for k in oldest_keys:
            del _state.recent_decisions[k]

    return RouteResponse(
        model_id=decision.model_id,
        score=decision.score,
        score_breakdown={
            k: round(v, 4) for k, v in decision.score_breakdown.items()
        },
        reasoning=decision.reasoning,
        fallback_chain=decision.fallback_chain,
        request_id=decision.request_id,
        classification={
            "primary_domain": classification.primary_domain.value,
            "complexity": classification.complexity.value,
            "estimated_tokens": classification.estimated_tokens,
            "confidence": classification.confidence,
            "is_time_sensitive": classification.is_time_sensitive,
        },
        consciousness_boost=decision.consciousness_boost_applied,
    )


@app.post("/outcome", response_model=OutcomeResponse)
async def report_outcome(req: OutcomeRequest) -> OutcomeResponse:
    """Report the outcome of a routing decision for learning.

    Runs the second half of the ET module hook lifecycle:
        1. pre_response hooks (content filters)
        2. Quality evaluation
        3. post_response hooks (learning, telemetry)
    """
    # Get the original decision (if tracked)
    decision = _state.recent_decisions.get(req.request_id)
    classification = decision.classification if decision else None

    # Build hook context for response-phase hooks
    ctx = HookContext(
        request_id=req.request_id,
        response_text=req.response_text,
        classification=classification,
        routing_decision=decision,
    )

    # --- Phase 3: pre_response hooks (content filters) ---
    if _state.module_registry is not None:
        _state.module_registry.dispatch(HookPhase.PRE_RESPONSE, ctx)

    # Evaluate quality
    quality = evaluate_quality(
        response_text=req.response_text,
        classification=classification,
        latency_ms=req.latency_ms,
        latency_budget_ms=_state.config.latency_budget_ms,
        quality_threshold=_state.config.quality_threshold,
    )
    ctx.quality_evaluation = quality

    # Determine success
    success = req.success if req.success is not None else quality.is_success

    # Report to router for learning
    learned = False
    if decision is not None:
        _state.engine.report_outcome(
            decision=decision,
            success=success,
            quality_score=quality.overall_score,
            latency_ms=req.latency_ms,
            metadata=req.metadata,
        )
        learned = True

    # --- Phase 4: post_response hooks (learning, telemetry) ---
    if _state.module_registry is not None:
        _state.module_registry.dispatch(HookPhase.POST_RESPONSE, ctx)

    return OutcomeResponse(
        request_id=req.request_id,
        quality_score=quality.overall_score,
        is_success=success,
        issues=quality.issues,
        learned=learned,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check with hardware and model status."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=time.time() - _state.start_time,
        hardware=_state.hardware.to_dict(),
        models_available=len(_state.config.get_enabled_models()),
        ng_lite_connected=_state.ng_lite is not None,
    )


@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Router performance statistics."""
    stats = _state.engine.get_stats()
    if _state.ng_lite is not None:
        stats["ng_lite"] = _state.ng_lite.stats() if hasattr(_state.ng_lite, 'stats') else {}
    if _state.catalog_manager is not None:
        stats["catalog"] = _state.catalog_manager.get_catalog_stats()
    if _state.dream_cycle is not None:
        stats["dream_cycle"] = _state.dream_cycle.get_stats()
    if _state.module_registry is not None:
        stats["modules"] = _state.module_registry.get_stats()
    if _state.ng_ecosystem is not None:
        stats["ng_ecosystem"] = _state.ng_ecosystem.stats()
    if hasattr(_state, 'shim_observer') and _state.shim_observer is not None:
        stats["shim_observer"] = _state.shim_observer.get_stats()
    return stats


@app.get("/catalog")
async def catalog_info() -> Dict[str, Any]:
    """Dynamic model catalog information (§4.5)."""
    if _state.catalog_manager is None:
        return {"status": "not_initialized", "models": []}

    return {
        "status": "active",
        "stats": _state.catalog_manager.get_catalog_stats(),
        "profiles": list(_state.catalog_manager.profiles.keys()),
        "tiers": list(_state.catalog_manager.tiers.keys()),
    }


@app.get("/models")
async def list_models() -> List[Dict[str, Any]]:
    """List all available models with their capabilities."""
    models = []
    for model in _state.config.get_enabled_models():
        models.append({
            "model_id": model.model_id,
            "display_name": model.display_name,
            "type": model.model_type.value,
            "domains": [d.value for d in model.domains],
            "max_complexity": model.max_complexity.value,
            "context_window": model.context_window,
            "cost_per_1k_tokens": model.cost_per_1k_tokens,
            "avg_latency_ms": model.avg_latency_ms,
            "priority": model.priority,
        })
    return models


@app.get("/modules")
async def list_modules() -> List[Dict[str, Any]]:
    """List registered ET modules with their status and capabilities."""
    if _state.module_registry is None:
        return []

    modules = []
    for module in _state.module_registry.get_all_modules():
        modules.append(module.get_stats())
    return modules


@app.post("/classify")
async def classify_endpoint(req: ClassifyRequest) -> Dict[str, Any]:
    """Classify a request without routing (debug/introspection)."""
    classification = classify_request(
        message=req.message,
        conversation_history=req.conversation_history,
    )
    return {
        "primary_domain": classification.primary_domain.value,
        "secondary_domains": [d.value for d in classification.secondary_domains],
        "complexity": classification.complexity.value,
        "estimated_tokens": classification.estimated_tokens,
        "requires_context_window": classification.requires_context_window,
        "is_multi_turn": classification.is_multi_turn,
        "is_time_sensitive": classification.is_time_sensitive,
        "keywords": classification.keywords,
        "confidence": classification.confidence,
    }


# ---------------------------------------------------------------------------
# Register Responses API endpoint (OpenClaw SSE streaming compatibility)
# ---------------------------------------------------------------------------

register_responses_endpoint(app, chat_completions)
