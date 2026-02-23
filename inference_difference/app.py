"""
FastAPI application for The Inference Difference.

TID operates as an ET Module host — a proxy/router that runs pluggable
ET modules through a standardized hook lifecycle on every request. The
modules (TrollGuard, OpenClaw, CTEM, etc.) plug into four lifecycle
phases: pre_route, post_route, pre_response, post_response.

The routing pipeline:
    1. Receive request
    2. Run pre_route hooks (TrollGuard scans, OpenClaw checks compliance)
    3. Classify request
    4. Route to best model
    5. Run post_route hooks (logging, auditing)
    6. [Caller executes the model — TID is a router, not a caller]
    7. Receive outcome via /outcome
    8. Run pre_response hooks (content filters)
    9. Evaluate quality
    10. Run post_response hooks (learning, telemetry)

Endpoints:
    POST /route          — Route a request to the best model
    POST /outcome        — Report routing outcome for learning
    GET  /health         — Health check with hardware/model status
    GET  /stats          — Router statistics and performance data
    GET  /models         — List available models
    GET  /modules        — List registered ET modules
    POST /classify       — Classify a request (debug/introspection)

Changelog (Grok audit response, 2026-02-19):
- ADDED: Optional API key auth via TID_API_KEY env var (audit: "no auth").
- ADDED: score_breakdown field to RouteResponse (audit: "no full reasoning
  trace"). Exposes per-factor scores for transparency.
- ADDED: Production exception handler (audit: "error leaks"). In production
  (TID_ENV=production), stack traces are suppressed in HTTP responses.
- KEPT: Synchronous classify_request in async endpoint (audit: "blocking").
  classify_request is a pure-CPU heuristic that runs in <1ms.

Changelog (ET Module Integration, 2026-02-23):
- ADDED: ET Module system — ModuleRegistry, hook lifecycle dispatch.
- ADDED: NG Ecosystem coordinator for cross-module learning.
- ADDED: TrollGuard module auto-registration on startup.
- ADDED: OpenClaw adapter auto-registration on startup.
- ADDED: /modules endpoint for module introspection.
- REFACTORED: /route now runs pre_route and post_route hooks.
- REFACTORED: /outcome now runs pre_response and post_response hooks.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference_difference.catalog_manager import CatalogManager
from inference_difference.classifier import classify_request
from inference_difference.config import (
    InferenceDifferenceConfig,
    ModelEntry,
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
from inference_difference.quality import evaluate_quality
from inference_difference.router import RoutingEngine

logger = logging.getLogger("inference_difference.app")

# ---------------------------------------------------------------------------
# Pydantic models for API
# ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    """Request body for /route."""
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

    # Set default model (prefer local if available)
    if _state.hardware.has_gpu:
        _state.config.default_model = "ollama/llama3.1:8b"
    else:
        _state.config.default_model = "anthropic/claude-haiku-4-5-20251001"

    # Initialize NG-Lite for learning
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from ng_lite import NGLite

        _state.ng_lite = NGLite(module_id="inference_difference")

        # Load persisted state if available
        state_path = _state.config.ng_lite_state_path
        if os.path.exists(state_path):
            _state.ng_lite.load(state_path)
            logger.info("NG-Lite state restored from %s", state_path)
    except Exception as e:
        logger.warning("NG-Lite initialization failed: %s", e)
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

    # Initialize DreamCycle for model property correlation analysis (§4.5.5)
    _state.dream_cycle = DreamCycle()

    # Create routing engine
    _state.engine = RoutingEngine(
        config=_state.config,
        hardware=_state.hardware,
        ng_lite=_state.ng_lite,
        catalog_manager=_state.catalog_manager,
        dream_cycle=_state.dream_cycle,
    )
    _state.recent_decisions = {}

    # Initialize NG Ecosystem coordinator for cross-module learning
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from ng_ecosystem import NGEcosystem

        _state.ng_ecosystem = NGEcosystem()

        # Register TID itself as a module in the ecosystem
        if _state.ng_lite is not None:
            _state.ng_ecosystem._modules["inference_difference"] = (
                __import__("ng_ecosystem").ModuleNGState(
                    module_id="inference_difference",
                    ng_lite=_state.ng_lite,
                    tier=1,
                )
            )

        logger.info("NG Ecosystem coordinator initialized")
    except Exception as e:
        logger.warning("NG Ecosystem initialization failed: %s", e)
        _state.ng_ecosystem = None

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

    # Register OpenClaw adapter (standalone mode)
    try:
        from inference_difference.et_module import ETModuleManifest
        from inference_difference.openclaw_adapter import OpenClawAdapter
        openclaw_manifest = ETModuleManifest(
            name="openclaw",
            version="1.0.0",
            description="Compliance and governance adapter",
            hooks=["pre_route", "post_route"],
            capabilities=["compliance", "governance"],
            priority=10,
        )
        openclaw = OpenClawAdapter(openclaw_manifest)
        _state.module_registry.register(openclaw)
        logger.info("OpenClaw adapter registered (standalone mode)")
    except Exception as e:
        logger.warning("OpenClaw adapter registration failed: %s", e)

    # Connect NG Ecosystem peers if multiple modules registered
    if _state.ng_ecosystem is not None:
        try:
            connections = _state.ng_ecosystem.connect_peers()
            if connections > 0:
                logger.info(
                    "NG Ecosystem: %d peer connections established",
                    connections,
                )
        except Exception as e:
            logger.warning("NG Ecosystem peer connection failed: %s", e)

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
            _state.ng_ecosystem.save_all(base_dir)
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

    # Persist NG-Lite state on shutdown
    if _state.ng_lite is not None:
        try:
            _state.ng_lite.save(_state.config.ng_lite_state_path)
            logger.info("NG-Lite state saved on shutdown")
        except Exception as e:
            logger.warning("NG-Lite save failed: %s", e)


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/route", response_model=RouteResponse)
async def route_request(req: RouteRequest) -> RouteResponse:
    """Route an inference request to the best available model.

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
        stats["ng_lite"] = _state.ng_lite.get_stats()
    if _state.catalog_manager is not None:
        stats["catalog"] = _state.catalog_manager.get_catalog_stats()
    if _state.dream_cycle is not None:
        stats["dream_cycle"] = _state.dream_cycle.get_stats()
    if _state.module_registry is not None:
        stats["modules"] = _state.module_registry.get_stats()
    if _state.ng_ecosystem is not None:
        stats["ng_ecosystem"] = _state.ng_ecosystem.get_stats()
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
