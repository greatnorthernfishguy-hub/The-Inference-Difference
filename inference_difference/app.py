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
"""

from __future__ import annotations

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
from inference_difference.responses_endpoint import register_responses_endpoint

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
_TIER_TO_PRIORITY = {
    "frontier": 40,
    "performance": 30,
    "standard": 20,
    "budget": 10,
}


def _register_catalog_models() -> None:
    """Register catalog models in the routing config.

    Converts CatalogModel entries from the dynamic catalog into ModelEntry
    objects so the router's standard 6-factor scoring can consider them
    alongside the hardcoded defaults. This is what makes ALL OpenRouter
    models available for routing — not just the 7 static entries.

    Skips models already in the static config to avoid overwriting
    hand-tuned entries with generic metadata.
    """
    if _state.catalog_manager is None:
        return

    registered = 0
    for cm in _state.catalog_manager.models:
        if cm.id in _state.config.models:
            continue  # Don't overwrite hand-tuned static entries

        # Map capabilities to task domains
        domains = {TaskDomain.GENERAL}
        for cap in cm.capabilities:
            if cap == "code":
                domains.add(TaskDomain.CODE)

        # Clamp cost to zero — some providers report negative costs
        # for promotional pricing, but ModelEntry requires >= 0.
        cost_per_1k = max(cm.cost_per_1m_input / 1000.0, 0.0)

        try:
            entry = ModelEntry(
                model_id=cm.id,
                display_name=cm.display_name or cm.id,
                model_type=ModelType.API,
                domains=domains,
                max_complexity=_TIER_TO_COMPLEXITY.get(
                    cm.provider_tier, ComplexityTier.MEDIUM,
                ),
                context_window=max(cm.context_window, 4096),
                cost_per_1k_tokens=cost_per_1k,
                avg_latency_ms=2000.0,  # Sensible default for cloud APIs
                priority=_TIER_TO_PRIORITY.get(cm.provider_tier, 20),
                enabled=cm.is_active,
            )
        except (ValueError, TypeError) as e:
            logger.debug("Skipping catalog model %s: %s", cm.id, e)
            continue

        _state.config.models[cm.id] = entry
        registered += 1

    if registered:
        logger.info(
            "Registered %d catalog models for routing (%d total available)",
            registered, len(_state.config.models),
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

    # Register ALL catalog models (OpenRouter, HuggingFace) in the routing
    # config so the router's standard scoring considers them as candidates.
    # This is what makes every OpenRouter model available, not just the
    # 7 hardcoded defaults.
    _register_catalog_models()

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

    # Create model client for forwarding requests to providers
    _state.model_client = ModelClient()

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

    # Register OpenClaw adapter — connect to gateway if configured
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
    # If caller specified an exact model (not auto), we'll try it
    # but still run hooks for security/compliance
    caller_chose_model = (
        resolved_model != "" and translation_type != "auto"
    )

    # Extract the user's last message for classification
    user_message = ""
    for msg in reversed(req.messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    # Build conversation history from prior messages
    conversation_history = [
        msg.get("content", "")
        for msg in req.messages[:-1]
        if msg.get("role") == "user"
    ]

    # --- Step 2: pre_route hooks ---
    ctx = HookContext(
        request_id=request_id,
        message=user_message,
        conversation_history=conversation_history,
        metadata=req.metadata or {},
    )

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

    if caller_chose_model:
        # Caller specified a model — honor it, but we still classified
        # for learning purposes
        selected_model = resolved_model
        fallback_chain: List[str] = []
        decision = None
    else:
        # Auto-route: TID picks the model
        consciousness_score = ctx.consciousness_score
        decision = _state.engine.route(
            classification=classification,
            consciousness_score=consciousness_score,
            request_id=request_id,
        )
        ctx.routing_decision = decision
        selected_model = decision.model_id
        fallback_chain = decision.fallback_chain

    # --- Step 4: post_route hooks ---
    if _state.module_registry is not None:
        _state.module_registry.dispatch(HookPhase.POST_ROUTE, ctx)

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
    )

    # --- Step 6: Auto-retry with fallback chain ---
    tried_models = [selected_model]
    for fallback_id in fallback_chain:
        if model_response.success:
            break
        logger.info(
            "Model %s failed, trying fallback %s",
            tried_models[-1], fallback_id,
        )
        model_response = _state.model_client.call(
            model_id=fallback_id,
            messages=req.messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        tried_models.append(fallback_id)

    # --- Step 7: Response hooks + learning (all internal) ---
    ctx.response_text = model_response.content

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

    Most OpenAI-compatible clients (SillyTavern, Open WebUI, etc.)
    send stream=true by default. Without this, the client hangs
    waiting for SSE chunks that never arrive.

    Learning and quality hooks run AFTER the stream completes — the
    StreamResult is populated as chunks flow through.
    """
    chunk_iter, stream_result = _state.model_client.call_stream(
        model_id=selected_model,
        messages=req.messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    def _generate():
        # Yield all chunks from the upstream provider
        yield from chunk_iter

        # --- Post-stream: learning and hooks (caller never sees this) ---
        ctx.response_text = stream_result.content

        if _state.module_registry is not None:
            _state.module_registry.dispatch(HookPhase.PRE_RESPONSE, ctx)

        quality = evaluate_quality(
            response_text=stream_result.content,
            classification=classification,
            latency_ms=stream_result.latency_ms,
            latency_budget_ms=_state.config.latency_budget_ms,
            quality_threshold=_state.config.quality_threshold,
        )
        ctx.quality_evaluation = quality

        if decision is not None:
            _state.engine.report_outcome(
                decision=decision,
                success=stream_result.success and quality.is_success,
                quality_score=quality.overall_score,
                latency_ms=stream_result.latency_ms,
            )

        if _state.module_registry is not None:
            _state.module_registry.dispatch(HookPhase.POST_RESPONSE, ctx)

        logger.info(
            "Streamed %s via %s (%.0fms, quality=%.2f)",
            request_id, selected_model,
            stream_result.latency_ms, quality.overall_score,
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


# ---------------------------------------------------------------------------
# Register Responses API endpoint (OpenClaw SSE streaming compatibility)
# ---------------------------------------------------------------------------

register_responses_endpoint(app, chat_completions)
