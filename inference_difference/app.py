"""
FastAPI application for The Inference Difference.

Exposes the routing engine as an HTTP API. Other modules (Cricket,
ClawGuard, Observatory, etc.) call this to route their inference
requests to optimal models.

Endpoints:
    POST /route          — Route a request to the best model
    POST /outcome        — Report routing outcome for learning
    GET  /health         — Health check with hardware/model status
    GET  /stats          — Router statistics and performance data
    GET  /models         — List available models
    GET  /classify       — Classify a request (debug/introspection)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference_difference.classifier import classify_request
from inference_difference.config import (
    InferenceDifferenceConfig,
    ModelEntry,
    default_api_models,
    default_local_models,
    default_openrouter_models,
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

    # Build model registry (local + OpenRouter + direct API)
    _state.config = InferenceDifferenceConfig()
    _state.config.models = {
        **default_local_models(),
        **default_openrouter_models(),
        **default_api_models(),
    }

    # Set default model.
    # Syl's configured default is DeepSeek via OpenRouter — respect that.
    # Fall back to a local model if one is available (free and fast).
    if _state.hardware.has_gpu and _state.hardware.ollama_available:
        _state.config.default_model = "ollama/llama3.1:8b"
    else:
        _state.config.default_model = "openrouter/deepseek/deepseek-chat"

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

    # Create routing engine
    _state.engine = RoutingEngine(
        config=_state.config,
        hardware=_state.hardware,
        ng_lite=_state.ng_lite,
    )
    _state.recent_decisions = {}

    logger.info(
        "The Inference Difference started: %d models, GPU=%s",
        len(_state.config.get_enabled_models()),
        _state.hardware.has_gpu,
    )

    yield

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
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/route", response_model=RouteResponse)
async def route_request(req: RouteRequest) -> RouteResponse:
    """Route an inference request to the best available model.

    Classifies the request, scores all candidate models, and returns
    the best match with a fallback chain and full reasoning trace.
    """
    # Classify
    classification = classify_request(
        message=req.message,
        conversation_history=req.conversation_history,
        metadata=req.metadata,
    )

    # Route
    decision = _state.engine.route(
        classification=classification,
        consciousness_score=req.consciousness_score,
        request_id=req.request_id,
    )

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

    Evaluates response quality and feeds the result back into
    NG-Lite to improve future routing decisions.
    """
    # Get the original decision (if tracked)
    decision = _state.recent_decisions.get(req.request_id)
    classification = decision.classification if decision else None

    # Evaluate quality
    quality = evaluate_quality(
        response_text=req.response_text,
        classification=classification,
        latency_ms=req.latency_ms,
        latency_budget_ms=_state.config.latency_budget_ms,
        quality_threshold=_state.config.quality_threshold,
    )

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
    return stats


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
