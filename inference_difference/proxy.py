"""
Transparent OpenAI-compatible proxy for The Inference Difference.

This is the heart of TID's ambitious vision: agents point their
OPENAI_BASE_URL at TID and every inference request is automatically
classified, routed to the optimal model, executed, quality-evaluated,
and learned from. No integration code needed — just one env var.

Endpoints:
    POST /v1/chat/completions  — OpenAI-compatible chat completion proxy
    GET  /v1/models            — OpenAI-compatible model listing

How it works:
    1. Receive OpenAI-format request
    2. Classify the request (domain, complexity, urgency)
    3. Route to the best model (or honor explicit model choice)
    4. Call the upstream provider (Ollama local, OpenRouter, etc.)
    5. Evaluate response quality
    6. If quality is below threshold, retry with fallback model
    7. Report outcome to NG-Lite for learning
    8. Return OpenAI-format response with X-TID-* transparency headers
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference_difference.classifier import classify_request
from inference_difference.config import ModelEntry, ModelType
from inference_difference.quality import evaluate_quality

logger = logging.getLogger("inference_difference.proxy")

proxy_router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic models (OpenAI-compatible)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str = "user"
    content: Optional[str] = ""
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Any] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ModelObject(BaseModel):
    """OpenAI-compatible model listing entry."""
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "inference-difference"


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response."""
    object: str = "list"
    data: List[ModelObject] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level state (set by init_proxy)
# ---------------------------------------------------------------------------

_app_state = None  # Set by init_proxy()


# ---------------------------------------------------------------------------
# Model Client — calls upstream providers
# ---------------------------------------------------------------------------

class ModelClient:
    """Async HTTP client for calling upstream model providers.

    Routes calls to the correct upstream based on ModelType:
      - LOCAL  → http://localhost:11434/v1/chat/completions (ollama)
      - OPENROUTER / API → https://openrouter.ai/api/v1/chat/completions
    """

    OLLAMA_BASE = "http://localhost:11434/v1"
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def startup(self):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def shutdown(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def can_call(self, model: ModelEntry) -> bool:
        """Check if we have the credentials to call this model."""
        if model.model_type == ModelType.LOCAL:
            return True  # Ollama doesn't need auth
        if model.model_type in (ModelType.OPENROUTER, ModelType.API):
            return bool(os.environ.get("OPENROUTER_API_KEY"))
        return False

    def _resolve_upstream(self, model: ModelEntry) -> tuple[str, str, dict[str, str]]:
        """Resolve upstream URL, model name, and headers for a model.

        Returns:
            (base_url, model_name, headers)
        """
        if model.model_type == ModelType.LOCAL:
            ollama_name = model.model_id.removeprefix("ollama/")
            return (
                f"{self.OLLAMA_BASE}/chat/completions",
                ollama_name,
                {"Content-Type": "application/json"},
            )

        # OpenRouter and API models go through OpenRouter
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        openrouter_id = model.metadata.get("openrouter_id", "")
        if not openrouter_id:
            # For direct API models, use the model_id as-is
            openrouter_id = model.model_id
        return (
            f"{self.OPENROUTER_BASE}/chat/completions",
            openrouter_id,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/greatnorthernfishguy-hub/The-Inference-Difference",
                "X-Title": "The Inference Difference",
            },
        )

    async def call(
        self,
        model: ModelEntry,
        request: ChatCompletionRequest,
    ) -> Optional[dict]:
        """Call an upstream model (non-streaming).

        Returns the raw JSON response dict, or None on failure.
        """
        if not self._client:
            logger.error("ModelClient not started")
            return None

        url, model_name, headers = self._resolve_upstream(model)

        # Build upstream payload
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = request.stop

        try:
            resp = await self._client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(
                "Upstream %s returned %d: %s",
                model.model_id, resp.status_code, resp.text[:200],
            )
            return None
        except Exception as e:
            logger.warning("Upstream call to %s failed: %s", model.model_id, e)
            return None

    async def call_stream(
        self,
        model: ModelEntry,
        request: ChatCompletionRequest,
    ) -> Optional[AsyncIterator[bytes]]:
        """Call an upstream model with streaming.

        Returns an async iterator of SSE bytes, or None on failure.
        Also returns the httpx Response object so we can close it properly.
        """
        if not self._client:
            logger.error("ModelClient not started")
            return None

        url, model_name, headers = self._resolve_upstream(model)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "stream": True,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = request.stop

        try:
            req = self._client.build_request("POST", url, json=payload, headers=headers)
            resp = await self._client.send(req, stream=True)
            if resp.status_code == 200:
                return resp
            logger.warning(
                "Upstream stream %s returned %d",
                model.model_id, resp.status_code,
            )
            await resp.aclose()
            return None
        except Exception as e:
            logger.warning("Upstream stream to %s failed: %s", model.model_id, e)
            return None


# Singleton client
model_client = ModelClient()


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def _resolve_model(requested: Optional[str]) -> Optional[ModelEntry]:
    """Resolve a client-specified model name to a registry entry.

    Resolution order:
      1. Exact model_id match
      2. Match by openrouter_id in metadata
      3. Match by ollama name (without prefix)
      4. None (let the router decide)
    """
    if not requested or not _app_state:
        return None

    config = _app_state.config

    # 1. Exact match
    entry = config.get_model(requested)
    if entry:
        return entry

    # 2. Match with "openrouter/" prefix
    prefixed = f"openrouter/{requested}"
    entry = config.get_model(prefixed)
    if entry:
        return entry

    # 3. Match by openrouter_id metadata
    for model in config.get_enabled_models():
        if model.metadata.get("openrouter_id") == requested:
            return model

    # 4. Match by ollama name
    ollama_prefixed = f"ollama/{requested}"
    entry = config.get_model(ollama_prefixed)
    if entry:
        return entry

    return None


# ---------------------------------------------------------------------------
# Proxy endpoint: POST /v1/chat/completions
# ---------------------------------------------------------------------------

@proxy_router.post("/v1/chat/completions")
async def proxy_chat_completions(req: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completion endpoint.

    This is the transparent proxy. Agents call this exactly like they
    would call OpenAI, and TID handles classification, routing, upstream
    calling, quality evaluation, and learning automatically.
    """
    if not _app_state:
        return _error_response("TID proxy not initialized", status=503)

    start_time = time.time()
    request_id = f"tid_{uuid.uuid4().hex[:12]}"

    # Extract last user message for classification
    user_message = ""
    for msg in reversed(req.messages):
        if msg.role == "user" and msg.content:
            user_message = msg.content
            break

    # Classify the request
    conversation_history = [
        m.content for m in req.messages[:-1]
        if m.content
    ]
    classification = classify_request(
        message=user_message,
        conversation_history=conversation_history,
    )

    # Resolve model: explicit choice or router decision
    explicit_model = _resolve_model(req.model)
    if explicit_model and model_client.can_call(explicit_model):
        chosen_model = explicit_model
        routing_decision = None
        routing_method = "explicit"
    else:
        # Let the router decide
        routing_decision = _app_state.engine.route(
            classification=classification,
            request_id=request_id,
        )
        chosen_model = routing_decision.model_entry

        # Walk the fallback chain if we can't call the primary
        if chosen_model and not model_client.can_call(chosen_model):
            chosen_model = None
            for fallback_id in routing_decision.fallback_chain:
                fb_entry = _app_state.config.get_model(fallback_id)
                if fb_entry and model_client.can_call(fb_entry):
                    chosen_model = fb_entry
                    break

        if not chosen_model:
            return _error_response(
                "No callable model available. Set OPENROUTER_API_KEY for "
                "cloud models, or ensure ollama is running for local models.",
                status=503,
            )
        routing_method = "routed"

    logger.info(
        "Proxy %s: %s -> %s (%s, domain=%s, complexity=%s)",
        request_id, req.model or "(auto)", chosen_model.model_id,
        routing_method,
        classification.primary_domain.value,
        classification.complexity.value,
    )

    # Streaming path
    if req.stream:
        return await _handle_streaming(
            req, chosen_model, request_id, routing_decision,
            classification, start_time,
        )

    # Non-streaming path
    return await _handle_non_streaming(
        req, chosen_model, request_id, routing_decision,
        classification, start_time,
    )


# ---------------------------------------------------------------------------
# Non-streaming handler
# ---------------------------------------------------------------------------

async def _handle_non_streaming(
    req: ChatCompletionRequest,
    chosen_model: ModelEntry,
    request_id: str,
    routing_decision,
    classification,
    start_time: float,
):
    """Handle a non-streaming chat completion request."""
    # Try primary model
    upstream_resp = await model_client.call(chosen_model, req)
    latency_ms = (time.time() - start_time) * 1000

    # Quality evaluation and retry logic
    if upstream_resp:
        response_text = _extract_response_text(upstream_resp)
        quality = evaluate_quality(
            response_text=response_text,
            classification=classification,
            latency_ms=latency_ms,
            quality_threshold=_app_state.config.quality_threshold,
        )

        # If quality is below threshold, try fallback
        if (
            not quality.is_success
            and routing_decision
            and routing_decision.fallback_chain
        ):
            logger.info(
                "Proxy %s: quality %.2f below threshold, trying fallback",
                request_id, quality.overall_score,
            )
            for fallback_id in routing_decision.fallback_chain[:_app_state.config.max_retries]:
                fb_entry = _app_state.config.get_model(fallback_id)
                if fb_entry and model_client.can_call(fb_entry):
                    fb_start = time.time()
                    fb_resp = await model_client.call(fb_entry, req)
                    fb_latency = (time.time() - fb_start) * 1000

                    if fb_resp:
                        fb_text = _extract_response_text(fb_resp)
                        fb_quality = evaluate_quality(
                            response_text=fb_text,
                            classification=classification,
                            latency_ms=fb_latency,
                            quality_threshold=_app_state.config.quality_threshold,
                        )
                        if fb_quality.overall_score > quality.overall_score:
                            upstream_resp = fb_resp
                            chosen_model = fb_entry
                            quality = fb_quality
                            latency_ms = fb_latency
                            response_text = fb_text
                            logger.info(
                                "Proxy %s: fallback %s improved quality to %.2f",
                                request_id, fb_entry.model_id,
                                fb_quality.overall_score,
                            )
                            break

        # Report outcome for learning
        _report_outcome(
            routing_decision, chosen_model, quality.overall_score,
            quality.is_success, latency_ms, request_id,
        )
    else:
        # Primary failed entirely — try fallbacks
        if routing_decision and routing_decision.fallback_chain:
            for fallback_id in routing_decision.fallback_chain[:_app_state.config.max_retries]:
                fb_entry = _app_state.config.get_model(fallback_id)
                if fb_entry and model_client.can_call(fb_entry):
                    fb_start = time.time()
                    upstream_resp = await model_client.call(fb_entry, req)
                    if upstream_resp:
                        chosen_model = fb_entry
                        latency_ms = (time.time() - fb_start) * 1000
                        logger.info(
                            "Proxy %s: primary failed, fallback %s succeeded",
                            request_id, fb_entry.model_id,
                        )
                        break

        if not upstream_resp:
            return _error_response(
                f"All models failed for request {request_id}",
                status=502,
            )

    # Build OpenAI-compatible response
    total_latency = (time.time() - start_time) * 1000
    return _build_response(upstream_resp, chosen_model, request_id, total_latency)


# ---------------------------------------------------------------------------
# Streaming handler
# ---------------------------------------------------------------------------

async def _handle_streaming(
    req: ChatCompletionRequest,
    chosen_model: ModelEntry,
    request_id: str,
    routing_decision,
    classification,
    start_time: float,
):
    """Handle a streaming chat completion request.

    Passes SSE chunks through to the client. Collects content for
    post-stream quality evaluation and learning.
    """
    resp = await model_client.call_stream(chosen_model, req)
    if resp is None:
        # Try fallbacks
        if routing_decision and routing_decision.fallback_chain:
            for fallback_id in routing_decision.fallback_chain[:_app_state.config.max_retries]:
                fb_entry = _app_state.config.get_model(fallback_id)
                if fb_entry and model_client.can_call(fb_entry):
                    resp = await model_client.call_stream(fb_entry, req)
                    if resp is not None:
                        chosen_model = fb_entry
                        break

        if resp is None:
            return _error_response(
                f"All models failed for streaming request {request_id}",
                status=502,
            )

    async def stream_generator():
        """Yield SSE chunks and collect content for post-stream eval."""
        collected_content = []
        try:
            async for line in resp.aiter_lines():
                yield f"{line}\n\n".encode()
                # Collect content from data chunks for quality eval
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            collected_content.append(delta)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
        finally:
            await resp.aclose()

            # Post-stream quality evaluation and learning
            full_text = "".join(collected_content)
            if full_text:
                latency_ms = (time.time() - start_time) * 1000
                quality = evaluate_quality(
                    response_text=full_text,
                    classification=classification,
                    latency_ms=latency_ms,
                    quality_threshold=_app_state.config.quality_threshold,
                )
                _report_outcome(
                    routing_decision, chosen_model, quality.overall_score,
                    quality.is_success, latency_ms, request_id,
                )

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "X-TID-Request-ID": request_id,
            "X-TID-Model": chosen_model.model_id,
            "X-TID-Routed": "true",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Model listing endpoint: GET /v1/models
# ---------------------------------------------------------------------------

@proxy_router.get("/v1/models")
async def list_models_openai():
    """OpenAI-compatible model listing.

    Returns all models TID can route to, so agents can see what's available.
    """
    if not _app_state:
        return ModelListResponse()

    models = []
    for entry in _app_state.config.get_enabled_models():
        models.append(ModelObject(
            id=entry.model_id,
            created=int(time.time()),
            owned_by=entry.model_type.value,
        ))

        # Also expose the openrouter_id as an alias
        or_id = entry.metadata.get("openrouter_id")
        if or_id:
            models.append(ModelObject(
                id=or_id,
                created=int(time.time()),
                owned_by=entry.model_type.value,
            ))

    return ModelListResponse(data=models)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_response_text(upstream_resp: dict) -> str:
    """Extract the assistant message text from an upstream response."""
    try:
        return (
            upstream_resp.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except (IndexError, KeyError):
        return ""


def _report_outcome(
    routing_decision,
    chosen_model: ModelEntry,
    quality_score: float,
    success: bool,
    latency_ms: float,
    request_id: str,
):
    """Report routing outcome for NG-Lite learning."""
    if not _app_state:
        return

    if routing_decision is not None:
        _app_state.engine.report_outcome(
            decision=routing_decision,
            success=success,
            quality_score=quality_score,
            latency_ms=latency_ms,
            metadata={"proxy": True, "model_used": chosen_model.model_id},
        )
        # Store for /outcome endpoint compatibility
        _app_state.recent_decisions[request_id] = routing_decision


def _build_response(
    upstream_resp: dict,
    chosen_model: ModelEntry,
    request_id: str,
    latency_ms: float,
) -> dict:
    """Build final response, injecting TID metadata.

    Returns the upstream response as-is with TID headers added
    via model and id fields. This preserves any provider-specific
    fields the client might rely on.
    """
    # Overlay TID metadata onto the upstream response
    resp = dict(upstream_resp)
    resp["id"] = request_id
    # Preserve upstream model name but add TID info
    if "model" not in resp:
        resp["model"] = chosen_model.model_id

    # Add TID metadata as an extension field
    resp["tid_metadata"] = {
        "routed_to": chosen_model.model_id,
        "model_type": chosen_model.model_type.value,
        "latency_ms": round(latency_ms, 1),
        "request_id": request_id,
    }

    return resp


def _error_response(message: str, status: int = 500) -> dict:
    """Build an OpenAI-compatible error response."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "server_error",
                "code": status,
            }
        },
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_proxy(app_state):
    """Initialize the proxy with the application state.

    Called from app.py lifespan. Avoids circular imports by receiving
    the state object rather than importing it.
    """
    global _app_state
    _app_state = app_state
    logger.info("Proxy initialized with %d models", len(app_state.config.get_enabled_models()))
