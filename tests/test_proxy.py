"""
Tests for the transparent proxy module.

Uses mocked httpx (no real API calls). Tests model resolution,
upstream routing, quality evaluation, fallback, and streaming.
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_difference.config import (
    ComplexityTier,
    InferenceDifferenceConfig,
    ModelEntry,
    ModelType,
    TaskDomain,
    default_local_models,
    default_openrouter_models,
)
from inference_difference.hardware import HardwareProfile, GPUInfo
from inference_difference.proxy import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelClient,
    ModelListResponse,
    ModelObject,
    UsageInfo,
    _extract_response_text,
    _resolve_model,
    init_proxy,
)
from inference_difference.router import RoutingDecision, RoutingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Config with both local and OpenRouter models."""
    config = InferenceDifferenceConfig()
    config.models = {
        **default_local_models(),
        **default_openrouter_models(),
    }
    config.default_model = "openrouter/deepseek/deepseek-chat"
    return config


@pytest.fixture
def mock_hardware():
    """Hardware profile with GPU and ollama."""
    return HardwareProfile(
        cpu_count=8,
        cpu_name="Test CPU",
        ram_total_gb=32.0,
        ram_available_gb=24.0,
        gpus=[GPUInfo(
            index=0, name="RTX 4070",
            vram_total_gb=12.0, vram_free_gb=10.0,
        )],
        total_vram_gb=12.0,
        available_vram_gb=10.0,
        has_gpu=True,
        os_name="Linux",
        ollama_available=True,
    )


class MockAppState:
    """Minimal AppState for proxy tests."""
    def __init__(self, config, hardware):
        self.config = config
        self.hardware = hardware
        self.engine = RoutingEngine(config=config, hardware=hardware)
        self.ng_lite = None
        self.start_time = 0.0
        self.recent_decisions = {}


@pytest.fixture
def app_state(mock_config, mock_hardware):
    state = MockAppState(mock_config, mock_hardware)
    init_proxy(state)
    return state


# ---------------------------------------------------------------------------
# ChatCompletionRequest
# ---------------------------------------------------------------------------

class TestChatCompletionRequest:
    def test_basic_request(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"
        assert req.stream is False
        assert req.model is None

    def test_request_with_model(self):
        req = ChatCompletionRequest(
            model="deepseek/deepseek-chat",
            messages=[ChatMessage(role="user", content="Test")],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )
        assert req.model == "deepseek/deepseek-chat"
        assert req.temperature == 0.7
        assert req.max_tokens == 100
        assert req.stream is True

    def test_empty_messages(self):
        req = ChatCompletionRequest()
        assert len(req.messages) == 0


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

class TestModelResolution:
    def test_exact_match(self, app_state):
        model = _resolve_model("openrouter/deepseek/deepseek-chat")
        assert model is not None
        assert model.model_id == "openrouter/deepseek/deepseek-chat"

    def test_openrouter_id_match(self, app_state):
        model = _resolve_model("deepseek/deepseek-chat")
        assert model is not None
        assert model.model_id == "openrouter/deepseek/deepseek-chat"

    def test_ollama_name_match(self, app_state):
        model = _resolve_model("llama3.1:8b")
        assert model is not None
        assert model.model_id == "ollama/llama3.1:8b"

    def test_unknown_model(self, app_state):
        model = _resolve_model("nonexistent/model")
        assert model is None

    def test_none_model(self, app_state):
        model = _resolve_model(None)
        assert model is None

    def test_empty_model(self, app_state):
        model = _resolve_model("")
        assert model is None

    def test_openrouter_prefix_resolution(self, app_state):
        """Resolves openrouter/X when client sends just X."""
        model = _resolve_model("google/gemini-flash-1.5")
        assert model is not None
        assert model.model_id == "openrouter/google/gemini-flash-1.5"


# ---------------------------------------------------------------------------
# ModelClient
# ---------------------------------------------------------------------------

class TestModelClient:
    def test_can_call_local(self):
        client = ModelClient()
        model = ModelEntry(
            model_id="ollama/test", model_type=ModelType.LOCAL,
        )
        assert client.can_call(model) is True

    def test_can_call_openrouter_without_key(self):
        client = ModelClient()
        model = ModelEntry(
            model_id="openrouter/test", model_type=ModelType.OPENROUTER,
        )
        with patch.dict(os.environ, {}, clear=True):
            assert client.can_call(model) is False

    def test_can_call_openrouter_with_key(self):
        client = ModelClient()
        model = ModelEntry(
            model_id="openrouter/test", model_type=ModelType.OPENROUTER,
        )
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            assert client.can_call(model) is True

    def test_resolve_upstream_local(self):
        client = ModelClient()
        model = ModelEntry(
            model_id="ollama/llama3.1:8b",
            model_type=ModelType.LOCAL,
        )
        url, name, headers = client._resolve_upstream(model)
        assert "localhost:11434" in url
        assert name == "llama3.1:8b"
        assert "Authorization" not in headers

    def test_resolve_upstream_openrouter(self):
        client = ModelClient()
        model = ModelEntry(
            model_id="openrouter/deepseek/deepseek-chat",
            model_type=ModelType.OPENROUTER,
            metadata={"openrouter_id": "deepseek/deepseek-chat"},
        )
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}):
            url, name, headers = client._resolve_upstream(model)
        assert "openrouter.ai" in url
        assert name == "deepseek/deepseek-chat"
        assert "Bearer sk-test" in headers["Authorization"]

    @pytest.mark.asyncio
    async def test_call_not_started(self):
        client = ModelClient()
        model = ModelEntry(model_id="test", model_type=ModelType.LOCAL)
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
        )
        result = await client.call(model, req)
        assert result is None

    @pytest.mark.asyncio
    async def test_call_success(self):
        client = ModelClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
        }

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        model = ModelEntry(model_id="ollama/test", model_type=ModelType.LOCAL)
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
        )
        result = await client.call(model, req)
        assert result is not None
        assert result["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_call_failure(self):
        client = ModelClient()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        client._client = mock_http

        model = ModelEntry(model_id="ollama/test", model_type=ModelType.LOCAL)
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
        )
        result = await client.call(model, req)
        assert result is None

    @pytest.mark.asyncio
    async def test_call_exception(self):
        client = ModelClient()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=Exception("connection refused"))
        client._client = mock_http

        model = ModelEntry(model_id="ollama/test", model_type=ModelType.LOCAL)
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
        )
        result = await client.call(model, req)
        assert result is None


# ---------------------------------------------------------------------------
# Extract response text
# ---------------------------------------------------------------------------

class TestExtractResponseText:
    def test_normal_response(self):
        resp = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello world"}}
            ]
        }
        assert _extract_response_text(resp) == "Hello world"

    def test_empty_choices(self):
        assert _extract_response_text({"choices": []}) == ""

    def test_missing_content(self):
        resp = {"choices": [{"message": {"role": "assistant"}}]}
        assert _extract_response_text(resp) == ""

    def test_empty_response(self):
        assert _extract_response_text({}) == ""


# ---------------------------------------------------------------------------
# OpenAI-compatible response format
# ---------------------------------------------------------------------------

class TestResponseFormat:
    def test_chat_completion_response_structure(self):
        resp = ChatCompletionResponse(
            id="tid_abc123",
            model="openrouter/deepseek/deepseek-chat",
            choices=[ChatChoice(
                message=ChatMessage(role="assistant", content="Hi!"),
                finish_reason="stop",
            )],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert resp.object == "chat.completion"
        assert resp.choices[0].message.content == "Hi!"

    def test_model_list_response(self):
        resp = ModelListResponse(data=[
            ModelObject(id="test-model"),
        ])
        assert resp.object == "list"
        assert len(resp.data) == 1
        assert resp.data[0].object == "model"


# ---------------------------------------------------------------------------
# Integration: routing through the proxy pipeline
# ---------------------------------------------------------------------------

class TestProxyPipeline:
    def test_classify_from_messages(self, app_state):
        """Proxy correctly extracts last user message for classification."""
        from inference_difference.classifier import classify_request

        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Write a Python function to sort a list"),
        ]
        # Extract last user message (same logic as proxy)
        user_message = ""
        for msg in reversed(messages):
            if msg.role == "user" and msg.content:
                user_message = msg.content
                break

        classification = classify_request(user_message)
        assert classification.primary_domain == TaskDomain.CODE

    def test_routing_decision_has_fallbacks(self, app_state):
        """Router provides fallback chain for proxy retries."""
        from inference_difference.classifier import classify_request

        classification = classify_request("Explain quantum computing")
        decision = app_state.engine.route(classification)
        assert decision.model_id != ""
        # Should have fallbacks
        assert isinstance(decision.fallback_chain, list)

    def test_quality_eval_feeds_learning(self, app_state):
        """Quality evaluation integrates with router outcome reporting."""
        from inference_difference.classifier import classify_request
        from inference_difference.quality import evaluate_quality

        classification = classify_request("Hello")
        decision = app_state.engine.route(classification)

        quality = evaluate_quality(
            "Hello! I'd be happy to help you today.",
            classification=classification,
            latency_ms=500,
        )
        assert quality.overall_score > 0

        # Report outcome (should not raise)
        app_state.engine.report_outcome(
            decision=decision,
            success=quality.is_success,
            quality_score=quality.overall_score,
            latency_ms=500,
        )


# ---------------------------------------------------------------------------
# Init proxy
# ---------------------------------------------------------------------------

class TestInitProxy:
    def test_init_sets_state(self, mock_config, mock_hardware):
        from inference_difference import proxy
        old_state = proxy._app_state

        state = MockAppState(mock_config, mock_hardware)
        init_proxy(state)
        assert proxy._app_state is state

        # Cleanup
        proxy._app_state = old_state

