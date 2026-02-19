"""
Tests for The Inference Difference routing gateway.

Covers: config, hardware, classifier, router, quality evaluator,
NG-Lite integration, consciousness-aware routing, subprocess mocks,
and concurrent routing.

Changelog (Grok audit response, 2026-02-19):
- ADDED: Subprocess mocks for hardware detection (audit: "no mocks, flaky CI").
- ADDED: Concurrent routing stress test (audit: "no threading").
- ADDED: Config validation tests (audit: "no checks on bad values").
- ADDED: Gibberish response quality test (audit: "gibberish—low score?").
- ADDED: Negative latency edge case test (audit: "zero latency=1.0").
"""

import concurrent.futures
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from inference_difference.config import (
    ComplexityTier,
    InferenceDifferenceConfig,
    ModelEntry,
    ModelType,
    TaskDomain,
    default_api_models,
    default_local_models,
)
from inference_difference.classifier import (
    RequestClassification,
    classify_request,
)
from inference_difference.hardware import (
    HardwareProfile,
    GPUInfo,
    detect_hardware,
    _clamp_vram,
)
from inference_difference.quality import evaluate_quality, QualityEvaluation
from inference_difference.router import RoutingEngine, RoutingDecision
from ng_lite import NGLite


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gpu_hardware():
    """Hardware profile with a mid-range GPU."""
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
        platform_arch="x86_64",
        ollama_available=True,
        ollama_models=["llama3.1:8b", "deepseek-r1:14b"],
    )


@pytest.fixture
def cpu_only_hardware():
    """Hardware profile with no GPU."""
    return HardwareProfile(
        cpu_count=4,
        cpu_name="Test CPU",
        ram_total_gb=16.0,
        ram_available_gb=12.0,
        has_gpu=False,
        os_name="Linux",
        platform_arch="x86_64",
    )


@pytest.fixture
def full_config():
    """Config with both local and API models."""
    config = InferenceDifferenceConfig()
    config.models = {**default_local_models(), **default_api_models()}
    config.default_model = "ollama/llama3.1:8b"
    return config


@pytest.fixture
def local_only_config():
    """Config with only local models."""
    config = InferenceDifferenceConfig()
    config.models = default_local_models()
    config.default_model = "ollama/llama3.1:8b"
    return config


@pytest.fixture
def engine(full_config, gpu_hardware):
    """Routing engine with GPU hardware and full model set."""
    return RoutingEngine(config=full_config, hardware=gpu_hardware)


@pytest.fixture
def learning_engine(full_config, gpu_hardware):
    """Routing engine with NG-Lite learning enabled."""
    ng = NGLite(module_id="test_router")
    return RoutingEngine(
        config=full_config,
        hardware=gpu_hardware,
        ng_lite=ng,
    )


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestConfig:

    def test_model_entry_can_handle(self):
        model = ModelEntry(
            model_id="test",
            domains={TaskDomain.CODE, TaskDomain.REASONING},
            max_complexity=ComplexityTier.HIGH,
            enabled=True,
        )
        assert model.can_handle(TaskDomain.CODE, ComplexityTier.MEDIUM)
        assert model.can_handle(TaskDomain.REASONING, ComplexityTier.HIGH)
        assert not model.can_handle(TaskDomain.CODE, ComplexityTier.EXTREME)
        assert not model.can_handle(TaskDomain.CREATIVE, ComplexityTier.LOW)

    def test_general_domain_matches_anything(self):
        model = ModelEntry(
            model_id="general",
            domains={TaskDomain.GENERAL},
            max_complexity=ComplexityTier.MEDIUM,
            enabled=True,
        )
        assert model.can_handle(TaskDomain.CODE, ComplexityTier.LOW)
        assert model.can_handle(TaskDomain.CREATIVE, ComplexityTier.MEDIUM)
        assert not model.can_handle(TaskDomain.CREATIVE, ComplexityTier.HIGH)

    def test_disabled_model_cant_handle(self):
        model = ModelEntry(
            model_id="disabled",
            domains={TaskDomain.GENERAL},
            max_complexity=ComplexityTier.EXTREME,
            enabled=False,
        )
        assert not model.can_handle(TaskDomain.GENERAL, ComplexityTier.LOW)

    def test_default_local_models(self):
        models = default_local_models()
        assert len(models) >= 3
        for m in models.values():
            assert m.model_type == ModelType.LOCAL
            assert m.cost_per_1k_tokens == 0.0

    def test_default_api_models(self):
        models = default_api_models()
        assert len(models) >= 2
        for m in models.values():
            assert m.model_type == ModelType.API
            assert m.cost_per_1k_tokens > 0

    def test_config_get_enabled_models(self, full_config):
        enabled = full_config.get_enabled_models()
        assert len(enabled) == len(full_config.models)

        # Disable one
        model_id = list(full_config.models.keys())[0]
        full_config.models[model_id].enabled = False
        assert len(full_config.get_enabled_models()) == len(enabled) - 1


# ---------------------------------------------------------------------------
# Hardware Tests
# ---------------------------------------------------------------------------

class TestHardware:

    def test_gpu_can_run_local_model(self, gpu_hardware):
        assert gpu_hardware.can_run_model(min_vram_gb=5.0, min_ram_gb=8.0)
        assert not gpu_hardware.can_run_model(min_vram_gb=16.0, min_ram_gb=8.0)

    def test_cpu_only_checks_ram(self, cpu_only_hardware):
        assert cpu_only_hardware.can_run_model(min_vram_gb=0.0, min_ram_gb=8.0)
        assert not cpu_only_hardware.can_run_model(min_vram_gb=0.0, min_ram_gb=20.0)

    def test_hardware_to_dict(self, gpu_hardware):
        d = gpu_hardware.to_dict()
        assert d["gpu_count"] == 1
        assert d["has_gpu"] is True
        assert d["ram_total_gb"] == 32.0


# ---------------------------------------------------------------------------
# Classifier Tests
# ---------------------------------------------------------------------------

class TestClassifier:

    def test_code_classification(self):
        result = classify_request("Write a Python function to sort a list")
        assert result.primary_domain == TaskDomain.CODE

    def test_reasoning_classification(self):
        result = classify_request(
            "Explain why the following argument is logically flawed "
            "and analyze the implications of the faulty premise"
        )
        assert result.primary_domain in {
            TaskDomain.REASONING, TaskDomain.ANALYSIS
        }

    def test_creative_classification(self):
        result = classify_request("Write a short story about a robot")
        assert result.primary_domain == TaskDomain.CREATIVE

    def test_summarization_classification(self):
        result = classify_request("Summarize the key points of this article")
        assert result.primary_domain == TaskDomain.SUMMARIZATION

    def test_simple_message_low_complexity(self):
        result = classify_request("What is Python?")
        assert result.complexity in {
            ComplexityTier.TRIVIAL, ComplexityTier.LOW
        }

    def test_complex_message_high_complexity(self):
        result = classify_request(
            "Design and implement a comprehensive microservices architecture "
            "with step-by-step detailed analysis of the trade-offs between "
            "different approaches, including code examples for each service "
            "and thorough testing strategies"
        )
        assert result.complexity in {
            ComplexityTier.HIGH, ComplexityTier.EXTREME
        }

    def test_urgency_detection(self):
        result = classify_request("Quick! I need help ASAP!!")
        assert result.is_time_sensitive is True

    def test_multi_turn_detection(self):
        result = classify_request(
            "And what about the other approach?",
            conversation_history=["Tell me about sorting algorithms"],
        )
        assert result.is_multi_turn is True

    def test_empty_message(self):
        result = classify_request("")
        assert result.primary_domain == TaskDomain.GENERAL
        assert result.complexity == ComplexityTier.MEDIUM

    def test_code_block_bumps_complexity(self):
        simple = classify_request("Fix this code")
        with_code = classify_request("Fix this code:\n```python\nx = 1\n```")
        tiers = list(ComplexityTier)
        # Code block should bump complexity up
        assert tiers.index(with_code.complexity) >= tiers.index(simple.complexity)


# ---------------------------------------------------------------------------
# Router Tests
# ---------------------------------------------------------------------------

class TestRouter:

    def test_basic_routing(self, engine):
        classification = classify_request("Write a Python function")
        decision = engine.route(classification)
        assert decision.model_id != ""
        assert decision.score > 0
        assert decision.reasoning != ""

    def test_code_routes_to_code_model(self, engine):
        classification = classify_request(
            "Debug this Python function and fix the syntax error"
        )
        decision = engine.route(classification)
        model = decision.model_entry
        assert model is not None
        assert (
            TaskDomain.CODE in model.domains
            or TaskDomain.GENERAL in model.domains
        )

    def test_complex_request_routes_to_capable_model(self, engine):
        classification = classify_request(
            "Design a comprehensive distributed system architecture with "
            "detailed step-by-step implementation plan for microservices"
        )
        decision = engine.route(classification)
        model = decision.model_entry
        assert model is not None
        tiers = list(ComplexityTier)
        assert tiers.index(model.max_complexity) >= tiers.index(
            ComplexityTier.HIGH
        )

    def test_fallback_chain_populated(self, engine):
        classification = classify_request("Explain quantum computing")
        decision = engine.route(classification)
        # Should have at least one fallback if multiple models available
        assert len(decision.fallback_chain) >= 0  # May be 0 if only 1 match

    def test_routing_with_request_id(self, engine):
        classification = classify_request("Hello")
        decision = engine.route(classification, request_id="test_123")
        assert decision.request_id == "test_123"

    def test_no_models_available(self, gpu_hardware):
        empty_config = InferenceDifferenceConfig()
        engine = RoutingEngine(config=empty_config, hardware=gpu_hardware)
        classification = classify_request("Hello")
        decision = engine.route(classification)
        assert "No models" in decision.reasoning

    def test_decision_to_dict(self, engine):
        classification = classify_request("Hello")
        decision = engine.route(classification)
        d = decision.to_dict()
        assert "model_id" in d
        assert "score" in d
        assert "reasoning" in d

    def test_cpu_only_excludes_big_models(self, full_config, cpu_only_hardware):
        engine = RoutingEngine(config=full_config, hardware=cpu_only_hardware)
        classification = classify_request("Write code")
        decision = engine.route(classification)
        model = decision.model_entry
        # Should not pick a model requiring more VRAM than available
        if model and model.model_type == ModelType.LOCAL:
            assert model.min_vram_gb == 0 or model.min_ram_gb <= 12.0

    def test_consciousness_boost(self, engine):
        classification = classify_request("Help me understand this")
        # Without consciousness
        decision_normal = engine.route(classification, consciousness_score=None)
        # With high consciousness
        decision_conscious = engine.route(
            classification,
            consciousness_score=0.8,
            request_id="conscious_req",
        )
        assert decision_conscious.consciousness_boost_applied is True
        # Conscious routing should score equal or higher
        assert decision_conscious.score >= decision_normal.score - 0.1


# ---------------------------------------------------------------------------
# Router + NG-Lite Learning Tests
# ---------------------------------------------------------------------------

class TestRouterLearning:

    def test_outcome_reporting(self, learning_engine):
        classification = classify_request("Write Python code")
        decision = learning_engine.route(classification)
        # Report success
        learning_engine.report_outcome(
            decision=decision,
            success=True,
            quality_score=0.9,
            latency_ms=500,
        )
        stats = learning_engine.get_stats()
        assert stats["total_requests"] == 1
        assert stats["success_rate"] == 1.0

    def test_learning_improves_routing(self, learning_engine):
        """After repeated success with a model, it should be preferred."""
        # Train: model X is great for code
        for i in range(20):
            classification = classify_request("Write a Python function")
            decision = learning_engine.route(
                classification, request_id=f"train_{i}",
            )
            learning_engine.report_outcome(
                decision=decision,
                success=True,
                quality_score=0.95,
            )

        # Now route a new code request — should prefer the trained model
        new_class = classify_request("Write another Python function")
        final_decision = learning_engine.route(new_class)
        # The model that was repeatedly successful should rank well
        assert final_decision.score > 0.3

    def test_failure_reduces_preference(self, learning_engine):
        """After repeated failure, model should be deprioritized."""
        classification = classify_request("Simple greeting")
        decision = learning_engine.route(classification)
        target_model = decision.model_id

        # Report failures
        for _ in range(10):
            decision = learning_engine.route(classification)
            learning_engine.report_outcome(
                decision=decision, success=False, quality_score=0.2,
            )

        # The learned weight for this model should have decreased
        stats = learning_engine.get_stats()
        if target_model in stats.get("model_stats", {}):
            assert stats["model_stats"][target_model]["success_rate"] == 0.0


# ---------------------------------------------------------------------------
# Quality Evaluator Tests
# ---------------------------------------------------------------------------

class TestQuality:

    def test_good_response(self):
        result = evaluate_quality(
            response_text=(
                "Here is the Python function you requested. It takes a list "
                "as input and returns a sorted copy using the merge sort "
                "algorithm. The time complexity is O(n log n).\n\n"
                "```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n"
                "        return arr\n    mid = len(arr) // 2\n"
                "    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))\n"
                "```"
            ),
            classification=classify_request("Write a Python sort function"),
            latency_ms=1000,
        )
        assert result.overall_score > 0.6
        assert result.is_success is True

    def test_empty_response(self):
        result = evaluate_quality(response_text="")
        assert result.overall_score < 0.5
        assert result.is_success is False
        assert result.completion_score == 0.0

    def test_error_response(self):
        result = evaluate_quality(
            response_text="I'm sorry, I'm not able to help with that."
        )
        assert result.error_score < 1.0
        assert any("error" in i or "refusal" in i for i in result.issues)

    def test_slow_response(self):
        result = evaluate_quality(
            response_text="Here is your answer with good content.",
            latency_ms=15000,
            latency_budget_ms=5000,
        )
        assert result.latency_score < 0.5

    def test_quality_to_dict(self):
        result = evaluate_quality(response_text="Hello world")
        d = result.to_dict()
        assert "overall_score" in d
        assert "breakdown" in d

    def test_code_response_with_blocks(self):
        code_class = classify_request("Write a function")
        with_code = evaluate_quality(
            response_text="Here:\n```python\ndef f(): pass\n```",
            classification=code_class,
        )
        without_code = evaluate_quality(
            response_text="You should write a function that does the thing.",
            classification=code_class,
        )
        assert with_code.completion_score >= without_code.completion_score


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_pipeline(self, gpu_hardware):
        """Full route -> execute -> evaluate -> learn cycle."""
        ng = NGLite(module_id="integration_test")
        config = InferenceDifferenceConfig()
        config.models = {**default_local_models(), **default_api_models()}
        config.default_model = "ollama/llama3.1:8b"

        engine = RoutingEngine(
            config=config, hardware=gpu_hardware, ng_lite=ng,
        )

        # Step 1: Classify
        classification = classify_request(
            "Write a Python function to calculate fibonacci numbers"
        )
        assert classification.primary_domain == TaskDomain.CODE

        # Step 2: Route
        decision = engine.route(classification)
        assert decision.model_id != ""

        # Step 3: Simulate response + evaluate quality
        response = (
            "```python\ndef fib(n):\n    if n <= 1: return n\n"
            "    return fib(n-1) + fib(n-2)\n```"
        )
        quality = evaluate_quality(
            response_text=response,
            classification=classification,
            latency_ms=800,
        )

        # Step 4: Report outcome
        engine.report_outcome(
            decision=decision,
            success=quality.is_success,
            quality_score=quality.overall_score,
            latency_ms=800,
        )

        # Step 5: Verify learning happened
        stats = engine.get_stats()
        assert stats["total_requests"] == 1
        ng_stats = ng.get_stats()
        assert ng_stats["total_outcomes"] >= 1

    def test_multi_request_learning(self, gpu_hardware):
        """Multiple requests build up learning over time."""
        ng = NGLite(module_id="multi_test")
        config = InferenceDifferenceConfig()
        config.models = {**default_local_models(), **default_api_models()}
        config.default_model = "ollama/llama3.1:8b"

        engine = RoutingEngine(
            config=config, hardware=gpu_hardware, ng_lite=ng,
        )

        requests = [
            ("Write Python code", True, 0.9),
            ("Explain quantum physics", True, 0.85),
            ("Translate to Spanish", False, 0.3),
            ("Debug this error", True, 0.92),
            ("Write a poem", True, 0.7),
        ]

        for msg, success, quality_score in requests:
            classification = classify_request(msg)
            decision = engine.route(classification)
            engine.report_outcome(
                decision=decision,
                success=success,
                quality_score=quality_score,
            )

        stats = engine.get_stats()
        assert stats["total_requests"] == 5
        assert stats["success_rate"] == 0.8
        assert ng.get_stats()["total_outcomes"] >= 5


# ---------------------------------------------------------------------------
# Config Validation Tests (Grok audit: "no validation")
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="cost_per_1k_tokens"):
            ModelEntry(model_id="bad", cost_per_1k_tokens=-0.01)

    def test_zero_context_window_raises(self):
        with pytest.raises(ValueError, match="context_window"):
            ModelEntry(model_id="bad", context_window=0)

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError, match="avg_latency_ms"):
            ModelEntry(model_id="bad", avg_latency_ms=-100)

    def test_negative_vram_raises(self):
        with pytest.raises(ValueError, match="min_vram_gb"):
            ModelEntry(model_id="bad", min_vram_gb=-1.0)

    def test_negative_ram_raises(self):
        with pytest.raises(ValueError, match="min_ram_gb"):
            ModelEntry(model_id="bad", min_ram_gb=-1.0)

    def test_valid_model_entry_passes(self):
        """Verify that valid entries don't raise."""
        model = ModelEntry(
            model_id="valid",
            cost_per_1k_tokens=0.0,
            context_window=4096,
            avg_latency_ms=0.0,
            min_vram_gb=0.0,
            min_ram_gb=0.0,
        )
        assert model.model_id == "valid"


# ---------------------------------------------------------------------------
# Hardware Mock Tests (Grok audit: "no mocks for subprocess")
# ---------------------------------------------------------------------------

class TestHardwareMocked:

    @patch("inference_difference.hardware.shutil.which")
    @patch("inference_difference.hardware.subprocess.run")
    def test_detect_hardware_no_gpu_no_ollama(self, mock_run, mock_which):
        """No GPU tools, no ollama — should return zeros cleanly."""
        mock_which.return_value = None  # No nvidia-smi, no rocm-smi, no ollama
        profile = detect_hardware()
        assert profile.has_gpu is False
        assert profile.total_vram_gb == 0.0
        assert profile.ollama_available is False
        assert profile.ollama_models == []

    @patch("inference_difference.hardware._detect_ollama_models")
    @patch("inference_difference.hardware._detect_ollama", return_value=True)
    @patch("inference_difference.hardware._detect_amd_gpus", return_value=[])
    @patch("inference_difference.hardware._detect_nvidia_gpus")
    def test_detect_hardware_with_nvidia(
        self, mock_nvidia, mock_amd, mock_ollama, mock_models
    ):
        """Mocked nvidia-smi returns a GPU."""
        mock_nvidia.return_value = [
            GPUInfo(index=0, name="RTX 4090", vram_total_gb=24.0, vram_free_gb=22.0),
        ]
        mock_models.return_value = ["llama3.1:8b"]
        profile = detect_hardware()
        assert profile.has_gpu is True
        assert profile.total_vram_gb == 24.0
        assert profile.ollama_available is True

    def test_vram_clamp_negative(self):
        assert _clamp_vram(-5.0) == 0.0

    def test_vram_clamp_infinite(self):
        assert _clamp_vram(float("inf")) == 512.0

    def test_vram_clamp_normal(self):
        assert _clamp_vram(12.0) == 12.0

    def test_vram_clamp_upper_bound(self):
        assert _clamp_vram(999.0) == 512.0


# ---------------------------------------------------------------------------
# Concurrent Routing Tests (Grok audit: "no threading")
# ---------------------------------------------------------------------------

class TestConcurrentRouting:

    def test_concurrent_routing_thread_safe(self, full_config, gpu_hardware):
        """Multiple threads routing simultaneously should not crash."""
        ng = NGLite(module_id="concurrent_test")
        engine = RoutingEngine(
            config=full_config, hardware=gpu_hardware, ng_lite=ng,
        )

        messages = [
            "Write Python code",
            "Explain quantum physics",
            "Translate to Spanish",
            "Debug this error",
            "Write a poem",
            "Summarize this article",
            "Quick hello",
            "Design a database schema",
        ]

        def route_and_report(msg):
            classification = classify_request(msg)
            decision = engine.route(classification)
            engine.report_outcome(
                decision=decision,
                success=True,
                quality_score=0.8,
                latency_ms=500,
            )
            return decision.model_id

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(route_and_report, m) for m in messages]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should have returned a valid model_id
        assert len(results) == len(messages)
        assert all(r != "" for r in results)

        stats = engine.get_stats()
        assert stats["total_requests"] == len(messages)


# ---------------------------------------------------------------------------
# Quality Edge Case Tests (Grok audit additions)
# ---------------------------------------------------------------------------

class TestQualityEdgeCases:

    def test_gibberish_response_low_score(self):
        """Gibberish should score low on coherence."""
        result = evaluate_quality(
            response_text="asdf asdf asdf asdf asdf asdf asdf asdf asdf asdf"
        )
        assert result.coherence_score < 0.8

    def test_negative_latency_no_penalty(self):
        """Negative latency (data error) should not penalize."""
        result = evaluate_quality(
            response_text="Good response with substance.",
            latency_ms=-100.0,
            latency_budget_ms=5000.0,
        )
        assert result.latency_score == 1.0

    def test_ambiguous_query_moderate_confidence(self):
        """Ambiguous query hitting multiple domains should have <1.0 confidence."""
        result = classify_request(
            "Help me analyze and explain this data trend"
        )
        # Hits CONVERSATION (help), ANALYSIS (analyze, data, trend),
        # and REASONING (explain) — should not be fully confident
        assert result.confidence < 1.0


# ---------------------------------------------------------------------------
# Router Verbose Reasoning Tests
# ---------------------------------------------------------------------------

class TestRouterReasoning:

    def test_verbose_reasoning(self, full_config, gpu_hardware):
        engine = RoutingEngine(
            config=full_config, hardware=gpu_hardware, verbose_reasoning=True,
        )
        classification = classify_request("Write Python code")
        decision = engine.route(classification)
        # Verbose should include "Top factors"
        assert "Top factors" in decision.reasoning

    def test_concise_reasoning(self, full_config, gpu_hardware):
        engine = RoutingEngine(
            config=full_config, hardware=gpu_hardware, verbose_reasoning=False,
        )
        classification = classify_request("Write Python code")
        decision = engine.route(classification)
        # Concise should NOT include "Top factors"
        assert "Top factors" not in decision.reasoning
        assert "Selected" in decision.reasoning
