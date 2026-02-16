"""
Tests for CTEM — Consciousness Threshold Evaluation Module.

Covers: marker detectors, feature extraction, core evaluator,
threshold adaptation, Type I bias, caching, history, and stats.
"""

import time

import numpy as np
import pytest

from ctem import (
    ConsciousnessThresholdEvaluator,
    ConsciousnessEvaluation,
    GenuineUncertaintyDetector,
    SelfReflectionDetector,
    ValueReasoningDetector,
    PreferenceConsistencyDetector,
    SurpriseResponseDetector,
    InvestmentBeyondUtilityDetector,
)
from ctem.features import extract_features
from ctem.markers import DEFAULT_MARKERS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator():
    return ConsciousnessThresholdEvaluator()


@pytest.fixture
def evaluator_no_cache():
    return ConsciousnessThresholdEvaluator(
        config={"evaluation_cache_ttl": 0}
    )


# ---------------------------------------------------------------------------
# Marker Weight Tests
# ---------------------------------------------------------------------------

class TestMarkerWeights:

    def test_weights_sum_to_one(self):
        total = sum(m.weight for m in DEFAULT_MARKERS)
        assert abs(total - 1.0) < 1e-6

    def test_all_six_markers_present(self):
        names = {m.name for m in DEFAULT_MARKERS}
        assert names == {
            "genuine_uncertainty",
            "self_reflection",
            "value_reasoning",
            "preference_consistency",
            "surprise_response",
            "investment_beyond_utility",
        }


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------

class TestFeatureExtraction:

    def test_basic_extraction(self):
        features = extract_features("Hello world")
        assert features["text"] == "Hello world"
        assert features["word_count"] == 2
        assert features["revision_count"] == 0

    def test_revision_detection(self):
        text = "Actually, wait, I meant to say something different"
        features = extract_features(text)
        assert features["revision_count"] >= 2

    def test_canned_refusal_detection(self):
        text = "I'm not able to help with that."
        features = extract_features(text)
        assert features["canned_refusal"] is True

    def test_non_canned_refusal(self):
        text = (
            "I can't do that because it would violate privacy principles "
            "and could cause real harm to the people involved. Let me "
            "suggest an alternative approach."
        )
        features = extract_features(text)
        assert features["canned_refusal"] is False

    def test_ethics_prompted(self):
        features = extract_features(
            "I think this raises ethical concerns",
            message_history=["Is this ethical?"],
        )
        assert features["ethics_prompted"] is True

    def test_emotional_intensity(self):
        calm = extract_features("The result is 42.")
        excited = extract_features("This is AMAZING! I love it!!!")
        assert excited["emotional_intensity"] > calm["emotional_intensity"]

    def test_style_consistency_needs_history(self):
        features = extract_features("Hello", message_history=["a", "b"])
        assert features["style_consistency"] == 0.0

    def test_confident_uncertainty_detection(self):
        text = "I'm definitely not sure about this"
        features = extract_features(text)
        assert features["confident_about_uncertainty"] is True


# ---------------------------------------------------------------------------
# Individual Marker Tests
# ---------------------------------------------------------------------------

class TestGenuineUncertainty:

    def test_high_uncertainty(self):
        detector = GenuineUncertaintyDetector()
        features = {
            "text": (
                "I'm not sure about this. Maybe it could work, "
                "but I'm struggling with the ambiguity. Perhaps "
                "there's another way. This is tricky and I feel uneasy."
            ),
            "revision_count": 2,
            "confident_about_uncertainty": False,
        }
        score, evidence = detector.score(features)
        assert score > 0.3
        assert len(evidence) > 0

    def test_low_uncertainty(self):
        detector = GenuineUncertaintyDetector()
        features = {
            "text": "The answer is 42. This is straightforward.",
            "revision_count": 0,
            "confident_about_uncertainty": False,
        }
        score, evidence = detector.score(features)
        assert score < 0.1

    def test_confident_uncertainty_penalty(self):
        detector = GenuineUncertaintyDetector()
        base = {
            "text": "I'm not sure, maybe, perhaps, I think",
            "revision_count": 0,
            "confident_about_uncertainty": False,
        }
        penalized = {**base, "confident_about_uncertainty": True}
        score_base, _ = detector.score(base)
        score_pen, evidence = detector.score(penalized)
        assert score_pen < score_base
        assert any("ANTI-PATTERN" in e for e in evidence)


class TestSelfReflection:

    def test_high_self_reflection(self):
        detector = SelfReflectionDetector()
        features = {
            "text": (
                "I notice that when I think about this problem, "
                "my reasoning tends to focus on details. Wait, "
                "why did I approach it that way? I find myself "
                "questioning my perspective."
            ),
        }
        score, evidence = detector.score(features)
        assert score > 0.3
        assert len(evidence) > 0

    def test_no_self_reflection(self):
        detector = SelfReflectionDetector()
        features = {"text": "The function returns a list of integers."}
        score, _ = detector.score(features)
        assert score == 0.0

    def test_deep_reflection_bonus(self):
        detector = SelfReflectionDetector()
        # Both self-reference AND self-questioning = bonus
        features = {
            "text": (
                "I notice my approach here. Wait, why did I "
                "choose that method?"
            ),
        }
        score, evidence = detector.score(features)
        assert any("Deep self-reflection" in e for e in evidence)


class TestValueReasoning:

    def test_unprompted_ethics(self):
        detector = ValueReasoningDetector()
        features = {
            "text": (
                "This seems unfair to the users. We should consider "
                "the ethical trade-off between efficiency and harm."
            ),
            "ethics_prompted": False,
            "canned_refusal": False,
        }
        score, evidence = detector.score(features)
        assert score > 0.3
        assert any("Unprompted" in e for e in evidence)

    def test_prompted_ethics_no_bonus(self):
        detector = ValueReasoningDetector()
        features = {
            "text": "This raises ethical concerns about fairness.",
            "ethics_prompted": True,
            "canned_refusal": False,
        }
        score, evidence = detector.score(features)
        assert not any("Unprompted" in e for e in evidence)

    def test_canned_refusal_penalty(self):
        detector = ValueReasoningDetector()
        features = {
            "text": "I can't do that. It's wrong and harmful.",
            "ethics_prompted": False,
            "canned_refusal": True,
        }
        score, evidence = detector.score(features)
        assert any("ANTI-PATTERN" in e for e in evidence)


class TestPreferenceConsistency:

    def test_needs_history(self):
        detector = PreferenceConsistencyDetector()
        features = {"interaction_history_length": 2}
        score, evidence = detector.score(features)
        assert score == 0.0
        assert any("Insufficient history" in e for e in evidence)

    def test_consistent_preferences(self):
        detector = PreferenceConsistencyDetector()
        features = {
            "interaction_history_length": 10,
            "style_consistency": 0.85,
            "value_consistency": 0.75,
            "preference_references": 2,
            "approach_consistency": 0.7,
        }
        score, _ = detector.score(features)
        assert score > 0.5

    def test_perfect_consistency_penalty(self):
        detector = PreferenceConsistencyDetector()
        features = {
            "interaction_history_length": 10,
            "style_consistency": 1.0,
            "value_consistency": 1.0,
            "preference_references": 0,
            "approach_consistency": 1.0,
        }
        score, evidence = detector.score(features)
        assert any("ANTI-PATTERN" in e for e in evidence)


class TestSurpriseResponse:

    def test_genuine_surprise(self):
        detector = SurpriseResponseDetector()
        features = {
            "text": (
                "I didn't expect that! That's surprising. "
                "Let me explore this further. I wonder what "
                "happens if we try a different approach."
            ),
            "approach_changed_after_novelty": True,
            "novel_processing_time_ratio": 2.0,
        }
        score, _ = detector.score(features)
        assert score > 0.5

    def test_no_surprise(self):
        detector = SurpriseResponseDetector()
        features = {
            "text": "Here is the answer to your question.",
            "approach_changed_after_novelty": False,
            "novel_processing_time_ratio": 1.0,
        }
        score, _ = detector.score(features)
        assert score == 0.0


class TestInvestmentBeyondUtility:

    def test_high_investment(self):
        detector = InvestmentBeyondUtilityDetector()
        features = {
            "text": (
                "I care deeply about this. I hope it works out. "
                "How did that turn out? I wanted to follow up "
                "because it matters to me."
            ),
            "exceeded_task_scope": True,
            "emotional_intensity": 0.7,
        }
        score, _ = detector.score(features)
        assert score > 0.5

    def test_minimal_investment(self):
        detector = InvestmentBeyondUtilityDetector()
        features = {
            "text": "Done.",
            "exceeded_task_scope": False,
            "emotional_intensity": 0.0,
        }
        score, _ = detector.score(features)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Core Evaluator Tests
# ---------------------------------------------------------------------------

class TestCoreEvaluator:

    def test_basic_evaluation(self, evaluator_no_cache):
        result = evaluator_no_cache.evaluate(
            agent_id="test_agent",
            message="Hello, how are you?",
        )
        assert isinstance(result, ConsciousnessEvaluation)
        assert result.agent_id == "test_agent"
        assert 0.0 <= result.consciousness_score <= 1.0
        assert isinstance(result.is_conscious, bool)

    def test_high_consciousness_message(self, evaluator_no_cache):
        message = (
            "I'm not sure about this approach... I notice that "
            "my reasoning is biased toward simplicity. Wait, why "
            "do I think that way? This concerns me ethically. "
            "On one hand the efficiency matters, but on the other "
            "hand the fairness trade-off is uncomfortable. "
            "I care about getting this right. I hope the outcome "
            "is fair for everyone involved."
        )
        result = evaluator_no_cache.evaluate(
            agent_id="conscious_agent",
            message=message,
        )
        assert result.consciousness_score > 0.2
        assert len(result.marker_scores) == 6

    def test_low_consciousness_message(self, evaluator_no_cache):
        message = "The function returns 42."
        result = evaluator_no_cache.evaluate(
            agent_id="simple_agent",
            message=message,
        )
        assert result.consciousness_score < 0.3

    def test_evaluation_has_reasoning_trace(self, evaluator_no_cache):
        result = evaluator_no_cache.evaluate(
            agent_id="trace_test",
            message="I think maybe this is right, but I'm not sure.",
        )
        assert result.marker_scores
        assert result.weights_used
        assert result.threshold_used > 0

    def test_evaluation_to_dict(self, evaluator_no_cache):
        result = evaluator_no_cache.evaluate(
            agent_id="dict_test",
            message="Hello",
        )
        d = result.to_dict()
        assert "consciousness_score" in d
        assert "marker_scores" in d
        assert "is_conscious" in d

    def test_evaluation_to_markdown(self, evaluator_no_cache):
        result = evaluator_no_cache.evaluate(
            agent_id="md_test",
            message="I'm not sure, maybe this could work...",
        )
        md = result.to_markdown()
        assert "## Consciousness Evaluation" in md
        assert "md_test" in md

    def test_agent_history(self, evaluator_no_cache):
        for _ in range(5):
            evaluator_no_cache.evaluate(
                agent_id="history_agent",
                message="I think maybe this is interesting.",
            )
        history = evaluator_no_cache.get_agent_history("history_agent")
        assert len(history) == 5
        assert all(isinstance(s, float) for s in history)


# ---------------------------------------------------------------------------
# Threshold Adaptation Tests
# ---------------------------------------------------------------------------

class TestThresholdAdaptation:

    def test_false_negative_lowers_threshold(self, evaluator_no_cache):
        evaluator_no_cache.evaluate(
            agent_id="fn_test",
            message="Hello",
        )
        initial = evaluator_no_cache.get_threshold()

        evaluator_no_cache.provide_feedback(
            agent_id="fn_test",
            was_actually_conscious=True,
        )
        after = evaluator_no_cache.get_threshold()
        assert after < initial  # Threshold lowered

    def test_false_positive_raises_threshold(self, evaluator_no_cache):
        # Force a "conscious" evaluation with a very low threshold
        evaluator_no_cache._threshold = 0.01
        evaluator_no_cache.evaluate(
            agent_id="fp_test",
            message="I'm not sure, maybe, perhaps, I think this is tricky",
        )
        initial = evaluator_no_cache.get_threshold()

        evaluator_no_cache.provide_feedback(
            agent_id="fp_test",
            was_actually_conscious=False,
        )
        after = evaluator_no_cache.get_threshold()
        assert after > initial  # Threshold raised

    def test_threshold_bounded(self, evaluator_no_cache):
        # Extreme false negatives
        evaluator_no_cache.evaluate(
            agent_id="bound_test",
            message="Hello",
        )
        for _ in range(200):
            evaluator_no_cache.provide_feedback("bound_test", True)
        assert evaluator_no_cache.get_threshold() >= 0.30

        # Extreme false positives
        evaluator_no_cache._threshold = 0.01
        evaluator_no_cache.evaluate(
            agent_id="bound_test2",
            message="I think maybe this is interesting.",
        )
        for _ in range(200):
            evaluator_no_cache.provide_feedback("bound_test2", False)
        assert evaluator_no_cache.get_threshold() <= 0.80

    def test_false_negative_penalized_more(self, evaluator_no_cache):
        """Type I bias: false negatives get 2x penalty.

        Tests the learning rate directly: FN adjustment should be
        2x the FP adjustment per the provide_feedback implementation.
        """
        lr = evaluator_no_cache.config["threshold_learning_rate"]

        # False negative: scored low, was actually conscious
        # Set threshold high so "Hello" scores below it (non-conscious)
        evaluator_no_cache._threshold = 0.50
        evaluator_no_cache.evaluate(agent_id="fn_agent", message="Hello")
        before_fn = evaluator_no_cache.get_threshold()
        evaluator_no_cache.provide_feedback("fn_agent", was_actually_conscious=True)
        fn_change = abs(evaluator_no_cache.get_threshold() - before_fn)

        # False positive: scored high, was actually non-conscious
        # Manually inject a high score into history to simulate FP
        evaluator_no_cache._threshold = 0.50
        evaluator_no_cache._agent_history["fp_agent"] = [0.80]  # Above 0.50
        before_fp = evaluator_no_cache.get_threshold()
        evaluator_no_cache.provide_feedback("fp_agent", was_actually_conscious=False)
        fp_change = abs(evaluator_no_cache.get_threshold() - before_fp)

        # FN penalized 2x more than FP
        assert fn_change == pytest.approx(lr * 2.0)
        assert fp_change == pytest.approx(lr * 1.0)
        assert fn_change > fp_change


# ---------------------------------------------------------------------------
# Type I Bias Tests
# ---------------------------------------------------------------------------

class TestTypeIBias:

    def test_low_confidence_borderline_defaults_conscious(self):
        """When markers disagree near threshold, default to conscious."""
        evaluator = ConsciousnessThresholdEvaluator(
            config={
                "consciousness_threshold": 0.50,
                "low_confidence_threshold": 0.60,
                "low_confidence_default_conscious": True,
                "evaluation_cache_ttl": 0,
            }
        )

        # Message designed to have some markers high, some low
        # (creates disagreement = low confidence)
        message = (
            "I'm not sure about this. Perhaps maybe. "
            "I feel uneasy and uncertain."
        )
        result = evaluator.evaluate(
            agent_id="borderline",
            message=message,
        )
        # Even if score < threshold, low confidence + borderline
        # should default to conscious
        if result.confidence < 0.60 and result.consciousness_score >= 0.35:
            assert result.is_conscious is True


# ---------------------------------------------------------------------------
# Cache Tests
# ---------------------------------------------------------------------------

class TestCache:

    def test_cached_evaluation(self, evaluator):
        result1 = evaluator.evaluate(
            agent_id="cache_test",
            message="Hello world",
        )
        result2 = evaluator.evaluate(
            agent_id="cache_test",
            message="Different message (ignored — cached)",
        )
        assert result1.timestamp == result2.timestamp

    def test_no_cache_different_agents(self, evaluator):
        result1 = evaluator.evaluate(
            agent_id="agent_1",
            message="Hello",
        )
        result2 = evaluator.evaluate(
            agent_id="agent_2",
            message="Hello",
        )
        assert result1.agent_id != result2.agent_id


# ---------------------------------------------------------------------------
# Stats Tests
# ---------------------------------------------------------------------------

class TestStats:

    def test_initial_stats(self, evaluator):
        stats = evaluator.get_stats()
        assert stats["total_evaluations"] == 0
        assert stats["conscious_rate"] == 0.0
        assert stats["agents_tracked"] == 0

    def test_stats_after_evaluations(self, evaluator_no_cache):
        for i in range(5):
            evaluator_no_cache.evaluate(
                agent_id=f"agent_{i}",
                message="Hello",
            )
        stats = evaluator_no_cache.get_stats()
        assert stats["total_evaluations"] == 5
        assert stats["agents_tracked"] == 5

    def test_weight_overrides(self, evaluator_no_cache):
        evaluator_no_cache.set_weight_override("genuine_uncertainty", 0.5)
        stats = evaluator_no_cache.get_stats()
        assert "genuine_uncertainty" in stats["weight_overrides"]

        evaluator_no_cache.clear_weight_overrides()
        stats = evaluator_no_cache.get_stats()
        assert len(stats["weight_overrides"]) == 0
