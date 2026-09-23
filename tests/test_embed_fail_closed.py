"""
Fail-closed behavior tests for ng_embed callers (LAW 4/7).

Canonical ng_embed raises EmbeddingUnavailableError when the embedding
model cannot load — it never returns a fabricated vector. These tests
verify every TID caller handles that failure without writing forged
data to the substrate and without silently swallowing the failure.

# ---- Changelog ----
# [2026-09-23] Claude Code — add fail-closed caller tests
#   What: Tests for classifier._semantic_embed / classify_request propagation,
#     ShimObserver.observe/query_confidence, and DreamCycle._teach_substrate
#     handling of EmbeddingUnavailableError.
#   Why:  ng_embed revendor removed the SHA hash fallback (LAW 2/4/7); caller
#     behavior on model-unavailable had zero test coverage.
#   How:  Force the real singleton unavailable via _model_failed flag (no
#     vendored-file modification), or patch ng_embed.embed; assert zero
#     record_outcome calls and correct propagation/neutral defaults.
# -------------------
"""

from unittest.mock import MagicMock, patch

import pytest

import ng_embed
from ng_embed import EmbeddingUnavailableError
from inference_difference.classifier import _semantic_embed, classify_request
from inference_difference.dream_cycle import DreamCycle, PropertyInsight
from inference_difference.translation_shim import ShimObserver


@pytest.fixture(autouse=True)
def _reset_embed_singleton():
    """Isolate the NGEmbed singleton across tests."""
    ng_embed.NGEmbed.reset_instance()
    yield
    ng_embed.NGEmbed.reset_instance()


def _force_model_failed() -> None:
    """Force the real singleton into the model-unavailable state."""
    inst = ng_embed.NGEmbed.get_instance()
    inst._model_failed = True


class TestClassifierFailClosed:
    """classifier.py must propagate embed failure — no forged vectors."""

    def test_semantic_embed_raises_when_model_unavailable(self):
        _force_model_failed()
        with pytest.raises(EmbeddingUnavailableError):
            _semantic_embed("hello world")

    def test_classify_request_propagates_no_hash_fallback(self):
        _force_model_failed()
        with pytest.raises(EmbeddingUnavailableError):
            classify_request("Write a Python function")

    def test_semantic_embed_no_exception_is_real_vector(self):
        # Sanity: with the model available, embed returns a real 768-dim vector.
        vec = _semantic_embed("hello world")
        assert vec.shape == (768,)


class TestShimObserverFailClosed:
    """translation_shim.py ShimObserver — zero graph writes on embed failure."""

    def _make_observer(self):
        ng = MagicMock()
        return ShimObserver(ng_ecosystem=ng), ng

    def test_observe_no_record_outcome_on_embed_failure(self):
        obs, ng = self._make_observer()
        with patch.object(ng_embed, "embed", side_effect=EmbeddingUnavailableError("x")):
            # Must not raise, must not write
            obs.observe(model_id="m", operation="alias_resolve", did_apply=True)
        ng.record_outcome.assert_not_called()

    def test_query_confidence_returns_neutral_on_embed_failure(self):
        obs, ng = self._make_observer()
        with patch.object(ng_embed, "embed", side_effect=EmbeddingUnavailableError("x")):
            result = obs.query_confidence(model_id="m", operation="alias_resolve")
        assert result == obs._neutral
        ng.get_recommendations.assert_not_called()


class TestDreamCycleFailClosed:
    """dream_cycle.py _teach_substrate — zero graph writes on embed failure."""

    def test_teach_substrate_no_record_outcome_on_embed_failure(self):
        ng = MagicMock()
        # embed_fn is bound unwrapped exactly as app.py wires it:
        # `from ng_embed import embed as _ng_embed; _dc_embed_fn = _ng_embed`
        # (inference_difference/app.py:908-909). No try/except wrapper, so a
        # raised failure can never reach record_outcome as a None embedding.
        cycle = DreamCycle(ng_ecosystem=ng, embed_fn=ng_embed.embed)
        insights = {
            "coding": [
                PropertyInsight(
                    property_name="context_window",
                    observation="insight text",
                    recommendation="more context",
                    confidence=0.9,
                    sample_size=20,
                )
            ]
        }
        # embed_fn is bound at construction (mirrors app.py), so patch the
        # class method — the bound module-level function routes through it.
        with patch.object(ng_embed.NGEmbed, "embed", side_effect=EmbeddingUnavailableError("x")):
            # Must not raise, must not write
            cycle._teach_substrate(insights)
        ng.record_outcome.assert_not_called()
        assert cycle._substrate_teach_count == 0


class TestAppFailClosed:
    """app.py — 503 handler and _substrate_tier_mapping fail-closed behavior.

    Skipped in environments without fastapi (pre-existing gap in the system
    interpreter); runs anywhere the app is importable.
    """

    def test_substrate_tier_mapping_defaults_on_embed_failure(self):
        pytest.importorskip("fastapi")
        from unittest.mock import MagicMock as _M
        from inference_difference import app as tid_app
        from inference_difference.config import InferenceDifferenceConfig

        tid_app._state.config = InferenceDifferenceConfig()
        tid_app._state.ng_ecosystem = _M()
        with patch.object(
            ng_embed.NGEmbed, "embed", side_effect=EmbeddingUnavailableError("x"),
        ):
            complexity, priority = tid_app._substrate_tier_mapping("performance")
        # Bootstrap defaults returned, no exception, no graph write.
        assert priority == tid_app._state.config.tier_priority_performance
        tid_app._state.ng_ecosystem.get_recommendations.assert_not_called()

    def test_classify_endpoint_returns_503_when_embedding_unavailable(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient
        from inference_difference import app as tid_app

        client = TestClient(tid_app.app, raise_server_exceptions=False)
        _force_model_failed()
        try:
            resp = client.post("/classify", json={"message": "Write a Python function"})
        finally:
            ng_embed.NGEmbed.reset_instance()
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()
