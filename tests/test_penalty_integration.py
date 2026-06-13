# ---- Changelog ----
# [2026-06-12] Claude Code (Opus 4.8, Anima/Codemine CC) — BLK-TID-PB-2 integration tests
# What: Verify the penalty box wired into RoutingEngine — TTL-EXEMPT persistence (law-enforcer
#   CRITICAL), report_outcome climb/descend with provider-402 exclusion, manual clear, and the
#   structural veto-only guardrail (penalty never appears in a scoring method).
# Why: prove the wiring, not just the unit logic. The TTL test is the true regression guard for the
#   recurring bug — a plain round-trip would pass while the >24h-discard bug shipped.
# How: real RoutingEngine on CPU-only hardware, sidecar redirected to tmp_path.
# -------------------
import time

import msgpack

from inference_difference.config import InferenceDifferenceConfig
from inference_difference.hardware import HardwareProfile
from inference_difference.router import RoutingEngine, RoutingDecision


def _engine(tmp_path):
    eng = RoutingEngine(
        config=InferenceDifferenceConfig(),
        hardware=HardwareProfile(has_gpu=False, available_vram_gb=0.0),
    )
    eng._stats_cache_path = str(tmp_path / "model_stats_cache.msgpack")
    return eng


def test_blacklist_survives_24h_ttl_discard_while_stats_drop(tmp_path):
    """Law-enforcer CRITICAL: penalty/blacklist is TTL-exempt; stale success-stats are discarded."""
    eng = _engine(tmp_path)
    for _ in range(7):
        eng._model_penalty.record_failure("dead/model")        # climb to L7 blacklist
    assert eng._model_penalty.is_blacklisted("dead/model")
    eng._model_success_stats["some/model"] = [True, False, True]
    eng._stats_dirty = True
    eng.save_stats()

    # Backdate the sidecar's saved_at to >24h ago (the discard path the round-trip never exercises).
    with open(eng._stats_cache_path, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)
    data["saved_at"] = time.time() - 90000
    with open(eng._stats_cache_path, "wb") as f:
        f.write(msgpack.packb(data))

    fresh = _engine(tmp_path)
    fresh._stats_cache_path = eng._stats_cache_path
    fresh.load_stats()
    assert fresh._model_success_stats == {}                     # stale stats DROPPED (correct)
    assert fresh._model_penalty.is_blacklisted("dead/model")    # blacklist SURVIVES (the fix)


def test_penalty_round_trips_through_real_sidecar(tmp_path):
    eng = _engine(tmp_path)
    eng._model_penalty.record_failure("a")                      # L1
    for _ in range(3):
        eng._model_penalty.record_failure("b")                  # L3
    eng._stats_dirty = True
    eng.save_stats()
    fresh = _engine(tmp_path)
    fresh._stats_cache_path = eng._stats_cache_path
    fresh.load_stats()
    assert fresh._model_penalty.level("a") == 1
    assert fresh._model_penalty.level("b") == 3


def test_report_outcome_climbs_on_model_failure_and_descends_on_success(tmp_path):
    eng = _engine(tmp_path)
    dec = RoutingDecision(model_id="m")
    eng.report_outcome(dec, success=False, metadata={"error": "400 the requested model does not exist"})
    assert eng._model_penalty.level("m") == 1
    eng.report_outcome(dec, success=True, metadata={})
    assert eng._model_penalty.level("m") == 0


def test_report_outcome_does_not_climb_on_provider_402(tmp_path):
    eng = _engine(tmp_path)
    dec = RoutingDecision(model_id="m")
    eng.report_outcome(dec, success=False, metadata={"error": "402 insufficient balance"})
    assert eng._model_penalty.level("m") == 0                   # provider-wide → no climb
    eng.report_outcome(dec, success=False, metadata={"provider_blocked": True})
    assert eng._model_penalty.level("m") == 0


def test_clear_penalty_resets(tmp_path):
    eng = _engine(tmp_path)
    for _ in range(7):
        eng._model_penalty.record_failure("m")
    assert eng._model_penalty.is_blacklisted("m")
    assert eng.clear_penalty("m") == 1
    assert eng._model_penalty.level("m") == 0


def test_veto_only_penalty_absent_from_every_scoring_method():
    """Law-enforcer Q2 guardrail (permanent): _model_penalty never appears in a _score_* method."""
    import inspect
    from inference_difference import router as r
    score_src = ""
    for name, fn in inspect.getmembers(r.RoutingEngine, predicate=inspect.isfunction):
        if name.startswith("_score"):
            score_src += inspect.getsource(fn)
    assert score_src, "no _score_* methods found — check the assertion target"
    assert "_model_penalty" not in score_src
