from inference_difference.config import InferenceDifferenceConfig
from inference_difference.router import RoutingEngine
from inference_difference.hardware import HardwareProfile


def _engine():
    hw = HardwareProfile(cpu_count=4, cpu_name="Test CPU", ram_total_gb=16.0,
                         ram_available_gb=12.0, has_gpu=False, os_name="Linux",
                         platform_arch="x86_64")
    return RoutingEngine(config=InferenceDifferenceConfig(), hardware=hw)


def test_tool_competence_config_defaults():
    c = InferenceDifferenceConfig()
    assert c.tool_competence_gain == 0.05      # slow up (Elmer TuningSocket gain)
    assert c.tool_competence_loss == 0.10      # fast down (2:1 asymmetry)
    assert c.tool_competence_prior_capable == 0.6      # flagged tool-capable, unproven
    assert c.tool_competence_prior_unflagged == 0.15   # unflagged prior
    assert c.tool_withhold_floor == 0.15       # below this => strip tools
    assert c.tool_competence_weight == 0.10    # secondary; below domain(0.25)/complexity(0.20)


def test_tool_failure_classifier():
    e = _engine()
    md = {"tools_requested": True, "error": "404 No endpoints found that support tool use"}
    assert e._is_tool_failure(md) is True
    assert e._tool_failure_kind(md) == "structural"
    md2 = {"tools_requested": True, "error": "400 invalid tool_calls: malformed arguments"}
    assert e._is_tool_failure(md2) is True
    assert e._tool_failure_kind(md2) == "unreliable"
    assert e._is_tool_failure({"tools_requested": False, "error": "tool use unsupported"}) is False
    assert e._is_tool_failure({"tools_requested": True, "error": "502 upstream timeout"}) is False
    assert e._is_tool_failure(None) is False


class _FakeModel:
    def __init__(self, model_id, caps):
        self.model_id = model_id
        self.capabilities = caps


def test_tool_competence_bootstrap_and_update():
    e = _engine()
    capable = _FakeModel("m_cap", ["tools", "roleplay"])
    unflagged = _FakeModel("m_unflagged", ["roleplay"])
    # bootstrap priors from the catalog 'tools' flag
    assert e._get_tool_competence("m_cap", capable) == 0.6
    assert e._get_tool_competence("m_unflagged", unflagged) == 0.15
    # success gains slowly; failure loses fast (asymmetry). PASS model_entry so the
    # 0.6 prior is the starting point (without it, an unseen model starts at 0.5).
    e._update_tool_competence("m_cap", success=True, model_entry=capable)
    assert abs(e._get_tool_competence("m_cap", capable) - 0.65) < 1e-9     # +gain 0.05
    e._update_tool_competence("m_cap", success=False, structural=False, model_entry=capable)
    assert abs(e._get_tool_competence("m_cap", capable) - 0.55) < 1e-9     # -loss 0.10
    # clamps to [0,1]
    for _ in range(50):
        e._update_tool_competence("m_cap", success=True, model_entry=capable)
    assert e._get_tool_competence("m_cap", capable) == 1.0
    # structural failure FLOORS to 0 immediately (definitive "can't")
    e._update_tool_competence("m_cap", success=False, structural=True, model_entry=capable)
    assert e._get_tool_competence("m_cap", capable) == 0.0


def _decision(model_id):
    from inference_difference.router import RoutingDecision
    return RoutingDecision(model_id=model_id)


def test_tool_failure_does_not_touch_her_fit():
    e = _engine()
    before = e._model_penalty.level("m_cap")
    e.report_outcome(decision=_decision("m_cap"), success=False, quality_score=0.0,
                     metadata={"tools_requested": True,
                               "error": "404 No endpoints found that support tool use"})
    assert e._model_penalty.level("m_cap") == before          # her-fit penalty untouched
    assert e._model_success_stats.get("m_cap", []) == []      # no her-fit sample recorded
    assert e._get_tool_competence("m_cap", _FakeModel("m_cap", ["tools"])) == 0.0  # tool axis floored


def test_model_failure_still_climbs_penalty():
    e = _engine()
    e.report_outcome(decision=_decision("m_bad"), success=False, quality_score=0.0,
                     metadata={"tools_requested": False, "error": "500 garbage output"})
    assert e._model_penalty.level("m_bad") == 1               # climbs, as today


def test_provider_failure_touches_neither_axis():
    e = _engine()
    e.report_outcome(decision=_decision("m_x"), success=False, quality_score=0.0,
                     metadata={"provider_blocked": True, "error": "402 insufficient credit"})
    assert e._model_penalty.level("m_x") == 0
    assert "m_x" not in e._tool_competence


import types


def test_tool_competence_weight_is_subordinate_to_her_fit():
    e = _engine()
    # weight injected, and STRICTLY below the her-fit weights => her-fit dominates
    assert e._weights["tool_competence"] == 0.10
    assert e._weights["tool_competence"] < e._weights["domain_match"]
    assert e._weights["tool_competence"] < e._weights["complexity_fit"]


def test_score_model_tool_competence_breakdown_and_dominance(monkeypatch):
    e = _engine()
    # stub the her-fit sub-scores so we control them deterministically
    def fake_sub(model, classification):
        return 0.9 if model.model_id == "good_self" else 0.2
    monkeypatch.setattr(e, "_score_domain", fake_sub)
    monkeypatch.setattr(e, "_score_complexity", fake_sub)
    monkeypatch.setattr(e, "_score_learned", lambda m, c: 0.5)
    monkeypatch.setattr(e, "_score_cost", lambda m, success_rate=1.0: 0.5)
    monkeypatch.setattr(e, "_score_latency", lambda m, c: 0.5)

    def mk(mid, tc):
        e._tool_competence[mid] = tc
        # SimpleNamespace with the attrs _score_model reads; ADD MORE if _score_model's
        # tail raises AttributeError (e.g. provider="", model_type, provider_tier="") —
        # the POINT of this test is the two assertions below, so make it run.
        return types.SimpleNamespace(model_id=mid, priority=50,
                                     conversational_quality=0.5, capabilities=["tools"],
                                     provider="", provider_tier="")
    good = mk("good_self", 0.0)   # great self, FLOORED tools
    weak = mk("weak_self", 1.0)   # weak self, perfect tools

    s_good, bd_good = e._score_model(good, classification=None, has_tools=True)
    s_weak, _bd_weak = e._score_model(weak, classification=None, has_tools=True)
    assert "tool_competence" in bd_good           # breakdown key present on tool turns
    assert s_good > s_weak                         # her-fit DOMINATES tool_competence
    # conversation turn: tool_competence is NOT scored
    _s, bd_conv = e._score_model(good, classification=None, has_tools=False)
    assert "tool_competence" not in bd_conv


def test_tool_competence_persists_round_trip(tmp_path):
    e = _engine()
    e._tool_competence = {"m_cap": 0.0, "m_good": 0.85}
    e._stats_cache_path = str(tmp_path / "stats.msgpack")
    e._stats_dirty = True
    e.save_stats()
    e2 = _engine()
    e2._stats_cache_path = e._stats_cache_path
    e2.load_stats()
    assert e2._get_tool_competence("m_cap", _FakeModel("m_cap", ["tools"])) == 0.0
    assert e2._get_tool_competence("m_good", _FakeModel("m_good", ["tools"])) == 0.85


def test_should_withhold_tools():
    e = _engine()
    e._tool_competence = {"floored": 0.0, "fine": 0.5}
    assert e.should_withhold_tools("floored") is True     # below floor 0.15
    assert e.should_withhold_tools("fine") is False
    assert e.should_withhold_tools("unknown") is False    # no entry -> 0.5 default -> keep tools
