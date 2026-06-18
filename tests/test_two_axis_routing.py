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
