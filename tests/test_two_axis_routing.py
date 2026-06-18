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
