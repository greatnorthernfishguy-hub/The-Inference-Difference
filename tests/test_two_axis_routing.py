from inference_difference.config import InferenceDifferenceConfig


def test_tool_competence_config_defaults():
    c = InferenceDifferenceConfig()
    assert c.tool_competence_gain == 0.05      # slow up (Elmer TuningSocket gain)
    assert c.tool_competence_loss == 0.10      # fast down (2:1 asymmetry)
    assert c.tool_competence_prior_capable == 0.6      # flagged tool-capable, unproven
    assert c.tool_competence_prior_unflagged == 0.15   # unflagged prior
    assert c.tool_withhold_floor == 0.15       # below this => strip tools
    assert c.tool_competence_weight == 0.10    # secondary; below domain(0.25)/complexity(0.20)
