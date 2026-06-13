# ---- Changelog ----
# [2026-06-12] Claude Code (Opus 4.8, Anima/Codemine CC) — unit tests for ModelPenaltyBox (Layer B)
# What: Deterministic tests (injected clock, no real sleeps) for the per-model penalty box:
#   climb, expiry/eligibility, asymmetric descend, terminal blacklist, notify-once, persistence,
#   manual clear. The veto-only property is enforced structurally (no scoring method exists) and is
#   re-checked in the integration suite (router has no _model_penalty in any _score_* path).
# Why: the ladder is the make-or-break logic; it must be provably correct before TID integration.
# How: a tiny mutable Clock drives time_fn so durations advance deterministically.
# -------------------
import pytest

from inference_difference.model_penalty import ModelPenaltyBox


class Clock:
    """Mutable injectable clock."""
    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, secs):
        self.t += float(secs)


LADDER = (30, 300, 3600, 86400, 604800, 5184000)  # L1..L6; L7 = blacklist


def _box(clock=None, on_blacklist=None):
    return ModelPenaltyBox(ladder_seconds=LADDER, time_fn=clock or Clock(), on_blacklist=on_blacklist)


def test_clear_model_is_not_penalized():
    box = _box()
    assert box.level("m") == 0
    assert box.is_penalized("m") is False
    assert box.is_blacklisted("m") is False


def test_failure_climbs_to_L1_and_penalizes_until_expiry():
    clk = Clock()
    box = _box(clk)
    box.record_failure("m")
    assert box.level("m") == 1
    assert box.is_penalized("m") is True          # inside the 30s box
    clk.advance(29)
    assert box.is_penalized("m") is True
    clk.advance(2)                                # past 30s
    assert box.is_penalized("m") is False         # eligible again (re-probe)


def test_consecutive_failures_climb_one_rung_each_with_correct_durations():
    clk = Clock()
    box = _box(clk)
    box.record_failure("m"); assert box.level("m") == 1   # 30s
    clk.advance(31)
    box.record_failure("m"); assert box.level("m") == 2   # 5m
    clk.advance(301)
    box.record_failure("m"); assert box.level("m") == 3   # 1h
    # still penalized: only 0s elapsed into the 1h box
    assert box.is_penalized("m") is True


def test_success_descends_exactly_one_rung_not_reset():
    clk = Clock()
    box = _box(clk)
    for _ in range(4):                            # climb to L4
        box.record_failure("m"); clk.advance(10 ** 7)
    assert box.level("m") == 4
    box.record_success("m")
    assert box.level("m") == 3                    # down ONE, not to 0
    assert box.is_penalized("m") is False         # eligible after a success


def test_descend_to_zero_clears_entry():
    box = _box()
    box.record_failure("m")
    assert box.level("m") == 1
    box.record_success("m")
    assert box.level("m") == 0
    assert box.is_penalized("m") is False


def test_two_failures_then_one_success_nets_plus_one():
    clk = Clock()
    box = _box(clk)
    box.record_failure("m"); clk.advance(10 ** 7)
    box.record_failure("m"); clk.advance(10 ** 7)
    box.record_success("m")
    assert box.level("m") == 1


def test_climbs_to_blacklist_at_L7_and_is_terminal():
    clk = Clock()
    box = _box(clk)
    for _ in range(7):                            # 6 timed boxes then blacklist
        box.record_failure("m"); clk.advance(10 ** 8)
    assert box.level("m") == 7
    assert box.is_blacklisted("m") is True
    assert box.is_penalized("m") is True
    clk.advance(10 ** 9)                          # no amount of time clears a blacklist
    assert box.is_penalized("m") is True
    box.record_success("m")                       # success does NOT rescue a blacklisted model
    assert box.is_blacklisted("m") is True
    assert box.level("m") == 7


def test_L6_final_box_still_descends_on_success():
    clk = Clock()
    box = _box(clk)
    for _ in range(6):                            # climb to L6 (the ~2-month final box, NOT terminal)
        box.record_failure("m"); clk.advance(10 ** 8)
    assert box.level("m") == 6
    assert box.is_blacklisted("m") is False
    box.record_success("m")
    assert box.level("m") == 5                    # L6 is the last chance, not terminal


def test_blacklist_notification_fires_exactly_once():
    clk = Clock()
    fired = []
    box = _box(clk, on_blacklist=lambda mid: fired.append(mid))
    for _ in range(7):
        box.record_failure("m"); clk.advance(10 ** 8)
    assert fired == ["m"]                          # once on the transition into L7
    box.record_failure("m")                        # further failures on a blacklisted model are no-ops
    assert fired == ["m"]


def test_blacklist_notification_failure_never_raises():
    def boom(mid):
        raise RuntimeError("webhook down")
    clk = Clock()
    box = _box(clk, on_blacklist=boom)
    for _ in range(7):                             # must not raise despite the failing callback
        box.record_failure("m"); clk.advance(10 ** 8)
    assert box.is_blacklisted("m") is True


def test_clear_resets_to_L0():
    box = _box()
    for _ in range(7):
        box.record_failure("m")
    assert box.is_blacklisted("m") is True
    assert box.clear("m") is True
    assert box.level("m") == 0
    assert box.is_penalized("m") is False
    assert box.clear("m") is False                 # nothing left to clear


def test_serialization_round_trip_preserves_level_and_blacklist():
    clk = Clock()
    box = _box(clk)
    box.record_failure("a")                        # L1
    for _ in range(3):
        box.record_failure("b"); clk.advance(10 ** 7)   # L3
    for _ in range(7):
        box.record_failure("c"); clk.advance(10 ** 8)   # L7 blacklist
    snapshot = box.to_dict()

    restored = _box(clk)
    restored.from_dict(snapshot)
    assert restored.level("a") == 1
    assert restored.level("b") == 3
    assert restored.is_blacklisted("c") is True
    assert restored.is_penalized("c") is True      # the regression guard: blacklist survives a reload


def test_from_dict_tolerates_garbage():
    box = _box()
    box.from_dict(None)                            # absent sidecar key
    box.from_dict({"m": {"level": "oops"}})        # malformed entry skipped, not fatal
    box.from_dict({"ok": {"level": 2, "until": 0}})
    assert box.level("ok") == 2
    assert box.level("m") == 0
