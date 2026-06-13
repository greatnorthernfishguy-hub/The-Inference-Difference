# ---- Changelog ----
# [2026-06-12] Claude Code (Opus 4.8, Anima/Codemine CC) — TID routing substrate-authority fix, Layer B
# What: ModelPenaltyBox — a per-model escalating-backoff circuit breaker (penalty box) for TID routing.
#   A model that returns model-attributable failures is EXCLUDED from candidate selection for an
#   escalating duration (L0 clear → L1..L6 timed boxes → L7 terminal blacklist), climbs one rung per
#   failure, descends one rung per success (asymmetric: lost fast, earned back slow), and persists.
# Why: TID's substrate learning is reward-only (router._score_learned returns neutral for any model not
#   in the top-k winners), so it can boost a good model but never DEMOTE a broken one — a 0%-success
#   model survived 91 selections, routing Syl to models that cannot embody her. This is the negative
#   authority (veto) half of the fix; positive routing authority stays IN the substrate (Layer A).
#   Design + Law analysis: docs/prd/2026-06-12-tid-model-penalty-box-design.md (Tier-1 Law-checked,
#   neurograph-law-enforcer COMPLIANT 2026-06-12).
# How: pure operational state machine (the Competence-Model category, like Elmer's TuningSocket — local
#   per-parameter accumulator, NOT substrate topology). VETO-ONLY BY CONSTRUCTION: this class exposes
#   only state + boolean eligibility queries; there is deliberately NO scoring method, so it can never
#   contribute a positive selection score (the law-enforcer Q2 guardrail, enforced structurally).
#   Time is injected (time_fn) so the ladder is deterministically testable with no real sleeps.
#   Persistence (to_dict/from_dict) is TTL-agnostic — the router MUST load it UNCONDITIONALLY, exempt
#   from the #282 sidecar's 24h discard, or long-box/blacklist state evaporates on a quiet-day restart
#   (the recurring bug). See design §3.2.
# -------------------
"""Per-model penalty box (escalating-backoff circuit breaker) for TID routing — Layer B.

The substrate holds raw failure *experience* (deposited unclassified via ``report_outcome`` →
``ng_lite.record_outcome``); this box holds the operational *consequence* — a backoff schedule
that excludes a failing model from selection. The two are distinct: experience → substrate (raw,
LAW 7); control-flow state machine → here (local, the same category the ecosystem already accepts
for ``model_client._provider_blocked``). Forcing this timer state machine into the Hebbian graph
would violate LAW 6/7; local placement is correct.

Ladder (``ladder_seconds`` = the six timed-box durations, L1..L6):
    L0  clear / eligible
    L1..L6  timed boxes; duration = ``ladder_seconds[level - 1]``  (L6 ≈ 2 months)
    L7  BLACKLIST — terminal, no timer, never auto-descends; cleared only by ``clear()``.
"""

from typing import Any, Callable, Dict, Optional
import time as _time


class ModelPenaltyBox:
    """Escalating-backoff veto state, keyed by ``model_id``.

    VETO-ONLY: exposes eligibility (``is_penalized``) and state (``level`` / ``is_blacklisted``),
    never a score. Callers exclude penalized models from candidates; they must never read this to
    *prefer* a model (that authority belongs to the substrate — Layer A).
    """

    def __init__(
        self,
        ladder_seconds=(30, 300, 3600, 86400, 604800, 5184000),
        time_fn: Optional[Callable[[], float]] = None,
        on_blacklist: Optional[Callable[[str], None]] = None,
    ) -> None:
        # The six timed-box durations (L1..L6). L6 ≈ 2 months. Bootstrap scaffolding (LAW 5 /
        # Competence Model) — substrate-tunable under Layer A.
        self._ladder = list(ladder_seconds)
        self._blacklist_level = len(self._ladder) + 1  # L7 when ladder has 6 entries
        self._time = time_fn or _time.time
        self._on_blacklist = on_blacklist
        # model_id -> {"level": int, "until": float}. Absent key == L0 / clear.
        self._state: Dict[str, Dict[str, float]] = {}

    # ----- internals -----------------------------------------------------

    def _now(self) -> float:
        return float(self._time())

    def _entry(self, model_id: str) -> Optional[Dict[str, float]]:
        return self._state.get(model_id)

    # ----- mutation (called from RouterEngine.report_outcome) -------------

    def record_failure(self, model_id: str) -> None:
        """Climb one rung on a MODEL-ATTRIBUTABLE failure.

        The caller is responsible for NOT calling this on provider-wide failures (402/credit/
        provider-blocked) — those belong to the provider circuit breaker, not the model ladder.
        """
        cur = self.level(model_id)
        if cur >= self._blacklist_level:
            return  # already blacklisted — terminal, no-op
        new = cur + 1
        if new >= self._blacklist_level:
            # Transition into terminal blacklist (fires the notify exactly once).
            self._state[model_id] = {"level": float(self._blacklist_level), "until": 0.0}
            if self._on_blacklist is not None:
                try:
                    self._on_blacklist(model_id)
                except Exception:
                    pass  # notification is best-effort; never break routing
            return
        self._state[model_id] = {
            "level": float(new),
            "until": self._now() + self._ladder[new - 1],
        }

    def record_success(self, model_id: str) -> None:
        """Descend one rung on success (asymmetric recovery; never descends from blacklist)."""
        cur = self.level(model_id)
        if cur == 0 or cur >= self._blacklist_level:
            return  # already clear, or terminal blacklist (manual clear only)
        new = cur - 1
        if new <= 0:
            self._state.pop(model_id, None)  # back to L0 / clear
            return
        # Descend a rung but remain eligible (the model just succeeded → until in the past).
        self._state[model_id] = {"level": float(new), "until": 0.0}

    def clear(self, model_id: str) -> bool:
        """Manual reset to L0 (the /routing/penalty/clear endpoint). True if a penalty was cleared."""
        return self._state.pop(model_id, None) is not None

    def clear_all(self) -> int:
        n = len(self._state)
        self._state.clear()
        return n

    # ----- queries (called from RouterEngine._filter_candidates) ----------

    def level(self, model_id: str) -> int:
        e = self._entry(model_id)
        return int(e["level"]) if e else 0

    def is_blacklisted(self, model_id: str) -> bool:
        return self.level(model_id) >= self._blacklist_level

    def is_penalized(self, model_id: str) -> bool:
        """True iff the model is currently excluded from selection (in a timed box or blacklisted)."""
        e = self._entry(model_id)
        if e is None:
            return False
        if int(e["level"]) >= self._blacklist_level:
            return True
        return float(e["until"]) > self._now()

    # ----- persistence (TTL-EXEMPT — see design §3.2) --------------------

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Serialize for the sidecar. Must be loaded back UNCONDITIONALLY (never age-discarded)."""
        return {
            mid: {"level": int(e["level"]), "until": float(e["until"])}
            for mid, e in self._state.items()
        }

    def from_dict(self, data: Optional[Dict[str, Dict[str, Any]]]) -> None:
        """Restore from the sidecar. Tolerant of absent/malformed input (backward-compatible)."""
        self._state = {}
        if not isinstance(data, dict):
            return
        for mid, e in data.items():
            try:
                self._state[str(mid)] = {
                    "level": float(int(e["level"])),
                    "until": float(e.get("until", 0.0)),
                }
            except (KeyError, TypeError, ValueError):
                continue  # skip malformed entries rather than fail the whole load
