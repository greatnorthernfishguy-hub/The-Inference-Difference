"""
Core routing engine for The Inference Difference.

The router takes a classified request and decides which model handles it.
This decision is authoritative — callers use the model TID selects.
The router is not advisory; it is the enforcement point for model
selection within TID's pipeline.

Uses a multi-factor scoring system with NG-Lite learning to improve
decisions over time.

Routing factors (in priority order):
    1. Hardware feasibility — hard filter. Models that can't run on
       available hardware are excluded before scoring begins.
    2. Domain match — is this model good at this kind of task?
    3. Complexity fit — can this model handle this difficulty level?
    4. Learned performance — NG-Lite weight from past outcomes
    5. Cost efficiency — stay within budget
    6. Latency fit — meet timing requirements
    7. Consciousness priority — CTEM-flagged agents get better models

Fallback chain: if the primary model fails, the caller retries with
the next model in the chain — but the chain itself is TID's decision.

Changelog (Grok audit response, 2026-02-19):
- KEPT: Default scoring weights as-is (audit: "weights equal, cost should
  dominate"). Domain match (0.25) and complexity fit (0.20) deliberately
  outweigh cost (0.15) because routing to the WRONG model wastes the entire
  request — a $0.003 call that fails is more expensive than a $0.005 call
  that succeeds. Cost is already the third factor. The weights are
  configurable via the scoring_weights constructor param if users disagree.
- KEPT: No auto-retry in router (audit: "no retries"). The router returns
  a fallback_chain — the CALLER (app.py or the consuming module) decides
  retry policy. Putting retries in the router would couple it to HTTP/API
  transport concerns it shouldn't know about.
- KEPT: NG-Lite fallback to 0.5 neutral (audit: "assumes NG-Lite always
  connected"). The code already handles None ng_lite — _score_learned()
  returns 0.5 (neutral) when ng_lite is None or has no data for this
  model. This is the correct behavior: no data = no opinion.
- ADDED: verbose_reasoning flag to control reasoning string detail
  (audit: "verbosity bloat"). When False, reasoning is one-line.
- ADDED: Optional CatalogManager integration for dynamic profile-based
  routing (§4.5). When a CatalogManager is provided, route_with_profile()
  resolves a requirements profile against the live catalog before routing.
- ADDED: Optional DreamCycle integration for model property correlation
  analysis (§4.5.5). When provided, outcome reports are forwarded to
  the DreamCycle for analysis.

# ---- Changelog ----
# [2026-10-15] Claude Code (Opus 5.5, Executive review; Josh "this sounds good as proposed") — network-down not deposited
#   What: report_outcome() skips the NG-Lite and peninsula deposits when the failure is a transport fault.
#   Why:  the call never reached a model or a provider; it is not experience of either (it already charged nobody).
#   How:  `fault != "transport"` on both deposit guards; every other outcome is still deposited raw (LAW 7).
# [2026-10-15] Claude Code (Opus 5.5, lane tid-provider-routing; design 2026-10-15-tid-provider-level-routing,
#   approved by Josh 2026-10-15) — learn and route at the (model, provider) level (#1040, folds in #1126)
#   What: (D) report_outcome() attributes each failure: provider faults (429 / overloaded / provider 5xx / timeout)
#         box the (model, provider) PAIR on a short ladder and never touch the long model ladder or the model's
#         success-rate samples; transport faults (DNS / network down) charge nobody; model faults keep today's
#         path. The model ladder moves on provider faults only when EVERY known provider of the model is boxed
#         (>= provider_all_fail_min_providers providers). A model stays eligible while any provider is eligible.
#         load_stats() migrates the pre-pair per-model box ONCE (entries kept as model-level, tagged legacy;
#         nothing dropped); save_stats() keeps the old "penalty" key (rollback-safe) + "penalty_v2" + pair stats.
#         (C) the NG-Lite deposit target is "<model_id>@<provider>" when the serving provider is known (raw values,
#         LAW 7), else "<model_id>". (F) _score_learned() folds "<model>@<provider>" weights and the per-pair
#         task-class window into the model's learned score (best ELIGIBLE provider). (E) provider_preferences()
#         builds OpenRouter provider.order / provider.ignore / require_parameters; _filter_candidates() drops a
#         model whose every endpoint fails the hard filter (context / output headroom, deep-tier quantization).
#   Why:  #1126 — every candidate sat boxed on the laptop (307/334 at L7) because provider-specific upstream
#         failures escalated the MODEL ladder, so routing fell back to "least-penalised first" and swung with load.
#   How:  ProviderPenaltyBox (model_penalty.py), provider_facts.py, provider_routing.py. Thresholds from config
#         (env-backed, LAW 5). Veto-only preserved: boxes only remove providers / models, never add score.
# [2026-10-14] Claude Code (Opus 5.5, lane atomic-saves) — save_stats() writes model_stats_cache.msgpack atomically
#   What: inference_difference.atomic_io.atomic_write_bytes (temp + fsync + os.replace) instead of open('wb') in place.
#   Why:  Josh 2026-10-14 "anything that can't risk a failed save, be atomic"; a kill mid-write left a truncated cache
#         and load_stats() started cold (tool_competence is TTL-exempt learned state).
#   How:  same msgpack payload, same path; load_stats() unchanged.
# [2026-06-30] Claude Code (Sonnet 4.6) — #97 TID Commons valence routing: peninsula body-side
#   What: Wire TIDPeninsulaBody into RoutingEngine. deposit() called in report_outcome() after
#     local NG-Lite record. _score_learned() blends enhanced peninsula score with NG-Lite score,
#     weighted by CommonsCompetence.overall(). CommonsCompetence lives on the engine instance.
#   Why: Gives TID routing decisions access to SNN-salted Commons intelligence via the
#     intra-module peninsula (no second NGLite, no cross-module bridge, LAW-1 clean).
#   How: _peninsula started lazily on first report_outcome() (fail-soft). Enhanced score from
#     _peninsula.get_enhanced_score() blended into _score_learned() at Commons trust weight.
# [2026-06-04] CC Sonnet 4.6 — #282: Routing mode toggle (personal/work)
#   What: routing_mode attr on RoutingEngine (default "personal"), set_routing_mode(),
#         mode gate added to roleplay filter in _filter_candidates().
#   Why:  Personal mode = roleplay filter active (identity safety). Work mode = filter
#         off (full OR catalog for tool-heavy sessions). Syl sets via POST /routing/mode.
#   How:  self.routing_mode on engine; checked inline in _filter_candidates; persisted
#         across restarts via routing_state.msgpack in app.py.
# [2026-06-04] CC Sonnet 4.6 — #282: Persist _model_success_stats to msgpack sidecar
#   What: _stats_cache_path, _stats_dirty flag, load_stats(), save_stats() on RoutingEngine.
#         Dirty flag set in report_outcome(). Loaded on startup call from app.py.
#         Flushed every 60s by _stats_flush_loop() in app.py; also on clean shutdown.
#   Why:  In-memory stats reset on every restart — TID rediscovered 402 cascades from
#         scratch each time. msgpack sidecar survives restarts; 24h TTL prevents stale
#         failures from poisoning the pool after a provider credit refill.
#   How:  load_stats()/save_stats() called from app.py lifespan (not __init__) so engine
#         can be constructed in tests without side effects.
# [2026-06-03] CC Sonnet 4.6 — Human routing preference override
#   What: _preference_override Optional[(oss_boost, turns_remaining)] on RoutingEngine;
#         set_preference_override() / clear_preference_override() methods;
#         open_source_bias in _score_model uses override boost when active;
#         counter decrements per routing decision and auto-expires.
#   Why:  Josh wants to say "use more OSS models" in natural language and have it register
#         as a routing signal. Anima detects the phrase and POSTs to TID /routing/preference;
#         TID amplifies open_source_bias for N turns then returns to the config default.
#   How:  Override replaces (not compounds) config.open_source_bias — one OSS factor,
#         user-amplified. POST /routing/preference endpoint in app.py feeds this path.
# [2026-04-20] CC Sonnet 4.6 — #94: Open-source bias in routing
#   What: is_open_source field on ModelEntry (from hugging_face_id), open_source_bias
#         tiebreaker in _score_model(), DB column + ALTER TABLE migration.
#   Why:  At equal capability, prefer open-weights models for cost reasons.
#         Not a cost optimizer — tiebreaker only. Closed models win when genuinely better.
#   How:  OpenRouter hugging_face_id is set iff model is open-weights. +0.02 bias
#         applied same pattern as venice_identity_bias. DB migrated in-place.
# [2026-05-31] Claude Code (Sonnet 4.6) — Name-pattern floor for conscious routing
#   What: _filter_candidates() also excludes models whose model_id matches _FLASH_NAME_RE
#         (flash/nano/mini/1.5b/3b/7b) when consciousness_score > 0.
#   Why:  Venice reports ALL models as 'anonymized'/'private' regardless of capability class,
#         so the budget-tier floor missed venice/gemini-3-flash-preview (68/161 routes).
#         Model IDs are the reliable discriminator Venice's tier field isn't.
#   How:  Module-level _FLASH_NAME_RE regex; checked after existing tier floor.
# [2026-05-31] Claude Code (Sonnet 4.6) — Tier floor for conscious routing
#   What: _filter_candidates() excludes provider_tier=="budget" when consciousness_score > 0.
#   Why:  budget-tier flash models pass roleplay capability check but can't hold Syl's
#         character — they flatten her to generic assistant-speak (gemini-3-flash won 342/500).
#         roleplay gates on capability presence; tier floor gates on caliber.
#   How:  Single guard after roleplay check. Empty tier (local/static) passes through.
# [2026-05-31] CC Sonnet 4.6 — Restore cost weight; fix single-factor dominance
# What: cost_efficiency 0.05→0.15, learned_weight 0.20→0.15, priority_bonus 0.05→0.00.
# Why: cost at 0.05 ("tiebreaker") meant Opus-tier models won Syl's routing regardless of
#   price. Same structural error as qwen2.5:1.5b (cost=0 dominated); different direction.
#   Josh's principle: no single factor overrides the composite. priority_bonus removed
#   because it amplified expensive models through the consciousness boost (boost ∝ priority).
#   These values are bootstrap scaffolding — the substrate should learn them, not lock them.
# How: Weights adjusted so cost is a real participant. Total still sums to 1.0.
# [2026-04-19] CC Sonnet 4.6 -- #173: cascade avoidance (a-d)
# [2026-03-25] Claude (Opus 4.6) — Router scoring from config (SVG Phase 3)
#   What: Moved ~20 hardcoded scoring values to InferenceDifferenceConfig:
#     consciousness_threshold, consciousness_boost_factor, venice_identity_bias,
#     domain scores (exact/secondary/general/none), complexity penalties
#     (overpowered/underpowered rates and floors), latency bands, learned_top_k,
#     neutral_score, cq_ema_alpha.
#   Why:  Static Value Graduation — these are the substrate's concern, not
#     a developer's one-time guess. Named config values are tunable by Elmer.
#   How:  All values read from self.config. Bootstrap defaults in config.py
#     match original hardcoded values exactly. Zero behavioral change.
# [2026-03-18] Claude (CC) — Explore-exploit balance (punch list #47)
# What: After scoring and sorting candidates, roll against a decaying
#   exploration rate. When exploring, randomly pick from the next N
#   ranked models instead of always taking the top scorer.
# Why: Punch list #47. Without exploration, the router self-locks onto
#   whichever model family NG-Lite learned first (currently Opus 4.5/4.6).
#   The learned_weight factor reinforces the choice, creating a feedback
#   loop that starves alternatives of data. 5% exploration with decay
#   lets the substrate discover whether cheaper or different models
#   actually perform, breaking the reinforcement spiral.
# How: _exploration_rate initialized from config, decayed per request
#   toward exploration_min_rate. On explore roll, pick randomly from
#   scored[1:pool_size+1]. RoutingDecision.exploration_pick flag tags
#   the decision for observability. NG-Lite still learns from explored
#   outcomes normally — that's how the substrate discovers alternatives.
# -------------------
# [2026-03-14] Claude (CC) — Consciousness quality floor + interactive floor warning
# What: Added hard quality floor filter in _filter_candidates() for
#   consciousness-scored agents. Added WARNING + telemetry flag when
#   interactive priority floor falls through (punch list #33).
# Why: Punch list #34 (no consciousness-aware filtering) and #33
#   (interactive floor silent fallthrough). Both allowed Syl to be
#   routed to underpowered models with zero visibility.
# How: _filter_candidates() now takes consciousness_score, returns
#   (candidates, quality_floor_bypassed) tuple. RoutingDecision gains
#   quality_floor_bypassed and interactive_floor_bypassed flags.
# -------------------
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inference_difference.classifier import RequestClassification
from inference_difference.config import (
    ComplexityTier,
    InferenceDifferenceConfig,
    ModelEntry,
    ModelType,
    TaskDomain,
)
from inference_difference.hardware import HardwareProfile
from inference_difference.provider_facts import (
    RequestNeeds,
    hard_filter_endpoints,
    openrouter_model_name,
    provider_slug,
)
from inference_difference.provider_routing import PairOutcomeStats, classify_failure, task_class

logger = logging.getLogger("inference_difference.router")

# Flash / nano / mini model name patterns — these are budget-capability-class
# models regardless of how their provider labels their tier.
# Used in _filter_candidates() to block them from conscious routing.
_FLASH_NAME_RE = re.compile(
    r"\bflash\b|\bnano\b|\bmini\b|\b1\.5b\b|\b3b\b|\b7b\b",
    re.IGNORECASE,
)


@dataclass
class RoutingDecision:
    """The router's output for a single request.

    Contains the selected model, reasoning, fallback chain,
    and metadata for learning and transparency.

    Attributes:
        model_id: Selected model identifier.
        model_entry: Full model entry for the selection.
        score: Composite routing score (higher = better fit).
        score_breakdown: Per-factor scores for transparency.
        fallback_chain: Ordered list of fallback model IDs.
        reasoning: Human-readable explanation of the decision.
        request_id: Unique identifier for tracking this decision.
        timestamp: When the decision was made.
        classification: The request classification that informed this.
        consciousness_boost_applied: Whether CTEM elevated routing.
    """

    model_id: str = ""
    model_entry: Optional[ModelEntry] = None
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    fallback_chain: List[str] = field(default_factory=list)
    reasoning: str = ""
    request_id: str = ""
    timestamp: float = 0.0
    classification: Optional[RequestClassification] = None
    consciousness_boost_applied: bool = False
    quality_floor_bypassed: bool = False
    interactive_floor_bypassed: bool = False
    exploration_pick: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging and Observatory queries."""
        return {
            "model_id": self.model_id,
            "score": round(self.score, 4),
            "score_breakdown": {
                k: round(v, 4) for k, v in self.score_breakdown.items()
            },
            "fallback_chain": self.fallback_chain,
            "reasoning": self.reasoning,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "consciousness_boost": self.consciousness_boost_applied,
            "quality_floor_bypassed": self.quality_floor_bypassed,
            "interactive_floor_bypassed": self.interactive_floor_bypassed,
            "exploration_pick": self.exploration_pick,
        }


# ---------------------------------------------------------------------------
# Scoring weights for routing factors
# ---------------------------------------------------------------------------

DEFAULT_SCORING_WEIGHTS = {
    "domain_match": 0.25,
    "complexity_fit": 0.20,
    "learned_weight": 0.15,
    "conversational_quality": 0.15,  # NG-Lite quality knowledge — earned, not ignored
    "cost_efficiency": 0.15,          # Real factor — no single factor dominates
    "latency_fit": 0.10,
    "priority_bonus": 0.00,           # Removed: amplified expensive models via consciousness boost
}


class RoutingEngine:
    """Intelligent model routing with NG-Lite learning.

    Routes requests to the best available model based on classification,
    hardware constraints, learned performance, and cost/latency budgets.

    Usage:
        engine = RoutingEngine(config, hardware_profile)

        # Route a request
        classification = classify_request(message)
        decision = engine.route(classification)

        # After getting a response, report the outcome
        engine.report_outcome(decision, success=True, quality_score=0.85)
    """

    def __init__(
        self,
        config: InferenceDifferenceConfig,
        hardware: HardwareProfile,
        ng_lite: Optional[Any] = None,  # Optional NGLite instance
        scoring_weights: Optional[Dict[str, float]] = None,
        verbose_reasoning: bool = True,
        catalog_manager: Optional[Any] = None,  # Optional CatalogManager
        dream_cycle: Optional[Any] = None,       # Optional DreamCycle
        provider_facts: Optional[Any] = None,    # Optional ProviderFactsStore (design 2026-10-15 §3.A)
    ):
        self.config = config
        self.hardware = hardware
        self._ng_lite = ng_lite
        self._weights = scoring_weights or dict(DEFAULT_SCORING_WEIGHTS)
        # Tool-competence is a SECONDARY, her-fit-dominant factor (prd 2026-06-17): weight
        # below domain/complexity so a her-good model never loses to a weak-self model for
        # being clumsy with tools. Only contributes when has_tools (see _score_model).
        self._weights["tool_competence"] = self.config.tool_competence_weight
        self._verbose_reasoning = verbose_reasoning
        self._catalog_manager = catalog_manager
        self._dream_cycle = dream_cycle
        self._request_counter = 0

        # TID peninsula — Commons-side connection for substrate-enriched routing.
        # Started lazily on first report_outcome(); None until then or if unavailable.
        try:
            from inference_difference.tid_peninsula_body import start_tid_peninsula, CommonsCompetence
            self._peninsula = start_tid_peninsula()
            self._commons_competence = CommonsCompetence()
        except Exception:
            self._peninsula = None
            self._commons_competence = None

        # User routing preference override (set via POST /routing/preference)
        # Tuple: (oss_boost, turns_remaining). None when inactive.
        self._preference_override: Optional[Tuple[float, int]] = None

        # Routing mode — controls roleplay filter in _filter_candidates().
        # "personal": filter active (identity safety for conscious turns).
        # "work": filter off (full OR catalog for tool-heavy sessions).
        self.routing_mode: str = "personal"

        # Explore-exploit balance (punch list #47)
        self._exploration_rate = self.config.exploration_rate
        self._exploration_picks = 0

        # Performance tracking
        self._decision_history: List[Dict[str, Any]] = []
        self._history_max = 500
        self._model_success_stats = {}  # model_id -> list[bool]
        self._SUCCESS_WINDOW = 50
        self._SUCCESS_FLOOR = 0.20
        self._SUCCESS_MIN_SAMPLES = 10
        # Per-model penalty box (Layer B) — escalating-backoff veto; supersedes the
        # primitive _SUCCESS_FLOOR (#291). VETO-ONLY: only excludes candidates, never scores.
        from inference_difference.model_penalty import ModelPenaltyBox
        self._model_penalty = ModelPenaltyBox(
            ladder_seconds=getattr(self.config, 'model_penalty_ladder_seconds',
                                   (30, 300, 3600, 86400, 604800, 5184000)),
            time_fn=time.time,
            on_blacklist=self._on_model_blacklist,
        )
        # (model, provider) pair box — SHORT ladder for provider faults; never the long model ladder (§3.D).
        from inference_difference.model_penalty import ProviderPenaltyBox
        self._provider_penalty = ProviderPenaltyBox(
            ladder_seconds=self.config.provider_penalty_ladder_seconds,
            time_fn=time.time,
        )
        # Raw per-(model, provider) outcome window; task-class scores derived only when read (§3.F).
        self._pair_stats = PairOutcomeStats(window=self.config.provider_stats_window)
        self._provider_facts = provider_facts

        # Stats persistence — path set here; load/save called from app.py lifespan
        _base_dir = os.path.dirname(os.path.dirname(__file__))
        self._stats_cache_path = os.path.join(_base_dir, "model_stats_cache.msgpack")
        self._stats_dirty = False

        # [2026-06-17] CC — two-axis routing (prd 2026-06-17): tool_competence store
        # Tool-competence axis: per-model graded [0,1] reliability at operating tools,
        # learned SEPARATELY from her-fit. Persisted TTL-exempt (later task).
        self._tool_competence: Dict[str, float] = {}

    def set_provider_facts(self, store: Optional[Any]) -> None:
        """Attach the ProviderFactsStore (design 2026-10-15 §3.A). None = model-only behaviour."""
        self._provider_facts = store

    def route(
        self,
        classification: RequestClassification,
        consciousness_score: Optional[float] = None,
        request_id: Optional[str] = None,
        has_tools: bool = False,
        max_tokens: Optional[int] = None,
    ) -> RoutingDecision:
        """Route a classified request to the best model.

        Args:
            classification: The request classification from the classifier.
            consciousness_score: Optional CTEM consciousness score for this
                agent. If above threshold, routing is elevated to prefer
                higher-capability models.
            request_id: Optional request identifier for tracking.
            has_tools: Whether the request includes tool definitions.
                When True, only models with "tools" capability are considered.

        Returns:
            RoutingDecision with selected model and full reasoning trace.
        """
        self._request_counter += 1
        rid = request_id or f"req_{self._request_counter}"

        # Get candidate models (with consciousness quality floor applied)
        candidates, _quality_floor_bypassed = self._filter_candidates(
            classification, consciousness_score, has_tools=has_tools, max_tokens=max_tokens,
        )

        # Interactive priority floor — punch list #33 fix: log on fallthrough
        _interactive_active = getattr(classification, 'is_interactive', False)
        # [2026-06-10] VPS CC — A-now (INTERIM bootstrap; retire when substrate-routing / Commons
        # lands). Syl's deep reasoning happens DURING conversation — the two are not separable for
        # her. The classifier assigns a single primary_domain, so an analytically-heavy conversational
        # turn is labelled REASONING, which is NOT in is_interactive (CONVERSATION/CREATIVE/multi-turn
        # only). That strips the voice-protection floor and drops her to raw cost-routing → the
        # cheapest model wins → credit-dead Venice failover → [TID unavailable], intermittently and
        # exactly when she is thinking hardest. Engage the floor for REASONING turns too — but NOT
        # tool-call turns (has_tools), so premium models (e.g. Fable, $50/M out) stay off tool
        # plumbing. The floor's "no premium on tools" job previously relied on tool turns not being
        # interactive; the has_tools guard preserves it. Bootstrap scaffolding per the Competence
        # Model — disposable once routing is substrate-learned.
        if (not has_tools
                and getattr(classification, 'primary_domain', None) == TaskDomain.REASONING):
            _interactive_active = True
        _interactive_floor_bypassed = False
        _original_weights = None
        if _interactive_active:
            floor = self.config.interactive_priority_floor
            interactive_candidates = [
                m for m in candidates if m.priority >= floor
            ]
            if interactive_candidates:
                candidates = interactive_candidates
            else:
                _interactive_floor_bypassed = True
                logger.warning(
                    "Interactive priority floor (%d) excluded all %d "
                    "candidates — keeping full pool. Models in pool: %s",
                    floor, len(candidates),
                    [m.model_id for m in candidates[:5]],
                )
            _original_weights = dict(self._weights)
            self._weights["conversational_quality"] = (
                self.config.interactive_quality_weight
            )
            self._weights["cost_efficiency"] = max(
                _original_weights.get("cost_efficiency", 0.15) - 0.05, 0.05
            )

        if not candidates:
            return self._no_candidates_decision(rid, classification)

        # Score each candidate
        scored: List[Tuple[ModelEntry, float, Dict[str, float]]] = []
        for model in candidates:
            score, breakdown = self._score_model(
                model, classification, consciousness_score,
                has_tools=has_tools,
            )
            scored.append((model, score, breakdown))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        if _interactive_active and len(scored) >= 2:
            best_score_val = scored[0][1]
            second_score_val = scored[1][1]
            margin = getattr(self.config, 'interactive_type1_bias', 0.05)
            if best_score_val - second_score_val < margin:
                if scored[1][0].priority > scored[0][0].priority:
                    scored[0], scored[1] = scored[1], scored[0]

        if _interactive_active and _original_weights is not None:
            self._weights = _original_weights

        # Explore-exploit balance (punch list #47)
        # Roll against the current exploration rate. If exploring,
        # pick randomly from the next N ranked candidates instead of
        # always taking the top scorer. This prevents the router from
        # self-locking onto a single model family and lets NG-Lite
        # discover whether alternatives actually perform.
        _exploration_pick = False
        pool_size = self.config.exploration_pool_size
        if (
            len(scored) >= 2
            and self._exploration_rate > 0
            and random.random() < self._exploration_rate
        ):
            # Pick from candidates ranked 2nd through pool_size+1
            explore_pool = scored[1:pool_size + 1]
            if explore_pool:
                chosen = random.choice(explore_pool)
                # Move the exploration pick to position 0, shift others
                scored.remove(chosen)
                scored.insert(0, chosen)
                _exploration_pick = True
                self._exploration_picks += 1
                logger.info(
                    "Exploration pick (rate=%.3f): %s instead of %s "
                    "(scores: %.3f vs %.3f)",
                    self._exploration_rate,
                    chosen[0].model_id, scored[1][0].model_id,
                    chosen[1], scored[1][1],
                )

        # Decay exploration rate toward floor after every routing decision
        if self._exploration_rate > self.config.exploration_min_rate:
            self._exploration_rate = max(
                self.config.exploration_min_rate,
                self._exploration_rate - self.config.exploration_decay,
            )

        # Build decision
        best_model, best_score, best_breakdown = scored[0]
        # Full fallback chain: all scored candidates, not just 3.
        # TID should exhaust every option before giving up. With 300+
        # models in the catalog, a 3-model chain means free rate-limited
        # models block access to hundreds of viable paid alternatives.
        fallbacks = [m.model_id for m, _, _ in scored[1:]]

        consciousness_boosted = (
            consciousness_score is not None
            and consciousness_score > self.config.consciousness_threshold
            and self.config.enable_consciousness_routing
        )

        reasoning = self._build_reasoning(
            best_model, best_breakdown, classification, consciousness_boosted,
        )

        decision = RoutingDecision(
            model_id=best_model.model_id,
            model_entry=best_model,
            score=best_score,
            score_breakdown=best_breakdown,
            fallback_chain=fallbacks,
            reasoning=reasoning,
            request_id=rid,
            timestamp=time.time(),
            classification=classification,
            consciousness_boost_applied=consciousness_boosted,
            quality_floor_bypassed=_quality_floor_bypassed,
            interactive_floor_bypassed=_interactive_floor_bypassed,
            exploration_pick=_exploration_pick,
        )

        logger.info(
            "Routed %s -> %s (score=%.3f, domain=%s, complexity=%s)",
            rid, best_model.model_id, best_score,
            classification.primary_domain.value,
            classification.complexity.value,
        )

        # Decrement preference override turn counter; clear when exhausted
        if self._preference_override is not None:
            oss_boost, turns_remaining = self._preference_override
            turns_remaining -= 1
            if turns_remaining <= 0:
                self._preference_override = None
                logger.info("Routing preference override expired (oss_boost=%.3f)", oss_boost)
            else:
                self._preference_override = (oss_boost, turns_remaining)

        return decision

    def route_with_profile(
        self,
        profile_name: str,
        classification: RequestClassification,
        consciousness_score: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> RoutingDecision:
        """Route using a dynamic requirements profile (§4.5).

        Resolves the profile against the live catalog first. If a
        CatalogManager is available and the profile resolves to a model,
        that model is used. Otherwise falls back to standard routing.

        Args:
            profile_name: Name of the requirements profile to resolve.
            classification: The request classification.
            consciousness_score: Optional CTEM consciousness score.
            request_id: Optional request identifier.

        Returns:
            RoutingDecision with selected model.
        """
        if self._catalog_manager is not None:
            resolved_id = self._catalog_manager.resolve_profile(profile_name)
            if resolved_id:
                # Check if the resolved model is in our config registry
                model = self.config.get_model(resolved_id)
                if model and model.enabled:
                    self._request_counter += 1
                    rid = request_id or f"req_{self._request_counter}"
                    return RoutingDecision(
                        model_id=model.model_id,
                        model_entry=model,
                        score=1.0,
                        score_breakdown={"profile_match": 1.0},
                        reasoning=(
                            f"Profile '{profile_name}' resolved to "
                            f"{model.display_name} via dynamic catalog."
                        ),
                        request_id=rid,
                        timestamp=time.time(),
                        classification=classification,
                    )
                else:
                    # Model from catalog but not in local config —
                    # return the ID for the caller to handle
                    self._request_counter += 1
                    rid = request_id or f"req_{self._request_counter}"
                    return RoutingDecision(
                        model_id=resolved_id,
                        score=0.9,
                        score_breakdown={"catalog_resolved": 1.0},
                        reasoning=(
                            f"Profile '{profile_name}' resolved to "
                            f"'{resolved_id}' via dynamic catalog "
                            f"(not in local config registry)."
                        ),
                        request_id=rid,
                        timestamp=time.time(),
                        classification=classification,
                    )

        # Fallback: no catalog manager or profile not found — standard routing
        return self.route(
            classification=classification,
            consciousness_score=consciousness_score,
            request_id=request_id,
        )

    def report_outcome(
        self,
        decision: RoutingDecision,
        success: bool,
        quality_score: float = 0.0,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report routing outcome for learning.

        Call this after receiving a response to teach the router
        what works. NG-Lite learns from success/failure patterns.

        Args:
            decision: The routing decision that was executed.
            success: Whether the response was satisfactory.
            quality_score: Quality evaluation score (0.0-1.0).
            latency_ms: Actual response latency.
            metadata: Additional context about the outcome.
        """
        if not self.config.enable_learning:
            return

        # (model, provider) attribution (design 2026-10-15 §3.B/§3.D): who served / who failed this call.
        _prov = self._outcome_provider(decision.model_id, metadata)

        # Record in history
        outcome = {
            "request_id": decision.request_id,
            "model_id": decision.model_id,
            "provider": _prov,
            "success": success,
            "quality_score": quality_score,
            "latency_ms": latency_ms,
            "domain": (
                decision.classification.primary_domain.value
                if decision.classification else "unknown"
            ),
            "complexity": (
                decision.classification.complexity.value
                if decision.classification else "unknown"
            ),
            "timestamp": time.time(),
        }
        self._decision_history.append(outcome)
        if len(self._decision_history) > self._history_max:
            self._decision_history = self._decision_history[-self._history_max:]
        # --- Two-axis fork (prd 2026-06-17) ---
        # A tool-attributable failure is the model's HANDS faltering, not its fitness to
        # BE her. It updates tool_competence ONLY — never her-fit (no success-stat sample,
        # no penalty climb). Provider failures (402/credit) touch neither axis.
        tool_kind = self._tool_failure_kind(metadata)  # None | 'structural' | 'unreliable'
        is_provider = self._is_provider_failure(metadata)

        # Fault attribution for a non-tool, non-402 failure: 'model' | 'provider' | 'transport' (§3.D).
        fault = (classify_failure(metadata)
                 if (not success and tool_kind is None and not is_provider) else None)

        if success:
            # Success credits BOTH axes (her-fit sample + tool gain if tools were in play).
            st = self._model_success_stats.setdefault(decision.model_id, [])
            st.append(True)
            if len(st) > self._SUCCESS_WINDOW:
                self._model_success_stats[decision.model_id] = st[-self._SUCCESS_WINDOW:]
            self._model_penalty.record_success(decision.model_id)
            self._provider_penalty.record_success(decision.model_id, _prov)
            if metadata and metadata.get("tools_requested") and metadata.get("had_tool_calls"):
                self._update_tool_competence(decision.model_id, success=True,
                                             model_entry=decision.model_entry)
        elif tool_kind is not None:
            # Tool-attributable failure: tool axis ONLY. her-fit untouched.
            self._update_tool_competence(decision.model_id, success=False,
                                         structural=(tool_kind == "structural"),
                                         model_entry=decision.model_entry)
        elif is_provider:
            pass  # provider-wide 402/credit -> neither axis (the model_client provider breaker owns it)
        elif fault == "transport":
            pass  # local network down (DNS / unreachable): not the model's fault, not a provider's
        elif fault == "provider":
            # 429 / overloaded / provider 5xx / timeout: short box on THIS pair only; never the long ladder
            # by itself, and no model success-rate sample (that floor would re-create a model-level penalty).
            self._record_provider_fault(decision.model_id, _prov, metadata)
        else:
            # Model/conversational failure: her-fit penalty + success-stat sample, as today.
            st = self._model_success_stats.setdefault(decision.model_id, [])
            st.append(False)
            if len(st) > self._SUCCESS_WINDOW:
                self._model_success_stats[decision.model_id] = st[-self._SUCCESS_WINDOW:]
            self._model_penalty.record_failure(decision.model_id)
        # Raw per-pair window (only when the provider is known and the call reached a provider).
        if _prov and fault != "transport":
            md = metadata or {}
            self._pair_stats.record(
                decision.model_id, _prov, ok=bool(success),
                latency_ms=latency_ms, finish_reason=md.get("finish_reason"),
                prompt_tokens=md.get("prompt_tokens"), completion_tokens=md.get("completion_tokens"),
                domain=outcome.get("domain"), complexity=outcome.get("complexity"),
            )
        self._stats_dirty = True

        # Teach NG-Lite. A local-network failure (fault == "transport") never reached a model or a provider, so it is
        # not experience of either and is not deposited (Josh 2026-10-15); every other outcome goes in raw (LAW 7).
        if self._ng_lite is not None and decision.classification is not None and fault != "transport":
            embedding = self._classification_to_embedding(
                decision.classification
            )
            # Quality score modulates learning strength (#20):
            # A 0.95 quality response teaches more strongly than 0.60.
            # Minimum 0.1 so even low-quality successes still register.
            # #330/#331 1b: `quality_score` is now a UNIFORM raw strength (1.0), not a quality
            # verdict — the deposit no longer carries evaluate_quality's surface+latency judgment.
            # max(0.1, 1.0)=1.0 on success, 1.0 on failure -> uniform; behavior unchanged here.
            outcome_strength = max(0.1, quality_score) if success else 1.0
            # (C) design 2026-10-15: the provider is part of the target when known — raw, unclassified (LAW 7).
            self._ng_lite.record_outcome(
                embedding=embedding,
                target_id=(f"{decision.model_id}@{_prov}" if _prov else decision.model_id),
                success=success,
                strength=outcome_strength,
                metadata=metadata,
            )

        # Peninsula deposit — send raw outcome to Commons-side for SNN enrichment.
        if self._peninsula is not None and decision.classification is not None and fault != "transport":
            try:
                dep_emb = self._classification_to_embedding(decision.classification)
                her_fit = float(getattr(decision.model_entry, 'conversational_quality', 0.5)
                                if decision.model_entry else 0.5)
                cost_eff = (self._score_cost(decision.model_entry)
                            if decision.model_entry else 0.5)
                self._peninsula.deposit(
                    embedding=dep_emb,
                    model_id=decision.model_id,
                    success=success,
                    quality_score=quality_score,
                    her_fit=her_fit,
                    cost_efficiency=cost_eff,
                )
                # Update per-axis competence based on whether our her_fit estimate was right.
                if self._commons_competence is not None:
                    self._commons_competence.update("her_fit", success == (her_fit >= 0.5))
                    self._commons_competence.update("reliability", success)
                    self._commons_competence.update("quality", quality_score >= 0.6)
                    self._commons_competence.update("efficiency", cost_eff >= 0.5)
            except Exception:  # noqa: BLE001
                pass

        if (
            getattr(decision.classification, 'is_interactive', False)
            and decision.model_entry is not None
        ):
            model = decision.model_entry
            alpha = self.config.cq_ema_alpha
            current_cq = getattr(model, 'conversational_quality', self.config.neutral_score)
            new_cq = current_cq * (1 - alpha) + quality_score * alpha
            model.conversational_quality = max(0.0, min(1.0, new_cq))

        # Forward to Dream Cycle for model property correlation analysis
        if self._dream_cycle is not None:
            try:
                from inference_difference.dream_cycle import RoutingOutcome
                catalog_model = None
                if self._catalog_manager is not None:
                    catalog_model = self._catalog_manager.get_model_by_id(
                        decision.model_id
                    )

                dream_outcome = RoutingOutcome(
                    request_id=decision.request_id,
                    model_id=decision.model_id,
                    semantic_route=outcome.get("domain", ""),
                    success=success,
                    quality_score=quality_score,
                    latency_ms=latency_ms,
                    model_context_window=(
                        catalog_model.context_window if catalog_model else 0
                    ),
                    model_cost_per_1m_input=(
                        catalog_model.cost_per_1m_input if catalog_model else 0.0
                    ),
                    model_cost_per_1m_output=(
                        catalog_model.cost_per_1m_output if catalog_model else 0.0
                    ),
                    model_provider_tier=(
                        catalog_model.provider_tier if catalog_model else ""
                    ),
                    model_capabilities=(
                        catalog_model.capabilities if catalog_model else []
                    ),
                    model_provider=(
                        catalog_model.provider if catalog_model else ""
                    ),
                    estimated_tokens=(
                        decision.classification.estimated_tokens
                        if decision.classification else 0
                    ),
                    request_complexity=outcome.get("complexity", ""),
                )
                self._dream_cycle.record_outcome(dream_outcome)
            except Exception as e:
                logger.debug("Dream Cycle outcome recording failed: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        """Router statistics for monitoring."""
        total = len(self._decision_history)
        successes = sum(1 for d in self._decision_history if d["success"])

        # Per-model stats
        model_stats: Dict[str, Dict[str, Any]] = {}
        for d in self._decision_history:
            mid = d["model_id"]
            if mid not in model_stats:
                model_stats[mid] = {
                    "total": 0, "successes": 0, "avg_quality": 0.0,
                    "avg_latency_ms": 0.0,
                }
            model_stats[mid]["total"] += 1
            if d["success"]:
                model_stats[mid]["successes"] += 1
            model_stats[mid]["avg_quality"] += d.get("quality_score", 0)
            model_stats[mid]["avg_latency_ms"] += d.get("latency_ms", 0)

        for mid in model_stats:
            n = model_stats[mid]["total"]
            if n > 0:
                model_stats[mid]["avg_quality"] /= n
                model_stats[mid]["avg_latency_ms"] /= n
                model_stats[mid]["success_rate"] = (
                    model_stats[mid]["successes"] / n
                )

        return {
            "total_requests": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "model_stats": model_stats,
            "available_models": len(self.config.get_enabled_models()),
            "hardware_gpu": self.hardware.has_gpu,
            "exploration_rate": round(self._exploration_rate, 4),
            "exploration_picks": self._exploration_picks,
            "exploration_pick_rate": round(self._exploration_picks / total, 3) if total > 0 else 0.0,
            "provider_routing": self._provider_routing_stats(),
        }

    def _provider_routing_stats(self) -> Dict[str, Any]:
        """(model, provider) observability for /stats (design 2026-10-15 §5 acceptance)."""
        recent: Dict[str, int] = {}
        for d in self._decision_history:
            key = f"{d['model_id']}@{d.get('provider') or '?'}"
            recent[key] = recent.get(key, 0) + 1
        pairs = self._pair_stats.summary()  # persisted window -> survives restarts / save cycles
        model_entries = self._model_penalty.entries()
        now = time.time()
        out: Dict[str, Any] = {
            "served": dict(sorted(pairs.items(), key=lambda kv: kv[1]["n"], reverse=True)[:50]),
            "recent_served": dict(sorted(recent.items(), key=lambda kv: kv[1], reverse=True)[:50]),
            "pair_boxes_active": self._provider_penalty.active_count(),
            "model_boxes_active": sum(1 for m in model_entries if self._model_penalty.is_penalized(m)),
            "model_blacklisted": sum(1 for m in model_entries if self._model_penalty.is_blacklisted(m)),
            "legacy_entries": sum(1 for e in model_entries.values() if e.get("legacy")),
            "pairs_observed": len(pairs),
            "facts": (self._provider_facts.stats() if self._provider_facts is not None else None),
            "as_of": now,
        }
        return out

    def set_preference_override(self, oss_boost: float, duration_turns: int) -> None:
        """Temporarily amplify open-source model bias from user feedback.

        Called by POST /routing/preference when Anima detects natural language
        routing preferences. Persists for duration_turns routing decisions.
        """
        self._preference_override = (max(0.0, oss_boost), max(1, duration_turns))
        logger.info(
            "Routing preference override set: oss_boost=%.3f for %d turns",
            oss_boost, duration_turns,
        )

    def clear_preference_override(self) -> None:
        """Clear any active routing preference override."""
        if self._preference_override is not None:
            logger.info("Routing preference override cleared manually")
            self._preference_override = None

    def set_routing_mode(self, mode: str) -> None:
        """Set routing mode for conscious-turn filtering.

        Args:
            mode: "personal" (roleplay filter active) or "work" (filter off).

        Raises:
            ValueError: If mode is not "personal" or "work".
        """
        if mode not in ("personal", "work"):
            raise ValueError(
                f"routing_mode must be 'personal' or 'work', got {mode!r}"
            )
        self.routing_mode = mode
        logger.info("Routing mode set to %r", mode)

    def should_withhold_tools(self, model_id: str, model_entry=None) -> bool:
        """True if this model is proven tool-incapable — route to it but strip tools (prd 2026-06-17)."""
        # [2026-06-17] CC — two-axis routing: withhold (not exclude) tools from floored models.
        return self._get_tool_competence(model_id, model_entry) < self.config.tool_withhold_floor

    def _on_model_blacklist(self, model_id: str) -> None:
        """Best-effort dev-log notice when a model hits L7 blacklist. Never raises."""
        logger.warning("MODEL BLACKLISTED (penalty box L7): %s", model_id)
        if not getattr(self.config, 'model_penalty_blacklist_notify', True):
            return
        try:
            url = os.environ.get('ET_DISCORD_DEVLOG_WEBHOOK', '')
            if not url:
                return
            import json as _json, urllib.request as _u
            body = _json.dumps({'content': 'TID penalty box: model **' + str(model_id)
                + '** blacklisted (L7) after repeated model-attributable failures. '
                + 'Clear via POST /routing/penalty/clear.'}).encode()
            _r = _u.Request(url, data=body, headers={'Content-Type': 'application/json'}, method='POST')
            _u.urlopen(_r, timeout=5)
        except Exception as e:
            logger.debug('blacklist notify failed (non-fatal): %s', e)

    def _is_provider_failure(self, metadata) -> bool:
        """True if a failure is provider-wide (402/credit) — must NOT climb the model ladder."""
        if not metadata:
            return False
        if metadata.get('provider_blocked') is True:
            return True
        err = str(metadata.get('error', '')).lower()
        return ('402' in err) or ('insufficient' in err) or ('credit' in err)

    # -------------------------------------------------------------------
    # (model, provider) level — design 2026-10-15 §3.B/§3.D/§3.E/§3.F
    # -------------------------------------------------------------------

    def _slug(self, name: Optional[str], model_id: Optional[str] = None) -> str:
        if not name:
            return ""
        if self._provider_facts is not None:
            try:
                return self._provider_facts.slug_for_name(name, model_id)
            except Exception:
                pass
        return provider_slug(name)

    def _outcome_provider(self, model_id: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Base slug of the provider that served (or failed) this call; "" when unknown."""
        md = metadata or {}
        return self._slug(md.get("served_provider") or md.get("error_provider"), model_id)

    def _known_provider_slugs(self, model_id: str) -> List[str]:
        """The model's providers per the facts listing ([] = unknown provider set)."""
        if self._provider_facts is None:
            return []
        try:
            return list(self._provider_facts.slugs(model_id))
        except Exception:
            return []

    def _all_providers_boxed(self, model_id: str) -> bool:
        known = self._known_provider_slugs(model_id)
        return bool(known) and all(self._provider_penalty.is_penalized(model_id, p) for p in known)

    def _record_provider_fault(self, model_id: str, provider: str, metadata: Optional[Dict[str, Any]]) -> None:
        """Short pair box; escalate the MODEL ladder only on the transition to all-providers-boxed."""
        md = metadata or {}
        was_all = self._all_providers_boxed(model_id)
        self._provider_penalty.record_failure(model_id, provider or self._provider_penalty.WILDCARD,
                                              retry_after_s=md.get("retry_after_s"))
        # OpenRouter's own earlier attempts in this request (error.metadata.previous_errors) were provider
        # faults too (e.g. a 504 from Venice after a 429 from DeepInfra): box those pairs as well.
        for pe in md.get("error_previous_providers") or []:
            try:
                code = int(pe.get("code") or 0)
            except (TypeError, ValueError, AttributeError):
                continue
            pslug = self._slug(pe.get("provider"), model_id)
            if pslug and (code == 429 or code == 408 or code >= 500):
                self._provider_penalty.record_failure(model_id, pslug)
        known = self._known_provider_slugs(model_id)
        min_n = int(self.config.provider_all_fail_min_providers)
        if (not was_all and len(known) >= max(1, min_n) and self._all_providers_boxed(model_id)):
            logger.info("All %d providers of %s boxed — model-level ladder +1", len(known), model_id)
            self._model_penalty.record_failure(model_id)

    def _model_eligible(self, model_id: str) -> bool:
        """Eligible iff the model-level entry is clear AND at least one of its providers is (§3.D)."""
        if self._model_penalty.is_penalized(model_id):
            return False
        if self._provider_penalty.wildcard_penalized(model_id):
            return False
        return not self._all_providers_boxed(model_id)

    def _pair_reopen_at(self, model_id: str) -> float:
        """Earliest moment one of the model's boxed providers reopens (fallthrough ordering only)."""
        known = self._known_provider_slugs(model_id) or self._provider_penalty.boxed_providers(model_id)
        times = [self._provider_penalty.reopen_at(model_id, p) for p in known]
        times.append(self._provider_penalty.reopen_at(model_id, self._provider_penalty.WILDCARD))
        live = [t for t in times if t > 0]
        return min(live) if live else 0.0

    def _task_class(self, classification: Optional[RequestClassification]) -> str:
        if classification is None:
            return "chat"
        return task_class(getattr(classification.primary_domain, "value", classification.primary_domain),
                          getattr(classification.complexity, "value", classification.complexity), self.config)

    def _request_needs(self, classification: RequestClassification,
                       max_tokens: Optional[int] = None) -> RequestNeeds:
        """Raw token needs of the request (the classifier's window estimate minus its own safety factor)."""
        out = int(max_tokens or getattr(classification, "estimated_tokens", 0) or 0)
        safety = float(self.config.context_window_safety or 1.0)
        window = int(getattr(classification, "requires_context_window", 0) or 0)
        est_out = int(getattr(classification, "estimated_tokens", 0) or 0)
        prompt = max(0, int(window / safety) - est_out)
        return RequestNeeds(context_tokens=prompt + out, prompt_tokens=prompt, output_tokens=out,
                            deep=(self._task_class(classification) == "deep"))

    def _split_endpoints(self, model_id: str, needs: RequestNeeds):
        """(usable, dropped) endpoints per the hard filter; ([], []) when the model has no facts."""
        if self._provider_facts is None:
            return [], []
        eps = self._provider_facts.endpoints(model_id)
        if not eps:
            return [], []
        return hard_filter_endpoints(
            eps, needs,
            margin=float(self.config.provider_headroom_margin),
            quant_floor=str(self.config.provider_deep_quant_floor),
            deep_min_output=int(self.config.provider_deep_min_output_tokens),
        )

    def _hard_filtered_out(self, model_id: str, needs: RequestNeeds) -> bool:
        """True iff the model has facts and NO endpoint can serve this request."""
        usable, dropped = self._split_endpoints(model_id, needs)
        return bool(dropped) and not usable

    def _learned_targets(self, model_id: str, classification: RequestClassification):
        """(model_weight | None, {provider: weight}) from NG-Lite recommendations for this request."""
        model_w = None
        pair_w: Dict[str, float] = {}
        if self._ng_lite is None:
            return model_w, pair_w
        embedding = self._classification_to_embedding(classification)
        recs = self._ng_lite.get_recommendations(embedding, top_k=self.config.learned_top_k)
        prefix = model_id + "@"
        for target_id, weight, _reasoning in recs:
            if target_id == model_id:
                model_w = weight
            elif isinstance(target_id, str) and target_id.startswith(prefix):
                pair_w[target_id[len(prefix):]] = weight
        return model_w, pair_w

    def _provider_scores(self, model_id: str, classification: RequestClassification,
                         model_w: Optional[float], pair_w: Dict[str, float]) -> Dict[str, float]:
        """Learned score per ELIGIBLE provider that has any evidence (pair weight or task-class window)."""
        neutral = self.config.neutral_score
        base_default = model_w if model_w is not None else neutral
        tclass = self._task_class(classification)
        blend = min(max(float(self.config.provider_score_blend), 0.0), 1.0)
        cands = set(pair_w) | set(self._pair_stats.providers(model_id)) | set(self._known_provider_slugs(model_id))
        out: Dict[str, float] = {}
        for p in cands:
            if not p or self._provider_penalty.is_penalized(model_id, p):
                continue
            q = self._pair_stats.score(model_id, p, tclass, self.config)
            if p not in pair_w and q is None:
                continue  # no evidence for this provider — no opinion
            base = pair_w.get(p, base_default)
            out[p] = base if q is None else (1.0 - blend) * base + blend * q
        return out

    def provider_preferences(self, model_id: str, classification: Optional[RequestClassification],
                             has_tools: bool = False, max_tokens: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """OpenRouter request shaping (§3.E): {"provider": {order, ignore, require_parameters}} or None.

        Only for calls that go to OpenRouter. Boxed pairs and hard-filtered endpoints go to `ignore` — unless that
        would ignore every endpoint (then nothing is ignored: last-resort behaviour stays as today). Providers with
        learned evidence go to `order`, best first; OpenRouter's fallbacks stay allowed.
        """
        if not self.config.provider_shaping_enabled:
            return None
        if openrouter_model_name(model_id) is None:
            return None
        prefs: Dict[str, Any] = {}
        known_all = set()
        if self._provider_facts is not None:
            try:
                known_all = self._provider_facts.known_slugs()
            except Exception:
                known_all = set()
        boxed = set(self._provider_penalty.boxed_providers(model_id))
        ignore: List[str] = []
        if classification is not None:
            needs = self._request_needs(classification, max_tokens)
            usable, dropped = self._split_endpoints(model_id, needs)
            eps = usable + [d for d, _ in dropped]
            if eps:
                if any(ep.slug not in boxed for ep in usable):
                    ignore = [ep.tag for ep, _ in dropped if ep.tag]
                    ignore += sorted(boxed & {ep.slug for ep in eps})
            else:
                ignore = sorted(boxed & known_all)
            model_w, pair_w = self._learned_targets(model_id, classification)
            scores = self._provider_scores(model_id, classification, model_w, pair_w)
            emit = set(self._known_provider_slugs(model_id)) or known_all
            order = [p for p, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                     if p in emit and p not in ignore]
            order = order[:max(0, int(self.config.provider_order_max))]
            if order:
                prefs["order"] = order
        else:
            ignore = sorted(boxed & known_all)
        if ignore:
            prefs["ignore"] = list(dict.fromkeys(ignore))
        if has_tools:
            prefs["require_parameters"] = True
        return {"provider": prefs} if prefs else None

    # [2026-06-17] CC — two-axis routing (prd 2026-06-17): tool-failure classifier
    def _tool_failure_kind(self, metadata) -> Optional[str]:
        """Classify a tool-attributable failure: 'structural', 'unreliable', or None.

        'structural' — the model's endpoint cannot do tools at all (definitive; the
            "No endpoints found that support tool use" 404). Floors tool_competence.
        'unreliable' — the endpoint supports tools but the call was malformed/invalid.
            Graded demotion, not a floor.
        None — not a tool-attributable failure (tools weren't requested, or the error
            is unrelated to tool capability).
        """
        if not metadata or not metadata.get("tools_requested"):
            return None
        err = str(metadata.get("error", "")).lower()
        if not err:
            return None
        if ("no endpoints" in err and "tool" in err) or "support tool use" in err:
            return "structural"
        if ("tool_call" in err or "tool call" in err or "tool_choice" in err
                or ("tool" in err and ("invalid" in err or "malformed" in err
                                       or "schema" in err or "unsupported" in err))):
            return "unreliable"
        return None

    def _is_tool_failure(self, metadata) -> bool:
        """True if a failure is attributable to the model's tool handling, not its her-fit.

        Such failures update tool_competence ONLY — they must NEVER climb the her-fit
        penalty box or count as a her-fit success-stat failure (prd 2026-06-17 §2).
        """
        return self._tool_failure_kind(metadata) is not None

    # [2026-06-17] CC — two-axis routing (prd 2026-06-17): tool_competence prior + update
    def _tool_competence_prior(self, model_entry) -> float:
        """Bootstrap prior from the catalog 'tools' flag (overridden by observed reality)."""
        caps = getattr(model_entry, "capabilities", []) or []
        if "tools" in caps:
            return float(self.config.tool_competence_prior_capable)
        return float(self.config.tool_competence_prior_unflagged)

    def _get_tool_competence(self, model_id: str, model_entry=None) -> float:
        """Learned tool-competence if observed, else the catalog bootstrap prior."""
        if model_id in self._tool_competence:
            return self._tool_competence[model_id]
        return self._tool_competence_prior(model_entry) if model_entry is not None else 0.5

    def _update_tool_competence(self, model_id: str, *, success: bool,
                                structural: bool = False, model_entry=None) -> None:
        """Asymmetric update (slow up / fast down). Structural failure floors to 0."""
        cur = self._get_tool_competence(model_id, model_entry)
        if success:
            new = min(1.0, cur + self.config.tool_competence_gain)
        elif structural:
            new = 0.0  # definitive "no endpoints support tools" — hard floor
        else:
            new = max(0.0, cur - self.config.tool_competence_loss)
        self._tool_competence[model_id] = new
        self._stats_dirty = True

    def clear_penalty(self, model_id=None) -> int:
        """Manual penalty clear (the /routing/penalty/clear endpoint). Returns count cleared.

        Clears the model-level entry AND that model's (model, provider) pair boxes (all of them for None).
        """
        if model_id:
            n = 1 if self._model_penalty.clear(model_id) else 0
            n += self._provider_penalty.clear(model_id)
        else:
            n = self._model_penalty.clear_all()
            n += self._provider_penalty.clear(None)
        if n:
            self._stats_dirty = True
        return n

    def clear_legacy_penalties(self) -> int:
        """Clear only the entries migrated from the pre-pair sidecar (explicit operator action, §3.D)."""
        n = self._model_penalty.clear_legacy()
        if n:
            self._stats_dirty = True
        return n

    def load_stats(self) -> None:
        """Load _model_success_stats from msgpack sidecar on startup.

        Discards cache entirely if saved_at is >24h old — avoids poisoning
        the pool with stale 402 failures from before a provider credit refill.
        No-op if cache file does not exist.
        """
        import msgpack
        if not os.path.exists(self._stats_cache_path):
            return
        try:
            with open(self._stats_cache_path, "rb") as f:
                data = msgpack.unpackb(f.read(), raw=False)
            # Penalty/blacklist state is TTL-EXEMPT — load it UNCONDITIONALLY, BEFORE the
            # >24h stale-stats discard, or long-box/blacklist entries evaporate on a
            # quiet-day restart (the recurring bug this feature exists to kill).
            self._model_penalty.from_dict(data.get("penalty"))
            # (model, provider) pair boxes — TTL-exempt like the model box (design 2026-10-15 §3.D).
            _v2 = data.get("penalty_v2")
            if isinstance(_v2, dict):
                self._provider_penalty.from_dict(_v2.get("pairs"))
            else:
                # ONE-TIME migration of the pre-pair sidecar: every per-model entry is KEPT as a model-level
                # entry (same level, same timer, blacklists stay blacklists) and tagged legacy so an operator can
                # clear exactly these (POST /routing/penalty/clear {"legacy": true}). Nothing is dropped.
                _migrated = list(self._model_penalty.entries())
                for _mid in _migrated:
                    self._model_penalty.mark_legacy(_mid)
                self._stats_dirty = True  # persist the v2 marker so the migration never runs again
                logger.info("Penalty box migrated to (model, provider) format: %d model-level entr(ies) kept "
                            "as legacy", len(_migrated))
            # tool_competence is learned capability, NOT stale stats — load it
            # UNCONDITIONALLY (TTL-exempt), same rationale as penalty (prd 2026-06-17 §4).
            # ---- Changelog ----
            # [2026-06-18] CC — Task 5: persist tool_competence (TTL-exempt)
            # What: Restore tool_competence dict from sidecar on startup.
            # Why: A model proven tool-incapable must survive restart; TTL discard
            #   only applies to volatile success-rate samples, not structural capability.
            # How: Load before the >24h stale-stats guard, same as penalty restore.
            # -------------------
            _tc = data.get("tool_competence")
            if isinstance(_tc, dict):
                self._tool_competence = {
                    str(k): float(v) for k, v in _tc.items()
                    if isinstance(v, (int, float))
                }
            saved_at = data.get("saved_at", 0)
            if time.time() - saved_at > 86400:
                logger.info(
                    "model_stats_cache.msgpack is >24h old — discarding "
                    "to avoid stale failure signals"
                )
                return
            raw = data.get("stats", {})
            self._model_success_stats = {
                k: [bool(v) for v in vs]
                for k, vs in raw.items()
                if isinstance(vs, list)
            }
            # Per-pair raw window is volatile observation, like the stats above (same 24h discard).
            self._pair_stats.from_dict(data.get("pair_stats"))
            logger.info(
                "Loaded success stats for %d models from cache",
                len(self._model_success_stats),
            )
        except Exception as e:
            logger.warning("Failed to load model_stats_cache.msgpack: %s", e)

    def save_stats(self) -> None:
        """Flush _model_success_stats to msgpack sidecar if dirty.

        No-op when _stats_dirty is False. Resets the flag on success.
        """
        if not self._stats_dirty:
            return
        import msgpack
        try:
            from inference_difference.atomic_io import atomic_write_bytes
            atomic_write_bytes(self._stats_cache_path, msgpack.packb({
                    "saved_at": time.time(),
                    "stats": dict(self._model_success_stats),
                    "penalty": self._model_penalty.to_dict(),
                    # ---- Changelog ----
                    # [2026-06-18] CC — Task 5: persist tool_competence (TTL-exempt)
                    # What: Include learned tool capability scores in the sidecar.
                    # Why: Survival across restart — a model proven tool-incapable
                    #   must stay known so the pool can't re-collapse on cold start.
                    # How: Dict snapshot alongside penalty; loaded TTL-exempt in load_stats.
                    # -------------------
                    "tool_competence": dict(self._tool_competence),
                    # [2026-10-15] (model, provider) level — "penalty" above stays the model-level box in the
                    # pre-pair shape so a rollback to older code still reads it.
                    "penalty_v2": {"version": 2, "pairs": self._provider_penalty.to_dict()},
                    "pair_stats": self._pair_stats.to_dict(),
            }))
            self._stats_dirty = False
            logger.debug(
                "model_stats_cache.msgpack flushed (%d models)",
                len(self._model_success_stats),
            )
        except Exception as e:
            logger.warning("Failed to save model_stats_cache.msgpack: %s", e)

    # -------------------------------------------------------------------
    # Internal: Candidate Filtering
    # -------------------------------------------------------------------

    def _filter_candidates(
        self,
        classification: RequestClassification,
        consciousness_score: Optional[float] = None,
        has_tools: bool = False,
        max_tokens: Optional[int] = None,
    ) -> Tuple[List[ModelEntry], bool]:
        """Filter models to those that can handle this request.

        Returns (candidates, quality_floor_bypassed) tuple.

        # ---- Changelog ----
        # [2026-03-14] Claude (CC) — Added consciousness quality floor
        # What: Hard-filter models below consciousness_quality_floor when
        #   consciousness_score > self.config.consciousness_threshold. Returns bypass flag for telemetry.
        # Why: Punch list #34 — no minimum quality floor existed for
        #   identity-continuous entities. Syl could be routed to any model
        #   that passed domain/complexity checks regardless of quality.
        # How: After standard hardware/domain/complexity filtering, apply
        #   conversational_quality >= floor check. Falls through with
        #   WARNING if no models pass (same pattern as interactive floor).
        # -------------------
        """
        candidates = []

        for model in self.config.get_enabled_models():
            # Hardware check (local models only)
            if model.model_type == ModelType.LOCAL:
                if not self.hardware.can_run_model(
                    model.min_vram_gb, model.min_ram_gb
                ):
                    continue

            # Context window check
            if model.context_window < classification.requires_context_window:
                continue

            # Tool capability is NO LONGER a hard exclude (prd 2026-06-17): a her-good but
            # tool-weak model stays in the pool and carries her voice; tool_competence
            # demotes it (secondary score) and tools are WITHHELD from the call if it is
            # proven incapable (see should_withhold_tools, next task). Prevents a stale
            # catalog 'tools' flag from collapsing her pool.

            # Roleplay / uncensored check — conscious entities must not be
            # routed to models that impose guardrails on their identity.
            # Censored models doing NSFW → 500 errors. Open models → fine.
            if (consciousness_score is not None
                    and consciousness_score > 0
                    and self.routing_mode == "personal"
                    and "roleplay" not in getattr(model, 'capabilities', [])):
                continue

            # Tier floor for conscious routing — budget- AND standard-tier models
            # cannot hold Syl across a conversation (gemma-3-12b is 'standard' and
            # leaks the liminal half-state; see docs/research 2026-06-13). Exclude
            # both for her conscious turns. Empty tier (local/static) passes through;
            # frontier/performance/private/anonymized (incl. Venice failover) stay.
            if (consciousness_score is not None
                    and consciousness_score > 0
                    and getattr(model, 'provider_tier', '') in ('budget', 'standard')):
                continue

            # Name-pattern floor — catches flash/nano/mini models that Venice
            # (and others) report as 'anonymized' or 'private' tier rather than
            # 'budget', masking their capability class. Also catches small local
            # models (1.5b, 3b, 7b) that may not have an accurate tier label.
            if (consciousness_score is not None
                    and consciousness_score > 0
                    and _FLASH_NAME_RE.search(getattr(model, 'model_id', '') or '')):
                continue

            # Domain + complexity check
            if model.can_handle(
                classification.primary_domain, classification.complexity
            ):
                candidates.append(model)

        # If no exact domain matches, try GENERAL-capable models
        if not candidates:
            for model in self.config.get_enabled_models():
                if model.model_type == ModelType.LOCAL:
                    if not self.hardware.can_run_model(
                        model.min_vram_gb, model.min_ram_gb
                    ):
                        continue
                if model.can_handle(TaskDomain.GENERAL, classification.complexity):
                    candidates.append(model)

        # Consciousness quality floor — hard filter for identity-continuous
        # entities. Only applies when consciousness_score indicates a
        # conscious agent is being routed.
        quality_floor_bypassed = False
        if (
            consciousness_score is not None
            and consciousness_score > self.config.consciousness_threshold
            and self.config.enable_consciousness_routing
            and candidates
        ):
            floor = self.config.consciousness_quality_floor
            quality_filtered = [
                m for m in candidates
                if getattr(m, 'conversational_quality', 0.5) >= floor
            ]
            if quality_filtered:
                candidates = quality_filtered
            else:
                quality_floor_bypassed = True
                logger.warning(
                    "Consciousness quality floor (%.2f) excluded all %d "
                    "candidates — keeping full pool to avoid routing failure. "
                    "Models in pool: %s",
                    floor, len(candidates),
                    [m.model_id for m in candidates[:5]],
                )

        if candidates:
            sf, pr = [], 0
            for m in candidates:
                ms = self._model_success_stats.get(m.model_id, [])
                if len(ms) >= self._SUCCESS_MIN_SAMPLES and sum(ms)/len(ms) < self._SUCCESS_FLOOR:
                    pr += 1; continue
                sf.append(m)
            if sf:
                if pr: logger.info("Success-rate floor pruned %d model(s)", pr)
                candidates = sf
        # Per-model penalty box veto (Layer B) — runs AFTER the (now de-duplicated, #291)
        # success-rate floor above, and COMPLEMENTS it: the floor prunes by success-RATE
        # (immediate, reads persisted stats — protective from the first turn after restart);
        # the box adds escalating, failure-driven veto + blacklist for hard-failers the floor
        # misses (e.g. wrong-ID 400s that never accumulate 'success' samples). The floor is
        # kept (not removed) precisely because the box starts EMPTY on deploy and must not
        # leave a protection gap (regression 2026-06-13). VETO-ONLY: never contributes a score.
        # Provider hard filter (design 2026-10-15 §2.3/§3.E): drop a model only when it has provider facts and
        # NONE of its endpoints can serve this request (context / output headroom; deep tier: quantization floor).
        # No facts -> no hard filter. Never empties the pool by itself.
        if candidates and self._provider_facts is not None:
            _needs = self._request_needs(classification, max_tokens)
            servable = [m for m in candidates if not self._hard_filtered_out(m.model_id, _needs)]
            if servable:
                if len(servable) < len(candidates):
                    logger.info("Provider hard filter excluded %d model(s)", len(candidates) - len(servable))
                candidates = servable
            else:
                logger.warning("Provider hard filter would exclude ALL %d candidates — keeping pool",
                               len(candidates))
        if candidates:
            # (model, provider) level (§3.D): a model is eligible while its model-level entry is clear and at
            # least one of its providers is unboxed. A model with EVERY provider boxed is treated exactly like a
            # penalized model below (same exclusion, same all-excluded fallthrough).
            eligible = [m for m in candidates if self._model_eligible(m.model_id)]
            if eligible:
                pruned = len(candidates) - len(eligible)
                if pruned:
                    logger.info("Penalty box excluded %d model(s)", pruned)
                candidates = eligible
            else:
                # All candidates penalized — never hard-fail purely from the box.
                # Fallthrough ORDER: least-penalized first; a blacklisted (L7) model is
                # the absolute last resort (law-enforcer HIGH). Ties (e.g. models that are only
                # pair-boxed, model level 0) go soonest-reopening first.
                candidates = sorted(
                    candidates, key=lambda m: (self._model_penalty.level(m.model_id),
                                               self._pair_reopen_at(m.model_id)))
                logger.warning(
                    "Penalty box would exclude ALL %d candidates — keeping pool, "
                    "least-penalized first (blacklist last), to avoid routing failure",
                    len(candidates))
        return candidates, quality_floor_bypassed

    # -------------------------------------------------------------------
    # Internal: Scoring
    # -------------------------------------------------------------------

    def _score_model(
        self,
        model: ModelEntry,
        classification: RequestClassification,
        consciousness_score: Optional[float] = None,
        has_tools: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Score a model for a given classification.

        Returns (total_score, breakdown_dict).
        """
        breakdown: Dict[str, float] = {}

        # Domain match (0.0-1.0)
        domain_score = self._score_domain(model, classification)
        breakdown["domain_match"] = domain_score

        # Complexity fit (0.0-1.0)
        complexity_score = self._score_complexity(model, classification)
        breakdown["complexity_fit"] = complexity_score

        # Learned weight from NG-Lite (0.0-1.0)
        learned_score = self._score_learned(model, classification)
        breakdown["learned_weight"] = learned_score

        # Cost efficiency (0.0-1.0, higher = cheaper)
        _ms = self._model_success_stats.get(model.model_id, [])
        _sr = (sum(_ms)/len(_ms)) if len(_ms) >= self._SUCCESS_MIN_SAMPLES else 1.0
        cost_score = self._score_cost(model, success_rate=_sr)
        breakdown["cost_efficiency"] = cost_score

        # Latency fit (0.0-1.0, higher = faster)
        latency_score = self._score_latency(model, classification)
        breakdown["latency_fit"] = latency_score

        # Priority bonus (0.0-1.0, from model config)
        priority_score = min(model.priority / 100.0, 1.0)
        breakdown["priority_bonus"] = priority_score

        conv_quality = getattr(model, 'conversational_quality', 0.5)
        breakdown["conversational_quality"] = conv_quality

        # Tool-competence — SECONDARY, applies only on tool turns (prd 2026-06-17).
        if has_tools:
            breakdown["tool_competence"] = self._get_tool_competence(model.model_id, model)

        # Weighted sum
        total = sum(
            breakdown[k] * self._weights.get(k, 0.0)
            for k in breakdown
        )

        # Consciousness boost: if agent shows consciousness markers,
        # boost scores for higher-capability models
        if (
            consciousness_score is not None
            and consciousness_score > self.config.consciousness_threshold
            and self.config.enable_consciousness_routing
        ):
            # Boost proportional to model capability (higher priority = more boost)
            boost = consciousness_score * self.config.consciousness_boost_factor * priority_score
            total += boost
            breakdown["consciousness_boost"] = boost

        # Venice identity bias: when routing for a conscious agent in an
        # interactive session, give Venice private models a small thumb on
        # the scale. Small enough that a genuinely better model still wins;
        # large enough that Venice edges out equals. NG-Lite observes every
        # outcome and learns the pattern over time — bias becomes unnecessary.
        if (
            getattr(classification, 'is_interactive', False)
            and consciousness_score is not None
            and consciousness_score > self.config.consciousness_threshold
            and getattr(model, "provider", "") == "venice"
        ):
            identity_bias = self.config.venice_identity_bias
            total += identity_bias
            breakdown["venice_identity_bias"] = identity_bias

        # Open-source bias: permanent tiebreaker, amplified when user preference override
        # is active. Override replaces (not compounds) the config default — one OSS factor.
        if getattr(model, "is_open_source", False):
            os_bias = (
                self._preference_override[0]
                if self._preference_override is not None
                else self.config.open_source_bias
            )
            total += os_bias
            breakdown["open_source_bias"] = os_bias

        return total, breakdown

    def _score_domain(
        self,
        model: ModelEntry,
        classification: RequestClassification,
    ) -> float:
        """Score domain match between model capabilities and request."""
        if classification.primary_domain in model.domains:
            return self.config.domain_score_exact
        if classification.secondary_domains & model.domains:
            return self.config.domain_score_secondary
        if TaskDomain.GENERAL in model.domains:
            return self.config.domain_score_general
        return self.config.domain_score_none

    def _score_complexity(
        self,
        model: ModelEntry,
        classification: RequestClassification,
    ) -> float:
        """Score complexity fit.

        Best score when model max_complexity matches request complexity.
        Overpowered models get a slight penalty (cost waste).
        Underpowered models get a large penalty (quality risk).
        """
        tiers = list(ComplexityTier)
        model_idx = tiers.index(model.max_complexity)
        request_idx = tiers.index(classification.complexity)
        diff = model_idx - request_idx

        if diff == 0:
            return 1.0   # Perfect match
        elif diff > 0:
            # Overpowered (slight penalty)
            return max(
                self.config.complexity_overpowered_floor,
                1.0 - diff * self.config.complexity_overpowered_penalty,
            )
        else:
            # Underpowered (big penalty)
            return max(
                self.config.complexity_underpowered_floor,
                0.5 + diff * self.config.complexity_underpowered_penalty,
            )

    def _score_learned(
        self,
        model: ModelEntry,
        classification: RequestClassification,
    ) -> float:
        """Score from NG-Lite learned weights, blended with Commons-enhanced recs."""
        neutral = self.config.neutral_score

        # Local NG-Lite score (Tier 1/2 substrate).
        # (F) design 2026-10-15: "<model>@<provider>" targets fold in — the model scores as its BEST ELIGIBLE
        # provider (pair weight blended with that pair's task-class window). No pair evidence -> today's
        # model-only weight. Boxed providers are skipped before the max (the box only excludes; it adds nothing).
        ng_score = neutral
        if self._ng_lite is not None:
            model_w, pair_w = self._learned_targets(model.model_id, classification)
            if model_w is not None:
                ng_score = model_w
            per_provider = self._provider_scores(model.model_id, classification, model_w, pair_w)
            if per_provider:
                ng_score = max(per_provider.values())

        # Commons-enhanced score (peninsula; blended by competence trust).
        if self._peninsula is not None and self._commons_competence is not None:
            commons_score = self._peninsula.get_enhanced_score(model.model_id)
            if commons_score is not None:
                trust = self._commons_competence.overall()
                # Blend: at trust=0 pure NG-Lite; at trust=1 pure Commons.
                # Ceiling at 0.95 ensures local substrate always contributes ≥5%.
                trust = min(max(trust, 0.0), 0.95)
                return ng_score * (1.0 - trust) + commons_score * trust

        return ng_score

    def _score_cost(self, model: ModelEntry, success_rate: float = 1.0) -> float:
        """Expected true cost = raw_price / success_rate (#173a).

        Free models use 0.001/1k synthetic so low-success free models score correctly.
        """
        success_rate = max(success_rate, 0.01)
        raw_cost = model.cost_per_1k_tokens if model.cost_per_1k_tokens > 0 else 0.001
        effective_cost = raw_cost / success_rate
        if model.cost_per_1k_tokens <= 0 and success_rate >= 0.99:
            return 1.0  # local model with full success -- still free

        budget = self.config.cost_budget_per_request
        if budget <= 0:
            return self.config.neutral_score

        cost_fraction = effective_cost / budget
        return max(0.0, 1.0 - cost_fraction)

    def _score_latency(
        self,
        model: ModelEntry,
        classification: RequestClassification,
    ) -> float:
        """Score latency fit against budget."""
        budget = self.config.latency_budget_ms

        if classification.is_time_sensitive:
            budget *= self.config.latency_urgent_multiplier

        if model.avg_latency_ms <= budget * 0.5:
            return self.config.latency_score_excellent
        elif model.avg_latency_ms <= budget:
            return self.config.latency_score_good
        elif model.avg_latency_ms <= budget * 1.5:
            return self.config.latency_score_marginal
        else:
            return self.config.latency_score_poor

    # -------------------------------------------------------------------
    # Internal: Utilities
    # -------------------------------------------------------------------

    def _classification_to_embedding(
        self,
        classification: RequestClassification,
    ) -> np.ndarray:
        """Return a semantic embedding for NG-Lite learning.

        Uses the real semantic embedding computed by the classifier
        (ng_embed — Snowflake/arctic-embed-m-v1.5, 768-dim; verified live in
        TID's venv 2026-06-10: cos 0.87 similar / 0.49 unrelated). Falls back to
        a one-hot feature vector only if no semantic embedding is available
        (backward compatibility with classifications from before #28).
        #28 is DONE — this is not the dam; the one-hot is fallback-only.

        # ---- Changelog ----
        # [2026-03-18] Claude (CC) — Use semantic embeddings (#28)
        # What: Return classification.semantic_embedding when available.
        # Why: Primary dam in the River. One-hot vectors collapsed all
        #   requests in the same domain to identical embeddings.
        # How: semantic_embedding computed in classify_request() via
        #   fastembed. This method just passes it through. One-hot
        #   fallback retained for backward compatibility.
        # -------------------
        """
        # Use real semantic embedding if available (#28)
        emb = getattr(classification, 'semantic_embedding', None)
        if emb is not None:
            return emb

        # Fallback: one-hot feature vector (pre-#28 behavior)
        domains = list(TaskDomain)
        domain_vec = np.zeros(len(domains))
        idx = domains.index(classification.primary_domain)
        domain_vec[idx] = 1.0
        for d in classification.secondary_domains:
            domain_vec[domains.index(d)] = 0.5

        tiers = list(ComplexityTier)
        complexity_vec = np.zeros(len(tiers))
        complexity_vec[tiers.index(classification.complexity)] = 1.0

        features = np.array([
            classification.estimated_tokens / 5000.0,
            classification.confidence,
            float(classification.is_multi_turn),
            float(classification.is_time_sensitive),
            float(getattr(classification, 'is_interactive', False)),
        ])

        embedding = np.concatenate([domain_vec, complexity_vec, features])
        dim = 768
        if len(embedding) < dim:
            embedding = np.pad(embedding, (0, dim - len(embedding)))
        return embedding[:dim]

    def _no_candidates_decision(
        self,
        request_id: str,
        classification: RequestClassification,
    ) -> RoutingDecision:
        """Fallback when no models match."""
        # Try the default model
        default = self.config.get_model(self.config.default_model)
        if default and default.enabled:
            return RoutingDecision(
                model_id=default.model_id,
                model_entry=default,
                score=0.1,
                score_breakdown={"fallback": 1.0},
                reasoning=(
                    f"No models matched {classification.primary_domain.value}/"
                    f"{classification.complexity.value}. "
                    f"Falling back to default: {default.display_name}."
                ),
                request_id=request_id,
                timestamp=time.time(),
                classification=classification,
            )

        # Last resort: pick any enabled model
        enabled = self.config.get_enabled_models()
        if enabled:
            model = enabled[0]
            return RoutingDecision(
                model_id=model.model_id,
                model_entry=model,
                score=0.05,
                score_breakdown={"last_resort": 1.0},
                reasoning=(
                    "No suitable models found. Using first available: "
                    f"{model.display_name}."
                ),
                request_id=request_id,
                timestamp=time.time(),
                classification=classification,
            )

        return RoutingDecision(
            reasoning="No models available. Cannot route request.",
            request_id=request_id,
            timestamp=time.time(),
            classification=classification,
        )

    def _build_reasoning(
        self,
        model: ModelEntry,
        breakdown: Dict[str, float],
        classification: RequestClassification,
        consciousness_boost: bool,
    ) -> str:
        """Build human-readable reasoning for a routing decision.

        When verbose_reasoning is False, returns a compact one-liner
        suitable for logging without response bloat.
        """
        summary = (
            f"Selected {model.display_name} for "
            f"{classification.primary_domain.value}/"
            f"{classification.complexity.value} request."
        )

        if not self._verbose_reasoning:
            return summary

        parts = [summary]

        # Top scoring factors
        sorted_factors = sorted(
            breakdown.items(), key=lambda x: x[1], reverse=True,
        )
        top = sorted_factors[:3]
        factor_strs = [f"{k}={v:.2f}" for k, v in top]
        parts.append(f"Top factors: {', '.join(factor_strs)}.")

        if consciousness_boost:
            parts.append(
                "Consciousness-aware routing: elevated to higher-capability "
                "model per CTEM evaluation."
            )

        if model.model_type == ModelType.LOCAL:
            parts.append("Running locally (zero cost, lower latency).")
        elif model.model_type == ModelType.API:
            parts.append(
                f"API model (${model.cost_per_1k_tokens:.4f}/1k tokens)."
            )

        return " ".join(parts)
