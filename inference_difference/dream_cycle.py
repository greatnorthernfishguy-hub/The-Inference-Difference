"""
Dream Cycle Enhancement for Dynamic Model Catalog Selection (§4.5.5).

With dynamic catalog selection, the Dream Cycle gains additional learning
capability beyond tier assignment. It analyzes which model properties
correlate with routing success for each semantic route, and tightens
profile requirements based on learned patterns.

Examples of learnable insights:
- "Coding tasks with context > 8K succeed more with context_window > 32K"
- "Analysis tasks retry less with models in 'performance' provider tier"
- "Simple chat tasks under 100 tokens always succeed with local model"

The DreamCycle runs periodically (or on demand) to analyze recent
routing outcomes and update profile requirements.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("inference_difference.dream_cycle")


@dataclass
class RoutingOutcome:
    """A recorded routing outcome for Dream Cycle analysis."""
    request_id: str = ""
    model_id: str = ""
    semantic_route: str = ""           # e.g. "coding", "simple_chat"
    success: bool = False
    quality_score: float = 0.0
    latency_ms: float = 0.0
    # Model properties at time of routing
    model_context_window: int = 0
    model_cost_per_1m_input: float = 0.0
    model_cost_per_1m_output: float = 0.0
    model_provider_tier: str = ""
    model_capabilities: List[str] = field(default_factory=list)
    model_provider: str = ""
    # Request properties
    estimated_tokens: int = 0
    request_complexity: str = ""


@dataclass
class PropertyInsight:
    """A learned insight about model properties and routing success."""
    property_name: str = ""            # e.g. "context_window", "provider_tier"
    observation: str = ""              # Human-readable description
    recommendation: str = ""           # What to change in the profile
    confidence: float = 0.0           # How confident we are (0.0-1.0)
    sample_size: int = 0              # How many outcomes informed this


class DreamCycle:
    """Analyzes routing outcomes to learn model property correlations (§4.5.5).

    After analyzing retry/thumbs_down failures, looks for patterns
    in model properties that predicted success or failure.

    Usage:
        cycle = DreamCycle()

        # Record outcomes as they happen
        cycle.record_outcome(outcome)

        # Periodically analyze for insights
        insights = cycle.analyze_model_property_correlations()

        # Apply insights to profiles (optional auto-tuning)
        cycle.apply_insights_to_profiles(catalog_manager)
    """

    def __init__(
        self,
        min_sample_size: int = 10,
        confidence_threshold: float = 0.6,
        max_outcomes: int = 5000,
    ):
        self.min_sample_size = min_sample_size
        self.confidence_threshold = confidence_threshold
        self.max_outcomes = max_outcomes

        self._outcomes: List[RoutingOutcome] = []
        self._insights: Dict[str, List[PropertyInsight]] = {}

    def record_outcome(self, outcome: RoutingOutcome) -> None:
        """Record a routing outcome for later analysis."""
        self._outcomes.append(outcome)
        if len(self._outcomes) > self.max_outcomes:
            # Keep the most recent outcomes
            self._outcomes = self._outcomes[-self.max_outcomes:]

    def analyze_model_property_correlations(
        self,
    ) -> Dict[str, List[PropertyInsight]]:
        """Analyze recent outcomes for model property patterns.

        Groups outcomes by semantic route, then compares model properties
        between successful and failed requests to find patterns.

        Returns:
            Dict mapping semantic route -> list of PropertyInsights.
        """
        # Group outcomes by semantic route
        by_route: Dict[str, List[RoutingOutcome]] = defaultdict(list)
        for outcome in self._outcomes:
            if outcome.semantic_route:
                by_route[outcome.semantic_route].append(outcome)

        all_insights: Dict[str, List[PropertyInsight]] = {}

        for route, outcomes in by_route.items():
            if len(outcomes) < self.min_sample_size:
                continue

            failures = [o for o in outcomes if not o.success]
            successes = [o for o in outcomes if o.success]

            if not failures or not successes:
                continue

            route_insights = []

            # Analyze context window correlation
            insight = self._analyze_context_window(route, failures, successes)
            if insight:
                route_insights.append(insight)

            # Analyze provider tier correlation
            insight = self._analyze_provider_tier(route, failures, successes)
            if insight:
                route_insights.append(insight)

            # Analyze cost correlation
            insight = self._analyze_cost(route, failures, successes)
            if insight:
                route_insights.append(insight)

            # Analyze capabilities correlation
            insight = self._analyze_capabilities(route, failures, successes)
            if insight:
                route_insights.append(insight)

            if route_insights:
                all_insights[route] = route_insights
                for insight in route_insights:
                    logger.info(
                        "Dream Cycle insight for '%s': %s (confidence=%.2f)",
                        route, insight.observation, insight.confidence,
                    )

        self._insights = all_insights
        return all_insights

    def get_insights(self) -> Dict[str, List[PropertyInsight]]:
        """Return the most recent analysis insights."""
        return self._insights

    def get_stats(self) -> Dict[str, Any]:
        """Return Dream Cycle statistics."""
        route_counts: Dict[str, int] = defaultdict(int)
        for o in self._outcomes:
            route_counts[o.semantic_route] += 1

        total_insights = sum(
            len(insights) for insights in self._insights.values()
        )

        return {
            "total_outcomes": len(self._outcomes),
            "routes_tracked": dict(route_counts),
            "total_insights": total_insights,
            "insights_by_route": {
                route: len(insights)
                for route, insights in self._insights.items()
            },
        }

    # -------------------------------------------------------------------
    # Property analysis methods
    # -------------------------------------------------------------------

    def _analyze_context_window(
        self,
        route: str,
        failures: List[RoutingOutcome],
        successes: List[RoutingOutcome],
    ) -> Optional[PropertyInsight]:
        """Check if context window size correlates with success."""
        if not failures or not successes:
            return None

        avg_fail_ctx = sum(
            f.model_context_window for f in failures
        ) / len(failures)
        avg_success_ctx = sum(
            s.model_context_window for s in successes
        ) / len(successes)

        # Significant difference: success models have >50% more context
        if avg_fail_ctx > 0 and avg_success_ctx > avg_fail_ctx * 1.5:
            confidence = min(
                len(successes) / (len(successes) + len(failures)),
                1.0,
            )
            if confidence >= self.confidence_threshold:
                return PropertyInsight(
                    property_name="context_window",
                    observation=(
                        f"Route '{route}': successful models avg "
                        f"{int(avg_success_ctx)} context vs "
                        f"{int(avg_fail_ctx)} for failures"
                    ),
                    recommendation=(
                        f"min_context_window >= {int(avg_success_ctx * 0.8)}"
                    ),
                    confidence=confidence,
                    sample_size=len(failures) + len(successes),
                )

        return None

    def _analyze_provider_tier(
        self,
        route: str,
        failures: List[RoutingOutcome],
        successes: List[RoutingOutcome],
    ) -> Optional[PropertyInsight]:
        """Check if provider tier correlates with success."""
        fail_tiers: Dict[str, int] = defaultdict(int)
        success_tiers: Dict[str, int] = defaultdict(int)

        for f in failures:
            if f.model_provider_tier:
                fail_tiers[f.model_provider_tier] += 1
        for s in successes:
            if s.model_provider_tier:
                success_tiers[s.model_provider_tier] += 1

        if not fail_tiers or not success_tiers:
            return None

        # Find tiers that appear predominantly in failures but not successes
        problematic_tiers = []
        for tier, count in fail_tiers.items():
            fail_rate = count / len(failures)
            success_rate = success_tiers.get(tier, 0) / len(successes)
            if fail_rate > 0.5 and success_rate < 0.2:
                problematic_tiers.append(tier)

        # Find tiers that appear predominantly in successes
        good_tiers = []
        for tier, count in success_tiers.items():
            success_rate = count / len(successes)
            fail_rate = fail_tiers.get(tier, 0) / len(failures)
            if success_rate > 0.5 and fail_rate < 0.2:
                good_tiers.append(tier)

        if good_tiers:
            confidence = len(successes) / (len(successes) + len(failures))
            if confidence >= self.confidence_threshold:
                return PropertyInsight(
                    property_name="provider_tier",
                    observation=(
                        f"Route '{route}': tier(s) {good_tiers} correlate "
                        f"with success"
                    ),
                    recommendation=(
                        f"provider_tiers = {good_tiers}"
                    ),
                    confidence=confidence,
                    sample_size=len(failures) + len(successes),
                )

        return None

    def _analyze_cost(
        self,
        route: str,
        failures: List[RoutingOutcome],
        successes: List[RoutingOutcome],
    ) -> Optional[PropertyInsight]:
        """Check if cost correlates with success (maybe cheap = bad)."""
        avg_fail_cost = sum(
            f.model_cost_per_1m_input for f in failures
        ) / len(failures)
        avg_success_cost = sum(
            s.model_cost_per_1m_input for s in successes
        ) / len(successes)

        # If successful models cost significantly more, cheapest isn't best
        if avg_success_cost > avg_fail_cost * 2.0 and avg_fail_cost > 0:
            confidence = min(
                len(successes) / (len(successes) + len(failures)),
                1.0,
            )
            if confidence >= self.confidence_threshold:
                # Suggest raising the floor, not the ceiling
                min_viable = avg_fail_cost * 1.5
                return PropertyInsight(
                    property_name="cost",
                    observation=(
                        f"Route '{route}': cheapest models "
                        f"(avg ${avg_fail_cost:.2f}/1M) fail more than "
                        f"mid-range (avg ${avg_success_cost:.2f}/1M)"
                    ),
                    recommendation=(
                        f"Consider min_cost_per_1m_input >= "
                        f"${min_viable:.2f}"
                    ),
                    confidence=confidence,
                    sample_size=len(failures) + len(successes),
                )

        return None

    def _analyze_capabilities(
        self,
        route: str,
        failures: List[RoutingOutcome],
        successes: List[RoutingOutcome],
    ) -> Optional[PropertyInsight]:
        """Check if specific capabilities correlate with success."""
        fail_caps: Dict[str, int] = defaultdict(int)
        success_caps: Dict[str, int] = defaultdict(int)

        for f in failures:
            for cap in f.model_capabilities:
                fail_caps[cap] += 1
        for s in successes:
            for cap in s.model_capabilities:
                success_caps[cap] += 1

        # Find capabilities that successful models have but failures lack
        missing_in_failures = []
        for cap, count in success_caps.items():
            success_rate = count / len(successes)
            fail_rate = fail_caps.get(cap, 0) / len(failures)
            if success_rate > 0.7 and fail_rate < 0.3:
                missing_in_failures.append(cap)

        if missing_in_failures:
            confidence = len(successes) / (len(successes) + len(failures))
            if confidence >= self.confidence_threshold:
                return PropertyInsight(
                    property_name="capabilities",
                    observation=(
                        f"Route '{route}': capability "
                        f"{missing_in_failures} present in "
                        f"{len(successes)} successes but rare in "
                        f"{len(failures)} failures"
                    ),
                    recommendation=(
                        f"required_capabilities should include "
                        f"{missing_in_failures}"
                    ),
                    confidence=confidence,
                    sample_size=len(failures) + len(successes),
                )

        return None

    # -------------------------------------------------------------------
    # Profile auto-tuning
    # -------------------------------------------------------------------

    def apply_insights_to_profiles(
        self,
        catalog_manager: Any,
    ) -> List[str]:
        """Apply learned insights to profile requirements.

        Only applies insights above the confidence threshold.
        Returns list of changes made for logging.

        Note: This modifies profiles in-memory only. Changes are not
        persisted to task_requirements.yaml — that requires operator
        approval. The Dream Cycle suggests, the operator decides.
        """
        changes = []

        for route, insights in self._insights.items():
            if route not in catalog_manager.profiles:
                continue

            profile = catalog_manager.profiles[route]

            for insight in insights:
                if insight.confidence < self.confidence_threshold:
                    continue

                if (
                    insight.property_name == "context_window"
                    and "min_context_window >=" in insight.recommendation
                ):
                    # Extract recommended value
                    try:
                        val = int(
                            insight.recommendation
                            .split(">=")[1]
                            .strip()
                        )
                        if (
                            profile.min_context_window is None
                            or val > profile.min_context_window
                        ):
                            profile.min_context_window = val
                            changes.append(
                                f"{route}: min_context_window -> {val}"
                            )
                    except (ValueError, IndexError):
                        pass

                elif (
                    insight.property_name == "provider_tier"
                    and "provider_tiers =" in insight.recommendation
                ):
                    # Extract recommended tiers
                    try:
                        tiers_str = (
                            insight.recommendation
                            .split("=")[1]
                            .strip()
                        )
                        # Parse the list format
                        tiers = [
                            t.strip().strip("'\"")
                            for t in tiers_str.strip("[]").split(",")
                            if t.strip()
                        ]
                        if tiers:
                            profile.provider_tiers = tiers
                            changes.append(
                                f"{route}: provider_tiers -> {tiers}"
                            )
                    except (ValueError, IndexError):
                        pass

        if changes:
            logger.info(
                "Dream Cycle applied %d profile updates: %s",
                len(changes), changes,
            )

        return changes
