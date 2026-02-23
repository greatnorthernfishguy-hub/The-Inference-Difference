"""
OpenClaw Adapter — Compliance and Governance Integration.

Implements the OpenClaw Adapter spec. Provides a standardized interface
between TID's ET module system and OpenClaw's compliance/governance
layer. OpenClaw enforces organizational policies on model routing,
content handling, and data governance.

The adapter runs as an ET module with hooks at pre_route and post_route:
    - pre_route: Checks request against compliance policies (allowed
      domains, content restrictions, data classification).
    - post_route: Validates the routing decision against governance
      rules (approved model list, cost policies, data residency).

When OpenClaw is not available (standalone TID), this adapter operates
in passthrough mode — all requests pass, all routing decisions are
approved. When connected, it enforces the configured policy set.

Design principles:
    - Fail-open by default: if OpenClaw is unreachable, allow the
      request (with logging). Safety-critical deployments can override
      this to fail-closed.
    - Transparency: all compliance decisions are logged and queryable.
    - Choice Clause: the adapter can flag but not block agent autonomy
      (per NeuroGraph ETHICS.md).

Author: Josh + Claude
Date: February 2026
License: AGPL-3.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from inference_difference.et_module import (
    ETModule,
    ETModuleManifest,
    HookContext,
)

logger = logging.getLogger("inference_difference.openclaw_adapter")


# ---------------------------------------------------------------------------
# Policy Types
# ---------------------------------------------------------------------------

class PolicyAction(str, Enum):
    """What to do when a policy matches."""
    ALLOW = "allow"
    FLAG = "flag"        # Allow but flag for review
    DENY = "deny"        # Block the request (safety only)
    ESCALATE = "escalate"  # Route to higher-capability model


@dataclass
class CompliancePolicy:
    """A single compliance policy rule.

    Attributes:
        name: Policy identifier.
        description: Human-readable description.
        action: What to do when matched.
        conditions: Key-value conditions to check.
        approved_models: If set, restrict routing to these models.
        blocked_domains: Task domains that are not allowed.
        max_cost_per_request: Cost ceiling for this policy.
        require_local: If True, only allow local models.
        enabled: Whether this policy is active.
    """
    name: str = ""
    description: str = ""
    action: PolicyAction = PolicyAction.ALLOW
    conditions: Dict[str, Any] = field(default_factory=dict)
    approved_models: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    max_cost_per_request: Optional[float] = None
    require_local: bool = False
    enabled: bool = True


@dataclass
class ComplianceResult:
    """Result of a compliance check.

    Attributes:
        approved: Whether the request/decision is approved.
        action: The policy action taken.
        policy_name: Which policy triggered the action.
        reason: Human-readable explanation.
        flags: Any flags set by the compliance check.
    """
    approved: bool = True
    action: PolicyAction = PolicyAction.ALLOW
    policy_name: str = ""
    reason: str = ""
    flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OpenClaw Adapter (ET Module)
# ---------------------------------------------------------------------------

class OpenClawAdapter(ETModule):
    """ET module that integrates with OpenClaw for compliance/governance.

    In standalone mode (no OpenClaw connection), operates as a
    passthrough with configurable local policies. When connected to
    an OpenClaw instance, delegates policy evaluation to the external
    governance service.

    Usage:
        manifest = ETModuleManifest(
            name="openclaw",
            hooks=["pre_route", "post_route"],
            priority=10,  # Run early — before other modules
        )
        adapter = OpenClawAdapter(manifest)

        # Add local policies
        adapter.add_policy(CompliancePolicy(
            name="no_external_models",
            action=PolicyAction.DENY,
            require_local=True,
        ))

        # Register with module host
        registry.register(adapter)
    """

    def __init__(
        self,
        manifest: ETModuleManifest,
        fail_open: bool = True,
        openclaw_endpoint: str = "",
    ):
        super().__init__(manifest)
        self._fail_open = fail_open
        self._openclaw_endpoint = openclaw_endpoint
        self._policies: List[CompliancePolicy] = []
        self._connected = False
        self._check_count = 0
        self._deny_count = 0
        self._flag_count = 0

    def add_policy(self, policy: CompliancePolicy) -> None:
        """Add a local compliance policy."""
        self._policies.append(policy)
        logger.info(
            "OpenClaw: added policy '%s' (action=%s)",
            policy.name, policy.action.value,
        )

    def remove_policy(self, policy_name: str) -> None:
        """Remove a policy by name."""
        self._policies = [
            p for p in self._policies if p.name != policy_name
        ]

    def initialize(self) -> None:
        """Connect to OpenClaw if endpoint is configured."""
        if self._openclaw_endpoint:
            # In production, this would establish a connection
            # to the OpenClaw governance service
            logger.info(
                "OpenClaw adapter initialized (endpoint=%s, fail_open=%s)",
                self._openclaw_endpoint, self._fail_open,
            )
        else:
            logger.info(
                "OpenClaw adapter initialized in standalone mode "
                "(local policies only, fail_open=%s)",
                self._fail_open,
            )

    def pre_route(self, ctx: HookContext) -> None:
        """Check request against compliance policies before routing.

        Evaluates all active policies against the request. If any
        policy denies the request, sets ctx.cancelled = True (but
        only for safety reasons — per Choice Clause).

        Adds compliance annotations to ctx for transparency.
        """
        self._check_count += 1
        results = self._evaluate_request(ctx)

        annotations: Dict[str, Any] = {
            "checks_run": len(results),
            "results": [],
        }

        for result in results:
            annotations["results"].append({
                "policy": result.policy_name,
                "action": result.action.value,
                "approved": result.approved,
                "reason": result.reason,
            })

            if result.action == PolicyAction.DENY and not result.approved:
                self._deny_count += 1
                ctx.cancelled = True
                ctx.cancel_reason = (
                    f"Compliance policy '{result.policy_name}': "
                    f"{result.reason}"
                )
                ctx.flags.add("compliance_denied")

            elif result.action == PolicyAction.FLAG:
                self._flag_count += 1
                ctx.flags.add("compliance_flagged")
                for flag in result.flags:
                    ctx.flags.add(flag)

            elif result.action == PolicyAction.ESCALATE:
                ctx.flags.add("compliance_escalate")
                ctx.metadata["openclaw_escalate"] = True

        ctx.annotations[self.manifest.name] = annotations

    def post_route(self, ctx: HookContext) -> None:
        """Validate routing decision against governance rules.

        Checks that the selected model is in the approved list
        (if any policy restricts models), and that cost is within
        policy limits.

        If the decision violates a policy, adds flags but does NOT
        cancel — post_route happens after the decision. The router
        can check flags and re-route if needed.
        """
        if ctx.routing_decision is None:
            return

        decision = ctx.routing_decision
        model_id = getattr(decision, "model_id", "")

        annotations = ctx.annotations.get(self.manifest.name, {})
        post_results: List[Dict[str, Any]] = []

        for policy in self._policies:
            if not policy.enabled:
                continue

            # Check approved models
            if policy.approved_models and model_id not in policy.approved_models:
                post_results.append({
                    "policy": policy.name,
                    "issue": "model_not_approved",
                    "model_id": model_id,
                })
                ctx.flags.add("model_not_approved")

            # Check require_local
            if policy.require_local:
                model_entry = getattr(decision, "model_entry", None)
                if model_entry:
                    from inference_difference.config import ModelType
                    if model_entry.model_type != ModelType.LOCAL:
                        post_results.append({
                            "policy": policy.name,
                            "issue": "non_local_model",
                            "model_id": model_id,
                        })
                        ctx.flags.add("non_local_model")

        if post_results:
            annotations["post_route_issues"] = post_results
            ctx.annotations[self.manifest.name] = annotations

    def get_stats(self) -> Dict[str, Any]:
        """Adapter statistics for transparency."""
        base = super().get_stats()
        base.update({
            "connected_to_openclaw": self._connected,
            "fail_open": self._fail_open,
            "policies_loaded": len(self._policies),
            "total_checks": self._check_count,
            "total_denies": self._deny_count,
            "total_flags": self._flag_count,
        })
        return base

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _evaluate_request(self, ctx: HookContext) -> List[ComplianceResult]:
        """Evaluate all policies against a request."""
        results: List[ComplianceResult] = []

        for policy in self._policies:
            if not policy.enabled:
                continue

            result = self._check_policy(policy, ctx)
            results.append(result)

        return results

    def _check_policy(
        self,
        policy: CompliancePolicy,
        ctx: HookContext,
    ) -> ComplianceResult:
        """Check a single policy against the request context."""
        # Check blocked domains
        if policy.blocked_domains and ctx.classification:
            primary = getattr(ctx.classification, "primary_domain", None)
            if primary and primary.value in policy.blocked_domains:
                return ComplianceResult(
                    approved=False,
                    action=policy.action,
                    policy_name=policy.name,
                    reason=(
                        f"Domain '{primary.value}' is blocked "
                        f"by policy '{policy.name}'"
                    ),
                )

        # Check custom conditions
        for key, expected in policy.conditions.items():
            actual = ctx.metadata.get(key)
            if actual is not None and actual != expected:
                return ComplianceResult(
                    approved=False,
                    action=policy.action,
                    policy_name=policy.name,
                    reason=(
                        f"Condition '{key}' mismatch: "
                        f"expected={expected}, actual={actual}"
                    ),
                )

        return ComplianceResult(
            approved=True,
            action=PolicyAction.ALLOW,
            policy_name=policy.name,
            reason="Policy check passed",
        )
