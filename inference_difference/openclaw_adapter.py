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

Connection protocol:
    The OpenClaw Gateway multiplexes WebSocket and HTTP on a single port
    (default 18789). This adapter uses the HTTP hooks surface for
    synchronous compliance checks:
        - POST /hooks/agent — route compliance queries to agents
    Authentication is via Bearer token (OPENCLAW_GATEWAY_TOKEN env var).
    Health probing uses a plain HTTP GET to detect gateway liveness.

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

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from inference_difference.et_module import (
    ETModule,
    ETModuleManifest,
    HookContext,
)

logger = logging.getLogger("inference_difference.openclaw_adapter")

# Gateway HTTP timeout for compliance checks (seconds).
_GATEWAY_TIMEOUT = 5


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
    governance service and runs local policies as a secondary layer.

    Usage:
        manifest = ETModuleManifest(
            name="openclaw",
            hooks=["pre_route", "post_route"],
            priority=10,  # Run early — before other modules
        )
        adapter = OpenClawAdapter(
            manifest,
            openclaw_endpoint="http://127.0.0.1:18789",
            openclaw_token="your-gateway-token",
        )

        # Add local policies (always evaluated, even when connected)
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
        openclaw_token: str = "",
    ):
        super().__init__(manifest)
        self._fail_open = fail_open
        self._openclaw_endpoint = openclaw_endpoint.rstrip("/")
        self._openclaw_token = openclaw_token
        self._policies: List[CompliancePolicy] = []
        self._connected = False
        self._check_count = 0
        self._deny_count = 0
        self._flag_count = 0
        self._gateway_checks = 0
        self._gateway_errors = 0

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
        """Connect to OpenClaw gateway if endpoint is configured.

        Probes the gateway with an HTTP GET to verify it is reachable.
        The gateway serves its web UI on all paths, so any HTTP response
        (even HTML) confirms the process is alive. Sets _connected=True
        on success.

        If the probe fails, logs a warning and stays in standalone mode.
        The adapter will re-probe on each hook call so it can recover
        if the gateway comes up later.
        """
        if not self._openclaw_endpoint:
            logger.info(
                "OpenClaw adapter initialized in standalone mode "
                "(no endpoint configured, fail_open=%s)",
                self._fail_open,
            )
            return

        self._connected = self._probe_gateway()
        if self._connected:
            logger.info(
                "OpenClaw adapter connected to gateway at %s "
                "(fail_open=%s)",
                self._openclaw_endpoint, self._fail_open,
            )
        else:
            logger.warning(
                "OpenClaw adapter could not reach gateway at %s — "
                "starting in standalone mode (fail_open=%s). "
                "Will re-probe on each request.",
                self._openclaw_endpoint, self._fail_open,
            )

    def shutdown(self) -> None:
        """Clean up on shutdown."""
        if self._connected:
            logger.info("OpenClaw adapter disconnecting from gateway")
        self._connected = False

    def pre_route(self, ctx: HookContext) -> None:
        """Check request against compliance policies before routing.

        When connected to the gateway, sends a compliance query via
        POST /hooks/agent and interprets the response. Falls back to
        local policy evaluation if the gateway is unreachable.

        Always runs local policies as a secondary layer.
        """
        self._check_count += 1

        annotations: Dict[str, Any] = {
            "gateway_connected": self._connected,
            "results": [],
        }

        # --- Gateway compliance check (if connected) ---
        if self._openclaw_endpoint:
            gw_result = self._gateway_compliance_check(ctx)
            if gw_result is not None:
                annotations["gateway_response"] = gw_result
                self._apply_gateway_result(ctx, gw_result, annotations)
            elif not self._fail_open:
                # Gateway unreachable in fail-closed mode
                self._deny_count += 1
                ctx.cancelled = True
                ctx.cancel_reason = (
                    "OpenClaw gateway unreachable (fail_closed mode)"
                )
                ctx.flags.add("compliance_denied")
                ctx.annotations[self.manifest.name] = annotations
                return

        # --- Local policy evaluation (always runs) ---
        results = self._evaluate_request(ctx)
        annotations["local_checks_run"] = len(results)

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

        When connected to the gateway, sends the routing decision for
        validation. Falls back to local policy checks if unreachable.
        """
        if ctx.routing_decision is None:
            return

        decision = ctx.routing_decision
        model_id = getattr(decision, "model_id", "")

        annotations = ctx.annotations.get(self.manifest.name, {})
        post_results: List[Dict[str, Any]] = []

        # --- Gateway validation (if connected) ---
        if self._openclaw_endpoint:
            gw_result = self._gateway_routing_validation(ctx)
            if gw_result is not None:
                annotations["gateway_post_route"] = gw_result

        # --- Local policy validation (always runs) ---
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
            "endpoint": self._openclaw_endpoint or "(standalone)",
            "policies_loaded": len(self._policies),
            "total_checks": self._check_count,
            "total_denies": self._deny_count,
            "total_flags": self._flag_count,
            "gateway_checks": self._gateway_checks,
            "gateway_errors": self._gateway_errors,
        })
        return base

    # -------------------------------------------------------------------
    # Gateway HTTP communication
    # -------------------------------------------------------------------

    def _probe_gateway(self) -> bool:
        """Check if the OpenClaw gateway is reachable via HTTP.

        The gateway serves its web UI on all paths, so any response
        (even HTML) confirms the process is alive. We also accept 4xx
        responses (auth issues) as proof of liveness.
        """
        try:
            req = urllib.request.Request(
                self._openclaw_endpoint,
                method="GET",
            )
            if self._openclaw_token:
                req.add_header(
                    "Authorization", f"Bearer {self._openclaw_token}",
                )
            with urllib.request.urlopen(req, timeout=_GATEWAY_TIMEOUT):
                return True
        except urllib.error.HTTPError as e:
            # 4xx = gateway is alive but rejected our request (auth, etc.)
            if e.code < 500:
                return True
            logger.debug("Gateway probe got %d", e.code)
            return False
        except Exception as e:
            logger.debug("Gateway probe failed: %s", e)
            return False

    def _gateway_post(
        self, path: str, body: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """POST JSON to the OpenClaw gateway, return parsed response.

        Returns None on any failure (network, timeout, non-JSON response).
        Callers fall back to local policies when this returns None.
        """
        url = f"{self._openclaw_endpoint}{path}"
        data = json.dumps(body).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            if self._openclaw_token:
                req.add_header(
                    "Authorization", f"Bearer {self._openclaw_token}",
                )
            with urllib.request.urlopen(
                req, timeout=_GATEWAY_TIMEOUT,
            ) as resp:
                resp_body = resp.read().decode("utf-8")
                try:
                    return json.loads(resp_body)
                except json.JSONDecodeError:
                    # Gateway returned non-JSON (probably HTML from SPA).
                    # Treat as "endpoint not implemented yet" — not an error.
                    logger.debug(
                        "Gateway %s returned non-JSON response", path,
                    )
                    return None
        except urllib.error.HTTPError as e:
            self._gateway_errors += 1
            logger.debug("Gateway POST %s got HTTP %d", path, e.code)
            return None
        except Exception as e:
            self._gateway_errors += 1
            # Check if gateway went down — re-probe on next call
            if self._connected:
                self._connected = self._probe_gateway()
                if not self._connected:
                    logger.warning(
                        "OpenClaw gateway lost contact at %s — "
                        "falling back to standalone mode",
                        self._openclaw_endpoint,
                    )
            logger.debug("Gateway POST %s failed: %s", path, e)
            return None

    def _gateway_compliance_check(
        self, ctx: HookContext,
    ) -> Optional[Dict[str, Any]]:
        """Ask the gateway to evaluate a request for compliance.

        Sends a structured compliance query to POST /hooks/agent.
        The gateway routes this to the appropriate compliance agent.
        """
        # Re-probe if we lost connection
        if not self._connected and self._openclaw_endpoint:
            self._connected = self._probe_gateway()

        if not self._connected:
            return None

        self._gateway_checks += 1
        return self._gateway_post("/hooks/agent", {
            "type": "compliance_check",
            "phase": "pre_route",
            "source": "tid",
            "request": {
                "message": ctx.message,
                "request_id": ctx.request_id,
                "metadata": ctx.metadata,
                "flags": list(ctx.flags),
            },
        })

    def _gateway_routing_validation(
        self, ctx: HookContext,
    ) -> Optional[Dict[str, Any]]:
        """Ask the gateway to validate a routing decision."""
        if not self._connected:
            return None

        decision = ctx.routing_decision
        self._gateway_checks += 1
        return self._gateway_post("/hooks/agent", {
            "type": "compliance_check",
            "phase": "post_route",
            "source": "tid",
            "decision": {
                "model_id": getattr(decision, "model_id", ""),
                "score": getattr(decision, "score", 0.0),
                "request_id": ctx.request_id,
            },
        })

    def _apply_gateway_result(
        self,
        ctx: HookContext,
        gw_result: Dict[str, Any],
        annotations: Dict[str, Any],
    ) -> None:
        """Apply a gateway compliance response to the hook context.

        The gateway may return structured compliance data. If the
        response contains actionable fields, they are applied.
        Unknown response shapes are logged but not acted on.
        """
        action = gw_result.get("action", "allow")
        reason = gw_result.get("reason", "")

        if action == "deny":
            self._deny_count += 1
            ctx.cancelled = True
            ctx.cancel_reason = (
                f"OpenClaw gateway denied: {reason}"
            )
            ctx.flags.add("compliance_denied")
            ctx.flags.add("gateway_denied")

        elif action == "flag":
            self._flag_count += 1
            ctx.flags.add("compliance_flagged")
            ctx.flags.add("gateway_flagged")
            for flag in gw_result.get("flags", []):
                ctx.flags.add(flag)

        elif action == "escalate":
            ctx.flags.add("compliance_escalate")
            ctx.metadata["openclaw_escalate"] = True

    # -------------------------------------------------------------------
    # Local policy evaluation
    # -------------------------------------------------------------------

    def _evaluate_request(self, ctx: HookContext) -> List[ComplianceResult]:
        """Evaluate all local policies against a request."""
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
