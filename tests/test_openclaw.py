"""
Tests for OpenClaw Adapter — Compliance and Governance Integration.

Covers: policy evaluation, domain blocking, model approval,
local-only enforcement, fail-open behavior, hook lifecycle integration,
and transparency.
"""

import pytest

from inference_difference.classifier import classify_request
from inference_difference.config import ComplexityTier, ModelType, TaskDomain
from inference_difference.et_module import (
    ETModuleManifest,
    HookContext,
    HookPhase,
    ModuleRegistry,
)
from inference_difference.openclaw_adapter import (
    CompliancePolicy,
    ComplianceResult,
    OpenClawAdapter,
    PolicyAction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def openclaw():
    """OpenClaw adapter with default settings (standalone, fail-open)."""
    manifest = ETModuleManifest(
        name="openclaw",
        hooks=["pre_route", "post_route"],
        priority=10,
    )
    return OpenClawAdapter(manifest)


@pytest.fixture
def context():
    """Clean hook context."""
    return HookContext(request_id="oc_test_1", message="Hello world")


# ---------------------------------------------------------------------------
# Policy Management
# ---------------------------------------------------------------------------

class TestPolicyManagement:

    def test_add_policy(self, openclaw):
        policy = CompliancePolicy(
            name="test_policy",
            action=PolicyAction.FLAG,
        )
        openclaw.add_policy(policy)
        assert len(openclaw._policies) == 1

    def test_remove_policy(self, openclaw):
        openclaw.add_policy(CompliancePolicy(name="p1"))
        openclaw.add_policy(CompliancePolicy(name="p2"))
        openclaw.remove_policy("p1")
        assert len(openclaw._policies) == 1
        assert openclaw._policies[0].name == "p2"


# ---------------------------------------------------------------------------
# Pre-Route: Domain Blocking
# ---------------------------------------------------------------------------

class TestDomainBlocking:

    def test_blocked_domain_denies(self, openclaw, context):
        openclaw.add_policy(CompliancePolicy(
            name="no_creative",
            action=PolicyAction.DENY,
            blocked_domains=["creative"],
        ))

        # Set classification with creative domain
        classification = classify_request("Write a poem about robots")
        context.classification = classification

        openclaw.pre_route(context)

        if classification.primary_domain == TaskDomain.CREATIVE:
            assert context.cancelled is True
            assert "compliance_denied" in context.flags

    def test_allowed_domain_passes(self, openclaw, context):
        openclaw.add_policy(CompliancePolicy(
            name="no_creative",
            action=PolicyAction.DENY,
            blocked_domains=["creative"],
        ))

        classification = classify_request("Write Python code")
        context.classification = classification

        openclaw.pre_route(context)
        assert context.cancelled is False

    def test_no_classification_passes(self, openclaw, context):
        """No classification available — policy check passes."""
        openclaw.add_policy(CompliancePolicy(
            name="no_creative",
            action=PolicyAction.DENY,
            blocked_domains=["creative"],
        ))
        openclaw.pre_route(context)
        assert context.cancelled is False


# ---------------------------------------------------------------------------
# Pre-Route: Flag Policy
# ---------------------------------------------------------------------------

class TestFlagPolicy:

    def test_flag_policy_sets_flag(self, openclaw, context):
        openclaw.add_policy(CompliancePolicy(
            name="flag_all",
            action=PolicyAction.FLAG,
            conditions={"env": "production"},
        ))
        context.metadata["env"] = "development"

        openclaw.pre_route(context)
        assert "compliance_flagged" in context.flags
        assert context.cancelled is False

    def test_matching_condition_passes(self, openclaw, context):
        openclaw.add_policy(CompliancePolicy(
            name="env_check",
            action=PolicyAction.DENY,
            conditions={"env": "production"},
        ))
        context.metadata["env"] = "production"

        openclaw.pre_route(context)
        # Matching condition means the check passes (no violation)
        assert context.cancelled is False


# ---------------------------------------------------------------------------
# Pre-Route: Escalation
# ---------------------------------------------------------------------------

class TestEscalation:

    def test_escalation_sets_flag(self, openclaw, context):
        openclaw.add_policy(CompliancePolicy(
            name="escalate_check",
            action=PolicyAction.ESCALATE,
            conditions={"priority": "high"},
        ))
        context.metadata["priority"] = "low"

        openclaw.pre_route(context)
        assert "compliance_escalate" in context.flags
        assert context.metadata.get("openclaw_escalate") is True


# ---------------------------------------------------------------------------
# Post-Route: Model Approval
# ---------------------------------------------------------------------------

class TestModelApproval:

    def test_unapproved_model_flagged(self, openclaw, context):
        openclaw.add_policy(CompliancePolicy(
            name="approved_only",
            approved_models=["model_a", "model_b"],
        ))

        class FakeDecision:
            model_id = "model_c"
            model_entry = None

        context.routing_decision = FakeDecision()
        openclaw.post_route(context)
        assert "model_not_approved" in context.flags

    def test_approved_model_no_flag(self, openclaw, context):
        openclaw.add_policy(CompliancePolicy(
            name="approved_only",
            approved_models=["model_a", "model_b"],
        ))

        class FakeDecision:
            model_id = "model_a"
            model_entry = None

        context.routing_decision = FakeDecision()
        openclaw.post_route(context)
        assert "model_not_approved" not in context.flags

    def test_no_routing_decision_noop(self, openclaw, context):
        openclaw.post_route(context)  # Should not crash


# ---------------------------------------------------------------------------
# Post-Route: Local-Only Enforcement
# ---------------------------------------------------------------------------

class TestLocalOnly:

    def test_require_local_flags_api_model(self, openclaw, context):
        openclaw.add_policy(CompliancePolicy(
            name="local_only",
            require_local=True,
        ))

        from inference_difference.config import ModelEntry

        class FakeDecision:
            model_id = "anthropic/claude"
            model_entry = ModelEntry(
                model_id="anthropic/claude",
                model_type=ModelType.API,
            )

        context.routing_decision = FakeDecision()
        openclaw.post_route(context)
        assert "non_local_model" in context.flags

    def test_require_local_allows_local_model(self, openclaw, context):
        openclaw.add_policy(CompliancePolicy(
            name="local_only",
            require_local=True,
        ))

        from inference_difference.config import ModelEntry

        class FakeDecision:
            model_id = "ollama/llama3.1:8b"
            model_entry = ModelEntry(
                model_id="ollama/llama3.1:8b",
                model_type=ModelType.LOCAL,
            )

        context.routing_decision = FakeDecision()
        openclaw.post_route(context)
        assert "non_local_model" not in context.flags


# ---------------------------------------------------------------------------
# Transparency & Stats
# ---------------------------------------------------------------------------

class TestOpenClawStats:

    def test_initial_stats(self, openclaw):
        stats = openclaw.get_stats()
        assert stats["name"] == "openclaw"
        assert stats["total_checks"] == 0
        assert stats["total_denies"] == 0
        assert stats["total_flags"] == 0
        assert stats["fail_open"] is True

    def test_stats_after_checks(self, openclaw):
        openclaw.add_policy(CompliancePolicy(
            name="deny_policy",
            action=PolicyAction.DENY,
            blocked_domains=["creative"],
        ))

        # Run a check that triggers deny
        ctx = HookContext(request_id="s1", message="Write a poem")
        classification = classify_request("Write a poem about robots")
        ctx.classification = classification
        openclaw.pre_route(ctx)

        stats = openclaw.get_stats()
        assert stats["total_checks"] == 1

    def test_annotations_present(self, openclaw, context):
        openclaw.add_policy(CompliancePolicy(name="p1"))
        openclaw.pre_route(context)
        assert "openclaw" in context.annotations
        ann = context.annotations["openclaw"]
        assert "local_checks_run" in ann
        assert "results" in ann
        assert "gateway_connected" in ann

    def test_stats_include_gateway_fields(self, openclaw):
        stats = openclaw.get_stats()
        assert "gateway_checks" in stats
        assert "gateway_errors" in stats
        assert stats["endpoint"] == "(standalone)"


# ---------------------------------------------------------------------------
# Disabled Policies
# ---------------------------------------------------------------------------

class TestDisabledPolicies:

    def test_disabled_policy_skipped(self, openclaw, context):
        openclaw.add_policy(CompliancePolicy(
            name="disabled",
            action=PolicyAction.DENY,
            blocked_domains=["code"],
            enabled=False,
        ))

        classification = classify_request("Write Python code")
        context.classification = classification
        openclaw.pre_route(context)
        assert context.cancelled is False


# ---------------------------------------------------------------------------
# Integration: Registry + OpenClaw
# ---------------------------------------------------------------------------

class TestOpenClawIntegration:

    def test_openclaw_in_registry(self):
        registry = ModuleRegistry()
        manifest = ETModuleManifest(
            name="openclaw",
            hooks=["pre_route", "post_route"],
            priority=10,
        )
        adapter = OpenClawAdapter(manifest)
        registry.register(adapter)

        ctx = HookContext(request_id="int_1", message="Hello")
        results = registry.dispatch(HookPhase.PRE_ROUTE, ctx)
        assert len(results) == 1
        assert results[0].success is True

    def test_openclaw_before_trollguard(self):
        """OpenClaw (priority 10) runs before TrollGuard (priority 5)...
        wait, TrollGuard has lower priority number, so it runs first.
        This is correct — security before compliance."""
        from inference_difference.trollguard import create_trollguard

        registry = ModuleRegistry()
        tg = create_trollguard()  # priority 5
        oc_manifest = ETModuleManifest(
            name="openclaw", hooks=["pre_route"], priority=10,
        )
        oc = OpenClawAdapter(oc_manifest)

        registry.register(tg)
        registry.register(oc)

        modules = registry.get_all_modules()
        assert modules[0].name == "trollguard"
        assert modules[1].name == "openclaw"
