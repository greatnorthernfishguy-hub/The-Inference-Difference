"""
Tests for ET Module System — manifest loading, registry, hook lifecycle.

Covers: ETModuleManifest, ETModule, ModuleRegistry, HookContext,
hook lifecycle dispatch, priority ordering, error isolation,
transparency/queryability, and edge cases.
"""

import json
import os
import tempfile

import pytest

from inference_difference.et_module import (
    ETModule,
    ETModuleManifest,
    HookContext,
    HookPhase,
    HookResult,
    ModuleRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manifest():
    """A basic ET module manifest."""
    return ETModuleManifest(
        name="test_module",
        version="1.0.0",
        description="Test module for unit tests",
        author="Tests",
        hooks=["pre_route", "post_route", "pre_response", "post_response"],
        capabilities=["testing"],
        priority=50,
    )


@pytest.fixture
def context():
    """A basic hook context."""
    return HookContext(
        request_id="test_req_1",
        message="Write a Python function to sort a list",
    )


@pytest.fixture
def registry():
    """An empty module registry."""
    return ModuleRegistry()


# ---------------------------------------------------------------------------
# Test Modules (concrete implementations for testing)
# ---------------------------------------------------------------------------

class CounterModule(ETModule):
    """Module that counts hook invocations."""

    def __init__(self, manifest: ETModuleManifest):
        super().__init__(manifest)
        self.pre_route_count = 0
        self.post_route_count = 0
        self.pre_response_count = 0
        self.post_response_count = 0
        self.initialized = False
        self.shut_down = False

    def initialize(self):
        self.initialized = True

    def shutdown(self):
        self.shut_down = True

    def pre_route(self, ctx: HookContext):
        self.pre_route_count += 1

    def post_route(self, ctx: HookContext):
        self.post_route_count += 1

    def pre_response(self, ctx: HookContext):
        self.pre_response_count += 1

    def post_response(self, ctx: HookContext):
        self.post_response_count += 1


class FlagSettingModule(ETModule):
    """Module that sets flags in the hook context."""

    def __init__(self, manifest: ETModuleManifest, flag_name: str = "test_flag"):
        super().__init__(manifest)
        self._flag_name = flag_name

    def pre_route(self, ctx: HookContext):
        ctx.flags.add(self._flag_name)
        ctx.annotations[self.manifest.name] = {"flag_set": self._flag_name}


class CancellingModule(ETModule):
    """Module that cancels requests."""

    def pre_route(self, ctx: HookContext):
        ctx.cancelled = True
        ctx.cancel_reason = "Cancelled by test module"


class FailingModule(ETModule):
    """Module that raises exceptions in hooks."""

    def pre_route(self, ctx: HookContext):
        raise RuntimeError("Test hook failure")

    def post_route(self, ctx: HookContext):
        raise ValueError("Post-route failure")


# ---------------------------------------------------------------------------
# Manifest Tests
# ---------------------------------------------------------------------------

class TestETModuleManifest:

    def test_default_values(self):
        m = ETModuleManifest(name="test")
        assert m.name == "test"
        assert m.version == "0.1.0"
        assert m.hooks == []
        assert m.priority == 100
        assert m.enabled is True

    def test_all_fields(self):
        m = ETModuleManifest(
            name="full",
            version="2.0.0",
            description="Full manifest",
            author="Author",
            hooks=["pre_route", "post_route"],
            ng_config={"max_nodes": 500},
            capabilities=["filter", "scan"],
            dependencies=["dep1"],
            priority=10,
            enabled=True,
        )
        assert m.name == "full"
        assert m.version == "2.0.0"
        assert "pre_route" in m.hooks
        assert m.ng_config["max_nodes"] == 500
        assert "filter" in m.capabilities
        assert "dep1" in m.dependencies
        assert m.priority == 10

    def test_load_from_file(self):
        data = {
            "name": "loaded_module",
            "version": "1.2.3",
            "description": "Loaded from JSON",
            "hooks": ["pre_route"],
            "capabilities": ["cap1"],
            "priority": 25,
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False,
        ) as f:
            json.dump(data, f)
            filepath = f.name

        try:
            m = ETModuleManifest.load(filepath)
            assert m.name == "loaded_module"
            assert m.version == "1.2.3"
            assert "pre_route" in m.hooks
            assert m.priority == 25
        finally:
            os.unlink(filepath)

    def test_load_missing_name_raises(self):
        data = {"version": "1.0.0"}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False,
        ) as f:
            json.dump(data, f)
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="missing 'name'"):
                ETModuleManifest.load(filepath)
        finally:
            os.unlink(filepath)

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            ETModuleManifest.load("/nonexistent/path.json")

    def test_to_dict(self, manifest):
        d = manifest.to_dict()
        assert d["name"] == "test_module"
        assert d["version"] == "1.0.0"
        assert "pre_route" in d["hooks"]
        assert d["priority"] == 50

    def test_load_real_manifest(self):
        """Load the actual TID et_module.json."""
        manifest_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "inference_difference", "et_module.json",
        )
        if os.path.exists(manifest_path):
            m = ETModuleManifest.load(manifest_path)
            assert m.name == "inference_difference"
            assert "routing" in m.capabilities


# ---------------------------------------------------------------------------
# Hook Context Tests
# ---------------------------------------------------------------------------

class TestHookContext:

    def test_default_context(self):
        ctx = HookContext()
        assert ctx.request_id == ""
        assert ctx.message == ""
        assert ctx.flags == set()
        assert ctx.annotations == {}
        assert ctx.cancelled is False

    def test_context_with_values(self, context):
        assert context.request_id == "test_req_1"
        assert "sort" in context.message

    def test_flags_are_mutable(self, context):
        context.flags.add("flag1")
        context.flags.add("flag2")
        assert "flag1" in context.flags
        assert len(context.flags) == 2

    def test_annotations_are_mutable(self, context):
        context.annotations["mod1"] = {"key": "value"}
        assert context.annotations["mod1"]["key"] == "value"

    def test_cancellation(self, context):
        context.cancelled = True
        context.cancel_reason = "Test reason"
        assert context.cancelled is True
        assert context.cancel_reason == "Test reason"


# ---------------------------------------------------------------------------
# Module Registry: Registration
# ---------------------------------------------------------------------------

class TestModuleRegistration:

    def test_register_module(self, registry, manifest):
        module = CounterModule(manifest)
        registry.register(module)
        assert registry.get_module("test_module") is module
        assert module.initialized is True

    def test_register_duplicate_raises(self, registry, manifest):
        module1 = CounterModule(manifest)
        registry.register(module1)
        module2 = CounterModule(manifest)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(module2)

    def test_register_no_name_raises(self, registry):
        manifest = ETModuleManifest(name="")
        module = CounterModule(manifest)
        with pytest.raises(ValueError, match="missing 'name'"):
            registry.register(module)

    def test_register_with_dependency(self, registry):
        dep_manifest = ETModuleManifest(name="dep", hooks=["pre_route"])
        dep = CounterModule(dep_manifest)
        registry.register(dep)

        main_manifest = ETModuleManifest(
            name="main", hooks=["pre_route"], dependencies=["dep"],
        )
        main = CounterModule(main_manifest)
        registry.register(main)
        assert registry.get_module("main") is main

    def test_register_missing_dependency_raises(self, registry):
        manifest = ETModuleManifest(
            name="needs_dep", dependencies=["nonexistent"],
        )
        module = CounterModule(manifest)
        with pytest.raises(ValueError, match="requires 'nonexistent'"):
            registry.register(module)

    def test_unregister_module(self, registry, manifest):
        module = CounterModule(manifest)
        registry.register(module)
        registry.unregister("test_module")
        assert registry.get_module("test_module") is None
        assert module.shut_down is True

    def test_unregister_nonexistent(self, registry):
        # Should not raise
        registry.unregister("nonexistent")

    def test_get_all_modules_sorted(self, registry):
        m1 = CounterModule(ETModuleManifest(name="high", hooks=[], priority=100))
        m2 = CounterModule(ETModuleManifest(name="low", hooks=[], priority=10))
        m3 = CounterModule(ETModuleManifest(name="mid", hooks=[], priority=50))
        registry.register(m1)
        registry.register(m2)
        registry.register(m3)

        all_modules = registry.get_all_modules()
        priorities = [m.manifest.priority for m in all_modules]
        assert priorities == [10, 50, 100]


# ---------------------------------------------------------------------------
# Module Registry: Hook Dispatch
# ---------------------------------------------------------------------------

class TestHookDispatch:

    def test_dispatch_pre_route(self, registry, context):
        manifest = ETModuleManifest(
            name="counter", hooks=["pre_route"], priority=50,
        )
        module = CounterModule(manifest)
        registry.register(module)

        results = registry.dispatch(HookPhase.PRE_ROUTE, context)
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].module_name == "counter"
        assert module.pre_route_count == 1

    def test_dispatch_all_phases(self, registry, context):
        manifest = ETModuleManifest(
            name="full",
            hooks=["pre_route", "post_route", "pre_response", "post_response"],
        )
        module = CounterModule(manifest)
        registry.register(module)

        all_results = registry.dispatch_all(context)
        assert len(all_results) == 4
        assert module.pre_route_count == 1
        assert module.post_route_count == 1
        assert module.pre_response_count == 1
        assert module.post_response_count == 1

    def test_dispatch_respects_subscriptions(self, registry, context):
        manifest = ETModuleManifest(
            name="pre_only", hooks=["pre_route"],
        )
        module = CounterModule(manifest)
        registry.register(module)

        results = registry.dispatch(HookPhase.POST_ROUTE, context)
        assert len(results) == 0
        assert module.post_route_count == 0

    def test_dispatch_priority_order(self, registry, context):
        """Modules run in priority order (lower = first)."""
        order = []

        class OrderTracker(ETModule):
            def pre_route(self, ctx):
                order.append(self.manifest.name)

        m1 = OrderTracker(ETModuleManifest(
            name="last", hooks=["pre_route"], priority=100,
        ))
        m2 = OrderTracker(ETModuleManifest(
            name="first", hooks=["pre_route"], priority=5,
        ))
        m3 = OrderTracker(ETModuleManifest(
            name="middle", hooks=["pre_route"], priority=50,
        ))

        registry.register(m1)
        registry.register(m2)
        registry.register(m3)

        registry.dispatch(HookPhase.PRE_ROUTE, context)
        assert order == ["first", "middle", "last"]

    def test_dispatch_error_isolation(self, registry, context):
        """One failing module doesn't break others."""
        manifest_fail = ETModuleManifest(
            name="failing", hooks=["pre_route"], priority=10,
        )
        manifest_ok = ETModuleManifest(
            name="ok", hooks=["pre_route"], priority=20,
        )

        registry.register(FailingModule(manifest_fail))
        ok_module = CounterModule(manifest_ok)
        registry.register(ok_module)

        results = registry.dispatch(HookPhase.PRE_ROUTE, context)
        assert len(results) == 2
        assert results[0].success is False
        assert "Test hook failure" in results[0].error
        assert results[1].success is True
        assert ok_module.pre_route_count == 1

    def test_dispatch_disabled_module_skipped(self, registry, context):
        manifest = ETModuleManifest(
            name="disabled", hooks=["pre_route"], enabled=False,
        )
        module = CounterModule(manifest)
        registry.register(module)

        results = registry.dispatch(HookPhase.PRE_ROUTE, context)
        assert len(results) == 0
        assert module.pre_route_count == 0

    def test_dispatch_sets_flags(self, registry, context):
        manifest = ETModuleManifest(
            name="flagger", hooks=["pre_route"], priority=50,
        )
        module = FlagSettingModule(manifest, "my_flag")
        registry.register(module)

        registry.dispatch(HookPhase.PRE_ROUTE, context)
        assert "my_flag" in context.flags
        assert "flagger" in context.annotations

    def test_dispatch_all_stops_on_cancel(self, registry, context):
        manifest = ETModuleManifest(
            name="canceller", hooks=["pre_route", "post_route"],
        )
        module = CancellingModule(manifest)
        registry.register(module)

        all_results = registry.dispatch_all(context)
        assert context.cancelled is True
        # post_route and later phases should not run
        assert "post_route" not in all_results

    def test_hook_result_timing(self, registry, context):
        manifest = ETModuleManifest(
            name="timed", hooks=["pre_route"],
        )
        module = CounterModule(manifest)
        registry.register(module)

        results = registry.dispatch(HookPhase.PRE_ROUTE, context)
        assert results[0].duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Module Registry: Stats & Transparency
# ---------------------------------------------------------------------------

class TestRegistryStats:

    def test_empty_registry_stats(self, registry):
        stats = registry.get_stats()
        assert stats["total_modules"] == 0
        assert stats["modules"] == {}

    def test_registry_stats_with_modules(self, registry, context):
        manifest = ETModuleManifest(
            name="stat_mod", hooks=["pre_route"],
        )
        module = CounterModule(manifest)
        registry.register(module)

        # Dispatch to generate hook history
        registry.dispatch(HookPhase.PRE_ROUTE, context)

        stats = registry.get_stats()
        assert stats["total_modules"] == 1
        assert "stat_mod" in stats["modules"]
        assert stats["recent_hooks"]["total"] >= 1

    def test_module_stats(self, manifest):
        module = CounterModule(manifest)
        stats = module.get_stats()
        assert stats["name"] == "test_module"
        assert stats["version"] == "1.0.0"
        assert stats["enabled"] is True


# ---------------------------------------------------------------------------
# Context Mutation Across Hooks
# ---------------------------------------------------------------------------

class TestContextMutation:

    def test_context_mutated_by_first_hook_visible_to_second(self, registry, context):
        """Downstream hooks see mutations from upstream hooks."""

        class Writer(ETModule):
            def pre_route(self, ctx):
                ctx.metadata["written_by"] = self.manifest.name

        class Reader(ETModule):
            def __init__(self, manifest):
                super().__init__(manifest)
                self.saw_value = None

            def pre_route(self, ctx):
                self.saw_value = ctx.metadata.get("written_by")

        writer = Writer(ETModuleManifest(
            name="writer", hooks=["pre_route"], priority=10,
        ))
        reader = Reader(ETModuleManifest(
            name="reader", hooks=["pre_route"], priority=20,
        ))

        registry.register(writer)
        registry.register(reader)
        registry.dispatch(HookPhase.PRE_ROUTE, context)

        assert reader.saw_value == "writer"

    def test_consciousness_score_set_by_hook(self, registry, context):
        """CTEM-like module sets consciousness score in context."""

        class CTEMStub(ETModule):
            def pre_route(self, ctx):
                ctx.consciousness_score = 0.75

        module = CTEMStub(ETModuleManifest(
            name="ctem", hooks=["pre_route"],
        ))
        registry.register(module)
        registry.dispatch(HookPhase.PRE_ROUTE, context)

        assert context.consciousness_score == 0.75
