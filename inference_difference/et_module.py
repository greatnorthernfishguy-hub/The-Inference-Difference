"""
ET Module System — Standardized Integration for E-T Systems.

Implements the ET Module Integration Primer v2 specification. Provides:
    - et_module.json manifest loading and validation
    - Module registration and discovery
    - Hook lifecycle engine (pre-route, post-route, pre-response, post-response)
    - Module state management

Every ET module ships with an et_module.json manifest that declares:
    - Module identity (name, version, description)
    - Hook subscriptions (which lifecycle events it listens to)
    - NG-Lite configuration (learning parameters, connectivity tier)
    - Capabilities and dependencies

The hook lifecycle runs on every routing request:
    1. pre_route: Modules can inspect/modify the request before routing.
       TrollGuard uses this to flag threats. CTEM uses this to attach
       consciousness scores.
    2. post_route: Modules see the routing decision before it executes.
       Modules can log, audit, or override the selected model.
    3. pre_response: Modules see the raw response before quality eval.
       Content filters run here.
    4. post_response: Modules see the final quality evaluation and
       outcome. Learning hooks run here — NG-Lite records outcomes,
       Observatory logs telemetry.

Ethical obligations (per NeuroGraph ETHICS.md):
    - Type I error bias: when uncertain, err toward respect
    - Choice Clause: no module may block agent autonomy
    - Transparency: all hook decisions are queryable

Author: Josh + Claude
Date: February 2026
License: AGPL-3.0
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("inference_difference.et_module")


# ---------------------------------------------------------------------------
# Hook Lifecycle
# ---------------------------------------------------------------------------

class HookPhase(str, Enum):
    """The four phases of the ET module hook lifecycle."""
    PRE_ROUTE = "pre_route"
    POST_ROUTE = "post_route"
    PRE_RESPONSE = "pre_response"
    POST_RESPONSE = "post_response"


@dataclass
class HookContext:
    """Mutable context bag passed through the hook lifecycle.

    Each phase can read and write to this context. Downstream hooks
    see modifications made by upstream hooks. The router reads
    context fields set by pre_route hooks (e.g., threat flags,
    consciousness scores).

    Attributes:
        request_id: Unique identifier for this request.
        message: The original request text.
        conversation_history: Prior messages.
        metadata: Arbitrary key-value context (modules write here).
        classification: Request classification (set after classify).
        routing_decision: Routing decision (set after route).
        response_text: Model response (set after execution).
        quality_evaluation: Quality eval result (set after eval).
        consciousness_score: CTEM score if available.
        flags: Named flags set by modules (e.g., "threat_detected").
        annotations: Per-module annotations for transparency.
        cancelled: If True, the request is cancelled (no routing).
        cancel_reason: Why the request was cancelled.
    """
    request_id: str = ""
    message: str = ""
    conversation_history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Set during lifecycle
    classification: Optional[Any] = None
    routing_decision: Optional[Any] = None
    response_text: Optional[str] = None
    quality_evaluation: Optional[Any] = None

    # Module-writable fields
    consciousness_score: Optional[float] = None
    flags: Set[str] = field(default_factory=set)
    annotations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Cancellation (Choice Clause: modules can flag, but only the
    # host can actually cancel — and only for safety reasons)
    cancelled: bool = False
    cancel_reason: str = ""


@dataclass
class HookResult:
    """Result from a single hook invocation.

    Attributes:
        module_name: Which module produced this result.
        phase: Which lifecycle phase this was.
        success: Whether the hook ran without error.
        duration_ms: How long the hook took.
        modifications: What the hook changed (for transparency).
        error: Error message if the hook failed.
    """
    module_name: str = ""
    phase: HookPhase = HookPhase.PRE_ROUTE
    success: bool = True
    duration_ms: float = 0.0
    modifications: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


# ---------------------------------------------------------------------------
# ET Module Manifest
# ---------------------------------------------------------------------------

@dataclass
class ETModuleManifest:
    """Parsed et_module.json manifest.

    Attributes:
        name: Module identifier (e.g., "trollguard", "ctem").
        version: Semantic version string.
        description: Human-readable description.
        author: Module author.
        hooks: Which lifecycle phases this module subscribes to.
        ng_config: NG-Lite configuration overrides for this module.
        capabilities: What this module provides (e.g., ["content_filter"]).
        dependencies: Other modules this one requires.
        priority: Hook execution priority (lower = runs first).
        enabled: Whether this module is active.
    """
    name: str = ""
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    hooks: List[str] = field(default_factory=list)
    ng_config: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 100
    enabled: bool = True

    @staticmethod
    def load(path: str) -> ETModuleManifest:
        """Load a manifest from an et_module.json file.

        Args:
            path: Path to the et_module.json file.

        Returns:
            Parsed ETModuleManifest.

        Raises:
            FileNotFoundError: If the manifest file doesn't exist.
            ValueError: If required fields are missing.
        """
        with open(path, "r") as f:
            data = json.load(f)

        name = data.get("name", "")
        if not name:
            raise ValueError(f"et_module.json at {path} missing 'name' field")

        return ETModuleManifest(
            name=name,
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            hooks=data.get("hooks", []),
            ng_config=data.get("ng_config", {}),
            capabilities=data.get("capabilities", []),
            dependencies=data.get("dependencies", []),
            priority=data.get("priority", 100),
            enabled=data.get("enabled", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest for logging and transparency."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "hooks": self.hooks,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "enabled": self.enabled,
        }


# ---------------------------------------------------------------------------
# ET Module Base Class
# ---------------------------------------------------------------------------

class ETModule:
    """Base class for all ET modules.

    Subclass this and implement the hook methods you need. Register
    hook subscriptions in your et_module.json manifest.

    The module host calls hooks in priority order (lower = first).
    If a hook raises an exception, the host logs the error and
    continues to the next module — one broken module doesn't take
    down the system.

    Usage:
        class TrollGuard(ETModule):
            def pre_route(self, ctx: HookContext) -> None:
                if self._detect_threat(ctx.message):
                    ctx.flags.add("threat_detected")
                    ctx.annotations[self.manifest.name] = {
                        "threat_score": 0.95,
                    }
    """

    def __init__(self, manifest: ETModuleManifest):
        self.manifest = manifest
        self._ng_lite: Optional[Any] = None

    @property
    def name(self) -> str:
        return self.manifest.name

    def initialize(self) -> None:
        """Called once when the module is registered.

        Override to set up resources, load models, connect to
        services, etc. Called after the manifest is validated but
        before any hooks run.
        """
        pass

    def shutdown(self) -> None:
        """Called when the module host is shutting down.

        Override to clean up resources, save state, close
        connections, etc.
        """
        pass

    def pre_route(self, ctx: HookContext) -> None:
        """Called before routing. Inspect/modify the request."""
        pass

    def post_route(self, ctx: HookContext) -> None:
        """Called after routing decision. Inspect/log/override."""
        pass

    def pre_response(self, ctx: HookContext) -> None:
        """Called after model response, before quality eval."""
        pass

    def post_response(self, ctx: HookContext) -> None:
        """Called after quality eval. Learning hooks run here."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Return module statistics for transparency/Observatory."""
        return {
            "name": self.manifest.name,
            "version": self.manifest.version,
            "enabled": self.manifest.enabled,
        }


# ---------------------------------------------------------------------------
# Module Registry & Hook Dispatcher
# ---------------------------------------------------------------------------

class ModuleRegistry:
    """Registry of loaded ET modules and hook lifecycle dispatcher.

    The registry:
    1. Loads modules from directories containing et_module.json
    2. Validates manifests and checks dependencies
    3. Dispatches hook lifecycle events in priority order
    4. Tracks hook execution for transparency and debugging

    Usage:
        registry = ModuleRegistry()
        registry.register(trollguard_module)
        registry.register(ctem_module)

        # Run hook lifecycle
        ctx = HookContext(message="Hello", request_id="req_1")
        results = registry.dispatch(HookPhase.PRE_ROUTE, ctx)
    """

    def __init__(self):
        self._modules: Dict[str, ETModule] = {}
        self._hook_history: List[HookResult] = []
        self._history_max = 1000

    def register(self, module: ETModule) -> None:
        """Register an ET module.

        Validates the manifest, checks dependencies, calls
        initialize(), and adds to the registry.

        Args:
            module: The ET module instance to register.

        Raises:
            ValueError: If manifest is invalid or dependencies unmet.
        """
        manifest = module.manifest
        if not manifest.name:
            raise ValueError("Module manifest missing 'name'")

        if manifest.name in self._modules:
            raise ValueError(
                f"Module '{manifest.name}' already registered"
            )

        # Check dependencies
        for dep in manifest.dependencies:
            if dep not in self._modules:
                raise ValueError(
                    f"Module '{manifest.name}' requires '{dep}' "
                    f"which is not registered"
                )

        # Initialize the module
        try:
            module.initialize()
        except Exception as e:
            raise ValueError(
                f"Module '{manifest.name}' initialization failed: {e}"
            ) from e

        self._modules[manifest.name] = module
        logger.info(
            "Registered ET module: %s v%s (priority=%d, hooks=%s)",
            manifest.name, manifest.version,
            manifest.priority, manifest.hooks,
        )

    def unregister(self, module_name: str) -> None:
        """Unregister and shut down a module.

        Args:
            module_name: The module to remove.
        """
        if module_name not in self._modules:
            return

        module = self._modules[module_name]
        try:
            module.shutdown()
        except Exception as e:
            logger.warning(
                "Module '%s' shutdown error: %s", module_name, e,
            )

        del self._modules[module_name]
        logger.info("Unregistered ET module: %s", module_name)

    def dispatch(
        self,
        phase: HookPhase,
        ctx: HookContext,
    ) -> List[HookResult]:
        """Dispatch a hook lifecycle event to all subscribed modules.

        Modules are called in priority order (lower priority number
        runs first). If a module raises an exception, the error is
        logged and the next module runs — one broken module doesn't
        take down the system.

        Args:
            phase: Which lifecycle phase to dispatch.
            ctx: The mutable context bag.

        Returns:
            List of HookResults from all modules that ran.
        """
        results: List[HookResult] = []

        # Get subscribed modules, sorted by priority
        subscribed = self._get_subscribed(phase)

        for module in subscribed:
            if not module.manifest.enabled:
                continue

            start = time.monotonic()
            result = HookResult(
                module_name=module.name,
                phase=phase,
            )

            try:
                handler = self._get_handler(module, phase)
                if handler:
                    handler(ctx)
                    result.success = True
            except Exception as e:
                result.success = False
                result.error = str(e)
                logger.warning(
                    "Hook %s.%s failed: %s",
                    module.name, phase.value, e,
                )

            result.duration_ms = (time.monotonic() - start) * 1000
            results.append(result)

            # Record in history
            self._record_history(result)

        return results

    def dispatch_all(self, ctx: HookContext) -> Dict[str, List[HookResult]]:
        """Dispatch all four lifecycle phases in order.

        Convenience method for the full lifecycle. Stops early if
        the context is cancelled after pre_route.

        Args:
            ctx: The mutable context bag.

        Returns:
            Dict mapping phase name to list of HookResults.
        """
        all_results: Dict[str, List[HookResult]] = {}

        for phase in HookPhase:
            results = self.dispatch(phase, ctx)
            all_results[phase.value] = results

            # If cancelled during pre_route, stop the lifecycle
            if phase == HookPhase.PRE_ROUTE and ctx.cancelled:
                break

        return all_results

    def get_module(self, name: str) -> Optional[ETModule]:
        """Look up a registered module by name."""
        return self._modules.get(name)

    def get_all_modules(self) -> List[ETModule]:
        """Return all registered modules, sorted by priority."""
        return sorted(
            self._modules.values(),
            key=lambda m: m.manifest.priority,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Registry statistics for transparency."""
        module_stats = {}
        for name, module in self._modules.items():
            module_stats[name] = module.get_stats()

        # Recent hook execution summary
        recent = self._hook_history[-100:]
        phase_counts: Dict[str, int] = {}
        error_count = 0
        for result in recent:
            phase_counts[result.phase.value] = (
                phase_counts.get(result.phase.value, 0) + 1
            )
            if not result.success:
                error_count += 1

        return {
            "total_modules": len(self._modules),
            "modules": module_stats,
            "recent_hooks": {
                "total": len(recent),
                "by_phase": phase_counts,
                "errors": error_count,
            },
            "hook_history_size": len(self._hook_history),
        }

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _get_subscribed(self, phase: HookPhase) -> List[ETModule]:
        """Get modules subscribed to this phase, sorted by priority."""
        subscribed = [
            m for m in self._modules.values()
            if phase.value in m.manifest.hooks and m.manifest.enabled
        ]
        subscribed.sort(key=lambda m: m.manifest.priority)
        return subscribed

    @staticmethod
    def _get_handler(
        module: ETModule, phase: HookPhase,
    ) -> Optional[Callable]:
        """Get the hook method for a phase."""
        handler_map = {
            HookPhase.PRE_ROUTE: module.pre_route,
            HookPhase.POST_ROUTE: module.post_route,
            HookPhase.PRE_RESPONSE: module.pre_response,
            HookPhase.POST_RESPONSE: module.post_response,
        }
        return handler_map.get(phase)

    def _record_history(self, result: HookResult) -> None:
        """Append to bounded hook history."""
        self._hook_history.append(result)
        if len(self._hook_history) > self._history_max:
            self._hook_history = self._hook_history[-self._history_max:]
