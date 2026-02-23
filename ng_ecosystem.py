"""
NG Ecosystem — NeuroGraph Ecosystem Coordinator.

Manages NG-Lite instances across co-located ET modules and coordinates
peer-to-peer learning between them. This is the Tier 2 (peer-pooled)
implementation described in the NG Ecosystem spec.

Architecture:
    Tier 1 — Isolated: Each module has its own NG-Lite. No coordinator.
    Tier 2 — Peer-pooled: This coordinator connects co-located modules
             via NGPeerBridge, allowing shared learning without SaaS.
    Tier 3 — Full SaaS: The coordinator connects all modules to
             NeuroGraph SaaS via NGSaaSBridge.

The ecosystem coordinator:
    1. Maintains a registry of all NG-Lite instances by module ID
    2. Creates NGPeerBridge connections between co-located modules
    3. Provides a unified API for recording outcomes and getting
       recommendations that routes through the appropriate tier
    4. Handles transparent tier transitions (1→2→3 and back)

Ethical obligations (per NeuroGraph ETHICS.md):
    - Transparency: all cross-module learning is logged and queryable
    - Choice Clause: modules can opt out of shared learning
    - Type I error bias: when uncertain about sharing, don't share

Canonical source: https://github.com/greatnorthernfishguy-hub/NeuroGraph
License: AGPL-3.0

Author: Josh + Claude
Date: February 2026
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ng_lite import NGBridge, NGLite

logger = logging.getLogger("ng_ecosystem")


# ---------------------------------------------------------------------------
# NGPeerBridge — Tier 2 connectivity
# ---------------------------------------------------------------------------

class NGPeerBridge(NGBridge):
    """Connects two co-located NG-Lite instances for shared learning.

    When modules run together (e.g., TID + TrollGuard on the same host),
    they pool their pattern knowledge for mutual benefit. The bridge
    forwards outcomes and recommendations between peers.

    The module doesn't know or care whether its bridge partner is a
    sibling module or the full SaaS — both use the same NGBridge
    interface. Tier transitions are transparent.

    Design:
        - Outcomes are recorded in the local instance AND forwarded
          to the peer, tagged with the source module_id.
        - Recommendations merge local and peer suggestions, giving
          local knowledge a slight priority (it's more relevant).
        - Novelty detection considers patterns from both instances.

    Attributes:
        peer: The NG-Lite instance on the other side of the bridge.
        peer_module_id: Module ID of the peer.
        local_weight: Weight given to local recommendations (0.0-1.0).
        peer_weight: Weight given to peer recommendations (0.0-1.0).
    """

    def __init__(
        self,
        peer: NGLite,
        peer_module_id: str = "",
        local_weight: float = 0.6,
        peer_weight: float = 0.4,
    ):
        self._peer = peer
        self._peer_module_id = peer_module_id or peer.module_id
        self._local_weight = local_weight
        self._peer_weight = peer_weight
        self._connected = True
        self._outcomes_forwarded = 0
        self._recommendations_served = 0

    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Activate the bridge connection."""
        self._connected = True
        logger.info(
            "NGPeerBridge connected to peer '%s'",
            self._peer_module_id,
        )

    def disconnect(self) -> None:
        """Deactivate the bridge connection."""
        self._connected = False
        logger.info(
            "NGPeerBridge disconnected from peer '%s'",
            self._peer_module_id,
        )

    def record_outcome(
        self,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        module_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Forward outcome to peer for cross-module learning.

        The peer learns from this module's experience. The target_id
        is prefixed with the source module to avoid namespace collisions.
        """
        if not self._connected:
            return None

        # Tag the outcome with source module for transparency
        peer_metadata = dict(metadata or {})
        peer_metadata["source_module"] = module_id
        peer_metadata["cross_module"] = True

        try:
            peer_result = self._peer.record_outcome(
                embedding=embedding,
                target_id=target_id,
                success=success,
                metadata=peer_metadata,
            )
            self._outcomes_forwarded += 1
            return {
                "peer_module": self._peer_module_id,
                "peer_weight": peer_result.get("weight_after", 0.0),
                "cross_module": True,
            }
        except Exception as e:
            logger.warning(
                "NGPeerBridge record_outcome to '%s' failed: %s",
                self._peer_module_id, e,
            )
            return None

    def get_recommendations(
        self,
        embedding: np.ndarray,
        module_id: str,
        top_k: int = 3,
    ) -> Optional[List[Tuple[str, float, str]]]:
        """Get recommendations from peer, weighted by peer_weight.

        Returns (target_id, confidence, reasoning) tuples. The
        confidence is scaled by peer_weight since peer knowledge
        is less directly relevant than local knowledge.
        """
        if not self._connected:
            return None

        try:
            peer_recs = self._peer.get_recommendations(
                embedding=embedding, top_k=top_k,
            )
            self._recommendations_served += 1

            return [
                (
                    target_id,
                    weight * self._peer_weight,
                    f"Peer recommendation from '{self._peer_module_id}'",
                )
                for target_id, weight in peer_recs
            ]
        except Exception as e:
            logger.warning(
                "NGPeerBridge get_recommendations from '%s' failed: %s",
                self._peer_module_id, e,
            )
            return None

    def detect_novelty(
        self,
        embedding: np.ndarray,
        module_id: str,
    ) -> Optional[float]:
        """Get novelty score from peer's perspective.

        If the peer has seen this pattern before, the novelty is
        lower — cross-module pattern recognition.
        """
        if not self._connected:
            return None

        try:
            return self._peer.detect_novelty(embedding)
        except Exception as e:
            logger.warning(
                "NGPeerBridge detect_novelty from '%s' failed: %s",
                self._peer_module_id, e,
            )
            return None

    def sync_state(
        self,
        local_state: Dict[str, Any],
        module_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Sync is a no-op for peer bridges.

        Peer bridges forward outcomes in real-time. There's no
        batched state sync needed — unlike SaaS, which might
        accumulate offline and sync periodically.
        """
        return {"synced": True, "peer": self._peer_module_id}

    def get_stats(self) -> Dict[str, Any]:
        """Bridge statistics for transparency."""
        return {
            "type": "peer",
            "peer_module": self._peer_module_id,
            "connected": self._connected,
            "outcomes_forwarded": self._outcomes_forwarded,
            "recommendations_served": self._recommendations_served,
            "local_weight": self._local_weight,
            "peer_weight": self._peer_weight,
        }


# ---------------------------------------------------------------------------
# NGSaaSBridge — Tier 3 connectivity (stub)
# ---------------------------------------------------------------------------

class NGSaaSBridge(NGBridge):
    """Connects to full NeuroGraph SaaS for cross-module intelligence.

    Tier 3 connectivity. When NeuroGraph SaaS is available, this bridge
    delegates to the full substrate for:
        - Cross-module STDP (spike-timing dependent plasticity)
        - Hypergraph capabilities
        - Predictive coding
        - Global pattern recognition across all connected modules

    This is a stub implementation that returns None for all operations,
    causing NG-Lite to fall back to local learning. When NeuroGraph SaaS
    is built, this bridge will connect to it via HTTP/gRPC.
    """

    def __init__(self, endpoint: str = "", api_key: str = ""):
        self._endpoint = endpoint
        self._api_key = api_key
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def record_outcome(
        self,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        module_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Forward outcome to NeuroGraph SaaS. Stub: returns None."""
        return None

    def get_recommendations(
        self,
        embedding: np.ndarray,
        module_id: str,
        top_k: int = 3,
    ) -> Optional[List[Tuple[str, float, str]]]:
        """Get recommendations from NeuroGraph SaaS. Stub: returns None."""
        return None

    def detect_novelty(
        self,
        embedding: np.ndarray,
        module_id: str,
    ) -> Optional[float]:
        """Get novelty from NeuroGraph SaaS. Stub: returns None."""
        return None

    def sync_state(
        self,
        local_state: Dict[str, Any],
        module_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Sync with NeuroGraph SaaS. Stub: returns None."""
        return None


# ---------------------------------------------------------------------------
# NG Ecosystem Coordinator
# ---------------------------------------------------------------------------

@dataclass
class ModuleNGState:
    """Tracks NG-Lite state for a single module within the ecosystem."""
    module_id: str
    ng_lite: NGLite
    tier: int = 1  # 1=isolated, 2=peer-pooled, 3=saas
    peers: List[str] = field(default_factory=list)
    opt_out_sharing: bool = False


class NGEcosystem:
    """Coordinates NG-Lite instances across co-located ET modules.

    The ecosystem coordinator is a singleton that all modules register
    with. It manages:
        1. NG-Lite instance creation and lifecycle
        2. Peer bridge connections between co-located modules
        3. Tier transitions (1→2→3 and fallback)
        4. Unified stats and transparency

    Usage:
        eco = NGEcosystem()

        # Register modules
        tid_ng = eco.register_module("inference_difference")
        tg_ng = eco.register_module("trollguard")

        # Ecosystem auto-connects peers at Tier 2
        eco.connect_peers()

        # Modules use their NG-Lite instances normally
        tid_ng.record_outcome(embedding, "model_a", success=True)

        # Get ecosystem-wide stats
        stats = eco.get_stats()
    """

    def __init__(self):
        self._modules: Dict[str, ModuleNGState] = {}
        self._saas_bridge: Optional[NGSaaSBridge] = None

    def register_module(
        self,
        module_id: str,
        ng_config: Optional[Dict[str, Any]] = None,
        opt_out_sharing: bool = False,
    ) -> NGLite:
        """Register a module and create its NG-Lite instance.

        Args:
            module_id: Unique module identifier.
            ng_config: NG-Lite config overrides for this module.
            opt_out_sharing: If True, this module won't participate
                in peer-to-peer learning (Choice Clause).

        Returns:
            The module's NG-Lite instance.
        """
        if module_id in self._modules:
            return self._modules[module_id].ng_lite

        ng = NGLite(module_id=module_id, config=ng_config)

        state = ModuleNGState(
            module_id=module_id,
            ng_lite=ng,
            tier=1,
            opt_out_sharing=opt_out_sharing,
        )
        self._modules[module_id] = state

        logger.info(
            "NG Ecosystem: registered module '%s' at Tier 1%s",
            module_id,
            " (opt-out sharing)" if opt_out_sharing else "",
        )

        return ng

    def unregister_module(self, module_id: str) -> None:
        """Remove a module from the ecosystem.

        Disconnects all peer bridges and removes the module's
        NG-Lite instance.
        """
        if module_id not in self._modules:
            return

        state = self._modules[module_id]

        # Disconnect bridge if connected
        state.ng_lite.disconnect_bridge()
        state.peers.clear()

        del self._modules[module_id]
        logger.info("NG Ecosystem: unregistered module '%s'", module_id)

    def get_ng_lite(self, module_id: str) -> Optional[NGLite]:
        """Get the NG-Lite instance for a module."""
        state = self._modules.get(module_id)
        return state.ng_lite if state else None

    # -------------------------------------------------------------------
    # Peer Connectivity (Tier 2)
    # -------------------------------------------------------------------

    def connect_peers(self) -> int:
        """Connect all eligible co-located modules as peers.

        Creates NGPeerBridge connections between all modules that
        haven't opted out of sharing. Each module gets a bridge to
        every other eligible module.

        Returns:
            Number of peer connections created.
        """
        eligible = [
            s for s in self._modules.values()
            if not s.opt_out_sharing
        ]

        if len(eligible) < 2:
            return 0

        connections = 0
        for i, state_a in enumerate(eligible):
            for state_b in eligible[i + 1:]:
                # Create bidirectional bridges
                bridge_a_to_b = NGPeerBridge(
                    peer=state_b.ng_lite,
                    peer_module_id=state_b.module_id,
                )
                bridge_b_to_a = NGPeerBridge(
                    peer=state_a.ng_lite,
                    peer_module_id=state_a.module_id,
                )

                # Connect bridges (last bridge wins — for simplicity,
                # connect the first peer only. Multi-peer bridging
                # would require a multi-bridge wrapper.)
                if not state_a.peers:
                    state_a.ng_lite.connect_bridge(bridge_a_to_b)
                if not state_b.peers:
                    state_b.ng_lite.connect_bridge(bridge_b_to_a)

                state_a.peers.append(state_b.module_id)
                state_b.peers.append(state_a.module_id)
                state_a.tier = 2
                state_b.tier = 2
                connections += 1

                logger.info(
                    "NG Ecosystem: peer bridge %s <-> %s",
                    state_a.module_id, state_b.module_id,
                )

        return connections

    def disconnect_peers(self) -> None:
        """Disconnect all peer bridges, falling back to Tier 1."""
        for state in self._modules.values():
            if state.tier >= 2:
                state.ng_lite.disconnect_bridge()
                state.peers.clear()
                state.tier = 1

        logger.info("NG Ecosystem: all peer bridges disconnected")

    # -------------------------------------------------------------------
    # SaaS Connectivity (Tier 3)
    # -------------------------------------------------------------------

    def connect_saas(
        self,
        endpoint: str,
        api_key: str = "",
    ) -> bool:
        """Connect all modules to NeuroGraph SaaS (Tier 3).

        Upgrades from Tier 1 or 2 to Tier 3. The SaaS bridge
        replaces peer bridges — SaaS provides superset functionality.

        Returns:
            True if connection succeeded.
        """
        self._saas_bridge = NGSaaSBridge(
            endpoint=endpoint, api_key=api_key,
        )

        # SaaS bridge is a stub — would connect here in production
        # For now, always return False (SaaS not yet available)
        logger.info(
            "NG Ecosystem: SaaS bridge created (endpoint=%s) — "
            "stub implementation, falling back to local/peer",
            endpoint,
        )
        return False

    # -------------------------------------------------------------------
    # Ecosystem Operations
    # -------------------------------------------------------------------

    def record_outcome(
        self,
        module_id: str,
        embedding: np.ndarray,
        target_id: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record an outcome for a module through the ecosystem.

        Convenience method that routes through the module's NG-Lite
        instance (which may forward through bridges).

        Args:
            module_id: Which module is recording.
            embedding: The input pattern embedding.
            target_id: What was chosen.
            success: Whether the outcome was successful.
            metadata: Additional context.

        Returns:
            Learning result dict, or None if module not found.
        """
        state = self._modules.get(module_id)
        if not state:
            return None

        return state.ng_lite.record_outcome(
            embedding=embedding,
            target_id=target_id,
            success=success,
            metadata=metadata,
        )

    def get_recommendations(
        self,
        module_id: str,
        embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Get recommendations for a module through the ecosystem.

        Args:
            module_id: Which module is asking.
            embedding: The input pattern embedding.
            top_k: Maximum recommendations.

        Returns:
            List of (target_id, confidence) tuples.
        """
        state = self._modules.get(module_id)
        if not state:
            return []

        return state.ng_lite.get_recommendations(
            embedding=embedding, top_k=top_k,
        )

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save_all(self, base_dir: str) -> None:
        """Save all module NG-Lite states to disk.

        Each module's state is saved to {base_dir}/{module_id}_ng.json.

        Args:
            base_dir: Directory to save state files.
        """
        for module_id, state in self._modules.items():
            filepath = os.path.join(base_dir, f"{module_id}_ng.json")
            try:
                state.ng_lite.save(filepath)
            except Exception as e:
                logger.warning(
                    "Failed to save NG state for '%s': %s",
                    module_id, e,
                )

    def load_all(self, base_dir: str) -> None:
        """Load all module NG-Lite states from disk.

        Each module's state is loaded from {base_dir}/{module_id}_ng.json
        if the file exists.

        Args:
            base_dir: Directory containing state files.
        """
        for module_id, state in self._modules.items():
            filepath = os.path.join(base_dir, f"{module_id}_ng.json")
            if os.path.exists(filepath):
                try:
                    state.ng_lite.load(filepath)
                    logger.info(
                        "Loaded NG state for '%s' from %s",
                        module_id, filepath,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to load NG state for '%s': %s",
                        module_id, e,
                    )

    # -------------------------------------------------------------------
    # Stats & Transparency
    # -------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Ecosystem-wide statistics for transparency/Observatory."""
        module_stats = {}
        for module_id, state in self._modules.items():
            module_stats[module_id] = {
                "tier": state.tier,
                "peers": state.peers,
                "opt_out_sharing": state.opt_out_sharing,
                "ng_stats": state.ng_lite.get_stats(),
            }

        return {
            "total_modules": len(self._modules),
            "modules": module_stats,
            "saas_connected": (
                self._saas_bridge is not None
                and self._saas_bridge.is_connected()
            ),
        }


# Need os for save/load paths
import os
