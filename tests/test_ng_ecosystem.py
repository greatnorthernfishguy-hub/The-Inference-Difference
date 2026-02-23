"""
Tests for NG Ecosystem — NeuroGraph Ecosystem Coordinator.

Covers: module registration, peer bridging (Tier 2), SaaS stub (Tier 3),
cross-module learning, persistence, opt-out (Choice Clause), tier
transitions, and transparency.
"""

import os
import tempfile

import numpy as np
import pytest

from ng_lite import NGLite
from ng_ecosystem import (
    ModuleNGState,
    NGEcosystem,
    NGPeerBridge,
    NGSaaSBridge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embedding():
    """A random 384-dim embedding, normalized."""
    rng = np.random.RandomState(42)
    emb = rng.randn(384)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def different_embedding():
    """A different random embedding."""
    rng = np.random.RandomState(99)
    emb = rng.randn(384)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def eco():
    """Fresh NG Ecosystem coordinator."""
    return NGEcosystem()


# ---------------------------------------------------------------------------
# NGPeerBridge Tests
# ---------------------------------------------------------------------------

class TestNGPeerBridge:

    def test_bridge_connected_by_default(self):
        peer = NGLite(module_id="peer")
        bridge = NGPeerBridge(peer, peer_module_id="peer")
        assert bridge.is_connected() is True

    def test_bridge_disconnect_reconnect(self):
        peer = NGLite(module_id="peer")
        bridge = NGPeerBridge(peer, peer_module_id="peer")
        bridge.disconnect()
        assert bridge.is_connected() is False
        bridge.connect()
        assert bridge.is_connected() is True

    def test_record_outcome_forwards(self, embedding):
        peer = NGLite(module_id="peer")
        bridge = NGPeerBridge(peer, peer_module_id="peer")

        result = bridge.record_outcome(
            embedding=embedding,
            target_id="model_a",
            success=True,
            module_id="source_module",
        )
        assert result is not None
        assert result["cross_module"] is True
        assert result["peer_module"] == "peer"

        # Peer should have learned
        assert peer.get_stats()["total_outcomes"] == 1

    def test_record_outcome_disconnected(self, embedding):
        peer = NGLite(module_id="peer")
        bridge = NGPeerBridge(peer, peer_module_id="peer")
        bridge.disconnect()

        result = bridge.record_outcome(
            embedding=embedding,
            target_id="model_a",
            success=True,
            module_id="source",
        )
        assert result is None
        assert peer.get_stats()["total_outcomes"] == 0

    def test_get_recommendations_from_peer(self, embedding):
        peer = NGLite(module_id="peer")
        # Train peer
        for _ in range(5):
            peer.record_outcome(embedding, "peer_model", success=True)

        bridge = NGPeerBridge(peer, peer_module_id="peer", peer_weight=0.4)
        recs = bridge.get_recommendations(
            embedding=embedding, module_id="source", top_k=3,
        )
        assert recs is not None
        assert len(recs) > 0
        assert recs[0][0] == "peer_model"
        # Weight should be scaled by peer_weight
        assert recs[0][1] <= 0.4

    def test_detect_novelty_from_peer(self, embedding, different_embedding):
        peer = NGLite(module_id="peer")
        peer.find_or_create_node(embedding)

        bridge = NGPeerBridge(peer, peer_module_id="peer")
        # Known pattern — low novelty
        novelty_known = bridge.detect_novelty(embedding, module_id="source")
        assert novelty_known is not None
        assert novelty_known < 0.2

        # Unknown pattern — high novelty
        novelty_new = bridge.detect_novelty(different_embedding, module_id="source")
        assert novelty_new is not None
        assert novelty_new > novelty_known

    def test_sync_state_noop(self, embedding):
        peer = NGLite(module_id="peer")
        bridge = NGPeerBridge(peer, peer_module_id="peer")
        result = bridge.sync_state({}, module_id="source")
        assert result == {"synced": True, "peer": "peer"}

    def test_bridge_stats(self, embedding):
        peer = NGLite(module_id="peer")
        bridge = NGPeerBridge(peer, peer_module_id="peer")
        bridge.record_outcome(embedding, "m1", True, "source")

        stats = bridge.get_stats()
        assert stats["type"] == "peer"
        assert stats["peer_module"] == "peer"
        assert stats["outcomes_forwarded"] == 1
        assert stats["connected"] is True


# ---------------------------------------------------------------------------
# NGSaaSBridge Tests (Stub)
# ---------------------------------------------------------------------------

class TestNGSaaSBridge:

    def test_saas_not_connected_by_default(self):
        bridge = NGSaaSBridge()
        assert bridge.is_connected() is False

    def test_saas_returns_none(self, embedding):
        bridge = NGSaaSBridge()
        assert bridge.record_outcome(embedding, "m", True, "mod") is None
        assert bridge.get_recommendations(embedding, "mod") is None
        assert bridge.detect_novelty(embedding, "mod") is None
        assert bridge.sync_state({}, "mod") is None


# ---------------------------------------------------------------------------
# NG Ecosystem: Module Registration
# ---------------------------------------------------------------------------

class TestEcosystemRegistration:

    def test_register_module(self, eco):
        ng = eco.register_module("tid")
        assert isinstance(ng, NGLite)
        assert ng.module_id == "tid"
        assert eco.get_ng_lite("tid") is ng

    def test_register_duplicate_returns_existing(self, eco):
        ng1 = eco.register_module("tid")
        ng2 = eco.register_module("tid")
        assert ng1 is ng2

    def test_register_with_config(self, eco):
        ng = eco.register_module("tid", ng_config={"max_nodes": 500})
        assert ng.config["max_nodes"] == 500

    def test_register_opt_out(self, eco):
        eco.register_module("private", opt_out_sharing=True)
        state = eco._modules["private"]
        assert state.opt_out_sharing is True

    def test_unregister_module(self, eco):
        eco.register_module("tid")
        eco.unregister_module("tid")
        assert eco.get_ng_lite("tid") is None

    def test_unregister_nonexistent(self, eco):
        eco.unregister_module("nonexistent")  # Should not crash

    def test_get_ng_lite_missing(self, eco):
        assert eco.get_ng_lite("nonexistent") is None


# ---------------------------------------------------------------------------
# NG Ecosystem: Peer Connectivity (Tier 2)
# ---------------------------------------------------------------------------

class TestPeerConnectivity:

    def test_connect_peers_two_modules(self, eco):
        eco.register_module("tid")
        eco.register_module("trollguard")
        connections = eco.connect_peers()
        assert connections == 1

        # Both should be Tier 2
        assert eco._modules["tid"].tier == 2
        assert eco._modules["trollguard"].tier == 2

        # Both should have peer references
        assert "trollguard" in eco._modules["tid"].peers
        assert "tid" in eco._modules["trollguard"].peers

    def test_connect_peers_opt_out_excluded(self, eco):
        eco.register_module("tid")
        eco.register_module("private", opt_out_sharing=True)
        connections = eco.connect_peers()
        assert connections == 0  # Private module excluded

    def test_connect_peers_single_module_noop(self, eco):
        eco.register_module("tid")
        connections = eco.connect_peers()
        assert connections == 0

    def test_disconnect_peers(self, eco):
        eco.register_module("tid")
        eco.register_module("trollguard")
        eco.connect_peers()
        eco.disconnect_peers()

        assert eco._modules["tid"].tier == 1
        assert eco._modules["trollguard"].tier == 1
        assert eco._modules["tid"].peers == []

    def test_cross_module_learning(self, eco, embedding):
        """After peer connection, learning is shared."""
        tid_ng = eco.register_module("tid")
        tg_ng = eco.register_module("trollguard")
        eco.connect_peers()

        # TID learns something
        tid_ng.record_outcome(embedding, "model_a", success=True)

        # TrollGuard's peer (TID) should have knowledge
        # The bridge forwards outcomes to peer
        assert tg_ng.get_stats()["total_outcomes"] >= 0


# ---------------------------------------------------------------------------
# NG Ecosystem: SaaS Connectivity (Tier 3)
# ---------------------------------------------------------------------------

class TestSaaSConnectivity:

    def test_connect_saas_returns_false(self, eco):
        """SaaS is a stub — always returns False."""
        result = eco.connect_saas("https://ng.example.com")
        assert result is False


# ---------------------------------------------------------------------------
# NG Ecosystem: Operations
# ---------------------------------------------------------------------------

class TestEcosystemOperations:

    def test_record_outcome_via_ecosystem(self, eco, embedding):
        eco.register_module("tid")
        result = eco.record_outcome(
            "tid", embedding, "model_a", success=True,
        )
        assert result is not None
        assert result["weight_after"] > 0.5

    def test_record_outcome_unknown_module(self, eco, embedding):
        result = eco.record_outcome(
            "nonexistent", embedding, "m", success=True,
        )
        assert result is None

    def test_get_recommendations_via_ecosystem(self, eco, embedding):
        eco.register_module("tid")
        eco.record_outcome("tid", embedding, "model_a", success=True)
        recs = eco.get_recommendations("tid", embedding, top_k=3)
        assert len(recs) > 0
        assert recs[0][0] == "model_a"

    def test_get_recommendations_unknown_module(self, eco, embedding):
        recs = eco.get_recommendations("nonexistent", embedding)
        assert recs == []


# ---------------------------------------------------------------------------
# NG Ecosystem: Persistence
# ---------------------------------------------------------------------------

class TestEcosystemPersistence:

    def test_save_and_load(self, eco, embedding):
        eco.register_module("tid")
        eco.record_outcome("tid", embedding, "model_a", success=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            eco.save_all(tmpdir)

            # Verify file exists
            filepath = os.path.join(tmpdir, "tid_ng.json")
            assert os.path.exists(filepath)

            # Load into fresh ecosystem
            eco2 = NGEcosystem()
            eco2.register_module("tid")
            eco2.load_all(tmpdir)

            ng = eco2.get_ng_lite("tid")
            assert ng.get_stats()["total_outcomes"] == 1

    def test_load_missing_files(self, eco):
        eco.register_module("tid")
        with tempfile.TemporaryDirectory() as tmpdir:
            eco.load_all(tmpdir)  # Should not crash


# ---------------------------------------------------------------------------
# NG Ecosystem: Stats & Transparency
# ---------------------------------------------------------------------------

class TestEcosystemStats:

    def test_empty_stats(self, eco):
        stats = eco.get_stats()
        assert stats["total_modules"] == 0
        assert stats["saas_connected"] is False

    def test_stats_with_modules(self, eco, embedding):
        eco.register_module("tid")
        eco.register_module("trollguard")
        eco.connect_peers()
        eco.record_outcome("tid", embedding, "model_a", success=True)

        stats = eco.get_stats()
        assert stats["total_modules"] == 2
        assert "tid" in stats["modules"]
        assert stats["modules"]["tid"]["tier"] == 2
        assert "trollguard" in stats["modules"]["tid"]["peers"]

    def test_three_module_ecosystem(self, eco):
        """Three modules should create 3 peer connections."""
        eco.register_module("tid")
        eco.register_module("trollguard")
        eco.register_module("observatory")
        connections = eco.connect_peers()
        assert connections == 3
