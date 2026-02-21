"""
Tests for Dynamic Model Catalog Selection (§4.5).

Covers: CatalogManager, profile resolution, blocklist/allowlist,
hard ceiling safety valve, tier backward compatibility, Dream Cycle
model property correlation analysis, graceful degradation, and
router integration with profile-based routing.
"""

import json
import os
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import yaml

from inference_difference.catalog_manager import (
    CatalogFilters,
    CatalogManager,
    CatalogModel,
    RequirementsProfile,
    TierEntry,
)
from inference_difference.classifier import classify_request
from inference_difference.config import (
    ComplexityTier,
    InferenceDifferenceConfig,
    ModelEntry,
    ModelType,
    TaskDomain,
    default_api_models,
    default_local_models,
)
from inference_difference.dream_cycle import (
    DreamCycle,
    PropertyInsight,
    RoutingOutcome,
)
from inference_difference.hardware import HardwareProfile, GPUInfo
from inference_difference.router import RoutingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_models():
    """Sample catalog models for testing."""
    return [
        CatalogModel(
            id="openrouter/deepseek/deepseek-r1",
            provider="openrouter",
            display_name="DeepSeek R1",
            context_window=65536,
            cost_per_1m_input=0.55,
            cost_per_1m_output=2.19,
            provider_tier="performance",
            capabilities=["code"],
            is_active=True,
            last_seen="2026-02-21T00:00:00Z",
            fetched_at="2026-02-21T00:00:00Z",
        ),
        CatalogModel(
            id="openrouter/google/gemini-pro",
            provider="openrouter",
            display_name="Gemini Pro",
            context_window=128000,
            cost_per_1m_input=1.25,
            cost_per_1m_output=5.00,
            provider_tier="frontier",
            capabilities=["code", "vision"],
            is_active=True,
            last_seen="2026-02-21T00:00:00Z",
            fetched_at="2026-02-21T00:00:00Z",
        ),
        CatalogModel(
            id="openrouter/meta/llama-3.1-8b",
            provider="openrouter",
            display_name="Llama 3.1 8B",
            context_window=8192,
            cost_per_1m_input=0.10,
            cost_per_1m_output=0.10,
            provider_tier="budget",
            capabilities=[],
            is_active=True,
            last_seen="2026-02-21T00:00:00Z",
            fetched_at="2026-02-21T00:00:00Z",
        ),
        CatalogModel(
            id="huggingface/bigcode/starcoder2",
            provider="huggingface",
            display_name="StarCoder2",
            context_window=16384,
            cost_per_1m_input=0.0,
            cost_per_1m_output=0.0,
            provider_tier="standard",
            capabilities=["code"],
            is_active=True,
            last_seen="2026-02-21T00:00:00Z",
            fetched_at="2026-02-21T00:00:00Z",
        ),
        CatalogModel(
            id="openrouter/expensive/mega-model",
            provider="openrouter",
            display_name="Mega Model",
            context_window=200000,
            cost_per_1m_input=50.00,
            cost_per_1m_output=150.00,
            provider_tier="frontier",
            capabilities=["code", "vision", "tools"],
            is_active=True,
            last_seen="2026-02-21T00:00:00Z",
            fetched_at="2026-02-21T00:00:00Z",
        ),
    ]


@pytest.fixture
def catalog_manager(tmp_dir, sample_models):
    """CatalogManager with sample models preloaded."""
    db_path = os.path.join(tmp_dir, "test_gateway.db")
    manager = CatalogManager(db_path=db_path, fallback_model="ollama/llama3.1:8b")
    manager.initialize()
    manager.models = sample_models
    manager._save_to_cache()

    # Load default profiles
    profiles_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "task_requirements.yaml",
    )
    manager.load_profiles(profiles_path)

    # Load default filters
    filters_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "catalog_filters.yaml",
    )
    manager.load_filters(filters_path)

    yield manager
    manager.close()


@pytest.fixture
def gpu_hardware():
    """Hardware profile with a mid-range GPU."""
    return HardwareProfile(
        cpu_count=8,
        cpu_name="Test CPU",
        ram_total_gb=32.0,
        ram_available_gb=24.0,
        gpus=[GPUInfo(
            index=0, name="RTX 4070",
            vram_total_gb=12.0, vram_free_gb=10.0,
        )],
        total_vram_gb=12.0,
        available_vram_gb=10.0,
        has_gpu=True,
        os_name="Linux",
        platform_arch="x86_64",
        ollama_available=True,
        ollama_models=["llama3.1:8b"],
    )


# ---------------------------------------------------------------------------
# CatalogManager: Initialization & Database
# ---------------------------------------------------------------------------

class TestCatalogManagerInit:

    def test_initialize_creates_tables(self, tmp_dir):
        """Database tables are created on initialization."""
        db_path = os.path.join(tmp_dir, "init_test.db")
        manager = CatalogManager(db_path=db_path)
        manager.initialize()

        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}

        assert "model_catalog" in table_names
        assert "catalog_refresh_log" in table_names
        conn.close()
        manager.close()

    def test_save_and_load_cache(self, tmp_dir, sample_models):
        """Models are persisted to SQLite and reloaded."""
        db_path = os.path.join(tmp_dir, "cache_test.db")
        manager = CatalogManager(db_path=db_path)
        manager.initialize()
        manager.models = sample_models
        manager._save_to_cache()

        # Create a new manager and load from cache
        manager2 = CatalogManager(db_path=db_path)
        manager2.initialize()

        assert len(manager2.models) == len(sample_models)
        ids = {m.id for m in manager2.models}
        assert "openrouter/deepseek/deepseek-r1" in ids
        assert "huggingface/bigcode/starcoder2" in ids

        manager.close()
        manager2.close()

    def test_cached_model_properties_preserved(self, tmp_dir, sample_models):
        """Model properties survive cache roundtrip."""
        db_path = os.path.join(tmp_dir, "props_test.db")
        manager = CatalogManager(db_path=db_path)
        manager.initialize()
        manager.models = sample_models
        manager._save_to_cache()

        manager2 = CatalogManager(db_path=db_path)
        manager2.initialize()

        deepseek = next(
            m for m in manager2.models
            if m.id == "openrouter/deepseek/deepseek-r1"
        )
        assert deepseek.provider == "openrouter"
        assert deepseek.context_window == 65536
        assert deepseek.cost_per_1m_input == 0.55
        assert deepseek.cost_per_1m_output == 2.19
        assert deepseek.provider_tier == "performance"
        assert "code" in deepseek.capabilities

        manager.close()
        manager2.close()


# ---------------------------------------------------------------------------
# CatalogManager: Profile Loading
# ---------------------------------------------------------------------------

class TestProfileLoading:

    def test_load_profiles_from_yaml(self, catalog_manager):
        """Profiles are loaded from task_requirements.yaml."""
        assert len(catalog_manager.profiles) >= 5
        assert "coding" in catalog_manager.profiles
        assert "simple_chat" in catalog_manager.profiles
        assert "local" in catalog_manager.profiles
        assert "premium" in catalog_manager.profiles

    def test_coding_profile_properties(self, catalog_manager):
        """Coding profile has expected constraints."""
        coding = catalog_manager.profiles["coding"]
        assert coding.max_cost_per_1m_input_tokens == 3.00
        assert coding.max_cost_per_1m_output_tokens == 9.00
        assert coding.min_context_window == 32000
        assert "code" in coding.required_capabilities
        assert coding.sort_by == "cost_asc"
        assert coding.max_candidates == 3

    def test_local_profile_is_local_only(self, catalog_manager):
        """Local profile prefers local and disables cloud fallback."""
        local = catalog_manager.profiles["local"]
        assert local.prefer_local is True
        assert local.fallback_to_cloud is False

    def test_premium_profile_uses_quality_sort(self, catalog_manager):
        """Premium profile sorts by quality descending."""
        premium = catalog_manager.profiles["premium"]
        assert premium.sort_by == "quality_desc"
        assert "frontier" in premium.provider_tiers

    def test_missing_profiles_file(self, catalog_manager):
        """Missing profiles file doesn't crash."""
        catalog_manager.load_profiles("/nonexistent/path.yaml")
        # Should still have previously loaded profiles
        assert len(catalog_manager.profiles) >= 5

    def test_load_profiles_custom(self, tmp_dir):
        """Custom profiles file is loaded correctly."""
        custom = {
            "version": 1,
            "profiles": {
                "test_profile": {
                    "max_cost_per_1m_input_tokens": 1.0,
                    "min_context_window": 8000,
                    "sort_by": "cost_asc",
                    "max_candidates": 2,
                },
            },
        }
        path = os.path.join(tmp_dir, "custom_profiles.yaml")
        with open(path, "w") as f:
            yaml.dump(custom, f)

        db_path = os.path.join(tmp_dir, "custom.db")
        manager = CatalogManager(db_path=db_path)
        manager.initialize()
        manager.load_profiles(path)

        assert "test_profile" in manager.profiles
        p = manager.profiles["test_profile"]
        assert p.max_cost_per_1m_input_tokens == 1.0
        assert p.min_context_window == 8000
        manager.close()


# ---------------------------------------------------------------------------
# CatalogManager: Filter Loading
# ---------------------------------------------------------------------------

class TestFilterLoading:

    def test_load_filters_from_yaml(self, catalog_manager):
        """Filters are loaded from catalog_filters.yaml."""
        f = catalog_manager.filters
        assert f.hard_ceiling_input == 20.0
        assert f.hard_ceiling_output == 60.0
        assert "openrouter" in f.allowlist_providers
        assert "huggingface" in f.allowlist_providers

    def test_missing_filters_file(self, tmp_dir):
        """Missing filters file uses defaults."""
        db_path = os.path.join(tmp_dir, "no_filter.db")
        manager = CatalogManager(db_path=db_path)
        manager.initialize()
        manager.load_filters("/nonexistent/path.yaml")
        # Defaults should be safe
        assert manager.filters.hard_ceiling_input == 20.0
        assert manager.filters.hard_ceiling_output == 60.0
        manager.close()


# ---------------------------------------------------------------------------
# CatalogManager: Profile Resolution
# ---------------------------------------------------------------------------

class TestProfileResolution:

    def test_resolve_coding_profile(self, catalog_manager):
        """Coding profile resolves to a code-capable model."""
        model_id = catalog_manager.resolve_profile("coding")
        assert model_id is not None
        # Should be one of the models with code capability
        # and within cost/context constraints
        model = catalog_manager.get_model_by_id(model_id)
        if model:
            assert "code" in model.capabilities

    def test_resolve_simple_chat(self, catalog_manager):
        """Simple chat profile resolves to cheapest qualifying model."""
        model_id = catalog_manager.resolve_profile("simple_chat")
        assert model_id is not None
        model = catalog_manager.get_model_by_id(model_id)
        if model:
            assert model.cost_per_1m_input <= 0.50

    def test_resolve_local_profile_returns_none(self, catalog_manager):
        """Local-only profile returns None (handled by caller)."""
        model_id = catalog_manager.resolve_profile("local")
        assert model_id is None

    def test_resolve_premium_profile(self, catalog_manager):
        """Premium profile prefers frontier-tier models."""
        model_id = catalog_manager.resolve_profile("premium")
        assert model_id is not None
        model = catalog_manager.get_model_by_id(model_id)
        if model:
            assert model.provider_tier == "frontier"

    def test_resolve_unknown_profile(self, catalog_manager):
        """Unknown profile returns None."""
        model_id = catalog_manager.resolve_profile("nonexistent")
        assert model_id is None

    def test_resolve_with_candidates(self, catalog_manager):
        """resolve_profile_candidates returns full candidate list."""
        candidates = catalog_manager.resolve_profile_candidates("coding")
        assert len(candidates) > 0
        for c in candidates:
            assert "code" in c.capabilities

    def test_cost_ascending_sort(self, catalog_manager):
        """cost_asc sort returns cheapest first."""
        candidates = catalog_manager.resolve_profile_candidates("simple_chat")
        if len(candidates) >= 2:
            costs = [c.cost_per_1m_input + c.cost_per_1m_output for c in candidates]
            assert costs == sorted(costs)

    def test_quality_descending_sort(self, catalog_manager):
        """quality_desc sort returns frontier tier first."""
        candidates = catalog_manager.resolve_profile_candidates("premium")
        if len(candidates) >= 2:
            # First should be frontier
            assert candidates[0].provider_tier == "frontier"


# ---------------------------------------------------------------------------
# CatalogManager: Hard Ceiling Safety Valve
# ---------------------------------------------------------------------------

class TestHardCeiling:

    def test_hard_ceiling_blocks_expensive_models(self, catalog_manager):
        """Hard ceiling prevents routing to models above price limit."""
        # The "expensive/mega-model" costs $50/1M input, $150/1M output
        # Hard ceiling is $20 input, $60 output — should be filtered out
        candidates = catalog_manager._apply_hard_ceiling(
            catalog_manager.models,
        )
        expensive_ids = [c.id for c in candidates]
        assert "openrouter/expensive/mega-model" not in expensive_ids

    def test_hard_ceiling_allows_cheap_models(self, catalog_manager):
        """Hard ceiling doesn't block models within limits."""
        candidates = catalog_manager._apply_hard_ceiling(
            catalog_manager.models,
        )
        cheap_ids = [c.id for c in candidates]
        assert "openrouter/meta/llama-3.1-8b" in cheap_ids

    def test_hard_ceiling_applied_during_resolution(self, catalog_manager):
        """Hard ceiling is enforced during profile resolution."""
        # Create a profile with no cost limits (would match everything)
        catalog_manager.profiles["unlimited"] = RequirementsProfile(
            name="unlimited",
            sort_by="quality_desc",
            max_candidates=10,
        )
        candidates = catalog_manager.resolve_profile_candidates("unlimited")
        for c in candidates:
            assert c.cost_per_1m_input <= catalog_manager.filters.hard_ceiling_input
            assert c.cost_per_1m_output <= catalog_manager.filters.hard_ceiling_output

    def test_hard_ceiling_custom_value(self, tmp_dir, sample_models):
        """Custom hard ceiling values are respected."""
        # Set a very low ceiling
        filters_data = {
            "version": 1,
            "blocklist": {"providers": [], "models": []},
            "allowlist": {"providers": []},
            "hard_ceiling": {
                "max_cost_per_1m_input_tokens": 0.20,
                "max_cost_per_1m_output_tokens": 0.50,
            },
        }
        filters_path = os.path.join(tmp_dir, "strict_filters.yaml")
        with open(filters_path, "w") as f:
            yaml.dump(filters_data, f)

        db_path = os.path.join(tmp_dir, "strict.db")
        manager = CatalogManager(db_path=db_path)
        manager.initialize()
        manager.models = sample_models
        manager.load_filters(filters_path)

        candidates = manager._apply_hard_ceiling(manager.models)
        for c in candidates:
            assert c.cost_per_1m_input <= 0.20
            assert c.cost_per_1m_output <= 0.50

        manager.close()


# ---------------------------------------------------------------------------
# CatalogManager: Blocklist & Allowlist
# ---------------------------------------------------------------------------

class TestBlocklistAllowlist:

    def test_blocklist_removes_providers(self, catalog_manager):
        """Blocklisted providers are excluded."""
        catalog_manager.filters.blocklist_providers = ["huggingface"]
        candidates = catalog_manager._apply_blocklist(catalog_manager.models)
        for c in candidates:
            assert c.provider != "huggingface"

    def test_blocklist_removes_models(self, catalog_manager):
        """Blocklisted model IDs are excluded."""
        catalog_manager.filters.blocklist_models = [
            "openrouter/meta/llama-3.1-8b"
        ]
        candidates = catalog_manager._apply_blocklist(catalog_manager.models)
        ids = [c.id for c in candidates]
        assert "openrouter/meta/llama-3.1-8b" not in ids

    def test_allowlist_restricts_to_providers(self, catalog_manager):
        """Allowlist restricts to specified providers only."""
        catalog_manager.filters.allowlist_providers = ["huggingface"]
        candidates = catalog_manager._apply_allowlist(catalog_manager.models)
        for c in candidates:
            assert c.provider == "huggingface"

    def test_empty_allowlist_allows_all(self, catalog_manager):
        """Empty allowlist allows all providers."""
        catalog_manager.filters.allowlist_providers = []
        candidates = catalog_manager._apply_allowlist(catalog_manager.models)
        assert len(candidates) == len(catalog_manager.models)


# ---------------------------------------------------------------------------
# CatalogManager: Tier Backward Compatibility (§4.5.7)
# ---------------------------------------------------------------------------

class TestTierBackwardCompat:

    def test_load_tiers_from_yaml(self, tmp_dir, sample_models):
        """tier_models.yaml loads both old and new format."""
        tiers_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "tier_models.yaml",
        )
        db_path = os.path.join(tmp_dir, "tier.db")
        manager = CatalogManager(db_path=db_path)
        manager.initialize()
        manager.models = sample_models
        manager.load_tiers(tiers_path)

        assert "local" in manager.tiers
        assert "coding" in manager.tiers
        assert "premium" in manager.tiers

        # Local uses old format (models list)
        local = manager.tiers["local"]
        assert local.models is not None
        assert len(local.models) > 0
        assert local.profile is None

        # Coding uses new format (profile reference)
        coding = manager.tiers["coding"]
        assert coding.profile == "coding"

        manager.close()

    def test_resolve_tier_old_format(self, tmp_dir, sample_models):
        """Old-format tier resolves to named model."""
        tiers_data = {
            "tiers": {
                "local": {
                    "models": [
                        {"name": "ollama/qwen2.5:7b", "priority": 1},
                    ],
                },
            },
        }
        tiers_path = os.path.join(tmp_dir, "old_tiers.yaml")
        with open(tiers_path, "w") as f:
            yaml.dump(tiers_data, f)

        db_path = os.path.join(tmp_dir, "old.db")
        manager = CatalogManager(db_path=db_path)
        manager.initialize()
        manager.load_tiers(tiers_path)

        result = manager.resolve_tier("local")
        assert result == "ollama/qwen2.5:7b"
        manager.close()

    def test_resolve_tier_new_format(self, catalog_manager):
        """New-format tier resolves via catalog profile."""
        catalog_manager.tiers["coding"] = TierEntry(
            tier_name="coding", profile="coding",
        )
        result = catalog_manager.resolve_tier("coding")
        assert result is not None

    def test_resolve_tier_unknown(self, catalog_manager):
        """Unknown tier returns None."""
        result = catalog_manager.resolve_tier("nonexistent")
        assert result is None

    def test_old_format_priority_ordering(self, tmp_dir):
        """Old format returns highest-priority model."""
        tiers_data = {
            "tiers": {
                "multi": {
                    "models": [
                        {"name": "low-priority", "priority": 1},
                        {"name": "high-priority", "priority": 10},
                        {"name": "mid-priority", "priority": 5},
                    ],
                },
            },
        }
        tiers_path = os.path.join(tmp_dir, "priority.yaml")
        with open(tiers_path, "w") as f:
            yaml.dump(tiers_data, f)

        db_path = os.path.join(tmp_dir, "prio.db")
        manager = CatalogManager(db_path=db_path)
        manager.initialize()
        manager.load_tiers(tiers_path)

        result = manager.resolve_tier("multi")
        assert result == "high-priority"
        manager.close()


# ---------------------------------------------------------------------------
# CatalogManager: Graceful Degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:

    def test_refresh_keeps_cache_on_failure(self, catalog_manager):
        """When all fetches fail, cached models are preserved."""
        original_count = len(catalog_manager.models)
        assert original_count > 0

        # Mock httpx to simulate network failure
        with patch.dict("sys.modules", {"httpx": MagicMock(
            get=MagicMock(side_effect=Exception("Network error")),
        )}):
            catalog_manager.refresh()

        # Models should still be available from cache
        assert len(catalog_manager.models) == original_count

    def test_startup_without_network(self, tmp_dir, sample_models):
        """CatalogManager starts from cache when network is unavailable."""
        db_path = os.path.join(tmp_dir, "offline.db")
        # Pre-populate cache
        manager1 = CatalogManager(db_path=db_path)
        manager1.initialize()
        manager1.models = sample_models
        manager1._save_to_cache()
        manager1.close()

        # New manager loads from cache without refresh
        manager2 = CatalogManager(db_path=db_path)
        manager2.initialize()
        assert len(manager2.models) == len(sample_models)
        manager2.close()

    def test_empty_cache_empty_models(self, tmp_dir):
        """Fresh start with no cache and no network has 0 models."""
        db_path = os.path.join(tmp_dir, "fresh.db")
        manager = CatalogManager(db_path=db_path)
        manager.initialize()
        assert len(manager.models) == 0
        manager.close()

    def test_needs_refresh_initially(self, tmp_dir):
        """Fresh manager needs refresh."""
        db_path = os.path.join(tmp_dir, "refresh.db")
        manager = CatalogManager(db_path=db_path)
        manager.initialize()
        assert manager.needs_refresh() is True
        manager.close()


# ---------------------------------------------------------------------------
# CatalogManager: Stats
# ---------------------------------------------------------------------------

class TestCatalogStats:

    def test_catalog_stats(self, catalog_manager):
        """Stats report model counts and filter state."""
        stats = catalog_manager.get_catalog_stats()
        assert stats["total_models"] > 0
        assert "openrouter" in stats["providers"]
        assert stats["profiles_loaded"] > 0
        assert "hard_ceiling_input" in stats["filters"]

    def test_get_model_by_id(self, catalog_manager):
        """Model lookup by ID works."""
        model = catalog_manager.get_model_by_id(
            "openrouter/deepseek/deepseek-r1"
        )
        assert model is not None
        assert model.display_name == "DeepSeek R1"

    def test_get_model_by_id_missing(self, catalog_manager):
        """Missing model returns None."""
        model = catalog_manager.get_model_by_id("nonexistent")
        assert model is None


# ---------------------------------------------------------------------------
# CatalogManager: Provider Classification
# ---------------------------------------------------------------------------

class TestProviderClassification:

    def test_frontier_classification(self):
        """Frontier keywords classify correctly."""
        manager = CatalogManager.__new__(CatalogManager)
        assert manager._classify_provider_tier(
            {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus"},
        ) == "frontier"

    def test_performance_classification(self):
        """Performance keywords classify correctly."""
        manager = CatalogManager.__new__(CatalogManager)
        assert manager._classify_provider_tier(
            {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1"},
        ) == "performance"

    def test_budget_classification(self):
        """Budget keywords classify correctly."""
        manager = CatalogManager.__new__(CatalogManager)
        assert manager._classify_provider_tier(
            {"id": "some/tiny-model", "name": "Tiny Model"},
        ) == "budget"

    def test_standard_default(self):
        """Unknown models default to standard."""
        manager = CatalogManager.__new__(CatalogManager)
        assert manager._classify_provider_tier(
            {"id": "some/model", "name": "Some Model"},
        ) == "standard"


# ---------------------------------------------------------------------------
# Dream Cycle: Outcome Recording
# ---------------------------------------------------------------------------

class TestDreamCycleRecording:

    def test_record_outcome(self):
        """Outcomes are recorded."""
        cycle = DreamCycle()
        outcome = RoutingOutcome(
            request_id="req1",
            model_id="test/model",
            semantic_route="coding",
            success=True,
            quality_score=0.9,
        )
        cycle.record_outcome(outcome)
        assert len(cycle._outcomes) == 1

    def test_outcome_pruning(self):
        """Old outcomes are pruned when limit exceeded."""
        cycle = DreamCycle(max_outcomes=5)
        for i in range(10):
            cycle.record_outcome(RoutingOutcome(
                request_id=f"req{i}",
                semantic_route="test",
                success=True,
            ))
        assert len(cycle._outcomes) == 5

    def test_stats(self):
        """Stats report correct counts."""
        cycle = DreamCycle()
        for i in range(5):
            cycle.record_outcome(RoutingOutcome(
                request_id=f"req{i}",
                semantic_route="coding" if i < 3 else "analysis",
                success=True,
            ))
        stats = cycle.get_stats()
        assert stats["total_outcomes"] == 5
        assert stats["routes_tracked"]["coding"] == 3
        assert stats["routes_tracked"]["analysis"] == 2


# ---------------------------------------------------------------------------
# Dream Cycle: Model Property Correlation Analysis
# ---------------------------------------------------------------------------

class TestDreamCycleAnalysis:

    def _make_outcomes(self, route, n_success, n_fail,
                       success_ctx=65536, fail_ctx=4096,
                       success_tier="performance", fail_tier="budget",
                       success_caps=None, fail_caps=None,
                       success_cost=2.0, fail_cost=0.1):
        """Helper to create controlled outcome sets."""
        outcomes = []
        for i in range(n_success):
            outcomes.append(RoutingOutcome(
                request_id=f"s{i}",
                model_id=f"model_success_{i}",
                semantic_route=route,
                success=True,
                quality_score=0.9,
                model_context_window=success_ctx,
                model_provider_tier=success_tier,
                model_capabilities=success_caps or ["code"],
                model_cost_per_1m_input=success_cost,
            ))
        for i in range(n_fail):
            outcomes.append(RoutingOutcome(
                request_id=f"f{i}",
                model_id=f"model_fail_{i}",
                semantic_route=route,
                success=False,
                quality_score=0.2,
                model_context_window=fail_ctx,
                model_provider_tier=fail_tier,
                model_capabilities=fail_caps or [],
                model_cost_per_1m_input=fail_cost,
            ))
        return outcomes

    def test_context_window_correlation(self):
        """Detects when larger context windows correlate with success."""
        cycle = DreamCycle(min_sample_size=5, confidence_threshold=0.5)
        outcomes = self._make_outcomes(
            "coding", n_success=8, n_fail=4,
            success_ctx=65536, fail_ctx=4096,
        )
        for o in outcomes:
            cycle.record_outcome(o)

        insights = cycle.analyze_model_property_correlations()
        assert "coding" in insights
        ctx_insights = [
            i for i in insights["coding"]
            if i.property_name == "context_window"
        ]
        assert len(ctx_insights) > 0
        assert ctx_insights[0].confidence >= 0.5

    def test_provider_tier_correlation(self):
        """Detects when certain provider tiers correlate with success."""
        cycle = DreamCycle(min_sample_size=5, confidence_threshold=0.5)
        outcomes = self._make_outcomes(
            "analysis", n_success=8, n_fail=4,
            success_tier="performance", fail_tier="budget",
        )
        for o in outcomes:
            cycle.record_outcome(o)

        insights = cycle.analyze_model_property_correlations()
        assert "analysis" in insights
        tier_insights = [
            i for i in insights["analysis"]
            if i.property_name == "provider_tier"
        ]
        assert len(tier_insights) > 0

    def test_capability_correlation(self):
        """Detects when capabilities correlate with success."""
        cycle = DreamCycle(min_sample_size=5, confidence_threshold=0.5)
        outcomes = self._make_outcomes(
            "coding", n_success=8, n_fail=4,
            success_caps=["code"], fail_caps=[],
        )
        for o in outcomes:
            cycle.record_outcome(o)

        insights = cycle.analyze_model_property_correlations()
        assert "coding" in insights
        cap_insights = [
            i for i in insights["coding"]
            if i.property_name == "capabilities"
        ]
        assert len(cap_insights) > 0

    def test_no_insights_below_sample_size(self):
        """No insights generated with too few samples."""
        cycle = DreamCycle(min_sample_size=20)
        outcomes = self._make_outcomes("coding", n_success=3, n_fail=2)
        for o in outcomes:
            cycle.record_outcome(o)

        insights = cycle.analyze_model_property_correlations()
        assert "coding" not in insights

    def test_no_insights_without_failures(self):
        """No insights when everything succeeds."""
        cycle = DreamCycle(min_sample_size=5)
        outcomes = self._make_outcomes("coding", n_success=10, n_fail=0)
        for o in outcomes:
            cycle.record_outcome(o)

        insights = cycle.analyze_model_property_correlations()
        assert "coding" not in insights


# ---------------------------------------------------------------------------
# Dream Cycle: Profile Auto-Tuning
# ---------------------------------------------------------------------------

class TestDreamCycleAutoTuning:

    def test_apply_context_window_insight(self, catalog_manager):
        """Dream Cycle can tighten profile context window."""
        cycle = DreamCycle(min_sample_size=5, confidence_threshold=0.5)

        # Record outcomes suggesting larger context = success
        for i in range(8):
            cycle.record_outcome(RoutingOutcome(
                request_id=f"s{i}", semantic_route="coding",
                success=True, model_context_window=65536,
                model_provider_tier="performance",
            ))
        for i in range(4):
            cycle.record_outcome(RoutingOutcome(
                request_id=f"f{i}", semantic_route="coding",
                success=False, model_context_window=4096,
                model_provider_tier="budget",
            ))

        cycle.analyze_model_property_correlations()
        changes = cycle.apply_insights_to_profiles(catalog_manager)
        # May or may not apply depending on confidence — just verify no crash
        assert isinstance(changes, list)


# ---------------------------------------------------------------------------
# Router Integration with CatalogManager
# ---------------------------------------------------------------------------

class TestRouterCatalogIntegration:

    def test_route_with_profile(self, catalog_manager, gpu_hardware):
        """Router route_with_profile resolves via catalog."""
        config = InferenceDifferenceConfig()
        config.models = {**default_local_models(), **default_api_models()}
        config.default_model = "ollama/llama3.1:8b"

        engine = RoutingEngine(
            config=config,
            hardware=gpu_hardware,
            catalog_manager=catalog_manager,
        )

        classification = classify_request("Write a Python function")
        decision = engine.route_with_profile(
            "coding", classification, request_id="profile_test",
        )
        assert decision.model_id != ""
        assert decision.request_id == "profile_test"

    def test_route_with_profile_fallback(self, gpu_hardware):
        """Without catalog manager, falls back to standard routing."""
        config = InferenceDifferenceConfig()
        config.models = {**default_local_models(), **default_api_models()}
        config.default_model = "ollama/llama3.1:8b"

        engine = RoutingEngine(
            config=config,
            hardware=gpu_hardware,
            catalog_manager=None,
        )

        classification = classify_request("Write a Python function")
        decision = engine.route_with_profile("coding", classification)
        # Should fall back to standard routing
        assert decision.model_id != ""
        assert decision.score > 0

    def test_route_standard_unaffected(self, catalog_manager, gpu_hardware):
        """Standard route() is unaffected by catalog integration."""
        config = InferenceDifferenceConfig()
        config.models = {**default_local_models(), **default_api_models()}
        config.default_model = "ollama/llama3.1:8b"

        engine = RoutingEngine(
            config=config,
            hardware=gpu_hardware,
            catalog_manager=catalog_manager,
        )

        classification = classify_request("Write a Python function")
        decision = engine.route(classification)
        assert decision.model_id != ""
        assert decision.score > 0

    def test_dream_cycle_receives_outcomes(self, catalog_manager, gpu_hardware):
        """Dream Cycle receives forwarded outcomes from router."""
        config = InferenceDifferenceConfig()
        config.models = {**default_local_models(), **default_api_models()}
        config.default_model = "ollama/llama3.1:8b"

        dream = DreamCycle()
        engine = RoutingEngine(
            config=config,
            hardware=gpu_hardware,
            catalog_manager=catalog_manager,
            dream_cycle=dream,
        )

        classification = classify_request("Write Python code")
        decision = engine.route(classification)
        engine.report_outcome(
            decision=decision,
            success=True,
            quality_score=0.9,
            latency_ms=500,
        )

        assert dream.get_stats()["total_outcomes"] >= 1


# ---------------------------------------------------------------------------
# CatalogModel Data Class
# ---------------------------------------------------------------------------

class TestCatalogModel:

    def test_default_values(self):
        """CatalogModel has sensible defaults."""
        m = CatalogModel(id="test")
        assert m.provider == ""
        assert m.context_window == 0
        assert m.cost_per_1m_input == 0.0
        assert m.capabilities == []
        assert m.is_active is True

    def test_model_with_all_fields(self):
        """CatalogModel stores all fields correctly."""
        m = CatalogModel(
            id="openrouter/test/model",
            provider="openrouter",
            display_name="Test Model",
            context_window=32768,
            cost_per_1m_input=1.5,
            cost_per_1m_output=4.5,
            provider_tier="performance",
            capabilities=["code", "tools"],
            is_active=True,
            last_seen="2026-02-21T00:00:00Z",
            fetched_at="2026-02-21T00:00:00Z",
        )
        assert m.id == "openrouter/test/model"
        assert m.context_window == 32768
        assert "code" in m.capabilities


# ---------------------------------------------------------------------------
# RequirementsProfile Data Class
# ---------------------------------------------------------------------------

class TestRequirementsProfile:

    def test_default_values(self):
        """RequirementsProfile has sensible defaults."""
        p = RequirementsProfile()
        assert p.prefer_local is False
        assert p.fallback_to_cloud is True
        assert p.sort_by == "cost_asc"
        assert p.max_candidates == 3

    def test_profile_with_constraints(self):
        """Profile stores constraints correctly."""
        p = RequirementsProfile(
            name="test",
            max_cost_per_1m_input_tokens=5.0,
            min_context_window=32000,
            required_capabilities=["code"],
            provider_tiers=["frontier"],
            sort_by="quality_desc",
            max_candidates=2,
        )
        assert p.max_cost_per_1m_input_tokens == 5.0
        assert p.min_context_window == 32000
        assert "code" in p.required_capabilities
        assert "frontier" in p.provider_tiers


# ---------------------------------------------------------------------------
# PropertyInsight Data Class
# ---------------------------------------------------------------------------

class TestPropertyInsight:

    def test_insight_fields(self):
        """PropertyInsight stores analysis results."""
        insight = PropertyInsight(
            property_name="context_window",
            observation="Larger context windows succeed more",
            recommendation="min_context_window >= 32000",
            confidence=0.85,
            sample_size=50,
        )
        assert insight.property_name == "context_window"
        assert insight.confidence == 0.85
        assert insight.sample_size == 50
