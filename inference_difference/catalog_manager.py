"""
Dynamic Model Catalog Selection for The Inference Difference (§4.5).

The CatalogManager fetches, caches, and queries available model catalogs
from OpenRouter and HuggingFace Inference API. Instead of routing to a
named model, TID routes to a requirements profile and resolves it against
the live catalog at request time.

Key design decisions:
- Graceful degradation: if a provider is unreachable at startup or refresh,
  TID falls back to the last cached catalog rather than failing.
- Hard ceiling safety valve: catalog_filters.yaml defines a hard cost
  ceiling that is ALWAYS enforced, regardless of profile configuration.
  This prevents bugs in profile resolution from routing cheap requests
  to expensive models.
- SQLite cache: the catalog is persisted in gateway.db so TID can start
  without network access using stale-but-valid data.
- Backward compatibility: static tier_models.yaml entries (old format)
  remain valid alongside dynamic profile references (new format).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

logger = logging.getLogger("inference_difference.catalog_manager")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CatalogModel:
    """A model entry from the provider catalog.

    Normalized representation combining OpenRouter and HuggingFace
    model metadata into a common schema for filtering and sorting.
    """
    id: str                              # e.g. "openrouter/deepseek/deepseek-r1"
    provider: str = ""                   # "openrouter" | "huggingface"
    display_name: str = ""
    context_window: int = 0
    cost_per_1m_input: float = 0.0       # USD per 1M input tokens
    cost_per_1m_output: float = 0.0      # USD per 1M output tokens
    provider_tier: str = ""              # "frontier"|"performance"|"standard"|"budget"
    capabilities: List[str] = field(default_factory=list)  # ["code","vision","tools"]
    is_active: bool = True
    last_seen: str = ""                  # ISO timestamp
    fetched_at: str = ""                 # ISO timestamp


@dataclass
class RequirementsProfile:
    """A requirements profile describing what a task needs (§4.5.2).

    Loaded from task_requirements.yaml. The CatalogManager resolves
    profiles against the cached catalog to find the best model.
    """
    name: str = ""
    prefer_local: bool = False
    fallback_to_cloud: bool = True
    max_cost_per_1m_input_tokens: Optional[float] = None
    max_cost_per_1m_output_tokens: Optional[float] = None
    min_context_window: Optional[int] = None
    required_capabilities: List[str] = field(default_factory=list)
    provider_tiers: List[str] = field(default_factory=list)
    sort_by: str = "cost_asc"            # "cost_asc" | "quality_desc"
    max_candidates: int = 3


@dataclass
class CatalogFilters:
    """Operator-level overrides from catalog_filters.yaml (§4.5.4)."""
    blocklist_providers: List[str] = field(default_factory=list)
    blocklist_models: List[str] = field(default_factory=list)
    allowlist_providers: List[str] = field(default_factory=list)
    hard_ceiling_input: float = 20.0     # USD per 1M input tokens
    hard_ceiling_output: float = 60.0    # USD per 1M output tokens


@dataclass
class TierEntry:
    """A single entry in tier_models.yaml, supporting both formats (§4.5.7).

    Old format: models list with name + priority (passthrough).
    New format: profile reference (resolved dynamically via catalog).
    """
    tier_name: str = ""
    profile: Optional[str] = None        # New format: profile name
    models: List[Dict[str, Any]] = field(default_factory=list)  # Old format


# ---------------------------------------------------------------------------
# CatalogManager
# ---------------------------------------------------------------------------

class CatalogManager:
    """Fetches, caches, and queries available model catalogs
    from OpenRouter and HuggingFace (§4.5.3).

    Usage:
        manager = CatalogManager(db_path="gateway.db")
        manager.load_profiles("task_requirements.yaml")
        manager.load_filters("catalog_filters.yaml")
        manager.refresh()  # Fetch from providers (graceful on failure)

        model_id = manager.resolve_profile("coding")
    """

    OPENROUTER_CATALOG_URL = "https://openrouter.ai/api/v1/models"
    HF_CATALOG_URL = "https://api-inference.huggingface.co/models"

    def __init__(
        self,
        db_path: str = "gateway.db",
        fallback_model: str = "ollama/llama3.1:8b",
        refresh_interval_seconds: int = 21600,  # 6 hours
    ):
        self.db_path = db_path
        self.fallback_model = fallback_model
        self.refresh_interval_seconds = refresh_interval_seconds

        self.profiles: Dict[str, RequirementsProfile] = {}
        self.filters: CatalogFilters = CatalogFilters()
        self.tiers: Dict[str, TierEntry] = {}
        self.models: List[CatalogModel] = []

        self._last_refresh: float = 0.0
        self._db: Optional[sqlite3.Connection] = None

    # -------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------

    def initialize(self) -> None:
        """Set up database tables and load cached catalog."""
        self._init_db()
        self._load_from_cache()
        logger.info(
            "CatalogManager initialized: %d cached models", len(self.models),
        )

    def _init_db(self) -> None:
        """Create database tables if they don't exist (§4.5.6)."""
        self._db = sqlite3.connect(self.db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")

        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS model_catalog (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                display_name TEXT,
                context_window INTEGER,
                cost_per_1m_input_tokens REAL,
                cost_per_1m_output_tokens REAL,
                provider_tier TEXT,
                capabilities TEXT,
                is_active INTEGER DEFAULT 1,
                last_seen TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS catalog_refresh_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                provider TEXT NOT NULL,
                models_found INTEGER,
                models_added INTEGER,
                models_removed INTEGER,
                duration_ms INTEGER
            );
        """)
        self._db.commit()

    def _load_from_cache(self) -> None:
        """Load models from the SQLite cache into memory."""
        if self._db is None:
            return

        rows = self._db.execute(
            "SELECT * FROM model_catalog WHERE is_active = 1"
        ).fetchall()

        self.models = []
        for row in rows:
            caps = []
            if row["capabilities"]:
                try:
                    caps = json.loads(row["capabilities"])
                except (json.JSONDecodeError, TypeError):
                    caps = []

            self.models.append(CatalogModel(
                id=row["id"],
                provider=row["provider"],
                display_name=row["display_name"] or "",
                context_window=row["context_window"] or 0,
                cost_per_1m_input=row["cost_per_1m_input_tokens"] or 0.0,
                cost_per_1m_output=row["cost_per_1m_output_tokens"] or 0.0,
                provider_tier=row["provider_tier"] or "",
                capabilities=caps,
                is_active=bool(row["is_active"]),
                last_seen=row["last_seen"] or "",
                fetched_at=row["fetched_at"] or "",
            ))

        if self.models:
            logger.info(
                "Loaded %d models from catalog cache", len(self.models),
            )

    # -------------------------------------------------------------------
    # Configuration loading
    # -------------------------------------------------------------------

    def load_profiles(self, path: str) -> None:
        """Load requirements profiles from task_requirements.yaml."""
        if not os.path.exists(path):
            logger.warning("Profiles file not found: %s", path)
            return

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data or "profiles" not in data:
            logger.warning("No profiles found in %s", path)
            return

        for name, spec in data["profiles"].items():
            if spec is None:
                spec = {}
            profile = RequirementsProfile(
                name=name,
                prefer_local=spec.get("prefer_local", False),
                fallback_to_cloud=spec.get("fallback_to_cloud", True),
                max_cost_per_1m_input_tokens=spec.get(
                    "max_cost_per_1m_input_tokens"
                ),
                max_cost_per_1m_output_tokens=spec.get(
                    "max_cost_per_1m_output_tokens"
                ),
                min_context_window=spec.get("min_context_window"),
                required_capabilities=spec.get("required_capabilities", []),
                provider_tiers=spec.get("provider_tiers", []),
                sort_by=spec.get("sort_by", "cost_asc"),
                max_candidates=spec.get("max_candidates", 3),
            )
            self.profiles[name] = profile

        logger.info("Loaded %d profiles from %s", len(self.profiles), path)

    def load_filters(self, path: str) -> None:
        """Load catalog filters from catalog_filters.yaml."""
        if not os.path.exists(path):
            logger.warning("Filters file not found: %s", path)
            return

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            logger.warning("Empty filters file: %s", path)
            return

        blocklist = data.get("blocklist", {})
        allowlist = data.get("allowlist", {})
        ceiling = data.get("hard_ceiling", {})

        self.filters = CatalogFilters(
            blocklist_providers=blocklist.get("providers", []) or [],
            blocklist_models=blocklist.get("models", []) or [],
            allowlist_providers=allowlist.get("providers", []) or [],
            hard_ceiling_input=ceiling.get(
                "max_cost_per_1m_input_tokens", 20.0,
            ),
            hard_ceiling_output=ceiling.get(
                "max_cost_per_1m_output_tokens", 60.0,
            ),
        )
        logger.info("Loaded catalog filters from %s", path)

    def load_tiers(self, path: str) -> None:
        """Load tier_models.yaml with backward-compatible format (§4.5.7).

        Supports both old format (named model list) and new format
        (profile reference). Both work simultaneously.
        """
        if not os.path.exists(path):
            logger.info("No tier_models.yaml at %s (optional)", path)
            return

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data or "tiers" not in data:
            return

        for tier_name, tier_spec in data["tiers"].items():
            if tier_spec is None:
                continue

            entry = TierEntry(tier_name=tier_name)

            # New format: profile reference
            if "profile" in tier_spec:
                entry.profile = tier_spec["profile"]

            # Old format: explicit model list
            if "models" in tier_spec:
                entry.models = tier_spec["models"]

            self.tiers[tier_name] = entry

        logger.info("Loaded %d tiers from %s", len(self.tiers), path)

    # -------------------------------------------------------------------
    # Catalog refresh
    # -------------------------------------------------------------------

    def refresh(self) -> None:
        """Fetch latest catalog from all configured providers (§4.5.3).

        Graceful degradation: if a provider is unreachable, TID falls
        back to the last cached catalog rather than failing to start.
        """
        start = time.monotonic()
        new_models: List[CatalogModel] = []
        now = datetime.now(timezone.utc).isoformat()

        # Fetch from OpenRouter
        or_models = self._fetch_openrouter_catalog(now)
        new_models.extend(or_models)
        self._log_refresh(
            "openrouter", len(or_models), start, now,
        )

        # Fetch from HuggingFace
        hf_models = self._fetch_huggingface_catalog(now)
        new_models.extend(hf_models)
        self._log_refresh(
            "huggingface", len(hf_models), start, now,
        )

        if new_models:
            self.models = new_models
            self._save_to_cache()
            logger.info(
                "Catalog refreshed: %d models available", len(self.models),
            )
        else:
            # Graceful degradation: keep cached models
            logger.warning(
                "All catalog fetches failed. Using %d cached models.",
                len(self.models),
            )

        self._last_refresh = time.time()

    def needs_refresh(self) -> bool:
        """Check if the catalog is due for a refresh."""
        if self._last_refresh == 0.0:
            return True
        return (time.time() - self._last_refresh) > self.refresh_interval_seconds

    def _fetch_openrouter_catalog(self, now: str) -> List[CatalogModel]:
        """Fetch model catalog from OpenRouter.

        Returns empty list on failure (graceful degradation).
        """
        try:
            import httpx
        except ImportError:
            logger.warning(
                "httpx not installed — cannot fetch OpenRouter catalog"
            )
            return []

        try:
            resp = httpx.get(
                self.OPENROUTER_CATALOG_URL,
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("OpenRouter catalog fetch failed: %s", e)
            return []

        models = []
        for item in data.get("data", []):
            model_id = item.get("id", "")
            if not model_id:
                continue

            # Parse pricing (OpenRouter returns per-token, we store per-1M)
            pricing = item.get("pricing", {})
            prompt_price = float(pricing.get("prompt", "0") or "0")
            completion_price = float(pricing.get("completion", "0") or "0")

            # Determine provider tier from OpenRouter metadata
            tier = self._classify_provider_tier(item)

            # Extract capabilities
            capabilities = self._extract_capabilities(item)

            models.append(CatalogModel(
                id=f"openrouter/{model_id}",
                provider="openrouter",
                display_name=item.get("name", model_id),
                context_window=item.get("context_length", 0) or 0,
                cost_per_1m_input=prompt_price * 1_000_000,
                cost_per_1m_output=completion_price * 1_000_000,
                provider_tier=tier,
                capabilities=capabilities,
                is_active=True,
                last_seen=now,
                fetched_at=now,
            ))

        return models

    def _fetch_huggingface_catalog(self, now: str) -> List[CatalogModel]:
        """Fetch model catalog from HuggingFace Inference API.

        Returns empty list on failure (graceful degradation).
        """
        try:
            import httpx
        except ImportError:
            logger.warning(
                "httpx not installed — cannot fetch HuggingFace catalog"
            )
            return []

        try:
            resp = httpx.get(
                self.HF_CATALOG_URL,
                timeout=30.0,
                params={
                    "pipeline_tag": "text-generation",
                    "inference": "warm",
                    "limit": 100,
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("HuggingFace catalog fetch failed: %s", e)
            return []

        models = []
        if not isinstance(data, list):
            return models

        for item in data:
            model_id = item.get("id", "") or item.get("modelId", "")
            if not model_id:
                continue

            # HuggingFace Inference API models are typically free tier
            models.append(CatalogModel(
                id=f"huggingface/{model_id}",
                provider="huggingface",
                display_name=model_id.split("/")[-1] if "/" in model_id else model_id,
                context_window=self._hf_context_window(item),
                cost_per_1m_input=0.0,   # HF Inference API free tier
                cost_per_1m_output=0.0,
                provider_tier="standard",
                capabilities=self._hf_capabilities(item),
                is_active=True,
                last_seen=now,
                fetched_at=now,
            ))

        return models

    def _classify_provider_tier(self, item: Dict[str, Any]) -> str:
        """Classify a model into a provider tier based on metadata."""
        model_id = item.get("id", "").lower()
        name = item.get("name", "").lower()

        # Frontier models (major providers' flagship models)
        frontier_keywords = [
            "gpt-4", "claude-3", "claude-opus", "claude-sonnet",
            "gemini-pro", "gemini-ultra", "o1", "o3",
        ]
        if any(kw in model_id or kw in name for kw in frontier_keywords):
            return "frontier"

        # Performance tier
        perf_keywords = [
            "deepseek-r1", "llama-3.1-70b", "llama-3.1-405b",
            "mixtral-8x22b", "qwen-72b", "yi-large",
            "command-r-plus",
        ]
        if any(kw in model_id or kw in name for kw in perf_keywords):
            return "performance"

        # Budget tier (very small or explicitly cheap)
        budget_keywords = [
            "tiny", "mini", "nano", "1b", "3b", "phi-2",
        ]
        if any(kw in model_id or kw in name for kw in budget_keywords):
            return "budget"

        return "standard"

    def _extract_capabilities(self, item: Dict[str, Any]) -> List[str]:
        """Extract model capabilities from OpenRouter metadata."""
        caps = []
        model_id = item.get("id", "").lower()
        name = item.get("name", "").lower()

        # Code capability
        code_keywords = ["code", "coder", "codestral", "deepseek-coder"]
        if any(kw in model_id or kw in name for kw in code_keywords):
            caps.append("code")

        # Vision capability
        if item.get("architecture", {}).get("modality", "") == "multimodal":
            caps.append("vision")

        # Tool use
        desc = item.get("description", "").lower()
        if "function calling" in desc or "tool" in desc:
            caps.append("tools")

        return caps

    def _hf_context_window(self, item: Dict[str, Any]) -> int:
        """Extract context window from HuggingFace model metadata."""
        # HF doesn't consistently expose this; use a sensible default
        config = item.get("config", {})
        if isinstance(config, dict):
            max_len = config.get("max_position_embeddings", 0)
            if max_len and isinstance(max_len, int):
                return max_len
        return 4096  # Conservative default

    def _hf_capabilities(self, item: Dict[str, Any]) -> List[str]:
        """Extract capabilities from HuggingFace model metadata."""
        caps = []
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        if "code" in tags or "code-generation" in tags:
            caps.append("code")

        model_id = str(item.get("id", "")).lower()
        if "code" in model_id or "coder" in model_id:
            caps.append("code")

        return list(set(caps))

    # -------------------------------------------------------------------
    # Profile resolution (§4.5.3)
    # -------------------------------------------------------------------

    def resolve_profile(self, profile_name: str) -> Optional[str]:
        """Given a profile name, return the best available model ID.

        Applies hard ceiling, blocklist/allowlist, filters by requirements,
        sorts by criteria, returns winner. Returns None if the profile
        is not found or is a local-only profile.

        The hard ceiling is ALWAYS enforced first — this is the safety
        valve against routing simple requests to expensive models.
        """
        if profile_name not in self.profiles:
            logger.warning("Unknown profile: '%s'", profile_name)
            return None

        profile = self.profiles[profile_name]

        # Local-only profiles don't use the catalog
        if profile.prefer_local and not profile.fallback_to_cloud:
            return None

        candidates = self._filter_catalog(profile)
        candidates = self._apply_hard_ceiling(candidates)
        candidates = self._apply_blocklist(candidates)
        candidates = self._apply_allowlist(candidates)
        candidates = self._sort(candidates, profile.sort_by)

        # Trim to max_candidates
        candidates = candidates[:profile.max_candidates]

        if not candidates:
            logger.warning(
                "No models matched profile '%s', falling back to '%s'",
                profile_name, self.fallback_model,
            )
            return self.fallback_model

        winner = candidates[0]
        logger.info(
            "Profile '%s' resolved to '%s' ($%.2f/1M input tokens)",
            profile_name, winner.id, winner.cost_per_1m_input,
        )
        return winner.id

    def resolve_profile_candidates(
        self, profile_name: str,
    ) -> List[CatalogModel]:
        """Resolve a profile and return the full candidate list.

        Useful for the Dream Cycle to analyze which models were
        considered and their properties.
        """
        if profile_name not in self.profiles:
            return []

        profile = self.profiles[profile_name]
        if profile.prefer_local and not profile.fallback_to_cloud:
            return []

        candidates = self._filter_catalog(profile)
        candidates = self._apply_hard_ceiling(candidates)
        candidates = self._apply_blocklist(candidates)
        candidates = self._apply_allowlist(candidates)
        candidates = self._sort(candidates, profile.sort_by)
        return candidates[:profile.max_candidates]

    def _filter_catalog(self, profile: RequirementsProfile) -> List[CatalogModel]:
        """Filter catalog models by profile requirements."""
        candidates = [m for m in self.models if m.is_active]

        if profile.max_cost_per_1m_input_tokens is not None:
            candidates = [
                m for m in candidates
                if m.cost_per_1m_input <= profile.max_cost_per_1m_input_tokens
            ]

        if profile.max_cost_per_1m_output_tokens is not None:
            candidates = [
                m for m in candidates
                if m.cost_per_1m_output <= profile.max_cost_per_1m_output_tokens
            ]

        if profile.min_context_window is not None:
            candidates = [
                m for m in candidates
                if m.context_window >= profile.min_context_window
            ]

        if profile.required_capabilities:
            candidates = [
                m for m in candidates
                if all(
                    cap in m.capabilities
                    for cap in profile.required_capabilities
                )
            ]

        if profile.provider_tiers:
            candidates = [
                m for m in candidates
                if m.provider_tier in profile.provider_tiers
            ]

        return candidates

    def _apply_hard_ceiling(
        self, candidates: List[CatalogModel],
    ) -> List[CatalogModel]:
        """Apply hard cost ceiling — the safety valve (§4.5.4).

        No request ever routes to a model above this price,
        regardless of profile configuration.
        """
        return [
            m for m in candidates
            if (m.cost_per_1m_input <= self.filters.hard_ceiling_input
                and m.cost_per_1m_output <= self.filters.hard_ceiling_output)
        ]

    def _apply_blocklist(
        self, candidates: List[CatalogModel],
    ) -> List[CatalogModel]:
        """Remove blocklisted providers and models."""
        if not self.filters.blocklist_providers and not self.filters.blocklist_models:
            return candidates

        return [
            m for m in candidates
            if (m.provider not in self.filters.blocklist_providers
                and m.id not in self.filters.blocklist_models)
        ]

    def _apply_allowlist(
        self, candidates: List[CatalogModel],
    ) -> List[CatalogModel]:
        """If allowlist is set, only keep models from allowed providers."""
        if not self.filters.allowlist_providers:
            return candidates  # No allowlist = allow all

        return [
            m for m in candidates
            if m.provider in self.filters.allowlist_providers
        ]

    def _sort(
        self,
        candidates: List[CatalogModel],
        sort_by: str,
    ) -> List[CatalogModel]:
        """Sort candidates by the specified criteria."""
        if sort_by == "cost_asc":
            return sorted(
                candidates,
                key=lambda m: m.cost_per_1m_input + m.cost_per_1m_output,
            )
        elif sort_by == "quality_desc":
            # Quality heuristic: frontier > performance > standard > budget
            tier_order = {
                "frontier": 0, "performance": 1,
                "standard": 2, "budget": 3,
            }
            return sorted(
                candidates,
                key=lambda m: (
                    tier_order.get(m.provider_tier, 99),
                    -(m.context_window or 0),
                ),
            )
        else:
            return candidates

    # -------------------------------------------------------------------
    # Tier resolution (§4.5.7 backward compatibility)
    # -------------------------------------------------------------------

    def resolve_tier(self, tier_name: str) -> Optional[str]:
        """Resolve a tier to a model ID, supporting both formats.

        Old format (model name list): returns the highest-priority model.
        New format (profile reference): resolves via catalog.
        """
        if tier_name not in self.tiers:
            return None

        tier = self.tiers[tier_name]

        # New format: profile reference
        if tier.profile:
            return self.resolve_profile(tier.profile)

        # Old format: explicit model list (passthrough)
        if tier.models:
            sorted_models = sorted(
                tier.models,
                key=lambda m: m.get("priority", 0),
                reverse=True,
            )
            if sorted_models:
                return sorted_models[0].get("name")

        return None

    # -------------------------------------------------------------------
    # Cache persistence
    # -------------------------------------------------------------------

    def _save_to_cache(self) -> None:
        """Save current catalog to SQLite cache."""
        if self._db is None:
            return

        for model in self.models:
            self._db.execute("""
                INSERT OR REPLACE INTO model_catalog
                    (id, provider, display_name, context_window,
                     cost_per_1m_input_tokens, cost_per_1m_output_tokens,
                     provider_tier, capabilities, is_active,
                     last_seen, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.id,
                model.provider,
                model.display_name,
                model.context_window,
                model.cost_per_1m_input,
                model.cost_per_1m_output,
                model.provider_tier,
                json.dumps(model.capabilities),
                1 if model.is_active else 0,
                model.last_seen,
                model.fetched_at,
            ))

        self._db.commit()
        logger.info("Saved %d models to catalog cache", len(self.models))

    def _log_refresh(
        self,
        provider: str,
        models_found: int,
        start_time: float,
        now: str,
    ) -> None:
        """Log a catalog refresh event."""
        if self._db is None:
            return

        duration_ms = int((time.monotonic() - start_time) * 1000)
        self._db.execute("""
            INSERT INTO catalog_refresh_log
                (id, timestamp, provider, models_found,
                 models_added, models_removed, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid4()),
            now,
            provider,
            models_found,
            models_found,  # Simplified: treat all as added on full refresh
            0,
            duration_ms,
        ))
        self._db.commit()

    # -------------------------------------------------------------------
    # Query helpers
    # -------------------------------------------------------------------

    def get_catalog_stats(self) -> Dict[str, Any]:
        """Return catalog statistics for the /health endpoint."""
        provider_counts: Dict[str, int] = {}
        for m in self.models:
            provider_counts[m.provider] = provider_counts.get(m.provider, 0) + 1

        return {
            "total_models": len(self.models),
            "providers": provider_counts,
            "profiles_loaded": len(self.profiles),
            "tiers_loaded": len(self.tiers),
            "last_refresh": self._last_refresh,
            "filters": {
                "blocklist_providers": self.filters.blocklist_providers,
                "blocklist_models": self.filters.blocklist_models,
                "allowlist_providers": self.filters.allowlist_providers,
                "hard_ceiling_input": self.filters.hard_ceiling_input,
                "hard_ceiling_output": self.filters.hard_ceiling_output,
            },
        }

    def get_model_by_id(self, model_id: str) -> Optional[CatalogModel]:
        """Look up a catalog model by ID."""
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None
