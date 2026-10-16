# ---- Changelog ----
# [2026-10-15] Claude Code (Opus 5.5, Executive; Josh "Good catches and calls") — no frontier model per token
# What: pins catalog_filters.yaml's blocklist.model_patterns against every frontier-lab id that was in the live
#       laptop pool on 2026-10-15, and against the open-weights ids from the same labs that must stay eligible.
# Why:  Josh's standing rule — frontier models only through subscriptions; OpenRouter is for open-source only.
#       TID routed real traffic to Haiku 5.5 / Sonnet 4.6 / GPT-5.1-codex-mini per token on 2026-10-14/15.
# How:  loads the repo's real catalog_filters.yaml; no network.
# -------------------
"""A frontier model must never be routable per token (Josh's standing rule)."""
import os
import tempfile

import pytest

from inference_difference.catalog_manager import CatalogManager, CatalogModel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every frontier-lab id seen in the live laptop pool 2026-10-15 (plus Grok / Gemini, which the account had blocked).
FRONTIER = [
    "openrouter/anthropic/claude-haiku-4.5", "openrouter/anthropic/claude-haiku-5.5",
    "openrouter/anthropic/claude-opus-4.5", "openrouter/anthropic/claude-opus-4.6",
    "openrouter/anthropic/claude-opus-5.5", "openrouter/anthropic/claude-sonnet-4.5",
    "openrouter/anthropic/claude-sonnet-4.6", "openrouter/anthropic/claude-sonnet-5.5",
    "openrouter/openai/gpt-3.5-turbo-0613", "openrouter/openai/gpt-4o", "openrouter/openai/gpt-4o-mini",
    "openrouter/openai/gpt-4.1", "openrouter/openai/gpt-4.1-nano", "openrouter/openai/gpt-5",
    "openrouter/openai/gpt-5-nano", "openrouter/openai/gpt-5.1-codex-mini", "openrouter/openai/gpt-5.4",
    "openrouter/openai/o3", "openrouter/x-ai/grok-4.6", "venice/grok-4-20-multi-agent-beta",
    "openrouter/google/gemini-3-pro", "openrouter/amazon/nova-premier-v1", "openrouter/amazon/nova-micro-v1",
    "openrouter/cohere/command-a-plus", "openrouter/mistralai/mistral-large-2407",
    "openrouter/mistralai/mistral-large-2512", "openrouter/mistralai/mistral-large-4-0",
    "openrouter/openrouter/auto", "openrouter/openrouter/auto-beta", "openrouter/openrouter/pareto-code",
]

# Open-weights models (some from the same labs) that must stay eligible.
OPEN = [
    "openrouter/google/gemma-3-27b-it", "openrouter/google/gemma-4-31b-it",
    "openrouter/openai/gpt-oss-120b", "openrouter/openai/gpt-oss-20b",
    "openrouter/meta-llama/llama-3.3-70b-instruct", "openrouter/qwen/qwen3-coder-next",
    "openrouter/mistralai/mistral-small-2603", "openrouter/mistralai/devstral-2512",
    "openrouter/deepseek/deepseek-v3.2", "openrouter/z-ai/glm-5", "openrouter/openrouter/free",
]


@pytest.fixture
def manager():
    with tempfile.TemporaryDirectory() as d:
        m = CatalogManager(db_path=os.path.join(d, "t.db"), fallback_model="ollama/llama3.1:8b")
        m.initialize()
        m.load_filters(os.path.join(REPO, "catalog_filters.yaml"))
        yield m
        m.close()


@pytest.mark.parametrize("model_id", FRONTIER)
def test_frontier_model_is_blocked(manager, model_id):
    assert manager.is_blocklisted(model_id, model_id.split("/")[0]), model_id


@pytest.mark.parametrize("model_id", OPEN)
def test_open_weights_model_stays_eligible(manager, model_id):
    assert not manager.is_blocklisted(model_id, model_id.split("/")[0]), model_id


def test_apply_blocklist_drops_frontier_from_a_candidate_pool(manager):
    pool = [CatalogModel(id=i, provider=i.split("/")[0]) for i in FRONTIER + OPEN]
    kept = {m.id for m in manager._apply_blocklist(pool)}
    assert kept == set(OPEN)


def test_invalid_pattern_is_skipped_not_fatal(manager):
    manager.filters.blocklist_model_patterns = ["(", "^openrouter/anthropic/"]
    assert manager.is_blocklisted("openrouter/anthropic/claude-opus-5.5")
    assert not manager.is_blocklisted("openrouter/qwen/qwen3-coder-next")


def test_caller_named_frontier_model_is_not_forwarded(manager, monkeypatch):
    """An explicitly requested model used to skip the filters; a blocklisted one must be auto-routed instead."""
    from inference_difference import app as tid_app
    monkeypatch.setattr(tid_app._state, "catalog_manager", manager, raising=False)
    assert tid_app._caller_model_allowed("openrouter/anthropic/claude-opus-5.5") is False
    assert tid_app._caller_model_allowed("openrouter/openai/gpt-5.4") is False
    assert tid_app._caller_model_allowed("openrouter/qwen/qwen3-coder-next") is True
    assert tid_app._caller_model_allowed("openrouter/openai/gpt-oss-120b") is True
    monkeypatch.setattr(tid_app._state, "catalog_manager", None, raising=False)
    assert tid_app._caller_model_allowed("openrouter/anthropic/claude-opus-5.5") is True  # no catalog: unchanged
