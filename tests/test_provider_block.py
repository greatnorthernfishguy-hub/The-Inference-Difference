# ---- Changelog ----
# [2026-06-07] CC (Opus 4.8) — #TID-402: provider-level credit block + refund probe
# What: Tests for ModelClient provider-block: 402 blocks the whole provider
#       (persisted), blocked providers instant-skip with no network call, and a
#       successful probe after the interval clears the block (refund detected).
# Why: TID kept routing Syl to an unfunded Venice (per-model learning couldn't
#      catch a provider-wide 402); provider-block makes the teaching signal land.
# How: Drive ModelClient state directly; assert instant-skip + cache roundtrip.
# -------------------
import json
import time

import inference_difference.model_client as mc
from inference_difference.model_client import ModelClient, _provider_from_base_url


def test_provider_from_base_url():
    assert _provider_from_base_url("https://api.venice.ai/api") == "venice"
    assert _provider_from_base_url("https://openrouter.ai/api") == "openrouter"
    assert _provider_from_base_url("http://127.0.0.1:11434") == "ollama"
    assert _provider_from_base_url("https://api.anthropic.com") == "anthropic"


def test_blocked_provider_instant_skips_without_network(monkeypatch):
    # Guard: if any network call is attempted, fail loudly.
    def _boom(*a, **k):
        raise AssertionError("network call attempted for a blocked provider")
    c = ModelClient()
    monkeypatch.setattr(c, "_call_openai_compat", _boom)
    monkeypatch.setattr(c, "_call_anthropic", _boom)
    c._provider_blocked = {"venice": time.time()}  # just blocked → within interval
    resp = c.call("venice/deepseek-v3.2", [{"role": "user", "content": "hi"}])
    assert resp.success is False
    assert "credit-blocked" in resp.error
    assert resp.latency_ms == 0.0


def test_unblocked_provider_is_attempted(monkeypatch):
    # A provider NOT in the block set must reach the call path (not instant-skip).
    from inference_difference.model_client import ModelResponse
    reached = {}
    def _ok(base_url, api_key, model_name, *a, **k):
        reached["model"] = model_name
        return ModelResponse(model=model_name, latency_ms=5.0, success=True, content="ok",
                             usage={"total_tokens": 3})
    c = ModelClient()
    monkeypatch.setattr(c, "_call_openai_compat", _ok)
    c._provider_blocked = {"venice": time.time()}  # venice blocked, openrouter is not
    resp = c.call("openrouter/openai/gpt-4.1-nano", [{"role": "user", "content": "hi"}])
    assert resp.success is True
    assert reached.get("model")  # the call path was reached


def test_successful_probe_clears_block(monkeypatch):
    # After the probe interval, a SUCCESSFUL call clears the provider block (refund).
    from inference_difference.model_client import ModelResponse
    def _ok(base_url, api_key, model_name, *a, **k):
        return ModelResponse(model=model_name, latency_ms=5.0, success=True, content="ok",
                             usage={"total_tokens": 3})
    c = ModelClient()
    monkeypatch.setattr(c, "_call_openai_compat", _ok)
    monkeypatch.setattr(c, "_save_provider_blocked_cache", lambda: None)
    c._provider_blocked = {"venice": time.time() - mc._PROVIDER_PROBE_INTERVAL_S - 1}  # past interval
    resp = c.call("venice/deepseek-v3.2", [{"role": "user", "content": "hi"}])
    assert resp.success is True
    assert "venice" not in c._provider_blocked  # block cleared by successful probe


def test_provider_blocked_cache_roundtrip(tmp_path, monkeypatch):
    p = str(tmp_path / "provider_blocked_cache.json")
    monkeypatch.setattr(mc, "_PROVIDER_BLOCKED_CACHE_PATH", p)
    c = ModelClient()
    c._provider_blocked = {"venice": 12345.0}
    c._save_provider_blocked_cache()
    assert json.load(open(p)) == {"venice": 12345.0}
    c2 = ModelClient()  # loads from the patched path on init
    assert c2._provider_blocked.get("venice") == 12345.0


def test_write_credit_notice(tmp_path, monkeypatch):
    # Funded-provider 402 drops a one-shot notice file for Anima to surface.
    p = str(tmp_path / "credit_notice.json")
    monkeypatch.setattr(mc, "_CREDIT_NOTICE_PATH", p)
    mc._write_credit_notice("openrouter")
    d = json.load(open(p))
    assert d["provider"] == "openrouter"
    assert "out of credits" in d["text"]
    assert d["text"].startswith("⚠")
