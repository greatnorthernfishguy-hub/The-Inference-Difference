# ---- Changelog ----
# [2026-10-15] Claude Code (Opus 5.5, lane tid-provider-routing; design 2026-10-15-tid-provider-level-routing §5,
#   approved by Josh 2026-10-15) — unit tests for (model, provider)-level routing (#1040 / #1126)
# What: one block per design component (A facts + refresher, B served provider capture, C deposit target,
#   D pair penalty box + migration, E request shaping + hard filters, F scoring) including the four named
#   acceptance cases: (1) a 429 from one provider boxes only that pair and the model stays routable through another;
#   (2) a model with every provider boxed falls back exactly as today; (3) hard filters drop an endpoint whose
#   context or output cap is too small; (4) the persisted box migrates without loss.
# Why: design §5 acceptance.
# How: real RoutingEngine / ModelClient / ProviderFactsStore; network replaced by fakes (no test touches the network);
#   time injected where a ladder is exercised.
# -------------------
import io
import json
import threading
import time
import urllib.error
from unittest.mock import patch

import msgpack
import pytest

from inference_difference.classifier import RequestClassification
from inference_difference.config import (
    ComplexityTier,
    InferenceDifferenceConfig,
    ModelEntry,
    ModelType,
    TaskDomain,
)
from inference_difference.hardware import HardwareProfile
from inference_difference.model_penalty import ModelPenaltyBox, ProviderPenaltyBox
from inference_difference.provider_facts import (
    EndpointFact,
    ProviderFactsRefresher,
    ProviderFactsStore,
    RequestNeeds,
    hard_filter_endpoints,
    openrouter_model_name,
    provider_slug,
)
from inference_difference.provider_routing import PairOutcomeStats, classify_failure, task_class
from inference_difference.router import RoutingDecision, RoutingEngine

M1 = "openrouter/acme/model-one"
M2 = "openrouter/acme/model-two"


def _ep(tag, name, ctx=131072, out=16384, quant="bf16", params=("tools", "temperature")):
    return EndpointFact(tag=tag, slug=provider_slug(tag), provider_name=name, context_length=ctx,
                        max_completion_tokens=out, quantization=quant, supported_parameters=list(params))


def _entry(mid, **kw):
    base = dict(model_id=mid, display_name=mid, model_type=ModelType.API,
                domains={TaskDomain.GENERAL, TaskDomain.CONVERSATION, TaskDomain.REASONING},
                max_complexity=ComplexityTier.EXTREME, context_window=131072, cost_per_1k_tokens=0.001,
                avg_latency_ms=1000, priority=30)
    base.update(kw)
    return ModelEntry(**base)


def _engine(tmp_path, models=(M1, M2), facts=None):
    cfg = InferenceDifferenceConfig()
    cfg.models = {m: _entry(m) for m in models}
    cfg.exploration_rate = 0.0
    cfg.exploration_min_rate = 0.0
    eng = RoutingEngine(config=cfg, hardware=HardwareProfile(has_gpu=False, available_vram_gb=0.0))
    eng._peninsula = None
    eng._commons_competence = None
    eng._stats_cache_path = str(tmp_path / "model_stats_cache.msgpack")
    if facts is not None:
        store = ProviderFactsStore(str(tmp_path / "provider_facts.msgpack"))
        for mid, eps in facts.items():
            store.put(mid, eps)
        eng.set_provider_facts(store)
    return eng


def _cls(domain=TaskDomain.CONVERSATION, complexity=ComplexityTier.MEDIUM, est=500, window=4096):
    return RequestClassification(primary_domain=domain, complexity=complexity, estimated_tokens=est,
                                 requires_context_window=window)


TWO_PROVIDERS = {
    M1: [_ep("deepinfra/turbo", "DeepInfra"), _ep("nebius", "Nebius")],
    M2: [_ep("together", "Together")],
}

E429 = {"error": "HTTP 429: rate limited", "http_status": 429, "error_type": "rate_limit_exceeded"}


# ---------------------------------------------------------------- A. provider facts + refresher

def test_endpoint_fact_parses_openrouter_listing_shape():
    ep = EndpointFact.from_api({
        "tag": "deepinfra/turbo", "provider_name": "DeepInfra", "context_length": 262144,
        "max_completion_tokens": 65536, "max_prompt_tokens": None, "quantization": "fp4",
        "supported_parameters": ["tools", "max_tokens"], "pricing": {"prompt": "0.0000003"}, "status": 0,
        "uptime_last_30m": 100, "uptime_last_1d": 99.1,
        "latency_last_30m": {"p50": 940.5}, "throughput_last_30m": {"p50": 45.0},
    })
    assert (ep.slug, ep.provider_name, ep.context_length, ep.max_completion_tokens) == (
        "deepinfra", "DeepInfra", 262144, 65536)
    assert ep.max_prompt_tokens is None and ep.quantization == "fp4"
    assert ep.latency_p50_ms == 940.5 and ep.throughput_p50 == 45.0 and "tools" in ep.supported_parameters


def test_facts_store_round_trips_atomically(tmp_path):
    path = str(tmp_path / "pf.msgpack")
    s = ProviderFactsStore(path)
    s.put(M1, TWO_PROVIDERS[M1])
    s.save()
    t = ProviderFactsStore(path)
    t.load()
    assert [e.tag for e in t.endpoints(M1)] == ["deepinfra/turbo", "nebius"]
    assert t.slug_for_name("DeepInfra", M1) == "deepinfra"
    assert not list(tmp_path.glob("*.tmp-*"))  # temp file replaced, not left behind


def test_facts_store_load_tolerates_corrupt_file(tmp_path):
    p = tmp_path / "pf.msgpack"
    p.write_bytes(b"\x00garbage")
    s = ProviderFactsStore(str(p))
    s.load()
    assert s.endpoints(M1) == []


def test_refresher_sweep_is_fail_soft_and_keeps_stale_facts(tmp_path):
    store = ProviderFactsStore(str(tmp_path / "pf.msgpack"))
    store.put(M2, [_ep("together", "Together")], fetched_at=1.0)
    calls = []

    def fetch(or_model, key):
        calls.append(or_model)
        if or_model == "acme/model-two":
            raise RuntimeError("HTTP 503")
        return [_ep("nebius", "Nebius")]

    r = ProviderFactsRefresher(store, models_fn=lambda: [M1, M2, "venice/x", "ollama/y"], min_spacing_s=0,
                               fetch_fn=fetch, api_key_fn=lambda: "sk-test")
    assert r.sweep_once() == 1
    assert sorted(calls) == ["acme/model-one", "acme/model-two"]      # only OpenRouter-bound models
    assert [e.tag for e in store.endpoints(M2)] == ["together"]       # failure kept the stale facts
    assert [e.tag for e in store.endpoints(M1)] == ["nebius"]
    assert (tmp_path / "pf.msgpack").exists()


def test_refresher_without_key_makes_no_calls(tmp_path):
    called = []
    r = ProviderFactsRefresher(ProviderFactsStore(str(tmp_path / "pf")), models_fn=lambda: [M1],
                               fetch_fn=lambda m, k: called.append(m) or [], api_key_fn=lambda: "")
    assert r.sweep_once() == 0 and called == []


def test_refresher_runs_on_its_own_clock_without_any_request(tmp_path):
    """LAW 8: the refresher sweeps on a wall-clock timer — nothing conversational drives it."""
    swept = threading.Event()

    def fetch(m, k):
        swept.set()
        return [_ep("nebius", "Nebius")]

    r = ProviderFactsRefresher(ProviderFactsStore(str(tmp_path / "pf")), models_fn=lambda: [M1],
                               interval_s=1, initial_delay_s=0, min_spacing_s=0, fetch_fn=fetch,
                               api_key_fn=lambda: "sk-test")
    r.start()
    try:
        assert swept.wait(5)
    finally:
        r.stop()
    assert r.sweeps >= 1


def test_openrouter_model_name_matches_model_client_resolution():
    assert openrouter_model_name("openrouter/acme/x") == "acme/x"
    assert openrouter_model_name("deepseek/deepseek-chat") == "deepseek/deepseek-chat"
    assert openrouter_model_name("venice/deepseek-v3.2") is None
    assert openrouter_model_name("ollama/llama3.1:8b") is None


# ---------------------------------------------------------------- B. served provider per call

class _FakeResp:
    def __init__(self, body, status=200, headers=None):
        self._b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self._status = status
        self._headers = headers or {}
        self._sent = False

    def read(self, n=-1):
        if n is None or n < 0:
            out, self._b = self._b, b""
            return out
        out, self._b = self._b[:n], self._b[n:]
        return out

    def getheaders(self):
        return list(self._headers.items())

    def getcode(self):
        return self._status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _client():
    from inference_difference.model_client import ModelClient
    c = ModelClient()
    c._policy_blocked, c._policy_blocked_expiry, c._provider_blocked = set(), {}, {}
    return c


def test_blocking_call_captures_served_provider_and_tokens():
    body = {"id": "gen-1", "provider": "DeepInfra", "model": "acme/model-one",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
    with patch("urllib.request.urlopen", return_value=_FakeResp(body)):
        r = _client().call(M1, [{"role": "user", "content": "x"}])
    assert r.success and r.provider == "DeepInfra" and r.finish_reason == "length"
    assert r.usage["completion_tokens"] == 7 and r.http_status == 200


def test_blocking_call_falls_back_to_generation_lookup_when_provider_missing(monkeypatch):
    body = {"id": "gen-abc", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    seen = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        seen.append(url)
        if "/generation?id=" in url:
            return _FakeResp({"data": {"provider_name": "Nebius"}})
        return _FakeResp(body)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        r = _client().call(M1, [{"role": "user", "content": "x"}])
    assert r.provider == "Nebius"
    assert any("generation?id=gen-abc" in u for u in seen)


def test_blocking_http_error_carries_typed_error_provider_and_retry_after():
    err_body = json.dumps({"error": {"code": 504, "message": "aborted", "metadata": {
        "provider_name": "Venice", "error_type": "timeout",
        "previous_errors": [{"code": 429, "provider_name": "DeepInfra"}]}}}).encode()

    def boom(req, timeout=None):
        raise urllib.error.HTTPError("u", 504, "x", {"Retry-After": "7"}, io.BytesIO(err_body))

    with patch("urllib.request.urlopen", side_effect=boom):
        r = _client().call(M1, [{"role": "user", "content": "x"}])
    assert not r.success and r.http_status == 504 and r.error_type == "timeout"
    assert r.error_provider == "Venice" and r.retry_after_s == 7.0
    assert r.error_previous_providers == [{"provider": "DeepInfra", "code": 429}]


def test_in_body_error_envelope_is_not_a_silent_success():
    body = {"id": "gen-2", "provider": "Nebius", "choices": [],
            "error": {"code": 502, "message": "bad", "metadata": {"error_type": "provider_unavailable"}}}
    with patch("urllib.request.urlopen", return_value=_FakeResp(body)):
        r = _client().call(M1, [{"role": "user", "content": "x"}])
    assert r.success is False and r.error_type == "provider_unavailable" and r.provider == "Nebius"


def test_stream_captures_provider_usage_and_finish_reason():
    lines = [
        {"id": "gen-s", "provider": "Together", "choices": [{"delta": {"content": "he"}}]},
        {"id": "gen-s", "provider": "Together", "choices": [{"delta": {"content": "y"}, "finish_reason": "stop"}]},
        {"id": "gen-s", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
    ]
    raw = b"".join(b"data: " + json.dumps(x).encode() + b"\n\n" for x in lines) + b"data: [DONE]\n\n"
    with patch("urllib.request.urlopen", return_value=_FakeResp(raw)):
        it, res = _client().call_stream(M2, [{"role": "user", "content": "x"}])
        list(it)
    assert res.success and res.content == "hey" and res.provider == "Together"
    assert res.usage["completion_tokens"] == 2 and res.generation_id == "gen-s" and res.finish_reason == "stop"


def test_shaping_reaches_the_wire_body():
    sent = {}

    def cap(req, timeout=None):
        sent.update(json.loads(req.data.decode()))
        return _FakeResp({"id": "g", "provider": "Nebius", "choices": [{"message": {"content": "ok"}}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    with patch("urllib.request.urlopen", side_effect=cap):
        _client().call(M1, [{"role": "user", "content": "x"}],
                       extra_params={"provider": {"order": ["nebius"], "ignore": ["deepinfra"]}})
    assert sent["provider"] == {"order": ["nebius"], "ignore": ["deepinfra"]}


# ---------------------------------------------------------------- C. substrate deposit target

class _FakeNG:
    def __init__(self, recs=None):
        self.deposits = []
        self.recs = recs or []

    def record_outcome(self, embedding, target_id, success, strength, metadata=None):
        self.deposits.append((target_id, success, dict(metadata or {})))

    def get_recommendations(self, embedding, top_k=20):
        return list(self.recs)


def test_deposit_target_includes_provider_when_known_raw(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    eng._ng_lite = _FakeNG()
    dec = RoutingDecision(model_id=M1, classification=_cls())
    eng.report_outcome(dec, success=True, latency_ms=900,
                       metadata={"served_provider": "DeepInfra", "completion_tokens": 9, "finish_reason": "stop"})
    eng.report_outcome(dec, success=False, metadata={"error": "<urlopen error Name or service not known>"})
    eng.report_outcome(dec, success=False, metadata={"error": "HTTP 400: bad", "http_status": 400})
    (t1, s1, md1), (t2, s2, _md2) = eng._ng_lite.deposits   # the network-down outcome is NOT among them
    assert t1 == f"{M1}@deepinfra" and s1 is True
    assert md1["served_provider"] == "DeepInfra" and md1["completion_tokens"] == 9  # raw values, unclassified
    assert t2 == M1 and s2 is False  # provider unknown -> model-only target


def test_network_down_is_not_deposited(tmp_path):
    """Josh 2026-10-15: a local-network failure never reached a model or provider, so it is not experience of either."""
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    eng._ng_lite = _FakeNG()
    eng.report_outcome(RoutingDecision(model_id=M1, classification=_cls()), success=False,
                       metadata={"error": "<urlopen error [Errno -2] Name or service not known>"})
    assert eng._ng_lite.deposits == []


# ---------------------------------------------------------------- D. pair penalty box

def test_named_case_1_429_boxes_only_that_pair_and_model_stays_routable(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    dec = RoutingDecision(model_id=M1, classification=_cls())
    eng.report_outcome(dec, success=False, metadata={**E429, "error_provider": "DeepInfra"})
    assert eng._provider_penalty.is_penalized(M1, "deepinfra")
    assert not eng._provider_penalty.is_penalized(M1, "nebius")
    assert eng._model_penalty.level(M1) == 0                 # the long ladder did not move
    assert eng._model_success_stats.get(M1, []) == []        # no model-level failure sample either
    cands, _ = eng._filter_candidates(_cls())
    assert M1 in [m.model_id for m in cands]                 # still routable (through nebius)
    prefs = eng.provider_preferences(M1, _cls())
    assert prefs["provider"]["ignore"] == ["deepinfra"]


def test_provider_fault_never_climbs_the_long_ladder_alone(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    dec = RoutingDecision(model_id=M1, classification=_cls())
    for _ in range(50):
        eng.report_outcome(dec, success=False, metadata={**E429, "error_provider": "DeepInfra"})
    assert eng._model_penalty.level(M1) == 0
    assert eng._provider_penalty.level(M1, "deepinfra") == len(eng.config.provider_penalty_ladder_seconds)


def test_named_case_2_every_provider_boxed_falls_back_exactly_as_today(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    dec1 = RoutingDecision(model_id=M1, classification=_cls())
    eng.report_outcome(dec1, success=False, metadata={**E429, "error_provider": "DeepInfra"})
    eng.report_outcome(dec1, success=False, metadata={**E429, "error_provider": "Nebius"})
    # every provider of M1 boxed -> M1 excluded while M2 is available
    assert [m.model_id for m in eng._filter_candidates(_cls())[0]] == [M2]
    # M1's second provider fault completed "all providers failing" -> model-level ladder +1 (exactly once)
    assert eng._model_penalty.level(M1) == 1
    # box M2's only provider too: every candidate excluded -> today's fallthrough (keep pool, least-penalised first)
    eng.report_outcome(RoutingDecision(model_id=M2, classification=_cls()), success=False,
                       metadata={**E429, "error_provider": "Together"})
    assert eng._model_penalty.level(M2) == 0                  # single-provider model: stays a provider fault
    with patch("inference_difference.router.logger") as lg:
        cands, _ = eng._filter_candidates(_cls())
    assert [m.model_id for m in cands] == [M2, M1]            # level 0 before level 1
    assert any("would exclude ALL" in str(c) for c in lg.warning.call_args_list)
    # and with no shaping possible (every endpoint boxed) nothing is ignored: last resort sends as today
    assert "ignore" not in (eng.provider_preferences(M1, _cls()) or {}).get("provider", {})


def test_model_fault_still_climbs_model_ladder(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    eng.report_outcome(RoutingDecision(model_id=M1), success=False,
                       metadata={"error": "HTTP 404: No endpoints found for acme/model-one", "http_status": 404})
    assert eng._model_penalty.level(M1) == 1


def test_transport_fault_charges_nobody(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    eng.report_outcome(RoutingDecision(model_id=M1), success=False,
                       metadata={"error": "<urlopen error [Errno -2] Name or service not known>"})
    assert eng._model_penalty.level(M1) == 0 and eng._provider_penalty.active_count() == 0
    assert eng._model_success_stats.get(M1, []) == []


def test_unattributed_429_is_a_short_model_wide_box_not_the_ladder(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    eng.report_outcome(RoutingDecision(model_id=M1), success=False, metadata=dict(E429))
    assert eng._provider_penalty.wildcard_penalized(M1) and eng._model_penalty.level(M1) == 0
    assert M1 not in [m.model_id for m in eng._filter_candidates(_cls())[0]]
    eng.report_outcome(RoutingDecision(model_id=M1), success=True, metadata={"served_provider": "Nebius"})
    assert not eng._provider_penalty.wildcard_penalized(M1)


def test_previous_errors_box_their_providers_too(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    eng.report_outcome(RoutingDecision(model_id=M1), success=False, metadata={
        "error": "HTTP 504: aborted", "http_status": 504, "error_provider": "Nebius",
        "error_previous_providers": [{"provider": "DeepInfra", "code": 429}]})
    assert eng._provider_penalty.boxed_providers(M1) == ["deepinfra", "nebius"]


def test_unnamed_5xx_keeps_todays_model_path():
    assert classify_failure({"error": "500 garbage output"}) == "model"
    assert classify_failure({"error": "HTTP 503: x", "http_status": 503, "error_provider": "Nebius"}) == "provider"
    assert classify_failure({"error": "x", "error_type": "provider_overloaded"}) == "provider"
    assert classify_failure({"error": "HTTP 400: bad", "http_status": 400}) == "model"
    assert classify_failure({"call_ok": True, "finish_reason": "length", "served_provider": "X"}) == "provider"


def test_pair_box_honours_retry_after_capped_and_clears_on_success():
    t = [1000.0]
    box = ProviderPenaltyBox(ladder_seconds=(30, 60, 600), time_fn=lambda: t[0])
    box.record_failure("m", "p", retry_after_s=90)
    assert box.reopen_at("m", "p") == 1090.0
    box.record_failure("m", "p", retry_after_s=10 ** 6)
    assert box.reopen_at("m", "p") == 1600.0                # capped at the last rung
    t[0] = 1700.0
    assert not box.is_penalized("m", "p")
    box.record_success("m", "p")
    assert box.level("m", "p") == 0


def test_named_case_4_persisted_box_migrates_without_loss(tmp_path):
    now = time.time()
    old = {"saved_at": now, "stats": {}, "tool_competence": {},
           "penalty": {"a": {"level": 7, "until": 0.0}, "b": {"level": 4, "until": now + 86400},
                       "c": {"level": 1, "until": now - 5}}}
    path = tmp_path / "model_stats_cache.msgpack"
    path.write_bytes(msgpack.packb(old))
    eng = _engine(tmp_path)
    eng.load_stats()
    for mid, e in old["penalty"].items():                 # every entry kept, same level + timer
        assert eng._model_penalty.level(mid) == e["level"]
        assert eng._model_penalty.entries()[mid]["until"] == e["until"]
        assert eng._model_penalty.is_legacy(mid)
    assert eng._model_penalty.is_blacklisted("a") and eng._model_penalty.is_penalized("b")
    eng.save_stats()
    data = msgpack.unpackb(path.read_bytes(), raw=False)
    assert data["penalty"]["a"]["level"] == 7 and "penalty_v2" in data   # old key still readable (rollback)
    # pre-pair reader still loads the model box from the new file
    legacy_reader = ModelPenaltyBox()
    legacy_reader.from_dict(data["penalty"])
    assert legacy_reader.level("b") == 4
    # second load: no re-migration, nothing lost; legacy entries can be cleared selectively
    eng2 = _engine(tmp_path)
    eng2.load_stats()
    assert eng2._model_penalty.level("a") == 7 and eng2._stats_dirty is False
    eng2._model_penalty.record_failure("fresh")
    assert eng2.clear_legacy_penalties() == 3
    assert eng2._model_penalty.level("fresh") == 1


def test_pair_box_and_window_round_trip(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    eng.report_outcome(RoutingDecision(model_id=M1, classification=_cls()), success=False,
                       metadata={**E429, "error_provider": "DeepInfra"})
    eng.report_outcome(RoutingDecision(model_id=M1, classification=_cls()), success=True, latency_ms=800,
                       metadata={"served_provider": "Nebius", "finish_reason": "stop"})
    eng.save_stats()
    eng2 = _engine(tmp_path, facts=TWO_PROVIDERS)
    eng2.load_stats()
    assert eng2._provider_penalty.is_penalized(M1, "deepinfra")
    assert eng2._pair_stats.records(M1, "nebius")[0]["ok"] is True
    assert eng2.get_stats()["provider_routing"]["served"]  # stats exposes served pairs (in-process history)


# ---------------------------------------------------------------- E. request shaping + hard filters

def test_named_case_3_hard_filter_drops_endpoint_with_too_small_context_or_output():
    eps = [_ep("tiny-ctx", "TinyCtx", ctx=8192), _ep("tiny-out", "TinyOut", out=1024), _ep("ok", "Ok")]
    needs = RequestNeeds(context_tokens=20000, prompt_tokens=18000, output_tokens=2000)
    keep, drop = hard_filter_endpoints(eps, needs, margin=0.1, quant_floor="fp8", deep_min_output=4096)
    assert [e.tag for e in keep] == ["ok"]
    assert {d.tag: why.split()[0] for d, why in drop} == {"tiny-ctx": "context", "tiny-out": "output"}


def test_hard_filter_quantization_floor_only_for_deep_and_unknown_passes():
    eps = [_ep("q4", "Q4", quant="fp4"), _ep("q8", "Q8", quant="fp8"), _ep("unk", "Unk", quant="unknown")]
    keep, _ = hard_filter_endpoints(eps, RequestNeeds(deep=False), margin=0.1, quant_floor="fp8",
                                    deep_min_output=0)
    assert len(keep) == 3
    keep, drop = hard_filter_endpoints(eps, RequestNeeds(deep=True), margin=0.1, quant_floor="fp8",
                                       deep_min_output=0)
    assert [e.tag for e in keep] == ["q8", "unk"] and drop[0][0].tag == "q4"


def test_hard_filter_excludes_model_with_no_servable_endpoint_and_ignores_dropped_tags(tmp_path):
    facts = {M1: [_ep("small", "Small", ctx=8192), _ep("big", "Big", ctx=200000)],
             M2: [_ep("small2", "Small2", ctx=8192)]}
    eng = _engine(tmp_path, facts=facts)
    big = _cls(window=60000, est=1000)  # ~39k prompt tokens + 1k out
    assert [m.model_id for m in eng._filter_candidates(big)[0]] == [M1]
    assert eng.provider_preferences(M1, big)["provider"]["ignore"] == ["small"]


def test_require_parameters_with_tools_and_no_shaping_off_openrouter(tmp_path):
    eng = _engine(tmp_path, models=(M1, "venice/x"), facts=TWO_PROVIDERS)
    assert eng.provider_preferences(M1, _cls(), has_tools=True)["provider"]["require_parameters"] is True
    assert eng.provider_preferences("venice/x", _cls(), has_tools=True) is None
    assert eng.provider_preferences(M1, _cls()) is None          # nothing learned / boxed -> unshaped
    eng.config.provider_shaping_enabled = False
    assert eng.provider_preferences(M1, _cls(), has_tools=True) is None


def test_order_lists_best_learned_providers_first(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    eng._ng_lite = _FakeNG(recs=[(f"{M1}@nebius", 0.9, ""), (f"{M1}@deepinfra", 0.4, ""), (M1, 0.5, "")])
    assert eng.provider_preferences(M1, _cls())["provider"]["order"] == ["nebius", "deepinfra"]


# ---------------------------------------------------------------- F. scoring

def test_learned_score_is_best_eligible_provider(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    eng._ng_lite = _FakeNG(recs=[(f"{M1}@nebius", 0.9, ""), (f"{M1}@deepinfra", 0.3, ""), (M1, 0.5, "")])
    model = eng.config.models[M1]
    assert eng._score_learned(model, _cls()) == pytest.approx(0.9)
    eng._provider_penalty.record_failure(M1, "nebius")           # best provider boxed -> next eligible
    assert eng._score_learned(model, _cls()) == pytest.approx(0.3)
    eng._ng_lite = _FakeNG(recs=[(M1, 0.7, "")])                 # no pair evidence -> today's model weight
    assert eng._score_learned(model, _cls()) == pytest.approx(0.7)


def test_task_class_scores_from_raw_window():
    cfg = InferenceDifferenceConfig()
    st = PairOutcomeStats(window=10)
    for _ in range(4):
        st.record("m", "slow", ok=True, latency_ms=16000, finish_reason="stop", prompt_tokens=1,
                  completion_tokens=1, domain="conversation", complexity="medium")
        st.record("m", "cut", ok=True, latency_ms=500, finish_reason="length", prompt_tokens=1,
                  completion_tokens=1, domain="reasoning", complexity="high")
    assert st.score("m", "slow", "chat", cfg) == pytest.approx(cfg.provider_chat_latency_floor_ms / 16000)
    assert st.score("m", "cut", "deep", cfg) == 0.0              # completed-but-cut-off is not "good" for deep
    assert st.score("m", "slow", "deep", cfg) == 1.0             # latency floor applies to chat only
    assert task_class("reasoning", "low", cfg) == "deep"
    assert task_class("summarization", "high", cfg) == "extraction"
    assert task_class("conversation", "medium", cfg) == "chat"


def test_veto_only_pair_box_absent_from_scoring_methods():
    """The pair box may only REMOVE providers before scoring; no _score_* method reads it directly."""
    import inspect
    from inference_difference import router as r
    src = "".join(inspect.getsource(fn) for name, fn in inspect.getmembers(r.RoutingEngine, inspect.isfunction)
                  if name.startswith("_score"))
    assert "_provider_penalty" not in src and "_model_penalty" not in src


# ---------------------------------------------------------------- app wiring helpers

def test_attempt_decision_targets_the_model_actually_tried():
    from inference_difference import app
    cfg = InferenceDifferenceConfig()
    cfg.models = {M1: _entry(M1), M2: _entry(M2, conversational_quality=0.9)}
    with patch.object(app._state, "config", cfg, create=True):
        primary = RoutingDecision(model_id=M1, model_entry=cfg.models[M1])
        d = app._attempt_decision(primary, M2)
    assert d.model_id == M2 and d.model_entry is cfg.models[M2]
    assert primary.model_id == M1                                # the original decision is untouched


def test_provider_meta_extracts_raw_facts():
    from inference_difference import app
    from inference_difference.model_client import ModelResponse
    r = ModelResponse(provider="Nebius", http_status=200, finish_reason="length",
                      usage={"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14})
    md = app._provider_meta(r)
    assert md["served_provider"] == "Nebius" and md["completion_tokens"] == 9 and md["finish_reason"] == "length"


def test_named_provider_429_does_not_start_whole_model_cooldown():
    """Law-review HIGH: model_client's per-model 429 cooldown must not bench every provider of the model."""
    named = json.dumps({"error": {"code": 429, "message": "Provider returned error",
                                  "metadata": {"provider_name": "DeepInfra", "raw": "rate-limited upstream"}}}).encode()
    unnamed = json.dumps({"error": {"code": 429, "message": "rate limited"}}).encode()
    c = _client()
    for body in (named, unnamed):
        def boom(req, timeout=None, _b=body):
            raise urllib.error.HTTPError("u", 429, "x", {}, io.BytesIO(_b))
        mid = M1 if body is named else M2
        with patch("urllib.request.urlopen", side_effect=boom):
            c.call(mid, [{"role": "user", "content": "x"}])
    assert not c.is_rate_limited_by_id(M1)                   # named -> the router's pair box owns it
    assert c.is_rate_limited_by_id(M2)                       # unattributed -> today's cooldown
    assert c._rate_limit_hits["acme/model-two"] == 1         # one hit per 429 (duplicate block removed)


def test_generation_lookup_respects_config_switch(monkeypatch):
    from inference_difference.model_client import ModelClient
    body = {"id": "gen-x", "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    seen = []

    def fake(req, timeout=None):
        seen.append(req.full_url)
        return _FakeResp(body)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    c = ModelClient(generation_lookup=False)
    c._policy_blocked, c._policy_blocked_expiry, c._provider_blocked = set(), {}, {}
    with patch("urllib.request.urlopen", side_effect=fake):
        r = c.call(M1, [{"role": "user", "content": "x"}])
    assert r.provider == "" and not any("generation" in u for u in seen)


def test_read_timeout_is_a_short_box_not_the_long_ladder(tmp_path):
    eng = _engine(tmp_path, facts=TWO_PROVIDERS)
    eng.report_outcome(RoutingDecision(model_id=M1), success=False, metadata={"error": "The read operation timed out"})
    assert eng._model_penalty.level(M1) == 0 and eng._provider_penalty.wildcard_penalized(M1)
