# BUILD — TID (model, provider)-level routing

Branch `cc-laptop-tid-provider-routing-20261015`, base `7466fea` (`origin/cc-laptop-tid-pool-20261003`).
Design: `~/docs/superpowers/specs/2026-10-15-tid-provider-level-routing-design.md` (approved by Josh 2026-10-15;
punch list #1040, folds in #1126). **Built, not installed.** Nothing here is live.

## What changed, per component

**A. Provider facts on their own clock (LAW 8)**
- New file `inference_difference/provider_facts.py`:
  - `ProviderFactsStore` (`:160`) caches endpoint facts per model and saves them atomically as msgpack (`save` `:223`, `load` `:235`, using `atomic_io`).
  - `ProviderFactsRefresher` (`:271`) is a daemon thread. It sleeps on `Event.wait(interval)` and sweeps every OpenRouter-bound model in the enabled pool (`sweep_once` `:317`).
    - The sweep is rate-limited (`min_spacing_s`) and fail-soft: when a fetch fails, the old facts are kept.
    - With no `OPENROUTER_API_KEY` it makes no calls at all, which also keeps the tests off the network.
  - `fetch_endpoints_openrouter` (`:261`) calls `GET /api/v1/models/{author}/{slug}/endpoints`, using the same account-key pattern as `catalog_manager`.
  - `EndpointFact.from_api` (`:89`) parses the response. Its shape was checked against the OpenRouter API reference and one live read-only GET:
    - `tag` is the routing slug (e.g. `deepinfra/turbo`).
    - `provider_name` is the display name that chat responses report.
- `inference_difference/app.py:1036`: the lifespan step loads `provider_facts.msgpack`, attaches the store to the router and starts the refresher. Shutdown stops it.

**B. Served provider per call** (`inference_difference/model_client.py`)
- `ModelResponse` and `StreamResult` gained these fields: `provider`, `http_status`, `error_type`, `error_provider`, `error_previous_providers`, `retry_after_s`, plus `generation_id` and `usage` on streams.
- Blocking success path (`:799`):
  - Takes `provider` from the response body.
  - If the body has none, falls back to `GET /api/v1/generation?id=…` (`_lookup_generation_provider` `:230`). This can be switched off in config.
  - A 200 whose body is an OpenRouter error envelope with no content is now `success=False`. Before, it was a silent empty success.
- Error facts:
  - `_or_error_facts` (`:192`) parses the full error body, not the 200-character log excerpt.
  - `_retry_after` (`:221`) reads the `Retry-After` header.
- Streaming: per-chunk capture at `:1291` (provider, generation id, usage, mid-stream error) and an error-body parse at `:1349`.
- 429 handling (`:842`):
  - A 429 that names its provider no longer starts the whole-model cooldown, which used to bench every provider of the model. The router's pair box handles it now.
  - The duplicated 429 block from `c119c31` (2026-04-19) is removed. It counted two hits per 429, so the first cooldown was 120s instead of 60s.

**C. Substrate deposit** (`inference_difference/router.py:755`)
- The NG-Lite target is `<model_id>@<provider_slug>` when the provider is known, otherwise `<model_id>`.
- The metadata carries the raw per-call facts: served or error provider, status, error_type, tokens, finish_reason. Nothing is pre-classified (LAW 7).
- No vendored file was touched.

**D. Penalty box keyed by pair**
- `inference_difference/model_penalty.py:199` adds `ProviderPenaltyBox`:
  - Short, capped ladder, default 30/60/120/300/600s. A `Retry-After` value is honoured, up to the cap.
  - It never escalates by itself.
  - The `*` slot covers provider faults whose provider can't be identified.
- `ModelPenaltyBox` gained a legacy tag and `clear_legacy()` (`:131`, `:141`).
- Fault attribution is `classify_failure`, `inference_difference/provider_routing.py:48`:

  | Failure | Classed as |
  |---|---|
  | 429 / 408 | provider |
  | OpenRouter `rate_limit_exceeded` / `provider_overloaded` / `provider_unavailable` / `timeout` | provider |
  | 5xx with a **named** provider | provider |
  | read timeout | provider (wildcard box) |
  | completed but cut off with nothing usable | provider |
  | DNS / network unreachable | transport |
  | everything else (including an unnamed 5xx) | model, as today |

- `router.report_outcome` (`:698`):
  - Provider fault → `_record_provider_fault` (`:1005`). This boxes the pair, plus the `previous_errors` providers. It records no model success-rate sample.
  - The model ladder moves by +1 only on the transition to "every known provider boxed", and only when the model has at least `provider_all_fail_min_providers` providers.
  - Transport fault → nobody is charged.
- Eligibility, `_model_eligible` (`:1027`): a model is eligible while its model-level entry is clear and at least one provider is unboxed.
- The "every provider boxed" case goes through the existing all-excluded fallthrough unchanged: pool kept, least-penalised first, ties broken by soonest reopen.
- Migration, `load_stats` (`:1264`), runs once:
  - Every pre-pair per-model entry is kept as a model-level entry with the same level and timer, tagged `legacy`. Nothing is dropped.
  - `save_stats` (`:1334`) still writes the old `penalty` key in its old shape, so rollback stays safe. It adds `penalty_v2` (the pair boxes) and `pair_stats`.
- `inference_difference/app.py:2270`: `POST /routing/penalty/clear {"legacy": true}` clears only the migrated entries.

**Attribution fix (prerequisite for D)**
- `inference_difference/app.py:1559,1663` (`_attempt_decision` `:299`): the blocking fallback loop now charges each attempt to the model actually tried.
- Before, it charged the **primary** for every fallback failure and credited the primary with a fallback's success, so one bad cascade climbed the primary several rungs.
- Attempt decisions now carry the attempted model's own `ModelEntry`. The stream path already re-targeted `model_id` but kept the primary's entry, so the conversational-quality EMA was updating the wrong model.

**E. Request shaping + hard filters**
- `RoutingEngine.provider_preferences` (`router.py:1114`) builds `{"provider": {...}}` for OpenRouter calls only:
  - `order`: the best learned providers.
  - `ignore`: boxed pairs plus endpoint tags removed by the hard filter. If that would ignore every endpoint, nothing is ignored (last resort, same as today).
  - `require_parameters: true` when tools are sent.
- `app.py:1529,1607,1821` passes it through `extra_params` on the primary call, the blocking fallbacks and every stream attempt.
- Hard filters, `provider_facts.endpoint_fails` (`:127`) / `hard_filter_endpoints` (`:148`), drop an endpoint when:
  - its context is below need × (1 + margin);
  - its prompt cap is too small;
  - its output cap is below max(requested output × (1 + margin), deep-tier minimum);
  - in the deep tier only, its quantization is below the floor.
- A cap or quantization reported as unknown never filters.
- `_filter_candidates` (`router.py:1481`) drops a model only when it has facts and no endpoint can serve the request. It never empties the pool.

**F. Scoring**
- `_score_learned` (`router.py:1668`) now scores the model as its best **eligible** provider (`_provider_scores` `:1095`). Each provider's score is its `<model>@<provider>` NG-Lite weight, blended at `provider_score_blend` with that pair's task-class score.
- Task-class scores come from `PairOutcomeStats.score` (`provider_routing.py:102`), derived when read:
  - chat: success rate × latency factor, with the latency floor applied to chat only;
  - deep: completed and not cut off (`finish_reason != "length"`);
  - extraction: success and well-formed (not cut off, errored or content-filtered).
- With no pair evidence, the model-only weight is used exactly as before.
- Both boxes stay veto-only: no `_score_*` method reads either box, and a test enforces this.

**Observability**
- `/stats` → `provider_routing`, built in `router.py:885`. It reports:
  - `served`: per `model@provider` from the persisted window, so it survives restarts and save cycles;
  - `recent_served`;
  - active pair and model boxes, blacklisted count, legacy count;
  - provider facts sweep status.

## Tests

- New file `tests/test_provider_routing.py`, 38 tests, covering every component. The four §5 named cases are:
  - `test_named_case_1_429_boxes_only_that_pair_and_model_stays_routable`
  - `test_named_case_2_every_provider_boxed_falls_back_exactly_as_today`
  - `test_named_case_3_hard_filter_drops_endpoint_with_too_small_context_or_output`
  - `test_named_case_4_persisted_box_migrates_without_loss`
- `tests/test_atomic_saves.py`: the stub engine gained `_provider_penalty` and `_pair_stats`, because `save_stats` now writes them.
- No test touches the network.
- Full suite on the final code head `56a9513`: see "Full-suite result" below.

## Env / config knobs added

All are fields on `InferenceDifferenceConfig`. Each default is read from the environment variable at construction (LAW 5). **None are set in the env file today.** To pin a value, add it to `~/.config/inference-difference/env`.

| Env var | Default | Meaning |
|---|---|---|
| `TID_PROVIDER_FACTS_ENABLED` | `1` | Start the facts refresher |
| `TID_PROVIDER_FACTS_REFRESH_S` | `3600` | Sweep interval (own clock) |
| `TID_PROVIDER_FACTS_INITIAL_DELAY_S` | `20` | Delay before the first sweep after start |
| `TID_PROVIDER_FACTS_MIN_SPACING_S` | `1.0` | Rate limit between endpoint-listing calls |
| `TID_PROVIDER_FACTS_FETCH_TIMEOUT_S` | `20` | Timeout for each listing call |
| `TID_PROVIDER_FACTS_PATH` | repo root `provider_facts.msgpack` | Facts cache path |
| `TID_PROVIDER_GENERATION_LOOKUP` | `1` | `GET /generation` fallback when a response lacks `provider` |
| `TID_PROVIDER_GENERATION_LOOKUP_TIMEOUT_S` | `3` | Timeout for that lookup |
| `TID_PROVIDER_PENALTY_LADDER_S` | `30,60,120,300,600` | Pair-box ladder (capped) |
| `TID_PROVIDER_ALL_FAIL_MIN_PROVIDERS` | `2` | Minimum known providers before all-boxed moves the model ladder |
| `TID_PROVIDER_HEADROOM_MARGIN` | `0.10` | Hard-filter margin on context, prompt and output needs |
| `TID_PROVIDER_DEEP_QUANT_FLOOR` | `fp8` | Deep-tier quantization floor |
| `TID_PROVIDER_DEEP_MIN_OUTPUT_TOKENS` | `4096` | Deep-tier minimum endpoint output cap |
| `TID_PROVIDER_SHAPING_ENABLED` | `1` | Kill switch for `provider.order` / `ignore` / `require_parameters` |
| `TID_PROVIDER_ORDER_MAX` | `5` | Maximum providers in `order` |
| `TID_PROVIDER_CHAT_LATENCY_FLOOR_MS` | `8000` | Chat-tier latency floor |
| `TID_PROVIDER_DEEP_DOMAINS` | `reasoning` | Domains classed as deep |
| `TID_PROVIDER_DEEP_COMPLEXITIES` | `high,extreme` | Complexities classed as deep |
| `TID_PROVIDER_EXTRACTION_DOMAINS` | `summarization,translation` | Domains classed as extraction |
| `TID_PROVIDER_STATS_WINDOW` | `50` | Per-pair raw window size |
| `TID_PROVIDER_STATS_MIN_SAMPLES` | `3` | Minimum samples before a pair's task-class score counts |
| `TID_PROVIDER_SCORE_BLEND` | `0.5` | Pair-window weight vs. NG-Lite pair weight |

## Install (Executive) — what changes

1. **Code.** The live laptop TID runs from `~/The-Inference-Difference-vps`, on `cc-laptop-tid-pool-20261003` at `7466fea`. Fast-forward that checkout to this branch's head. This branch is `7466fea` + 4 commits, so it is a pure fast-forward.
2. **Restart: yes.** `systemctl --user restart inference-difference`. It is a user unit on the laptop. Follow the usual "old PID fully dead" check.
3. **State on first start:**
   - `model_stats_cache.msgpack` is migrated once. The 334 per-model entries (307 at L7 as of 2026-10-15 13:39) stay as model-level entries tagged `legacy`, and `penalty_v2` and `pair_stats` are added.
   - `provider_facts.msgpack` is created after the first sweep (about 20s plus about 1s per OpenRouter model, at the defaults). It is gitignored.
   - No env change is required.
4. **The acceptance bar needs one operator decision.** The migration keeps the 307 legacy blacklists, as the design requires, so they keep blocking. The "would exclude ALL" warning rate will **not** fall unless they are cleared. After install, if Josh agrees: `curl -sX POST localhost:7437/routing/penalty/clear -H 'Content-Type: application/json' -d '{"legacy":true}'`. That clears only the migrated entries.
5. **Verify:**
   - `GET /stats` → `provider_routing.served` fills with `model@provider` keys;
   - `facts.models` > 0 after the first sweep;
   - `pair_boxes_active` stays small;
   - watch the "would exclude ALL" warning count for at least 2 stats save cycles (60s each) and 2 facts sweeps.

**Rollback**
- Check out `7466fea` in `~/The-Inference-Difference-vps` and restart.
- The old code reads the `penalty` key unchanged; the extra `legacy` field is ignored. It also ignores `penalty_v2`, `pair_stats` and `provider_facts.msgpack`.
- One thing to know: an old-code save drops `penalty_v2`. Re-installing later would therefore re-run the migration, which only re-tags the then-current entries as legacy.
- Softer levers that need no code rollback:
  - `TID_PROVIDER_SHAPING_ENABLED=0` stops request shaping.
  - `TID_PROVIDER_FACTS_ENABLED=0` stops the refresher. With no facts there are no hard filters and no provider names in requests.

## Full-suite result

The full suite ran once on code head `56a9513`, with `NEUROGRAPH_CANONICAL_PATH=$HOME/NeuroGraph-worktrees/trial-s4-20261002` and `venv/bin/python3 -m pytest -q -p no:cacheprovider --tb=short -rfE`, in 37m44s:

**7 failed, 457 passed, 22 errors.** This is exactly the pre-existing baseline set. There are **no new failures**.

| Kind | Tests |
|---|---|
| errors (22) | `tests/test_ng_ecosystem.py`, the whole set |
| failed | `test_ng_ecosystem.py::TestNGSaaSBridge::test_saas_not_connected_by_default` |
| failed | `test_ng_ecosystem.py::TestNGSaaSBridge::test_saas_returns_none` |
| failed | `test_inference_difference.py::TestRouter::test_consciousness_boost` |
| failed | `test_inference_difference.py::TestExploreExploit::test_exploration_config_defaults` |
| failed | `test_transparent_proxy.py::TestCatalogModelRegistration::test_register_catalog_models_converts_correctly` |
| failed | `test_transparent_proxy.py::TestCatalogModelRegistration::test_v1_models_includes_static_models` |
| failed | `test_transparent_proxy.py::TestFallbackChain::test_fallback_on_failure` |

`test_fallback_on_failure` exercises the fallback loop this branch changed. It was re-run on its own at `7466fea` in a throwaway worktree and failed identically there (`assert 502 == 200`, line 345), so its failure is pre-existing and unchanged.

Before that run, the targeted suites gave 81 passed: `test_provider_routing`, `test_penalty_integration`, `test_two_axis_routing`, `test_atomic_saves`, `test_model_penalty`, `test_provider_block`, `test_raw_outcome`.

After the run, `git status` showed no vendored-file changes, and `git diff 7466fea` over the vendored files is empty.

## Open items / flags (not fixed here)

- **Transport failures still reach NG-Lite.** A DNS or network-down failure is still deposited with `success=False` against `<model_id>`, as raw experience. Only the box and stats paths skip it. Josh should decide whether a call that never reached a model should be deposited at all.
- **Install acceptance needs the legacy clear** (Install, step 4). Without it, the 307 migrated blacklists keep the "would exclude ALL" warnings going.
- **Pair windows have no time decay.** A provider pushed down `order` gathers few new samples. Its window stays as it was until the 24h discard on load. This is a sample window, not a countdown.
- **Streaming tool-call-only responses are treated as failures (pre-existing).** `_generate` keys success on `stream_result.content`, which never collects `delta.tool_calls`. A streamed response that only calls tools is therefore counted as failed and the next model is tried. This is out of scope here; it belongs on the punch list.
- **The NG-Lite state path is hardcoded (pre-existing).** `app.py` hardcodes `/home/josh/The-Inference-Difference/ng_lite_state.json`. The live laptop service runs from `~/The-Inference-Difference-vps`, so it writes NG state into the main checkout. That may be intentional; it should be confirmed.
- **The laws review asked for a decision on the local pair window.** The window feeds routing through a static `provider_score_blend`. Design §3.F asks for `<model>@<provider>` outcomes "per task class"; this window is how the per-task-class part was built. Confirm it is what was intended.
