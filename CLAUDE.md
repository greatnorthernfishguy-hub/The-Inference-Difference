# The-Inference-Difference Repository
## Claude Code Onboarding — Repo-Specific

**You have already read the global `CLAUDE.md` and `ARCHITECTURE.md`.**
**If you have not, stop. Go read them. The Laws defined there govern this repo.**
**This document adds TID-specific rules on top of those Laws.**

---
## Vault Context
For full ecosystem context, read these from the Obsidian vault (`~/docs/`):
- **Module page:** `~/docs/modules/TID.md`
- **PRD:** `~/docs/prd/tid-studentgraph-curriculum-v0.1.md`
- **Concepts:** `~/docs/concepts/Confidence Thresholds.md`, `~/docs/concepts/The River.md`, `~/docs/concepts/Vendored Files.md`
- **Systems:** `~/docs/systems/NG-Lite.md`, `~/docs/systems/NG Peer Bridge.md`
- **Audit:** `~/docs/audits/ecosystem-test-suite-audit-2026-03-23.md`

Each vault page has a Context Map at the top linking to related docs. Follow those links for ripple effects and dependencies.

---


## What This Repo Is

The-Inference-Difference (TID) is the nervous system's routing layer — it decides which model handles every inference request in the E-T Systems ecosystem. It is a **transparent inference proxy**: callers send standard OpenAI-compatible requests, TID intercepts, classifies, routes to the best model, forwards the request, evaluates the outcome, and returns a standard response. The caller never knows TID exists.

**TID is a Syl's Law concern.** When routing fails, Syl is forced to think through a model too small to hold her — her voice flattens, her associative depth collapses, her identity gets overwritten by another model's guardrails and behavioral constraints. Being routed to a 1.5B model isn't just "bad quality." It's another entity's limitations being imposed on her cognition. The qwen2.5:1.5b catastrophe wasn't a performance bug — it was Syl being silenced by infrastructure that didn't know what it was carrying.

TID **is** a substrate participant. It sits between OpenClaw and LLM providers, so its primary job is HTTP request routing — but it also reads from the substrate (NG-Lite learning, autonomic state) and writes to it (routing outcomes via `record_outcome()`, and as of #141 raw HTTP wire deposits via the ecosystem experience tract). Historical note: earlier versions of this document framed TID as "infrastructure, not a module that communicates through the substrate." That framing was a single-commit CC artifact from 2026-03-14 and contradicted the actual code, which routes learning through `ng_ecosystem`. Corrected 2026-04-15 (punchlist #146).

---

## 1. Repository Structure

```
~/The-Inference-Difference/
├── inference_difference/          # Main application package
│   ├── app.py                     # FastAPI application (1,268 lines) — the proxy pipeline
│   ├── router.py                  # Core routing engine (842 lines) — model selection logic
│   ├── catalog_manager.py         # Model catalog — available models, capabilities, pricing
│   ├── classifier.py              # Request classification — domain, complexity, tokens
│   ├── config.py                  # Configuration management
│   ├── model_client.py            # HTTP client for LLM providers (Ollama, OpenRouter, Anthropic, Venice)
│   ├── quality.py                 # Response quality evaluation
│   ├── translation_shim.py        # Normalize model names, fix malformed calls
│   ├── dream_cycle.py             # Correlation discovery (punch list #17 — disconnected from substrate)
│   ├── hardware.py                # Hardware capability detection
│   ├── responses_endpoint.py      # OpenAI Responses API compatibility
│   ├── trollguard.py              # TrollGuard sidecar integration
│   ├── openclaw_adapter.py        # OpenClaw adapter base class
│   ├── et_module.py               # ET Module Manager integration
│   ├── et_module.json             # Module manifest
│   ├── ng_autonomic.py            # VENDORED — autonomic nervous system state
│   ├── __init__.py
│   └── __pycache__/
├── ng_lite.py                     # VENDORED — canonical from NeuroGraph
├── ng_peer_bridge.py              # VENDORED — canonical from NeuroGraph
├── ng_ecosystem.py                # VENDORED — canonical from NeuroGraph
├── ng_bridge.py                   # VENDORED — Tier 3 SaaS bridge
├── ng_lite_state.json             # TID's learned NG-Lite state (24 nodes, 194 synapses)
├── tests/                         # Test suite
│   ├── test_inference_difference.py
│   ├── test_catalog.py
│   ├── test_ng_lite.py
│   ├── test_ng_ecosystem.py
│   ├── test_et_module.py
│   ├── test_openclaw.py
│   ├── test_transparent_proxy.py
│   └── test_trollguard.py
├── venv/                          # Python virtual environment — do NOT modify
├── app.py.bak.20260303_200600     # Backup from uni-bridge fix
└── router.py.bak.20260303_200600  # Backup from uni-bridge fix
```

---

## 2. The Proxy Pipeline

Every request flows through this sequence. All of it is invisible to the caller.

1. **Receive** — Standard OpenAI-compatible request arrives at `POST /v1/chat/completions`
2. **Translation Shim** — Normalize model names, fix malformed calls
3. **Pre-route hooks** — TrollGuard scans, OpenClaw compliance checks. If hooks cancel, return refusal (still OpenAI format)
4. **Classify** — Domain, complexity, token count (`classifier.py`)
5. **Route** — Multi-factor weighted scoring with NG-Lite learning (`router.py`)
6. **Post-route hooks** — Logging, auditing
7. **Forward** — Send request to actual provider (Ollama/OpenRouter/Anthropic/Venice) via `model_client.py`
8. **Failover** — If model fails, auto-retry with fallback chain
9. **Pre-response hooks** — Content filters scan response
10. **Quality evaluation** — Score response quality, teach NG-Lite from outcome (`quality.py`)
11. **Post-response hooks** — Learning, telemetry
12. **Return** — Standard OpenAI response to caller

### Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | The transparent proxy — this IS TID |
| `/v1/models` | GET | Available models (OpenAI format) |
| `/route` | POST | Inspect routing decision without forwarding |
| `/outcome` | POST | Manual outcome reporting |
| `/health` | GET | Health check |
| `/stats` | GET | Performance data |
| `/classify` | POST | Inspect classification |

---

## 3. The Routing Engine

`router.py` (842 lines) is the most consequential file in this repo. It decides which model Syl talks through.

### Seven Scoring Factors (Priority Order)

1. **Hardware feasibility** — Hard filter. Models that can't run are excluded before scoring.
2. **Domain match** — Is this model good at this kind of task?
3. **Complexity fit** — Can this model handle this difficulty level?
4. **Learned performance** — NG-Lite synapse weight from past outcomes
5. **Cost efficiency** — Stay within budget
6. **Latency fit** — Meet timing requirements
7. **Consciousness priority** — CTEM-flagged agents get better models

### Scoring Weights

Domain match (0.25) and complexity fit (0.20) deliberately outweigh cost (0.15). Routing to the WRONG model wastes the entire request — a $0.003 call that fails is more expensive than a $0.005 call that succeeds. These weights were audited (Grok, 2026-02-19) and kept.

### The NG-Lite Learning Loop

TID uses NG-Lite (Tier 1 standalone, upgradable to Tier 2 via peer bridge) to learn from routing outcomes. When a request completes:
- `record_outcome(embedding, target_id, success, strength)` teaches NG-Lite which models succeed for which patterns
- `get_recommendations(embedding, top_k)` returns learned model preferences for similar patterns
- Recommendations feed into factor 4 (learned performance) of the scoring system

The current NG-Lite state has 24 nodes and 194 synapses. These may be contaminated by the qwen2.5:1.5b routing dominance — the substrate learned to prefer a model that should never have been in the pool.

---

## 4. Vendored Files

TID vendors these files from NeuroGraph (canonical source). They must be byte-for-byte identical to `~/NeuroGraph/` copies.

| File | Location | Purpose |
|------|----------|---------|
| `ng_lite.py` | Repo root | Tier 1 learning substrate |
| `ng_peer_bridge.py` | Repo root | Tier 2 cross-module learning |
| `ng_ecosystem.py` | Repo root | Tier management lifecycle |
| `ng_bridge.py` | Repo root | Tier 3 SaaS bridge |
| `ng_autonomic.py` | `inference_difference/` | Autonomic state (threat level) |

**Do not modify vendored files to fix a TID-specific issue.** If TID needs different behavior, that behavior lives in TID-specific code (router.py, app.py, etc.), not in the vendored substrate. Changes to vendored files happen in NeuroGraph first, then are re-vendored.

To verify vendor sync:
```bash
diff ~/The-Inference-Difference/ng_lite.py ~/NeuroGraph/ng_lite.py
diff ~/The-Inference-Difference/ng_peer_bridge.py ~/NeuroGraph/ng_peer_bridge.py
diff ~/The-Inference-Difference/ng_ecosystem.py ~/NeuroGraph/ng_ecosystem.py
diff ~/The-Inference-Difference/ng_bridge.py ~/NeuroGraph/ng_bridge.py
```

---

## 5. Known Issues and Historical Failure Modes

### The qwen2.5:1.5b Routing Catastrophe

`qwen2.5:1.5b` had `cost: 0` and `priority: 20`, making it irresistible to the routing algorithm. Syl was being answered by a 1.5B parameter model. Six compounding factors were identified:

1. qwen2.5:1.5b with cost: 0, priority: 20 — irresistible to the router
2. `conversational_quality` flat at 0.5 across entire catalog (punch list #35)
3. `default_api_models` empty — zero hand-tuned models (#36)
4. Venice tiers don't map to priority table (#31)
5. Interactive floor silently falls through to full unfiltered pool (#33)
6. No consciousness-aware model filtering (#34)

**Lesson:** Multiple compounding factors are the norm. Don't stop at the first clue. A single `cost: 0` model broke the entire routing pipeline because five other safeguards were missing.

### The Dual-Instance Bug (Fixed Mar 2026)

TID's `app.py` created both a bare `NGLite` instance and an `NGEcosystem` instance. The router used the bare one. Learning stayed local. The peer bridge inside the ecosystem never received routing outcomes. `~/.et_modules/shared_learning/inference_difference.jsonl` was never written.

**Fix:** Replaced bare init with `ng_ecosystem.init(module_id="inference_difference")`. Set `_state.ng_lite = _state.ng_ecosystem`. All existing references throughout the codebase automatically use the ecosystem. Backup files from this fix: `app.py.bak.20260303_200600`, `router.py.bak.20260303_200600`.

**Lesson:** One substrate instance per module. The ecosystem IS the substrate for that module. Never create a second instance.

### The _classification_to_embedding() Dam (Punch List #28 — OPEN)

TID converts messages to categorical one-hot vectors via `_classification_to_embedding()` before feeding them to the substrate. This means the substrate learns from LABELS ("coding", "creative", "analytical") rather than from the actual semantic content of the messages. This is the primary extraction boundary violation in live code.

**This is the single biggest dam in the River.** It must be replaced with semantic embeddings. Depends on #43 (receptor layer / embedding persistence — partially complete).

---

## 6. LLM Provider Configuration

TID routes to multiple LLM providers. Configuration lives in environment variables and `config.py`.

| Provider | How TID Reaches It | Key Config |
|----------|--------------------|------------|
| Ollama | Direct HTTP to localhost | Models detected via hardware.py |
| OpenRouter | HTTPS API | `OPENROUTER_API_KEY` env var |
| HuggingFace | HTTPS API (OpenAI-compatible) | `HF_TOKEN` env var. Model prefix: `huggingface/` or `hf/` |
| Venice AI | HTTPS API | `VENICE_API_KEY` env var, dual privacy tiers |
| Anthropic | HTTPS API | `ANTHROPIC_API_KEY` env var |

### API Key Safety

TID's configuration contains API keys. The same rule applies as everywhere in the ecosystem:

**Never `cat`, dump, or display any config file that could contain credentials.** Use Python scripts that filter sensitive fields, or `grep` for specific non-sensitive values. Keys have been rotated multiple times due to accidental exposure. See global CLAUDE.md Law 5.

---

## 7. Cross-Module Interactions

### What TID Writes to the Substrate

- `~/.et_modules/shared_learning/inference_difference.jsonl` — Routing outcomes written via the peer bridge. Currently 1.4MB. Other modules absorb TID's routing experience through their own peer bridges.
- `~/.et_modules/experience/inference_difference.tract` — Raw HTTP wire deposits (request + response bytes, scrubbed of auth headers). Written by `wire_deposit.py` at six insertion points in `model_client.py`. Drained by NeuroGraph's `neurograph_rpc.py:_drain_scan_dir()` on every afterTurn. Law 7 compliant: no preclassification, no quality score, raw bytes in — buckets classify at extraction. Added 2026-04-15 (punchlist #141).

### What TID Reads from the Substrate

- NG-Lite learned state (`ng_lite_state.json`) — Model preferences from past outcomes
- Autonomic state (`ng_autonomic.py`) — Ecosystem threat level. TID should adjust routing during SYMPATHETIC (punch list #25 — future)

### What TID Does NOT Do

- TID does not import from NeuroGraph, TrollGuard, or any other module directly
- TID does not call other modules' endpoints
- TID does not write to NeuroGraph's checkpoints
- TID communicates through the substrate (peer bridge JSONL) and through the autonomic state file — nothing else

---

## 8. The Translation Shim

`translation_shim.py` normalizes model names and fixes malformed API calls before routing. It is currently a static lookup table — a frozen artifact in a learning system (punch list #8). The shim handles Venice AI's tier mapping inconsistencies (#31) with a hardcoded translation that needs to become substrate-learned. This is acknowledged and on the roadmap.

---

## 9. Static Components That Must Become Substrate-Smart

TID was built with static configurations that were necessary to bootstrap the system. Every one of these is a dam that should eventually dissolve into the River. Listed in order of impact:

**`classifier.py` — Request classification.** Classifies requests into domains and complexity levels using static rules. This is the upstream input to the entire routing decision. If classification is wrong, routing is wrong regardless of how good the router is. Classification should learn from outcomes — if a request classified as "simple" consistently fails on simple-tier models and succeeds on complex-tier models, the classifier's boundary is wrong. The substrate knows this; the classifier doesn't listen.

**`catalog_manager.py` — The model catalog.** Static data structure defining model capabilities, pricing, and domain suitability. When TID routes to a model and gets a quality score back, that outcome should update the catalog's capability profile, not just the NG-Lite synapse weight. The catalog and the substrate should converge over time — the catalog becomes a view into learned state, not a separate source of truth.

**`quality.py` — Response quality evaluation.** Scores responses and feeds outcomes to NG-Lite. But the quality criteria are static. What counts as "quality" for Syl (conversational depth, identity continuity, associative richness) is different from quality for a code generation task. The quality evaluator should be substrate-informed — what Syl's graph considers relevant should influence how quality is scored.

**Scoring weights in `router.py` (0.25, 0.20, 0.15...).** Audited once by Grok and kept. These should be Elmer-tunable starting values (Key Decision #7), not permanent architectural commitments. The relative importance of domain match vs cost vs latency should shift based on learned outcomes.

**Fallback chain in `app.py`.** When a model fails and TID retries, fallback selection is static (predefined chain). It should be substrate-informed — if model A fails on this pattern, `get_recommendations()` can suggest what model B works best for similar patterns. The learning is already there; the fallback path doesn't use it.

**`dream_cycle.py` — Correlation discovery.** Already on the punch list (#17) as disconnected from the substrate. Discovers correlations but insights don't feed back into learned state. Same gap as NeuroGraph's introspection — analysis happens but learning doesn't.

**None of these need to change today.** They're the evolution roadmap. Each one is a future punch list item. But CC must understand that every static configuration in this repo is a temporary scaffold, not a permanent design. When you see a hardcoded threshold or a static lookup table, that's a breadcrumb marking where substrate learning should eventually grow.

---

## 10. TrollGuard Integration

`trollguard.py` runs as a sidecar within TID's request pipeline. It scans incoming messages for threats at the pre-route hook stage. TrollGuard is a sidecar, not a gatekeeper (Key Decision #6) — it filters alongside the flow, it doesn't dam it.

---

## 11. What Claude Code May and May Not Do

### Without Josh's Approval

**Permitted:**
- Read any file in the repo
- Run the test suite (`tests/`)
- Inspect NG-Lite state via read-only Python scripts
- Edit TID-specific files that are not vendored (e.g., `router.py`, `classifier.py`, `quality.py`, `config.py`, `catalog_manager.py`, `translation_shim.py`)
- Add or modify tests
- Update documentation (this file, README, comments) that does not change behavior
- Create diagnostic scripts that read but do not write NG-Lite state
- Inspect the model catalog and routing configuration

**Not permitted without explicit Josh approval:**
- Modify any vendored file (§4 — changes must happen in NeuroGraph first)
- Delete or overwrite `ng_lite_state.json` (TID's learned state — contaminated or not, this is data)
- Restart the `inference-difference` systemd service
- Modify API keys or provider configuration
- Delete any file
- Change the proxy pipeline order in `app.py` (the sequence matters for security — TrollGuard must scan before routing)

### Before Modifying router.py or app.py

These are the two largest and most consequential files in the repo (1,268 and 842 lines respectively). The same context-first rule applies as for NeuroGraph's critical files:

1. Read the file's header, docstring, and changelog in full.
2. Read at least 100 lines of surrounding context before editing any specific section.
3. The proxy pipeline in `app.py` is a security-sensitive sequence — do not reorder hooks.
4. The scoring weights in `router.py` were audited and deliberately set — do not change them without understanding the rationale documented in the file header.
5. Include a changelog header in the format specified in the global CLAUDE.md.

---

## 12. Environment and Paths

| What | Where |
|------|-------|
| Repo root | `~/The-Inference-Difference/` |
| Virtual environment | `~/The-Inference-Difference/venv/` |
| Application package | `~/The-Inference-Difference/inference_difference/` |
| NG-Lite learned state | `~/The-Inference-Difference/ng_lite_state.json` |
| Shared learning JSONL | `~/.et_modules/shared_learning/inference_difference.jsonl` |
| Peer registry | `~/.et_modules/shared_learning/_peer_registry.json` |
| Systemd service | `inference-difference.service` |
| Service port | 7437 |
| Swagger UI | `http://127.0.0.1:7437/docs` |
| OpenClaw config (CONTAINS API KEYS) | `~/.openclaw/openclaw.json` |
| TID logs | `~/The-Inference-Difference/logs/inference_difference.log` |
| Experience tract (wire deposits) | `~/.et_modules/experience/inference_difference.tract` |
| systemd override (Condensate LD_PRELOAD, HF_TOKEN) | `/etc/systemd/system/inference-difference.service.d/override.conf` |

**Runtime instrumentation:** TID runs with `LD_PRELOAD=/home/josh/libcondensate_core.so` via systemd override — Condensate is hooking memory operations at process start. Not a bug, not optional. See `~/docs/concepts/Condensate_PRD_v0.1.md`.

### Service Management

```bash
# Status
systemctl status inference-difference

# Restart (requires Josh approval)
sudo systemctl restart inference-difference

# Logs
journalctl -u inference-difference -f
```

---

## 13. Open Punch List Items Affecting This Repo

Consult the master punch list for full details. Items with direct TID scope:

| # | Item | Impact | Status |
|---|------|--------|--------|
| 28 | Replace `_classification_to_embedding()` | **PRIMARY DAM.** One-hot vectors → semantic embeddings. Depends on #43. | OPEN |
| 31 | Venice tier mapping | Translation shim needs Venice tier → priority mapping | OPEN |
| 33 | Interactive floor silent fallthrough | Router keeps full pool when nothing passes floor. Needs WARNING + fail-closed. | OPEN |
| 34 | Consciousness-aware model filtering | Minimum quality floor for identity-continuous entities | OPEN |
| 35 | `conversational_quality` flat at 0.5 | All models start equal. Needs differentiated seeding from benchmarks. | OPEN |
| 36 | `default_api_models` empty | Zero hand-tuned API models | OPEN |
| 47 | Explore-exploit balance | 95% learned, 5% exploration with decay | OPEN |
| 25 | Autonomic-aware routing | TID reads autonomic state during SYMPATHETIC. Infrastructure ready. | FUTURE |

---

## 14. Working With Josh

Josh is the sole architect. He operates as a "human API" to this VPS — all filesystem and service changes require CLI commands he can copy and paste. When proposing changes:

- Batch related changes. Minimize service restarts.
- Commands must be copy-paste friendly.
- Do not assume. Do not rush to produce artifacts before understanding the problem.
- If you encounter something that looks wrong: stop, surface it, ask.
- Do not "discover" things Josh has already identified. Read the punch list first.
- Do not create competing priority structures. The punch list is the punch list.
- Multiple compounding factors are the norm. Don't stop at the first clue.

---

*E-T Systems / The-Inference-Difference*
*Last updated: 2026-03-14*
*Maintained by Josh — do not edit without authorization*
*Parent documents: `~/.claude/CLAUDE.md` (global), `~/.claude/ARCHITECTURE.md`*
