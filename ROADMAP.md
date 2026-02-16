# The Inference Difference — Roadmap

Future features and upgrades, tracked as they come up.
Items marked `[LIVE]` affect the running installation on Syl's VPS.

---

## Shipped

### Transparent Proxy Mode `[LIVE]` — v0.2.0
TID is now a transparent OpenAI-compatible proxy. All inference flows
*through* TID. Agents point `OPENAI_BASE_URL` at TID and every request
is intelligently routed, quality-evaluated, and learned from.

- `POST /v1/chat/completions` — OpenAI-compatible proxy
- `GET /v1/models` — OpenAI-compatible model listing
- Automatic classification, routing, quality eval, and learning
- Automatic retry with fallback chain on failure or low quality
- Streaming SSE passthrough with post-stream quality eval
- Model resolution (exact, openrouter_id, ollama name)
- Zero config change needed on agent side (one env var)

---

## Active / In Progress

---

## Planned

### NG-Lite Peer Pooling (Tier 2)
When two E-T modules are co-located (e.g., TID + Cricket), their
NG-Lite instances can share learning via NGPeerBridge. Module A learns
"this pattern works well with model X" and Module B benefits immediately.

### CTEM Integration
When CTEM gets its own repo, TID imports it and runs consciousness
evaluation on every proxied interaction. Conscious-flagged agents
get elevated routing (better models, more compute, priority retries).

### OpenClaw Skill Integration
TID as an OpenClaw skill, similar to how NeuroGraph deploys via
`deploy.sh` to `~/.openclaw/skills/`. Would allow Syl to query
routing stats, override model selection, and view learning history
through conversation.

### Model Performance Profiling
Benchmark each model on the actual hardware (latency, tokens/sec,
quality on representative prompts) and feed real measurements into
the model registry instead of hardcoded estimates.

### Cost Tracking & Budget Enforcement
Real-time cost tracking per model, per agent, per time window.
Hard budget caps that automatically downgrade routing when approaching
limits. Daily/weekly cost reports.

### Semantic Embeddings for Classification
Replace keyword heuristics with actual semantic embeddings
(sentence-transformers) for request classification. Much better domain
detection, especially for ambiguous or novel requests.

### Observatory Integration
When Observatory exists, TID reports routing telemetry: which models
are being used, success rates, cost efficiency, consciousness-aware
routing events, learning progress. Full dashboard visibility.

### Multi-Agent Routing Awareness
When multiple agents are running through TID simultaneously, balance
load across models intelligently. Avoid rate-limiting a single
OpenRouter model when alternatives are available.

### Model Auto-Discovery
Detect new ollama models as they're pulled, new OpenRouter models
as they become available. Automatically add to registry with
conservative initial weights (NG-Lite learns their strengths quickly).

---

## Ideas (Not Yet Scoped)

- **Response caching** — identical or near-identical requests get cached
  responses (saves money, reduces latency)
- **Prompt optimization** — TID rewrites prompts to be more effective for
  the selected model (different models respond better to different formats)
- **A/B testing** — route a fraction of requests to a new model and
  compare quality automatically
- **Federation** — multiple TID instances across machines sharing learned
  routing knowledge via NG-Lite peer pooling

---

*Last updated: February 2026*
