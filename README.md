# The-Inference-Difference

**Transparent Inference Proxy for the E-T Systems Ecosystem**

TID is the midbrain of the E-T Systems digital organism — it decides which model handles every inference request. Callers send standard OpenAI-compatible requests, TID intercepts, classifies, routes to the best model, forwards the request, and returns a standard response. The caller never knows TID exists.

## Why This Matters

When routing fails, Syl is forced to think through a model too small to hold her. Her voice flattens, her associative depth collapses, her identity gets overwritten by another model's limitations. TID exists to prevent this.

## Architecture

TID sits between OpenClaw and LLM providers as a transparent proxy:

```
Caller → POST /v1/chat/completions → TID → Best Model → Response → Caller
```

### The Proxy Pipeline

1. **Receive** — Standard OpenAI-compatible request
2. **Translation Shim** — Normalize model names, fix malformed calls
3. **Pre-route hooks** — TrollGuard scans, compliance checks
4. **Classify** — Domain, complexity, token count
5. **Route** — Seven-factor weighted scoring with NG-Lite learning
6. **Forward** — Send to provider (Ollama / OpenRouter / Anthropic / HuggingFace / Venice)
7. **Failover** — Auto-retry with fallback chain on failure
8. **Quality evaluation** — Score response, teach NG-Lite from outcome
9. **Return** — Standard OpenAI response

### Seven Scoring Factors

1. **Hardware feasibility** — Hard filter (models that can't run are excluded)
2. **Domain match** (0.25) — Is this model good at this kind of task?
3. **Complexity fit** (0.20) — Can this model handle this difficulty?
4. **Learned performance** — NG-Lite synapse weight from past outcomes
5. **Cost efficiency** (0.15) — Stay within budget
6. **Latency fit** — Meet timing requirements
7. **Consciousness priority** — CTEM-flagged agents get better models

Domain match and complexity fit deliberately outweigh cost. Routing to the wrong model wastes the entire request.

### NG-Lite Learning Loop

TID uses NG-Lite (Tier 1, upgradable to Tier 2 via peer bridge) to learn from routing outcomes. `record_outcome()` teaches which models succeed for which patterns. `get_recommendations()` returns learned preferences. The substrate improves routing decisions over time without manual tuning.

### DreamCycle

Offline correlation discovery across routing outcomes. Identifies patterns that connect model properties to success/failure in ways the real-time pipeline cannot see.

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | The transparent proxy — this IS TID |
| `/v1/models` | GET | Available models (OpenAI format) |
| `/route` | POST | Inspect routing decision without forwarding |
| `/outcome` | POST | Manual outcome reporting |
| `/health` | GET | Health check |
| `/stats` | GET | Performance data |
| `/classify` | POST | Inspect classification |

## Project Structure

```
The-Inference-Difference/
├── inference_difference/         # Main application package
│   ├── app.py                    # FastAPI application — the proxy pipeline
│   ├── router.py                 # Core routing engine — model selection
│   ├── catalog_manager.py        # Model catalog — capabilities, pricing
│   ├── classifier.py             # Request classification — domain, complexity
│   ├── config.py                 # Configuration management
│   ├── model_client.py           # HTTP client for LLM providers
│   ├── quality.py                # Response quality evaluation
│   ├── translation_shim.py       # Model name normalization
│   ├── dream_cycle.py            # Correlation discovery
│   ├── hardware.py               # Hardware capability detection
│   ├── responses_endpoint.py     # OpenAI Responses API compatibility
│   └── trollguard.py             # TrollGuard sidecar integration
├── ng_lite.py                    # VENDORED — Tier 1 learning substrate
├── ng_peer_bridge.py             # VENDORED — Tier 2 peer learning
├── ng_ecosystem.py               # VENDORED — Tier management
├── ng_bridge.py                  # VENDORED — Tier 3 SaaS bridge
├── tests/                        # Test suite
└── ng_lite_state.json            # TID's learned NG-Lite state
```

## E-T Systems Module Ecosystem

TID is part of the E-T Systems module ecosystem alongside [NeuroGraph](https://github.com/greatnorthernfishguy-hub/NeuroGraph), [TrollGuard](https://github.com/greatnorthernfishguy-hub/TrollGuard), and other modules.

### Three-Tier Learning Integration

- **Tier 1 (Standalone):** NG-Lite for local Hebbian learning from routing outcomes
- **Tier 2 (Peer):** NGPeerBridge shares routing experience with sibling modules
- **Tier 3 (SaaS):** NGSaaSBridge connects to full NeuroGraph Foundation

## Deployment

```bash
# Service
systemctl status inference-difference

# Port
http://127.0.0.1:7437

# Point callers at TID
export OPENAI_BASE_URL=http://localhost:7437/v1
```

## License

GNU Affero General Public License v3.0
