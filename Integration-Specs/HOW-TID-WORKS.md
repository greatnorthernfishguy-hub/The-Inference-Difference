# How TID Actually Works — The Full Mechanical Flow

**Audience:** Syl, or anyone integrating with TID from outside.
**Source of truth:** `inference_difference/app.py`, the FastAPI endpoints.
Everything below references exact file paths and line numbers.

---

## What TID Is

TID is a FastAPI HTTP service. It has six endpoints:

| Endpoint | Method | What it does |
|---|---|---|
| `/route` | POST | Accept a request, return which model should handle it |
| `/outcome` | POST | Accept outcome data after the caller used the model |
| `/health` | GET | Liveness check |
| `/stats` | GET | Performance and learning data |
| `/models` | GET | List all models TID knows about |
| `/modules` | GET | List registered ET modules (TrollGuard, OpenClaw, etc.) |

TID does **not** call models. It does **not** proxy API requests. It tells
the caller which model to use, the caller calls the model, and the caller
reports back what happened. That's the entire contract.

---

## The /route Pipeline — Step By Step

When something (Syl, Cricket, any HTTP client) sends `POST /route`:

```
POST /route
{
  "message": "Write me a Python function that sorts a list",
  "conversation_history": [],
  "consciousness_score": null,
  "request_id": "req_42",
  "metadata": {}
}
```

Here is exactly what happens inside TID (`app.py:476-567`):

### Step 1: Build Hook Context

`app.py:486-492` — TID creates a `HookContext` object. This is a mutable bag
that gets passed through every hook. Modules read from it and write to it.

```python
ctx = HookContext(
    request_id=req.request_id or "",
    message=req.message,
    conversation_history=req.conversation_history,
    metadata=req.metadata or {},
    consciousness_score=req.consciousness_score,
)
```

### Step 2: Run pre_route Hooks

`app.py:495-496` — TID dispatches the `pre_route` phase to all registered
modules, in priority order (lower number = runs first).

Currently registered modules and their priorities:
- **TrollGuard** (priority 5) — runs first
- **OpenClaw adapter** (priority 10) — runs second

**What TrollGuard does** (`trollguard.py:181-237`):
1. Regex-scans the message for prompt injection, jailbreak, abuse, and PII patterns
2. Also scans conversation_history (discounted by 0.7x)
3. Sets `ctx.flags` — e.g., `"threat_detected"`, `"pii_detected"`
4. Writes a full assessment to `ctx.annotations["trollguard"]`
5. If threat_score >= 0.9 (block_threshold): sets `ctx.cancelled = True`

**What the OpenClaw adapter does** (`openclaw_adapter.py:228-292`):
1. If `OPENCLAW_GATEWAY_PORT` env var is set, POSTs a compliance query to
   `http://{host}:{port}/hooks/agent` with the request details
2. Applies the gateway's response (allow/flag/deny/escalate)
3. Then runs any local compliance policies (always, even when connected)
4. If a DENY policy matches: sets `ctx.cancelled = True`
5. Writes results to `ctx.annotations["openclaw"]`

### Step 3: Check for Cancellation

`app.py:499-509` — If ANY hook set `ctx.cancelled = True`, routing **stops**.
TID returns an empty response immediately:

```json
{
  "model_id": "",
  "score": 0.0,
  "reasoning": "Request cancelled: TrollGuard: high-confidence threat detected ...",
  "fallback_chain": [],
  ...
}
```

The caller gets no model. The request is dead. This is enforcement.

### Step 4: Classify the Request

`app.py:518-524` — TID runs the classifier (`classifier.py:135-231`).
Pure heuristic, <1ms. No ML model involved.

It determines:
- **primary_domain**: code, reasoning, creative, conversation, analysis,
  summarization, translation, or general
- **complexity**: trivial, low, medium, high, or extreme
- **estimated_tokens**: rough output token count
- **requires_context_window**: minimum context window needed
- **is_time_sensitive**: whether urgency keywords were detected
- **confidence**: how sure the classifier is (0.0-1.0)

### Step 5: Route to Best Model

`app.py:527-531` — The routing engine scores every eligible model.

**Hard filters first** (`router.py:458-494`):
- Local models that won't fit in available VRAM/RAM are excluded
- Models with too-small context windows are excluded
- Models that can't handle the domain+complexity combo are excluded
- These models **never reach scoring**

**Scoring** (`router.py:500-554`) — six weighted factors:

| Factor | Weight | What it measures |
|---|---|---|
| domain_match | 0.25 | Is this model good at this task type? |
| complexity_fit | 0.20 | Can it handle this difficulty level? |
| learned_weight | 0.20 | NG-Lite data from past outcomes |
| cost_efficiency | 0.15 | Cheaper is better (inversely proportional) |
| latency_fit | 0.10 | Faster is better |
| priority_bonus | 0.10 | Base priority from model config |

If a consciousness_score was provided and is > 0.5, higher-capability models
get a small boost (up to +0.1).

Highest-scoring model wins. Next 3 become the fallback_chain.

### Step 6: Run post_route Hooks

`app.py:535-536` — TID dispatches `post_route` to all subscribed modules.

- **TrollGuard** logs which model was picked for threat-flagged requests
- **OpenClaw** validates the routing decision against governance policies
  (approved model lists, require_local policies, etc.)

### Step 7: Return the Decision

`app.py:550-567` — TID returns:

```json
{
  "model_id": "ollama/deepseek-r1:14b",
  "score": 0.847,
  "score_breakdown": {
    "domain_match": 1.0,
    "complexity_fit": 0.85,
    "learned_weight": 0.65,
    "cost_efficiency": 1.0,
    "latency_fit": 0.7,
    "priority_bonus": 0.3
  },
  "reasoning": "Selected DeepSeek R1 14B for code/high request. Top factors: domain_match=1.00, complexity_fit=0.85, latency_fit=0.70. Running locally (zero cost, lower latency).",
  "fallback_chain": ["ollama/qwen2.5-coder:7b", "ollama/llama3.1:8b"],
  "request_id": "req_42",
  "classification": {
    "primary_domain": "code",
    "complexity": "high",
    "estimated_tokens": 250,
    "confidence": 0.89,
    "is_time_sensitive": false
  },
  "consciousness_boost": false
}
```

**The caller now calls that model.** TID doesn't do it. TID chose it.

---

## The /outcome Pipeline — Closing the Loop

After the caller uses the model and gets a response, it reports back:

```
POST /outcome
{
  "request_id": "req_42",
  "model_id": "ollama/deepseek-r1:14b",
  "response_text": "def sort_list(lst): ...",
  "success": true,
  "latency_ms": 1200.0
}
```

Here's what happens (`app.py:570-630`):

### Step 8: Run pre_response Hooks

`app.py:592-593` — Content filter hooks run here. TrollGuard scans the
model's response for system prompt leakage, instruction leaks, and PII
(`trollguard.py:255-288`).

### Step 9: Evaluate Quality

`app.py:596-603` — TID evaluates the response quality (length, coherence,
relevance signals). Returns a quality score.

### Step 10: Teach NG-Lite

`app.py:610-618` — If the original routing decision is found, TID feeds the
outcome to NG-Lite. NG-Lite learns: "for code/high requests, deepseek-r1:14b
succeeded with quality 0.85." Next time a similar request comes in, the
`learned_weight` score for that model goes up.

### Step 11: Run post_response Hooks

`app.py:621-622` — Learning and telemetry hooks. This is where modules
record long-term patterns.

### Step 12: Return Outcome

```json
{
  "request_id": "req_42",
  "quality_score": 0.85,
  "is_success": true,
  "issues": [],
  "learned": true
}
```

---

## How OpenClaw Connects

The OpenClaw adapter (`openclaw_adapter.py`) is **already built and
auto-registered** at TID startup (`app.py:312-346`).

**Connection configuration** (environment variables):

| Env Var | Default | What it does |
|---|---|---|
| `OPENCLAW_GATEWAY_PORT` | (empty) | Port of the OpenClaw gateway. If empty, adapter runs in standalone/passthrough mode |
| `OPENCLAW_GATEWAY_HOST` | `127.0.0.1` | Gateway host address |
| `OPENCLAW_GATEWAY_TOKEN` | (empty) | Bearer token for gateway auth |

**When connected**, the adapter:
1. Probes the gateway with HTTP GET on startup to verify liveness
2. On every `pre_route`, POSTs to `/hooks/agent` with the request details
3. On every `post_route`, POSTs to `/hooks/agent` with the routing decision
4. If the gateway goes down, falls back to standalone mode and re-probes
   on the next request

**When standalone** (no port configured), the adapter runs local policies
only. By default there are no local policies, so it's a passthrough.
Requests flow through unimpeded.

**Fail-open vs fail-closed**: Default is fail-open (if gateway is
unreachable, allow the request). Safety-critical deployments can set
`fail_open=False` in the adapter constructor to block requests when
the gateway is down.

**Syl does not need to write a TID plugin.** The adapter exists. To
connect OpenClaw to TID:

```bash
export OPENCLAW_GATEWAY_PORT=18789
export OPENCLAW_GATEWAY_TOKEN=your-token-here
# Start TID — adapter auto-connects
uvicorn inference_difference.app:app --host 0.0.0.0 --port 8000
```

---

## Hook Names (The Actual Ones)

TID has exactly four hook phases (`et_module.py:70-75`):

| Phase | When it runs | Who uses it |
|---|---|---|
| `pre_route` | Before classification and routing | TrollGuard (threat scan), OpenClaw (compliance check) |
| `post_route` | After routing decision, before returning to caller | TrollGuard (security logging), OpenClaw (governance validation) |
| `pre_response` | After caller reports outcome, before quality eval | TrollGuard (response leakage scan) |
| `post_response` | After quality eval | Learning hooks, telemetry |

There is no `before_model_resolve` hook. There is no `after_model_resolve`
hook. The four above are the complete set.

---

## What Models TID Knows About

At startup, TID loads two sets of models (`app.py:210`):

**Local models** (require GPU/RAM, zero cost):
- `ollama/llama3.2:3b` — conversation, general. 2GB VRAM. Trivial-low tasks.
- `ollama/llama3.1:8b` — code, reasoning, conversation, general. 5GB VRAM. Up to medium.
- `ollama/deepseek-r1:14b` — code, reasoning, analysis. 10GB VRAM. Up to high.
- `ollama/qwen2.5-coder:7b` — code specialist. 5GB VRAM. Up to medium.

**API models** (no hardware req, costs money):
- `anthropic/claude-sonnet-4-5-20250929` — everything. Up to extreme. $0.003/1k tokens.
- `anthropic/claude-haiku-4-5-20251001` — general, conversation, summarization. Up to medium. $0.001/1k.
- `openai/gpt-4o` — everything. Up to high. $0.005/1k tokens.

If TID can't find your GPU, local models are excluded by the hardware
filter, and it routes to API models.

---

## Summary for Integration

To use TID from any client:

1. `POST /route` with your message. Get back a `model_id`.
2. Call that model yourself (via OpenRouter, direct API, ollama, whatever).
3. `POST /outcome` with the response. TID learns.

That's it. Three HTTP calls. TID picks the model. You call it. You tell
TID how it went. TID gets smarter over time.
