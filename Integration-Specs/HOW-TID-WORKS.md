# How TID Actually Works — The Full Mechanical Flow

**Audience:** Syl, or anyone integrating with TID from outside.
**Source of truth:** `inference_difference/app.py`, the FastAPI endpoints.

---

## What TID Is

TID is a **transparent inference proxy**. It looks like an OpenAI API
server. Callers point `OPENAI_BASE_URL` at TID and make normal API calls.
TID intercepts the call, picks the best model, forwards the request to
the actual provider, and returns the response. **The caller never knows
TID exists.**

```bash
# This is the entire integration:
export OPENAI_BASE_URL=http://localhost:4001/v1
# Now every OpenAI-compatible call goes through TID automatically.
```

---

## One Endpoint That Matters

```
POST /v1/chat/completions
```

That's it. Standard OpenAI chat completions. Send your messages, get
back a response. TID handles everything else internally.

---

## The Full Pipeline — What Happens Inside

When a caller sends `POST /v1/chat/completions`:

```json
{
  "model": "auto",
  "messages": [
    {"role": "user", "content": "Write me a Python function that sorts a list"}
  ],
  "temperature": 0.7
}
```

### Step 1: Translation Shim

`translation_shim.py` — Normalizes the model name.

- `"auto"` or `"default"` or `""` → let TID route automatically
- `"gpt-4"` → `"openai/gpt-4o"`
- `"claude"` → `"anthropic/claude-sonnet-4-5-20250929"`
- `"deepseek"` → `"deepseek/deepseek-chat"`
- `"llama"` → `"ollama/llama3.1:8b"`
- Already qualified (`"ollama/deepseek-r1:14b"`) → pass through

If the caller specifies a real model, TID honors it. If they say "auto"
(or anything that resolves to auto), TID picks the best one.

### Step 2: pre_route Hooks (Security + Compliance)

Two modules run, in priority order:

**TrollGuard** (priority 5, runs first):
- Regex-scans the message for prompt injection, jailbreaks, abuse, PII
- If threat_score >= 0.9 → cancels the request (no model called)
- If threat_score >= 0.5 → flags it but allows routing

**OpenClaw Adapter** (priority 10, runs second):
- If connected to the OpenClaw gateway, POSTs a compliance query
- Gateway can allow / flag / deny / escalate
- If deny → cancels the request

### Step 3: Cancellation Check

If any hook cancelled the request, the caller gets a **refusal response**
in standard OpenAI format:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "I'm unable to process this request. Reason: ..."
    }
  }]
}
```

Still a valid OpenAI response. Still 200 OK. The caller's code handles
it normally — it just sees the model refused. No special error handling
needed.

### Step 4: Classify the Request

`classifier.py` — Pure heuristic, <1ms. Determines:
- **domain**: code, reasoning, creative, conversation, analysis, etc.
- **complexity**: trivial → low → medium → high → extreme
- **estimated_tokens**, **context window needs**, **urgency**

### Step 5: Route to Best Model

If the caller said `model: "auto"`, TID picks the model:

**Hard filters** (models that can't work are removed before scoring):
- Local models that don't fit in VRAM/RAM → excluded
- Models with too-small context windows → excluded
- Models that can't handle the domain+complexity → excluded

**Scoring** (six weighted factors on remaining models):

| Factor | Weight | What it measures |
|---|---|---|
| domain_match | 0.25 | Is this model good at this task type? |
| complexity_fit | 0.20 | Can it handle this difficulty level? |
| learned_weight | 0.20 | NG-Lite data from past outcomes |
| cost_efficiency | 0.15 | Cheaper = better |
| latency_fit | 0.10 | Faster = better |
| priority_bonus | 0.10 | Base priority from model config |

Highest score wins. Next 3 become the fallback chain.

### Step 6: post_route Hooks

- TrollGuard logs which model was picked for flagged requests
- OpenClaw validates the decision against governance policies

### Step 7: Forward to Provider

`model_client.py` — TID makes the actual API call:

- `ollama/*` → `http://localhost:11434/v1/chat/completions`
- `openai/*` → `https://api.openai.com/v1/chat/completions`
- `anthropic/*` → `https://api.anthropic.com/v1/messages` (translated to OpenAI format)
- Everything else → `https://openrouter.ai/api/v1/chat/completions`
- If `LITELLM_BASE_URL` is set → all traffic goes through LiteLLM

The caller's original messages, temperature, max_tokens, etc. are
forwarded as-is. TID only changes which model handles it.

### Step 8: Auto-Retry on Failure

If the model fails, TID automatically tries the next model in the
fallback chain. The caller never sees the retry — they just get the
response from whichever model succeeded.

### Step 9: Quality Eval + Learning (Internal)

After getting the response, TID:
1. Runs pre_response hooks (TrollGuard scans for leaked system prompts)
2. Evaluates response quality
3. Teaches NG-Lite: "for code/high requests, this model worked well"
4. Runs post_response hooks (telemetry)

**The caller never sees any of this.** They already have their response.

### Step 10: Return the Response

The caller gets a standard OpenAI chat completion response:

```json
{
  "id": "chatcmpl-tid-1708888000",
  "object": "chat.completion",
  "created": 1708888000,
  "model": "ollama/deepseek-r1:14b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "def sort_list(lst):\n    return sorted(lst)\n"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 22,
    "total_tokens": 37
  },
  "routing_info": {
    "routed_by": "tid",
    "model_selected": "ollama/deepseek-r1:14b",
    "classification": {"domain": "code", "complexity": "high"},
    "quality_score": 0.85
  }
}
```

The `routing_info` field is a TID extension — callers can ignore it.
Everything else is standard OpenAI format.

---

## How OpenClaw Connects

The adapter is **already built and auto-registered** at startup.

**Environment variables:**

| Env Var | Default | What it does |
|---|---|---|
| `OPENCLAW_GATEWAY_PORT` | (empty) | Gateway port. Empty = standalone mode |
| `OPENCLAW_GATEWAY_HOST` | `127.0.0.1` | Gateway host |
| `OPENCLAW_GATEWAY_TOKEN` | (empty) | Bearer token for auth |

```bash
export OPENCLAW_GATEWAY_PORT=18789
export OPENCLAW_GATEWAY_TOKEN=your-token-here
uvicorn inference_difference.app:app --host 0.0.0.0 --port 4001
```

The adapter POSTs compliance queries to `/hooks/agent` on every request.
The gateway responds with allow/flag/deny/escalate. TID enforces it.

**Syl does not need to write a TID plugin.** The adapter exists.

---

## Hook Phases

| Phase | When | Who |
|---|---|---|
| `pre_route` | Before routing | TrollGuard, OpenClaw |
| `post_route` | After routing, before forwarding | TrollGuard, OpenClaw |
| `pre_response` | After model responds | TrollGuard (response scan) |
| `post_response` | After quality eval | Learning, telemetry |

These are the only four. No others exist.

---

## What Models TID Knows About

**Local** (zero cost, need GPU):
- `ollama/llama3.2:3b` — conversation, general. 2GB VRAM.
- `ollama/llama3.1:8b` — code, reasoning, general. 5GB VRAM.
- `ollama/deepseek-r1:14b` — code, reasoning, analysis. 10GB VRAM.
- `ollama/qwen2.5-coder:7b` — code specialist. 5GB VRAM.

**API** (costs money, no hardware req):
- `anthropic/claude-sonnet-4-5-20250929` — everything. $0.003/1k tokens.
- `anthropic/claude-haiku-4-5-20251001` — general, conversation. $0.001/1k.
- `openai/gpt-4o` — everything. $0.005/1k tokens.

---

## Debug/Introspection Endpoints

These exist for development and debugging. Normal callers never use them.

| Endpoint | What |
|---|---|
| `POST /route` | See routing decision without forwarding |
| `POST /outcome` | Manual outcome reporting |
| `POST /classify` | See how TID classifies a message |
| `GET /health` | Liveness |
| `GET /stats` | Performance data |
| `GET /models` | TID model list |
| `GET /modules` | ET module list |

---

## Summary

```
export OPENAI_BASE_URL=http://localhost:4001/v1
```

That's the integration. One env var. Every call goes through TID.
TID picks the model, calls it, returns the response, learns from
the outcome. The caller is oblivious. As designed.
