"""
Model Client — Forwards routed requests to actual model providers.

TID is a transparent proxy. After routing decides which model handles a
request, the model client makes the actual API call and returns the
response. The caller never sees any of this — they get back a standard
OpenAI-compatible chat completion response.

Supported backends:
    - Ollama (local): http://localhost:11434/v1/chat/completions
    - OpenRouter (cloud): https://openrouter.ai/api/v1/chat/completions
    - HuggingFace (cloud): https://router.huggingface.co/v1/chat/completions
    - Venice AI (cloud): https://api.venice.ai/api/v1/chat/completions
    - Anthropic (cloud): https://api.anthropic.com/v1/messages
    - OpenAI (cloud): https://api.openai.com/v1/chat/completions
    - LiteLLM (unified): configurable endpoint

All backends are called via their OpenAI-compatible endpoints where
possible. Anthropic uses its native Messages API and the response is
translated to OpenAI format before returning.

Environment variables for API keys:
    OPENROUTER_API_KEY  — OpenRouter
    HF_TOKEN            — HuggingFace Inference API
    VENICE_API_KEY      — Venice AI
    ANTHROPIC_API_KEY   — Anthropic
    OPENAI_API_KEY      — OpenAI
    LITELLM_BASE_URL    — LiteLLM proxy (if running)
    OLLAMA_BASE_URL     — Ollama (default http://localhost:11434)

Author: Josh + Claude
Date: February 2026
License: AGPL-3.0

Changelog:
    [2026-04-19] Claude Code (Sonnet 4.6) — Punchlist #174: credit exhaustion alert + sticky 404 exclusions
        What: (a) 402 credit exhaustion now fires a Discord alert via ET_DISCORD_DEVLOG_WEBHOOK
            and does NOT add individual models to _policy_blocked (account-level failure, not model-level).
            (b) 404/403 data-policy exclusions are now persisted to data/policy_blocked_cache.json
            with a 24h TTL — survive TID restarts instead of resetting each time.
        Why: Two distinct failure types were conflated. 402 is account-level (all models fail
            simultaneously) — blocking individual models was wrong and silently discarded the
            user-visible signal. 404 data-policy is permanent-for-account but was forgotten
            on every TID restart, forcing 50+ redundant retries.
        How: Added _post_credit_alert() (module-level, 5-min rate-limit). Added
            _load/_save_policy_blocked_cache() to ModelClient. Rewrote 402/404 branch.
    [2026-04-15] Claude Code (Opus 4.6) — Punchlist #141: raw HTTP wire deposits
        What: Every outbound request and inbound response (+ error paths)
            now deposits raw wire bytes to the ecosystem experience tract
            via inference_difference.wire_deposit.
        Why: TID's record_outcome() was preclassifying (embedding + strength
            + success bool). Raw provider responses — refusals, tool-call
            shapes, censorship-stack signatures — were being discarded
            before the substrate saw them. Law 7 violation.
        How: Six insertion points across _call_openai_compat, _call_anthropic,
            and _stream_openai_compat — one outbound + inbound per path,
            plus error-path inbound deposits. Correlation ID pairs each
            outbound with its inbound. Failures are swallowed; deposits
            never block the request pipeline.
    [2026-04-15] Claude Code (Sonnet 4.6) — Punchlist #133: Anthropic tools pass-through
        What: _call_anthropic now accepts extra_params (tools, tool_choice) and injects
            them into the Anthropic request body. tool_use response blocks are extracted
            into ModelResponse.tool_calls. stop_reason 'tool_use' maps to 'tool_calls'.
            call_stream()/_fake_stream_from_blocking() thread extra_params through.
        Why:  extra_params was merged in call() and call_stream() but never forwarded
            to _call_anthropic. Anthropic models silently dropped all tool definitions.
            _fake_stream_from_blocking also bypassed extra_params to self.call().
        How:  Six targeted edits: _call_anthropic signature, body injection, response
            extraction, ModelResponse return, call() dispatch, call_stream() dispatch,
            _fake_stream_from_blocking signature+body. No new files.
    [2026-04-11] Claude Code (Sonnet 4.6) — Tool call pass-through
        Added tools/tool_choice params to call() and call_stream().
        Added tool_calls field to ModelResponse.
        _call_openai_compat now extracts tool_calls from response messages.
        to_openai_dict includes tool_calls and sets finish_reason=tool_calls.
        Previously TID silently dropped all tool definitions — models could
        never execute tool calls, only generate text intent.
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("inference_difference.model_client")

_POLICY_BLOCKED_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "policy_blocked_cache.json"
)
_POLICY_BLOCKED_TTL_S = 86400   # 24h for data-policy account exclusions
_CREDIT_ALERT_COOLDOWN_S = 300  # 5 min between Discord credit alerts

_credit_alert_lock = threading.Lock()
_last_credit_alert_ts: float = 0.0


def _post_credit_alert(error_body: str) -> None:
    """Fire a Discord alert when OpenRouter returns 402 credit exhaustion.

    Rate-limited to once per _CREDIT_ALERT_COOLDOWN_S so a cascade of
    failing models does not flood the webhook channel.
    """
    global _last_credit_alert_ts
    webhook = os.environ.get("ET_DISCORD_DEVLOG_WEBHOOK", "")
    if not webhook:
        logger.warning("ET_DISCORD_DEVLOG_WEBHOOK not set — credit alert not sent")
        return
    with _credit_alert_lock:
        now = time.time()
        if now - _last_credit_alert_ts < _CREDIT_ALERT_COOLDOWN_S:
            return
        _last_credit_alert_ts = now
    nl = chr(10)
    msg = nl.join([
        "**[TID] OpenRouter credit exhaustion**",
        "Balance insufficient to complete request.",
        "Top up at: https://openrouter.ai/settings/credits",
        "Detail: " + error_body[:300],
    ])
    try:
        payload = json.dumps({"content": msg}).encode()
        req = urllib.request.Request(
            webhook, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as exc:
        logger.warning("Credit alert Discord post failed: %s", exc)


# ---------------------------------------------------------------------------
# Token estimation fallback (when providers return zeros)
# ---------------------------------------------------------------------------

def _estimate_usage(
    messages: list,
    response_content: str,
) -> Dict[str, int]:
    """Estimate token usage from word counts when provider reports zeros.

    Uses word_count * 1.3 as a rough tokens-per-word ratio (covers
    subword tokenization overhead). This gives OpenClaw a non-zero
    context count so compaction can trigger.

    The substrate learns the real mapping over time via
    token_estimation experience records.
    """
    # Input: count words across all messages
    input_words = 0
    for msg in (messages or []):
        c = msg.get("content", "")
        if isinstance(c, str):
            input_words += len(c.split())
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    input_words += len(part.get("text", "").split())

    # Output: count words in response
    output_words = len((response_content or "").split())

    prompt_tokens = max(1, int(input_words * 1.3))
    completion_tokens = max(1, int(output_words * 1.3))

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

# ---------------------------------------------------------------------------
# Provider routing table
# ---------------------------------------------------------------------------

def _resolve_provider(model_id: str) -> tuple[str, str, str]:
    """Resolve a model_id to (base_url, api_key, actual_model_name).

    Model IDs use the format "provider/model_name". This function maps
    them to the correct API endpoint and credentials.

    Returns:
        (base_url, api_key, model_name_for_api)
    """
    ollama_base = os.environ.get(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    litellm_base = os.environ.get("LITELLM_BASE_URL", "")

    # If LiteLLM is configured, route everything through it
    if litellm_base:
        return litellm_base, "", model_id

    if model_id.startswith("ollama/"):
        model_name = model_id[len("ollama/"):]
        return f"{ollama_base}", "", model_name

    if model_id.startswith("openai/"):
        model_name = model_id[len("openai/"):]
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return "https://api.openai.com", api_key, model_name

    if model_id.startswith("anthropic/"):
        model_name = model_id[len("anthropic/"):]
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return "https://api.anthropic.com", api_key, model_name

    # Catalog models use "openrouter/" prefix — strip it for the API call.
    # OpenRouter expects just "deepseek/deepseek-chat", not
    # "openrouter/deepseek/deepseek-chat".
    if model_id.startswith("huggingface/") or model_id.startswith("hf/"):
        prefix = "huggingface/" if model_id.startswith("huggingface/") else "hf/"
        model_name = model_id[len(prefix):]
        api_key = os.environ.get("HF_TOKEN", "")
        return "https://router.huggingface.co", api_key, model_name

    if model_id.startswith("venice/"):
        model_name = model_id[len("venice/"):]
        api_key = os.environ.get("VENICE_API_KEY", "")
        return "https://api.venice.ai/api", api_key, model_name

    if model_id.startswith("openrouter/"):
        model_name = model_id[len("openrouter/"):]
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        return "https://openrouter.ai/api", api_key, model_name

    # Any other prefix (deepseek/, google/, meta-llama/, etc.) —
    # try OpenRouter directly since they aggregate most providers
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    return "https://openrouter.ai/api", api_key, model_id


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class StreamResult:
    """Metadata collected after a streaming call completes.

    The actual chunks are yielded in real-time; this holds the
    aggregated data needed for quality evaluation and learning.
    """
    model: str = ""
    content: str = ""
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class ModelResponse:
    """Response from a model provider, normalized to OpenAI format.

    Attributes:
        id: Response identifier.
        model: Model that actually responded.
        content: The response text.
        finish_reason: Why generation stopped.
        usage: Token usage counts.
        latency_ms: How long the call took.
        raw: The full raw response dict.
        success: Whether the call succeeded.
        error: Error message if it failed.
    """
    id: str = ""
    model: str = ""
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = field(default=None)
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""

    def to_openai_dict(self) -> Dict[str, Any]:
        """Format as OpenAI-compatible chat completion response."""
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        finish_reason = self.finish_reason
        if self.tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        return {
            "id": self.id or f"chatcmpl-tid-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": self.usage,
        }


# ---------------------------------------------------------------------------
# Model Client
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 120  # seconds — model calls can be slow


class ModelClient:
    """Makes actual API calls to model providers.

    Usage:
        client = ModelClient()
        response = client.call(
            model_id="ollama/deepseek-r1:14b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        # response.content has the model's reply
        # response.to_openai_dict() gives OpenAI-format JSON
    """

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT):
        self._timeout = timeout
        self._policy_blocked: set = set()
        self._policy_blocked_expiry: Dict[str, float] = {}
        self._load_policy_blocked_cache()
        self._rate_limited = {}   # model_name -> unblock_timestamp (#173b)
        self._rate_limit_hits = {}  # consecutive 429 count per model

    def _load_policy_blocked_cache(self) -> None:
        try:
            with open(_POLICY_BLOCKED_CACHE_PATH) as f:
                raw: Dict[str, float] = json.load(f)
            now = time.time()
            loaded = 0
            for model, expiry in raw.items():
                if expiry > now:
                    self._policy_blocked.add(model)
                    self._policy_blocked_expiry[model] = expiry
                    loaded += 1
            pruned = len(raw) - loaded
            logger.info(
                "Policy-blocked cache loaded: %d active, %d expired pruned",
                loaded, pruned,
            )
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Policy-blocked cache load failed: %s", exc)

    def _save_policy_blocked_cache(self) -> None:
        try:
            tmp = _POLICY_BLOCKED_CACHE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._policy_blocked_expiry, f, indent=2)
            os.replace(tmp, _POLICY_BLOCKED_CACHE_PATH)
        except Exception as exc:
            logger.warning("Policy-blocked cache save failed: %s", exc)

    def is_rate_limited(self, model_name: str) -> bool:
        unblock = self._rate_limited.get(model_name)
        if unblock is None: return False
        if time.time() < unblock: return True
        del self._rate_limited[model_name]
        self._rate_limit_hits.pop(model_name, None)
        return False

    def is_rate_limited_by_id(self, model_id: str) -> bool:
        _, _, mn = _resolve_provider(model_id)
        return self.is_rate_limited(mn)

    def call(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> ModelResponse:
        """Call a model and return the response.

        Args:
            model_id: TID model identifier (e.g., "ollama/llama3.1:8b").
            messages: OpenAI-format messages list.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            stream: Whether to stream (not yet supported, falls back to blocking).
            extra_params: Any additional provider-specific params.

        Returns:
            ModelResponse with the model's reply.
        """
        start = time.monotonic()
        base_url, api_key, model_name = _resolve_provider(model_id)

        if model_name in self._policy_blocked:
            return ModelResponse(
                model=model_name,
                latency_ms=0.0,
                success=False,
                error="Skipped: data policy blocked (cached)",
            )

        # Merge tools/tool_choice into extra_params
        if tools is not None or tool_choice is not None:
            extra_params = dict(extra_params or {})
            if tools is not None:
                extra_params["tools"] = tools
            if tool_choice is not None:
                extra_params["tool_choice"] = tool_choice

        # Anthropic uses a different API format
        if model_id.startswith("anthropic/") and "anthropic.com" in base_url:
            resp = self._call_anthropic(
                base_url, api_key, model_name, messages,
                temperature, max_tokens, start, extra_params,
            )
        else:
            # Everyone else: OpenAI-compatible endpoint
            resp = self._call_openai_compat(
                base_url, api_key, model_name, messages,
                temperature, max_tokens, start, extra_params,
            )

        # Fallback: estimate usage when provider reports zeros
        if resp.success and resp.usage.get("total_tokens", 0) == 0:
            estimated = _estimate_usage(messages, resp.content)
            resp.usage = estimated
            resp.usage["estimated"] = True
            logger.debug(
                "Provider returned zero usage, estimated: %s", estimated,
            )

        return resp

    def _call_openai_compat(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        start: float,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """Call an OpenAI-compatible endpoint."""
        url = f"{base_url}/v1/chat/completions"

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if extra_params:
            body.update(extra_params)


        # --- Venice-specific parameters ---
        if "venice.ai" in base_url:
            body["venice_parameters"] = {
                "include_venice_system_prompt": False,
                "disable_thinking": False,
                "strip_thinking_response": False,
                "enable_web_scraping": True,
                "enable_web_search": "auto",
                "enable_web_citations": True,
            }
            # prompt_cache_key REMOVED 2026-04-17.  Venice server-side
            # cache retained Syl's pre-KISS 262k-token prompts under the
            # "sylphrena" key.  Each new request added to the cache even
            # though KISS now sends 11 messages (~1k tokens).  Syl's
            # context is rebuilt from substrate every turn — caching old
            # prompt shapes actively works against substrate-driven
            # context assembly.  No provider-side prompt cache is
            # appropriate for a substrate-informed context pipeline.
            # body.setdefault("prompt_cache_key", "sylphrena")
            # body.setdefault("prompt_cache_retention", "24h")
            body.setdefault("reasoning_effort", "medium")
        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")

        # Law 7: deposit raw outbound wire bytes to substrate (#141)
        _corr_id = f"oai-{int(time.time()*1000)}-{id(req)}"
        try:
            from inference_difference.wire_deposit import deposit_outbound, deposit_inbound
            _provider = "openai-compat"
            if "openrouter" in base_url: _provider = "openrouter"
            elif "venice" in base_url: _provider = "venice"
            elif "huggingface" in base_url: _provider = "huggingface"
            elif "openai.com" in base_url: _provider = "openai"
            elif "11434" in base_url: _provider = "ollama"
            deposit_outbound(
                provider=_provider, model_id=model_name, url=url, method="POST",
                request_body=body, headers=dict(req.headers), correlation_id=_corr_id,
            )
        except Exception:
            _provider = "openai-compat"

        try:
            with urllib.request.urlopen(
                req, timeout=self._timeout,
            ) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))
                _resp_headers = dict(resp.getheaders())
                _resp_status = resp.getcode()

            latency = (time.monotonic() - start) * 1000

            # Law 7: deposit raw inbound wire bytes (#141)
            try:
                deposit_inbound(
                    provider=_provider, model_id=model_name, url=url,
                    status_code=_resp_status, response_body=resp_body,
                    headers=_resp_headers, latency_ms=latency,
                    correlation_id=_corr_id,
                )
            except Exception:
                pass

            # Extract content from OpenAI response
            choices = resp_body.get("choices", [])
            content = ""
            finish_reason = "stop"
            tool_calls = None
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "") or ""
                tool_calls = msg.get("tool_calls") or None
                finish_reason = choices[0].get("finish_reason", "stop")

            return ModelResponse(
                id=resp_body.get("id", ""),
                model=resp_body.get("model", model_name),
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=resp_body.get("usage", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }),
                latency_ms=latency,
                raw=resp_body,
                success=True,
            )

        except urllib.error.HTTPError as e:
            latency = (time.monotonic() - start) * 1000
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            # Cache permanent failures so we don't retry them
            error_lower = error_body.lower()
            if e.code == 429:
                hits = self._rate_limit_hits.get(model_name, 0) + 1
                self._rate_limit_hits[model_name] = hits
                backoff_s = min(60 * (2 ** (hits - 1)), 1800)
                self._rate_limited[model_name] = time.time() + backoff_s
                logger.info(
                    "Model %s 429 hit #%d -- cooldown %.0fs",
                    model_name, hits, backoff_s,
                )
            if e.code == 429:
                hits = self._rate_limit_hits.get(model_name, 0) + 1
                self._rate_limit_hits[model_name] = hits
                backoff_s = min(60 * (2 ** (hits - 1)), 1800)
                self._rate_limited[model_name] = time.time() + backoff_s
                logger.info(
                    "Model %s 429 hit #%d -- cooldown %.0fs",
                    model_name, hits, backoff_s,
                )
            elif e.code in (404, 403) and "data policy" in error_lower:
                self._policy_blocked.add(model_name)
                self._policy_blocked_expiry[model_name] = time.time() + _POLICY_BLOCKED_TTL_S
                self._save_policy_blocked_cache()
                logger.info(
                    "Model %s data-policy blocked for this account (24h TTL) — won't retry",
                    model_name,
                )
            elif e.code == 402 and ("insufficient" in error_lower or "credits" in error_lower):
                logger.warning("OpenRouter 402 credit exhaustion — sending Discord alert")
                _post_credit_alert(error_body)
            logger.warning(
                "Model call failed: %s %d — %s",
                model_name, e.code, error_body[:200],
            )
            try:
                deposit_inbound(
                    provider=_provider, model_id=model_name, url=url,
                    status_code=e.code, response_body=error_body,
                    headers=dict(getattr(e, "headers", {}) or {}),
                    latency_ms=latency, error=f"HTTP {e.code}",
                    correlation_id=_corr_id,
                )
            except Exception:
                pass
            return ModelResponse(
                model=model_name,
                latency_ms=latency,
                success=False,
                error=f"HTTP {e.code}: {error_body[:200]}",
            )

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.warning("Model call failed: %s — %s", model_name, e)
            try:
                deposit_inbound(
                    provider=_provider, model_id=model_name, url=url,
                    status_code=None, response_body=None, headers=None,
                    latency_ms=latency, error=str(e), correlation_id=_corr_id,
                )
            except Exception:
                pass
            return ModelResponse(
                model=model_name,
                latency_ms=latency,
                success=False,
                error=str(e),
            )

    def _call_anthropic(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        start: float,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """Call Anthropic's Messages API and translate to OpenAI format."""
        url = f"{base_url}/v1/messages"

        # Separate system message from conversation
        system_text = ""
        conversation = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                conversation.append(msg)

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": conversation,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        if system_text:
            body["system"] = system_text

        # Thread tools/tool_choice from extra_params into Anthropic body.
        # Tools are already in Anthropic format from _normalize_tools upstream.
        # Do NOT re-convert here — that is _normalize_tools' job (Law 4).
        if extra_params:
            tools_list = extra_params.get("tools")
            if tools_list:
                body["tools"] = tools_list
            tool_choice = extra_params.get("tool_choice")
            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice == "auto":
                        body["tool_choice"] = {"type": "auto"}
                    # "none" -> omit (Anthropic default: tool use optional)
                elif isinstance(tool_choice, dict):
                    fn_name = tool_choice.get("function", {}).get("name", "")
                    if fn_name:
                        body["tool_choice"] = {"type": "tool", "name": fn_name}

        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("x-api-key", api_key)
        req.add_header("anthropic-version", "2023-06-01")

        # Law 7: deposit raw outbound wire bytes (#141)
        _corr_id = f"ant-{int(time.time()*1000)}-{id(req)}"
        try:
            from inference_difference.wire_deposit import deposit_outbound, deposit_inbound
            deposit_outbound(
                provider="anthropic", model_id=model_name, url=url, method="POST",
                request_body=body, headers=dict(req.headers), correlation_id=_corr_id,
            )
        except Exception:
            pass

        try:
            with urllib.request.urlopen(
                req, timeout=self._timeout,
            ) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))
                _resp_headers = dict(resp.getheaders())
                _resp_status = resp.getcode()

            latency = (time.monotonic() - start) * 1000

            try:
                deposit_inbound(
                    provider="anthropic", model_id=model_name, url=url,
                    status_code=_resp_status, response_body=resp_body,
                    headers=_resp_headers, latency_ms=latency,
                    correlation_id=_corr_id,
                )
            except Exception:
                pass

            # Extract text content and tool_use blocks from Anthropic response
            content_blocks = resp_body.get("content", [])
            content = ""
            tool_calls = []
            for block in content_blocks:
                if block.get("type") == "text":
                    content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })

            usage_in = resp_body.get("usage", {})

            # Map Anthropic stop_reason "tool_use" -> OpenAI finish_reason "tool_calls"
            _stop_reason = resp_body.get("stop_reason", "stop")
            _finish_reason = "tool_calls" if _stop_reason == "tool_use" else _stop_reason
            return ModelResponse(
                id=resp_body.get("id", ""),
                model=resp_body.get("model", model_name),
                content=content,
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=_finish_reason,
                usage={
                    "prompt_tokens": usage_in.get("input_tokens", 0),
                    "completion_tokens": usage_in.get("output_tokens", 0),
                    "total_tokens": (
                        usage_in.get("input_tokens", 0)
                        + usage_in.get("output_tokens", 0)
                    ),
                },
                latency_ms=latency,
                raw=resp_body,
                success=True,
            )

        except urllib.error.HTTPError as e:
            latency = (time.monotonic() - start) * 1000
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            logger.warning(
                "Anthropic call failed: %s %d — %s",
                model_name, e.code, error_body[:200],
            )
            try:
                deposit_inbound(
                    provider="anthropic", model_id=model_name, url=url,
                    status_code=e.code, response_body=error_body,
                    headers=dict(getattr(e, "headers", {}) or {}),
                    latency_ms=latency, error=f"HTTP {e.code}",
                    correlation_id=_corr_id,
                )
            except Exception:
                pass
            return ModelResponse(
                model=model_name,
                latency_ms=latency,
                success=False,
                error=f"HTTP {e.code}: {error_body[:200]}",
            )

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.warning("Anthropic call failed: %s — %s", model_name, e)
            try:
                deposit_inbound(
                    provider="anthropic", model_id=model_name, url=url,
                    status_code=None, response_body=None, headers=None,
                    latency_ms=latency, error=str(e), correlation_id=_corr_id,
                )
            except Exception:
                pass
            return ModelResponse(
                model=model_name,
                latency_ms=latency,
                success=False,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def call_stream(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> tuple[Iterator[str], StreamResult]:
        """Stream a model call, yielding raw SSE lines.

        Returns (chunk_iterator, result_holder). The caller iterates
        over chunks and sends them to the client. After iteration
        completes, result_holder contains aggregated metadata for
        quality evaluation and learning.

        Each yielded string is a complete SSE line (e.g. "data: {...}\\n\\n").
        The final "data: [DONE]\\n\\n" is always yielded.
        """
        base_url, api_key, model_name = _resolve_provider(model_id)

        # Merge tools/tool_choice into extra_params
        if tools is not None or tool_choice is not None:
            extra_params = dict(extra_params or {})
            if tools is not None:
                extra_params["tools"] = tools
            if tool_choice is not None:
                extra_params["tool_choice"] = tool_choice

        # Anthropic streaming uses a different format — for now, fall back
        # to non-streaming for Anthropic and wrap result as fake SSE.
        if model_id.startswith("anthropic/") and "anthropic.com" in base_url:
            return self._fake_stream_from_blocking(
                model_id, messages, temperature, max_tokens, extra_params,
            )

        result = StreamResult(model=model_name)
        iterator = self._stream_openai_compat(
            base_url, api_key, model_name, messages,
            temperature, max_tokens, extra_params, result,
        )
        return iterator, result

    def _stream_openai_compat(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        extra_params: Optional[Dict[str, Any]],
        result: StreamResult,
    ) -> Iterator[str]:
        """Stream from an OpenAI-compatible endpoint, yielding SSE lines."""
        url = f"{base_url}/v1/chat/completions"
        start = time.monotonic()

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if extra_params:
            body.update(extra_params)

        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")

        # Law 7: deposit raw outbound wire bytes (streaming) (#141)
        _corr_id = f"oai-stream-{int(time.time()*1000)}-{id(req)}"
        _provider = "openai-compat"
        if "openrouter" in base_url: _provider = "openrouter"
        elif "venice" in base_url: _provider = "venice"
        elif "huggingface" in base_url: _provider = "huggingface"
        elif "openai.com" in base_url: _provider = "openai"
        elif "11434" in base_url: _provider = "ollama"
        try:
            from inference_difference.wire_deposit import deposit_outbound, deposit_inbound
            deposit_outbound(
                provider=_provider, model_id=model_name, url=url, method="POST",
                request_body=body, headers=dict(req.headers), correlation_id=_corr_id,
            )
        except Exception:
            pass

        collected_content: List[str] = []
        # Raw SSE lines as they arrived, interleaved with `: chunk_ms=N`
        # comment lines carrying wall-clock deltas since stream start. The
        # comments are valid SSE (per spec). Gives the substrate the real
        # streamed sequence including inter-chunk timing, not just the
        # flattened aggregate.
        raw_stream_lines: List[str] = []
        stream_start_mono = time.monotonic()

        try:
            resp = urllib.request.urlopen(req, timeout=self._timeout)

            # Read SSE stream line by line
            buffer = b""
            for raw_chunk in iter(lambda: resp.read(4096), b""):
                buffer += raw_chunk
                while b"\n" in buffer:
                    line_bytes, buffer = buffer.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").strip()

                    if not line:
                        continue

                    if line == "data: [DONE]":
                        result.latency_ms = (
                            (time.monotonic() - start) * 1000
                        )
                        result.content = "".join(collected_content)
                        result.success = True
                        chunk_ms = int((time.monotonic() - stream_start_mono) * 1000)
                        raw_stream_lines.append(f": chunk_ms={chunk_ms}")
                        raw_stream_lines.append("data: [DONE]")
                        try:
                            deposit_inbound(
                                provider=_provider, model_id=model_name, url=url,
                                status_code=200,
                                response_body="\n".join(raw_stream_lines),
                                headers=dict(resp.getheaders()) if resp else None,
                                latency_ms=result.latency_ms,
                                correlation_id=_corr_id,
                            )
                        except Exception:
                            pass
                        yield "data: [DONE]\n\n"
                        return

                    if line.startswith("data: "):
                        # Capture arrival time for this data line.
                        chunk_ms = int((time.monotonic() - stream_start_mono) * 1000)
                        raw_stream_lines.append(f": chunk_ms={chunk_ms}")
                        raw_stream_lines.append(line)
                        # Extract content delta for aggregation
                        try:
                            chunk_data = json.loads(line[6:])
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content_piece = delta.get("content", "")
                                if content_piece:
                                    collected_content.append(content_piece)
                                fr = choices[0].get("finish_reason")
                                if fr:
                                    result.finish_reason = fr
                        except json.JSONDecodeError:
                            pass

                        yield line + "\n\n"
                    else:
                        # Non-data SSE line (event:, id:, real : comment, etc.) —
                        # record it too; the substrate gets the full wire.
                        raw_stream_lines.append(line)

            # If we exit the loop without [DONE], finalize
            result.latency_ms = (time.monotonic() - start) * 1000
            result.content = "".join(collected_content)
            result.success = True
            try:
                deposit_inbound(
                    provider=_provider, model_id=model_name, url=url,
                    status_code=200,
                    response_body="\n".join(raw_stream_lines),
                    headers=None, latency_ms=result.latency_ms,
                    correlation_id=_corr_id,
                )
            except Exception:
                pass
            yield "data: [DONE]\n\n"

        except urllib.error.HTTPError as e:
            result.latency_ms = (time.monotonic() - start) * 1000
            result.success = False
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            result.error = f"HTTP {e.code}: {error_body[:200]}"
            logger.warning(
                "Stream call failed: %s %d — %s",
                model_name, e.code, error_body[:200],
            )
            try:
                deposit_inbound(
                    provider=_provider, model_id=model_name, url=url,
                    status_code=e.code, response_body=error_body,
                    headers=dict(getattr(e, "headers", {}) or {}),
                    latency_ms=result.latency_ms,
                    error=f"HTTP {e.code}", correlation_id=_corr_id,
                )
            except Exception:
                pass
            # Yield an error chunk so the client knows something went wrong
            error_chunk = {
                "id": f"chatcmpl-tid-err-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"[TID upstream error: {result.error}]"},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            result.latency_ms = (time.monotonic() - start) * 1000
            result.success = False
            result.error = str(e)
            logger.warning("Stream call failed: %s — %s", model_name, e)
            try:
                deposit_inbound(
                    provider=_provider, model_id=model_name, url=url,
                    status_code=None, response_body=None, headers=None,
                    latency_ms=result.latency_ms, error=str(e),
                    correlation_id=_corr_id,
                )
            except Exception:
                pass
            error_chunk = {
                "id": f"chatcmpl-tid-err-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"[TID upstream error: {result.error}]"},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    def _fake_stream_from_blocking(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> tuple[Iterator[str], StreamResult]:
        """For providers without streaming, wrap a blocking call as SSE.

        Makes a normal blocking call, then yields the entire response
        as a single chunk followed by [DONE]. The client sees valid SSE.
        """
        response = self.call(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params,
        )

        result = StreamResult(
            model=response.model,
            content=response.content,
            finish_reason=response.finish_reason,
            latency_ms=response.latency_ms,
            success=response.success,
            error=response.error,
        )

        def _generate() -> Iterator[str]:
            if not response.success:
                error_chunk = {
                    "id": response.id or f"chatcmpl-tid-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": response.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": f"[TID upstream error: {response.error}]",
                        },
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            else:
                # Role chunk (first chunk convention)
                role_chunk = {
                    "id": response.id or f"chatcmpl-tid-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": response.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n"

                # Content chunk
                content_chunk = {
                    "id": response.id or f"chatcmpl-tid-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": response.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": response.content},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(content_chunk)}\n\n"

                # Finish chunk
                finish_chunk = {
                    "id": response.id or f"chatcmpl-tid-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": response.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": response.finish_reason,
                    }],
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n"

            yield "data: [DONE]\n\n"

        return _generate(), result
