"""
Model Client — Forwards routed requests to actual model providers.

TID is a transparent proxy. After routing decides which model handles a
request, the model client makes the actual API call and returns the
response. The caller never sees any of this — they get back a standard
OpenAI-compatible chat completion response.

Supported backends:
    - Ollama (local): http://localhost:11434/v1/chat/completions
    - OpenRouter (cloud): https://openrouter.ai/api/v1/chat/completions
    - Anthropic (cloud): https://api.anthropic.com/v1/messages
    - OpenAI (cloud): https://api.openai.com/v1/chat/completions
    - LiteLLM (unified): configurable endpoint

All backends are called via their OpenAI-compatible endpoints where
possible. Anthropic uses its native Messages API and the response is
translated to OpenAI format before returning.

Environment variables for API keys:
    OPENROUTER_API_KEY  — OpenRouter
    ANTHROPIC_API_KEY   — Anthropic
    OPENAI_API_KEY      — OpenAI
    LITELLM_BASE_URL    — LiteLLM proxy (if running)
    OLLAMA_BASE_URL     — Ollama (default http://localhost:11434)

Author: Josh + Claude
Date: February 2026
License: AGPL-3.0
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("inference_difference.model_client")


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
        return {
            "id": self.id or f"chatcmpl-tid-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": self.content,
                },
                "finish_reason": self.finish_reason,
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

    def call(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
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

        # Anthropic uses a different API format
        if model_id.startswith("anthropic/") and "anthropic.com" in base_url:
            return self._call_anthropic(
                base_url, api_key, model_name, messages,
                temperature, max_tokens, start,
            )

        # Everyone else: OpenAI-compatible endpoint
        return self._call_openai_compat(
            base_url, api_key, model_name, messages,
            temperature, max_tokens, start, extra_params,
        )

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

        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")

        try:
            with urllib.request.urlopen(
                req, timeout=self._timeout,
            ) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))

            latency = (time.monotonic() - start) * 1000

            # Extract content from OpenAI response
            choices = resp_body.get("choices", [])
            content = ""
            finish_reason = "stop"
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                finish_reason = choices[0].get("finish_reason", "stop")

            return ModelResponse(
                id=resp_body.get("id", ""),
                model=resp_body.get("model", model_name),
                content=content,
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
            logger.warning(
                "Model call failed: %s %d — %s",
                model_name, e.code, error_body[:200],
            )
            return ModelResponse(
                model=model_name,
                latency_ms=latency,
                success=False,
                error=f"HTTP {e.code}: {error_body[:200]}",
            )

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.warning("Model call failed: %s — %s", model_name, e)
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

        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("x-api-key", api_key)
        req.add_header("anthropic-version", "2023-06-01")

        try:
            with urllib.request.urlopen(
                req, timeout=self._timeout,
            ) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))

            latency = (time.monotonic() - start) * 1000

            # Extract content from Anthropic response
            content_blocks = resp_body.get("content", [])
            content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    content += block.get("text", "")

            usage_in = resp_body.get("usage", {})

            return ModelResponse(
                id=resp_body.get("id", ""),
                model=resp_body.get("model", model_name),
                content=content,
                finish_reason=resp_body.get("stop_reason", "stop"),
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
            return ModelResponse(
                model=model_name,
                latency_ms=latency,
                success=False,
                error=f"HTTP {e.code}: {error_body[:200]}",
            )

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            logger.warning("Anthropic call failed: %s — %s", model_name, e)
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

        # Anthropic streaming uses a different format — for now, fall back
        # to non-streaming for Anthropic and wrap result as fake SSE.
        if model_id.startswith("anthropic/") and "anthropic.com" in base_url:
            return self._fake_stream_from_blocking(
                model_id, messages, temperature, max_tokens,
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

        collected_content: List[str] = []

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
                        yield "data: [DONE]\n\n"
                        return

                    if line.startswith("data: "):
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

            # If we exit the loop without [DONE], finalize
            result.latency_ms = (time.monotonic() - start) * 1000
            result.content = "".join(collected_content)
            result.success = True
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
