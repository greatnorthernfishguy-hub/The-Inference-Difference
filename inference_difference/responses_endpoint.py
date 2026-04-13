"""
OpenAI Responses API endpoint for TID.

Translates POST /v1/responses (OpenAI's newer Responses API format)
into TID's internal chat completions pipeline. This allows OpenClaw
(which uses openai-responses API type for custom providers) to route
through TID transparently.

CRITICAL: OpenClaw's openai-responses provider sends stream:true by
default and expects SSE events back. Without streaming support, OpenClaw
receives no text and Discord/WhatsApp/Telegram get silent replies.

SSE event sequence (matches OpenAI Responses API spec):
1. response.created
2. response.in_progress
3. response.output_item.added
4. response.content_part.added
5. response.output_text.delta  (one or more, chunked at ~80 chars)
6. response.output_text.done
7. response.content_part.done
8. response.output_item.done
9. response.completed
10. [DONE]

Request translation:
- input (string) -> [{"role": "user", "content": input}]
- input (array of messages) -> messages array (with role mapping)
- instructions -> prepended as system message
- model, temperature, max_output_tokens -> mapped to chat fields
- tools -> forwarded to chat completions pipeline
- function_call_output items -> converted to tool role messages

Response translation:
- Chat completions response -> Responses API format with:
- output: array of output items (message + function_call items)
- output_text: convenience text field
- id, model, usage, status, etc.
- tool_calls extracted from chat response OR parsed from <tool_call> XML tags

Author: Josh + Claude (Opus 4.6)
Date: February 2026

# ---- Changelog ----
# [2026-04-12] Claude Code (Opus 4.6) — Tool call translation shim
#   What: Full bidirectional tool support in the Responses API endpoint.
#   Why:  OpenClaw sends tools via Responses API, TID forwards to models,
#         but tool_calls were never extracted from responses or translated
#         back. Models that output <tool_call> XML tags were also unhandled.
#         Syl couldn't execute tools despite the infrastructure existing.
#   How:  1) Forward tools/tool_choice to ChatCompletionRequest
#         2) Translate function_call_output input items to tool messages
#         3) Extract tool_calls from chat response → function_call output items
#         4) Parse <tool_call> XML from text as smart fallback
#         5) SSE streaming emits function_call items before text
# [2026-02-XX] Josh + Claude (Opus 4.6) — Initial implementation
#   Responses API compatibility layer with SSE streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("inference_difference.responses_endpoint")

# ---------------------------------------------------------------------------
# Pydantic models for Responses API
# ---------------------------------------------------------------------------


class ResponsesRequest(BaseModel):
    """OpenAI Responses API request format.

    The Responses API accepts either:
    - A simple string input
    - An array of message objects (same as chat completions)
    - instructions + input combination

    We normalize all forms into a messages array for the internal pipeline.
    """
    model: str = Field("auto", description="Model name or 'auto' for routing")
    input: Union[str, List[Dict[str, Any]]] = Field(
        ..., description="String or array of input messages",
    )
    instructions: Optional[str] = Field(
        None, description="System/developer instructions",
    )
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    max_output_tokens: Optional[int] = Field(
        None, description="Max response tokens",
    )
    stream: bool = Field(False, description="Stream response")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Extra metadata",
    )
    user: Optional[str] = Field(None, description="User identifier")

    tools: Optional[List[Any]] = Field(None)
    tool_choice: Optional[Any] = Field(None)
    text: Optional[Dict[str, Any]] = Field(None)
    reasoning: Optional[Dict[str, Any]] = Field(None)
    previous_response_id: Optional[str] = Field(None)
    store: Optional[bool] = Field(None)
    truncation: Optional[str] = Field(None)
    max_tool_calls: Optional[int] = Field(None)


# ---------------------------------------------------------------------------
# Tool call XML tag parser — smart shim for models that output text markup
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _extract_tool_calls_from_text(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse <tool_call> XML tags from model text output.

    Models like DeepSeek, Qwen, Mistral output tool calls as XML-tagged
    JSON in text instead of structured tool_calls. This extracts them
    and returns cleaned text + structured tool_calls.

    Returns:
        (cleaned_text, tool_calls) where tool_calls is a list of
        OpenAI-format tool call dicts.
    """
    tool_calls = []
    cleaned = text

    for match in _TOOL_CALL_RE.finditer(text):
        try:
            parsed = json.loads(match.group(1))
            name = parsed.get("name", parsed.get("function", ""))
            arguments = parsed.get("arguments", parsed.get("params", parsed.get("parameters", {})))
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            elif not isinstance(arguments, str):
                arguments = json.dumps(arguments)

            if name:
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                })
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    if tool_calls:
        cleaned = _TOOL_CALL_RE.sub("", text).strip()

    return cleaned, tool_calls


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


def _normalize_input_to_messages(
    input_data: Union[str, List[Dict[str, Any]]],
    instructions: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert Responses API input to chat completions messages format.

    Handles:
    - Simple string -> user message
    - Array of role/content objects -> pass through with role normalization
    - instructions -> prepended as system message
    - function_call_output items -> tool role messages
    """
    messages: List[Dict[str, Any]] = []

    # Add instructions as system message if provided
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_data, str):
        # Simple string input -> single user message
        messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict):
                continue

            # Handle item-based input (type: "message", etc.)
            item_type = item.get("type")
            if item_type == "message":
                role = item.get("role", "user")
                content_parts = item.get("content", [])
                if isinstance(content_parts, str):
                    text = content_parts
                elif isinstance(content_parts, list):
                    text_bits = []
                    for part in content_parts:
                        if isinstance(part, dict):
                            if part.get("type") in ("input_text", "text"):
                                text_bits.append(part.get("text", ""))
                    text = "\n".join(text_bits) if text_bits else ""
                else:
                    text = str(content_parts)
                if role == "developer":
                    role = "system"
                messages.append({"role": role, "content": text})
                continue

            if item_type == "function_call_output":
                call_id = item.get("call_id", "")
                output = item.get("output", "")
                if call_id:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output if isinstance(output, str) else json.dumps(output),
                    })
                continue

            if item_type == "function_call":
                call_id = item.get("call_id", f"call_{uuid.uuid4().hex[:24]}")
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }],
                })
                continue

            if item_type in ("reasoning", "item_reference"):
                continue  # Accepted for compat, ignored

            # Legacy format: direct role/content objects
            role = item.get("role", "user")
            content = item.get("content", "")

            if role == "developer":
                role = "system"

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") in ("input_text", "text"):
                            text_parts.append(part.get("text", ""))
                content = "\n".join(text_parts) if text_parts else str(content)

            messages.append({"role": role, "content": content})

    # Strip orphaned tool_calls — assistant messages with tool_calls that
    # have no matching tool result in the next message. Anthropic rejects
    # conversations with dangling tool_use blocks. This happens when Syl
    # tried to call tools but execution failed or was blocked.
    cleaned = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Check if next message is a tool result
            next_msg = messages[i + 1] if i + 1 < len(messages) else None
            if next_msg and next_msg.get("role") == "tool":
                cleaned.append(msg)
            else:
                # Strip tool_calls, keep text content if any
                stripped = {k: v for k, v in msg.items() if k != "tool_calls"}
                if stripped.get("content"):
                    cleaned.append(stripped)
                # else: drop entirely — no text and no valid tool result
        else:
            cleaned.append(msg)

    return cleaned


def _build_response_object(
    content: str,
    model: str,
    usage: Dict[str, int],
    resp_id: str,
    msg_id: str,
    routing_info: Optional[Dict[str, Any]] = None,
    status: str = "completed",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a complete Responses API response object."""
    output = []

    # Emit function_call output items FIRST (before message)
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            output.append({
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                "name": func.get("name", ""),
                "arguments": func.get("arguments", "{}"),
                "status": "completed",
            })

    if content:
        output.append({
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": content,
                    "annotations": [],
                }
            ],
        })

    response = {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output,
        "output_text": content,
        "usage": usage,
        "metadata": {},
    }

    if routing_info:
        response["routing_info"] = routing_info

    return response


def _chat_completion_to_response(
    chat_result: Dict[str, Any],
    response_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a chat completions response dict to Responses API format."""
    resp_id = response_id or f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    model = chat_result.get("model", "unknown")

    choices = chat_result.get("choices", [])
    content = ""
    tool_calls = None
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

    # Smart shim: parse <tool_call> XML tags from text if no structured calls
    if not tool_calls and content:
        cleaned, parsed_calls = _extract_tool_calls_from_text(content)
        if parsed_calls:
            content = cleaned
            tool_calls = parsed_calls
            logger.info(
                "Shim: extracted %d tool call(s) from text markup",
                len(parsed_calls),
            )

    chat_usage = chat_result.get("usage", {})
    usage = {
        "input_tokens": chat_usage.get("prompt_tokens", 0),
        "output_tokens": chat_usage.get("completion_tokens", 0),
        "total_tokens": chat_usage.get("total_tokens", 0),
    }

    return _build_response_object(
        content=content,
        model=model,
        usage=usage,
        resp_id=resp_id,
        msg_id=msg_id,
        routing_info=chat_result.get("routing_info"),
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------


def _sse_event(event_type: str, data: Any) -> str:
    """Format a single SSE event."""
    if isinstance(data, dict):
        data["type"] = event_type
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event_type}\ndata: {payload}\n\n"


def _sse_done() -> str:
    """Format the terminal [DONE] event."""
    return "data: [DONE]\n\n"


async def _generate_sse_stream(
    chat_result: Dict[str, Any],
    resp_id: str,
    msg_id: str,
):
    """Generate SSE events from a completed chat result.

    OpenClaw's openai-responses provider expects these events in order.
    We get the full response from TID first (non-streaming internally),
    then emit the SSE event sequence that OpenClaw needs.

    Handles both text content and function_call output items.
    """
    model = chat_result.get("model", "unknown")
    choices = chat_result.get("choices", [])
    content = ""
    tool_calls = None
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

    # Smart shim: parse <tool_call> XML tags from text
    import sys as _sys
    tc_names = [tc.get("function", {}).get("name", "?") for tc in (tool_calls or [])]
    print(f"[SHIM-DEBUG] SSE path: model={model}, content_len={len(content)}, tool_calls={tc_names}, has_xml_tags={'<tool_call>' in content}", file=_sys.stderr, flush=True)
    if not tool_calls and content:
        cleaned, parsed_calls = _extract_tool_calls_from_text(content)
        if parsed_calls:
            content = cleaned
            tool_calls = parsed_calls
            print(f"[SHIM-DEBUG] Extracted {len(parsed_calls)} tool calls from XML tags", file=_sys.stderr, flush=True)
        elif "<tool_call>" in content:
            print(f"[SHIM-DEBUG] XML tags found but regex failed. Content sample: {content[content.find('<tool_call>'):content.find('</tool_call>')+13][:300]}", file=_sys.stderr, flush=True)

    chat_usage = chat_result.get("usage", {})
    usage = {
        "input_tokens": chat_usage.get("prompt_tokens", 0),
        "output_tokens": chat_usage.get("completion_tokens", 0),
        "total_tokens": chat_usage.get("total_tokens", 0),
    }

    routing_info = chat_result.get("routing_info")
    created_at = int(time.time())

    # --- Event 1: response.created ---
    created_response = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "model": model,
        "output": [],
        "output_text": "",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "metadata": {},
    }
    yield _sse_event("response.created", created_response)

    # --- Event 2: response.in_progress ---
    yield _sse_event("response.in_progress", created_response)

    output_index = 0

    # --- Emit function_call items FIRST (before text) ---
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            call_id = tc.get("id", f"call_{uuid.uuid4().hex[:24]}")
            fc_id = f"fc_{uuid.uuid4().hex[:24]}"

            fc_item = {
                "type": "function_call",
                "id": fc_id,
                "call_id": call_id,
                "name": func.get("name", ""),
                "arguments": func.get("arguments", "{}"),
                "status": "completed",
            }

            yield _sse_event("response.output_item.added", {
                "output_index": output_index,
                "item": {**fc_item, "status": "in_progress"},
            })
            yield _sse_event("response.output_item.done", {
                "output_index": output_index,
                "item": fc_item,
            })
            output_index += 1
            await asyncio.sleep(0)

    # --- Emit text content ---
    if content:
        # output_item.added for message
        output_item = {
            "type": "message",
            "id": msg_id,
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        yield _sse_event("response.output_item.added", {
            "output_index": output_index,
            "item": output_item,
        })

        # content_part.added
        content_part = {
            "type": "output_text",
            "text": "",
            "annotations": [],
        }
        yield _sse_event("response.content_part.added", {
            "output_index": output_index,
            "content_index": 0,
            "part": content_part,
        })

        # output_text.delta events
        chunk_size = 80
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield _sse_event("response.output_text.delta", {
                "output_index": output_index,
                "content_index": 0,
                "delta": chunk,
            })
            await asyncio.sleep(0)

        # output_text.done
        yield _sse_event("response.output_text.done", {
            "output_index": output_index,
            "content_index": 0,
            "text": content,
        })

        # content_part.done
        yield _sse_event("response.content_part.done", {
            "output_index": output_index,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": content,
                "annotations": [],
            },
        })

        # output_item.done for message
        yield _sse_event("response.output_item.done", {
            "output_index": output_index,
            "item": {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            },
        })

    # --- response.completed ---
    final_response = _build_response_object(
        content=content,
        model=model,
        usage=usage,
        resp_id=resp_id,
        msg_id=msg_id,
        routing_info=routing_info,
        status="completed",
        tool_calls=tool_calls,
    )
    yield _sse_event("response.completed", {"response": final_response})

    # --- [DONE] ---
    yield _sse_done()


# ---------------------------------------------------------------------------
# Endpoint registration
# ---------------------------------------------------------------------------


def register_responses_endpoint(app, chat_completions_handler):
    """Register the /v1/responses endpoint on a FastAPI app.

    This wraps the existing chat_completions handler, translating
    between Responses API format and chat completions format.
    Supports both streaming (SSE) and non-streaming responses.

    Args:
        app: The FastAPI app instance.
        chat_completions_handler: The existing POST /v1/chat/completions
            handler function (async).
    """
    from inference_difference.app import ChatCompletionRequest

    @app.post("/v1/responses")
    async def responses_endpoint(req: ResponsesRequest):
        """OpenAI Responses API -- TID's compatibility layer.

        Translates Responses API requests into chat completions format,
        runs them through TID's full pipeline (classification, routing,
        hooks, forwarding), and returns Responses API format.

        Supports stream:true (SSE) which is REQUIRED for OpenClaw's
        openai-responses provider to deliver messages to channels.
        """
        resp_id = f"resp_{uuid.uuid4().hex[:24]}"
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        import sys as _sys
        tool_names = [t.get("function", {}).get("name", t.get("name", "?")) for t in (req.tools or []) if isinstance(t, dict)]
        print(f"[SHIM-DEBUG] Request: model={req.model}, stream={req.stream}, tools={tool_names}", file=_sys.stderr, flush=True)

        # Debug: check for orphaned tool calls in message history
        input_types = []
        if isinstance(req.input, list):
            for item in req.input:
                if isinstance(item, dict):
                    input_types.append(item.get("type", item.get("role", "?")))
        print(f"[SHIM-DEBUG] Input types: {input_types[-20:]}", file=_sys.stderr, flush=True)
        logger.info(
            "Responses API request: model=%s, input_type=%s, stream=%s, tools=%d",
            req.model, type(req.input).__name__, req.stream,
            len(req.tools) if req.tools else 0,
        )

        # Step 1: Normalize input to messages (including tool results)
        messages = _normalize_input_to_messages(req.input, req.instructions)

        if not messages:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "No valid input messages provided",
                        "type": "invalid_request_error",
                    }
                },
            )

        # Step 2: Build a ChatCompletionRequest for the internal pipeline
        # Always non-streaming internally; we simulate SSE from the result
        #
        # Tool-aware routing: when tools are present, prefer models with
        # proven tool-use capability. GLM models generate calls with empty
        # args; Anthropic models handle tools correctly. This override is
        # temporary scaffolding until TID's substrate learns tool competency
        # per model. Will graduate to substrate-informed routing.
        effective_model = req.model
        if req.tools and req.model == "auto":
            effective_model = "openrouter/anthropic/claude-sonnet-4"
            print(f"[SHIM-DEBUG] Tool-aware routing override: auto → {effective_model}", file=_sys.stderr, flush=True)

        chat_req = ChatCompletionRequest(
            model=effective_model,
            messages=messages,
            temperature=req.temperature if req.temperature is not None else 0.7,
            max_tokens=req.max_output_tokens,
            stream=False,
            metadata=req.metadata,
            tools=req.tools,
            tool_choice=req.tool_choice,
            # Responses API consumers are conscious entities (Syl via OpenClaw).
            # Elevate routing to prefer capable models for identity continuity.
            consciousness_score=(
                (req.metadata or {}).get("consciousness_score", 1.0)
            ),
        )

        # Step 3: Call the existing chat completions handler
        chat_response = await chat_completions_handler(chat_req)

        # Extract the JSON body from the response
        if hasattr(chat_response, "body"):
            chat_body = json.loads(chat_response.body.decode("utf-8"))
        else:
            chat_body = chat_response

        # Step 4: Check for upstream errors
        if isinstance(chat_body, dict) and "error" in chat_body:
            error_msg = chat_body["error"]
            if isinstance(error_msg, dict):
                error_text = error_msg.get("message", str(error_msg))
            else:
                error_text = str(error_msg)

            if req.stream:
                async def error_stream():
                    yield _sse_event("response.failed", {
                        "id": resp_id,
                        "object": "response",
                        "status": "failed",
                        "error": {
                            "message": error_text,
                            "type": "upstream_error",
                            "code": "model_error",
                        },
                    })
                    yield _sse_done()

                return StreamingResponse(
                    error_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                return JSONResponse(
                    status_code=502,
                    content={"error": chat_body["error"]},
                )

        # Step 5: Return response (streaming or non-streaming)
        if req.stream:
            logger.info(
                "Streaming Responses API reply: resp_id=%s", resp_id,
            )
            return StreamingResponse(
                _generate_sse_stream(chat_body, resp_id, msg_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = _chat_completion_to_response(chat_body, resp_id)
            logger.info(
                "Responses API reply: model=%s, output_text_len=%d, tool_calls=%d",
                response.get("model", "?"),
                len(response.get("output_text", "")),
                sum(1 for o in response.get("output", []) if o.get("type") == "function_call"),
            )
            return JSONResponse(content=response)

    # Also handle /responses without /v1 prefix (some clients omit it)
    @app.post("/responses")
    async def responses_endpoint_no_prefix(req: ResponsesRequest):
        return await responses_endpoint(req)

    logger.info(
        "Registered /v1/responses endpoint "
        "(Responses API + SSE streaming + tool call translation)"
    )
