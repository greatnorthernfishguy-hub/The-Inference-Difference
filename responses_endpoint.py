“””
OpenAI Responses API endpoint for TID.

Translates POST /v1/responses (OpenAI’s newer Responses API format)
into TID’s internal chat completions pipeline. This allows OpenClaw
(which uses openai-responses API type for custom providers) to route
through TID transparently.

Request translation:
- input (string) → [{“role”: “user”, “content”: input}]
- input (array of messages) → messages array (with role mapping)
- instructions → prepended as system message
- model, temperature, max_output_tokens → mapped to chat completions fields

Response translation:
- Chat completions response → Responses API format with:
- output: array of output items
- output_text: convenience text field
- id, model, usage, status, etc.

Author: Josh + Claude (Opus 4.6)
Date: February 2026
License: AGPL-3.0
“””

from **future** import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(“inference_difference.responses_endpoint”)

# —————————————————————————

# Pydantic models for Responses API

# —————————————————————————

class ResponsesRequest(BaseModel):
“”“OpenAI Responses API request format.

```
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

# Fields we accept but don't use (for compatibility)
tools: Optional[List[Any]] = Field(None)
tool_choice: Optional[Any] = Field(None)
text: Optional[Dict[str, Any]] = Field(None)
reasoning: Optional[Dict[str, Any]] = Field(None)
previous_response_id: Optional[str] = Field(None)
store: Optional[bool] = Field(None)
truncation: Optional[str] = Field(None)
```

# —————————————————————————

# Translation helpers

# —————————————————————————

def _normalize_input_to_messages(
input_data: Union[str, List[Dict[str, Any]]],
instructions: Optional[str] = None,
) -> List[Dict[str, Any]]:
“”“Convert Responses API input to chat completions messages format.

```
Handles:
- Simple string → user message
- Array of role/content objects → pass through with role normalization
- instructions → prepended as system message
"""
messages: List[Dict[str, Any]] = []

# Add instructions as system message if provided
if instructions:
    messages.append({"role": "system", "content": instructions})

if isinstance(input_data, str):
    # Simple string input → single user message
    messages.append({"role": "user", "content": input_data})
elif isinstance(input_data, list):
    for item in input_data:
        if not isinstance(item, dict):
            continue

        role = item.get("role", "user")
        content = item.get("content", "")

        # Normalize "developer" role to "system" (Responses API uses "developer")
        if role == "developer":
            role = "system"

        # Handle content that's an array of content parts
        if isinstance(content, list):
            # Extract text from content parts
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "input_text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
            content = "\n".join(text_parts) if text_parts else str(content)

        messages.append({"role": role, "content": content})

return messages
```

def _chat_completion_to_response(
chat_result: Dict[str, Any],
response_id: Optional[str] = None,
) -> Dict[str, Any]:
“”“Convert a chat completions response dict to Responses API format.

```
Maps:
- choices[0].message.content → output[0] as message item + output_text
- usage → usage (with slight field name adjustments)
- model, id → preserved
"""
resp_id = response_id or f"resp_{uuid.uuid4().hex[:24]}"
model = chat_result.get("model", "unknown")

# Extract content from chat completion
choices = chat_result.get("choices", [])
content = ""
if choices:
    msg = choices[0].get("message", {})
    content = msg.get("content", "")

# Build output items (Responses API format)
output = []
if content:
    output.append({
        "type": "message",
        "id": f"msg_{uuid.uuid4().hex[:24]}",
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

# Map usage fields
chat_usage = chat_result.get("usage", {})
usage = {
    "input_tokens": chat_usage.get("prompt_tokens", 0),
    "output_tokens": chat_usage.get("completion_tokens", 0),
    "total_tokens": chat_usage.get("total_tokens", 0),
}

# Build the Responses API response object
response = {
    "id": resp_id,
    "object": "response",
    "created_at": int(time.time()),
    "status": "completed",
    "model": model,
    "output": output,
    "output_text": content,
    "usage": usage,
    "metadata": {},
}

# Preserve TID routing info if present
if "routing_info" in chat_result:
    response["routing_info"] = chat_result["routing_info"]

return response
```

# —————————————————————————

# Endpoint registration

# —————————————————————————

def register_responses_endpoint(app, chat_completions_handler):
“”“Register the /v1/responses endpoint on a FastAPI app.

```
This wraps the existing chat_completions handler, translating
between Responses API format and chat completions format.

Args:
    app: The FastAPI app instance.
    chat_completions_handler: The existing POST /v1/chat/completions
        handler function (async or sync).
"""
from inference_difference.app import ChatCompletionRequest

@app.post("/v1/responses")
async def responses_endpoint(req: ResponsesRequest) -> JSONResponse:
    """OpenAI Responses API — TID's compatibility layer.

    Translates Responses API requests into chat completions format,
    runs them through TID's full pipeline (classification, routing,
    hooks, forwarding), and returns Responses API format.
    """
    logger.info(
        "Responses API request: model=%s, input_type=%s",
        req.model, type(req.input).__name__,
    )

    # Step 1: Normalize input to messages
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
    chat_req = ChatCompletionRequest(
        model=req.model,
        messages=messages,
        temperature=req.temperature if req.temperature is not None else 0.7,
        max_tokens=req.max_output_tokens,
        stream=False,  # Handle streaming separately if needed
        metadata=req.metadata,
    )

    # Step 3: Call the existing chat completions handler
    # The handler returns a JSONResponse, so we need to extract the body
    chat_response = await chat_completions_handler(chat_req)

    # Extract the JSON body from the response
    if hasattr(chat_response, 'body'):
        import json
        chat_body = json.loads(chat_response.body.decode('utf-8'))
    else:
        chat_body = chat_response

    # Step 4: Check for upstream errors
    if isinstance(chat_body, dict) and "error" in chat_body:
        return JSONResponse(
            status_code=502,
            content={
                "error": chat_body["error"],
            },
        )

    # Step 5: Translate to Responses API format
    response = _chat_completion_to_response(chat_body)

    logger.info(
        "Responses API reply: model=%s, output_text_len=%d",
        response.get("model", "?"),
        len(response.get("output_text", "")),
    )

    return JSONResponse(content=response)

logger.info("Registered /v1/responses endpoint (Responses API compatibility)")
```
