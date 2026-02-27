"""
Translation Shim — Fixes malformed API calls before routing.

Per spec section 7. Callers send whatever model name they want
(or "auto", or garbage). The shim normalizes it before routing.

Tier 1: Pattern-based (always active, <1ms)
    - Model name normalization ("gpt4" → "openai/gpt-4o")
    - "auto" / "default" → let router decide
    - Known aliases and shortcuts

Author: Josh + Claude
Date: February 2026
License: AGPL-3.0
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("inference_difference.translation_shim")


# ---------------------------------------------------------------------------
# Model name aliases
# ---------------------------------------------------------------------------

MODEL_ALIASES: Dict[str, str] = {
    # Auto/default → empty (let router decide)
    "auto": "",
    "tid/auto": "",
    "default": "",

    # OpenAI shortcuts
    "gpt-4": "openai/gpt-4o",
    "gpt4": "openai/gpt-4o",
    "gpt-4o": "openai/gpt-4o",
    "gpt4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "chatgpt": "openai/gpt-4o",

    # Anthropic shortcuts
    "claude": "anthropic/claude-sonnet-4-5-20250929",
    "claude-sonnet": "anthropic/claude-sonnet-4-5-20250929",
    "claude-haiku": "anthropic/claude-haiku-4-5-20251001",
    "claude-opus": "anthropic/claude-opus-4-5-20250929",
    "sonnet": "anthropic/claude-sonnet-4-5-20250929",
    "haiku": "anthropic/claude-haiku-4-5-20251001",
    "opus": "anthropic/claude-opus-4-5-20250929",

    # DeepSeek shortcuts
    "deepseek": "deepseek/deepseek-chat",
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-coder": "deepseek/deepseek-chat",

    # Local model shortcuts
    "llama": "ollama/llama3.1:8b",
    "llama3": "ollama/llama3.2:3b",
    "llama3.1": "ollama/llama3.1:8b",
    "llama3.2": "ollama/llama3.2:3b",
    "qwen": "ollama/qwen2.5-coder:7b",
    "qwen-coder": "ollama/qwen2.5-coder:7b",
    "deepseek-local": "ollama/deepseek-r1:14b",
}


def translate_request(
    model: str,
    messages: List[Dict[str, str]],
) -> Tuple[str, Optional[str]]:
    """Normalize the model name and detect issues.

    Args:
        model: The model name from the request.
        messages: The messages list (for malformation detection).

    Returns:
        (normalized_model, translation_type)
        translation_type is None if no translation was needed,
        "alias" if a model alias was resolved,
        "auto" if model was "auto"/"default"/empty.
    """
    if not model or model.strip() == "":
        return "", "auto"

    lower = model.strip().lower()

    # Check alias table
    if lower in MODEL_ALIASES:
        resolved = MODEL_ALIASES[lower]
        if resolved == "":
            logger.debug("Model '%s' → auto-route", model)
            return "", "auto"
        logger.debug("Model alias '%s' → '%s'", model, resolved)
        return resolved, "alias"

    # Already a fully-qualified model_id (has a / separator)
    if "/" in model:
        return model, None

    # Unrecognized bare name — pass through, router will handle it
    logger.debug("Unrecognized model name '%s' — passing through", model)
    return model, None
