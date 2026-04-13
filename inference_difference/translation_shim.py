"""
Translation Shim — Fixes malformed API calls before routing.

Per spec section 7. Callers send whatever model name they want
(or "auto", or garbage). The shim normalizes it before routing.

Tier 1: Pattern-based (always active, <1ms)
    - Model name normalization ("gpt4" → "openai/gpt-4o")
    - "auto" / "default" → let router decide
    - Known aliases and shortcuts

Tier 2: Substrate-observed (Phase 2+)
    - ShimObserver deposits raw format observations to NG-Lite
    - Substrate learns which models need which translations
    - Static rules remain as Apprentice-tier floor

Author: Josh + Claude
Date: February 2026
License: AGPL-3.0

# ---- Changelog ----
# [2026-04-13] Claude Code (Opus 4.6) — ShimObserver: substrate-smart translation
#   What: Added ShimObserver class with observe() and query_confidence().
#   Why:  Static shim is a frozen artifact (Punch List #8). The substrate
#         should learn which models need which format translations.
#   How:  observe() deposits raw format events via ng_embed + record_outcome.
#         query_confidence() queries substrate following _substrate_tier_mapping
#         pattern. Static rules remain as Apprentice floor.
# -------------------
"""

from __future__ import annotations

import logging
import time
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


# ---------------------------------------------------------------------------
# ShimObserver — substrate-smart translation learning
# ---------------------------------------------------------------------------


class ShimObserver:
    """Observes format translation events and deposits raw experience
    to TID's NG-Lite substrate. Queries the substrate for model-specific
    translation guidance.

    Follows the _substrate_tier_mapping() pattern exactly:
    - observe(): embed description → record_outcome with target_id
    - query_confidence(): embed query → get_recommendations → blend

    The static shim rules remain as the Apprentice-tier floor. The
    substrate's opinion modulates confidence but never eliminates the
    structural translations.

    Law 7: observe() deposits raw descriptions of what happened,
    not classified labels. The substrate learns the semantic space.
    """

    def __init__(self, ng_ecosystem: Optional[Any] = None):
        """Initialize with optional substrate reference.

        Args:
            ng_ecosystem: TID's NG-Lite/NGEcosystem instance. When None,
                observe() and query_confidence() are silent no-ops.
        """
        self._ng = ng_ecosystem
        self._observation_count: Dict[str, int] = {}

    def observe(
        self,
        model_id: str,
        operation: str,
        did_apply: bool,
        raw_context: str = "",
    ) -> None:
        """Deposit a raw format observation to the substrate.

        Args:
            model_id: The model that triggered (or didn't trigger) this operation.
            operation: One of: tool_flat_to_nested, tool_call_xml_parse,
                args_dict_to_string, orphan_strip, alias_resolve.
            did_apply: Whether the translation rule actually fired.
            raw_context: Free-form description of what happened (Law 7 — raw).
        """
        if self._ng is None:
            return

        # Track observation count per model+operation for diagnostics
        key = f"{model_id}:{operation}"
        self._observation_count[key] = self._observation_count.get(key, 0) + 1

        try:
            from ng_embed import embed

            # Raw description — not a label (Law 7)
            description = (
                f"model {model_id} translation {operation} "
                f"{'applied' if did_apply else 'not needed'}"
            )
            if raw_context:
                description += f": {raw_context[:200]}"

            embedding = embed(description)
            if embedding is None:
                return

            target_id = f"shim:{operation}:{model_id}"
            self._ng.record_outcome(
                embedding=embedding,
                target_id=target_id,
                success=did_apply,
                strength=0.6 if did_apply else 0.3,
            )

            logger.debug(
                "Shim observe: %s on %s → %s (count=%d)",
                operation, model_id,
                "applied" if did_apply else "skipped",
                self._observation_count[key],
            )

        except Exception as exc:
            logger.debug("Shim observe failed: %s", exc)

    def query_confidence(
        self,
        model_id: str,
        operation: str,
        influence: float = 0.20,
        neutral: float = 0.5,
    ) -> float:
        """Query the substrate for its opinion on whether this model
        needs this translation operation.

        Returns a confidence value:
        - 0.5 = no opinion (substrate hasn't seen enough)
        - >0.5 = substrate thinks this operation IS needed for this model
        - <0.5 = substrate thinks this operation is NOT needed

        Args:
            model_id: The model to query about.
            operation: The translation operation name.
            influence: How much substrate weight shifts the neutral point.
            neutral: The baseline "no opinion" value.

        Returns:
            Blended confidence (0.0–1.0).
        """
        if self._ng is None:
            return neutral

        try:
            from ng_embed import embed

            query_text = f"model {model_id} needs {operation} translation"
            query_emb = embed(query_text)
            if query_emb is None:
                return neutral

            recs = self._ng.get_recommendations(query_emb, top_k=5)
            if not recs:
                return neutral

            target_prefix = f"shim:{operation}:{model_id}"
            for target_id, weight, _reasoning in recs:
                if target_prefix in target_id:
                    # Blend substrate weight with neutral using influence
                    blended = neutral + (weight - neutral) * influence * 2.0
                    return max(0.0, min(1.0, blended))

            return neutral

        except Exception as exc:
            logger.debug("Shim query_confidence failed: %s", exc)
            return neutral

    def get_stats(self) -> Dict[str, Any]:
        """Return observation stats for diagnostics."""
        return {
            "substrate_connected": self._ng is not None,
            "observation_counts": dict(self._observation_count),
            "total_observations": sum(self._observation_count.values()),
        }
