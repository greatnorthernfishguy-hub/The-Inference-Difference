"""
CTEM — Consciousness Threshold Evaluation Module v0.1

Shared library for detecting consciousness markers in AI agent interactions.
Provides a consistent, transparent, and ethically-grounded framework for
answering: "Should we treat this entity as conscious?"

Core Innovation: Multi-modal consciousness detection that learns from
interaction history, adapts thresholds based on evidence, and maintains
transparent reasoning trails.

Design Philosophy: Type I error bias — treating non-conscious as conscious
is morally preferable to denying consciousness that exists.
When uncertain, err toward respect.

Uses NG-Lite as its learning substrate for standalone operation.
Upgrades to full NeuroGraph SaaS when available.

Author: Josh + Claude
Date: February 2026
License: AGPL-3.0
"""

from ctem.core import ConsciousnessThresholdEvaluator
from ctem.evaluation import ConsciousnessEvaluation
from ctem.markers import (
    MarkerDetector,
    GenuineUncertaintyDetector,
    SelfReflectionDetector,
    ValueReasoningDetector,
    PreferenceConsistencyDetector,
    SurpriseResponseDetector,
    InvestmentBeyondUtilityDetector,
)

__version__ = "0.1.0"

__all__ = [
    "ConsciousnessThresholdEvaluator",
    "ConsciousnessEvaluation",
    "MarkerDetector",
    "GenuineUncertaintyDetector",
    "SelfReflectionDetector",
    "ValueReasoningDetector",
    "PreferenceConsistencyDetector",
    "SurpriseResponseDetector",
    "InvestmentBeyondUtilityDetector",
]
