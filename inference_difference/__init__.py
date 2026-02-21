"""
The Inference Difference — Intelligent Routing Gateway

Routes inference requests to optimal models based on hardware capability,
request complexity, learned performance history, and consciousness-aware
priorities. Uses NG-Lite for Hebbian learning from routing outcomes.

Author: Josh + Claude
Date: February 2026
License: AGPL-3.0

Changelog (Grok audit response, 2026-02-19):
- KEPT: No version guard / compatibility shim (audit: "breaking changes?").
  This is v0.1.0 — pre-release. There are zero external consumers to break.
  Adding semver guards to __init__.py before we have a stable API would be
  premature. When we hit 1.0, we'll add deprecation warnings. __all__ is
  already exhaustive as the audit itself acknowledged.
"""

from inference_difference.catalog_manager import CatalogManager, CatalogModel
from inference_difference.config import InferenceDifferenceConfig, ModelEntry
from inference_difference.dream_cycle import DreamCycle
from inference_difference.hardware import HardwareProfile, detect_hardware
from inference_difference.classifier import RequestClassification, classify_request
from inference_difference.router import RoutingEngine, RoutingDecision
from inference_difference.quality import QualityEvaluation, evaluate_quality

__version__ = "0.1.0"

__all__ = [
    "CatalogManager",
    "CatalogModel",
    "DreamCycle",
    "InferenceDifferenceConfig",
    "ModelEntry",
    "HardwareProfile",
    "detect_hardware",
    "RequestClassification",
    "classify_request",
    "RoutingEngine",
    "RoutingDecision",
    "QualityEvaluation",
    "evaluate_quality",
]
