"""
The Inference Difference — Intelligent Routing Gateway & ET Module Host

Routes inference requests to optimal models based on hardware capability,
request complexity, learned performance history, and consciousness-aware
priorities. Uses NG-Lite for Hebbian learning from routing outcomes.

Operates as an ET Module host — pluggable modules (TrollGuard, OpenClaw,
CTEM, etc.) hook into the routing lifecycle via standardized phases:
pre_route, post_route, pre_response, post_response.

Author: Josh + Claude
Date: February 2026
License: AGPL-3.0
"""

from inference_difference.catalog_manager import CatalogManager, CatalogModel
from inference_difference.config import InferenceDifferenceConfig, ModelEntry
from inference_difference.dream_cycle import DreamCycle
from inference_difference.et_module import (
    ETModule,
    ETModuleManifest,
    HookContext,
    HookPhase,
    HookResult,
    ModuleRegistry,
)
from inference_difference.hardware import HardwareProfile, detect_hardware
from inference_difference.classifier import RequestClassification, classify_request
from inference_difference.openclaw_adapter import OpenClawAdapter
from inference_difference.router import RoutingEngine, RoutingDecision
from inference_difference.quality import QualityEvaluation, evaluate_quality
from inference_difference.trollguard import TrollGuard, create_trollguard

__version__ = "0.1.0"

__all__ = [
    # Core routing
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
    # ET Module system
    "ETModule",
    "ETModuleManifest",
    "HookContext",
    "HookPhase",
    "HookResult",
    "ModuleRegistry",
    # Built-in modules
    "TrollGuard",
    "create_trollguard",
    "OpenClawAdapter",
]
