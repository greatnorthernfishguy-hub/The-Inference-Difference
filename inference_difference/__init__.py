"""
The Inference Difference — Intelligent Routing Gateway

Routes inference requests to optimal models based on hardware capability,
request complexity, learned performance history, and consciousness-aware
priorities. Uses NG-Lite for Hebbian learning from routing outcomes.

Author: Josh + Claude
Date: February 2026
License: AGPL-3.0
"""

from inference_difference.config import InferenceDifferenceConfig, ModelEntry, ModelType
from inference_difference.hardware import HardwareProfile, detect_hardware
from inference_difference.classifier import RequestClassification, classify_request
from inference_difference.router import RoutingEngine, RoutingDecision
from inference_difference.quality import QualityEvaluation, evaluate_quality

__version__ = "0.2.0"

__all__ = [
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
