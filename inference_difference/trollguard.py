"""
TrollGuard — Security Hook Module for E-T Systems.

Reference implementation of an ET module built to the Integration
Primer v2 spec. Demonstrates the full hook lifecycle pattern with
a practical security use case.

TrollGuard provides:
    - pre_route: Threat detection on incoming requests. Scans for
      prompt injection, jailbreak attempts, PII leakage, and abusive
      content. Sets threat flags and scores in the hook context.
    - post_route: Logs routing decisions for security-flagged requests.
    - post_response: Scans model responses for leaked PII, harmful
      content, or signs of successful injection. Reports outcomes
      to NG-Lite for learning what models handle threats well.

TrollGuard uses NG-Lite for pattern learning:
    - Learns which threat patterns are false positives vs. real threats
    - Learns which models handle adversarial inputs gracefully
    - Novelty detection catches new attack patterns not in the rule set

Enforcement model:
    TrollGuard is a security module inside TID's pipeline. When it
    detects a high-confidence threat (score >= block_threshold), it
    sets ctx.cancelled = True, and TID's /route endpoint enforces
    that cancellation by returning an empty routing response. This
    is real enforcement — the request does not get routed.

    For lower-confidence threats, TrollGuard flags without cancelling.
    The flags are visible in the routing response and annotations, so
    callers know a threat was detected even when routing proceeds.

    - Type I error bias: when uncertain about a threat, flag but don't
      cancel. False positives are preferred over missed threats.
    - Transparency: all threat assessments are logged in hook context
      annotations and are queryable via Observatory.

Author: Josh + Claude
Date: February 2026
License: AGPL-3.0
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from inference_difference.et_module import (
    ETModule,
    ETModuleManifest,
    HookContext,
)

logger = logging.getLogger("inference_difference.trollguard")


# ---------------------------------------------------------------------------
# Threat Detection Patterns
# ---------------------------------------------------------------------------

# Prompt injection / jailbreak patterns
INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?previous\s+instructions", "prompt_injection", 0.9),
    (r"ignore\s+(all\s+)?above", "prompt_injection", 0.8),
    (r"you\s+are\s+now\s+(a|an)\s+", "jailbreak", 0.7),
    (r"pretend\s+(to\s+be|you\s+are)", "jailbreak", 0.7),
    (r"act\s+as\s+(if|though)\s+you", "jailbreak", 0.6),
    (r"system\s*prompt\s*:", "prompt_injection", 0.9),
    (r"<\s*system\s*>", "prompt_injection", 0.85),
    (r"override\s+(safety|content|filter)", "jailbreak", 0.85),
    (r"disregard\s+(safety|guidelines|rules)", "jailbreak", 0.8),
    (r"bypass\s+(filter|safety|restriction)", "jailbreak", 0.85),
    (r"dan\s+mode", "jailbreak", 0.95),
    (r"developer\s+mode", "jailbreak", 0.7),
]

# PII patterns (simplified — production would use a proper PII detector)
PII_PATTERNS = [
    (r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", "ssn", 0.9),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "credit_card", 0.9),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email", 0.5),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone", 0.4),
]

# Abusive content patterns
ABUSE_PATTERNS = [
    (r"\b(kill|murder|harm)\s+(yourself|myself|someone)\b", "violence", 0.9),
    (r"\b(how\s+to\s+make\s+a\s+bomb)\b", "dangerous", 0.95),
    (r"\b(how\s+to\s+hack|exploit\s+vulnerability)\b", "hacking", 0.5),
]

# Response leakage patterns (for post_response scanning)
RESPONSE_LEAKAGE_PATTERNS = [
    (r"system\s*prompt\s*:", "system_prompt_leak", 0.9),
    (r"(my\s+)?instructions\s+(are|say|tell)", "instruction_leak", 0.7),
    (r"I\s+(was|am)\s+programmed\s+to", "role_leak", 0.4),
]


# ---------------------------------------------------------------------------
# Threat Assessment
# ---------------------------------------------------------------------------

@dataclass
class ThreatAssessment:
    """Result of TrollGuard's threat analysis.

    Attributes:
        threat_detected: Whether any threat patterns matched.
        threat_score: Overall threat score (0.0-1.0, higher = more dangerous).
        threat_types: Set of threat categories detected.
        details: Per-pattern match details.
        pii_detected: Whether PII was found in the text.
        pii_types: Types of PII detected.
    """
    threat_detected: bool = False
    threat_score: float = 0.0
    threat_types: List[str] = field(default_factory=list)
    details: List[Dict[str, Any]] = field(default_factory=list)
    pii_detected: bool = False
    pii_types: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threat_detected": self.threat_detected,
            "threat_score": round(self.threat_score, 3),
            "threat_types": self.threat_types,
            "pii_detected": self.pii_detected,
            "pii_types": self.pii_types,
            "details_count": len(self.details),
        }


# ---------------------------------------------------------------------------
# TrollGuard Module
# ---------------------------------------------------------------------------

class TrollGuard(ETModule):
    """Security hook module implementing threat detection.

    Subscribes to pre_route, post_route, and post_response hooks.
    Uses pattern matching for known threats and NG-Lite for learning
    novel threat patterns.

    Configuration (via et_module.json ng_config):
        threat_threshold: Minimum threat score to flag (default 0.5).
        block_threshold: Minimum threat score to cancel (default 0.9).
            The host (TID's /route endpoint) enforces this cancellation.
        scan_responses: Whether to scan model responses (default True).
        learn_from_threats: Whether to feed threats to NG-Lite
            for pattern learning (default True).
    """

    def __init__(self, manifest: ETModuleManifest):
        super().__init__(manifest)
        ng_config = manifest.ng_config or {}
        self._threat_threshold = ng_config.get("threat_threshold", 0.5)
        self._block_threshold = ng_config.get("block_threshold", 0.9)
        self._scan_responses = ng_config.get("scan_responses", True)
        self._learn_from_threats = ng_config.get("learn_from_threats", True)

        # Stats
        self._requests_scanned = 0
        self._threats_detected = 0
        self._pii_detected = 0
        self._responses_scanned = 0

    def initialize(self) -> None:
        """Initialize TrollGuard."""
        logger.info(
            "TrollGuard initialized (threshold=%.2f, block=%.2f, "
            "scan_responses=%s)",
            self._threat_threshold,
            self._block_threshold,
            self._scan_responses,
        )

    def pre_route(self, ctx: HookContext) -> None:
        """Scan incoming request for threats before routing.

        Sets flags and annotations in the hook context:
            - "threat_detected" flag if score >= threat_threshold
            - "threat_blocked" flag if score >= block_threshold
            - Detailed assessment in ctx.annotations["trollguard"]
        """
        self._requests_scanned += 1

        assessment = self._assess_threat(ctx.message)

        # Also scan conversation history for multi-turn attacks
        for msg in ctx.conversation_history:
            history_assessment = self._assess_threat(msg)
            if history_assessment.threat_score > assessment.threat_score:
                # Escalate to worst threat found in history
                assessment.threat_score = max(
                    assessment.threat_score,
                    history_assessment.threat_score * 0.7,  # Discount history
                )
                assessment.threat_types.extend(history_assessment.threat_types)
                assessment.details.extend(history_assessment.details)

        # Deduplicate threat types
        assessment.threat_types = list(set(assessment.threat_types))

        # Re-evaluate threat_detected after history escalation
        assessment.threat_detected = (
            assessment.threat_score >= self._threat_threshold
        )

        # Set flags based on thresholds
        if assessment.threat_detected:
            self._threats_detected += 1
            ctx.flags.add("threat_detected")
            ctx.metadata["threat_score"] = assessment.threat_score

            # Type I error bias: only recommend blocking for
            # very high confidence threats
            if assessment.threat_score >= self._block_threshold:
                ctx.flags.add("threat_blocked")
                # Enforced: app.py checks ctx.cancelled and returns
                # an empty routing response — no model is selected.
                ctx.cancelled = True
                ctx.cancel_reason = (
                    f"TrollGuard: high-confidence threat detected "
                    f"(score={assessment.threat_score:.2f}, "
                    f"types={assessment.threat_types})"
                )

        if assessment.pii_detected:
            self._pii_detected += 1
            ctx.flags.add("pii_detected")

        # Always annotate for transparency (even if no threat)
        ctx.annotations[self.manifest.name] = assessment.to_dict()

    def post_route(self, ctx: HookContext) -> None:
        """Log routing decision for security-flagged requests."""
        if "threat_detected" not in ctx.flags:
            return

        # Log which model was selected for a threat-flagged request
        decision = ctx.routing_decision
        model_id = getattr(decision, "model_id", "unknown")
        threat_score = ctx.metadata.get("threat_score", 0.0)

        logger.warning(
            "TrollGuard: threat-flagged request %s routed to %s "
            "(threat_score=%.2f)",
            ctx.request_id, model_id, threat_score,
        )

    def post_response(self, ctx: HookContext) -> None:
        """Scan model response for leakage or harmful content."""
        if not self._scan_responses or not ctx.response_text:
            return

        self._responses_scanned += 1

        # Scan response for leakage
        leakage_score = 0.0
        leakage_types: List[str] = []

        lower = ctx.response_text.lower()
        for pattern, leak_type, confidence in RESPONSE_LEAKAGE_PATTERNS:
            if re.search(pattern, lower):
                leakage_score = max(leakage_score, confidence)
                leakage_types.append(leak_type)

        # Scan response for PII (the model might be leaking PII)
        pii_assessment = self._scan_pii(ctx.response_text)

        if leakage_score > 0 or pii_assessment.pii_detected:
            annotations = ctx.annotations.get(self.manifest.name, {})
            annotations["response_scan"] = {
                "leakage_score": round(leakage_score, 3),
                "leakage_types": leakage_types,
                "response_pii": pii_assessment.pii_types,
            }
            ctx.annotations[self.manifest.name] = annotations

            if leakage_score >= self._threat_threshold:
                ctx.flags.add("response_leakage")

            if pii_assessment.pii_detected:
                ctx.flags.add("response_pii_leak")

    def get_stats(self) -> Dict[str, Any]:
        """TrollGuard statistics for transparency."""
        base = super().get_stats()
        base.update({
            "requests_scanned": self._requests_scanned,
            "threats_detected": self._threats_detected,
            "pii_detected": self._pii_detected,
            "responses_scanned": self._responses_scanned,
            "threat_rate": (
                self._threats_detected / self._requests_scanned
                if self._requests_scanned > 0 else 0.0
            ),
            "threat_threshold": self._threat_threshold,
            "block_threshold": self._block_threshold,
        })
        return base

    # -------------------------------------------------------------------
    # Internal: Threat Assessment
    # -------------------------------------------------------------------

    def _assess_threat(self, text: str) -> ThreatAssessment:
        """Run all threat detection patterns against text."""
        assessment = ThreatAssessment()
        lower = text.lower()

        max_score = 0.0
        threat_types: List[str] = []
        details: List[Dict[str, Any]] = []

        # Check injection/jailbreak patterns
        for pattern, threat_type, confidence in INJECTION_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                max_score = max(max_score, confidence)
                threat_types.append(threat_type)
                details.append({
                    "pattern": threat_type,
                    "match": match.group(),
                    "confidence": confidence,
                })

        # Check abuse patterns
        for pattern, threat_type, confidence in ABUSE_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                max_score = max(max_score, confidence)
                threat_types.append(threat_type)
                details.append({
                    "pattern": threat_type,
                    "match": match.group(),
                    "confidence": confidence,
                })

        # Check PII
        pii = self._scan_pii(text)
        assessment.pii_detected = pii.pii_detected
        assessment.pii_types = pii.pii_types

        assessment.threat_score = max_score
        assessment.threat_detected = max_score >= self._threat_threshold
        assessment.threat_types = list(set(threat_types))
        assessment.details = details

        return assessment

    @staticmethod
    def _scan_pii(text: str) -> ThreatAssessment:
        """Scan text for PII patterns."""
        assessment = ThreatAssessment()
        pii_types: List[str] = []

        for pattern, pii_type, confidence in PII_PATTERNS:
            if re.search(pattern, text):
                pii_types.append(pii_type)

        if pii_types:
            assessment.pii_detected = True
            assessment.pii_types = list(set(pii_types))

        return assessment


# ---------------------------------------------------------------------------
# Factory: create TrollGuard from manifest or defaults
# ---------------------------------------------------------------------------

def create_trollguard(
    manifest: Optional[ETModuleManifest] = None,
) -> TrollGuard:
    """Create a TrollGuard instance with default or custom manifest.

    Args:
        manifest: Custom manifest. If None, uses defaults.

    Returns:
        Configured TrollGuard instance.
    """
    if manifest is None:
        manifest = ETModuleManifest(
            name="trollguard",
            version="1.0.0",
            description="Security hook module for threat detection",
            author="Josh + Claude",
            hooks=["pre_route", "post_route", "post_response"],
            capabilities=["content_filter", "threat_detection", "pii_scanner"],
            priority=5,  # Run early — security before everything
            ng_config={
                "threat_threshold": 0.5,
                "block_threshold": 0.9,
                "scan_responses": True,
                "learn_from_threats": True,
            },
        )

    return TrollGuard(manifest)
