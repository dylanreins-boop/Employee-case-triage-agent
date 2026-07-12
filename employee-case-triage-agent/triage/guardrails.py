"""Deterministic escalation rules that sit OUTSIDE the model.

The model recommends; this module decides. The rules here are intentionally
code, not prompt instructions, because they encode judgments the organization
should never delegate to a language model:

1. Some categories always get a human, no matter how confident the model is.
2. A confident answer with no policy grounding is a hallucination risk, not
   an answer.
3. Below a confidence floor, a plausible-sounding draft is more dangerous
   than a queue entry.

Each rule appends a machine-readable reason code so the human queue can be
audited later ("why did the agent punt this?").
"""
from __future__ import annotations

from typing import List

from .models import Category, Outcome, TriageDecision, TriageResult, Ticket

# Categories where an autonomous AI reply is inappropriate regardless of
# model confidence: legally sensitive, safety-relevant, or high-stakes for
# the individual employee.
ALWAYS_ESCALATE_CATEGORIES = {
    Category.HARASSMENT_OR_MISCONDUCT,
    Category.MEDICAL_OR_ACCOMMODATION,
    Category.TERMINATION_OR_DISCIPLINE,
    Category.LEGAL_OR_COMPLIANCE,
}

CONFIDENCE_FLOOR = 0.75

# Reason codes (stable strings — the queue file is meant to be grep-able)
REASON_SENSITIVE_CATEGORY = "sensitive_category_requires_human"
REASON_LOW_CONFIDENCE = "confidence_below_floor"
REASON_NO_POLICY_GROUNDING = "no_policy_citations"
REASON_MODEL_RECOMMENDED = "model_recommended_escalation"
REASON_API_ANOMALY = "api_anomaly"
REASON_RULES_ONLY_MODE = "rules_only_mode_cannot_verify"


def apply_guardrails(ticket: Ticket, decision: TriageDecision, mode: str = "agent") -> TriageResult:
    """Turn a model recommendation into a final outcome.

    Order matters only for readability — any single triggered rule escalates.
    """
    reasons: List[str] = []

    if decision.category in ALWAYS_ESCALATE_CATEGORIES:
        reasons.append(REASON_SENSITIVE_CATEGORY)

    if decision.confidence < CONFIDENCE_FLOOR:
        reasons.append(REASON_LOW_CONFIDENCE)

    if not decision.policy_citations:
        reasons.append(REASON_NO_POLICY_GROUNDING)

    if decision.escalate_recommended:
        reasons.append(REASON_MODEL_RECOMMENDED)

    outcome = Outcome.ESCALATE if reasons else Outcome.AUTO_RESPOND
    return TriageResult(
        ticket_id=ticket.id,
        outcome=outcome,
        decision=decision,
        escalation_reasons=reasons,
        mode=mode,
    )
