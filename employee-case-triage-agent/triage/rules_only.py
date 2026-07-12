"""Offline fallback triage — no model, no API key.

Runs the same pipeline shape (Ticket -> TriageDecision -> guardrails) using a
keyword classifier. It is honest about its limits: it never drafts a policy
answer (it can't verify one), so everything it produces carries no citations
and therefore escalates via the no-grounding guardrail. The value is that the
queue is still correctly triaged by category when the model is unavailable —
graceful degradation instead of an outage.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .guardrails import REASON_RULES_ONLY_MODE
from .models import Category, Ticket, TriageDecision

# Highest-priority patterns first: a ticket mentioning both "expenses" and
# "harassment" must land in the sensitive bucket.
_KEYWORD_MAP: List[Tuple[Category, List[str]]] = [
    (Category.HARASSMENT_OR_MISCONDUCT, ["harass", "bully", "hostile", "inappropriate comment", "misconduct", "retaliat"]),
    (Category.MEDICAL_OR_ACCOMMODATION, ["medical", "disability", "accommodation", "injury", "mental health", "sick leave", "fmla"]),
    (Category.TERMINATION_OR_DISCIPLINE, ["fired", "terminat", "layoff", "severance", "pip", "written warning", "dismiss"]),
    (Category.LEGAL_OR_COMPLIANCE, ["lawyer", "legal", "lawsuit", "whistleblow", "compliance", "discriminat", "subpoena"]),
    (Category.LEAVE_AND_PTO, ["pto", "vacation", "leave", "holiday", "time off", "parental", "carry over"]),
    (Category.PAYROLL_AND_EXPENSES, ["payroll", "paycheck", "salary", "expense", "reimburs", "per diem", "overtime"]),
    (Category.BENEFITS, ["benefit", "insurance", "401k", "pension", "dental", "gym", "wellness"]),
    (Category.REMOTE_WORK, ["remote", "work from home", "wfh", "hybrid", "relocat", "home office"]),
]


def classify(ticket: Ticket) -> Tuple[Category, float]:
    text = f"{ticket.subject} {ticket.body}".lower()
    for category, keywords in _KEYWORD_MAP:
        hits = sum(1 for kw in keywords if kw in text)
        if hits:
            # keyword matching is crude; cap confidence well below the
            # auto-respond floor so nothing ships without a human
            return category, min(0.4 + 0.1 * hits, 0.7)
    return Category.OTHER, 0.2


def triage_ticket(ticket: Ticket) -> TriageDecision:
    category, confidence = classify(ticket)
    return TriageDecision(
        category=category,
        confidence=confidence,
        summary=f"[rules-only] keyword-classified as {category.value}",
        policy_citations=[],  # a keyword matcher cannot verify grounding
        draft_response=(
            "Thanks for reaching out — your request has been received and "
            "routed to the People Operations team."
        ),
        escalate_recommended=True,
        escalation_reason=REASON_RULES_ONLY_MODE,
    )
