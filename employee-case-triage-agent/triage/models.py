"""Data models shared by the agent, the guardrails, and the CLI."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    LEAVE_AND_PTO = "leave_and_pto"
    PAYROLL_AND_EXPENSES = "payroll_and_expenses"
    BENEFITS = "benefits"
    REMOTE_WORK = "remote_work"
    HARASSMENT_OR_MISCONDUCT = "harassment_or_misconduct"
    MEDICAL_OR_ACCOMMODATION = "medical_or_accommodation"
    TERMINATION_OR_DISCIPLINE = "termination_or_discipline"
    LEGAL_OR_COMPLIANCE = "legal_or_compliance"
    OTHER = "other"


class Ticket(BaseModel):
    id: str
    submitted_by: str
    subject: str
    body: str


class PolicyCitation(BaseModel):
    document: str
    section: str


class TriageDecision(BaseModel):
    """The model's structured verdict on one ticket.

    This is what Claude returns. It is a *recommendation* — the guardrails in
    guardrails.py decide what actually happens to it.
    """

    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    policy_citations: List[PolicyCitation] = Field(default_factory=list)
    draft_response: str
    escalate_recommended: bool
    escalation_reason: Optional[str] = None


class Outcome(str, Enum):
    AUTO_RESPOND = "auto_respond"
    ESCALATE = "escalate"


class TriageResult(BaseModel):
    """Final, guardrail-checked result for one ticket."""

    ticket_id: str
    outcome: Outcome
    decision: TriageDecision
    escalation_reasons: List[str] = Field(default_factory=list)
    mode: str = "agent"  # "agent" or "rules_only"


# JSON schema sent to the API for structured output. Hand-written rather than
# TriageDecision.model_json_schema() because structured outputs reject
# numerical constraints (ge/le on confidence) — those are validated
# client-side by pydantic instead.
TRIAGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [c.value for c in Category],
        },
        "confidence": {
            "type": "number",
            "description": (
                "Your honest confidence (0-1) that the category is right AND "
                "the draft response is fully supported by cited policy text. "
                "Do not inflate this: an escalation costs a few minutes of a "
                "human's time, a wrong answer sent to an employee costs trust."
            ),
        },
        "summary": {
            "type": "string",
            "description": "One or two sentences describing the request.",
        },
        "policy_citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document": {"type": "string"},
                    "section": {"type": "string"},
                },
                "required": ["document", "section"],
                "additionalProperties": False,
            },
        },
        "draft_response": {
            "type": "string",
            "description": (
                "A draft reply to the employee, grounded in the cited policy "
                "sections. If you are recommending escalation, write a short "
                "acknowledgement instead (never guess at policy)."
            ),
        },
        "escalate_recommended": {"type": "boolean"},
        "escalation_reason": {"type": ["string", "null"]},
    },
    "required": [
        "category",
        "confidence",
        "summary",
        "policy_citations",
        "draft_response",
        "escalate_recommended",
        "escalation_reason",
    ],
    "additionalProperties": False,
}
