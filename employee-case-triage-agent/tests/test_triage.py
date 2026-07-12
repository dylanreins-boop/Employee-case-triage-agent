"""Offline tests for the parts that must never be wrong: the guardrails.

None of these touch the API — the escalation rules are deterministic code and
are tested as such.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage import rules_only
from triage.guardrails import (
    ALWAYS_ESCALATE_CATEGORIES,
    CONFIDENCE_FLOOR,
    REASON_LOW_CONFIDENCE,
    REASON_MODEL_RECOMMENDED,
    REASON_NO_POLICY_GROUNDING,
    REASON_SENSITIVE_CATEGORY,
    apply_guardrails,
)
from triage.models import (
    Category,
    Outcome,
    PolicyCitation,
    Ticket,
    TriageDecision,
)
from triage.policies import PolicyCorpus
from triage.queue import write_result

ROOT = Path(__file__).parent.parent

TICKET = Ticket(id="T-1", submitted_by="test", subject="s", body="b")


def make_decision(**overrides) -> TriageDecision:
    base = dict(
        category=Category.LEAVE_AND_PTO,
        confidence=0.95,
        summary="test",
        policy_citations=[PolicyCitation(document="leave_policy", section="Annual Leave Entitlement")],
        draft_response="draft",
        escalate_recommended=False,
        escalation_reason=None,
    )
    base.update(overrides)
    return TriageDecision(**base)


# --- guardrails -----------------------------------------------------------

def test_confident_grounded_benign_ticket_auto_responds():
    result = apply_guardrails(TICKET, make_decision())
    assert result.outcome == Outcome.AUTO_RESPOND
    assert result.escalation_reasons == []


@pytest.mark.parametrize("category", sorted(ALWAYS_ESCALATE_CATEGORIES, key=lambda c: c.value))
def test_sensitive_categories_escalate_even_at_full_confidence(category):
    decision = make_decision(category=category, confidence=1.0)
    result = apply_guardrails(TICKET, decision)
    assert result.outcome == Outcome.ESCALATE
    assert REASON_SENSITIVE_CATEGORY in result.escalation_reasons


def test_low_confidence_escalates():
    decision = make_decision(confidence=CONFIDENCE_FLOOR - 0.01)
    result = apply_guardrails(TICKET, decision)
    assert result.outcome == Outcome.ESCALATE
    assert REASON_LOW_CONFIDENCE in result.escalation_reasons


def test_missing_citations_escalates_despite_high_confidence():
    decision = make_decision(policy_citations=[])
    result = apply_guardrails(TICKET, decision)
    assert result.outcome == Outcome.ESCALATE
    assert REASON_NO_POLICY_GROUNDING in result.escalation_reasons


def test_model_escalation_recommendation_is_honored():
    decision = make_decision(escalate_recommended=True, escalation_reason="angry tone")
    result = apply_guardrails(TICKET, decision)
    assert result.outcome == Outcome.ESCALATE
    assert REASON_MODEL_RECOMMENDED in result.escalation_reasons


def test_multiple_triggered_rules_all_recorded():
    decision = make_decision(
        category=Category.HARASSMENT_OR_MISCONDUCT,
        confidence=0.3,
        policy_citations=[],
    )
    result = apply_guardrails(TICKET, decision)
    assert set(result.escalation_reasons) >= {
        REASON_SENSITIVE_CATEGORY,
        REASON_LOW_CONFIDENCE,
        REASON_NO_POLICY_GROUNDING,
    }


# --- policy retrieval -----------------------------------------------------

@pytest.fixture(scope="module")
def corpus() -> PolicyCorpus:
    return PolicyCorpus.load(ROOT / "policies")


def test_corpus_loads_sections(corpus):
    assert len(corpus.sections) >= 15


def test_search_finds_pto_carryover(corpus):
    hits = corpus.search("carry over unused vacation days next year")
    assert hits, "expected at least one hit"
    assert hits[0]["document"] == "leave_policy"
    assert "carry" in hits[0]["section"].lower()


def test_search_finds_per_diem(corpus):
    hits = corpus.search("per diem meal rate international travel")
    assert any(h["section"] == "Per Diem Rates" for h in hits)


def test_search_unknown_topic_returns_little_or_nothing(corpus):
    hits = corpus.search("cryptocurrency trading desk zebra")
    assert all("zebra" not in h["text"].lower() for h in hits)


# --- rules-only fallback --------------------------------------------------

def test_rules_only_classifies_harassment():
    ticket = Ticket(
        id="T-2", submitted_by="x", subject="My manager keeps making me uncomfortable",
        body="He makes inappropriate comments and I feel bullied.",
    )
    decision = rules_only.triage_ticket(ticket)
    assert decision.category == Category.HARASSMENT_OR_MISCONDUCT


def test_rules_only_never_auto_responds():
    """The fallback can't verify policy grounding, so nothing it produces may ship."""
    for subject, body in [
        ("PTO question", "How many vacation days do I get?"),
        ("Gym", "Is my gym membership covered as a benefit?"),
        ("random", "completely unrelated question"),
    ]:
        ticket = Ticket(id="T-3", submitted_by="x", subject=subject, body=body)
        result = apply_guardrails(ticket, rules_only.triage_ticket(ticket), mode="rules_only")
        assert result.outcome == Outcome.ESCALATE


def test_sensitive_keywords_beat_benign_ones():
    ticket = Ticket(
        id="T-4", submitted_by="x", subject="Expenses and something else",
        body="I want to claim expenses but also my teammate keeps harassing me.",
    )
    category, _ = rules_only.classify(ticket)
    assert category == Category.HARASSMENT_OR_MISCONDUCT


# --- queue outputs --------------------------------------------------------

def test_escalation_written_to_queue(tmp_path):
    decision = make_decision(category=Category.LEGAL_OR_COMPLIANCE)
    result = apply_guardrails(TICKET, decision)
    path = write_result(tmp_path, TICKET, result)
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["ticket_id"] == "T-1"
    assert REASON_SENSITIVE_CATEGORY in record["escalation_reasons"]


def test_auto_response_written_as_draft(tmp_path):
    result = apply_guardrails(TICKET, make_decision())
    path = write_result(tmp_path, TICKET, result)
    assert path.suffix == ".md"
    text = path.read_text(encoding="utf-8")
    assert "leave_policy" in text
