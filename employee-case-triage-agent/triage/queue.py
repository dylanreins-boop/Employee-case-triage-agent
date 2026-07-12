"""Output sinks: the human escalation queue and the auto-response drafts.

Escalations go to a JSONL queue (one auditable record per line, stable reason
codes). Auto-responses are written as draft files — this demo deliberately
stops at drafting; actually sending mail is a decision left to a human or a
downstream system with its own controls.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import Outcome, Ticket, TriageResult


def write_result(outputs_dir: Path, ticket: Ticket, result: TriageResult) -> Path:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if result.outcome == Outcome.ESCALATE:
        return _append_to_queue(outputs_dir / "human_queue.jsonl", ticket, result)
    return _write_draft(outputs_dir / "drafts", ticket, result)


def _append_to_queue(queue_path: Path, ticket: Ticket, result: TriageResult) -> Path:
    record = {
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "ticket_id": ticket.id,
        "subject": ticket.subject,
        "category": result.decision.category.value,
        "confidence": result.decision.confidence,
        "escalation_reasons": result.escalation_reasons,
        "model_summary": result.decision.summary,
        "mode": result.mode,
    }
    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return queue_path


def _write_draft(drafts_dir: Path, ticket: Ticket, result: TriageResult) -> Path:
    drafts_dir.mkdir(parents=True, exist_ok=True)
    path = drafts_dir / f"{ticket.id}.md"
    citations = "\n".join(
        f"- {c.document} § {c.section}" for c in result.decision.policy_citations
    )
    path.write_text(
        f"# Draft reply — {ticket.id}: {ticket.subject}\n\n"
        f"**Category:** {result.decision.category.value}  \n"
        f"**Confidence:** {result.decision.confidence:.2f}\n\n"
        f"{result.decision.draft_response}\n\n"
        f"## Policy citations\n{citations}\n",
        encoding="utf-8",
    )
    return path
