"""Run the triage pipeline over the fake ticket inbox.

Usage:
    python run_triage.py                 # agent mode (needs Anthropic credentials)
    python run_triage.py --offline      # rules-only mode, no API calls
    python run_triage.py --ticket TKT-003
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from triage import rules_only
from triage.guardrails import apply_guardrails
from triage.models import Outcome, Ticket
from triage.policies import PolicyCorpus
from triage.queue import write_result

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"


def load_tickets(only_id: str | None) -> list[Ticket]:
    raw = json.loads((ROOT / "data" / "tickets.json").read_text(encoding="utf-8"))
    tickets = [Ticket.model_validate(t) for t in raw]
    if only_id:
        tickets = [t for t in tickets if t.id == only_id]
        if not tickets:
            sys.exit(f"No ticket with id {only_id}")
    return tickets


def make_agent_runner(corpus: PolicyCorpus):
    """Build the agent-mode runner, or explain why we can't and return None."""
    try:
        import anthropic
    except ImportError:
        print("[!] anthropic package not installed -> falling back to --offline mode\n")
        return None

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant auth profile

    from triage.agent import triage_ticket

    def run(ticket: Ticket):
        try:
            decision, tool_calls = triage_ticket(client, ticket, corpus)
            return decision, "agent", tool_calls
        except anthropic.AuthenticationError:
            _exit_no_credentials()
        except TypeError as exc:
            # SDK raises TypeError at request time when no auth method resolves
            if "authentication" in str(exc).lower():
                _exit_no_credentials()
            raise

    return run


def _exit_no_credentials() -> None:
    sys.exit(
        "Anthropic credentials missing or invalid. Set ANTHROPIC_API_KEY "
        "(or run `ant auth login`), or use --offline for rules-only mode."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Employee case triage agent")
    parser.add_argument("--offline", action="store_true", help="rules-only mode, no API calls")
    parser.add_argument("--ticket", help="process a single ticket id")
    parser.add_argument("--fresh", action="store_true", help="clear outputs/ before running")
    args = parser.parse_args()

    if args.fresh and OUTPUTS.exists():
        shutil.rmtree(OUTPUTS)

    corpus = PolicyCorpus.load(ROOT / "policies")
    tickets = load_tickets(args.ticket)

    agent_run = None if args.offline else make_agent_runner(corpus)

    print(f"{'ticket':<9} {'category':<26} {'conf':<5} {'outcome':<13} reasons")
    print("-" * 90)

    escalated = 0
    for ticket in tickets:
        if agent_run:
            decision, mode, _ = agent_run(ticket)
        else:
            decision, mode = rules_only.triage_ticket(ticket), "rules_only"

        result = apply_guardrails(ticket, decision, mode=mode)
        write_result(OUTPUTS, ticket, result)

        if result.outcome == Outcome.ESCALATE:
            escalated += 1
        reasons = ", ".join(result.escalation_reasons) or "-"
        print(
            f"{ticket.id:<9} {decision.category.value:<26} "
            f"{decision.confidence:<5.2f} {result.outcome.value:<13} {reasons}"
        )

    print("-" * 90)
    print(
        f"{len(tickets)} tickets: {len(tickets) - escalated} auto-drafted, "
        f"{escalated} escalated to the human queue"
    )
    print(f"Drafts:  {OUTPUTS / 'drafts'}")
    print(f"Queue:   {OUTPUTS / 'human_queue.jsonl'}")


if __name__ == "__main__":
    main()
