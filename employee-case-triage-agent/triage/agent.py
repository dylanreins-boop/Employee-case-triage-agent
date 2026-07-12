"""The Claude-backed triage agent.

One agentic loop per ticket:
  - the model may call `search_policy` (strict tool) against the corpus,
  - its final turn is constrained to TRIAGE_OUTPUT_SCHEMA via structured
    outputs, then validated client-side with pydantic (which also enforces
    the 0-1 confidence bounds the API-side schema can't express).

Any anomaly — refusal, token cap, unparseable output — degrades to an
escalation, never to a made-up answer. The guardrails in guardrails.py are
applied by the caller on top of whatever this returns.
"""
from __future__ import annotations

import json
from typing import Tuple

import anthropic
from pydantic import ValidationError

from .guardrails import REASON_API_ANOMALY
from .models import Category, Ticket, TriageDecision, TRIAGE_OUTPUT_SCHEMA
from .policies import PolicyCorpus

MODEL = "claude-opus-4-8"
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are an HR case triage assistant. For each incoming employee ticket you:

1. Classify it into exactly one category.
2. Search the governed policy corpus with the search_policy tool before drafting anything. \
Search at least once; refine your query if the first results miss.
3. Draft a reply grounded ONLY in policy text the tool returned. Cite every document and \
section you relied on in policy_citations. Never state a policy detail the tool did not return.
4. Report an honest confidence score. Confidence means: the category is right AND every \
claim in the draft is supported by a citation. If the ticket is ambiguous, mixes topics, \
or the corpus doesn't cover it, say so with a low score.
5. Set escalate_recommended=true whenever a human should review before anything is sent — \
sensitive situations (harassment, medical, legal, discipline), angry or distressed tone, \
requests for exceptions to policy, or anything the policies don't clearly answer.

You are a triage layer, not the final authority. A separate rule engine will force-escalate \
sensitive categories and low-confidence results regardless of your recommendation, so there \
is no benefit to overstating confidence — and real harm in an ungrounded draft reaching an \
employee. When in doubt, recommend escalation and keep the draft to a brief acknowledgement."""

SEARCH_POLICY_TOOL = {
    "name": "search_policy",
    "description": (
        "Search the company HR policy corpus. Returns the most relevant policy "
        "sections as (document, section, text). Call this before drafting any "
        "reply; call it again with a refined query if results look irrelevant. "
        "An empty result means the corpus does not cover the topic."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords describing the policy topic, e.g. 'parental leave duration'",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def triage_ticket(
    client: anthropic.Anthropic, ticket: Ticket, corpus: PolicyCorpus
) -> Tuple[TriageDecision, int]:
    """Run the agent loop for one ticket.

    Returns (decision, tool_calls_made). Never raises on model-side anomalies —
    those come back as a low-confidence escalate decision so the pipeline keeps
    moving and the ticket lands in the human queue.
    """
    user_content = (
        f"Ticket {ticket.id} from {ticket.submitted_by}\n"
        f"Subject: {ticket.subject}\n\n{ticket.body}"
    )
    messages = [{"role": "user", "content": user_content}]
    tool_calls = 0

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[SEARCH_POLICY_TOOL],
            output_config={"format": {"type": "json_schema", "schema": TRIAGE_OUTPUT_SCHEMA}},
            messages=messages,
        )

        if response.stop_reason == "refusal":
            return _anomaly_decision("model refused the request"), tool_calls

        if response.stop_reason == "max_tokens":
            return _anomaly_decision("response truncated at token limit"), tool_calls

        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls += 1
                    hits = corpus.search(block.input["query"])
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps({"results": hits}),
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            continue

        # end_turn: structured output guarantees the text block is schema-valid JSON
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return TriageDecision.model_validate_json(text), tool_calls
        except ValidationError as exc:
            return _anomaly_decision(f"output failed validation: {exc.errors()[0]['msg']}"), tool_calls

    return _anomaly_decision(f"no final answer after {MAX_TOOL_ROUNDS} tool rounds"), tool_calls


def _anomaly_decision(detail: str) -> TriageDecision:
    """Fail closed: an API/parsing anomaly becomes an escalation, not a guess."""
    return TriageDecision(
        category=Category.OTHER,
        confidence=0.0,
        summary=f"Automatic triage failed: {detail}",
        policy_citations=[],
        draft_response=(
            "Thanks for reaching out — your request has been routed to the "
            "People Operations team for a personal response."
        ),
        escalate_recommended=True,
        escalation_reason=f"{REASON_API_ANOMALY}: {detail}",
    )
