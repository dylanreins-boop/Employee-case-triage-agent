# Employee Case Triage Agent

An agent that triages a messy HR ticket inbox: it classifies each request, pulls
relevant policy context through a tool call, drafts a grounded reply — and
**escalates to a human queue whenever it shouldn't act autonomously**.

The interesting part is not the model call. It's the boundary around it: a
deterministic rule layer decides what the AI is allowed to finish on its own,
and that layer is code, not prompt.

## What it demonstrates

| Capability | Where |
|---|---|
| Agentic tool use (`search_policy` over a governed corpus) | `triage/agent.py`, `triage/policies.py` |
| Structured outputs (JSON schema constrained response, pydantic-validated) | `triage/models.py` |
| **Human-in-the-loop escalation with auditable reason codes** | `triage/guardrails.py`, `outputs/human_queue.jsonl` |
| Fail-closed error handling (refusal / truncation / bad JSON → escalate, never guess) | `triage/agent.py` `_anomaly_decision` |
| Graceful degradation without the model (rules-only mode) | `triage/rules_only.py` |
| Tests for the safety-critical logic, all offline | `tests/test_triage.py` |

## The escalation design (the part most agent demos skip)

The model returns a *recommendation* — category, confidence, citations, draft.
A separate guardrail layer then applies four rules, any one of which forces the
ticket into the human queue:

1. **Sensitive category** — harassment, medical/accommodation,
   termination/discipline, and legal matters escalate at *any* confidence.
   These are judgments an organization should never delegate to a language
   model, so the rule lives in code where the model can't be talked out of it.
2. **Confidence floor** — below 0.75, a fluent draft is a liability, not a
   deliverable.
3. **No policy grounding** — a confident answer with zero citations is a
   hallucination risk by definition.
4. **The model asked for a human** — the prompt tells Claude there's no reward
   for overconfidence (the rule engine will catch it anyway), which makes its
   self-reported escalations more honest, not less frequent.

Every escalation carries stable reason codes
(`sensitive_category_requires_human`, `confidence_below_floor`, …) so the queue
is auditable: you can answer "why did the agent punt this?" months later.

The demo money shot: **TKT-003** (a harassment report) produces a correct
classification and a sympathetic draft — and still lands in the human queue,
because no confidence score makes an AI the right first responder for that
conversation. Meanwhile **TKT-001** (PTO carry-over) auto-drafts a reply citing
`leave_policy § Carrying Over Unused PTO`.

## What this deliberately does *not* do

- **Send anything.** Auto-responses stop at draft files in `outputs/drafts/`.
- **Answer from the model's world knowledge.** Claims must be grounded in
  tool-returned policy text; the corpus not covering a topic is an escalation,
  not an invitation to improvise.
- **Trust the model's confidence blindly.** Confidence is one input to a rule
  engine, not the decision itself.

## Quick start

```bash
pip install -r requirements.txt

# Interactive UI (auto-detects credentials; falls back to rules-only mode)
streamlit run app.py

# With Anthropic credentials (ANTHROPIC_API_KEY or `ant auth login`):
python run_triage.py            # full agent: classify + search + draft
python run_triage.py --ticket TKT-003

# Without credentials — rules-only keyword triage, everything escalates
# with correct categories (graceful degradation, not an outage):
python run_triage.py --offline

python -m pytest tests/ -q      # offline; tests the guardrails, not the model
```

Outputs land in `outputs/drafts/*.md` (auto-responses awaiting a human send
decision) and `outputs/human_queue.jsonl` (escalations with reason codes).

## Architecture

```
tickets.json ─▶ agent loop (claude-opus-4-8) ──▶ TriageDecision ─▶ guardrails ─▶ drafts/
                 │  search_policy tool  ▲            (pydantic)      (code)   └▶ human_queue.jsonl
                 └──────▶ policy corpus ┘
```

- One `messages.create` loop per ticket with a strict `search_policy` tool and
  a JSON-schema-constrained final answer (`output_config.format`); adaptive
  thinking on; system prompt cached across tickets via `cache_control`.
- Numerical bounds the API schema can't express (confidence ∈ [0,1]) are
  enforced client-side with pydantic.
- The retriever is intentionally naive keyword scoring — swapping in embeddings
  changes nothing about the agent contract or the guardrails.
