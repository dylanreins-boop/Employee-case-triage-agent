"""Streamlit UI for the employee case triage agent.

    streamlit run app.py

Same pipeline as run_triage.py — Ticket -> (agent | rules-only) -> guardrails —
rendered interactively. Results live in session state; nothing is written to
outputs/ from the UI.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from triage import rules_only
from triage.guardrails import (
    ALWAYS_ESCALATE_CATEGORIES,
    CONFIDENCE_FLOOR,
    apply_guardrails,
)
from triage.models import Outcome, Ticket, TriageResult
from triage.policies import PolicyCorpus

ROOT = Path(__file__).parent

st.set_page_config(page_title="Employee Case Triage Agent", page_icon="🗂️", layout="wide")


# --- cached resources -------------------------------------------------------

@st.cache_resource
def load_corpus() -> PolicyCorpus:
    return PolicyCorpus.load(ROOT / "policies")


@st.cache_data
def load_tickets() -> list[Ticket]:
    raw = json.loads((ROOT / "data" / "tickets.json").read_text(encoding="utf-8"))
    return [Ticket.model_validate(t) for t in raw]


@st.cache_resource
def get_client():
    """Anthropic client, or None if the SDK/credentials are unavailable.

    Construction succeeds even without credentials (the SDK only errors at
    request time), so explicitly check that some auth method resolved.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
        has_auth = any(
            getattr(client, attr, None) for attr in ("api_key", "auth_token", "credentials")
        )
        return client if has_auth else None
    except Exception:
        return None


# --- triage ------------------------------------------------------------------

def run_triage(ticket: Ticket, use_agent: bool) -> TriageResult:
    corpus = load_corpus()
    if use_agent:
        import anthropic
        from triage.agent import triage_ticket

        client = get_client()
        try:
            decision, _ = triage_ticket(client, ticket, corpus)
            return apply_guardrails(ticket, decision, mode="agent")
        except (anthropic.AuthenticationError, TypeError) as exc:
            st.error(
                "Anthropic credentials missing or invalid — set `ANTHROPIC_API_KEY` "
                "in the environment before launching Streamlit, or switch to "
                f"rules-only mode in the sidebar. ({exc.__class__.__name__})"
            )
            st.stop()
    decision = rules_only.triage_ticket(ticket)
    return apply_guardrails(ticket, decision, mode="rules_only")


def render_result(result: TriageResult) -> None:
    d = result.decision
    left, right = st.columns([1, 2])
    with left:
        if result.outcome == Outcome.AUTO_RESPOND:
            st.success("✅ Auto-respond — draft ready for send approval")
        else:
            st.warning("🙋 Escalated to the human queue")
        st.metric("Category", d.category.value)
        st.progress(d.confidence, text=f"Model confidence: {d.confidence:.2f}")
        st.caption(f"Mode: `{result.mode}`")
        if result.escalation_reasons:
            st.markdown("**Escalation reasons**")
            for reason in result.escalation_reasons:
                st.markdown(f"- `{reason}`")
    with right:
        st.markdown(f"**Summary:** {d.summary}")
        st.markdown("**Draft reply**")
        st.info(d.draft_response)
        if d.policy_citations:
            st.markdown("**Policy citations**")
            for c in d.policy_citations:
                st.markdown(f"- `{c.document}` § {c.section}")
        else:
            st.caption("No policy citations — the no-grounding guardrail blocks auto-response.")


# --- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.title("🗂️ Case Triage")
    client_available = get_client() is not None
    mode = st.radio(
        "Triage mode",
        ["Claude agent", "Rules-only (offline)"],
        index=0 if client_available else 1,
        help="Agent mode classifies, searches policy, and drafts with Claude. "
        "Rules-only is the keyword fallback — nothing it produces auto-ships.",
    )
    use_agent = mode == "Claude agent"
    if use_agent and not client_available:
        st.warning("No Anthropic SDK/credentials detected — runs will fail until "
                   "`ANTHROPIC_API_KEY` is set. Rules-only mode always works.")

    st.divider()
    st.markdown("**Guardrails (code, not prompt)**")
    st.markdown(
        f"- Confidence floor: **{CONFIDENCE_FLOOR}**\n"
        + "- Always-human categories:\n"
        + "\n".join(f"  - `{c.value}`" for c in sorted(ALWAYS_ESCALATE_CATEGORIES, key=lambda c: c.value))
    )
    st.caption(
        "The model recommends; these rules decide. Any triggered rule routes "
        "the ticket to a human regardless of model confidence."
    )

# --- main --------------------------------------------------------------------

st.title("Employee Case Triage Agent")
st.caption(
    "Classifies HR tickets, grounds draft replies in policy via tool use, and "
    "escalates anything an AI shouldn't finish on its own."
)

results: dict[str, TriageResult] = st.session_state.setdefault("results", {})

inbox_tab, custom_tab, queue_tab = st.tabs(["📥 Inbox", "✍️ New ticket", "🙋 Human queue"])

with inbox_tab:
    tickets = load_tickets()
    if st.button(f"Triage all {len(tickets)} tickets", type="primary"):
        progress = st.progress(0.0)
        for i, ticket in enumerate(tickets):
            results[ticket.id] = run_triage(ticket, use_agent)
            progress.progress((i + 1) / len(tickets))
        progress.empty()

    for ticket in tickets:
        done = ticket.id in results
        icon = (
            ("✅" if results[ticket.id].outcome == Outcome.AUTO_RESPOND else "🙋")
            if done
            else "▫️"
        )
        with st.expander(f"{icon} **{ticket.id}** — {ticket.subject}", expanded=False):
            st.markdown(f"*From `{ticket.submitted_by}`:*")
            st.markdown(f"> {ticket.body}")
            if st.button("Triage this ticket", key=f"btn_{ticket.id}"):
                with st.spinner("Triaging…"):
                    results[ticket.id] = run_triage(ticket, use_agent)
            if ticket.id in results:
                st.divider()
                render_result(results[ticket.id])

with custom_tab:
    with st.form("custom_ticket"):
        subject = st.text_input("Subject")
        body = st.text_area("Message", height=150)
        submitted = st.form_submit_button("Triage", type="primary")
    if submitted:
        if not subject.strip() or not body.strip():
            st.error("Subject and message are both required.")
        else:
            n = st.session_state.get("custom_count", 0) + 1
            st.session_state["custom_count"] = n
            ticket = Ticket(
                id=f"CUSTOM-{n:03d}", submitted_by="you", subject=subject.strip(), body=body.strip()
            )
            with st.spinner("Triaging…"):
                results[ticket.id] = run_triage(ticket, use_agent)
            render_result(results[ticket.id])

with queue_tab:
    escalated = [r for r in results.values() if r.outcome == Outcome.ESCALATE]
    drafted = [r for r in results.values() if r.outcome == Outcome.AUTO_RESPOND]
    st.markdown(
        f"**{len(escalated)}** escalated · **{len(drafted)}** auto-drafted "
        f"(of {len(results)} triaged)"
    )
    if escalated:
        st.dataframe(
            [
                {
                    "ticket": r.ticket_id,
                    "category": r.decision.category.value,
                    "confidence": round(r.decision.confidence, 2),
                    "reasons": ", ".join(r.escalation_reasons),
                    "summary": r.decision.summary,
                }
                for r in escalated
            ],
            width="stretch",
        )
    else:
        st.caption("Nothing in the queue yet — triage some tickets in the Inbox tab.")
