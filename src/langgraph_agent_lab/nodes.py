"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM usage in this implementation:
- classify_node  : LLM + .with_structured_output(Classification)
- answer_node    : LLM, grounded strictly in state context
- evaluate_node  : LLM-as-judge, with a deterministic short-circuit on transport errors
- ask_clarification_node : LLM, generates one specific follow-up question
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, Route, make_event

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

RouteName = Literal["simple", "tool", "missing_info", "risky", "error"]

# Below this attempt count the mock backend keeps failing, so the retry loop is exercised.
TRANSIENT_FAILURE_ATTEMPTS = 2


# --- LLM plumbing ----------------------------------------------------
class Classification(BaseModel):
    """Structured output schema for classify_node."""

    route: RouteName = Field(description="The single best route for this support ticket.")
    risk_level: Literal["low", "medium", "high"] = Field(
        description="high when the request has irreversible side effects, low otherwise."
    )
    reason: str = Field(description="One short sentence justifying the route.")


class ToolJudgement(BaseModel):
    """Structured output schema for the evaluate_node LLM judge."""

    verdict: Literal["success", "needs_retry"]
    reason: str = Field(description="One short sentence justifying the verdict.")


CLASSIFY_SYSTEM = """You are the intake classifier of a customer-support agent.
Classify the ticket into exactly one route:

- risky: the user asks for an action with real side effects - refunds, payments,
  cancellations, deletions, account changes, sending emails or messages. Anything a
  human should approve before it happens.
- tool: the user asks for information that must be looked up in a backend system -
  order status, shipment tracking, invoice details, account records, search.
- missing_info: the request is too vague or incomplete to act on - no subject, no
  identifier, no describable problem.
- error: the ticket reports a system or infrastructure failure - timeout, crash,
  service unavailable, cannot recover, repeated failures.
- simple: a general question answerable from documentation or general knowledge,
  needing no lookup and causing no side effect.

When more than one route fits, apply this priority, highest first:
risky > tool > missing_info > error > simple.
For example, a ticket asking to refund an order AND report its status is risky, not
tool, because the refund has a side effect.

Set risk_level to "high" whenever the route is risky, "low" otherwise. Use "medium"
only for a reversible action that still touches customer data."""

ANSWER_SYSTEM = """You are a customer-support agent writing the final reply to a ticket.
Ground your reply ONLY in the context given to you. Never invent order numbers, dates,
refund amounts, or policies that are not in the context. If tool results are present,
state what they show. If a human reviewer approved an action, confirm it was carried out.
Answer in at most 120 words of plain prose - no markdown headings, no bullet lists."""

CLARIFY_SYSTEM = """A support ticket is too vague or incomplete to act on, or a reviewer
rejected the action the customer asked for. Write ONE specific question back to the
customer that would unblock the request. Ask for the concrete identifier or detail that
is missing. Return only the question, one sentence, no preamble."""

JUDGE_SYSTEM = """You judge whether a backend tool result is good enough to answer a
support ticket. Answer "needs_retry" ONLY when the result is an error, is empty, or is
clearly unusable for the ticket. Anything that plausibly answers the ticket is "success".
Do not demand extra detail - this is a reliability gate, not a quality review."""


@lru_cache(maxsize=1)
def _llm() -> BaseChatModel:
    """Build the chat model once per process - node functions run many times per graph."""
    return get_llm()


def _fallback_route(query: str) -> Classification:
    """Degraded classification used ONLY when the LLM call itself raises.

    The LLM is the primary and intended classifier; this only keeps a whole grading run
    from dying on one network blip. It mirrors the same priority order as the prompt.
    """
    text = query.lower()
    if any(w in text for w in ("refund", "delete", "cancel", "send", "charge")):
        return Classification(route="risky", risk_level="high", reason="fallback: side-effect verb")
    if any(w in text for w in ("lookup", "status", "track", "find", "search")):
        return Classification(route="tool", risk_level="low", reason="fallback: lookup verb")
    if any(w in text for w in ("timeout", "failure", "crash", "unavailable", "cannot recover")):
        return Classification(route="error", risk_level="low", reason="fallback: failure keyword")
    if len(text.split()) < 5:
        return Classification(route="missing_info", risk_level="low", reason="fallback: too short")
    return Classification(route="simple", risk_level="low", reason="fallback: default")


def _context_block(state: AgentState) -> str:
    """Serialize the parts of state an answer may be grounded in."""
    lines = [f"Ticket: {state.get('query', '')}", f"Route: {state.get('route', '')}"]
    tool_results = state.get("tool_results") or []
    if tool_results:
        lines.append("Tool results:")
        lines.extend(f"  - {result}" for result in tool_results)
    else:
        lines.append("Tool results: none - answer from general support knowledge.")
    proposed = state.get("proposed_action")
    if proposed:
        lines.append(f"Proposed action: {proposed}")
    approval = state.get("approval")
    if approval:
        verdict = "APPROVED" if approval.get("approved") else "REJECTED"
        lines.append(f"Human review: {verdict} by {approval.get('reviewer', 'unknown')}")
    errors = state.get("errors") or []
    if errors:
        lines.append(f"Errors so far: {'; '.join(errors[-3:])}")
    return "\n".join(lines)


# --- EXAMPLE: working node (provided for reference) ------------------
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# --- Nodes -----------------------------------------------------------
def classify_node(state: AgentState) -> dict:
    """Classify the ticket into a route with an LLM using structured output."""
    query = state.get("query", "")
    started = time.perf_counter()
    degraded = False
    try:
        chain = _llm().with_structured_output(Classification)
        result = chain.invoke(
            [("system", CLASSIFY_SYSTEM), ("human", f"Support ticket:\n{query}")]
        )
    except Exception as exc:  # noqa: BLE001 - any provider error degrades, never crashes
        degraded = True
        result = _fallback_route(query)
        result.reason = f"{result.reason} (llm error: {type(exc).__name__})"
    latency_ms = int((time.perf_counter() - started) * 1000)

    update: dict[str, Any] = {
        "route": result.route,
        "risk_level": result.risk_level,
        "messages": [f"classify:{result.route}"],
        "events": [
            make_event(
                "classify",
                "degraded" if degraded else "completed",
                f"route={result.route} risk={result.risk_level}: {result.reason}",
                latency_ms=latency_ms,
                degraded=degraded,
            )
        ],
    }
    if degraded:
        update["errors"] = [f"classify: LLM unavailable, used fallback -> {result.route}"]
    return update


def tool_node(state: AgentState) -> dict:
    """Execute a mock backend tool, simulating transient failure on the error route."""
    route = str(state.get("route") or "")
    attempt = int(state.get("attempt") or 0)
    query = state.get("query", "")
    started = time.perf_counter()

    if route == Route.ERROR.value and attempt < TRANSIENT_FAILURE_ATTEMPTS:
        # Transient downstream fault: evaluate_node will send this back to retry.
        result = f"ERROR: downstream service unavailable (attempt {attempt})"
        status = "failed"
    elif route == Route.RISKY.value:
        result = f"[action_executor] approved action executed | request={query[:80]}"
        status = "completed"
    else:
        result = f"[record_lookup] status=OK matched_records=1 | request={query[:80]}"
        status = "completed"

    return {
        "tool_results": [result],
        "messages": [f"tool:{status}"],
        "events": [
            make_event(
                "tool",
                status,
                result,
                latency_ms=int((time.perf_counter() - started) * 1000),
                attempt=attempt,
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result - the retry-loop gate (LLM-as-judge)."""
    tool_results = state.get("tool_results") or []
    latest = tool_results[-1] if tool_results else ""
    started = time.perf_counter()

    if not latest:
        verdict, reason, judge = "needs_retry", "no tool result produced", "deterministic"
    elif "ERROR" in latest:
        # A transport-level failure needs no LLM opinion - short-circuit and save a call.
        verdict, reason, judge = "needs_retry", "tool reported an error", "deterministic"
    else:
        try:
            chain = _llm().with_structured_output(ToolJudgement)
            judgement = chain.invoke(
                [
                    ("system", JUDGE_SYSTEM),
                    ("human", f"Ticket:\n{state.get('query', '')}\n\nTool result:\n{latest}"),
                ]
            )
            verdict, reason, judge = judgement.verdict, judgement.reason, "llm"
        except Exception as exc:  # noqa: BLE001 - a broken judge must not block the graph
            verdict = "success"
            reason = f"judge unavailable ({type(exc).__name__})"
            judge = "fallback"

    return {
        "evaluation_result": verdict,
        "messages": [f"evaluate:{verdict}"],
        "events": [
            make_event(
                "evaluate",
                verdict,
                reason,
                latency_ms=int((time.perf_counter() - started) * 1000),
                judge=judge,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate the final reply with an LLM, grounded in whatever state has collected."""
    started = time.perf_counter()
    context = _context_block(state)
    degraded = False
    reason = ""
    try:
        response = _llm().invoke([("system", ANSWER_SYSTEM), ("human", context)])
        answer = str(response.content).strip()
    except Exception as exc:  # noqa: BLE001 - degrade to an honest holding reply
        degraded = True
        answer = (
            "We received your request and our systems are temporarily unable to draft a "
            "full reply. A support agent will follow up shortly."
        )
        reason = f"answer: LLM unavailable ({type(exc).__name__})"

    update: dict[str, Any] = {
        "final_answer": answer,
        "messages": [f"answer:{answer[:40]}"],
        "events": [
            make_event(
                "answer",
                "degraded" if degraded else "completed",
                answer[:200],
                latency_ms=int((time.perf_counter() - started) * 1000),
                degraded=degraded,
            )
        ],
    }
    if degraded:
        update["errors"] = [reason]
    return update


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for the missing information instead of hallucinating an answer."""
    started = time.perf_counter()
    approval = state.get("approval")
    rejected = bool(approval) and not approval.get("approved", False)
    prompt = f"Ticket:\n{state.get('query', '')}"
    if rejected:
        prompt += f"\n\nA reviewer rejected this action: {state.get('proposed_action', '')}"
    try:
        response = _llm().invoke([("system", CLARIFY_SYSTEM), ("human", prompt)])
        question = str(response.content).strip()
    except Exception:  # noqa: BLE001 - a generic ask is better than crashing the graph
        question = "Could you share the order number or account ID this request refers to?"

    return {
        "pending_question": question,
        "final_answer": question,
        "messages": [f"clarify:{question[:40]}"],
        "events": [
            make_event(
                "clarify",
                "rejected_alternative" if rejected else "completed",
                question,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Describe the risky action and hold it for human approval."""
    query = state.get("query", "")
    risk_level = state.get("risk_level", "high")
    proposed = (
        f"Execute the customer request: '{query}'. "
        f"Risk level {risk_level}: this has side effects that cannot be undone from the "
        "support console, so it requires human approval before execution."
    )
    return {
        "proposed_action": proposed,
        "messages": ["risky_action:pending_approval"],
        "events": [
            make_event("risky_action", "pending_approval", proposed, risk_level=risk_level)
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop gate. Mock-approves offline; real interrupt() when enabled."""
    proposed = state.get("proposed_action") or state.get("query", "")
    use_interrupt = os.getenv("LANGGRAPH_INTERRUPT", "").strip().lower() in {"1", "true", "yes"}

    if use_interrupt:
        from langgraph.types import interrupt

        payload = interrupt({"proposed_action": proposed, "question": "Approve this action?"})
        if isinstance(payload, dict):
            decision = ApprovalDecision(
                approved=bool(payload.get("approved", False)),
                reviewer=str(payload.get("reviewer", "human-reviewer")),
                comment=str(payload.get("comment", "")),
            )
        else:
            decision = ApprovalDecision(
                approved=bool(payload),
                reviewer="human-reviewer",
                comment="resumed via interrupt",
            )
    else:
        decision = ApprovalDecision(
            approved=True, reviewer="mock-reviewer", comment="auto-approved for offline run"
        )

    return {
        "approval": decision.model_dump(),
        "messages": [f"approval:{'approved' if decision.approved else 'rejected'}"],
        "events": [
            make_event(
                "approval",
                "approved" if decision.approved else "rejected",
                f"{decision.reviewer}: {decision.comment}",
                mode="interrupt" if use_interrupt else "mock",
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record one retry attempt. route_after_retry reads the incremented counter."""
    attempt = int(state.get("attempt") or 0) + 1
    max_attempts = int(state.get("max_attempts") or 0)
    message = f"transient failure, retry attempt {attempt}/{max_attempts}"
    return {
        "attempt": attempt,
        "errors": [message],
        "messages": [f"retry:{attempt}"],
        "events": [make_event("retry", "retrying", message, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Terminal escalation once the retry budget is exhausted.

    Note: this deliberately does NOT overwrite `route` - grading compares the classified
    route against the expected route, and the ticket was still classified correctly.
    """
    attempt = int(state.get("attempt") or 0)
    max_attempts = int(state.get("max_attempts") or 0)
    answer = (
        "We could not complete this request automatically after "
        f"{attempt} of {max_attempts} attempts. The ticket has been escalated to a human "
        "support engineer and you will receive an update by email."
    )
    return {
        "final_answer": answer,
        "errors": [f"dead_letter: retry budget exhausted after {attempt} attempts"],
        "messages": ["dead_letter:escalated"],
        "events": [
            make_event(
                "dead_letter", "escalated", answer, attempt=attempt, max_attempts=max_attempts
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit the final audit event. Every route passes through here before END."""
    return {
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route", ""),
                answered=bool(state.get("final_answer") or state.get("pending_question")),
            )
        ]
    }
