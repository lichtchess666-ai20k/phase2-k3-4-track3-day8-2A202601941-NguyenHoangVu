"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState, Route

#: Classification result -> node that handles it. Kept as data (not if/elif) so a new
#: route only needs one line here plus the matching node registration in graph.py.
ROUTE_TO_NODE: dict[str, str] = {
    Route.SIMPLE.value: "answer",
    Route.TOOL.value: "tool",
    Route.MISSING_INFO.value: "clarify",
    Route.RISKY.value: "risky_action",
    Route.ERROR.value: "retry",
}

#: Fallback when the LLM returns a route we do not know. "answer" is the safe default:
#: it always terminates and never triggers a side effect.
DEFAULT_NODE = "answer"


def route_after_classify(state: AgentState) -> str:
    """Map the classified route to the next graph node."""
    route = str(state.get("route") or "")
    return ROUTE_TO_NODE.get(route, DEFAULT_NODE)


def route_after_evaluate(state: AgentState) -> str:
    """Decide if the tool result is satisfactory or needs another attempt.

    This is the 'done?' check that closes the retry loop — the key LangGraph
    advantage over a linear LCEL chain.
    """
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry the tool or escalate to the dead-letter node.

    Bounded on purpose: `attempt` is already incremented by retry_or_fallback_node,
    so once it reaches max_attempts the loop exits instead of spinning forever.
    """
    attempt = int(state.get("attempt") or 0)
    max_attempts = int(state.get("max_attempts") or 0)
    if attempt < max_attempts:
        return "tool"
    return "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Route on the human approval decision: approved runs the action, rejected asks back."""
    approval = state.get("approval")
    if isinstance(approval, dict):
        approved = bool(approval.get("approved", False))
    else:
        # Tolerate an ApprovalDecision object in case approval_node returns the model.
        approved = bool(getattr(approval, "approved", False))
    if approved:
        return "tool"
    return "clarify"
