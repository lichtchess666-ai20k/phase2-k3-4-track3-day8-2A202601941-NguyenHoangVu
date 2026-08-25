"""Cross-process crash-resume evidence for the persistence track.

The point is that the SQLite checkpoint outlives the process that wrote it.

    python scripts/persistence_demo.py start    # runs until the approval interrupt, exits
    python scripts/persistence_demo.py resume   # a NEW process finishes the same thread
    python scripts/persistence_demo.py history  # lists the checkpoints that made it possible

`start` sets LANGGRAPH_INTERRUPT=true, so approval_node calls interrupt(). LangGraph
persists the pending state and stops. Nothing is held in memory afterwards: `resume` opens
the same file, finds the paused thread, and injects the human decision with Command().
"""

from __future__ import annotations

import os
import sys

DB_PATH = "outputs/demo_checkpoints.sqlite"
THREAD_ID = "thread-persistence-demo"
QUERY = "Refund order 12345 and email the customer a confirmation"


def _graph():  # noqa: ANN202 - local helper, the compiled type is a langgraph internal
    from langgraph_agent_lab.graph import build_graph
    from langgraph_agent_lab.persistence import build_checkpointer

    return build_graph(checkpointer=build_checkpointer("sqlite", DB_PATH))


def cmd_start() -> None:
    os.environ["LANGGRAPH_INTERRUPT"] = "true"
    from langgraph_agent_lab.state import Route, Scenario, initial_state

    graph = _graph()
    scenario = Scenario(id="persistence-demo", query=QUERY, expected_route=Route.RISKY)
    state = initial_state(scenario)
    state["thread_id"] = THREAD_ID
    config = {"configurable": {"thread_id": THREAD_ID}}

    result = graph.invoke(state, config=config)
    snapshot = graph.get_state(config)

    print(f"[start] pid={os.getpid()} db={DB_PATH}")
    print(f"[start] route classified as: {result.get('route')}")
    print(f"[start] graph paused at nodes: {snapshot.next}")
    print(f"[start] interrupt payload: {[i.value for i in snapshot.interrupts]}")
    print(f"[start] final_answer so far: {result.get('final_answer')!r}  <- still unanswered")
    print("[start] process is about to exit; state lives only in the sqlite file now")


def cmd_resume() -> None:
    os.environ["LANGGRAPH_INTERRUPT"] = "true"
    from langgraph.types import Command

    graph = _graph()
    config = {"configurable": {"thread_id": THREAD_ID}}
    before = graph.get_state(config)
    if not before.next:
        print("[resume] nothing pending — run 'start' first (or delete the db to redo it)")
        return

    print(f"[resume] pid={os.getpid()} reopened {DB_PATH}")
    print(f"[resume] recovered a thread paused at: {before.next}")
    print(f"[resume] events already persisted: {len(before.values.get('events', []))}")

    decision = {"approved": True, "reviewer": "demo-human", "comment": "approved after resume"}
    result = graph.invoke(Command(resume=decision), config=config)

    print(f"[resume] approval recorded: {result.get('approval')}")
    print(f"[resume] tool results: {result.get('tool_results')}")
    print(f"[resume] final_answer: {result.get('final_answer')}")
    print(f"[resume] total events after completion: {len(result.get('events', []))}")


def cmd_history() -> None:
    graph = _graph()
    config = {"configurable": {"thread_id": THREAD_ID}}
    history = list(graph.get_state_history(config))
    print(f"[history] {len(history)} checkpoints for {THREAD_ID} (newest first)")
    for snapshot in history:
        checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id", "?")
        nodes = ",".join(snapshot.next) or "END"
        events = len(snapshot.values.get("events", []))
        print(f"  {checkpoint_id}  next={nodes:<14} events={events}")


COMMANDS = {"start": cmd_start, "resume": cmd_resume, "history": cmd_history}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in COMMANDS:
        print(f"usage: python scripts/persistence_demo.py [{'|'.join(COMMANDS)}]")
        raise SystemExit(1)
    COMMANDS[name]()
