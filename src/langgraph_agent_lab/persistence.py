"""Checkpointer adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

DEFAULT_SQLITE_PATH = "outputs/checkpoints.sqlite"


def _sqlite_path(database_url: str | None) -> Path:
    """Accept either a bare path or a sqlite:/// URL and return a filesystem path."""
    raw = database_url or DEFAULT_SQLITE_PATH
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///", "sqlite://"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    - "none"   : no persistence, the graph keeps state only for the duration of invoke()
    - "memory" : per-process persistence, enough for state-history replay inside one run
    - "sqlite" : durable persistence on disk, survives a process crash
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        path = _sqlite_path(database_url)
        # check_same_thread=False: LangGraph may touch the connection from a worker thread.
        conn = sqlite3.connect(str(path), check_same_thread=False)
        # WAL lets a reader (the resume process) work while a writer holds the db.
        conn.execute("PRAGMA journal_mode=WAL")
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")


def verify_state_history(graph: CompiledStateGraph, thread_id: str) -> bool:
    """Check that the checkpointer really recorded a replayable history for one thread.

    Returns True when more than one checkpoint exists and the earliest one can be read
    back — that is the property crash-resume and time-travel both depend on.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        history = list(graph.get_state_history(config))
    except Exception:  # noqa: BLE001 - "none" checkpointer, or a backend that cannot list
        return False
    if len(history) < 2:
        return False
    earliest = history[-1]
    return earliest.config.get("configurable", {}).get("checkpoint_id") is not None
