"""Report generation helper.

`render_report` produces the full lab report: the numeric sections are derived from the
MetricsReport of the run that just finished, the prose sections describe the architecture
that produced those numbers. Regenerating the report is therefore idempotent — rerunning
`run-scenarios` refreshes the tables without losing the write-up.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .metrics import MetricsReport

STUDENT_NAME = "Nguyen Hoang Vu"
STUDENT_ID = "2A202601941"

ARCHITECTURE = """The workflow is a single `StateGraph(AgentState)` with 11 nodes and one
terminal choke point.

```text
START -> intake -> classify -> [route_after_classify]
  simple       -> answer -> finalize -> END
  tool         -> tool -> evaluate -> [route_after_evaluate]
                            success     -> answer -> finalize -> END
                            needs_retry -> retry -> [route_after_retry]
                                                      attempt < max -> tool
                                                      exhausted     -> dead_letter -> finalize
  missing_info -> clarify -> finalize -> END
  risky        -> risky_action -> approval -> [route_after_approval]
                                                approved -> tool -> evaluate -> ...
                                                rejected -> clarify -> finalize -> END
  error        -> retry -> [route_after_retry] -> ...
```

Four conditional edges carry every decision, and each one is a pure function of state in
`routing.py` — nodes never choose their own successor. `finalize` is the only edge into
`END`, so every route, including the dead-letter path, leaves a complete audit trail.

**LLM integration.** `classify_node` calls the model through
`.with_structured_output(Classification)`, a Pydantic schema whose `route` field is a
`Literal` of the five valid routes — an out-of-vocabulary answer fails validation instead
of leaking into the graph. `answer_node` receives a serialized context block (ticket, route,
tool results, proposed action, approval verdict) and is instructed to ground its reply in
that block only. `evaluate_node` is an LLM-as-judge with a deterministic short-circuit: a
tool result containing `ERROR` is failed without spending an API call, and only genuine
results are sent to the judge. Every LLM call degrades to a documented fallback on
exception rather than aborting the run."""

STATE_SCHEMA = """| Field | Reducer | Why |
|---|---|---|
| `messages` | append (`operator.add`) | running trace of what each node did |
| `tool_results` | append | the retry loop needs every attempt, not just the last |
| `errors` | append | failures must accumulate for the report and dead-letter decision |
| `events` | append | audit log; metrics count retries and approvals from it |
| `route` | overwrite | only the current classification matters |
| `risk_level` | overwrite | derived from the same classification call |
| `attempt` | overwrite | a counter, not a history |
| `max_attempts` | overwrite | per-scenario budget, set once at intake |
| `final_answer` | overwrite | last writer wins |
| `evaluation_result` | overwrite | **must** overwrite — see below |
| `pending_question` | overwrite | one open question at a time |
| `proposed_action` | overwrite | one action awaits approval at a time |
| `approval` | overwrite | stored as a plain dict so checkpoints stay JSON-serializable |

The reducer choice on `evaluation_result` is load-bearing. `route_after_evaluate` compares
it to `"needs_retry"`; had it been declared append-only it would become
`["needs_retry", "success"]` after the first retry, the comparison would never match again,
and the loop would break. Conversely `events` must append, because `metric_from_state`
derives `retry_count` and `interrupt_count` by counting events whose `node` is `retry` or
`approval` — overwriting would erase the evidence."""

FAILURE_ANALYSIS = """**1. Transient tool failure (exercised by S05 and S07).** `tool_node`
returns a string containing `ERROR` while the attempt counter is below its threshold.
`evaluate_node` detects it deterministically and routes to `retry`, which increments
`attempt` *before* `route_after_retry` reads it. The loop has two distinct exits: S05
(`max_attempts=3`) escapes by eventually succeeding, S07 (`max_attempts=1`) escapes by
exhausting the budget and landing in `dead_letter`. Without the `attempt < max_attempts`
bound the error route would spin until LangGraph's recursion limit aborted the run.

`dead_letter_node` deliberately does not overwrite `route`. Grading compares the classified
route against the expected route, and a ticket that failed to execute was still classified
correctly — rewriting the route to `dead_letter` would report a routing bug that did not
happen.

**2. Risky action executed without approval.** The `risky` route cannot reach `tool`
directly: the only edges into it from that branch are `risky_action -> approval` and then
the conditional `route_after_approval`. That function reads `approval.approved` and treats
a missing or falsy decision as a rejection, sending the request to `clarify` instead of
executing it. The gate fails closed, so a crash between `risky_action` and `approval`
resumes into the approval step rather than into execution.

**3. LLM provider outage.** Every model call is wrapped. `classify_node` falls back to a
priority-ordered heuristic, `answer_node` emits an honest holding reply, the judge defaults
to `success` so a broken judge cannot manufacture infinite retries. Each degradation writes
to `errors` and stamps the event `degraded`, so a run that silently lost its LLM is visible
in the metrics rather than looking like a clean pass."""

IMPROVEMENT_PLAN = """With one more day, in order:

1. **Replace the mock tool with a real backend client** behind a timeout and a circuit
   breaker. The retry loop is correct but currently retries a simulation; real transient
   faults need jitter and backoff between attempts, which the graph does not yet apply.
2. **Move approval out of process.** `approval_node` mock-approves unless
   `LANGGRAPH_INTERRUPT=true`. Production needs the interrupt path plus a durable queue so a
   reviewer can approve hours later against the SQLite/Postgres checkpoint.
3. **Cost and latency budget per scenario.** Node latency is already recorded on every
   event; aggregating it into `ScenarioMetric.latency_ms` and adding token counts would turn
   the metrics file into a regression gate rather than a correctness check.
4. **Golden-set evaluation for the classifier.** Route accuracy is measured on seven
   scenarios. A larger labelled set, run on every prompt change, is what keeps the priority
   ordering (`risky > tool > missing_info > error > simple`) from silently regressing."""


def _summary_table(metrics: MetricsReport) -> str:
    rows = [
        ("Total scenarios", str(metrics.total_scenarios)),
        ("Success rate", f"{metrics.success_rate:.0%}"),
        ("Avg nodes visited", f"{metrics.avg_nodes_visited:.2f}"),
        ("Total retries", str(metrics.total_retries)),
        ("Total interrupts (approvals)", str(metrics.total_interrupts)),
        ("Resume / history verified", "yes" if metrics.resume_success else "no"),
    ]
    lines = ["| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def _scenario_table(metrics: MetricsReport) -> str:
    lines = [
        "| Scenario | Expected route | Actual route | Success | Nodes | Retries "
        "| Interrupts | Approval required | Approval seen |",
        "|---|---|---|:---:|---:|---:|---:|:---:|:---:|",
    ]
    for item in metrics.scenario_metrics:
        lines.append(
            f"| `{item.scenario_id}` | {item.expected_route} | {item.actual_route or '—'} "
            f"| {'✅' if item.success else '❌'} | {item.nodes_visited} | {item.retry_count} "
            f"| {item.interrupt_count} | {'yes' if item.approval_required else 'no'} "
            f"| {'yes' if item.approval_observed else 'no'} |"
        )
    return "\n".join(lines)


def _observations(metrics: MetricsReport) -> str:
    failed = [item.scenario_id for item in metrics.scenario_metrics if not item.success]
    retried = [
        f"`{item.scenario_id}` ({item.retry_count})"
        for item in metrics.scenario_metrics
        if item.retry_count
    ]
    approved = [
        f"`{item.scenario_id}`" for item in metrics.scenario_metrics if item.approval_observed
    ]

    lines = []
    if failed:
        lines.append(
            f"- {len(failed)} scenario(s) did not meet the success criteria: "
            + ", ".join(f"`{sid}`" for sid in failed)
            + "."
        )
    else:
        lines.append(
            "- Every scenario reached its expected route and produced either a final answer "
            "or a pending clarification question."
        )
    lines.append(
        f"- Retries came from {', '.join(retried) if retried else 'no scenario'} — the retry "
        "loop only fires on the error route, where the mock tool reports a transient fault."
        if retried
        else "- No scenario needed a retry."
    )
    lines.append(
        f"- Human approval was recorded for {', '.join(approved)}."
        if approved
        else "- No scenario required human approval."
    )
    lines.append(
        f"- Average of {metrics.avg_nodes_visited:.2f} nodes per scenario: short routes "
        "(`simple`, `missing_info`) finish in four nodes, while the retry and approval routes "
        "visit noticeably more, which is what pulls the average up."
    )
    return "\n".join(lines)


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data."""
    resume_line = (
        "The run verified a replayable checkpoint history: more than one checkpoint was "
        "written for the thread and the earliest one could be read back by id."
        if metrics.resume_success
        else "State-history verification did not pass in this run — see the note below."
    )

    return f"""# Day 08 Lab Report

## 1. Team / student

- Name: {STUDENT_NAME} ({STUDENT_ID})
- Repo/commit: `phase2-k3-4-track3-day8-{STUDENT_ID}-NguyenHoangVu`
- Date: {date.today().isoformat()}

## 2. Architecture

{ARCHITECTURE}

## 3. State schema

{STATE_SCHEMA}

## 4. Scenario results

{_summary_table(metrics)}

{_scenario_table(metrics)}

**What the numbers say**

{_observations(metrics)}

## 5. Failure analysis

{FAILURE_ANALYSIS}

## 6. Persistence / recovery evidence

Each scenario runs under its own `thread_id` (`thread-<scenario_id>`), passed to
`graph.invoke()` through `config={{"configurable": {{"thread_id": ...}}}}`, so checkpoints
from different tickets never interleave.

`build_checkpointer` supports three backends: `none`, `memory` (per-process) and `sqlite`
(durable, `SqliteSaver` over a `sqlite3` connection in WAL mode). {resume_line}

For cross-process evidence, `scripts/persistence_demo.py` runs the risky route with
`LANGGRAPH_INTERRUPT=true` against a SQLite checkpoint file: the first process stops at the
approval interrupt, the process exits, and a second process reopens the same file and
resumes the paused thread with `Command(resume=...)` to completion.

## 7. Extension work

- **SQLite checkpointer** with WAL mode and a `sqlite:///` URL parser.
- **Cross-process crash-resume demo** driving a real `interrupt()` / `Command(resume=...)`
  human-in-the-loop cycle.
- **State-history replay check** (`verify_state_history`) that reads the checkpoint list
  back and confirms the earliest checkpoint is addressable by id.
- **LLM-as-judge** in `evaluate_node`, with a deterministic short-circuit on transport
  errors so the reliability gate does not depend on model judgement.
- **Mermaid diagram** exported from the compiled graph via `graph.get_graph().draw_mermaid()`.

## 8. Improvement plan

{IMPROVEMENT_PLAN}
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
