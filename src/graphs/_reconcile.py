"""Shared async-task reconciliation helpers (pure functions).

Converts the deepagents ``async_tasks`` channel (LangGraph SDK run statuses,
keyed by ``task_id``) into the dashboard's Orchestrator_State view
(``dispatch_statuses`` / ``accumulated_results``, keyed by ``agent_name``).

Used by both the AgentCore ingress (src/agentcore/server.py, end-of-stream
reconciliation) and the OrchestratorMiddleware
(src/graphs/orchestrator_state.py, per-turn state persistence). Keep this
module dependency-free (stdlib only) so it imports cheaply into both.
See design.md "Reconciliation mapping" and Correctness Properties 1 & 2.
"""

from __future__ import annotations

import json
import re
from typing import Any

# The four dashboard-facing statuses the frontend understands.
VALID_DISPATCH_STATUSES = frozenset({"running", "success", "failed", "cancelled"})

# Reconciliation table (design.md): LangGraph SDK run status -> dashboard status.
RUN_STATUS_TO_DISPATCH = {
    "running": "running",
    "pending": "running",
    "success": "success",
    "error": "failed",
    "timeout": "failed",
    "interrupted": "failed",
    "cancelled": "cancelled",
}

# Safe, non-terminal default for unrecognized run statuses: an unknown status
# most likely means the run is still in flight, so we report "running" rather
# than prematurely marking it terminal. This keeps derivation total (Property 1).
DEFAULT_DISPATCH_STATUS = "running"

# When a single agent has multiple tracked tasks, this precedence decides the
# reported dashboard status. "success" wins so the aggregate status stays
# consistent with accumulated_results (which is populated on any success).
DISPATCH_STATUS_PRECEDENCE = {
    "success": 3,
    "failed": 2,
    "cancelled": 1,
    "running": 0,
}

# The async task tool that launches a background sub-agent run (Req 12.2).
LAUNCH_TOOL_NAME = "start_async_task"

# Max characters retained for an accumulated_results summary.
RESULT_SUMMARY_LIMIT = 500


def derive_dispatch_status(run_status: Any) -> str:
    """Map a LangGraph SDK run status onto a dashboard dispatch status.

    Returns one of ``running``/``success``/``failed``/``cancelled`` for every
    possible input (the mapping is total). Known statuses follow the design's
    reconciliation table; any unrecognized value (or non-string input) maps to
    the safe non-terminal default ``running``.
    """
    if not isinstance(run_status, str):
        return DEFAULT_DISPATCH_STATUS
    return RUN_STATUS_TO_DISPATCH.get(run_status, DEFAULT_DISPATCH_STATUS)


def coerce_tool_args(args: Any) -> dict:
    """Best-effort coercion of tool-call args into a dict.

    Accepts an already-parsed dict, or a (possibly partial) JSON string as
    streamed via tool_call_chunks. Falls back to a targeted regex for
    ``subagent_type`` when the JSON is incomplete/unparseable.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            match = re.search(r'"subagent_type"\s*:\s*"([a-zA-Z0-9_-]+)', args)
            if match:
                return {"subagent_type": match.group(1)}
    return {}


def launch_agent_name(args: Any) -> str | None:
    """Extract the launched Sub_Agent's ``agent_name`` from launch tool args.

    deepagents' ``start_async_task`` takes ``subagent_type``; we also tolerate
    the legacy ``agent_name``/``name`` keys. Returns ``None`` when no name is
    present.
    """
    parsed = coerce_tool_args(args)
    name = (
        parsed.get("subagent_type")
        or parsed.get("agent_name")
        or parsed.get("name")
    )
    return name if isinstance(name, str) and name else None


def _message_text_content(msg: Any) -> str | None:
    """Return a message's textual content if it is a plain string, else None."""
    if isinstance(msg, dict):
        content = msg.get("content")
    else:
        content = getattr(msg, "content", None)
    return content if isinstance(content, str) else None


def extract_result_summaries(messages: Any) -> dict[str, str]:
    """Recover success summaries from ``check_async_task`` results in history.

    ``check_async_task`` returns ``json.dumps({"status", "thread_id",
    "result"|"error", ...})`` as its ToolMessage. We scan the persisted message
    history for such JSON payloads and, for successful runs, return a
    ``{task_id (== thread_id): truncated summary}`` map to feed ``reconcile``.
    """
    summaries: dict[str, str] = {}
    for msg in messages or []:
        content = _message_text_content(msg)
        if not content:
            continue
        try:
            data = json.loads(content)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        thread_id = data.get("thread_id")
        if thread_id and data.get("status") == "success":
            summaries[str(thread_id)] = str(data.get("result", ""))[
                :RESULT_SUMMARY_LIMIT
            ]
    return summaries


def reconcile(async_tasks: dict, summaries: dict) -> tuple[dict, dict]:
    """Reconcile the ``async_tasks`` channel into dashboard state.

    Args:
        async_tasks: Snapshot of the ``async_tasks`` channel keyed by
            ``task_id``. Each entry is a mapping containing at least
            ``agent_name`` and ``status`` (the LangGraph SDK run status).
        summaries: Per-task result summaries, keyed by ``task_id`` (with a
            fallback lookup by ``agent_name``).

    Returns:
        ``(dispatch_statuses, accumulated_results)``, both keyed by
        ``agent_name``. ``dispatch_statuses`` has exactly one entry per distinct
        agent_name present in ``async_tasks``, each a valid derived status. An
        ``accumulated_results`` entry is present for an agent if and only if
        that agent has at least one task whose derived status is ``success``
        (Property 2).
    """
    summaries = summaries or {}
    dispatch_statuses: dict = {}
    accumulated_results: dict = {}

    for task_id, task in (async_tasks or {}).items():
        if not isinstance(task, dict):
            continue
        agent_name = task.get("agent_name")
        if not agent_name:
            continue

        derived = derive_dispatch_status(task.get("status"))

        # Aggregate per agent by precedence so success surfaces on the dashboard.
        current = dispatch_statuses.get(agent_name)
        if current is None or (
            DISPATCH_STATUS_PRECEDENCE[derived] > DISPATCH_STATUS_PRECEDENCE[current]
        ):
            dispatch_statuses[agent_name] = derived

        if derived == "success":
            summary = summaries.get(str(task_id))
            if summary is None:
                summary = summaries.get(agent_name)
            accumulated_results[agent_name] = summary if summary is not None else ""

    return dispatch_statuses, accumulated_results
