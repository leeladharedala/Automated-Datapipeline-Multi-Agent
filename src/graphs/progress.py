"""Progress emission for sub-agent nodes.

Delegates to ``log_subagent_progress``, which emits to two sinks:
  1. The in-process realtime log bridge (dev / same-process fallback).
  2. The LangGraph custom stream via ``get_stream_writer()`` for
     cross-process streaming (consumed via ``join_stream`` with
     ``stream_mode="custom"``).
"""

from src.graphs.realtime_logs import log_subagent_progress


def emit_progress(agent_name: str, message: str) -> None:
    """Emit sub-agent progress to the in-process bridge and the custom stream.

    ``log_subagent_progress`` already publishes the line to the LangGraph
    custom stream (consumed cross-process via join_stream stream_mode="custom"),
    so this delegates entirely to it. Do NOT also write to the stream here — a
    second ``get_stream_writer()`` write duplicated every progress line on the
    dashboard (the relay keys off the shared ``message`` field).

    Example:
        emit_progress("iac-agent", "[validate] Running terraform plan...")
    """
    log_subagent_progress(agent_name, message)


def resolve_with_heartbeat(
    future, agent_name: str, phase: str, interval: float = 6.0
):
    """Block on a concurrent.futures Future, emitting a heartbeat every
    ``interval`` seconds while it is still pending.

    A sub-agent node's real work (a single model ``ainvoke``) runs silently for
    tens of seconds with no progress events, so the dashboard panel looks frozen
    or empty. Polling the future with a timeout lets us emit "still working" on
    the NODE thread — where the LangGraph stream-writer contextvar is set, so
    the line actually reaches the dashboard (a background thread would not have
    the context and the write would be a silent no-op).

    Best-effort: the heartbeat never affects the result. On any error emitting,
    or on the fallback path with no pollable future, callers just get the
    value. Uses ``concurrent.futures.TimeoutError`` (what a run_coroutine_
    threadsafe future raises), NOT asyncio's.
    """
    import concurrent.futures as _cf

    elapsed = 0.0
    while True:
        try:
            return future.result(timeout=interval)
        except _cf.TimeoutError:
            elapsed += interval
            try:
                emit_progress(
                    agent_name,
                    f"[{phase}] still working… ({int(elapsed)}s elapsed)",
                )
            except Exception:
                pass  # heartbeat is advisory; never let it break the node
