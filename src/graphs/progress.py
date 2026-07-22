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
