"""Dual-sink progress emission for sub-agent nodes.

Emits progress to two sinks:
  1. The in-process realtime log bridge (dev / same-process fallback).
  2. The LangGraph custom stream via ``get_stream_writer()`` for
     cross-process streaming (consumed via ``join_stream`` with
     ``stream_mode="custom"``).
"""

from langgraph.config import get_stream_writer

from src.graphs.realtime_logs import log_subagent_progress  # kept for local/dev


def emit_progress(agent_name: str, message: str) -> None:
    """Emit sub-agent progress to both the in-process bridge and custom stream.

    Example:
        emit_progress("iac-agent", "[validate] Running terraform plan...")
    """
    # In-process bridge (dev / same-process fallback)
    log_subagent_progress(agent_name, message)

    # Cross-process custom stream (consumed via join_stream stream_mode="custom")
    try:
        writer = get_stream_writer()
        if writer is not None:
            writer({"agent_name": agent_name, "message": message})
    except Exception:
        pass  # no active stream context (e.g. unit invocation)
