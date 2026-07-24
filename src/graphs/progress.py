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
    future, agent_name: str, phase: str, interval: float = 15.0
):
    """Block on a concurrent.futures Future, emitting a heartbeat every
    ``interval`` seconds while it is still pending.

    A sub-agent node's real work (a single model ``ainvoke``) runs silently for
    tens of seconds with no progress events, so the dashboard panel looks frozen
    or empty. Polling the future with a timeout lets us emit "still working" on
    the NODE thread — where the LangGraph stream-writer contextvar is set, so
    the line actually reaches the dashboard (a background thread would not have
    the context and the write would be a silent no-op).

    Prefer ``run_agent_with_live_progress`` for DeepAgent nodes: it streams the
    agent's actual internal tool calls as they happen instead of only this
    coarse "still working" tick. This heartbeat-only helper remains for callers
    that hold an opaque future they cannot stream.

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
                    f"[{phase}] still working ({int(elapsed)}s)",
                )
            except Exception:
                pass  # heartbeat is advisory; never let it break the node


# ---------------------------------------------------------------------------
# Live streaming of a sub-agent's internal tool calls
# ---------------------------------------------------------------------------

def _truncate(value, limit: int = 90) -> str:
    """One-line, length-capped rendering of an arbitrary tool argument."""
    s = str(value).replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _describe_tool_call(name: str, args: dict) -> str:
    """Turn a single tool call into a short, human-readable activity line.

    Kept intentionally terse — one line per action — so the dashboard shows the
    real sequence of work (write main.tf, run terraform plan, call an MCP tool)
    rather than a wall of raw tool payloads.
    """
    args = args or {}
    path = args.get("file_path") or args.get("path") or args.get("filename")
    if name in ("write_file", "edit_file", "read_file", "create_file", "delete_file"):
        return f"{name} {path}" if path else name
    if name in ("execute", "run", "shell", "bash", "execute_command", "run_command"):
        cmd = args.get("command") or args.get("cmd") or args.get("code") or ""
        return f"execute: {_truncate(cmd)}" if cmd else name
    if name in ("ls", "glob", "grep"):
        arg = args.get("path") or args.get("pattern") or ""
        return f"{name} {_truncate(arg)}".strip()
    # MCP / other tools: show the tool name plus its first scalar argument.
    if args:
        k = next(iter(args))
        return f"{name}({k}={_truncate(args[k], 60)})"
    return name


def _render_agent_events(msg) -> list[str]:
    """Render newly-produced messages from a sub-agent step into log lines.

    Only tool CALLS are surfaced (the informative action). Tool results, raw
    model text and thinking are skipped — the enclosing node emits its own
    start/finish summaries, and echoing full payloads would flood the panel.
    """
    try:
        from langchain_core.messages import AIMessage
    except Exception:
        return []
    lines: list[str] = []
    try:
        if isinstance(msg, AIMessage):
            for tc in (getattr(msg, "tool_calls", None) or []):
                line = _describe_tool_call(tc.get("name") or "tool", tc.get("args") or {})
                if line:
                    lines.append(line)
    except Exception:
        pass
    return lines


def _drain(log_q, agent_name: str, phase: str) -> int:
    """Emit every queued log line on the CURRENT (node) thread; return count.

    Draining here — not on the background loop that produced the lines — is what
    makes them reach the dashboard: the LangGraph stream-writer contextvar is
    only set on the node thread.
    """
    n = 0
    while True:
        try:
            line = log_q.get_nowait()
        except Exception:
            break
        try:
            emit_progress(agent_name, f"[{phase}] → {line}")
        except Exception:
            pass
        n += 1
    return n


def run_agent_with_live_progress(
    agent,
    user_message: str,
    bg_loop,
    agent_name: str,
    phase: str,
    invoke_config: dict | None = None,
    interval: float = 15.0,
    poll: float = 0.5,
):
    """Run a DeepAgent via ``astream`` on ``bg_loop``, emitting each internal
    tool call as a realtime progress line, and return the final state dict.

    The agent's real work is many internal LLM turns and tool calls hidden
    inside one ``ainvoke``; streaming with ``stream_mode="values"`` lets us diff
    the message list after every super-step and surface each new tool call the
    instant it happens. Lines are handed off from the background loop to the
    node thread through a thread-safe queue and flushed here (see ``_drain``),
    so they actually reach the dashboard. A coarse "still working (Ns)" tick is
    still emitted, but only after ``interval`` seconds with no other activity.

    The returned value is the same final state ``ainvoke`` would return
    (``{"messages": [...], "files": {...}, ...}``), so callers are unchanged.
    """
    import asyncio
    import queue as _queue
    import concurrent.futures as _cf
    from langchain_core.messages import HumanMessage

    log_q: "_queue.Queue" = _queue.Queue()

    async def _astream():
        stream_kwargs = {"stream_mode": "values"}
        if invoke_config:
            stream_kwargs["config"] = invoke_config
        seen = 0
        final = None
        async for state in agent.astream(
            {"messages": [HumanMessage(content=user_message)]}, **stream_kwargs
        ):
            final = state
            msgs = state.get("messages", []) if isinstance(state, dict) else []
            for m in msgs[seen:]:
                for line in _render_agent_events(m):
                    log_q.put(line)
            seen = len(msgs)
        return final

    fut = asyncio.run_coroutine_threadsafe(_astream(), bg_loop)

    elapsed = 0.0
    idle = 0.0
    while True:
        try:
            result = fut.result(timeout=poll)
            _drain(log_q, agent_name, phase)  # flush anything left
            return result
        except _cf.TimeoutError:
            drained = _drain(log_q, agent_name, phase)
            elapsed += poll
            idle = 0.0 if drained else idle + poll
            if idle >= interval:
                idle = 0.0
                try:
                    emit_progress(
                        agent_name, f"[{phase}] still working ({int(elapsed)}s)"
                    )
                except Exception:
                    pass
