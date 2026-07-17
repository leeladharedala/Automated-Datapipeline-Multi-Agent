"""
AgentCore Entrypoint — matches the official FAST LangGraph pattern.

Uses BedrockAgentCoreApp with async streaming entrypoint.
"""

import asyncio
import logging
import os
import sys

# Guarantee project root is in sys.path for absolute imports
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp, RequestContext

# Pure reconciliation helpers shared with the OrchestratorMiddleware
# (src/graphs/orchestrator_state.py). They convert the deepagents `async_tasks`
# channel (LangGraph SDK run statuses, keyed by task_id) into the dashboard's
# Orchestrator_State view (`dispatch_statuses` / `accumulated_results`, keyed
# by agent_name). See design.md "Reconciliation mapping".
from src.graphs._reconcile import (
    LAUNCH_TOOL_NAME as _LAUNCH_TOOL_NAME,
    derive_dispatch_status,
    extract_result_summaries as _extract_result_summaries,
    launch_agent_name as _launch_agent_name,
    reconcile,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _SuppressOtelDetachNoise(logging.Filter):
    """Drop the benign "Failed to detach context" ValueError.

    The langchain auto-instrumentation attaches an OTel context token in one
    asyncio task while our queue+producer streaming pattern closes the wrapped
    generator in another, so the token reset is refused by contextvars. It
    fires on successful runs after the response has streamed; suppressing it
    keeps ERROR-level logs meaningful for alerting.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Failed to detach context" not in record.getMessage()


logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachNoise())

app = BedrockAgentCoreApp()

_graph = None


# ---------------------------------------------------------------------------
# Async dispatch detection helpers (pure functions)
#
# These drive the SSE dispatch-status detection in `invocations`. Detection is
# keyed off the deepagents async task tool surface:
#   * `start_async_task` tool CALL  → the Sub_Agent is launched (running).
#   * `check_async_task` / `cancel_async_task` / `list_async_tasks` results
#     land in the `async_tasks` channel (status per task_id), which `reconcile`
#     turns into terminal dashboard statuses.
#
# NOTE on ToolMessage surfacing: the `stream_events(version="v3")` `messages`
# projection consumed by `_stream_producer` (task 8.1) yields the Supervisor's
# LLM messages — it does NOT surface ToolMessages. The async task tools write
# their outcome into the `async_tasks` state channel (and, for a successful
# `check_async_task`, a JSON result into the message history). So terminal
# statuses are surfaced by reconciling the persisted `async_tasks` channel
# (via `reconcile`) at end-of-stream, with success summaries recovered from the
# JSON `check_async_task` results in the message history. The `task_id →
# agent_name` map comes straight from the `async_tasks` channel entries, so
# every emitted event references the correct Sub_Agent (Req 11.4/12.4).
# ---------------------------------------------------------------------------

# Sub-agent types launchable as async tasks (matches the AsyncSubAgent specs).
_ASYNC_SUBAGENT_TYPES = frozenset({"iac-agent", "cicd-agent", "data-eng-agent"})


def _merge_tool_call_fragments(partial: dict, dumped: dict) -> list[dict]:
    """Merge streamed tool-call fragments (by index) and return the current view.

    ``stream_events`` may deliver a tool call either fully-formed (args already
    a dict / complete JSON string) or as incremental ``tool_call_chunks`` whose
    ``args`` fragments must be concatenated by ``index``. This accumulates into
    ``partial`` (keyed by index) and returns, for every fragment seen in this
    ``dumped`` chunk, a ``{"id", "name", "args"}`` view of the accumulated call.
    """
    import json as _json

    touched: list[dict] = []
    tool_calls = dumped.get("tool_calls") or dumped.get("tool_call_chunks") or []
    for tc in tool_calls:
        index = tc.get("index")
        if index is None:
            # Fully-formed tool call without a streaming index — key by id.
            index = f"full_{tc.get('id') or len(partial)}"

        acc = partial.setdefault(index, {"id": "", "name": "", "args": ""})

        if tc.get("id"):
            acc["id"] = tc["id"]
        if tc.get("name"):
            acc["name"] = tc["name"]

        incoming = tc.get("args")
        if incoming:
            if isinstance(incoming, str):
                acc["args"] += incoming
            elif isinstance(incoming, dict):
                merged: dict = {}
                if acc["args"]:
                    try:
                        merged = _json.loads(acc["args"])
                    except Exception:
                        merged = {}
                if not isinstance(merged, dict):
                    merged = {}
                merged.update(incoming)
                acc["args"] = _json.dumps(merged)

        touched.append(
            {
                "id": acc["id"] or f"index_{index}",
                "name": acc["name"],
                "args": acc["args"],
            }
        )
    return touched


def _task_agent_map(async_tasks: dict) -> dict:
    """Build the ``task_id → agent_name`` map from the ``async_tasks`` channel.

    This is the mapping that guarantees every emitted Pipeline_Status_Event
    references the correct Sub_Agent (Req 11.4 / 12.4).
    """
    mapping: dict = {}
    for task_id, task in (async_tasks or {}).items():
        if isinstance(task, dict):
            agent = task.get("agent_name")
            if isinstance(agent, str) and agent:
                mapping[str(task_id)] = agent
    return mapping


def _reconcile_state_values(state_values: dict) -> tuple[dict, dict]:
    """Reconcile persisted Supervisor state into dashboard dispatch statuses.

    Reads the ``async_tasks`` channel (task_id → {agent_name, status, ...}) and
    recovers success summaries from ``check_async_task`` JSON results in the
    message history, then delegates to ``reconcile`` to produce per-agent
    ``(dispatch_statuses, accumulated_results)``.
    """
    values = state_values or {}
    async_tasks = values.get("async_tasks") or {}
    messages = values.get("messages") or []
    summaries = _extract_result_summaries(messages)
    return reconcile(async_tasks, summaries)


def _status_event_from_state(
    state_values: dict, status_overrides: dict | None = None
) -> dict | None:
    """Build a complete ``__pipeline_status__`` event from persisted state.

    Reconciles the ``async_tasks`` channel into agent-keyed ``dispatch_statuses``
    / ``accumulated_results`` and additionally emits the per-task ``tasks``
    collection keyed by the real ``task_id`` — this is what lets the dashboard
    key Task cards (and cancel/steer directives) by the actual Task_ID instead
    of synthesizing one from the agent name.

    ``status_overrides`` maps ``task_id → run status`` observed live from the
    co-located server SDK; when present it supersedes the (possibly stale)
    channel status for that task.
    """
    values = state_values or {}
    async_tasks: dict = {}
    for task_id, task in (values.get("async_tasks") or {}).items():
        if isinstance(task, dict) and status_overrides and str(task_id) in status_overrides:
            task = {**task, "status": status_overrides[str(task_id)]}
        async_tasks[task_id] = task

    summaries = _extract_result_summaries(values.get("messages") or [])
    dispatch_statuses, accumulated_results = reconcile(async_tasks, summaries)
    if not dispatch_statuses:
        return None

    event: dict = {
        "__pipeline_status__": True,
        "dispatch_statuses": dispatch_statuses,
    }
    if accumulated_results:
        event["accumulated_results"] = accumulated_results

    tasks_payload: dict = {}
    for task_id, task in async_tasks.items():
        if not isinstance(task, dict):
            continue
        agent_name = task.get("agent_name")
        if not agent_name:
            continue
        tid = str(task_id)
        entry: dict = {
            "task_id": tid,
            "agent_name": agent_name,
            "status": derive_dispatch_status(task.get("status")),
        }
        if tid in summaries:
            entry["result_summary"] = summaries[tid]
        tasks_payload[tid] = entry
    if tasks_payload:
        event["tasks"] = tasks_payload
    return event


async def _sdk_status_overrides(client, async_tasks: dict) -> dict:
    """Query the co-located server for each tracked task's live run status.

    The ``async_tasks`` channel only advances when the Supervisor calls a
    check/list/cancel tool, so between turns its statuses go stale. This asks
    the server directly (``runs.get``) and returns a ``task_id → run status``
    override map. Best-effort: an unreachable run is simply omitted.
    """
    overrides: dict = {}
    for task_id, task in (async_tasks or {}).items():
        if not isinstance(task, dict):
            continue
        thread_id = task.get("thread_id")
        run_id = task.get("run_id")
        if not thread_id or not run_id:
            continue
        try:
            run = await client.runs.get(thread_id, run_id)
        except Exception:
            continue
        raw = run.get("status") if isinstance(run, dict) else getattr(run, "status", None)
        if isinstance(raw, str) and raw:
            overrides[str(task_id)] = raw
    return overrides


# ---------------------------------------------------------------------------
# Remote-run stream relay (Req 18) — forwards a background sub-agent run's
# progress (messages / updates / custom) from the co-located LangGraph server
# into the SSE queue as `realtime_logs` Pipeline_Status_Events (Req 12.5, 18.5).
# ---------------------------------------------------------------------------

# Stream modes relayed from a background sub-agent run (Req 18.4).
STREAM_RELAY_MODES = ("messages", "updates", "custom")


def _get_subagent_client():
    """Async LangGraph SDK client for the co-located server (Process B).

    Module-level factory so tests can inject a fake client.
    """
    from langgraph_sdk import get_client

    url = os.environ.get("SUBAGENT_SERVER_URL", "http://127.0.0.1:2024")
    return get_client(url=url)


def _relay_log_text(event: str, data) -> str | None:
    """Convert one relayed stream chunk into a human-readable progress line.

    * ``messages`` — the streamed token text (``(chunk, metadata)`` tuple or a
      message dict/object with ``content``).
    * ``updates`` — a per-node step marker, e.g. ``[research] step``.
    * ``custom`` — the ``emit_progress`` payload's ``message``.
    """
    import json as _json

    if event == "messages":
        item = data
        if isinstance(item, (tuple, list)) and item:
            item = item[0]
        if isinstance(item, str):
            return item or None
        if isinstance(item, dict):
            content = item.get("content")
        else:
            content = getattr(item, "content", None)
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
            return text or None
        return None

    if event == "updates":
        if isinstance(data, dict) and data:
            return " ".join(f"[{node}] step" for node in data)
        return None

    if event == "custom":
        if isinstance(data, dict):
            message = data.get("message")
            if isinstance(message, str) and message:
                return message
            return _json.dumps(data)
        if isinstance(data, str):
            return data or None
        return None

    return None


async def stream_run_relay(client, task: dict, queue: asyncio.Queue) -> None:
    """Relay a background sub-agent run's stream into the SSE queue.

    PRIMARY path (Req 18.2/18.7/18.8): ``client.threads.stream(thread_id,
    run_id, stream_mode=[messages, updates, custom], on_disconnect="continue")``
    — the SDK reconnects/replays transparently and never cancels the run.
    FALLBACK path (Req 18.3): ``client.runs.join_stream(...,
    stream_resumable=True)``. Progress is advisory (Req 18.9): if both paths
    fail the relay returns quietly and never breaks the top-level stream.
    """
    agent_name = task.get("agent_name") or ""
    thread_id = task.get("thread_id")
    run_id = task.get("run_id")
    if not agent_name or not thread_id:
        return

    def _put(text: str) -> None:
        queue.put_nowait(
            {"__pipeline_status__": True, "realtime_logs": {agent_name: text}}
        )

    async def _consume(stream) -> None:
        import inspect as _inspect

        if _inspect.isawaitable(stream):
            stream = await stream
        async for chunk in stream:
            text = _relay_log_text(
                getattr(chunk, "event", None) or "", getattr(chunk, "data", None)
            )
            if text:
                _put(text)

    import inspect as _inspect2

    def _supported(func, **kwargs) -> dict:
        """Drop kwargs the installed langgraph-sdk signature doesn't accept.

        The SDK's streaming API differs across versions (e.g. 0.4.x
        ``threads.stream`` has no ``run_id`` and ``runs.join_stream`` has no
        ``stream_resumable``), so filter to what this client supports rather
        than failing the relay on a TypeError.
        """
        try:
            params = _inspect2.signature(func).parameters
        except (TypeError, ValueError):
            return kwargs
        if any(p.kind is _inspect2.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}

    threads_kwargs = _supported(
        client.threads.stream,
        thread_id=thread_id,
        run_id=run_id,
        stream_mode=list(STREAM_RELAY_MODES),
        on_disconnect="continue",
    )
    # Only usable as an attach-to-run stream when the SDK accepts run_id;
    # without it, threads.stream would try to START a new run instead.
    if "run_id" in threads_kwargs:
        try:
            await _consume(client.threads.stream(**threads_kwargs))
            return
        except Exception as exc:
            logger.warning(
                "threads.stream relay failed for %s (%s); falling back to join_stream",
                agent_name,
                exc,
            )

    try:
        await _consume(
            client.runs.join_stream(
                **_supported(
                    client.runs.join_stream,
                    thread_id=thread_id,
                    run_id=run_id,
                    stream_mode=list(STREAM_RELAY_MODES),
                    cancel_on_disconnect=False,
                    stream_resumable=True,
                )
            )
        )
    except Exception as exc:
        logger.warning(
            "join_stream relay failed for %s (%s) — progress relay skipped",
            agent_name,
            exc,
        )


def _resolve_secrets():
    arn = os.environ.get("ANTHROPIC_API_KEY_SECRET_ARN", "")
    if arn and not os.environ.get("ANTHROPIC_API_KEY"):
        region = os.environ.get("AWS_REGION", "us-west-2")
        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=arn)
        os.environ["ANTHROPIC_API_KEY"] = resp["SecretString"]
        logger.info("Resolved ANTHROPIC_API_KEY")

    gh_arn = os.environ.get("GITHUB_TOKEN_SECRET_ARN", "")
    if gh_arn and not os.environ.get("GITHUB_TOKEN"):
        # Extract region from ARN (arn:aws:secretsmanager:<region>:...) so we
        # call the correct endpoint even if the secret is in another region.
        arn_parts = gh_arn.split(":")
        secret_region = arn_parts[3] if len(arn_parts) > 3 and arn_parts[3] else os.environ.get("AWS_REGION", "us-west-2")
        client = boto3.client("secretsmanager", region_name=secret_region)
        resp = client.get_secret_value(SecretId=gh_arn)
        os.environ["GITHUB_TOKEN"] = resp["SecretString"]
        logger.info("Resolved GITHUB_TOKEN from %s", secret_region)


async def _get_graph():
    """Lazy-build the agent graph."""
    global _graph
    if _graph is not None:
        return _graph

    _resolve_secrets()

    from src.main import build_agent
    logger.info("Building agent graph...")
    _graph = await build_agent()
    logger.info("Agent graph ready.")
    return _graph


@app.entrypoint
async def invocations(payload, context: RequestContext):
    """Async streaming entrypoint — yields message chunks."""
    # ── Lightweight status mode ──
    # `{"mode": "status"}` reads the checkpointed async_tasks channel (enriched
    # with live run statuses from the co-located server) and returns a single
    # reconciled Pipeline_Status_Event WITHOUT invoking the LLM. This lets the
    # frontend poll for terminal statuses between user turns.
    if isinstance(payload, dict) and payload.get("mode") == "status":
        try:
            graph = await _get_graph()
            session_id = getattr(context, "session_id", None) or payload.get(
                "runtimeSessionId", "default-session"
            )
            config = {"configurable": {"thread_id": session_id, "actor_id": session_id}}
            snapshot = await graph.aget_state(config)
            state_values = getattr(snapshot, "values", None) or {}
            overrides: dict = {}
            try:
                client = _get_subagent_client()
                overrides = await _sdk_status_overrides(
                    client, state_values.get("async_tasks") or {}
                )
            except Exception as exc:
                logger.warning("Status mode: SDK status lookup failed: %s", exc)
            event = _status_event_from_state(state_values, overrides)
            yield event or {"__pipeline_status__": True, "dispatch_statuses": {}}
        except Exception as exc:
            logger.exception("Status mode failed")
            yield {"status": "error", "error": str(exc)}
        return

    user_query = payload.get("prompt", "")
    if not user_query:
        msgs = payload.get("messages", [])
        if msgs:
            last = msgs[-1]
            user_query = last.get("content", "") if isinstance(last, dict) else str(last)

    if not user_query:
        yield {"status": "error", "error": "No prompt provided"}
        return

    try:
        graph = await _get_graph()

        session_id = getattr(context, "session_id", None) or payload.get("runtimeSessionId", "default-session")
        config = {"configurable": {"thread_id": session_id, "actor_id": session_id}}

        chunk_count = 0
        # tool_call_id → agent_name for start_async_task launches already
        # reported as "running" this stream (dedupes the running emission).
        launched_calls: dict[str, str] = {}
        # tool_index → accumulated tool-call fragment (id/name/args).
        partial_tool_calls: dict = {}
        logger.info("Starting astream for session=%s", session_id)

        # Retrieve active OpenTelemetry trace context if available
        from opentelemetry import trace
        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
        trace_id = ""
        if span_context.is_valid:
            raw_id = trace.format_trace_id(span_context.trace_id)
            if len(raw_id) == 32:
                trace_id = f"1-{raw_id[:8]}-{raw_id[8:]}"
            else:
                trace_id = raw_id
            logger.info("Active telemetry Trace ID: %s", trace_id)

        if trace_id:
            yield {
                "__pipeline_status__": True,
                "trace_id": trace_id
            }

        # Use a queue + keepalive task so the stream never goes idle
        # (idle streams cause BodyTimeoutError on the proxy side).
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()
        KEEPALIVE_INTERVAL = 15  # seconds

        async def _stream_producer():
            """Drive the Supervisor's `stream_events(version="v3")` run and push
            message-chunk items into the SSE queue (Req 18.1).

            Replaces the prior ``graph.astream(stream_mode="messages")``
            primitive. We consume the v3 typed ``messages`` projection —
            streaming text/reasoning deltas live and, per message, the
            finalized tool calls — and reconstruct ``AIMessageChunk`` items so
            the existing downstream consumer contract is preserved: each item
            is the same ``(message_chunk, metadata)`` tuple the consumer
            unpacks, ``model_dump()``s, and yields.
            """
            import json as _json

            from langchain_core.messages import AIMessageChunk

            def _emit(chunk, node):
                # Preserve the (message_chunk, metadata) tuple shape the
                # consumer expects from the former stream_mode="messages" path.
                queue.put_nowait((chunk, {"langgraph_node": node}))

            try:
                run = await graph.astream_events(
                    {"messages": [("user", user_query)]},
                    config=config,
                    version="v3",
                )
                async with run:
                    # Each item in run.messages is one LLM message stream.
                    async for msg in run.messages:
                        node = getattr(msg, "node", None)

                        # Stream text/reasoning deltas live, in wire order,
                        # from the per-message protocol events.
                        async for ev in msg:
                            if not isinstance(ev, dict):
                                continue
                            if ev.get("event") != "content-block-delta":
                                continue
                            delta = ev.get("delta")
                            if not isinstance(delta, dict):
                                continue
                            dtype = delta.get("type")
                            if dtype == "text-delta":
                                text = delta.get("text", "")
                                if text:
                                    _emit(AIMessageChunk(content=text), node)
                            elif dtype == "reasoning-delta":
                                reasoning = delta.get("reasoning", "")
                                if reasoning:
                                    _emit(
                                        AIMessageChunk(
                                            content="",
                                            additional_kwargs={
                                                "reasoning": reasoning
                                            },
                                        ),
                                        node,
                                    )

                        # Finalized tool calls are the robust dispatch
                        # signal (providers surface them via block deltas or
                        # whole-message content blocks). Re-emit them as
                        # indexed tool_call_chunks so the tuple contract and
                        # dispatch detection keep working.
                        final = await msg.output
                        tool_calls = getattr(final, "tool_calls", None) or []
                        if tool_calls:
                            chunks = []
                            for i, tc in enumerate(tool_calls):
                                args = tc.get("args")
                                chunks.append(
                                    {
                                        "type": "tool_call_chunk",
                                        "id": tc.get("id"),
                                        "name": tc.get("name"),
                                        "args": args
                                        if isinstance(args, str)
                                        else _json.dumps(args or {}),
                                        "index": i,
                                    }
                                )
                            _emit(
                                AIMessageChunk(
                                    content="", tool_call_chunks=chunks
                                ),
                                node,
                            )
            except Exception as exc:
                await queue.put(exc)
            finally:
                await queue.put(SENTINEL)

        async def _keepalive():
            """Inject periodic heartbeat events to prevent stream timeouts."""
            while True:
                await asyncio.sleep(KEEPALIVE_INTERVAL)
                await queue.put("__keepalive__")

        # --- Remote-run stream relays (Req 12.5 / 18) ---
        # One relay per launched sub-agent run: it resolves the run's
        # async_tasks entry (thread_id/run_id) from the checkpointed state, then
        # forwards the run's messages/updates/custom stream into this SSE queue
        # as realtime_logs events. A relay outliving the Supervisor turn keeps
        # the top-level stream open (keepalives continue) so live progress and
        # the final terminal statuses reach the dashboard without another turn.
        relay_tasks: set[asyncio.Task] = set()
        relayed_task_ids: set[str] = set()
        relay_client = None
        RELAY_RESOLVE_ATTEMPTS = 30  # ~30s for the channel entry to appear
        RELAY_MAX_WAIT = float(os.environ.get("SUBAGENT_RELAY_MAX_WAIT_SECONDS", "840"))

        async def _spawn_relay_for(agent_name: str) -> None:
            """Resolve the launched agent's task entry, announce its task_id,
            and relay its remote run stream into the queue."""
            nonlocal relay_client
            try:
                if relay_client is None:
                    relay_client = _get_subagent_client()
            except Exception as exc:
                logger.warning("Sub-agent SDK client unavailable, no relay: %s", exc)
                return
            for _ in range(RELAY_RESOLVE_ATTEMPTS):
                try:
                    snapshot = await graph.aget_state(config)
                    values = getattr(snapshot, "values", None) or {}
                except Exception:
                    values = {}
                for task_id, task in (values.get("async_tasks") or {}).items():
                    if not isinstance(task, dict):
                        continue
                    if task.get("agent_name") != agent_name:
                        continue
                    tid = str(task_id)
                    if tid in relayed_task_ids:
                        continue
                    relayed_task_ids.add(tid)
                    # Announce the real task_id → agent association so the
                    # dashboard can key this Task (and cancel/steer it) by id.
                    queue.put_nowait(
                        {
                            "__pipeline_status__": True,
                            "dispatch_statuses": {agent_name: "running"},
                            "tasks": {
                                tid: {
                                    "task_id": tid,
                                    "agent_name": agent_name,
                                    "status": "running",
                                }
                            },
                        }
                    )
                    await stream_run_relay(relay_client, task, queue)
                    return
                await asyncio.sleep(1.0)
            logger.warning("No async_tasks entry appeared for %s; relay skipped", agent_name)

        producer_task = asyncio.create_task(_stream_producer())
        keepalive_task = asyncio.create_task(_keepalive())

        producer_done = False
        relays_outlived_producer = False
        relay_deadline: float | None = None

        try:
            while True:
                if producer_done:
                    # The Supervisor turn is over; stay open only while relays
                    # are still streaming sub-agent progress (bounded).
                    active_relays = [t for t in relay_tasks if not t.done()]
                    if (not active_relays and queue.empty()) or (
                        relay_deadline is not None
                        and asyncio.get_event_loop().time() > relay_deadline
                    ):
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                else:
                    item = await queue.get()
                if item is SENTINEL:
                    # The Supervisor run has finished and its state is
                    # checkpointed. Reconcile the persisted `async_tasks`
                    # channel into terminal per-agent dispatch statuses so
                    # check/cancel/list outcomes surface even though the v3
                    # `messages` projection does not carry ToolMessages
                    # (Req 12.3/12.4, 11.4).
                    try:
                        snapshot = await graph.aget_state(config)
                        state_values = getattr(snapshot, "values", None) or {}
                    except Exception as exc:
                        logger.warning(
                            "Failed to read final state for reconciliation: %s", exc
                        )
                        state_values = {}

                    task_map = _task_agent_map(state_values.get("async_tasks") or {})
                    event = _status_event_from_state(state_values)
                    if event:
                        logger.info(
                            "Reconciled dispatch statuses %s (task→agent map: %s)",
                            event.get("dispatch_statuses"),
                            task_map,
                        )
                        yield event

                    producer_done = True
                    relays_outlived_producer = any(
                        not t.done() for t in relay_tasks
                    )
                    if relays_outlived_producer:
                        relay_deadline = (
                            asyncio.get_event_loop().time() + RELAY_MAX_WAIT
                        )
                        logger.info(
                            "Supervisor turn complete; keeping stream open for "
                            "%d active sub-agent relay(s)",
                            sum(1 for t in relay_tasks if not t.done()),
                        )
                    continue
                if isinstance(item, Exception):
                    raise item
                if item == "__keepalive__":
                    yield {"__keepalive__": True}
                    continue

                if isinstance(item, dict) and "__pipeline_status__" in item:
                    yield item
                    continue

                message_chunk, metadata = item
                dumped = message_chunk.model_dump()
                chunk_count += 1

                # Log the first few chunks to diagnose stream format
                if chunk_count <= 5:
                    content_preview = str(dumped.get("content", ""))[:200]
                    logger.info(
                        "Chunk #%d type=%s content_preview=%s tool_calls=%s",
                        chunk_count,
                        dumped.get("type", "unknown"),
                        content_preview,
                        bool(dumped.get("tool_calls")),
                    )

                # --- Detect sub-agent launch: start_async_task tool call ---
                # Key detection off the async launch tool. Tool calls may arrive
                # fully-formed or as incremental tool_call_chunks, so merge
                # fragments by index before inspecting name/args. Emit a
                # "running" dispatch status once per launch (Req 12.2), keyed to
                # the Sub_Agent named by `subagent_type` (Req 11.4/12.4), and
                # spawn a stream relay that forwards the launched run's live
                # progress into this stream as realtime_logs (Req 12.5 / 18.5).
                for tc in _merge_tool_call_fragments(partial_tool_calls, dumped):
                    if tc["name"] != _LAUNCH_TOOL_NAME:
                        continue
                    agent_name = _launch_agent_name(tc["args"])
                    if agent_name not in _ASYNC_SUBAGENT_TYPES:
                        continue
                    tc_id = tc["id"]
                    if tc_id in launched_calls:
                        continue
                    launched_calls[tc_id] = agent_name
                    logger.info("Sub-agent launched: %s → running", agent_name)
                    yield {
                        "__pipeline_status__": True,
                        "dispatch_statuses": {agent_name: "running"},
                    }
                    relay_tasks.add(asyncio.create_task(_spawn_relay_for(agent_name)))

                yield dumped

            # Relays outlived the Supervisor turn: the sub-agent runs have now
            # finished streaming, so emit a final SDK-verified terminal event
            # (the async_tasks channel itself stays stale until the Supervisor
            # next calls check_async_task).
            if relays_outlived_producer:
                try:
                    snapshot = await graph.aget_state(config)
                    state_values = getattr(snapshot, "values", None) or {}
                except Exception as exc:
                    logger.warning("Failed to re-read state after relays: %s", exc)
                    state_values = {}
                overrides: dict = {}
                if relay_client is not None:
                    try:
                        overrides = await _sdk_status_overrides(
                            relay_client, state_values.get("async_tasks") or {}
                        )
                    except Exception as exc:
                        logger.warning("Post-relay SDK status lookup failed: %s", exc)
                event = _status_event_from_state(state_values, overrides)
                if event:
                    logger.info(
                        "Post-relay terminal statuses: %s", event.get("dispatch_statuses")
                    )
                    yield event
        finally:
            keepalive_task.cancel()
            for t in relay_tasks:
                t.cancel()
            await producer_task

        logger.info("Stream complete: yielded %d chunks", chunk_count)

        # If zero chunks were yielded, send a fallback so the UI isn't blank
        if chunk_count == 0:
            logger.warning("No chunks yielded — sending fallback")
            yield {"content": "Agent completed but produced no streaming output.", "type": "ai"}

    except Exception as exc:
        logger.exception("Agent run failed")
        yield {"status": "error", "error": str(exc)}


if __name__ == "__main__":
    app.run()
