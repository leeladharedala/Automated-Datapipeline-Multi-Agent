"""
AgentCore Entrypoint — matches the official FAST LangGraph pattern.

Uses BedrockAgentCoreApp with async streaming entrypoint.
"""

import asyncio
import logging
import os
import sys
import traceback

# Guarantee project root is in sys.path for absolute imports
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp, RequestContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp()

_graph = None


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
        dispatched_agents: dict[str, str] = {}  # tool_call_id → agent_name
        partial_tool_calls: dict[int, dict] = {}  # tool_index → accumulated fragment
        logger.info("Starting astream for session=%s", session_id)

        # Use a queue + keepalive task so the stream never goes idle
        # (idle streams cause BodyTimeoutError on the proxy side).
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()
        KEEPALIVE_INTERVAL = 15  # seconds

        async def _stream_producer():
            """Read graph.astream and push chunks into the queue."""
            try:
                async for event in graph.astream(
                    {"messages": [("user", user_query)]},
                    config=config,
                    stream_mode="messages",
                ):
                    await queue.put(event)
            except Exception as exc:
                await queue.put(exc)
            finally:
                await queue.put(SENTINEL)

        async def _keepalive():
            """Inject periodic heartbeat events to prevent stream timeouts."""
            while True:
                await asyncio.sleep(KEEPALIVE_INTERVAL)
                await queue.put("__keepalive__")

        # Subscribe to realtime subagent logs and route them to our SSE stream queue
        loop = asyncio.get_running_loop()

        def log_listener(agent_name: str, message: str):
            try:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "__pipeline_status__": True,
                        "realtime_logs": {agent_name: message}
                    }
                )
            except Exception as e:
                logger.warning("Failed to queue log event: %s", e)

        from src.graphs.realtime_logs import subscribe_realtime_logs, unsubscribe_realtime_logs
        subscribe_realtime_logs(log_listener)

        producer_task = asyncio.create_task(_stream_producer())
        keepalive_task = asyncio.create_task(_keepalive())

        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
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

                # --- Detect sub-agent dispatch (AIMessageChunk with tool_calls) ---
                # LangGraph stream_mode="messages" delivers tool calls incrementally:
                #   Chunk 1: name="task", args={}
                #   Chunk 2: name="",     args={"agent_name": "iac-agent"}
                # We accumulate fragments by index and defer emission until both
                # name and agent_name are available.
                tool_calls = dumped.get("tool_calls") or dumped.get("tool_call_chunks") or []
                for tc in tool_calls:
                    tc_index = tc.get("index")
                    if tc_index is None:
                        continue

                    # Merge this fragment into the accumulator
                    partial = partial_tool_calls.setdefault(tc_index, {"id": "", "name": "", "args": ""})
                    
                    incoming_id = tc.get("id")
                    if incoming_id:
                        partial["id"] = incoming_id

                    incoming_name = tc.get("name")
                    if incoming_name:
                        partial["name"] = incoming_name

                    incoming_args = tc.get("args")
                    if incoming_args:
                        if isinstance(incoming_args, str):
                            partial["args"] += incoming_args
                        elif isinstance(incoming_args, dict):
                            import json
                            try:
                                # Merge dict keys into existing partial["args"] if possible
                                existing_dict = {}
                                if partial.get("args"):
                                    try:
                                        existing_dict = json.loads(partial["args"])
                                    except Exception:
                                        pass
                                existing_dict.update(incoming_args)
                                partial["args"] = json.dumps(existing_dict)
                            except Exception:
                                partial["args"] = json.dumps(incoming_args)

                    # Emit only once we have name="task" or direct subagent name AND a non-empty agent_name
                    tc_name = partial.get("name", "")
                    agent_name = ""
                    args_str = partial.get("args", "")
                    tc_id = partial.get("id") or f"index_{tc_index}"

                    if tc_name == "task":
                        if args_str:
                            import json
                            try:
                                args_dict = json.loads(args_str)
                                agent_name = args_dict.get("subagent_type") or args_dict.get("agent_name") or args_dict.get("name")
                            except Exception:
                                import re
                                # Highly robust regex that matches even without trailing quote
                                match_sub = re.search(r'"subagent_type"\s*:\s*"([a-zA-Z0-9_-]+)', args_str)
                                if match_sub:
                                    agent_name = match_sub.group(1)
                                else:
                                    match = re.search(r'"agent_name"\s*:\s*"([a-zA-Z0-9_-]+)', args_str)
                                    if match:
                                        agent_name = match.group(1)
                                    else:
                                        match_name = re.search(r'"name"\s*:\s*"([a-zA-Z0-9_-]+)', args_str)
                                        if match_name:
                                            agent_name = match_name.group(1)
                    elif tc_name in {"iac-agent", "cicd-agent", "data-eng-agent"}:
                        agent_name = tc_name

                    VALID_AGENT_NAMES = {"iac-agent", "cicd-agent", "data-eng-agent"}
                    if agent_name in VALID_AGENT_NAMES:
                        if tc_id not in dispatched_agents:
                            dispatched_agents[tc_id] = agent_name
                            # Do not clear the index completely yet in case more chunks arrive,
                            # but mark it as dispatched to prevent duplicate events.
                            logger.info("Sub-agent dispatched: %s → running", agent_name)
                            yield {
                                "__pipeline_status__": True,
                                "dispatch_statuses": {agent_name: "running"},
                            }
                        real_id = partial.get("id")
                        if real_id and real_id not in dispatched_agents:
                            dispatched_agents[real_id] = agent_name

                # --- Detect sub-agent completion (ToolMessage) ---
                msg_type = dumped.get("type", "")
                if msg_type == "tool":
                    tc_id = dumped.get("tool_call_id", "")
                    agent_name = dispatched_agents.get(tc_id, "")
                    if agent_name:
                        result_text = str(dumped.get("content", ""))
                        passed = (
                            "PASSED" in result_text.upper() or
                            '"validation_passed": true' in result_text.lower() or
                            '"validation_passed":true' in result_text.lower()
                        )
                        status = "success" if passed else "failed"
                        logger.info("Sub-agent completed: %s → %s", agent_name, status)
                        yield {
                            "__pipeline_status__": True,
                            "dispatch_statuses": {agent_name: status},
                            "accumulated_results": {agent_name: result_text[:500]},
                        }

                yield dumped
        finally:
            unsubscribe_realtime_logs(log_listener)
            keepalive_task.cancel()
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
