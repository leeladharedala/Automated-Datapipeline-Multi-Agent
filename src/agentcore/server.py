"""
AgentCore Entrypoint — matches the official FAST LangGraph pattern.

Uses BedrockAgentCoreApp with async streaming entrypoint.
"""

import asyncio
import logging
import os
import traceback

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
        logger.info("Starting astream for session=%s", session_id)

        async for event in graph.astream(
            {"messages": [("user", user_query)]},
            config=config,
            stream_mode="messages",
        ):
            message_chunk, metadata = event
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
            tool_calls = dumped.get("tool_calls") or []
            for tc in tool_calls:
                tc_name = tc.get("name", "")
                if tc_name == "task":
                    args = tc.get("args", {})
                    agent_name = args.get("agent_name") or args.get("name", "")
                    tc_id = tc.get("id", "")
                    if agent_name:
                        dispatched_agents[tc_id] = agent_name
                        logger.info("Sub-agent dispatched: %s → running", agent_name)
                        yield {
                            "__pipeline_status__": True,
                            "dispatch_statuses": {agent_name: "running"},
                        }

            # --- Detect sub-agent completion (ToolMessage) ---
            msg_type = dumped.get("type", "")
            if msg_type == "tool":
                tc_id = dumped.get("tool_call_id", "")
                agent_name = dispatched_agents.get(tc_id, "")
                if agent_name:
                    result_text = str(dumped.get("content", ""))
                    passed = "PASSED" in result_text.upper()
                    status = "success" if passed else "failed"
                    logger.info("Sub-agent completed: %s → %s", agent_name, status)
                    yield {
                        "__pipeline_status__": True,
                        "dispatch_statuses": {agent_name: status},
                        "accumulated_results": {agent_name: result_text[:500]},
                    }

            yield dumped

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
