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

        async for event in graph.astream(
            {"messages": [("user", user_query)]},
            config=config,
            stream_mode="messages",
        ):
            message_chunk, metadata = event
            yield message_chunk.model_dump()

    except Exception as exc:
        logger.exception("Agent run failed")
        yield {"status": "error", "error": str(exc)}


if __name__ == "__main__":
    app.run()
