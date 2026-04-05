"""
AgentCore Entrypoint — BedrockAgentCoreApp with full orchestrator agent.

Initialization happens at module load time (before app.run()) so the
agent is ready when the first invocation arrives, avoiding the 120s
cold start timeout.
"""

print("BOOT: Server module loading...", flush=True)

import asyncio
import logging
import os
import traceback
import uuid

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_core.messages import HumanMessage

print("BOOT: Core imports OK.", flush=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = BedrockAgentCoreApp()
_agent_graph = None
_init_error = None


def _build_agent_sync():
    """Build the agent graph synchronously at startup."""
    global _agent_graph, _init_error
    errors = []

    # Resolve secrets
    try:
        arn = os.environ.get("ANTHROPIC_API_KEY_SECRET_ARN", "")
        if arn and not os.environ.get("ANTHROPIC_API_KEY"):
            region = os.environ.get("AWS_REGION", "us-west-2")
            client = boto3.client("secretsmanager", region_name=region)
            resp = client.get_secret_value(SecretId=arn)
            os.environ["ANTHROPIC_API_KEY"] = resp["SecretString"]
            print("BOOT: Resolved ANTHROPIC_API_KEY", flush=True)
    except Exception as exc:
        errors.append(f"Secrets: {exc}")
        print(f"ERROR: Secret resolution failed: {exc}", flush=True)

    # Build agent
    try:
        from src.main import build_agent
        print("BOOT: Building agent graph...", flush=True)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _agent_graph = loop.run_until_complete(build_agent())
        loop.close()
        print("BOOT: Agent graph ready.", flush=True)
    except Exception as exc:
        errors.append(f"Build: {type(exc).__name__}: {exc}")
        print(f"ERROR: Agent build failed: {exc}", flush=True)
        traceback.print_exc()

    _init_error = "; ".join(errors) if errors else None


# Initialize at module load time — before app.run()
print("BOOT: Starting agent initialization...", flush=True)
_build_agent_sync()
print(f"BOOT: Init complete. Agent ready: {_agent_graph is not None}", flush=True)


@app.entrypoint
def invoke(payload, context=None):
    """Process user input through the orchestrator agent."""
    if _agent_graph is None:
        return {"error": f"Agent failed to initialize: {_init_error or 'unknown'}"}

    # Extract prompt
    prompt = payload.get("prompt", "")
    if not prompt:
        prompt = payload.get("input", {}).get("prompt", "")
    if not prompt:
        messages = payload.get("messages", [])
        if messages:
            last = messages[-1]
            prompt = last.get("content", "") if isinstance(last, dict) else str(last)

    if not prompt:
        return {"error": "No prompt found in payload."}

    try:
        session_id = None
        if context:
            session_id = getattr(context, "session_id", None)
        if not session_id:
            session_id = str(uuid.uuid4())

        config = {
            "configurable": {
                "thread_id": session_id,
                "actor_id": session_id,
            }
        }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            _agent_graph.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )
        )
        loop.close()

        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            content = getattr(last_msg, "content", str(last_msg))
            return {"result": content}

        return {"result": str(result)}

    except Exception as exc:
        print(f"ERROR: Invocation failed: {exc}", flush=True)
        traceback.print_exc()
        return {"error": f"Invocation failed: {exc}"}


if __name__ == "__main__":
    print("BOOT: Starting BedrockAgentCoreApp...", flush=True)
    app.run()
