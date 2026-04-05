"""
AgentCore Entrypoint — BedrockAgentCoreApp with full orchestrator agent.

Uses the official SDK (@app.entrypoint) which works on AgentCore.
Agent initialization happens lazily on first invocation.
"""

print("BOOT: Server module loading...", flush=True)

import asyncio
import logging
import os
import traceback

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp

print("BOOT: Core imports OK.", flush=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = BedrockAgentCoreApp()

_agent_graph = None
_initialized = False
_init_error = None


def _resolve_secrets():
    arn = os.environ.get("ANTHROPIC_API_KEY_SECRET_ARN", "")
    if arn and not os.environ.get("ANTHROPIC_API_KEY"):
        region = os.environ.get("AWS_REGION", "us-west-2")
        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=arn)
        os.environ["ANTHROPIC_API_KEY"] = resp["SecretString"]
        print("BOOT: Resolved ANTHROPIC_API_KEY", flush=True)


def _init_sync():
    """Synchronous wrapper for lazy initialization."""
    global _agent_graph, _initialized, _init_error

    if _initialized:
        return

    print("BOOT: Starting initialization...", flush=True)
    errors = []

    try:
        _resolve_secrets()
    except Exception as exc:
        msg = f"Secret resolution failed: {exc}"
        print(f"ERROR: {msg}", flush=True)
        errors.append(msg)

    try:
        from src.tracing import setup_tracing
        setup_tracing()
        print("BOOT: Tracing initialized.", flush=True)
    except Exception as exc:
        msg = f"Tracing setup failed: {exc}"
        print(f"WARNING: {msg}", flush=True)
        errors.append(msg)

    try:
        from src.main import build_agent
        print("BOOT: Building agent graph...", flush=True)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        _agent_graph = loop.run_until_complete(
            asyncio.wait_for(build_agent(), timeout=120.0)
        )
        print("BOOT: Agent graph ready.", flush=True)
    except Exception as exc:
        msg = f"Agent build failed: {type(exc).__name__}: {exc}"
        print(f"ERROR: {msg}", flush=True)
        traceback.print_exc()
        errors.append(msg)

    _init_error = "; ".join(errors) if errors else None
    _initialized = True


@app.entrypoint
def invoke(payload, context=None):
    """Process user input through the orchestrator agent."""
    _init_sync()

    if _agent_graph is None:
        return {"error": f"Agent failed to initialize: {_init_error or 'unknown'}"}

    # Extract prompt from various payload formats
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
        import uuid
        from langchain_core.messages import HumanMessage

        # AgentCore passes session ID via context; use it as thread_id
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

        # Use existing event loop if available, otherwise create one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            _agent_graph.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )
        )

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
