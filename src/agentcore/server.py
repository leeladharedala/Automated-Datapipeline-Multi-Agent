"""
AgentCore Entrypoint — FastAPI AG-UI server with CopilotKit.

Initialization happens at module load time so the agent is ready
when AgentCore starts polling /ping.
"""

print("BOOT: Server module loading...", flush=True)

import asyncio
import logging
import os
import traceback

import boto3
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from copilotkit import CopilotKitMiddleware
from copilotkit.langgraph import LangGraphAGUIAgent

print("BOOT: All imports OK.", flush=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_agent_graph = None
_copilotkit_middleware = None
_init_error = None


def _build_sync():
    """Build agent at module load time."""
    global _agent_graph, _copilotkit_middleware, _init_error
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
        print(f"ERROR: {exc}", flush=True)

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
        errors.append(f"Build: {exc}")
        print(f"ERROR: Agent build failed: {exc}", flush=True)
        traceback.print_exc()

    # Setup CopilotKit middleware
    if _agent_graph:
        try:
            agent = LangGraphAGUIAgent(graph=_agent_graph, name="orchestrator")
            _copilotkit_middleware = CopilotKitMiddleware(agents=[agent])
            print("BOOT: CopilotKit middleware ready.", flush=True)
        except Exception as exc:
            errors.append(f"CopilotKit: {exc}")
            print(f"ERROR: CopilotKit setup failed: {exc}", flush=True)

    _init_error = "; ".join(errors) if errors else None


# Initialize at module load
print("BOOT: Starting init...", flush=True)
_build_sync()
print(f"BOOT: Done. Agent ready: {_agent_graph is not None}", flush=True)

app = FastAPI()


@app.get("/ping")
async def ping():
    return {"status": "Healthy", "agent_ready": _agent_graph is not None}


@app.post("/invocations")
async def invocations(request: Request):
    if _copilotkit_middleware is None:
        return JSONResponse(status_code=503, content={"error": f"Not ready: {_init_error}"})
    return await _copilotkit_middleware.handle_request(request)
