"""
AgentCore Entrypoint — AG-UI protocol server for CopilotKit frontend.

Uses FastAPI + uvicorn on port 8080 with /invocations (POST) and /ping (GET)
as required by AgentCore's AGUI protocol support. The CopilotKitMiddleware
handles AG-UI event encoding and SSE streaming.
"""

print("BOOT: AgentCore server module loading...", flush=True)

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

print("BOOT: Core imports OK.", flush=True)

from copilotkit import CopilotKitMiddleware
from copilotkit.langgraph import LangGraphAGUIAgent

print("BOOT: CopilotKit imports OK.", flush=True)

from src.main import build_agent
from src.tracing import setup_tracing, shutdown_tracing
from src.tracing.agui import wrap_agui_handler
from src.tracing.server import create_invocations_tracing_middleware

print("BOOT: All imports successful.", flush=True)

logger = logging.getLogger(__name__)

_agent_graph = None
_copilotkit_middleware = None


def _resolve_secrets():
    """Resolve API keys from Secrets Manager ARNs into env vars at boot."""
    arn = os.environ.get("ANTHROPIC_API_KEY_SECRET_ARN", "")
    if arn and not os.environ.get("ANTHROPIC_API_KEY"):
        region = os.environ.get("AWS_REGION", "us-west-2")
        try:
            client = boto3.client("secretsmanager", region_name=region)
            resp = client.get_secret_value(SecretId=arn)
            os.environ["ANTHROPIC_API_KEY"] = resp["SecretString"]
            print("BOOT: Resolved ANTHROPIC_API_KEY from Secrets Manager", flush=True)
        except Exception as exc:
            print(f"ERROR: Failed to resolve Anthropic API key: {exc}", flush=True)
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize tracing, secrets, and the orchestrator graph at startup."""
    global _agent_graph, _copilotkit_middleware

    try:
        setup_tracing()
        print("BOOT: Tracing initialized.", flush=True)
    except Exception as exc:
        print(f"WARNING: Tracing setup failed: {exc}", flush=True)

    try:
        _resolve_secrets()
    except Exception as exc:
        print(f"ERROR: Secret resolution failed — continuing without API key: {exc}", flush=True)

    try:
        print("BOOT: Building orchestrator agent graph...", flush=True)
        _agent_graph = await asyncio.wait_for(build_agent(), timeout=120.0)

        agent = LangGraphAGUIAgent(
            graph=_agent_graph,
            name="orchestrator",
        )
        _copilotkit_middleware = CopilotKitMiddleware(agents=[agent])

        # Wrap handle_request with AG-UI stream tracing
        _copilotkit_middleware.handle_request = wrap_agui_handler(
            _copilotkit_middleware.handle_request
        )

        print("BOOT: Agent graph initialized and ready.", flush=True)
    except asyncio.TimeoutError:
        print("ERROR: Agent build timed out after 120s (MCP tools may be hanging).", flush=True)
    except Exception as exc:
        print(f"ERROR: Agent build failed: {exc}", flush=True)
        logger.error("Agent graph initialization failed: %s", exc, exc_info=True)

    yield

    _agent_graph = None
    _copilotkit_middleware = None
    try:
        shutdown_tracing()
    except Exception:
        pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(create_invocations_tracing_middleware())


@app.get("/ping")
async def ping():
    """Health check — always return 200 so AgentCore keeps the container alive."""
    return {"status": "Healthy", "agent_ready": _agent_graph is not None}


@app.post("/invocations")
async def invocations(request: Request):
    """AG-UI protocol handler — delegates to CopilotKitMiddleware for SSE streaming."""
    if _copilotkit_middleware is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent graph not yet initialized"},
        )

    # Delegate to CopilotKitMiddleware which handles AG-UI event encoding
    # and SSE streaming back to the frontend.
    return await _copilotkit_middleware.handle_request(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
