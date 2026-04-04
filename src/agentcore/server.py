"""
AgentCore Entrypoint — FastAPI AG-UI protocol server.

Exposes POST /invocations and GET /ping for the AgentCore Runtime.
Wraps the DeepAgent orchestrator graph with LangGraphAGUIAgent
(AG-UI event encoding) and CopilotKitMiddleware (frontend state sync).
"""

import logging
import os
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from copilotkit import CopilotKitMiddleware
from copilotkit.langgraph import LangGraphAGUIAgent

from src.main import build_agent
from src.tracing import setup_tracing, shutdown_tracing
from src.tracing.agui import wrap_agui_handler
from src.tracing.server import create_invocations_tracing_middleware

logger = logging.getLogger(__name__)

# Module-level state for the initialized agent graph
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
            logger.info("Resolved ANTHROPIC_API_KEY from Secrets Manager")
        except Exception as exc:
            logger.error("Failed to resolve Anthropic API key from %s: %s", arn, exc)
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize tracing and the orchestrator graph once at startup.

    IMPORTANT: This must never crash or the container dies before the
    healthcheck endpoint becomes reachable, causing AgentCore to kill
    it with a 424 RuntimeClientError and zero CloudWatch logs.
    """
    global _agent_graph, _copilotkit_middleware

    try:
        setup_tracing()
    except Exception as exc:
        print(f"WARNING: Tracing setup failed: {exc}", flush=True)

    try:
        _resolve_secrets()
    except Exception as exc:
        print(f"ERROR: Secret resolution failed: {exc}", flush=True)

    try:
        logger.info("Building orchestrator agent graph...")
        _agent_graph = await build_agent()

        agent = LangGraphAGUIAgent(
            graph=_agent_graph,
            name="orchestrator",
        )
        _copilotkit_middleware = CopilotKitMiddleware(agents=[agent])

        # Wrap handle_request with AG-UI stream tracing
        _copilotkit_middleware.handle_request = wrap_agui_handler(
            _copilotkit_middleware.handle_request
        )

        logger.info("Agent graph initialized and ready.")
    except Exception as exc:
        # Log but do NOT re-raise — let the server boot so /ping is reachable
        # and AgentCore doesn't kill us before we can emit any logs.
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
    """Health check endpoint required by AgentCore Runtime."""
    if _agent_graph is None:
        return JSONResponse(
            status_code=503,
            content={"status": "Unavailable", "detail": "Agent graph not yet initialized"},
        )
    return {"status": "Healthy"}


@app.post("/invocations")
async def invocations(request: Request):
    """AG-UI protocol handler — receives RunAgentInput, streams AG-UI events."""
    if _copilotkit_middleware is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent graph not yet initialized"},
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON payload"},
        )

    # Validate required fields per AG-UI RunAgentInput schema
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "Payload must be a JSON object"},
        )

    missing = []
    for field in ("thread_id", "run_id", "messages"):
        if field not in body:
            missing.append(field)
    if missing:
        return JSONResponse(
            status_code=400,
            content={"error": f"Missing required fields: {', '.join(missing)}"},
        )

    if not isinstance(body.get("messages"), list):
        return JSONResponse(
            status_code=400,
            content={"error": "Field 'messages' must be a list"},
        )

    if not isinstance(body.get("thread_id"), str) or not body["thread_id"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Field 'thread_id' must be a non-empty string"},
        )

    if not isinstance(body.get("run_id"), str) or not body["run_id"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Field 'run_id' must be a non-empty string"},
        )

    # Delegate to CopilotKitMiddleware which handles AG-UI streaming
    return await _copilotkit_middleware.handle_request(request)
