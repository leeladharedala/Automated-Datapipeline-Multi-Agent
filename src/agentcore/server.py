"""
AgentCore Entrypoint — FastAPI AG-UI server with CopilotKit.

Uvicorn boots fast. OTEL configured programmatically (no auto-instrument wrapper).
Agent builds lazily on first request.
"""

import asyncio
import logging
import os
import traceback

import boto3
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from copilotkit import CopilotKitMiddleware
from copilotkit.langgraph import LangGraphAGUIAgent

# Programmatic OTEL setup (replaces opentelemetry-instrument wrapper)
try:
    if os.environ.get("AGENT_OBSERVABILITY_ENABLED", "").lower() == "true":
        from aws_opentelemetry_distro.aws_opentelemetry_configurator import AwsOpenTelemetryConfigurator
        configurator = AwsOpenTelemetryConfigurator()
        configurator.configure()
except Exception:
    pass  # OTEL setup is best-effort

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

_agent_graph = None
_copilotkit_middleware = None
_initialized = False
_init_error = None


async def _ensure_init():
    """Lazy init — build agent on first request."""
    global _agent_graph, _copilotkit_middleware, _initialized, _init_error

    if _initialized:
        return

    errors = []

    # Resolve secrets
    try:
        arn = os.environ.get("ANTHROPIC_API_KEY_SECRET_ARN", "")
        if arn and not os.environ.get("ANTHROPIC_API_KEY"):
            region = os.environ.get("AWS_REGION", "us-west-2")
            client = boto3.client("secretsmanager", region_name=region)
            resp = client.get_secret_value(SecretId=arn)
            os.environ["ANTHROPIC_API_KEY"] = resp["SecretString"]
    except Exception as exc:
        errors.append(f"Secrets: {exc}")

    # Build agent
    try:
        from src.main import build_agent
        _agent_graph = await build_agent()
    except Exception as exc:
        errors.append(f"Build: {exc}")
        traceback.print_exc()

    # Setup CopilotKit
    if _agent_graph:
        try:
            agent = LangGraphAGUIAgent(graph=_agent_graph, name="orchestrator")
            _copilotkit_middleware = CopilotKitMiddleware(agents=[agent])
        except Exception as exc:
            errors.append(f"CopilotKit: {exc}")

    _init_error = "; ".join(errors) if errors else None
    _initialized = True


@app.get("/ping")
async def ping():
    return {"status": "Healthy", "agent_ready": _agent_graph is not None}


@app.post("/invocations")
async def invocations(request: Request):
    await _ensure_init()
    if _copilotkit_middleware is None:
        return JSONResponse(status_code=503, content={"error": f"Not ready: {_init_error}"})
    return await _copilotkit_middleware.handle_request(request)
