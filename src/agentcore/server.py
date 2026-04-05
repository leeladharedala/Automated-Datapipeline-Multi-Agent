"""
AgentCore AG-UI server — FastAPI + uvicorn + CopilotKit.

Uses async lifespan for initialization so uvicorn starts immediately
and /ping returns 200 while the agent builds in the background.
"""

import asyncio
import logging
import os
import traceback
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from copilotkit import CopilotKitMiddleware
from copilotkit.langgraph import LangGraphAGUIAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_copilotkit = None
_init_error = None


def _resolve_secrets():
    arn = os.environ.get("ANTHROPIC_API_KEY_SECRET_ARN", "")
    if arn and not os.environ.get("ANTHROPIC_API_KEY"):
        region = os.environ.get("AWS_REGION", "us-west-2")
        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=arn)
        os.environ["ANTHROPIC_API_KEY"] = resp["SecretString"]
        print("BOOT: Resolved ANTHROPIC_API_KEY", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _copilotkit, _init_error

    try:
        _resolve_secrets()
    except Exception as exc:
        print(f"ERROR: Secrets failed: {exc}", flush=True)

    try:
        from src.main import build_agent
        print("BOOT: Building agent...", flush=True)
        graph = await build_agent()
        agent = LangGraphAGUIAgent(graph=graph, name="orchestrator")
        _copilotkit = CopilotKitMiddleware(agents=[agent])
        print("BOOT: AG-UI ready.", flush=True)
    except Exception as exc:
        _init_error = str(exc)
        print(f"ERROR: Agent build failed: {exc}", flush=True)
        traceback.print_exc()

    yield
    _copilotkit = None


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
async def ping():
    return {"status": "Healthy"}


@app.post("/invocations")
async def invocations(request: Request):
    if _copilotkit is None:
        return JSONResponse(status_code=503, content={"error": _init_error or "not ready"})
    return await _copilotkit.handle_request(request)
