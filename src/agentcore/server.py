"""
AgentCore AG-UI server — matches the official AWS docs pattern exactly.

Uses uvicorn.run() inline (not python -m uvicorn) which is the pattern
shown in the AWS AgentCore AG-UI docs.
"""

import asyncio
import logging
import os
import traceback

import boto3
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from copilotkit import CopilotKitMiddleware
from copilotkit.langgraph import LangGraphAGUIAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_copilotkit = None

app = FastAPI()


@app.get("/ping")
async def ping():
    return JSONResponse({"status": "Healthy"})


@app.post("/invocations")
async def invocations(request: Request):
    global _copilotkit

    # Lazy init on first request
    if _copilotkit is None:
        try:
            _copilotkit = await _build_copilotkit()
        except Exception as exc:
            logger.error("Init failed: %s", exc)
            return JSONResponse(status_code=503, content={"error": str(exc)})

    return await _copilotkit.handle_request(request)


async def _build_copilotkit():
    """Build agent and CopilotKit middleware."""
    # Resolve secrets
    arn = os.environ.get("ANTHROPIC_API_KEY_SECRET_ARN", "")
    if arn and not os.environ.get("ANTHROPIC_API_KEY"):
        region = os.environ.get("AWS_REGION", "us-west-2")
        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=arn)
        os.environ["ANTHROPIC_API_KEY"] = resp["SecretString"]
        logger.info("Resolved ANTHROPIC_API_KEY")

    # Build agent
    from src.main import build_agent
    logger.info("Building agent graph...")
    graph = await build_agent()
    logger.info("Agent graph ready.")

    # Wrap with CopilotKit AG-UI
    agent = LangGraphAGUIAgent(graph=graph, name="orchestrator")
    return CopilotKitMiddleware(agents=[agent])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
