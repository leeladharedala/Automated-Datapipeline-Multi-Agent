"""
AgentCore Entrypoint — FastAPI AG-UI server.

Only FastAPI is imported at module level. All heavy imports (CopilotKit,
LangChain, etc.) are deferred to first request so uvicorn boots fast.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

_middleware = None
_initialized = False
_init_error = None


async def _ensure_init():
    global _middleware, _initialized, _init_error
    if _initialized:
        return

    import asyncio
    import os
    import traceback
    errors = []

    # Secrets
    try:
        import boto3
        arn = os.environ.get("ANTHROPIC_API_KEY_SECRET_ARN", "")
        if arn and not os.environ.get("ANTHROPIC_API_KEY"):
            region = os.environ.get("AWS_REGION", "us-west-2")
            client = boto3.client("secretsmanager", region_name=region)
            resp = client.get_secret_value(SecretId=arn)
            os.environ["ANTHROPIC_API_KEY"] = resp["SecretString"]
    except Exception as exc:
        errors.append(f"Secrets: {exc}")

    # Build agent + CopilotKit
    try:
        from src.main import build_agent
        from copilotkit import CopilotKitMiddleware
        from copilotkit.langgraph import LangGraphAGUIAgent

        graph = await build_agent()
        agent = LangGraphAGUIAgent(graph=graph, name="orchestrator")
        _middleware = CopilotKitMiddleware(agents=[agent])
    except Exception as exc:
        errors.append(f"Build: {exc}")
        traceback.print_exc()

    _init_error = "; ".join(errors) if errors else None
    _initialized = True


@app.get("/ping")
async def ping():
    return {"status": "Healthy"}


@app.post("/invocations")
async def invocations(request: Request):
    await _ensure_init()
    if _middleware is None:
        return JSONResponse(status_code=503, content={"error": f"Not ready: {_init_error}"})
    return await _middleware.handle_request(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
