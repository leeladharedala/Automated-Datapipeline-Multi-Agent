"""
AgentCore diagnostic server — tests imports one by one.

This server boots with BedrockAgentCoreApp (which we know works) and
tests each heavy import individually, reporting which ones succeed
and which ones crash. Check CloudWatch logs after invoking.
"""

print("BOOT: Diagnostic server starting...", flush=True)

import sys
import traceback

from bedrock_agentcore.runtime import BedrockAgentCoreApp

print("BOOT: BedrockAgentCoreApp imported OK.", flush=True)

app = BedrockAgentCoreApp()

# Track import results
import_results = []


def _try_import(description, import_fn):
    """Try an import and record the result."""
    try:
        import_fn()
        msg = f"OK: {description}"
        print(f"BOOT: {msg}", flush=True)
        import_results.append(msg)
    except Exception as exc:
        msg = f"FAIL: {description} -> {type(exc).__name__}: {exc}"
        print(f"BOOT: {msg}", flush=True)
        traceback.print_exc()
        import_results.append(msg)


# Test each import group
_try_import("boto3", lambda: __import__("boto3"))
_try_import("yaml", lambda: __import__("yaml"))
_try_import("fastapi", lambda: __import__("fastapi"))
_try_import("uvicorn", lambda: __import__("uvicorn"))
_try_import("opentelemetry.api", lambda: __import__("opentelemetry"))
_try_import("opentelemetry.sdk", lambda: __import__("opentelemetry.sdk.trace"))
_try_import("opentelemetry.exporter.grpc",
            lambda: __import__("opentelemetry.exporter.otlp.proto.grpc"))
_try_import("opentelemetry.exporter.http",
            lambda: __import__("opentelemetry.exporter.otlp.proto.http"))
_try_import("opentelemetry.propagators.aws",
            lambda: __import__("opentelemetry.propagators.aws"))
_try_import("langchain_core",
            lambda: __import__("langchain_core"))
_try_import("langchain",
            lambda: __import__("langchain"))
_try_import("langchain.agents.middleware.types",
            lambda: __import__("langchain.agents.middleware.types"))
_try_import("langchain_anthropic",
            lambda: __import__("langchain_anthropic"))
_try_import("langgraph",
            lambda: __import__("langgraph"))
_try_import("langgraph_checkpoint_aws",
            lambda: __import__("langgraph_checkpoint_aws"))
_try_import("deepagents",
            lambda: __import__("deepagents"))
_try_import("copilotkit",
            lambda: __import__("copilotkit"))
_try_import("copilotkit.langgraph",
            lambda: __import__("copilotkit.langgraph"))
_try_import("langchain_mcp_adapters",
            lambda: __import__("langchain_mcp_adapters"))
_try_import("github (PyGithub)",
            lambda: __import__("github"))
_try_import("starlette",
            lambda: __import__("starlette"))

# Now try the actual src imports
_try_import("src.tracing",
            lambda: __import__("src.tracing"))
_try_import("src.main",
            lambda: __import__("src.main"))

print(f"BOOT: Import diagnostics complete. {len(import_results)} tests run.", flush=True)


@app.entrypoint
def invoke(payload, context=None):
    """Return import diagnostic results."""
    return {
        "result": "\n".join(import_results),
        "total": len(import_results),
        "failures": [r for r in import_results if r.startswith("FAIL")],
    }


if __name__ == "__main__":
    print("BOOT: Starting diagnostic server...", flush=True)
    app.run()
