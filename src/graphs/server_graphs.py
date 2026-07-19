"""Compiled sub-agent graphs exposed to the co-located LangGraph API server.

`langgraph.json` (repo root) references the module-level variables defined
below. Each is a compiled ``StateGraph`` produced by the existing factories
(`build_iac_graph`, `build_cicd_graph`, `build_data_eng_graph`) — their nodes,
edges, and extended state are reused **unchanged** (Req 1.5).

These graphs are compiled **without** ``AgentCoreMemorySaver``: the co-located
LangGraph API server (Process B) owns their persistence and supplies its own
checkpointer/store (Req 13.3, 13.4). This is the single place where the
sub-agent graphs are compiled for the co-located server (Req 2.3).
"""

import logging
import os

from langchain_anthropic import ChatAnthropic

from src.graphs import (
    build_iac_graph,
    build_cicd_graph,
    build_data_eng_graph,
)


class _SuppressOtelDetachNoise(logging.Filter):
    """Drop the benign "Failed to detach context" ValueError.

    Now that this process runs under opentelemetry-instrument, the langchain
    auto-instrumentation wrapping ``Pregel.astream`` conflicts with the
    graphs' background-loop ``run_coroutine_threadsafe`` pattern: the OTel
    context token is reset from a different contextvars Context, producing a
    full-stack ERROR log per run. Same suppression as the ingress
    (src/agentcore/server.py) so ERROR-level logs stay meaningful.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Failed to detach context" not in record.getMessage()


logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachNoise())

# Env-driven model defaults, consistent with src/main.py's Supervisor build.
_MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


def _resolve_anthropic_key() -> None:
    """Resolve ANTHROPIC_API_KEY from Secrets Manager for THIS process.

    The ingress (src/agentcore/server.py) resolves the key into its own
    process env only. The co-located server is a separate process, so without
    this every sub-agent run fails its first LLM call with "Could not resolve
    authentication method". Best-effort: local dev may set the key directly.
    """
    arn = os.environ.get("ANTHROPIC_API_KEY_SECRET_ARN", "")
    if not arn or os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        import boto3

        region = os.environ.get("AWS_REGION", "us-west-2")
        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=arn)
        os.environ["ANTHROPIC_API_KEY"] = resp["SecretString"]
    except Exception as exc:  # pragma: no cover - depends on AWS environment
        import logging

        logging.getLogger(__name__).warning(
            "Could not resolve ANTHROPIC_API_KEY from Secrets Manager: %s", exc
        )


_resolve_anthropic_key()

# One shared model for every sub-agent graph on the co-located server.
_model = ChatAnthropic(model=_MODEL_NAME)

# Module-level compiled graph handles referenced by langgraph.json.
# graph_id values ("iac" / "cicd" / "data-eng") match the AsyncSubAgent specs.
# MCP/browser tools are loaded lazily inside the sub-agent nodes, so the server
# handles compile with empty tool lists here (Req 2.3, 1.5).
iac_graph = build_iac_graph(model=_model, tools=[])
cicd_graph = build_cicd_graph(model=_model)
data_eng_graph = build_data_eng_graph(model=_model, tools=[])


def supervisor_graph():
    """Zero-arg factory exposing the Supervisor as a launchable graph (Req 2.4).

    The Supervisor is built asynchronously (``src.main.build_agent`` is a
    coroutine). The import is deferred to call time to avoid a circular import
    with ``src.main`` at server start (``src.main`` imports from ``src.graphs``).

    ``langgraph.json`` may reference this callable as the ``supervisor`` graph;
    the LangGraph API server invokes zero-arg factories to obtain the compiled
    graph. Returns the compiled Supervisor graph.
    """
    import asyncio

    from src.main import build_agent

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # An event loop is already running (rare for the server's import-time
        # factory call); run the coroutine on a dedicated loop instead.
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(build_agent())
        finally:
            new_loop.close()

    return loop.run_until_complete(build_agent())
