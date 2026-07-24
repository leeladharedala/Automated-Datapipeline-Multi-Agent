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

from src.anthropic_model import SanitizedChatAnthropic

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


def _setup_otlp_log_export() -> None:
    """Ship this process's log records to CloudWatch's OTLP endpoint
    (the ``otel-rt-logs`` stream) WITHOUT auto-instrumentation.

    ``opentelemetry-instrument`` cannot run here — the ADOT configurator
    patches botocore unconditionally and every boto3 call (Secrets Manager,
    S3 data sampling) dies with "maximum recursion depth exceeded". But log
    EXPORT needs no instrumentation at all: hand-construct the same
    SigV4-signing exporter the distro uses, attach it as a stdlib logging
    handler, and leave every library unpatched. The endpoint and
    log-group/stream headers are injected into the container env by the
    AgentCore platform; if they're absent (local dev), this is a no-op and
    logs keep flowing to stdout only.
    """
    try:
        region = os.environ.get("AWS_REGION", "us-west-2")
        # Preferred: explicit configuration passed by our own Terraform
        # (SUBAGENT_OTLP_LOG_GROUP / _STREAM). Fallback: the standard OTLP
        # env pair, when the platform provides it.
        log_group = os.environ.get("SUBAGENT_OTLP_LOG_GROUP", "")
        log_stream = os.environ.get("SUBAGENT_OTLP_LOG_STREAM", "otel-rt-logs")
        endpoint = os.environ.get(
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
            f"https://logs.{region}.amazonaws.com/v1/logs",
        )
        if not log_group:
            headers_env = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "")
            pairs = dict(p.split("=", 1) for p in headers_env.split(",") if "=" in p)
            log_group = pairs.get("x-aws-log-group") or ""
            log_stream = pairs.get("x-aws-log-stream") or log_stream
        if not (endpoint and log_group and log_stream):
            logging.getLogger(__name__).warning(
                "Sub-agent OTLP log export disabled: no log group configured "
                "(set SUBAGENT_OTLP_LOG_GROUP)"
            )
            return

        from botocore.session import Session
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource

        try:
            from amazon.opentelemetry.distro.exporter.otlp.aws.logs.otlp_aws_log_record_exporter import (  # noqa: E501
                OTLPAwsLogRecordExporter as _Exporter,
            )
        except ImportError:  # class name differs across distro versions
            from amazon.opentelemetry.distro.exporter.otlp.aws.logs.otlp_aws_log_record_exporter import (  # noqa: E501
                OTLPAwsLogExporter as _Exporter,
            )

        raw_attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
        base_service = dict(
            p.split("=", 1) for p in raw_attrs.split(",") if "=" in p
        ).get("service.name", "multi-agent-pipeline")
        resource = Resource.create(
            {
                "service.name": f"{base_service}-subagents",
                "aws.service.type": "gen_ai_agent",
            }
        )
        exporter = _Exporter(
            aws_region=os.environ.get("AWS_REGION", "us-west-2"),
            session=Session(),
            log_group=log_group,
            log_stream=log_stream,
            endpoint=endpoint,
        )
        provider = LoggerProvider(resource=resource)
        provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
        logging.getLogger().addHandler(handler)
        logging.getLogger(__name__).info(
            "Sub-agent OTLP log export active → %s (%s)", log_group, log_stream
        )
    except Exception as exc:  # pragma: no cover - depends on platform env
        logging.getLogger(__name__).warning(
            "Sub-agent OTLP log export not configured (%s); stdout only", exc
        )


_setup_otlp_log_export()

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
# Thinking is explicitly DISABLED for latency (adaptive thinking generates
# reasoning tokens before the visible output). NOTE: disabling it previously
# degraded generation quality — iac burned all 3 fix attempts on runs that
# used to pass first try — so watch the self-heal loops; re-enable by removing
# the `thinking` kwarg (model default is adaptive thinking ON) if quality drops.
# SanitizedChatAnthropic still strips any empty thinking blocks from replayed
# history as a safety net — see src/anthropic_model.py.
# max_tokens MUST be explicit: without langchain-model-profiles installed,
# ChatAnthropic falls back to 4096, and responses truncate mid-fence
# ("Generated 0 workflow file(s)") and burn entire self-heal budgets.
# Verified: the same generate prompt yields both complete workflow files at
# 32k where it produced a 370-char stub at the 4096 fallback.
# Set to the claude-sonnet-5 output ceiling (128000) — the API requires the
# field, so this is effectively "uncapped". It's an upper bound, not a target:
# raising it does not speed up responses that finish before the cap.
_model = SanitizedChatAnthropic(
    model=_MODEL_NAME, max_tokens=128000, thinking={"type": "disabled"}
)

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
