"""TracerProvider setup, shutdown, and NoOp fallback.

Configures the OTel TracerProvider at application startup with OTLP export
to an ADOT Collector sidecar. All behaviour is controlled via environment
variables so that tracing can be enabled, disabled, or reconfigured without
code changes.
"""

from __future__ import annotations

import logging
import os

from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NoOpTracerProvider

logger = logging.getLogger(__name__)

# Module-level reference so shutdown_tracing() can flush the provider that
# setup_tracing() created, even if someone replaces the global provider later.
_provider: TracerProvider | None = None


def setup_tracing() -> None:
    """Initialise the global TracerProvider with an OTLP exporter.

    Reads configuration from environment variables:
      OTEL_TRACING_ENABLED   – "true" (default) or "false"
      OTEL_SERVICE_NAME      – defaults to "multi-agent-data-pipeline"
      OTEL_EXPORTER_OTLP_ENDPOINT  – defaults to "http://localhost:4317"
      OTEL_EXPORTER_OTLP_PROTOCOL  – "grpc" (default) or "http/protobuf"
      OTEL_TRACES_SAMPLER          – e.g. "parentbased_traceidratio"
      OTEL_TRACES_SAMPLER_ARG      – e.g. "0.1"
      ENVIRONMENT                  – defaults to "development"
    """
    global _provider

    enabled = os.environ.get("OTEL_TRACING_ENABLED", "true").lower()
    if enabled == "false":
        trace.set_tracer_provider(NoOpTracerProvider())
        logger.info("Tracing disabled via OTEL_TRACING_ENABLED=false")
        return

    service_name = os.environ.get("OTEL_SERVICE_NAME", "multi-agent-data-pipeline")
    environment = os.environ.get("ENVIRONMENT", "development")
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    default_port = "4318" if protocol == "http/protobuf" else "4317"
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", f"http://localhost:{default_port}")
    sampler_name = os.environ.get("OTEL_TRACES_SAMPLER", "")
    sampler_arg = os.environ.get("OTEL_TRACES_SAMPLER_ARG", "1.0")

    # --- Resource ---
    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": environment,
            "cloud.provider": "aws",
            "cloud.platform": "aws_ecs",
        }
    )

    # --- Sampler ---
    sampler = _build_sampler(sampler_name, sampler_arg)

    # --- TracerProvider ---
    id_generator = None
    id_gen_env = os.environ.get("OTEL_PYTHON_ID_GENERATOR", "").lower()
    if id_gen_env in ("xray", "aws_xray"):
        try:
            from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator
            id_generator = AwsXRayIdGenerator()
            logger.info("Using AwsXRayIdGenerator for X-Ray compatibility")
        except ImportError:
            logger.warning("AwsXRayIdGenerator requested but extension package not installed.")

    provider = TracerProvider(
        resource=resource,
        sampler=sampler,
        id_generator=id_generator,
    )

    # --- Exporter + Processor ---
    try:
        exporter = _build_exporter(protocol, endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception:
        logger.warning(
            "Failed to configure OTLP exporter at %s – spans will not be exported",
            endpoint,
            exc_info=True,
        )

    _provider = provider
    trace.set_tracer_provider(provider)

    # --- Propagators (W3C TraceContext + X-Ray) ---
    _configure_propagators()

    logger.info(
        "Tracing initialised: service=%s env=%s endpoint=%s protocol=%s",
        service_name,
        environment,
        endpoint,
        protocol,
    )


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the exporter (5 s timeout)."""
    global _provider
    if _provider is not None:
        _provider.force_flush(timeout_millis=5000)
        _provider.shutdown()
        _provider = None
        logger.info("Tracing shut down")


def get_tracer(name: str = "multi-agent-pipeline") -> trace.Tracer:
    """Return a Tracer from the global TracerProvider."""
    return trace.get_tracer(name)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_exporter(protocol: str, endpoint: str):
    """Create an OTLP span exporter for the given protocol."""
    if protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPExporter,
        )
        return HTTPExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")

    # Default to gRPC
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCExporter,
    )
    return GRPCExporter(endpoint=endpoint, insecure=True)


def _build_sampler(sampler_name: str, sampler_arg: str):
    """Return an OTel sampler based on env-var configuration."""
    from opentelemetry.sdk.trace.sampling import (
        ALWAYS_ON,
        ParentBasedTraceIdRatio,
        TraceIdRatioBased,
    )

    if not sampler_name:
        return ALWAYS_ON

    try:
        ratio = float(sampler_arg)
    except (ValueError, TypeError):
        ratio = 1.0

    name = sampler_name.lower()
    if name == "traceidratio":
        return TraceIdRatioBased(ratio)
    if name == "parentbased_traceidratio":
        return ParentBasedTraceIdRatio(ratio)

    logger.warning("Unknown sampler '%s', falling back to ALWAYS_ON", sampler_name)
    return ALWAYS_ON


def _configure_propagators() -> None:
    """Set W3C TraceContext and AWS X-Ray composite propagator."""
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )
    from opentelemetry.propagators.aws import AwsXRayPropagator

    set_global_textmap(
        CompositePropagator(
            [
                TraceContextTextMapPropagator(),  # W3C TraceContext
                AwsXRayPropagator(),              # X-Ray trace header
            ]
        )
    )
