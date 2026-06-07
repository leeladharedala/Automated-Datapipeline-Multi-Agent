"""OpenTelemetry tracing module for the multi-agent data pipeline."""

from src.tracing.parser import traced_classify_and_extract, traced_parse_pipeline_document
from src.tracing.provider import get_tracer, setup_tracing, shutdown_tracing
from src.tracing.utils import record_exception, traced, traced_span

__all__ = [
    "setup_tracing",
    "shutdown_tracing",
    "get_tracer",
    "traced",
    "traced_span",
    "record_exception",
    "traced_parse_pipeline_document",
    "traced_classify_and_extract",
]
