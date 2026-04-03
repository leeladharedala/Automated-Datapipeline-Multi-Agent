"""OpenTelemetry tracing module for the multi-agent data pipeline."""

from src.tracing.agui import wrap_agui_handler
from src.tracing.llm import traced_llm
from src.tracing.memory import (
    traced_post_model_hook,
    traced_pre_model_hook,
    traced_store_put,
    traced_store_search,
)
from src.tracing.middleware import instrument_middleware
from src.tracing.parser import traced_classify_and_extract, traced_parse_pipeline_document
from src.tracing.provider import get_tracer, setup_tracing, shutdown_tracing
from src.tracing.retry import traced_retry_loop
from src.tracing.tools import trace_tools, traced_tool
from src.tracing.utils import record_exception, traced, traced_span

__all__ = [
    "setup_tracing",
    "shutdown_tracing",
    "get_tracer",
    "traced",
    "traced_span",
    "record_exception",
    "wrap_agui_handler",
    "traced_llm",
    "traced_tool",
    "trace_tools",
    "instrument_middleware",
    "traced_parse_pipeline_document",
    "traced_classify_and_extract",
    "traced_pre_model_hook",
    "traced_post_model_hook",
    "traced_store_search",
    "traced_store_put",
    "traced_retry_loop",
]
