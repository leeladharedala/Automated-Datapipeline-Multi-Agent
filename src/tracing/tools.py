"""Tool call tracing wrappers.

Wraps LangChain ``BaseTool`` instances so that every invocation produces a
``"tool:{tool_name}"`` span with input/output size, category, latency, and
success attributes.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from typing import Any

from langchain_core.tools import BaseTool
from opentelemetry.trace import StatusCode

from src.tracing.provider import get_tracer
from src.tracing.utils import record_exception

logger = logging.getLogger(__name__)

TOOL_CATEGORIES: dict[str, str] = {
    "parse_document_tool": "custom",
    "submit_pr": "custom",
    "write_file": "vfs",
    "edit_file": "vfs",
    "read_file": "vfs",
    "execute": "sandbox",
    "code_interpreter": "sandbox",
}


def _get_category(tool_name: str) -> str:
    """Return the category for *tool_name*, defaulting to ``"mcp"``."""
    return TOOL_CATEGORIES.get(tool_name, "mcp")


def _serialized_size(value: Any) -> int:
    """Best-effort byte length of a JSON-serialised value."""
    try:
        return len(json.dumps(value, default=str).encode())
    except Exception:
        return len(str(value).encode())


def traced_tool(tool: BaseTool) -> BaseTool:
    """Return *tool* with ``invoke`` / ``ainvoke`` wrapped in tracing spans.

    Span name: ``"tool:{tool_name}"``

    Input attributes:
        ``tool.name``, ``tool.input_size``, ``tool.category``

    Output attributes:
        ``tool.output_size``, ``tool.success``, ``tool.latency_ms``
    """
    tool_name = tool.name
    category = _get_category(tool_name)
    span_name = f"tool:{tool_name}"

    original_invoke = tool.invoke
    original_ainvoke = tool.ainvoke

    @functools.wraps(original_invoke)
    def traced_invoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.input_size", _serialized_size(input))
            span.set_attribute("tool.category", category)
            start = time.monotonic()
            try:
                result = original_invoke(input, config=config, **kwargs)
                latency_ms = (time.monotonic() - start) * 1000
                span.set_attribute("tool.output_size", _serialized_size(result))
                span.set_attribute("tool.success", True)
                span.set_attribute("tool.latency_ms", latency_ms)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                latency_ms = (time.monotonic() - start) * 1000
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.latency_ms", latency_ms)
                record_exception(span, exc)
                raise

    @functools.wraps(original_ainvoke)
    async def traced_ainvoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.input_size", _serialized_size(input))
            span.set_attribute("tool.category", category)
            start = time.monotonic()
            try:
                result = await original_ainvoke(input, config=config, **kwargs)
                latency_ms = (time.monotonic() - start) * 1000
                span.set_attribute("tool.output_size", _serialized_size(result))
                span.set_attribute("tool.success", True)
                span.set_attribute("tool.latency_ms", latency_ms)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                latency_ms = (time.monotonic() - start) * 1000
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.latency_ms", latency_ms)
                record_exception(span, exc)
                raise

    # Use object.__setattr__ to bypass Pydantic v2 field validation on tools
    object.__setattr__(tool, "invoke", traced_invoke)
    object.__setattr__(tool, "ainvoke", traced_ainvoke)
    return tool


def trace_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """Wrap every tool in *tools* with tracing spans."""
    return [traced_tool(t) for t in tools]
