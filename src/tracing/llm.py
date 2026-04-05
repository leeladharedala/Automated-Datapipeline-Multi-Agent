"""LLM call tracing wrapper.

Wraps ``ChatAnthropic`` (or any ``BaseChatModel``) so that every ``invoke``
and ``ainvoke`` call produces an ``"llm:{model_name}"`` span with token
counts, latency, and error classification attributes.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from opentelemetry.trace import StatusCode

from src.tracing.provider import get_tracer
from src.tracing.utils import record_exception

logger = logging.getLogger(__name__)


def traced_llm(model: BaseChatModel) -> BaseChatModel:
    """Return *model* with ``invoke`` / ``ainvoke`` wrapped in tracing spans.

    Span name: ``"llm:{model_name}"``

    Input attributes:
        ``llm.model_name``, ``llm.temperature``,
        ``llm.input_message_count``, ``llm.input_token_estimate``

    Output attributes:
        ``llm.output_tokens``, ``llm.input_tokens``, ``llm.total_tokens``,
        ``llm.latency_ms``, ``llm.stop_reason``

    Error attributes:
        ``llm.error_type`` (``"rate_limit"``, ``"timeout"``, ``"api_error"``)
    """
    model_name = getattr(model, "model_name", None) or getattr(model, "model", "unknown")
    temperature = getattr(model, "temperature", None)
    span_name = f"llm:{model_name}"

    original_invoke = model.invoke
    original_ainvoke = model.ainvoke

    @functools.wraps(original_invoke)
    def traced_invoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span(span_name) as span:
            _set_input_attributes(span, model_name, temperature, input)
            start = time.monotonic()
            try:
                result = original_invoke(input, config=config, **kwargs)
                latency_ms = (time.monotonic() - start) * 1000
                _set_output_attributes(span, result, latency_ms)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                latency_ms = (time.monotonic() - start) * 1000
                span.set_attribute("llm.latency_ms", latency_ms)
                span.set_attribute("llm.error_type", _classify_error(exc))
                record_exception(span, exc)
                raise

    @functools.wraps(original_ainvoke)
    async def traced_ainvoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span(span_name) as span:
            _set_input_attributes(span, model_name, temperature, input)
            start = time.monotonic()
            try:
                result = await original_ainvoke(input, config=config, **kwargs)
                latency_ms = (time.monotonic() - start) * 1000
                _set_output_attributes(span, result, latency_ms)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                latency_ms = (time.monotonic() - start) * 1000
                span.set_attribute("llm.latency_ms", latency_ms)
                span.set_attribute("llm.error_type", _classify_error(exc))
                record_exception(span, exc)
                raise

    # Use object.__setattr__ to bypass Pydantic v2 field validation
    object.__setattr__(model, "invoke", traced_invoke)
    object.__setattr__(model, "ainvoke", traced_ainvoke)
    return model


def _estimate_tokens(messages: Any) -> int:
    """Rough token estimate: ~4 chars per token across all message content."""
    if not messages:
        return 0
    total_chars = 0
    if isinstance(messages, (list, tuple)):
        for msg in messages:
            content = getattr(msg, "content", None) or (msg if isinstance(msg, str) else "")
            total_chars += len(str(content))
    else:
        total_chars = len(str(messages))
    return max(1, total_chars // 4)


def _message_count(input: Any) -> int:
    """Return the number of messages in the input."""
    if isinstance(input, (list, tuple)):
        return len(input)
    return 1


def _set_input_attributes(
    span: Any,
    model_name: str,
    temperature: float | None,
    input: Any,
) -> None:
    """Set LLM input attributes on the span."""
    span.set_attribute("llm.model_name", model_name)
    if temperature is not None:
        span.set_attribute("llm.temperature", temperature)
    span.set_attribute("llm.input_message_count", _message_count(input))
    span.set_attribute("llm.input_token_estimate", _estimate_tokens(input))


def _set_output_attributes(span: Any, result: Any, latency_ms: float) -> None:
    """Set LLM output attributes on the span from the response."""
    span.set_attribute("llm.latency_ms", latency_ms)

    # Extract token usage from response_metadata or usage_metadata
    usage = getattr(result, "usage_metadata", None) or {}
    if not usage:
        meta = getattr(result, "response_metadata", {}) or {}
        usage = meta.get("usage", {})

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    span.set_attribute("llm.input_tokens", input_tokens)
    span.set_attribute("llm.output_tokens", output_tokens)
    span.set_attribute("llm.total_tokens", input_tokens + output_tokens)

    # Stop reason
    stop_reason = getattr(result, "response_metadata", {}).get("stop_reason", "")
    if stop_reason:
        span.set_attribute("llm.stop_reason", stop_reason)


def _classify_error(exc: Exception) -> str:
    """Classify an LLM exception into a category string."""
    exc_type = type(exc).__name__.lower()
    exc_msg = str(exc).lower()

    if "rate" in exc_type or "rate" in exc_msg or "429" in exc_msg:
        return "rate_limit"
    if "timeout" in exc_type or "timeout" in exc_msg or "timed out" in exc_msg:
        return "timeout"
    return "api_error"
