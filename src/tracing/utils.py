"""Decorators and helpers for span creation."""

from __future__ import annotations

import asyncio
import functools
from contextlib import contextmanager
from typing import Any, Callable, Generator

from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode


def traced(
    span_name: str | None = None,
    attributes: dict[str, str] | None = None,
) -> Callable:
    """Decorator that wraps a function in a span.

    If *span_name* is ``None``, the function's qualified name is used.
    Supports both sync and async functions.
    Records exceptions and sets ERROR status on failure, OK on success.
    """

    def decorator(fn: Callable) -> Callable:
        name = span_name or fn.__qualname__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = trace.get_tracer("multi-agent-pipeline")
                with tracer.start_as_current_span(name) as span:
                    if attributes:
                        for k, v in attributes.items():
                            span.set_attribute(k, v)
                    try:
                        result = await fn(*args, **kwargs)
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as exc:
                        record_exception(span, exc)
                        raise

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer("multi-agent-pipeline")
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                try:
                    result = fn(*args, **kwargs)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as exc:
                    record_exception(span, exc)
                    raise

        return sync_wrapper

    return decorator


@contextmanager
def traced_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    tracer_name: str = "multi-agent-pipeline",
) -> Generator[Span, None, None]:
    """Context manager that creates a span.

    Records exceptions and sets ERROR status on failure, OK on success.

    Usage::

        with traced_span("my_operation", {"key": "value"}) as span:
            result = do_work()
            span.set_attribute("result.size", len(result))
    """
    tracer = trace.get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        try:
            yield span
            span.set_status(StatusCode.OK)
        except Exception as exc:
            record_exception(span, exc)
            raise


def record_exception(span: Span, exc: Exception) -> None:
    """Record an exception on a span and set ERROR status."""
    span.record_exception(exc)
    span.set_status(StatusCode.ERROR, str(exc))
