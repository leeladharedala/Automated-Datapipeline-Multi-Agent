"""Retry loop tracing wrapper.

Instruments the validate→fix retry cycle in sub-agent graphs with
OpenTelemetry spans so that each retry attempt and the overall loop
are visible in the trace waterfall.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from opentelemetry.trace import StatusCode

from src.tracing.provider import get_tracer
from src.tracing.utils import record_exception


def traced_retry_loop(
    graph_name: str,
    validate_fn: Callable,
    fix_fn: Callable,
    should_retry_fn: Callable,
) -> tuple[Callable, Callable, Callable]:
    """Return instrumented versions of *validate*, *fix*, and *should_retry*.

    Creates:
    - ``"retry_loop:{graph_name}"`` span encompassing all retry attempts.
    - ``"retry:{graph_name}.attempt_{n}"`` child spans for each fix attempt.

    Retry span attributes:
        ``retry.attempt``, ``retry.max_attempts``,
        ``retry.validation_error`` (truncated 512 chars), ``retry.graph_name``

    Loop span attributes:
        ``retry.total_attempts``, ``retry.final_result`` (``"passed"`` | ``"exhausted"``),
        ``retry.total_duration_ms``

    The loop span is started lazily on the first validation call and ended
    when the routing decision is *not* ``"fix"`` (i.e. the loop concludes).
    """

    # Mutable state shared across the three wrapped functions within a single
    # graph execution.  Each graph invocation gets its own closure via
    # ``traced_retry_loop``, so there is no cross-invocation leakage.
    _loop_ctx: dict[str, Any] = {
        "span": None,
        "token": None,
        "start_ns": 0,
        "attempt": 0,
        "attempt_span": None,
        "attempt_token": None,
    }

    tracer = get_tracer("multi-agent-pipeline")

    def _ensure_loop_span() -> None:
        """Start the loop span if it hasn't been started yet."""
        if _loop_ctx["span"] is None:
            span = tracer.start_span(f"retry_loop:{graph_name}")
            span.set_attribute("retry.graph_name", graph_name)
            _loop_ctx["span"] = span
            _loop_ctx["start_ns"] = time.monotonic_ns()
            _loop_ctx["attempt"] = 0

    def _end_loop_span(final_result: str) -> None:
        """End the loop span with summary attributes."""
        span = _loop_ctx.get("span")
        if span is None:
            return
        elapsed_ms = (time.monotonic_ns() - _loop_ctx["start_ns"]) / 1_000_000
        span.set_attribute("retry.total_attempts", _loop_ctx["attempt"])
        span.set_attribute("retry.final_result", final_result)
        span.set_attribute("retry.total_duration_ms", round(elapsed_ms, 2))
        span.set_status(StatusCode.OK)
        span.end()
        _loop_ctx["span"] = None

    # -- wrapped validate --------------------------------------------------

    @functools.wraps(validate_fn)
    def wrapped_validate(state: Any, *args: Any, **kwargs: Any) -> Any:
        _ensure_loop_span()
        try:
            return validate_fn(state, *args, **kwargs)
        except Exception as exc:
            loop_span = _loop_ctx.get("span")
            if loop_span is not None:
                record_exception(loop_span, exc)
            raise

    # -- wrapped fix -------------------------------------------------------

    @functools.wraps(fix_fn)
    def wrapped_fix(state: Any, *args: Any, **kwargs: Any) -> Any:
        _ensure_loop_span()
        _loop_ctx["attempt"] += 1
        attempt = _loop_ctx["attempt"]

        max_attempts = state.get("max_attempts", 3) if isinstance(state, dict) else 3
        validation_error = (
            state.get("validation_output", "") if isinstance(state, dict) else ""
        )

        span_name = f"retry:{graph_name}.attempt_{attempt}"
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("retry.attempt", attempt)
            span.set_attribute("retry.max_attempts", max_attempts)
            span.set_attribute(
                "retry.validation_error", str(validation_error)[:512]
            )
            span.set_attribute("retry.graph_name", graph_name)
            try:
                result = fix_fn(state, *args, **kwargs)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                record_exception(span, exc)
                raise

    # -- wrapped should_retry ----------------------------------------------

    @functools.wraps(should_retry_fn)
    def wrapped_should_retry(state: Any, *args: Any, **kwargs: Any) -> Any:
        decision = should_retry_fn(state, *args, **kwargs)

        # If the decision is not "fix", the retry loop is over.
        if str(decision) != "fix":
            final_result = (
                "passed" if state.get("validation_passed", False) else "exhausted"
            ) if isinstance(state, dict) else str(decision)
            _end_loop_span(final_result)

        return decision

    return wrapped_validate, wrapped_fix, wrapped_should_retry
