"""OrchestratorMiddleware tracing instrumentation.

Returns a subclass of the given middleware class that adds tracing spans
around ``modify_request``, ``_extract_dispatch_updates``,
``wrap_model_call``, and ``awrap_model_call``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.trace import StatusCode

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

_TRACER_NAME = "multi-agent-pipeline"


def instrument_middleware(middleware_cls: type) -> type:
    """Return a subclass of *middleware_cls* with tracing spans added.

    Instrumented methods:
    - ``modify_request``  → ``"middleware:modify_request"`` span
    - ``wrap_model_call`` / ``awrap_model_call`` → ``"middleware:model_call"`` span
    - ``after_model`` / ``aafter_model`` → ``"middleware:dispatch_update"`` span
      (only when dispatch updates are detected)
    """

    class InstrumentedMiddleware(middleware_cls):  # type: ignore[valid-type]
        """Tracing-aware subclass of the original middleware."""

        # -- modify_request ------------------------------------------------

        def modify_request(self, request: Any) -> Any:
            tracer = trace.get_tracer(_TRACER_NAME)
            phase = request.state.get("pipeline_phase", "planning") if hasattr(request, "state") else "unknown"
            with tracer.start_as_current_span("middleware:modify_request") as span:
                span.set_attribute("middleware.pipeline_phase", phase)
                try:
                    result = super().modify_request(request)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        # -- wrap_model_call (sync) ----------------------------------------

        def wrap_model_call(
            self,
            request: Any,
            handler: Callable,
        ) -> Any:
            tracer = trace.get_tracer(_TRACER_NAME)
            with tracer.start_as_current_span("middleware:model_call") as span:
                try:
                    result = super().wrap_model_call(request, handler)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        # -- awrap_model_call (async) --------------------------------------

        async def awrap_model_call(
            self,
            request: Any,
            handler: Callable[..., Awaitable],
        ) -> Any:
            tracer = trace.get_tracer(_TRACER_NAME)
            with tracer.start_as_current_span("middleware:model_call") as span:
                try:
                    result = await super().awrap_model_call(request, handler)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        # -- after_model (sync) – dispatch update tracking -----------------

        def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
            updates = super().after_model(state, runtime)
            if updates and "dispatch_statuses" in updates:
                tracer = trace.get_tracer(_TRACER_NAME)
                with tracer.start_as_current_span("middleware:dispatch_update") as span:
                    agents = list(updates["dispatch_statuses"].keys())
                    # OTel attributes must be scalar or sequence of single type;
                    # join list to comma-separated string to avoid 400 Bad Request.
                    span.set_attribute("middleware.agents_updated", ", ".join(agents))
                    span.set_attribute("middleware.agent_count", len(agents))
                    span.set_status(StatusCode.OK)
            return updates

        # -- aafter_model (async) – dispatch update tracking ---------------

        async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
            updates = await super().aafter_model(state, runtime)
            if updates and "dispatch_statuses" in updates:
                tracer = trace.get_tracer(_TRACER_NAME)
                with tracer.start_as_current_span("middleware:dispatch_update") as span:
                    agents = list(updates["dispatch_statuses"].keys())
                    # OTel attributes must be scalar or sequence of single type;
                    # join list to comma-separated string to avoid 400 Bad Request.
                    span.set_attribute("middleware.agents_updated", ", ".join(agents))
                    span.set_attribute("middleware.agent_count", len(agents))
                    span.set_status(StatusCode.OK)
            return updates

    InstrumentedMiddleware.__name__ = f"Instrumented{middleware_cls.__name__}"
    InstrumentedMiddleware.__qualname__ = f"Instrumented{middleware_cls.__qualname__}"

    return InstrumentedMiddleware
