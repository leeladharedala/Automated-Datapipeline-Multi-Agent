"""HTTP request tracing middleware for the /invocations endpoint.

Creates a root span for each POST /invocations request and child spans for
request deserialization and response serialization.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.tracing.provider import get_tracer

logger = logging.getLogger(__name__)


def create_invocations_tracing_middleware() -> type[BaseHTTPMiddleware]:
    """Return a Starlette middleware class that instruments POST /invocations.

    For each request the middleware creates:
    1. A root span ``"POST /invocations"`` with HTTP and AgentCore attributes.
    2. A child span ``"http:deserialize_request"`` during body parsing.
    3. A child span ``"http:serialize_response"`` wrapping response delivery.

    Non-/invocations routes (e.g. ``/ping``) are passed through untouched.
    """

    class InvocationsTracingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            # Only instrument POST /invocations
            if request.url.path != "/invocations" or request.method != "POST":
                return await call_next(request)

            tracer = get_tracer("multi-agent-pipeline")

            with tracer.start_as_current_span("POST /invocations") as root_span:
                root_span.set_attribute("http.method", "POST")
                root_span.set_attribute("http.route", "/invocations")

                # --- Deserialize request body ---
                body_bytes: bytes = b""
                try:
                    with tracer.start_as_current_span("http:deserialize_request") as deser_span:
                        body_bytes = await request.body()
                        deser_span.set_attribute("http.request_content_length", len(body_bytes))
                except Exception as exc:
                    root_span.record_exception(exc)
                    root_span.set_status(StatusCode.ERROR, str(exc))
                    raise

                root_span.set_attribute("http.request_content_length", len(body_bytes))

                # Extract AgentCore identifiers from the body
                _set_agentcore_attributes(root_span, body_bytes)

                # --- Call the actual endpoint handler ---
                try:
                    response = await call_next(request)
                except Exception as exc:
                    root_span.record_exception(exc)
                    root_span.set_status(StatusCode.ERROR, str(exc))
                    raise

                # --- Serialize response ---
                try:
                    with tracer.start_as_current_span("http:serialize_response") as ser_span:
                        if isinstance(response, StreamingResponse):
                            # For streaming responses we record what we can
                            content_length = response.headers.get("content-length")
                            if content_length is not None:
                                ser_span.set_attribute(
                                    "http.response_content_length", int(content_length)
                                )
                            else:
                                ser_span.set_attribute("http.response_content_length", 0)
                        else:
                            body = getattr(response, "body", b"")
                            ser_span.set_attribute(
                                "http.response_content_length", len(body) if body else 0
                            )
                except Exception as exc:
                    root_span.record_exception(exc)
                    root_span.set_status(StatusCode.ERROR, str(exc))
                    raise

                # Record final status
                status_code = getattr(response, "status_code", 200)
                root_span.set_attribute("http.status_code", status_code)

                if 200 <= status_code < 400:
                    root_span.set_status(StatusCode.OK)
                else:
                    root_span.set_status(StatusCode.ERROR, f"HTTP {status_code}")

                return response

    return InvocationsTracingMiddleware


def _set_agentcore_attributes(span: trace.Span, body_bytes: bytes) -> None:
    """Parse the request body and set thread_id / run_id on the span."""
    try:
        body = json.loads(body_bytes)
        if isinstance(body, dict):
            thread_id = body.get("thread_id", "")
            run_id = body.get("run_id", "")
            if thread_id:
                span.set_attribute("agentcore.thread_id", str(thread_id))
            if run_id:
                span.set_attribute("agentcore.run_id", str(run_id))
    except (json.JSONDecodeError, TypeError):
        # Body isn't valid JSON — nothing to extract
        pass
