"""AG-UI event stream tracing.

Wraps the CopilotKitMiddleware event stream to create spans for the overall
SSE stream and for each individual AG-UI event emitted to the client.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from typing import Any, AsyncIterator, Callable

from opentelemetry.trace import StatusCode

from src.tracing.provider import get_tracer
from src.tracing.utils import record_exception

logger = logging.getLogger(__name__)


def wrap_agui_handler(original_handler: Callable) -> Callable:
    """Wrap a CopilotKitMiddleware ``handle_request`` method with tracing.

    Creates:
    1. An ``"agui:stream"`` span encompassing the full SSE stream.
    2. ``"agui:event:{event_type}"`` child spans for each emitted event.

    The wrapper preserves the original streaming response semantics — it
    replaces the response body iterator with a tracing-aware version that
    yields events unchanged while recording spans.
    """

    @functools.wraps(original_handler)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")

        with tracer.start_as_current_span("agui:stream") as stream_span:
            stream_start = time.monotonic()
            event_index = 0

            try:
                response = await original_handler(*args, **kwargs)
            except Exception as exc:
                record_exception(stream_span, exc)
                raise

            # If the response has a streaming body, wrap it to trace events.
            original_body = getattr(response, "body_iterator", None)
            if original_body is not None:
                response.body_iterator = _traced_event_iterator(
                    original_body,
                    tracer,
                    stream_span,
                    stream_start,
                    event_index,
                )
            else:
                # Non-streaming response — just finalise the stream span.
                duration_ms = (time.monotonic() - stream_start) * 1000
                stream_span.set_attribute("agui.total_events", 0)
                stream_span.set_attribute("agui.stream_duration_ms", duration_ms)
                stream_span.set_status(StatusCode.OK)

            return response

    return wrapper


async def _traced_event_iterator(
    original_iterator: AsyncIterator[bytes | str],
    tracer: Any,
    stream_span: Any,
    stream_start: float,
    event_index: int,
) -> AsyncIterator[bytes | str]:
    """Yield events from *original_iterator* while creating per-event spans.

    Each SSE chunk is inspected for an ``event:`` field to determine the
    AG-UI event type. If the chunk cannot be parsed, it is still yielded
    unchanged — tracing never blocks delivery.
    """
    try:
        async for chunk in original_iterator:
            event_type = _extract_event_type(chunk)
            chunk_size = len(chunk) if isinstance(chunk, (bytes, str)) else 0

            if event_type:
                with tracer.start_as_current_span(f"agui:event:{event_type}") as evt_span:
                    evt_span.set_attribute("agui.event_type", event_type)
                    evt_span.set_attribute("agui.event_index", event_index)
                    evt_span.set_attribute("agui.event_size", chunk_size)
                    evt_span.set_status(StatusCode.OK)
            else:
                # Non-event chunk (e.g. keep-alive comment) — still count it.
                pass

            event_index += 1
            yield chunk

        # Stream completed successfully.
        duration_ms = (time.monotonic() - stream_start) * 1000
        stream_span.set_attribute("agui.total_events", event_index)
        stream_span.set_attribute("agui.stream_duration_ms", duration_ms)
        stream_span.set_status(StatusCode.OK)

    except Exception as exc:
        duration_ms = (time.monotonic() - stream_start) * 1000
        stream_span.set_attribute("agui.total_events", event_index)
        stream_span.set_attribute("agui.stream_duration_ms", duration_ms)
        record_exception(stream_span, exc)
        raise


def _extract_event_type(chunk: bytes | str) -> str | None:
    """Best-effort extraction of the AG-UI event type from an SSE chunk.

    SSE chunks typically look like::

        event: text_message_content
        data: {"type":"text_message_content", ...}

    or the data payload alone may carry a ``type`` field.  We try the
    ``event:`` line first, then fall back to parsing the ``data:`` JSON.
    Returns ``None`` when the event type cannot be determined.
    """
    text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk

    # Try the SSE "event:" field first.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("event:"):
            return stripped[len("event:"):].strip()

    # Fall back to the "data:" JSON payload.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("data:"):
            data_str = stripped[len("data:"):].strip()
            try:
                data = json.loads(data_str)
                if isinstance(data, dict) and "type" in data:
                    return str(data["type"])
            except (json.JSONDecodeError, TypeError):
                pass

    return None
