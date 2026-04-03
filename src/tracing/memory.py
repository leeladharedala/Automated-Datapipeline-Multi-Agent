"""Memory hook tracing wrappers.

Wraps ``pre_model_hook`` / ``post_model_hook`` and individual
``store.search`` / ``store.put`` operations with OpenTelemetry spans so
that every memory interaction is visible in the trace waterfall.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable

from opentelemetry.trace import StatusCode

from src.tracing.provider import get_tracer
from src.tracing.utils import record_exception

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook wrappers
# ---------------------------------------------------------------------------

def traced_pre_model_hook(original_fn: Callable) -> Callable:
    """Wrap *original_fn* (``pre_model_hook``) in a ``"memory:pre_model_hook"`` span.

    Attributes set:
        ``memory.actor_id``, ``memory.thread_id``,
        ``memory.namespaces_searched``, ``memory.memories_retrieved``
    """

    @functools.wraps(original_fn)
    def wrapper(state: dict, config: Any, *, store: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span("memory:pre_model_hook") as span:
            actor_id = config.get("configurable", {}).get("actor_id", "")
            thread_id = config.get("configurable", {}).get("thread_id", "")
            span.set_attribute("memory.actor_id", actor_id)
            span.set_attribute("memory.thread_id", thread_id)

            # Wrap the store temporarily to count search/put calls
            counter = _SearchCounter()
            wrapped_store = _counting_store(store, counter)

            try:
                result = original_fn(state, config, store=wrapped_store)
                span.set_attribute("memory.namespaces_searched", counter.search_count)
                span.set_attribute("memory.memories_retrieved", counter.results_total)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_attribute("memory.namespaces_searched", counter.search_count)
                span.set_attribute("memory.memories_retrieved", counter.results_total)
                record_exception(span, exc)
                raise

    return wrapper


def traced_post_model_hook(original_fn: Callable) -> Callable:
    """Wrap *original_fn* (``post_model_hook``) in a ``"memory:post_model_hook"`` span.

    Attributes set:
        ``memory.actor_id``, ``memory.message_saved``
    """

    @functools.wraps(original_fn)
    def wrapper(state: dict, config: Any, *, store: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span("memory:post_model_hook") as span:
            actor_id = config.get("configurable", {}).get("actor_id", "")
            span.set_attribute("memory.actor_id", actor_id)

            # Track whether a put (message save) occurs
            counter = _SearchCounter()
            wrapped_store = _counting_store(store, counter)

            try:
                result = original_fn(state, config, store=wrapped_store)
                span.set_attribute("memory.message_saved", counter.put_count > 0)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_attribute("memory.message_saved", counter.put_count > 0)
                record_exception(span, exc)
                raise

    return wrapper


# ---------------------------------------------------------------------------
# Store operation wrappers
# ---------------------------------------------------------------------------

def traced_store_search(store: Any) -> Any:
    """Return *store* with ``search`` wrapped in ``"memory:store.search"`` child spans.

    Attributes set:
        ``memory.namespace``, ``memory.query`` (truncated 128 chars),
        ``memory.limit``, ``memory.results_count``
    """
    original_search = store.search

    @functools.wraps(original_search)
    def wrapped_search(namespace: Any, *, query: str = "", limit: int = 10, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span("memory:store.search") as span:
            span.set_attribute("memory.namespace", _format_namespace(namespace))
            span.set_attribute("memory.query", query[:128])
            span.set_attribute("memory.limit", limit)
            try:
                results = original_search(namespace, query=query, limit=limit, **kwargs)
                span.set_attribute("memory.results_count", len(results) if results else 0)
                span.set_status(StatusCode.OK)
                return results
            except Exception as exc:
                record_exception(span, exc)
                raise

    store.search = wrapped_search  # type: ignore[assignment]
    return store


def traced_store_put(store: Any) -> Any:
    """Return *store* with ``put`` wrapped in ``"memory:store.put"`` child spans.

    Attributes set:
        ``memory.namespace``, ``memory.key``, ``memory.value_size``
    """
    original_put = store.put

    @functools.wraps(original_put)
    def wrapped_put(namespace: Any, key: str, value: Any, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span("memory:store.put") as span:
            span.set_attribute("memory.namespace", _format_namespace(namespace))
            span.set_attribute("memory.key", key)
            span.set_attribute("memory.value_size", _value_size(value))
            try:
                result = original_put(namespace, key, value, **kwargs)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                record_exception(span, exc)
                raise

    store.put = wrapped_put  # type: ignore[assignment]
    return store


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _SearchCounter:
    """Lightweight counter used by hook wrappers to tally store operations."""

    __slots__ = ("search_count", "results_total", "put_count")

    def __init__(self) -> None:
        self.search_count = 0
        self.results_total = 0
        self.put_count = 0


def _counting_store(store: Any, counter: _SearchCounter) -> Any:
    """Return a thin proxy around *store* that increments *counter* on each call.

    This is used inside the hook wrappers so that the hook-level span can
    report aggregate counts (namespaces searched, memories retrieved) without
    duplicating the per-operation tracing that ``traced_store_search`` /
    ``traced_store_put`` already provide.
    """

    class _Proxy:
        """Delegates everything to the real store, counting search/put calls."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def search(self, namespace: Any, *, query: str = "", limit: int = 10, **kw: Any) -> Any:
            counter.search_count += 1
            results = self._inner.search(namespace, query=query, limit=limit, **kw)
            counter.results_total += len(results) if results else 0
            return results

        def put(self, namespace: Any, key: str, value: Any, **kw: Any) -> Any:
            counter.put_count += 1
            return self._inner.put(namespace, key, value, **kw)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    return _Proxy(store)


def _format_namespace(namespace: Any) -> str:
    """Convert a namespace tuple to a readable string."""
    if isinstance(namespace, (list, tuple)):
        return "/".join(str(part) for part in namespace)
    return str(namespace)


def _value_size(value: Any) -> int:
    """Best-effort byte length of a serialised value."""
    try:
        return len(json.dumps(value, default=str).encode())
    except Exception:
        return len(str(value).encode())
