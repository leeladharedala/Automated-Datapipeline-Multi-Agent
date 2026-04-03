"""Graph node and edge tracing wrappers.

Provides wrapper functions that instrument LangGraph ``StateGraph`` node
execution and edge routing with OpenTelemetry spans.  The main entry point
is :func:`instrument_graph`, which accepts a ``StateGraph`` builder and
wraps all registered nodes and edges *before* compilation.
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
# Node wrapper
# ---------------------------------------------------------------------------

def traced_node(
    graph_name: str,
    node_name: str,
    node_type: str,
    node_fn: Callable,
) -> Callable:
    """Wrap *node_fn* to create a ``"node:{graph_name}.{node_name}"`` span.

    Attributes set on the span:
        ``graph.name``, ``node.name``, ``node.type``

    On completion the span records ``node.output_keys`` (the dict keys
    returned by the node function).  On error the exception is recorded
    and the span status is set to ERROR.
    """
    span_name = f"node:{graph_name}.{node_name}"

    @functools.wraps(node_fn)
    def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("graph.name", graph_name)
            span.set_attribute("node.name", node_name)
            span.set_attribute("node.type", node_type)
            try:
                result = node_fn(state, *args, **kwargs)
                if isinstance(result, dict):
                    span.set_attribute("node.output_keys", list(result.keys()))
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                record_exception(span, exc)
                raise

    return wrapper


# ---------------------------------------------------------------------------
# Edge wrappers
# ---------------------------------------------------------------------------

def traced_edge(
    graph_name: str,
    edge_fn: Callable,
    source_node: str,
) -> Callable:
    """Wrap a conditional edge function to create an ``"edge:{graph_name}.{fn}"`` span.

    Attributes:
        ``edge.source_node``, ``edge.graph_name``

    On completion:
        ``edge.decision`` (the routing value returned),
        ``edge.target_node`` (same as decision),
        ``edge.state_snapshot`` (JSON of state keys that influence routing)
    """
    fn_name = getattr(edge_fn, "__name__", "unknown")
    span_name = f"edge:{graph_name}.{fn_name}"

    @functools.wraps(edge_fn)
    def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("edge.source_node", source_node)
            span.set_attribute("edge.graph_name", graph_name)
            try:
                decision = edge_fn(state, *args, **kwargs)
                span.set_attribute("edge.decision", str(decision))
                span.set_attribute("edge.target_node", str(decision))
                span.set_attribute(
                    "edge.state_snapshot",
                    _state_snapshot(state),
                )
                span.set_status(StatusCode.OK)
                return decision
            except Exception as exc:
                record_exception(span, exc)
                raise

    return wrapper


def traced_static_edge(
    graph_name: str,
    source_node: str,
    target_node: str,
) -> Callable:
    """Return a no-op function that creates an ``"edge:{graph_name}.{src}->{tgt}"`` span.

    The span has ``edge.type`` set to ``"static"``.

    This is intended to be inserted as a node wrapper that fires a span
    *after* the source node completes, since LangGraph static edges don't
    have a user-callable function.
    """
    span_name = f"edge:{graph_name}.{source_node}->{target_node}"

    def emit_span() -> None:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("edge.type", "static")
            span.set_attribute("edge.source_node", source_node)
            span.set_attribute("edge.target_node", target_node)
            span.set_attribute("edge.graph_name", graph_name)
            span.set_status(StatusCode.OK)

    return emit_span


# ---------------------------------------------------------------------------
# instrument_graph — main entry point
# ---------------------------------------------------------------------------

def instrument_graph(
    graph_builder: Any,
    graph_name: str,
    node_types: dict[str, str],
) -> Any:
    """Instrument all nodes and edges in a ``StateGraph`` builder with tracing.

    This must be called **before** ``graph_builder.compile()``.

    Args:
        graph_builder: A ``langgraph.graph.StateGraph`` instance.
        graph_name: Human-readable graph name (e.g. ``"iac"``, ``"cicd"``).
        node_types: Mapping of node name → type string
            (``"agent"``, ``"function"``, or ``"conditional"``).

    Returns:
        The same *graph_builder* (mutated in place) for chaining.
    """
    # --- Wrap registered nodes ---
    # StateGraph stores nodes in graph_builder.nodes (dict[str, NodeSpec] or
    # dict[str, Callable] depending on LangGraph version).  We replace each
    # callable with a traced version.
    for name in list(graph_builder.nodes):
        node_spec = graph_builder.nodes[name]
        node_type = node_types.get(name, "function")

        # LangGraph may store the callable directly or inside a wrapper
        # object.  Handle both.
        if callable(node_spec):
            graph_builder.nodes[name] = traced_node(
                graph_name, name, node_type, node_spec,
            )
        elif hasattr(node_spec, "runnable"):
            # NodeSpec-style: wrap the inner runnable
            original = node_spec.runnable
            if callable(original):
                node_spec.runnable = traced_node(
                    graph_name, name, node_type, original,
                )

    # --- Wrap conditional edges ---
    # StateGraph stores conditional edges in graph_builder.branches
    # (dict[str, dict[str, Branch]]).  Each Branch has a .path callable.
    if hasattr(graph_builder, "branches"):
        for source_node, branch_map in graph_builder.branches.items():
            for key, branch in branch_map.items():
                if hasattr(branch, "path") and callable(branch.path):
                    branch.path = traced_edge(
                        graph_name, branch.path, source_node,
                    )

    # --- Emit entry-point edge span ---
    # Create a START->{entry_node} span by wrapping the entry node so that
    # a span fires before its first execution.
    entry_point = getattr(graph_builder, "entry_point", None)
    if entry_point and entry_point in graph_builder.nodes:
        _wrap_entry_span(graph_builder, graph_name, entry_point)

    return graph_builder


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ROUTING_STATE_KEYS = frozenset({
    "validation_passed", "attempt", "max_attempts",
})


def _state_snapshot(state: Any) -> str:
    """Return a JSON string of routing-relevant state keys."""
    if isinstance(state, dict):
        snapshot = {
            k: v for k, v in state.items()
            if k in _ROUTING_STATE_KEYS
        }
    else:
        snapshot = {}
    try:
        return json.dumps(snapshot, default=str)
    except Exception:
        return "{}"


def _wrap_entry_span(
    graph_builder: Any,
    graph_name: str,
    entry_node: str,
) -> None:
    """Wrap the entry node so that a ``START->{entry_node}`` edge span fires first."""
    span_name = f"edge:{graph_name}.START->{entry_node}"
    node_fn = graph_builder.nodes[entry_node]

    if not callable(node_fn):
        return

    @functools.wraps(node_fn)
    def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("edge.type", "static")
            span.set_attribute("edge.source_node", "START")
            span.set_attribute("edge.target_node", entry_node)
            span.set_attribute("edge.graph_name", graph_name)
            span.set_status(StatusCode.OK)
        return node_fn(state, *args, **kwargs)

    graph_builder.nodes[entry_node] = wrapper
