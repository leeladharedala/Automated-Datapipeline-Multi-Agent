"""Orchestrator custom state and middleware for pipeline-level tracking.

Extends the DeepAgent's AgentState with typed fields for pipeline phase,
sub-agent dispatch statuses, accumulated results, and metadata. The
OrchestratorMiddleware injects pipeline context before model calls and
updates dispatch/result tracking after model responses.

Concurrent sub-agent updates are handled via the merge_dicts reducer
to avoid INVALID_CONCURRENT_GRAPH_UPDATE errors.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from deepagents.middleware._utils import append_to_system_message

# The deepagents ``AsyncSubAgentMiddleware`` tracks background sub-agent runs in
# the ``async_tasks`` state channel (keyed by ``task_id``, each entry carrying
# ``agent_name``/``thread_id``/``run_id``/``status`` — see AsyncTask). This
# middleware reconciles that channel into the Orchestrator_State dashboard view
# (``dispatch_statuses`` / ``accumulated_results``, keyed by ``agent_name``) so
# the reconciled view persists in Supervisor memory across turns/compaction.
# The reconciliation helpers are shared with the AgentCore ingress
# (src/agentcore/server.py) via src/graphs/_reconcile.py.
from src.graphs._reconcile import (
    LAUNCH_TOOL_NAME,
    extract_result_summaries,
    launch_agent_name,
    reconcile,
)

logger = logging.getLogger(__name__)


def merge_dicts(left: dict, right: dict) -> dict:
    """Reducer that merges two dicts for concurrent sub-agent updates.

    Later values overwrite earlier ones for the same key.
    """
    return {**left, **right}


class OrchestratorState(AgentState):
    """Custom orchestrator state extending AgentState with pipeline tracking.

    Fields:
        pipeline_phase: Current execution phase
            ("planning", "executing", "reviewing", "complete").
        dispatch_statuses: Per-agent status tracking
            (e.g. {"iac-agent": "running", "cicd-agent": "success"}).
            Uses merge_dicts reducer for safe concurrent updates.
        accumulated_results: Per-agent result summaries
            (e.g. {"iac-agent": "VALIDATION: PASSED ..."}).
            Uses merge_dicts reducer for safe concurrent updates.
        pipeline_metadata: User-provided context and parsed pipeline document
            (e.g. {"environment": "prod", "pipeline_document": {...}}).
    """

    pipeline_phase: NotRequired[str]
    dispatch_statuses: NotRequired[Annotated[dict[str, str], merge_dicts]]
    accumulated_results: NotRequired[Annotated[dict[str, str], merge_dicts]]
    pipeline_metadata: NotRequired[dict[str, Any]]


_PIPELINE_CONTEXT_TEMPLATE = """\
<pipeline_context>
Phase: {phase}
Dispatch statuses: {statuses}
Accumulated results: {results}
Metadata: {metadata}
{document_section}</pipeline_context>
"""


def _format_pipeline_document_summary(doc: dict[str, Any]) -> str:
    """Format a parsed pipeline document into a readable context summary."""
    lines: list[str] = ["<pipeline_document>"]

    # Data source section
    data_source = doc.get("data_source", {})
    if data_source:
        lines.append("Data Source:")
        if "uri" in data_source:
            lines.append(f"  URI: {data_source['uri']}")
        if "format" in data_source:
            lines.append(f"  Format: {data_source['format']}")

    # Transformations section
    transformations = doc.get("transformations", [])
    if transformations:
        lines.append("Transformations:")
        for t in transformations:
            name = t.get("name", "unnamed")
            desc = t.get("description", "")
            lines.append(f"  - {name}: {desc}")

    # Architecture section
    architecture = doc.get("architecture", {})
    if architecture:
        lines.append(f"Architecture: {json.dumps(architecture)}")

    lines.append("</pipeline_document>")
    return "\n".join(lines)


class OrchestratorMiddleware(AgentMiddleware[OrchestratorState, Any, Any]):
    """Middleware that injects pipeline state into model context and tracks dispatch.

    - before_model / wrap_model_call: Injects current pipeline phase, dispatch
      statuses, and accumulated results into the system prompt so the LLM is
      aware of pipeline progress.
    - after_model: Parses tool-call responses to detect sub-agent dispatch and
      completion events, updating dispatch_statuses and accumulated_results.
    """

    state_schema = OrchestratorState

    # --- before_model: inject pipeline context ---

    def _build_context_snippet(self, state: OrchestratorState) -> str:
        """Format pipeline state fields into a context string."""
        phase = state.get("pipeline_phase", "planning")
        statuses = state.get("dispatch_statuses", {})
        results = state.get("accumulated_results", {})
        metadata = state.get("pipeline_metadata", {})

        # Build pipeline document summary if present
        document_section = ""
        if isinstance(metadata, dict) and "pipeline_document" in metadata:
            document_section = _format_pipeline_document_summary(
                metadata["pipeline_document"]
            )

        return _PIPELINE_CONTEXT_TEMPLATE.format(
            phase=phase,
            statuses=json.dumps(statuses) if statuses else "none",
            results=json.dumps(results) if results else "none",
            metadata=json.dumps(metadata) if metadata else "none",
            document_section=document_section,
        )

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Inject pipeline context into the system message."""
        snippet = self._build_context_snippet(request.state)
        new_system = append_to_system_message(request.system_message, snippet)
        return request.override(system_message=new_system)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject pipeline context before model call (sync)."""
        modified = self.modify_request(request)
        return handler(modified)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject pipeline context before model call (async)."""
        modified = self.modify_request(request)
        return await handler(modified)

    # --- after_model: reconcile async task state from tool calls/results ---

    def _extract_dispatch_updates(
        self, state: OrchestratorState
    ) -> dict[str, Any] | None:
        """Reconcile async sub-agent task state into Orchestrator_State.

        Recognizes the deepagents async task tool surface:

        - ``start_async_task`` tool CALL (in the AI message the model just
          produced) -> the Sub_Agent is launched, so its ``agent_name`` (from
          ``subagent_type``) is marked ``running`` immediately. This fires before
          the launch tool writes the ``async_tasks`` channel, so freshly launched
          agents surface without waiting a turn.
        - ``check_async_task`` / ``list_async_tasks`` / ``cancel_async_task`` /
          ``update_async_task`` results land in the ``async_tasks`` channel (a
          per-``task_id`` status). Reconciling that channel derives terminal
          ``success``/``failed``/``cancelled`` statuses via the same derivation
          table as the server. On success, a truncated result summary (recovered
          from the ``check_async_task`` JSON ToolMessage, matched by
          ``task_id``) is recorded under the resolved ``agent_name``.

        Writing these into ``dispatch_statuses`` / ``accumulated_results`` (both
        merge_dicts-reduced) persists the reconciled view in Supervisor memory
        across turns and context compaction.
        """
        messages = state.get("messages", [])
        async_tasks = state.get("async_tasks") or {}

        # Reconcile the persisted async_tasks channel (authoritative statuses
        # written by the check/cancel/list/update tools) into per-agent state.
        summaries = extract_result_summaries(messages)
        dispatch_statuses, accumulated_results = reconcile(async_tasks, summaries)

        # Mark freshly launched agents running from start_async_task calls in the
        # AI message the model just produced. The channel is authoritative, so a
        # launch only fills a gap for an agent not already resolved from it.
        if messages:
            last_msg = messages[-1]
            tool_calls = getattr(last_msg, "tool_calls", None) or []
            for tc in tool_calls:
                if tc.get("name") != LAUNCH_TOOL_NAME:
                    continue
                agent_name = launch_agent_name(tc.get("args"))
                if agent_name and agent_name not in dispatch_statuses:
                    dispatch_statuses[agent_name] = "running"
                    logger.debug("Async launch detected: %s -> running", agent_name)

        if not dispatch_statuses and not accumulated_results:
            return None

        updates: dict[str, Any] = {}
        if dispatch_statuses:
            updates["dispatch_statuses"] = dispatch_statuses
        if accumulated_results:
            updates["accumulated_results"] = accumulated_results
        return updates

    def after_model(
        self, state: OrchestratorState, runtime: Any
    ) -> dict[str, Any] | None:
        """Update dispatch statuses after model call (sync)."""
        return self._extract_dispatch_updates(state)

    async def aafter_model(
        self, state: OrchestratorState, runtime: Any
    ) -> dict[str, Any] | None:
        """Update dispatch statuses after model call (async)."""
        return self._extract_dispatch_updates(state)
