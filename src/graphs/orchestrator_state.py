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
    PrivateStateAttr,
)
from langchain_core.messages import SystemMessage
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from deepagents.middleware._utils import append_to_system_message

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

    # --- after_model: update dispatch statuses from tool calls ---

    def _extract_dispatch_updates(
        self, state: OrchestratorState
    ) -> dict[str, Any] | None:
        """Parse the latest AI message for sub-agent dispatch/completion signals.

        Looks for tool calls named 'task' (DeepAgent's sub-agent dispatch tool)
        and updates dispatch_statuses accordingly. When a sub-agent result appears
        in messages, updates accumulated_results.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        status_updates: dict[str, str] = {}
        result_updates: dict[str, str] = {}

        # Check for tool calls dispatching to sub-agents
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                if tc.get("name") == "task":
                    agent_name = tc.get("args", {}).get("agent_name", "")
                    if agent_name:
                        status_updates[agent_name] = "running"
                        logger.debug("Dispatch detected: %s -> running", agent_name)

        # Check for tool messages returning sub-agent results
        if hasattr(last_msg, "type") and last_msg.type == "tool":
            name = getattr(last_msg, "name", "")
            if name == "task":
                content = getattr(last_msg, "content", "")
                # Try to extract agent name from the tool call context
                tool_call_id = getattr(last_msg, "tool_call_id", "")
                # Look back for the matching AI message with the tool call
                for msg in reversed(messages[:-1]):
                    if hasattr(msg, "tool_calls"):
                        for tc in msg.tool_calls:
                            if tc.get("id") == tool_call_id:
                                agent_name = tc.get("args", {}).get(
                                    "agent_name", ""
                                )
                                if agent_name:
                                    passed = "PASSED" in content.upper()
                                    status_updates[agent_name] = (
                                        "success" if passed else "failed"
                                    )
                                    # Truncate result for summary
                                    summary = (
                                        content[:500] + "..."
                                        if len(content) > 500
                                        else content
                                    )
                                    result_updates[agent_name] = summary
                                break
                    break  # Only check the immediately preceding message

        if not status_updates and not result_updates:
            return None

        updates: dict[str, Any] = {}
        if status_updates:
            updates["dispatch_statuses"] = status_updates
        if result_updates:
            updates["accumulated_results"] = result_updates
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
