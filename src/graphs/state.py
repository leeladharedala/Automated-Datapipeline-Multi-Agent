"""Base and per-graph state schemas for LangGraph sub-agents."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class BaseSubAgentState(TypedDict, total=False):
    """Common state fields shared across all sub-agent graphs.

    Includes `messages` with the add_messages reducer, required by the
    CompiledSubAgent contract so the orchestrator can pass task descriptions
    in and receive final reports out via the messages list.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    task_description: str
    validation_output: str
    validation_passed: bool
    attempt: int
    max_attempts: int
    report: str


class IaCState(BaseSubAgentState, total=False):
    """IaC sub-agent state with Terraform-specific fields."""

    research_context: str
    tf_artifacts: dict[str, str]


class CICDState(BaseSubAgentState, total=False):
    """CI/CD sub-agent state with GitHub Actions workflow fields."""

    workflow_artifacts: dict[str, str]


class DataEngState(BaseSubAgentState, total=False):
    """Data Engineering sub-agent state with transformation code fields."""

    code_artifacts: dict[str, str]
    code_content: dict[str, str]   # filename -> actual file content (passed to validator)
    inferred_schema: dict[str, Any]
    data_sample_status: str


# Default values for state initialization
BASE_STATE_DEFAULTS: BaseSubAgentState = {
    "messages": [],
    "task_description": "",
    "validation_output": "",
    "validation_passed": False,
    "attempt": 0,
    "max_attempts": 3,
    "report": "",
}

IAC_STATE_DEFAULTS: IaCState = {
    **BASE_STATE_DEFAULTS,
    "research_context": "",
    "tf_artifacts": {},
}

CICD_STATE_DEFAULTS: CICDState = {
    **BASE_STATE_DEFAULTS,
    "workflow_artifacts": {},
}

DATA_ENG_STATE_DEFAULTS: DataEngState = {
    **BASE_STATE_DEFAULTS,
    "code_artifacts": {},
    "inferred_schema": {},
    "data_sample_status": "skipped",
}
