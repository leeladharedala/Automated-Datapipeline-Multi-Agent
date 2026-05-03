"""
Multi-Agent Data Pipeline — Entry Point

Creates the root orchestrator agent using DeepAgents, backed by
AgentCore Memory for short-term checkpoint persistence.
"""

import logging
import os

from deepagents import create_deep_agent, CompiledSubAgent
from langchain_anthropic import ChatAnthropic
from langgraph_checkpoint_aws import AgentCoreMemorySaver

from src.prompts.orchestrator_prompt import ORCHESTRATOR_PROMPT
from src.graphs import (
    build_iac_graph,
    build_cicd_graph,
    build_data_eng_graph,
    OrchestratorMiddleware,
    OrchestratorState,
)
from src.tools.gateway import load_gateway_tools
from src.tools.submit_pr import submit_pr
from src.document_parser import parse_document_tool

REGION = os.environ.get("AWS_REGION", "us-west-2")
MEMORY_ID = os.environ.get("AGENTCORE_MEMORY_ID", "")

logger = logging.getLogger(__name__)


async def build_agent():
    """Build and return the compiled orchestrator agent."""
    if not MEMORY_ID:
        raise RuntimeError("AGENTCORE_MEMORY_ID environment variable is required")

    model = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
    )

    # MCP tools (Terraform Registry + AWS Docs) are loaded lazily on first
    # IaC research node invocation to avoid AgentCore's 120s init timeout.
    # The npx/uvx downloads happen at research time, not at startup.
    gateway_tools = []

    # Build compiled sub-agent graphs
    iac_graph = build_iac_graph(model=model, tools=gateway_tools)
    cicd_graph = build_cicd_graph(model=model)
    data_eng_graph = build_data_eng_graph(model=model, tools=[])

    # Wrap as CompiledSubAgent objects (not plain dicts)
    subagents = [
        CompiledSubAgent(
            name="iac-agent",
            description="Generates Terraform infrastructure code for AWS resources.",
            runnable=iac_graph,
        ),
        CompiledSubAgent(
            name="cicd-agent",
            description="Generates GitHub Actions CI/CD workflows.",
            runnable=cicd_graph,
        ),
        CompiledSubAgent(
            name="data-eng-agent",
            description="Generates data transformation code with tests.",
            runnable=data_eng_graph,
        ),
    ]

    # Short-term memory: use the remote saver directly for the orchestrator so
    # every turn is persisted immediately.  The ThrottledCheckpointSaver is
    # intentionally NOT used here — throttling caused the final checkpoint of
    # a pipeline run to be skipped, leaving the next turn with no conversation
    # history and causing the LLM to re-dispatch all subagents from scratch.
    # Subagents run to completion in a single shot and don't need a checkpointer.
    checkpointer = AgentCoreMemorySaver(memory_id=MEMORY_ID, region_name=REGION)

    # Pass tools directly — skip tracing wrappers to avoid
    # Pydantic v2 compatibility issues during agent compilation
    agent = create_deep_agent(
        model=model,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=subagents,
        tools=[submit_pr, parse_document_tool],
        checkpointer=checkpointer,
        middleware=[OrchestratorMiddleware()],
        state_schema=OrchestratorState,
    )

    return agent
