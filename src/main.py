"""
Multi-Agent Data Pipeline — Entry Point

Creates the root orchestrator agent using DeepAgents, backed by
AgentCore Memory for both short-term (checkpoint) and long-term
(preferences/facts) persistence. Wires compiled LangGraph sub-agent
graphs (IaC, CI/CD, DataEng) as CompiledSubAgents with
OrchestratorMiddleware for pipeline-level state tracking.
"""

import logging
import os
from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore

from src.prompts.orchestrator_prompt import ORCHESTRATOR_PROMPT
from src.graphs import (
    build_iac_graph,
    build_cicd_graph,
    build_data_eng_graph,
    OrchestratorMiddleware,
)
from src.tools.gateway import load_gateway_tools
from src.tools.submit_pr import submit_pr
from src.document_parser import parse_document_tool
from src.memory.hooks import pre_model_hook, post_model_hook
from src.tracing import (
    traced_llm,
    trace_tools,
    traced_pre_model_hook,
    traced_post_model_hook,
    traced_store_search,
    traced_store_put,
    instrument_middleware,
)

# AgentCore Memory configuration
REGION = os.environ.get("AWS_REGION", "us-west-2")
MEMORY_ID = os.environ.get("AGENTCORE_MEMORY_ID", "")

logger = logging.getLogger(__name__)


async def build_agent():
    """Build and return the compiled orchestrator agent.

    Memory architecture:
    - AgentCoreMemorySaver (short-term): Persists conversation state,
      VFS artifacts, and graph execution checkpoints per session.
    - AgentCoreMemoryStore (long-term): Extracts preferences and facts
      across sessions via pre/post model hooks.
    - pre_model_hook: Saves user messages, retrieves past preferences.
    - post_model_hook: Saves AI responses for pattern extraction.
    """
    if not MEMORY_ID:
        raise RuntimeError("AGENTCORE_MEMORY_ID environment variable is required")

    model = ChatAnthropic(
        model="claude-sonnet-4-6-20250514",
        temperature=0,
    )
    model = traced_llm(model)

    # Load MCP tools from Terraform Registry + AWS Docs servers
    try:
        gateway_tools = await load_gateway_tools()
    except Exception as exc:
        logger.warning("MCP tool loading failed, continuing without gateway tools: %s", exc)
        gateway_tools = []

    # Build compiled sub-agent graphs
    iac_graph = build_iac_graph(model=model, tools=gateway_tools)
    cicd_graph = build_cicd_graph(model=model)
    data_eng_graph = build_data_eng_graph(model=model, tools=[])  # browser tools injected at runtime

    # Wrap as CompiledSubAgent dicts for create_deep_agent
    subagents = [
        {
            "name": "iac-agent",
            "description": "Generates Terraform infrastructure code for AWS resources.",
            "runnable": iac_graph,
        },
        {
            "name": "cicd-agent",
            "description": "Generates GitHub Actions CI/CD workflows.",
            "runnable": cicd_graph,
        },
        {
            "name": "data-eng-agent",
            "description": "Generates data transformation code with tests.",
            "runnable": data_eng_graph,
        },
    ]

    # Short-term memory: checkpoints for state persistence + VFS
    checkpointer = AgentCoreMemorySaver(MEMORY_ID, region_name=REGION)

    # Long-term memory: preferences and facts across sessions
    store = AgentCoreMemoryStore(MEMORY_ID, region_name=REGION)
    store = traced_store_search(store)
    store = traced_store_put(store)

    # Wrap memory hooks with tracing
    traced_pre_hook = traced_pre_model_hook(pre_model_hook)
    traced_post_hook = traced_post_model_hook(post_model_hook)

    # Instrument middleware with tracing
    TracedOrchestratorMiddleware = instrument_middleware(OrchestratorMiddleware)

    agent = create_deep_agent(
        model=model,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=subagents,
        tools=trace_tools([submit_pr, parse_document_tool]),
        checkpointer=checkpointer,
        store=store,
        pre_model_hook=traced_pre_hook,
        post_model_hook=traced_post_hook,
        middleware=[TracedOrchestratorMiddleware],
    )

    return agent
