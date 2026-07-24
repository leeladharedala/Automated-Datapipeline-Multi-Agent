"""
Multi-Agent Data Pipeline — Entry Point

Creates the root orchestrator agent using DeepAgents, backed by
AgentCore Memory for short-term checkpoint persistence.

Sub-agents are launched as non-blocking background runs on a co-located
LangGraph API server via deepagents' AsyncSubAgent (preview API, 0.6.x).
The three sub-agent graphs live only in the co-located server process
(see src/graphs/server_graphs.py + langgraph.json); this process no longer
instantiates them.
"""

import logging
import os

from deepagents import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgent  # preview API (0.6.x)

from src.anthropic_model import SanitizedChatAnthropic
from src.checkpointer import FastAgentCoreMemorySaver
from src.prompts.orchestrator_prompt import ORCHESTRATOR_PROMPT
from src.graphs import OrchestratorMiddleware
from src.tools.submit_pr import submit_pr
from src.document_parser import parse_document_tool

REGION = os.environ.get("AWS_REGION", "us-west-2")
MEMORY_ID = os.environ.get("AGENTCORE_MEMORY_ID", "")

# Co-located LangGraph API server on localhost inside the same container.
SUBAGENT_SERVER_URL = os.environ.get("SUBAGENT_SERVER_URL", "http://127.0.0.1:2024")

# The middleware injects `x-auth-scheme: langsmith` by default. Our co-located
# server is self-hosted, so override the header per spec to a scheme the server
# ignores/accepts (see Requirement 14).
_LOCAL_HEADERS = {"x-auth-scheme": "noop"}

logger = logging.getLogger(__name__)


async def build_agent():
    """Build and return the compiled orchestrator agent."""
    if not MEMORY_ID:
        raise RuntimeError("AGENTCORE_MEMORY_ID environment variable is required")

    # Thinking is explicitly DISABLED for latency: adaptive thinking generates
    # reasoning tokens before any visible output, which dominated response
    # start-time. SanitizedChatAnthropic still strips any empty thinking blocks
    # from replayed history as a safety net — see src/anthropic_model.py.
    # (Re-enable by removing the `thinking` kwarg if generation quality drops;
    # the model default is adaptive thinking ON.)
    # max_tokens MUST be explicit (the profiles package is absent, so the
    # library falls back to 4096 — far too small for the Supervisor's
    # submit_pr call, whose args inline every generated file). Set to the
    # claude-sonnet-5 output ceiling (128000) — the API requires the field, so
    # this is "uncapped" in practice. Note: it's an upper bound, not a target;
    # raising it does not change latency for responses that finish sooner.
    model = SanitizedChatAnthropic(
        model="claude-sonnet-5",
        max_tokens=128000,
        thinking={"type": "disabled"},
    )

    # Async sub-agent specs. A spec containing `graph_id` triggers deepagents to
    # auto-append AsyncSubAgentMiddleware, so we do NOT instantiate it ourselves.
    # `graph_id` must match the keys registered in langgraph.json for the
    # co-located server; `url` targets that localhost server.
    async_subagents: list[AsyncSubAgent] = [
        AsyncSubAgent(
            name="iac-agent",
            description="Generates Terraform infrastructure code for AWS resources.",
            graph_id="iac",  # must match langgraph.json key
            url=SUBAGENT_SERVER_URL,
            headers=_LOCAL_HEADERS,
        ),
        AsyncSubAgent(
            name="cicd-agent",
            description="Generates GitHub Actions CI/CD workflows.",
            graph_id="cicd",
            url=SUBAGENT_SERVER_URL,
            headers=_LOCAL_HEADERS,
        ),
        AsyncSubAgent(
            name="data-eng-agent",
            description="Generates data transformation code with tests.",
            graph_id="data-eng",
            url=SUBAGENT_SERVER_URL,
            headers=_LOCAL_HEADERS,
        ),
    ]

    # Short-term memory: use the remote saver directly for the orchestrator so
    # every turn is persisted immediately.  Do NOT throttle checkpoint writes
    # here — a throttled saver previously skipped the final checkpoint of a
    # pipeline run, leaving the next turn with no conversation history and
    # causing the LLM to re-dispatch all subagents from scratch.
    # FastAgentCoreMemorySaver keeps DeltaChannel (messages) history loads at
    # one event fetch instead of one per ancestor checkpoint — without it every
    # state load takes minutes once a thread has real history.
    # Subagent persistence is owned by the co-located server, not this process.
    checkpointer = FastAgentCoreMemorySaver(memory_id=MEMORY_ID, region_name=REGION)

    # Pass tools directly — skip tracing wrappers to avoid
    # Pydantic v2 compatibility issues during agent compilation.
    # `graph_id` presence in the specs auto-adds AsyncSubAgentMiddleware.
    agent = create_deep_agent(
        model=model,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=async_subagents,
        tools=[submit_pr, parse_document_tool],
        checkpointer=checkpointer,
        middleware=[OrchestratorMiddleware()],
    )

    return agent
