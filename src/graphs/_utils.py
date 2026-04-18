"""Shared utilities for subagent graph nodes."""

from __future__ import annotations

from langchain_core.messages import HumanMessage


def get_task_description(state: dict) -> str:
    """Extract the task description from state, falling back to messages.

    CompiledSubAgent passes the task content via the messages list (as a
    HumanMessage), but graph nodes read from state["task_description"] which
    defaults to an empty string. This helper checks task_description first,
    then falls back to extracting the content from the first HumanMessage.
    """
    task = state.get("task_description", "")
    if task:
        return task

    # Fallback: extract from the first HumanMessage in messages
    messages = state.get("messages", [])
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            return _content_to_str(msg.content)

    return ""


def _content_to_str(content) -> str:
    """Coerce message content to a plain string.

    In newer LangChain versions, ``content`` can be a list of content
    blocks (dicts with ``type``/``text`` keys) instead of a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
        )
    return str(content)
