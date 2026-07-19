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


def vfs_file_text(data) -> str | None:
    """Extract text from a deepagents VFS ``files`` entry.

    ``FileData`` carries ``content: str`` (legacy data may carry a list of
    lines); tolerate a bare string too.
    """
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        content = data.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(line) for line in content)
    return None


def merge_vfs_files(
    content: dict, vfs: dict | None, expected: tuple[str, ...]
) -> dict:
    """Fill missing expected files (matched by basename) from a mini-agent's VFS.

    The generate/fix mini-agents sometimes WRITE their output via the
    filesystem tools instead of (or in addition to) printing fenced code
    blocks — observed in production as "Generated 0 workflow file(s)" while
    the agent claimed it had created the files. Parsed fenced blocks win;
    VFS files only fill the gaps.
    """
    for path, data in (vfs or {}).items():
        name = str(path).rsplit("/", 1)[-1]
        if name not in expected or name in content:
            continue
        text = vfs_file_text(data)
        if isinstance(text, str) and text.strip():
            content[name] = text.strip()
    return content
