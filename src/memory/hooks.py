"""
Pre/post model hooks for AgentCore long-term memory.

Works with three AgentCore Memory strategies configured in setup.py:
1. UserPreference — user's architectural preferences (region, framework, etc.)
2. Semantic — factual knowledge (past pipeline architectures, schemas)
3. Summary — session summaries for recapping long conversations

The pre_model_hook retrieves relevant memories from all three namespaces
and injects them into the conversation context before each LLM call.
The post_model_hook saves both user and AI messages for async extraction.
"""

import uuid
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.store.base import BaseStore
from langgraph.types import RunnableConfig

# Max memories to retrieve per namespace
MAX_RESULTS_PER_NS = 3


def _build_memory_context(
    store: BaseStore,
    actor_id: str,
    query: str,
) -> str | None:
    """Search all three memory namespaces and build a context block."""
    sections = []

    # 1. User preferences
    prefs = store.search(
        ("users", actor_id, "preferences"),
        query=query,
        limit=MAX_RESULTS_PER_NS,
    )
    if prefs:
        lines = []
        for mem in prefs:
            val = mem.value
            text = val.get("message", val) if isinstance(val, dict) else str(val)
            lines.append(f"  - {text}")
        sections.append("User Preferences:\n" + "\n".join(lines))

    # 2. Semantic facts
    facts = store.search(
        ("users", actor_id, "facts"),
        query=query,
        limit=MAX_RESULTS_PER_NS,
    )
    if facts:
        lines = []
        for mem in facts:
            val = mem.value
            text = val.get("message", val) if isinstance(val, dict) else str(val)
            lines.append(f"  - {text}")
        sections.append("Architectural Facts:\n" + "\n".join(lines))

    # 3. Session summaries (from past sessions, not current)
    summaries = store.search(
        ("summaries", actor_id),
        query=query,
        limit=2,  # Only most relevant past summaries
    )
    if summaries:
        lines = []
        for mem in summaries:
            val = mem.value
            text = val.get("message", val) if isinstance(val, dict) else str(val)
            lines.append(f"  - {text}")
        sections.append("Past Session Summaries:\n" + "\n".join(lines))

    if not sections:
        return None

    return (
        "\n[Long-Term Memory — Recalled from previous sessions]\n"
        + "\n\n".join(sections)
        + "\n[End of Long-Term Memory]\n"
    )


def pre_model_hook(state: dict, config: RunnableConfig, *, store: BaseStore):
    """Runs before each LLM call.

    1. Saves the latest user message to the store for async extraction
       (AgentCore processes it in the background using the configured strategies).
    2. Retrieves relevant memories from preferences, facts, and summaries.
    3. Injects them as a SystemMessage at the start of the conversation.
    """
    actor_id = config["configurable"]["actor_id"]
    thread_id = config["configurable"]["thread_id"]
    messages = list(state.get("messages", []))

    # Find the last human message
    last_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg
            break

    if not last_human:
        return {"llm_input_messages": messages}

    # Save user message for async extraction by AgentCore
    session_ns = (actor_id, thread_id)
    store.put(session_ns, str(uuid.uuid4()), {"message": last_human.content})

    # Retrieve relevant long-term memories
    memory_context = _build_memory_context(store, actor_id, last_human.content)

    if memory_context:
        injected = SystemMessage(content=memory_context)
        messages = [injected] + messages

    return {"llm_input_messages": messages}


def post_model_hook(state: dict, config: RunnableConfig, *, store: BaseStore):
    """Runs after each LLM call.

    Saves the AI response to the store so AgentCore can extract
    architectural decisions, naming patterns, and session summaries.
    """
    actor_id = config["configurable"]["actor_id"]
    thread_id = config["configurable"]["thread_id"]
    messages = state.get("messages", [])

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            session_ns = (actor_id, thread_id)
            store.put(session_ns, str(uuid.uuid4()), {"message": msg.content})
            break

    return {"messages": messages}
