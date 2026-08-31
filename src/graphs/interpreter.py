"""Programmatic tool calling (PTC) via the QuickJS interpreter middleware.

`langchain_quickjs.CodeInterpreterMiddleware` adds an ``eval`` tool to a
DeepAgent: the model writes JavaScript, the middleware runs it in an embedded
QuickJS context, and only the value of the last expression comes back into the
model context. With a PTC allowlist, selected LangChain tools are bridged into
that context as ``tools.<camelCaseName>(input)`` async functions, so the model
can loop / branch / fan out over them **in code** and return a distilled result
instead of replaying every raw tool payload through the conversation.

That is exactly the shape of the iac agent's research node: a dozen Terraform
Registry + AWS Docs MCP lookups whose raw output (whole provider schema pages)
is far larger than the summary the generator actually needs.

Everything here degrades to normal tool calling. `build_ptc_middleware` returns
``(middleware, direct_tools)`` and exactly one of the two is ever non-empty, so
a call site can hand both straight to `create_deep_agent` regardless of whether
the interpreter is available:

    middleware, direct_tools = build_ptc_middleware(mcp_tools, agent="iac-agent")
    create_deep_agent(model=..., tools=direct_tools, middleware=middleware or None)

Notes on the pin (see also AGENTS.md §9): `langchain-quickjs` 0.3.3+ requires
``deepagents>=0.6.12`` and this image is pinned to 0.6.8, so requirements.txt
caps it at ``<0.3.3``. Interpreters are a **beta** API upstream — every import
and construction path below is best-effort on purpose.
"""

import logging
import os
import re
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# `tools.<name>` must be a legal JS identifier after camel-casing, else
# `filter_tools_for_ptc` raises — and it raises from inside `wrap_model_call`,
# i.e. mid-run, not at construction. Pre-filter here so a badly named MCP tool
# costs us that one tool instead of the whole research node.
_JS_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")

# The subagent dispatch tool is reserved by the middleware (it is always the
# `task()` global inside the REPL); passing it through `ptc=` raises ValueError.
_RESERVED_PTC_NAMES = frozenset({"task"})

# Defaults, all overridable via env (§7). The upstream defaults assume
# fast in-process tools; MCP lookups over stdio are neither fast nor small.
_DEFAULT_TIMEOUT_SECONDS = 300.0   # upstream default is 5s — one MCP call blows that
_DEFAULT_MAX_RESULT_CHARS = 24000  # upstream 4000 truncates a research digest
_DEFAULT_MEMORY_LIMIT_MB = 128     # upstream 64MB; provider docs are large strings
_DEFAULT_MAX_PTC_CALLS = 64        # upstream 256; a research pass needs ~10-20


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_number(name: str, default: float, cast) -> Any:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %r", name, raw, default)
        return default


def to_camel_case(name: str) -> str:
    """``snake_case`` / ``kebab-case`` → ``camelCase`` (mirrors the middleware)."""
    parts = re.split(r"[_-]+", name)
    head, *rest = [p for p in parts if p] or [""]
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def is_ptc_exposable(tool: Any) -> bool:
    """Whether ``tool`` can be exposed as ``tools.<camelCaseName>``."""
    name = getattr(tool, "name", None)
    if not isinstance(name, str) or not name:
        return False
    if name in _RESERVED_PTC_NAMES:
        return False
    return bool(_JS_IDENTIFIER.match(to_camel_case(name)))


def _load_middleware_class():
    """Import `CodeInterpreterMiddleware`, or return None if unavailable.

    Kept as a seam: the package is an optional dependency (and a beta one), so
    both the deployment and the tests need to exercise the absent path.
    """
    try:
        from langchain_quickjs import CodeInterpreterMiddleware
    except Exception as exc:  # ImportError, or a broken quickjs-rs/wasmtime load
        logger.warning("QuickJS interpreter unavailable (%s); PTC disabled", exc)
        return None
    return CodeInterpreterMiddleware


def build_ptc_middleware(
    tools: Sequence[Any],
    *,
    agent: str,
    enabled: bool | None = None,
) -> tuple[list, list]:
    """Bridge ``tools`` into a QuickJS interpreter for programmatic tool calling.

    Args:
        tools: The tools to expose. Passed as `BaseTool` instances rather than
            names so they do **not** have to be on the agent's own tool list —
            `filter_tools_for_ptc` includes explicit instances directly.
        agent: Agent name, for log lines only.
        enabled: Override the env flag (tests, and callers that gate per node).

    Returns:
        ``(middleware, direct_tools)``. On success: ``([middleware], [])`` —
        the tools reach the model only through ``tools.*`` inside ``eval``, so
        their payloads never enter the transcript. On any failure, or when PTC
        is switched off: ``([], list(tools))`` — plain tool calling, i.e. the
        behavior this node had before PTC existed.
    """
    fallback: tuple[list, list] = ([], list(tools))

    if not tools:
        return fallback
    if enabled is None:
        enabled = _env_flag("IAC_RESEARCH_PTC", True)
    if not enabled:
        logger.info("[%s] PTC disabled by config; using direct tool calling", agent)
        return fallback

    middleware_cls = _load_middleware_class()
    if middleware_cls is None:
        return fallback

    exposed = [t for t in tools if is_ptc_exposable(t)]
    skipped = [getattr(t, "name", "?") for t in tools if not is_ptc_exposable(t)]
    if skipped:
        logger.warning(
            "[%s] %d tool(s) not exposable via PTC, dropped from the allowlist: %s",
            agent, len(skipped), ", ".join(map(str, skipped)),
        )
    if not exposed:
        return fallback

    memory_limit_mb = _env_number("PTC_MEMORY_LIMIT_MB", _DEFAULT_MEMORY_LIMIT_MB, int)
    try:
        middleware = middleware_cls(
            ptc=exposed,
            # One wall-clock budget covers the whole eval, awaited PTC calls
            # included — an MCP fan-out needs minutes, not the 5s default.
            timeout=_env_number("PTC_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS, float),
            max_result_chars=_env_number(
                "PTC_MAX_RESULT_CHARS", _DEFAULT_MAX_RESULT_CHARS, int
            ),
            memory_limit=memory_limit_mb * 1024 * 1024,
            max_ptc_calls=_env_number("PTC_MAX_CALLS", _DEFAULT_MAX_PTC_CALLS, int),
            # These mini agents are single-shot and built with checkpointer=False;
            # "turn" skips the cross-turn snapshot (which would otherwise be
            # serialized into state we throw away) and evicts the QuickJS runtime
            # in `after_agent` instead of leaking one per research call.
            mode="turn",
            subagents=False,
        )
    except Exception as exc:  # beta API — a signature change must not kill the node
        logger.warning(
            "[%s] Could not build the PTC interpreter (%s); using direct tool calling",
            agent, exc,
        )
        return fallback

    logger.info(
        "[%s] PTC enabled: %d tool(s) bridged into the QuickJS interpreter (%s)",
        agent, len(exposed), ", ".join(to_camel_case(t.name) for t in exposed),
    )
    return [middleware], []
