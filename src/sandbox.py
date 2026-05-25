"""Sandbox backends for subagent validation nodes.

- Code Interpreter toolkit (langchain-aws): Used by data-eng for structural
  validation in an isolated MicroVM. Provides execute_code, install_packages, etc.
- LocalShellBackend: Used by IaC and CI/CD for terraform/actionlint which
  are pre-installed in the AgentCore Runtime container.
"""

import asyncio
import logging
import os
import threading

from deepagents.backends import LocalShellBackend

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "us-west-2")

# Module-level caches
_code_interpreter_cache: dict = {}
_local_shell_cache: dict = {}


def reset_code_interpreter_tools() -> list:
    """Clear the Code Interpreter cache and start a fresh session.

    Called when the existing session is detected as inactive
    (ValidationException: session not active). Clears the cached tools
    and toolkit, then calls get_code_interpreter_tools() to spin up a
    new MicroVM session.

    The background event loop is intentionally preserved — it is long-lived
    and shared with _invoke_agent in data_eng_graph. Only the toolkit and
    tools entries are cleared so a new session is created on the same loop.
    """
    _code_interpreter_cache.pop("tools", None)
    _code_interpreter_cache.pop("toolkit", None)
    logger.info("Code Interpreter cache cleared — starting fresh session")
    return get_code_interpreter_tools()


def get_code_interpreter_tools() -> list:
    """Get or create Code Interpreter tools for data-eng validation.

    Uses langchain-aws's create_code_interpreter_toolkit which provides
    execute_code, install_packages, write_files, etc. as LangChain tools.

    Validation uses only stdlib (ast module), so no packages are pre-installed.

    Returns an empty list if setup fails (callers fall back gracefully).
    """
    if "tools" in _code_interpreter_cache:
        return _code_interpreter_cache["tools"]

    try:
        from langchain_aws.tools import create_code_interpreter_toolkit

        # create_code_interpreter_toolkit is async, run it in a
        # background loop since graph builders are sync.
        loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        async def _setup():
            # Use the custom Code Interpreter if configured (has execution
            # role with S3 access), otherwise fall back to the default.
            ci_id = os.environ.get("AGENTCORE_CODE_INTERPRETER_ID")
            kwargs = {"region": REGION}
            if ci_id:
                kwargs["code_interpreter_identifier"] = ci_id
            toolkit, tools = await create_code_interpreter_toolkit(**kwargs)
            return toolkit, tools

        future = asyncio.run_coroutine_threadsafe(_setup(), loop)
        toolkit, tools = future.result(timeout=180)

        _code_interpreter_cache["tools"] = tools
        _code_interpreter_cache["toolkit"] = toolkit
        _code_interpreter_cache["loop"] = loop
        logger.info("Code Interpreter toolkit ready with %d tools", len(tools))
        return tools
    except Exception as exc:
        logger.warning("Failed to create Code Interpreter toolkit: %s", exc)
        return []


def get_local_shell_backend(agent_name: str = "iac-agent") -> LocalShellBackend:
    """Get or create LocalShellBackend.

    Used by IaC (terraform validate/fmt) and CI/CD (actionlint) nodes
    since those binaries are pre-installed in the container.
    """
    cache_key = f"backend_{agent_name}"
    if cache_key in _local_shell_cache:
        return _local_shell_cache[cache_key]

    root_dir = "/tmp/agent-workspace"
    os.makedirs(root_dir, exist_ok=True)
    backend = LocalShellBackend(root_dir=root_dir)
    _local_shell_cache[cache_key] = backend
    logger.info("LocalShellBackend ready for %s at %s", agent_name, root_dir)
    return backend
