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


def get_local_shell_backend() -> LocalShellBackend:
    """Get or create a shared LocalShellBackend.

    Used by IaC (terraform validate/fmt) and CI/CD (actionlint) nodes
    since those binaries are pre-installed in the container.
    """
    if "backend" in _local_shell_cache:
        return _local_shell_cache["backend"]

    root_dir = "/tmp/agent-workspace"
    os.makedirs(root_dir, exist_ok=True)
    backend = LocalShellBackend(root_dir=root_dir)
    _local_shell_cache["backend"] = backend
    logger.info("LocalShellBackend ready at %s", root_dir)
    return backend
