"""AgentCore Code Interpreter sandbox and local shell backends for subagent validation.

- Code Interpreter (MicroVM): Used by data-eng for pytest/pyspark validation.
- LocalShellBackend: Used by IaC and CI/CD for terraform/actionlint which
  are pre-installed in the AgentCore Runtime container.
"""

import logging
import os

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from langchain_agentcore_codeinterpreter import AgentCoreSandbox
from deepagents.backends import LocalShellBackend

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "us-west-2")
CODE_INTERPRETER_ID = os.environ.get("AGENTCORE_CODE_INTERPRETER_ID", "")

# Module-level caches
_sandbox_cache: dict = {}
_local_shell_cache: dict = {}


def get_sandbox_backend() -> AgentCoreSandbox | None:
    """Get or create a shared AgentCoreSandbox backend (Code Interpreter).

    Used by data-eng validate/fix nodes for pytest + pyspark.
    Returns None if CODE_INTERPRETER_ID is not configured.
    """
    if not CODE_INTERPRETER_ID:
        logger.warning("AGENTCORE_CODE_INTERPRETER_ID not set, sandbox unavailable")
        return None

    if "backend" in _sandbox_cache:
        return _sandbox_cache["backend"]

    try:
        interpreter = CodeInterpreter(region=REGION)
        interpreter.start(identifier=CODE_INTERPRETER_ID)

        # Pre-install validation dependencies so validate/fix nodes
        # don't waste time installing on every invocation.
        logger.info("Pre-installing sandbox dependencies...")
        interpreter.install_packages([
            "pytest",
            "pandas",
            "pyspark",
            "pyyaml",
        ])
        logger.info("Sandbox dependencies installed")

        backend = AgentCoreSandbox(interpreter=interpreter)
        _sandbox_cache["backend"] = backend
        _sandbox_cache["interpreter"] = interpreter
        logger.info("Code Interpreter sandbox started: %s", CODE_INTERPRETER_ID)
        return backend
    except Exception as exc:
        logger.warning("Failed to start Code Interpreter sandbox: %s", exc)
        return None


def get_local_shell_backend() -> LocalShellBackend:
    """Get or create a shared LocalShellBackend.

    Used by IaC (terraform validate/fmt) and CI/CD (actionlint) nodes
    since those binaries are pre-installed in the container.
    """
    if "backend" in _local_shell_cache:
        return _local_shell_cache["backend"]

    backend = LocalShellBackend(root_dir="/tmp/agent-workspace")
    _local_shell_cache["backend"] = backend
    logger.info("LocalShellBackend ready at /tmp/agent-workspace")
    return backend
