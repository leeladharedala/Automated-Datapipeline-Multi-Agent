"""CI/CD SubAgent Graph — GitHub Actions workflow generation with self-healing validation.

Builds a compiled LangGraph StateGraph that:
1. Generates GitHub Actions workflow files via a mini DeepAgent with write_file (agent node)
2. Validates with actionlint via a mini DeepAgent with execute (agent node)
3. Self-heals on validation failure via a mini DeepAgent with edit_file (agent node)
4. Produces a final pass/fail report (pure function)

No research node — CI/CD generation starts directly from the task description.
Agent nodes use create_deep_agent to get VFS (write_file, edit_file, read_file)
and sandbox execution (execute) — artifacts are persisted to AgentCore short-term
memory via the shared VFS.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from src.graphs.state import CICDState
from src.graphs._utils import get_task_description
from src.tracing.graphs import instrument_graph
from src.tracing.utils import traced_span

# --- System prompts for agent nodes ---

_GENERATE_PROMPT = """\
You are a GitHub Actions workflow generator.

Given the task description below, generate production-grade GitHub Actions \
workflow files. Respond with EXACTLY two fenced code blocks:

```yaml:deploy.yml
<full content of deploy.yml here>
```

```yaml:destroy.yml
<full content of destroy.yml here>
```

DEPLOY WORKFLOW (deploy.yml) rules:
- Trigger on push to main AND workflow_dispatch with environment input.
- Use a concurrency group per environment to prevent parallel deploys.
- Jobs: lint → plan → apply (with manual approval gate for prod via environment protection).
- Cache Terraform plugin directory (.terraform/providers) between runs.
- Run `terraform fmt -check` and `terraform validate` before plan.
- Use `terraform plan -out=tfplan` and `terraform apply tfplan` (never auto-approve from plan).
- Upload the plan file as a workflow artifact for audit trail.
- Use OIDC for AWS auth (aws-actions/configure-aws-credentials with role-to-assume), not static keys.
- Pin all action versions to full SHA, not tags.
- Set `timeout-minutes` on each job to prevent hung runners.

DESTROY WORKFLOW (destroy.yml) rules:
- Trigger ONLY on workflow_dispatch with required environment input and confirmation input.
- Require a confirmation string (e.g., "destroy-<environment>") to prevent accidental runs.
- Use the same OIDC auth and concurrency group as deploy.
- Run `terraform plan -destroy -out=tfplan` then `terraform apply tfplan`.
- Upload the destroy plan as a workflow artifact.
- Add a final cleanup step to remove the Terraform state if fully destroyed.
- Set `timeout-minutes` generous enough for resource deletion (some AWS resources take time).

GENERAL rules:
- Parameterize environment, region, and resource names via workflow inputs.
- Use official GitHub Actions where possible (actions/checkout, aws-actions, hashicorp/setup-terraform).
- Never hardcode credentials — use GitHub secrets and OIDC.
- Use `terraform init -backend-config` for dynamic backend selection per environment.
- Respond with BOTH fenced code blocks above, nothing else before or after.
"""

_VALIDATE_PROMPT = """\
You are a GitHub Actions workflow linter. You have access to the execute tool \
which runs commands in the AgentCore Runtime sandbox (actionlint is pre-installed).

The generated workflow files have already been written to /.github/workflows/ \
in the sandbox. Run this command:
  execute("actionlint /.github/workflows/deploy.yml /.github/workflows/destroy.yml")

After running the command, report the results. Include the full output \
(both stdout and stderr). State clearly whether validation PASSED or FAILED.
"""

_FIX_PROMPT = """\
You are a GitHub Actions debugging expert. You have access to read_file, \
edit_file, and execute tools that operate on a shared VFS.

You will receive the actionlint error output. Your job:
1. Read the broken workflow file(s) using read_file
2. Analyze the error and identify the root cause
3. Fix ONLY the broken file(s) using edit_file — do not regenerate everything
4. Respond with a summary of what you fixed and why
"""



def _build_agent(model, system_prompt: str, backend=None):
    """Build a single DeepAgent instance with checkpointing disabled.

    Agents must be built once per pipeline run and reused across node
    invocations to avoid unnecessary re-instantiation overhead.
    """
    from deepagents import create_deep_agent
    kwargs = dict(
        model=model,
        system_prompt=system_prompt,
        checkpointer=False,
    )
    if backend is not None:
        kwargs["backend"] = backend
    return create_deep_agent(**kwargs)


def _invoke_agent(agent, user_message: str, bg_loop_cache: dict | None = None) -> str:
    """Invoke a pre-built agent and return the final AI message text.

    Schedules work on a long-lived background loop via run_coroutine_threadsafe
    so that the loop is never closed between calls — avoids httpx TLS cleanup
    errors that occur when asyncio.run() tears down a short-lived loop.
    """
    import asyncio
    import threading

    async def _ainvoke():
        return await agent.ainvoke({"messages": [HumanMessage(content=user_message)]})

    cache = bg_loop_cache or {}

    # Ensure a long-lived background loop exists — never use asyncio.run().
    if "loop" not in cache or not cache["loop"].is_running():
        _ready = threading.Event()
        _bg_loop = asyncio.new_event_loop()

        def _run_bg():
            asyncio.set_event_loop(_bg_loop)
            _bg_loop.call_soon_threadsafe(_ready.set)
            _bg_loop.run_forever()

        threading.Thread(target=_run_bg, daemon=True).start()
        _ready.wait(timeout=5)
        cache["loop"] = _bg_loop

    result = asyncio.run_coroutine_threadsafe(_ainvoke(), cache["loop"]).result()

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            from src.graphs._utils import _content_to_str
            return _content_to_str(msg.content)
    return ""


def _parse_yaml_blocks(response: str) -> dict[str, str]:
    """Extract named fenced YAML code blocks from the generate response.

    Looks for blocks like:
      ```yaml:deploy.yml
      <content>
      ```
    Returns a dict of {filename: content}.
    Falls back to extracting plain ```yaml blocks in order if named blocks
    are not found.
    """
    import re as _re
    content: dict[str, str] = {}

    # Try named blocks first: ```yaml:filename.yml
    named = _re.findall(r"```(?:yaml:)([\w./]+\.yml)\n(.*?)```", response, _re.DOTALL)
    for fname, code in named:
        content[fname] = code.strip()

    if "deploy.yml" in content:
        return content

    # Fallback: grab plain ```yaml blocks in order
    plain = _re.findall(r"```(?:yaml|yml)?\n(.*?)```", response, _re.DOTALL)
    expected = ["deploy.yml", "destroy.yml"]
    for i, code in enumerate(plain[:2]):
        content[expected[i]] = code.strip()

    return content


def _generate(state: CICDState, agent, bg_loop_cache: dict | None = None) -> dict[str, Any]:
    """Agent node: generate GitHub Actions workflow files and store content in state.

    Accepts a pre-built agent so the same instance is reused across calls.
    """
    task = get_task_description(state)

    with traced_span("agent:cicd.generate", {
        "agent.graph": "cicd",
        "agent.node": "generate",
        "agent.role": "code_generator",
    }):
        response = _invoke_agent(agent, f"## Task\n{task}", bg_loop_cache=bg_loop_cache)

    workflow_content = _parse_yaml_blocks(response)

    workflow_artifacts = {
        "deploy.yml": "/.github/workflows/deploy.yml",
        "destroy.yml": "/.github/workflows/destroy.yml",
    }
    return {
        "workflow_artifacts": workflow_artifacts,
        "workflow_content": workflow_content,
        "messages": [AIMessage(content=response)],
    }


def _validate(state: CICDState, agent, bg_loop_cache: dict | None = None) -> dict[str, Any]:
    """Agent node: write generated workflow files into sandbox then run actionlint.

    Accepts a pre-built agent so the same instance is reused across validate/fix cycles.
    """
    task = get_task_description(state)
    artifacts = state.get("workflow_artifacts", {})
    workflow_content = state.get("workflow_content", {})

    deploy_content = workflow_content.get("deploy.yml", "")
    destroy_content = workflow_content.get("destroy.yml", "")
    write_cmds = (
        "mkdir -p /.github/workflows"
        f" && python3 -c \"open('/.github/workflows/deploy.yml', 'w').write({repr(deploy_content)})\""
        f" && python3 -c \"open('/.github/workflows/destroy.yml', 'w').write({repr(destroy_content)})\""
    )

    files_list = ", ".join(artifacts.values()) if artifacts else "unknown"

    user_msg = (
        f"## Step 1: Write generated files into the sandbox\n"
        f"Run this exact command using execute:\n"
        f"```\n{write_cmds}\n```\n\n"
        f"## Step 2: Run actionlint validation\n"
        f"## Task\n{task}\n\n"
        f"## Generated Files\n{files_list}\n\n"
        "After writing the files, run actionlint validation now."
    )
    with traced_span("agent:cicd.validate", {
        "agent.graph": "cicd",
        "agent.node": "validate",
        "agent.role": "validator",
    }):
        response = _invoke_agent(agent, user_msg, bg_loop_cache=bg_loop_cache)

    upper = response.upper()
    if "VALIDATION FAILED" in upper or "VALIDATION: FAILED" in upper:
        passed = False
    elif "VALIDATION PASSED" in upper or "VALIDATION: PASSED" in upper:
        passed = True
    else:
        passed = "PASSED" in upper and "FAILED" not in upper
    return {
        "validation_passed": passed,
        "validation_output": response,
    }


def _fix(state: CICDState, agent, bg_loop_cache: dict | None = None) -> dict[str, Any]:
    """Agent node: fix actionlint errors and update workflow_content in state.

    Accepts a pre-built agent so the same instance is reused across validate/fix cycles.
    """
    attempt = state.get("attempt", 0) + 1
    error_output = state.get("validation_output", "")
    task = get_task_description(state)

    user_msg = (
        f"## Original Task\n{task}\n\n"
        f"## Actionlint Error (attempt {attempt})\n{error_output}\n\n"
        "Fix the broken workflow files in /.github/workflows/. "
        "Read the files using execute (e.g. execute('cat /.github/workflows/deploy.yml')), "
        "fix the issues, then respond with the corrected file content in fenced code blocks:\n\n"
        "```yaml:deploy.yml\n<corrected content>\n```\n\n"
        "```yaml:destroy.yml\n<corrected content>\n```"
    )
    with traced_span("agent:cicd.fix", {
        "agent.graph": "cicd",
        "agent.node": "fix",
        "agent.role": "debugger",
        "agent.attempt": attempt,
    }):
        response = _invoke_agent(agent, user_msg, bg_loop_cache=bg_loop_cache)

    fixed_content = _parse_yaml_blocks(response)
    current_content = state.get("workflow_content", {})
    updated_content = {**current_content, **fixed_content}

    return {
        "attempt": attempt,
        "workflow_content": updated_content,
        "messages": [AIMessage(content=response)],
    }


def _report(state: CICDState) -> dict[str, Any]:
    """Pure function: format final report and set messages for CompiledSubAgent return."""
    import json as _json
    passed = state.get("validation_passed", False)
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    validation_output = state.get("validation_output", "")
    artifacts = state.get("workflow_artifacts", {})
    workflow_content = state.get("workflow_content", {})

    files_list = ", ".join(artifacts.keys()) if artifacts else "none"

    # Human-readable fenced blocks for context
    file_contents = ""
    for fname in ["deploy.yml", "destroy.yml"]:
        content = workflow_content.get(fname, "")
        if content:
            file_contents += f"\n\n### {fname}\n```yaml\n{content}\n```"
        else:
            file_contents += f"\n\n### {fname}\n(content unavailable)"

    if passed:
        status_line = "VALIDATION: PASSED\nactionlint completed successfully."
    else:
        status_line = f"VALIDATION: FAILED ({attempt}/{max_attempts} attempts exhausted)\n\nLAST ERROR:\n{validation_output}"

    # Structured JSON block at the end — the orchestrator extracts files from
    # this block directly instead of parsing fenced YAML blocks.
    pr_files = {
        f".github/workflows/{fname}": workflow_content.get(fname, "")
        for fname in ["deploy.yml", "destroy.yml"]
        if workflow_content.get(fname)
    }
    structured_block = (
        "\n\n<!-- PR_FILES_JSON\n"
        + _json.dumps(pr_files, indent=2)
        + "\n-->"
    )

    report = (
        f"{status_line}\n"
        f"Files generated: {files_list}\n"
        f"Attempts used: {attempt}"
        f"{file_contents}"
        f"{structured_block}"
    )

    return {
        "report": report,
        "messages": [AIMessage(content=report)],
    }


def _should_retry_or_report(state: CICDState) -> str:
    """Conditional edge: decide whether to fix, or go to report."""
    if state.get("validation_passed", False):
        return "report"
    if state.get("attempt", 0) < state.get("max_attempts", 3):
        return "fix"
    return "report"


def build_cicd_graph(model):
    """Factory: build and compile the CI/CD SubAgent StateGraph.

    Agents are built ONCE here and reused across all node invocations to
    avoid unnecessary re-instantiation overhead per call.

    Args:
        model: A ChatAnthropic (or compatible) LLM instance.

    Returns:
        A compiled LangGraph StateGraph ready for invocation as a CompiledSubAgent.
    """
    graph = StateGraph(CICDState)

    # Load local shell backend (actionlint pre-installed in container)
    from src.sandbox import get_local_shell_backend
    sandbox = get_local_shell_backend()

    # Shared background loop cache — ensures all agents in this graph reuse
    # the same long-lived loop for asyncio.run_coroutine_threadsafe calls.
    _bg_loop_cache: dict = {}

    # Build agents ONCE per pipeline run and close over them in node lambdas.
    # generate_agent: pure LLM workflow generation, no sandbox tools.
    # validate_agent: LocalShellBackend for actionlint.
    # fix_agent: LocalShellBackend for reading + fixing workflow files.
    generate_agent = _build_agent(model, _GENERATE_PROMPT)
    validate_agent = _build_agent(model, _VALIDATE_PROMPT, backend=sandbox)
    fix_agent = _build_agent(model, _FIX_PROMPT, backend=sandbox)

    def generate(state: CICDState) -> dict[str, Any]:
        return _generate(state, generate_agent, bg_loop_cache=_bg_loop_cache)

    def validate(state: CICDState) -> dict[str, Any]:
        return _validate(state, validate_agent, bg_loop_cache=_bg_loop_cache)

    def fix(state: CICDState) -> dict[str, Any]:
        return _fix(state, fix_agent, bg_loop_cache=_bg_loop_cache)

    def report(state: CICDState) -> dict[str, Any]:
        return _report(state)

    # Add nodes
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("fix", fix)
    graph.add_node("report", report)

    # Wire edges: generate → validate → conditional → fix/report
    graph.set_entry_point("generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        _should_retry_or_report,
        {"fix": "fix", "report": "report"},
    )
    graph.add_edge("fix", "validate")
    graph.add_edge("report", END)

    instrument_graph(graph, "cicd", {
        "generate": "agent",
        "validate": "agent",
        "fix": "agent",
        "report": "function",
    })

    return graph.compile()
