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



def _run_agent(system_prompt: str, user_message: str, model, backend=None) -> str:
    """Create a mini DeepAgent, invoke it, and return the final AI message text.

    Uses ainvoke (async) for consistency with other graphs and to support
    any async-only tools that may be added in the future.

    Schedules work on the sandbox background loop via run_coroutine_threadsafe
    to avoid "Event loop is closed" errors when httpx cleans up TLS connections
    after asyncio.run() tears down a short-lived loop.
    """
    import asyncio
    from deepagents import create_deep_agent
    from src.sandbox import _code_interpreter_cache

    kwargs = dict(model=model, system_prompt=system_prompt)
    if backend is not None:
        kwargs["backend"] = backend
    kwargs["checkpointer"] = False

    agent = create_deep_agent(**kwargs)

    async def _ainvoke():
        return await agent.ainvoke({"messages": [HumanMessage(content=user_message)]})

    # Ensure a long-lived background loop exists.  Reuse the sandbox loop if
    # already initialised, otherwise create a dedicated one for this module.
    if "loop" not in _code_interpreter_cache:
        import threading
        _bg_loop = asyncio.new_event_loop()

        def _run_bg():
            asyncio.set_event_loop(_bg_loop)
            _bg_loop.run_forever()

        threading.Thread(target=_run_bg, daemon=True).start()
        _code_interpreter_cache["loop"] = _bg_loop

    bg_loop = _code_interpreter_cache.get("loop")
    if bg_loop is not None and bg_loop.is_running():
        result = asyncio.run_coroutine_threadsafe(_ainvoke(), bg_loop).result()
    else:
        result = asyncio.run(_ainvoke())

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


def _generate(state: CICDState, model) -> dict[str, Any]:
    """Agent node: generate GitHub Actions workflow files and store content in state.

    The generated code is stored in state['workflow_content'] as a dict of
    {filename: content}. The validate node reads this and writes the files
    into the sandbox before running actionlint — this bridges the filesystem
    gap between the generate node (VFS) and the sandbox.
    """
    from src.graphs.realtime_logs import log_subagent_progress
    log_subagent_progress("cicd-agent", "[system] Starting CI/CD Agent generation node...")
    log_subagent_progress("cicd-agent", "[research] Extracting Terraform resource names from IaC configuration...")
    log_subagent_progress("cicd-agent", "[generate] Designing deploy.yml (Linting → TF Plan → S3 Upload → TF Apply)...")
    log_subagent_progress("cicd-agent", "[generate] Designing destroy.yml manual teardown workflow with safety gates...")

    task = get_task_description(state)

    with traced_span("agent:cicd.generate", {
        "agent.graph": "cicd",
        "agent.node": "generate",
        "agent.role": "code_generator",
    }):
        response = _run_agent(_GENERATE_PROMPT, f"## Task\n{task}", model)

    workflow_content = _parse_yaml_blocks(response)

    from src.graphs.realtime_logs import log_subagent_progress
    log_subagent_progress("cicd-agent", f"[generate] Generated workflow files: {', '.join(workflow_content.keys())}")

    workflow_artifacts = {
        "deploy.yml": "/.github/workflows/deploy.yml",
        "destroy.yml": "/.github/workflows/destroy.yml",
    }
    return {
        "workflow_artifacts": workflow_artifacts,
        "workflow_content": workflow_content,
        "messages": [AIMessage(content=response)],
    }


def _validate(state: CICDState, model, sandbox=None) -> dict[str, Any]:
    """Agent node: write generated workflow files into sandbox then run actionlint."""
    from src.graphs.realtime_logs import log_subagent_progress
    log_subagent_progress("cicd-agent", "[validate] Validating GitHub Actions YAML syntax and schema triggers...")
    log_subagent_progress("cicd-agent", "[validate] Checking OIDC IAM role validation variables...")

    task = get_task_description(state)
    artifacts = state.get("workflow_artifacts", {})
    workflow_content = state.get("workflow_content", {})

    # Build shell commands that write the generated files into the sandbox.
    # Use python3 -c to write files — avoids all shell quoting issues with YAML content.
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
        response = _run_agent(_VALIDATE_PROMPT, user_msg, model, backend=sandbox)

    # Check for explicit PASSED/FAILED keywords from the prompt.
    # Avoid matching generic "success" which can appear in failure context.
    upper = response.upper()
    if "VALIDATION FAILED" in upper or "VALIDATION: FAILED" in upper:
        passed = False
    elif "VALIDATION PASSED" in upper or "VALIDATION: PASSED" in upper:
        passed = True
    else:
        passed = "PASSED" in upper and "FAILED" not in upper

    from src.graphs.realtime_logs import log_subagent_progress
    if passed:
        log_subagent_progress("cicd-agent", "[validate] CI/CD Workflow Validation PASSED. Files are ready!")
    else:
        log_subagent_progress("cicd-agent", f"[validate] CI/CD Workflow Validation FAILED:\n{response}")

    return {
        "validation_passed": passed,
        "validation_output": response,
    }


def _fix(state: CICDState, model, sandbox=None) -> dict[str, Any]:
    """Agent node: fix actionlint errors and update workflow_content in state."""
    from src.graphs.realtime_logs import log_subagent_progress
    attempt = state.get("attempt", 0) + 1
    log_subagent_progress("cicd-agent", f"[fix] Initiating self-healing loop (attempt {attempt})...")
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
        response = _run_agent(_FIX_PROMPT, user_msg, model, backend=sandbox)

    # Parse corrected YAML blocks and update workflow_content in state
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

    Args:
        model: A ChatAnthropic (or compatible) LLM instance.

    Returns:
        A compiled LangGraph StateGraph ready for invocation as a CompiledSubAgent.
    """
    graph = StateGraph(CICDState)

    # Load local shell backend for validate/fix nodes (terraform + actionlint
    # are pre-installed in the container)
    from src.sandbox import get_local_shell_backend
    sandbox = get_local_shell_backend(agent_name="cicd-agent")

    def generate(state: CICDState) -> dict[str, Any]:
        return _generate(state, model)

    def validate(state: CICDState) -> dict[str, Any]:
        return _validate(state, model, sandbox=sandbox)

    def fix(state: CICDState) -> dict[str, Any]:
        return _fix(state, model, sandbox=sandbox)

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
