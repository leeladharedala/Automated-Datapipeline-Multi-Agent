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
You are a GitHub Actions workflow generator. You have access to write_file and \
read_file tools that write to a shared VFS (persisted by AgentCore Memory).

Given the task description below, generate production-grade GitHub Actions \
workflow files. Write the following files using write_file:
- /.github/workflows/deploy.yml — Deploy workflow
- /.github/workflows/destroy.yml — Destroy/teardown workflow

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
- Write BOTH files, then respond with a summary of what you wrote.
"""

_VALIDATE_PROMPT = """\
You are a GitHub Actions workflow linter. You have access to the execute tool \
which runs commands in the AgentCore Runtime sandbox (actionlint is pre-installed).

Run this command:
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
    """Create a mini DeepAgent, invoke it, and return the final AI message text."""
    from deepagents import create_deep_agent

    kwargs = dict(model=model, system_prompt=system_prompt)
    if backend is not None:
        kwargs["backend"] = backend

    agent = create_deep_agent(**kwargs)
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            from src.graphs._utils import _content_to_str
            return _content_to_str(msg.content)
    return ""


def _generate(state: CICDState, model) -> dict[str, Any]:
    """Agent node: create_deep_agent with write_file to generate workflow files."""
    task = get_task_description(state)

    with traced_span("agent:cicd.generate", {
        "agent.graph": "cicd",
        "agent.node": "generate",
        "agent.role": "code_generator",
    }):
        response = _run_agent(_GENERATE_PROMPT, f"## Task\n{task}", model)

    workflow_artifacts = {
        "deploy.yml": "/.github/workflows/deploy.yml",
        "destroy.yml": "/.github/workflows/destroy.yml",
    }
    return {"workflow_artifacts": workflow_artifacts, "messages": [AIMessage(content=response)]}


def _validate(state: CICDState, model, sandbox=None) -> dict[str, Any]:
    """Agent node: create_deep_agent with execute to run actionlint."""
    task = get_task_description(state)
    artifacts = state.get("workflow_artifacts", {})
    files_list = ", ".join(artifacts.values()) if artifacts else "unknown"

    user_msg = (
        f"## Task\n{task}\n\n"
        f"## Generated Files\n{files_list}\n\n"
        "Run actionlint validation now."
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
    return {
        "validation_passed": passed,
        "validation_output": response,
    }


def _fix(state: CICDState, model, sandbox=None) -> dict[str, Any]:
    """Agent node: create_deep_agent with edit_file to fix actionlint errors."""
    attempt = state.get("attempt", 0) + 1
    error_output = state.get("validation_output", "")
    task = get_task_description(state)

    user_msg = (
        f"## Original Task\n{task}\n\n"
        f"## Actionlint Error (attempt {attempt})\n{error_output}\n\n"
        "Fix the broken workflow files in /.github/workflows/."
    )
    with traced_span("agent:cicd.fix", {
        "agent.graph": "cicd",
        "agent.node": "fix",
        "agent.role": "debugger",
        "agent.attempt": attempt,
    }):
        response = _run_agent(_FIX_PROMPT, user_msg, model, backend=sandbox)

    return {"attempt": attempt, "messages": [AIMessage(content=response)]}


def _report(state: CICDState) -> dict[str, Any]:
    """Pure function: format final report and set messages for CompiledSubAgent return."""
    passed = state.get("validation_passed", False)
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    validation_output = state.get("validation_output", "")
    artifacts = state.get("workflow_artifacts", {})

    files_list = ", ".join(artifacts.keys()) if artifacts else "none"

    if passed:
        report = (
            "VALIDATION: PASSED\n"
            "actionlint completed successfully.\n"
            f"Files generated: {files_list}\n"
            f"Attempts used: {attempt}"
        )
    else:
        report = (
            f"VALIDATION: FAILED ({attempt}/{max_attempts} attempts exhausted)\n\n"
            f"LAST ERROR:\n{validation_output}\n\n"
            f"Files generated: {files_list}"
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
    sandbox = get_local_shell_backend()

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
