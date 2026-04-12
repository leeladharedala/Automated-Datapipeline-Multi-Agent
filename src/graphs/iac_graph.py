"""IaC SubAgent Graph — Terraform generation with self-healing validation loop.

Builds a compiled LangGraph StateGraph that:
1. Researches AWS resource schemas via Gateway MCP tools (agent node — Sonnet 4.6)
2. Generates Terraform HCL files via a mini DeepAgent with write_file (agent node)
3. Validates with terraform init/validate via a mini DeepAgent with execute (agent node)
4. Self-heals on validation failure via a mini DeepAgent with edit_file (agent node)
5. Produces a final pass/fail report (pure function)

Agent nodes use create_deep_agent to get VFS (write_file, edit_file, read_file)
and sandbox execution (execute) — artifacts are persisted to AgentCore short-term
memory via the shared VFS. The research node uses Sonnet 4.6 to intelligently
call Terraform Registry and AWS Docs MCP tools with correct input schemas.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from src.graphs.state import IaCState
from src.tracing.graphs import instrument_graph
from src.tracing.utils import traced_span

# --- System prompts for agent nodes ---

_RESEARCH_PROMPT = """\
You are a Terraform research specialist. You have access to MCP tools from \
the Terraform Registry and AWS Documentation servers.

Given the task description below, research the AWS resource types, arguments, \
and best practices needed to implement the requested infrastructure. Use the \
available tools to:
1. Look up the correct Terraform resource types and their required/optional arguments
2. Find AWS documentation for the services involved
3. Identify any dependencies between resources (e.g., IAM roles needed for Lambda)

Be thorough — the code generator that follows will rely entirely on your research \
to produce valid Terraform. Include resource type names, required arguments, \
argument types, and any gotchas or constraints from the docs.

Respond with a structured summary of your findings.
"""

_GENERATE_PROMPT = """\
You are a Terraform IaC code generator. You have access to write_file and \
read_file tools that write to a shared VFS (persisted by AgentCore Memory).

Given the task description and research context below, generate production-grade \
Terraform HCL code. Write the following files using write_file:
- /infra/provider.tf — AWS provider and backend config
- /infra/variables.tf — Parameterized variables
- /infra/main.tf — Core resource definitions
- /infra/outputs.tf — Output values

Rules:
- Parameterize region, environment, and resource names via variables.
- Include proper tagging on all resources.
- Never hardcode credentials or account IDs.
- Use only resource types and arguments confirmed by the research context.
- DESTROY COMPATIBILITY: All generated Terraform MUST be cleanly destroyable via `terraform destroy`.
  - Never use `prevent_destroy = true` in lifecycle blocks.
  - Avoid circular dependencies between resources — use explicit `depends_on` where needed.
  - For S3 buckets, set `force_destroy = true` so non-empty buckets can be destroyed.
  - For RDS instances, set `skip_final_snapshot = true` and `deletion_protection = false`.
  - For CloudWatch Log Groups, set explicit `retention_in_days` to avoid orphaned resources.
  - For IAM roles/policies, ensure inline policies are used or `force_detach_policies = true`.
  - Never create resources with manual deletion requirements (e.g., non-empty ECR repos without `force_delete = true`).
- Write ALL four files, then respond with a summary of what you wrote.
"""

_VALIDATE_PROMPT = """\
You are a Terraform validation runner. You have access to the execute tool \
which runs commands in the AgentCore Runtime sandbox (terraform is pre-installed).

Given the task description below, verify that the generated Terraform code fully \
implements the requested architectural design and is logically/syntactically valid.

Run these steps:
1. execute("cd /infra && terraform init -backend=false")
2. execute("cd /infra && terraform validate")
3. execute("cd /infra && terraform plan -input=false") to verify the execution plan.

Analyze the terraform output and file contents. Does the generated code actually match \
the requested architecture in the task? Are there any missing resources or logic errors?
If anything is wrong, missing, or fails validation, explain the exact failures and state \
clearly that validation FAILED.
If everything works perfectly and perfectly matches the task description, state clearly \
that validation PASSED.
"""

_FIX_PROMPT = """\
You are a Terraform debugging expert. You have access to read_file, edit_file, \
and execute tools that operate on a shared VFS.

You will receive the validation error output. Your job:
1. Read the broken .tf file(s) using read_file
2. Analyze the error and identify the root cause
3. Fix ONLY the broken file(s) using edit_file — do not regenerate everything
4. Respond with a summary of what you fixed and why
"""


def _run_agent(system_prompt: str, user_message: str, model, tools=None) -> str:
    """Create a mini DeepAgent, invoke it, and return the final AI message text.

    This gives the agent access to the full DeepAgent tool stack:
    write_file, edit_file, read_file, execute, ls, glob, grep.
    Tools passed via the `tools` parameter are added on top.
    """
    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],
    )
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})
    # Extract the last AI message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def _research(state: IaCState, model, tools: list) -> dict[str, Any]:
    """Agent node: create_deep_agent with MCP tools to research AWS resource schemas.

    Uses Sonnet 4.6 to intelligently call Terraform Registry and AWS Docs MCP
    tools with the correct input schemas, extracting resource types, arguments,
    and best practices needed for code generation.
    """
    task = state.get("task_description", "")
    if not tools or not task:
        return {"research_context": "No research tools available or no task provided."}

    with traced_span("agent:iac.research", {
        "agent.graph": "iac",
        "agent.node": "research",
        "agent.role": "researcher",
        "agent.tool_count": len(tools),
    }):
        response = _run_agent(
            _RESEARCH_PROMPT,
            f"## Task\n{task}\n\nResearch the AWS resources and Terraform configuration needed for this task.",
            model,
            tools=tools,
        )
    return {"research_context": response, "messages": [AIMessage(content=response)]}


def _generate(state: IaCState, model) -> dict[str, Any]:
    """Agent node: create_deep_agent with write_file to generate Terraform files."""
    task = state.get("task_description", "")
    research = state.get("research_context", "")

    user_msg = f"## Task\n{task}\n\n## Research Context\n{research}"
    with traced_span("agent:iac.generate", {
        "agent.graph": "iac",
        "agent.node": "generate",
        "agent.role": "code_generator",
        "agent.research_context_length": len(research),
    }):
        response = _run_agent(_GENERATE_PROMPT, user_msg, model)

    # Track the expected artifact filenames
    tf_artifacts = {
        "provider.tf": "/infra/provider.tf",
        "variables.tf": "/infra/variables.tf",
        "main.tf": "/infra/main.tf",
        "outputs.tf": "/infra/outputs.tf",
    }
    return {"tf_artifacts": tf_artifacts, "messages": [AIMessage(content=response)]}


def _validate(state: IaCState, model) -> dict[str, Any]:
    """Agent node: create_deep_agent with execute to run terraform validate."""
    task = state.get("task_description", "")
    research = state.get("research_context", "")
    artifacts = state.get("tf_artifacts", {})
    files_list = ", ".join(artifacts.values()) if artifacts else "unknown"

    user_msg = (
        f"## Task\n{task}\n\n"
        f"## Research Context Summary\n{research[:500]}\n\n"
        f"## Generated Files\n{files_list}\n\n"
        "Run terraform validation and plan now."
    )
    with traced_span("agent:iac.validate", {
        "agent.graph": "iac",
        "agent.node": "validate",
        "agent.role": "validator",
    }):
        response = _run_agent(_VALIDATE_PROMPT, user_msg, model)

    # Check for explicit PASSED/FAILED keywords from the prompt.
    # Avoid matching generic "success" which can appear in failure context.
    upper = response.upper()
    if "VALIDATION FAILED" in upper or "VALIDATION: FAILED" in upper:
        passed = False
    elif "VALIDATION PASSED" in upper or "VALIDATION: PASSED" in upper:
        passed = True
    else:
        # Fallback: absence of explicit FAILED with presence of PASSED
        passed = "PASSED" in upper and "FAILED" not in upper
    return {
        "validation_passed": passed,
        "validation_output": response,
    }


def _fix(state: IaCState, model) -> dict[str, Any]:
    """Agent node: create_deep_agent with edit_file to fix broken Terraform files."""
    attempt = state.get("attempt", 0) + 1
    error_output = state.get("validation_output", "")
    task = state.get("task_description", "")

    user_msg = (
        f"## Original Task\n{task}\n\n"
        f"## Validation Error (attempt {attempt})\n{error_output}\n\n"
        "Fix the broken Terraform files in /infra/."
    )
    with traced_span("agent:iac.fix", {
        "agent.graph": "iac",
        "agent.node": "fix",
        "agent.role": "debugger",
        "agent.attempt": attempt,
    }):
        response = _run_agent(_FIX_PROMPT, user_msg, model)

    return {"attempt": attempt, "messages": [AIMessage(content=response)]}


def _report(state: IaCState) -> dict[str, Any]:
    """Pure function: format final report and set messages for CompiledSubAgent return."""
    passed = state.get("validation_passed", False)
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    validation_output = state.get("validation_output", "")
    artifacts = state.get("tf_artifacts", {})

    files_list = ", ".join(artifacts.keys()) if artifacts else "none"

    if passed:
        report = (
            "VALIDATION: PASSED\n"
            "terraform validate completed successfully.\n"
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


def _should_retry_or_report(state: IaCState) -> str:
    """Conditional edge: decide whether to fix, or go to report."""
    if state.get("validation_passed", False):
        return "report"
    if state.get("attempt", 0) < state.get("max_attempts", 3):
        return "fix"
    return "report"


def build_iac_graph(model, tools=None):
    """Factory: build and compile the IaC SubAgent StateGraph.

    Args:
        model: A ChatAnthropic (or compatible) LLM instance.
        tools: Optional list of Gateway MCP tools for resource schema research.

    Returns:
        A compiled LangGraph StateGraph ready for invocation as a CompiledSubAgent.
    """
    graph = StateGraph(IaCState)

    # Closures capture model and tools from factory args
    def research(state: IaCState) -> dict[str, Any]:
        return _research(state, model, tools or [])

    def generate(state: IaCState) -> dict[str, Any]:
        return _generate(state, model)

    def validate(state: IaCState) -> dict[str, Any]:
        return _validate(state, model)

    def fix(state: IaCState) -> dict[str, Any]:
        return _fix(state, model)

    def report(state: IaCState) -> dict[str, Any]:
        return _report(state)

    # Add nodes
    graph.add_node("research", research)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("fix", fix)
    graph.add_node("report", report)

    # Wire edges
    graph.set_entry_point("research")
    graph.add_edge("research", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        _should_retry_or_report,
        {"fix": "fix", "report": "report"},
    )
    graph.add_edge("fix", "validate")
    graph.add_edge("report", END)

    instrument_graph(graph, "iac", {
        "research": "agent",
        "generate": "agent",
        "validate": "agent",
        "fix": "agent",
        "report": "function",
    })

    return graph.compile()
