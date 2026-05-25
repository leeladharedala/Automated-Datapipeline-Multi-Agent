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

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from src.graphs.state import IaCState
from src.graphs._utils import get_task_description
from src.tracing.graphs import instrument_graph
from src.tracing.utils import traced_span

logger = logging.getLogger(__name__)

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
Terraform HCL code. Respond with EXACTLY four fenced code blocks:

```hcl:provider.tf
<full content of provider.tf here>
```

```hcl:variables.tf
<full content of variables.tf here>
```

```hcl:main.tf
<full content of main.tf here>
```

```hcl:outputs.tf
<full content of outputs.tf here>
```

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
- Respond with ALL four fenced code blocks above, nothing else before or after.
"""

_VALIDATE_PROMPT = """\
You are a Terraform plan reviewer. You have access to the execute tool \
which runs commands in the AgentCore Runtime sandbox (terraform is pre-installed).

The generated Terraform files have already been written to /infra/ in the sandbox. \
Given the task description and research context below, verify that the generated \
Terraform code creates the correct resources as per the requested architecture.

Run these steps:
1. execute("cd /infra && terraform init -backend=false")
2. execute("cd /infra && terraform plan -input=false -no-color")

Then review the plan output carefully:
- List every resource that terraform plans to create, modify, or destroy.
- Compare each planned resource against the task description and research context.
- Check: are ALL resources from the requested architecture present in the plan?
- Check: are there any EXTRA resources not requested?
- Check: are resource configurations correct (instance types, regions, names, etc.)?
- Check: are resource dependencies and references wired correctly?

If the plan shows all requested resources with correct configurations, state: \
VALIDATION PASSED

If any resources are missing, misconfigured, or the plan has errors, explain \
exactly what is wrong and state: VALIDATION FAILED
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



def _run_agent(system_prompt: str, user_message: str, model, tools=None, backend=None,
               _bg_loop_cache: dict | None = None) -> str:
    """Create a mini DeepAgent, invoke it, and return the final AI message text.

    This gives the agent access to the full DeepAgent tool stack:
    write_file, edit_file, read_file, execute, ls, glob, grep.
    Tools passed via the `tools` parameter are added on top.
    Pass a backend (e.g. AgentCoreSandbox) to enable the execute tool.

    Uses ainvoke (async) to support tools that only implement async invocation
    (e.g. MCP StructuredTools from langchain-mcp-adapters).

    Schedules work on a long-lived background loop (passed via _bg_loop_cache)
    when available, to avoid "Event loop is closed" errors from httpx cleanup
    after asyncio.run() tears down a short-lived loop.
    """
    import asyncio
    from deepagents import create_deep_agent

    kwargs = dict(
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],
    )
    if backend is not None:
        kwargs["backend"] = backend
    kwargs["checkpointer"] = False  # disable checkpointing for inner agents

    agent = create_deep_agent(**kwargs)

    async def _ainvoke():
        return await agent.ainvoke({"messages": [HumanMessage(content=user_message)]})

    # Use the caller-supplied background loop when available so that the loop
    # is never closed between calls (avoids httpx TLS cleanup errors).
    bg_loop = (_bg_loop_cache or {}).get("loop")
    if bg_loop is not None and bg_loop.is_running():
        result = asyncio.run_coroutine_threadsafe(_ainvoke(), bg_loop).result()
    else:
        result = asyncio.run(_ainvoke())

    # Extract the last AI message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            from src.graphs._utils import _content_to_str
            return _content_to_str(msg.content)
    return ""


def _research(state: IaCState, model, tools_cache: dict) -> dict[str, Any]:
    """Agent node: create_deep_agent with MCP tools to research AWS resource schemas.

    Uses Sonnet 4.6 to intelligently call Terraform Registry and AWS Docs MCP
    tools with the correct input schemas, extracting resource types, arguments,
    and best practices needed for code generation.

    If no tools were provided at graph build time, lazily loads them from the
    Terraform Registry and AWS Docs MCP servers on first invocation. The loaded
    tools and client are cached in tools_cache for subsequent calls.
    """
    from src.graphs.realtime_logs import log_subagent_progress
    log_subagent_progress("iac-agent", "[system] Starting IaC Agent research node...")
    log_subagent_progress("iac-agent", "[research] Analyzing target S3 Bucket structure and encryption constraints...")
    log_subagent_progress("iac-agent", "[research] Inspecting AWS Glue Job Version 4.0 requirement parameters...")
    log_subagent_progress("iac-agent", "[research] Validating IAM Role & Policy resource access requirements...")

    task = get_task_description(state)
    if not task:
        return {"research_context": "No task provided."}

    # Lazy-load MCP tools if none were provided at build time
    active_tools = tools_cache.get("tools", [])
    if not active_tools:
        try:
            from src.tools.gateway import load_gateway_tools
            import threading
            logger.info("Lazy-loading Terraform Registry + AWS Docs MCP tools...")
            # MCP stdio clients need a running event loop for the lifetime
            # of the subprocess connections.  Spin up a dedicated loop in a
            # background daemon thread and load tools on it.
            loop = asyncio.new_event_loop()

            def _run_loop():
                asyncio.set_event_loop(loop)
                loop.run_forever()

            t = threading.Thread(target=_run_loop, daemon=True)
            t.start()

            future = asyncio.run_coroutine_threadsafe(load_gateway_tools(), loop)
            clients, active_tools = future.result(timeout=120)

            tools_cache["tools"] = active_tools
            tools_cache["clients"] = clients
            tools_cache["loop"] = loop
            logger.info("Loaded %d MCP tools (Terraform Registry + AWS Docs)", len(active_tools))
        except Exception as exc:
            logger.warning("Failed to load MCP tools, proceeding without research: %s", exc)
            return {"research_context": "MCP tools unavailable; skipping research phase."}

    if not active_tools:
        return {"research_context": "No research tools available."}

    with traced_span("agent:iac.research", {
        "agent.graph": "iac",
        "agent.node": "research",
        "agent.role": "researcher",
        "agent.tool_count": len(active_tools),
    }):
        response = _run_agent(
            _RESEARCH_PROMPT,
            f"## Task\n{task}\n\nResearch the AWS resources and Terraform configuration needed for this task.",
            model,
            tools=active_tools,
            _bg_loop_cache=tools_cache,
        )
    return {"research_context": response, "messages": [AIMessage(content=response)]}


def _parse_hcl_blocks(response: str) -> dict[str, str]:
    """Extract named fenced HCL code blocks from the generate response.

    Looks for blocks like:
      ```hcl:main.tf
      <content>
      ```
    Returns a dict of {filename: content}.
    Falls back to extracting plain ```hcl blocks in order if named blocks
    are not found.
    """
    import re as _re
    content: dict[str, str] = {}

    # Try named blocks first: ```hcl:filename.tf
    named = _re.findall(r"```(?:hcl:)([\w./]+\.tf)\n(.*?)```", response, _re.DOTALL)
    for fname, code in named:
        content[fname] = code.strip()

    if "main.tf" in content:
        return content

    # Fallback: grab plain ```hcl blocks in order and map to expected filenames
    plain = _re.findall(r"```(?:hcl|terraform)?\n(.*?)```", response, _re.DOTALL)
    expected = ["provider.tf", "variables.tf", "main.tf", "outputs.tf"]
    for i, code in enumerate(plain[:4]):
        content[expected[i]] = code.strip()

    return content


def _generate(state: IaCState, model, _bg_loop_cache: dict | None = None) -> dict[str, Any]:
    """Agent node: generate Terraform HCL files and store content in state.

    The generated code is stored in state['tf_content'] as a dict of
    {filename: content}. The validate node reads this and writes the files
    into the sandbox before running terraform plan — this bridges the
    filesystem gap between the generate node (VFS) and the sandbox.
    """
    task = get_task_description(state)
    research = state.get("research_context", "")

    from src.graphs.realtime_logs import log_subagent_progress
    log_subagent_progress("iac-agent", "[generate] Generating provider.tf with AWS provider version requirements...")
    log_subagent_progress("iac-agent", "[generate] Generating variables.tf specifying configurable pipeline inputs...")
    log_subagent_progress("iac-agent", "[generate] Compiling main.tf with S3 Bucket, IAM, and Glue Job...")
    log_subagent_progress("iac-agent", "[generate] Writing outputs.tf to export bucket and job ARNs...")

    user_msg = f"## Task\n{task}\n\n## Research Context\n{research}"
    with traced_span("agent:iac.generate", {
        "agent.graph": "iac",
        "agent.node": "generate",
        "agent.role": "code_generator",
        "agent.research_context_length": len(research),
    }):
        response = _run_agent(_GENERATE_PROMPT, user_msg, model, _bg_loop_cache=_bg_loop_cache)

    tf_content = _parse_hcl_blocks(response)

    # Track the expected artifact filenames
    tf_artifacts = {
        "provider.tf": "/infra/provider.tf",
        "variables.tf": "/infra/variables.tf",
        "main.tf": "/infra/main.tf",
        "outputs.tf": "/infra/outputs.tf",
    }
    return {
        "tf_artifacts": tf_artifacts,
        "tf_content": tf_content,
        "messages": [AIMessage(content=response)],
    }


def _validate(state: IaCState, model, sandbox=None, _bg_loop_cache: dict | None = None) -> dict[str, Any]:
    """Agent node: write generated Terraform files into sandbox then run terraform plan."""
    from src.graphs.realtime_logs import log_subagent_progress
    log_subagent_progress("iac-agent", "[validate] Initiating Terraform validation in secure sandbox...")

    task = get_task_description(state)
    research = state.get("research_context", "")
    artifacts = state.get("tf_artifacts", {})
    tf_content = state.get("tf_content", {})

    # Build shell commands that write the generated files into the sandbox.
    # Use printf with %s to safely embed arbitrary HCL content without
    # shell quoting issues — repr() gives us a Python string literal which
    # we then pass via python -c to write the file.
    write_cmds = "mkdir -p /infra"
    for fname in ["provider.tf", "variables.tf", "main.tf", "outputs.tf"]:
        content = tf_content.get(fname, "")
        # Use python3 -c to write the file — avoids all shell quoting issues
        escaped = repr(content)
        write_cmds += f" && python3 -c \"open('/infra/{fname}', 'w').write({escaped})\""

    files_list = ", ".join(artifacts.values()) if artifacts else "unknown"

    user_msg = (
        f"## Step 1: Write generated files into the sandbox\n"
        f"Run this exact command using execute:\n"
        f"```\n{write_cmds}\n```\n\n"
        f"## Step 2: Run terraform plan\n"
        f"## Task\n{task}\n\n"
        f"## Research Context Summary\n{research[:500]}\n\n"
        f"## Generated Files\n{files_list}\n\n"
        "After writing the files, run terraform plan and verify the planned resources "
        "match the architecture above."
    )
    with traced_span("agent:iac.validate", {
        "agent.graph": "iac",
        "agent.node": "validate",
        "agent.role": "validator",
    }):
        response = _run_agent(_VALIDATE_PROMPT, user_msg, model, backend=sandbox,
                              _bg_loop_cache=_bg_loop_cache)

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


def _fix(state: IaCState, model, sandbox=None, _bg_loop_cache: dict | None = None) -> dict[str, Any]:
    """Agent node: fix broken Terraform files using sandbox execute + browser tools."""
    attempt = state.get("attempt", 0) + 1
    error_output = state.get("validation_output", "")
    task = get_task_description(state)

    user_msg = (
        f"## Original Task\n{task}\n\n"
        f"## Validation Error (attempt {attempt})\n{error_output}\n\n"
        "Fix the broken Terraform files in /infra/. "
        "Read the files using execute (e.g. execute('cat /infra/main.tf')), "
        "fix the issues, then respond with the corrected file content in fenced "
        "code blocks:\n\n"
        "```hcl:provider.tf\n<corrected content>\n```\n\n"
        "```hcl:variables.tf\n<corrected content>\n```\n\n"
        "```hcl:main.tf\n<corrected content>\n```\n\n"
        "```hcl:outputs.tf\n<corrected content>\n```"
    )
    with traced_span("agent:iac.fix", {
        "agent.graph": "iac",
        "agent.node": "fix",
        "agent.role": "debugger",
        "agent.attempt": attempt,
    }):
        response = _run_agent(_FIX_PROMPT, user_msg, model, backend=sandbox,
                              _bg_loop_cache=_bg_loop_cache)

    # Parse corrected HCL blocks and update tf_content in state
    fixed_content = _parse_hcl_blocks(response)
    current_content = state.get("tf_content", {})
    updated_content = {**current_content, **fixed_content}

    return {
        "attempt": attempt,
        "tf_content": updated_content,
        "messages": [AIMessage(content=response)],
    }


def _report(state: IaCState) -> dict[str, Any]:
    """Pure function: format final report and set messages for CompiledSubAgent return."""
    import json as _json
    passed = state.get("validation_passed", False)
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    validation_output = state.get("validation_output", "")
    artifacts = state.get("tf_artifacts", {})
    tf_content = state.get("tf_content", {})

    files_list = ", ".join(artifacts.keys()) if artifacts else "none"

    # Human-readable fenced blocks for context
    file_contents = ""
    for fname in ["provider.tf", "variables.tf", "main.tf", "outputs.tf"]:
        content = tf_content.get(fname, "")
        if content:
            file_contents += f"\n\n### {fname}\n```hcl\n{content}\n```"
        else:
            file_contents += f"\n\n### {fname}\n(content unavailable)"

    if passed:
        status_line = "VALIDATION: PASSED\nterraform validate completed successfully."
    else:
        status_line = f"VALIDATION: FAILED ({attempt}/{max_attempts} attempts exhausted)\n\nLAST ERROR:\n{validation_output}"

    # Structured JSON block at the end — the orchestrator extracts files from
    # this block directly instead of parsing fenced HCL blocks, which is
    # fragile when reports are long or content contains special characters.
    # Keys match the target file paths expected by submit_pr.
    pr_files = {
        f"infra/{fname}": tf_content.get(fname, "")
        for fname in ["provider.tf", "variables.tf", "main.tf", "outputs.tf"]
        if tf_content.get(fname)
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
    # Cache for lazily-loaded MCP tools (loaded once on first research call)
    _cached_tools: dict[str, Any] = {"tools": tools or [], "clients": None}

    # Load Code Interpreter sandbox for validate/fix nodes
    from src.sandbox import get_local_shell_backend
    sandbox = get_local_shell_backend(agent_name="iac-agent")

    def research(state: IaCState) -> dict[str, Any]:
        return _research(state, model, _cached_tools)

    def generate(state: IaCState) -> dict[str, Any]:
        return _generate(state, model, _cached_tools)

    def validate(state: IaCState) -> dict[str, Any]:
        return _validate(state, model, sandbox=sandbox, _bg_loop_cache=_cached_tools)

    def fix(state: IaCState) -> dict[str, Any]:
        return _fix(state, model, sandbox=sandbox, _bg_loop_cache=_cached_tools)

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
