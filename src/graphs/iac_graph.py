"""IaC SubAgent Graph — Terraform generation with self-healing validation loop.

Builds a compiled LangGraph StateGraph that:
1. Researches AWS resource schemas via Gateway MCP tools (agent node — Sonnet 5)
2. Generates Terraform HCL files via a mini DeepAgent with write_file (agent node)
3. Validates with terraform init/validate via a mini DeepAgent with execute (agent node)
4. Self-heals on validation failure via a mini DeepAgent with edit_file (agent node)
5. Produces a final pass/fail report (pure function)

Agent nodes use create_deep_agent to get VFS (write_file, edit_file, read_file)
and sandbox execution (execute) — artifacts are persisted to AgentCore short-term
memory via the shared VFS. The research node uses Sonnet 5 to intelligently
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

# The four files the generator must always produce; the missing-guard and the
# report's integrity check require these. The model MAY additionally split
# resources into extra files (e.g. iam.tf, s3.tf) — those are captured by
# _parse_hcl_blocks and must be written to the sandbox and included in the PR
# too, so validate/report iterate over ALL of tf_content, not just these four.
_REQUIRED_TF_FILES = ("provider.tf", "variables.tf", "main.tf", "outputs.tf")

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
- NO PLAN-TIME LOCAL FILE DEPENDENCIES: the PySpark/Glue script is produced by a
  SEPARATE agent and is NOT present in this sandbox at `terraform plan` time.
  - Never call any `file*()` function on a local path — `filemd5()`,
    `filebase64sha256()`, `file()`, `filebase64()` — these are evaluated at plan
    time and will hard-fail because the referenced file does not exist.
  - Never set `source = "<local path>"` (e.g. `${path.module}/scripts/...`) on an
    `aws_s3_object`. Do not upload the script via Terraform at all.
  - Model the Glue job's `command.script_location` (and any code reference) as an
    S3 URI, e.g. `"s3://${aws_s3_bucket.<bucket>.id}/${var.scripts_prefix}transform.py"`.
    Assume the script object is uploaded out-of-band by the CI/CD pipeline, not by Terraform.
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

Constraint: the PySpark/Glue script is produced by a SEPARATE agent and is NOT
present in this sandbox. If the error is a missing local file (e.g. a `filemd5()`
or `source = "${path.module}/scripts/..."` reference on an `aws_s3_object`), the
fix is to REMOVE that plan-time local-file dependency — do not try to create the
script. Delete the script-upload `aws_s3_object` resource and point the Glue job's
`script_location` at an S3 URI instead (the CI/CD pipeline uploads the script).
Never introduce any `file*()` call on a local path.
"""



def _run_agent(system_prompt: str, user_message: str, model, tools=None, backend=None,
               _bg_loop_cache: dict | None = None,
               heartbeat: tuple[str, str] | None = None) -> tuple[str, dict]:
    """Create a mini DeepAgent, invoke it, and return (final AI text, VFS files).

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
        fut = asyncio.run_coroutine_threadsafe(_ainvoke(), bg_loop)
        if heartbeat is not None:
            from src.graphs.progress import resolve_with_heartbeat
            result = resolve_with_heartbeat(fut, heartbeat[0], heartbeat[1])
        else:
            result = fut.result()
    else:
        result = asyncio.run(_ainvoke())

    # Extract the last AI message (plus the mini-agent's VFS output files)
    vfs_files = result.get("files") or {}
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            from src.graphs._utils import _content_to_str
            return _content_to_str(msg.content), vfs_files
    return "", vfs_files


def _research(state: IaCState, model, tools_cache: dict) -> dict[str, Any]:
    """Agent node: create_deep_agent with MCP tools to research AWS resource schemas.

    Uses Sonnet 5 to intelligently call Terraform Registry and AWS Docs MCP
    tools with the correct input schemas, extracting resource types, arguments,
    and best practices needed for code generation.

    If no tools were provided at graph build time, lazily loads them from the
    Terraform Registry and AWS Docs MCP servers on first invocation. The loaded
    tools and client are cached in tools_cache for subsequent calls.
    """
    from src.graphs.progress import emit_progress
    emit_progress("iac-agent", "[research] Starting IaC agent research node...")

    task = get_task_description(state)
    if not task:
        emit_progress("iac-agent", "[research] No task description provided; skipping research.")
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
            emit_progress("iac-agent", "[research] MCP research tools unavailable; skipping research phase.")
            return {"research_context": "MCP tools unavailable; skipping research phase."}

    if not active_tools:
        emit_progress("iac-agent", "[research] No research tools available; skipping research phase.")
        return {"research_context": "No research tools available."}

    emit_progress(
        "iac-agent",
        f"[research] Researching Terraform resource schemas and AWS docs "
        f"({len(active_tools)} MCP tools available)...",
    )

    with traced_span("agent:iac.research", {
        "agent.graph": "iac",
        "agent.node": "research",
        "agent.role": "researcher",
        "agent.tool_count": len(active_tools),
    }):
        response, _ = _run_agent(
            _RESEARCH_PROMPT,
            f"## Task\n{task}\n\nResearch the AWS resources and Terraform configuration needed for this task.",
            model,
            tools=active_tools,
            _bg_loop_cache=tools_cache,
            heartbeat=("iac-agent", "research"),
        )
    emit_progress(
        "iac-agent",
        f"[research] Research complete ({len(response)} chars of findings).",
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

    # Try named blocks first: ```hcl:filename.tf. Normalize to the basename:
    # the model sometimes writes path-qualified names (```hcl:infra/outputs.tf),
    # and every downstream consumer (validate guard, sandbox writes, report,
    # PR_FILES_JSON) keys by the bare filename — a prefixed key made outputs.tf
    # "missing" through three self-heal attempts in production.
    named = _re.findall(r"```(?:hcl:)([\w./]+\.tf)\n(.*?)```", response, _re.DOTALL)
    for fname, code in named:
        content[fname.rsplit("/", 1)[-1]] = code.strip()

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

    from src.graphs.progress import emit_progress
    emit_progress(
        "iac-agent",
        "[generate] Generating Terraform files "
        "(provider.tf, variables.tf, main.tf, outputs.tf)...",
    )

    user_msg = f"## Task\n{task}\n\n## Research Context\n{research}"
    with traced_span("agent:iac.generate", {
        "agent.graph": "iac",
        "agent.node": "generate",
        "agent.role": "code_generator",
        "agent.research_context_length": len(research),
    }):
        response, vfs_files = _run_agent(_GENERATE_PROMPT, user_msg, model, _bg_loop_cache=_bg_loop_cache,
                                         heartbeat=("iac-agent", "generate"))

    tf_content = _parse_hcl_blocks(response)
    # The mini-agent sometimes writes its output with the filesystem tools
    # instead of printing fenced blocks — recover those files from its VFS.
    from src.graphs._utils import merge_vfs_files
    tf_content = merge_vfs_files(
        tf_content, vfs_files,
        ("provider.tf", "variables.tf", "main.tf", "outputs.tf", "iam.tf"),
    )
    emit_progress(
        "iac-agent",
        f"[generate] Generated {len(tf_content)} file(s): "
        f"{', '.join(tf_content) or 'none'}",
    )

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
    from src.graphs.progress import emit_progress
    emit_progress("iac-agent", "[validate] Running terraform init and plan on the generated files...")

    task = get_task_description(state)
    research = state.get("research_context", "")
    artifacts = state.get("tf_artifacts", {})
    tf_content = state.get("tf_content", {})

    # Guard: nothing to validate. Without this, empty files get written into
    # the sandbox and terraform plan can pass on a partial config, producing
    # a PASSED report with missing files. Fail fast so the fix loop
    # regenerates the fenced blocks instead.
    missing = [f for f in _REQUIRED_TF_FILES if not tf_content.get(f)]
    if missing:
        output = (
            "VALIDATION: FAILED\n"
            f"No Terraform content was parsed for: {', '.join(missing)}. "
            "The previous response did not contain the required fenced blocks. "
            "Respond with the complete file content in fenced blocks named "
            "```hcl:provider.tf, ```hcl:variables.tf, ```hcl:main.tf and "
            "```hcl:outputs.tf."
        )
        emit_progress("iac-agent", f"[validate] {output}")
        return {"validation_passed": False, "validation_output": output}

    # Build shell commands that write the generated files into the sandbox.
    # HCL is full of double quotes and ${...} interpolations, both of which
    # the shell mangles inside a double-quoted command. Base64 is pure
    # [A-Za-z0-9+/=], so the payload survives the shell untouched.
    # Write EVERY generated file — not just the required four. The model often
    # splits resources into extra files (iam.tf, s3.tf, …); dropping them here
    # left main.tf/outputs.tf referencing undeclared resources, so terraform
    # plan failed and burned the whole self-heal budget.
    import base64
    write_cmds = "mkdir -p /infra"
    written_files = [f for f, c in tf_content.items() if c]
    for fname in written_files:
        content = tf_content.get(fname, "")
        b64 = base64.b64encode(content.encode()).decode()
        write_cmds += (
            f" && python3 -c 'import base64;"
            f' open("/infra/{fname}", "wb")'
            f'.write(base64.b64decode("{b64}"))\''
        )

    files_list = ", ".join(f"/infra/{f}" for f in written_files) or "unknown"

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
        response, _ = _run_agent(_VALIDATE_PROMPT, user_msg, model, backend=sandbox,
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

    if passed:
        emit_progress("iac-agent", "[validate] Terraform validation PASSED.")
    else:
        emit_progress("iac-agent", f"[validate] Terraform validation FAILED:\n{response}")

    return {
        "validation_passed": passed,
        "validation_output": response,
    }


def _fix(state: IaCState, model, sandbox=None, _bg_loop_cache: dict | None = None) -> dict[str, Any]:
    """Agent node: fix broken Terraform files using sandbox execute + browser tools."""
    from src.graphs.progress import emit_progress
    attempt = state.get("attempt", 0) + 1
    emit_progress("iac-agent", f"[fix] Initiating self-healing loop (attempt {attempt})...")
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
        response, vfs_files = _run_agent(_FIX_PROMPT, user_msg, model, backend=sandbox,
                                         _bg_loop_cache=_bg_loop_cache)

    # Parse corrected HCL blocks and update tf_content in state
    fixed_content = _parse_hcl_blocks(response)
    from src.graphs._utils import merge_vfs_files
    fixed_content = merge_vfs_files(
        fixed_content, vfs_files,
        ("provider.tf", "variables.tf", "main.tf", "outputs.tf", "iam.tf"),
    )
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

    # Order the required files first, then any extra generated files (iam.tf, …).
    ordered_files = list(_REQUIRED_TF_FILES) + [
        f for f in tf_content if f not in _REQUIRED_TF_FILES
    ]

    files_list = ", ".join(f for f in ordered_files if tf_content.get(f)) or "none"

    # Human-readable fenced blocks for context — every generated file.
    file_contents = ""
    for fname in ordered_files:
        content = tf_content.get(fname, "")
        if content:
            file_contents += f"\n\n### {fname}\n```hcl\n{content}\n```"
        elif fname in _REQUIRED_TF_FILES:
            file_contents += f"\n\n### {fname}\n(content unavailable)"

    # Structured JSON block at the end — the orchestrator extracts files from
    # this block directly instead of parsing fenced HCL blocks, which is
    # fragile when reports are long or content contains special characters.
    # Keys match the target file paths expected by submit_pr. Include EVERY
    # generated file (iam.tf, s3.tf, …), not just the required four — otherwise
    # a passing config with a split-out iam.tf would open an incomplete PR.
    pr_files = {
        f"infra/{fname}": content
        for fname, content in tf_content.items()
        if content
    }

    # Integrity guard: a PASSED report missing any REQUIRED file is a lie the
    # orchestrator would turn into a broken PR. Downgrade to FAILED. Extra
    # files beyond the required four are allowed (subset check, not equality).
    required_paths = {f"infra/{f}" for f in _REQUIRED_TF_FILES}
    if passed and not required_paths.issubset(set(pr_files)):
        passed = False
        missing_files = ", ".join(sorted(p.rsplit("/", 1)[-1] for p in required_paths - set(pr_files)))
        validation_output = (
            f"Validation reported success but file content is missing for: {missing_files}. "
            "The generated Terraform content was never captured — do not submit these files."
        )

    if passed:
        status_line = "VALIDATION: PASSED\nterraform plan validation completed successfully."
    else:
        status_line = f"VALIDATION: FAILED ({attempt}/{max_attempts} attempts exhausted)\n\nLAST ERROR:\n{validation_output}"
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
