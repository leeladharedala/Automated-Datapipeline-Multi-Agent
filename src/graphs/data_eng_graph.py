"""Data Engineering SubAgent Graph — transformation code generation with structural validation.

Builds a compiled LangGraph StateGraph that:
1. Samples data schema directly from S3 using boto3/pandas in-process (no agent)
2. Generates data transformation code via a mini DeepAgent with browser tools (pure LLM, no CI)
3. Validates code structure via a mini DeepAgent with Code Interpreter tools:
   - Writes generated code into the MicroVM via execute_code
   - Runs a structural check script (AST parse, required files, main() presence, pyspark imports)
4. Self-heals on validation failure via a mini DeepAgent with browser + Code Interpreter tools
5. Produces a final pass/fail report (pure function)

No research node — DataEng generation starts directly from the task description.
The generate node uses only browser tools for on-demand framework documentation lookup.
The validate and fix nodes use Code Interpreter tools to operate on the MicroVM filesystem.
Generated code is stored in state['code_content'] and written into the MicroVM by validate,
avoiding the VFS vs MicroVM filesystem split.
"""

import io
import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from src.graphs.state import DataEngState
from src.graphs._utils import get_task_description
from src.tracing.graphs import instrument_graph
from src.tracing.utils import traced_span

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 100

# --- System prompts for agent nodes ---

_GENERATE_PROMPT = (
    "You are a data transformation code generator. You have access to browser tools "
    "for looking up framework documentation on-demand.\n\n"
    "Generate production-grade PySpark data transformation code for the task below. "
    "Respond with EXACTLY two fenced code blocks:\n\n"
    "```python:transform.py\n"
    "<full content of transform.py here>\n"
    "```\n\n"
    "```python:__init__.py\n"
    "<full content of __init__.py here, can be empty>\n"
    "```\n\n"
    "TRANSFORMATION CODE rules:\n"
    "- Use PySpark DataFrames for ALL transformations (read from and write to S3 paths).\n"
    "- Structure transforms as pure functions: input DataFrame(s) in, output DataFrame out.\n"
    "- Handle nulls explicitly with .na.drop(), .na.fill(), or .when(col.isNull(), ...).\n"
    "- Use .select() and .withColumn() over .map()/.rdd.\n"
    "- Avoid collect() or toPandas() on large datasets.\n"
    "- Add proper column type casting (e.g., .cast('timestamp')).\n"
    "- Use partitionBy on writes for efficient downstream reads.\n"
    "- Write output as Parquet with snappy compression unless task specifies otherwise.\n"
    "- Include a main() entry point that accepts input/output S3 paths as arguments.\n"
    "- Add logging with Python's logging module, not print statements.\n"
    "- If unsure about PySpark API details, use the browser tool to look up official docs.\n"
)

_FIX_PROMPT = """\
You are a data engineering debugging expert. You have access to execute_code \
(Code Interpreter tool) and browser tools.

The files are already in /tmp/transformations/ in the MicroVM. \
Use execute_code to read and fix them:

To read: execute_code with: print(open('/tmp/transformations/transform.py').read())
To fix:  execute_code with: open('/tmp/transformations/transform.py', 'w').write(new_code)

After fixing, respond with the corrected file content in fenced code blocks:

```python:transform.py
<corrected transform.py content>
```

```python:__init__.py
<corrected __init__.py content>
```

You will receive the validation failure output. Your job:
1. Read the broken file(s) using execute_code
2. Analyze the error and identify the root cause
3. Fix ONLY the broken file(s) using execute_code
4. Respond with the corrected content in the fenced blocks above

Common issues: missing main() entry point, syntax errors, missing pyspark imports.
"""


def _invoke_agent(agent, user_message: str, thread_id: str | None = None,
                  heartbeat: tuple[str, str] | None = None) -> str:
    """Invoke a pre-built agent and return the final AI message text.

    Schedules work on the sandbox background loop via run_coroutine_threadsafe
    so that CI tools' asyncio primitives are always accessed from their owning
    loop — no "bound to a different event loop" or "Event loop is closed" errors.

    The agent must be built once and reused across calls so that the Code
    Interpreter toolkit's internal session cache (keyed on thread_id + tool
    instance id) hits on every call rather than spinning up a new MicroVM
    session each time.
    """
    import asyncio
    from src.sandbox import _code_interpreter_cache

    invoke_config = None
    if thread_id:
        invoke_config = {"configurable": {"thread_id": thread_id}}

    async def _ainvoke():
        if invoke_config:
            return await agent.ainvoke(
                {"messages": [HumanMessage(content=user_message)]},
                config=invoke_config,
            )
        return await agent.ainvoke({"messages": [HumanMessage(content=user_message)]})

    # Ensure a long-lived background loop exists for scheduling work.
    # We use the sandbox background loop if it's already been initialised
    # (by get_code_interpreter_tools()), otherwise create a dedicated one.
    #
    # IMPORTANT: We always use the background loop — never asyncio.run().
    # asyncio.run() creates a new event loop, runs the coroutine, then closes
    # the loop. When httpx cleans up TLS connections after the coroutine
    # completes, it calls loop.call_soon() on the now-closed loop, raising
    # "RuntimeError: Event loop is closed". The background loop stays open
    # for the lifetime of the process, so httpx cleanup always succeeds.
    if "loop" not in _code_interpreter_cache:
        import threading
        _ready = threading.Event()
        _bg_loop = asyncio.new_event_loop()

        def _run_bg():
            asyncio.set_event_loop(_bg_loop)
            _bg_loop.call_soon_threadsafe(_ready.set)
            _bg_loop.run_forever()

        threading.Thread(target=_run_bg, daemon=True).start()
        _ready.wait(timeout=5)  # wait until run_forever() is actually running
        _code_interpreter_cache["loop"] = _bg_loop

    bg_loop = _code_interpreter_cache["loop"]
    if heartbeat is not None:
        # Stream the agent so its internal tool calls (sample data, write_file,
        # execute PySpark checks) reach the dashboard in realtime rather than a
        # single "still working" tick per phase.
        from src.graphs.progress import run_agent_with_live_progress
        result = run_agent_with_live_progress(
            agent, user_message, bg_loop, heartbeat[0], heartbeat[1],
            invoke_config=invoke_config,
        )
    else:
        fut = asyncio.run_coroutine_threadsafe(_ainvoke(), bg_loop)
        result = fut.result()

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            from src.graphs._utils import _content_to_str
            return _content_to_str(msg.content)
    return ""


def _build_agent(model, tools: list, system_prompt: str):
    """Build a single DeepAgent instance with checkpointing disabled.

    Agents must be built once per pipeline run and reused across node
    invocations. Building a new agent per call creates a new toolkit
    instance with a new internal UUID, which defeats the Code Interpreter
    toolkit's session cache (keyed on thread_id + tool instance id) and
    causes a fresh MicroVM session to be started on every LLM call.
    """
    from deepagents import create_deep_agent
    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        checkpointer=False,
    )


def _sample_data(state: DataEngState) -> dict[str, Any]:
    """Sample data schema directly from S3 using boto3 + pandas in-process.

    Runs in the container where IAM role credentials are available.
    Neither the LocalShellBackend nor the Code Interpreter MicroVM have
    S3 credentials, so we sample directly using the container's execution
    role instead of delegating to a DeepAgent.
    Falls back gracefully if S3 access fails.
    """
    from src.graphs.progress import emit_progress

    task = get_task_description(state)
    s3_match = re.search(r"s3://\S+", task)
    if not s3_match:
        emit_progress("data-eng-agent", "[sample] No S3 URI found in the task; skipping data sampling.")
        logger.info("No S3 URI found, skipping sampling.")
        return {"inferred_schema": {}, "data_sample_status": "skipped"}

    import boto3
    import pandas as pd

    s3_uri = s3_match.group(0).rstrip("/")

    # Parse bucket and key from s3://bucket/key
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    if not key:
        emit_progress("data-eng-agent", f"[sample] S3 URI {s3_uri} has no object key; skipping data sampling.")
        logger.warning("S3 URI %s has no object key, skipping sampling", s3_uri)
        return {"inferred_schema": {}, "data_sample_status": "skipped"}

    # Auto-detect format from task description
    task_lower = task.lower()
    if "csv" in task_lower:
        fmt = "csv"
    elif "json" in task_lower:
        fmt = "json"
    else:
        fmt = "parquet"

    emit_progress(
        "data-eng-agent",
        f"[sample] Sampling up to {DEFAULT_SAMPLE_SIZE} rows from {s3_uri} "
        f"(format: {fmt})...",
    )

    try:
        with traced_span("agent:data_eng.sample_data", {
            "agent.graph": "data_eng",
            "agent.node": "sample_data",
            "sample.s3_uri": s3_uri,
            "sample.format": fmt,
            "sample.size": DEFAULT_SAMPLE_SIZE,
        }):
            region = os.environ.get("AWS_REGION", "us-west-2")
            s3 = boto3.client("s3", region_name=region)

            # If the key looks like a prefix (no file extension), list objects
            # and pick the first matching file to sample from.
            _, ext = os.path.splitext(key)
            if not ext:
                resp = s3.list_objects_v2(Bucket=bucket, Prefix=key.rstrip("/") + "/", MaxKeys=10)
                contents = resp.get("Contents", [])
                # Filter to files with a recognized extension
                fmt_ext_map = {"csv": ".csv", "json": ".json", "parquet": ".parquet"}
                target_ext = fmt_ext_map.get(fmt, "")
                candidates = [
                    obj_meta["Key"] for obj_meta in contents
                    if not obj_meta["Key"].endswith("/")
                    and (not target_ext or obj_meta["Key"].endswith(target_ext))
                ]
                if not candidates:
                    emit_progress(
                        "data-eng-agent",
                        f"[sample] No sampleable files found under s3://{bucket}/{key}; skipping.",
                    )
                    logger.warning(
                        "No sampleable files found under s3://%s/%s, skipping",
                        bucket, key,
                    )
                    return {"inferred_schema": {}, "data_sample_status": "skipped"}
                key = candidates[0]
                logger.info("Resolved prefix to object: s3://%s/%s", bucket, key)

            obj = s3.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()

            if fmt == "csv":
                df = pd.read_csv(io.BytesIO(body), nrows=DEFAULT_SAMPLE_SIZE)
            elif fmt == "json":
                df = pd.read_json(io.BytesIO(body), lines=True, nrows=DEFAULT_SAMPLE_SIZE)
            else:
                df = pd.read_parquet(io.BytesIO(body)).head(DEFAULT_SAMPLE_SIZE)

            schema = []
            for col in df.columns:
                schema.append({
                    "name": col,
                    "type": str(df[col].dtype),
                    "nullable": bool(df[col].isnull().any()),
                })

            emit_progress(
                "data-eng-agent",
                f"[sample] Inferred schema: {len(schema)} column(s) from "
                f"{len(df)} sampled row(s).",
            )
            return {
                "inferred_schema": {
                    "columns": schema,
                    "row_count": len(df),
                },
                "data_sample_status": "success",
            }
    except Exception as exc:
        emit_progress("data-eng-agent", f"[sample] Data sampling failed: {exc}")
        logger.error("Data sampling failed: %s", exc)
        return {"inferred_schema": {}, "data_sample_status": "failed"}


def _parse_code_blocks(response: str) -> dict[str, str]:
    """Extract named fenced code blocks from the generate/fix response.

    Handles all fence formats the LLM may produce:
      ```python:transform.py   <- named with language prefix (most common)
      ```transform.py          <- named without language prefix
      ```python                <- plain python block (fallback)
      ```                      <- bare block (last resort)

    Returns a dict of {filename: content}.
    Falls back to positional assignment (first block = transform.py,
    second = __init__.py) when no named blocks are found.
    """
    import re as _re
    content: dict[str, str] = {}

    # Pass 1: named blocks with optional language prefix
    # Matches: ```python:transform.py  OR  ```transform.py
    named = _re.findall(
        r"```(?:[a-zA-Z]+:)?([\w./]+\.py)\s*\n(.*?)```",
        response,
        _re.DOTALL,
    )
    for fname, code in named:
        # Basename: the model sometimes writes path-qualified names
        # (```python:tests/test_transform.py) while every downstream consumer
        # keys by the bare filename.
        content[fname.rsplit("/", 1)[-1]] = code.strip()

    if "transform.py" in content:
        return content

    # Pass 2: plain ```python or ``` blocks — assign positionally
    plain = _re.findall(r"```(?:python)?\s*\n(.*?)```", response, _re.DOTALL)
    if plain:
        content["transform.py"] = plain[0].strip()
    if len(plain) > 1:
        content["__init__.py"] = plain[1].strip()

    return content


def _generate(state: DataEngState, agent, tool_count: int = 0) -> dict[str, Any]:
    """Agent node: pure LLM code generation — no Code Interpreter needed.

    The generated code is stored in state['code_content'] as a dict of
    {filename: content}. The validate node reads this and writes the files
    into the MicroVM via execute_code before running the structural check.

    Accepts a pre-built agent so the same toolkit instance is reused across
    calls — critical for Code Interpreter session reuse.
    """
    from src.graphs.progress import emit_progress

    task = get_task_description(state)
    inferred_schema = state.get("inferred_schema", {})

    if inferred_schema and inferred_schema.get("columns"):
        emit_progress(
            "data-eng-agent",
            f"[generate] Generating PySpark transformation code using the "
            f"inferred schema ({len(inferred_schema['columns'])} columns)...",
        )
    else:
        emit_progress(
            "data-eng-agent",
            "[generate] Generating PySpark transformation code from the task "
            "description (no data sample available)...",
        )

    user_msg = f"## Task\n{task}"

    if inferred_schema and inferred_schema.get("columns"):
        schema_lines = [
            f"- {col['name']} ({col['type']}, nullable={col['nullable']})"
            for col in inferred_schema["columns"]
        ]
        user_msg += (
            f"\n\n## Inferred Schema (sampled {inferred_schema.get('row_count', '?')} rows)\n"
            + "\n".join(schema_lines)
            + "\n\nUse this schema for input column references, data types, and null-handling logic."
        )
    else:
        user_msg += (
            "\n\n## Note\n"
            "No data sample was available. Generate transformation code based solely "
            "on the transformations section from the task description."
        )

    with traced_span("agent:data_eng.generate", {
        "agent.graph": "data_eng",
        "agent.node": "generate",
        "agent.role": "code_generator",
        "agent.has_schema": bool(inferred_schema and inferred_schema.get("columns")),
        "agent.tool_count": tool_count,
    }):
        # Generate uses only browser tools — no Code Interpreter needed.
        # The code is returned as text and stored in state for the validate node.
        response = _invoke_agent(agent, user_msg, heartbeat=("data-eng-agent", "generate"))

    code_content = _parse_code_blocks(response)
    emit_progress(
        "data-eng-agent",
        f"[generate] Generated {len(code_content)} file(s): "
        f"{', '.join(code_content) or 'none'}",
    )
    if not code_content.get("transform.py"):
        logger.warning(
            "generate: _parse_code_blocks found no transform.py in response "
            "(response length=%d). Raw response prefix: %.200s",
            len(response),
            response,
        )
    code_artifacts = {
        "transform.py": "/tmp/transformations/transform.py",
        "__init__.py": "/tmp/transformations/__init__.py",
    }
    return {
        "code_artifacts": code_artifacts,
        "code_content": code_content,
        "messages": [AIMessage(content=response)],
    }


_VALIDATE_PROMPT = """\
You are a code structure validator. You have access to execute_code \
(Code Interpreter tool) which runs Python in an isolated MicroVM sandbox.

CRITICAL: Each execute_code call runs in a completely isolated session. \
Files written in one call are NOT visible in the next call. \
You MUST run the write step and the validation step in a SINGLE execute_code call.

You will receive a single Python code block. Run it as ONE execute_code call. \
Do not split it into multiple calls.

Report the full output exactly as printed. State clearly whether validation PASSED or FAILED.
"""


def _validate(state: DataEngState) -> dict[str, Any]:
    """Validate generated code structure directly in Python — no LLM needed.

    The generate node stores code as text in state['code_content'].
    We run the AST structural check directly here without involving the
    Code Interpreter MicroVM or an LLM agent. This avoids the fundamental
    problem of execute_code sessions being isolated per-call (files written
    in one call are not visible in the next).

    The Code Interpreter is only needed for fix attempts that require
    actually running the code — structural validation (AST parse, required
    functions, imports) is pure Python and needs no sandbox.
    """
    from src.graphs.progress import emit_progress
    emit_progress(
        "data-eng-agent",
        "[validate] Running structural checks on the generated code "
        "(syntax, main() entry point, PySpark imports)...",
    )

    code_content = state.get("code_content", {})
    artifacts = state.get("code_artifacts", {})

    transform_code = code_content.get("transform.py", "")
    init_code = code_content.get("__init__.py", "")

    errors = []
    func_names = []

    with traced_span("agent:data_eng.validate", {
        "agent.graph": "data_eng",
        "agent.node": "validate",
        "agent.artifact_count": len(artifacts),
        "agent.has_code_content": bool(code_content),
    }):
        # Check transform.py
        if not transform_code:
            errors.append("MISSING: transform.py (no code generated)")
        else:
            import ast as _ast
            try:
                tree = _ast.parse(transform_code)
                func_names = [
                    n.name for n in _ast.walk(tree)
                    if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                ]
                if not func_names:
                    errors.append("NO FUNCTIONS found in transform.py")
                if "main" not in func_names:
                    errors.append("MISSING main() entry point in transform.py")
                imports = [
                    n for n in _ast.walk(tree)
                    if isinstance(n, (_ast.Import, _ast.ImportFrom))
                ]
                has_pyspark = any(
                    (isinstance(n, _ast.ImportFrom) and n.module and "pyspark" in n.module)
                    or (isinstance(n, _ast.Import) and any("pyspark" in a.name for a in n.names))
                    for n in imports
                )
                if not has_pyspark:
                    errors.append("NO pyspark imports found - expected PySpark DataFrame code")
            except SyntaxError as e:
                errors.append(f"SYNTAX ERROR in transform.py: {e}")

        # Check __init__.py (just needs to be parseable — can be empty)
        if init_code is None:
            errors.append("MISSING: __init__.py (no code generated)")
        else:
            import ast as _ast
            try:
                _ast.parse(init_code)
            except SyntaxError as e:
                errors.append(f"SYNTAX ERROR in __init__.py: {e}")

    if errors:
        output = "VALIDATION: FAILED\n" + "\n".join(f"  - {e}" for e in errors)
        passed = False
    else:
        output = (
            "VALIDATION: PASSED\n"
            f"  - Functions found: {', '.join(func_names)}\n"
            "  - All required files present and syntactically valid"
        )
        passed = True
    emit_progress("data-eng-agent", f"[validate] Structural validation result:\n{output}")

    return {
        "validation_passed": passed,
        "validation_output": output,
    }


_FIX_FALLBACK_PROMPT = """\
You are a data engineering debugging expert. You do NOT have access to a live \
Code Interpreter session right now, so you must fix the code purely from the \
source provided below.

You will receive:
- The original task description
- The current (broken) file contents
- The validation error output

Your job:
1. Analyze the error and identify the root cause
2. Produce corrected file content in fenced code blocks:

```python:transform.py
<corrected transform.py content>
```

```python:__init__.py
<corrected __init__.py content>
```

Common issues: missing main() entry point, syntax errors, missing pyspark imports.
"""


def _fix(state: DataEngState, model) -> dict[str, Any]:
    """Fix validation failures using a single stateless LLM call.

    Since _validate is pure Python AST (no Code Interpreter), _fix doesn't
    need a DeepAgent or tool loop — it just needs one LLM call with the
    broken code and the error. Using the model directly avoids:
    - DeepAgent internal message history accumulating across fix attempts
    - Growing context from repeated tool-use turns
    - The 60k+ token blowup seen when fix_agent accumulates prior messages
    """
    import asyncio
    from src.sandbox import _code_interpreter_cache
    from langchain_core.messages import SystemMessage

    from src.graphs.progress import emit_progress
    attempt = state.get("attempt", 0) + 1
    emit_progress("data-eng-agent", f"[fix] Initiating self-healing loop (attempt {attempt})...")
    error_output = state.get("validation_output", "")
    task = get_task_description(state)
    inferred_schema = state.get("inferred_schema", {})
    code_content = state.get("code_content", {})

    schema_lines = [
        f"- {col['name']} ({col['type']}, nullable={col['nullable']})"
        for col in inferred_schema.get("columns", [])
    ] if inferred_schema else []

    # Build a focused, minimal prompt — no accumulated history
    user_msg = f"## Original Task\n{task}\n\n"
    if schema_lines:
        user_msg += "## Inferred Schema\n" + "\n".join(schema_lines) + "\n\n"
    user_msg += f"## Validation Error (attempt {attempt})\n{error_output}\n\n"
    user_msg += "## Current File Contents\n"
    for fname, content in code_content.items():
        if content:
            user_msg += f"\n### {fname}\n```python\n{content}\n```\n"
        else:
            user_msg += f"\n### {fname}\n(empty — needs to be generated)\n"
    user_msg += (
        "\nFix the code above so it passes all validation checks. "
        "Respond with the corrected content in fenced code blocks as instructed."
    )

    messages = [
        SystemMessage(content=_FIX_FALLBACK_PROMPT),
        HumanMessage(content=user_msg),
    ]

    with traced_span("agent:data_eng.fix", {
        "agent.graph": "data_eng",
        "agent.node": "fix",
        "agent.role": "debugger",
        "agent.attempt": attempt,
    }):
        # Single stateless LLM call — no agent, no tool loop, no history accumulation
        async def _ainvoke():
            return await model.ainvoke(messages)

        # Use the shared background loop (same pattern as _invoke_agent)
        if "loop" not in _code_interpreter_cache:
            import threading
            _ready = threading.Event()
            _bg_loop = asyncio.new_event_loop()

            def _run_bg():
                asyncio.set_event_loop(_bg_loop)
                _bg_loop.call_soon_threadsafe(_ready.set)
                _bg_loop.run_forever()

            threading.Thread(target=_run_bg, daemon=True).start()
            _ready.wait(timeout=5)
            _code_interpreter_cache["loop"] = _bg_loop

        bg_loop = _code_interpreter_cache["loop"]
        ai_msg = asyncio.run_coroutine_threadsafe(_ainvoke(), bg_loop).result()

        from src.graphs._utils import _content_to_str
        response = _content_to_str(ai_msg.content) if ai_msg.content else ""

    # Parse corrected code blocks and merge into state
    fixed_content = _parse_code_blocks(response)
    updated_content = {**code_content, **fixed_content}

    return {
        "attempt": attempt,
        "code_content": updated_content,
        "messages": [AIMessage(content=response)],
    }


def _report(state: DataEngState) -> dict[str, Any]:
    """Pure function: format final report and set messages for CompiledSubAgent return."""
    import json as _json
    passed = state.get("validation_passed", False)
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    validation_output = state.get("validation_output", "")
    artifacts = state.get("code_artifacts", {})
    code_content = state.get("code_content", {})

    files_list = ", ".join(artifacts.keys()) if artifacts else "none"

    # Human-readable fenced blocks for context
    file_contents = ""
    for fname, content in code_content.items():
        if content:
            file_contents += f"\n\n### {fname}\n```python\n{content}\n```"

    if passed:
        status_line = "VALIDATION: PASSED\nStructural validation completed successfully."
    else:
        status_line = f"VALIDATION: FAILED ({attempt}/{max_attempts} attempts exhausted)\n\nLAST ERROR:\n{validation_output}"

    # Structured JSON block at the end — the orchestrator extracts files from
    # this block directly instead of parsing fenced Python blocks.
    pr_files = {
        f"src/transformations/{fname}": content
        for fname, content in code_content.items()
        if content
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


def _should_retry_or_report(state: DataEngState) -> str:
    """Conditional edge: decide whether to fix, or go to report."""
    if state.get("validation_passed", False):
        return "report"
    if state.get("attempt", 0) < state.get("max_attempts", 3):
        return "fix"
    return "report"


def build_data_eng_graph(model, tools=None):
    """Factory: build and compile the Data Engineering SubAgent StateGraph.

    Args:
        model: A ChatAnthropic (or compatible) LLM instance.
        tools: Optional list of browser tools for on-demand documentation lookup.

    Returns:
        A compiled LangGraph StateGraph ready for invocation as a CompiledSubAgent.
    """
    browser_tools = tools or []

    # validate uses pure Python AST — no Code Interpreter or agent needed.
    # fix uses a single stateless model.ainvoke() call — no DeepAgent, no history
    # accumulation, no token blowup across fix attempts.
    generate_agent = _build_agent(model, browser_tools, _GENERATE_PROMPT)

    graph = StateGraph(DataEngState)

    def sample_data(state: DataEngState) -> dict[str, Any]:
        return _sample_data(state)

    def generate(state: DataEngState) -> dict[str, Any]:
        return _generate(state, generate_agent, tool_count=len(browser_tools))

    def validate(state: DataEngState) -> dict[str, Any]:
        return _validate(state)

    def fix(state: DataEngState) -> dict[str, Any]:
        return _fix(state, model)

    def report(state: DataEngState) -> dict[str, Any]:
        return _report(state)

    # Add nodes
    graph.add_node("sample_data", sample_data)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("fix", fix)
    graph.add_node("report", report)

    # Wire edges: sample_data → generate → validate → conditional → fix/report
    graph.set_entry_point("sample_data")
    graph.add_edge("sample_data", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        _should_retry_or_report,
        {"fix": "fix", "report": "report"},
    )
    graph.add_edge("fix", "validate")
    graph.add_edge("report", END)

    instrument_graph(graph, "data_eng", {
        "sample_data": "agent",
        "generate": "agent",
        "validate": "agent",
        "fix": "agent",
        "report": "function",
    })

    return graph.compile()
