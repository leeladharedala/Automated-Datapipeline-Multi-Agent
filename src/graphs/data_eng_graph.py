"""Data Engineering SubAgent Graph — transformation code generation with pytest validation.

Builds a compiled LangGraph StateGraph that:
1. Samples data schema via a mini DeepAgent with execute (agent node)
2. Generates data transformation code via a mini DeepAgent with write_file + browser tools (agent node)
3. Validates with pytest via a mini DeepAgent with execute (agent node)
4. Self-heals on validation failure via a mini DeepAgent with edit_file + browser tools (agent node)
5. Produces a final pass/fail report (pure function)

No research node — DataEng generation starts directly from the task description.
The generate and fix agent nodes have browser tools bound for on-demand framework
documentation lookup (PySpark, Pandas, dbt).
Agent nodes use create_deep_agent to get VFS (write_file, edit_file, read_file)
and sandbox execution (execute) — artifacts are persisted to AgentCore short-term
memory via the shared VFS.
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

The files are already in /src/transformations/ in the MicroVM. \
Use execute_code to read and fix them:

To read: execute_code with: print(open('/src/transformations/transform.py').read())
To fix:  execute_code with: open('/src/transformations/transform.py', 'w').write(new_code)

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


def _run_agent(system_prompt: str, user_message: str, model, tools=None, backend=None,
               ci_tools=None, thread_id: str | None = None) -> str:
    """Create a mini DeepAgent, invoke it, and return the final AI message text.

    Extra tools (e.g. browser tools) are passed on top of the built-in
    DeepAgent tool stack (write_file, edit_file, read_file, execute, etc.).
    Pass a backend (e.g. AgentCoreSandbox) to enable the execute tool.

    Uses ainvoke (async) to support tools that only implement async invocation
    (e.g. Code Interpreter StructuredTools from langchain-aws).

    Always schedules work on the sandbox background loop via
    run_coroutine_threadsafe so that:
    1. CI tools' asyncio primitives (bound to that loop) are accessed from
       their owning loop — no "bound to a different event loop" errors.
    2. The loop is never closed between calls — no "Event loop is closed"
       errors when httpx tries to clean up TLS connections after asyncio.run().

    Pass thread_id to reuse the same Code Interpreter session across all
    validate/fix calls within a single pipeline run, avoiding the overhead
    of starting a new MicroVM session per invocation.
    """
    import asyncio
    from deepagents import create_deep_agent
    from src.sandbox import _code_interpreter_cache

    kwargs = dict(
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],
    )
    if backend is not None:
        kwargs["backend"] = backend
    kwargs["checkpointer"] = False  # disable checkpointing for inner agents

    agent = create_deep_agent(**kwargs)

    # Build a stable config so the Code Interpreter toolkit reuses the same
    # MicroVM session for all validate/fix calls within a pipeline run.
    # Without this, each ainvoke() gets a fresh auto-generated thread_id and
    # the toolkit starts a new session every time.
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
    # (by get_code_interpreter_tools()), otherwise create a dedicated one
    # for this graph module so we never fall back to asyncio.run().
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
    # Always schedule on the background loop — never use asyncio.run().
    result = asyncio.run_coroutine_threadsafe(_ainvoke(), bg_loop).result()

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            from src.graphs._utils import _content_to_str
            return _content_to_str(msg.content)
    return ""


_SAMPLE_DATA_PROMPT = """\
You are a data sampling specialist. You have access to the execute tool \
which runs commands in the AgentCore Runtime sandbox.

Given the S3 URI and format below, sample up to {sample_size} records and \
infer the schema. Run a PySpark script using the execute tool.

IMPORTANT: Respond with ONLY the JSON output from the script, nothing else. \
Do not add any commentary before or after the JSON.
"""


def _sample_data(state: DataEngState, model, backend=None, ci_tools=None) -> dict[str, Any]:
    """Sample data schema directly from S3 using boto3 + pandas in-process.

    Runs in the container where IAM role credentials are available.
    Neither the LocalShellBackend nor the Code Interpreter MicroVM have
    S3 credentials, so we sample directly using the container's execution
    role instead of delegating to a DeepAgent.
    Falls back gracefully if S3 access fails.
    """
    task = get_task_description(state)
    s3_match = re.search(r"s3://\S+", task)
    if not s3_match:
        return {"inferred_schema": {}, "data_sample_status": "skipped"}

    import io
    import os

    import boto3
    import pandas as pd

    s3_uri = s3_match.group(0).rstrip("/")

    # Parse bucket and key from s3://bucket/key
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    if not key:
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

            return {
                "inferred_schema": {
                    "columns": schema,
                    "row_count": len(df),
                },
                "data_sample_status": "success",
            }
    except Exception as exc:
        logger.error("Data sampling failed: %s", exc)
        return {"inferred_schema": {}, "data_sample_status": "failed"}


def _parse_code_blocks(response: str) -> dict[str, str]:
    """Extract named fenced code blocks from the generate response.

    Looks for blocks like:
      ```python:transform.py
      <content>
      ```
    Returns a dict of {filename: content}.
    Falls back to extracting the first two plain ```python blocks if
    named blocks are not found.
    """
    import re as _re
    content: dict[str, str] = {}

    # Try named blocks first: ```python:filename.py
    named = _re.findall(r"```(?:python:)?([\w./]+\.py)\n(.*?)```", response, _re.DOTALL)
    for fname, code in named:
        # Strip the leading "python:" prefix if present in the filename match
        fname = fname.lstrip("python:")
        content[fname] = code.strip()

    if "transform.py" in content:
        return content

    # Fallback: grab all plain ```python blocks in order
    plain = _re.findall(r"```(?:python)?\n(.*?)```", response, _re.DOTALL)
    if plain:
        content["transform.py"] = plain[0].strip()
    if len(plain) > 1:
        content["__init__.py"] = plain[1].strip()

    return content


def _generate(state: DataEngState, model, tools: list,
              ci_tools=None, thread_id: str | None = None) -> dict[str, Any]:
    """Agent node: pure LLM code generation — no Code Interpreter needed.

    The generated code is stored in state['code_content'] as a dict of
    {filename: content}. The validate node reads this and writes the files
    into the MicroVM via execute_code before running the structural check.
    """
    task = get_task_description(state)
    inferred_schema = state.get("inferred_schema", {})

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
        "agent.tool_count": len(tools),
    }):
        # Generate uses only browser tools — no Code Interpreter needed.
        # The code is returned as text and stored in state for the validate node.
        response = _run_agent(_GENERATE_PROMPT, user_msg, model, tools=tools)

    code_content = _parse_code_blocks(response)
    code_artifacts = {
        "transform.py": "/src/transformations/transform.py",
        "__init__.py": "/src/transformations/__init__.py",
    }
    return {
        "code_artifacts": code_artifacts,
        "code_content": code_content,
        "messages": [AIMessage(content=response)],
    }


_VALIDATE_PROMPT = """\
You are a code structure validator. You have access to execute_code \
(Code Interpreter tool) which runs Python in an isolated MicroVM sandbox.

The generated code has already been written to /src/transformations/ in the \
MicroVM. Run the following validation script using execute_code:

```python
import ast, sys, os

errors = []
func_names = []

required = ["/src/transformations/transform.py", "/src/transformations/__init__.py"]
for f in required:
    if not os.path.exists(f):
        errors.append(f"MISSING: {f}")

tf = "/src/transformations/transform.py"
if os.path.exists(tf):
    source = open(tf).read()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        errors.append(f"SYNTAX ERROR in transform.py: {e}")
        tree = None
    if tree:
        func_names = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not func_names:
            errors.append("NO FUNCTIONS found in transform.py")
        if "main" not in func_names:
            errors.append("MISSING main() entry point in transform.py")
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        has_pyspark = any(
            (isinstance(n, ast.ImportFrom) and n.module and "pyspark" in n.module)
            or (isinstance(n, ast.Import) and any("pyspark" in a.name for a in n.names))
            for n in imports
        )
        if not has_pyspark:
            errors.append("NO pyspark imports found - expected PySpark DataFrame code")

init = "/src/transformations/__init__.py"
if os.path.exists(init):
    try:
        ast.parse(open(init).read())
    except SyntaxError as e:
        errors.append(f"SYNTAX ERROR in __init__.py: {e}")

if errors:
    print("VALIDATION: FAILED")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("VALIDATION: PASSED")
    print(f"  - Functions found: {', '.join(func_names)}")
    print("  - All required files present and syntactically valid")
```

Report the full output. State clearly whether validation PASSED or FAILED.
"""


def _validate(state: DataEngState, model, ci_tools=None, thread_id: str | None = None) -> dict[str, Any]:
    """Agent node: write generated code into MicroVM then run structural validation.

    The generate node stores code as text in state['code_content'].
    We write those files into the MicroVM via execute_code before running
    the validation script — this bridges the filesystem gap between the
    generate node (pure LLM, no Code Interpreter) and the MicroVM.
    """
    code_content = state.get("code_content", {})
    artifacts = state.get("code_artifacts", {})

    # Build a Python snippet that writes the generated files into the MicroVM.
    # This runs as the first execute_code call so the validation script finds them.
    transform_code = code_content.get("transform.py", "")
    init_code = code_content.get("__init__.py", "")

    # Escape backslashes and triple-quotes so the content embeds safely
    def _escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    setup_code = (
        "import os\n"
        "os.makedirs('/src/transformations', exist_ok=True)\n"
        f'with open("/src/transformations/transform.py", "w") as f:\n'
        f'    f.write("""{_escape(transform_code)}""")\n'
        f'with open("/src/transformations/__init__.py", "w") as f:\n'
        f'    f.write("""{_escape(init_code)}""")\n'
        "print('Files written to /src/transformations/')\n"
        "print(os.listdir('/src/transformations/'))\n"
    )

    user_msg = (
        f"## Step 1: Write generated files into the MicroVM\n"
        f"Run this exact Python code using execute_code:\n```python\n{setup_code}\n```\n\n"
        f"## Step 2: Run the validation script\n"
        f"After the files are written, run the validation script from the prompt."
    )

    try:
        with traced_span("agent:data_eng.validate", {
            "agent.graph": "data_eng",
            "agent.node": "validate",
            "agent.artifact_count": len(artifacts),
            "agent.has_code_content": bool(code_content),
        }):
            response = _run_agent(_VALIDATE_PROMPT, user_msg, model, tools=ci_tools,
                                  ci_tools=ci_tools, thread_id=thread_id)
    except Exception as exc:
        response = str(exc)

    output = response
    upper = output.upper()
    if "VALIDATION FAILED" in upper or "VALIDATION: FAILED" in upper:
        passed = False
    elif "VALIDATION PASSED" in upper or "VALIDATION: PASSED" in upper:
        passed = True
    else:
        output_lower = output.lower()
        has_passed = "passed" in output_lower
        has_failed = "failed" in output_lower or "error" in output_lower
        passed = has_passed and not has_failed

    return {
        "validation_passed": passed,
        "validation_output": output,
    }


def _fix(state: DataEngState, model, tools: list, ci_tools=None, thread_id: str | None = None) -> dict[str, Any]:
    """Agent node: fix pytest failures using Code Interpreter + browser tools."""
    attempt = state.get("attempt", 0) + 1
    error_output = state.get("validation_output", "")
    task = get_task_description(state)
    inferred_schema = state.get("inferred_schema", {})

    user_msg = f"## Original Task\n{task}\n\n"
    if inferred_schema and inferred_schema.get("columns"):
        schema_lines = [
            f"- {col['name']} ({col['type']}, nullable={col['nullable']})"
            for col in inferred_schema["columns"]
        ]
        user_msg += "## Inferred Schema\n" + "\n".join(schema_lines) + "\n\n"
    user_msg += (
        f"## Pytest Failure (attempt {attempt})\n{error_output}\n\n"
        "Fix the broken transformation files in /src/transformations/ using execute_code, "
        "then respond with the corrected content in fenced code blocks as instructed."
    )
    with traced_span("agent:data_eng.fix", {
        "agent.graph": "data_eng",
        "agent.node": "fix",
        "agent.role": "debugger",
        "agent.attempt": attempt,
        "agent.tool_count": len(tools),
    }):
        response = _run_agent(_FIX_PROMPT, user_msg, model, tools=tools, ci_tools=ci_tools,
                              thread_id=thread_id)

    # Parse corrected code blocks from the fix response and update code_content in state
    fixed_content = _parse_code_blocks(response)
    current_content = state.get("code_content", {})
    updated_content = {**current_content, **fixed_content}

    return {
        "attempt": attempt,
        "code_content": updated_content,
        "messages": [AIMessage(content=response)],
    }


def _report(state: DataEngState) -> dict[str, Any]:
    """Pure function: format final report and set messages for CompiledSubAgent return."""
    passed = state.get("validation_passed", False)
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    validation_output = state.get("validation_output", "")
    artifacts = state.get("code_artifacts", {})
    code_content = state.get("code_content", {})

    files_list = ", ".join(artifacts.keys()) if artifacts else "none"

    # Include the actual generated code so the orchestrator can pass it back
    # on followup/re-dispatch requests without needing to regenerate from scratch.
    file_contents = ""
    for fname, content in code_content.items():
        if content:
            file_contents += f"\n\n### {fname}\n```python\n{content}\n```"

    if passed:
        report = (
            "VALIDATION: PASSED\n"
            "pytest completed successfully.\n"
            f"Files generated: {files_list}\n"
            f"Attempts used: {attempt}"
            f"{file_contents}"
        )
    else:
        report = (
            f"VALIDATION: FAILED ({attempt}/{max_attempts} attempts exhausted)\n\n"
            f"LAST ERROR:\n{validation_output}\n\n"
            f"Files generated: {files_list}"
            f"{file_contents}"
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
    import uuid
    browser_tools = tools or []

    # Load Code Interpreter tools for validate/fix nodes (execute_code,
    # install_packages, write_files, etc.)
    from src.sandbox import get_code_interpreter_tools, get_local_shell_backend
    ci_tools = get_code_interpreter_tools()
    sandbox = get_local_shell_backend()

    # Stable thread_id for this pipeline run — ensures all validate/fix calls
    # reuse the same Code Interpreter MicroVM session instead of starting a
    # new one per invocation (which adds ~1s latency each time).
    ci_thread_id = f"data-eng-{uuid.uuid4()}"

    graph = StateGraph(DataEngState)

    def sample_data(state: DataEngState) -> dict[str, Any]:
        return _sample_data(state, model)

    def generate(state: DataEngState) -> dict[str, Any]:
        # Generate is pure LLM + browser tools — no Code Interpreter needed.
        # Code content is stored in state and written to MicroVM by validate.
        return _generate(state, model, browser_tools)

    def validate(state: DataEngState) -> dict[str, Any]:
        return _validate(state, model, ci_tools=ci_tools, thread_id=ci_thread_id)

    def fix(state: DataEngState) -> dict[str, Any]:
        return _fix(state, model, browser_tools + ci_tools, ci_tools=ci_tools,
                    thread_id=ci_thread_id)

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
