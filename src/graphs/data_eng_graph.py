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

_GENERATE_PROMPT = """\
You are a data transformation code generator. You have access to write_file, \
read_file tools that write to a shared VFS (persisted by AgentCore Memory), \
and browser tools for looking up framework documentation on-demand.

Given the task description below, generate production-grade PySpark data \
transformation code. Write the following files using write_file:
- /src/transformations/transform.py — Core PySpark DataFrame transformation logic
- /src/transformations/__init__.py — Package init (import the main transform)

TRANSFORMATION CODE rules:
- Use PySpark DataFrames for ALL transformations (read from and write to S3 paths).
- Structure transforms as pure functions: input DataFrame(s) in, output DataFrame out.
- Handle nulls explicitly — use .na.drop(), .na.fill(), or .when(col.isNull(), ...) as appropriate.
- Use .select() and .withColumn() over .map()/.rdd — stay in the DataFrame API for Catalyst optimization.
- Avoid collect() or toPandas() on large datasets — keep everything distributed.
- Add proper column type casting (e.g., .cast("timestamp")) rather than relying on inference.
- Use partitionBy on writes for efficient downstream reads (partition by date, region, etc.).
- Write output as Parquet with snappy compression (default) unless task specifies otherwise.
- Include a main() entry point that accepts input/output S3 paths as arguments for CLI invocation.
- Add logging with Python's logging module, not print statements.
- If unsure about PySpark API details, use the browser tool to look up official docs.

Write ALL files, then respond with a summary of what you wrote.
"""

_FIX_PROMPT = """\
You are a data engineering debugging expert. You have access to read_file, \
edit_file tools that operate on a shared VFS, and browser tools for looking \
up framework documentation on-demand.

You will receive the validation failure output. Your job:
1. Read the broken file(s) using read_file
2. Analyze the error and identify the root cause
3. If unsure about an API or behavior, use the browser tool to look up docs
4. Fix ONLY the broken file(s) using edit_file — do not regenerate everything
5. Respond with a summary of what you fixed and why

Common issues: missing main() entry point, syntax errors, missing pyspark imports, \
missing required files.
"""


def _run_agent(system_prompt: str, user_message: str, model, tools=None, backend=None) -> str:
    """Create a mini DeepAgent, invoke it, and return the final AI message text.

    Extra tools (e.g. browser tools) are passed on top of the built-in
    DeepAgent tool stack (write_file, edit_file, read_file, execute, etc.).
    Pass a backend (e.g. AgentCoreSandbox) to enable the execute tool.

    Uses ainvoke (async) to support tools that only implement async invocation
    (e.g. Code Interpreter StructuredTools from langchain-aws).
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
    kwargs["checkpointer"] = False  # FIX: disable checkpointing for inner agents

    agent = create_deep_agent(**kwargs)

    async def _ainvoke():
        return await agent.ainvoke({"messages": [HumanMessage(content=user_message)]})

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        new_loop = asyncio.new_event_loop()  # FIX: isolated loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(new_loop.run_until_complete, _ainvoke()).result()
        # Do NOT close new_loop — the model's httpx client may hold
        # transport connections bound to this loop.  Closing it would
        # destroy those transports and cause "Event loop is closed" on
        # the next _run_agent() call when httpx tries to reuse them.
    else:
        new_loop = asyncio.new_event_loop()  # FIX: isolated loop
        result = new_loop.run_until_complete(_ainvoke())

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


def _generate(state: DataEngState, model, tools: list) -> dict[str, Any]:
    """Agent node: create_deep_agent with write_file + browser tools to generate code."""
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
        response = _run_agent(_GENERATE_PROMPT, user_msg, model, tools=tools)

    code_artifacts = {
        "transform.py": "/src/transformations/transform.py",
        "__init__.py": "/src/transformations/__init__.py",
    }
    return {"code_artifacts": code_artifacts, "messages": [AIMessage(content=response)]}


_VALIDATE_PROMPT = """\
You are a code structure validator. You have access to the execute tool \
which runs commands in the AgentCore Runtime sandbox.

Validate the generated PySpark transformation code by running a structural \
check script. Do NOT attempt to run PySpark — it is not available in this sandbox.

Run the following validation script using the execute tool:

```python
import ast, sys, os

errors = []
func_names = []

# 1. Check required files exist
required = ["/src/transformations/transform.py", "/src/transformations/__init__.py"]
for f in required:
    if not os.path.exists(f):
        errors.append(f"MISSING: {f}")

# 2. Parse transform.py for syntax and structure
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

# 3. Parse __init__.py for syntax
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


def _validate(state: DataEngState, model, ci_tools=None) -> dict[str, Any]:
    """Agent node: use Code Interpreter tools to run pytest validation.

    Uses execute_code/execute_command from the Code Interpreter toolkit
    to run pytest in an isolated MicroVM sandbox.
    """
    artifacts = state.get("code_artifacts", {})
    files_list = ", ".join(artifacts.values()) if artifacts else "unknown"

    user_msg = (
        f"## Generated Files\n{files_list}\n\n"
        "Run pytest validation now."
    )

    try:
        with traced_span("agent:data_eng.validate", {
            "agent.graph": "data_eng",
            "agent.node": "validate",
            "agent.artifact_count": len(artifacts),
        }):
            response = _run_agent(_VALIDATE_PROMPT, user_msg, model, tools=ci_tools)
    except Exception as exc:
        response = str(exc)

    output = response

    # Check for explicit pytest result markers.
    # pytest output contains "X passed" and/or "X failed" — check both.
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


def _fix(state: DataEngState, model, tools: list) -> dict[str, Any]:
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
        "Fix the broken transformation or test files in /src/transformations/ and /tests/."
    )
    with traced_span("agent:data_eng.fix", {
        "agent.graph": "data_eng",
        "agent.node": "fix",
        "agent.role": "debugger",
        "agent.attempt": attempt,
        "agent.tool_count": len(tools),
    }):
        response = _run_agent(_FIX_PROMPT, user_msg, model, tools=tools)

    return {"attempt": attempt, "messages": [AIMessage(content=response)]}


def _report(state: DataEngState) -> dict[str, Any]:
    """Pure function: format final report and set messages for CompiledSubAgent return."""
    passed = state.get("validation_passed", False)
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    validation_output = state.get("validation_output", "")
    artifacts = state.get("code_artifacts", {})

    files_list = ", ".join(artifacts.keys()) if artifacts else "none"

    if passed:
        report = (
            "VALIDATION: PASSED\n"
            "pytest completed successfully.\n"
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

    # Load Code Interpreter tools for validate/fix nodes (execute_code,
    # install_packages, write_files, etc.)
    from src.sandbox import get_code_interpreter_tools, get_local_shell_backend
    ci_tools = get_code_interpreter_tools()
    sandbox = get_local_shell_backend()

    graph = StateGraph(DataEngState)

    def sample_data(state: DataEngState) -> dict[str, Any]:
        return _sample_data(state, model)

    def generate(state: DataEngState) -> dict[str, Any]:
        return _generate(state, model, browser_tools)

    def validate(state: DataEngState) -> dict[str, Any]:
        return _validate(state, model, ci_tools=ci_tools)

    def fix(state: DataEngState) -> dict[str, Any]:
        return _fix(state, model, browser_tools + ci_tools)

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
