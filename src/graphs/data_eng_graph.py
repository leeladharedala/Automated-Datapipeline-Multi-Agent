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

import json
import logging
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
transformation code with tests. Write the following files using write_file:
- /src/transformations/transform.py — Core PySpark DataFrame transformation logic
- /src/transformations/__init__.py — Package init (import the main transform)
- /tests/test_transform.py — pytest test suite using a local SparkSession

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

TEST SUITE rules:
- Create a local SparkSession fixture with .master("local[2]") for parallelism testing.
- Cover: happy path, empty DataFrame, null handling, schema mismatch, and duplicate rows.
- Use small inline test data (createDataFrame with explicit schema), not external files.
- Assert on both schema (column names + types) and row-level values.
- Use pytest.mark.parametrize for testing multiple transformation scenarios.
- Test that output partitioning is correct when applicable.
- Never hardcode credentials in transformation code or tests.

Write ALL three files, then respond with a summary of what you wrote.
"""

_FIX_PROMPT = """\
You are a data engineering debugging expert. You have access to read_file, \
edit_file tools that operate on a shared VFS, and browser tools for looking \
up framework documentation on-demand.

You will receive the pytest failure output. Your job:
1. Read the broken file(s) using read_file
2. Analyze the error and identify the root cause
3. If unsure about an API or behavior, use the browser tool to look up docs
4. Fix ONLY the broken file(s) using edit_file — do not regenerate everything
5. Respond with a summary of what you fixed and why
"""


def _run_agent(system_prompt: str, user_message: str, model, tools=None) -> str:
    """Create a mini DeepAgent, invoke it, and return the final AI message text.

    Extra tools (e.g. browser tools) are passed on top of the built-in
    DeepAgent tool stack (write_file, edit_file, read_file, execute, etc.).
    """
    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],
    )
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


_SAMPLE_DATA_PROMPT = """\
You are a data sampling specialist. You have access to the execute tool \
which runs commands in the AgentCore Runtime sandbox.

Given the S3 URI and format below, sample up to {sample_size} records and \
infer the schema. Run a PySpark script using the execute tool.

IMPORTANT: Respond with ONLY the JSON output from the script, nothing else. \
Do not add any commentary before or after the JSON.
"""


def _sample_data(state: DataEngState, model) -> dict[str, Any]:
    """Agent node: use execute tool via DeepAgent to sample data from S3.

    Creates a mini DeepAgent that runs a PySpark script in the AgentCore
    Runtime sandbox to read sample records and infer schema.
    Falls back gracefully if S3 access fails.
    """
    task = get_task_description(state)
    s3_match = re.search(r"s3://\S+", task)
    if not s3_match:
        return {"inferred_schema": {}, "data_sample_status": "skipped"}

    s3_uri = s3_match.group(0)

    # Auto-detect format from task description
    task_lower = task.lower()
    if "csv" in task_lower:
        read_method = "csv"
        read_opts = '.option("header", "true").option("inferSchema", "true")'
    elif "json" in task_lower:
        read_method = "json"
        read_opts = ""
    else:
        read_method = "parquet"
        read_opts = ""

    sampling_script = (
        "from pyspark.sql import SparkSession\n"
        "import json\n"
        "\n"
        "try:\n"
        '    spark = SparkSession.builder \\\n'
        '        .appName("data_sampling") \\\n'
        '        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \\\n'
        "        .getOrCreate()\n"
        "\n"
        f'    df = spark.read.format("{read_method}"){read_opts}.load("{s3_uri}").limit({DEFAULT_SAMPLE_SIZE})\n'
        "\n"
        "    schema = []\n"
        "    for field in df.schema.fields:\n"
        "        schema.append({\n"
        '            "name": field.name,\n'
        '            "type": str(field.dataType),\n'
        '            "nullable": field.nullable,\n'
        "        })\n"
        "\n"
        "    row_count = df.count()\n"
        '    print(json.dumps({"status": "success", "columns": schema, "row_count": row_count}))\n'
        "    spark.stop()\n"
        "except Exception as e:\n"
        '    print(json.dumps({"status": "failed", "error": str(e)}))\n'
    )

    user_msg = (
        f"Run the following PySpark script using the execute tool. "
        f"First run: execute(\"pip install pyspark\")\n"
        f"Then run the script:\n\n```python\n{sampling_script}\n```\n\n"
        f"Respond with ONLY the JSON output line from the script."
    )

    try:
        with traced_span("agent:data_eng.sample_data", {
            "agent.graph": "data_eng",
            "agent.node": "sample_data",
            "sample.s3_uri": s3_uri,
            "sample.format": read_method,
            "sample.size": DEFAULT_SAMPLE_SIZE,
        }):
            response = _run_agent(
                _SAMPLE_DATA_PROMPT.format(sample_size=DEFAULT_SAMPLE_SIZE),
                user_msg,
                model,
            )

        # Try to extract JSON from the response
        # The agent may include extra text around the JSON
        json_match = re.search(r'\{[^{}]*"status"\s*:\s*"[^"]*"[^{}]*\}', response)
        if json_match:
            output = json.loads(json_match.group(0))
        else:
            output = json.loads(response.strip())

        if output.get("status") == "success":
            return {
                "inferred_schema": {
                    "columns": output["columns"],
                    "row_count": output["row_count"],
                },
                "data_sample_status": "success",
            }
        else:
            logger.error("Data sampling failed: %s", output.get("error", "unknown"))
            return {"inferred_schema": {}, "data_sample_status": "failed"}
    except Exception as exc:
        logger.error("Data sampling exception: %s", exc)
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
        "test_transform.py": "/tests/test_transform.py",
    }
    return {"code_artifacts": code_artifacts, "messages": [AIMessage(content=response)]}


_VALIDATE_PROMPT = """\
You are a pytest validation runner. You have access to the execute tool \
which runs commands in the AgentCore Runtime sandbox.

Run the following steps:
1. execute("pip install pytest pandas pyspark")
2. execute("python -m pytest /tests/test_transform.py -v --tb=long")

Report the full pytest output. State clearly whether validation PASSED or FAILED.
"""


def _validate(state: DataEngState, model) -> dict[str, Any]:
    """Agent node: use execute tool via DeepAgent to run pytest validation.

    Creates a mini DeepAgent that installs deps and runs pytest in the
    AgentCore Runtime sandbox. Parses test output and sets
    validation_passed / validation_output.
    """
    artifacts = state.get("code_artifacts", {})
    files_list = ", ".join(artifacts.values()) if artifacts else "unknown"

    user_msg = (
        f"## Generated Files\n{files_list}\n\n"
        "Install dependencies and run pytest validation now."
    )

    try:
        with traced_span("agent:data_eng.validate", {
            "agent.graph": "data_eng",
            "agent.node": "validate",
            "agent.artifact_count": len(artifacts),
        }):
            response = _run_agent(_VALIDATE_PROMPT, user_msg, model)
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
    """Agent node: create_deep_agent with edit_file + browser tools to fix pytest failures."""
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

    graph = StateGraph(DataEngState)

    def sample_data(state: DataEngState) -> dict[str, Any]:
        return _sample_data(state, model)

    def generate(state: DataEngState) -> dict[str, Any]:
        return _generate(state, model, browser_tools)

    def validate(state: DataEngState) -> dict[str, Any]:
        return _validate(state, model)

    def fix(state: DataEngState) -> dict[str, Any]:
        return _fix(state, model, browser_tools)

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
