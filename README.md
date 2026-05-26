# Multi-Agent Data Pipeline

### [🎬 Watch the Walkthrough Demo Video](YOUR_GOOGLE_DRIVE_LINK_HERE)



## 🚀 Overview
This multi-agent system is designed to autonomously generate, validate, and deploy production-grade cloud infrastructure (Terraform), CI/CD pipelines (GitHub Actions), and data engineering workloads (PySpark). 

Instead of relying on a single, massive LLM prompt, the system utilizes a hierarchical architecture where a master **Orchestrator Agent** routes complex user requests to highly specialized **Subagents**.

---

## 🛡️ Reliability Guarantees

- **Isolated Execution**: By sandboxing all tool executions (`terraform plan`, `actionlint`), the host infrastructure remains entirely protected against hallucinated code and malicious commands.
- **Self-Healing Validation Loop**: The integration of the deterministic LangGraph `StateGraph` ensures that every failure dynamically triggers the `fix` node. No flawed code can bypass the `validate` node until it mathematically passes or exhausts its retry limits.
- **Stateless Debugging (Context Management)**: To prevent LLM context-window bloat and "prompt drift" during multiple fix attempts, the debugging nodes are built to be stateless. They inject only the broken file and the exact error trace into a fresh prompt, ensuring the agent remains hyper-focused on the fix.
- **AST Structural Validation**: Instead of relying on LLM self-evaluations or brittle regex, PySpark code is validated via Python's native Abstract Syntax Tree (AST). This guarantees that essential functions like `main()` are present, syntax is flawless, and required library imports exist.
- **Destroy-Compatible Generation**: The IaC agent strictly enforces CI/CD lifecycle compatibility by ensuring infrastructure is cleanly destroyable (e.g., forcing `force_destroy = true` on S3 buckets, avoiding `prevent_destroy`), preventing dangling cloud resources.
- **Format Integrity & Serialization Boundaries**: The structured `<!-- PR_FILES_JSON ... -->` block serves as a strict serialization boundary. This guarantees that file payloads are extracted precisely via JSON parsing, entirely avoiding the fragility of parsing varying Markdown formats.
- **Agent Loop Reuse**: The background `asyncio` loop is cached and reused within the subagent instances. This prevents connection teardown errors and allows the sandbox cache to persist session IDs securely across internal LLM tool calls.

---

## 📝 Expected Input Document Format
To trigger the pipeline, users typically submit a structured .txt file outlining the data sources, required transformations, and target infrastructure. The orchestrator uses this document to route tasks correctly. 

**Example Input:**
```
data_source:
  uri: s3://multi-agent-pipeline-dev-raw-data/raw_data/
  format: jsonl
  options:
    schema:
      site_id: string
      timestamp: string
      energy_generated_kwh: float
      energy_consumed_kwh: float

transformations:
  - name: calculate_net_energy
    description: >
      Add column net_energy_kwh = energy_generated_kwh - energy_consumed_kwh
  - name: flag_negative_energy
    description: >
      Add column negative_energy_flag set to 1 if energy_generated_kwh < 0
      or energy_consumed_kwh < 0, otherwise 0

architecture:
  compute: AWS Glue (PySpark)
  source: S3 raw_data/ prefix (JSONL)
  sink: S3 transformed_data/ prefix (Parquet, overwrite mode)
  flow: Read from S3 -> Transform in Glue -> Write to S3
```

---

## 🏛️ Main Architecture

The core multi-agent platform is driven by the **Main Orchestrator Agent**, powered by **Claude 4.6 Sonnet**. The Orchestrator does not write the complex code itself; instead, it acts as a project manager—understanding the user's intent, maintaining conversation flow, delegating tasks to subagents, and deploying the final artifacts.

### 🌟 Orchestrator Features & Capabilities

1. **Intelligent Subagent Routing**: Wraps subagent graphs as `CompiledSubAgent` tools. The Orchestrator decides whether a user prompt requires Infrastructure, Data Engineering, or CI/CD work (or a combination) and invokes the relevant subagent.
2. **Persistent Memory**: Uses AWS AgentCore (`AgentCoreMemorySaver`) for robust short-term checkpoint persistence, ensuring the orchestrator remembers the entire conversation history seamlessly across interactions without skipping turns.
3. **Automated Deployment**: Armed with a `submit_pr` tool, the Orchestrator aggregates the results from the subagents and natively opens a GitHub Pull Request with the generated codebase.
4. **Middleware Interception**: Custom `OrchestratorMiddleware` allows for pre-processing of interactions and telemetry capture.
5. **Observability & Tracing**: 
   - Integrated with OpenTelemetry (`aws-opentelemetry-distro` and `opentelemetry-instrumentation-langchain`).
   - Extensive use of `@traced_span` and `instrument_graph()` around every node (e.g., `agent:iac.generate`, `agent:data_eng.validate`) to capture deep observability into LLM latency, tool invocations, token counts, and internal ReAct loops.
6. **Robust Output Parsing**: The Orchestrator strips a specialized hidden JSON block (`<!-- PR_FILES_JSON ... -->`) from the end of the subagent reports to cleanly extract the code payloads, eliminating the brittleness of regex parsing markdown code blocks.

---

## 🤖 Detailed Subagent Architecture

The system features three specialized subagents. Crucially, none of these are generic open-ended loops. They are implemented as strictly defined **LangGraph `StateGraphs`** that follow a deterministic `Research -> Generate -> Validate -> Fix -> Report` lifecycle.

### 1. Data Engineering Agent
**Purpose:** Generate robust, functional PySpark data transformations with proper null-handling and schema awareness.

* **`sample_data` (Node)**: Instead of delegating to an LLM, this node executes Python (`boto3` + `pandas`) directly in-process using the container's IAM role. It samples data from target S3 buckets (CSV, JSON, Parquet) and dynamically infers the exact schema for the prompt.
* **`generate` (Node)**: A specialized `DeepAgent` generates pure PySpark code (`transform.py` & `__init__.py`) using functional programming paradigms, avoiding `.rdd` or `.map()`.
* **`validate` (Node)**: Utilizes Python's native `ast` module to run structural validation without spinning up a sandbox. It mathematically guarantees the code has a `main()` entry point, no syntax errors, and proper `pyspark` imports.
* **`fix` (Node)**: A stateless, highly-focused LLM call. To prevent "context window bloat" across multiple failures, it passes only the broken file and error message, surgically patching the code up to 3 times before returning to `validate`.

### 2. Infrastructure as Code (IaC) Agent
**Purpose:** Generate parameterised, destroy-compatible Terraform configurations representing AWS cloud resources.

* **`research` (Node)**: Leverages **Model Context Protocol (MCP)** servers (Terraform Registry, AWS Docs) lazily loaded onto an isolated event loop. It dynamically researches exact Terraform resource types, required arguments, and best practices.
* **`generate` (Node)**: Outputs the exact HCL structure (`provider.tf`, `variables.tf`, `main.tf`, `outputs.tf`), strictly enforcing policies like `force_destroy = true` and `skip_final_snapshot = true` to guarantee CI/CD tear-down compatibility.
* **`validate` (Node)**: Bridges the file system to the **AgentCore Runtime Sandbox**. It writes the files to an isolated microVM and executes `terraform init -backend=false` followed by `terraform plan`. It acts as an automated Code Reviewer, checking the plan output to ensure all requested resources are correctly mapped.
* **`fix` (Node)**: A mini DeepAgent armed with `read_file`, `edit_file`, and `execute` tools. It parses the sandbox Terraform errors and surgically modifies specific `.tf` files.

### 3. CI/CD Agent
**Purpose:** Generate secure, OIDC-compliant GitHub Actions workflows.

* **`generate` (Node)**: Outputs YAML files (`deploy.yml`, `destroy.yml`) featuring OIDC AWS authentication, parameterized environments, concurrency groups, and manual approval gates for production.
* **`validate` (Node)**: Uses the **AgentCore Runtime Sandbox** to run `actionlint` directly against the generated GitHub Actions workflows to catch YAML validation errors or action-specific bugs before they ever reach GitHub.
* **`fix` (Node)**: Similar to IaC, this node intercepts any `actionlint` failures and uses surgical `edit_file` tools to patch YAML syntax without regenerating the entire workflow from scratch.

---

## ⚠️ Disclaimer
While this multi-agent system features robust self-healing and validation mechanisms, **the generated code should never be merged blindly**. A human engineer must always review the Pull Request submitted by the Orchestrator to ensure architectural accuracy, security compliance, and business logic correctness before deploying to production environments.
