#!/usr/bin/env bash
set -euo pipefail

# --- Resolve ANTHROPIC_API_KEY BEFORE starting either process ---
# Must happen here, uninstrumented: fetching the secret inside the
# opentelemetry-instrument'ed langgraph server fails with "maximum recursion
# depth exceeded" (auto-instrumentation recursing through boto3 at graph
# import time), which left the sub-agent ChatAnthropic with no key and every
# run dying at its first LLM call. Exporting the key here lets the in-process
# resolvers (server_graphs._resolve_anthropic_key, server._resolve_secrets)
# short-circuit without ever touching Secrets Manager under instrumentation.
if [ -n "${ANTHROPIC_API_KEY_SECRET_ARN:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  for attempt in 1 2 3; do
    _KEY="$(python - <<'PY' 2>/dev/null || true
import os, boto3
arn = os.environ["ANTHROPIC_API_KEY_SECRET_ARN"]
region = os.environ.get("AWS_REGION", "us-west-2")
print(boto3.client("secretsmanager", region_name=region).get_secret_value(SecretId=arn)["SecretString"], end="")
PY
)"
    if [ -n "$_KEY" ]; then
      export ANTHROPIC_API_KEY="$_KEY"
      echo "Resolved ANTHROPIC_API_KEY in entrypoint (attempt $attempt)"
      break
    fi
    echo "ANTHROPIC_API_KEY resolution attempt $attempt failed; retrying" >&2
    sleep 2
  done
  if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "FATAL: could not resolve ANTHROPIC_API_KEY from Secrets Manager" >&2
    exit 1
  fi
fi

# --- Route sub-agent OTLP logs to the platform log group when discoverable ---
# The instrumented ingress learns its log destination from
# OTEL_EXPORTER_OTLP_LOGS_HEADERS (x-aws-log-group / x-aws-log-stream) when
# the platform injects it. If present, reuse it for Process B's hand-wired
# exporter so sub-agent records land in the SAME group/stream as the ingress
# records (the platform runtime group's name embeds a random suffix Terraform
# cannot self-reference). Otherwise the Terraform-provided
# SUBAGENT_OTLP_LOG_GROUP (/agentcore/<project>-<env>) fallback applies.
_PLATFORM_GROUP="$(printf '%s' "${OTEL_EXPORTER_OTLP_LOGS_HEADERS:-}" | tr ',' '\n' | sed -n 's/^x-aws-log-group=//p' | head -1)"
_PLATFORM_STREAM="$(printf '%s' "${OTEL_EXPORTER_OTLP_LOGS_HEADERS:-}" | tr ',' '\n' | sed -n 's/^x-aws-log-stream=//p' | head -1)"
if [ -n "$_PLATFORM_GROUP" ]; then
  export SUBAGENT_OTLP_LOG_GROUP="$_PLATFORM_GROUP"
  export SUBAGENT_OTLP_LOG_STREAM="${_PLATFORM_STREAM:-${SUBAGENT_OTLP_LOG_STREAM:-otel-rt-logs}}"
  echo "Sub-agent OTLP logs → platform group ${SUBAGENT_OTLP_LOG_GROUP} (${SUBAGENT_OTLP_LOG_STREAM})"
else
  echo "Sub-agent OTLP logs → ${SUBAGENT_OTLP_LOG_GROUP:-<none configured>} (${SUBAGENT_OTLP_LOG_STREAM:-otel-rt-logs})"
fi
# Boot diagnostic (names only, no values): which observability env keys the
# platform actually injects into the container.
echo "Observability env keys: $(env | grep -E '^(OTEL_|AGENT_|AWS_BEDROCK|BEDROCK_)' | cut -d= -f1 | sort | tr '\n' ' ')"

# --- Process B: co-located LangGraph API server (localhost only) ---
# Worker pool sized N+1 (N=3 sub-agents => >=4 concurrent slots) so a full
# fan-out never queues (Req 2.5 / 2.6 / 9.1).
#
# This is the in-memory langgraph-api server: sub-agent threads/runs live in
# process RAM and do NOT survive a container restart (the Supervisor's own
# state is unaffected — it checkpoints to AgentCore Memory). A durable Process
# B would need the Postgres runtime (LangSmith self-hosted; `langgraph up` is
# not usable here since AgentCore containers cannot run Docker). --no-reload
# disables the file watcher so the server is never restarted (and its state
# wiped) mid-run.
# NOT instrumented by default. Auto-instrumentation of this process is
# incompatible with the ADOT distro: even with
# OTEL_PYTHON_DISABLED_INSTRUMENTATIONS covering botocore/requests/urllib3,
# the aws_configurator applies its own botocore patches unconditionally, so
# every boto3 call here (Secrets Manager at import, S3 data sampling at run
# time) dies with "maximum recursion depth exceeded"; the langchain
# instrumentation additionally mutated replayed messages into thinking-block
# 400s. Verified empirically on 2026-07-19 across three configurations.
# Structured logs reach CloudWatch via the hand-wired OTLP exporter in
# server_graphs (plus stdout); the dashboard gets progress via the
# run-stream relays. Set SUBAGENT_INSTRUMENT=true (Terraform env) to wrap
# this process in opentelemetry-instrument anyway — expect the failures
# above until ADOT stops unconditional botocore patching.
if [ "${SUBAGENT_INSTRUMENT:-false}" = "true" ]; then
  echo "SUBAGENT_INSTRUMENT=true — wrapping langgraph dev in opentelemetry-instrument"
  opentelemetry-instrument langgraph dev \
    --host 127.0.0.1 --port 2024 \
    --n-jobs-per-worker 4 \
    --no-reload \
    --no-browser &
else
  langgraph dev \
    --host 127.0.0.1 --port 2024 \
    --n-jobs-per-worker 4 \
    --no-reload \
    --no-browser &
fi
LG_PID=$!

# --- Wait for readiness before ingress accepts work (Req 3.5/3.6) ---
for i in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:2024/ok >/dev/null 2>&1; then
    echo "Co-located LangGraph server ready"; break
  fi
  if ! kill -0 "$LG_PID" 2>/dev/null; then
    echo "Co-located server died during startup" >&2; exit 1
  fi
  sleep 1
done

# --- Process A: AgentCore ingress (unchanged contract, port 8080) ---
exec opentelemetry-instrument python src/agentcore/server.py
