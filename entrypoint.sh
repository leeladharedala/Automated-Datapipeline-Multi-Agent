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
# Deliberately NOT instrumented. Auto-instrumentation of this process is
# incompatible with the ADOT distro: even with
# OTEL_PYTHON_DISABLED_INSTRUMENTATIONS covering botocore/requests/urllib3,
# the aws_configurator applies its own botocore patches unconditionally, so
# every boto3 call here (Secrets Manager at import, S3 data sampling at run
# time) dies with "maximum recursion depth exceeded"; the langchain
# instrumentation additionally mutated replayed messages into thinking-block
# 400s. Verified empirically on 2026-07-19 across three configurations.
# Sub-agent logs reach CloudWatch via stdout ([runtime-logs] streams) and
# reach the dashboard via the run-stream relays; span-level telemetry for
# this process would need a hand-configured OTel SDK (no auto patches).
langgraph dev \
  --host 127.0.0.1 --port 2024 \
  --n-jobs-per-worker 4 \
  --no-reload \
  --no-browser &
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
