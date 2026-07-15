#!/usr/bin/env bash
set -euo pipefail

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
