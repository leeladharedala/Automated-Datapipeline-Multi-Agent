"""
Minimal AgentCore test server using the official SDK.

Deploy this INSTEAD of the real server to isolate whether the issue
is infrastructure (Terraform/IAM/ECR) or application code.

Usage in Dockerfile CMD:
  CMD ["python", "src/agentcore/server_minimal.py"]
"""

print("BOOT: Minimal test server starting...", flush=True)

from bedrock_agentcore.runtime import BedrockAgentCoreApp

print("BOOT: bedrock-agentcore SDK imported OK.", flush=True)

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload, context=None):
    """Minimal echo handler to test AgentCore connectivity."""
    prompt = payload.get("prompt", payload.get("input", {}).get("prompt", "no prompt"))
    return {"result": f"Echo: {prompt}"}


if __name__ == "__main__":
    print("BOOT: Starting BedrockAgentCoreApp...", flush=True)
    app.run()
