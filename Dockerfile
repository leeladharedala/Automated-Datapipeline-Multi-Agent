# AgentCore Runtime container
FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

# Install system deps + terraform CLI + actionlint + Node.js (for npx MCP servers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget unzip ca-certificates curl gnupg && \
    rm -rf /var/lib/apt/lists/*

# Node.js 22 LTS (provides npx for Terraform MCP server)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Pre-install MCP server npm packages so npx doesn't download at runtime
RUN npm install -g @hashicorp/terraform-mcp-server || true

# Pre-install AWS Docs MCP server via pip (uvx fails under QEMU ARM64 emulation)
RUN pip install --no-cache-dir "awslabs.aws-documentation-mcp-server"

# Terraform
RUN wget -q https://releases.hashicorp.com/terraform/1.9.8/terraform_1.9.8_linux_arm64.zip && \
    unzip terraform_1.9.8_linux_arm64.zip -d /usr/local/bin/ && \
    rm terraform_1.9.8_linux_arm64.zip

# Actionlint
RUN wget -q https://github.com/rhysd/actionlint/releases/download/v1.7.7/actionlint_1.7.7_linux_arm64.tar.gz && \
    tar -xzf actionlint_1.7.7_linux_arm64.tar.gz -C /usr/local/bin/ actionlint && \
    rm actionlint_1.7.7_linux_arm64.tar.gz

# Python deps — use pip directly (uv can silently fail under QEMU ARM64 emulation)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verify critical packages are actually installed and show versions
RUN python -c "\
from importlib.metadata import version; \
import deepagents; print(f'deepagents {deepagents.__version__}'); \
import copilotkit; print('copilotkit OK'); \
import langchain; print(f'langchain {langchain.__version__}'); \
import langgraph; print(f'langgraph {version(\"langgraph\")}'); \
import langgraph_checkpoint_aws; print('langgraph-checkpoint-aws OK'); \
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore; print('Memory classes OK'); \
AgentCoreMemorySaver(memory_id='test', region_name='us-west-2'); print('AgentCoreMemorySaver OK'); \
import fastapi; print(f'fastapi {fastapi.__version__}'); \
import bedrock_agentcore; print('bedrock-agentcore OK'); \
from langchain_anthropic import ChatAnthropic; print('ChatAnthropic OK'); \
from langchain_mcp_adapters.client import MultiServerMCPClient; print('MCP adapters OK'); \
from langchain.agents.middleware.types import AgentMiddleware; print('AgentMiddleware OK'); \
from deepagents import create_deep_agent; print('create_deep_agent OK'); \
from deepagents import CompiledSubAgent; print('CompiledSubAgent OK'); \
from deepagents.middleware._utils import append_to_system_message; print('deepagents middleware OK'); \
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse; print('AgentMiddleware OK'); \
print('All critical packages verified.'); \
"

COPY src/ ./src/


EXPOSE 8080
ENV PORT=8080
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1



# NOTE: Do NOT add a Docker HEALTHCHECK — AgentCore manages health
# checking itself by polling /ping. A Docker-level HEALTHCHECK can
# conflict with AgentCore's container lifecycle management.

CMD ["opentelemetry-instrument", "python", "src/agentcore/server.py"]