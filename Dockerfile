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

RUN pip install --no-cache-dir uv

# Pre-install AWS Docs MCP server so uvx doesn't download at runtime
# (avoids network timeouts / proxy issues in production)
RUN python -m uv tool install "awslabs.aws-documentation-mcp-server@latest" || true

# Verify critical packages are actually installed and show versions
RUN python -c "\
from importlib.metadata import version; \
import deepagents; print(f'deepagents {deepagents.__version__}'); \
assert deepagents.__version__.startswith('0.6'), f'expected deepagents 0.6.x, got {deepagents.__version__}'; \
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
from langchain_quickjs import CodeInterpreterMiddleware; print('CodeInterpreterMiddleware (PTC) OK'); \
from langchain.agents.middleware.types import AgentMiddleware; print('AgentMiddleware OK'); \
from deepagents import create_deep_agent; print('create_deep_agent OK'); \
from deepagents import AsyncSubAgent, AsyncSubAgentMiddleware; print('AsyncSubAgent OK'); \
from langgraph_sdk import get_client, get_sync_client; print('langgraph_sdk OK'); \
from deepagents.middleware._utils import append_to_system_message; print('deepagents middleware OK'); \
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse; print('AgentMiddleware OK'); \
print('All critical packages verified.'); \
"

COPY src/ ./src/

# Co-located LangGraph API server config + in-container process launcher
COPY langgraph.json ./
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh


EXPOSE 8080
ENV PORT=8080
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# AWS Docs MCP server config — per https://github.com/awslabs/mcp
ENV AWS_DOCUMENTATION_PARTITION=aws
ENV MCP_USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"



# NOTE: Do NOT add a Docker HEALTHCHECK — AgentCore manages health
# checking itself by polling /ping. A Docker-level HEALTHCHECK can
# conflict with AgentCore's container lifecycle management.

CMD ["./entrypoint.sh"]