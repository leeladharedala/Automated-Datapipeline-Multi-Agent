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

# Terraform
RUN wget -q https://releases.hashicorp.com/terraform/1.9.8/terraform_1.9.8_linux_arm64.zip && \
    unzip terraform_1.9.8_linux_arm64.zip -d /usr/local/bin/ && \
    rm terraform_1.9.8_linux_arm64.zip

# Actionlint
RUN wget -q https://github.com/rhysd/actionlint/releases/download/v1.7.7/actionlint_1.7.7_linux_arm64.tar.gz && \
    tar -xzf actionlint_1.7.7_linux_arm64.tar.gz -C /usr/local/bin/ actionlint && \
    rm actionlint_1.7.7_linux_arm64.tar.gz

# Python deps (install uv first to bypass pip backtracking hell, then use uv to resolve deps instantly)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip uv && \
    uv pip install --system --no-cache -r requirements.txt uvicorn fastapi

COPY src/ ./src/

EXPOSE 8080
ENV PORT=8080
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1


CMD ["python", "-m", "uvicorn", "src.agentcore.server:app", "--host", "0.0.0.0", "--port", "8080"]
