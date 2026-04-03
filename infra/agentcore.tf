# ──────────────────────────────────────────────────────────────
# AgentCore — Runtime, Memory, Gateway
# ──────────────────────────────────────────────────────────────

# ── Agent Runtime ─────────────────────────────────────────────

resource "aws_bedrockagentcore_agent_runtime" "orchestrator" {
  agent_runtime_name = replace("${var.project_name}_${var.environment}", "-", "_")
  description        = "Multi-agent data pipeline orchestrator (LangGraph + DeepAgents)"

  agent_runtime_artifact {
    container_configuration {
      container_uri = "${aws_ecr_repository.agent.repository_url}:${var.ecr_image_tag}"
    }
  }

  role_arn = aws_iam_role.agentcore_execution.arn

  environment_variables = {
    # AWS
    AWS_REGION = var.aws_region

    # Memory
    AGENTCORE_MEMORY_ID = aws_bedrockagentcore_memory.pipeline.id

    # Anthropic API key — resolved from Secrets Manager at boot
    ANTHROPIC_API_KEY_SECRET_ARN = aws_secretsmanager_secret.anthropic_api_key.arn

    # Model
    BEDROCK_MODEL_ID = var.agentcore_model_id

    # Observability (OTEL / ADOT)
    AGENT_OBSERVABILITY_ENABLED = "true"
    OTEL_TRACES_EXPORTER        = "otlp"
    OTEL_EXPORTER_OTLP_ENDPOINT = "https://otlp.${var.aws_region}.amazonaws.com"
    OTEL_RESOURCE_ATTRIBUTES    = "service.name=${var.project_name}-${var.environment}"

    # GitHub — used by submit_pr tool to create PRs on a target repo
    GITHUB_REPO             = var.github_target_repo
    GITHUB_BASE_BRANCH      = var.github_base_branch
    GITHUB_TOKEN_SECRET_ARN = var.github_token_secret_arn
  }

  network_configuration {
    network_mode = "PUBLIC"
  }

  # ── ADOT Collector Sidecar ──────────────────────────────────
  #
  # NOTE: sidecar_container is not yet supported by the Terraform
  # AWS provider for aws_bedrockagentcore_agent_runtime.
  # The OTEL env vars above will export traces via the AgentCore
  # built-in observability. Re-add the sidecar block when the
  # provider schema supports it.
}

# ── Agent Runtime Endpoint ────────────────────────────────────

resource "aws_bedrockagentcore_agent_runtime_endpoint" "orchestrator" {
  name             = replace("${var.project_name}_${var.environment}_endpoint", "-", "_")
  agent_runtime_id = aws_bedrockagentcore_agent_runtime.orchestrator.agent_runtime_id
  description      = "Public endpoint for the orchestrator agent runtime"
}

# ── Agent Memory ──────────────────────────────────────────────

resource "aws_bedrockagentcore_memory" "pipeline" {
  name        = replace("${var.project_name}_${var.environment}_memory", "-", "_")
  description = "Short-term checkpoints and long-term preferences for the pipeline agents"

  event_expiry_duration = var.memory_event_expiry_days
}

# ── MCP Gateway ───────────────────────────────────────────────

resource "aws_bedrockagentcore_gateway" "mcp" {
  name        = replace("${var.project_name}_${var.environment}_mcp_gateway", "-", "_")
  description = "MCP gateway for Terraform Registry and AWS Docs"

  protocol_type   = "MCP"
  role_arn        = aws_iam_role.agentcore_execution.arn
  authorizer_type = "NONE"
}

# Gateway target: Terraform Registry MCP server
resource "aws_bedrockagentcore_gateway_target" "terraform_registry" {
  gateway_identifier = aws_bedrockagentcore_gateway.mcp.gateway_id
  name               = "terraform_registry"
  description        = "Terraform Registry MCP — module schemas, provider templates"

  target_configuration {
    mcp {
      mcp_server {
        endpoint = "https://registry.terraform.io/mcp"
      }
    }
  }
}

# Gateway target: AWS Documentation MCP server
resource "aws_bedrockagentcore_gateway_target" "aws_docs" {
  gateway_identifier = aws_bedrockagentcore_gateway.mcp.gateway_id
  name               = "aws_docs"
  description        = "AWS Docs MCP — live AWS resource property documentation"

  target_configuration {
    mcp {
      mcp_server {
        endpoint = "https://docs.aws.amazon.com/mcp"
      }
    }
  }
}
