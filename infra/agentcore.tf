# ──────────────────────────────────────────────────────────────
# AgentCore — Runtime, Memory, Gateway
# ──────────────────────────────────────────────────────────────

# ── Agent Runtime ─────────────────────────────────────────────

resource "aws_bedrockagentcore_agent_runtime" "orchestrator" {
  depends_on         = [time_sleep.wait_for_iam]
  agent_runtime_name = replace("${var.project_name}_${var.environment}", "-", "_")
  description        = "Multi-agent data pipeline orchestrator (LangGraph + DeepAgents)"

  network_configuration {
    network_mode = "PUBLIC"
  }

  agent_runtime_artifact {
    container_configuration {
      container_uri = "${data.aws_ecr_repository.agent.repository_url}:${var.ecr_image_tag}"
    }
  }

  protocol_configuration {
    server_protocol = "HTTP"
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


    # Observability (ADOT)
    AGENT_OBSERVABILITY_ENABLED                        = "true"
    OTEL_PYTHON_DISTRO                                 = "aws_distro"
    OTEL_PYTHON_CONFIGURATOR                           = "aws_configurator"
    OTEL_EXPORTER_OTLP_PROTOCOL                        = "http/protobuf"
    OTEL_TRACES_EXPORTER                               = "otlp"
    OTEL_RESOURCE_ATTRIBUTES                           = "service.name=${var.project_name}-${var.environment}"
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = "false"
    TRACELOOP_TRACE_CONTENT                            = "false"
    OTEL_PYTHON_ID_GENERATOR                           = "xray"

    # GitHub — used by submit_pr tool to create PRs on a target repo
    GITHUB_REPO             = var.github_target_repo
    GITHUB_BASE_BRANCH      = var.github_base_branch
    GITHUB_TOKEN_SECRET_ARN = startswith(var.github_token_secret_arn, "arn:") ? var.github_token_secret_arn : aws_secretsmanager_secret.github_pat.arn
  }

  lifecycle_configuration {
    idle_runtime_session_timeout = 900   # 15 min — keep warm sessions alive
    max_lifetime                 = 28800 # 8 hours
  }

  # ── ADOT Collector Sidecar ──────────────────────────────────
  #
  # NOTE: sidecar_container is not yet supported by the Terraform
  # AWS provider for aws_bedrockagentcore_agent_runtime.
  # The OTEL env vars above will export traces via the AgentCore
  # built-in observability. Re-add the sidecar block when the
  # provider schema supports it.
}

# ── Agent Memory ──────────────────────────────────────────────

resource "aws_bedrockagentcore_memory" "pipeline" {
  depends_on  = [time_sleep.wait_for_iam]
  name        = replace("${var.project_name}_${var.environment}_memory", "-", "_")
  description = "Short-term checkpoints and long-term preferences for the pipeline agents"

  event_expiry_duration = var.memory_event_expiry_days
}

