# ──────────────────────────────────────────────────────────────
# Built-in Tools — Code Interpreter & Browser
#
# Both are fully managed AWS resources running in isolated
# Firecracker microVMs. No Lambda/ECS needed.
# ──────────────────────────────────────────────────────────────

# ── Wait for IAM Propagation ────────────────────────────────────
resource "time_sleep" "wait_for_iam" {
  depends_on = [
    aws_iam_role.agentcore_execution,
    aws_iam_role_policy.bedrock_invoke,
    aws_iam_role_policy.agentcore_memory,
    aws_iam_role_policy.ecr_pull,
    aws_iam_role_policy.s3_access,
    aws_iam_role_policy.secrets_read
  ]
  create_duration = "15s"
}

# ── Code Interpreter ──────────────────────────────────────────
# Used by the Data Engineering subagent for running pytest
# against generated transformation code.

resource "aws_bedrockagentcore_code_interpreter" "data_eng" {
  depends_on  = [time_sleep.wait_for_iam]
  name        = replace("${var.project_name}_${var.environment}_code_interpreter", "-", "_")
  description = "Managed Python sandbox for Data Engineering subagent — runs pytest on generated code"

  execution_role_arn = aws_iam_role.agentcore_execution.arn

  network_configuration {
    network_mode = "PUBLIC"
  }
}

# ── Browser Tool ──────────────────────────────────────────────
# Used by the Data Engineering subagent for web search and
# documentation lookups.

resource "aws_bedrockagentcore_browser" "data_eng" {
  depends_on  = [time_sleep.wait_for_iam]
  name        = replace("${var.project_name}_${var.environment}_browser", "-", "_")
  description = "Managed Chromium browser for Data Engineering subagent — web search and docs"

  network_configuration {
    network_mode = "PUBLIC"
  }

  execution_role_arn = aws_iam_role.agentcore_execution.arn

  # Session recording — enable if S3 bucket is configured
  dynamic "recording" {
    for_each = var.browser_recording_s3_bucket != "" ? [1] : []
    content {
      enabled = true
      s3_location {
        bucket = var.browser_recording_s3_bucket
        prefix = "${var.project_name}-${var.environment}/browser-sessions/"
      }
    }
  }
}
