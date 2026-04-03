# ──────────────────────────────────────────────────────────────
# Outputs
# ──────────────────────────────────────────────────────────────

# ── ECR ───────────────────────────────────────────────────────

output "ecr_repository_url" {
  description = "ECR repository URL — use in CI/CD for docker push"
  value       = aws_ecr_repository.agent.repository_url
}

# ── AgentCore Runtime ─────────────────────────────────────────

output "agent_runtime_id" {
  description = "AgentCore runtime identifier"
  value       = aws_bedrockagentcore_agent_runtime.orchestrator.agent_runtime_id
}

output "agent_runtime_arn" {
  description = "AgentCore runtime ARN"
  value       = aws_bedrockagentcore_agent_runtime.orchestrator.agent_runtime_arn
}

output "agent_runtime_endpoint" {
  description = "AgentCore runtime endpoint ARN — used by the CopilotKit frontend proxy"
  value       = aws_bedrockagentcore_agent_runtime_endpoint.orchestrator.agent_runtime_endpoint_arn
}

# ── Memory ────────────────────────────────────────────────────

output "memory_id" {
  description = "AgentCore Memory ID — set as AGENTCORE_MEMORY_ID env var"
  value       = aws_bedrockagentcore_memory.pipeline.id
}

# ── Gateway ───────────────────────────────────────────────────


# ── Tools ─────────────────────────────────────────────────────

output "code_interpreter_id" {
  description = "Code interpreter resource ID"
  value       = aws_bedrockagentcore_code_interpreter.data_eng.code_interpreter_id
}

output "browser_arn" {
  description = "Browser tool resource ARN"
  value       = aws_bedrockagentcore_browser.data_eng.browser_arn
}

# ── Secrets ───────────────────────────────────────────────────

output "anthropic_secret_arn" {
  description = "Secrets Manager ARN for Anthropic API key — target for GitHub Actions put-secret-value"
  value       = aws_secretsmanager_secret.anthropic_api_key.arn
}

# ── Observability ─────────────────────────────────────────────

output "cloudwatch_log_group_name" {
  description = "CloudWatch log group for agent runtime logs"
  value       = aws_cloudwatch_log_group.agent_runtime.name
}

output "cloudwatch_dashboard_url" {
  description = "Direct link to the CloudWatch dashboard"
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.agent_pipeline.dashboard_name}"
}

# ── IAM ───────────────────────────────────────────────────────

output "execution_role_arn" {
  description = "AgentCore execution role ARN"
  value       = aws_iam_role.agentcore_execution.arn
}

# ── S3 Raw Data ───────────────────────────────────────────────

output "raw_data_bucket_name" {
  description = "S3 bucket name for raw data storage — used by pipeline sync step"
  value       = aws_s3_bucket.raw_data.id
}

output "raw_data_bucket_arn" {
  description = "S3 bucket ARN for raw data storage — for IAM policy references"
  value       = aws_s3_bucket.raw_data.arn
}
