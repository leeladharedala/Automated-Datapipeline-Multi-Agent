# ──────────────────────────────────────────────────────────────
# Input variables
# ──────────────────────────────────────────────────────────────

# ── General ───────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Naming prefix used across all resources"
  type        = string
  default     = "multi-agent-pipeline"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod"
  }
}

# ── ECR ───────────────────────────────────────────────────────

variable "ecr_image_tag" {
  description = "Container image tag to deploy (overridden by CI/CD)"
  type        = string
  default     = "latest"
}

# ── AgentCore ─────────────────────────────────────────────────

variable "agentcore_model_id" {
  description = "Bedrock model ID for the orchestrator agent"
  type        = string
  default     = "anthropic.claude-sonnet-4-6-20250514-v1:0"
}

variable "memory_event_expiry_days" {
  description = "Number of days before AgentCore Memory events expire"
  type        = number
  default     = 90
}

# ── Secrets ───────────────────────────────────────────────────

variable "github_token_secret_arn" {
  description = "ARN of existing Secrets Manager secret for GitHub PAT (used by submit_pr tool). Leave empty to skip."
  type        = string
  default     = ""
}

variable "github_target_repo" {
  description = "GitHub repo where the agent creates PRs, in org/repo format (e.g. myorg/infra-repo)"
  type        = string
  default     = "your-org/your-target-repo" # ← change this
}

variable "github_base_branch" {
  description = "Base branch for PRs created by the agent"
  type        = string
  default     = "main"
}

# ── Browser Tool ──────────────────────────────────────────────

variable "browser_recording_s3_bucket" {
  description = "S3 bucket name for browser session recordings. Leave empty to disable recording."
  type        = string
  default     = ""
}

# ── Observability ─────────────────────────────────────────────

variable "log_retention_days" {
  description = "CloudWatch log group retention in days"
  type        = number
  default     = 30
}

variable "xray_sampling_rate" {
  description = "X-Ray sampling rate (1.0 = 100%, 0.1 = 10%)"
  type        = number
  default     = 1.0

  validation {
    condition     = var.xray_sampling_rate >= 0 && var.xray_sampling_rate <= 1
    error_message = "xray_sampling_rate must be between 0 and 1"
  }
}

variable "adot_collector_image" {
  description = "ADOT Collector container image for the X-Ray sidecar"
  type        = string
  default     = "amazon/aws-otel-collector:latest"
}

# ── Frontend ──────────────────────────────────────────────────

variable "frontend_origin" {
  description = "Frontend origin URL for CORS on the ADOT collector HTTP receiver (e.g. https://app.example.com)"
  type        = string
  default     = "http://localhost:3000"
}

# ── Data Source S3 Access ─────────────────────────────────────

variable "data_source_s3_arns" {
  description = "S3 bucket ARNs the Data Engineering agent can read from for data sampling"
  type        = list(string)
  default     = []
}
