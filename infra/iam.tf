# ──────────────────────────────────────────────────────────────
# IAM — AgentCore execution role & policies
# ──────────────────────────────────────────────────────────────

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = var.aws_region
}

# ── Trust policy: allow AgentCore to assume this role ─────────

data "aws_iam_policy_document" "agentcore_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type = "Service"
      identifiers = [
        "bedrock.amazonaws.com",
        "lambda.amazonaws.com",
        "ecs-tasks.amazonaws.com"
      ]
    }

  }
}

resource "aws_iam_role" "agentcore_execution" {
  name               = "${var.project_name}-${var.environment}-agentcore-exec"
  assume_role_policy = data.aws_iam_policy_document.agentcore_assume_role.json
}

# ── Bedrock model invocation ──────────────────────────────────

data "aws_iam_policy_document" "bedrock_invoke" {
  statement {
    sid    = "BedrockInvoke"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    resources = ["arn:aws:bedrock:${local.region}::foundation-model/*"]
  }
}

resource "aws_iam_role_policy" "bedrock_invoke" {
  name   = "bedrock-invoke"
  role   = aws_iam_role.agentcore_execution.id
  policy = data.aws_iam_policy_document.bedrock_invoke.json
}

# ── ECR pull ──────────────────────────────────────────────────

data "aws_iam_policy_document" "ecr_pull" {
  statement {
    sid    = "ECRAuth"
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "ECRPull"
    effect = "Allow"
    actions = [
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability",
    ]
    resources = [data.aws_ecr_repository.agent.arn]
  }
}

resource "aws_iam_role_policy" "ecr_pull" {
  name   = "ecr-pull"
  role   = aws_iam_role.agentcore_execution.id
  policy = data.aws_iam_policy_document.ecr_pull.json
}

# ── AgentCore Memory ──────────────────────────────────────────

data "aws_iam_policy_document" "agentcore_memory" {
  statement {
    sid    = "AgentCoreMemory"
    effect = "Allow"
    actions = [
      "bedrock:GetAgentMemory",
      "bedrock:PutAgentMemory",
      "bedrock:DeleteAgentMemory",
    ]
    resources = [
      "arn:aws:bedrock:${local.region}:${local.account_id}:agent-memory/*",
    ]
  }
}

resource "aws_iam_role_policy" "agentcore_memory" {
  name   = "agentcore-memory"
  role   = aws_iam_role.agentcore_execution.id
  policy = data.aws_iam_policy_document.agentcore_memory.json
}

# ── CloudWatch Logs ───────────────────────────────────────────

data "aws_iam_policy_document" "cloudwatch_logs" {
  statement {
    sid    = "CloudWatchLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = [
      "arn:aws:logs:${local.region}:${local.account_id}:log-group:/agentcore/${var.project_name}-${var.environment}*",
    ]
  }
}

resource "aws_iam_role_policy" "cloudwatch_logs" {
  name   = "cloudwatch-logs"
  role   = aws_iam_role.agentcore_execution.id
  policy = data.aws_iam_policy_document.cloudwatch_logs.json
}

# ── Secrets Manager (Anthropic API key + GitHub PAT) ──────────

data "aws_iam_policy_document" "secrets_read" {
  statement {
    sid    = "SecretsRead"
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret",
    ]
    resources = compact([
      aws_secretsmanager_secret.anthropic_api_key.arn,
      var.github_token_secret_arn,
    ])
  }
}

resource "aws_iam_role_policy" "secrets_read" {
  name   = "secrets-read"
  role   = aws_iam_role.agentcore_execution.id
  policy = data.aws_iam_policy_document.secrets_read.json
}

# ── S3 (browser recordings + optional artifact storage) ───────

data "aws_iam_policy_document" "s3_access" {
  statement {
    sid    = "S3Access"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      "arn:aws:s3:::${var.project_name}-*",
      "arn:aws:s3:::${var.project_name}-*/*",
    ]
  }
}

resource "aws_iam_role_policy" "s3_access" {
  name   = "s3-access"
  role   = aws_iam_role.agentcore_execution.id
  policy = data.aws_iam_policy_document.s3_access.json
}

# ── S3 Data Source Read (for Data Engineering sampling) ───────

data "aws_iam_policy_document" "s3_data_source_read" {
  count = length(var.data_source_s3_arns) > 0 ? 1 : 0

  statement {
    sid    = "S3DataSourceRead"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
    ]
    resources = concat(
      var.data_source_s3_arns,
      [for arn in var.data_source_s3_arns : "${arn}/*"]
    )
  }
}

resource "aws_iam_role_policy" "s3_data_source_read" {
  count  = length(var.data_source_s3_arns) > 0 ? 1 : 0
  name   = "s3-data-source-read"
  role   = aws_iam_role.agentcore_execution.id
  policy = data.aws_iam_policy_document.s3_data_source_read[0].json
}

# ── X-Ray Tracing ─────────────────────────────────────────────

resource "aws_iam_role_policy" "xray_write" {
  name   = "xray-write"
  role   = aws_iam_role.agentcore_execution.id
  policy = data.aws_iam_policy_document.xray_write.json
}
