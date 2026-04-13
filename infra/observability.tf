# ──────────────────────────────────────────────────────────────
# Observability — CloudWatch Logs, Dashboard, OTEL config
#
# Note: No dedicated aws_bedrockagentcore_observability resource
# exists yet. Observability is enabled via:
#   1. CloudWatch log groups (this file)
#   2. Runtime env vars for OTEL/ADOT (in agentcore.tf)
#   3. CloudWatch dashboard (this file)
# ──────────────────────────────────────────────────────────────

# ── X-Ray IAM Policy Document ─────────────────────────────────

data "aws_iam_policy_document" "xray_write" {
  statement {
    sid    = "XRayWrite"
    effect = "Allow"
    actions = [
      "xray:PutTraceSegments",
      "xray:PutTelemetryRecords",
      "xray:GetSamplingRules",
    ]
    resources = ["*"]
  }
}

# ── X-Ray Sampling Rule ──────────────────────────────────────

resource "aws_xray_sampling_rule" "agent_pipeline" {
  rule_name      = "${var.project_name}-${var.environment}"
  priority       = 1000
  reservoir_size = 1
  fixed_rate     = var.xray_sampling_rate
  host           = "*"
  http_method    = "*"
  service_name   = "${var.project_name}-${var.environment}"
  service_type   = "*"
  url_path       = "*"
  version        = 1
  resource_arn   = "*"
}

# ── X-Ray Group ──────────────────────────────────────────────

resource "aws_xray_group" "agent_pipeline" {
  group_name        = "${var.project_name}-${var.environment}"
  filter_expression = "service(\"${var.project_name}-${var.environment}\")"
}

# ── X-Ray Indexing Rule (Transaction Search) ─────────────────

resource "aws_xray_indexing_rule" "agent_pipeline" {
  rule_name = "Default"

  rule {
    probabilistic {
      desired_rule_percentage = 100
    }
  }
}

# ── CloudWatch Log Group ──────────────────────────────────────

resource "aws_cloudwatch_log_group" "agent_runtime" {
  name              = "/agentcore/${var.project_name}-${var.environment}"
  retention_in_days = var.log_retention_days
}

# ── CloudWatch Dashboard ─────────────────────────────────────

resource "aws_cloudwatch_dashboard" "agent_pipeline" {
  dashboard_name = "${var.project_name}-${var.environment}"

  dashboard_body = jsonencode({
    widgets = [
      # ── Row 1: Invocation metrics ──
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Agent Invocation Count"
          region = var.aws_region
          metrics = [
            ["AWS/BedrockAgentCore", "InvocationCount", "RuntimeName", "${var.project_name}-${var.environment}"]
          ]
          stat   = "Sum"
          period = 300
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Agent Invocation Latency (p50/p90/p99)"
          region = var.aws_region
          metrics = [
            ["AWS/BedrockAgentCore", "InvocationLatency", "RuntimeName", "${var.project_name}-${var.environment}", { stat = "p50" }],
            ["...", { stat = "p90" }],
            ["...", { stat = "p99" }]
          ]
          period = 300
          view   = "timeSeries"
        }
      },

      # ── Row 2: Errors & tool usage ──
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "Error Count"
          region = var.aws_region
          metrics = [
            ["AWS/BedrockAgentCore", "ErrorCount", "RuntimeName", "${var.project_name}-${var.environment}"]
          ]
          stat   = "Sum"
          period = 300
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "Code Interpreter & Browser Tool Invocations"
          region = var.aws_region
          metrics = [
            ["AWS/BedrockAgentCore", "ToolInvocationCount", "ToolName", "code_interpreter"],
            ["AWS/BedrockAgentCore", "ToolInvocationCount", "ToolName", "browser"]
          ]
          stat   = "Sum"
          period = 300
          view   = "timeSeries"
        }
      },

      # ── Row 3: Memory & logs ──
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 12
        height = 6
        properties = {
          title  = "Memory Read/Write Operations"
          region = var.aws_region
          metrics = [
            ["AWS/BedrockAgentCore", "MemoryReadCount", "MemoryName", aws_bedrockagentcore_memory.pipeline.name],
            ["AWS/BedrockAgentCore", "MemoryWriteCount", "MemoryName", aws_bedrockagentcore_memory.pipeline.name]
          ]
          stat   = "Sum"
          period = 300
          view   = "timeSeries"
        }
      },
      {
        type   = "log"
        x      = 12
        y      = 12
        width  = 12
        height = 6
        properties = {
          title  = "Recent Agent Logs"
          region = var.aws_region
          query  = "SOURCE '${aws_cloudwatch_log_group.agent_runtime.name}' | fields @timestamp, @message | sort @timestamp desc | limit 50"
          view   = "table"
        }
      }
    ]
  })
}

# ── ADOT Collector CloudWatch Log Group ───────────────────────

resource "aws_cloudwatch_log_group" "adot_collector" {
  name              = "/agentcore/${var.project_name}-${var.environment}/adot-collector"
  retention_in_days = var.log_retention_days
}