# ──────────────────────────────────────────────────────────────
# Secrets Manager — Anthropic API key
#
# Flow:
#   1. Terraform creates the secret shell with a placeholder.
#   2. GitHub Actions CI/CD populates the real value:
#        aws secretsmanager put-secret-value \
#          --secret-id <secret_id> \
#          --secret-string "${{ secrets.ANTHROPIC_API_KEY }}"
#   3. AgentCore runtime reads the secret at boot via its ARN.
# ──────────────────────────────────────────────────────────────

resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name        = "${var.project_name}-${var.environment}/anthropic-api-key"
  description = "Anthropic API key for the multi-agent pipeline. Populated by GitHub Actions CI/CD."

  recovery_window_in_days = var.environment == "prod" ? 30 : 0
}

resource "aws_secretsmanager_secret_version" "anthropic_api_key" {
  secret_id     = aws_secretsmanager_secret.anthropic_api_key.id
  secret_string = "CHANGE_ME"

  # GitHub Actions overwrites this value — don't let Terraform revert it
  lifecycle {
    ignore_changes = [secret_string]
  }
}

# ── GitHub PAT ────────────────────────────────────────────────
#
# Same flow as the Anthropic key:
#   1. Terraform creates the secret shell in the deployment region.
#   2. GitHub Actions populates the real value:
#        aws secretsmanager put-secret-value \
#          --secret-id <secret_id> \
#          --secret-string "${{ secrets.GH_PAT }}"
#   3. AgentCore runtime reads the secret at boot via its ARN.
# ──────────────────────────────────────────────────────────────

resource "aws_secretsmanager_secret" "github_pat" {
  name        = "${var.project_name}-${var.environment}/github-pat"
  description = "GitHub PAT for the submit_pr tool. Populated by GitHub Actions CI/CD."

  recovery_window_in_days = var.environment == "prod" ? 30 : 0
}

resource "aws_secretsmanager_secret_version" "github_pat" {
  secret_id     = aws_secretsmanager_secret.github_pat.id
  secret_string = "CHANGE_ME"

  lifecycle {
    ignore_changes = [secret_string]
  }
}
