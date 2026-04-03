# ──────────────────────────────────────────────────────────────
# ECR Repository — agent runtime container image
# ──────────────────────────────────────────────────────────────

data "aws_ecr_repository" "agent" {
  name = "${var.project_name}-${var.environment}"
}

# Lifecycle policy — keep only the last 10 images
resource "aws_ecr_lifecycle_policy" "agent" {
  repository = data.aws_ecr_repository.agent.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Retain last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
