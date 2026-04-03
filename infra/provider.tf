# ──────────────────────────────────────────────────────────────
# Provider & Terraform version constraints
# ──────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.86.0"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.11"
    }
  }

  # Remote state — values provided via -backend-config in CI
  # or via infra/backend.hcl locally
  backend "s3" {
    # Partial configuration — these are injected at init time:
    #   bucket, key, region, dynamodb_table
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
