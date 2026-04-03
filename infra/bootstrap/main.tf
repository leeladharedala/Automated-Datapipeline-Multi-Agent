# ──────────────────────────────────────────────────────────────
# Bootstrap — State Infrastructure
#
# Run this ONCE to create the S3 bucket and DynamoDB table
# that store Terraform remote state for the main infra.
#
# Usage:
#   cd infra/bootstrap
#   terraform init
#   terraform apply -var="aws_region=us-west-2" -var="project_name=multi-agent-pipeline"
#
# This is intentionally separate and uses LOCAL state.
# ──────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.90"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "us-west-2"
}

variable "project_name" {
  type    = string
  default = "multi-agent-pipeline"
}

provider "aws" {
  region = var.aws_region
}

# ── S3 Bucket for Terraform State ─────────────────────────────

resource "aws_s3_bucket" "terraform_state" {
  bucket = "${var.project_name}-tfstate-${data.aws_caller_identity.current.account_id}"

  # Prevent accidental deletion
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── DynamoDB Table for State Locking ──────────────────────────

resource "aws_dynamodb_table" "terraform_locks" {
  name         = "${var.project_name}-tfstate-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

# ── Data & Outputs ────────────────────────────────────────────

data "aws_caller_identity" "current" {}

output "state_bucket_name" {
  description = "S3 bucket name for Terraform state — use in backend config"
  value       = aws_s3_bucket.terraform_state.id
}

output "lock_table_name" {
  description = "DynamoDB table name for state locking — use in backend config"
  value       = aws_dynamodb_table.terraform_locks.name
}

output "backend_config" {
  description = "Paste this into your backend config or pass via -backend-config"
  value       = <<-EOT
    bucket         = "${aws_s3_bucket.terraform_state.id}"
    key            = "${var.project_name}/terraform.tfstate"
    region         = "${var.aws_region}"
    dynamodb_table = "${aws_dynamodb_table.terraform_locks.name}"
    encrypt        = true
  EOT
}
