# ──────────────────────────────────────────────────────────────
# S3 Bucket — raw data storage
# ──────────────────────────────────────────────────────────────

resource "aws_s3_bucket" "raw_data" {
  bucket        = "${var.project_name}-${var.environment}-raw-data"
  force_destroy = var.environment != "prod"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
