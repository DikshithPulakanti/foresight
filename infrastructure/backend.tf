# Foresight — Terraform Backend
# Part of the Foresight AI Financial OS infrastructure
# github.com/DikshithPulakanti/foresight

terraform {
  backend "s3" {
    bucket         = "foresight-terraform-state"
    key            = "foresight/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "foresight-terraform-locks"
    encrypt        = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  required_version = ">= 1.7.0"
}
