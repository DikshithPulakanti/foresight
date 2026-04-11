# Foresight — Prod Environment
# Part of the Foresight AI Financial OS infrastructure
# github.com/DikshithPulakanti/foresight

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.7.0"
}

provider "aws" {
  region = "us-east-1"
}

locals {
  project = "foresight"
  env     = "prod"
}

# ── Networking ───────────────────────────────────────────────────────────────

module "networking" {
  source  = "../../modules/networking"
  project = local.project
  env     = local.env
}

# ── Storage ──────────────────────────────────────────────────────────────────

module "storage" {
  source  = "../../modules/storage"
  project = local.project
  env     = local.env
}

# ── Database ─────────────────────────────────────────────────────────────────

module "database" {
  source             = "../../modules/database"
  project            = local.project
  env                = local.env
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  db_password        = var.db_password
  redis_node_type    = "cache.t3.small"
}

# ── ECS ──────────────────────────────────────────────────────────────────────

module "ecs" {
  source               = "../../modules/ecs"
  project              = local.project
  env                  = local.env
  vpc_id               = module.networking.vpc_id
  private_subnet_ids   = module.networking.private_subnet_ids
  ecr_repository_url   = module.storage.ecr_repository_url
  api_cpu              = "1024"
  api_memory           = "2048"
  desired_count        = 2
  neo4j_uri            = var.neo4j_uri
  postgres_url         = "postgresql://foresight_admin:${var.db_password}@${module.database.rds_endpoint}:5432/foresight"
  redis_url            = "redis://${module.database.redis_endpoint}:6379"
  anthropic_secret_arn = var.anthropic_secret_arn
  plaid_secret_arn     = var.plaid_secret_arn
}
