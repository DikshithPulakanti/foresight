# Foresight — ECS Module Variables
# Part of the Foresight AI Financial OS infrastructure
# github.com/DikshithPulakanti/foresight

variable "project" {
  type        = string
  description = "Project name used for resource naming"
}

variable "env" {
  type        = string
  description = "Environment name (dev, prod)"
}

variable "vpc_id" {
  type        = string
  description = "ID of the VPC to deploy into"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "List of private subnet IDs for ECS tasks"
}

variable "ecr_repository_url" {
  type        = string
  description = "URL of the ECR repository for the API image"
}

variable "api_cpu" {
  type        = string
  default     = "512"
  description = "CPU units for the API task (1024 = 1 vCPU)"
}

variable "api_memory" {
  type        = string
  default     = "1024"
  description = "Memory in MiB for the API task"
}

variable "desired_count" {
  type        = number
  default     = 1
  description = "Number of API task instances to run"
}

variable "neo4j_uri" {
  type        = string
  description = "Neo4j connection URI (bolt://host:7687)"
}

variable "postgres_url" {
  type        = string
  description = "PostgreSQL connection URL"
  sensitive   = true
}

variable "redis_url" {
  type        = string
  description = "Redis connection URL"
}

variable "anthropic_secret_arn" {
  type        = string
  description = "ARN of the Secrets Manager secret for Anthropic API key"
}

variable "plaid_secret_arn" {
  type        = string
  description = "ARN of the Secrets Manager secret for Plaid credentials"
}
