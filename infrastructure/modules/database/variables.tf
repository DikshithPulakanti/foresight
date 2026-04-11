# Foresight — Database Module Variables
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
  description = "ID of the VPC"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "List of private subnet IDs for database placement"
}

variable "db_password" {
  type        = string
  sensitive   = true
  description = "Master password for the Aurora PostgreSQL cluster"
}

variable "redis_node_type" {
  type        = string
  default     = "cache.t3.micro"
  description = "ElastiCache node type for Redis"
}
