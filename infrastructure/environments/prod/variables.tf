# Foresight — Prod Environment Variables
# Part of the Foresight AI Financial OS infrastructure
# github.com/DikshithPulakanti/foresight

variable "db_password" {
  type        = string
  sensitive   = true
  description = "Master password for the Aurora PostgreSQL cluster"
}

variable "neo4j_uri" {
  type        = string
  description = "Neo4j connection URI (bolt://host:7687)"
}

variable "anthropic_secret_arn" {
  type        = string
  description = "ARN of the Secrets Manager secret for Anthropic API key"
}

variable "plaid_secret_arn" {
  type        = string
  description = "ARN of the Secrets Manager secret for Plaid credentials"
}
