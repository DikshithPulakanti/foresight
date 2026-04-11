# Foresight — Networking Module Variables
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

variable "region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region for availability zone selection"
}
