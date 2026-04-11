# Foresight — ECS Module Outputs
# Part of the Foresight AI Financial OS infrastructure
# github.com/DikshithPulakanti/foresight

output "cluster_id" {
  description = "ID of the ECS cluster"
  value       = aws_ecs_cluster.main.id
}

output "cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

output "service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.api.name
}

output "task_definition_arn" {
  description = "ARN of the task definition"
  value       = aws_ecs_task_definition.api.arn
}
