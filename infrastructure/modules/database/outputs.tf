# Foresight — Database Module Outputs
# Part of the Foresight AI Financial OS infrastructure
# github.com/DikshithPulakanti/foresight

output "rds_endpoint" {
  description = "Endpoint of the Aurora PostgreSQL cluster"
  value       = aws_rds_cluster.main.endpoint
}

output "rds_port" {
  description = "Port of the Aurora PostgreSQL cluster"
  value       = aws_rds_cluster.main.port
}

output "redis_endpoint" {
  description = "Endpoint of the ElastiCache Redis cluster"
  value       = aws_elasticache_cluster.redis.cache_nodes[0].address
}

output "redis_port" {
  description = "Port of the ElastiCache Redis cluster"
  value       = aws_elasticache_cluster.redis.cache_nodes[0].port
}
