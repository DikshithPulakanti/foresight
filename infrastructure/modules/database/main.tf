# Foresight — Database Module
# Part of the Foresight AI Financial OS infrastructure
# github.com/DikshithPulakanti/foresight

# ── RDS Subnet Group ─────────────────────────────────────────────────────────

resource "aws_db_subnet_group" "main" {
  name       = "${var.project}-${var.env}-db-subnet"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name = "${var.project}-${var.env}-db-subnet"
  }
}

# ── Security Group — RDS ─────────────────────────────────────────────────────

resource "aws_security_group" "rds" {
  name        = "${var.project}-${var.env}-rds"
  description = "Allow PostgreSQL access from ECS tasks"
  vpc_id      = var.vpc_id

  ingress {
    description = "PostgreSQL from VPC"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project}-${var.env}-rds-sg"
  }
}

# ── Aurora PostgreSQL Cluster ────────────────────────────────────────────────

resource "aws_rds_cluster" "main" {
  cluster_identifier = "${var.project}-${var.env}-aurora"
  engine             = "aurora-postgresql"
  engine_version     = "15.4"
  database_name      = "foresight"
  master_username    = "foresight_admin"
  master_password    = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  skip_final_snapshot     = var.env == "dev"
  backup_retention_period = 7

  tags = {
    Name = "${var.project}-${var.env}-aurora"
  }
}

resource "aws_rds_cluster_instance" "main" {
  identifier         = "${var.project}-${var.env}-aurora-1"
  cluster_identifier = aws_rds_cluster.main.id
  instance_class     = "db.t3.medium"
  engine             = aws_rds_cluster.main.engine
  engine_version     = aws_rds_cluster.main.engine_version

  tags = {
    Name = "${var.project}-${var.env}-aurora-instance-1"
  }
}

# ── ElastiCache Subnet Group ────────────────────────────────────────────────

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project}-${var.env}-redis-subnet"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name = "${var.project}-${var.env}-redis-subnet"
  }
}

# ── Security Group — Redis ───────────────────────────────────────────────────

resource "aws_security_group" "redis" {
  name        = "${var.project}-${var.env}-redis"
  description = "Allow Redis access from ECS tasks"
  vpc_id      = var.vpc_id

  ingress {
    description = "Redis from VPC"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project}-${var.env}-redis-sg"
  }
}

# ── ElastiCache Redis ────────────────────────────────────────────────────────

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.project}-${var.env}-redis"
  engine               = "redis"
  node_type            = var.redis_node_type
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]

  tags = {
    Name = "${var.project}-${var.env}-redis"
  }
}
