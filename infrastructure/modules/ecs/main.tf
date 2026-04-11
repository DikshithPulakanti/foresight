# Foresight — ECS Fargate Module
# Part of the Foresight AI Financial OS infrastructure
# github.com/DikshithPulakanti/foresight

# ── ECS Cluster ──────────────────────────────────────────────────────────────

resource "aws_ecs_cluster" "main" {
  name = "${var.project}-${var.env}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.project}-${var.env}-cluster"
  }
}

# ── IAM Role — Task Execution ────────────────────────────────────────────────

data "aws_iam_policy_document" "ecs_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_task_execution" {
  name               = "${var.project}-${var.env}-ecs-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json

  tags = {
    Name = "${var.project}-${var.env}-ecs-exec"
  }
}

resource "aws_iam_role_policy_attachment" "ecs_exec_policy" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy_attachment" "ecs_secrets_policy" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/SecretsManagerReadWrite"
}

# ── Security Group — ECS Tasks ───────────────────────────────────────────────

resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project}-${var.env}-ecs-tasks"
  description = "Allow inbound traffic to ECS tasks"
  vpc_id      = var.vpc_id

  ingress {
    description = "API port"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project}-${var.env}-ecs-tasks-sg"
  }
}

# ── CloudWatch Log Group ─────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project}-${var.env}"
  retention_in_days = 7

  tags = {
    Name = "${var.project}-${var.env}-logs"
  }
}

# ── Task Definition ──────────────────────────────────────────────────────────

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project}-${var.env}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.api_cpu
  memory                   = var.api_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([{
    name  = "foresight-api"
    image = "${var.ecr_repository_url}:latest"

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "APP_ENV", value = var.env },
      { name = "NEO4J_URI", value = var.neo4j_uri },
      { name = "POSTGRES_URL", value = var.postgres_url },
      { name = "REDIS_URL", value = var.redis_url },
    ]

    secrets = [
      { name = "ANTHROPIC_API_KEY", valueFrom = var.anthropic_secret_arn },
      { name = "PLAID_CLIENT_ID", valueFrom = var.plaid_secret_arn },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${var.project}-${var.env}"
        "awslogs-region"        = "us-east-1"
        "awslogs-stream-prefix" = "api"
      }
    }

    essential = true
  }])

  tags = {
    Name = "${var.project}-${var.env}-api-task"
  }
}

# ── ECS Service ──────────────────────────────────────────────────────────────

resource "aws_ecs_service" "api" {
  name            = "${var.project}-${var.env}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  depends_on = [aws_iam_role_policy_attachment.ecs_exec_policy]

  tags = {
    Name = "${var.project}-${var.env}-api-service"
  }
}
