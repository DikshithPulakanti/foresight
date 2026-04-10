# Foresight

> Proactive AI financial operating system — see what's coming before it costs you.

![CI](https://github.com/DikshithPulakanti/foresight/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![AWS](https://img.shields.io/badge/deployed-AWS-orange)

Foresight monitors your finances 24/7 across bank accounts, email, and calendar — warning you about problems before they happen.

## What it does

- Detects duplicate charges, forgotten subscriptions, and unusual spending
- Predicts your account balance 30 and 60 days into the future
- Reads your inbox for renewal notices before they hit your card
- Responds to voice queries in natural language
- Generates a weekly audio financial briefing

## Architecture

| Layer | Tech |
|---|---|
| Agents | LangGraph — 12 specialized agents |
| MCP Servers | 6 custom servers (Plaid, Gmail, Calendar, Graph, Vision, Voice) |
| AI | Claude API + Whisper STT + Kokoro TTS |
| Knowledge graph | Neo4j |
| Vector search | Qdrant |
| ML model | SpendingCategoryBERT (HuggingFace) |
| Backend | FastAPI + asyncpg + Redis |
| Frontend | Next.js 14 PWA + voice interface |
| Cloud | AWS — ECS Fargate, Lambda, SageMaker, RDS |
| IaC | Terraform — modular, dev + prod |
| CI/CD | GitHub Actions |

## Getting started

    cp .env.example .env
    docker-compose up -d
    curl http://localhost:8000/health

## Project structure

    foresight/
    services/
        api/              FastAPI backend
        mcp-servers/      6 custom MCP servers
        agents/           12 LangGraph agents
    frontend/             Next.js PWA
    infrastructure/       Terraform 9 modules
    ml/                   Fine-tuning pipeline
    tests/                Unit + integration + e2e
    docs/                 Architecture docs

## Author

Dikshith Pulakanti — https://github.com/DikshithPulakanti
