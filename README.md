# Foresight

> Proactive AI financial operating system — see what's coming before it costs you.

![CI](https://github.com/DikshithPulakanti/foresight/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![AWS](https://img.shields.io/badge/deployed-AWS-orange)
![HuggingFace](https://img.shields.io/badge/model-HuggingFace-yellow)

Foresight monitors your finances 24/7 across bank accounts, email, and
calendar — warning you about financial problems **before** they happen,
not after the money is already gone.

## Demo

[Demo Video](YOUR_DEMO_URL) | [Live App](YOUR_APP_URL)

## What it does

- Detects duplicate charges, unusual spending, and forgotten subscriptions the moment they occur
- Predicts your exact account balance 30 and 60 days into the future
- Reads your inbox for renewal notices before they charge your card
- Responds to natural language voice queries: "How much did I spend on food this month?"
- Delivers a personalized 5-minute weekly audio financial briefing every Sunday

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Interfaces                    │
│         Next.js PWA · Voice (Whisper/TTS)            │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              12 LangGraph Agents                     │
│  Transaction Monitor · Subscription Auditor          │
│  Bill Negotiator · Cashflow Prophet · Receipt Scanner│
│  Email Monitor · Calendar Planner · Goal Tracker     │
│  Alert Sentinel · Voice Orchestrator                 │
│  Document Analyst · Advisor                          │
└──────┬─────────────────────────────┬────────────────┘
       │ call_tool()                 │ call_tool()
┌──────▼──────┐               ┌──────▼──────┐
│  6 Custom   │               │  6 Custom   │
│ MCP Servers │               │ MCP Servers │
│ plaid-mcp   │               │ gmail-mcp   │
│ graph-mcp   │               │calendar-mcp │
│ vision-mcp  │               │ voice-mcp   │
└──────┬──────┘               └──────┬──────┘
       │                             │
┌──────▼─────────────────────────────▼──────┐
│              Data Layer                    │
│  Neo4j · PostgreSQL · Redis · Qdrant       │
└────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI agents | LangGraph + Claude API (Anthropic) |
| MCP servers | 6 custom servers, 20 tools |
| Voice | Whisper STT + gTTS |
| Computer vision | Claude Vision API |
| Knowledge graph | Neo4j + GDS |
| Vector search | Qdrant |
| ML model | SpendingCategoryBERT (fine-tuned BERT, HuggingFace) |
| Backend | FastAPI + asyncpg + Redis |
| Frontend | Next.js 14 PWA + Tailwind |
| Cloud | AWS (ECS Fargate, Lambda, RDS Aurora, SageMaker) |
| IaC | Terraform — modular, dev + prod |
| CI/CD | GitHub Actions |
| Observability | LangSmith + MLflow + CloudWatch |

## Project structure

```
foresight/
├── services/
│   ├── api/              FastAPI backend — 5 routers
│   ├── mcp-servers/      6 custom MCP servers, 20 tools
│   └── agents/           12 LangGraph agents + orchestrator
├── frontend/             Next.js 14 PWA — 5 pages + voice
├── infrastructure/       Terraform — 5 modules, dev + prod
├── ml/                   Dataset generation + BERT fine-tuning
└── tests/                47 tests — unit + integration
```

## Getting started

```bash
git clone https://github.com/DikshithPulakanti/foresight.git
cd foresight
cp .env.example .env        # fill in API keys
docker-compose up -d        # start all services
curl localhost:8000/health  # verify API is running
cd frontend && npm install && npm run dev  # start frontend
```

## ML Model

SpendingCategoryBERT classifies raw bank transaction strings into 8
spending categories with high accuracy. Trained on 50,000 generated
examples using Claude Haiku.

```python
from transformers import pipeline
classifier = pipeline("text-classification",
                      model="DikshithPulakanti/SpendingCategoryBERT")
classifier("WHOLEFDS MKT #1523")
# [{'label': 'grocery', 'score': 0.98}]
```

Published on HuggingFace: [SpendingCategoryBERT](https://huggingface.co/DikshithPulakanti/SpendingCategoryBERT)

## The interview pitch

"Foresight is a proactive AI financial OS with 12 specialized LangGraph agents
that communicate exclusively through 6 custom MCP servers — Anthropic's protocol
for structured tool access. It predicts bank balance 60 days ahead, scans Gmail
for financial signals, reads receipt photos with Claude Vision, and delivers a
weekly audio briefing. SpendingCategoryBERT, my fine-tuned BERT model, is
published on HuggingFace. The whole system deploys on AWS via modular Terraform
with GitHub Actions CI/CD."

## Documentation

- [Architecture decisions](docs/architecture.md) — why MCP, LangGraph, Neo4j
- [Agent reference](docs/agents.md) — what each of the 12 agents does
- [MCP server reference](docs/mcp-servers.md) — 6 servers, 20 tools

## Author

Dikshith Pulakanti
[GitHub](https://github.com/DikshithPulakanti) ·
[LinkedIn](https://linkedin.com/in/DikshithPulakanti)
