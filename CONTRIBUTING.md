# Contributing to Foresight

## Branching strategy
- main — production only, protected
- feat/your-feature — all new work goes on a branch
- Never push directly to main

## Commit message format
- feat: new feature
- fix: bug fix
- chore: maintenance
- docs: documentation
- test: adding tests

## Running locally
    cp .env.example .env
    docker-compose up -d
    curl http://localhost:8000/health

## Before opening a PR
- Run: make lint
- Run: make test
- Add tests for new functionality
