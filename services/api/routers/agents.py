"""AI agent orchestration endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.agents.orchestrator import agent_orchestrator

router = APIRouter(prefix="/agents", tags=["agents"])


# ------------------------------------------------------------------
# Shared schemas
# ------------------------------------------------------------------

class AgentRunRequest(BaseModel):
    """Body for triggering an agent run."""
    user_id: str


class AgentRunResponse(BaseModel):
    """Envelope returned after an agent completes."""
    status: str
    session_id: str
    output: dict[str, Any]
    steps: list[str]
    error: str | None = None


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.get("/status")
async def agent_status() -> dict[str, Any]:
    """Return health and metadata for every registered agent."""
    return {
        "status": "ok",
        "agents": agent_orchestrator.list_agents(),
    }


@router.post(
    "/transaction-monitor/run",
    response_model=AgentRunResponse,
)
async def run_transaction_monitor(body: AgentRunRequest) -> AgentRunResponse:
    """Execute the Transaction Monitor agent for a given user.

    The agent fetches recent transactions from Plaid, analyses them for
    anomalies (duplicates, large purchases, unusual amounts, new merchants),
    creates alerts in the knowledge graph, and returns a Claude-generated
    summary.
    """
    try:
        result = await agent_orchestrator.run_agent(
            "transaction_monitor",
            user_id=body.user_id,
            input_data={},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return AgentRunResponse(
        status=result.get("status", "unknown"),
        session_id=result.get("session_id", ""),
        output=result.get("output", {}),
        steps=result.get("steps", []),
        error=result.get("error"),
    )
