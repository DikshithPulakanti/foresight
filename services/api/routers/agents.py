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
    return await _dispatch("transaction_monitor", body)


@router.post(
    "/subscription-auditor/run",
    response_model=AgentRunResponse,
)
async def run_subscription_auditor(body: AgentRunRequest) -> AgentRunResponse:
    """Execute the Subscription Auditor agent for a given user.

    Cross-references recurring bank charges (Plaid) with inbox receipts
    (Gmail), deduplicates via fuzzy matching, and returns a unified
    subscription inventory with cancellation/savings opportunities.
    """
    return await _dispatch("subscription_auditor", body)


@router.post(
    "/bill-negotiator/run",
    response_model=AgentRunResponse,
)
async def run_bill_negotiator(body: AgentRunRequest) -> AgentRunResponse:
    """Execute the Bill Negotiator agent for a given user.

    Identifies bills the user is overpaying for (internet, phone, insurance,
    cable), researches competitor pricing, and returns ready-to-use phone
    negotiation scripts with estimated savings.
    """
    return await _dispatch("bill_negotiator", body)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

async def _dispatch(agent_name: str, body: AgentRunRequest) -> AgentRunResponse:
    """Run an agent by name and wrap the result in ``AgentRunResponse``."""
    try:
        result = await agent_orchestrator.run_agent(
            agent_name,
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
