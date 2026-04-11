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


class EmailMonitorRequest(AgentRunRequest):
    """Body for the email-monitor endpoint (extends the base request)."""
    days_back: int = 7


class ReceiptScannerRequest(AgentRunRequest):
    """Body for the receipt-scanner endpoint (extends the base request)."""
    image_base64: str
    image_type: str | None = None


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


@router.post(
    "/cashflow-prophet/run",
    response_model=AgentRunResponse,
)
async def run_cashflow_prophet(body: AgentRunRequest) -> AgentRunResponse:
    """Execute the Cashflow Prophet agent for a given user.

    Predicts bank balance 30 and 60 days out using time-series forecasting
    across historical cashflow, recurring charges, payday schedule, and
    calendar events.  Fires proactive alerts when a shortfall is projected.
    """
    return await _dispatch("cashflow_prophet", body)


@router.post(
    "/email-monitor/run",
    response_model=AgentRunResponse,
)
async def run_email_monitor(body: EmailMonitorRequest) -> AgentRunResponse:
    """Execute the Email Monitor agent for a given user.

    Scans Gmail for financial signals (bills due, renewals, price increases,
    overdue notices), classifies urgency, extracts action items, creates
    knowledge-graph alerts, and returns a concise digest.
    """
    return await _dispatch(
        "email_monitor",
        body,
        input_data={"days_back": body.days_back},
    )


@router.post(
    "/receipt-scanner/run",
    response_model=AgentRunResponse,
)
async def run_receipt_scanner(body: ReceiptScannerRequest) -> AgentRunResponse:
    """Execute the Receipt Scanner agent for a given user.

    Accepts a base64-encoded receipt photo, extracts merchant/items/total
    via Claude Vision, categorises the expense, stores it in the knowledge
    graph, and returns a confirmation message.
    """
    input_data: dict[str, Any] = {"image_base64": body.image_base64}
    if body.image_type is not None:
        input_data["image_type"] = body.image_type
    return await _dispatch("receipt_scanner", body, input_data=input_data)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

async def _dispatch(
    agent_name: str,
    body: AgentRunRequest,
    *,
    input_data: dict[str, Any] | None = None,
) -> AgentRunResponse:
    """Run an agent by name and wrap the result in ``AgentRunResponse``."""
    try:
        result = await agent_orchestrator.run_agent(
            agent_name,
            user_id=body.user_id,
            input_data=input_data or {},
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
