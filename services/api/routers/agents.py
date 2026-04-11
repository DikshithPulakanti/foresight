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


class CalendarPlannerRequest(AgentRunRequest):
    """Body for the calendar-planner endpoint (extends the base request)."""
    days_ahead: int = 30


class EmailMonitorRequest(AgentRunRequest):
    """Body for the email-monitor endpoint (extends the base request)."""
    days_back: int = 7


class ReceiptScannerRequest(AgentRunRequest):
    """Body for the receipt-scanner endpoint (extends the base request)."""
    image_base64: str
    image_type: str | None = None


class DocumentAnalystRequest(AgentRunRequest):
    """Body for the document-analyst endpoint (extends the base request)."""
    image_base64: str
    document_type: str | None = None


class AlertSentinelRequest(AgentRunRequest):
    """Body for the alert-sentinel endpoint (extends the base request)."""
    dismissed_types: list[str] = []


class VoiceRequest(AgentRunRequest):
    """Body for the voice orchestrator endpoint.

    Either ``audio_base64`` or ``text_query`` must be provided.
    """
    audio_base64: str | None = None
    text_query: str | None = None
    audio_format: str = "wav"
    voice_speed: float = 1.0


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
    "/advisor/run",
    response_model=AgentRunResponse,
)
async def run_advisor(body: AgentRunRequest) -> AgentRunResponse:
    """Execute the Advisor agent for a given user.

    Runs four specialist agents in parallel (transaction monitor,
    subscription auditor, cashflow prophet, goal tracker), synthesises
    their outputs into a weekly summary with a 0-100 health score,
    writes a ~650-word briefing script optimised for spoken delivery,
    converts it to audio, and returns the full briefing.
    """
    return await _dispatch("advisor", body)


@router.post(
    "/alert-sentinel/run",
    response_model=AgentRunResponse,
)
async def run_alert_sentinel(body: AlertSentinelRequest) -> AgentRunResponse:
    """Execute the Alert Sentinel agent for a given user.

    Collects all pending alerts from Neo4j and Postgres, scores them on a
    0-100 urgency scale, deduplicates within a 6-hour window, selects up
    to 3 push notifications per run (critical alerts always bypass the
    budget), and returns push-ready payloads.
    """
    input_data: dict[str, Any] = {}
    if body.dismissed_types:
        input_data["dismissed_types"] = body.dismissed_types
    return await _dispatch("alert_sentinel", body, input_data=input_data)


@router.post(
    "/document-analyst/run",
    response_model=AgentRunResponse,
)
async def run_document_analyst(body: DocumentAnalystRequest) -> AgentRunResponse:
    """Execute the Document Analyst agent for a given user.

    Accepts a base64-encoded document image (lease, insurance policy,
    credit-card agreement, medical bill, etc.), classifies it, extracts
    structured data via Vision MCP, runs a document-type-specific deep
    risk analysis, creates alerts and calendar reminders, and returns a
    plain-English summary.
    """
    input_data: dict[str, Any] = {"image_base64": body.image_base64}
    if body.document_type is not None:
        input_data["document_type"] = body.document_type
    return await _dispatch("document_analyst", body, input_data=input_data)


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
    "/calendar-planner/run",
    response_model=AgentRunResponse,
)
async def run_calendar_planner(body: CalendarPlannerRequest) -> AgentRunResponse:
    """Execute the Calendar Planner agent for a given user.

    Looks ahead at upcoming calendar events, estimates their cost (using
    Claude when no amount is known), checks whether the budget can absorb
    each expense, and creates proactive calendar reminders for at-risk items.
    """
    return await _dispatch(
        "calendar_planner",
        body,
        input_data={"days_ahead": body.days_ahead},
    )


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
    "/goal-tracker/run",
    response_model=AgentRunResponse,
)
async def run_goal_tracker(body: AgentRunRequest) -> AgentRunResponse:
    """Execute the Goal Tracker agent for a given user.

    Fetches savings goals from the knowledge graph, calculates whether the
    user is on track for each one, generates spending-adjustment
    recommendations for goals that are behind, and returns an encouraging
    Claude-generated summary.
    """
    return await _dispatch("goal_tracker", body)


@router.post(
    "/voice/run",
    response_model=AgentRunResponse,
)
async def run_voice_orchestrator(body: VoiceRequest) -> AgentRunResponse:
    """Execute the Voice Orchestrator agent for a given user.

    Accepts either ``audio_base64`` (raw speech) or ``text_query`` (typed
    input).  The agent transcribes, classifies intent, routes to the
    appropriate data source or sub-agent, formulates a natural spoken
    response, and returns synthesised audio.
    """
    if not body.audio_base64 and not body.text_query:
        raise HTTPException(
            status_code=422,
            detail="Either audio_base64 or text_query must be provided",
        )

    input_data: dict[str, Any] = {"voice_speed": body.voice_speed}
    if body.audio_base64:
        input_data["audio_base64"] = body.audio_base64
        input_data["audio_format"] = body.audio_format
    if body.text_query:
        input_data["text_query"] = body.text_query
    return await _dispatch("voice_orchestrator", body, input_data=input_data)


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
