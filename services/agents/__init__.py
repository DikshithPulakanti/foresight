"""Foresight AI agent package — LangGraph-based agents backed by MCP servers.

Concrete agent classes and the orchestrator are imported lazily to avoid
circular imports (advisor → orchestrator → advisor).
"""

from services.agents.base_agent import AgentState, BaseAgent

__all__ = [
    "AgentState",
    "BaseAgent",
]


def __getattr__(name: str):
    """Lazy-import concrete agents and the orchestrator on first access."""
    _lazy = {
        "AdvisorAgent": "services.agents.advisor.agent",
        "AlertSentinelAgent": "services.agents.alert_sentinel.agent",
        "BillNegotiatorAgent": "services.agents.bill_negotiator.agent",
        "CalendarPlannerAgent": "services.agents.calendar_planner.agent",
        "CashflowProphetAgent": "services.agents.cashflow_prophet.agent",
        "DocumentAnalystAgent": "services.agents.document_analyst.agent",
        "EmailMonitorAgent": "services.agents.email_monitor.agent",
        "GoalTrackerAgent": "services.agents.goal_tracker.agent",
        "ReceiptScannerAgent": "services.agents.receipt_scanner.agent",
        "SubscriptionAuditorAgent": "services.agents.subscription_auditor.agent",
        "TransactionMonitorAgent": "services.agents.transaction_monitor.agent",
        "VoiceOrchestratorAgent": "services.agents.voice_orchestrator.agent",
        "AgentOrchestrator": "services.agents.orchestrator",
        "agent_orchestrator": "services.agents.orchestrator",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
