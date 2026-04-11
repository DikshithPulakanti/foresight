"""Agent orchestrator — central registry and dispatcher for all Foresight agents.

The orchestrator is a singleton that owns every ``BaseAgent`` instance.  API
routes and background schedulers interact with agents exclusively through
``agent_orchestrator.run_agent(name, user_id, input_data)``.

On import, every shipped agent is instantiated and registered so that the
rest of the application never has to know which concrete classes exist.
"""

from __future__ import annotations

import logging
from typing import Any

from services.agents.base_agent import AgentState, BaseAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Holds references to every agent and dispatches ``run`` calls by name.

    Usage::

        from services.agents.orchestrator import agent_orchestrator

        result = await agent_orchestrator.run_agent(
            "transaction_monitor", user_id="abc-123", input_data={}
        )
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Add an agent to the orchestrator.

        Raises
        ------
        ValueError
            If an agent with the same ``name`` is already registered.
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' is already registered")
        self._agents[agent.name] = agent
        logger.info(
            "Registered agent '%s' (%s)",
            agent.name,
            type(agent).__name__,
        )

    def get(self, name: str) -> BaseAgent:
        """Look up a registered agent by name.

        Raises
        ------
        KeyError
            If no agent with that name has been registered.
        """
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' is not registered")
        return self._agents[name]

    def list_agents(self) -> list[dict[str, str]]:
        """Return metadata for every registered agent."""
        return [
            {
                "name": agent.name,
                "description": agent.description,
                "type": type(agent).__name__,
            }
            for agent in self._agents.values()
        ]

    async def run_agent(
        self,
        name: str,
        user_id: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentState:
        """Execute an agent by name and return its final state.

        Parameters
        ----------
        name:
            Registered agent name (e.g. ``"transaction_monitor"``).
        user_id:
            The authenticated user this run belongs to.
        input_data:
            Arbitrary payload forwarded to the agent's initial state.
        metadata:
            Extra context persisted alongside the run.

        Returns
        -------
        AgentState
            The terminal state produced by the agent graph.
        """
        agent = self.get(name)
        logger.info("Orchestrator dispatching '%s' for user %s", name, user_id)
        return await agent.run(user_id, input_data, metadata)


# ------------------------------------------------------------------
# Module-level singleton & agent registration
# ------------------------------------------------------------------

agent_orchestrator = AgentOrchestrator()
"""Module-level singleton — import this, never instantiate directly."""

# Register every shipped agent on first import.
from services.agents.bill_negotiator.agent import BillNegotiatorAgent  # noqa: E402
from services.agents.cashflow_prophet.agent import CashflowProphetAgent  # noqa: E402
from services.agents.subscription_auditor.agent import SubscriptionAuditorAgent  # noqa: E402
from services.agents.transaction_monitor.agent import TransactionMonitorAgent  # noqa: E402

agent_orchestrator.register(BillNegotiatorAgent())
agent_orchestrator.register(CashflowProphetAgent())
agent_orchestrator.register(SubscriptionAuditorAgent())
agent_orchestrator.register(TransactionMonitorAgent())
