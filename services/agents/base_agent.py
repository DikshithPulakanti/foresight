"""Base agent class — the foundation every Foresight agent inherits from.

Every agent in Foresight is a **LangGraph StateGraph** that communicates with
the outside world exclusively through MCP servers.  ``BaseAgent`` provides:

* A canonical ``AgentState`` schema shared by all agents.
* Automatic graph construction with an ``initialise → process → finalise``
  skeleton that subclasses extend by overriding ``define_nodes`` and
  ``define_edges``.
* Built-in MCP access through the global ``mcp_registry`` singleton, so
  subclasses never import individual MCP servers directly.
* Run-level observability: every execution is assigned a unique ``session_id``,
  timed, and logged.

Subclass contract
-----------------
1. Override ``define_nodes(builder)`` — add your domain-specific nodes.
2. Override ``define_edges(builder)`` — wire the nodes together.
3. Optionally override ``process(state)`` to inject shared pre-processing.

Architecture::

    ┌───────────────────────────────────────────────┐
    │  BaseAgent (ABC)                              │
    │  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
    │  │initialise │─▶│ process  │─▶│  finalise  │  │
    │  └──────────┘  └──────────┘  └────────────┘  │
    │       │              │              │         │
    │       ▼              ▼              ▼         │
    │          mcp_registry.call(server, tool, {})  │
    └───────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

try:
    from mcp_servers.registry import mcp_registry
except ImportError:
    try:
        from services.mcp_servers.registry import mcp_registry
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp-servers'))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp_servers'))
        from registry import mcp_registry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared state schema
# ------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Canonical state dictionary carried through every LangGraph node.

    All agents read and write to this structure.  Fields use ``total=False``
    so that nodes may set them incrementally.
    """

    user_id: str
    session_id: str
    input: dict[str, Any]
    output: dict[str, Any]
    status: str                # "running" | "completed" | "failed"
    error: Optional[str]
    steps: list[str]
    started_at: float
    metadata: dict[str, Any]


# ------------------------------------------------------------------
# Base agent
# ------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base for every Foresight LangGraph agent.

    Parameters
    ----------
    name:
        Machine-readable identifier used in logs and the ``agent_runs`` table
        (e.g. ``"spending_analyzer"``).
    description:
        Human-readable summary shown in the API ``/agents/status`` endpoint.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._graph = self._build_graph()
        self._compiled = self._graph.compile()
        logger.info("Agent '%s' ready", self.name)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Assemble the LangGraph ``StateGraph`` from the agent skeleton.

        The skeleton has three phases:

        1. **initialise** — stamps ``session_id``, ``started_at``, status.
        2. **process** — default no-op; subclasses override or add nodes.
        3. **finalise** — records completion time and status.

        Subclasses inject domain logic by overriding ``define_nodes`` and
        ``define_edges``.
        """
        builder = StateGraph(AgentState)

        builder.add_node("initialise", self._initialise)
        builder.add_node("process", self.process)
        builder.add_node("finalise", self._finalise)

        builder.set_entry_point("initialise")

        self.define_nodes(builder)
        self.define_edges(builder)

        if not self._has_custom_edges:
            builder.add_edge("initialise", "process")
            builder.add_edge("process", "finalise")
            builder.add_edge("finalise", END)

        return builder

    @property
    def _has_custom_edges(self) -> bool:
        """Return *True* if the subclass overrides ``define_edges``."""
        return type(self).define_edges is not BaseAgent.define_edges

    # ------------------------------------------------------------------
    # Lifecycle nodes (built-in)
    # ------------------------------------------------------------------

    @staticmethod
    def _initialise(state: AgentState) -> dict[str, Any]:
        return {
            "session_id": state.get("session_id") or str(uuid.uuid4()),
            "started_at": time.time(),
            "status": "running",
            "steps": [],
            "output": {},
            "error": None,
        }

    @staticmethod
    def _finalise(state: AgentState) -> dict[str, Any]:
        status = "failed" if state.get("error") else "completed"
        elapsed = time.time() - state.get("started_at", time.time())
        return {
            "status": status,
            "metadata": {
                **state.get("metadata", {}),
                "elapsed_seconds": round(elapsed, 3),
            },
        }

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def define_nodes(self, builder: StateGraph) -> None:
        """Register domain-specific nodes on *builder*.

        Called during ``__init__``.  Subclasses **must** implement this and
        add at least one node that performs the agent's core work.
        """

    def define_edges(self, builder: StateGraph) -> None:
        """Wire nodes together on *builder*.

        The default implementation connects
        ``initialise → process → finalise → END``.  Override to create
        branching, conditional routing, or cycles.
        """

    def process(self, state: AgentState) -> dict[str, Any]:
        """Default processing node — a no-op pass-through.

        Simple agents override this single method.  More complex agents add
        their own nodes in ``define_nodes`` and ignore ``process`` entirely.
        """
        return {}

    # ------------------------------------------------------------------
    # MCP helpers
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Call an MCP tool via the global registry and return its data.

        This is the **only** way agent nodes should interact with external
        services.  It ensures that every I/O operation is routed through MCP,
        keeping agents decoupled from implementation details.

        Raises
        ------
        RuntimeError
            If the tool invocation fails (``ToolResult.success is False``).
        """
        result = await mcp_registry.call(
            server_name,
            tool_name,
            params or {},
        )

        if not result.success:
            raise RuntimeError(
                f"MCP tool {server_name}/{tool_name} failed: {result.error}"
            )

        return result.data

    def available_tools(self, server_name: str) -> list[dict[str, Any]]:
        """List the tools exposed by a registered MCP server."""
        server = mcp_registry.get(server_name)
        return server.list_tools()

    def available_servers(self) -> list[str]:
        """Return the names of all registered MCP servers."""
        return mcp_registry.list_all()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @staticmethod
    def add_step(state: AgentState, step: str) -> dict[str, Any]:
        """Append a step label to the state's audit trail.

        Use inside a node to track progress::

            return {**self.add_step(state, "fetched transactions"), ...}
        """
        return {"steps": [*state.get("steps", []), step]}

    @staticmethod
    def set_error(state: AgentState, error: str) -> dict[str, Any]:
        """Mark the run as failed with an error message."""
        return {
            "error": error,
            "status": "failed",
            "steps": [*state.get("steps", []), f"ERROR: {error}"],
        }

    @staticmethod
    def _complete(state: AgentState, output: dict[str, Any]) -> dict[str, Any]:
        """Set the final output payload for the agent run.

        Typically called in the last domain node before ``finalise``::

            return {**self._complete(state, {"summary": "..."}),
                    **self.add_step(state, "done")}
        """
        return {"output": {**state.get("output", {}), **output}}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(
        self,
        user_id: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentState:
        """Execute the agent graph end-to-end.

        Parameters
        ----------
        user_id:
            The authenticated user this run belongs to.
        input_data:
            Arbitrary payload the first node can read from ``state["input"]``.
        metadata:
            Extra context (e.g. trigger source) persisted in state.

        Returns
        -------
        AgentState
            The final state after the graph reaches ``END``.
        """
        initial_state: AgentState = {
            "user_id": user_id,
            "session_id": str(uuid.uuid4()),
            "input": input_data or {},
            "output": {},
            "status": "pending",
            "error": None,
            "steps": [],
            "started_at": time.time(),
            "metadata": metadata or {},
        }

        logger.info(
            "Running agent '%s' for user %s (session %s)",
            self.name,
            user_id,
            initial_state["session_id"],
        )

        try:
            final_state = await self._compiled.ainvoke(initial_state)
        except Exception as exc:
            logger.exception("Agent '%s' crashed", self.name)
            final_state = {
                **initial_state,
                "status": "failed",
                "error": str(exc),
                "steps": [*initial_state["steps"], f"CRASH: {exc}"],
            }

        logger.info(
            "Agent '%s' finished — status=%s, steps=%d, elapsed=%.2fs",
            self.name,
            final_state.get("status"),
            len(final_state.get("steps", [])),
            time.time() - initial_state["started_at"],
        )

        return final_state

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
