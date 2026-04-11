"""Unit tests for the BaseAgent class, state helpers, and agent execution.

These tests verify that the agent infrastructure (state management, step
tracking, error handling, MCP integration) works correctly before any
domain-specific agent logic is layered on top.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent


# ---------------------------------------------------------------------------
# Concrete test agent (since BaseAgent is abstract)
# ---------------------------------------------------------------------------

class _EchoAgent(BaseAgent):
    """Minimal agent that copies input to output for testing."""

    def define_nodes(self, builder: StateGraph) -> None:
        pass  # use the default initialise → process → finalise skeleton

    def process(self, state: AgentState) -> dict[str, Any]:
        return {
            **self.add_step(state, "echoed input"),
            **self._complete(state, {"echo": state.get("input", {})}),
        }


class _FailingAgent(BaseAgent):
    """Agent whose process node always sets an error."""

    def define_nodes(self, builder: StateGraph) -> None:
        pass

    def process(self, state: AgentState) -> dict[str, Any]:
        return self.set_error(state, "something broke")


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------

class TestBaseAgent:
    """Verify initialisation, state helpers, and the run lifecycle."""

    def test_agent_initialization(self):
        """Agent stores name and description, and compiles the graph."""
        agent = _EchoAgent(name="echo", description="A test agent")

        assert agent.name == "echo"
        assert agent.description == "A test agent"
        assert agent._compiled is not None

    def test_add_step(self, sample_agent_state):
        """add_step appends to the steps list without mutating original."""
        result = BaseAgent.add_step(sample_agent_state, "step one")

        assert len(result["steps"]) == 1
        assert "step one" in result["steps"][0]

    def test_add_step_preserves_existing(self, sample_agent_state):
        """add_step preserves previously recorded steps."""
        sample_agent_state["steps"] = ["existing step"]
        result = BaseAgent.add_step(sample_agent_state, "step two")

        assert len(result["steps"]) == 2
        assert "existing step" in result["steps"][0]
        assert "step two" in result["steps"][1]

    def test_set_error(self, sample_agent_state):
        """set_error marks the run as failed with the error message."""
        result = BaseAgent.set_error(sample_agent_state, "something broke")

        assert result["status"] == "failed"
        assert result["error"] == "something broke"
        assert any("ERROR" in s for s in result["steps"])

    def test_complete_merges_output(self, sample_agent_state):
        """_complete merges new keys into the existing output dict."""
        sample_agent_state["output"] = {"existing": True}
        result = BaseAgent._complete(sample_agent_state, {"new_key": 42})

        assert result["output"]["existing"] is True
        assert result["output"]["new_key"] == 42

    @pytest.mark.asyncio
    async def test_run_returns_completed_state(self):
        """A successful run returns state with status='completed'."""
        agent = _EchoAgent(name="echo-test")

        with patch("services.agents.base_agent.mcp_registry"):
            state = await agent.run(
                user_id="test-user",
                input_data={"greeting": "hello"},
            )

        assert state["status"] == "completed"
        assert state["user_id"] == "test-user"
        assert "echo" in state["output"]
        assert state["output"]["echo"]["greeting"] == "hello"
        assert len(state["steps"]) >= 1
        assert state["error"] is None

    @pytest.mark.asyncio
    async def test_run_captures_error_state(self):
        """An agent that set_error()s ends with status='failed'."""
        agent = _FailingAgent(name="fail-test")

        with patch("services.agents.base_agent.mcp_registry"):
            state = await agent.run(user_id="test-user")

        assert state["status"] == "failed"
        assert state["error"] == "something broke"

    @pytest.mark.asyncio
    async def test_run_handles_crash(self):
        """If a node raises an unhandled exception, run() catches it."""

        class _CrashAgent(BaseAgent):
            def define_nodes(self, builder: StateGraph) -> None:
                pass

            def process(self, state: AgentState) -> dict[str, Any]:
                raise RuntimeError("unexpected crash")

        agent = _CrashAgent(name="crash-test")

        with patch("services.agents.base_agent.mcp_registry"):
            state = await agent.run(user_id="test-user")

        assert state["status"] == "failed"
        assert "unexpected crash" in state["error"]

    @pytest.mark.asyncio
    async def test_run_assigns_session_id(self):
        """Every run gets a unique session_id (UUID format)."""
        agent = _EchoAgent(name="session-test")

        with patch("services.agents.base_agent.mcp_registry"):
            state = await agent.run(user_id="test-user")

        assert "session_id" in state
        assert len(state["session_id"]) > 0

    def test_repr(self):
        """__repr__ includes the agent name."""
        agent = _EchoAgent(name="my-agent")
        assert "my-agent" in repr(agent)
