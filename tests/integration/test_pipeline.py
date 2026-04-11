"""Integration tests for the MCP registry + BaseAgent pipeline.

These tests verify that the plumbing between the MCPRegistry, individual
MCP servers, and BaseAgent.call_tool() works end-to-end — ensuring that
an agent can discover, invoke, and receive results from MCP tools.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from langgraph.graph import StateGraph

from base import BaseMCPServer, ToolResult
from registry import MCPRegistry
from services.agents.base_agent import AgentState, BaseAgent


# ---------------------------------------------------------------------------
# Test MCP server
# ---------------------------------------------------------------------------

class _TestServer(BaseMCPServer):
    """MCP server with a single tool that returns test data."""

    async def setup(self) -> None:
        self.register_tool(
            name="get_balance",
            description="Get account balance",
            input_schema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
            handler=self._get_balance,
        )

    @staticmethod
    async def _get_balance(params: dict) -> dict:
        return {"user_id": params["user_id"], "balance": 4250.00}


# ---------------------------------------------------------------------------
# MCP Registry integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_registry_call_routing():
    """Registry correctly routes a call to the right server and tool,
    returning a ToolResult with the expected data."""
    registry = MCPRegistry()
    server = _TestServer("banking")
    await server.setup()
    registry.register(server)

    result = await registry.call("banking", "get_balance", {"user_id": "u1"})

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data["balance"] == 4250.00
    assert result.data["user_id"] == "u1"
    assert result.execution_time_ms >= 0


@pytest.mark.asyncio
async def test_registry_call_to_missing_server_raises():
    """Calling a server that isn't registered raises KeyError."""
    registry = MCPRegistry()

    with pytest.raises(KeyError, match="not registered"):
        await registry.call("ghost", "any_tool", {})


# ---------------------------------------------------------------------------
# Agent → MCP pipeline integration
# ---------------------------------------------------------------------------

class _BalanceAgent(BaseAgent):
    """Agent that fetches a balance via MCP in its process node."""

    def define_nodes(self, builder: StateGraph) -> None:
        pass

    def process(self, state: AgentState) -> dict[str, Any]:
        # Synchronous node — we can't await here in the default skeleton,
        # so we test the MCP wiring via the run() path instead
        return {
            **self.add_step(state, "process executed"),
            **self._complete(state, {"processed": True}),
        }


@pytest.mark.asyncio
async def test_agent_calls_mcp_through_base():
    """An agent can call an MCP tool via the inherited call_tool method,
    and the result flows through the registry to the correct server."""
    # Set up a real MCP registry with our test server
    registry = MCPRegistry()
    server = _TestServer("banking")
    await server.setup()
    registry.register(server)

    agent = _BalanceAgent(name="balance-agent")

    with patch("services.agents.base_agent.mcp_registry", registry):
        # Verify call_tool works through the base class
        data = await agent.call_tool("banking", "get_balance", {"user_id": "u1"})

        assert data["balance"] == 4250.00

        # Verify the agent can also list servers and tools
        assert "banking" in agent.available_servers()
        tools = agent.available_tools("banking")
        assert len(tools) == 1
        assert tools[0]["name"] == "get_balance"


@pytest.mark.asyncio
async def test_agent_mcp_failure_raises_runtime_error():
    """When an MCP tool fails, call_tool raises RuntimeError."""
    registry = MCPRegistry()
    server = _TestServer("banking")
    await server.setup()
    registry.register(server)

    agent = _BalanceAgent(name="fail-agent")

    with patch("services.agents.base_agent.mcp_registry", registry):
        with pytest.raises(RuntimeError, match="failed"):
            await agent.call_tool("banking", "nonexistent_tool", {})


@pytest.mark.asyncio
async def test_full_agent_run_with_mocked_registry():
    """A full agent.run() invocation completes successfully with a
    patched MCP registry, producing a valid final state."""
    agent = _BalanceAgent(name="full-run-test")

    with patch("services.agents.base_agent.mcp_registry"):
        state = await agent.run(
            user_id="test-user",
            input_data={"account": "checking"},
        )

    assert state["status"] == "completed"
    assert state["output"]["processed"] is True
    assert any("process executed" in s for s in state["steps"])
    assert state["error"] is None
