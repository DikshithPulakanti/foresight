"""Unit tests for the MCP server base class, tool registration, and registry.

These tests verify the core infrastructure that all 6 MCP servers rely on:
tool discovery, invocation, parameter validation, timing, and error handling.
"""

from __future__ import annotations

import pytest

from base import BaseMCPServer, ToolResult
from registry import MCPRegistry


# ---------------------------------------------------------------------------
# Concrete test server (since BaseMCPServer is abstract)
# ---------------------------------------------------------------------------

class _StubServer(BaseMCPServer):
    """Minimal concrete server for testing the base class."""

    async def setup(self) -> None:
        self.register_tool(
            name="echo",
            description="Returns params as-is",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            handler=self._echo,
        )

    @staticmethod
    async def _echo(params: dict) -> dict:
        return {"echoed": params.get("message")}


class _FailingServer(BaseMCPServer):
    """Server whose tool always raises."""

    async def setup(self) -> None:
        self.register_tool(
            name="boom",
            description="Always fails",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=self._boom,
        )

    @staticmethod
    async def _boom(_params: dict) -> None:
        raise ValueError("bad input")


# ---------------------------------------------------------------------------
# BaseMCPServer tests
# ---------------------------------------------------------------------------

class TestBaseMCPServer:
    """Verify tool registration, listing, invocation, and validation."""

    def test_tool_registration(self):
        """Registering a tool makes it appear in list_tools with correct keys."""
        server = _StubServer("test")
        # setup() hasn't been called yet — no tools registered
        assert server.list_tools() == []

    @pytest.mark.asyncio
    async def test_tool_registration_after_setup(self):
        """After setup(), the registered tool appears in list_tools."""
        server = _StubServer("test")
        await server.setup()

        tools = server.list_tools()
        assert len(tools) == 1

        tool = tools[0]
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
        assert tool["name"] == "echo"

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """Calling a non-existent tool returns a failed ToolResult."""
        server = _StubServer("test")
        await server.setup()

        result = await server.call_tool("nonexistent", {})

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_tool_execution_success(self):
        """A successful tool call returns data and records execution time."""
        server = _StubServer("test")
        await server.setup()

        result = await server.call_tool("echo", {"message": "hello"})

        assert result.success is True
        assert result.data == {"echoed": "hello"}
        assert result.execution_time_ms > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_tool_execution_failure(self):
        """A tool that raises returns a failed ToolResult with the error message."""
        server = _FailingServer("fail-server")
        await server.setup()

        result = await server.call_tool("boom", {})

        assert result.success is False
        assert "bad input" in result.error
        assert result.execution_time_ms > 0

    def test_validate_params_missing_required(self):
        """Validation fails when a required parameter is absent."""
        server = _StubServer("test")
        # Manually register to avoid async setup in a sync test
        server.register_tool(
            name="need_id",
            description="Needs user_id",
            input_schema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
            handler=lambda p: p,
        )

        valid, msg = server.validate_params("need_id", {})

        assert valid is False
        assert "user_id" in msg

    def test_validate_params_success(self):
        """Validation passes when all required parameters are provided."""
        server = _StubServer("test")
        server.register_tool(
            name="need_id",
            description="Needs user_id",
            input_schema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
            handler=lambda p: p,
        )

        valid, msg = server.validate_params("need_id", {"user_id": "abc"})

        assert valid is True

    def test_validate_params_unknown_tool(self):
        """Validation fails gracefully for a tool that doesn't exist."""
        server = _StubServer("test")
        valid, msg = server.validate_params("ghost", {})
        assert valid is False
        assert "not found" in msg


# ---------------------------------------------------------------------------
# MCPRegistry tests
# ---------------------------------------------------------------------------

class TestMCPRegistry:
    """Verify server registration, lookup, and duplicate detection."""

    @pytest.mark.asyncio
    async def test_register_and_get(self):
        """A registered server can be retrieved by name."""
        registry = MCPRegistry()
        server = _StubServer("plaid")
        await server.setup()

        registry.register(server)
        retrieved = registry.get("plaid")

        assert retrieved is server
        assert retrieved.name == "plaid"

    @pytest.mark.asyncio
    async def test_duplicate_registration_raises(self):
        """Registering the same server name twice raises ValueError."""
        registry = MCPRegistry()
        server1 = _StubServer("plaid")
        server2 = _StubServer("plaid")
        await server1.setup()
        await server2.setup()

        registry.register(server1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(server2)

    @pytest.mark.asyncio
    async def test_list_all(self):
        """list_all returns names of every registered server."""
        registry = MCPRegistry()
        for name in ["plaid", "gmail", "calendar"]:
            server = _StubServer(name)
            await server.setup()
            registry.register(server)

        names = registry.list_all()

        assert sorted(names) == ["calendar", "gmail", "plaid"]

    def test_get_unregistered_raises(self):
        """Getting a server that doesn't exist raises KeyError."""
        registry = MCPRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    @pytest.mark.asyncio
    async def test_call_dispatches_to_server(self):
        """registry.call() dispatches to the correct server and tool."""
        registry = MCPRegistry()
        server = _StubServer("echo-server")
        await server.setup()
        registry.register(server)

        result = await registry.call("echo-server", "echo", {"message": "hi"})

        assert result.success is True
        assert result.data == {"echoed": "hi"}
