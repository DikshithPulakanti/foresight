"""Global MCP server registry (singleton).

The ``MCPRegistry`` acts as the single look-up table that the agent
orchestrator queries to discover and invoke tools across *all* MCP servers.

Typical startup flow::

    from mcp_servers.registry import mcp_registry
    from mcp_servers.plaid_mcp.server import PlaidMCPServer

    plaid = PlaidMCPServer()
    await plaid.setup()
    mcp_registry.register(plaid)

    # Later, from inside an agent:
    result = await mcp_registry.call("plaid", "get_transactions", {"days": 30})

The module-level ``mcp_registry`` instance is the singleton that should be
imported everywhere — never instantiate ``MCPRegistry`` directly.
"""

from __future__ import annotations

import logging
from typing import Any

from base import BaseMCPServer, ToolResult

logger = logging.getLogger(__name__)


class MCPRegistry:
    """Holds references to every active MCP server and dispatches tool calls.

    This class is designed as a **singleton** — import the module-level
    ``mcp_registry`` instance rather than creating your own.
    """

    def __init__(self) -> None:
        self._servers: dict[str, BaseMCPServer] = {}

    def register(self, server: BaseMCPServer) -> None:
        """Add a server to the registry.

        Args:
            server: A fully initialised (``setup()`` already called) MCP
                    server instance.

        Raises:
            ValueError: If a server with the same name is already registered.
        """
        if server.name in self._servers:
            raise ValueError(
                f"MCP server '{server.name}' is already registered"
            )
        self._servers[server.name] = server
        logger.info("Registered MCP server '%s' (%d tools)", server.name, len(server.list_tools()))

    def get(self, name: str) -> BaseMCPServer:
        """Retrieve a server by name.

        Args:
            name: The server's ``name`` attribute (e.g. ``"plaid"``).

        Raises:
            KeyError: If no server with that name has been registered.
        """
        if name not in self._servers:
            raise KeyError(f"MCP server '{name}' is not registered")
        return self._servers[name]

    def list_all(self) -> list[str]:
        """Return the names of every registered server."""
        return list(self._servers.keys())

    async def call(
        self,
        server_name: str,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """Dispatch a tool call to the appropriate server.

        This is the primary entry-point used by agent code. It resolves the
        server, delegates to ``BaseMCPServer.call_tool``, and returns the
        ``ToolResult``.

        Args:
            server_name: Registered name of the target MCP server.
            tool_name:   Tool to invoke on that server.
            params:      Parameters forwarded to the tool handler.

        Returns:
            A ``ToolResult`` with the outcome of the invocation.
        """
        server = self.get(server_name)
        return await server.call_tool(tool_name, params)


mcp_registry = MCPRegistry()
"""Module-level singleton — import this instead of creating new instances."""
