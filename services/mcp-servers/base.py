"""Base MCP (Model Context Protocol) server infrastructure for Foresight.

What is MCP?
============
The **Model Context Protocol** is Anthropic's open standard that defines how AI
models (like Claude) communicate with external tools and data sources.  Instead
of hard-coding tool calls inside application code, MCP provides a uniform JSON
interface that any compliant model can discover and invoke:

1. **Tool Discovery** – the model calls ``list_tools()`` to learn which
   capabilities a server exposes, including human-readable descriptions and
   JSON-Schema parameter definitions.
2. **Tool Invocation** – the model calls ``call_tool(name, params)`` and
   receives a structured ``ToolResult`` containing either the output data or
   an error description.

Why a base class?
-----------------
Foresight connects to many external services (Plaid, Gmail, Google Calendar,
Neo4j, etc.).  Each service is wrapped in its own MCP server subclass that
inherits from ``BaseMCPServer``.  The base class provides:

* A **tool registry** so that subclasses only need to call ``register_tool``
  inside their ``setup()`` method.
* **Automatic timing and logging** for every invocation.
* **Parameter validation** against JSON Schema ``required`` fields.
* A **consistent result envelope** (``ToolResult``) that upstream agents can
  rely on regardless of which server they are talking to.

Architecture
------------
::

    ┌─────────────┐     list_tools / call_tool     ┌────────────────────┐
    │  AI Agent    │ ──────────────────────────────▶│  BaseMCPServer     │
    │  (LangGraph) │ ◀────────────── ToolResult ───│   ├─ PlaidMCP      │
    └─────────────┘                                │   ├─ GmailMCP      │
                                                   │   ├─ CalendarMCP   │
                                                   │   ├─ GraphMCP      │
                                                   │   ├─ VectorMCP     │
                                                   │   └─ VoiceMCP      │
                                                   └────────────────────┘
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDefinition:
    """Describes a single tool that an MCP server exposes.

    Attributes:
        name:         Unique identifier within the server (e.g. ``get_transactions``).
        description:  Human-readable explanation shown to the AI model so it
                      knows *when* and *why* to use the tool.
        input_schema: A JSON-Schema dict defining the expected parameters.
                      Must include ``"type": "object"`` at the top level and a
                      ``"required"`` list for mandatory fields.
        handler:      The async callable that implements the tool logic.
                      Signature: ``async (params: dict) -> Any``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class ToolResult:
    """Uniform envelope returned by every tool invocation.

    Agents inspect ``success`` to decide whether to use ``data`` or surface
    ``error`` to the user.  ``execution_time_ms`` is recorded for
    observability.
    """

    success: bool
    data: Any
    error: str | None = None
    execution_time_ms: float = field(default=0.0)


# ------------------------------------------------------------------
# Base server
# ------------------------------------------------------------------

class BaseMCPServer(ABC):
    """Abstract base for all Foresight MCP servers.

    Subclasses **must** implement :pymethod:`setup`, where they call
    :pymethod:`register_tool` for each capability they want to expose.

    Example::

        class PlaidMCPServer(BaseMCPServer):
            async def setup(self) -> None:
                self.register_tool(
                    name="get_transactions",
                    description="Fetch recent bank transactions via Plaid",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "access_token": {"type": "string"},
                            "days": {"type": "integer", "default": 30},
                        },
                        "required": ["access_token"],
                    },
                    handler=self._get_transactions,
                )

            async def _get_transactions(self, params: dict) -> list[dict]:
                ...
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._tools: dict[str, ToolDefinition] = {}
        logger.info("MCP server '%s' initialised", name)

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register a tool so that it becomes discoverable via ``list_tools``.

        Args:
            name:         Unique tool identifier within this server.
            description:  Explanation the AI model reads to decide when to call
                          the tool.
            input_schema: JSON-Schema object describing expected parameters.
            handler:      Async callable ``(params) -> Any``.
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )
        logger.debug("Registered tool '%s' on server '%s'", name, self.name)

    # ------------------------------------------------------------------
    # MCP protocol surface
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tool descriptors in the format the MCP protocol expects.

        Each entry contains ``name``, ``description``, and ``inputSchema``
        (note the camelCase key — this matches the MCP/Claude wire format).
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Invoke a registered tool by name.

        The method handles:
        * Lookup validation – returns an error ``ToolResult`` if the tool does
          not exist.
        * Timing – records wall-clock milliseconds in
          ``ToolResult.execution_time_ms``.
        * Error capture – any exception raised by the handler is caught and
          surfaced in ``ToolResult.error`` rather than propagated, so that the
          calling agent always receives a well-formed response.

        Args:
            tool_name: The registered name of the tool.
            params:    Dictionary of input parameters to pass to the handler.

        Returns:
            A ``ToolResult`` indicating success or failure.
        """
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found on server '{self.name}'",
            )

        tool = self._tools[tool_name]
        start = time.perf_counter()

        try:
            result_data = await tool.handler(params)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Tool called: %s | params: %s | time: %.1fms | success: True",
                tool_name,
                params,
                elapsed_ms,
            )
            return ToolResult(success=True, data=result_data, execution_time_ms=elapsed_ms)

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "Tool called: %s | params: %s | time: %.1fms | success: False | error: %s",
                tool_name,
                params,
                elapsed_ms,
                exc,
                exc_info=True,
            )
            return ToolResult(
                success=False,
                data=None,
                error=str(exc),
                execution_time_ms=elapsed_ms,
            )

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def validate_params(self, tool_name: str, params: dict[str, Any]) -> tuple[bool, str]:
        """Check that *params* satisfies the tool's ``required`` fields.

        This is intentionally a lightweight check (presence only, not type
        coercion) because the AI model is expected to produce well-typed
        values.  Full JSON-Schema validation can be layered on top if needed.

        Args:
            tool_name: The registered tool to validate against.
            params:    The candidate parameters.

        Returns:
            A ``(valid, message)`` tuple.  If ``valid`` is ``False``,
            ``message`` explains which required fields are missing.
        """
        if tool_name not in self._tools:
            return False, f"Tool '{tool_name}' not found"

        schema = self._tools[tool_name].input_schema
        required: list[str] = schema.get("required", [])
        missing = [field for field in required if field not in params]

        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"
        return True, "ok"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def setup(self) -> None:
        """Perform post-init setup — register tools and acquire resources.

        Subclasses must override this method.  It is called once during
        application startup (before the first ``call_tool`` invocation).
        """
