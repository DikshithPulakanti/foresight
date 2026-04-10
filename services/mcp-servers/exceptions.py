"""MCP server exceptions.

These exception types allow callers to distinguish between different failure
modes when interacting with the MCP tool layer:

* **ToolNotFoundException** – the requested tool name does not exist in the
  server's registry.  This typically indicates a typo or a missing
  ``register_tool`` call in the server's ``setup()`` method.

* **ToolExecutionError** – the tool was found and invoked, but its handler
  raised an unrecoverable error at runtime (e.g. a downstream API returned
  an HTTP 500, or a database query timed out).

* **InvalidParamsError** – the parameters supplied to ``call_tool`` fail
  schema validation (missing required fields, wrong types, etc.).
"""

from __future__ import annotations


class ToolNotFoundException(Exception):
    """Raised when a tool name cannot be found in the server's tool map."""

    def __init__(self, tool_name: str, server_name: str = "") -> None:
        ctx = f" on server '{server_name}'" if server_name else ""
        super().__init__(f"Tool '{tool_name}' not found{ctx}")
        self.tool_name = tool_name
        self.server_name = server_name


class ToolExecutionError(Exception):
    """Raised when a tool handler fails at runtime."""

    def __init__(self, tool_name: str, cause: Exception) -> None:
        super().__init__(f"Tool '{tool_name}' execution failed: {cause}")
        self.tool_name = tool_name
        self.cause = cause


class InvalidParamsError(Exception):
    """Raised when tool parameters fail validation against the input schema."""

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(f"Invalid params for tool '{tool_name}': {reason}")
        self.tool_name = tool_name
        self.reason = reason
