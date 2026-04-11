"""Shared pytest fixtures used across all Foresight tests.

Provides mock clients for Anthropic, Neo4j, Redis, and Plaid — as well as
a canonical ``sample_agent_state`` that mirrors the real ``AgentState``
TypedDict used by every agent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure import paths work regardless of where pytest is invoked from.
# The MCP servers use relative imports from services/mcp-servers/ and the API
# uses imports like ``from routers import ...``, so we add both directories.
# ---------------------------------------------------------------------------

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "services" / "mcp-servers"))
sys.path.insert(0, str(_repo / "services" / "api"))
sys.path.insert(0, str(_repo / "services"))
sys.path.insert(0, str(_repo))

# The on-disk directory is "mcp-servers" (hyphen) but code imports
# "mcp_servers" (underscore).  Create a virtual package alias so both work.
import types as _types

_mcp_pkg = _types.ModuleType("mcp_servers")
_mcp_pkg.__path__ = [str(_repo / "services" / "mcp-servers")]
sys.modules.setdefault("mcp_servers", _mcp_pkg)


# ---------------------------------------------------------------------------
# Mock external service clients
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic_client():
    """Patch ``anthropic.Anthropic`` and return a mock that yields JSON."""
    with patch("anthropic.Anthropic") as mock_cls:
        client = MagicMock()
        client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"category": "grocery", "confidence": 0.95}')]
        )
        mock_cls.return_value = client
        yield client


@pytest.fixture
def mock_neo4j_client():
    """Async mock for the Neo4j graph client."""
    client = AsyncMock()
    client.execute_read.return_value = [
        {"merchant": "Whole Foods", "total": 245.50, "frequency": 8}
    ]
    client.execute_write.return_value = {"id": "test-123"}
    return client


@pytest.fixture
def mock_redis_client():
    """Async mock for the Redis cache client."""
    client = AsyncMock()
    client.get_json.return_value = None
    client.set_json.return_value = True
    return client


@pytest.fixture
def mock_plaid_response():
    """Sample Plaid transaction response matching the real API shape."""
    return {
        "transactions": [
            {
                "id": "t1",
                "amount": 87.43,
                "merchant_name": "Whole Foods",
                "category": "grocery",
                "date": "2025-04-10",
                "pending": False,
            },
            {
                "id": "t2",
                "amount": 15.99,
                "merchant_name": "Netflix",
                "category": "entertainment",
                "date": "2025-04-09",
                "pending": False,
            },
        ]
    }


@pytest.fixture
def sample_agent_state():
    """Canonical AgentState dict used as test input for agent nodes."""
    return {
        "user_id": "test-user-123",
        "session_id": "sess-abc",
        "input": {},
        "output": {},
        "status": "running",
        "error": None,
        "steps": [],
        "started_at": 1234567890.0,
        "metadata": {},
    }
