"""Integration tests for the FastAPI endpoints.

Uses ``httpx.AsyncClient`` with an ASGI transport to exercise the real
FastAPI app without starting a network server. These tests verify route
existence, status codes, response shapes, and input validation.
"""

from __future__ import annotations

import httpx
import pytest

from main import app


@pytest.fixture(scope="module")
async def client():
    """Create an async HTTPX client wired directly to the FastAPI ASGI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Verify the liveness probe returns structured status information."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        """GET /health succeeds and includes status, version, and env."""
        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "env" in body

    @pytest.mark.asyncio
    async def test_health_version_format(self, client):
        """The version field is a semver-style string."""
        response = await client.get("/health")
        body = response.json()
        assert "." in body["version"]


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------

class TestAgentEndpoints:
    """Verify agent route registration, validation, and error handling."""

    @pytest.mark.asyncio
    async def test_agent_status_returns_200(self, client):
        """GET /agents/status is reachable and returns 200."""
        response = await client.get("/agents/status")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_transaction_monitor_requires_user_id(self, client):
        """POST without user_id should fail validation (422)."""
        response = await client.post("/agents/transaction-monitor/run", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_transaction_monitor_with_user_id(self, client):
        """POST with valid user_id is accepted (may return 200 or 500 if
        downstream services are unavailable — we just verify the route exists
        and validates input correctly)."""
        response = await client.post(
            "/agents/transaction-monitor/run",
            json={"user_id": "test-user-123"},
        )
        assert response.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_receipt_scanner_requires_image(self, client):
        """Receipt scanner needs both user_id and image_base64."""
        response = await client.post(
            "/agents/receipt-scanner/run",
            json={"user_id": "test"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_voice_accepts_text_query(self, client):
        """Voice endpoint should accept a text_query as alternative to audio."""
        response = await client.post(
            "/agents/voice/run",
            json={"user_id": "test", "text_query": "How much did I spend?"},
        )
        assert response.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_unknown_agent_returns_404_or_422(self, client):
        """A request to a non-existent agent path returns 404 or similar."""
        response = await client.post(
            "/agents/nonexistent-agent/run",
            json={"user_id": "test"},
        )
        assert response.status_code in (404, 405, 422)


# ---------------------------------------------------------------------------
# Graph endpoints
# ---------------------------------------------------------------------------

class TestGraphEndpoints:
    """Verify the knowledge-graph stats endpoint."""

    @pytest.mark.asyncio
    async def test_graph_stats_returns_200(self, client):
        """GET /graph/stats is reachable and returns JSON."""
        response = await client.get("/graph/stats")

        assert response.status_code == 200
        assert isinstance(response.json(), (dict, list))


# ---------------------------------------------------------------------------
# Voice endpoints
# ---------------------------------------------------------------------------

class TestVoiceEndpoints:
    """Verify the voice module status endpoint."""

    @pytest.mark.asyncio
    async def test_voice_status_returns_200(self, client):
        """GET /voice/status is reachable."""
        response = await client.get("/voice/status")
        assert response.status_code == 200
