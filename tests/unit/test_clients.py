"""Unit tests for the Redis and Neo4j database clients.

These tests mock the underlying connection libraries (redis.asyncio and
neo4j) to verify that our wrappers correctly serialise data, generate
cache keys, and handle edge cases — without requiring running databases.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.redis_client import RedisClient


# ---------------------------------------------------------------------------
# RedisClient tests
# ---------------------------------------------------------------------------

class TestRedisClient:
    """Verify JSON helpers, key generation, and missing-key handling."""

    @pytest.mark.asyncio
    async def test_set_json_serialises_to_string(self):
        """set_json should JSON-encode the value and call the underlying set."""
        client = RedisClient()
        mock_redis = AsyncMock()
        client._redis = mock_redis

        await client.set_json("mykey", {"data": 123}, ttl=60)

        mock_redis.set.assert_awaited_once_with(
            "mykey",
            json.dumps({"data": 123}),
            ex=60,
        )

    @pytest.mark.asyncio
    async def test_get_json_deserialises(self):
        """get_json should parse the stored JSON string back into a dict."""
        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = '{"data": 123}'
        client._redis = mock_redis

        result = await client.get_json("mykey")

        assert result == {"data": 123}

    @pytest.mark.asyncio
    async def test_get_json_returns_none_for_missing_key(self):
        """get_json returns None when the key doesn't exist in Redis."""
        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        client._redis = mock_redis

        result = await client.get_json("missing-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_calls_underlying(self):
        """delete() proxies to the Redis delete command."""
        client = RedisClient()
        mock_redis = AsyncMock()
        client._redis = mock_redis

        await client.delete("old-key")

        mock_redis.delete.assert_awaited_once_with("old-key")

    @pytest.mark.asyncio
    async def test_exists_returns_bool(self):
        """exists() returns True when the key is present."""
        client = RedisClient()
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1
        client._redis = mock_redis

        assert await client.exists("some-key") is True

    def test_user_transactions_key(self):
        """Cache key helper embeds the user_id correctly."""
        key = RedisClient.user_transactions_key("user-abc")
        assert "user-abc" in key
        assert key == "user:user-abc:transactions"

    def test_user_balances_key(self):
        """Balance cache key helper embeds the user_id correctly."""
        key = RedisClient.user_balances_key("user-abc")
        assert "user-abc" in key
        assert key == "user:user-abc:balances"

    def test_not_connected_raises_runtime_error(self):
        """Accessing the underlying Redis before connect() raises RuntimeError."""
        client = RedisClient()
        with pytest.raises(RuntimeError, match="not connected"):
            _ = client._r


# ---------------------------------------------------------------------------
# Neo4jClient tests
# ---------------------------------------------------------------------------

class TestNeo4jClient:
    """Verify parameterised queries and retry behaviour."""

    @pytest.mark.asyncio
    async def test_execute_read_returns_records(self):
        """execute_read should return whatever the mock driver yields."""
        from unittest.mock import AsyncMock

        # Build a lightweight mock that quacks like neo4j.AsyncSession
        mock_session = AsyncMock()
        mock_session.run = AsyncMock()

        from clients.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client._driver = MagicMock()

        mock_result = AsyncMock()
        mock_result.data.return_value = [{"name": "Whole Foods"}]

        async def fake_execute_read(fn, **kwargs):
            return [{"name": "Whole Foods"}]

        mock_session.execute_read = fake_execute_read
        client._driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        client._driver.session.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await client.execute_read("MATCH (n) RETURN n.name AS name")
        assert isinstance(result, list)
        assert result[0]["name"] == "Whole Foods"
