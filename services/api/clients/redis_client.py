"""Redis caching client for Foresight."""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis wrapper with JSON helpers and key-generation utilities."""

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None
        logger.info("RedisClient initialised")

    @property
    def _r(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError("RedisClient is not connected — call connect() first")
        return self._redis

    async def connect(self, url: str) -> None:
        """Open a connection to Redis."""
        self._redis = aioredis.from_url(url, decode_responses=True)
        await self._redis.ping()
        logger.info("Redis connection verified (%s)", url)

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis connection closed")

    # ------------------------------------------------------------------
    # Primitive operations
    # ------------------------------------------------------------------

    async def get(self, key: str) -> str | None:
        """Get a string value by key."""
        return await self._r.get(key)

    async def set(self, key: str, value: str, ttl: int = 300) -> None:
        """Set a string value with a TTL in seconds."""
        await self._r.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        await self._r.delete(key)

    async def exists(self, key: str) -> bool:
        """Check whether a key exists."""
        return bool(await self._r.exists(key))

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    async def get_json(self, key: str) -> dict[str, Any] | None:
        """Deserialise a JSON string stored at *key*."""
        raw = await self.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set_json(self, key: str, value: dict[str, Any], ttl: int = 300) -> None:
        """Serialise *value* as JSON and store it with a TTL."""
        await self.set(key, json.dumps(value), ttl=ttl)

    # ------------------------------------------------------------------
    # Cache key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def user_transactions_key(user_id: str) -> str:
        """Build the cache key for a user's recent transactions."""
        return f"user:{user_id}:transactions"

    @staticmethod
    def user_balances_key(user_id: str) -> str:
        """Build the cache key for a user's account balances."""
        return f"user:{user_id}:balances"
