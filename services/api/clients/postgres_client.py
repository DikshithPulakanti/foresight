"""PostgreSQL client backed by asyncpg for relational persistence."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY,
    email       TEXT UNIQUE NOT NULL,
    plaid_access_token TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id           UUID PRIMARY KEY,
    agent_name   TEXT NOT NULL,
    user_id      UUID NOT NULL,
    status       TEXT NOT NULL,
    started_at   TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    output_json  JSONB
);

CREATE TABLE IF NOT EXISTS alerts (
    id          UUID PRIMARY KEY,
    user_id     UUID NOT NULL,
    alert_type  TEXT NOT NULL,
    title       TEXT NOT NULL,
    message     TEXT NOT NULL,
    severity    TEXT NOT NULL,
    is_read     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS goals (
    id             UUID PRIMARY KEY,
    user_id        UUID NOT NULL,
    name           TEXT NOT NULL,
    target_amount  NUMERIC,
    current_amount NUMERIC,
    deadline       DATE,
    created_at     TIMESTAMPTZ
);
"""


class PostgresClient:
    """Async PostgreSQL client using ``asyncpg``."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        logger.info("PostgresClient initialised")

    @property
    def _active_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresClient is not connected — call connect() first")
        return self._pool

    async def connect(self, url: str) -> None:
        """Create a connection pool and verify the database is reachable."""
        self._pool = await asyncpg.create_pool(dsn=url, min_size=2, max_size=10)
        logger.info("PostgreSQL connection pool created (%s)", url.split("@")[-1])

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    async def create_tables(self) -> None:
        """Ensure the application schema exists."""
        async with self._active_pool.acquire() as conn:
            await conn.execute(_CREATE_TABLES_SQL)
        logger.info("Database tables ensured")

    async def upsert_user(self, user_id: str, email: str) -> None:
        """Insert a user or update the email if the id already exists."""
        async with self._active_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (id, email)
                VALUES ($1, $2)
                ON CONFLICT (id) DO UPDATE SET email = EXCLUDED.email
                """,
                uuid.UUID(user_id),
                email,
            )
        logger.info("Upserted user %s", user_id)

    async def log_agent_run(
        self,
        agent_name: str,
        user_id: str,
        status: str,
        output: dict[str, Any],
    ) -> str:
        """Record an agent execution and return its generated id."""
        run_id = uuid.uuid4()
        async with self._active_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_runs (id, agent_name, user_id, status, started_at, output_json)
                VALUES ($1, $2, $3, $4, NOW(), $5)
                """,
                run_id,
                agent_name,
                uuid.UUID(user_id),
                status,
                json.dumps(output),
            )
        logger.info("Logged agent run %s (%s)", run_id, agent_name)
        return str(run_id)

    async def create_alert(
        self,
        user_id: str,
        alert_type: str,
        title: str,
        message: str,
        severity: str,
    ) -> str:
        """Create a new alert and return its id."""
        alert_id = uuid.uuid4()
        async with self._active_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO alerts (id, user_id, alert_type, title, message, severity)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                alert_id,
                uuid.UUID(user_id),
                alert_type,
                title,
                message,
                severity,
            )
        logger.info("Created alert %s for user %s", alert_id, user_id)
        return str(alert_id)

    async def get_unread_alerts(self, user_id: str) -> list[dict[str, Any]]:
        """Return all unread alerts for a user, newest first."""
        async with self._active_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, alert_type, title, message, severity, created_at
                FROM alerts
                WHERE user_id = $1 AND is_read = FALSE
                ORDER BY created_at DESC
                """,
                uuid.UUID(user_id),
            )
        return [dict(row) for row in rows]

    async def mark_alerts_read(self, alert_ids: list[str]) -> None:
        """Mark one or more alerts as read."""
        uuids = [uuid.UUID(aid) for aid in alert_ids]
        async with self._active_pool.acquire() as conn:
            await conn.execute(
                "UPDATE alerts SET is_read = TRUE WHERE id = ANY($1::uuid[])",
                uuids,
            )
        logger.info("Marked %d alert(s) as read", len(uuids))
