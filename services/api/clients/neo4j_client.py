"""Neo4j graph database client for the Foresight financial knowledge graph."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction
from neo4j.exceptions import ServiceUnavailable

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 1.0


class Neo4jClient:
    """Async wrapper around the Neo4j Python driver.

    All Cypher queries use parameterised inputs — string interpolation is
    never used for query construction.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: AsyncDriver | None = None
        logger.info("Neo4jClient initialised (uri=%s, user=%s)", uri, user)

    async def connect(self) -> None:
        """Create the async driver and verify connectivity."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        await self._driver.verify_connectivity()
        logger.info("Neo4j connection verified at %s", self._uri)

    async def close(self) -> None:
        """Gracefully close the driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    async def __aenter__(self) -> Neo4jClient:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    @property
    def _active_driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4jClient is not connected — call connect() first")
        return self._driver

    async def _retry(self, coro_factory: Any) -> Any:
        """Retry *coro_factory* on ``ServiceUnavailable`` with back-off."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await coro_factory()
            except ServiceUnavailable as exc:
                last_exc = exc
                logger.warning(
                    "ServiceUnavailable (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_BACKOFF_S * attempt)
        raise last_exc  # type: ignore[misc]

    async def execute_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read transaction and return all records as dicts."""
        params = params or {}
        logger.debug("execute_read: %s | params=%s", query, params)

        async def _run() -> list[dict[str, Any]]:
            async with self._active_driver.session() as session:

                async def _work(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
                    result = await tx.run(query, params)
                    return [record.data() async for record in result]

                return await session.execute_read(_work)

        return await self._retry(_run)

    async def execute_write(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a write transaction and return the summary counters."""
        params = params or {}
        logger.debug("execute_write: %s | params=%s", query, params)

        async def _run() -> dict[str, Any]:
            async with self._active_driver.session() as session:

                async def _work(tx: AsyncManagedTransaction) -> dict[str, Any]:
                    result = await tx.run(query, params)
                    summary = await result.consume()
                    return dict(summary.counters.__dict__)

                return await session.execute_write(_work)

        return await self._retry(_run)

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------

    async def create_user(self, user_id: str, email: str) -> None:
        """Merge a User node in the graph."""
        await self.execute_write(
            "MERGE (u:User {id: $user_id}) "
            "SET u.email = $email, u.updated_at = datetime()",
            {"user_id": user_id, "email": email},
        )
        logger.info("Merged User node %s", user_id)

    async def create_transaction(self, tx_data: dict[str, Any]) -> None:
        """Merge a Transaction node from a data dict (must contain ``id``)."""
        tx_id = tx_data["id"]
        props = {k: v for k, v in tx_data.items() if k != "id"}
        await self.execute_write(
            "MERGE (t:Transaction {id: $id}) SET t += $props",
            {"id": tx_id, "props": props},
        )
        logger.info("Merged Transaction node %s", tx_id)

    async def create_merchant(self, merchant_data: dict[str, Any]) -> None:
        """Merge a Merchant node from a data dict (must contain ``name``)."""
        name = merchant_data["name"]
        props = {k: v for k, v in merchant_data.items() if k != "name"}
        await self.execute_write(
            "MERGE (m:Merchant {name: $name}) SET m += $props",
            {"name": name, "props": props},
        )
        logger.info("Merged Merchant node %s", name)

    async def link_transaction_to_merchant(
        self,
        tx_id: str,
        merchant_name: str,
    ) -> None:
        """Create a ``PAID_TO`` relationship between a transaction and merchant."""
        await self.execute_write(
            "MATCH (t:Transaction {id: $tx_id}), (m:Merchant {name: $merchant_name}) "
            "MERGE (t)-[:PAID_TO]->(m)",
            {"tx_id": tx_id, "merchant_name": merchant_name},
        )
        logger.info("Linked Transaction %s → Merchant %s", tx_id, merchant_name)

    async def get_spending_by_category(
        self,
        user_id: str,
        days: int,
    ) -> list[dict[str, Any]]:
        """Aggregate spending per category for a user over the last *days* days."""
        return await self.execute_read(
            "MATCH (u:User {id: $user_id})-[:MADE]->(t:Transaction) "
            "WHERE t.date >= date() - duration({days: $days}) "
            "RETURN t.category AS category, sum(t.amount) AS total "
            "ORDER BY total DESC",
            {"user_id": user_id, "days": days},
        )

    async def find_recurring_merchants(
        self,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """Find merchants the user has transacted with 2+ times in the last 30 days."""
        return await self.execute_read(
            "MATCH (u:User {id: $user_id})-[:MADE]->(t:Transaction)-[:PAID_TO]->(m:Merchant) "
            "WHERE t.date >= date() - duration({days: 30}) "
            "WITH m, count(t) AS tx_count "
            "WHERE tx_count >= 2 "
            "RETURN m.name AS merchant, tx_count "
            "ORDER BY tx_count DESC",
            {"user_id": user_id},
        )
