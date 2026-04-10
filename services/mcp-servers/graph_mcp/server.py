"""Graph MCP server — read/write access to the Foresight Neo4j knowledge graph.

The knowledge graph is the long-term memory of the entire system.  Every
transaction, merchant, subscription, alert, and savings goal is represented
as a node with typed relationships between them.  This server exposes that
graph to AI agents through five parameterised-query tools so they can
retrieve spending patterns, manage alerts, track goals, and pull cashflow
history without writing raw Cypher.

All Cypher statements use ``$param`` placeholders — string interpolation is
never used for query construction.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseMCPServer

logger = logging.getLogger(__name__)


class GraphMCPServer(BaseMCPServer):
    """MCP server backed by a :class:`Neo4jClient`.

    Tools registered:

    1. **get_spending_patterns** — top merchants, categories, and trends
    2. **find_subscription_graph** — detected subscriptions for a user
    3. **create_alert_node** — write a financial alert into the graph
    4. **get_cashflow_data** — historical income/expense time series
    5. **update_goal_progress** — update current amount on a savings goal
    """

    def __init__(self, neo4j_client: Any) -> None:
        super().__init__(name="graph")
        self._neo4j = neo4j_client
        logger.info("GraphMCPServer created")
        self.setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Register the five graph tools."""

        self.register_tool(
            name="get_spending_patterns",
            description=(
                "Get the user's historical spending patterns from the "
                "knowledge graph — top merchants, categories, trends"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "days_back": {"type": "integer", "default": 90},
                },
                "required": ["user_id"],
            },
            handler=self._get_spending_patterns_handler,
        )

        self.register_tool(
            name="find_subscription_graph",
            description=(
                "Find all detected subscription patterns for a user "
                "from the graph"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"],
            },
            handler=self._find_subscription_graph_handler,
        )

        self.register_tool(
            name="create_alert_node",
            description=(
                "Create a financial alert in the graph and mark it as pending"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "alert_type": {"type": "string"},
                    "title": {"type": "string"},
                    "message": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                    },
                    "amount": {"type": "number"},
                },
                "required": ["user_id", "alert_type", "title", "message", "severity"],
            },
            handler=self._create_alert_node_handler,
        )

        self.register_tool(
            name="get_cashflow_data",
            description=(
                "Get historical income and expense data for cashflow "
                "forecasting"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "months_back": {"type": "integer", "default": 6},
                },
                "required": ["user_id"],
            },
            handler=self._get_cashflow_data_handler,
        )

        self.register_tool(
            name="update_goal_progress",
            description="Update the current progress on a savings goal",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "goal_id": {"type": "string"},
                    "current_amount": {"type": "number"},
                },
                "required": ["user_id", "goal_id", "current_amount"],
            },
            handler=self._update_goal_progress_handler,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _assert_user_exists(self, user_id: str) -> None:
        """Raise if the user node is not in the graph."""
        rows = await self._neo4j.execute_read(
            "MATCH (u:User {id: $user_id}) RETURN u.id AS id",
            {"user_id": user_id},
        )
        if not rows:
            raise ValueError(f"User '{user_id}' not found in the knowledge graph")

    # ------------------------------------------------------------------
    # Tool 1: get_spending_patterns
    # ------------------------------------------------------------------

    async def _get_spending_patterns_handler(
        self,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return top merchants by total spend over the lookback window."""
        user_id: str = params["user_id"]
        days_back: int = params.get("days_back", 90)

        await self._assert_user_exists(user_id)

        rows = await self._neo4j.execute_read(
            "MATCH (u:User {id: $user_id})-[:MADE]->(t:Transaction)-[:AT]->(m:Merchant) "
            "WHERE t.date >= date() - duration({days: $days_back}) "
            "RETURN m.name AS merchant, count(t) AS frequency, "
            "       sum(t.amount) AS total, avg(t.amount) AS avg_amount "
            "ORDER BY total DESC LIMIT 20",
            {"user_id": user_id, "days_back": days_back},
        )

        logger.info("get_spending_patterns: user=%s days=%d results=%d", user_id, days_back, len(rows))
        return rows

    # ------------------------------------------------------------------
    # Tool 2: find_subscription_graph
    # ------------------------------------------------------------------

    async def _find_subscription_graph_handler(
        self,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return all subscription nodes linked to the user."""
        user_id: str = params["user_id"]

        await self._assert_user_exists(user_id)

        rows = await self._neo4j.execute_read(
            "MATCH (u:User {id: $user_id})-[:HAS_SUBSCRIPTION]->(s:Subscription) "
            "RETURN s.name AS name, s.amount AS amount, s.frequency AS frequency, "
            "       s.next_charge_date AS next_charge_date, s.category AS category",
            {"user_id": user_id},
        )

        logger.info("find_subscription_graph: user=%s subscriptions=%d", user_id, len(rows))
        return rows

    # ------------------------------------------------------------------
    # Tool 3: create_alert_node
    # ------------------------------------------------------------------

    async def _create_alert_node_handler(
        self,
        params: dict[str, Any],
    ) -> dict[str, str]:
        """Create an Alert node and link it to the user."""
        user_id: str = params["user_id"]
        alert_type: str = params["alert_type"]
        title: str = params["title"]
        message: str = params["message"]
        severity: str = params["severity"]
        amount: float | None = params.get("amount")

        await self._assert_user_exists(user_id)

        alert_id = str(uuid.uuid4())

        query = (
            "MERGE (a:Alert {id: $alert_id}) "
            "SET a.type = $type, a.title = $title, a.message = $message, "
            "    a.severity = $severity, a.created_at = datetime(), "
            "    a.status = 'pending'"
        )
        query_params: dict[str, Any] = {
            "alert_id": alert_id,
            "type": alert_type,
            "title": title,
            "message": message,
            "severity": severity,
        }

        if amount is not None:
            query += ", a.amount = $amount"
            query_params["amount"] = amount

        query += (
            " WITH a "
            "MATCH (u:User {id: $user_id}) "
            "CREATE (u)-[:HAS_ALERT]->(a)"
        )
        query_params["user_id"] = user_id

        await self._neo4j.execute_write(query, query_params)

        logger.info("Created alert %s (type=%s, severity=%s) for user %s", alert_id, alert_type, severity, user_id)
        return {"alert_id": alert_id}

    # ------------------------------------------------------------------
    # Tool 4: get_cashflow_data
    # ------------------------------------------------------------------

    async def _get_cashflow_data_handler(
        self,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return income/expense time series for cashflow forecasting."""
        user_id: str = params["user_id"]
        months_back: int = params.get("months_back", 6)

        await self._assert_user_exists(user_id)

        rows = await self._neo4j.execute_read(
            "MATCH (u:User {id: $user_id})-[:MADE]->(t:Transaction) "
            "WHERE t.date >= date() - duration({months: $months_back}) "
            "RETURN t.date AS date, t.amount AS amount, t.category AS category, "
            "       CASE WHEN t.amount < 0 THEN 'income' ELSE 'expense' END AS type "
            "ORDER BY t.date",
            {"user_id": user_id, "months_back": months_back},
        )

        logger.info("get_cashflow_data: user=%s months=%d records=%d", user_id, months_back, len(rows))
        return rows

    # ------------------------------------------------------------------
    # Tool 5: update_goal_progress
    # ------------------------------------------------------------------

    async def _update_goal_progress_handler(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Update the saved amount on a goal and return progress."""
        user_id: str = params["user_id"]
        goal_id: str = params["goal_id"]
        current_amount: float = params["current_amount"]

        await self._assert_user_exists(user_id)

        rows = await self._neo4j.execute_read(
            "MATCH (u:User {id: $user_id})-[:HAS_GOAL]->(g:Goal {id: $goal_id}) "
            "RETURN g.id AS id",
            {"user_id": user_id, "goal_id": goal_id},
        )
        if not rows:
            raise ValueError(f"Goal '{goal_id}' not found for user '{user_id}'")

        await self._neo4j.execute_write(
            "MATCH (u:User {id: $user_id})-[:HAS_GOAL]->(g:Goal {id: $goal_id}) "
            "SET g.current_amount = $current_amount, g.updated_at = datetime()",
            {"user_id": user_id, "goal_id": goal_id, "current_amount": current_amount},
        )

        result = await self._neo4j.execute_read(
            "MATCH (u:User {id: $user_id})-[:HAS_GOAL]->(g:Goal {id: $goal_id}) "
            "RETURN g.name AS name, g.target_amount AS target_amount, "
            "       g.current_amount AS current_amount, "
            "       round((g.current_amount / g.target_amount) * 100) AS percent_complete",
            {"user_id": user_id, "goal_id": goal_id},
        )

        logger.info("Updated goal %s for user %s → $%.2f", goal_id, user_id, current_amount)
        return result[0] if result else {}
