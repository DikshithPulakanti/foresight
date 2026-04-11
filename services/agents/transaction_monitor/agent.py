"""Transaction Monitor — watches every new bank transaction and fires alerts.

This agent runs on a schedule (or on-demand) and performs four steps:

1. **Fetch** the latest transactions from Plaid via ``plaid_mcp``.
2. **Analyse** them for anomalies — duplicate charges, unusual amounts,
   first-time merchants, and large purchases.
3. **Create alerts** in the Neo4j knowledge graph via ``graph_mcp`` and
   persist each transaction as a graph node.
4. **Generate a summary** using Claude so the user gets a concise,
   plain-English explanation of what was flagged and why.

The graph edges it creates::

    (User)-[:MADE]->(Transaction)
    (User)-[:HAS_ALERT]->(Alert)

All external I/O goes through MCP ``call_tool`` — the agent never imports
Plaid, Neo4j, or Anthropic clients directly.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent

logger = logging.getLogger(__name__)

_LARGE_PURCHASE_THRESHOLD = 200.0
_HIGH_SEVERITY_THRESHOLD = 500.0


class TransactionMonitorAgent(BaseAgent):
    """Monitors bank transactions in real time and fires anomaly alerts.

    Detects four categories of issues:

    * **Duplicate charges** — same merchant + amount + date appearing twice.
    * **Unusual amounts** — statistical outliers flagged by Plaid's anomaly
      detection tool.
    * **New merchants** — first-time transactions with an unknown merchant.
    * **Large purchases** — any single transaction above
      ``$LARGE_PURCHASE_THRESHOLD``.
    """

    def __init__(self) -> None:
        super().__init__(
            name="transaction_monitor",
            description=(
                "Monitors bank transactions in real time — detects duplicates, "
                "unusual amounts, new merchants, and large purchases"
            ),
        )
        self._llm = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the four processing nodes."""
        builder.add_node("fetch_transactions", self._fetch_transactions)
        builder.add_node("analyze_transactions", self._analyze_transactions)
        builder.add_node("create_alerts", self._create_alerts)
        builder.add_node("generate_summary", self._generate_summary)

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline with custom edges."""
        builder.add_edge("initialise", "fetch_transactions")
        builder.add_edge("fetch_transactions", "analyze_transactions")
        builder.add_edge("analyze_transactions", "create_alerts")
        builder.add_edge("create_alerts", "generate_summary")
        builder.add_edge("generate_summary", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: fetch_transactions
    # ------------------------------------------------------------------

    async def _fetch_transactions(self, state: AgentState) -> dict[str, Any]:
        """Pull the last 7 days of transactions from Plaid."""
        try:
            transactions: list[dict[str, Any]] = await self.call_tool(
                "plaid_mcp",
                "get_transactions",
                {
                    "user_id": state["user_id"],
                    "days_back": 7,
                    "limit": 100,
                },
            )
        except RuntimeError as exc:
            logger.error("fetch_transactions failed: %s", exc)
            return self.set_error(state, str(exc))

        step = f"Fetched {len(transactions)} transactions"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "transactions": transactions},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: analyze_transactions
    # ------------------------------------------------------------------

    async def _analyze_transactions(self, state: AgentState) -> dict[str, Any]:
        """Run anomaly detection, duplicate checking, and large-purchase scan."""
        if state.get("status") == "failed":
            return {}

        transactions: list[dict[str, Any]] = (
            state.get("input", {}).get("transactions", [])
        )

        # --- Plaid statistical anomaly detection ---
        try:
            plaid_flagged: list[dict[str, Any]] = await self.call_tool(
                "plaid_mcp",
                "flag_unusual_transactions",
                {
                    "user_id": state["user_id"],
                    "lookback_days": 90,
                    "sensitivity": 0.7,
                },
            )
        except RuntimeError as exc:
            logger.warning("flag_unusual_transactions failed: %s", exc)
            plaid_flagged = []

        flagged_ids: set[str] = set()
        flagged: list[dict[str, Any]] = []

        for item in plaid_flagged:
            tx = item.get("transaction", {})
            flagged.append({
                "id": tx.get("id"),
                "merchant_name": tx.get("merchant_name", "Unknown"),
                "amount": tx.get("amount", 0),
                "date": tx.get("date"),
                "reason": item.get("reason", "Unusual transaction"),
                "confidence": item.get("confidence_score", 0),
            })
            flagged_ids.add(tx.get("id"))

        # --- Duplicate detection (same merchant + amount + date) ---
        groups: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
        for tx in transactions:
            key = (
                tx.get("merchant_name", ""),
                tx.get("amount", 0),
                tx.get("date", ""),
            )
            groups[key].append(tx)

        for key, group in groups.items():
            if len(group) <= 1:
                continue
            for tx in group:
                if tx.get("id") in flagged_ids:
                    continue
                flagged.append({
                    "id": tx.get("id"),
                    "merchant_name": key[0],
                    "amount": key[1],
                    "date": key[2],
                    "reason": f"Duplicate charge — {len(group)} identical transactions",
                    "confidence": 0.9,
                })
                flagged_ids.add(tx.get("id"))

        # --- Large-purchase detection ---
        for tx in transactions:
            if tx.get("id") in flagged_ids:
                continue
            amount = abs(tx.get("amount", 0))
            if amount > _LARGE_PURCHASE_THRESHOLD:
                flagged.append({
                    "id": tx.get("id"),
                    "merchant_name": tx.get("merchant_name", "Unknown"),
                    "amount": amount,
                    "date": tx.get("date"),
                    "reason": f"Large purchase — ${amount:.2f} exceeds ${_LARGE_PURCHASE_THRESHOLD:.0f} threshold",
                    "confidence": 0.8,
                })
                flagged_ids.add(tx.get("id"))

        step = f"Analyzed transactions, found {len(flagged)} issues"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "flagged": flagged},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: create_alerts
    # ------------------------------------------------------------------

    async def _create_alerts(self, state: AgentState) -> dict[str, Any]:
        """Persist an alert node and a transaction node in Neo4j for each flag."""
        if state.get("status") == "failed":
            return {}

        flagged: list[dict[str, Any]] = state.get("input", {}).get("flagged", [])
        transactions: list[dict[str, Any]] = state.get("input", {}).get("transactions", [])
        user_id: str = state["user_id"]
        alerts_created = 0

        for item in flagged:
            merchant = item.get("merchant_name", "Unknown")
            amount = abs(item.get("amount", 0))
            reason = item.get("reason", "Flagged transaction")
            severity = "high" if amount > _HIGH_SEVERITY_THRESHOLD else "medium"

            # Create alert node in the knowledge graph
            try:
                await self.call_tool(
                    "graph_mcp",
                    "create_alert_node",
                    {
                        "user_id": user_id,
                        "alert_type": "transaction_anomaly",
                        "title": f"Unusual transaction: {merchant}",
                        "message": f"${amount:.2f} at {merchant} flagged: {reason}",
                        "severity": severity,
                        "amount": amount,
                    },
                )
                alerts_created += 1
            except RuntimeError as exc:
                logger.warning("Failed to create alert for %s: %s", merchant, exc)

        # Persist every transaction as a graph node linked to the user
        for tx in transactions:
            try:
                await self.call_tool(
                    "graph_mcp",
                    "store_transaction_node",
                    {
                        "user_id": user_id,
                        "transaction_id": tx.get("id"),
                        "properties": {
                            "amount": tx.get("amount"),
                            "date": tx.get("date"),
                            "merchant_name": tx.get("merchant_name"),
                            "category": tx.get("category"),
                            "pending": tx.get("pending", False),
                        },
                    },
                )
            except RuntimeError as exc:
                logger.debug("store_transaction_node skipped: %s", exc)

        step = f"Created {alerts_created} alerts"
        logger.info(step)
        return {
            "input": {
                **state.get("input", {}),
                "alerts_created": alerts_created,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: generate_summary
    # ------------------------------------------------------------------

    async def _generate_summary(self, state: AgentState) -> dict[str, Any]:
        """Use Claude to produce a plain-English summary of flagged items."""
        if state.get("status") == "failed":
            return {}

        flagged: list[dict[str, Any]] = state.get("input", {}).get("flagged", [])
        transactions: list[dict[str, Any]] = state.get("input", {}).get("transactions", [])
        alerts_created: int = state.get("input", {}).get("alerts_created", 0)

        if not flagged:
            summary = "No unusual transactions detected in the past 7 days."
        else:
            prompt = (
                f"Summarize these {len(flagged)} flagged transactions for a "
                f"user in 2-3 sentences. Be specific about amounts and "
                f"merchants. Transactions: {flagged}"
            )
            try:
                response = self._llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                summary = response.content[0].text
            except Exception as exc:
                logger.warning("Claude summarisation failed: %s", exc)
                summary = (
                    f"{len(flagged)} transactions flagged — "
                    f"review your alerts for details."
                )

        output = {
            "summary": summary,
            "alerts_created": alerts_created,
            "transactions_analyzed": len(transactions),
            "flagged_count": len(flagged),
        }

        step = "Generated summary"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }
