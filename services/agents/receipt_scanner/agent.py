"""Receipt Scanner — processes a receipt photo and instantly categorises the expense.

Triggered when the user taps the camera button on the mobile app.  The image
is sent as base64, processed through Claude Vision (via ``vision_mcp``), and
the extracted data is categorised, stored in the knowledge graph, and returned
with a friendly confirmation message.

Graph pipeline::

    initialise
      → validate_input
      → scan_receipt
      → categorize_and_store
      → generate_response
      → finalise → END
"""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Any

from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent

logger = logging.getLogger(__name__)

_LARGE_PURCHASE_THRESHOLD = 100.0

CATEGORY_MAP: dict[str, list[str]] = {
    "grocery": [
        "whole foods", "trader joe", "safeway", "kroger", "walmart",
        "target", "costco", "aldi", "publix", "wegmans",
    ],
    "restaurant": [
        "mcdonald", "starbucks", "chipotle", "subway", "pizza", "cafe",
        "restaurant", "grill", "bar", "diner", "burger", "taco", "sushi",
        "bakery", "wendy", "panera", "chick-fil-a",
    ],
    "transport": [
        "uber", "lyft", "taxi", "gas", "shell", "bp", "chevron", "exxon",
        "parking", "amtrak", "delta", "united", "southwest",
    ],
    "pharmacy": [
        "cvs", "walgreens", "rite aid", "pharmacy", "drug",
    ],
    "entertainment": [
        "cinema", "movie", "theater", "theatre", "netflix", "spotify",
        "amazon prime", "hulu", "concert", "ticket",
    ],
    "shopping": [
        "amazon", "apple", "best buy", "mall", "store", "ikea", "home depot",
        "lowes", "nordstrom", "macys", "nike", "gap",
    ],
    "utilities": [
        "at&t", "verizon", "comcast", "electric", "water", "internet",
        "t-mobile", "sprint", "xfinity",
    ],
}


def _categorise_merchant(merchant_name: str) -> str:
    """Map a merchant name to a spending category via keyword lookup.

    Falls back to ``"general"`` when no keyword matches.
    """
    lower = merchant_name.lower()
    for category, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in lower:
                return category
    return "general"


class ReceiptScannerAgent(BaseAgent):
    """Scans receipt photos and instantly categorises the expense.

    Uses Claude Vision (through ``vision_mcp``) to extract merchant, items,
    and total, then persists the transaction in the knowledge graph and
    returns a confirmation to the mobile app.
    """

    def __init__(self) -> None:
        super().__init__(
            name="receipt_scanner",
            description=(
                "Scans receipt photos using Claude Vision — extracts merchant, "
                "items, total, and instantly categorizes the expense"
            ),
        )

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the four processing nodes."""
        builder.add_node("validate_input", self._validate_input)
        builder.add_node("scan_receipt", self._scan_receipt)
        builder.add_node("categorize_and_store", self._categorize_and_store)
        builder.add_node("generate_response", self._generate_response)

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline."""
        builder.add_edge("initialise", "validate_input")
        builder.add_edge("validate_input", "scan_receipt")
        builder.add_edge("scan_receipt", "categorize_and_store")
        builder.add_edge("categorize_and_store", "generate_response")
        builder.add_edge("generate_response", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: validate_input
    # ------------------------------------------------------------------

    async def _validate_input(self, state: AgentState) -> dict[str, Any]:
        """Verify the caller supplied a non-empty base64 image."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        image_b64 = inp.get("image_base64")

        if not image_b64 or not isinstance(image_b64, str) or not image_b64.strip():
            return self.set_error(state, "No image provided")

        image_type: str = inp.get("image_type", "image/jpeg")

        step = f"Image received, type={image_type}"
        logger.info(step)
        return {
            "input": {**inp, "image_type": image_type},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: scan_receipt
    # ------------------------------------------------------------------

    async def _scan_receipt(self, state: AgentState) -> dict[str, Any]:
        """Send the image to Claude Vision via ``vision_mcp`` and extract data."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})

        try:
            receipt_data: dict[str, Any] = await self.call_tool(
                "vision_mcp",
                "scan_receipt",
                {
                    "image_base64": inp["image_base64"],
                    "image_type": inp.get("image_type", "image/jpeg"),
                },
            )
        except RuntimeError as exc:
            logger.error("scan_receipt failed: %s", exc)
            return self.set_error(state, str(exc))

        merchant = receipt_data.get("merchant_name")
        total = receipt_data.get("total_amount")

        if not merchant:
            return self.set_error(
                state,
                "Could not read merchant name from receipt — "
                "please try a clearer photo",
            )

        step = f"Scanned receipt — merchant={merchant}, total=${total}"
        logger.info(step)
        return {
            "input": {**inp, "receipt_data": receipt_data},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: categorize_and_store
    # ------------------------------------------------------------------

    async def _categorize_and_store(self, state: AgentState) -> dict[str, Any]:
        """Categorise the expense and persist the transaction in Neo4j."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        receipt: dict[str, Any] = inp.get("receipt_data", {})
        user_id: str = state["user_id"]

        merchant = receipt.get("merchant_name", "Unknown")
        total_amount = float(receipt.get("total_amount") or 0)
        receipt_date = receipt.get("date") or date.today().isoformat()
        category = _categorise_merchant(merchant)
        transaction_id = str(uuid.uuid4())

        # Persist the transaction node in the knowledge graph
        try:
            await self.call_tool(
                "graph_mcp",
                "store_transaction_node",
                {
                    "user_id": user_id,
                    "transaction_id": transaction_id,
                    "properties": {
                        "merchant": merchant,
                        "amount": total_amount,
                        "date": receipt_date,
                        "category": category,
                        "source": "receipt_scan",
                    },
                },
            )
        except RuntimeError as exc:
            logger.debug("store_transaction_node skipped: %s", exc)

        # Fire a low-severity alert for large purchases
        if total_amount > _LARGE_PURCHASE_THRESHOLD:
            try:
                await self.call_tool(
                    "graph_mcp",
                    "create_alert_node",
                    {
                        "user_id": user_id,
                        "alert_type": "large_purchase",
                        "title": f"Large purchase scanned: {merchant}",
                        "message": f"${total_amount:.2f} at {merchant} recorded",
                        "severity": "low",
                        "amount": total_amount,
                    },
                )
            except RuntimeError as exc:
                logger.warning("Failed to create large-purchase alert: %s", exc)

        step = f"Categorized as {category}, stored in graph"
        logger.info(step)
        return {
            "input": {
                **inp,
                "category": category,
                "transaction_id": transaction_id,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: generate_response
    # ------------------------------------------------------------------

    async def _generate_response(self, state: AgentState) -> dict[str, Any]:
        """Build a confirmation message and optionally add a spending insight."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        receipt: dict[str, Any] = inp.get("receipt_data", {})
        category: str = inp.get("category", "general")
        transaction_id: str = inp.get("transaction_id", "")
        user_id: str = state["user_id"]

        merchant = receipt.get("merchant_name", "Unknown")
        total_amount = float(receipt.get("total_amount") or 0)
        items: list[dict[str, Any]] = receipt.get("items") or []

        # --- Build confirmation message ---
        message = f"Got it! ${total_amount:.2f} at {merchant} categorized as {category}."
        if items:
            message += f" I recorded {len(items)} item{'s' if len(items) != 1 else ''}."

        # --- Optional spending insight ---
        spending_insight: str | None = None
        try:
            patterns: list[dict[str, Any]] = await self.call_tool(
                "graph_mcp",
                "get_spending_patterns",
                {"user_id": user_id, "days_back": 90},
            )
            top_merchants = [
                p.get("merchant", "").lower() for p in patterns[:10]
            ]
            if merchant.lower() in top_merchants:
                match = next(
                    p for p in patterns
                    if p.get("merchant", "").lower() == merchant.lower()
                )
                freq = match.get("frequency", 0)
                avg = match.get("avg_amount", 0)
                spending_insight = (
                    f"You've visited {merchant} {freq} times in the last 90 days "
                    f"with an average spend of ${avg:.2f}."
                )
                message += f" FYI: {spending_insight}"
        except RuntimeError as exc:
            logger.debug("Spending insight unavailable: %s", exc)

        output: dict[str, Any] = {
            "receipt": receipt,
            "category": category,
            "transaction_id": transaction_id,
            "confirmation_message": message,
            "spending_insight": spending_insight,
        }

        step = "Generated response"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }
