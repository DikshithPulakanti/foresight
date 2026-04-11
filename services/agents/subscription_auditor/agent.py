"""Subscription Auditor — cross-references bank charges and inbox receipts.

This agent hunts down every recurring charge the user is paying for by
pulling data from two independent sources:

* **Bank account** (Plaid ``get_recurring_transactions``) — hard proof of
  money leaving the account on a regular cadence.
* **Inbox** (Gmail ``find_subscription_emails``) — renewal confirmations,
  welcome emails, and receipts that reveal services the user signed up for.

It then cross-references the two lists using fuzzy name matching, builds a
single unified subscription inventory, and flags actionable opportunities:
forgotten subscriptions, price increases, and duplicate services.

Graph pipeline::

    initialise
      → fetch_bank_subscriptions
      → fetch_email_subscriptions
      → cross_reference
      → identify_opportunities
      → generate_report
      → finalise → END
"""

from __future__ import annotations

import logging
import re
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent

logger = logging.getLogger(__name__)

_DOLLAR_RE = re.compile(r"\$?([\d,]+\.?\d*)")

_FREQUENCY_TO_MONTHLY: dict[str, float] = {
    "weekly": 4.33,
    "biweekly": 2.17,
    "semi_monthly": 2.0,
    "monthly": 1.0,
    "quarterly": 1.0 / 3,
    "semi_annually": 1.0 / 6,
    "annually": 1.0 / 12,
    "yearly": 1.0 / 12,
    "unknown": 1.0,
}

_FORGOTTEN_DAYS_THRESHOLD = 60
_FORGOTTEN_AMOUNT_THRESHOLD = 5.00
_DUPLICATE_CATEGORIES: list[list[str]] = [
    ["netflix", "hulu", "disney", "hbo", "paramount", "peacock", "apple tv", "prime video"],
    ["spotify", "apple music", "tidal", "youtube music", "amazon music", "deezer"],
    ["dropbox", "google one", "icloud", "onedrive"],
    ["zoom", "teams", "webex", "google meet"],
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_dollar(raw: str | float | int | None) -> float:
    """Extract a numeric dollar value from various formats.

    Handles ``"$15.49"``, ``"15.49"``, ``15.49``, and ``None`` → 0.0.
    """
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    match = _DOLLAR_RE.search(str(raw))
    if match:
        return float(match.group(1).replace(",", ""))
    return 0.0


def _fuzzy_match(a: str, b: str, min_run: int = 3) -> bool:
    """Return *True* if *a* and *b* share ``min_run``+ consecutive characters.

    Comparison is case-insensitive and ignores non-alphanumeric characters so
    that "Netflix" matches "netflix.com" and "Apple TV+" matches "apple tv".
    """
    a_clean = re.sub(r"[^a-z0-9]", "", a.lower())
    b_clean = re.sub(r"[^a-z0-9]", "", b.lower())
    if not a_clean or not b_clean:
        return False

    shorter, longer = (a_clean, b_clean) if len(a_clean) <= len(b_clean) else (b_clean, a_clean)
    for i in range(len(shorter) - min_run + 1):
        if shorter[i : i + min_run] in longer:
            return True
    return False


def _normalise_monthly(amount: float, frequency: str) -> float:
    """Convert an amount at *frequency* to its monthly equivalent."""
    multiplier = _FREQUENCY_TO_MONTHLY.get(frequency.lower(), 1.0)
    return round(amount * multiplier, 2)


def _days_since(date_str: str | None) -> int | None:
    """Return the number of days between *date_str* and today, or None."""
    if not date_str:
        return None
    from datetime import date, datetime

    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    return (date.today() - d).days


class SubscriptionAuditorAgent(BaseAgent):
    """Finds all recurring charges across bank + email and surfaces savings.

    Detects three opportunity types:

    * **forgotten** — charged but seemingly unused (old or tiny amount).
    * **price_increase** — email says a different amount than the bank.
    * **duplicate** — two subscriptions in the same category (streaming, etc.).
    """

    def __init__(self) -> None:
        super().__init__(
            name="subscription_auditor",
            description=(
                "Finds all recurring charges across bank accounts and email "
                "— surfaces forgotten subscriptions, price increases, and "
                "cancellation opportunities"
            ),
        )
        self._llm = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the five processing nodes."""
        builder.add_node("fetch_bank_subscriptions", self._fetch_bank_subscriptions)
        builder.add_node("fetch_email_subscriptions", self._fetch_email_subscriptions)
        builder.add_node("cross_reference", self._cross_reference)
        builder.add_node("identify_opportunities", self._identify_opportunities)
        builder.add_node("generate_report", self._generate_report)

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline."""
        builder.add_edge("initialise", "fetch_bank_subscriptions")
        builder.add_edge("fetch_bank_subscriptions", "fetch_email_subscriptions")
        builder.add_edge("fetch_email_subscriptions", "cross_reference")
        builder.add_edge("cross_reference", "identify_opportunities")
        builder.add_edge("identify_opportunities", "generate_report")
        builder.add_edge("generate_report", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: fetch_bank_subscriptions
    # ------------------------------------------------------------------

    async def _fetch_bank_subscriptions(self, state: AgentState) -> dict[str, Any]:
        """Pull recurring charges from Plaid."""
        try:
            bank_subs: list[dict[str, Any]] = await self.call_tool(
                "plaid_mcp",
                "get_recurring_transactions",
                {"user_id": state["user_id"]},
            )
        except RuntimeError as exc:
            logger.error("fetch_bank_subscriptions failed: %s", exc)
            return self.set_error(state, str(exc))

        step = f"Found {len(bank_subs)} recurring charges from bank"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "bank_subscriptions": bank_subs},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: fetch_email_subscriptions
    # ------------------------------------------------------------------

    async def _fetch_email_subscriptions(self, state: AgentState) -> dict[str, Any]:
        """Scan Gmail for subscription-related emails (best-effort)."""
        if state.get("status") == "failed":
            return {}

        try:
            email_subs: list[dict[str, Any]] = await self.call_tool(
                "gmail_mcp",
                "find_subscription_emails",
                {"months_back": 3},
            )
        except RuntimeError as exc:
            logger.warning("fetch_email_subscriptions failed (non-fatal): %s", exc)
            email_subs = []

        step = f"Found {len(email_subs)} subscription emails"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "email_subscriptions": email_subs},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: cross_reference
    # ------------------------------------------------------------------

    async def _cross_reference(self, state: AgentState) -> dict[str, Any]:
        """Merge bank and email subscriptions into a deduplicated inventory.

        Algorithm
        ---------
        1. Seed the unified list from bank subscriptions (these have the
           strongest signal — real money left the account).
        2. For each email subscription, try to match it against an existing
           bank entry using ``_fuzzy_match``.  If a match is found, upgrade
           the source to ``"both"`` and fill in any fields the bank entry
           was missing (e.g. email-only renewal dates).
        3. If no bank match exists, add the email subscription as a
           standalone ``"email"``-sourced entry.
        4. Normalise every subscription's amount to a monthly cadence and
           compute ``total_monthly_cost``.
        5. Sort the list by amount descending so the most expensive charges
           appear first.
        """
        if state.get("status") == "failed":
            return {}

        bank_subs: list[dict[str, Any]] = (
            state.get("input", {}).get("bank_subscriptions", [])
        )
        email_subs: list[dict[str, Any]] = (
            state.get("input", {}).get("email_subscriptions", [])
        )

        # --- Step 1: seed from bank entries ---
        unified: list[dict[str, Any]] = []
        for bs in bank_subs:
            raw_amount = _parse_dollar(bs.get("amount"))
            frequency = bs.get("frequency", "monthly")
            unified.append({
                "service_name": bs.get("merchant_name", "Unknown"),
                "amount": raw_amount,
                "monthly_amount": _normalise_monthly(raw_amount, frequency),
                "frequency": frequency,
                "source": "bank",
                "last_charged": bs.get("last_date"),
                "next_expected": bs.get("next_expected_date"),
                "confidence_score": 0.95,
            })

        # --- Step 2 & 3: merge email entries ---
        matched_bank_indices: set[int] = set()

        for es in email_subs:
            email_name = es.get("service_name", "")
            email_amount = _parse_dollar(es.get("amount"))
            best_idx: int | None = None

            for idx, entry in enumerate(unified):
                if idx in matched_bank_indices:
                    continue
                if _fuzzy_match(email_name, entry["service_name"]):
                    best_idx = idx
                    break

            if best_idx is not None:
                matched_bank_indices.add(best_idx)
                unified[best_idx]["source"] = "both"
                unified[best_idx]["confidence_score"] = 0.99
                if email_amount and not unified[best_idx]["amount"]:
                    unified[best_idx]["amount"] = email_amount
                    unified[best_idx]["monthly_amount"] = email_amount
                if es.get("renewal_date") and not unified[best_idx]["next_expected"]:
                    unified[best_idx]["next_expected"] = es["renewal_date"]
            else:
                unified.append({
                    "service_name": email_name,
                    "amount": email_amount,
                    "monthly_amount": email_amount,
                    "frequency": "monthly",
                    "source": "email",
                    "last_charged": None,
                    "next_expected": es.get("renewal_date"),
                    "confidence_score": 0.70,
                })

        # --- Step 4 & 5: totals and sort ---
        total_monthly = round(
            sum(s["monthly_amount"] for s in unified), 2
        )
        unified.sort(key=lambda s: s["monthly_amount"], reverse=True)

        step = (
            f"Cross-referenced — {len(unified)} unique subscriptions, "
            f"${total_monthly:.2f}/month total"
        )
        logger.info(step)
        return {
            "input": {
                **state.get("input", {}),
                "unified_subscriptions": unified,
                "total_monthly_cost": total_monthly,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: identify_opportunities
    # ------------------------------------------------------------------

    async def _identify_opportunities(self, state: AgentState) -> dict[str, Any]:
        """Flag forgotten, price-increased, and duplicate subscriptions."""
        if state.get("status") == "failed":
            return {}

        unified: list[dict[str, Any]] = (
            state.get("input", {}).get("unified_subscriptions", [])
        )
        bank_subs: list[dict[str, Any]] = (
            state.get("input", {}).get("bank_subscriptions", [])
        )
        email_subs: list[dict[str, Any]] = (
            state.get("input", {}).get("email_subscriptions", [])
        )

        opportunities: list[dict[str, Any]] = []

        # --- Forgotten subscriptions ---
        for sub in unified:
            days = _days_since(sub.get("last_charged"))
            amount = sub.get("monthly_amount", 0)
            if days is not None and days > _FORGOTTEN_DAYS_THRESHOLD:
                opportunities.append({
                    "service": sub["service_name"],
                    "opportunity_type": "forgotten",
                    "potential_savings": amount,
                    "action": (
                        f"Last charged {days} days ago — consider cancelling "
                        f"{sub['service_name']} to save ${amount:.2f}/mo"
                    ),
                })
            elif 0 < amount < _FORGOTTEN_AMOUNT_THRESHOLD:
                opportunities.append({
                    "service": sub["service_name"],
                    "opportunity_type": "forgotten",
                    "potential_savings": amount,
                    "action": (
                        f"Small recurring charge of ${amount:.2f}/mo — "
                        f"verify you still use {sub['service_name']}"
                    ),
                })

        # --- Price increases (email amount differs from bank amount) ---
        bank_by_name: dict[str, float] = {
            bs.get("merchant_name", ""): _parse_dollar(bs.get("amount"))
            for bs in bank_subs
        }
        for es in email_subs:
            email_name = es.get("service_name", "")
            email_amount = _parse_dollar(es.get("amount"))
            if not email_amount:
                continue
            for bank_name, bank_amount in bank_by_name.items():
                if not bank_amount:
                    continue
                if _fuzzy_match(email_name, bank_name) and email_amount != bank_amount:
                    diff = abs(email_amount - bank_amount)
                    opportunities.append({
                        "service": email_name,
                        "opportunity_type": "price_increase",
                        "potential_savings": diff,
                        "action": (
                            f"{email_name} was ${bank_amount:.2f}, now "
                            f"${email_amount:.2f} — review or downgrade plan"
                        ),
                    })
                    break

        # --- Duplicate services in the same category ---
        for category_group in _DUPLICATE_CATEGORIES:
            active_in_group: list[dict[str, Any]] = []
            for sub in unified:
                name_lower = sub["service_name"].lower()
                if any(keyword in name_lower for keyword in category_group):
                    active_in_group.append(sub)
            if len(active_in_group) >= 2:
                cheapest = min(active_in_group, key=lambda s: s["monthly_amount"])
                for dup in active_in_group:
                    if dup is cheapest:
                        continue
                    opportunities.append({
                        "service": dup["service_name"],
                        "opportunity_type": "duplicate",
                        "potential_savings": dup["monthly_amount"],
                        "action": (
                            f"You also pay for {cheapest['service_name']} — "
                            f"cancel {dup['service_name']} to save "
                            f"${dup['monthly_amount']:.2f}/mo"
                        ),
                    })

        step = f"Found {len(opportunities)} cancellation/savings opportunities"
        logger.info(step)
        return {
            "input": {
                **state.get("input", {}),
                "opportunities": opportunities,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 5: generate_report
    # ------------------------------------------------------------------

    async def _generate_report(self, state: AgentState) -> dict[str, Any]:
        """Ask Claude for a friendly financial summary and set final output."""
        if state.get("status") == "failed":
            return {}

        unified: list[dict[str, Any]] = (
            state.get("input", {}).get("unified_subscriptions", [])
        )
        opportunities: list[dict[str, Any]] = (
            state.get("input", {}).get("opportunities", [])
        )
        total_monthly: float = state.get("input", {}).get("total_monthly_cost", 0)
        total_annual = round(total_monthly * 12, 2)
        potential_savings = round(
            sum(o.get("potential_savings", 0) for o in opportunities), 2
        )

        prompt = (
            f"The user has {len(unified)} active subscriptions costing "
            f"${total_monthly:.2f}/month (${total_annual:.2f}/year).\n\n"
            f"Subscriptions: {unified}\n"
            f"Opportunities: {opportunities}\n\n"
            "Write a friendly 3-4 sentence financial summary. Mention the "
            "total cost, highlight the biggest opportunity to save money, "
            "and suggest one specific action they should take today."
        )

        try:
            response = self._llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            summary: str = response.content[0].text
        except Exception as exc:
            logger.warning("Claude summarisation failed: %s", exc)
            summary = (
                f"You have {len(unified)} active subscriptions totalling "
                f"${total_monthly:.2f}/month. We found {len(opportunities)} "
                f"opportunities that could save you ${potential_savings:.2f}/mo."
            )

        output = {
            "subscriptions": unified,
            "total_monthly_cost": total_monthly,
            "total_annual_cost": total_annual,
            "opportunities": opportunities,
            "potential_savings": potential_savings,
            "summary": summary,
        }

        step = "Generated report"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }
