"""Bill Negotiator — identifies overpaid bills and writes negotiation scripts.

This agent scans the user's spending (Plaid) and inbox (Gmail) for bills in
categories that are routinely negotiable — internet, phone, insurance, cable,
gym memberships — then uses Claude to:

1. Research competitive market rates for each bill.
2. Identify bills where the user is paying above the market average.
3. Generate a ready-to-use, verbatim phone script the user can read aloud
   to their provider's retention department to get the bill lowered.

Graph pipeline::

    initialise
      → fetch_bills
      → research_market_rates
      → generate_scripts
      → store_recommendations
      → compile_report
      → finalise → END
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent

logger = logging.getLogger(__name__)

_DOLLAR_RE = re.compile(r"\$?([\d,]+\.?\d*)")

NEGOTIABLE_CATEGORIES: list[str] = [
    "internet",
    "cable",
    "phone",
    "wireless",
    "insurance",
    "utilities",
    "telecom",
    "gym",
]

_MINIMUM_SAVINGS_THRESHOLD = 5.00


def _parse_dollar(raw: str | float | int | None) -> float:
    """Extract a numeric dollar value from ``"$89.00"``, ``89.0``, or ``None``."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    match = _DOLLAR_RE.search(str(raw))
    if match:
        return float(match.group(1).replace(",", ""))
    return 0.0


def _category_matches(text: str) -> str | None:
    """Return the first NEGOTIABLE category that appears in *text* (case-insensitive)."""
    lower = text.lower()
    for cat in NEGOTIABLE_CATEGORIES:
        if cat in lower:
            return cat
    return None


def _extract_provider(sender: str) -> str:
    """Best-effort provider name from an email ``From`` header."""
    if "<" in sender:
        name = sender.split("<")[0].strip().strip('"')
        if name:
            return name
    domain = sender.split("@")[-1].split(">")[0]
    return domain.split(".")[0].title()


class BillNegotiatorAgent(BaseAgent):
    """Identifies overpaid bills and generates negotiation scripts.

    Covers internet, phone, insurance, cable, utilities, and gym
    memberships.  For each bill above the market average the agent produces
    a step-by-step phone script the user can read verbatim to the
    provider's retention department.
    """

    def __init__(self) -> None:
        super().__init__(
            name="bill_negotiator",
            description=(
                "Identifies overpaid bills and generates negotiation scripts "
                "to lower them — internet, phone, insurance, cable"
            ),
        )
        self._llm = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the five processing nodes."""
        builder.add_node("fetch_bills", self._fetch_bills)
        builder.add_node("research_market_rates", self._research_market_rates)
        builder.add_node("generate_scripts", self._generate_scripts)
        builder.add_node("store_recommendations", self._store_recommendations)
        builder.add_node("compile_report", self._compile_report)

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline."""
        builder.add_edge("initialise", "fetch_bills")
        builder.add_edge("fetch_bills", "research_market_rates")
        builder.add_edge("research_market_rates", "generate_scripts")
        builder.add_edge("generate_scripts", "store_recommendations")
        builder.add_edge("store_recommendations", "compile_report")
        builder.add_edge("compile_report", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: fetch_bills
    # ------------------------------------------------------------------

    async def _fetch_bills(self, state: AgentState) -> dict[str, Any]:
        """Pull 90-day spending by category from Plaid and recent bill emails.

        Merges both sources and keeps only categories in
        ``NEGOTIABLE_CATEGORIES``.
        """
        user_id: str = state["user_id"]
        today = date.today()
        start = today - timedelta(days=90)

        # --- Plaid: spending by category over the last 90 days ---
        try:
            spending: dict[str, float] = await self.call_tool(
                "plaid_mcp",
                "get_spending_by_category",
                {
                    "user_id": user_id,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": today.strftime("%Y-%m-%d"),
                },
            )
        except RuntimeError as exc:
            logger.error("fetch_bills: Plaid spending failed: %s", exc)
            return self.set_error(state, str(exc))

        # --- Gmail: recent bill-related emails (best-effort) ---
        email_bills: list[dict[str, Any]] = []
        try:
            email_bills = await self.call_tool(
                "gmail_mcp",
                "scan_financial_emails",
                {"days_back": 30},
            )
        except RuntimeError as exc:
            logger.warning("fetch_bills: Gmail scan failed (non-fatal): %s", exc)

        # --- Filter to negotiable categories ---
        bills: list[dict[str, Any]] = []
        seen_providers: set[str] = set()

        for category, total_90d in spending.items():
            matched_cat = _category_matches(category)
            if matched_cat is None:
                continue
            monthly = round(total_90d / 3, 2)
            provider = category.title()
            if provider.lower() in seen_providers:
                continue
            seen_providers.add(provider.lower())
            bills.append({
                "provider": provider,
                "category": matched_cat,
                "monthly_amount": monthly,
                "source": "bank",
            })

        for email in email_bills:
            sender = email.get("sender", "")
            subject = email.get("subject", "")
            combined = f"{sender} {subject}"
            matched_cat = _category_matches(combined)
            if matched_cat is None:
                continue
            provider = _extract_provider(sender)
            if provider.lower() in seen_providers:
                continue
            seen_providers.add(provider.lower())
            amount = _parse_dollar(email.get("amount_mentioned"))
            if amount <= 0:
                continue
            bills.append({
                "provider": provider,
                "category": matched_cat,
                "monthly_amount": amount,
                "source": "email",
            })

        total = round(sum(b["monthly_amount"] for b in bills), 2)
        step = f"Found {len(bills)} negotiable bills totaling ${total:.2f}/month"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "bills": bills},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: research_market_rates
    # ------------------------------------------------------------------

    async def _research_market_rates(self, state: AgentState) -> dict[str, Any]:
        """Use Claude to estimate competitive pricing for each bill category.

        Bills where the potential saving is below ``$_MINIMUM_SAVINGS_THRESHOLD``
        are dropped — not worth the user's time to negotiate.
        """
        if state.get("status") == "failed":
            return {}

        bills: list[dict[str, Any]] = state.get("input", {}).get("bills", [])
        negotiable: list[dict[str, Any]] = []
        total_potential = 0.0

        for bill in bills:
            prompt = (
                f"What is the typical monthly price range for {bill['category']} "
                f"service from a competing provider in the US in 2024? "
                f"The user currently pays ${bill['monthly_amount']:.2f}/month "
                f"to {bill['provider']}.\n\n"
                f"Return ONLY valid JSON with these exact keys:\n"
                f'{{"min_price": <float>, "max_price": <float>, '
                f'"average_price": <float>, '
                f'"top_3_competitors": ["name1", "name2", "name3"]}}'
            )

            market_rates: dict[str, Any] = {
                "min_price": 0,
                "max_price": 0,
                "average_price": 0,
                "top_3_competitors": [],
            }

            try:
                response = self._llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=250,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    market_rates = json.loads(json_match.group())
            except Exception as exc:
                logger.warning(
                    "Market-rate lookup failed for %s: %s",
                    bill["provider"],
                    exc,
                )

            avg_price = float(market_rates.get("average_price", 0))
            potential_savings = round(bill["monthly_amount"] - avg_price, 2)

            if potential_savings < _MINIMUM_SAVINGS_THRESHOLD:
                continue

            negotiable.append({
                **bill,
                "market_rates": market_rates,
                "potential_savings": potential_savings,
            })
            total_potential += potential_savings

        total_potential = round(total_potential, 2)
        step = (
            f"Researched rates — {len(negotiable)} bills worth negotiating, "
            f"${total_potential:.2f} potential monthly savings"
        )
        logger.info(step)
        return {
            "input": {
                **state.get("input", {}),
                "negotiable_bills": negotiable,
                "total_potential_savings": total_potential,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: generate_scripts
    # ------------------------------------------------------------------

    async def _generate_scripts(self, state: AgentState) -> dict[str, Any]:
        """Generate a verbatim phone negotiation script for each negotiable bill.

        Each script is structured as a five-part dialogue the user can read
        aloud to the provider's retention department:

        1. **Opening** — polite greeting with account context.
        2. **Leverage** — cite competitor pricing and loyalty tenure.
        3. **The ask** — request a specific dollar amount.
        4. **Objection handling** — what to say if the rep pushes back.
        5. **Closing** — confirm the new rate and get a reference number.
        """
        if state.get("status") == "failed":
            return {}

        negotiable: list[dict[str, Any]] = (
            state.get("input", {}).get("negotiable_bills", [])
        )
        scripts: list[dict[str, Any]] = []

        for bill in negotiable:
            provider = bill["provider"]
            category = bill["category"]
            amount = bill["monthly_amount"]
            rates = bill.get("market_rates", {})
            avg_price = rates.get("average_price", 0)
            competitors = rates.get("top_3_competitors", [])

            prompt = (
                f"Write a phone negotiation script for a customer calling "
                f"{provider} to lower their {category} bill.\n\n"
                f"Current bill: ${amount:.2f}/month\n"
                f"Market average: ${avg_price:.2f}/month\n"
                f"Competitors offering lower rates: {', '.join(competitors) if competitors else 'various competitors'}\n\n"
                f"Structure the script with these exact sections:\n"
                f"1. OPENING — A polite greeting that mentions being a loyal "
                f"customer who wants to discuss their current rate.\n"
                f"2. LEVERAGE — Cite specific competitor prices. Mention "
                f"you've been comparing options and found better rates.\n"
                f"3. THE ASK — Request a specific lower rate "
                f"(${avg_price:.2f}/month or close to it). Be direct.\n"
                f"4. IF THEY PUSH BACK — What to say if the first rep "
                f"says no. Ask for the retention/cancellation department.\n"
                f"5. CLOSING — Confirm the new rate, ask for a confirmation "
                f"number, and ask when it takes effect.\n\n"
                f"Write it as a verbatim script the user reads word-for-word. "
                f"Use 'I' voice. Keep it under 300 words."
            )

            try:
                response = self._llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=600,
                    messages=[{"role": "user", "content": prompt}],
                )
                script_text: str = response.content[0].text
            except Exception as exc:
                logger.warning("Script generation failed for %s: %s", provider, exc)
                script_text = (
                    f"Call {provider} and ask to speak with their retention "
                    f"department. Mention that competitors like "
                    f"{', '.join(competitors[:2]) or 'others'} offer "
                    f"{category} for ~${avg_price:.2f}/month, and request "
                    f"they match that rate or apply a loyalty discount."
                )

            scripts.append({
                "provider": provider,
                "category": category,
                "current_amount": amount,
                "target_amount": round(avg_price, 2),
                "potential_savings": bill["potential_savings"],
                "competitors": competitors,
                "script": script_text,
            })

        step = f"Generated {len(scripts)} negotiation scripts"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "scripts": scripts},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: store_recommendations
    # ------------------------------------------------------------------

    async def _store_recommendations(self, state: AgentState) -> dict[str, Any]:
        """Persist each negotiation recommendation as an alert in the graph."""
        if state.get("status") == "failed":
            return {}

        scripts: list[dict[str, Any]] = state.get("input", {}).get("scripts", [])
        user_id: str = state["user_id"]
        stored = 0

        for item in scripts:
            try:
                await self.call_tool(
                    "graph_mcp",
                    "create_alert_node",
                    {
                        "user_id": user_id,
                        "alert_type": "bill_negotiation",
                        "title": f"Negotiate {item['provider']} {item['category']} bill",
                        "message": (
                            f"You pay ${item['current_amount']:.2f}/mo — "
                            f"market average is ${item['target_amount']:.2f}/mo. "
                            f"Potential savings: ${item['potential_savings']:.2f}/mo."
                        ),
                        "severity": "high" if item["potential_savings"] > 20 else "medium",
                        "amount": item["current_amount"],
                    },
                )
                stored += 1
            except RuntimeError as exc:
                logger.warning(
                    "Failed to store recommendation for %s: %s",
                    item["provider"],
                    exc,
                )

        step = f"Stored {stored} recommendations in knowledge graph"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "recommendations_stored": stored},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 5: compile_report
    # ------------------------------------------------------------------

    async def _compile_report(self, state: AgentState) -> dict[str, Any]:
        """Produce a Claude-generated executive summary and set final output."""
        if state.get("status") == "failed":
            return {}

        bills: list[dict[str, Any]] = state.get("input", {}).get("bills", [])
        scripts: list[dict[str, Any]] = state.get("input", {}).get("scripts", [])
        total_potential: float = state.get("input", {}).get(
            "total_potential_savings", 0
        )
        annual_potential = round(total_potential * 12, 2)

        if scripts:
            biggest = max(scripts, key=lambda s: s["potential_savings"])
            prompt = (
                f"The user has {len(bills)} recurring bills. After researching "
                f"market rates we identified {len(scripts)} that are worth "
                f"negotiating, with a combined potential savings of "
                f"${total_potential:.2f}/month (${annual_potential:.2f}/year).\n\n"
                f"The biggest opportunity is {biggest['provider']} "
                f"({biggest['category']}): currently ${biggest['current_amount']:.2f}/mo, "
                f"market average ${biggest['target_amount']:.2f}/mo.\n\n"
                f"Write a friendly 3-4 sentence summary. Mention the total "
                f"potential savings, highlight the single biggest opportunity, "
                f"and encourage them to start with that one call today."
            )
            try:
                response = self._llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                summary: str = response.content[0].text
            except Exception as exc:
                logger.warning("Summary generation failed: %s", exc)
                summary = (
                    f"We found {len(scripts)} bills you can negotiate to save "
                    f"${total_potential:.2f}/month (${annual_potential:.2f}/year). "
                    f"Start with {biggest['provider']} — you could save "
                    f"${biggest['potential_savings']:.2f}/month with one phone call."
                )
        else:
            summary = (
                "Good news — your bills look competitive with current market "
                "rates.  No negotiation opportunities found right now."
            )

        output = {
            "bills_analyzed": len(bills),
            "negotiable_count": len(scripts),
            "total_potential_monthly_savings": total_potential,
            "total_potential_annual_savings": annual_potential,
            "scripts": scripts,
            "summary": summary,
        }

        step = "Compiled report"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }
