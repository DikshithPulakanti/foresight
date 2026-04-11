"""Calendar Planner — connects upcoming events to spending and budget reality.

This agent looks ahead at the user's calendar, estimates the cost of every
upcoming financial event, and cross-checks whether the money will actually be
there when the bill lands.  Events that are "at risk" get an automatic
calendar reminder two weeks early so the user has time to adjust.

The cost-estimation step is one of the cleverest uses of LLMs in Foresight:
when a calendar event has no dollar amount attached (e.g. "Flight to NYC"),
Claude is asked to produce a realistic min/max/likely cost estimate based on
the event title and category.  This lets the planner reason about expenses
that haven't been formally quoted yet.

Graph pipeline::

    initialise
      → fetch_upcoming_events
      → fetch_spending_context
      → estimate_event_costs
      → check_budget_alignment
      → generate_plan
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

_HISTORICAL_DAYS = 90
_BUDGET_TIGHT_MULTIPLIER = 1.2
_REMINDER_LEAD_DAYS = 14


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except (ValueError, TypeError):
        return None


def _is_checking(account: dict[str, Any]) -> bool:
    acct_type = str(account.get("type", "")).lower()
    subtype = str(account.get("subtype", "")).lower()
    return "depository" in acct_type or "checking" in subtype


class CalendarPlannerAgent(BaseAgent):
    """Connects upcoming calendar events to spending and budget reality.

    For each event in the planning window the agent produces a
    ``budget_status`` of ``"comfortable"``, ``"tight"``, or ``"at_risk"``
    and creates proactive calendar reminders for at-risk events.
    """

    def __init__(self) -> None:
        super().__init__(
            name="calendar_planner",
            description=(
                "Connects upcoming calendar events to spending patterns — "
                "estimates costs, checks budget, and creates reminders "
                "before big expenses hit"
            ),
        )
        self._llm = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the five processing nodes."""
        builder.add_node("fetch_upcoming_events", self._fetch_upcoming_events)
        builder.add_node("fetch_spending_context", self._fetch_spending_context)
        builder.add_node("estimate_event_costs", self._estimate_event_costs)
        builder.add_node("check_budget_alignment", self._check_budget_alignment)
        builder.add_node("generate_plan", self._generate_plan)

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline."""
        builder.add_edge("initialise", "fetch_upcoming_events")
        builder.add_edge("fetch_upcoming_events", "fetch_spending_context")
        builder.add_edge("fetch_spending_context", "estimate_event_costs")
        builder.add_edge("estimate_event_costs", "check_budget_alignment")
        builder.add_edge("check_budget_alignment", "generate_plan")
        builder.add_edge("generate_plan", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: fetch_upcoming_events
    # ------------------------------------------------------------------

    async def _fetch_upcoming_events(self, state: AgentState) -> dict[str, Any]:
        """Pull financial calendar events and the payday schedule."""
        days_ahead: int = state.get("input", {}).get("days_ahead", 30)

        try:
            events: list[dict[str, Any]] = await self.call_tool(
                "calendar_mcp",
                "get_upcoming_financial_events",
                {"days_ahead": days_ahead},
            )
        except RuntimeError as exc:
            logger.error("fetch_upcoming_events failed: %s", exc)
            return self.set_error(state, str(exc))

        payday_info: dict[str, Any] = {}
        try:
            payday_info = await self.call_tool(
                "calendar_mcp",
                "get_payday_schedule",
                {},
            )
        except RuntimeError as exc:
            logger.warning("Payday schedule unavailable: %s", exc)

        payday_date = payday_info.get("next_payday", "unknown")

        step = f"Found {len(events)} upcoming financial events, next payday: {payday_date}"
        logger.info(step)
        return {
            "input": {
                **state.get("input", {}),
                "events": events,
                "payday_info": payday_info,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: fetch_spending_context
    # ------------------------------------------------------------------

    async def _fetch_spending_context(self, state: AgentState) -> dict[str, Any]:
        """Fetch 90-day spending by category and current account balances."""
        if state.get("status") == "failed":
            return {}

        user_id: str = state["user_id"]
        today = date.today()
        start = today - timedelta(days=_HISTORICAL_DAYS)

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
            logger.error("fetch_spending_context: spending failed: %s", exc)
            return self.set_error(state, str(exc))

        try:
            accounts: list[dict[str, Any]] = await self.call_tool(
                "plaid_mcp",
                "get_account_balances",
                {"user_id": user_id},
            )
        except RuntimeError as exc:
            logger.error("fetch_spending_context: balances failed: %s", exc)
            return self.set_error(state, str(exc))

        current_balance = round(
            sum(
                a.get("balance_current", 0) or 0
                for a in accounts
                if _is_checking(a)
            ),
            2,
        )

        total_spend = sum(spending.values())
        avg_daily_net = round(-total_spend / max(_HISTORICAL_DAYS, 1), 2)

        step = f"Fetched 90-day spending context, current balance ${current_balance:.2f}"
        logger.info(step)
        return {
            "input": {
                **state.get("input", {}),
                "spending_by_category": spending,
                "accounts": accounts,
                "current_balance": current_balance,
                "avg_daily_net": avg_daily_net,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: estimate_event_costs
    # ------------------------------------------------------------------

    async def _estimate_event_costs(self, state: AgentState) -> dict[str, Any]:
        """Attach a dollar estimate to every upcoming event.

        Events that already carry an ``estimated_amount`` from the calendar
        are marked ``cost_confidence="known"``.

        Events *without* a known amount are sent to Claude, which estimates
        a realistic min/max/likely cost based on the event title and
        category.  This is an intentional use of LLM judgment: calendar
        events like "Flight to NYC" or "Car insurance" are common enough
        that Claude can produce a useful ballpark, and the planner only
        needs directional accuracy to flag budget risk — not penny-perfect
        forecasts.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        events: list[dict[str, Any]] = inp.get("events", [])
        enriched: list[dict[str, Any]] = []

        for event in events:
            known_amount = event.get("estimated_amount")

            if known_amount is not None and known_amount > 0:
                enriched.append({
                    **event,
                    "estimated_cost": float(known_amount),
                    "cost_confidence": "known",
                    "cost_range": {
                        "min_cost": float(known_amount),
                        "max_cost": float(known_amount),
                        "likely_cost": float(known_amount),
                    },
                })
                continue

            # --- LLM cost estimation for events without a dollar figure ---
            title = event.get("title", "Unknown event")
            event_type = event.get("type", "unknown")

            prompt = (
                f'Estimate the typical cost for this financial event: "{title}"\n'
                f"Category: {event_type}\n"
                f"Based on typical US spending, give a realistic cost estimate.\n"
                f"Return ONLY valid JSON:\n"
                f'{{"min_cost": <float>, "max_cost": <float>, '
                f'"likely_cost": <float>, "reasoning": "<1 sentence>"}}'
            )

            cost_range: dict[str, Any] = {
                "min_cost": 0,
                "max_cost": 0,
                "likely_cost": 0,
                "reasoning": "Could not estimate",
            }

            try:
                response = self._llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    cost_range = json.loads(json_match.group())
            except Exception as exc:
                logger.warning("Cost estimation failed for '%s': %s", title, exc)

            enriched.append({
                **event,
                "estimated_cost": float(cost_range.get("likely_cost", 0)),
                "cost_confidence": "estimated",
                "cost_range": cost_range,
            })

        total_upcoming = round(sum(e["estimated_cost"] for e in enriched), 2)

        step = f"Estimated costs for {len(enriched)} events — total upcoming: ${total_upcoming:.2f}"
        logger.info(step)
        return {
            "input": {
                **inp,
                "enriched_events": enriched,
                "total_upcoming_spend": total_upcoming,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: check_budget_alignment
    # ------------------------------------------------------------------

    async def _check_budget_alignment(self, state: AgentState) -> dict[str, Any]:
        """For each event, project the balance on that date and flag risk.

        Budget statuses:

        * ``"at_risk"`` — projected balance < event cost.
        * ``"tight"`` — projected balance < event cost × 1.2 (no cushion).
        * ``"comfortable"`` — enough headroom.

        At-risk events get an automatic calendar reminder placed two weeks
        before the event date.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        enriched: list[dict[str, Any]] = inp.get("enriched_events", [])
        current_balance: float = inp.get("current_balance", 0)
        avg_daily_net: float = inp.get("avg_daily_net", 0)
        user_id: str = state["user_id"]
        today = date.today()

        budget_plan: list[dict[str, Any]] = []
        counts: dict[str, int] = {"comfortable": 0, "tight": 0, "at_risk": 0}
        reminders_created = 0

        for event in enriched:
            event_date = _parse_date(event.get("date"))
            days_until = event.get("days_until") or (
                (event_date - today).days if event_date else 30
            )
            cost = event.get("estimated_cost", 0)

            projected_balance = round(
                current_balance + (days_until * avg_daily_net), 2
            )

            if projected_balance < cost:
                status = "at_risk"
            elif projected_balance < cost * _BUDGET_TIGHT_MULTIPLIER:
                status = "tight"
            else:
                status = "comfortable"

            counts[status] += 1

            # Create a calendar reminder for at-risk events
            if status == "at_risk" and event_date:
                reminder_date = event_date - timedelta(days=_REMINDER_LEAD_DAYS)
                if reminder_date <= today:
                    reminder_date = today + timedelta(days=1)
                try:
                    await self.call_tool(
                        "calendar_mcp",
                        "add_financial_reminder",
                        {
                            "title": (
                                f"Budget alert: {event.get('title', 'Event')} "
                                f"costs ${cost:.2f}"
                            ),
                            "date": reminder_date.strftime("%Y-%m-%d"),
                            "description": (
                                f"You have {event.get('title', 'an event')} "
                                f"coming up costing ~${cost:.2f}. "
                                f"Projected balance on that day: "
                                f"${projected_balance:.2f}. Start saving now."
                            ),
                            "amount": cost,
                        },
                    )
                    reminders_created += 1
                except RuntimeError as exc:
                    logger.warning("Failed to create reminder: %s", exc)

            budget_plan.append({
                "event": event.get("title", "Unknown"),
                "event_type": event.get("type", "unknown"),
                "cost": cost,
                "cost_confidence": event.get("cost_confidence", "estimated"),
                "event_date": event.get("date"),
                "days_until": days_until,
                "budget_status": status,
                "projected_balance_on_date": projected_balance,
            })

        step = (
            f"Budget check: {counts['comfortable']} comfortable, "
            f"{counts['tight']} tight, {counts['at_risk']} at risk"
        )
        logger.info(step)
        return {
            "input": {
                **inp,
                "budget_plan": budget_plan,
                "budget_counts": counts,
                "reminders_created": reminders_created,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 5: generate_plan
    # ------------------------------------------------------------------

    async def _generate_plan(self, state: AgentState) -> dict[str, Any]:
        """Ask Claude for a practical calendar-budget summary and set output."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        budget_plan: list[dict[str, Any]] = inp.get("budget_plan", [])
        counts: dict[str, int] = inp.get("budget_counts", {})
        total_upcoming: float = inp.get("total_upcoming_spend", 0)
        current_balance: float = inp.get("current_balance", 0)
        payday_date: str = inp.get("payday_info", {}).get("next_payday", "unknown")
        reminders_created: int = inp.get("reminders_created", 0)

        comfortable_events = [
            e for e in budget_plan if e["budget_status"] == "comfortable"
        ]
        tight_events = [
            e for e in budget_plan if e["budget_status"] == "tight"
        ]
        at_risk_events = [
            e for e in budget_plan if e["budget_status"] == "at_risk"
        ]

        prompt = (
            f"The user has {len(budget_plan)} upcoming financial events "
            f"totaling ${total_upcoming:.2f}.\n"
            f"Current balance: ${current_balance:.2f}\n"
            f"Next payday: {payday_date}\n\n"
            f"Budget breakdown:\n"
            f"- Comfortable ({len(comfortable_events)}): "
            f"{[e['event'] for e in comfortable_events]}\n"
            f"- Tight ({len(tight_events)}): "
            f"{[e['event'] + ' $' + f'{e[\"cost\"]:.0f}' for e in tight_events]}\n"
            f"- At risk ({len(at_risk_events)}): "
            f"{[e['event'] + ' $' + f'{e[\"cost\"]:.0f}' for e in at_risk_events]}\n\n"
            f"Write a practical 3-4 sentence financial calendar summary. "
            f"Mention which events need the most attention and one specific "
            f"saving tip for the tightest upcoming expense."
        )

        try:
            response = self._llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            summary: str = response.content[0].text
        except Exception as exc:
            logger.warning("Plan summary generation failed: %s", exc)
            if at_risk_events:
                biggest_risk = max(at_risk_events, key=lambda e: e["cost"])
                summary = (
                    f"You have ${total_upcoming:.2f} in upcoming expenses "
                    f"against a ${current_balance:.2f} balance. "
                    f"Your biggest risk is {biggest_risk['event']} "
                    f"(${biggest_risk['cost']:.2f}) — start setting aside "
                    f"money now. Next payday: {payday_date}."
                )
            else:
                summary = (
                    f"You have {len(budget_plan)} upcoming events totaling "
                    f"${total_upcoming:.2f}. With a ${current_balance:.2f} "
                    f"balance, you're in good shape."
                )

        output: dict[str, Any] = {
            "upcoming_events": budget_plan,
            "total_upcoming_spend": total_upcoming,
            "current_balance": current_balance,
            "next_payday": payday_date,
            "at_risk_count": counts.get("at_risk", 0),
            "tight_count": counts.get("tight", 0),
            "comfortable_count": counts.get("comfortable", 0),
            "reminders_created": reminders_created,
            "summary": summary,
        }

        step = "Generated plan"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }
