"""Cashflow Prophet — predicts bank balance 30 & 60 days out.

This is the centrepiece forecasting agent.  It assembles every signal
Foresight has about the user's money — historical cashflow from the
knowledge graph, current balances from Plaid, recurring charges, payday
schedule, and upcoming calendar events — then rolls a day-by-day balance
projection forward for the next 60 days.

If the projection shows the balance crossing a danger threshold the agent
fires a proactive alert *before* the user has any idea a shortfall is
coming.

Algorithm (NODE 3: run_forecast)
--------------------------------
1.  Start with today's checking-account balance.
2.  Build a sparse ``daily_cashflows`` map from known future events
    (recurring charges, paydays, calendar events).
3.  For every day with no known event, fall back to a
    ``average_daily_net`` computed from the last 6 months of actuals.
4.  Roll forward, accumulating projected balance day by day.
5.  Extract milestones: 30-day balance, 60-day balance, lowest point,
    first day below zero, first day below the $500 safety buffer.

Risk tiers (NODE 4: detect_risks)
---------------------------------
* **CRITICAL** — balance goes negative within 30 days.
* **HIGH** — balance drops below $100 within 30 days.
* **MEDIUM** — balance drops below $500 within 30 days.
* **LOW** — balance declines 30 %+ over 60 days but stays above $500.
* **NONE** — forecast looks healthy.

Graph pipeline::

    initialise
      → fetch_historical_data
      → fetch_upcoming_events
      → run_forecast
      → detect_risks
      → create_alerts
      → generate_forecast_report
      → finalise → END
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Optional

import anthropic
from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent

logger = logging.getLogger(__name__)

_SAFETY_BUFFER = 500.0
_DANGER_THRESHOLD = 100.0
_FORECAST_DAYS = 60
_HISTORICAL_MONTHS = 6
_HISTORICAL_DAYS = _HISTORICAL_MONTHS * 30


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_date(raw: str | None) -> date | None:
    """Best-effort ``YYYY-MM-DD`` → ``date``.  Returns *None* on failure."""
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except (ValueError, TypeError):
        return None


def _is_checking(account: dict[str, Any]) -> bool:
    """Heuristic: treat depository / checking accounts as the cash pool."""
    acct_type = str(account.get("type", "")).lower()
    subtype = str(account.get("subtype", "")).lower()
    return "depository" in acct_type or "checking" in subtype


class CashflowProphetAgent(BaseAgent):
    """Predicts bank balance 30 and 60 days out and fires shortfall alerts."""

    def __init__(self) -> None:
        super().__init__(
            name="cashflow_prophet",
            description=(
                "Predicts bank balance 30 and 60 days into the future — "
                "fires early warnings when a cash shortfall is coming"
            ),
        )
        self._llm = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the six processing nodes."""
        builder.add_node("fetch_historical_data", self._fetch_historical_data)
        builder.add_node("fetch_upcoming_events", self._fetch_upcoming_events)
        builder.add_node("run_forecast", self._run_forecast)
        builder.add_node("detect_risks", self._detect_risks)
        builder.add_node("create_alerts", self._create_alerts)
        builder.add_node("generate_forecast_report", self._generate_forecast_report)

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline."""
        builder.add_edge("initialise", "fetch_historical_data")
        builder.add_edge("fetch_historical_data", "fetch_upcoming_events")
        builder.add_edge("fetch_upcoming_events", "run_forecast")
        builder.add_edge("run_forecast", "detect_risks")
        builder.add_edge("detect_risks", "create_alerts")
        builder.add_edge("create_alerts", "generate_forecast_report")
        builder.add_edge("generate_forecast_report", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: fetch_historical_data
    # ------------------------------------------------------------------

    async def _fetch_historical_data(self, state: AgentState) -> dict[str, Any]:
        """Pull 6 months of cashflow history and current balances."""
        user_id: str = state["user_id"]

        try:
            transactions: list[dict[str, Any]] = await self.call_tool(
                "graph_mcp",
                "get_cashflow_data",
                {"user_id": user_id, "months_back": _HISTORICAL_MONTHS},
            )
        except RuntimeError as exc:
            logger.error("fetch_historical_data: cashflow failed: %s", exc)
            return self.set_error(state, str(exc))

        try:
            accounts: list[dict[str, Any]] = await self.call_tool(
                "plaid_mcp",
                "get_account_balances",
                {"user_id": user_id},
            )
        except RuntimeError as exc:
            logger.error("fetch_historical_data: balances failed: %s", exc)
            return self.set_error(state, str(exc))

        current_balance = round(
            sum(
                a.get("balance_current", 0) or 0
                for a in accounts
                if _is_checking(a)
            ),
            2,
        )

        step = (
            f"Fetched {_HISTORICAL_MONTHS} months of cashflow data, "
            f"current balance ${current_balance:.2f}"
        )
        logger.info(step)
        return {
            "input": {
                **state.get("input", {}),
                "transactions": transactions,
                "accounts": accounts,
                "current_balance": current_balance,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: fetch_upcoming_events
    # ------------------------------------------------------------------

    async def _fetch_upcoming_events(self, state: AgentState) -> dict[str, Any]:
        """Collect every known future cash-in / cash-out over the next 60 days.

        Sources:
        * Google Calendar financial events (rent, travel, etc.)
        * Inferred payday schedule
        * Recurring bank charges from Plaid
        """
        if state.get("status") == "failed":
            return {}

        user_id: str = state["user_id"]
        transactions: list[dict[str, Any]] = (
            state.get("input", {}).get("transactions", [])
        )

        # --- Calendar events ---
        calendar_events: list[dict[str, Any]] = []
        try:
            calendar_events = await self.call_tool(
                "calendar_mcp",
                "get_upcoming_financial_events",
                {"days_ahead": _FORECAST_DAYS},
            )
        except RuntimeError as exc:
            logger.warning("Calendar events unavailable: %s", exc)

        # --- Payday schedule ---
        payday_info: dict[str, Any] = {}
        try:
            payday_info = await self.call_tool(
                "calendar_mcp",
                "get_payday_schedule",
                {},
            )
        except RuntimeError as exc:
            logger.warning("Payday schedule unavailable: %s", exc)

        # --- Recurring charges ---
        recurring: list[dict[str, Any]] = []
        try:
            recurring = await self.call_tool(
                "plaid_mcp",
                "get_recurring_transactions",
                {"user_id": user_id},
            )
        except RuntimeError as exc:
            logger.warning("Recurring transactions unavailable: %s", exc)

        # --- Average monthly income (from historical data for payday amounts) ---
        total_income = sum(
            abs(t.get("amount", 0))
            for t in transactions
            if str(t.get("type", "")).lower() == "income"
        )
        avg_monthly_income = round(total_income / max(_HISTORICAL_MONTHS, 1), 2)

        # --- Build unified upcoming_events list ---
        today = date.today()
        upcoming: list[dict[str, Any]] = []

        for ev in calendar_events:
            ev_date = _parse_date(ev.get("date"))
            if ev_date is None or ev_date <= today:
                continue
            amount = ev.get("estimated_amount")
            ev_type = str(ev.get("type", "")).lower()
            if amount is not None:
                sign = 1.0 if ev_type in ("salary", "payday") else -1.0
                upcoming.append({
                    "date": ev_date.isoformat(),
                    "amount": round(sign * abs(amount), 2),
                    "source": "calendar",
                    "description": ev.get("title", "Calendar event"),
                })

        for charge in recurring:
            next_date = _parse_date(charge.get("next_expected_date"))
            if next_date is None or next_date <= today:
                continue
            if next_date > today + timedelta(days=_FORECAST_DAYS):
                continue
            raw_amount = charge.get("amount")
            if raw_amount is None:
                continue
            upcoming.append({
                "date": next_date.isoformat(),
                "amount": round(-abs(float(raw_amount)), 2),
                "source": "recurring",
                "description": charge.get("merchant_name", "Recurring charge"),
            })

        # --- Project paydays into the 60-day window ---
        next_payday_str = payday_info.get("next_payday")
        frequency = payday_info.get("frequency")
        if next_payday_str and frequency:
            payday = _parse_date(next_payday_str)
            if payday:
                freq_days = {"weekly": 7, "biweekly": 14, "monthly": 30}.get(
                    frequency, 30
                )
                while payday and payday <= today + timedelta(days=_FORECAST_DAYS):
                    if payday > today:
                        upcoming.append({
                            "date": payday.isoformat(),
                            "amount": round(avg_monthly_income / (30 / freq_days), 2),
                            "source": "payday",
                            "description": f"Payday ({frequency})",
                        })
                    payday += timedelta(days=freq_days)

        upcoming.sort(key=lambda e: e["date"])

        step = f"Found {len(upcoming)} upcoming financial events in next 60 days"
        logger.info(step)
        return {
            "input": {
                **state.get("input", {}),
                "upcoming_events": upcoming,
                "avg_monthly_income": avg_monthly_income,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: run_forecast  (THE CORE)
    # ------------------------------------------------------------------

    async def _run_forecast(self, state: AgentState) -> dict[str, Any]:
        """Roll a day-by-day balance projection for the next 60 days.

        Algorithm
        ---------
        1. Start with ``current_balance``.
        2. Build a sparse ``daily_cashflows`` map keyed by ISO date string.
           Each key maps to the sum of amounts from ``upcoming_events``
           landing on that day.
        3. Compute ``average_daily_net`` from 6 months of historical
           actuals:  ``total_net / 180``.  This captures the baseline
           drift on days with no known event.
        4. Walk forward day-by-day, accumulating:
           ``projected_balance += daily_cashflows.get(day, average_daily_net)``
        5. Record milestones (30-day, 60-day, lowest point, first
           negative day, first sub-$500 day).
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        current_balance: float = inp.get("current_balance", 0)
        upcoming_events: list[dict[str, Any]] = inp.get("upcoming_events", [])
        transactions: list[dict[str, Any]] = inp.get("transactions", [])

        # --- Step 1: average_daily_net from historical data ---
        total_net = sum(
            t.get("amount", 0) if str(t.get("type", "")).lower() == "income"
            else -abs(t.get("amount", 0))
            for t in transactions
        )
        average_daily_net = round(total_net / max(_HISTORICAL_DAYS, 1), 2)

        # --- Step 2: build daily_cashflows sparse map ---
        daily_cashflows: dict[str, float] = defaultdict(float)
        daily_events: dict[str, list[str]] = defaultdict(list)

        for ev in upcoming_events:
            d = ev.get("date", "")
            daily_cashflows[d] += ev.get("amount", 0)
            daily_events[d].append(ev.get("description", ""))

        # --- Step 3 & 4: roll forward ---
        today = date.today()
        projected_balance = current_balance
        daily_projections: list[dict[str, Any]] = []

        lowest_point = current_balance
        lowest_point_date = today.isoformat()
        balance_30d: float = current_balance
        balance_60d: float = current_balance
        days_until_negative: Optional[int] = None
        days_below_500: Optional[int] = None

        for day_offset in range(1, _FORECAST_DAYS + 1):
            d = today + timedelta(days=day_offset)
            d_str = d.isoformat()

            if d_str in daily_cashflows:
                projected_balance += daily_cashflows[d_str]
            else:
                projected_balance += average_daily_net

            projected_balance = round(projected_balance, 2)

            daily_projections.append({
                "date": d_str,
                "projected_balance": projected_balance,
                "events": daily_events.get(d_str, []),
            })

            if projected_balance < lowest_point:
                lowest_point = projected_balance
                lowest_point_date = d_str

            if day_offset == 30:
                balance_30d = projected_balance
            if day_offset == _FORECAST_DAYS:
                balance_60d = projected_balance

            if projected_balance < 0 and days_until_negative is None:
                days_until_negative = day_offset
            if projected_balance < _SAFETY_BUFFER and days_below_500 is None:
                days_below_500 = day_offset

        forecast: dict[str, Any] = {
            "daily_projections": daily_projections,
            "balance_30d": balance_30d,
            "balance_60d": balance_60d,
            "lowest_point": round(lowest_point, 2),
            "lowest_point_date": lowest_point_date,
            "days_until_negative": days_until_negative,
            "days_below_500": days_below_500,
            "average_daily_net": average_daily_net,
        }

        step = (
            f"Forecast complete — "
            f"30d: ${balance_30d:.2f}, "
            f"60d: ${balance_60d:.2f}, "
            f"lowest: ${lowest_point:.2f}"
        )
        logger.info(step)
        return {
            "input": {**inp, "forecast": forecast},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: detect_risks
    # ------------------------------------------------------------------

    async def _detect_risks(self, state: AgentState) -> dict[str, Any]:
        """Classify the forecast into a risk tier and build risk details.

        Risk tiers (highest to lowest priority):

        * **CRITICAL** — ``days_until_negative <= 30``
        * **HIGH** — lowest projected balance < $100 within 30 days
        * **MEDIUM** — balance drops below the $500 safety buffer within 30 days
        * **LOW** — 60-day balance < 70% of today's balance (slow bleed)
        * **NONE** — everything looks healthy
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        forecast: dict[str, Any] = inp.get("forecast", {})
        current_balance: float = inp.get("current_balance", 0)

        days_neg = forecast.get("days_until_negative")
        days_low = forecast.get("days_below_500")
        lowest = forecast.get("lowest_point", current_balance)
        lowest_date = forecast.get("lowest_point_date", "")
        bal_30 = forecast.get("balance_30d", current_balance)
        bal_60 = forecast.get("balance_60d", current_balance)

        # Check lowest point within the first 30 days specifically
        projections_30d = forecast.get("daily_projections", [])[:30]
        lowest_in_30d = min(
            (p["projected_balance"] for p in projections_30d),
            default=current_balance,
        )

        risks: list[dict[str, Any]] = []
        risk_level = "none"

        if days_neg is not None and days_neg <= 30:
            risk_level = "critical"
            risks.append({
                "level": "critical",
                "title": "Projected overdraft",
                "message": (
                    f"Your balance is projected to go negative in "
                    f"{days_neg} days (around {forecast.get('lowest_point_date', 'unknown')}). "
                    f"Lowest projected balance: ${lowest:.2f}."
                ),
                "days_until": days_neg,
            })

        if lowest_in_30d < _DANGER_THRESHOLD and risk_level != "critical":
            risk_level = max(risk_level, "high", key=_risk_order)
            risks.append({
                "level": "high",
                "title": "Dangerously low balance ahead",
                "message": (
                    f"Balance is projected to drop to ${lowest_in_30d:.2f} "
                    f"within 30 days — well below the ${_DANGER_THRESHOLD:.0f} "
                    f"danger line."
                ),
                "days_until": days_low or 30,
            })

        if days_low is not None and days_low <= 30 and risk_level not in ("critical", "high"):
            risk_level = max(risk_level, "medium", key=_risk_order)
            risks.append({
                "level": "medium",
                "title": "Balance below safety buffer",
                "message": (
                    f"Balance will drop below the ${_SAFETY_BUFFER:.0f} safety "
                    f"buffer in {days_low} days. Consider deferring "
                    f"non-essential spending."
                ),
                "days_until": days_low,
            })

        if (
            risk_level == "none"
            and current_balance > 0
            and bal_60 < current_balance * 0.70
        ):
            risk_level = "low"
            pct_decline = round((1 - bal_60 / current_balance) * 100, 1)
            risks.append({
                "level": "low",
                "title": "Gradual balance decline",
                "message": (
                    f"Your balance is trending down {pct_decline}% over "
                    f"60 days (${current_balance:.2f} → ${bal_60:.2f}). "
                    f"Not urgent, but worth monitoring."
                ),
                "days_until": 60,
            })

        step = f"Risk assessment: {risk_level.upper()} — {len(risks)} risk(s) identified"
        logger.info(step)
        return {
            "input": {
                **inp,
                "risk_level": risk_level,
                "risks": risks,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 5: create_alerts
    # ------------------------------------------------------------------

    async def _create_alerts(self, state: AgentState) -> dict[str, Any]:
        """Persist risk alerts in the knowledge graph for critical/high/medium."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        risks: list[dict[str, Any]] = inp.get("risks", [])
        user_id: str = state["user_id"]
        created = 0

        for risk in risks:
            if risk["level"] == "none":
                continue
            try:
                await self.call_tool(
                    "graph_mcp",
                    "create_alert_node",
                    {
                        "user_id": user_id,
                        "alert_type": "cashflow_forecast",
                        "title": risk["title"],
                        "message": risk["message"],
                        "severity": risk["level"],
                    },
                )
                created += 1
            except RuntimeError as exc:
                logger.warning("Failed to create forecast alert: %s", exc)

        step = f"Created {created} forecast alerts"
        logger.info(step)
        return {
            "input": {**inp, "alerts_created": created},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 6: generate_forecast_report
    # ------------------------------------------------------------------

    async def _generate_forecast_report(self, state: AgentState) -> dict[str, Any]:
        """Ask Claude for a plain-English forecast summary and set output."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        forecast: dict[str, Any] = inp.get("forecast", {})
        risks: list[dict[str, Any]] = inp.get("risks", [])
        risk_level: str = inp.get("risk_level", "none")
        current_balance: float = inp.get("current_balance", 0)
        bal_30 = forecast.get("balance_30d", 0)
        bal_60 = forecast.get("balance_60d", 0)
        lowest = forecast.get("lowest_point", 0)
        lowest_date = forecast.get("lowest_point_date", "")
        avg_daily = forecast.get("average_daily_net", 0)

        risk_descriptions = "\n".join(
            f"- [{r['level'].upper()}] {r['message']}" for r in risks
        ) or "No risks detected."

        prompt = (
            f"You are a financial advisor. The user's current checking balance "
            f"is ${current_balance:,.2f}.\n\n"
            f"Our 60-day forecast projects:\n"
            f"- 30-day balance: ${bal_30:,.2f}\n"
            f"- 60-day balance: ${bal_60:,.2f}\n"
            f"- Lowest point: ${lowest:,.2f} on {lowest_date}\n"
            f"- Average daily net: ${avg_daily:,.2f}/day\n"
            f"- Overall risk level: {risk_level.upper()}\n\n"
            f"Risks identified:\n{risk_descriptions}\n\n"
            f"Write a clear, friendly 3-5 sentence summary of the user's "
            f"financial outlook. If there are risks, be specific about the "
            f"timeline and suggest one concrete action. If everything looks "
            f"healthy, say so confidently."
        )

        try:
            response = self._llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            summary: str = response.content[0].text
        except Exception as exc:
            logger.warning("Forecast summary generation failed: %s", exc)
            if risk_level in ("critical", "high"):
                summary = (
                    f"Warning: your balance is projected to drop to "
                    f"${lowest:,.2f} by {lowest_date}. Current balance: "
                    f"${current_balance:,.2f}. Review upcoming charges and "
                    f"consider deferring non-essential spending."
                )
            else:
                summary = (
                    f"Your balance is projected at ${bal_30:,.2f} in 30 days "
                    f"and ${bal_60:,.2f} in 60 days. "
                    f"Current balance: ${current_balance:,.2f}."
                )

        output: dict[str, Any] = {
            "current_balance": current_balance,
            "balance_30d": bal_30,
            "balance_60d": bal_60,
            "lowest_point": lowest,
            "lowest_point_date": lowest_date,
            "days_until_negative": forecast.get("days_until_negative"),
            "days_below_500": forecast.get("days_below_500"),
            "average_daily_net": avg_daily,
            "risk_level": risk_level,
            "risks": risks,
            "daily_projections": forecast.get("daily_projections", []),
            "summary": summary,
        }

        step = "Generated forecast report"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }


# ------------------------------------------------------------------
# Module-private helpers
# ------------------------------------------------------------------

_RISK_LEVELS = ("none", "low", "medium", "high", "critical")


def _risk_order(level: str) -> int:
    """Return a numeric rank so ``max(..., key=_risk_order)`` picks the worst."""
    try:
        return _RISK_LEVELS.index(level)
    except ValueError:
        return -1
