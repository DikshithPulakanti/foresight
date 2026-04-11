"""Goal Tracker — monitors savings goals and celebrates milestones.

This agent checks the user's savings goals against reality: current balance,
recent spending patterns, and time remaining.  It classifies each goal as
on-track / behind / ahead / completed / overdue and generates personalised
recommendations.

The tone is deliberately "supportive friend, not financial advisor" — the
Claude prompt in NODE 5 enforces this to produce responses that feel
encouraging rather than preachy.

Goals are stored in Neo4j as::

    (User)-[:HAS_GOAL]->(Goal {id, name, target_amount, current_amount, deadline})

Graph pipeline::

    initialise
      → fetch_goals_and_balance
      → calculate_progress
      → generate_recommendations
      → create_alerts
      → generate_report
      → finalise → END
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent

logger = logging.getLogger(__name__)

DISCRETIONARY_CATEGORIES: list[str] = [
    "restaurant",
    "entertainment",
    "shopping",
    "subscriptions",
    "food and drink",
    "recreation",
    "travel",
]

_AHEAD_MULTIPLIER = 1.2


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


class GoalTrackerAgent(BaseAgent):
    """Tracks savings goals, calculates progress, and recommends adjustments.

    Classifies each goal into one of five statuses:

    * **completed** — target amount reached (triggers celebration).
    * **ahead** — saving faster than required pace by ≥ 20 %.
    * **on_track** — projected to finish by the deadline.
    * **behind** — projected to miss the deadline at current pace.
    * **overdue** — deadline has passed and goal is not complete.
    """

    def __init__(self) -> None:
        super().__init__(
            name="goal_tracker",
            description=(
                "Tracks savings goals, calculates if the user is on track, "
                "and recommends spending adjustments to hit targets"
            ),
        )
        self._llm = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the five processing nodes."""
        builder.add_node("fetch_goals_and_balance", self._fetch_goals_and_balance)
        builder.add_node("calculate_progress", self._calculate_progress)
        builder.add_node("generate_recommendations", self._generate_recommendations)
        builder.add_node("create_alerts", self._create_alerts)
        builder.add_node("generate_report", self._generate_report)

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline."""
        builder.add_edge("initialise", "fetch_goals_and_balance")
        builder.add_edge("fetch_goals_and_balance", "calculate_progress")
        builder.add_edge("calculate_progress", "generate_recommendations")
        builder.add_edge("generate_recommendations", "create_alerts")
        builder.add_edge("create_alerts", "generate_report")
        builder.add_edge("generate_report", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: fetch_goals_and_balance
    # ------------------------------------------------------------------

    async def _fetch_goals_and_balance(self, state: AgentState) -> dict[str, Any]:
        """Fetch the user's savings goals, current balance, and monthly spend.

        Goals are read from Neo4j via ``graph_mcp.get_user_goals``.  If
        that tool is not yet registered, the call will raise a RuntimeError
        and the agent will fail gracefully with a clear message.
        """
        user_id: str = state["user_id"]
        today = date.today()
        month_start = today - timedelta(days=30)

        # --- Goals from the knowledge graph ---
        try:
            goals: list[dict[str, Any]] = await self.call_tool(
                "graph_mcp",
                "get_user_goals",
                {"user_id": user_id},
            )
        except RuntimeError as exc:
            logger.error("fetch_goals_and_balance: goals query failed: %s", exc)
            return self.set_error(
                state,
                "No savings goals found. Create a goal first.",
            )

        if not goals:
            return self.set_error(
                state,
                "No savings goals found. Create a goal first.",
            )

        # --- Current account balances ---
        try:
            accounts: list[dict[str, Any]] = await self.call_tool(
                "plaid_mcp",
                "get_account_balances",
                {"user_id": user_id},
            )
        except RuntimeError as exc:
            logger.error("fetch_goals_and_balance: balances failed: %s", exc)
            return self.set_error(state, str(exc))

        current_balance = round(
            sum(
                a.get("balance_current", 0) or 0
                for a in accounts
                if _is_checking(a)
            ),
            2,
        )

        # --- Last 30 days spending by category ---
        monthly_spending: dict[str, float] = {}
        try:
            monthly_spending = await self.call_tool(
                "plaid_mcp",
                "get_spending_by_category",
                {
                    "user_id": user_id,
                    "start_date": month_start.strftime("%Y-%m-%d"),
                    "end_date": today.strftime("%Y-%m-%d"),
                },
            )
        except RuntimeError as exc:
            logger.warning("Monthly spending unavailable: %s", exc)

        step = f"Found {len(goals)} active goals, current balance ${current_balance:.2f}"
        logger.info(step)
        return {
            "input": {
                **state.get("input", {}),
                "goals": goals,
                "current_balance": current_balance,
                "monthly_spending": monthly_spending,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: calculate_progress
    # ------------------------------------------------------------------

    async def _calculate_progress(self, state: AgentState) -> dict[str, Any]:
        """Enrich each goal with progress metrics and a status classification.

        Status classification:

        * ``completed`` — ``percent_complete >= 100``
        * ``overdue`` — deadline passed and not complete
        * ``ahead`` — monthly savings > required pace × 1.2
        * ``on_track`` — projected completion ≤ deadline
        * ``behind`` — projected completion > deadline
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        goals: list[dict[str, Any]] = inp.get("goals", [])
        monthly_spending: dict[str, float] = inp.get("monthly_spending", {})
        today = date.today()

        total_monthly_spend = sum(monthly_spending.values())
        # Rough monthly savings estimate: assume take-home is 2× spending
        estimated_monthly_savings = max(total_monthly_spend * 0.10, 0)

        progress: list[dict[str, Any]] = []
        counts: dict[str, int] = {
            "on_track": 0, "behind": 0, "ahead": 0,
            "completed": 0, "overdue": 0,
        }

        for goal in goals:
            target = float(goal.get("target_amount") or 0)
            current = float(goal.get("current_amount") or 0)
            deadline = _parse_date(goal.get("deadline"))
            goal_name = goal.get("name", "Unnamed goal")
            goal_id = goal.get("id", "")

            percent_complete = round((current / target) * 100, 1) if target > 0 else 0
            amount_remaining = max(target - current, 0)

            days_remaining = (deadline - today).days if deadline else 365
            required_daily = (
                amount_remaining / max(days_remaining, 1)
                if amount_remaining > 0
                else 0
            )
            required_monthly = round(required_daily * 30, 2)

            if amount_remaining > 0 and estimated_monthly_savings > 0:
                months_to_complete = amount_remaining / estimated_monthly_savings
                projected_completion = today + timedelta(
                    days=int(months_to_complete * 30)
                )
            else:
                projected_completion = today

            # --- Status classification ---
            if percent_complete >= 100:
                goal_status = "completed"
            elif deadline and deadline < today:
                goal_status = "overdue"
            elif (
                estimated_monthly_savings
                >= required_monthly * _AHEAD_MULTIPLIER
                and required_monthly > 0
            ):
                goal_status = "ahead"
            elif deadline and projected_completion <= deadline:
                goal_status = "on_track"
            else:
                goal_status = "behind"

            counts[goal_status] = counts.get(goal_status, 0) + 1

            progress.append({
                "goal_id": goal_id,
                "goal_name": goal_name,
                "target_amount": target,
                "current_amount": current,
                "percent_complete": percent_complete,
                "amount_remaining": round(amount_remaining, 2),
                "deadline": deadline.isoformat() if deadline else None,
                "days_remaining": days_remaining,
                "required_monthly_savings": required_monthly,
                "estimated_monthly_savings": round(estimated_monthly_savings, 2),
                "projected_completion_date": projected_completion.isoformat(),
                "status": goal_status,
            })

        step = (
            f"Calculated progress — "
            f"{counts['on_track']} on track, "
            f"{counts['behind']} behind, "
            f"{counts['completed']} completed"
        )
        logger.info(step)
        return {
            "input": {**inp, "goal_progress": progress, "goal_counts": counts},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: generate_recommendations
    # ------------------------------------------------------------------

    async def _generate_recommendations(self, state: AgentState) -> dict[str, Any]:
        """Produce actionable advice for each goal based on its status.

        * **behind** — identify the top discretionary spending category and
          recommend reducing it by the shortfall amount.
        * **completed** — celebrate and update the goal in the graph.
        * **on_track** / **ahead** — positive reinforcement.
        * **overdue** — suggest extending the deadline.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        progress: list[dict[str, Any]] = inp.get("goal_progress", [])
        monthly_spending: dict[str, float] = inp.get("monthly_spending", {})
        user_id: str = state["user_id"]

        # Rank discretionary categories by total spend
        disc_sorted = sorted(
            (
                (cat, amt)
                for cat, amt in monthly_spending.items()
                if any(d in cat.lower() for d in DISCRETIONARY_CATEGORIES)
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        top_discretionary = disc_sorted[0] if disc_sorted else None

        recommendations: list[dict[str, Any]] = []

        for goal in progress:
            goal_name = goal["goal_name"]
            status = goal["status"]

            if status == "behind":
                gap = round(
                    goal["required_monthly_savings"]
                    - goal["estimated_monthly_savings"],
                    2,
                )
                if top_discretionary and gap > 0:
                    cat_name, cat_amount = top_discretionary
                    recommendation = (
                        f"Reduce {cat_name} spending by ${gap:.2f}/month "
                        f"to hit your {goal_name} goal. You currently spend "
                        f"${cat_amount:.2f}/month in that category."
                    )
                else:
                    recommendation = (
                        f"You need to save an extra ${gap:.2f}/month to "
                        f"stay on track for {goal_name}."
                    )
                recommendations.append({
                    "goal_name": goal_name,
                    "status": status,
                    "recommendation": recommendation,
                    "action_required": True,
                })

            elif status == "completed":
                # Update the goal in the knowledge graph
                try:
                    await self.call_tool(
                        "graph_mcp",
                        "update_goal_progress",
                        {
                            "user_id": user_id,
                            "goal_id": goal["goal_id"],
                            "current_amount": goal["target_amount"],
                        },
                    )
                except RuntimeError as exc:
                    logger.warning("Failed to update completed goal: %s", exc)

                recommendations.append({
                    "goal_name": goal_name,
                    "status": status,
                    "recommendation": (
                        f"You did it! {goal_name} goal reached — "
                        f"${goal['target_amount']:,.2f} saved. "
                        f"Time to set a new goal!"
                    ),
                    "action_required": False,
                })

            elif status == "overdue":
                recommendations.append({
                    "goal_name": goal_name,
                    "status": status,
                    "recommendation": (
                        f"{goal_name} deadline has passed with "
                        f"${goal['amount_remaining']:,.2f} remaining. "
                        f"Consider extending the deadline or adjusting the target."
                    ),
                    "action_required": True,
                })

            elif status == "ahead":
                recommendations.append({
                    "goal_name": goal_name,
                    "status": status,
                    "recommendation": (
                        f"Great pace on {goal_name}! You're saving faster "
                        f"than needed. Keep it up — you'll finish early."
                    ),
                    "action_required": False,
                })

            else:  # on_track
                recommendations.append({
                    "goal_name": goal_name,
                    "status": status,
                    "recommendation": (
                        f"{goal_name} is on track — "
                        f"{goal['percent_complete']:.0f}% complete with "
                        f"{goal['days_remaining']} days to go."
                    ),
                    "action_required": False,
                })

        step = f"Generated {len(recommendations)} recommendations"
        logger.info(step)
        return {
            "input": {**inp, "recommendations": recommendations},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: create_alerts
    # ------------------------------------------------------------------

    async def _create_alerts(self, state: AgentState) -> dict[str, Any]:
        """Persist goal-related alerts in the knowledge graph."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        progress: list[dict[str, Any]] = inp.get("goal_progress", [])
        user_id: str = state["user_id"]
        created = 0

        for goal in progress:
            status = goal["status"]
            goal_name = goal["goal_name"]

            if status == "behind" and goal["days_remaining"] < 30:
                try:
                    await self.call_tool(
                        "graph_mcp",
                        "create_alert_node",
                        {
                            "user_id": user_id,
                            "alert_type": "goal_at_risk",
                            "title": f"Goal at risk: {goal_name}",
                            "message": (
                                f"${goal['amount_remaining']:,.2f} needed "
                                f"in {goal['days_remaining']} days"
                            ),
                            "severity": "high",
                            "amount": goal["amount_remaining"],
                        },
                    )
                    created += 1
                except RuntimeError as exc:
                    logger.warning("Alert creation failed: %s", exc)

            elif status == "overdue":
                try:
                    await self.call_tool(
                        "graph_mcp",
                        "create_alert_node",
                        {
                            "user_id": user_id,
                            "alert_type": "goal_overdue",
                            "title": f"{goal_name} deadline passed",
                            "message": (
                                f"Consider extending deadline — "
                                f"${goal['amount_remaining']:,.2f} remaining"
                            ),
                            "severity": "medium",
                        },
                    )
                    created += 1
                except RuntimeError as exc:
                    logger.warning("Alert creation failed: %s", exc)

            elif status == "completed":
                try:
                    await self.call_tool(
                        "graph_mcp",
                        "create_alert_node",
                        {
                            "user_id": user_id,
                            "alert_type": "goal_achieved",
                            "title": f"Congratulations! {goal_name} goal reached!",
                            "message": (
                                f"${goal['target_amount']:,.2f} saved — "
                                f"amazing work!"
                            ),
                            "severity": "low",
                            "amount": goal["target_amount"],
                        },
                    )
                    created += 1
                except RuntimeError as exc:
                    logger.warning("Alert creation failed: %s", exc)

        step = f"Created {created} goal alerts"
        logger.info(step)
        return {
            "input": {**inp, "alerts_created": created},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 5: generate_report
    # ------------------------------------------------------------------

    async def _generate_report(self, state: AgentState) -> dict[str, Any]:
        """Ask Claude for an encouraging summary and set the final output.

        The tone instruction — "supportive friend, not financial advisor" —
        is critical.  It produces responses that celebrate wins, stay honest
        about gaps, and avoid the preachy moralising that turns users off.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        progress: list[dict[str, Any]] = inp.get("goal_progress", [])
        recommendations: list[dict[str, Any]] = inp.get("recommendations", [])
        counts: dict[str, int] = inp.get("goal_counts", {})
        alerts_created: int = inp.get("alerts_created", 0)

        total_saved = round(sum(g["current_amount"] for g in progress), 2)
        total_target = round(sum(g["target_amount"] for g in progress), 2)
        overall_pct = round(
            (total_saved / total_target) * 100, 1
        ) if total_target > 0 else 0

        prompt = (
            f"User's savings goal summary:\n"
            f"Goals: {progress}\n"
            f"Recommendations: {recommendations}\n\n"
            f"Write an encouraging 3-4 sentence summary.\n"
            f"If goals are on track: celebrate and reinforce good behavior.\n"
            f"If behind: be honest but motivating — give ONE specific "
            f"actionable tip.\n"
            f"If a goal was completed: make it feel like a real achievement.\n"
            f"Never be preachy. Be like a supportive friend, not a "
            f"financial advisor."
        )

        try:
            response = self._llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            summary: str = response.content[0].text
        except Exception as exc:
            logger.warning("Report generation failed: %s", exc)
            if counts.get("completed", 0) > 0:
                summary = (
                    f"Amazing — you've hit {counts['completed']} goal(s)! "
                    f"Overall you've saved ${total_saved:,.2f} of "
                    f"${total_target:,.2f} ({overall_pct:.0f}%)."
                )
            elif counts.get("behind", 0) > 0:
                summary = (
                    f"You're {overall_pct:.0f}% of the way to your goals "
                    f"(${total_saved:,.2f} / ${total_target:,.2f}). "
                    f"Some goals need a boost — check the recommendations."
                )
            else:
                summary = (
                    f"You're on track! ${total_saved:,.2f} saved of "
                    f"${total_target:,.2f} ({overall_pct:.0f}%)."
                )

        output: dict[str, Any] = {
            "goals": progress,
            "recommendations": recommendations,
            "on_track_count": counts.get("on_track", 0),
            "behind_count": counts.get("behind", 0),
            "ahead_count": counts.get("ahead", 0),
            "completed_count": counts.get("completed", 0),
            "overdue_count": counts.get("overdue", 0),
            "total_saved": total_saved,
            "total_target": total_target,
            "overall_progress_pct": overall_pct,
            "alerts_created": alerts_created,
            "summary": summary,
        }

        step = "Generated report"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }
