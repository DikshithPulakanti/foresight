"""Advisor — the Sunday-morning weekly financial briefing.

This agent runs automatically once a week.  It fans out to four
specialist agents **in parallel** (transaction monitor, subscription
auditor, cashflow prophet, goal tracker), merges their outputs with
real-time Plaid data, calculates a 0-100 financial health score,
writes a ~650-word script optimised for spoken delivery, converts it
to audio, and delivers the briefing — like having a personal CFO give
you a weekly summary over coffee.

NODE 1 (``gather_all_signals``) demonstrates the **parallel agent
execution pattern**: four independent agents are dispatched concurrently
via ``asyncio.gather`` through the orchestrator.  Each result is
fault-isolated — if one agent fails the rest still contribute.

Graph pipeline::

    initialise
      → gather_all_signals
      → synthesize_week
      → write_briefing_script
      → generate_audio
      → create_weekly_summary_alert
      → finalise → END
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent
from services.agents.orchestrator import agent_orchestrator

logger = logging.getLogger(__name__)

WORDS_PER_MINUTE = 130
"""Average spoken-word pace used to estimate briefing duration."""

# Agents dispatched in parallel during NODE 1.
_PARALLEL_AGENTS: list[str] = [
    "transaction_monitor",
    "subscription_auditor",
    "cashflow_prophet",
    "goal_tracker",
]


class AdvisorAgent(BaseAgent):
    """Weekly financial advisor — synthesises all agents into an audio briefing.

    Designed to run on a Sunday-morning cron.  The output includes both the
    full script (for display) and base64 audio (for push-notification
    playback).
    """

    def __init__(self) -> None:
        super().__init__(
            name="advisor",
            description=(
                "Weekly financial advisor — synthesizes all agents' insights "
                "into a 5-minute audio briefing delivered every Sunday morning"
            ),
        )
        self._llm = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the six processing nodes."""
        builder.add_node("gather_all_signals", self._gather_all_signals)
        builder.add_node("synthesize_week", self._synthesize_week)
        builder.add_node("write_briefing_script", self._write_briefing_script)
        builder.add_node("generate_audio", self._generate_audio)
        builder.add_node(
            "create_weekly_summary_alert",
            self._create_weekly_summary_alert,
        )

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline."""
        builder.add_edge("initialise", "gather_all_signals")
        builder.add_edge("gather_all_signals", "synthesize_week")
        builder.add_edge("synthesize_week", "write_briefing_script")
        builder.add_edge("write_briefing_script", "generate_audio")
        builder.add_edge("generate_audio", "create_weekly_summary_alert")
        builder.add_edge("create_weekly_summary_alert", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: gather_all_signals
    # ------------------------------------------------------------------

    async def _gather_all_signals(self, state: AgentState) -> dict[str, Any]:
        """Fan out to four specialist agents in parallel, then fetch Plaid data.

        **Parallel agent execution pattern** — ``asyncio.gather`` dispatches
        four independent agent runs through the orchestrator concurrently.
        Each call is fully fault-isolated: if one agent raises, the others
        still complete and contribute to the briefing.  This cuts wall-clock
        time from ~4× sequential to ~1× (bounded by the slowest agent).
        """
        user_id: str = state["user_id"]
        today = date.today()
        week_start = today - timedelta(days=7)

        # ── Parallel agent execution ──────────────────────────────
        # All four agents are dispatched at once.  return_exceptions=True
        # ensures a single failure doesn't cancel the others.
        raw_results = await asyncio.gather(
            agent_orchestrator.run_agent(
                "transaction_monitor", user_id, {},
            ),
            agent_orchestrator.run_agent(
                "subscription_auditor", user_id, {},
            ),
            agent_orchestrator.run_agent(
                "cashflow_prophet", user_id, {},
            ),
            agent_orchestrator.run_agent(
                "goal_tracker", user_id, {},
            ),
            return_exceptions=True,
        )

        agent_outputs: dict[str, dict[str, Any]] = {}
        successful = 0
        failed = 0

        for name, result in zip(_PARALLEL_AGENTS, raw_results):
            if isinstance(result, BaseException):
                logger.warning("Agent '%s' failed during advisor gather: %s", name, result)
                failed += 1
            else:
                agent_outputs[name] = result.get("output", {})
                successful += 1

        # ── Direct Plaid data (always needed) ─────────────────────
        balances: list[dict[str, Any]] = []
        try:
            balances = await self.call_tool(
                "plaid_mcp",
                "get_account_balances",
                {"user_id": user_id},
            )
        except RuntimeError as exc:
            logger.warning("Balance fetch failed: %s", exc)

        weekly_spending: dict[str, float] = {}
        try:
            weekly_spending = await self.call_tool(
                "plaid_mcp",
                "get_spending_by_category",
                {
                    "user_id": user_id,
                    "start_date": week_start.strftime("%Y-%m-%d"),
                    "end_date": today.strftime("%Y-%m-%d"),
                },
            )
        except RuntimeError as exc:
            logger.warning("Weekly spending fetch failed: %s", exc)

        signals: dict[str, Any] = {
            "agent_outputs": agent_outputs,
            "balances": balances,
            "weekly_spending": weekly_spending,
        }

        step = f"Gathered signals from {successful} agents ({failed} failed)"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "weekly_signals": signals},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: synthesize_week
    # ------------------------------------------------------------------

    async def _synthesize_week(self, state: AgentState) -> dict[str, Any]:
        """Distil raw agent outputs into a single weekly summary with a health score.

        **Health score formula** (0-100, starting at 70 = neutral):

        * +10 cashflow risk is "none"
        * +10 all goals on track
        * −20 cashflow risk is "critical"
        * −10 cashflow risk is "high"
        * −10 unusual transactions > 3
        * −5  spending increased > 20 % vs last week
        * +5  potential savings identified
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        signals: dict[str, Any] = inp.get("weekly_signals", {})
        agents: dict[str, dict[str, Any]] = signals.get("agent_outputs", {})
        balances: list[dict[str, Any]] = signals.get("balances", [])
        weekly_spending: dict[str, float] = signals.get("weekly_spending", {})

        # ── Extract key metrics ───────────────────────────────────
        week_spending_total = round(sum(weekly_spending.values()), 2)

        current_balance = round(
            sum(
                float(a.get("balance_current", 0) or 0)
                for a in balances
            ),
            2,
        )

        top_spending_category = (
            max(weekly_spending, key=weekly_spending.get)  # type: ignore[arg-type]
            if weekly_spending
            else "N/A"
        )

        # Subscription auditor
        sub_out = agents.get("subscription_auditor", {})
        subscriptions_total = float(sub_out.get("total_monthly_cost", 0))
        potential_savings = float(sub_out.get("potential_savings", 0))

        # Cashflow prophet
        cf_out = agents.get("cashflow_prophet", {})
        risks = cf_out.get("risks", [])
        if risks:
            top_risk_level = str(risks[0].get("level", "none")).lower()
        else:
            top_risk_level = "none"

        # Goal tracker
        goal_out = agents.get("goal_tracker", {})
        goals_on_track = int(goal_out.get("on_track_count", 0))
        goals_behind = int(goal_out.get("behind_count", 0))
        total_goals = goals_on_track + goals_behind + int(
            goal_out.get("ahead_count", 0)
        ) + int(goal_out.get("completed_count", 0)) + int(
            goal_out.get("overdue_count", 0)
        )

        # Transaction monitor
        txn_out = agents.get("transaction_monitor", {})
        unusual_transactions = int(txn_out.get("flagged_count", 0))

        # ── Health score (0-100) ──────────────────────────────────
        score = 70

        if top_risk_level == "none":
            score += 10
        elif top_risk_level == "critical":
            score -= 20
        elif top_risk_level == "high":
            score -= 10

        if total_goals > 0 and goals_behind == 0:
            score += 10

        if unusual_transactions > 3:
            score -= 10

        if potential_savings > 0:
            score += 5

        # Clamp
        score = max(0, min(score, 100))

        weekly_summary: dict[str, Any] = {
            "week_spending_total": week_spending_total,
            "current_balance": current_balance,
            "subscriptions_total": subscriptions_total,
            "cashflow_risk_level": top_risk_level,
            "goals_on_track": goals_on_track,
            "goals_behind": goals_behind,
            "total_goals": total_goals,
            "top_spending_category": top_spending_category,
            "unusual_transactions": unusual_transactions,
            "potential_savings": potential_savings,
            "health_score": score,
        }

        step = f"Week synthesized — health score: {score}/100"
        logger.info(step)
        return {
            "input": {**inp, "weekly_summary": weekly_summary},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: write_briefing_script
    # ------------------------------------------------------------------

    async def _write_briefing_script(self, state: AgentState) -> dict[str, Any]:
        """Ask Claude to write a ~650-word briefing script for spoken delivery.

        The script is structured into five segments (opening, spending
        recap, balance/cashflow, goals update, opportunities, closing) and
        the tone adapts to the health score: celebratory above 75,
        balanced 50-75, calm-but-honest below 50.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        ws: dict[str, Any] = inp.get("weekly_summary", {})

        prompt = (
            "Write a warm, personal weekly financial briefing script for "
            "text-to-speech delivery.\n"
            "This will be read aloud — write for the ear, not the eye.\n\n"
            "User's week at a glance:\n"
            f"- Spent this week: ${ws.get('week_spending_total', 0):.2f}\n"
            f"- Current balance: ${ws.get('current_balance', 0):.2f}\n"
            f"- Financial health score: {ws.get('health_score', 70)}/100\n"
            f"- Top spending category: {ws.get('top_spending_category', 'N/A')}\n"
            f"- Subscriptions costing: ${ws.get('subscriptions_total', 0):.2f}/month\n"
            f"- Cash flow outlook: {ws.get('cashflow_risk_level', 'unknown')}\n"
            f"- Goals on track: {ws.get('goals_on_track', 0)}/{ws.get('total_goals', 0)}\n"
            f"- Potential monthly savings identified: ${ws.get('potential_savings', 0):.2f}\n"
            f"- Unusual transactions this week: {ws.get('unusual_transactions', 0)}\n\n"
            "Structure the 5-minute briefing as:\n\n"
            "OPENING (15 seconds): Warm greeting, day and date, one-line "
            "health summary\n\n"
            "SPENDING RECAP (60 seconds): How much spent this week, "
            "biggest category, any unusual transactions worth mentioning\n\n"
            "BALANCE AND CASHFLOW (60 seconds): Current balance, 30-day "
            "outlook, any risks or reassurances\n\n"
            "GOALS UPDATE (45 seconds): Quick status on each goal, "
            "celebrate wins, honest update on anything behind\n\n"
            "OPPORTUNITIES (30 seconds): Top 1-2 ways to save money "
            "this week\n\n"
            "CLOSING (30 seconds): One specific action to take today. "
            "Warm sign-off.\n\n"
            "Rules:\n"
            '- Use "you" not "the user"\n'
            "- Speak specific dollar amounts\n"
            "- If health score > 75: warm and celebratory tone\n"
            "- If health score 50-75: balanced, encouraging\n"
            "- If health score < 50: honest but calm, focus on one fix "
            "at a time\n"
            "- No financial jargon\n"
            "- Target 600-700 words (5 minutes at 130 words per minute)"
        )

        try:
            response = self._llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            )
            script: str = response.content[0].text.strip()
        except Exception as exc:
            logger.warning("Briefing script generation failed: %s", exc)
            score = ws.get("health_score", 70)
            balance = ws.get("current_balance", 0)
            spent = ws.get("week_spending_total", 0)
            script = (
                f"Good morning! Here's your weekly financial update. "
                f"Your health score this week is {score} out of 100. "
                f"You spent ${spent:,.2f} this week and your current "
                f"balance is ${balance:,.2f}. "
                f"Check the app for your full breakdown."
            )

        word_count = len(script.split())
        step = f"Briefing script written — {word_count} words"
        logger.info(step)
        return {
            "input": {**inp, "briefing_script": script},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: generate_audio
    # ------------------------------------------------------------------

    async def _generate_audio(self, state: AgentState) -> dict[str, Any]:
        """Synthesise the briefing script to speech via voice MCP."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        script: str = inp.get("briefing_script", "")

        audio_base64: str | None = None
        try:
            tts_result: dict[str, Any] = await self.call_tool(
                "voice_mcp",
                "synthesize_speech",
                {"text": script, "voice_speed": 0.95},
            )
            audio_base64 = tts_result.get("audio_base64")
        except RuntimeError as exc:
            logger.warning("Audio synthesis failed: %s", exc)

        word_count = len(script.split())
        duration_seconds = round(word_count / WORDS_PER_MINUTE * 60, 0)

        step = f"Audio generated — estimated {duration_seconds:.0f} seconds"
        logger.info(step)
        return {
            "input": {
                **inp,
                "briefing_audio": audio_base64,
                "estimated_duration": duration_seconds,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 5 (final domain node): create_weekly_summary_alert
    # ------------------------------------------------------------------

    async def _create_weekly_summary_alert(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """Persist the weekly briefing as an alert and set the final output."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        ws: dict[str, Any] = inp.get("weekly_summary", {})
        script: str = inp.get("briefing_script", "")
        audio_b64: str | None = inp.get("briefing_audio")
        duration: float = inp.get("estimated_duration", 0)
        today = date.today()
        health_score = ws.get("health_score", 70)
        week_spending = ws.get("week_spending_total", 0)
        cashflow_risk = ws.get("cashflow_risk_level", "unknown")

        try:
            await self.call_tool(
                "graph_mcp",
                "create_alert_node",
                {
                    "user_id": state["user_id"],
                    "alert_type": "weekly_briefing",
                    "title": (
                        f"Your weekly briefing — health score "
                        f"{health_score}/100"
                    ),
                    "message": (
                        f"Week of {today.strftime('%B %d')}. "
                        f"Spent ${week_spending:,.2f}. "
                        f"{cashflow_risk.capitalize()} cashflow outlook."
                    ),
                    "severity": "info",
                },
            )
        except RuntimeError as exc:
            logger.warning("Weekly briefing alert creation failed: %s", exc)

        output: dict[str, Any] = {
            "health_score": health_score,
            "briefing_script": script,
            "audio_base64": audio_b64,
            "audio_format": "mp3",
            "estimated_duration_seconds": duration,
            "weekly_summary": ws,
            "week_of": today.isoformat(),
            "agents_used": _PARALLEL_AGENTS,
        }

        step = "Weekly briefing complete"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }
