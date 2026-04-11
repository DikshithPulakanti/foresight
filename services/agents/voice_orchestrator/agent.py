"""Voice Orchestrator — the conversational interface to Foresight.

When the user speaks (or types), this agent:

1. **Transcribes** the audio via ``voice_mcp``.
2. **Classifies intent** with Claude (spending, balance, forecast, …).
3. **Routes** to the correct MCP tool *or* delegates to a full sub-agent
   through the orchestrator singleton.
4. **Formulates** a natural spoken answer via Claude.
5. **Synthesises** speech and returns the audio.

NODE 3 (``route_and_execute``) is the architecturally interesting piece:
it's an **agent that calls other agents**.  Simple data look-ups go straight
to an MCP tool, while complex multi-step questions are delegated to
specialised agents (CashflowProphet, GoalTracker, AlertSentinel, …).  This
keeps the voice layer thin — it handles *conversation*, not *computation*.

Graph pipeline::

    initialise
      → transcribe_input
      → understand_intent
      → route_and_execute
      → formulate_response
      → synthesize_and_complete
      → finalise → END
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent
from services.agents.orchestrator import agent_orchestrator

logger = logging.getLogger(__name__)

MIN_TRANSCRIPTION_CONFIDENCE = 0.4

# Maps natural-language time references to (start_date, end_date) offsets.
_TIME_PERIOD_DAYS: dict[str, int] = {
    "today": 1,
    "yesterday": 2,
    "this week": 7,
    "last week": 14,
    "this month": 30,
    "last month": 60,
    "this quarter": 90,
    "this year": 365,
}

# Intents that are served by delegating to a full sub-agent rather than a
# single MCP tool call.  The value is the orchestrator agent name.
_AGENT_INTENTS: dict[str, str] = {
    "forecast_query": "cashflow_prophet",
    "goal_query": "goal_tracker",
    "alert_query": "alert_sentinel",
    "bill_query": "bill_negotiator",
}


def _resolve_time_period(raw: str | None) -> tuple[str, str]:
    """Turn a fuzzy time reference into an ISO date range.

    Falls back to the last 30 days when the period is unrecognised.
    """
    today = date.today()
    if raw:
        days = _TIME_PERIOD_DAYS.get(raw.lower().strip(), 30)
    else:
        days = 30
    start = today - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


class VoiceOrchestratorAgent(BaseAgent):
    """The voice interface to Foresight.

    Accepts either raw audio (``audio_base64``) or a text query
    (``text_query``), understands the user's intent, fetches the answer
    from the appropriate source, and returns both a spoken-language
    response and synthesised audio.
    """

    def __init__(self) -> None:
        super().__init__(
            name="voice_orchestrator",
            description=(
                "The voice interface to Foresight — listens, understands "
                "intent, routes to the right agent, and speaks the answer back"
            ),
        )
        self._llm = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the five processing nodes."""
        builder.add_node("transcribe_input", self._transcribe_input)
        builder.add_node("understand_intent", self._understand_intent)
        builder.add_node("route_and_execute", self._route_and_execute)
        builder.add_node("formulate_response", self._formulate_response)
        builder.add_node("synthesize_and_complete", self._synthesize_and_complete)

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline."""
        builder.add_edge("initialise", "transcribe_input")
        builder.add_edge("transcribe_input", "understand_intent")
        builder.add_edge("understand_intent", "route_and_execute")
        builder.add_edge("route_and_execute", "formulate_response")
        builder.add_edge("formulate_response", "synthesize_and_complete")
        builder.add_edge("synthesize_and_complete", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: transcribe_input
    # ------------------------------------------------------------------

    async def _transcribe_input(self, state: AgentState) -> dict[str, Any]:
        """Convert audio to text, or pass through a direct text query.

        Two input paths:

        * ``audio_base64`` present → call ``voice_mcp.transcribe_audio``.
        * ``text_query`` present → use the text verbatim (confidence = 1.0).
        """
        inp: dict[str, Any] = state.get("input", {})
        audio_b64: str | None = inp.get("audio_base64")
        text_query: str | None = inp.get("text_query")

        if audio_b64:
            try:
                result: dict[str, Any] = await self.call_tool(
                    "voice_mcp",
                    "transcribe_audio",
                    {
                        "audio_base64": audio_b64,
                        "audio_format": inp.get("audio_format", "wav"),
                    },
                )
                transcript = str(result.get("transcript", ""))
                confidence = float(result.get("confidence", 0.0))
            except RuntimeError as exc:
                logger.error("Transcription failed: %s", exc)
                return self.set_error(state, f"Transcription failed: {exc}")
        elif text_query:
            transcript = text_query.strip()
            confidence = 1.0
        else:
            return self.set_error(
                state,
                "No input provided — send audio_base64 or text_query",
            )

        if confidence < MIN_TRANSCRIPTION_CONFIDENCE:
            return self.set_error(
                state,
                "Could not understand audio — please speak clearly",
            )

        step = (
            f"Transcribed: {len(transcript)} chars, "
            f"confidence={confidence:.2f}"
        )
        logger.info(step)
        return {
            "input": {
                **inp,
                "transcript": transcript,
                "transcription_confidence": confidence,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: understand_intent
    # ------------------------------------------------------------------

    async def _understand_intent(self, state: AgentState) -> dict[str, Any]:
        """Classify the user's transcript into a structured intent via Claude.

        Returns a JSON object with ``intent``, ``params``, ``confidence``,
        and ``rephrased`` (the query rewritten as a clear instruction).
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        transcript: str = inp.get("transcript", "")

        prompt = (
            f'The user said: "{transcript}"\n\n'
            "Classify this into ONE of these intents and extract parameters:\n"
            "- spending_query: asking about spending "
            "(params: category, time_period)\n"
            "- balance_query: asking about account balance "
            "(params: account_type)\n"
            "- subscription_query: asking about subscriptions "
            "(params: action=list/cancel/find)\n"
            "- forecast_query: asking about future finances "
            "(params: horizon_days)\n"
            "- goal_query: asking about savings goals "
            "(params: goal_name)\n"
            "- alert_query: asking about alerts or warnings "
            "(params: none)\n"
            "- bill_query: asking about bills or negotiation "
            "(params: provider)\n"
            "- general_question: general financial question "
            "(params: question_text)\n\n"
            'Return JSON only: {"intent": str, "params": dict, '
            '"confidence": float, "rephrased": str}\n'
            "rephrased = the query rewritten as a clear instruction"
        )

        try:
            response = self._llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text.strip()

            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0]

            parsed: dict[str, Any] = json.loads(raw_text)
        except (json.JSONDecodeError, IndexError, Exception) as exc:
            logger.warning("Intent parse failed, defaulting: %s", exc)
            parsed = {
                "intent": "general_question",
                "params": {"question_text": transcript},
                "confidence": 0.5,
                "rephrased": transcript,
            }

        intent = parsed.get("intent", "general_question")
        confidence = float(parsed.get("confidence", 0.5))

        step = f"Intent: {intent} (confidence={confidence:.2f})"
        logger.info(step)
        return {
            "input": {
                **inp,
                "intent": intent,
                "intent_params": parsed.get("params", {}),
                "intent_confidence": confidence,
                "rephrased_query": parsed.get("rephrased", transcript),
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: route_and_execute
    # ------------------------------------------------------------------

    async def _route_and_execute(self, state: AgentState) -> dict[str, Any]:
        """Route the classified intent to the correct data source.

        This is an **agent that calls other agents**.  The routing strategy
        splits into two tiers:

        **Tier 1 — Direct MCP tool calls** for simple, single-step look-ups
        (spending, balance, subscriptions).  These are fast and don't need
        the overhead of a full agent graph.

        **Tier 2 — Sub-agent delegation** for complex multi-step workflows
        (cashflow forecast, goal tracking, alert triage, bill negotiation).
        These are dispatched through ``agent_orchestrator.run_agent()`` and
        return a full ``AgentState``, from which we extract the ``output``.

        **Tier 3 — Claude fallback** for general financial questions that
        don't map to a specific tool or agent.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        intent: str = inp.get("intent", "general_question")
        params: dict[str, Any] = inp.get("intent_params", {})
        user_id: str = state["user_id"]
        transcript: str = inp.get("transcript", "")

        query_result: Any = None
        handler_label: str = intent

        try:
            if intent == "spending_query":
                start, end = _resolve_time_period(params.get("time_period"))
                query_result = await self.call_tool(
                    "plaid_mcp",
                    "get_spending_by_category",
                    {
                        "user_id": user_id,
                        "start_date": start,
                        "end_date": end,
                    },
                )

            elif intent == "balance_query":
                query_result = await self.call_tool(
                    "plaid_mcp",
                    "get_account_balances",
                    {"user_id": user_id},
                )

            elif intent == "subscription_query":
                query_result = await self.call_tool(
                    "plaid_mcp",
                    "get_recurring_transactions",
                    {"user_id": user_id},
                )

            elif intent in _AGENT_INTENTS:
                # ── Tier 2: delegate to a full sub-agent ───────────
                # The orchestrator runs the entire agent graph and
                # returns a complete AgentState.  We only surface the
                # output dict to keep the voice layer thin.
                agent_name = _AGENT_INTENTS[intent]
                handler_label = f"{intent} → {agent_name}"
                sub_state = await agent_orchestrator.run_agent(
                    agent_name,
                    user_id=user_id,
                    input_data={},
                )
                query_result = sub_state.get("output", {})

            else:
                # ── Tier 3: general question via Claude ────────────
                handler_label = "general_question → Claude"
                query_result = await self._answer_general_question(
                    transcript, user_id,
                )

        except (RuntimeError, KeyError) as exc:
            logger.error("Route handler failed for %s: %s", intent, exc)
            query_result = {"error": str(exc)}

        step = f"Routed to {handler_label} handler, got result"
        logger.info(step)
        return {
            "input": {**inp, "query_result": query_result},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: formulate_response
    # ------------------------------------------------------------------

    async def _formulate_response(self, state: AgentState) -> dict[str, Any]:
        """Convert raw data into a natural spoken response via Claude.

        The prompt enforces a conversational tone ("talk to a friend"),
        caps the output at 60 words, and avoids robotic phrasing.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        transcript: str = inp.get("transcript", "")
        query_result: Any = inp.get("query_result", {})

        prompt = (
            f'The user asked: "{transcript}"\n'
            f"Here is the data: {query_result}\n\n"
            "Write a natural, conversational spoken response in 1-3 "
            "sentences.\nRules:\n"
            '- Speak as if talking to a friend, not reading a report\n'
            '- Use specific numbers when relevant ("you spent $342 on '
            'food this month")\n'
            "- If the news is bad (low balance, overspending): be honest "
            "but calm\n"
            '- Never say "Based on the data" or "According to my '
            'analysis"\n'
            "- End with one actionable suggestion if relevant\n"
            "- Max 60 words"
        )

        try:
            response = self._llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            spoken_response: str = response.content[0].text.strip()
        except Exception as exc:
            logger.warning("Response formulation failed: %s", exc)
            if isinstance(query_result, dict) and "summary" in query_result:
                spoken_response = str(query_result["summary"])
            else:
                spoken_response = (
                    "I found your information but had trouble putting it "
                    "into words. Check the app for full details."
                )

        step = f"Formulated {len(spoken_response)} char response"
        logger.info(step)
        return {
            "input": {**inp, "spoken_response": spoken_response},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 5: synthesize_and_complete
    # ------------------------------------------------------------------

    async def _synthesize_and_complete(self, state: AgentState) -> dict[str, Any]:
        """Synthesise speech from the spoken response and finalise output."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        spoken_response: str = inp.get("spoken_response", "")
        voice_speed: float = float(inp.get("voice_speed", 1.0))

        audio_base64: str | None = None
        try:
            tts_result: dict[str, Any] = await self.call_tool(
                "voice_mcp",
                "synthesize_speech",
                {
                    "text": spoken_response,
                    "voice_speed": voice_speed,
                },
            )
            audio_base64 = tts_result.get("audio_base64")
        except RuntimeError as exc:
            logger.warning("Speech synthesis failed: %s", exc)

        output: dict[str, Any] = {
            "transcript": inp.get("transcript", ""),
            "intent": inp.get("intent", ""),
            "spoken_response": spoken_response,
            "audio_base64": audio_base64,
            "audio_format": "mp3",
            "query_result": inp.get("query_result", {}),
        }

        step = "Synthesised speech and completed"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _answer_general_question(
        self,
        question: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Answer a general financial question with spending context.

        Fetches the last 30 days of spending as background, then asks
        Claude to answer concisely using that context.
        """
        today = date.today()
        start = today - timedelta(days=30)

        spending_summary: dict[str, Any] = {}
        try:
            spending_summary = await self.call_tool(
                "plaid_mcp",
                "get_spending_by_category",
                {
                    "user_id": user_id,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": today.strftime("%Y-%m-%d"),
                },
            )
        except RuntimeError as exc:
            logger.warning("Spending context unavailable: %s", exc)

        prompt = (
            "You are a friendly personal finance assistant. Answer this "
            "question concisely using the user's financial context:\n\n"
            f"Question: {question}\n\n"
            f"Context (last 30 days spending): {spending_summary}"
        )

        try:
            response = self._llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.content[0].text.strip()
        except Exception as exc:
            logger.warning("General question LLM call failed: %s", exc)
            answer = "Sorry, I couldn't process that question right now."

        return {"answer": answer, "spending_context": spending_summary}
