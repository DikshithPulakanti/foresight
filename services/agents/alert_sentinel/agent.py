"""Alert Sentinel — the brain of Foresight's notification system.

This agent is designed to run on a 15-minute cadence.  It collects every
pending alert produced by all other agents (transaction monitor, cashflow
prophet, email monitor, etc.), deduplicates them, scores each one on a
0-100 urgency scale, and decides which are worth a push notification vs.
which should be silently logged.

The scoring formula is intentionally transparent and tunable so it's easy
to explain and adjust::

    final_score = SEVERITY_BASE
                + AMOUNT_BOOST
                + RECENCY_BOOST
                + TYPE_BOOST
                - DISMISSED_PENALTY

Graph pipeline::

    initialise
      → collect_pending_alerts
      → score_and_deduplicate
      → select_notifications
      → update_alert_statuses
      → generate_notification_payload
      → finalise → END
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Scoring constants — kept at module level for easy tuning
# ------------------------------------------------------------------

SEVERITY_BASE: dict[str, int] = {
    "critical": 90,
    "high": 70,
    "medium": 45,
    "low": 20,
    "info": 5,
}
"""Base score derived from the alert's severity label."""

AMOUNT_THRESHOLDS: list[tuple[float, int]] = [
    (1_000, 15),
    (500, 10),
    (100, 5),
]
"""(min_amount, bonus) — checked top-down, first match wins."""

RECENCY_THRESHOLDS: list[tuple[int, int]] = [
    (1, 10),   # created < 1 hour ago
    (6, 5),    # created < 6 hours ago
]
"""(max_hours, bonus) — checked top-down, first match wins."""

TYPE_BOOST: dict[str, int] = {
    "overdraft_risk": 20,
    "cashflow_critical": 20,
    "unusual_transaction": 10,
    "goal_achieved": -10,
}
"""Per-type adjustment.  Negative values lower urgency (good news)."""

DISMISSED_PENALTY = 30
"""Score reduction when a type was dismissed 3+ times in the last 7 days."""

MAX_NOTIFICATIONS_PER_RUN = 3
"""Push notification budget per 15-minute cycle."""

CRITICAL_SCORE_THRESHOLD = 85
"""Alerts at or above this score bypass the budget — always notify."""

NOTIFY_FLOOR = 40
"""Minimum score for an alert to qualify as a push notification."""

DEDUP_WINDOW_HOURS = 6
"""Two alerts with the same type + merchant within this window are duplicates."""

RUN_INTERVAL_MINUTES = 15


def _parse_dt(raw: Any) -> datetime | None:
    """Best-effort ISO datetime parse, returning UTC-aware or *None*."""
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _hours_since(dt: datetime | None) -> float:
    if dt is None:
        return 999.0
    now = datetime.now(timezone.utc)
    return max((now - dt).total_seconds() / 3600, 0)


def _dedup_key(alert: dict[str, Any]) -> str:
    """Normalised key used to detect duplicate alerts."""
    alert_type = str(alert.get("type", "")).lower().strip()
    merchant = str(
        alert.get("merchant", alert.get("service", alert.get("title", "")))
    ).lower().strip()
    return f"{alert_type}::{merchant}"


class AlertSentinelAgent(BaseAgent):
    """Runs every 15 minutes — collects, scores, deduplicates, and dispatches alerts.

    The sentinel acts as a single funnel: every other Foresight agent writes
    raw alerts into Neo4j (and optionally Postgres).  The sentinel reads them,
    applies a deterministic scoring formula, respects the user's notification
    budget, and emits a ready-to-push payload.

    Scoring formula (NODE 2)
    ------------------------
    .. code-block:: text

        final_score = SEVERITY_BASE[severity]
                    + AMOUNT_BOOST(amount)
                    + RECENCY_BOOST(created_at)
                    + TYPE_BOOST.get(type, 0)
                    − DISMISSED_PENALTY  (if type dismissed 3+ times in 7 days)

    Each component is defined at module level for easy tuning.
    """

    def __init__(self) -> None:
        super().__init__(
            name="alert_sentinel",
            description=(
                "Runs every 15 minutes — collects all pending alerts, "
                "deduplicates, scores by urgency, and only notifies the "
                "user about what truly matters"
            ),
        )
        self._llm = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the five processing nodes."""
        builder.add_node("collect_pending_alerts", self._collect_pending_alerts)
        builder.add_node("score_and_deduplicate", self._score_and_deduplicate)
        builder.add_node("select_notifications", self._select_notifications)
        builder.add_node("update_alert_statuses", self._update_alert_statuses)
        builder.add_node(
            "generate_notification_payload",
            self._generate_notification_payload,
        )

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline."""
        builder.add_edge("initialise", "collect_pending_alerts")
        builder.add_edge("collect_pending_alerts", "score_and_deduplicate")
        builder.add_edge("score_and_deduplicate", "select_notifications")
        builder.add_edge("select_notifications", "update_alert_statuses")
        builder.add_edge("update_alert_statuses", "generate_notification_payload")
        builder.add_edge("generate_notification_payload", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: collect_pending_alerts
    # ------------------------------------------------------------------

    async def _collect_pending_alerts(self, state: AgentState) -> dict[str, Any]:
        """Gather all unread alerts from both Neo4j and Postgres.

        Alerts are merged by ``(type, title)`` similarity so the downstream
        scoring node never sees the same event twice from different stores.
        """
        user_id: str = state["user_id"]
        raw_alerts: list[dict[str, Any]] = []

        # --- Neo4j via graph_mcp ---
        try:
            graph_alerts: list[dict[str, Any]] = await self.call_tool(
                "graph_mcp",
                "get_pending_alerts",
                {"user_id": user_id, "hours_back": 24},
            )
            for alert in graph_alerts:
                alert["source"] = "graph"
            raw_alerts.extend(graph_alerts)
        except RuntimeError as exc:
            logger.warning("Neo4j alert fetch failed: %s", exc)

        # --- Postgres via graph_mcp (DB-stored alerts) ---
        try:
            db_alerts: list[dict[str, Any]] = await self.call_tool(
                "graph_mcp",
                "get_unread_db_alerts",
                {"user_id": user_id},
            )
            for alert in db_alerts:
                alert["source"] = "db"
            raw_alerts.extend(db_alerts)
        except RuntimeError as exc:
            logger.warning("Postgres alert fetch failed: %s", exc)

        # --- Deduplicate across sources by (type + title) ---
        seen: dict[str, dict[str, Any]] = {}
        for alert in raw_alerts:
            key = _dedup_key(alert)
            if key not in seen:
                seen[key] = alert
            else:
                existing_dt = _parse_dt(seen[key].get("created_at"))
                new_dt = _parse_dt(alert.get("created_at"))
                if new_dt and (not existing_dt or new_dt > existing_dt):
                    seen[key] = alert

        deduped = list(seen.values())
        step = f"Collected {len(deduped)} pending alerts from last 24 hours"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "raw_alerts": deduped},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: score_and_deduplicate
    # ------------------------------------------------------------------

    async def _score_and_deduplicate(self, state: AgentState) -> dict[str, Any]:
        """Score each alert 0-100 and remove time-window duplicates.

        **Scoring formula** (the core of the sentinel):

        .. code-block:: text

            final_score = SEVERITY_BASE[severity]    # 5–90 depending on label
                        + AMOUNT_BOOST               # +5/+10/+15 for large amounts
                        + RECENCY_BOOST              # +5/+10 for recent alerts
                        + TYPE_BOOST                 # +20/+10/−10 per alert type
                        − DISMISSED_PENALTY          # −30 if user dismissed this
                                                     #   type 3+ times in 7 days

        Each constant is defined at module level for easy tuning.

        **Deduplication** removes alerts that share the same ``type`` AND
        ``merchant/service`` within a 6-hour window, keeping only the
        highest-scored instance.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        raw_alerts: list[dict[str, Any]] = inp.get("raw_alerts", [])
        dismissed_types: list[str] = inp.get("dismissed_types", [])

        scored: list[dict[str, Any]] = []

        for alert in raw_alerts:
            severity = str(alert.get("severity", "low")).lower()
            amount = float(alert.get("amount", 0) or 0)
            created_at = _parse_dt(alert.get("created_at"))
            alert_type = str(alert.get("type", "")).lower()
            hours_age = _hours_since(created_at)

            # ── Step 1: Severity base ──────────────────────────────
            score = SEVERITY_BASE.get(severity, SEVERITY_BASE["low"])

            # ── Step 2: Amount boost (first threshold match wins) ──
            for threshold, boost in AMOUNT_THRESHOLDS:
                if amount > threshold:
                    score += boost
                    break

            # ── Step 3: Recency boost ──────────────────────────────
            for max_hours, boost in RECENCY_THRESHOLDS:
                if hours_age < max_hours:
                    score += boost
                    break

            # ── Step 4: Type-specific boost ────────────────────────
            score += TYPE_BOOST.get(alert_type, 0)

            # ── Step 5: Dismissed-type penalty ─────────────────────
            # If the user dismissed this alert type 3+ times in the
            # last 7 days, assume they don't want it.
            if dismissed_types.count(alert_type) >= 3:
                score -= DISMISSED_PENALTY

            score = max(0, min(score, 100))

            scored.append({
                **alert,
                "final_score": score,
                "created_at_iso": created_at.isoformat() if created_at else None,
                "hours_age": round(hours_age, 1),
            })

        # ── Time-window deduplication ──────────────────────────────
        # Within each dedup key, keep only the highest-scored alert
        # if both fell within DEDUP_WINDOW_HOURS of each other.
        dedup_buckets: dict[str, list[dict[str, Any]]] = {}
        for alert in scored:
            key = _dedup_key(alert)
            dedup_buckets.setdefault(key, []).append(alert)

        deduped: list[dict[str, Any]] = []
        removed = 0
        for bucket in dedup_buckets.values():
            if len(bucket) == 1:
                deduped.append(bucket[0])
                continue

            bucket.sort(key=lambda a: a["final_score"], reverse=True)
            best = bucket[0]
            best_dt = _parse_dt(best.get("created_at"))
            deduped.append(best)

            for other in bucket[1:]:
                other_dt = _parse_dt(other.get("created_at"))
                if best_dt and other_dt:
                    gap = abs((best_dt - other_dt).total_seconds()) / 3600
                    if gap <= DEDUP_WINDOW_HOURS:
                        removed += 1
                        continue
                deduped.append(other)

        deduped.sort(key=lambda a: a["final_score"], reverse=True)

        step = f"Scored {len(scored)} alerts, {removed} removed as duplicates"
        logger.info(step)
        return {
            "input": {**inp, "scored_alerts": deduped},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: select_notifications
    # ------------------------------------------------------------------

    async def _select_notifications(self, state: AgentState) -> dict[str, Any]:
        """Apply the notification budget and pick which alerts get pushed.

        Rules:

        * Alerts with ``score >= 85`` (critical financial events) **always**
          get pushed — they bypass the budget.
        * Remaining budget slots (up to ``MAX_NOTIFICATIONS_PER_RUN``) are
          filled by the highest-scored alerts above ``NOTIFY_FLOOR``.
        * Everything else is silently logged.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        scored: list[dict[str, Any]] = inp.get("scored_alerts", [])

        critical: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        logged: list[dict[str, Any]] = []

        for alert in scored:
            if alert["final_score"] >= CRITICAL_SCORE_THRESHOLD:
                critical.append({**alert, "disposition": "notify"})
            elif alert["final_score"] >= NOTIFY_FLOOR:
                candidates.append(alert)
            else:
                logged.append({**alert, "disposition": "logged"})

        remaining_budget = max(MAX_NOTIFICATIONS_PER_RUN - len(critical), 0)
        promoted = candidates[:remaining_budget]
        demoted = candidates[remaining_budget:]

        notifications = critical + [
            {**a, "disposition": "notify"} for a in promoted
        ]
        logged.extend({**a, "disposition": "logged"} for a in demoted)

        total = len(scored)
        step = f"Selected {len(notifications)} notifications out of {total} alerts"
        logger.info(step)
        return {
            "input": {
                **inp,
                "notifications": notifications,
                "logged_alerts": logged,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: update_alert_statuses
    # ------------------------------------------------------------------

    async def _update_alert_statuses(self, state: AgentState) -> dict[str, Any]:
        """Persist disposition back to Neo4j and Postgres.

        * Notified alerts → ``status = "notified"``, ``notified_at = now()``.
        * Logged alerts → ``status = "logged"``.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        notifications: list[dict[str, Any]] = inp.get("notifications", [])
        logged_alerts: list[dict[str, Any]] = inp.get("logged_alerts", [])
        user_id: str = state["user_id"]

        notified_count = 0
        logged_count = 0

        # --- Mark notified ---
        for alert in notifications:
            alert_id = alert.get("id")
            if not alert_id:
                continue
            try:
                await self.call_tool(
                    "graph_mcp",
                    "update_alert_status",
                    {
                        "alert_id": alert_id,
                        "status": "notified",
                        "user_id": user_id,
                    },
                )
                notified_count += 1
            except RuntimeError as exc:
                logger.warning("Failed to mark alert %s notified: %s", alert_id, exc)

        # --- Mark logged ---
        logged_ids = [a.get("id") for a in logged_alerts if a.get("id")]
        if logged_ids:
            try:
                await self.call_tool(
                    "graph_mcp",
                    "bulk_update_alert_status",
                    {
                        "alert_ids": logged_ids,
                        "status": "logged",
                        "user_id": user_id,
                    },
                )
                logged_count = len(logged_ids)
            except RuntimeError as exc:
                logger.warning("Bulk log-status update failed: %s", exc)

        step = f"Updated statuses — {notified_count} notified, {logged_count} logged"
        logger.info(step)
        return {
            "input": {
                **inp,
                "notified_count": notified_count,
                "logged_count": logged_count,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 5: generate_notification_payload
    # ------------------------------------------------------------------

    async def _generate_notification_payload(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """Build push-ready payloads and a one-line Claude summary.

        Each payload contains everything the mobile client needs to display
        a push notification: title (≤ 50 chars), body (≤ 100 chars),
        severity badge, deep-link URL, and the unread badge count.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        notifications: list[dict[str, Any]] = inp.get("notifications", [])
        logged_alerts: list[dict[str, Any]] = inp.get("logged_alerts", [])
        scored_total = len(inp.get("scored_alerts", []))
        notified_count: int = inp.get("notified_count", 0)
        logged_count: int = inp.get("logged_count", 0)
        total_unread = len(notifications) + len(logged_alerts)

        payloads: list[dict[str, Any]] = []
        for alert in notifications:
            title = str(alert.get("title", "Alert"))[:50]
            body = str(alert.get("message", ""))[:100]
            alert_id = alert.get("id", "")
            payloads.append({
                "alert_id": alert_id,
                "title": title,
                "body": body,
                "severity": alert.get("severity", "medium"),
                "score": alert.get("final_score", 0),
                "action_url": f"/alerts/{alert_id}",
                "badge_count": total_unread,
            })

        # --- Summary ---
        if not notifications:
            summary = "All clear — no urgent alerts in the last 24 hours"
        else:
            try:
                prompt = (
                    f"Summarize these {len(notifications)} financial alerts "
                    f"in one sentence under 80 characters for a push "
                    f"notification preview:\n"
                    + "\n".join(
                        f"- {a.get('title', '')}: {a.get('message', '')}"
                        for a in notifications
                    )
                )
                response = self._llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                summary = response.content[0].text.strip()[:80]
            except Exception as exc:
                logger.warning("Summary generation failed: %s", exc)
                summary = (
                    f"{len(notifications)} alert(s) need your attention"
                )

        output: dict[str, Any] = {
            "notifications": payloads,
            "logged_count": logged_count,
            "total_alerts_processed": scored_total,
            "notification_summary": summary,
            "next_run_in_minutes": RUN_INTERVAL_MINUTES,
        }

        step = "Generated notification payload"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }
