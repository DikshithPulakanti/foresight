"""Email Monitor — scans Gmail for financial signals and fires proactive alerts.

This agent runs on a background schedule (or on-demand) and performs five
steps:

1. **Scan** the user's inbox for financial emails and price-increase
   notifications via ``gmail_mcp``.
2. **Classify** each email by urgency (0-10 scale) based on category, due
   date proximity, and severity.
3. **Extract** concrete action items from the high-urgency emails — what
   needs to happen, by when, and how much.
4. **Create alerts** in the knowledge graph proportional to urgency.
5. **Generate a digest** via Claude — a concise bullet-point summary the
   user can glance at and know exactly what to do.

Urgency tiers::

    10  CRITICAL  bill due within 3 days
     9  CRITICAL  overdue notice
     8  HIGH      bill due within 7 days / renewal within 3 days
     7  HIGH      price increase effective within 14 days
     6  MEDIUM    renewal within 7 days
     4  LOW       default
     3  INFO      refund
     2  INFO      payment confirmation

Graph pipeline::

    initialise
      → scan_inbox
      → classify_urgency
      → extract_action_items
      → create_alerts
      → generate_digest
      → finalise → END
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent

logger = logging.getLogger(__name__)

_DOLLAR_RE = re.compile(r"\$?([\d,]+\.?\d*)")


# ------------------------------------------------------------------
# Urgency helpers
# ------------------------------------------------------------------

def _parse_date(raw: str | None) -> date | None:
    """Best-effort date parse from various formats."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw[:10] if len(raw) >= 10 else raw, fmt).date()
        except (ValueError, TypeError):
            continue
    try:
        return date.fromisoformat(raw[:10])
    except (ValueError, TypeError):
        return None


def _days_until(raw_date: str | None) -> int | None:
    """Days from today to *raw_date*.  Negative means past-due."""
    d = _parse_date(raw_date)
    if d is None:
        return None
    return (d - date.today()).days


def _parse_dollar(raw: str | float | int | None) -> float:
    """Extract a numeric dollar value."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    match = _DOLLAR_RE.search(str(raw))
    return float(match.group(1).replace(",", "")) if match else 0.0


def _extract_service_name(email: dict[str, Any]) -> str:
    """Best-effort service name from sender or subject."""
    sender = email.get("sender", "")
    if "<" in sender:
        name = sender.split("<")[0].strip().strip('"')
        if name:
            return name
    subject = email.get("subject", "")
    for prefix in ("Re: ", "Fwd: ", "Your ", "your "):
        if subject.startswith(prefix):
            subject = subject[len(prefix):]
    return subject.split(" — ")[0].split(" - ")[0][:40] or "Unknown service"


def _score_email(category: str, days_left: int | None) -> tuple[int, str]:
    """Return ``(urgency_score, severity_label)`` for an email.

    See module docstring for the full urgency table.
    """
    if category == "overdue_notice":
        return 9, "critical"

    if category == "bill_due":
        if days_left is not None and days_left <= 3:
            return 10, "critical"
        if days_left is not None and days_left <= 7:
            return 8, "high"
        return 6, "medium"

    if category == "price_increase":
        if days_left is not None and days_left <= 14:
            return 7, "high"
        return 5, "medium"

    if category == "subscription_renewal":
        if days_left is not None and days_left <= 3:
            return 8, "high"
        if days_left is not None and days_left <= 7:
            return 6, "medium"
        return 5, "medium"

    if category == "payment_confirmation":
        return 2, "low"
    if category == "refund":
        return 3, "low"

    return 4, "low"


def _action_for_category(category: str) -> str:
    """Map an email category to its required user action."""
    return {
        "bill_due": "pay_now",
        "overdue_notice": "pay_now",
        "subscription_renewal": "cancel_before",
        "price_increase": "review_price_change",
    }.get(category, "none")


class EmailMonitorAgent(BaseAgent):
    """Scans Gmail for financial signals and fires proactive alerts.

    Covers bills due, subscription renewals, price increases, overdue
    notices, payment confirmations, and refunds.  High-urgency items
    become knowledge-graph alerts; everything is rolled into a concise
    digest.
    """

    def __init__(self) -> None:
        super().__init__(
            name="email_monitor",
            description=(
                "Continuously scans Gmail for financial signals — bills due, "
                "subscription renewals, price increases — and fires alerts "
                "before they hit your account"
            ),
        )
        self._llm = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the five processing nodes."""
        builder.add_node("scan_inbox", self._scan_inbox)
        builder.add_node("classify_urgency", self._classify_urgency)
        builder.add_node("extract_action_items", self._extract_action_items)
        builder.add_node("create_alerts", self._create_alerts)
        builder.add_node("generate_digest", self._generate_digest)

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline."""
        builder.add_edge("initialise", "scan_inbox")
        builder.add_edge("scan_inbox", "classify_urgency")
        builder.add_edge("classify_urgency", "extract_action_items")
        builder.add_edge("extract_action_items", "create_alerts")
        builder.add_edge("create_alerts", "generate_digest")
        builder.add_edge("generate_digest", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: scan_inbox
    # ------------------------------------------------------------------

    async def _scan_inbox(self, state: AgentState) -> dict[str, Any]:
        """Fetch financial emails and price-increase notices, then merge."""
        days_back: int = state.get("input", {}).get("days_back", 7)

        try:
            financial_emails: list[dict[str, Any]] = await self.call_tool(
                "gmail_mcp",
                "scan_financial_emails",
                {"days_back": days_back, "max_results": 50},
            )
        except RuntimeError as exc:
            logger.error("scan_inbox: financial email scan failed: %s", exc)
            return self.set_error(state, str(exc))

        price_increases: list[dict[str, Any]] = []
        try:
            price_increases = await self.call_tool(
                "gmail_mcp",
                "check_price_increases",
                {"days_back": days_back},
            )
        except RuntimeError as exc:
            logger.warning("scan_inbox: price-increase scan failed (non-fatal): %s", exc)

        # Normalise price-increase records into the same shape as financial emails
        for pi in price_increases:
            financial_emails.append({
                "id": pi.get("email_id", ""),
                "subject": f"Price increase: {pi.get('service', 'Unknown')}",
                "sender": pi.get("service", ""),
                "date": pi.get("effective_date"),
                "category": "price_increase",
                "amount_mentioned": pi.get("new_price"),
                "urgency": pi.get("urgency", "medium"),
                "snippet": (
                    f"{pi.get('service', '')} increasing from "
                    f"{pi.get('old_price', '?')} to {pi.get('new_price', '?')}"
                ),
                "_old_price": pi.get("old_price"),
                "_new_price": pi.get("new_price"),
                "_effective_date": pi.get("effective_date"),
            })

        # Deduplicate by email id
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for email in financial_emails:
            eid = email.get("id", "")
            if eid and eid in seen:
                continue
            seen.add(eid)
            unique.append(email)

        step = f"Scanned inbox — found {len(unique)} financial emails"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "emails": unique},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: classify_urgency
    # ------------------------------------------------------------------

    async def _classify_urgency(self, state: AgentState) -> dict[str, Any]:
        """Assign a 0-10 urgency score and severity label to each email."""
        if state.get("status") == "failed":
            return {}

        emails: list[dict[str, Any]] = state.get("input", {}).get("emails", [])
        classified: list[dict[str, Any]] = []
        counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for email in emails:
            category = email.get("category", "other")

            # Estimate days until the financial event:
            # For price_increases use the effective_date; for everything
            # else use the email date as a proxy (bills are usually
            # emailed a few days before the due date).
            raw_date = email.get("_effective_date") or email.get("date")
            days_left = _days_until(raw_date)

            score, severity = _score_email(category, days_left)

            classified.append({
                **email,
                "urgency_score": score,
                "severity": severity,
                "days_left": days_left,
            })
            counts[severity] = counts.get(severity, 0) + 1

        classified.sort(key=lambda e: e["urgency_score"], reverse=True)

        step = (
            f"Classified {len(classified)} emails — "
            f"{counts['critical']} critical, {counts['high']} high priority"
        )
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "classified_emails": classified},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: extract_action_items
    # ------------------------------------------------------------------

    async def _extract_action_items(self, state: AgentState) -> dict[str, Any]:
        """Build a concrete action-item list from emails with urgency >= 6."""
        if state.get("status") == "failed":
            return {}

        classified: list[dict[str, Any]] = (
            state.get("input", {}).get("classified_emails", [])
        )

        action_items: list[dict[str, Any]] = []
        today = date.today()

        for email in classified:
            if email.get("urgency_score", 0) < 6:
                continue

            category = email.get("category", "other")
            service = _extract_service_name(email)
            amount = _parse_dollar(email.get("amount_mentioned"))
            action = _action_for_category(category)

            raw_date = email.get("_effective_date") or email.get("date")
            deadline_date = _parse_date(raw_date)
            if deadline_date is None:
                deadline_date = today + timedelta(days=7)

            action_items.append({
                "email_id": email.get("id", ""),
                "service": service,
                "action": action,
                "deadline": deadline_date.isoformat(),
                "amount": amount,
                "urgency": email.get("urgency_score", 0),
                "severity": email.get("severity", "medium"),
                "category": category,
            })

        action_items.sort(key=lambda a: a["deadline"])

        step = f"Extracted {len(action_items)} action items requiring attention"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "action_items": action_items},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: create_alerts
    # ------------------------------------------------------------------

    async def _create_alerts(self, state: AgentState) -> dict[str, Any]:
        """Persist alerts in the knowledge graph scaled by urgency."""
        if state.get("status") == "failed":
            return {}

        action_items: list[dict[str, Any]] = (
            state.get("input", {}).get("action_items", [])
        )
        user_id: str = state["user_id"]
        created = 0

        for item in action_items:
            score = item.get("urgency", 0)
            if score >= 8:
                severity = "critical"
            elif score >= 6:
                severity = "high"
            else:
                severity = "medium"

            action_label = {
                "pay_now": "Payment required",
                "cancel_before": "Review / cancel before renewal",
                "review_price_change": "Price change — review",
            }.get(item["action"], item["action"])

            try:
                await self.call_tool(
                    "graph_mcp",
                    "create_alert_node",
                    {
                        "user_id": user_id,
                        "alert_type": "email_financial_signal",
                        "title": f"{item['service']}: {action_label}",
                        "message": (
                            f"Due {item['deadline']} — "
                            f"${item['amount']:.2f}. "
                            f"Action: {item['action']}"
                        ),
                        "severity": severity,
                        "amount": item["amount"],
                    },
                )
                created += 1
            except RuntimeError as exc:
                logger.warning(
                    "Failed to create alert for %s: %s", item["service"], exc,
                )

        step = f"Created {created} alerts from email signals"
        logger.info(step)
        return {
            "input": {**state.get("input", {}), "alerts_created": created},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 5: generate_digest
    # ------------------------------------------------------------------

    async def _generate_digest(self, state: AgentState) -> dict[str, Any]:
        """Ask Claude for a concise bullet-point digest and set output."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        emails: list[dict[str, Any]] = inp.get("emails", [])
        action_items: list[dict[str, Any]] = inp.get("action_items", [])
        alerts_created: int = inp.get("alerts_created", 0)

        total_exposure = round(sum(a["amount"] for a in action_items), 2)
        critical_count = sum(1 for a in action_items if a.get("urgency", 0) >= 9)
        high_count = sum(1 for a in action_items if 7 <= a.get("urgency", 0) < 9)

        if action_items:
            prompt = (
                f"The user has {len(action_items)} financial emails requiring "
                f"attention:\n{action_items}\n\n"
                f"Write a concise email digest in 3-5 bullet points. Each "
                f"bullet should be one action: service name, what needs to "
                f"happen, and by when.\n"
                f"Start with the most urgent. Use plain English, no jargon.\n"
                f"End with the total financial exposure "
                f"(${total_exposure:.2f} across all items)."
            )
            try:
                response = self._llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                digest: str = response.content[0].text
            except Exception as exc:
                logger.warning("Digest generation failed: %s", exc)
                lines = [
                    f"• {a['service']}: {a['action']} by {a['deadline']} — ${a['amount']:.2f}"
                    for a in action_items[:5]
                ]
                digest = "\n".join(lines) + f"\n\nTotal exposure: ${total_exposure:.2f}"
        else:
            digest = "No urgent financial emails in the scanned period. You're all clear."

        output: dict[str, Any] = {
            "emails_scanned": len(emails),
            "action_items": action_items,
            "alerts_created": alerts_created,
            "critical_count": critical_count,
            "high_count": high_count,
            "total_financial_exposure": total_exposure,
            "digest": digest,
        }

        step = "Generated digest"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }
