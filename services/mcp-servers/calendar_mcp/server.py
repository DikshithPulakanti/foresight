"""Calendar MCP server — financial timing awareness for Foresight agents.

This server connects to Google Calendar and exposes tools that let AI agents
reason about the temporal dimension of a user's finances: when bills are due,
when the next payday lands, whether upcoming travel will create unusual
expenses, and so on.

Mock mode
---------
Pass ``mock=True`` to skip Google auth and return realistic fake calendar
data so the full agent pipeline can be exercised locally.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseMCPServer

logger = logging.getLogger(__name__)

_FINANCIAL_KEYWORDS: list[str] = [
    "rent", "mortgage", "bill", "payment", "salary", "payday",
    "insurance", "subscription", "travel", "flight", "hotel", "tax",
]

_PAYDAY_KEYWORDS: list[str] = [
    "payday", "salary", "direct deposit", "paycheck",
]


# ------------------------------------------------------------------
# Mock data
# ------------------------------------------------------------------

def _mock_upcoming_events() -> list[dict[str, Any]]:
    today = date.today()
    return [
        {"title": "Rent due", "date": str(today + timedelta(days=5)), "type": "rent", "estimated_amount": 1850.00, "days_until": 5},
        {"title": "Payday", "date": str(today + timedelta(days=10)), "type": "salary", "estimated_amount": 3200.00, "days_until": 10},
        {"title": "Car insurance", "date": str(today + timedelta(days=14)), "type": "insurance", "estimated_amount": 142.00, "days_until": 14},
        {"title": "Flight to NYC", "date": str(today + timedelta(days=21)), "type": "travel", "estimated_amount": 380.00, "days_until": 21},
        {"title": "Internet bill", "date": str(today + timedelta(days=25)), "type": "bill", "estimated_amount": 89.00, "days_until": 25},
    ]


def _mock_add_reminder(title: str, event_date: str) -> dict[str, Any]:
    return {
        "event_id": "mock-event-001",
        "calendar_link": f"https://calendar.google.com/calendar/event?eid=mock-event-001",
    }


def _mock_payday_schedule() -> dict[str, Any]:
    today = date.today()
    day = today.day
    if day <= 15:
        next_payday = today.replace(day=15)
    else:
        month = today.month + 1 if today.month < 12 else 1
        year = today.year if today.month < 12 else today.year + 1
        next_payday = date(year, month, 15)
    return {
        "frequency": "biweekly",
        "next_payday": str(next_payday),
        "confidence": 0.85,
    }


# ------------------------------------------------------------------
# Calendar MCP Server
# ------------------------------------------------------------------

class CalendarMCPServer(BaseMCPServer):
    """MCP server that exposes Google Calendar tools for financial timing.

    Tools registered:

    1. **get_upcoming_financial_events** — bills, payday, travel in the
       next N days
    2. **add_financial_reminder** — create a calendar event for a bill or
       renewal
    3. **get_payday_schedule** — infer pay frequency and next payday from
       calendar history
    """

    def __init__(
        self,
        credentials_json: dict[str, Any] | None = None,
        mock: bool = False,
    ) -> None:
        super().__init__(name="calendar")

        self._mock = mock

        if mock:
            self._service = None
            logger.info("CalendarMCPServer created in MOCK mode")
        else:
            creds = Credentials.from_authorized_user_info(credentials_json)  # type: ignore[arg-type]
            self._service = build("calendar", "v3", credentials=creds)
            logger.info("CalendarMCPServer created (live)")

        self.setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Register the three Calendar tools."""

        self.register_tool(
            name="get_upcoming_financial_events",
            description=(
                "Get calendar events in the next N days that are financially "
                "relevant — bills, payday, rent, travel"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "days_ahead": {"type": "integer", "default": 30},
                },
                "required": [],
            },
            handler=self._get_upcoming_financial_events_handler,
        )

        self.register_tool(
            name="add_financial_reminder",
            description=(
                "Add a calendar reminder for an upcoming financial event like "
                "a bill due date or subscription renewal"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "description": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["title", "date", "description"],
            },
            handler=self._add_financial_reminder_handler,
        )

        self.register_tool(
            name="get_payday_schedule",
            description=(
                "Infer the user's payday schedule from their calendar and "
                "past patterns"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "months_back": {"type": "integer", "default": 3},
                },
                "required": [],
            },
            handler=self._get_payday_schedule_handler,
        )

    # ------------------------------------------------------------------
    # Google Calendar helpers
    # ------------------------------------------------------------------

    def _fetch_events(self, time_min: datetime, time_max: datetime) -> list[dict[str, Any]]:
        """Fetch events from the primary calendar within a time window."""
        result = (
            self._service.events()  # type: ignore[union-attr]
            .list(
                calendarId="primary",
                timeMin=time_min.isoformat(),
                timeMax=time_max.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        return result.get("items", [])

    @staticmethod
    def _classify_event(title: str) -> str | None:
        """Return the first matching financial keyword, or ``None``."""
        lower = title.lower()
        for kw in _FINANCIAL_KEYWORDS:
            if kw in lower:
                return kw
        return None

    @staticmethod
    def _parse_event_date(event: dict[str, Any]) -> date | None:
        """Extract a ``date`` from either an all-day or timed event."""
        start = event.get("start", {})
        raw = start.get("date") or start.get("dateTime", "")
        if not raw:
            return None
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Tool 1: get_upcoming_financial_events
    # ------------------------------------------------------------------

    async def _get_upcoming_financial_events_handler(
        self,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return financially-relevant calendar events sorted by date."""
        days_ahead: int = params.get("days_ahead", 30)

        if self._mock:
            return _mock_upcoming_events()

        today = date.today()
        time_min = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)
        time_max = datetime.combine(today + timedelta(days=days_ahead), datetime.min.time()).replace(tzinfo=timezone.utc)

        raw_events = self._fetch_events(time_min, time_max)
        logger.debug("Fetched %d calendar events for next %d days", len(raw_events), days_ahead)

        results: list[dict[str, Any]] = []
        for ev in raw_events:
            title = ev.get("summary", "")
            event_type = self._classify_event(title)
            if event_type is None:
                continue

            ev_date = self._parse_event_date(ev)
            if ev_date is None:
                continue

            results.append({
                "title": title,
                "date": str(ev_date),
                "type": event_type,
                "estimated_amount": None,
                "days_until": (ev_date - today).days,
            })
            logger.debug("Financial event: %s on %s (type=%s)", title, ev_date, event_type)

        results.sort(key=lambda e: e["date"])
        return results

    # ------------------------------------------------------------------
    # Tool 2: add_financial_reminder
    # ------------------------------------------------------------------

    async def _add_financial_reminder_handler(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a calendar event with bill-reminder popups."""
        title: str = params["title"]
        event_date: str = params["date"]
        description: str = params["description"]
        amount: float | None = params.get("amount")

        if self._mock:
            return _mock_add_reminder(title, event_date)

        body_lines = [description]
        if amount is not None:
            body_lines.append(f"Amount: ${amount:,.2f}")

        event_body: dict[str, Any] = {
            "summary": title,
            "description": "\n".join(body_lines),
            "start": {"date": event_date},
            "end": {"date": event_date},
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": 3 * 24 * 60},
                    {"method": "popup", "minutes": 1 * 24 * 60},
                ],
            },
        }

        created = (
            self._service.events()  # type: ignore[union-attr]
            .insert(calendarId="primary", body=event_body)
            .execute()
        )

        logger.info("Created calendar reminder '%s' on %s (id=%s)", title, event_date, created["id"])
        return {
            "event_id": created["id"],
            "calendar_link": created.get("htmlLink", ""),
        }

    # ------------------------------------------------------------------
    # Tool 3: get_payday_schedule
    # ------------------------------------------------------------------

    async def _get_payday_schedule_handler(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Infer pay frequency and next payday from calendar history."""
        months_back: int = params.get("months_back", 3)

        if self._mock:
            return _mock_payday_schedule()

        today = date.today()
        time_min = datetime.combine(
            today - timedelta(days=months_back * 30), datetime.min.time(),
        ).replace(tzinfo=timezone.utc)
        time_max = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)

        raw_events = self._fetch_events(time_min, time_max)
        logger.debug("Fetched %d events for payday analysis (%d months back)", len(raw_events), months_back)

        payday_dates: list[date] = []
        for ev in raw_events:
            title = (ev.get("summary") or "").lower()
            if any(kw in title for kw in _PAYDAY_KEYWORDS):
                ev_date = self._parse_event_date(ev)
                if ev_date is not None:
                    payday_dates.append(ev_date)

        if not payday_dates:
            return {"frequency": None, "next_payday": None, "confidence": 0.0}

        payday_dates.sort()

        # Compute gaps between consecutive paydays
        gaps = [(payday_dates[i + 1] - payday_dates[i]).days for i in range(len(payday_dates) - 1)]

        if not gaps:
            # Only one payday found — assume monthly on the same day
            next_month = today.month + 1 if today.month < 12 else 1
            next_year = today.year if today.month < 12 else today.year + 1
            try:
                next_payday = date(next_year, next_month, payday_dates[0].day)
            except ValueError:
                next_payday = date(next_year, next_month, 28)
            return {"frequency": "monthly", "next_payday": str(next_payday), "confidence": 0.4}

        avg_gap = sum(gaps) / len(gaps)
        gap_counts = Counter(round(g / 7) for g in gaps)
        most_common_weeks = gap_counts.most_common(1)[0][0]

        if most_common_weeks == 1:
            frequency = "weekly"
            next_payday = payday_dates[-1] + timedelta(days=7)
        elif most_common_weeks == 2:
            frequency = "biweekly"
            next_payday = payday_dates[-1] + timedelta(days=14)
        else:
            frequency = "monthly"
            last = payday_dates[-1]
            next_month = last.month + 1 if last.month < 12 else 1
            next_year = last.year if last.month < 12 else last.year + 1
            try:
                next_payday = date(next_year, next_month, last.day)
            except ValueError:
                next_payday = date(next_year, next_month, 28)

        # Shift forward if the inferred date is in the past
        while next_payday <= today:
            if frequency == "weekly":
                next_payday += timedelta(days=7)
            elif frequency == "biweekly":
                next_payday += timedelta(days=14)
            else:
                m = next_payday.month + 1 if next_payday.month < 12 else 1
                y = next_payday.year if next_payday.month < 12 else next_payday.year + 1
                try:
                    next_payday = date(y, m, next_payday.day)
                except ValueError:
                    next_payday = date(y, m, 28)

        consistency = gap_counts.most_common(1)[0][1] / len(gaps) if gaps else 0.0
        confidence = round(min(1.0, 0.5 + consistency * 0.5), 2)

        logger.info("Inferred payday: frequency=%s, next=%s, confidence=%.2f", frequency, next_payday, confidence)
        return {
            "frequency": frequency,
            "next_payday": str(next_payday),
            "confidence": confidence,
        }
