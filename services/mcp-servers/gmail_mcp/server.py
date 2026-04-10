"""Gmail MCP server — scans a user's inbox for financial signals.

This server connects to the Gmail API and exposes tools that let AI agents
discover bills, subscription renewals, price increases, receipts, and overdue
notices buried in a user's email.

Privacy invariant
-----------------
**Email body text is never logged.**  Only metadata (subject line, sender,
date) and extracted financial data (amounts, service names, dates) appear in
log output.

Mock mode
---------
When ``mock=True`` is passed to the constructor the server skips Gmail
entirely and returns realistic fake data so that the full agent pipeline can
be exercised without real Google credentials.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from io import StringIO
from typing import Any

import anthropic
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseMCPServer

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_FINANCIAL_CATEGORIES = [
    "bill_due",
    "subscription_renewal",
    "price_increase",
    "payment_confirmation",
    "refund",
    "overdue_notice",
    "promotional",
    "other",
]

_DOLLAR_RE = re.compile(r"\$[\d,]+\.?\d*")
_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{4}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]


# ------------------------------------------------------------------
# HTML stripping helper
# ------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Minimal HTML-to-plain-text converter."""

    def __init__(self) -> None:
        super().__init__()
        self._buf = StringIO()

    def handle_data(self, data: str) -> None:
        self._buf.write(data)

    def get_text(self) -> str:
        return self._buf.getvalue()


def _strip_html(html: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


# ------------------------------------------------------------------
# Mock data for testing
# ------------------------------------------------------------------

def _mock_financial_emails() -> list[dict[str, Any]]:
    today = date.today()
    return [
        {"id": "mock-001", "subject": "Your Netflix subscription renewed", "sender": "info@netflix.com", "date": str(today - timedelta(days=1)), "category": "subscription_renewal", "amount_mentioned": "$15.49", "urgency": "low", "snippet": "Your monthly subscription has been renewed."},
        {"id": "mock-002", "subject": "Electricity bill due Jan 15", "sender": "billing@power.com", "date": str(today - timedelta(days=3)), "category": "bill_due", "amount_mentioned": "$142.30", "urgency": "high", "snippet": "Your electricity bill of $142.30 is due."},
        {"id": "mock-003", "subject": "Payment confirmation — Amazon order", "sender": "auto-confirm@amazon.com", "date": str(today - timedelta(days=5)), "category": "payment_confirmation", "amount_mentioned": "$67.99", "urgency": "low", "snippet": "Your order has been confirmed and payment processed."},
        {"id": "mock-004", "subject": "Spotify Premium price increase", "sender": "no-reply@spotify.com", "date": str(today - timedelta(days=7)), "category": "price_increase", "amount_mentioned": "$12.99", "urgency": "medium", "snippet": "Your plan will increase from $10.99 to $12.99."},
        {"id": "mock-005", "subject": "OVERDUE: Internet bill past due", "sender": "billing@isp.com", "date": str(today - timedelta(days=2)), "category": "overdue_notice", "amount_mentioned": "$89.00", "urgency": "high", "snippet": "Your internet bill of $89.00 is past due."},
    ]


def _mock_email_detail(email_id: str) -> dict[str, Any]:
    return {
        "id": email_id,
        "subject": "Mock email subject",
        "sender": "mock@example.com",
        "date": str(date.today()),
        "body_text": "This is mock email body content for testing purposes. Amount: $49.99 due by 2025-02-01.",
    }


def _mock_subscriptions() -> list[dict[str, Any]]:
    return [
        {"service_name": "Netflix", "amount": "$15.49", "renewal_date": "2025-02-01", "email_id": "mock-sub-001"},
        {"service_name": "Spotify", "amount": "$10.99", "renewal_date": "2025-01-28", "email_id": "mock-sub-002"},
        {"service_name": "Adobe Creative Cloud", "amount": "$54.99", "renewal_date": "2025-02-15", "email_id": "mock-sub-003"},
        {"service_name": "iCloud+", "amount": "$2.99", "renewal_date": "2025-01-20", "email_id": "mock-sub-004"},
    ]


def _mock_price_increases() -> list[dict[str, Any]]:
    return [
        {"service": "Spotify", "old_price": "$10.99", "new_price": "$12.99", "effective_date": "2025-03-01", "email_id": "mock-pi-001", "urgency": "medium"},
        {"service": "YouTube Premium", "old_price": "$11.99", "new_price": "$13.99", "effective_date": "2025-02-15", "email_id": "mock-pi-002", "urgency": "medium"},
    ]


# ------------------------------------------------------------------
# Gmail MCP Server
# ------------------------------------------------------------------

class GmailMCPServer(BaseMCPServer):
    """MCP server that exposes Gmail scanning tools for financial intelligence.

    Tools registered:

    1. **scan_financial_emails** — find bills, renewals, receipts in inbox
    2. **get_email_details** — fetch full content of a single email
    3. **find_subscription_emails** — discover active subscriptions
    4. **check_price_increases** — detect price-increase notifications
    """

    def __init__(
        self,
        credentials_json: dict[str, Any] | None = None,
        redis_client: Any | None = None,
        anthropic_api_key: str | None = None,
        mock: bool = False,
    ) -> None:
        super().__init__(name="gmail")

        self._mock = mock
        self.redis = redis_client
        self._claude: anthropic.AsyncAnthropic | None = None

        if anthropic_api_key:
            self._claude = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

        if mock:
            self._service = None
            logger.info("GmailMCPServer created in MOCK mode")
        else:
            creds = Credentials.from_authorized_user_info(credentials_json)  # type: ignore[arg-type]
            self._service = build("gmail", "v1", credentials=creds)
            logger.info("GmailMCPServer created (live)")

        self.setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Register the four Gmail tools."""

        self.register_tool(
            name="scan_financial_emails",
            description=(
                "Scan inbox for financial emails — bills, renewals, receipts, "
                "price changes — from the last N days"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "days_back": {"type": "integer", "default": 14},
                    "max_results": {"type": "integer", "default": 50},
                },
                "required": [],
            },
            handler=self._scan_financial_emails_handler,
        )

        self.register_tool(
            name="get_email_details",
            description="Get the full content of a specific email by ID",
            input_schema={
                "type": "object",
                "properties": {
                    "email_id": {"type": "string"},
                },
                "required": ["email_id"],
            },
            handler=self._get_email_details_handler,
        )

        self.register_tool(
            name="find_subscription_emails",
            description=(
                "Find all subscription-related emails to identify what "
                "services the user is paying for"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "months_back": {"type": "integer", "default": 3},
                },
                "required": [],
            },
            handler=self._find_subscription_emails_handler,
        )

        self.register_tool(
            name="check_price_increases",
            description=(
                "Find emails notifying the user of price increases on "
                "services they use"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "days_back": {"type": "integer", "default": 60},
                },
                "required": [],
            },
            handler=self._check_price_increases_handler,
        )

    # ------------------------------------------------------------------
    # Gmail helpers
    # ------------------------------------------------------------------

    async def _gmail_search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Run a Gmail search with exponential backoff on quota errors."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = (
                    self._service.users()  # type: ignore[union-attr]
                    .messages()
                    .list(userId="me", q=query, maxResults=max_results)
                    .execute()
                )
                return result.get("messages", [])
            except Exception as exc:
                if attempt == _MAX_RETRIES:
                    raise
                wait = 2 ** attempt
                logger.warning("Gmail API error (attempt %d/%d, retry in %ds): %s", attempt, _MAX_RETRIES, wait, exc)
                await asyncio.sleep(wait)
        return []

    async def _gmail_get_message(self, msg_id: str, fmt: str = "metadata") -> dict[str, Any]:
        """Fetch a single message with exponential backoff."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return (
                    self._service.users()  # type: ignore[union-attr]
                    .messages()
                    .get(userId="me", id=msg_id, format=fmt)
                    .execute()
                )
            except Exception as exc:
                if attempt == _MAX_RETRIES:
                    raise
                wait = 2 ** attempt
                logger.warning("Gmail API error (attempt %d/%d, retry in %ds): %s", attempt, _MAX_RETRIES, wait, exc)
                await asyncio.sleep(wait)
        return {}

    def _extract_header(self, headers: list[dict[str, str]], name: str) -> str:
        """Pull a header value by name (case-insensitive)."""
        lower = name.lower()
        for h in headers:
            if h.get("name", "").lower() == lower:
                return h.get("value", "")
        return ""

    def _decode_body(self, payload: dict[str, Any]) -> str:
        """Recursively extract and decode plain-text body from a message payload."""
        if payload.get("mimeType") == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

        if payload.get("mimeType", "").startswith("text/html"):
            data = payload.get("body", {}).get("data", "")
            if data:
                return _strip_html(base64.urlsafe_b64decode(data).decode("utf-8", errors="replace"))

        for part in payload.get("parts", []):
            text = self._decode_body(part)
            if text:
                return text
        return ""

    # ------------------------------------------------------------------
    # Claude classification helper
    # ------------------------------------------------------------------

    async def _classify_email(self, subject: str, sender: str, snippet: str) -> dict[str, Any]:
        """Use Claude Haiku to classify a financial email.

        Returns ``{"category": str, "amount_mentioned": str|None, "urgency": str}``.
        """
        if self._claude is None:
            amounts = _DOLLAR_RE.findall(snippet + " " + subject)
            return {
                "category": "other",
                "amount_mentioned": amounts[0] if amounts else None,
                "urgency": "low",
            }

        prompt = (
            f"Classify this email into exactly one category.\n"
            f"Categories: {', '.join(_FINANCIAL_CATEGORIES)}\n\n"
            f"Subject: {subject}\nFrom: {sender}\nSnippet: {snippet}\n\n"
            f"Respond ONLY with JSON: "
            f'{{"category": "...", "amount_mentioned": "$X.XX or null", "urgency": "high|medium|low"}}'
        )

        try:
            response = await self._claude.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            return json.loads(text)
        except Exception as exc:
            logger.warning("Claude classification failed: %s", exc)
            amounts = _DOLLAR_RE.findall(snippet + " " + subject)
            return {
                "category": "other",
                "amount_mentioned": amounts[0] if amounts else None,
                "urgency": "low",
            }

    async def _extract_price_change(self, subject: str, body: str, email_id: str) -> dict[str, Any] | None:
        """Use Claude Haiku to extract old/new price and service name."""
        if self._claude is None:
            return None

        prompt = (
            f"Extract price change information from this email.\n"
            f"Subject: {subject}\nBody (truncated): {body[:1000]}\n\n"
            f"Respond ONLY with JSON: "
            f'{{"service": "...", "old_price": "$X.XX", "new_price": "$X.XX", '
            f'"effective_date": "YYYY-MM-DD or null", "urgency": "high|medium|low"}}'
        )

        try:
            response = await self._claude.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(response.content[0].text.strip())
            data["email_id"] = email_id
            return data
        except Exception as exc:
            logger.warning("Claude price extraction failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    async def _get_cached(self, key: str) -> Any | None:
        if self.redis is None:
            return None
        raw = await self.redis.get(key)
        if raw is None:
            return None
        logger.debug("Cache HIT: %s", key)
        return json.loads(raw)

    async def _set_cached(self, key: str, data: Any, ttl: int) -> None:
        if self.redis is None:
            return
        await self.redis.set(key, json.dumps(data, default=str), ttl=ttl)
        logger.debug("Cache SET: %s (ttl=%ds)", key, ttl)

    # ------------------------------------------------------------------
    # Tool 1: scan_financial_emails
    # ------------------------------------------------------------------

    async def _scan_financial_emails_handler(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Scan inbox for financial emails and classify each one."""
        days_back: int = params.get("days_back", 14)
        max_results: int = params.get("max_results", 50)

        if self._mock:
            return _mock_financial_emails()

        cache_key = f"gmail:financial:{days_back}"
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        query = (
            "subject:(bill OR invoice OR receipt OR renewal OR subscription "
            "OR payment OR charged OR overdue OR statement) "
            f"newer_than:{days_back}d"
        )
        logger.info("Gmail scan_financial_emails (days=%d, max=%d)", days_back, max_results)

        message_stubs = await self._gmail_search(query, max_results)
        results: list[dict[str, Any]] = []

        for stub in message_stubs:
            msg = await self._gmail_get_message(stub["id"], fmt="metadata")
            headers = msg.get("payload", {}).get("headers", [])
            subject = self._extract_header(headers, "Subject")
            sender = self._extract_header(headers, "From")
            date_str = self._extract_header(headers, "Date")
            snippet = msg.get("snippet", "")

            classification = await self._classify_email(subject, sender, snippet)

            results.append({
                "id": msg["id"],
                "subject": subject,
                "sender": sender,
                "date": date_str,
                "category": classification.get("category", "other"),
                "amount_mentioned": classification.get("amount_mentioned"),
                "urgency": classification.get("urgency", "low"),
                "snippet": snippet,
            })
            logger.info("Classified email id=%s category=%s", msg["id"], classification.get("category"))

        await self._set_cached(cache_key, results, ttl=1800)
        return results

    # ------------------------------------------------------------------
    # Tool 2: get_email_details
    # ------------------------------------------------------------------

    async def _get_email_details_handler(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch and decode the full content of a single email."""
        email_id: str = params["email_id"]

        if self._mock:
            return _mock_email_detail(email_id)

        logger.info("Gmail get_email_details id=%s", email_id)
        msg = await self._gmail_get_message(email_id, fmt="full")
        headers = msg.get("payload", {}).get("headers", [])

        return {
            "id": msg["id"],
            "subject": self._extract_header(headers, "Subject"),
            "sender": self._extract_header(headers, "From"),
            "date": self._extract_header(headers, "Date"),
            "body_text": self._decode_body(msg.get("payload", {})),
        }

    # ------------------------------------------------------------------
    # Tool 3: find_subscription_emails
    # ------------------------------------------------------------------

    async def _find_subscription_emails_handler(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Discover active subscriptions from email history."""
        months_back: int = params.get("months_back", 3)

        if self._mock:
            return _mock_subscriptions()

        cache_key = f"gmail:subscriptions:{months_back}"
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        query = (
            "subject:(subscription OR membership OR renewal OR \"your plan\") "
            f"newer_than:{months_back * 30}d"
        )
        logger.info("Gmail find_subscriptions (months=%d)", months_back)

        message_stubs = await self._gmail_search(query, max_results=100)
        subscriptions: list[dict[str, Any]] = []
        seen_services: set[str] = set()

        for stub in message_stubs:
            msg = await self._gmail_get_message(stub["id"], fmt="full")
            headers = msg.get("payload", {}).get("headers", [])
            subject = self._extract_header(headers, "Subject")
            sender = self._extract_header(headers, "From")
            snippet = msg.get("snippet", "")
            body = self._decode_body(msg.get("payload", {}))
            combined_text = f"{subject} {snippet} {body}"

            service_name = sender.split("<")[0].strip() if "<" in sender else sender.split("@")[0]
            if service_name in seen_services:
                continue
            seen_services.add(service_name)

            amounts = _DOLLAR_RE.findall(combined_text)
            dates_found: list[str] = []
            for pattern in _DATE_PATTERNS:
                dates_found.extend(pattern.findall(combined_text))

            subscriptions.append({
                "service_name": service_name,
                "amount": amounts[0] if amounts else None,
                "renewal_date": dates_found[0] if dates_found else None,
                "email_id": msg["id"],
            })
            logger.info("Found subscription: %s", service_name)

        await self._set_cached(cache_key, subscriptions, ttl=1800)
        return subscriptions

    # ------------------------------------------------------------------
    # Tool 4: check_price_increases
    # ------------------------------------------------------------------

    async def _check_price_increases_handler(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Detect price-increase notification emails."""
        days_back: int = params.get("days_back", 60)

        if self._mock:
            return _mock_price_increases()

        cache_key = f"gmail:price_increases:{days_back}"
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        query = (
            "subject:(\"price increase\" OR \"rate change\" OR \"new price\" "
            "OR \"updated plan\" OR \"pricing change\") "
            f"newer_than:{days_back}d"
        )
        logger.info("Gmail check_price_increases (days=%d)", days_back)

        message_stubs = await self._gmail_search(query, max_results=30)
        increases: list[dict[str, Any]] = []

        for stub in message_stubs:
            msg = await self._gmail_get_message(stub["id"], fmt="full")
            headers = msg.get("payload", {}).get("headers", [])
            subject = self._extract_header(headers, "Subject")
            body = self._decode_body(msg.get("payload", {}))

            extracted = await self._extract_price_change(subject, body, msg["id"])
            if extracted is not None:
                increases.append(extracted)
            else:
                amounts = _DOLLAR_RE.findall(f"{subject} {body}")
                sender = self._extract_header(headers, "From")
                service_name = sender.split("<")[0].strip() if "<" in sender else sender.split("@")[0]
                increases.append({
                    "service": service_name,
                    "old_price": amounts[0] if len(amounts) >= 2 else None,
                    "new_price": amounts[1] if len(amounts) >= 2 else (amounts[0] if amounts else None),
                    "effective_date": None,
                    "email_id": msg["id"],
                    "urgency": "medium",
                })
            logger.info("Price increase detected in email id=%s", msg["id"])

        await self._set_cached(cache_key, increases, ttl=1800)
        return increases
