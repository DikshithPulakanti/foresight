"""Plaid MCP server — banking data access for Foresight AI agents.

This server wraps Plaid's API behind the MCP tool interface so that any
LangGraph agent can fetch transactions, balances, recurring charges, and
spending analytics through a single ``call_tool`` invocation without knowing
anything about Plaid's SDK or authentication model.

Sandbox testing
---------------
When ``env="sandbox"`` a hard-coded sandbox access token is used so the
server can be exercised without real bank credentials.

Caching
-------
An optional ``RedisClient`` can be injected at construction time.  When
present, every Plaid response is cached with a tool-specific TTL to avoid
redundant API round-trips (and Plaid rate-limits).
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import plaid
from plaid.api import plaid_api
from plaid.model.accounts_balance_get_request import AccountsBalanceGetRequest
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.transactions_recurring_get_request import TransactionsRecurringGetRequest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseMCPServer

logger = logging.getLogger(__name__)

_SANDBOX_ACCESS_TOKEN = "access-sandbox-de3ce8ef-33f8-452c-a7d4-05b18d3a97b8"

_ENV_MAP: dict[str, plaid.Environment] = {
    "sandbox": plaid.Environment.Sandbox,
    "development": plaid.Environment.Development,
    "production": plaid.Environment.Production,
}


def _mask_uid(user_id: str) -> str:
    """Return a privacy-safe representation of a user id (last 4 chars)."""
    return f"***{user_id[-4:]}" if len(user_id) > 4 else "****"


class PlaidMCPServer(BaseMCPServer):
    """MCP server that exposes Plaid banking tools to AI agents.

    Tools registered by this server:

    1. **get_transactions** — recent bank transactions
    2. **get_account_balances** — current balances across linked accounts
    3. **get_recurring_transactions** — subscriptions and recurring charges
    4. **get_spending_by_category** — spending totals grouped by category
    5. **flag_unusual_transactions** — anomaly detection on spending history
    """

    def __init__(
        self,
        client_id: str,
        secret: str,
        env: str = "sandbox",
        redis_client: Any | None = None,
    ) -> None:
        super().__init__(name="plaid")

        configuration = plaid.Configuration(
            host=_ENV_MAP.get(env, plaid.Environment.Sandbox),
            api_key={
                "clientId": client_id,
                "secret": secret,
            },
        )
        api_client = plaid.ApiClient(configuration)
        self._plaid = plaid_api.PlaidApi(api_client)
        self._env = env
        self.redis = redis_client

        logger.info("PlaidMCPServer created (env=%s)", env)
        self.setup()

    def _access_token(self, _user_id: str) -> str:
        """Resolve the Plaid access token for a user.

        In sandbox mode the hard-coded token is returned.  In production
        this would look up the token stored during the Plaid Link flow.
        """
        if self._env == "sandbox":
            return _SANDBOX_ACCESS_TOKEN
        # Production: retrieve from secure storage keyed by user_id
        raise NotImplementedError("Production token lookup not yet implemented")

    # ------------------------------------------------------------------
    # Setup — register all tools
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Register the five Plaid tools with the MCP base class."""

        self.register_tool(
            name="get_transactions",
            description=(
                "Fetch recent bank transactions for a user from their "
                "linked bank account"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "days_back": {"type": "integer", "default": 30},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["user_id"],
            },
            handler=self._get_transactions_handler,
        )

        self.register_tool(
            name="get_account_balances",
            description=(
                "Get current account balances for all accounts linked "
                "by the user"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"],
            },
            handler=self._get_balances_handler,
        )

        self.register_tool(
            name="get_recurring_transactions",
            description=(
                "Identify recurring charges and subscriptions the user "
                "is paying for"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"],
            },
            handler=self._get_recurring_handler,
        )

        self.register_tool(
            name="get_spending_by_category",
            description=(
                "Get total spending grouped by category for a given "
                "time period"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format",
                    },
                },
                "required": ["user_id", "start_date", "end_date"],
            },
            handler=self._spending_by_category_handler,
        )

        self.register_tool(
            name="flag_unusual_transactions",
            description=(
                "Detect transactions that are unusual compared to the "
                "user's normal spending patterns"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "lookback_days": {"type": "integer", "default": 90},
                    "sensitivity": {
                        "type": "number",
                        "default": 0.7,
                        "description": "0.0–1.0; higher = more sensitive",
                    },
                },
                "required": ["user_id"],
            },
            handler=self._flag_unusual_handler,
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    async def _get_cached(self, key: str) -> list[dict[str, Any]] | None:
        """Return cached JSON list or ``None`` if unavailable."""
        if self.redis is None:
            return None
        raw = await self.redis.get(key)
        if raw is None:
            return None
        logger.debug("Cache HIT: %s", key)
        return json.loads(raw)

    async def _set_cached(
        self,
        key: str,
        data: list[dict[str, Any]] | dict[str, Any],
        ttl: int,
    ) -> None:
        """Store JSON data in Redis with a TTL."""
        if self.redis is None:
            return
        await self.redis.set(key, json.dumps(data, default=str), ttl=ttl)
        logger.debug("Cache SET: %s (ttl=%ds)", key, ttl)

    # ------------------------------------------------------------------
    # Tool 1: get_transactions
    # ------------------------------------------------------------------

    async def _get_transactions_handler(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Fetch recent transactions from Plaid."""
        user_id: str = params["user_id"]
        days_back: int = params.get("days_back", 30)
        limit: int = params.get("limit", 100)

        cache_key = f"plaid:txns:{user_id}"
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        logger.info("Plaid get_transactions for user %s (days=%d)", _mask_uid(user_id), days_back)
        access_token = self._access_token(user_id)
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        try:
            request = TransactionsGetRequest(
                access_token=access_token,
                start_date=start_date,
                end_date=end_date,
                options=TransactionsGetRequestOptions(count=limit),
            )
            response = self._plaid.transactions_get(request)
        except plaid.ApiException as exc:
            raise RuntimeError(f"Plaid API error: {exc.body}") from exc

        transactions = [
            {
                "id": tx.transaction_id,
                "amount": float(tx.amount),
                "date": str(tx.date),
                "merchant_name": tx.merchant_name or tx.name,
                "category": tx.category[0] if tx.category else "Uncategorized",
                "subcategory": tx.category[1] if tx.category and len(tx.category) > 1 else None,
                "pending": tx.pending,
                "account_id": tx.account_id,
                "payment_channel": tx.payment_channel,
            }
            for tx in response.transactions
        ]

        await self._set_cached(cache_key, transactions, ttl=900)
        return transactions

    # ------------------------------------------------------------------
    # Tool 2: get_account_balances
    # ------------------------------------------------------------------

    async def _get_balances_handler(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Fetch current account balances from Plaid."""
        user_id: str = params["user_id"]

        cache_key = f"plaid:balances:{user_id}"
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        logger.info("Plaid get_balances for user %s", _mask_uid(user_id))
        access_token = self._access_token(user_id)

        try:
            request = AccountsBalanceGetRequest(access_token=access_token)
            response = self._plaid.accounts_balance_get(request)
        except plaid.ApiException as exc:
            raise RuntimeError(f"Plaid API error: {exc.body}") from exc

        accounts = [
            {
                "account_id": acct.account_id,
                "name": acct.name,
                "type": str(acct.type),
                "subtype": str(acct.subtype) if acct.subtype else None,
                "balance_current": float(acct.balances.current) if acct.balances.current is not None else None,
                "balance_available": float(acct.balances.available) if acct.balances.available is not None else None,
                "currency_code": acct.balances.iso_currency_code,
            }
            for acct in response.accounts
        ]

        await self._set_cached(cache_key, accounts, ttl=300)
        return accounts

    # ------------------------------------------------------------------
    # Tool 3: get_recurring_transactions
    # ------------------------------------------------------------------

    async def _get_recurring_handler(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify recurring charges and subscriptions."""
        user_id: str = params["user_id"]

        cache_key = f"plaid:recurring:{user_id}"
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        logger.info("Plaid get_recurring for user %s", _mask_uid(user_id))
        access_token = self._access_token(user_id)

        try:
            request = TransactionsRecurringGetRequest(
                access_token=access_token,
                account_ids=[],
            )
            response = self._plaid.transactions_recurring_get(request)
        except plaid.ApiException as exc:
            raise RuntimeError(f"Plaid API error: {exc.body}") from exc

        recurring: list[dict[str, Any]] = []
        for stream in (response.outflow_streams or []):
            recurring.append(
                {
                    "merchant_name": stream.merchant_name or stream.description,
                    "amount": float(stream.average_amount.amount) if stream.average_amount else None,
                    "frequency": str(stream.frequency) if stream.frequency else "unknown",
                    "last_date": str(stream.last_date) if stream.last_date else None,
                    "next_expected_date": (
                        str(stream.predicted_next_date) if hasattr(stream, "predicted_next_date") and stream.predicted_next_date else None
                    ),
                    "category": stream.category[0] if stream.category else "Uncategorized",
                    "is_active": stream.is_active if hasattr(stream, "is_active") else True,
                }
            )

        await self._set_cached(cache_key, recurring, ttl=3600)
        return recurring

    # ------------------------------------------------------------------
    # Tool 4: get_spending_by_category
    # ------------------------------------------------------------------

    async def _spending_by_category_handler(self, params: dict[str, Any]) -> dict[str, float]:
        """Aggregate spending by primary category over a date range."""
        user_id: str = params["user_id"]
        start_date_str: str = params["start_date"]
        end_date_str: str = params["end_date"]

        logger.info(
            "Plaid spending_by_category for user %s (%s → %s)",
            _mask_uid(user_id),
            start_date_str,
            end_date_str,
        )
        access_token = self._access_token(user_id)
        start_dt = date.fromisoformat(start_date_str)
        end_dt = date.fromisoformat(end_date_str)

        try:
            request = TransactionsGetRequest(
                access_token=access_token,
                start_date=start_dt,
                end_date=end_dt,
                options=TransactionsGetRequestOptions(count=500),
            )
            response = self._plaid.transactions_get(request)
        except plaid.ApiException as exc:
            raise RuntimeError(f"Plaid API error: {exc.body}") from exc

        totals: dict[str, float] = defaultdict(float)
        for tx in response.transactions:
            category = tx.category[0] if tx.category else "Uncategorized"
            totals[category] += abs(float(tx.amount))

        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

    # ------------------------------------------------------------------
    # Tool 5: flag_unusual_transactions
    # ------------------------------------------------------------------

    async def _flag_unusual_handler(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Detect transactions that deviate from the user's normal patterns.

        Flags two types of anomalies:
        * **Amount outliers** — transactions where the amount exceeds
          ``mean + 2 * stdev`` for that merchant.
        * **New merchants** — merchants the user has never transacted with
          before the lookback window.
        """
        user_id: str = params["user_id"]
        lookback_days: int = params.get("lookback_days", 90)
        sensitivity: float = params.get("sensitivity", 0.7)

        logger.info(
            "Plaid flag_unusual for user %s (lookback=%d, sensitivity=%.2f)",
            _mask_uid(user_id),
            lookback_days,
            sensitivity,
        )
        access_token = self._access_token(user_id)
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        try:
            request = TransactionsGetRequest(
                access_token=access_token,
                start_date=start_date,
                end_date=end_date,
                options=TransactionsGetRequestOptions(count=500),
            )
            response = self._plaid.transactions_get(request)
        except plaid.ApiException as exc:
            raise RuntimeError(f"Plaid API error: {exc.body}") from exc

        # Build per-merchant amount history
        merchant_amounts: dict[str, list[float]] = defaultdict(list)
        for tx in response.transactions:
            merchant = tx.merchant_name or tx.name or "Unknown"
            merchant_amounts[merchant].append(abs(float(tx.amount)))

        # Determine the recent window (last 7 days) for anomaly scanning
        recent_cutoff = end_date - timedelta(days=7)
        flagged: list[dict[str, Any]] = []

        for tx in response.transactions:
            tx_date = tx.date if isinstance(tx.date, date) else date.fromisoformat(str(tx.date))
            if tx_date < recent_cutoff:
                continue

            merchant = tx.merchant_name or tx.name or "Unknown"
            amount = abs(float(tx.amount))
            amounts = merchant_amounts[merchant]

            tx_summary = {
                "id": tx.transaction_id,
                "amount": amount,
                "date": str(tx.date),
                "merchant_name": merchant,
                "category": tx.category[0] if tx.category else "Uncategorized",
            }

            # New-merchant flag
            if len(amounts) == 1:
                flagged.append(
                    {
                        "transaction": tx_summary,
                        "reason": "First transaction with this merchant",
                        "confidence_score": round(sensitivity, 2),
                    }
                )
                continue

            # Amount-outlier flag
            if len(amounts) >= 2:
                mean = statistics.mean(amounts)
                stdev = statistics.stdev(amounts)
                if stdev > 0 and amount > mean + 2 * stdev:
                    score = min(1.0, (amount - mean) / (3 * stdev)) * sensitivity
                    flagged.append(
                        {
                            "transaction": tx_summary,
                            "reason": (
                                f"Amount ${amount:.2f} exceeds typical "
                                f"${mean:.2f} ± ${stdev:.2f} for {merchant}"
                            ),
                            "confidence_score": round(score, 2),
                        }
                    )

        return flagged
