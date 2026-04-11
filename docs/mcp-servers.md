# MCP Server Reference

Foresight uses 6 custom Model Context Protocol servers exposing 20 tools.
Each server wraps one external system behind a uniform `call_tool`
interface so that agents never import SDKs or construct API requests
directly.

Every tool accepts a `params` dict and returns a `ToolResult` with
`success`, `data`, `error`, and `execution_time_ms`.

---

## Plaid MCP

**Connects to:** Plaid API (bank account linking, transactions, balances)

**Why it's a separate server:** Banking data access requires OAuth token
management, pagination, rate limiting, and error retry logic that no agent
should own. Centralizing Plaid behind one MCP server means a single place
to handle access tokens, sandbox vs. production switching, and Plaid
webhook ingestion.

| Tool | Description |
|------|-------------|
| `get_transactions` | Fetch recent bank transactions for a user from their linked accounts |
| `get_account_balances` | Get current balances for all linked accounts |
| `get_recurring_transactions` | Identify recurring charges and subscription patterns |
| `get_spending_by_category` | Total spending grouped by category for a time period |
| `flag_unusual_transactions` | Detect transactions unusual vs. the user's normal patterns |

---

## Gmail MCP

**Connects to:** Google Gmail API (inbox scanning, email content)

**Why it's a separate server:** Gmail requires OAuth2 with refresh tokens,
scoped permissions, and careful pagination across potentially thousands of
emails. The server handles credential management, token refresh, and mock
mode for development — none of which agents should be aware of.

| Tool | Description |
|------|-------------|
| `scan_financial_emails` | Scan inbox for bills, renewals, receipts, and price changes from the last N days |
| `get_email_details` | Get the full content of a specific email by ID |
| `find_subscription_emails` | Find all subscription-related emails to identify paid services |
| `check_price_increases` | Find emails notifying the user of price increases |

---

## Calendar MCP

**Connects to:** Google Calendar API (events, reminders)

**Why it's a separate server:** Calendar integration requires the same
OAuth2 flow as Gmail but with different scopes and a distinct API surface.
The server normalizes calendar events into a financial context —
extracting cost estimates from event descriptions and inferring payday
schedules from recurring patterns.

| Tool | Description |
|------|-------------|
| `get_upcoming_financial_events` | Get events in the next N days that are financially relevant — bills, payday, rent, travel |
| `add_financial_reminder` | Add a calendar reminder for an upcoming bill or renewal |
| `get_payday_schedule` | Infer the user's payday schedule from calendar patterns |

---

## Graph MCP

**Connects to:** Neo4j knowledge graph (long-term financial memory)

**Why it's a separate server:** The knowledge graph is the single source
of truth for cross-agent data — spending patterns, subscription chains,
alert history, and goal progress. Centralizing all graph writes through
one MCP server prevents conflicting Cypher queries and ensures every write
uses parameterized queries (never string interpolation) to avoid injection.

| Tool | Description |
|------|-------------|
| `get_spending_patterns` | Historical spending patterns — top merchants, categories, trends |
| `find_subscription_graph` | All detected subscription patterns from graph relationships |
| `create_alert_node` | Create a financial alert node and mark it as pending |
| `get_cashflow_data` | Historical income and expense data for cashflow forecasting |
| `update_goal_progress` | Update the current progress on a savings goal |

---

## Vision MCP

**Connects to:** Anthropic Claude Vision API (image understanding)

**Why it's a separate server:** Vision requests require base64 encoding,
EXIF stripping for privacy, prompt engineering specific to financial
documents, and structured JSON extraction from free-form images. The
server encapsulates all of this behind three clean tools, and handles
the Claude API key, model selection, and token budgeting.

| Tool | Description |
|------|-------------|
| `scan_receipt` | Extract structured data from a receipt photo — merchant, items, total, date, payment method |
| `analyze_bill` | Analyze a bill image — utility, medical, insurance — and flag anomalies |
| `extract_document_info` | Extract financial information from any document — lease, policy, contract |

---

## Voice MCP

**Connects to:** OpenAI Whisper (local STT) + gTTS (text-to-speech)

**Why it's a separate server:** Voice processing involves audio codec
handling, Whisper model loading (which should happen once at startup, not
per-request), and TTS synthesis. The server manages the Whisper model
lifecycle and provides a unified pipeline tool that chains transcription
and intent classification in a single call.

| Tool | Description |
|------|-------------|
| `transcribe_audio` | Convert spoken audio to text using Whisper |
| `synthesize_speech` | Convert text to speech audio for voice responses |
| `process_voice_query` | Full pipeline: transcribe audio, classify intent, return response text |
