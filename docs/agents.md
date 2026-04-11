# Agent Reference

Foresight runs 12 specialized LangGraph agents. Each agent is a compiled
state graph that communicates with external services exclusively through
MCP servers via `call_tool`. The orchestrator dispatches agent runs by
name and manages concurrency.

## Transaction Monitor

Watches every new bank transaction and fires alerts for anomalies —
duplicate charges, unusually large purchases, new merchants, and
suspicious patterns. Fetches recent transactions from **plaid_mcp**, runs
anomaly detection, persists flagged transactions in the knowledge graph
via **graph_mcp**, and generates a Claude-powered summary. Output includes
the number of transactions analyzed, flags raised, and alerts created.

## Subscription Auditor

Cross-references bank charges and inbox receipts to build a complete
picture of recurring payments. Pulls recurring transactions from
**plaid_mcp** and subscription-related emails from **gmail_mcp**, then
uses fuzzy matching to unify them into a single inventory. Flags forgotten
subscriptions, price increases, and duplicate services. Output includes a
full subscription list, total monthly/annual cost, and potential savings
opportunities.

## Bill Negotiator

Identifies overpaid bills and generates phone scripts for retention
calls. Scans spending via **plaid_mcp** and inbox via **gmail_mcp** for
negotiable categories — internet, phone, insurance, cable, gym
memberships. Uses Claude to research competitive market rates and write
step-by-step negotiation scripts with specific dollar amounts to request.
Output includes per-bill scripts and estimated monthly/annual savings.

## Cashflow Prophet

Predicts the user's bank balance 30 and 60 days into the future. Assembles
historical cashflow from **graph_mcp**, current balances from **plaid_mcp**,
and upcoming financial events from **calendar_mcp**, then rolls a
day-by-day balance projection forward. Fires shortfall alerts when the
projected balance drops below safety thresholds. Output includes daily
projections, risk level, the lowest balance point, and days until negative.

## Receipt Scanner

Processes a receipt photo and instantly categorizes the expense. Triggered
when the user taps the camera button on the mobile app. The base64 image
is sent to Claude Vision via **vision_mcp** for structured extraction
(merchant, items, total, date, payment method), then categorized and
stored in the knowledge graph via **graph_mcp**. Output includes the
parsed receipt, assigned category, and a spending insight.

## Email Monitor

Scans Gmail for financial signals on a background schedule. Pulls recent
emails via **gmail_mcp**, classifies urgency (bills due, renewals, price
increases, overdue notices), extracts action items, and creates alerts in
the knowledge graph via **graph_mcp**. High-urgency items become push
notifications. Output includes the number of emails scanned, action items
found, alerts created, and a digest summary.

## Calendar Planner

Connects upcoming calendar events to spending and budget reality. Reads
the next N days of events from **calendar_mcp**, estimates the cost of
each financially relevant event, and cross-checks against current balance
from **plaid_mcp**. Events where the money may not be available get
automatic reminders two weeks early. Output includes upcoming events with
cost estimates, at-risk count, and a Claude-written summary.

## Goal Tracker

Monitors savings goals and celebrates milestones. Checks goal progress
against the knowledge graph via **graph_mcp** and current balance via
**plaid_mcp**. Classifies each goal as on-track, behind, ahead, completed,
or overdue. Generates personalized recommendations for catching up on
behind goals. Output includes per-goal status, total saved vs. target,
overall progress percentage, and alerts for milestones.

## Alert Sentinel

The brain of Foresight's notification system, designed to run on a
15-minute cadence. Collects every pending alert produced by all other
agents from **graph_mcp**, deduplicates them, scores each on a 0-100
urgency scale, and decides which deserve a push notification vs. silent
logging. Prevents alert fatigue by suppressing redundant or low-value
notifications. Output includes dispatched notifications, logged count, and
next scheduled run.

## Voice Orchestrator

The conversational interface to Foresight. Accepts spoken audio or typed
text, transcribes via **voice_mcp** (Whisper), classifies intent with
Claude, routes to the appropriate MCP tool or sub-agent, formulates a
spoken response, and synthesizes audio via **voice_mcp** (gTTS). Handles
queries like "How much did I spend on food this month?" end-to-end with
data from **plaid_mcp**. Output includes transcript, detected intent,
spoken response text, and audio.

## Document Analyst

Reads uploaded financial documents — leases, insurance policies,
credit-card agreements, medical bills — and surfaces hidden risks. Sends
the document image to Claude Vision via **vision_mcp** for deep analysis,
identifies hidden fees, auto-renewal traps, penalty triggers, and
coverage gaps. Creates alerts in **graph_mcp** and financial reminders in
**calendar_mcp** for important dates. Output includes extracted data, risk
analysis, red flags, and key amounts.

## Advisor

The Sunday-morning weekly financial briefing. Fans out to four specialist
agents in parallel (Transaction Monitor, Subscription Auditor, Cashflow
Prophet, Goal Tracker), merges their outputs with real-time Plaid data
from **plaid_mcp**, calculates a 0-100 financial health score, writes a
~650-word script optimized for spoken delivery, and converts it to audio
via **voice_mcp**. Persists the weekly summary in **graph_mcp**. Output
includes the health score, briefing script, audio, and a summary of which
agents contributed.
