# Architecture Decisions

This document explains the key technical choices behind Foresight and why
each one exists.

## Why MCP servers instead of direct function calls

The Model Context Protocol is Anthropic's open standard for how AI models
communicate with external tools. Every external service in Foresight —
Plaid, Gmail, Google Calendar, Neo4j, Claude Vision, Whisper — is wrapped
in its own MCP server rather than called directly from agent code.

This enforces strict separation of concerns. An agent never imports the
Plaid SDK or constructs a Gmail API request. It calls
`call_tool("plaid", "get_transactions", {"days": 30})` and receives a
uniform `ToolResult` envelope. If we swap Plaid for a different banking
provider tomorrow, only one MCP server changes — zero agents need to be
touched. The tool discovery protocol (`list_tools`) also means Claude can
autonomously discover what capabilities are available without hard-coded
knowledge, which is essential as the system grows beyond 20 tools.

## Why LangGraph instead of plain Python

LangGraph gives each agent a compiled state graph with typed state, node
functions, and edges that can branch, loop, or run in parallel. This
matters for two reasons.

First, several agents have non-trivial control flow. The Advisor fans out
to four sub-agents in parallel via `asyncio.gather`, merges their outputs,
and then runs a sequential write-script → synthesize-audio pipeline. The
Cashflow Prophet runs a 60-day forward simulation that conditionally
triggers risk alerts. Expressing these as explicit graph nodes makes the
logic auditable and testable — each node is a pure function from state to
state delta.

Second, LangGraph provides built-in checkpointing and replay. When a node
fails (e.g. Plaid rate limit), the graph can resume from the last
checkpoint instead of restarting the entire pipeline. This is critical for
the Advisor agent, which takes 30+ seconds to complete all four sub-agent
calls.

## Why Neo4j instead of just PostgreSQL

PostgreSQL stores transactional data: user accounts, agent run history,
and credentials. Neo4j stores the knowledge graph: merchants, spending
patterns, subscription chains, and alert relationships.

The distinction matters because the most valuable analyses require graph
traversal. "Find all merchants where the user's spending increased by more
than 20% this month compared to the three-month average" is a single
Cypher query against a `(:User)-[:SPENT_AT]->(:Merchant)` graph. In SQL,
this requires multiple self-joins, CTEs, and window functions. Pattern
detection — like finding that a Netflix charge always appears two days
after a Spotify charge — is a native graph path query but an awkward
procedural loop in relational SQL.

Neo4j also enables the Graph Data Science library for community detection
(clustering related merchants) and PageRank (identifying the most
impactful spending categories), which feed into the financial health score.

## Why hybrid on-device / cloud architecture

Financial data is uniquely sensitive. Foresight processes bank
transactions, Gmail contents, and calendar events — data that users
reasonably expect to stay private.

The architecture splits accordingly. Voice transcription runs locally via
Whisper (no audio leaves the device). Receipt scanning sends images to
Claude Vision but strips EXIF metadata first. The knowledge graph runs in
a private VPC with no public endpoint. Only aggregated, anonymized
analytics ever leave the data layer.

On the cloud side, ECS Fargate runs the API and agents in private subnets
behind a NAT gateway. RDS Aurora and ElastiCache live in the same private
subnets with security groups that only accept traffic from the ECS task
security group. No database has a public IP.

## The agent communication pattern

Every agent inherits from `BaseAgent` and communicates with the outside
world exclusively through `self.call_tool(server_name, tool_name, params)`.
This method routes through the global `MCPRegistry` singleton, which
dispatches to the correct MCP server. The result is always a `ToolResult`
with `success`, `data`, `error`, and `execution_time_ms`.

This means agents are completely decoupled from implementation details.
The Transaction Monitor does not know whether `plaid_mcp` talks to a real
Plaid sandbox or a mock server — it just asks for transactions and gets
them. This pattern makes the entire system testable: swap every MCP server
for a mock, and 12 agents run in complete isolation with deterministic
outputs.
