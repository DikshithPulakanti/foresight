"""Foresight connection diagnostics — tests all service dependencies.

Usage:
    python scripts/test_connections.py

Reads connection strings from environment variables (or .env via dotenv).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Load .env from project root if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
CHECK = f"{GREEN}\u2713{RESET}"
CROSS = f"{RED}\u2717{RESET}"

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    mark = CHECK if ok else CROSS
    status = f"connected ({detail})" if ok else f"failed: {detail}"
    print(f"  {mark} {name:<14} {status}")


# ------------------------------------------------------------------
# Neo4j
# ------------------------------------------------------------------

async def check_neo4j() -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "foresight")
    try:
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS n")
            await result.single()
        await driver.close()
        record("Neo4j", True, uri)
    except Exception as exc:
        record("Neo4j", False, str(exc))


# ------------------------------------------------------------------
# PostgreSQL
# ------------------------------------------------------------------

async def check_postgres() -> None:
    url = os.getenv("POSTGRES_URL", "postgresql://postgres:foresight123@localhost:5432/foresight")
    try:
        import asyncpg
        conn = await asyncpg.connect(url)
        await conn.fetchval("SELECT 1")
        await conn.close()
        host = url.split("@")[-1].split("/")[0] if "@" in url else url
        record("PostgreSQL", True, host)
    except Exception as exc:
        record("PostgreSQL", False, str(exc))


# ------------------------------------------------------------------
# Redis
# ------------------------------------------------------------------

async def check_redis() -> None:
    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(url, decode_responses=True)
        await r.ping()
        await r.aclose()
        record("Redis", True, url)
    except Exception as exc:
        record("Redis", False, str(exc))


# ------------------------------------------------------------------
# Qdrant
# ------------------------------------------------------------------

async def check_qdrant() -> None:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{url}/collections")
            resp.raise_for_status()
        record("Qdrant", True, url)
    except Exception as exc:
        record("Qdrant", False, str(exc))


# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------

async def check_fastapi() -> None:
    url = os.getenv("API_URL", "http://localhost:8000")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{url}/health")
            resp.raise_for_status()
            data = resp.json()
            detail = f"{url} (v{data.get('version', '?')}, {data.get('env', '?')})"
        record("FastAPI", True, detail)
    except Exception as exc:
        record("FastAPI", False, str(exc))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main() -> None:
    print(f"\n{BOLD}Foresight — Connection Diagnostics{RESET}\n")

    start = time.monotonic()

    await check_neo4j()
    await check_postgres()
    await check_redis()
    await check_qdrant()
    await check_fastapi()

    elapsed = time.monotonic() - start
    healthy = sum(1 for _, ok, _ in results if ok)
    total = len(results)

    print(f"\n{BOLD}{healthy}/{total} services healthy{RESET} ({elapsed:.1f}s)\n")

    if healthy < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
