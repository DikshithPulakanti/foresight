"""Foresight API — Proactive AI financial operating system."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import agents, graph, health, transactions, voice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Handle startup and shutdown lifecycle events."""
    logger.info("Foresight API starting")
    yield
    logger.info("Foresight API shutting down")


app = FastAPI(
    title="Foresight API",
    version="0.1.0",
    description="Proactive AI financial operating system",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(transactions.router)
app.include_router(agents.router)
app.include_router(graph.router)
app.include_router(voice.router)


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return a structured JSON error."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )
