"""Health-check endpoint."""

from fastapi import APIRouter

from config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return basic liveness information."""
    settings = get_settings()
    return {"status": "ok", "version": "0.1.0", "env": settings.APP_ENV}
