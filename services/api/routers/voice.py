"""Voice interaction endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/voice", tags=["voice"])


@router.get("/status")
async def voice_status() -> dict[str, str]:
    """Placeholder — will return voice pipeline status."""
    return {"message": "voice router"}
