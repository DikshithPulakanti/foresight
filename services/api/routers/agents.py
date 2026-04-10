"""AI agent orchestration endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/status")
async def agent_status() -> dict[str, str]:
    """Placeholder — will return agent health and status."""
    return {"message": "agents router"}
