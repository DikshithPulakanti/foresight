"""Knowledge-graph query endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/stats")
async def graph_stats() -> dict[str, str]:
    """Placeholder — will return graph statistics."""
    return {"message": "graph router"}
