"""Transaction management endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.get("")
async def list_transactions() -> dict[str, str]:
    """Placeholder — will return user transactions."""
    return {"message": "transactions router"}
