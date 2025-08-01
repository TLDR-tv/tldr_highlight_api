"""Authentication endpoints."""
from fastapi import APIRouter

router = APIRouter()


@router.post("/login")
async def login():
    """User login endpoint."""
    # TODO: Implement
    return {"message": "Login endpoint not implemented"}