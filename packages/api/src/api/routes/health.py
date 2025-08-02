"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..dependencies import get_session

router = APIRouter()


@router.get("/health")
async def health_check():
    """Perform basic health check.
    
    Returns:
        Dictionary with status and service name.
    """
    return {"status": "healthy", "service": "tldr-highlight-api"}


@router.get("/health/db")
async def database_health(session: AsyncSession = Depends(get_session)):
    """Check database connectivity.
    
    Args:
        session: Database session for testing connectivity.
        
    Returns:
        Dictionary with database connection status.
    """
    try:
        # Execute simple query
        result = await session.execute(text("SELECT 1"))
        result.scalar()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}
