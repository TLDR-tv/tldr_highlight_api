"""Database connection and session management.

This module provides both async and sync database session management using SQLAlchemy 2.0
with proper connection pooling and transaction handling.
"""

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

# Global variables for lazy initialization
engine = None
async_session = None
sync_engine = None
sync_session = None


def get_engine():
    """Get or create the database engine."""
    global engine
    if engine is None:
        from src.core.config import settings

        engine = create_async_engine(
            str(settings.database_url),
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            future=True,  # Use 2.0 style
        )
    return engine


def get_session_factory():
    """Get or create the session factory."""
    global async_session
    if async_session is None:
        async_session = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return async_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get database session.

    Yields:
        AsyncSession: Database session for the request

    Example:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database session outside of FastAPI requests.

    Yields:
        AsyncSession: Database session

    Example:
        async with get_db_context() as db:
            result = await db.execute(select(User))
            users = result.scalars().all()
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables.

    Creates all tables defined in the models if they don't exist.
    Should be called on application startup.
    """
    from src.infrastructure.persistence.models.base import Base

    # Import all models to ensure they're registered with Base
    from src.infrastructure.persistence.models import (  # noqa: F401
        APIKey,
        Batch,
        Highlight,
        Organization,
        Stream,
        UsageRecord,
        User,
        Webhook,
        WebhookEvent,
    )

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections.

    Should be called on application shutdown to properly
    dispose of the connection pool.
    """
    global engine
    if engine:
        await engine.dispose()
        engine = None


def get_sync_engine():
    """Get or create the synchronous database engine."""
    global sync_engine
    if sync_engine is None:
        from src.core.config import settings

        # Convert async postgresql URL to sync
        sync_url = str(settings.database_url).replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        
        sync_engine = create_engine(
            sync_url,
            echo=settings.database_echo,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
    return sync_engine


def get_sync_session_factory():
    """Get or create the synchronous session factory."""
    global sync_session
    if sync_session is None:
        sync_session = sessionmaker(
            bind=get_sync_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return sync_session


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for synchronous database session.
    
    This is used for Celery tasks and other synchronous contexts.
    
    Yields:
        Session: Database session
        
    Example:
        with get_db_session() as db:
            user = db.query(User).first()
    """
    session_factory = get_sync_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
