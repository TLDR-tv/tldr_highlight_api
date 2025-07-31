"""Database connection and session management.

This module provides both async and sync database session management using SQLAlchemy 2.0
with proper connection pooling and transaction handling, following Pythonic patterns.
"""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional

from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)
from sqlalchemy.orm import Session, sessionmaker

from src.infrastructure.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions using singleton pattern.

    This class encapsulates database connectivity as an infrastructure concern,
    providing both async and sync interfaces for different use cases.
    """

    def __init__(self):
        self._async_engine: Optional[AsyncEngine] = None
        self._async_sessionmaker: Optional[async_sessionmaker] = None
        self._sync_engine: Optional[Engine] = None
        self._sync_sessionmaker: Optional[sessionmaker] = None

    @property
    def async_engine(self) -> AsyncEngine:
        """Get or create async database engine."""
        if self._async_engine is None:
            self._async_engine = create_async_engine(
                str(settings.database_url),
                echo=settings.database_echo,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
                future=True,  # Use 2.0 style
            )
        return self._async_engine

    @property
    def async_sessionmaker(self) -> async_sessionmaker:
        """Get or create async session factory."""
        if self._async_sessionmaker is None:
            self._async_sessionmaker = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
        return self._async_sessionmaker

    @property
    def sync_engine(self) -> Engine:
        """Get or create sync database engine."""
        if self._sync_engine is None:
            # Convert async postgresql URL to sync
            sync_url = str(settings.database_url).replace(
                "postgresql+asyncpg://", "postgresql://"
            )

            self._sync_engine = create_engine(
                sync_url,
                echo=settings.database_echo,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        return self._sync_engine

    @property
    def sync_sessionmaker(self) -> sessionmaker:
        """Get or create sync session factory."""
        if self._sync_sessionmaker is None:
            self._sync_sessionmaker = sessionmaker(
                bind=self.sync_engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )
        return self._sync_sessionmaker

    async def init_db(self) -> None:
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

        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables initialized")

    async def close_async(self) -> None:
        """Close async database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
            self._async_engine = None
            self._async_sessionmaker = None
            logger.info("Async database connections closed")

    def close_sync(self) -> None:
        """Close sync database connections."""
        if self._sync_engine:
            self._sync_engine.dispose()
            self._sync_engine = None
            self._sync_sessionmaker = None
            logger.info("Sync database connections closed")


# Global database manager instance
_db_manager = DatabaseManager()


async def get_async_session() -> AsyncSession:
    """Get an async database session.

    Returns:
        AsyncSession: Database session for async operations
    """
    return _db_manager.async_sessionmaker()


def get_sync_session() -> Session:
    """Get a sync database session.

    Returns:
        Session: Database session for sync operations
    """
    return _db_manager.sync_sessionmaker()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get database session.

    This follows FastAPI's dependency injection pattern while
    maintaining clean separation of infrastructure concerns.

    Yields:
        AsyncSession: Database session for the request

    Example:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with _db_manager.async_sessionmaker() as session:
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

    This provides a Pythonic way to manage database sessions
    in background tasks, scripts, or tests.

    Yields:
        AsyncSession: Database session

    Example:
        async with get_db_context() as db:
            result = await db.execute(select(User))
            users = result.scalars().all()
    """
    async with _db_manager.async_sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for synchronous database session.

    This is used for Celery tasks and other synchronous contexts,
    providing the same transaction management as the async version.

    Yields:
        Session: Database session

    Example:
        with get_db_session() as db:
            user = db.query(User).first()
    """
    session = _db_manager.sync_sessionmaker()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Convenience functions for application lifecycle
async def init_db() -> None:
    """Initialize database tables (application startup)."""
    await _db_manager.init_db()


async def close_db() -> None:
    """Close database connections (application shutdown)."""
    await _db_manager.close_async()
    _db_manager.close_sync()
