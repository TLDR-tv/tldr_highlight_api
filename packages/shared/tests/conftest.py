"""Shared test configuration for pytest."""

import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from shared.domain.models.organization import Organization
from shared.domain.models.user import User, UserRole
from shared.domain.models.api_key import APIKey, APIScopes
from shared.domain.models.highlight import Highlight
from shared.domain.models.wake_word import WakeWord
from shared.infrastructure.database.database import Database


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_database():
    """Create test database."""
    # Use SQLite in-memory database for tests  
    from contextlib import asynccontextmanager
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from shared.infrastructure.database.database import Base
    
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    class TestDatabase:
        def __init__(self, engine):
            self.engine = engine
            self.async_session = async_sessionmaker(
                engine, expire_on_commit=False
            )
            
        @asynccontextmanager
        async def session(self):
            async with self.async_session() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise
                finally:
                    await session.close()
                    
        async def close(self):
            await self.engine.dispose()
    
    db = TestDatabase(engine)
    yield db
    await db.close()


@pytest.fixture
async def db_session(test_database: Database) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for tests."""
    async with test_database.session() as session:
        yield session


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.add = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def sample_organization():
    """Create a sample organization."""
    return Organization(
        id=uuid4(),
        name="Test Organization",
        slug="test-org",
        is_active=True,
        webhook_url="https://example.com/webhook",
        webhook_secret="secret123"
    )


@pytest.fixture
def sample_user(sample_organization):
    """Create a sample user."""
    return User(
        id=uuid4(),
        organization_id=sample_organization.id,
        email="test@example.com",
        hashed_password="hashed_password",
        name="Test User",
        role=UserRole.MEMBER,
        is_active=True
    )


@pytest.fixture
def sample_api_key(sample_organization):
    """Create a sample API key."""
    return APIKey(
        id=uuid4(),
        organization_id=sample_organization.id,
        name="Test API Key",
        key_hash="hashed_key_value",
        scopes={APIScopes.STREAMS_READ, APIScopes.STREAMS_WRITE},
        is_active=True
    )


@pytest.fixture
def sample_highlight(sample_organization):
    """Create a sample highlight."""
    return Highlight(
        id=uuid4(),
        stream_id=uuid4(),
        organization_id=sample_organization.id,
        start_time=10.0,
        end_time=25.0,
        title="Test Highlight",
        overall_score=8.5,
        clip_path="/path/to/clip.mp4"
    )


@pytest.fixture
def sample_wake_word(sample_organization):
    """Create a sample wake word."""
    return WakeWord(
        id=uuid4(),
        organization_id=sample_organization.id,
        phrase="hey assistant",
        case_sensitive=False,
        max_edit_distance=2,
        similarity_threshold=0.8,
        pre_roll_seconds=10,
        post_roll_seconds=30,
        is_active=True
    )