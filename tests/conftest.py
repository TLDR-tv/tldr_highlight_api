"""Pytest configuration and fixtures."""

import pytest
import pytest_asyncio
import os
from unittest.mock import AsyncMock, MagicMock

# Set test environment variables before importing modules
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test_db"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["S3_ACCESS_KEY_ID"] = "test"
os.environ["S3_SECRET_ACCESS_KEY"] = "test"
os.environ["S3_HIGHLIGHTS_BUCKET"] = "test-highlights"
os.environ["S3_THUMBNAILS_BUCKET"] = "test-thumbnails"
os.environ["S3_TEMP_BUCKET"] = "test-temp"
os.environ["ENVIRONMENT"] = "development"


@pytest.fixture
def mock_cache():
    """Create mock Redis cache."""
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.return_value = True
    cache.delete.return_value = True
    cache.exists.return_value = False
    cache.expire.return_value = True
    cache.increment.return_value = 1

    # Mock the context manager
    cache.get_client.return_value.__aenter__ = AsyncMock()
    cache.get_client.return_value.__aexit__ = AsyncMock()

    return cache


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    session = AsyncMock()
    session.execute.return_value.scalar_one_or_none.return_value = None
    session.execute.return_value.scalars.return_value.all.return_value = []
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture(autouse=True)
def mock_database_imports(monkeypatch):
    """Mock database imports to avoid connection issues in tests."""
    mock_engine = MagicMock()
    mock_session = MagicMock()

    monkeypatch.setattr("src.core.database.engine", mock_engine)
    monkeypatch.setattr("src.core.database.async_session", mock_session)
    monkeypatch.setattr("src.core.database.get_db", lambda: AsyncMock())

    # Mock cache import
    mock_cache = AsyncMock()
    monkeypatch.setattr("src.core.cache.cache", mock_cache)
    monkeypatch.setattr("src.core.cache.rate_limiter", AsyncMock())


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    from src.core.config import Settings

    test_settings = Settings(
        database_url="postgresql+asyncpg://test:test@localhost:5432/test_db",
        redis_url="redis://localhost:6379/15",
        jwt_secret_key="test-secret-key",
        s3_access_key_id="test",
        s3_secret_access_key="test",
        s3_highlights_bucket="test-highlights",
        s3_thumbnails_bucket="test-thumbnails",
        s3_temp_bucket="test-temp",
        environment="development",
    )

    monkeypatch.setattr("src.core.config.settings", test_settings)
    return test_settings


@pytest_asyncio.fixture
async def async_client():
    """Create async test client for API testing."""
    from httpx import AsyncClient, ASGITransport
    from src.api.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def db_session(mock_db_session):
    """Create async database session for testing."""
    # Just return the mock session from the mock_db_session fixture
    yield mock_db_session
