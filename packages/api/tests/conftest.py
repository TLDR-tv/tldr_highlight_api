"""Pytest configuration and fixtures for API tests."""

import asyncio
from typing import AsyncGenerator, Generator
from uuid import uuid4, UUID

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

from api.main import app
from api.dependencies import get_session, get_settings_dep
from shared.infrastructure.database.models import Base
from shared.infrastructure.config.config import Settings


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def postgres_container():
    """Create PostgreSQL container for testing."""
    container = PostgresContainer("postgres:16-alpine")
    container.start()

    yield container

    container.stop()


@pytest.fixture
def test_settings(postgres_container) -> Settings:
    """Test settings with overrides."""
    # Get PostgreSQL connection URL and convert to async
    postgres_url = postgres_container.get_connection_url()
    # Replace both possible formats
    if "postgresql+psycopg2://" in postgres_url:
        postgres_url = postgres_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    elif "postgresql://" in postgres_url:
        postgres_url = postgres_url.replace("postgresql://", "postgresql+asyncpg://")

    return Settings(
        environment="test",
        database_url=postgres_url,
        redis_url="redis://localhost:6379/15",  # Use test database
        jwt_secret_key="test_secret_key_for_testing_only",
        jwt_expiry_seconds=3600,
        cors_origins=["http://testserver"],
        gemini_api_key="test_gemini_key",
        aws_access_key_id="test_aws_key",
        aws_secret_access_key="test_aws_secret",
        s3_region="us-east-1",
        s3_bucket_name="test-bucket",
    )


@pytest.fixture
async def test_db(test_settings):
    """Create test database."""
    engine = create_async_engine(
        test_settings.database_url,
        echo=False,
        pool_pre_ping=True,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def test_session(test_db) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = async_sessionmaker(
        test_db,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session


@pytest.fixture
async def client(test_session, test_settings) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with database session override."""

    async def get_test_session():
        yield test_session

    async def get_test_settings():
        return test_settings

    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_settings_dep] = get_test_settings

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_org_id() -> UUID:
    """Sample organization ID for testing."""
    return uuid4()


@pytest.fixture
def sample_user_id() -> UUID:
    """Sample user ID for testing."""
    return uuid4()


@pytest.fixture
def auth_headers(sample_user_id: UUID, sample_org_id: UUID) -> dict:
    """Create auth headers with valid JWT token."""
    from shared.infrastructure.security.jwt_service import JWTService

    settings = Settings(
        jwt_secret_key="test_secret_key_for_testing_only",
        jwt_expiry_seconds=3600,
    )
    jwt_service = JWTService(settings)

    token = jwt_service.create_access_token(
        user_id=sample_user_id,
        organization_id=sample_org_id,
        email="test@example.com",
        role="admin",
    )

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api_key_headers():
    """Create headers with API key."""
    def _headers(api_key: str) -> dict:
        return {"X-API-Key": api_key}
    return _headers