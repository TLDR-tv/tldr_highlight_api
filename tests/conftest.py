"""Pytest configuration and fixtures."""

import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
from uuid import uuid4, UUID

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

from src.api.main import app
from src.api.dependencies import get_session, get_settings_dep
from src.infrastructure.storage.database import Base
from src.infrastructure.config import Settings


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
        async_postgres_url = postgres_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    else:
        async_postgres_url = postgres_url.replace("postgresql://", "postgresql+asyncpg://")
    
    return Settings(
        environment="test",
        database_url=async_postgres_url,
        redis_url="redis://localhost:6379/15",  # Use a different Redis DB for tests
        jwt_secret_key="test-secret-key-for-testing-only",
        jwt_algorithm="HS256",
        jwt_expiry_seconds=3600,
        s3_region="us-east-1",
        s3_bucket_name="test-bucket",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        gemini_api_key="test-gemini-key",
        log_level="DEBUG",
    )


@pytest_asyncio.fixture
async def test_db(test_settings):
    """Create test database with tables."""
    # Create async engine with PostgreSQL
    engine = create_async_engine(
        test_settings.database_url,
        echo=False,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session factory
    async_session_maker = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )
    
    yield async_session_maker
    
    # Clean up - drop all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_db) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async with test_db() as session:
        yield session
        await session.rollback()


# Alias for compatibility with existing tests
@pytest_asyncio.fixture
async def test_session(db_session) -> AsyncGenerator[AsyncSession, None]:
    """Alias for db_session for backward compatibility."""
    yield db_session


@pytest_asyncio.fixture
async def client(test_db, test_settings) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with database override."""
    # Override dependencies
    async def override_get_session():
        async with test_db() as session:
            yield session
    
    def override_get_settings():
        return test_settings
    
    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_settings_dep] = override_get_settings
    
    # Create client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    
    # Clean up
    app.dependency_overrides.clear()


# Alias for compatibility with existing tests
@pytest_asyncio.fixture
async def async_client(client) -> AsyncGenerator[AsyncClient, None]:
    """Alias for client for backward compatibility."""
    yield client


@pytest.fixture
def auth_headers():
    """Helper to create authorization headers."""
    def _auth_headers(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}
    return _auth_headers


@pytest.fixture
def api_key_headers():
    """Helper to create API key headers."""
    def _api_key_headers(api_key: str) -> dict[str, str]:
        return {"X-API-Key": api_key}
    return _api_key_headers


@pytest_asyncio.fixture
async def create_auth_token(client: AsyncClient, db_session: AsyncSession):
    """Helper to create authenticated user and return token."""
    from src.infrastructure.storage.repositories import OrganizationRepository, UserRepository
    from tests.factories import create_test_organization, create_test_user
    
    async def _create_auth_token(
        email: str = "test@example.com",
        password: str = "TestPass123!",
        role: str = "member",
        organization_name: str = "Test Org",
    ) -> tuple[str, UUID, UUID]:
        """Create user, login, and return (token, user_id, org_id)."""
        # Create organization and user
        org = create_test_organization(name=organization_name)
        user, pwd = create_test_user(
            organization_id=org.id,
            email=email,
            password=password,
            role=role,
        )
        
        org_repo = OrganizationRepository(db_session)
        user_repo = UserRepository(db_session)
        await org_repo.create(org)
        await user_repo.create(user)
        await db_session.commit()
        
        # Login
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )
        token = response.json()["access_token"]
        
        return token, user.id, org.id
    
    return _create_auth_token