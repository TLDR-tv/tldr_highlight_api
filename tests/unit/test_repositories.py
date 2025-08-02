"""Unit tests for repository layer."""

import pytest
from uuid import uuid4
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from src.domain.models.organization import Organization
from src.domain.models.user import User, UserRole
from src.infrastructure.storage.database import Base
from src.infrastructure.storage.repositories import (
    OrganizationRepository,
    UserRepository,
)
from src.infrastructure.security.password_service import PasswordService


@pytest.fixture
async def engine():
    """Create test database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def session(engine):
    """Create test session."""
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with async_session() as session:
        yield session
        await session.rollback()


class TestOrganizationRepository:
    """Test organization repository."""

    async def test_create_organization(self, session: AsyncSession):
        """Test creating organization."""
        repo = OrganizationRepository(session)

        org = Organization(
            id=uuid4(),
            name="Test Company",
            slug="test-company",
        )

        created = await repo.create(org)
        await session.commit()

        assert created.id == org.id
        assert created.name == "Test Company"
        assert created.slug == "test-company"

    async def test_get_organization_by_slug(self, session: AsyncSession):
        """Test getting organization by slug."""
        repo = OrganizationRepository(session)

        org = Organization(
            id=uuid4(),
            name="Test Company",
            slug="test-company",
        )

        await repo.create(org)
        await session.commit()

        found = await repo.get_by_slug("test-company")
        assert found is not None
        assert found.id == org.id
        assert found.name == "Test Company"

    async def test_update_organization(self, session: AsyncSession):
        """Test updating organization."""
        repo = OrganizationRepository(session)

        org = Organization(
            id=uuid4(),
            name="Test Company",
            slug="test-company",
        )

        created = await repo.create(org)
        await session.commit()

        # Update
        created.name = "Updated Company"
        created.webhook_url = "https://example.com/webhook"

        updated = await repo.update(created)
        await session.commit()

        assert updated.name == "Updated Company"
        assert updated.webhook_url == "https://example.com/webhook"


class TestUserRepository:
    """Test user repository."""

    async def test_create_user(self, session: AsyncSession):
        """Test creating user."""
        # First create organization
        org_repo = OrganizationRepository(session)
        org = Organization(
            id=uuid4(),
            name="Test Company",
            slug="test-company",
        )
        await org_repo.create(org)

        # Create user
        user_repo = UserRepository(session)
        password_service = PasswordService()

        user = User(
            id=uuid4(),
            organization_id=org.id,
            email="test@example.com",
            name="Test User",
            role=UserRole.ADMIN,
            hashed_password=password_service.hash_password("TestPass123!"),
        )

        created = await user_repo.create(user)
        await session.commit()

        assert created.id == user.id
        assert created.email == "test@example.com"
        assert created.organization_id == org.id

    async def test_get_user_by_email(self, session: AsyncSession):
        """Test getting user by email."""
        # First create organization
        org_repo = OrganizationRepository(session)
        org = Organization(
            id=uuid4(),
            name="Test Company",
            slug="test-company",
        )
        await org_repo.create(org)

        # Create user
        user_repo = UserRepository(session)
        password_service = PasswordService()

        user = User(
            id=uuid4(),
            organization_id=org.id,
            email="test@example.com",
            name="Test User",
            role=UserRole.ADMIN,
            hashed_password=password_service.hash_password("TestPass123!"),
        )

        await user_repo.create(user)
        await session.commit()

        found = await user_repo.get_by_email("test@example.com")
        assert found is not None
        assert found.id == user.id
        assert found.email == "test@example.com"

    async def test_user_authentication(self, session: AsyncSession):
        """Test user authentication flow."""
        # First create organization
        org_repo = OrganizationRepository(session)
        org = Organization(
            id=uuid4(),
            name="Test Company",
            slug="test-company",
        )
        await org_repo.create(org)

        # Create user
        user_repo = UserRepository(session)
        password_service = PasswordService()

        password = "TestPass123!"
        user = User(
            id=uuid4(),
            organization_id=org.id,
            email="test@example.com",
            name="Test User",
            role=UserRole.ADMIN,
            hashed_password=password_service.hash_password(password),
        )

        await user_repo.create(user)
        await session.commit()

        # Test authentication
        found = await user_repo.get_by_email("test@example.com")
        assert found is not None
        assert password_service.verify_password(password, found.hashed_password)
