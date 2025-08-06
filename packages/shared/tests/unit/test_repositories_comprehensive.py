"""Comprehensive repository tests using real database session."""

import pytest
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from shared.domain.models.organization import Organization
from shared.domain.models.user import User, UserRole
from shared.domain.models.api_key import APIKey, APIScopes
from shared.domain.models.highlight import Highlight
from shared.domain.models.stream import Stream, StreamStatus, StreamType, StreamSource
from shared.infrastructure.storage.repositories import (
    OrganizationRepository,
    UserRepository,
    APIKeyRepository,
    HighlightRepository,
    StreamRepository,
)


@pytest.mark.asyncio
class TestOrganizationRepositoryComprehensive:
    """Comprehensive test suite for OrganizationRepository."""

    async def test_organization_crud_operations(self, db_session: AsyncSession):
        """Test full CRUD operations for organizations."""
        repo = OrganizationRepository(db_session)
        
        # Create
        org = Organization(
            id=uuid4(),
            name="Test Company",
            slug="test-company",
            is_active=True,
            webhook_url="https://example.com/webhook",
            webhook_secret="secret123"
        )
        
        created = await repo.create(org)
        assert created.id == org.id
        assert created.name == "Test Company"
        assert created.slug == "test-company"
        
        # Read
        found = await repo.get(org.id)
        assert found is not None
        assert found.name == "Test Company"
        
        # Update
        found.name = "Updated Company"
        updated = await repo.update(found)
        assert updated.name == "Updated Company"
        
        # List
        orgs = await repo.list()
        assert len(orgs) >= 1
        assert any(o.id == org.id for o in orgs)
        
        # Get by slug
        by_slug = await repo.get_by_slug("test-company")
        assert by_slug is not None
        assert by_slug.id == org.id
        

@pytest.mark.asyncio
class TestUserRepositoryComprehensive:
    """Comprehensive test suite for UserRepository."""
    
    async def test_user_crud_operations(self, db_session: AsyncSession):
        """Test full CRUD operations for users."""
        org_repo = OrganizationRepository(db_session)
        user_repo = UserRepository(db_session)
        
        # Create organization first
        org = Organization(
            id=uuid4(),
            name="Test Org",
            slug="test-org"
        )
        await org_repo.create(org)
        
        # Create user
        user = User(
            id=uuid4(),
            organization_id=org.id,
            email="test@example.com",
            hashed_password="hashed_pass123",
            name="Test User",
            role=UserRole.MEMBER,
            is_active=True
        )
        
        created = await user_repo.create(user)
        assert created.email == "test@example.com"
        assert created.name == "Test User"
        
        # Get by email
        by_email = await user_repo.get_by_email("test@example.com")
        assert by_email is not None
        assert by_email.id == user.id
        
        # Get by ID
        by_id = await user_repo.get(user.id)
        assert by_id is not None
        assert by_id.email == "test@example.com"
        
        # List by organization
        org_users = await user_repo.list_by_organization(org.id)
        assert len(org_users) == 1
        assert org_users[0].id == user.id
        

@pytest.mark.asyncio  
class TestAPIKeyRepositoryComprehensive:
    """Comprehensive test suite for APIKeyRepository."""
    
    async def test_api_key_crud_operations(self, db_session: AsyncSession):
        """Test full CRUD operations for API keys."""
        org_repo = OrganizationRepository(db_session)
        api_key_repo = APIKeyRepository(db_session)
        
        # Create organization first
        org = Organization(
            id=uuid4(),
            name="Test Org",
            slug="test-org"
        )
        await org_repo.create(org)
        
        # Create API key
        api_key = APIKey(
            id=uuid4(),
            organization_id=org.id,
            name="Test API Key",
            key_hash="hashed_key_value",
            prefix="tldr_test",
            scopes={APIScopes.STREAMS_READ},
            is_active=True
        )
        
        created = await api_key_repo.create(api_key)
        assert created.name == "Test API Key"
        assert created.prefix == "tldr_test"
        
        # Get by key hash
        by_hash = await api_key_repo.get_by_hash("hashed_key_value")
        assert by_hash is not None
        assert by_hash.id == api_key.id
        
        # List by organization
        org_keys = await api_key_repo.list_by_organization(org.id)
        assert len(org_keys) == 1
        assert org_keys[0].id == api_key.id
        

@pytest.mark.asyncio
class TestHighlightRepositoryComprehensive:
    """Comprehensive test suite for HighlightRepository."""
    
    async def test_highlight_crud_operations(self, db_session: AsyncSession):
        """Test full CRUD operations for highlights."""
        org_repo = OrganizationRepository(db_session)
        stream_repo = StreamRepository(db_session)
        highlight_repo = HighlightRepository(db_session)
        
        # Create organization and stream first
        org = Organization(
            id=uuid4(),
            name="Test Org",
            slug="test-org"
        )
        await org_repo.create(org)
        
        stream = Stream(
            id=uuid4(),
            organization_id=org.id,
            url="https://example.com/stream",
            name="Test Stream",
            type=StreamType.LIVESTREAM,
            status=StreamStatus.PENDING,
            stream_fingerprint="test_stream"
        )
        await stream_repo.add(stream)
        
        # Create highlight
        highlight = Highlight(
            id=uuid4(),
            stream_id=stream.id,
            organization_id=org.id,
            start_time=10.0,
            end_time=25.0,
            title="Test Highlight",
            overall_score=8.5
        )
        
        created = await highlight_repo.create(highlight)
        assert created.title == "Test Highlight"
        assert created.overall_score == 8.5
        
        # Get by ID
        by_id = await highlight_repo.get(highlight.id)
        assert by_id is not None
        assert by_id.title == "Test Highlight"
        
        # List by stream
        stream_highlights = await highlight_repo.list_by_stream(stream.id)
        assert len(stream_highlights) == 1
        assert stream_highlights[0].id == highlight.id


@pytest.mark.asyncio
class TestStreamRepositoryComprehensive:
    """Comprehensive test suite for StreamRepository."""
    
    async def test_stream_crud_operations(self, db_session: AsyncSession):
        """Test full CRUD operations for streams."""
        org_repo = OrganizationRepository(db_session)
        stream_repo = StreamRepository(db_session)
        
        # Create organization first
        org = Organization(
            id=uuid4(),
            name="Test Org", 
            slug="test-org"
        )
        await org_repo.create(org)
        
        # Create stream
        stream = Stream(
            id=uuid4(),
            organization_id=org.id,
            url="https://example.com/stream",
            name="Test Stream",
            type=StreamType.LIVESTREAM,
            status=StreamStatus.PENDING,
            stream_fingerprint="test_stream_123",
            source_type=StreamSource.DIRECT_URL
        )
        
        created = await stream_repo.add(stream)
        assert created.url == "https://example.com/stream"
        assert created.stream_fingerprint == "test_stream_123"
        
        # Get by ID
        by_id = await stream_repo.get(stream.id)
        assert by_id is not None
        assert by_id.url == "https://example.com/stream"
        
        # List by organization
        org_streams = await stream_repo.list_by_organization(org.id)
        assert len(org_streams) >= 1
        assert any(s.id == stream.id for s in org_streams)
        
        # Count by organization
        count = await stream_repo.count_by_organization(org.id)
        assert count >= 1


@pytest.mark.asyncio
class TestRepositoriesIntegration:
    """Integration tests across multiple repositories."""
    
    async def test_multi_repository_scenario(self, db_session: AsyncSession):
        """Test a complete scenario using multiple repositories."""
        # Setup repositories
        org_repo = OrganizationRepository(db_session)
        user_repo = UserRepository(db_session)
        api_key_repo = APIKeyRepository(db_session)
        stream_repo = StreamRepository(db_session)
        highlight_repo = HighlightRepository(db_session)
        
        # Create organization
        org = Organization(
            id=uuid4(),
            name="Integration Test Org",
            slug="integration-test"
        )
        org = await org_repo.create(org)
        
        # Create user
        user = User(
            id=uuid4(),
            organization_id=org.id,
            email="integration@example.com",
            hashed_password="hashed",
            name="Integration User",
            role=UserRole.ADMIN
        )
        user = await user_repo.create(user)
        
        # Create API key
        api_key = APIKey(
            id=uuid4(),
            organization_id=org.id,
            name="Integration API Key",
            key_hash="integration_hash",
            scopes={APIScopes.STREAMS_READ, APIScopes.STREAMS_WRITE}
        )
        api_key = await api_key_repo.create(api_key)
        
        # Create stream  
        stream = Stream(
            id=uuid4(),
            organization_id=org.id,
            url="https://example.com/integration",
            name="Integration Stream",
            type=StreamType.LIVESTREAM,
            status=StreamStatus.PROCESSING,
            stream_fingerprint="integration_stream"
        )
        stream = await stream_repo.add(stream)
        
        # Create highlight
        highlight = Highlight(
            id=uuid4(),
            stream_id=stream.id,
            organization_id=org.id,
            start_time=5.0,
            end_time=15.0,
            title="Integration Highlight",
            overall_score=9.2
        )
        highlight = await highlight_repo.create(highlight)
        
        # Verify all entities are linked properly
        assert user.organization_id == org.id
        assert api_key.organization_id == org.id
        assert stream.organization_id == org.id
        assert highlight.organization_id == org.id
        assert highlight.stream_id == stream.id
        
        # Test cross-repository queries
        org_users = await user_repo.list_by_organization(org.id)
        org_keys = await api_key_repo.list_by_organization(org.id)
        org_streams = await stream_repo.list_by_organization(org.id)
        stream_highlights = await highlight_repo.list_by_stream(stream.id)
        
        assert len(org_users) == 1
        assert len(org_keys) == 1  
        assert len(org_streams) == 1
        assert len(stream_highlights) == 1