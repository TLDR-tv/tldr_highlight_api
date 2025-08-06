"""Simple unit tests for organization repository."""

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infrastructure.storage.repositories.organization import OrganizationRepository
from shared.domain.models.organization import Organization
from shared.infrastructure.database.models import OrganizationModel


class TestOrganizationRepositorySimple:
    """Test OrganizationRepository basic functionality."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_session):
        """Create repository instance."""
        return OrganizationRepository(mock_session)

    def test_repository_initialization(self, mock_session):
        """Test repository initialization."""
        repo = OrganizationRepository(mock_session)
        assert repo.session == mock_session

    def test_to_domain_conversion(self, repository):
        """Test converting database model to domain model."""
        # Create a mock organization model
        org_id = uuid4()
        model = OrganizationModel()
        model.id = org_id
        model.name = "Test Organization"
        model.slug = "test-org"
        model.is_active = True
        model.total_streams_processed = 5
        model.total_highlights_generated = 10
        model.total_processing_seconds = 120.0
        model.webhook_url = "https://example.com/webhook"
        model.webhook_secret = "secret123"
        
        # Convert to domain model
        domain_org = repository._to_domain(model)
        
        assert isinstance(domain_org, Organization)
        assert domain_org.id == org_id
        assert domain_org.name == "Test Organization"
        assert domain_org.slug == "test-org"
        assert domain_org.is_active is True
        assert domain_org.total_streams_processed == 5
        assert domain_org.total_highlights_generated == 10
        assert domain_org.total_processing_seconds == 120.0
        assert domain_org.webhook_url == "https://example.com/webhook"

    def test_to_model_conversion(self, repository):
        """Test converting domain model to database model."""
        # Create domain organization
        org_id = uuid4()
        domain_org = Organization(
            id=org_id,
            name="Test Organization",
            slug="test-org",
            is_active=True,
            total_streams_processed=5,
            total_highlights_generated=10,
            total_processing_seconds=120.0,
            webhook_url="https://example.com/webhook",
        )
        
        # Convert to database model
        model = repository._to_model(domain_org)
        
        assert isinstance(model, OrganizationModel)
        assert model.id == org_id
        assert model.name == "Test Organization"
        assert model.slug == "test-org"
        assert model.is_active is True
        assert model.total_streams_processed == 5
        assert model.total_highlights_generated == 10
        assert model.total_processing_seconds == 120.0
        assert model.webhook_url == "https://example.com/webhook"

    def test_to_domain_with_minimal_model(self, repository):
        """Test converting minimal database model to domain model."""
        org_id = uuid4()
        model = OrganizationModel()
        model.id = org_id
        model.name = "Minimal Org"
        model.slug = "minimal"
        model.is_active = True
        model.total_streams_processed = 0
        model.total_highlights_generated = 0
        model.total_processing_seconds = 0.0
        model.webhook_url = None
        model.webhook_secret = None
        
        domain_org = repository._to_domain(model)
        
        assert domain_org.id == org_id
        assert domain_org.name == "Minimal Org"
        assert domain_org.webhook_url is None

    def test_to_model_with_minimal_domain(self, repository):
        """Test converting minimal domain model to database model."""
        org_id = uuid4()
        domain_org = Organization(
            id=org_id,
            name="Minimal Org",
            slug="minimal",
        )
        
        model = repository._to_model(domain_org)
        
        assert model.id == org_id
        assert model.name == "Minimal Org"
        assert model.slug == "minimal"
        assert model.is_active is True  # Default value
        assert model.total_streams_processed == 0  # Default value