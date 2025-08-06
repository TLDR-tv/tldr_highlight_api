"""Mock-based unit tests for organization repository to increase coverage."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infrastructure.storage.repositories.organization import OrganizationRepository
from shared.domain.models.organization import Organization
from shared.infrastructure.database.models import OrganizationModel


class TestOrganizationRepositoryMock:
    """Mock-based tests for OrganizationRepository to increase coverage."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_session):
        """Create repository instance."""
        return OrganizationRepository(mock_session)

    @pytest.fixture
    def sample_organization(self):
        """Create sample organization."""
        return Organization(
            id=uuid4(),
            name="Test Organization",
            slug="test-org",
            is_active=True,
            total_streams_processed=5,
            total_highlights_generated=10,
            total_processing_seconds=120.0,
            webhook_url="https://example.com/webhook"
        )

    @pytest.fixture
    def sample_organization_model(self):
        """Create sample organization model."""
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
        model.rubric_name = "general"
        
        return model

    @pytest.mark.asyncio
    async def test_add_organization(self, repository, sample_organization, mock_session):
        """Test adding an organization."""
        # Mock the session operations
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        
        result = await repository.add(sample_organization)
        
        # Should have called session.add, commit, and refresh
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        
        # Result should be the same organization
        assert result == sample_organization

    @pytest.mark.asyncio
    async def test_get_organization(self, repository, sample_organization_model, mock_session):
        """Test getting an organization by ID."""
        org_id = uuid4()
        
        # Mock the query result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=sample_organization_model)
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await repository.get(org_id)
        
        # Should have executed query
        mock_session.execute.assert_called_once()
        mock_result.scalar_one_or_none.assert_called_once()
        
        # Should return converted organization
        assert result is not None
        assert isinstance(result, Organization)

    @pytest.mark.asyncio
    async def test_get_organization_not_found(self, repository, mock_session):
        """Test getting an organization that doesn't exist."""
        org_id = uuid4()
        
        # Mock empty result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await repository.get(org_id)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_update_organization(self, repository, sample_organization, mock_session):
        """Test updating an organization."""
        # Mock the session operations
        mock_session.merge = AsyncMock(return_value=sample_organization)
        mock_session.commit = AsyncMock()
        
        result = await repository.update(sample_organization)
        
        # Should have called merge and commit
        mock_session.merge.assert_called_once()
        mock_session.commit.assert_called_once()
        
        assert result == sample_organization

    @pytest.mark.asyncio
    async def test_delete_organization(self, repository, mock_session):
        """Test deleting an organization."""
        org_id = uuid4()
        
        # Mock finding the organization
        mock_org_model = MagicMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=mock_org_model)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.delete = MagicMock()
        mock_session.commit = AsyncMock()
        
        await repository.delete(org_id)
        
        # Should have found, deleted, and committed
        mock_session.execute.assert_called()
        mock_session.delete.assert_called_once_with(mock_org_model)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_organization_not_found(self, repository, mock_session):
        """Test deleting an organization that doesn't exist."""
        org_id = uuid4()
        
        # Mock empty result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.delete = MagicMock()
        mock_session.commit = AsyncMock()
        
        await repository.delete(org_id)
        
        # Should not have called delete or commit when organization not found
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_by_slug(self, repository, sample_organization_model, mock_session):
        """Test getting organization by slug."""
        slug = "test-org"
        
        # Mock the query result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=sample_organization_model)
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await repository.get_by_slug(slug)
        
        # Should have executed query
        mock_session.execute.assert_called_once()
        assert result is not None
        assert isinstance(result, Organization)

    @pytest.mark.asyncio  
    async def test_list_organizations(self, repository, mock_session):
        """Test listing organizations with filters."""
        # Mock the query result with organizations
        mock_models = [MagicMock() for _ in range(2)]
        for i, model in enumerate(mock_models):
            model.id = uuid4()
            model.name = f"Organization {i}"
            model.slug = f"org-{i}"
            model.is_active = True
            model.total_streams_processed = i * 5
            model.total_highlights_generated = i * 10
            model.total_processing_seconds = float(i * 60)
            model.webhook_url = f"https://example{i}.com/webhook"
            model.webhook_secret = f"secret{i}"
            model.rubric_name = "general"
        
        mock_result = AsyncMock()
        mock_result.scalars = AsyncMock()
        mock_result.scalars().all = AsyncMock(return_value=mock_models)
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        # Test with filters
        result = await repository.list(is_active=True)
        
        # Should have executed query and returned converted organizations
        mock_session.execute.assert_called_once()
        assert len(result) == 2
        assert all(isinstance(org, Organization) for org in result)

    @pytest.mark.asyncio
    async def test_list_organizations_no_filters(self, repository, mock_session):
        """Test listing organizations without filters."""
        mock_models = []
        mock_result = AsyncMock()
        mock_result.scalars = AsyncMock()
        mock_result.scalars().all = AsyncMock(return_value=mock_models)
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await repository.list()
        
        mock_session.execute.assert_called_once()
        assert len(result) == 0

    def test_to_entity_conversion(self, repository, sample_organization_model):
        """Test converting model to entity."""
        # Set up the model's wake_word_configs attribute to simulate loaded relationship
        sample_organization_model.__dict__["wake_word_configs"] = []
        
        result = repository._to_entity(sample_organization_model)
        
        assert isinstance(result, Organization)
        assert result.id == sample_organization_model.id
        assert result.name == sample_organization_model.name
        assert result.slug == sample_organization_model.slug
        assert result.is_active == sample_organization_model.is_active
        assert result.webhook_url == sample_organization_model.webhook_url

    def test_to_model_conversion(self, repository, sample_organization):
        """Test converting entity to model."""
        result = repository._to_model(sample_organization)
        
        assert isinstance(result, OrganizationModel)
        assert result.id == sample_organization.id
        assert result.name == sample_organization.name
        assert result.slug == sample_organization.slug
        assert result.is_active == sample_organization.is_active
        assert result.webhook_url == sample_organization.webhook_url