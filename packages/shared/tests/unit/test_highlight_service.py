"""Unit tests for highlight service."""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
from typing import Optional

from shared.application.services.highlight_service import HighlightService
from shared.domain.models.highlight import Highlight, DimensionScore
from shared.infrastructure.storage.repositories import HighlightRepository


class TestHighlightService:
    """Test suite for HighlightService."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock highlight repository."""
        repository = AsyncMock(spec=HighlightRepository)
        return repository

    @pytest.fixture
    def highlight_service(self, mock_repository):
        """Create a highlight service with mocked dependencies."""
        return HighlightService(highlight_repository=mock_repository)

    @pytest.fixture
    def sample_highlight(self):
        """Create a sample highlight for testing."""
        highlight = Highlight(
            id=uuid4(),
            stream_id=uuid4(),
            organization_id=uuid4(),
            start_time=0.0,
            end_time=30.0,
            title="Test Highlight",
            description="Test highlight description",
            tags=["test", "sample"],
        )
        highlight.add_dimension_score("action_intensity", 0.9, 0.95)
        highlight.add_dimension_score("educational_value", 0.8, 0.85)
        return highlight

    @pytest.mark.asyncio
    async def test_get_highlight_success(
        self, highlight_service, mock_repository, sample_highlight
    ):
        """Test successful highlight retrieval."""
        # Arrange
        highlight_id = sample_highlight.id
        org_id = sample_highlight.organization_id
        mock_repository.get.return_value = sample_highlight

        # Act
        result = await highlight_service.get_highlight(highlight_id, org_id)

        # Assert
        assert result == sample_highlight
        mock_repository.get.assert_called_once_with(highlight_id)

    @pytest.mark.asyncio
    async def test_get_highlight_not_found(
        self, highlight_service, mock_repository
    ):
        """Test highlight retrieval when highlight doesn't exist."""
        # Arrange
        highlight_id = uuid4()
        org_id = uuid4()
        mock_repository.get.return_value = None

        # Act
        result = await highlight_service.get_highlight(highlight_id, org_id)

        # Assert
        assert result is None
        mock_repository.get.assert_called_once_with(highlight_id)

    @pytest.mark.asyncio
    async def test_get_highlight_wrong_organization(
        self, highlight_service, mock_repository, sample_highlight
    ):
        """Test highlight retrieval with wrong organization ID."""
        # Arrange
        highlight_id = sample_highlight.id
        wrong_org_id = uuid4()  # Different from sample_highlight.organization_id
        mock_repository.get.return_value = sample_highlight

        # Act
        result = await highlight_service.get_highlight(highlight_id, wrong_org_id)

        # Assert
        assert result is None
        mock_repository.get.assert_called_once_with(highlight_id)

    @pytest.mark.asyncio
    async def test_list_highlights_basic(
        self, highlight_service, mock_repository
    ):
        """Test basic highlight listing."""
        # Arrange
        org_id = uuid4()
        highlights = []
        for i in range(3):
            h = Highlight(
                id=uuid4(),
                stream_id=uuid4(),
                organization_id=org_id,
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0,
                title=f"Highlight {i}",
            )
            h.add_dimension_score("action_intensity", 0.8 + i * 0.05, 0.9)
            highlights.append(h)
        mock_repository.list.return_value = highlights

        # Act
        result = await highlight_service.list_highlights(org_id)

        # Assert
        assert result["highlights"] == highlights
        assert result["total"] == 3
        assert result["limit"] == 100
        assert result["offset"] == 0
        assert result["has_more"] is False
        mock_repository.list.assert_called_once_with(
            organization_id=org_id,
            order_by="created_at",
            limit=100,
            offset=0,
        )

    @pytest.mark.asyncio
    async def test_list_highlights_with_filters(
        self, highlight_service, mock_repository
    ):
        """Test highlight listing with all filters."""
        # Arrange
        org_id = uuid4()
        stream_id = uuid4()
        highlights = []
        mock_repository.list.return_value = highlights

        # Act
        result = await highlight_service.list_highlights(
            organization_id=org_id,
            stream_id=stream_id,
            wake_word_triggered=True,
            min_score=0.7,
            order_by="score",
            limit=50,
            offset=10,
        )

        # Assert
        assert result["highlights"] == []
        assert result["total"] == 0
        assert result["limit"] == 50
        assert result["offset"] == 10
        assert result["has_more"] is False
        mock_repository.list.assert_called_once_with(
            organization_id=org_id,
            stream_id=stream_id,
            wake_word_triggered=True,
            min_score=0.7,
            order_by="score",
            limit=50,
            offset=10,
        )

    @pytest.mark.asyncio
    async def test_list_highlights_pagination(
        self, highlight_service, mock_repository
    ):
        """Test highlight listing with pagination (full page)."""
        # Arrange
        org_id = uuid4()
        limit = 10
        highlights = []
        for i in range(limit):
            h = Highlight(
                id=uuid4(),
                stream_id=uuid4(),
                organization_id=org_id,
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0,
                title=f"Highlight {i}",
            )
            h.add_dimension_score("action_intensity", 0.8, 0.9)
            highlights.append(h)
        mock_repository.list.return_value = highlights

        # Act
        result = await highlight_service.list_highlights(
            organization_id=org_id,
            limit=limit,
            offset=20,
        )

        # Assert
        assert len(result["highlights"]) == limit
        assert result["total"] == 31  # offset + limit + 1
        assert result["has_more"] is True

    @pytest.mark.asyncio
    async def test_list_highlights_empty_results(
        self, highlight_service, mock_repository
    ):
        """Test highlight listing with no results."""
        # Arrange
        org_id = uuid4()
        mock_repository.list.return_value = []

        # Act
        result = await highlight_service.list_highlights(org_id)

        # Assert
        assert result["highlights"] == []
        assert result["total"] == 0
        assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_get_stream_highlights_success(
        self, highlight_service, mock_repository
    ):
        """Test getting highlights for a specific stream."""
        # Arrange
        stream_id = uuid4()
        org_id = uuid4()
        other_org_id = uuid4()
        
        # Mix of highlights from different organizations
        h1 = Highlight(
            id=uuid4(),
            stream_id=stream_id,
            organization_id=org_id,
            start_time=0.0,
            end_time=30.0,
            title="Exciting Moment",
        )
        h1.add_dimension_score("action_intensity", 0.9, 0.95)
        
        h2 = Highlight(
            id=uuid4(),
            stream_id=stream_id,
            organization_id=other_org_id,  # Different org
            start_time=30.0,
            end_time=60.0,
            title="Funny Moment",
        )
        h2.add_dimension_score("action_intensity", 0.8, 0.85)
        
        h3 = Highlight(
            id=uuid4(),
            stream_id=stream_id,
            organization_id=org_id,
            start_time=60.0,
            end_time=90.0,
            title="Impressive Play",
        )
        h3.add_dimension_score("action_intensity", 0.95, 0.98)
        
        highlights = [h1, h2, h3]
        mock_repository.list_by_stream.return_value = highlights

        # Act
        result = await highlight_service.get_stream_highlights(stream_id, org_id)

        # Assert
        assert len(result) == 2  # Only org's highlights
        assert all(h.organization_id == org_id for h in result)
        assert result[0] == highlights[0]
        assert result[1] == highlights[2]
        mock_repository.list_by_stream.assert_called_once_with(stream_id)

    @pytest.mark.asyncio
    async def test_get_stream_highlights_empty(
        self, highlight_service, mock_repository
    ):
        """Test getting highlights for a stream with no highlights."""
        # Arrange
        stream_id = uuid4()
        org_id = uuid4()
        mock_repository.list_by_stream.return_value = []

        # Act
        result = await highlight_service.get_stream_highlights(stream_id, org_id)

        # Assert
        assert result == []
        mock_repository.list_by_stream.assert_called_once_with(stream_id)

    @pytest.mark.asyncio
    async def test_get_stream_highlights_no_org_matches(
        self, highlight_service, mock_repository
    ):
        """Test getting highlights when none match the organization."""
        # Arrange
        stream_id = uuid4()
        org_id = uuid4()
        other_org_id = uuid4()
        
        # All highlights from different organization
        h = Highlight(
            id=uuid4(),
            stream_id=stream_id,
            organization_id=other_org_id,
            start_time=0.0,
            end_time=30.0,
            title="Exciting Moment",
        )
        h.add_dimension_score("action_intensity", 0.9, 0.95)
        highlights = [h]
        mock_repository.list_by_stream.return_value = highlights

        # Act
        result = await highlight_service.get_stream_highlights(stream_id, org_id)

        # Assert
        assert result == []
        mock_repository.list_by_stream.assert_called_once_with(stream_id)

    @pytest.mark.asyncio
    async def test_list_highlights_with_none_filters(
        self, highlight_service, mock_repository
    ):
        """Test that None filters are not passed to repository."""
        # Arrange
        org_id = uuid4()
        mock_repository.list.return_value = []

        # Act
        await highlight_service.list_highlights(
            organization_id=org_id,
            stream_id=None,
            wake_word_triggered=None,
            min_score=None,
        )

        # Assert
        # Verify None values are not in the call
        call_args = mock_repository.list.call_args[1]
        assert "stream_id" not in call_args
        assert "wake_word_triggered" not in call_args
        assert "min_score" not in call_args

    @pytest.mark.asyncio
    async def test_list_highlights_repository_exception(
        self, highlight_service, mock_repository
    ):
        """Test that repository exceptions are propagated."""
        # Arrange
        org_id = uuid4()
        mock_repository.list.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await highlight_service.list_highlights(org_id)

    @pytest.mark.asyncio
    async def test_get_highlight_repository_exception(
        self, highlight_service, mock_repository
    ):
        """Test that repository exceptions are propagated in get_highlight."""
        # Arrange
        highlight_id = uuid4()
        org_id = uuid4()
        mock_repository.get.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await highlight_service.get_highlight(highlight_id, org_id)

    @pytest.mark.asyncio
    async def test_get_stream_highlights_repository_exception(
        self, highlight_service, mock_repository
    ):
        """Test that repository exceptions are propagated in get_stream_highlights."""
        # Arrange
        stream_id = uuid4()
        org_id = uuid4()
        mock_repository.list_by_stream.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await highlight_service.get_stream_highlights(stream_id, org_id)