"""Mock-based unit tests for stream repository to increase coverage."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infrastructure.storage.repositories.stream import StreamRepository
from shared.domain.models.stream import Stream, StreamStatus, StreamType
from shared.infrastructure.database.models import StreamModel


class TestStreamRepositoryMock:
    """Mock-based tests for StreamRepository to increase coverage."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_session):
        """Create repository instance."""
        return StreamRepository(mock_session)

    @pytest.fixture
    def sample_stream(self):
        """Create sample stream."""
        return Stream(
            id=uuid4(),
            organization_id=uuid4(),
            url="https://example.com/stream.m3u8",
            type=StreamType.LIVESTREAM,
            status=StreamStatus.PENDING
        )

    @pytest.fixture
    def sample_stream_model(self):
        """Create sample stream model."""
        stream_id = uuid4()
        org_id = uuid4()
        
        model = StreamModel()
        model.id = stream_id
        model.organization_id = org_id
        model.stream_url = "https://example.com/stream.m3u8"
        model.stream_fingerprint = "test_stream"
        model.status = StreamStatus.PENDING
        model.segments_processed = 0
        model.highlights_generated = 0
        model.retry_count = 0
        model.duration_seconds = 0.0
        
        return model

    @pytest.mark.asyncio
    async def test_add_stream(self, repository, sample_stream, mock_session):
        """Test adding a stream."""
        # Mock the session operations
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()
        
        result = await repository.add(sample_stream)
        
        # Should have called session.add, commit, and refresh
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        
        # Result should be the same stream
        assert result == sample_stream

    @pytest.mark.asyncio
    async def test_get_stream(self, repository, sample_stream_model, mock_session):
        """Test getting a stream by ID."""
        stream_id = uuid4()
        
        # Mock the query result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=sample_stream_model)
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await repository.get(stream_id)
        
        # Should have executed query
        mock_session.execute.assert_called_once()
        mock_result.scalar_one_or_none.assert_called_once()
        
        # Should return converted stream
        assert result is not None
        assert isinstance(result, Stream)

    @pytest.mark.asyncio
    async def test_get_stream_not_found(self, repository, mock_session):
        """Test getting a stream that doesn't exist."""
        stream_id = uuid4()
        
        # Mock empty result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await repository.get(stream_id)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_update_stream(self, repository, sample_stream, mock_session):
        """Test updating a stream."""
        # Mock the session operations
        mock_session.merge = AsyncMock(return_value=sample_stream)
        mock_session.commit = AsyncMock()
        
        result = await repository.update(sample_stream)
        
        # Should have called merge and commit
        mock_session.merge.assert_called_once()
        mock_session.commit.assert_called_once()
        
        assert result == sample_stream

    @pytest.mark.asyncio
    async def test_delete_stream(self, repository, mock_session):
        """Test deleting a stream."""
        stream_id = uuid4()
        
        # Mock finding the stream
        mock_stream_model = MagicMock()
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=mock_stream_model)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.delete = MagicMock()
        mock_session.commit = AsyncMock()
        
        await repository.delete(stream_id)
        
        # Should have found, deleted, and committed
        mock_session.execute.assert_called()
        mock_session.delete.assert_called_once_with(mock_stream_model)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_stream_not_found(self, repository, mock_session):
        """Test deleting a stream that doesn't exist."""
        stream_id = uuid4()
        
        # Mock empty result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.delete = MagicMock()
        mock_session.commit = AsyncMock()
        
        await repository.delete(stream_id)
        
        # Should not have called delete or commit when stream not found
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_by_fingerprint(self, repository, sample_stream_model, mock_session):
        """Test getting stream by fingerprint."""
        fingerprint = "test_stream"
        org_id = uuid4()
        
        # Mock the query result
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=sample_stream_model)
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await repository.get_by_fingerprint(fingerprint, org_id)
        
        # Should have executed query
        mock_session.execute.assert_called_once()
        assert result is not None
        assert isinstance(result, Stream)

    @pytest.mark.asyncio  
    async def test_list_active(self, repository, mock_session):
        """Test listing active streams."""
        org_id = uuid4()
        
        # Mock the query result with active streams
        mock_models = [MagicMock() for _ in range(3)]
        for i, model in enumerate(mock_models):
            model.id = uuid4()
            model.organization_id = org_id
            model.stream_url = f"https://stream{i}.example.com"
            model.stream_fingerprint = f"stream_{i}"
            model.status = StreamStatus.PROCESSING
            model.segments_processed = i * 10
            model.highlights_generated = i * 5
            model.retry_count = 0
            model.duration_seconds = float(i * 60)
        
        mock_result = AsyncMock()
        mock_result.scalars = AsyncMock()
        mock_result.scalars().all = AsyncMock(return_value=mock_models)
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await repository.list_active(org_id)
        
        # Should have executed query and returned converted streams
        mock_session.execute.assert_called_once()
        assert len(result) == 3
        assert all(isinstance(stream, Stream) for stream in result)

    def test_to_entity_conversion(self, repository, sample_stream_model):
        """Test converting model to entity."""
        result = repository._to_entity(sample_stream_model)
        
        assert isinstance(result, Stream)
        assert result.id == sample_stream_model.id
        assert result.organization_id == sample_stream_model.organization_id
        assert result.url == sample_stream_model.stream_url
        assert result.stream_fingerprint == sample_stream_model.stream_fingerprint