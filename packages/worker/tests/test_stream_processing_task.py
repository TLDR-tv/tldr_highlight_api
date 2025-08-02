"""Tests for stream processing Celery tasks."""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import uuid4
import pytest
import tempfile
from pathlib import Path

from worker.tasks.stream_processing import process_stream_task as process_stream
from shared.domain.models.stream import Stream, StreamType, StreamSource
from shared.domain.models.organization import Organization


@pytest.fixture
def mock_stream():
    """Create mock stream."""
    return Stream(
        id=uuid4(),
        organization_id=uuid4(),
        url="https://example.com/stream.m3u8",
        name="Test Stream",
        type=StreamType.LIVESTREAM,
        source_type=StreamSource.HLS
    )


@pytest.fixture
def mock_organization():
    """Create mock organization."""
    return Organization(
        id=uuid4(),
        name="Test Org",
        webhook_url="https://example.com/webhook",
        webhook_secret="test-secret"
    )


class TestStreamProcessingTask:
    """Test stream processing Celery task."""
    
    @pytest.mark.asyncio
    async def test_process_stream_success(self, mock_stream, mock_organization):
        """Test successful stream processing."""
        with patch("worker.tasks.stream_processing.get_database") as mock_get_db:
            # Mock database and repositories
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db
            
            mock_session = AsyncMock()
            mock_db.session.return_value.__aenter__.return_value = mock_session
            
            with patch("worker.tasks.stream_processing.StreamRepository") as mock_stream_repo:
                with patch("worker.tasks.stream_processing.OrganizationRepository") as mock_org_repo:
                    with patch("worker.tasks.stream_processing.HighlightRepository") as mock_highlight_repo:
                        # Set up repository mocks
                        stream_repo = mock_stream_repo.return_value
                        stream_repo.get.return_value = mock_stream
                        
                        org_repo = mock_org_repo.return_value
                        org_repo.get.return_value = mock_organization
                        
                        highlight_repo = mock_highlight_repo.return_value
                        
                        # Mock services
                        with patch("worker.tasks.stream_processing.FFmpegService") as mock_ffmpeg:
                            with patch("worker.tasks.stream_processing.GeminiScorer") as mock_scorer:
                                with patch("worker.tasks.stream_processing.celery_app.send_task") as mock_send_task:
                                    # Mock FFmpeg segmentation
                                    ffmpeg_service = mock_ffmpeg.return_value
                                    segments = [
                                        {"path": "/tmp/segment1.mp4", "start_time": 0, "duration": 10},
                                        {"path": "/tmp/segment2.mp4", "start_time": 10, "duration": 10}
                                    ]
                                    ffmpeg_service.segment_stream = AsyncMock(return_value=segments)
                                    
                                    # Mock Gemini scoring
                                    scorer = mock_scorer.return_value
                                    scorer.score_video_segment = AsyncMock(side_effect=[
                                        {"score": 0.8, "reason": "High action"},
                                        {"score": 0.6, "reason": "Moderate action"}
                                    ])
                                    
                                    # Run the task
                                    result = await _process_stream_async(str(mock_stream.id))
                                    
                                    # Verify results
                                    assert result["status"] == "completed"
                                    assert result["stream_id"] == str(mock_stream.id)
                                    assert result["highlights_found"] == 2
                                    
                                    # Verify FFmpeg was called
                                    ffmpeg_service.segment_stream.assert_called_once()
                                    
                                    # Verify scorer was called for each segment
                                    assert scorer.score_video_segment.call_count == 2
                                    
                                    # Verify highlights were created
                                    assert highlight_repo.create.call_count == 2
                                    
                                    # Verify webhook was sent
                                    mock_send_task.assert_called_with(
                                        "send_webhook",
                                        args=[
                                            mock_organization.webhook_url,
                                            mock_organization.webhook_secret,
                                            {
                                                "event": "stream.processed",
                                                "stream_id": str(mock_stream.id),
                                                "highlights_count": 2
                                            }
                                        ]
                                    )
    
    @pytest.mark.asyncio
    async def test_process_stream_not_found(self):
        """Test processing non-existent stream."""
        stream_id = str(uuid4())
        
        with patch("worker.tasks.stream_processing.get_database") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db
            
            mock_session = AsyncMock()
            mock_db.session.return_value.__aenter__.return_value = mock_session
            
            with patch("worker.tasks.stream_processing.StreamRepository") as mock_stream_repo:
                stream_repo = mock_stream_repo.return_value
                stream_repo.get.return_value = None
                
                # Should raise ValueError
                with pytest.raises(ValueError, match="Stream not found"):
                    await _process_stream_async(stream_id)
    
    @pytest.mark.asyncio
    async def test_process_stream_ffmpeg_error(self, mock_stream, mock_organization):
        """Test handling FFmpeg errors."""
        with patch("worker.tasks.stream_processing.get_database") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db
            
            mock_session = AsyncMock()
            mock_db.session.return_value.__aenter__.return_value = mock_session
            
            with patch("worker.tasks.stream_processing.StreamRepository") as mock_stream_repo:
                with patch("worker.tasks.stream_processing.OrganizationRepository") as mock_org_repo:
                    stream_repo = mock_stream_repo.return_value
                    stream_repo.get.return_value = mock_stream
                    
                    org_repo = mock_org_repo.return_value
                    org_repo.get.return_value = mock_organization
                    
                    with patch("worker.tasks.stream_processing.FFmpegService") as mock_ffmpeg:
                        # Mock FFmpeg to raise error
                        ffmpeg_service = mock_ffmpeg.return_value
                        ffmpeg_service.segment_stream = AsyncMock(
                            side_effect=Exception("FFmpeg failed to process stream")
                        )
                        
                        # Should raise the exception
                        with pytest.raises(Exception, match="FFmpeg failed"):
                            await _process_stream_async(str(mock_stream.id))
    
    @pytest.mark.asyncio
    async def test_process_stream_no_highlights(self, mock_stream, mock_organization):
        """Test processing stream with no highlights found."""
        with patch("worker.tasks.stream_processing.get_database") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db
            
            mock_session = AsyncMock()
            mock_db.session.return_value.__aenter__.return_value = mock_session
            
            with patch("worker.tasks.stream_processing.StreamRepository") as mock_stream_repo:
                with patch("worker.tasks.stream_processing.OrganizationRepository") as mock_org_repo:
                    with patch("worker.tasks.stream_processing.HighlightRepository") as mock_highlight_repo:
                        stream_repo = mock_stream_repo.return_value
                        stream_repo.get.return_value = mock_stream
                        
                        org_repo = mock_org_repo.return_value
                        org_repo.get.return_value = mock_organization
                        
                        highlight_repo = mock_highlight_repo.return_value
                        
                        with patch("worker.tasks.stream_processing.FFmpegService") as mock_ffmpeg:
                            with patch("worker.tasks.stream_processing.GeminiScorer") as mock_scorer:
                                with patch("worker.tasks.stream_processing.celery_app.send_task") as mock_send_task:
                                    # Mock FFmpeg segmentation
                                    ffmpeg_service = mock_ffmpeg.return_value
                                    segments = [
                                        {"path": "/tmp/segment1.mp4", "start_time": 0, "duration": 10}
                                    ]
                                    ffmpeg_service.segment_stream = AsyncMock(return_value=segments)
                                    
                                    # Mock Gemini scoring with low scores
                                    scorer = mock_scorer.return_value
                                    scorer.score_video_segment = AsyncMock(return_value={
                                        "score": 0.3,
                                        "reason": "Low action"
                                    })
                                    
                                    # Run the task
                                    result = await _process_stream_async(str(mock_stream.id))
                                    
                                    # Verify results
                                    assert result["status"] == "completed"
                                    assert result["highlights_found"] == 0
                                    
                                    # Verify no highlights were created
                                    highlight_repo.create.assert_not_called()
                                    
                                    # Verify webhook was still sent
                                    mock_send_task.assert_called_with(
                                        "send_webhook",
                                        args=[
                                            mock_organization.webhook_url,
                                            mock_organization.webhook_secret,
                                            {
                                                "event": "stream.processed",
                                                "stream_id": str(mock_stream.id),
                                                "highlights_count": 0
                                            }
                                        ]
                                    )
    
    def test_process_stream_celery_task(self):
        """Test Celery task wrapper."""
        stream_id = str(uuid4())
        
        with patch("worker.tasks.stream_processing.asyncio.run") as mock_run:
            mock_run.return_value = {
                "status": "completed",
                "stream_id": stream_id,
                "highlights_found": 5
            }
            
            # Call the Celery task
            result = process_stream(stream_id)
            
            # Verify asyncio.run was called
            mock_run.assert_called_once()
            
            # Verify result
            assert result["status"] == "completed"
            assert result["highlights_found"] == 5