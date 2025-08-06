"""Unit tests for Gemini scorer service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
import json

from worker.services.gemini_scorer import (
    VideoFile,
    GeminiFileManager, 
    GeminiVideoScorer,
    gemini_video_file
)
from worker.services.dimension_framework import (
    DimensionDefinition,
    DimensionType,
    ScoringRubric
)


class TestVideoFile:
    """Test VideoFile dataclass."""
    
    def test_video_file_creation(self):
        """Test creating a VideoFile."""
        video_file = VideoFile(
            uri="gs://example-bucket/video.mp4",
            name="video.mp4",
            mime_type="video/mp4",
            size_bytes=1024*1024
        )
        
        assert video_file.uri == "gs://example-bucket/video.mp4"
        assert video_file.name == "video.mp4"
        assert video_file.mime_type == "video/mp4"
        assert video_file.size_bytes == 1024*1024

    def test_file_data_property(self):
        """Test file_data property conversion."""
        video_file = VideoFile(
            uri="gs://example-bucket/video.mp4",
            name="video.mp4", 
            mime_type="video/mp4",
            size_bytes=1024
        )
        
        with patch('worker.services.gemini_scorer.types') as mock_types:
            mock_file_data = Mock()
            mock_types.FileData.return_value = mock_file_data
            
            result = video_file.file_data
            
            mock_types.FileData.assert_called_once_with(
                file_uri="gs://example-bucket/video.mp4",
                mime_type="video/mp4"
            )
            assert result == mock_file_data


class TestGeminiFileManager:
    """Test Gemini file manager."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock Gemini client."""
        return Mock()
    
    @pytest.fixture
    def file_manager(self, mock_client):
        """Create file manager instance."""
        return GeminiFileManager(mock_client)
    
    @pytest.fixture
    def sample_video_path(self, tmp_path):
        """Create sample video file path."""
        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"fake video content" * 1000)  # Small test file
        return video_path

    async def test_upload_video_success(self, file_manager, sample_video_path):
        """Test successful video upload."""
        mock_response = Mock()
        mock_response.uri = "gs://bucket/video.mp4"
        mock_response.name = "projects/123/locations/us/corpora/456/files/789"
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = mock_response
            
            result = await file_manager.upload_video(sample_video_path)
            
            assert isinstance(result, VideoFile)
            assert result.uri == "gs://bucket/video.mp4"
            assert result.name == "projects/123/locations/us/corpora/456/files/789"
            assert result.mime_type == "video/mp4"
            assert result.size_bytes > 0
            assert result in file_manager._uploaded_files

    async def test_upload_video_file_not_found(self, file_manager):
        """Test upload with non-existent file."""
        non_existent_path = Path("/nonexistent/video.mp4")
        
        with pytest.raises(ValueError, match="Video file not found"):
            await file_manager.upload_video(non_existent_path)

    async def test_upload_video_file_too_large(self, file_manager, tmp_path):
        """Test upload with file too large."""
        large_file = tmp_path / "large_video.mp4"
        
        # Mock stat to return large size
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = 3 * 1024 * 1024 * 1024  # 3GB
            
            with pytest.raises(ValueError, match="Video file too large"):
                await file_manager.upload_video(large_file)

    async def test_upload_video_api_error(self, file_manager, sample_video_path):
        """Test upload with API error."""
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = Exception("API error")
            
            with pytest.raises(RuntimeError, match="Video upload failed"):
                await file_manager.upload_video(sample_video_path)

    async def test_poll_until_active_success(self, file_manager):
        """Test successful file polling until active."""
        video_file = VideoFile(
            uri="gs://bucket/video.mp4",
            name="test_file",
            mime_type="video/mp4", 
            size_bytes=1024
        )
        
        mock_file_info = Mock()
        mock_file_info.state.name = "ACTIVE"
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = mock_file_info
            
            # Should complete without raising
            await file_manager.poll_until_active(video_file, max_wait_time=10)

    async def test_poll_until_active_failed_state(self, file_manager):
        """Test polling when file enters failed state."""
        video_file = VideoFile(
            uri="gs://bucket/video.mp4",
            name="test_file",
            mime_type="video/mp4",
            size_bytes=1024
        )
        
        mock_file_info = Mock()
        mock_file_info.state.name = "FAILED"
        mock_file_info.state.error_message = "Processing failed"
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = mock_file_info
            
            with pytest.raises(ValueError, match="failed to process"):
                await file_manager.poll_until_active(video_file, max_wait_time=10)

    async def test_poll_until_active_timeout(self, file_manager):
        """Test polling timeout."""
        video_file = VideoFile(
            uri="gs://bucket/video.mp4",
            name="test_file", 
            mime_type="video/mp4",
            size_bytes=1024
        )
        
        mock_file_info = Mock()
        mock_file_info.state.name = "PROCESSING"
        
        with patch('asyncio.to_thread') as mock_to_thread, \
             patch('asyncio.sleep'):  # Speed up test
            mock_to_thread.return_value = mock_file_info
            
            with pytest.raises(RuntimeError, match="did not become ACTIVE"):
                await file_manager.poll_until_active(video_file, max_wait_time=0.1)

    async def test_delete_file_success(self, file_manager):
        """Test successful file deletion."""
        video_file = VideoFile(
            uri="gs://bucket/video.mp4",
            name="test_file",
            mime_type="video/mp4",
            size_bytes=1024
        )
        file_manager._uploaded_files.append(video_file)
        
        with patch('asyncio.to_thread') as mock_to_thread:
            await file_manager.delete_file(video_file)
            
            mock_to_thread.assert_called_once()
            assert video_file not in file_manager._uploaded_files

    async def test_delete_file_error(self, file_manager):
        """Test file deletion with error."""
        video_file = VideoFile(
            uri="gs://bucket/video.mp4", 
            name="test_file",
            mime_type="video/mp4",
            size_bytes=1024
        )
        
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = Exception("Delete failed")
            
            # Should not raise, just log error
            await file_manager.delete_file(video_file)

    async def test_cleanup(self, file_manager):
        """Test cleanup of all files."""
        video_files = [
            VideoFile(
                uri=f"gs://bucket/video{i}.mp4",
                name=f"test_file_{i}",
                mime_type="video/mp4",
                size_bytes=1024
            )
            for i in range(3)
        ]
        file_manager._uploaded_files = video_files[:]
        
        with patch.object(file_manager, 'delete_file', new_callable=AsyncMock) as mock_delete:
            await file_manager.cleanup()
            
            assert mock_delete.call_count == 3
            for video_file in video_files:
                mock_delete.assert_any_call(video_file)

    def test_get_video_mime_type(self):
        """Test MIME type detection."""
        assert GeminiFileManager._get_video_mime_type(Path("test.mp4")) == "video/mp4"
        assert GeminiFileManager._get_video_mime_type(Path("test.mpeg")) == "video/mpeg"
        assert GeminiFileManager._get_video_mime_type(Path("test.webm")) == "video/webm"
        assert GeminiFileManager._get_video_mime_type(Path("test.avi")) == "video/x-msvideo"
        assert GeminiFileManager._get_video_mime_type(Path("test.mov")) == "video/quicktime"
        
        # Unknown extension defaults to mp4
        assert GeminiFileManager._get_video_mime_type(Path("test.unknown")) == "video/mp4"


class TestGeminiVideoFile:
    """Test gemini_video_file context manager."""

    async def test_context_manager_success(self, tmp_path):
        """Test successful context manager usage."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")
        
        mock_client = Mock()
        mock_video_file = VideoFile(
            uri="gs://bucket/video.mp4",
            name="test_file",
            mime_type="video/mp4",
            size_bytes=1024
        )
        
        with patch('worker.services.gemini_scorer.GeminiFileManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager.upload_video.return_value = mock_video_file
            mock_manager_class.return_value = mock_manager
            
            async with gemini_video_file(mock_client, video_path) as video_file:
                assert video_file == mock_video_file
                mock_manager.upload_video.assert_called_once_with(video_path)
                mock_manager.poll_until_active.assert_called_once_with(mock_video_file)
            
            # Should cleanup after use
            mock_manager.delete_file.assert_called_once_with(mock_video_file)

    async def test_context_manager_with_exception(self, tmp_path):
        """Test context manager cleanup on exception."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")
        
        mock_client = Mock()
        mock_video_file = VideoFile(
            uri="gs://bucket/video.mp4",
            name="test_file",
            mime_type="video/mp4",
            size_bytes=1024
        )
        
        with patch('worker.services.gemini_scorer.GeminiFileManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager.upload_video.return_value = mock_video_file
            mock_manager_class.return_value = mock_manager
            
            with pytest.raises(ValueError, match="test exception"):
                async with gemini_video_file(mock_client, video_path) as video_file:
                    raise ValueError("test exception")
            
            # Should still cleanup after exception
            mock_manager.delete_file.assert_called_once_with(mock_video_file)


class TestGeminiVideoScorer:
    """Test Gemini video scorer."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock client."""
        return Mock()
    
    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        with patch('worker.services.gemini_scorer.genai.Client'):
            return GeminiVideoScorer(api_key="test-key")
    
    @pytest.fixture
    def sample_rubric(self):
        """Create sample scoring rubric."""
        dimensions = [
            DimensionDefinition(
                name="action_intensity",
                description="Rate the action intensity",
                type=DimensionType.SCALE_1_4,
                weight=1.0,
                scoring_prompt="Rate action intensity",
                examples=[]
            ),
            DimensionDefinition(
                name="emotional_impact",
                description="Has emotional impact",
                type=DimensionType.BINARY,
                weight=0.8,
                scoring_prompt="Has emotional impact",
                examples=[]
            )
        ]
        
        return ScoringRubric(
            name="Test Rubric",
            description="Test rubric",
            dimensions=dimensions,
            highlight_threshold=7.0,
            highlight_confidence_threshold=0.8
        )

    async def test_score_success(self, scorer, sample_rubric, tmp_path):
        """Test successful video scoring."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")
        
        mock_response = Mock()
        mock_response.text = json.dumps({
            "dimensions": {
                "action_intensity": {"score": 3.5, "confidence": 0.9},
                "emotional_impact": {"score": 1.0, "confidence": 0.8}
            }
        })
        
        mock_video_file = VideoFile(
            uri="gs://bucket/video.mp4",
            name="test_file", 
            mime_type="video/mp4",
            size_bytes=1024
        )
        
        with patch('worker.services.gemini_scorer.gemini_video_file') as mock_context:
            mock_context.return_value.__aenter__.return_value = mock_video_file
            
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = mock_response
                
                result = await scorer.score(video_path, sample_rubric)
                
                assert "action_intensity" in result
                assert "emotional_impact" in result
                
                # Check normalized scores
                action_score, action_conf = result["action_intensity"]
                emotional_score, emotional_conf = result["emotional_impact"]
                
                assert 0 <= action_score <= 1  # Should be normalized from 1-4 to 0-1
                assert action_conf == 0.9
                assert emotional_score == 1.0
                assert emotional_conf == 0.8

    async def test_score_file_not_found(self, scorer, sample_rubric):
        """Test scoring with non-existent file."""
        non_existent_path = Path("/nonexistent/video.mp4")
        
        with pytest.raises(ValueError, match="Video file not found"):
            await scorer.score(non_existent_path, sample_rubric)

    async def test_score_api_error(self, scorer, sample_rubric, tmp_path):
        """Test scoring with API error."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")
        
        mock_video_file = VideoFile(
            uri="gs://bucket/video.mp4",
            name="test_file",
            mime_type="video/mp4",
            size_bytes=1024
        )
        
        with patch('worker.services.gemini_scorer.gemini_video_file') as mock_context:
            mock_context.return_value.__aenter__.return_value = mock_video_file
            
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.side_effect = Exception("API error")
                
                result = await scorer.score(video_path, sample_rubric)
                
                # Should return low confidence scores on error
                assert all(score == 0.0 and conf == 0.0 for score, conf in result.values())

    async def test_score_invalid_json_response(self, scorer, sample_rubric, tmp_path):
        """Test scoring with invalid JSON response."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")
        
        mock_response = Mock()
        mock_response.text = "invalid json"
        
        mock_video_file = VideoFile(
            uri="gs://bucket/video.mp4",
            name="test_file",
            mime_type="video/mp4", 
            size_bytes=1024
        )
        
        with patch('worker.services.gemini_scorer.gemini_video_file') as mock_context:
            mock_context.return_value.__aenter__.return_value = mock_video_file
            
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = mock_response
                
                result = await scorer.score(video_path, sample_rubric)
                
                # Should return low confidence scores on parse error
                assert all(score == 0.0 and conf == 0.0 for score, conf in result.values())

    async def test_score_with_timestamp(self, scorer, tmp_path):
        """Test scoring with timestamp."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")
        
        dimension = DimensionDefinition(
            name="action_intensity",
            description="Rate the action intensity",
            type=DimensionType.SCALE_1_4,
            weight=1.0,
            scoring_prompt="Rate action intensity",
            examples=[]
        )
        
        mock_response = Mock()
        mock_response.text = json.dumps({
            "score": 3.0,
            "confidence": 0.8,
            "reasoning": "High action sequence"
        })
        
        mock_video_file = VideoFile(
            uri="gs://bucket/video.mp4",
            name="test_file",
            mime_type="video/mp4",
            size_bytes=1024
        )
        
        with patch('worker.services.gemini_scorer.gemini_video_file') as mock_context:
            mock_context.return_value.__aenter__.return_value = mock_video_file
            
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = mock_response
                
                score, confidence, reasoning = await scorer.score_with_timestamp(
                    video_path, dimension, "02:30"
                )
                
                # Should normalize 1-4 scale to 0-1
                assert 0 <= score <= 1
                assert confidence == 0.8
                assert reasoning == "High action sequence"

    async def test_score_with_timestamp_error(self, scorer, tmp_path):
        """Test scoring with timestamp API error."""
        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")
        
        dimension = DimensionDefinition(
            name="action_intensity",
            description="Rate the action intensity", 
            type=DimensionType.SCALE_1_4,
            weight=1.0,
            scoring_prompt="Rate action intensity",
            examples=[]
        )
        
        mock_video_file = VideoFile(
            uri="gs://bucket/video.mp4",
            name="test_file",
            mime_type="video/mp4",
            size_bytes=1024
        )
        
        with patch('worker.services.gemini_scorer.gemini_video_file') as mock_context:
            mock_context.return_value.__aenter__.return_value = mock_video_file
            
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.side_effect = Exception("API error")
                
                score, confidence, reasoning = await scorer.score_with_timestamp(
                    video_path, dimension, "02:30"
                )
                
                assert score == 0.0
                assert confidence == 0.0
                assert "Error:" in reasoning

    def test_parse_response_success(self, scorer, sample_rubric):
        """Test successful response parsing."""
        mock_response = Mock()
        mock_response.text = json.dumps({
            "dimensions": {
                "action_intensity": {"score": 3.5, "confidence": 0.9},
                "emotional_impact": {"score": 1.0, "confidence": 0.8}
            }
        })
        
        result = scorer._parse_response(mock_response, sample_rubric)
        
        assert len(result) == 2
        assert "action_intensity" in result
        assert "emotional_impact" in result
        
        # Check score normalization
        action_score, action_conf = result["action_intensity"] 
        emotional_score, emotional_conf = result["emotional_impact"]
        
        # action_intensity should be normalized from 1-4 to 0-1
        expected_normalized = (3.5 - 1) / 3  # (score - 1) / 3
        assert abs(action_score - expected_normalized) < 0.001
        assert action_conf == 0.9
        
        # binary should remain as-is
        assert emotional_score == 1.0
        assert emotional_conf == 0.8

    def test_parse_response_missing_dimensions(self, scorer, sample_rubric):
        """Test parsing response with missing dimensions."""
        mock_response = Mock()
        mock_response.text = json.dumps({
            "dimensions": {
                "action_intensity": {"score": 3.0, "confidence": 0.8}
                # Missing emotional_impact
            }
        })
        
        result = scorer._parse_response(mock_response, sample_rubric)
        
        assert len(result) == 2
        assert result["action_intensity"] != (0.0, 0.0)  # Should have real values
        assert result["emotional_impact"] == (0.0, 0.0)  # Should be default for missing

    def test_parse_response_invalid_json(self, scorer, sample_rubric):
        """Test parsing invalid JSON response."""
        mock_response = Mock()
        mock_response.text = "invalid json"
        
        result = scorer._parse_response(mock_response, sample_rubric)
        
        # Should return default low-confidence scores
        assert all(score == 0.0 and conf == 0.0 for score, conf in result.values())