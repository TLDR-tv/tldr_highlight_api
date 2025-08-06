"""Comprehensive tests for Gemini scorer service."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import uuid4
from pathlib import Path
import tempfile
import json
import time

from worker.services.gemini_scorer import (
    GeminiFileManager,
    GeminiVideoScorer,
    HighlightBoundary
)
from worker.services.dimension_framework import (
    DimensionDefinition,
    DimensionType,
    ScoringRubric
)


class TestHighlightBoundary:
    """Test HighlightBoundary dataclass."""

    def test_highlight_boundary_creation(self):
        """Test creating a highlight boundary."""
        boundary = HighlightBoundary(
            start_timestamp="01:30",
            end_timestamp="02:45",
            confidence=0.85,
            reasoning="High action sequence with explosions"
        )
        
        assert boundary.start_timestamp == "01:30"
        assert boundary.end_timestamp == "02:45"
        assert boundary.confidence == 0.85
        assert boundary.reasoning == "High action sequence with explosions"

    def test_to_seconds_basic(self):
        """Test converting MM:SS timestamps to seconds."""
        boundary = HighlightBoundary(
            start_timestamp="01:30",
            end_timestamp="02:45",
            confidence=0.85,
            reasoning="Test"
        )
        
        start_secs, end_secs = boundary.to_seconds()
        assert start_secs == 90.0  # 1*60 + 30
        assert end_secs == 165.0   # 2*60 + 45

    def test_to_seconds_zero_minutes(self):
        """Test converting timestamps with zero minutes."""
        boundary = HighlightBoundary(
            start_timestamp="00:15",
            end_timestamp="00:45",
            confidence=0.9,
            reasoning="Test"
        )
        
        start_secs, end_secs = boundary.to_seconds()
        assert start_secs == 15.0
        assert end_secs == 45.0

    def test_to_seconds_invalid_format(self):
        """Test handling invalid timestamp format."""
        boundary = HighlightBoundary(
            start_timestamp="invalid",
            end_timestamp="02:45",
            confidence=0.85,
            reasoning="Test"
        )
        
        with pytest.raises(ValueError):
            boundary.to_seconds()

    def test_to_seconds_negative_seconds(self):
        """Test handling negative seconds in timestamp."""
        boundary = HighlightBoundary(
            start_timestamp="01:-30",  # Invalid negative seconds
            end_timestamp="02:45",
            confidence=0.85,
            reasoning="Test"
        )
        
        with pytest.raises(ValueError):
            boundary.to_seconds()


class TestGeminiFileManager:
    """Test GeminiFileManager class."""

    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'fake video content')
            return Path(f.name)

    @pytest.fixture
    def mock_genai_client(self):
        """Create a mock Gemini client."""
        with patch('worker.services.gemini_scorer.genai') as mock_genai:
            mock_client = Mock()
            mock_genai.configure.return_value = None
            mock_genai.upload_file.return_value = Mock(uri="gs://test-bucket/file123")
            mock_genai.get_file.return_value = Mock(state="ACTIVE")
            yield mock_client

    def test_file_manager_initialization(self, mock_genai_client):
        """Test file manager initialization."""
        api_key = "test-api-key"
        
        manager = GeminiFileManager(api_key)
        
        assert manager.api_key == api_key

    @patch('worker.services.gemini_scorer.genai')
    async def test_upload_file_success(self, mock_genai, temp_video_file):
        """Test successful file upload."""
        # Setup mock
        mock_file = Mock()
        mock_file.uri = "gs://test-bucket/uploaded123"
        mock_genai.upload_file.return_value = mock_file
        
        manager = GeminiFileManager("test-key")
        
        # Test
        result = await manager.upload_file(temp_video_file)
        
        # Verify
        assert result == mock_file
        mock_genai.upload_file.assert_called_once()
        
        # Cleanup
        temp_video_file.unlink()

    @patch('worker.services.gemini_scorer.genai')
    async def test_upload_file_with_mime_type(self, mock_genai, temp_video_file):
        """Test file upload with specific mime type."""
        mock_file = Mock()
        mock_file.uri = "gs://test-bucket/uploaded123"
        mock_genai.upload_file.return_value = mock_file
        
        manager = GeminiFileManager("test-key")
        
        # Test
        await manager.upload_file(temp_video_file, mime_type="video/mp4")
        
        # Verify mime type was passed
        call_args = mock_genai.upload_file.call_args
        assert call_args[1]['mime_type'] == "video/mp4"
        
        temp_video_file.unlink()

    @patch('worker.services.gemini_scorer.genai')
    async def test_upload_file_error(self, mock_genai, temp_video_file):
        """Test file upload error handling."""
        mock_genai.upload_file.side_effect = Exception("Upload failed")
        
        manager = GeminiFileManager("test-key")
        
        with pytest.raises(Exception, match="Upload failed"):
            await manager.upload_file(temp_video_file)
        
        temp_video_file.unlink()

    @patch('worker.services.gemini_scorer.genai')
    async def test_poll_until_active_success(self, mock_genai):
        """Test polling until file is active."""
        mock_file = Mock()
        mock_file.name = "files/test123"
        
        # Mock get_file to return PROCESSING then ACTIVE
        mock_states = [
            Mock(state="PROCESSING"),
            Mock(state="ACTIVE")
        ]
        mock_genai.get_file.side_effect = mock_states
        
        manager = GeminiFileManager("test-key")
        
        # Test
        result = await manager.poll_until_active(mock_file, timeout=5)
        
        # Verify
        assert result.state == "ACTIVE"
        assert mock_genai.get_file.call_count == 2

    @patch('worker.services.gemini_scorer.genai')
    async def test_poll_until_active_timeout(self, mock_genai):
        """Test polling timeout."""
        mock_file = Mock()
        mock_file.name = "files/test123"
        
        # Mock get_file to always return PROCESSING
        mock_genai.get_file.return_value = Mock(state="PROCESSING")
        
        manager = GeminiFileManager("test-key")
        
        # Test with very short timeout
        with pytest.raises(TimeoutError):
            await manager.poll_until_active(mock_file, timeout=0.1)

    @patch('worker.services.gemini_scorer.genai')
    async def test_poll_until_active_failed_state(self, mock_genai):
        """Test polling when file reaches failed state."""
        mock_file = Mock()
        mock_file.name = "files/test123"
        
        mock_genai.get_file.return_value = Mock(state="FAILED")
        
        manager = GeminiFileManager("test-key")
        
        with pytest.raises(RuntimeError, match="File processing failed"):
            await manager.poll_until_active(mock_file)

    @patch('worker.services.gemini_scorer.genai')
    async def test_delete_file_success(self, mock_genai):
        """Test successful file deletion."""
        mock_file = Mock()
        mock_file.name = "files/test123"
        mock_genai.delete_file.return_value = None
        
        manager = GeminiFileManager("test-key")
        
        # Test
        await manager.delete_file(mock_file)
        
        # Verify
        mock_genai.delete_file.assert_called_once_with(mock_file.name)

    @patch('worker.services.gemini_scorer.genai')
    async def test_delete_file_error(self, mock_genai):
        """Test file deletion error handling."""
        mock_file = Mock()
        mock_file.name = "files/test123"
        mock_genai.delete_file.side_effect = Exception("Delete failed")
        
        manager = GeminiFileManager("test-key")
        
        # Should not raise exception - just log error
        await manager.delete_file(mock_file)


class TestGeminiVideoScorer:
    """Test GeminiVideoScorer class."""

    @pytest.fixture
    def sample_rubric(self):
        """Create sample scoring rubric."""
        dimensions = [
            DimensionDefinition(
                name="action_intensity",
                description="Rate the action intensity",
                type=DimensionType.SCALE_1_4,
                weight=1.0,
                scoring_prompt="Rate the action intensity from 1-4",
                examples=[]
            ),
            DimensionDefinition(
                name="educational_value",
                description="Has educational content",
                type=DimensionType.BINARY,
                weight=0.8,
                scoring_prompt="Does this have educational value?",
                examples=[]
            )
        ]
        
        return ScoringRubric(
            name="Test Rubric",
            description="For testing",
            dimensions=dimensions,
            highlight_threshold=0.7,
            highlight_confidence_threshold=0.8
        )

    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'fake video content')
            return Path(f.name)

    def test_scorer_initialization_default(self):
        """Test scorer initialization with defaults."""
        api_key = "test-api-key"
        
        scorer = GeminiVideoScorer(api_key)
        
        assert scorer.api_key == api_key
        assert scorer.model_name == "gemini-1.5-pro"
        assert isinstance(scorer.file_manager, GeminiFileManager)

    def test_scorer_initialization_custom_model(self):
        """Test scorer initialization with custom model."""
        api_key = "test-api-key"
        model_name = "gemini-2.0-flash"
        
        scorer = GeminiVideoScorer(api_key, model_name=model_name)
        
        assert scorer.model_name == model_name

    @patch('worker.services.gemini_scorer.genai')
    async def test_score_basic_success(self, mock_genai, sample_rubric, temp_video_file):
        """Test basic successful scoring."""
        # Setup mocks
        mock_file = Mock()
        mock_file.uri = "gs://test-bucket/file123"
        
        mock_file_manager = AsyncMock()
        mock_file_manager.upload_file.return_value = mock_file
        mock_file_manager.poll_until_active.return_value = mock_file
        mock_file_manager.delete_file.return_value = None
        
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "action_intensity": {"score": 3, "confidence": 0.85, "reasoning": "High action"},
            "educational_value": {"score": 0, "confidence": 0.9, "reasoning": "No educational content"}
        })
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        scorer = GeminiVideoScorer("test-key")
        scorer.file_manager = mock_file_manager
        
        # Test
        result = await scorer.score(temp_video_file, sample_rubric)
        
        # Verify
        assert isinstance(result, dict)
        assert "action_intensity" in result
        assert "educational_value" in result
        assert result["action_intensity"][0] == 0.75  # 3/4 normalized
        assert result["action_intensity"][1] == 0.85  # confidence
        assert result["educational_value"][0] == 0.0  # binary score
        assert result["educational_value"][1] == 0.9   # confidence
        
        # Verify cleanup
        mock_file_manager.delete_file.assert_called_once_with(mock_file)
        
        temp_video_file.unlink()

    @patch('worker.services.gemini_scorer.genai')
    async def test_score_with_boundaries_success(self, mock_genai, sample_rubric, temp_video_file):
        """Test scoring with highlight boundaries."""
        # Setup mocks
        mock_file = Mock()
        mock_file.uri = "gs://test-bucket/file123"
        
        mock_file_manager = AsyncMock()
        mock_file_manager.upload_file.return_value = mock_file
        mock_file_manager.poll_until_active.return_value = mock_file
        mock_file_manager.delete_file.return_value = None
        
        # Mock Gemini response with boundaries
        mock_response = Mock()
        mock_response.text = json.dumps({
            "scores": {
                "action_intensity": {"score": 4, "confidence": 0.9, "reasoning": "Intense action"},
                "educational_value": {"score": 1, "confidence": 0.8, "reasoning": "Some education"}
            },
            "highlight_boundary": {
                "start_timestamp": "01:30",
                "end_timestamp": "02:15",
                "confidence": 0.85,
                "reasoning": "Peak action sequence"
            }
        })
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        scorer = GeminiVideoScorer("test-key")
        scorer.file_manager = mock_file_manager
        
        # Test
        scores, boundary = await scorer.score_with_boundaries(temp_video_file, sample_rubric)
        
        # Verify scores
        assert isinstance(scores, dict)
        assert "action_intensity" in scores
        assert scores["action_intensity"][0] == 1.0  # 4/4 normalized
        
        # Verify boundary
        assert isinstance(boundary, HighlightBoundary)
        assert boundary.start_timestamp == "01:30"
        assert boundary.end_timestamp == "02:15"
        assert boundary.confidence == 0.85
        
        temp_video_file.unlink()

    @patch('worker.services.gemini_scorer.genai')
    async def test_score_gemini_error(self, mock_genai, sample_rubric, temp_video_file):
        """Test handling of Gemini API errors."""
        # Setup mocks
        mock_file = Mock()
        mock_file.uri = "gs://test-bucket/file123"
        
        mock_file_manager = AsyncMock()
        mock_file_manager.upload_file.return_value = mock_file
        mock_file_manager.poll_until_active.return_value = mock_file
        mock_file_manager.delete_file.return_value = None
        
        # Mock Gemini to raise error
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API rate limit exceeded")
        mock_genai.GenerativeModel.return_value = mock_model
        
        scorer = GeminiVideoScorer("test-key")
        scorer.file_manager = mock_file_manager
        
        # Test
        with pytest.raises(Exception, match="API rate limit exceeded"):
            await scorer.score(temp_video_file, sample_rubric)
        
        # Should still cleanup file
        mock_file_manager.delete_file.assert_called_once_with(mock_file)
        
        temp_video_file.unlink()

    @patch('worker.services.gemini_scorer.genai')
    async def test_score_invalid_json_response(self, mock_genai, sample_rubric, temp_video_file):
        """Test handling of invalid JSON response."""
        # Setup mocks
        mock_file = Mock()
        mock_file_uri = "gs://test-bucket/file123"
        mock_file.uri = mock_file_uri
        
        mock_file_manager = AsyncMock()
        mock_file_manager.upload_file.return_value = mock_file
        mock_file_manager.poll_until_active.return_value = mock_file
        mock_file_manager.delete_file.return_value = None
        
        # Mock invalid JSON response
        mock_response = Mock()
        mock_response.text = "This is not valid JSON"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        scorer = GeminiVideoScorer("test-key")
        scorer.file_manager = mock_file_manager
        
        # Test
        with pytest.raises(json.JSONDecodeError):
            await scorer.score(temp_video_file, sample_rubric)
        
        temp_video_file.unlink()

    @patch('worker.services.gemini_scorer.genai')
    async def test_score_missing_dimension_scores(self, mock_genai, sample_rubric, temp_video_file):
        """Test handling of missing dimension scores in response."""
        # Setup mocks
        mock_file = Mock()
        mock_file.uri = "gs://test-bucket/file123"
        
        mock_file_manager = AsyncMock()
        mock_file_manager.upload_file.return_value = mock_file
        mock_file_manager.poll_until_active.return_value = mock_file
        mock_file_manager.delete_file.return_value = None
        
        # Mock response missing one dimension
        mock_response = Mock()
        mock_response.text = json.dumps({
            "action_intensity": {"score": 3, "confidence": 0.85, "reasoning": "High action"}
            # Missing educational_value
        })
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        scorer = GeminiVideoScorer("test-key")
        scorer.file_manager = mock_file_manager
        
        # Test
        result = await scorer.score(temp_video_file, sample_rubric)
        
        # Should return available scores
        assert "action_intensity" in result
        # Missing dimension should not be included or should have default values
        assert len(result) <= len(sample_rubric.dimensions)
        
        temp_video_file.unlink()

    @patch('worker.services.gemini_scorer.genai')
    async def test_score_with_context(self, mock_genai, sample_rubric, temp_video_file):
        """Test scoring with context segments."""
        # Setup mocks
        mock_file = Mock()
        mock_file.uri = "gs://test-bucket/file123"
        
        mock_file_manager = AsyncMock()
        mock_file_manager.upload_file.return_value = mock_file
        mock_file_manager.poll_until_active.return_value = mock_file
        mock_file_manager.delete_file.return_value = None
        
        mock_response = Mock()
        mock_response.text = json.dumps({
            "action_intensity": {"score": 2, "confidence": 0.8, "reasoning": "Moderate action"},
            "educational_value": {"score": 1, "confidence": 0.7, "reasoning": "Educational"}
        })
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        scorer = GeminiVideoScorer("test-key")
        scorer.file_manager = mock_file_manager
        
        # Test with context
        context = [{"segment": "previous", "timestamp": "00:00-01:00"}]
        result = await scorer.score(temp_video_file, sample_rubric, context=context)
        
        # Verify context was included in prompt (indirectly through successful execution)
        assert isinstance(result, dict)
        assert len(result) > 0
        
        temp_video_file.unlink()

    def test_normalize_score_scale_1_4(self):
        """Test score normalization for 1-4 scale."""
        scorer = GeminiVideoScorer("test-key")
        
        assert scorer._normalize_score(1, DimensionType.SCALE_1_4) == 0.0
        assert scorer._normalize_score(2, DimensionType.SCALE_1_4) == 0.333  # approximately
        assert scorer._normalize_score(3, DimensionType.SCALE_1_4) == 0.667  # approximately
        assert scorer._normalize_score(4, DimensionType.SCALE_1_4) == 1.0

    def test_normalize_score_binary(self):
        """Test score normalization for binary type."""
        scorer = GeminiVideoScorer("test-key")
        
        assert scorer._normalize_score(0, DimensionType.BINARY) == 0.0
        assert scorer._normalize_score(1, DimensionType.BINARY) == 1.0

    def test_normalize_score_numeric(self):
        """Test score normalization for numeric type."""
        scorer = GeminiVideoScorer("test-key")
        
        # Numeric scores should already be 0-1
        assert scorer._normalize_score(0.0, DimensionType.NUMERIC) == 0.0
        assert scorer._normalize_score(0.5, DimensionType.NUMERIC) == 0.5
        assert scorer._normalize_score(1.0, DimensionType.NUMERIC) == 1.0

    def test_normalize_score_clamps_values(self):
        """Test that score normalization clamps values to 0-1."""
        scorer = GeminiVideoScorer("test-key")
        
        # Test clamping
        assert scorer._normalize_score(-0.5, DimensionType.NUMERIC) == 0.0
        assert scorer._normalize_score(1.5, DimensionType.NUMERIC) == 1.0