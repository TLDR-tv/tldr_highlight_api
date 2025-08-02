"""Tests for Gemini AI scoring service."""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
from pathlib import Path
import pytest
import json

from worker.services.gemini_scorer import GeminiVideoScorer as GeminiScorer


class TestGeminiScorer:
    """Test Gemini AI scoring service."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.gemini_api_key = "test-api-key"
        return settings
    
    @pytest.fixture
    def gemini_scorer(self, mock_settings):
        """Create Gemini scorer instance."""
        with patch("worker.services.gemini_scorer.get_settings", return_value=mock_settings):
            return GeminiScorer()
    
    @pytest.mark.asyncio
    async def test_score_video_segment_success(self, gemini_scorer):
        """Test successful video segment scoring."""
        video_path = "/tmp/test_segment.mp4"
        
        with patch("worker.services.gemini_scorer.genai") as mock_genai:
            # Mock file upload
            mock_file = MagicMock()
            mock_file.uri = "https://generativelanguage.googleapis.com/v1/files/test-file"
            mock_genai.upload_file.return_value = mock_file
            
            # Mock model response
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "score": 0.85,
                "confidence": 0.9,
                "reason": "High action sequence with multiple players",
                "key_moments": ["Player elimination at 0:03", "Team fight at 0:07"]
            })
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            
            # Score the segment
            result = await gemini_scorer.score_video_segment(video_path)
            
            # Verify results
            assert result["score"] == 0.85
            assert result["confidence"] == 0.9
            assert "High action sequence" in result["reason"]
            assert len(result["key_moments"]) == 2
            
            # Verify Gemini API calls
            mock_genai.upload_file.assert_called_once_with(
                video_path,
                mime_type="video/mp4",
                display_name="video_segment"
            )
            mock_model.generate_content.assert_called_once()
            
            # Verify file was deleted after use
            mock_file.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_score_video_segment_low_score(self, gemini_scorer):
        """Test scoring video with low action."""
        video_path = "/tmp/test_segment.mp4"
        
        with patch("worker.services.gemini_scorer.genai") as mock_genai:
            # Mock file upload
            mock_file = MagicMock()
            mock_file.uri = "https://generativelanguage.googleapis.com/v1/files/test-file"
            mock_genai.upload_file.return_value = mock_file
            
            # Mock model response with low score
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "score": 0.2,
                "confidence": 0.95,
                "reason": "Low activity, mostly idle gameplay",
                "key_moments": []
            })
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            
            # Score the segment
            result = await gemini_scorer.score_video_segment(video_path)
            
            # Verify low score
            assert result["score"] == 0.2
            assert result["confidence"] == 0.95
            assert "Low activity" in result["reason"]
            assert len(result["key_moments"]) == 0
    
    @pytest.mark.asyncio
    async def test_score_video_segment_api_error(self, gemini_scorer):
        """Test handling Gemini API errors."""
        video_path = "/tmp/test_segment.mp4"
        
        with patch("worker.services.gemini_scorer.genai") as mock_genai:
            # Mock API error
            mock_genai.upload_file.side_effect = Exception("API rate limit exceeded")
            
            # Should raise exception
            with pytest.raises(Exception, match="API rate limit exceeded"):
                await gemini_scorer.score_video_segment(video_path)
    
    @pytest.mark.asyncio
    async def test_score_video_segment_invalid_response(self, gemini_scorer):
        """Test handling invalid model responses."""
        video_path = "/tmp/test_segment.mp4"
        
        with patch("worker.services.gemini_scorer.genai") as mock_genai:
            # Mock file upload
            mock_file = MagicMock()
            mock_file.uri = "https://generativelanguage.googleapis.com/v1/files/test-file"
            mock_genai.upload_file.return_value = mock_file
            
            # Mock model response with invalid JSON
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Invalid JSON response"
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            
            # Should handle gracefully and return default score
            result = await gemini_scorer.score_video_segment(video_path)
            
            # Should return default low score
            assert result["score"] == 0.0
            assert "Failed to parse" in result["reason"]
            
            # File should still be cleaned up
            mock_file.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_score_video_segment_file_cleanup_on_error(self, gemini_scorer):
        """Test that files are cleaned up even on errors."""
        video_path = "/tmp/test_segment.mp4"
        
        with patch("worker.services.gemini_scorer.genai") as mock_genai:
            # Mock file upload
            mock_file = MagicMock()
            mock_file.uri = "https://generativelanguage.googleapis.com/v1/files/test-file"
            mock_genai.upload_file.return_value = mock_file
            
            # Mock model to raise error
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("Model error")
            mock_genai.GenerativeModel.return_value = mock_model
            
            # Should raise exception
            with pytest.raises(Exception, match="Model error"):
                await gemini_scorer.score_video_segment(video_path)
            
            # File should still be cleaned up
            mock_file.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_file_context_manager(self, gemini_scorer):
        """Test the upload file context manager."""
        video_path = "/tmp/test_segment.mp4"
        
        with patch("worker.services.gemini_scorer.genai") as mock_genai:
            # Mock file upload
            mock_file = MagicMock()
            mock_file.uri = "https://generativelanguage.googleapis.com/v1/files/test-file"
            mock_genai.upload_file.return_value = mock_file
            
            # Use context manager
            async with gemini_scorer._upload_file(video_path) as uploaded_file:
                assert uploaded_file == mock_file
                # File should not be deleted yet
                mock_file.delete.assert_not_called()
            
            # File should be deleted after context exit
            mock_file.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_score_video_segment_with_metadata(self, gemini_scorer):
        """Test scoring with additional metadata."""
        video_path = "/tmp/test_segment.mp4"
        metadata = {
            "stream_id": "test-stream-123",
            "segment_index": 5,
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        with patch("worker.services.gemini_scorer.genai") as mock_genai:
            # Mock file upload
            mock_file = MagicMock()
            mock_file.uri = "https://generativelanguage.googleapis.com/v1/files/test-file"
            mock_genai.upload_file.return_value = mock_file
            
            # Mock model response
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "score": 0.75,
                "confidence": 0.85,
                "reason": "Moderate action with strategic gameplay",
                "key_moments": ["Objective captured at 0:05"]
            })
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            
            # Score with metadata
            result = await gemini_scorer.score_video_segment(video_path, metadata=metadata)
            
            # Verify prompt includes metadata context
            call_args = mock_model.generate_content.call_args
            prompt_content = str(call_args[0][0])
            assert "segment #5" in prompt_content or "segment index" in prompt_content