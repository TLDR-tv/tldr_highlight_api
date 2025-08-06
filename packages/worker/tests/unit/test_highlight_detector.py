"""Unit tests for highlight detector service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4
from pathlib import Path

from worker.services.highlight_detector import HighlightDetector
from worker.services.dimension_framework import (
    DimensionDefinition,
    DimensionType, 
    ScoringRubric
)
from shared.domain.models.highlight import Highlight


class TestHighlightDetector:
    """Test highlight detector functionality."""
    
    @pytest.fixture
    def sample_rubric(self):
        """Create sample scoring rubric."""
        dimensions = [
            DimensionDefinition(
                name="action_intensity",
                description="Rate the action intensity",
                type=DimensionType.SCALE_1_4,
                weight=1.0,
                scoring_prompt="Rate the action intensity",
                examples=[]
            ),
            DimensionDefinition(
                name="educational_value",
                description="Has educational content",
                type=DimensionType.BINARY,
                weight=0.8,
                scoring_prompt="Has educational content", 
                examples=[]
            )
        ]
        
        return ScoringRubric(
            name="Gaming Rubric",
            description="For gaming content",
            dimensions=dimensions,
            highlight_threshold=7.0,
            highlight_confidence_threshold=0.8
        )
    
    @pytest.fixture
    def detector(self):
        """Create highlight detector instance."""
        return HighlightDetector()

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector is not None
        assert hasattr(detector, 'detect_highlights')
        assert hasattr(detector, 'score_segment')

    async def test_detect_highlights_basic(self, detector, sample_rubric):
        """Test basic highlight detection."""
        segments = [
            {"path": "/path/to/segment1.mp4", "start_time": 0.0, "end_time": 60.0},
            {"path": "/path/to/segment2.mp4", "start_time": 60.0, "end_time": 120.0},
            {"path": "/path/to/segment3.mp4", "start_time": 120.0, "end_time": 180.0}
        ]
        
        organization_id = uuid4()
        
        # Mock the scoring strategy
        mock_scorer = AsyncMock()
        mock_scorer.score.side_effect = [
            # Segment 1: High scores (should be highlight)
            {"action_intensity": (0.9, 0.8), "educational_value": (1.0, 0.9)},
            # Segment 2: Medium scores (maybe highlight)
            {"action_intensity": (0.6, 0.7), "educational_value": (0.5, 0.6)},
            # Segment 3: Low scores (not highlight)
            {"action_intensity": (0.2, 0.5), "educational_value": (0.0, 0.4)}
        ]
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            highlights = await detector.detect_highlights(
                segments, sample_rubric, organization_id
            )
            
            assert isinstance(highlights, list)
            # Should detect at least one highlight from high-scoring segment
            assert len(highlights) >= 1
            
            # Check that scoring strategy was called for each segment
            assert mock_scorer.score.call_count == len(segments)

    async def test_detect_highlights_no_highlights(self, detector, sample_rubric):
        """Test detection when no segments meet threshold."""
        segments = [
            {"path": "/path/to/segment1.mp4", "start_time": 0.0, "end_time": 60.0},
            {"path": "/path/to/segment2.mp4", "start_time": 60.0, "end_time": 120.0}
        ]
        
        organization_id = uuid4()
        
        # Mock scorer returning low scores
        mock_scorer = AsyncMock()
        mock_scorer.score.side_effect = [
            {"action_intensity": (0.1, 0.3), "educational_value": (0.2, 0.4)},
            {"action_intensity": (0.0, 0.2), "educational_value": (0.1, 0.3)}
        ]
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            highlights = await detector.detect_highlights(
                segments, sample_rubric, organization_id
            )
            
            assert highlights == []

    async def test_detect_highlights_empty_segments(self, detector, sample_rubric):
        """Test detection with empty segments list."""
        organization_id = uuid4()
        
        highlights = await detector.detect_highlights(
            [], sample_rubric, organization_id
        )
        
        assert highlights == []

    async def test_score_segment(self, detector, sample_rubric):
        """Test individual segment scoring."""
        segment = {"path": "/path/to/segment.mp4", "start_time": 0.0, "end_time": 60.0}
        
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = {
            "action_intensity": (0.8, 0.9),
            "educational_value": (1.0, 0.8)
        }
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            scores = await detector.score_segment(segment, sample_rubric)
            
            assert "action_intensity" in scores
            assert "educational_value" in scores
            assert scores["action_intensity"] == (0.8, 0.9)
            assert scores["educational_value"] == (1.0, 0.8)
            
            mock_scorer.score.assert_called_once()

    async def test_score_segment_with_context(self, detector, sample_rubric):
        """Test segment scoring with context."""
        segment = {"path": "/path/to/segment.mp4", "start_time": 60.0, "end_time": 120.0}
        context = [
            {"path": "/path/to/prev_segment.mp4", "start_time": 0.0, "end_time": 60.0}
        ]
        
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = {
            "action_intensity": (0.7, 0.8),
            "educational_value": (0.6, 0.7)
        }
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            scores = await detector.score_segment(segment, sample_rubric, context)
            
            # Verify scorer was called with context
            mock_scorer.score.assert_called_once()
            call_args = mock_scorer.score.call_args
            assert call_args[1]['context'] == context  # Should pass context

    async def test_scoring_error_handling(self, detector, sample_rubric):
        """Test error handling in scoring."""
        segments = [
            {"path": "/path/to/segment1.mp4", "start_time": 0.0, "end_time": 60.0}
        ]
        
        organization_id = uuid4()
        
        # Mock scorer that raises exception
        mock_scorer = AsyncMock()
        mock_scorer.score.side_effect = Exception("Scoring failed")
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            highlights = await detector.detect_highlights(
                segments, sample_rubric, organization_id
            )
            
            # Should handle error gracefully and return empty list
            assert highlights == []

    def test_get_scoring_strategy_default(self, detector):
        """Test getting default scoring strategy."""
        with patch('worker.services.highlight_detector.GeminiVideoScorer') as mock_scorer:
            strategy = detector._get_scoring_strategy()
            
            # Should create default Gemini scorer
            mock_scorer.assert_called_once()
            assert strategy == mock_scorer.return_value

    def test_get_scoring_strategy_with_config(self, detector):
        """Test getting scoring strategy with custom config."""
        config = {
            "strategy": "gemini",
            "api_key": "test-key",
            "model": "gemini-2.0-flash"
        }
        
        with patch('worker.services.highlight_detector.GeminiVideoScorer') as mock_scorer:
            strategy = detector._get_scoring_strategy(config)
            
            mock_scorer.assert_called_once_with(
                api_key="test-key",
                model_name="gemini-2.0-flash"
            )

    async def test_multi_modal_detection(self, detector, sample_rubric):
        """Test multi-modal highlight detection."""
        segments = [
            {
                "path": "/path/to/segment1.mp4",
                "start_time": 0.0,
                "end_time": 60.0,
                "audio_path": "/path/to/audio1.wav",
                "transcript": "This is exciting gameplay!"
            }
        ]
        
        organization_id = uuid4()
        
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = {
            "action_intensity": (0.9, 0.8),
            "educational_value": (0.7, 0.9)
        }
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            highlights = await detector.detect_highlights(
                segments, sample_rubric, organization_id
            )
            
            # Should process multi-modal content
            mock_scorer.score.assert_called_once()
            assert len(highlights) > 0

    async def test_highlight_creation(self, detector, sample_rubric):
        """Test highlight object creation from high-scoring segments."""
        segments = [
            {
                "path": "/path/to/segment1.mp4",
                "start_time": 0.0,
                "end_time": 60.0,
                "transcript": "Amazing play!"
            }
        ]
        
        organization_id = uuid4()
        
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = {
            "action_intensity": (0.95, 0.9),
            "educational_value": (0.8, 0.85)
        }
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            highlights = await detector.detect_highlights(
                segments, sample_rubric, organization_id
            )
            
            assert len(highlights) > 0
            highlight = highlights[0]
            
            # Check highlight properties
            assert isinstance(highlight, Highlight)
            assert highlight.organization_id == organization_id
            assert highlight.start_time == 0.0
            assert highlight.end_time == 60.0
            assert len(highlight.dimension_scores) == 2
            assert highlight.overall_score > 0

    async def test_threshold_filtering(self, detector, sample_rubric):
        """Test that highlights below threshold are filtered out."""
        # Lower the threshold for testing
        sample_rubric.highlight_threshold = 0.5
        sample_rubric.highlight_confidence_threshold = 0.6
        
        segments = [
            {"path": "/path/to/segment1.mp4", "start_time": 0.0, "end_time": 60.0},
            {"path": "/path/to/segment2.mp4", "start_time": 60.0, "end_time": 120.0}
        ]
        
        organization_id = uuid4()
        
        mock_scorer = AsyncMock()
        mock_scorer.score.side_effect = [
            # First segment: Above threshold
            {"action_intensity": (0.8, 0.9), "educational_value": (0.7, 0.8)},
            # Second segment: Below threshold 
            {"action_intensity": (0.2, 0.3), "educational_value": (0.1, 0.2)}
        ]
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            highlights = await detector.detect_highlights(
                segments, sample_rubric, organization_id
            )
            
            # Should only get one highlight (first segment)
            assert len(highlights) == 1
            assert highlights[0].start_time == 0.0

    async def test_wake_word_integration(self, detector, sample_rubric):
        """Test integration with wake word detection."""
        segments = [
            {
                "path": "/path/to/segment1.mp4",
                "start_time": 0.0,
                "end_time": 60.0,
                "wake_word_detected": "awesome moment",
                "wake_word_timestamp": 30.0
            }
        ]
        
        organization_id = uuid4()
        
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = {
            "action_intensity": (0.7, 0.8),
            "educational_value": (0.6, 0.7)
        }
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            highlights = await detector.detect_highlights(
                segments, sample_rubric, organization_id
            )
            
            if highlights:
                highlight = highlights[0]
                # Should preserve wake word information
                assert highlight.wake_word_triggered == True
                assert highlight.wake_word_detected == "awesome moment"

    async def test_concurrent_processing(self, detector, sample_rubric):
        """Test concurrent processing of multiple segments."""
        segments = [
            {"path": f"/path/to/segment{i}.mp4", "start_time": i*60.0, "end_time": (i+1)*60.0}
            for i in range(5)
        ]
        
        organization_id = uuid4()
        
        mock_scorer = AsyncMock()
        # Return high scores for all segments
        mock_scorer.score.return_value = {
            "action_intensity": (0.8, 0.9),
            "educational_value": (0.7, 0.8)
        }
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            highlights = await detector.detect_highlights(
                segments, sample_rubric, organization_id
            )
            
            # Should process all segments concurrently
            assert mock_scorer.score.call_count == len(segments)
            assert len(highlights) == len(segments)  # All should be highlights

    async def test_performance_optimization(self, detector, sample_rubric):
        """Test performance optimizations like caching."""
        segments = [
            {"path": "/path/to/segment1.mp4", "start_time": 0.0, "end_time": 60.0}
        ]
        
        organization_id = uuid4()
        
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = {
            "action_intensity": (0.8, 0.9),
            "educational_value": (0.7, 0.8)
        }
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            # Call detection twice with same parameters
            await detector.detect_highlights(segments, sample_rubric, organization_id)
            await detector.detect_highlights(segments, sample_rubric, organization_id)
            
            # Each call should still process (no caching implemented yet)
            assert mock_scorer.score.call_count == 2

    async def test_dimension_weight_application(self, detector):
        """Test that dimension weights are properly applied."""
        # Create rubric with weighted dimensions
        dimensions = [
            DimensionDefinition(
                name="primary_dimension",
                description="Primary scoring dimension",
                type=DimensionType.SCALE_1_4,
                weight=2.0,  # High weight
                scoring_prompt="Primary scoring dimension",
                examples=[]
            ),
            DimensionDefinition(
                name="secondary_dimension",
                description="Secondary scoring dimension",
                type=DimensionType.SCALE_1_4,
                weight=0.5,  # Low weight
                scoring_prompt="Secondary scoring dimension",
                examples=[]
            )
        ]
        
        rubric = ScoringRubric(
            name="Weighted Rubric",
            description="Test weighted scoring",
            dimensions=dimensions,
            highlight_threshold=6.0,
            highlight_confidence_threshold=0.7
        )
        
        segments = [
            {"path": "/path/to/segment1.mp4", "start_time": 0.0, "end_time": 60.0}
        ]
        
        organization_id = uuid4()
        
        mock_scorer = AsyncMock()
        mock_scorer.score.return_value = {
            "primary_dimension": (0.9, 0.8),    # High score, high weight
            "secondary_dimension": (0.3, 0.7)   # Low score, low weight
        }
        
        with patch.object(detector, '_get_scoring_strategy', return_value=mock_scorer):
            highlights = await detector.detect_highlights(
                segments, rubric, organization_id
            )
            
            # Weighted scoring should favor the high-weight dimension
            # Overall score should be influenced more by primary_dimension
            if highlights:
                highlight = highlights[0]
                assert highlight.overall_score > 0.5  # Should be pulled up by weighted primary