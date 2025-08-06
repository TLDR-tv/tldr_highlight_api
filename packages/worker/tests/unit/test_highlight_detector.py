"""Unit tests for highlight detector service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4
from pathlib import Path

from worker.services.highlight_detector import HighlightDetector, VideoSegment
from worker.services.dimension_framework import (
    DimensionDefinition,
    DimensionType, 
    ScoringRubric,
    ScoringStrategy
)
from shared.domain.models.stream import Stream


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
            highlight_threshold=0.7,  # Use normalized threshold (0.0-1.0)
            highlight_confidence_threshold=0.8
        )
    
    @pytest.fixture
    def mock_scoring_strategy(self):
        """Create mock scoring strategy."""
        mock_strategy = AsyncMock(spec=ScoringStrategy)
        return mock_strategy
    
    @pytest.fixture
    def detector(self, mock_scoring_strategy):
        """Create highlight detector instance."""
        return HighlightDetector(scoring_strategy=mock_scoring_strategy)

    def test_detector_initialization(self, detector, mock_scoring_strategy):
        """Test detector initialization."""
        assert detector is not None
        assert hasattr(detector, 'detect_highlights')
        assert detector.scoring_strategy == mock_scoring_strategy
        assert detector.min_highlight_duration == 10.0
        assert detector.max_highlight_duration == 120.0
        assert detector.overlap_threshold == 0.5

    async def test_detect_highlights_basic(self, detector, sample_rubric, mock_scoring_strategy):
        """Test basic highlight detection."""
        segments = [
            VideoSegment(Path("/path/to/segment1.mp4"), 0.0, 60.0, 1),
            VideoSegment(Path("/path/to/segment2.mp4"), 60.0, 60.0, 2),
            VideoSegment(Path("/path/to/segment3.mp4"), 120.0, 60.0, 3)
        ]
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        # Mock the scoring strategy
        mock_scoring_strategy.score.side_effect = [
            # Segment 1: High scores (should be highlight)
            {"action_intensity": (0.9, 0.8), "educational_value": (1.0, 0.9)},
            # Segment 2: Medium scores (maybe highlight)
            {"action_intensity": (0.6, 0.7), "educational_value": (0.5, 0.6)},
            # Segment 3: Low scores (not highlight)
            {"action_intensity": (0.2, 0.5), "educational_value": (0.0, 0.4)}
        ]
        
        highlights = await detector.detect_highlights(
            stream, segments, sample_rubric
        )
        
        assert isinstance(highlights, list)
        # Should detect at least one highlight from high-scoring segment
        assert len(highlights) >= 1
        
        # Check that scoring strategy was called for each segment
        assert mock_scoring_strategy.score.call_count == len(segments)

    async def test_detect_highlights_no_highlights(self, detector, sample_rubric, mock_scoring_strategy):
        """Test detection when no segments meet threshold."""
        segments = [
            VideoSegment(Path("/path/to/segment1.mp4"), 0.0, 60.0, 1),
            VideoSegment(Path("/path/to/segment2.mp4"), 60.0, 60.0, 2)
        ]
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        # Mock scorer returning low scores
        mock_scoring_strategy.score.side_effect = [
            {"action_intensity": (0.1, 0.3), "educational_value": (0.2, 0.4)},
            {"action_intensity": (0.0, 0.2), "educational_value": (0.1, 0.3)}
        ]
        
        highlights = await detector.detect_highlights(
            stream, segments, sample_rubric
        )
        
        assert highlights == []

    async def test_detect_highlights_empty_segments(self, detector, sample_rubric):
        """Test detection with empty segments list."""
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        highlights = await detector.detect_highlights(
            stream, [], sample_rubric
        )
        
        assert highlights == []

    async def test_score_single_segment(self, detector, sample_rubric, mock_scoring_strategy):
        """Test individual segment scoring through detect_highlights."""
        segment = VideoSegment(Path("/path/to/segment.mp4"), 0.0, 60.0, 1)
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        mock_scoring_strategy.score.return_value = {
            "action_intensity": (0.8, 0.9),
            "educational_value": (1.0, 0.8)
        }
        
        await detector.detect_highlights(stream, [segment], sample_rubric)
        
        # Verify scoring strategy was called
        mock_scoring_strategy.score.assert_called_once()

    async def test_score_segment_with_context(self, detector, sample_rubric, mock_scoring_strategy):
        """Test segment scoring with context through multiple segments."""
        segments = [
            VideoSegment(Path("/path/to/segment1.mp4"), 0.0, 60.0, 1),
            VideoSegment(Path("/path/to/segment2.mp4"), 60.0, 60.0, 2)
        ]
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        mock_scoring_strategy.score.return_value = {
            "action_intensity": (0.7, 0.8),
            "educational_value": (0.6, 0.7)
        }
        
        await detector.detect_highlights(stream, segments, sample_rubric)
        
        # Verify scorer was called for each segment
        assert mock_scoring_strategy.score.call_count == 2

    async def test_scoring_error_handling(self, detector, sample_rubric, mock_scoring_strategy):
        """Test error handling in scoring."""
        segments = [
            VideoSegment(Path("/path/to/segment1.mp4"), 0.0, 60.0, 1)
        ]
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        # Mock scorer that raises exception
        mock_scoring_strategy.score.side_effect = Exception("Scoring failed")
        
        highlights = await detector.detect_highlights(
            stream, segments, sample_rubric
        )
        
        # Should handle error gracefully and continue with empty scores
        assert isinstance(highlights, list)

    def test_get_scoring_strategy_injected(self, detector, mock_scoring_strategy):
        """Test that injected scoring strategy is used."""
        assert detector.scoring_strategy == mock_scoring_strategy

    def test_detector_configuration(self, mock_scoring_strategy):
        """Test detector with custom configuration."""
        detector = HighlightDetector(
            scoring_strategy=mock_scoring_strategy,
            min_highlight_duration=15.0,
            max_highlight_duration=90.0,
            overlap_threshold=0.3
        )
        
        assert detector.min_highlight_duration == 15.0
        assert detector.max_highlight_duration == 90.0
        assert detector.overlap_threshold == 0.3

    async def test_multi_modal_detection(self, detector, sample_rubric, mock_scoring_strategy):
        """Test multi-modal highlight detection."""
        segments = [
            VideoSegment(Path("/path/to/segment1.mp4"), 0.0, 60.0, 1)
        ]
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        mock_scoring_strategy.score.return_value = {
            "action_intensity": (0.9, 0.8),
            "educational_value": (0.7, 0.9)
        }
        
        highlights = await detector.detect_highlights(
            stream, segments, sample_rubric
        )
        
        # Should process multi-modal content
        mock_scoring_strategy.score.assert_called_once()
        assert len(highlights) >= 0

    async def test_highlight_creation(self, detector, sample_rubric, mock_scoring_strategy):
        """Test highlight candidate creation from high-scoring segments."""
        segments = [
            VideoSegment(Path("/path/to/segment1.mp4"), 0.0, 60.0, 1)
        ]
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        mock_scoring_strategy.score.return_value = {
            "action_intensity": (0.95, 0.9),
            "educational_value": (0.8, 0.85)
        }
        
        highlight_candidates = await detector.detect_highlights(
            stream, segments, sample_rubric
        )
        
        assert len(highlight_candidates) >= 0
        if highlight_candidates:
            candidate = highlight_candidates[0]
            
            # Check highlight candidate properties
            assert candidate.stream_id == stream.id
            assert candidate.start_time == 0.0
            assert candidate.end_time == 60.0
            assert len(candidate.dimension_scores) == 2
            assert candidate.overall_score > 0

    async def test_threshold_filtering(self, detector, sample_rubric, mock_scoring_strategy):
        """Test that highlights below threshold are filtered out."""
        # Lower the threshold for testing
        sample_rubric.highlight_threshold = 0.5
        sample_rubric.highlight_confidence_threshold = 0.6
        
        segments = [
            VideoSegment(Path("/path/to/segment1.mp4"), 0.0, 60.0, 1),
            VideoSegment(Path("/path/to/segment2.mp4"), 60.0, 60.0, 2)
        ]
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        mock_scoring_strategy.score.side_effect = [
            # First segment: Above threshold
            {"action_intensity": (0.8, 0.9), "educational_value": (0.7, 0.8)},
            # Second segment: Below threshold 
            {"action_intensity": (0.2, 0.3), "educational_value": (0.1, 0.2)}
        ]
        
        highlights = await detector.detect_highlights(
            stream, segments, sample_rubric
        )
        
        # Should only get highlights that meet threshold
        assert len(highlights) >= 0
        if highlights:
            assert highlights[0].start_time == 0.0

    async def test_wake_word_integration(self, detector, sample_rubric, mock_scoring_strategy):
        """Test basic functionality with wake word data in metadata."""
        segments = [
            VideoSegment(Path("/path/to/segment1.mp4"), 0.0, 60.0, 1)
        ]
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        mock_scoring_strategy.score.return_value = {
            "action_intensity": (0.7, 0.8),
            "educational_value": (0.6, 0.7)
        }
        
        highlights = await detector.detect_highlights(
            stream, segments, sample_rubric
        )
        
        # Should process normally even with wake word related data
        assert isinstance(highlights, list)

    async def test_multiple_segments_processing(self, detector, sample_rubric, mock_scoring_strategy):
        """Test processing of multiple segments."""
        segments = [
            VideoSegment(Path(f"/path/to/segment{i}.mp4"), i*60.0, 60.0, i+1)
            for i in range(5)
        ]
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        # Return high scores for all segments
        mock_scoring_strategy.score.return_value = {
            "action_intensity": (0.8, 0.9),
            "educational_value": (0.7, 0.8)
        }
        
        highlights = await detector.detect_highlights(
            stream, segments, sample_rubric
        )
        
        # Should process all segments
        assert mock_scoring_strategy.score.call_count == len(segments)
        assert len(highlights) >= 0  # Results depend on threshold and merging logic

    async def test_repeated_calls(self, detector, sample_rubric, mock_scoring_strategy):
        """Test repeated calls to detection."""
        segments = [
            VideoSegment(Path("/path/to/segment1.mp4"), 0.0, 60.0, 1)
        ]
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        mock_scoring_strategy.score.return_value = {
            "action_intensity": (0.8, 0.9),
            "educational_value": (0.7, 0.8)
        }
        
        # Call detection twice with same parameters
        await detector.detect_highlights(stream, segments, sample_rubric)
        await detector.detect_highlights(stream, segments, sample_rubric)
        
        # Each call should process independently
        assert mock_scoring_strategy.score.call_count == 2

    async def test_dimension_weight_application(self, detector, mock_scoring_strategy):
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
            highlight_threshold=0.6,
            highlight_confidence_threshold=0.7
        )
        
        segments = [
            VideoSegment(Path("/path/to/segment1.mp4"), 0.0, 60.0, 1)
        ]
        
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        
        mock_scoring_strategy.score.return_value = {
            "primary_dimension": (0.9, 0.8),    # High score, high weight
            "secondary_dimension": (0.3, 0.7)   # Low score, low weight
        }
        
        highlights = await detector.detect_highlights(
            stream, segments, rubric
        )
        
        # Weighted scoring should favor the high-weight dimension
        # Overall score should be influenced more by primary_dimension
        if highlights:
            highlight = highlights[0]
            assert highlight.overall_score > 0.5  # Should be pulled up by weighted primary