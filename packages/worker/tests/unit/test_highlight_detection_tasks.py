"""Tests for highlight detection tasks."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4, UUID
from pathlib import Path

from worker.tasks.highlight_detection import process_segment_for_highlights
from shared.domain.models.stream import Stream
from shared.domain.models.organization import Organization
from worker.services.dimension_framework import ScoringRubric, DimensionDefinition, DimensionType


class TestProcessSegmentForHighlights:
    """Test the process_segment_for_highlights function."""

    @pytest.fixture
    def mock_stream(self):
        """Create a mock stream."""
        stream = Mock(spec=Stream)
        stream.id = uuid4()
        stream.organization_id = uuid4()
        stream.url = "rtmp://test.stream.com/live"
        return stream

    @pytest.fixture
    def mock_organization(self):
        """Create a mock organization."""
        org = Mock(spec=Organization)
        org.id = uuid4()
        org.name = "Test Organization"
        org.rubric_name = "gaming"
        return org

    @pytest.fixture
    def segment_data(self):
        """Create segment data."""
        return {
            "segment_id": str(uuid4()),
            "video_path": "/tmp/segment_001.mp4",
            "start_time": 0.0,
            "end_time": 120.0,
            "audio_path": "/tmp/segment_001.wav",
            "segment_number": 1
        }

    @pytest.fixture
    def processing_options(self):
        """Create processing options."""
        return {
            "dimension_set_id": "test_dimension_set",
            "type_registry_id": "test_registry",
            "fusion_strategy": "weighted",
            "confidence_threshold": 0.8
        }

    @pytest.fixture
    def context_segments(self):
        """Create context segments."""
        return [
            {
                "segment_id": str(uuid4()),
                "start_time": -120.0,
                "end_time": 0.0,
                "segment_number": 0
            }
        ]

    @pytest.fixture
    def sample_rubric(self):
        """Create a sample scoring rubric."""
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
            highlight_threshold=0.7,
            highlight_confidence_threshold=0.8
        )

    @pytest.mark.asyncio
    async def test_process_segment_for_highlights_success(self, mock_stream, mock_organization, 
                                                          segment_data, processing_options, 
                                                          context_segments, sample_rubric):
        """Test successful highlight processing."""
        stream_id = str(mock_stream.id)
        
        # Mock highlight candidates from detector
        mock_highlight_candidate = Mock()
        mock_highlight_candidate.stream_id = mock_stream.id
        mock_highlight_candidate.start_time = 30.0
        mock_highlight_candidate.end_time = 90.0
        mock_highlight_candidate.duration = 60.0
        mock_highlight_candidate.overall_score = 0.85
        mock_highlight_candidate.confidence = 0.9
        mock_highlight_candidate.dimension_scores = {"action_intensity": 0.9, "educational_value": 0.8}
        
        with patch('worker.tasks.highlight_detection.get_settings') as mock_settings, \
             patch('worker.tasks.highlight_detection.Database') as mock_database, \
             patch('worker.tasks.highlight_detection.StreamRepository') as mock_stream_repo, \
             patch('worker.tasks.highlight_detection.OrganizationRepository') as mock_org_repo, \
             patch('worker.tasks.highlight_detection.HighlightRepository') as mock_highlight_repo, \
             patch('worker.tasks.highlight_detection.RubricRegistry') as mock_registry, \
             patch('worker.tasks.highlight_detection.GeminiVideoScorer') as mock_scorer_class, \
             patch('worker.tasks.highlight_detection.HighlightDetector') as mock_detector_class:
            
            # Setup settings
            mock_settings.return_value.database_url = "sqlite:///test.db"
            mock_settings.return_value.gemini_api_key = "test_api_key"
            
            # Setup database
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            # Setup repositories
            mock_stream_repo_instance = AsyncMock()
            mock_stream_repo_instance.get.return_value = mock_stream
            mock_stream_repo.return_value = mock_stream_repo_instance
            
            mock_org_repo_instance = AsyncMock()
            mock_org_repo_instance.get.return_value = mock_organization
            mock_org_repo.return_value = mock_org_repo_instance
            
            mock_highlight_repo_instance = AsyncMock()
            mock_highlight_repo.return_value = mock_highlight_repo_instance
            
            # Setup rubric registry
            mock_registry.get_rubric.return_value = sample_rubric
            
            # Setup scorer
            mock_scorer = AsyncMock()
            mock_scorer_class.return_value = mock_scorer
            
            # Setup detector
            mock_detector = AsyncMock()
            mock_detector.detect_highlights.return_value = [mock_highlight_candidate]
            mock_detector_class.return_value = mock_detector
            
            # Test
            result = await process_segment_for_highlights(
                stream_id, segment_data, processing_options, context_segments
            )
            
            # Verify
            assert isinstance(result, list)
            mock_stream_repo_instance.get.assert_called_once_with(UUID(stream_id))
            mock_org_repo_instance.get.assert_called_once_with(mock_stream.organization_id)
            mock_registry.get_rubric.assert_called_once_with("gaming")
            mock_scorer_class.assert_called_once_with(api_key="test_api_key")
            mock_detector_class.assert_called_once()
            mock_detector.detect_highlights.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_segment_stream_not_found(self, segment_data, processing_options, context_segments):
        """Test processing when stream not found."""
        stream_id = str(uuid4())
        
        with patch('worker.tasks.highlight_detection.get_settings') as mock_settings, \
             patch('worker.tasks.highlight_detection.Database') as mock_database, \
             patch('worker.tasks.highlight_detection.StreamRepository') as mock_stream_repo:
            
            # Setup settings
            mock_settings.return_value.database_url = "sqlite:///test.db"
            
            # Setup database
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            # Setup repositories
            mock_stream_repo_instance = AsyncMock()
            mock_stream_repo_instance.get.return_value = None
            mock_stream_repo.return_value = mock_stream_repo_instance
            
            # Test
            with pytest.raises(ValueError, match=f"Stream {stream_id} not found"):
                await process_segment_for_highlights(
                    stream_id, segment_data, processing_options, context_segments
                )

    @pytest.mark.asyncio
    async def test_process_segment_organization_not_found(self, mock_stream, segment_data, 
                                                          processing_options, context_segments):
        """Test processing when organization not found."""
        stream_id = str(mock_stream.id)
        
        with patch('worker.tasks.highlight_detection.get_settings') as mock_settings, \
             patch('worker.tasks.highlight_detection.Database') as mock_database, \
             patch('worker.tasks.highlight_detection.StreamRepository') as mock_stream_repo, \
             patch('worker.tasks.highlight_detection.OrganizationRepository') as mock_org_repo:
            
            # Setup settings
            mock_settings.return_value.database_url = "sqlite:///test.db"
            
            # Setup database
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            # Setup repositories
            mock_stream_repo_instance = AsyncMock()
            mock_stream_repo_instance.get.return_value = mock_stream
            mock_stream_repo.return_value = mock_stream_repo_instance
            
            mock_org_repo_instance = AsyncMock()
            mock_org_repo_instance.get.return_value = None
            mock_org_repo.return_value = mock_org_repo_instance
            
            # Test
            with pytest.raises(ValueError, match=f"Organization not found for stream {stream_id}"):
                await process_segment_for_highlights(
                    stream_id, segment_data, processing_options, context_segments
                )

    @pytest.mark.asyncio
    async def test_process_segment_fallback_to_general_rubric(self, mock_stream, mock_organization,
                                                              segment_data, processing_options,
                                                              context_segments, sample_rubric):
        """Test fallback to general rubric when organization rubric not found."""
        stream_id = str(mock_stream.id)
        mock_organization.rubric_name = "nonexistent_rubric"
        
        # Create general rubric
        general_rubric = ScoringRubric(
            name="general",
            description="General purpose rubric",
            dimensions=[],
            highlight_threshold=0.7,
            highlight_confidence_threshold=0.8
        )
        
        with patch('worker.tasks.highlight_detection.get_settings') as mock_settings, \
             patch('worker.tasks.highlight_detection.Database') as mock_database, \
             patch('worker.tasks.highlight_detection.StreamRepository') as mock_stream_repo, \
             patch('worker.tasks.highlight_detection.OrganizationRepository') as mock_org_repo, \
             patch('worker.tasks.highlight_detection.HighlightRepository') as mock_highlight_repo, \
             patch('worker.tasks.highlight_detection.RubricRegistry') as mock_registry, \
             patch('worker.tasks.highlight_detection.GeminiVideoScorer') as mock_scorer_class, \
             patch('worker.tasks.highlight_detection.HighlightDetector') as mock_detector_class, \
             patch('worker.tasks.highlight_detection.logger') as mock_logger:
            
            # Setup settings
            mock_settings.return_value.database_url = "sqlite:///test.db"
            mock_settings.return_value.gemini_api_key = "test_api_key"
            
            # Setup database
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            # Setup repositories
            mock_stream_repo_instance = AsyncMock()
            mock_stream_repo_instance.get.return_value = mock_stream
            mock_stream_repo.return_value = mock_stream_repo_instance
            
            mock_org_repo_instance = AsyncMock()
            mock_org_repo_instance.get.return_value = mock_organization
            mock_org_repo.return_value = mock_org_repo_instance
            
            mock_highlight_repo_instance = AsyncMock()
            mock_highlight_repo.return_value = mock_highlight_repo_instance
            
            # Setup rubric registry - first call returns None, second returns general
            mock_registry.get_rubric.side_effect = [None, general_rubric]
            
            # Setup scorer and detector
            mock_scorer = AsyncMock()
            mock_scorer_class.return_value = mock_scorer
            
            mock_detector = AsyncMock()
            mock_detector.detect_highlights.return_value = []
            mock_detector_class.return_value = mock_detector
            
            # Test
            result = await process_segment_for_highlights(
                stream_id, segment_data, processing_options, context_segments
            )
            
            # Verify fallback behavior
            assert mock_registry.get_rubric.call_count == 2
            mock_registry.get_rubric.assert_any_call("nonexistent_rubric")
            mock_registry.get_rubric.assert_any_call("general")
            mock_logger.warning.assert_called_once()
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_process_segment_no_general_rubric(self, mock_stream, mock_organization,
                                                     segment_data, processing_options, context_segments):
        """Test error when no general rubric is available."""
        stream_id = str(mock_stream.id)
        mock_organization.rubric_name = "nonexistent_rubric"
        
        with patch('worker.tasks.highlight_detection.get_settings') as mock_settings, \
             patch('worker.tasks.highlight_detection.Database') as mock_database, \
             patch('worker.tasks.highlight_detection.StreamRepository') as mock_stream_repo, \
             patch('worker.tasks.highlight_detection.OrganizationRepository') as mock_org_repo, \
             patch('worker.tasks.highlight_detection.HighlightRepository') as mock_highlight_repo, \
             patch('worker.tasks.highlight_detection.RubricRegistry') as mock_registry:
            
            # Setup settings
            mock_settings.return_value.database_url = "sqlite:///test.db"
            
            # Setup database
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            # Setup repositories
            mock_stream_repo_instance = AsyncMock()
            mock_stream_repo_instance.get.return_value = mock_stream
            mock_stream_repo.return_value = mock_stream_repo_instance
            
            mock_org_repo_instance = AsyncMock()
            mock_org_repo_instance.get.return_value = mock_organization
            mock_org_repo.return_value = mock_org_repo_instance
            
            mock_highlight_repo_instance = AsyncMock()
            mock_highlight_repo.return_value = mock_highlight_repo_instance
            
            # Setup rubric registry - both calls return None
            mock_registry.get_rubric.return_value = None
            
            # Test
            with pytest.raises(ValueError, match="General rubric not found in registry"):
                await process_segment_for_highlights(
                    stream_id, segment_data, processing_options, context_segments
                )

    @pytest.mark.asyncio
    async def test_process_segment_detector_error(self, mock_stream, mock_organization,
                                                  segment_data, processing_options,
                                                  context_segments, sample_rubric):
        """Test handling of detector errors."""
        stream_id = str(mock_stream.id)
        
        with patch('worker.tasks.highlight_detection.get_settings') as mock_settings, \
             patch('worker.tasks.highlight_detection.Database') as mock_database, \
             patch('worker.tasks.highlight_detection.StreamRepository') as mock_stream_repo, \
             patch('worker.tasks.highlight_detection.OrganizationRepository') as mock_org_repo, \
             patch('worker.tasks.highlight_detection.HighlightRepository') as mock_highlight_repo, \
             patch('worker.tasks.highlight_detection.RubricRegistry') as mock_registry, \
             patch('worker.tasks.highlight_detection.GeminiVideoScorer') as mock_scorer_class, \
             patch('worker.tasks.highlight_detection.HighlightDetector') as mock_detector_class:
            
            # Setup settings
            mock_settings.return_value.database_url = "sqlite:///test.db"
            mock_settings.return_value.gemini_api_key = "test_api_key"
            
            # Setup database
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            # Setup repositories
            mock_stream_repo_instance = AsyncMock()
            mock_stream_repo_instance.get.return_value = mock_stream
            mock_stream_repo.return_value = mock_stream_repo_instance
            
            mock_org_repo_instance = AsyncMock()
            mock_org_repo_instance.get.return_value = mock_organization
            mock_org_repo.return_value = mock_org_repo_instance
            
            mock_highlight_repo_instance = AsyncMock()
            mock_highlight_repo.return_value = mock_highlight_repo_instance
            
            # Setup rubric registry
            mock_registry.get_rubric.return_value = sample_rubric
            
            # Setup scorer
            mock_scorer = AsyncMock()
            mock_scorer_class.return_value = mock_scorer
            
            # Setup detector to raise error
            mock_detector = AsyncMock()
            mock_detector.detect_highlights.side_effect = Exception("Detector failed")
            mock_detector_class.return_value = mock_detector
            
            # Test - should propagate the exception
            with pytest.raises(Exception, match="Detector failed"):
                await process_segment_for_highlights(
                    stream_id, segment_data, processing_options, context_segments
                )

    @pytest.mark.asyncio
    async def test_process_segment_with_empty_context(self, mock_stream, mock_organization,
                                                      segment_data, processing_options, sample_rubric):
        """Test processing with empty context segments."""
        stream_id = str(mock_stream.id)
        context_segments = []
        
        with patch('worker.tasks.highlight_detection.get_settings') as mock_settings, \
             patch('worker.tasks.highlight_detection.Database') as mock_database, \
             patch('worker.tasks.highlight_detection.StreamRepository') as mock_stream_repo, \
             patch('worker.tasks.highlight_detection.OrganizationRepository') as mock_org_repo, \
             patch('worker.tasks.highlight_detection.HighlightRepository') as mock_highlight_repo, \
             patch('worker.tasks.highlight_detection.RubricRegistry') as mock_registry, \
             patch('worker.tasks.highlight_detection.GeminiVideoScorer') as mock_scorer_class, \
             patch('worker.tasks.highlight_detection.HighlightDetector') as mock_detector_class:
            
            # Setup settings
            mock_settings.return_value.database_url = "sqlite:///test.db"
            mock_settings.return_value.gemini_api_key = "test_api_key"
            
            # Setup database
            mock_session = AsyncMock()
            mock_database.return_value.session.return_value.__aenter__.return_value = mock_session
            
            # Setup repositories
            mock_stream_repo_instance = AsyncMock()
            mock_stream_repo_instance.get.return_value = mock_stream
            mock_stream_repo.return_value = mock_stream_repo_instance
            
            mock_org_repo_instance = AsyncMock()
            mock_org_repo_instance.get.return_value = mock_organization
            mock_org_repo.return_value = mock_org_repo_instance
            
            mock_highlight_repo_instance = AsyncMock()
            mock_highlight_repo.return_value = mock_highlight_repo_instance
            
            # Setup rubric registry
            mock_registry.get_rubric.return_value = sample_rubric
            
            # Setup scorer and detector
            mock_scorer = AsyncMock()
            mock_scorer_class.return_value = mock_scorer
            
            mock_detector = AsyncMock()
            mock_detector.detect_highlights.return_value = []
            mock_detector_class.return_value = mock_detector
            
            # Test
            result = await process_segment_for_highlights(
                stream_id, segment_data, processing_options, context_segments
            )
            
            # Should still work with empty context
            assert isinstance(result, list)
            mock_detector.detect_highlights.assert_called_once()