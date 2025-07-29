"""
Unit tests for Google Gemini integration.

Tests the Gemini processor and detector components for video understanding
and highlight detection capabilities.
"""

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import numpy as np

from src.services.content_processing.gemini_processor import (
    GeminiProcessor,
    GeminiProcessorConfig,
    GeminiHighlight,
    GeminiProcessingResult,
    ProcessingMode,
    GeminiModel,
)
from src.services.highlight_detection.gemini_detector import (
    GeminiDetector,
    GeminiDetectionConfig,
)
from src.services.highlight_detection.base_detector import (
    ContentSegment,
    DetectionResult,
)


@pytest.fixture
def gemini_processor_config():
    """Create a test Gemini processor configuration."""
    return GeminiProcessorConfig(
        model_name=GeminiModel.FLASH_2_0,
        default_mode=ProcessingMode.FILE_API,
        chunk_duration_seconds=60,
        max_video_duration_seconds=600,
        temperature=0.3,
        max_retries=2,
    )


@pytest.fixture
def gemini_detection_config():
    """Create a test Gemini detection configuration."""
    return GeminiDetectionConfig(
        highlight_score_threshold=0.5,
        highlight_confidence_threshold=0.6,
        merge_window_seconds=5.0,
        min_highlight_duration=3.0,
        max_highlight_duration=30.0,
        enable_quality_boost=True,
    )


@pytest.fixture
def mock_genai():
    """Mock the Google Generative AI module."""
    with patch("src.services.content_processing.gemini_processor.genai") as mock:
        # Mock model
        model_instance = MagicMock()
        model_instance.generate_content = AsyncMock()
        model_instance.start_chat = AsyncMock()
        model_instance.start_live_session = MagicMock()

        # Mock module functions
        mock.GenerativeModel.return_value = model_instance
        mock.configure = MagicMock()
        mock.upload_file = MagicMock()
        mock.get_file = MagicMock()
        mock.delete_file = MagicMock()

        yield mock


@pytest.fixture
def mock_settings():
    """Mock settings with Gemini API key."""
    with patch("src.services.content_processing.gemini_processor.settings") as mock:
        mock.gemini_api_key = "test-api-key"
        yield mock


@pytest.fixture
def sample_gemini_response():
    """Create a sample Gemini API response."""
    return {
        "highlights": [
            {
                "start_time": 10.0,
                "end_time": 25.0,
                "score": 0.85,
                "confidence": 0.9,
                "reason": "Exciting gameplay moment with multiple eliminations",
                "category": "action",
                "key_moments": ["triple kill", "clutch play"],
                "transcription": "Oh my god! What a play!",
                "visual_description": "Fast-paced combat scene",
                "audio_description": "Intense music with crowd cheering",
            },
            {
                "start_time": 45.0,
                "end_time": 60.0,
                "score": 0.75,
                "confidence": 0.8,
                "reason": "Emotional victory celebration",
                "category": "emotional",
                "key_moments": ["victory", "team celebration"],
                "transcription": "We did it! We won!",
                "visual_description": "Players celebrating",
                "audio_description": "Victory music and cheering",
            },
        ],
        "overall_quality": 0.82,
        "content_summary": "Competitive gaming match with exciting moments and victory",
    }


@pytest.fixture
def sample_video_segment():
    """Create a sample video segment."""
    return ContentSegment(
        segment_id="test-segment-1",
        start_time=0.0,
        end_time=60.0,
        data="test_video.mp4",  # File path
        metadata={"source": "test", "fps": 30, "resolution": "1920x1080"},
    )


class TestGeminiProcessor:
    """Test cases for the Gemini processor."""

    @pytest.mark.asyncio
    async def test_processor_initialization(
        self, gemini_processor_config, mock_genai, mock_settings
    ):
        """Test Gemini processor initialization."""
        processor = GeminiProcessor(gemini_processor_config)

        assert processor.config == gemini_processor_config
        assert processor.model is not None
        mock_genai.configure.assert_called_once_with(api_key="test-api-key")

    @pytest.mark.asyncio
    async def test_process_video_file_with_file_api(
        self, gemini_processor_config, mock_genai, mock_settings, sample_gemini_response
    ):
        """Test processing video file using File API."""
        processor = GeminiProcessor(gemini_processor_config)

        # Mock file upload
        mock_file = MagicMock()
        mock_file.state.name = "ACTIVE"
        mock_file.uri = "gs://test-bucket/test-file"
        mock_file.name = "test-file"
        mock_genai.upload_file.return_value = mock_file
        mock_genai.get_file.return_value = mock_file

        # Mock model response
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_gemini_response)
        processor.model.generate_content = MagicMock(return_value=mock_response)

        # Mock get_media_info
        with patch("src.services.content_processing.gemini_processor.media_processor.get_media_info") as mock_media_info:
            mock_media_info.return_value = {
                "duration": 60.0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "codec": "h264"
            }
            
            # Process video
            result = await processor.process_video_file(
                source="test_video.mp4", mode=ProcessingMode.FILE_API
            )

        # Verify results
        assert isinstance(result, GeminiProcessingResult)
        assert len(result.highlights) == 2
        assert result.overall_quality == 0.82
        assert result.mode_used == ProcessingMode.FILE_API
        assert result.model_used == GeminiModel.FLASH_2_0

        # Verify API calls
        mock_genai.upload_file.assert_called_once()
        processor.model.generate_content.assert_called_once()
        mock_genai.delete_file.assert_called_once_with("test-file")

    @pytest.mark.asyncio
    async def test_process_youtube_url(
        self, gemini_processor_config, mock_genai, mock_settings, sample_gemini_response
    ):
        """Test processing YouTube URL directly."""
        processor = GeminiProcessor(gemini_processor_config)

        # Mock model response
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_gemini_response)
        processor.model.generate_content = MagicMock(return_value=mock_response)

        # Process YouTube URL
        url = "https://www.youtube.com/watch?v=test123"
        result = await processor.process_video_file(
            source=url, mode=ProcessingMode.DIRECT_URL
        )

        # Verify results
        assert result.mode_used == ProcessingMode.DIRECT_URL
        assert len(result.highlights) == 2
        assert result.metadata["url"] == url

        # Verify model was called with URL
        call_args = processor.model.generate_content.call_args[0][0]
        assert url in call_args

    @pytest.mark.asyncio
    async def test_process_video_stream(
        self, gemini_processor_config, mock_genai, mock_settings, sample_gemini_response
    ):
        """Test processing video stream with Live API."""
        processor = GeminiProcessor(gemini_processor_config)

        # Mock live session
        mock_session = MagicMock()
        mock_session.send_message = AsyncMock()
        mock_session.close = MagicMock()
        processor.model.start_live_session.return_value = mock_session

        # Create mock video frames
        async def mock_video_stream():
            for i in range(3):
                frame = MagicMock()
                frame.timestamp = i * 30.0
                frame.data = np.zeros((480, 640, 3), dtype=np.uint8)
                yield frame

        # Process stream
        results = []
        async for result in processor.process_video_stream(mock_video_stream()):
            results.append(result)

        # Verify stream processing
        assert len(results) > 0
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_highlights(
        self, gemini_processor_config, mock_genai, mock_settings
    ):
        """Test parsing highlights from Gemini response."""
        processor = GeminiProcessor(gemini_processor_config)

        response_data = {
            "highlights": [
                {
                    "start_time": 5.0,
                    "end_time": 10.0,
                    "score": 0.9,
                    "confidence": 0.85,
                    "reason": "Test highlight",
                    "category": "action",
                    "key_moments": ["moment1", "moment2"],
                }
            ]
        }

        highlights = processor._parse_highlights(response_data, time_offset=10.0)

        assert len(highlights) == 1
        highlight = highlights[0]
        assert highlight.start_time == 15.0  # With offset
        assert highlight.end_time == 20.0  # With offset
        assert highlight.score == 0.9
        assert highlight.confidence == 0.85
        assert highlight.category == "action"
        assert len(highlight.key_moments) == 2

    @pytest.mark.asyncio
    async def test_error_handling(
        self, gemini_processor_config, mock_genai, mock_settings
    ):
        """Test error handling in processor."""
        processor = GeminiProcessor(gemini_processor_config)

        # Mock API error
        processor.model.generate_content = MagicMock(side_effect=Exception("API Error"))

        with patch("src.services.content_processing.gemini_processor.media_processor.get_media_info") as mock_media_info:
            mock_media_info.return_value = {
                "duration": 60.0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "codec": "h264"
            }
            result = await processor.process_video_file("test_video.mp4")

        assert result.error is not None
        assert "API Error" in result.error
        assert len(result.highlights) == 0
        assert processor.processing_stats["failed_processes"] == 1

    @pytest.mark.asyncio
    async def test_processing_stats(
        self, gemini_processor_config, mock_genai, mock_settings
    ):
        """Test processing statistics tracking."""
        processor = GeminiProcessor(gemini_processor_config)

        # Initial stats
        stats = await processor.get_processing_stats()
        assert stats["total_videos_processed"] == 0

        # Process a video (mocked)
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "highlights": [
                    {
                        "start_time": 0,
                        "end_time": 10,
                        "score": 0.8,
                        "confidence": 0.9,
                        "reason": "test",
                        "category": "action",
                        "key_moments": [],
                    }
                ],
                "overall_quality": 0.8,
                "content_summary": "Test",
            }
        )
        processor.model.generate_content = MagicMock(return_value=mock_response)

        # Mock file operations for File API
        mock_file = MagicMock()
        mock_file.state.name = "ACTIVE"
        mock_file.uri = "gs://test-bucket/test-file"
        mock_file.name = "test-file"
        mock_genai.upload_file.return_value = mock_file
        mock_genai.get_file.return_value = mock_file

        with patch("src.services.content_processing.gemini_processor.media_processor.get_media_info") as mock_media_info:
            mock_media_info.return_value = {
                "duration": 60.0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "codec": "h264"
            }
            await processor.process_video_file("test.mp4")

        # Check updated stats
        stats = await processor.get_processing_stats()
        assert stats["total_videos_processed"] == 1
        assert stats["total_highlights_found"] == 1
        assert stats["successful_processes"] == 1


class TestGeminiDetector:
    """Test cases for the Gemini detector."""

    @pytest.mark.asyncio
    async def test_detector_initialization(
        self, gemini_detection_config, mock_genai, mock_settings
    ):
        """Test Gemini detector initialization."""
        mock_processor = MagicMock()
        
        # Mock the module-level gemini_processor as None initially
        with patch("src.services.highlight_detection.gemini_detector.gemini_processor", None):
            # Mock initialize_gemini_processor to set the global processor
            def mock_init_func(config):
                import src.services.highlight_detection.gemini_detector as gemini_detector_module
                gemini_detector_module.gemini_processor = mock_processor
                
            with patch("src.services.highlight_detection.gemini_detector.initialize_gemini_processor", side_effect=mock_init_func) as mock_init:
                detector = GeminiDetector(gemini_detection_config)

                assert detector.gemini_config == gemini_detection_config
                assert detector.algorithm_name == "GeminiUnifiedDetector"
                assert detector.algorithm_version == "2.0.0"
                assert detector.processor == mock_processor
                mock_init.assert_called_once_with(gemini_detection_config.processor_config)

    @pytest.mark.asyncio
    async def test_detect_features(
        self,
        gemini_detection_config,
        mock_genai,
        mock_settings,
        sample_video_segment,
        sample_gemini_response,
    ):
        """Test feature detection from video segment."""
        # Create detector with mocked processor
        with patch(
            "src.services.highlight_detection.gemini_detector.gemini_processor"
        ) as mock_processor:
            detector = GeminiDetector(gemini_detection_config)

            # Mock processor result
            mock_result = GeminiProcessingResult(
                highlights=[
                    GeminiHighlight(
                        start_time=10.0,
                        end_time=25.0,
                        score=0.85,
                        confidence=0.9,
                        reason="Test highlight",
                        category="action",
                        key_moments=["moment1"],
                        transcription="Test transcription",
                    )
                ],
                overall_quality=0.8,
                content_summary="Test summary",
                processing_time=1.0,
                mode_used=ProcessingMode.FILE_API,
                model_used=GeminiModel.FLASH_2_0,
                total_duration=60.0,
                metadata={},
            )
            mock_processor.process_video_file = AsyncMock(return_value=mock_result)
            detector.processor = mock_processor

            # Detect features
            results = await detector._detect_features(
                sample_video_segment, gemini_detection_config
            )

            assert len(results) == 1
            result = results[0]
            assert isinstance(result, DetectionResult)
            assert result.score > 0.85  # With category weight
            assert result.confidence == 0.9
            assert result.metadata["category"] == "action"
            assert result.metadata["transcription"] == "Test transcription"

    @pytest.mark.asyncio
    async def test_unified_highlight_detection(
        self, gemini_detection_config, mock_genai, mock_settings
    ):
        """Test unified highlight detection from file."""
        with patch(
            "src.services.highlight_detection.gemini_detector.gemini_processor"
        ) as mock_processor:
            detector = GeminiDetector(gemini_detection_config)

            # Mock processor result with multiple highlights
            mock_result = GeminiProcessingResult(
                highlights=[
                    GeminiHighlight(
                        start_time=10.0,
                        end_time=20.0,
                        score=0.8,
                        confidence=0.85,
                        reason="First highlight",
                        category="action",
                        key_moments=["moment1"],
                    ),
                    GeminiHighlight(
                        start_time=25.0,  # Close to previous, should merge
                        end_time=35.0,
                        score=0.75,
                        confidence=0.8,
                        reason="Second highlight",
                        category="action",
                        key_moments=["moment2"],
                    ),
                    GeminiHighlight(
                        start_time=100.0,  # Far from previous, separate
                        end_time=110.0,
                        score=0.9,
                        confidence=0.95,
                        reason="Third highlight",
                        category="emotional",
                        key_moments=["moment3"],
                    ),
                ],
                overall_quality=0.85,
                content_summary="Test",
                processing_time=2.0,
                mode_used=ProcessingMode.FILE_API,
                model_used=GeminiModel.FLASH_2_0,
                total_duration=120.0,
                metadata={},
            )
            mock_processor.process_video_file = AsyncMock(return_value=mock_result)
            detector.processor = mock_processor

            # Detect highlights
            candidates = await detector.detect_highlights_unified("test_video.mp4")

            # Should have 2 candidates after merging
            assert len(candidates) == 2

            # First candidate should be merged
            first = candidates[0]
            assert first.start_time == 10.0
            assert first.end_time == 35.0  # Merged
            assert first.features.get("merged") is True
            # Score should be the max of the merged highlights with category weight applied
            # 0.8 * 1.2 (action weight) = 0.96
            assert first.score == 0.96

            # Second candidate should be separate
            second = candidates[1]
            assert second.start_time == 100.0
            assert second.features.get("category") == "emotional"

    @pytest.mark.asyncio
    async def test_category_weights(
        self, gemini_detection_config, mock_genai, mock_settings
    ):
        """Test category weight application."""
        # Modify config with custom weights
        gemini_detection_config.category_weights = {
            "action": 1.5,
            "emotional": 0.8,
            "general": 1.0,
        }

        with patch(
            "src.services.highlight_detection.gemini_detector.gemini_processor"
        ) as mock_processor:
            detector = GeminiDetector(gemini_detection_config)

            # Create highlights with different categories
            highlights = [
                GeminiHighlight(
                    start_time=0,
                    end_time=10,
                    score=0.6,
                    confidence=0.8,
                    reason="Action",
                    category="action",
                    key_moments=[],
                ),
                GeminiHighlight(
                    start_time=20,
                    end_time=30,
                    score=0.6,
                    confidence=0.8,
                    reason="Emotional",
                    category="emotional",
                    key_moments=[],
                ),
                GeminiHighlight(
                    start_time=40,
                    end_time=50,
                    score=0.6,
                    confidence=0.8,
                    reason="General",
                    category="general",
                    key_moments=[],
                ),
            ]

            mock_result = GeminiProcessingResult(
                highlights=highlights,
                overall_quality=0.7,
                content_summary="Test",
                processing_time=1.0,
                mode_used=ProcessingMode.FILE_API,
                model_used=GeminiModel.FLASH_2_0,
                total_duration=60.0,
                metadata={},
            )
            mock_processor.process_video_file = AsyncMock(return_value=mock_result)
            detector.processor = mock_processor

            candidates = await detector.detect_highlights_unified("test.mp4")

            # Action should have boosted score (0.6 * 1.5 = 0.9)
            action_candidates = [
                c for c in candidates if c.features.get("category") == "action"
            ]
            assert len(action_candidates) == 1
            assert abs(action_candidates[0].score - 0.9) < 0.0001  # Handle floating point precision

            # Emotional should have reduced score (0.6 * 0.8 = 0.48)
            # This should be filtered out due to threshold (0.5)
            emotional_candidates = [
                c for c in candidates if c.features.get("category") == "emotional"
            ]
            assert len(emotional_candidates) == 0

            # General should have normal score (0.6 * 1.0 = 0.6)
            general_candidates = [
                c for c in candidates if c.features.get("category") == "general"
            ]
            assert len(general_candidates) == 1
            assert general_candidates[0].score == 0.6

    @pytest.mark.asyncio
    async def test_quality_boost(
        self, gemini_detection_config, mock_genai, mock_settings
    ):
        """Test quality boost feature."""
        gemini_detection_config.enable_quality_boost = True
        gemini_detection_config.quality_boost_factor = 0.2

        with patch(
            "src.services.highlight_detection.gemini_detector.gemini_processor"
        ) as mock_processor:
            detector = GeminiDetector(gemini_detection_config)

            # High quality content
            mock_result = GeminiProcessingResult(
                highlights=[
                    GeminiHighlight(
                        start_time=0,
                        end_time=10,
                        score=0.7,
                        confidence=0.8,
                        reason="Test",
                        category="action",
                        key_moments=[],
                    )
                ],
                overall_quality=0.9,  # High quality
                content_summary="High quality content",
                processing_time=1.0,
                mode_used=ProcessingMode.FILE_API,
                model_used=GeminiModel.FLASH_2_0,
                total_duration=60.0,
                metadata={},
            )
            mock_processor.process_video_file = AsyncMock(return_value=mock_result)
            detector.processor = mock_processor

            # Convert highlights
            segment = ContentSegment(
                segment_id="test",
                start_time=0,
                end_time=60,
                data="test.mp4",
                metadata={},
            )

            results = await detector._convert_highlights_to_results(
                mock_result.highlights, segment, mock_result, gemini_detection_config
            )

            # Check quality boost applied
            # Base score: 0.7 * 1.2 (action weight) = 0.84
            # Quality boost: (0.9 - 0.7) * 0.2 = 0.04
            # Final: 0.84 + 0.04 = 0.88
            assert len(results) == 1
            assert abs(results[0].score - 0.88) < 0.01

    @pytest.mark.asyncio
    async def test_streaming_mode(
        self, gemini_detection_config, mock_genai, mock_settings
    ):
        """Test streaming mode processing."""
        gemini_detection_config.enable_streaming = True

        with patch(
            "src.services.highlight_detection.gemini_detector.gemini_processor"
        ) as mock_processor:
            detector = GeminiDetector(gemini_detection_config)

            # Mock streaming results
            async def mock_stream_generator(video_chunks=None, audio_chunks=None):
                yield GeminiProcessingResult(
                    highlights=[
                        GeminiHighlight(
                            start_time=0,
                            end_time=10,
                            score=0.8,
                            confidence=0.85,
                            reason="Stream highlight",
                            category="action",
                            key_moments=[],
                        )
                    ],
                    overall_quality=0.8,
                    content_summary="Stream segment",
                    processing_time=0.5,
                    mode_used=ProcessingMode.LIVE_API,
                    model_used=GeminiModel.FLASH_2_0,
                    total_duration=10.0,
                    metadata={},
                )

            mock_processor.process_video_stream = mock_stream_generator
            detector.processor = mock_processor

            # Create mock streams  
            async def mock_video_stream(video_chunks=None):
                for i in range(3):
                    yield MagicMock(timestamp=i * 10.0)

            # Process stream
            candidates = []
            async for candidate in detector.process_stream_with_gemini(
                mock_video_stream()
            ):
                candidates.append(candidate)

            assert len(candidates) == 1
            assert candidates[0].features["is_live"] is True
            assert candidates[0].features["processing_mode"] == "live_streaming"

    @pytest.mark.asyncio
    async def test_cache_functionality(
        self, gemini_detection_config, mock_genai, mock_settings, sample_video_segment
    ):
        """Test segment caching functionality."""
        with patch(
            "src.services.highlight_detection.gemini_detector.gemini_processor"
        ) as mock_processor:
            detector = GeminiDetector(gemini_detection_config)

            mock_result = GeminiProcessingResult(
                highlights=[],
                overall_quality=0.5,
                content_summary="Test",
                processing_time=1.0,
                mode_used=ProcessingMode.FILE_API,
                model_used=GeminiModel.FLASH_2_0,
                total_duration=60.0,
                metadata={},
            )
            mock_processor.process_video_file = AsyncMock(return_value=mock_result)
            detector.processor = mock_processor

            # First call - should process
            results1 = await detector._detect_features(
                sample_video_segment, gemini_detection_config
            )
            assert mock_processor.process_video_file.call_count == 1

            # Second call - should use cache
            results2 = await detector._detect_features(
                sample_video_segment, gemini_detection_config
            )
            assert (
                mock_processor.process_video_file.call_count == 1
            )  # No additional call
            assert results1 == results2

    def test_performance_metrics(
        self, gemini_detection_config, mock_genai, mock_settings
    ):
        """Test performance metrics collection."""
        with patch(
            "src.services.highlight_detection.gemini_detector.gemini_processor"
        ) as mock_processor:
            detector = GeminiDetector(gemini_detection_config)

            # Mock processor stats
            mock_processor.get_processing_stats = AsyncMock(
                return_value={
                    "total_videos_processed": 10,
                    "total_highlights_found": 25,
                    "average_processing_time": 2.5,
                }
            )
            detector.processor = mock_processor

            metrics = detector.get_performance_metrics()

            assert metrics["algorithm"] == "GeminiUnifiedDetector"
            assert metrics["version"] == "2.0.0"
            assert metrics["processor_stats"]["total_videos_processed"] == 10
            assert metrics["cache_size"] == 0
            assert "config" in metrics
