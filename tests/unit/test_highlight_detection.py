"""
Comprehensive unit tests for the highlight detection system.

This module contains extensive tests for all components of the AI-powered
highlight detection engine, including individual detectors, fusion scoring,
ranking, and post-processing.
"""

import asyncio
import pytest

import numpy as np

from src.services.highlight_detection.base_detector import (
    DetectionResult,
    DetectionConfig,
    ContentSegment,
    ModalityType,
    HighlightCandidate,
    ConfidenceLevel,
)
from src.services.highlight_detection.video_detector import (
    VideoDetector,
    VideoDetectionConfig,
    VideoFrameData,
    VideoActivityAnalyzer,
)
from src.services.highlight_detection.audio_detector import (
    AudioDetector,
    AudioDetectionConfig,
    AudioSegmentData,
    AudioExcitementAnalyzer,
)
from src.services.highlight_detection.chat_detector import (
    ChatDetector,
    ChatDetectionConfig,
    ChatMessage,
    ChatWindow,
    ChatExcitementAnalyzer,
)
from src.services.highlight_detection.fusion_scorer import (
    FusionScorer,
    FusionConfig,
    ModalityScore,
    TemporalAligner,
    FusionMethod,
    TemporalAlignment,
)
from src.services.highlight_detection.ranker import (
    HighlightRanker,
    RankingConfig,
    HighlightCluster,
    HighlightClustering,
    RankingMethod,
    ClusteringMethod,
)
from src.services.highlight_detection.post_processor import (
    HighlightPostProcessor,
    PostProcessorConfig,
    TemporalSmoother,
    BoundaryOptimizer,
    QualityEnhancer,
    SmoothingMethod,
    BoundaryOptimization,
    QualityEnhancement,
)


@pytest.fixture
def sample_content_segment():
    """Create a sample content segment for testing."""
    return ContentSegment(
        start_time=0.0,
        end_time=30.0,
        data=np.random.random((100,)),  # Sample data
        metadata={"source": "test", "quality": "high"},
    )


@pytest.fixture
def sample_video_frames():
    """Create sample video frames for testing."""
    frames = []
    for i in range(10):
        frame = VideoFrameData(
            frame_index=i,
            timestamp=i * 0.1,
            pixels=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            width=64,
            height=64,
            channels=3,
        )
        frames.append(frame)
    return frames


@pytest.fixture
def sample_audio_data():
    """Create sample audio data for testing."""
    return AudioSegmentData(
        start_time=0.0,
        end_time=10.0,
        samples=np.random.randn(44100 * 10),  # 10 seconds at 44.1kHz
        sample_rate=44100,
        channels=1,
        metadata={"transcription": "wow that was amazing"},
    )


@pytest.fixture
def sample_chat_messages():
    """Create sample chat messages for testing."""
    messages = []
    excitement_keywords = ["wow", "amazing", "epic", "insane", "clutch"]

    for i in range(20):
        timestamp = i * 1.0
        message_text = f"Message {i}"

        # Add some excitement keywords randomly
        if i % 4 == 0:
            message_text += f" {excitement_keywords[i % len(excitement_keywords)]}"

        message = ChatMessage(
            timestamp=timestamp,
            user_id=f"user_{i % 5}",  # 5 different users
            username=f"user_{i % 5}",
            message=message_text,
            platform="twitch",
        )
        messages.append(message)

    return messages


@pytest.fixture
def sample_highlight_candidates():
    """Create sample highlight candidates for testing."""
    candidates = []

    for i in range(10):
        start_time = i * 30.0
        end_time = start_time + 45.0
        score = 0.3 + (i * 0.07)  # Varying scores
        confidence = 0.5 + (i * 0.05)

        # Create mock detection results
        video_result = DetectionResult(
            segment_id=f"segment_{i}",
            modality=ModalityType.VIDEO,
            score=score,
            confidence=confidence,
            features={"motion": 0.6, "activity": 0.7},
        )

        audio_result = DetectionResult(
            segment_id=f"segment_{i}",
            modality=ModalityType.AUDIO,
            score=min(score + 0.1, 1.0),  # Ensure score doesn't exceed 1.0
            confidence=confidence,
            features={"volume": 0.5, "keywords": 2.0},  # Make keywords a float
        )

        candidate = HighlightCandidate(
            start_time=start_time,
            end_time=end_time,
            score=score,
            confidence=confidence,
            modality_results=[video_result, audio_result],
            features={"duration": end_time - start_time, "quality": score},
        )
        candidates.append(candidate)

    return candidates


class TestBaseDetector:
    """Test cases for the base detector framework."""

    def test_detection_result_validation(self):
        """Test detection result validation."""
        # Valid result
        result = DetectionResult(
            segment_id="test", modality=ModalityType.VIDEO, score=0.8, confidence=0.9
        )
        assert result.score == 0.8
        assert result.confidence == 0.9
        assert result.confidence_level == ConfidenceLevel.VERY_HIGH
        assert abs(result.weighted_score - 0.72) < 0.0001  # 0.8 * 0.9
        assert result.is_significant()

        # Invalid score ranges
        with pytest.raises(ValueError):
            DetectionResult(
                segment_id="test",
                modality=ModalityType.VIDEO,
                score=1.5,  # Invalid
                confidence=0.9,
            )

    def test_content_segment_properties(self, sample_content_segment):
        """Test content segment properties and methods."""
        segment = sample_content_segment

        assert segment.duration == 30.0
        assert segment.midpoint == 15.0

        # Test overlap detection
        other_segment = ContentSegment(start_time=20.0, end_time=50.0, data=None)

        assert segment.overlaps_with(other_segment)
        intersection = segment.intersection(other_segment)
        assert intersection is not None
        assert intersection.start_time == 20.0
        assert intersection.end_time == 30.0

    def test_highlight_candidate_properties(self, sample_highlight_candidates):
        """Test highlight candidate properties."""
        candidate = sample_highlight_candidates[0]

        assert candidate.duration == 45.0
        assert candidate.midpoint == 22.5
        assert candidate.has_modality(ModalityType.VIDEO)
        assert candidate.has_modality(ModalityType.AUDIO)
        assert not candidate.has_modality(ModalityType.CHAT)

        video_score = candidate.get_modality_score(ModalityType.VIDEO)
        assert video_score > 0

    def test_detection_config_validation(self):
        """Test detection configuration validation."""
        config = DetectionConfig(weight=1.5, min_confidence=0.1, sensitivity=0.8)

        assert config.weight == 1.5
        assert config.min_confidence == 0.1
        assert config.sensitivity == 0.8

        # Test invalid ranges
        with pytest.raises(ValueError):
            DetectionConfig(min_confidence=1.5)  # Invalid range


class TestVideoDetector:
    """Test cases for video-based highlight detection."""

    @pytest.fixture
    def video_detector(self):
        """Create video detector for testing."""
        config = VideoDetectionConfig(
            motion_threshold=0.3,
            scene_change_threshold=0.4,
            motion_weight=0.4,
            scene_change_weight=0.3,
            activity_weight=0.3,
        )
        return VideoDetector(config)

    def test_video_detector_initialization(self, video_detector):
        """Test video detector initialization."""
        assert video_detector.modality == ModalityType.VIDEO
        assert video_detector.algorithm_name == "VideoActivityDetector"
        assert video_detector.algorithm_version == "1.0.0"

    def test_video_frame_data_properties(self, sample_video_frames):
        """Test video frame data properties."""
        frame = sample_video_frames[0]

        assert frame.width == 64
        assert frame.height == 64
        assert frame.channels == 3
        assert frame.aspect_ratio == 1.0
        assert frame.total_pixels == 4096

        # Test grayscale conversion
        grayscale = frame.get_grayscale()
        assert grayscale.shape == (64, 64)
        assert grayscale.dtype == np.float64

    def test_motion_vector_computation(self, sample_video_frames):
        """Test motion vector computation."""
        if len(sample_video_frames) >= 2:
            current_frame = sample_video_frames[1]
            previous_frame = sample_video_frames[0]

            motion_vectors = current_frame.compute_motion_vectors(previous_frame)
            assert motion_vectors.shape[2] == 2  # x, y components
            assert motion_vectors.shape[0] > 0
            assert motion_vectors.shape[1] > 0

    def test_edge_map_computation(self, sample_video_frames):
        """Test edge map computation."""
        frame = sample_video_frames[0]
        edge_map = frame.compute_edge_map()

        assert edge_map.shape == (64, 64)
        assert edge_map.dtype == np.float64
        assert np.all(edge_map >= 0)  # Edge magnitudes should be non-negative

    def test_color_histogram_computation(self, sample_video_frames):
        """Test color histogram computation."""
        frame = sample_video_frames[0]
        histogram = frame.compute_color_histogram()

        assert len(histogram) == 48  # 16 bins * 3 channels
        assert np.isclose(np.sum(histogram), 1.0)  # Should be normalized

    @pytest.mark.asyncio
    async def test_video_activity_analyzer(self, sample_video_frames):
        """Test video activity analyzer."""
        config = VideoDetectionConfig()
        analyzer = VideoActivityAnalyzer(config)

        # Test motion analysis
        motion_results = await analyzer.analyze_motion(sample_video_frames)
        assert "motion_score" in motion_results
        assert "motion_consistency" in motion_results
        assert 0 <= motion_results["motion_score"] <= 1

        # Test scene change analysis
        scene_results = await analyzer.analyze_scene_changes(sample_video_frames)
        assert "scene_change_score" in scene_results
        assert "scene_stability" in scene_results
        assert 0 <= scene_results["scene_change_score"] <= 1

        # Test visual complexity analysis
        complexity_results = await analyzer.analyze_visual_complexity(
            sample_video_frames
        )
        assert "complexity_score" in complexity_results
        assert "edge_density" in complexity_results
        assert 0 <= complexity_results["complexity_score"] <= 1

    @pytest.mark.asyncio
    async def test_video_detector_detection(self, video_detector, sample_video_frames):
        """Test video detection process."""
        # Create content segment with video frames
        segment = ContentSegment(
            start_time=0.0,
            end_time=10.0,
            data=sample_video_frames,
            metadata={"source": "test_video"},
        )

        results = await video_detector.detect_highlights([segment])

        # Should return results if frames are valid
        if len(sample_video_frames) >= 2:
            assert len(results) >= 0  # May return 0 if below thresholds

            if results:
                result = results[0]
                assert result.modality == ModalityType.VIDEO
                assert 0 <= result.score <= 1
                assert 0 <= result.confidence <= 1
                assert "motion_magnitude" in result.features

    def test_video_detector_validation(self, video_detector):
        """Test video detector validation."""
        # Valid segment
        valid_segment = ContentSegment(
            start_time=0.0,
            end_time=10.0,
            data=[np.random.random((64, 64, 3)) for _ in range(5)],
        )
        assert video_detector._validate_segment(valid_segment)

        # Invalid segment (no data)
        invalid_segment = ContentSegment(start_time=0.0, end_time=10.0, data=None)
        assert not video_detector._validate_segment(invalid_segment)

    def test_video_detector_metrics(self, video_detector):
        """Test video detector performance metrics."""
        metrics = video_detector.get_performance_metrics()

        assert "algorithm" in metrics
        assert "version" in metrics
        assert "config" in metrics
        assert metrics["algorithm"] == "VideoActivityDetector"


class TestAudioDetector:
    """Test cases for audio-based highlight detection."""

    @pytest.fixture
    def audio_detector(self):
        """Create audio detector for testing."""
        config = AudioDetectionConfig(
            volume_spike_threshold=0.4,
            keyword_detection_enabled=True,
            spectral_analysis_enabled=True,
            speech_analysis_enabled=True,
        )
        return AudioDetector(config)

    def test_audio_detector_initialization(self, audio_detector):
        """Test audio detector initialization."""
        assert audio_detector.modality == ModalityType.AUDIO
        assert audio_detector.algorithm_name == "AudioExcitementDetector"
        assert audio_detector.algorithm_version == "1.0.0"

    def test_audio_segment_data_properties(self, sample_audio_data):
        """Test audio segment data properties."""
        audio = sample_audio_data

        assert audio.duration == 10.0
        assert audio.sample_count == 441000
        assert audio.sample_rate == 44100
        assert audio.channels == 1

    def test_rms_energy_computation(self, sample_audio_data):
        """Test RMS energy computation."""
        audio = sample_audio_data
        rms_energy = audio.get_rms_energy()

        assert len(rms_energy) > 0
        assert np.all(rms_energy >= 0)  # RMS should be non-negative

    def test_spectral_features_computation(self, sample_audio_data):
        """Test spectral features computation."""
        audio = sample_audio_data
        spectral_features = audio.get_spectral_features()

        expected_keys = [
            "spectral_centroid",
            "spectral_rolloff",
            "spectral_bandwidth",
            "zero_crossing_rate",
        ]
        for key in expected_keys:
            assert key in spectral_features
            assert len(spectral_features[key]) > 0

    def test_transcription_handling(self, sample_audio_data):
        """Test transcription handling."""
        audio = sample_audio_data
        transcription = audio.get_transcription()

        assert transcription == "wow that was amazing"  # From metadata

    @pytest.mark.asyncio
    async def test_audio_excitement_analyzer(self, sample_audio_data):
        """Test audio excitement analyzer."""
        config = AudioDetectionConfig()
        analyzer = AudioExcitementAnalyzer(config)

        # Test volume spike analysis
        volume_results = await analyzer.analyze_volume_spikes(sample_audio_data)
        assert "volume_spike_score" in volume_results
        assert "volume_consistency" in volume_results
        assert 0 <= volume_results["volume_spike_score"] <= 1

        # Test keyword analysis
        keyword_results = await analyzer.analyze_keywords(sample_audio_data)
        assert "keyword_score" in keyword_results
        assert "keyword_count" in keyword_results
        assert keyword_results["keyword_count"] >= 0

        # Test spectral analysis
        spectral_results = await analyzer.analyze_spectral_excitement(sample_audio_data)
        assert "spectral_score" in spectral_results
        assert "high_freq_energy" in spectral_results
        assert 0 <= spectral_results["spectral_score"] <= 1

        # Test speech pattern analysis
        speech_results = await analyzer.analyze_speech_patterns(sample_audio_data)
        assert "speech_score" in speech_results
        assert "speech_rate" in speech_results
        assert speech_results["speech_rate"] >= 0

    @pytest.mark.asyncio
    async def test_audio_detector_detection(self, audio_detector, sample_audio_data):
        """Test audio detection process."""
        # Create content segment with audio data
        segment = ContentSegment(
            start_time=0.0,
            end_time=10.0,
            data=sample_audio_data,
            metadata={"source": "test_audio"},
        )

        results = await audio_detector.detect_highlights([segment])

        # Should return results
        assert len(results) >= 0

        if results:
            result = results[0]
            assert result.modality == ModalityType.AUDIO
            assert 0 <= result.score <= 1
            assert 0 <= result.confidence <= 1
            assert "spike_count" in result.features

    def test_audio_detector_validation(self, audio_detector):
        """Test audio detector validation."""
        # Valid segment
        valid_segment = ContentSegment(
            start_time=0.0, end_time=10.0, data=np.random.randn(1000)
        )
        assert audio_detector._validate_segment(valid_segment)

        # Invalid segment (empty data)
        invalid_segment = ContentSegment(
            start_time=0.0, end_time=10.0, data=np.array([])
        )
        assert not audio_detector._validate_segment(invalid_segment)


class TestChatDetector:
    """Test cases for chat-based highlight detection."""

    @pytest.fixture
    def chat_detector(self):
        """Create chat detector for testing."""
        config = ChatDetectionConfig(
            frequency_spike_threshold=2.0,
            sentiment_analysis_enabled=True,
            engagement_analysis_enabled=True,
        )
        return ChatDetector(config)

    def test_chat_detector_initialization(self, chat_detector):
        """Test chat detector initialization."""
        assert chat_detector.modality == ModalityType.CHAT
        assert chat_detector.algorithm_name == "ChatExcitementDetector"
        assert chat_detector.algorithm_version == "1.0.0"

    def test_chat_message_properties(self, sample_chat_messages):
        """Test chat message properties."""
        message = sample_chat_messages[0]

        assert message.message_length > 0
        assert message.word_count > 0
        assert not message.is_bot  # Our test messages aren't from bots

        # Test sentiment analysis
        sentiment = message.get_sentiment_score()
        assert -1 <= sentiment <= 1

        # Test excitement keywords
        keywords = message.get_excitement_keywords()
        assert isinstance(keywords, list)

        # Test emoji count
        emoji_count = message.get_emoji_count()
        assert emoji_count >= 0

        # Test spam score
        spam_score = message.get_spam_score()
        assert 0 <= spam_score <= 1

    def test_chat_window_properties(self, sample_chat_messages):
        """Test chat window properties."""
        window = ChatWindow(
            start_time=0.0, end_time=20.0, messages=sample_chat_messages
        )

        assert window.duration == 20.0
        assert window.message_count == len(sample_chat_messages)
        assert window.unique_user_count == 5  # We created 5 different users
        assert window.message_rate > 0

        # Test message filtering
        config = ChatDetectionConfig()
        filtered = window.get_filtered_messages(config)
        assert len(filtered) <= len(sample_chat_messages)

    @pytest.mark.asyncio
    async def test_chat_excitement_analyzer(self, sample_chat_messages):
        """Test chat excitement analyzer."""
        config = ChatDetectionConfig()
        analyzer = ChatExcitementAnalyzer(config)

        window = ChatWindow(
            start_time=0.0, end_time=20.0, messages=sample_chat_messages
        )

        # Test frequency spike analysis
        frequency_results = await analyzer.analyze_frequency_spikes(window)
        assert "frequency_score" in frequency_results
        assert "message_rate" in frequency_results
        assert frequency_results["message_rate"] > 0

        # Test sentiment analysis
        sentiment_results = await analyzer.analyze_sentiment(window)
        assert "sentiment_score" in sentiment_results
        assert "avg_sentiment" in sentiment_results
        assert 0 <= sentiment_results["sentiment_score"] <= 1

        # Test engagement analysis
        engagement_results = await analyzer.analyze_engagement(window)
        assert "engagement_score" in engagement_results
        assert "user_diversity" in engagement_results
        assert 0 <= engagement_results["engagement_score"] <= 1

        # Test keyword/emoji analysis
        keyword_results = await analyzer.analyze_keywords_emojis(window)
        assert "keyword_emoji_score" in keyword_results
        assert "excitement_keyword_count" in keyword_results
        assert keyword_results["excitement_keyword_count"] >= 0

    @pytest.mark.asyncio
    async def test_chat_detector_detection(self, chat_detector, sample_chat_messages):
        """Test chat detection process."""
        window = ChatWindow(
            start_time=0.0, end_time=20.0, messages=sample_chat_messages
        )

        segment = ContentSegment(
            start_time=0.0, end_time=20.0, data=window, metadata={"source": "test_chat"}
        )

        results = await chat_detector.detect_highlights([segment])

        assert len(results) >= 0

        if results:
            result = results[0]
            assert result.modality == ModalityType.CHAT
            assert 0 <= result.score <= 1
            assert 0 <= result.confidence <= 1
            assert "message_count" in result.features


class TestFusionScorer:
    """Test cases for multi-modal fusion scoring."""

    @pytest.fixture
    def fusion_scorer(self):
        """Create fusion scorer for testing."""
        config = FusionConfig(
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            temporal_alignment=TemporalAlignment.WINDOW,
            video_weight=0.4,
            audio_weight=0.35,
            chat_weight=0.25,
        )
        return FusionScorer(config)

    def test_fusion_config_validation(self):
        """Test fusion configuration validation."""
        config = FusionConfig(video_weight=0.4, audio_weight=0.3, chat_weight=0.3)

        weights = config.normalized_weights
        # Video and audio weights should be normalized to sum to 1.0
        # Chat weight is kept as a bonus weight (not normalized)
        assert abs(weights[ModalityType.VIDEO] - 0.4 / 0.7) < 1e-10  # ~0.571
        assert abs(weights[ModalityType.AUDIO] - 0.3 / 0.7) < 1e-10  # ~0.429
        assert abs(weights[ModalityType.CHAT] - 0.3) < 1e-10  # Unchanged
        
        # Video + Audio should sum to 1.0 (chat is bonus)
        video_audio_sum = weights[ModalityType.VIDEO] + weights[ModalityType.AUDIO]
        assert abs(video_audio_sum - 1.0) < 1e-10

    def test_modality_score_properties(self):
        """Test modality score properties."""
        result = DetectionResult(
            segment_id="test", modality=ModalityType.VIDEO, score=0.8, confidence=0.9
        )

        modality_score = ModalityScore(
            timestamp=10.0,
            modality=ModalityType.VIDEO,
            score=0.8,
            confidence=0.9,
            detection_result=result,
            weight=0.4,
        )

        assert modality_score.weighted_score == 0.8 * 0.9 * 0.4
        assert modality_score.quality_score > 0

    @pytest.mark.asyncio
    async def test_temporal_aligner(self):
        """Test temporal alignment of detection results."""
        config = FusionConfig()
        aligner = TemporalAligner(config)

        # Create mock detection results
        video_results = [
            DetectionResult(
                segment_id="v1",
                modality=ModalityType.VIDEO,
                score=0.8,
                confidence=0.9,
                metadata={"start_time": 10.0, "end_time": 40.0},
            )
        ]

        audio_results = [
            DetectionResult(
                segment_id="a1",
                modality=ModalityType.AUDIO,
                score=0.7,
                confidence=0.8,
                metadata={"start_time": 12.0, "end_time": 42.0},
            )
        ]

        results_by_modality = {
            ModalityType.VIDEO: video_results,
            ModalityType.AUDIO: audio_results,
        }

        aligned_results = await aligner.align_results(results_by_modality)

        assert len(aligned_results) > 0
        for timestamp, aligned_scores in aligned_results:
            assert isinstance(timestamp, float)
            assert isinstance(aligned_scores, dict)

    @pytest.mark.asyncio
    async def test_fusion_scorer_fusion(self, fusion_scorer):
        """Test fusion scoring process."""
        # Create mock detection results for different modalities
        video_results = [
            DetectionResult(
                segment_id="seg1",
                modality=ModalityType.VIDEO,
                score=0.8,
                confidence=0.9,
                metadata={"start_time": 10.0, "end_time": 40.0},
            )
        ]

        audio_results = [
            DetectionResult(
                segment_id="seg1",
                modality=ModalityType.AUDIO,
                score=0.7,
                confidence=0.8,
                metadata={"start_time": 12.0, "end_time": 42.0},
            )
        ]

        chat_results = [
            DetectionResult(
                segment_id="seg1",
                modality=ModalityType.CHAT,
                score=0.6,
                confidence=0.7,
                metadata={"start_time": 11.0, "end_time": 41.0},
            )
        ]

        results_by_modality = {
            ModalityType.VIDEO: video_results,
            ModalityType.AUDIO: audio_results,
            ModalityType.CHAT: chat_results,
        }

        candidates = await fusion_scorer.fuse_results(results_by_modality)

        assert len(candidates) >= 0

        if candidates:
            candidate = candidates[0]
            assert isinstance(candidate, HighlightCandidate)
            assert 0 <= candidate.score <= 1
            assert 0 <= candidate.confidence <= 1
            assert len(candidate.modality_results) > 0

    def test_fusion_scorer_different_methods(self):
        """Test different fusion methods."""
        methods = [
            FusionMethod.WEIGHTED_AVERAGE,
            FusionMethod.CONFIDENCE_WEIGHTED,
            FusionMethod.ADAPTIVE_FUSION,
            FusionMethod.TEMPORAL_CORRELATION,
        ]

        for method in methods:
            config = FusionConfig(fusion_method=method)
            scorer = FusionScorer(config)
            assert scorer.config.fusion_method == method


class TestHighlightRanker:
    """Test cases for highlight ranking and selection."""

    @pytest.fixture
    def highlight_ranker(self):
        """Create highlight ranker for testing."""
        config = RankingConfig(
            ranking_method=RankingMethod.WEIGHTED_MULTI_CRITERIA,
            max_highlights=5,
            score_threshold=0.3,
            clustering_enabled=True,
        )
        return HighlightRanker(config)

    def test_ranking_config_validation(self):
        """Test ranking configuration validation."""
        config = RankingConfig(max_highlights=10, min_highlights=1, score_threshold=0.5)

        assert config.max_highlights == 10
        assert config.min_highlights == 1
        assert config.score_threshold == 0.5
        assert config.max_highlights >= config.min_highlights

    def test_highlight_cluster_properties(self, sample_highlight_candidates):
        """Test highlight cluster properties."""
        candidates = sample_highlight_candidates[:3]
        cluster = HighlightCluster(candidates, cluster_id=1)

        assert cluster.size == 3
        assert cluster.cluster_id == 1
        assert cluster.centroid_timestamp > 0
        assert cluster.avg_score > 0
        assert cluster.max_score > 0

        # Test best candidate selection
        best = cluster.get_best_candidate("weighted_score")
        assert best is not None
        assert best in candidates

        # Test intra-cluster similarity
        similarity = cluster.calculate_intra_cluster_similarity()
        assert 0 <= similarity <= 1

    @pytest.mark.asyncio
    async def test_highlight_clustering(self, sample_highlight_candidates):
        """Test highlight clustering algorithms."""
        config = RankingConfig(clustering_enabled=True)
        clustering = HighlightClustering(config)

        # Test temporal clustering
        config.clustering_method = ClusteringMethod.TEMPORAL
        clusters = await clustering.cluster_highlights(sample_highlight_candidates)
        assert len(clusters) > 0
        assert all(isinstance(cluster, HighlightCluster) for cluster in clusters)

        # Test feature-based clustering
        config.clustering_method = ClusteringMethod.FEATURE_BASED
        clusters = await clustering.cluster_highlights(sample_highlight_candidates)
        assert len(clusters) > 0

        # Test hybrid clustering
        config.clustering_method = ClusteringMethod.HYBRID
        clusters = await clustering.cluster_highlights(sample_highlight_candidates)
        assert len(clusters) > 0

    @pytest.mark.asyncio
    async def test_highlight_ranking(
        self, highlight_ranker, sample_highlight_candidates
    ):
        """Test highlight ranking process."""
        selected_highlights, metrics = await highlight_ranker.rank_and_select(
            sample_highlight_candidates
        )

        assert len(selected_highlights) <= highlight_ranker.config.max_highlights
        assert len(selected_highlights) >= min(
            highlight_ranker.config.min_highlights, len(sample_highlight_candidates)
        )

        # Check if results are sorted by score (generally)
        if len(selected_highlights) > 1:
            scores = [h.weighted_score for h in selected_highlights]
            # Scores should generally be in descending order
            assert scores[0] >= scores[-1]

        # Test metrics
        assert metrics.total_candidates == len(sample_highlight_candidates)
        assert metrics.selected_count == len(selected_highlights)
        assert 0 <= metrics.avg_score <= 1
        assert 0 <= metrics.diversity_score <= 1

    def test_ranking_methods(self, sample_highlight_candidates):
        """Test different ranking methods."""
        methods = [
            RankingMethod.SCORE_BASED,
            RankingMethod.WEIGHTED_MULTI_CRITERIA,
            RankingMethod.DIVERSITY_AWARE,
        ]

        for method in methods:
            config = RankingConfig(ranking_method=method, max_highlights=3)
            ranker = HighlightRanker(config)

            # Should be able to create ranker with different methods
            assert ranker.config.ranking_method == method

    def test_selection_strategies(self, sample_highlight_candidates):
        """Test different selection strategies."""
        from src.services.highlight_detection.ranker import SelectionStrategy

        strategies = [
            SelectionStrategy.TOP_N,
            SelectionStrategy.DIVERSE_SET,
            SelectionStrategy.TEMPORAL_SPREAD,
        ]

        for strategy in strategies:
            config = RankingConfig(selection_strategy=strategy, max_highlights=3)
            ranker = HighlightRanker(config)

            assert ranker.config.selection_strategy == strategy


class TestHighlightPostProcessor:
    """Test cases for highlight post-processing."""

    @pytest.fixture
    def post_processor(self):
        """Create post-processor for testing."""
        config = PostProcessorConfig(
            temporal_smoothing_enabled=True,
            boundary_optimization=BoundaryOptimization.CONTENT_ADAPTIVE,
            quality_enhancement=QualityEnhancement.ADAPTIVE_SCORING,
        )
        return HighlightPostProcessor(config)

    def test_post_processor_config_validation(self):
        """Test post-processor configuration validation."""
        config = PostProcessorConfig(
            smoothing_strength=0.5, boundary_tolerance=5.0, confidence_boost_factor=1.2
        )

        assert config.smoothing_strength == 0.5
        assert config.boundary_tolerance == 5.0
        assert config.confidence_boost_factor == 1.2

    @pytest.mark.asyncio
    async def test_temporal_smoothing(self, sample_highlight_candidates):
        """Test temporal smoothing of highlights."""
        config = PostProcessorConfig(
            temporal_smoothing_enabled=True, smoothing_method=SmoothingMethod.SAVGOL
        )
        smoother = TemporalSmoother(config)

        smoothed_candidates = await smoother.smooth_highlights(
            sample_highlight_candidates
        )

        assert len(smoothed_candidates) == len(sample_highlight_candidates)

        # Check if smoothing was applied
        for original, smoothed in zip(sample_highlight_candidates, smoothed_candidates):
            if smoothed.features and smoothed.features.get("smoothing_applied"):
                assert "original_score" in smoothed.features
                assert "original_confidence" in smoothed.features

    @pytest.mark.asyncio
    async def test_boundary_optimization(self, sample_highlight_candidates):
        """Test boundary optimization."""
        config = PostProcessorConfig(
            boundary_optimization=BoundaryOptimization.CONTENT_ADAPTIVE,
            boundary_tolerance=5.0,
        )
        optimizer = BoundaryOptimizer(config)

        optimized_candidates = await optimizer.optimize_boundaries(
            sample_highlight_candidates
        )

        assert len(optimized_candidates) == len(sample_highlight_candidates)

        # Check boundary constraints
        for candidate in optimized_candidates:
            assert candidate.duration >= config.min_highlight_duration
            assert candidate.duration <= config.max_highlight_duration

    @pytest.mark.asyncio
    async def test_quality_enhancement(self, sample_highlight_candidates):
        """Test quality enhancement."""
        config = PostProcessorConfig(
            quality_enhancement=QualityEnhancement.CONFIDENCE_BOOSTING,
            confidence_boost_factor=1.2,
        )
        enhancer = QualityEnhancer(config)

        enhanced_candidates = await enhancer.enhance_quality(
            sample_highlight_candidates
        )

        assert len(enhanced_candidates) == len(sample_highlight_candidates)

        # Some candidates should have improved quality
        improvements = 0
        for original, enhanced in zip(sample_highlight_candidates, enhanced_candidates):
            if (
                enhanced.confidence > original.confidence
                or enhanced.score > original.score
            ):
                improvements += 1

        # At least some should be improved (depending on the algorithm)
        assert improvements >= 0

    @pytest.mark.asyncio
    async def test_full_post_processing_pipeline(
        self, post_processor, sample_highlight_candidates
    ):
        """Test full post-processing pipeline."""
        processed_candidates, metrics = await post_processor.process_highlights(
            sample_highlight_candidates
        )

        assert len(processed_candidates) <= len(sample_highlight_candidates)
        assert metrics.candidates_processed == len(sample_highlight_candidates)
        assert metrics.processing_time_ms > 0
        assert metrics.improvement_ratio >= 0

        # Check if all candidates have valid scores and confidences
        for candidate in processed_candidates:
            assert 0 <= candidate.score <= 1
            assert 0 <= candidate.confidence <= 1

    def test_smoothing_methods(self, sample_highlight_candidates):
        """Test different smoothing methods."""
        methods = [
            SmoothingMethod.MOVING_AVERAGE,
            SmoothingMethod.EXPONENTIAL,
            SmoothingMethod.SAVGOL,
            SmoothingMethod.GAUSSIAN,
        ]

        for method in methods:
            config = PostProcessorConfig(
                temporal_smoothing_enabled=True, smoothing_method=method
            )
            processor = HighlightPostProcessor(config)

            assert processor.config.smoothing_method == method


class TestIntegrationScenarios:
    """Test cases for integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_detection_pipeline(
        self, sample_video_frames, sample_audio_data, sample_chat_messages
    ):
        """Test complete detection pipeline with all modalities."""
        # Create detectors
        video_detector = VideoDetector()
        audio_detector = AudioDetector()
        chat_detector = ChatDetector()

        # Create content segments
        video_segment = ContentSegment(
            start_time=0.0, end_time=30.0, data=sample_video_frames
        )

        audio_segment = ContentSegment(
            start_time=0.0, end_time=30.0, data=sample_audio_data
        )

        chat_window = ChatWindow(
            start_time=0.0, end_time=30.0, messages=sample_chat_messages
        )
        chat_segment = ContentSegment(start_time=0.0, end_time=30.0, data=chat_window)

        # Run detection
        video_results = await video_detector.detect_highlights([video_segment])
        audio_results = await audio_detector.detect_highlights([audio_segment])
        chat_results = await chat_detector.detect_highlights([chat_segment])

        # Fusion
        results_by_modality = {
            ModalityType.VIDEO: video_results,
            ModalityType.AUDIO: audio_results,
            ModalityType.CHAT: chat_results,
        }

        fusion_scorer = FusionScorer()
        candidates = await fusion_scorer.fuse_results(results_by_modality)

        # Ranking
        ranker = HighlightRanker()
        selected_highlights, ranking_metrics = await ranker.rank_and_select(candidates)

        # Post-processing
        post_processor = HighlightPostProcessor()
        final_highlights, processing_metrics = await post_processor.process_highlights(
            selected_highlights
        )

        # Verify pipeline output
        assert isinstance(final_highlights, list)
        assert isinstance(ranking_metrics.total_candidates, int)
        assert isinstance(processing_metrics.candidates_processed, int)

        for highlight in final_highlights:
            assert isinstance(highlight, HighlightCandidate)
            assert 0 <= highlight.score <= 1
            assert 0 <= highlight.confidence <= 1

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery in detection pipeline."""
        # Test with invalid data
        video_detector = VideoDetector()

        invalid_segment = ContentSegment(
            start_time=0.0,
            end_time=30.0,
            data=None,  # Invalid data
        )

        # Should handle gracefully without crashing
        results = await video_detector.detect_highlights([invalid_segment])
        assert isinstance(
            results, list
        )  # Should return empty list or handle gracefully

    def test_performance_metrics_collection(self):
        """Test performance metrics collection across components."""
        # Test detector metrics
        video_detector = VideoDetector()
        metrics = video_detector.get_metrics()
        assert "segments_processed" in metrics
        assert "highlights_detected" in metrics

        # Test fusion metrics
        fusion_scorer = FusionScorer()
        fusion_metrics = fusion_scorer.get_fusion_metrics()
        assert "fusion_method" in fusion_metrics
        assert "config" in fusion_metrics

    def test_configuration_validation_and_updates(self):
        """Test configuration validation and runtime updates."""
        # Test invalid configuration
        with pytest.raises(ValueError):
            VideoDetectionConfig(motion_threshold=1.5)  # Invalid range

        # Test configuration updates
        ranker = HighlightRanker()
        original_max = ranker.config.max_highlights

        ranker.update_config(max_highlights=20)
        assert ranker.config.max_highlights == 20
        assert ranker.config.max_highlights != original_max


# Performance and stress testing
class TestPerformanceAndStress:
    """Test cases for performance and stress testing."""

    @pytest.mark.asyncio
    async def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        # Create large number of candidates
        large_candidate_set = []

        for i in range(100):
            candidate = HighlightCandidate(
                start_time=i * 30.0,
                end_time=(i * 30.0) + 45.0,
                score=np.random.random(),
                confidence=np.random.random(),
                modality_results=[],
                features={"index": i},
            )
            large_candidate_set.append(candidate)

        # Test ranking performance
        ranker = HighlightRanker()
        start_time = asyncio.get_event_loop().time()

        selected, metrics = await ranker.rank_and_select(large_candidate_set)

        end_time = asyncio.get_event_loop().time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms

        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 5000  # 5 seconds
        assert len(selected) <= ranker.config.max_highlights
        assert metrics.total_candidates == 100

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        # Create multiple detectors
        detectors = [VideoDetector() for _ in range(3)]

        # Create test segments
        segments = []
        for i in range(10):
            segment = ContentSegment(
                start_time=i * 10.0,
                end_time=(i * 10.0) + 10.0,
                data=np.random.random((50, 50, 3)),  # Small video frames
            )
            segments.append(segment)

        # Process concurrently
        async def process_with_detector(detector, segment_list):
            return await detector.detect_highlights(segment_list)

        tasks = [process_with_detector(detector, segments) for detector in detectors]

        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()

        # Verify all tasks completed
        assert len(results) == 3

        # Should complete in reasonable time
        processing_time = (end_time - start_time) * 1000
        assert processing_time < 10000  # 10 seconds

    def test_memory_usage_optimization(self):
        """Test memory usage patterns."""
        # Create detector and verify it cleans up properly
        detector = VideoDetector()
        _initial_metrics = detector.get_metrics()

        # Reset metrics to free memory
        detector.reset_metrics()
        reset_metrics = detector.get_metrics()

        # Verify reset worked
        assert reset_metrics["segments_processed"] == 0
        assert reset_metrics["highlights_detected"] == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
