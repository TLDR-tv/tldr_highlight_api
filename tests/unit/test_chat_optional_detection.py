"""
Unit tests for chat-optional highlight detection.

This module tests that the highlight detection system works correctly
when chat data is unavailable, ensuring chat analysis is only supplementary.
"""

import pytest

from src.services.highlight_detection.base_detector import (
    DetectionResult,
    ModalityType,
)
from src.services.highlight_detection.fusion_scorer import (
    FusionScorer,
    FusionConfig,
    FusionMethod,
)


class TestChatOptionalDetection:
    """Test suite for chat-optional highlight detection."""

    @pytest.fixture
    def fusion_config(self):
        """Create fusion configuration with chat as optional bonus."""
        return FusionConfig(
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            video_weight=0.5,
            audio_weight=0.5,
            chat_weight=0.0,  # Chat is bonus only
            min_fusion_score=0.3,
            min_fusion_confidence=0.4,
            require_multiple_modalities=False,
            missing_modality_penalty=0.1,
        )

    @pytest.fixture
    def sample_video_result(self):
        """Create sample video detection result."""
        return DetectionResult(
            segment_id="seg_001",
            modality=ModalityType.VIDEO,
            score=0.8,
            confidence=0.9,
            features={"motion_intensity": 0.85, "scene_changes": 3},
            metadata={"start_time": 10.0, "end_time": 20.0},
        )

    @pytest.fixture
    def sample_audio_result(self):
        """Create sample audio detection result."""
        return DetectionResult(
            segment_id="seg_001",
            modality=ModalityType.AUDIO,
            score=0.75,
            confidence=0.85,
            features={"volume_spike": 1.0, "speech_emotion": 0.8},
            metadata={"start_time": 10.0, "end_time": 20.0},
        )

    @pytest.fixture
    def sample_chat_result(self):
        """Create sample chat detection result."""
        return DetectionResult(
            segment_id="seg_001",
            modality=ModalityType.CHAT,
            score=0.9,
            confidence=0.95,
            features={
                "message_spike": 1.0,
                "sentiment_score": 0.8,
            },  # Features must be floats
            metadata={"start_time": 10.0, "end_time": 20.0, "sentiment": "positive"},
        )

    @pytest.mark.asyncio
    async def test_fusion_without_chat(
        self, fusion_config, sample_video_result, sample_audio_result
    ):
        """Test that fusion works correctly without chat data."""
        fusion_scorer = FusionScorer(fusion_config)

        # Create results with only video and audio
        results_by_modality = {
            ModalityType.VIDEO: [sample_video_result],
            ModalityType.AUDIO: [sample_audio_result],
        }

        # Fuse results
        candidates = await fusion_scorer.fuse_results(results_by_modality)

        # Verify results
        assert len(candidates) > 0
        candidate = candidates[0]

        # Score should be weighted average of video and audio only
        expected_score = (0.8 * 0.5 + 0.75 * 0.5) / (0.5 + 0.5)
        assert abs(candidate.score - expected_score) < 0.1  # Allow for penalties

        # Should have only 2 modality results
        assert len(candidate.modality_results) == 2
        assert candidate.features["has_chat"] is False

    @pytest.mark.asyncio
    async def test_fusion_with_chat_bonus(
        self,
        fusion_config,
        sample_video_result,
        sample_audio_result,
        sample_chat_result,
    ):
        """Test that chat provides bonus scoring when available."""
        # Update config to include chat weight
        fusion_config.chat_weight = 0.2
        fusion_scorer = FusionScorer(fusion_config)

        # Create results with all modalities
        results_by_modality = {
            ModalityType.VIDEO: [sample_video_result],
            ModalityType.AUDIO: [sample_audio_result],
            ModalityType.CHAT: [sample_chat_result],
        }

        # Fuse results
        candidates = await fusion_scorer.fuse_results(results_by_modality)

        # Verify results
        assert len(candidates) > 0
        candidate = candidates[0]

        # Score should be core average plus chat bonus
        core_score = (0.8 * 0.5 + 0.75 * 0.5) / (0.5 + 0.5)
        chat_bonus = 0.9 * 0.2
        expected_score = min(1.0, core_score + chat_bonus)
        assert abs(candidate.score - expected_score) < 0.1  # Allow for penalties

        # Should have all 3 modality results
        assert len(candidate.modality_results) == 3
        assert candidate.features["has_chat"] is True

    @pytest.mark.asyncio
    async def test_no_penalty_for_missing_chat(
        self, fusion_config, sample_video_result, sample_audio_result
    ):
        """Test that missing chat doesn't result in penalties."""
        fusion_scorer = FusionScorer(fusion_config)

        # Create results without chat
        results_by_modality = {
            ModalityType.VIDEO: [sample_video_result],
            ModalityType.AUDIO: [sample_audio_result],
        }

        candidates = await fusion_scorer.fuse_results(results_by_modality)
        candidate = candidates[0]

        # Calculate expected score without chat penalty
        base_score = (0.8 * 0.5 + 0.75 * 0.5) / (0.5 + 0.5)

        # Score should not be significantly penalized for missing chat
        assert (
            candidate.score >= base_score * 0.9
        )  # Allow max 10% reduction for other factors

    @pytest.mark.asyncio
    async def test_single_modality_detection(self, fusion_config, sample_video_result):
        """Test that detection works with only video modality."""
        fusion_scorer = FusionScorer(fusion_config)

        # Create results with only video
        results_by_modality = {
            ModalityType.VIDEO: [sample_video_result],
        }

        candidates = await fusion_scorer.fuse_results(results_by_modality)

        # Should still produce candidates (with penalties)
        assert len(candidates) > 0
        candidate = candidates[0]

        # Score will be penalized for missing audio
        assert candidate.score < sample_video_result.score
        assert len(candidate.modality_results) == 1

    @pytest.mark.asyncio
    async def test_multimodal_processing_without_chat(self):
        """Test multimodal processing task handles missing chat gracefully."""
        from src.services.async_processing.tasks import process_multimodal_content

        # Test data
        stream_id = 1
        ingestion_data = {
            "video_chunks": ["chunk1.mp4", "chunk2.mp4"],
            "audio_chunks": ["chunk1.wav", "chunk2.wav"],
        }

        # Call the actual function (it handles missing chat internally)
        result = process_multimodal_content(stream_id, ingestion_data)

        # Verify chat is marked as unavailable
        assert result["chat_analysis"]["available"] is False
        assert "reason" in result["chat_analysis"]
        assert result["chat_analysis"]["sentiment_score"] == 0.0
        assert result["chat_analysis"]["message_count"] == 0

        # Verify other modalities were processed
        assert "video_features" in result
        assert "audio_features" in result

        # Verify modalities list doesn't include chat
        assert "video" in result["processing_metadata"]["modalities_available"]
        assert "audio" in result["processing_metadata"]["modalities_available"]
        assert "chat" not in result["processing_metadata"]["modalities_available"]

    def test_fusion_config_defaults(self):
        """Test that default fusion config has correct weights."""
        config = FusionConfig()

        # Video and audio should have equal weight by default
        assert config.video_weight == 0.5
        assert config.audio_weight == 0.5

        # Chat should have no weight by default (bonus only)
        assert config.chat_weight == 0.0

        # Should not require multiple modalities
        assert config.require_multiple_modalities is False

    def test_normalized_weights_calculation(self):
        """Test that weight normalization handles chat separately."""
        config = FusionConfig(
            video_weight=0.4,
            audio_weight=0.6,
            chat_weight=0.3,  # Bonus weight
        )

        normalized = config.normalized_weights

        # Video and audio should be normalized to sum to 1.0
        # 0.4 / (0.4 + 0.6) = 0.4
        # 0.6 / (0.4 + 0.6) = 0.6
        assert abs(normalized[ModalityType.VIDEO] - 0.4) < 0.01
        assert abs(normalized[ModalityType.AUDIO] - 0.6) < 0.01

        # Chat weight should remain as bonus (not normalized)
        assert normalized[ModalityType.CHAT] == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
