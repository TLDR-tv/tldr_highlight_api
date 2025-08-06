"""Simple tests for wake word detection tasks."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4
from pathlib import Path

from worker.tasks.wake_word_detection import (
    detect_wake_words_task,
    WakeWordDetectionTask
)


class TestWakeWordDetectionConfig:
    """Test WakeWordDetectionConfig dataclass."""

    def test_config_creation_default(self):
        """Test creating config with defaults."""
        config = WakeWordDetectionConfig()
        
        assert config.enabled is True
        assert config.sensitivity == 0.5
        assert config.model == "openai/whisper-base"
        assert config.cooldown_seconds == 30.0
        assert len(config.wake_words) == 0

    def test_config_creation_with_params(self):
        """Test creating config with parameters."""
        wake_words = ["amazing", "incredible", "wow"]
        
        config = WakeWordDetectionConfig(
            enabled=True,
            sensitivity=0.8,
            model="openai/whisper-large",
            cooldown_seconds=60.0,
            wake_words=wake_words
        )
        
        assert config.enabled is True
        assert config.sensitivity == 0.8
        assert config.model == "openai/whisper-large"
        assert config.cooldown_seconds == 60.0
        assert config.wake_words == wake_words

    def test_config_disabled(self):
        """Test disabled config."""
        config = WakeWordDetectionConfig(enabled=False)
        
        assert config.enabled is False


class TestWakeWordResult:
    """Test WakeWordResult dataclass."""

    def test_result_creation(self):
        """Test creating a wake word result."""
        result = WakeWordResult(
            word="amazing",
            confidence=0.92,
            timestamp=45.5,
            audio_segment_path=Path("/tmp/audio.wav")
        )
        
        assert result.word == "amazing"
        assert result.confidence == 0.92
        assert result.timestamp == 45.5
        assert result.audio_segment_path == Path("/tmp/audio.wav")

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = WakeWordResult(
            word="incredible",
            confidence=0.88,
            timestamp=120.0,
            audio_segment_path=Path("/tmp/test.wav")
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["word"] == "incredible"
        assert result_dict["confidence"] == 0.88
        assert result_dict["timestamp"] == 120.0
        assert result_dict["audio_segment_path"] == str(Path("/tmp/test.wav"))


class TestDetectWakeWordsTask:
    """Test the detect_wake_words_task function."""

    @pytest.fixture
    def sample_config(self):
        """Create sample wake word config."""
        return WakeWordDetectionConfig(
            enabled=True,
            sensitivity=0.7,
            wake_words=["amazing", "incredible", "wow", "epic"]
        )

    @pytest.fixture
    def mock_task(self):
        """Create mock task instance."""
        task = Mock()
        task.request = Mock()
        task.request.retries = 0
        task.default_retry_delay = 60
        task.retry = Mock(side_effect=Exception("Retry called"))
        return task

    def test_detect_wake_words_task_disabled(self, mock_task):
        """Test wake word detection when disabled."""
        config = WakeWordDetectionConfig(enabled=False)
        audio_segments = [{"path": "/tmp/audio1.wav", "start_time": 0.0, "end_time": 30.0}]
        
        result = detect_wake_words_task(mock_task, audio_segments, config.to_dict())
        
        assert result == []

    @patch('worker.tasks.wake_word_detection.asyncio.run')
    def test_detect_wake_words_task_success(self, mock_run, mock_task, sample_config):
        """Test successful wake word detection."""
        audio_segments = [
            {"path": "/tmp/audio1.wav", "start_time": 0.0, "end_time": 30.0},
            {"path": "/tmp/audio2.wav", "start_time": 25.0, "end_time": 55.0}
        ]
        
        expected_results = [
            {
                "word": "amazing",
                "confidence": 0.9,
                "timestamp": 15.0,
                "audio_segment_path": "/tmp/audio1.wav"
            }
        ]
        
        mock_run.return_value = expected_results
        
        result = detect_wake_words_task(mock_task, audio_segments, sample_config.to_dict())
        
        assert result == expected_results
        mock_run.assert_called_once()

    @patch('worker.tasks.wake_word_detection.asyncio.run')
    @patch('worker.tasks.wake_word_detection.logger')
    def test_detect_wake_words_task_error(self, mock_logger, mock_run, mock_task, sample_config):
        """Test wake word detection with error."""
        audio_segments = [{"path": "/tmp/audio1.wav", "start_time": 0.0, "end_time": 30.0}]
        
        test_exception = Exception("Transcription failed")
        mock_run.side_effect = test_exception
        
        with pytest.raises(Exception, match="Retry called"):
            detect_wake_words_task(mock_task, audio_segments, sample_config.to_dict())
        
        mock_logger.error.assert_called_once()
        mock_task.retry.assert_called_once()

    def test_detect_wake_words_task_empty_segments(self, mock_task, sample_config):
        """Test wake word detection with empty segments."""
        result = detect_wake_words_task(mock_task, [], sample_config.to_dict())
        
        assert result == []

    def test_detect_wake_words_task_no_wake_words_configured(self, mock_task):
        """Test detection with no wake words configured."""
        config = WakeWordDetectionConfig(enabled=True, wake_words=[])
        audio_segments = [{"path": "/tmp/audio1.wav", "start_time": 0.0, "end_time": 30.0}]
        
        result = detect_wake_words_task(mock_task, audio_segments, config.to_dict())
        
        assert result == []

    def test_wake_word_detection_config_to_dict(self):
        """Test converting config to dictionary."""
        config = WakeWordDetectionConfig(
            enabled=True,
            sensitivity=0.8,
            model="custom-model",
            cooldown_seconds=45.0,
            wake_words=["test", "example"]
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["enabled"] is True
        assert config_dict["sensitivity"] == 0.8
        assert config_dict["model"] == "custom-model"
        assert config_dict["cooldown_seconds"] == 45.0
        assert config_dict["wake_words"] == ["test", "example"]

    @pytest.mark.parametrize("sensitivity", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_config_sensitivity_values(self, sensitivity):
        """Test config with different sensitivity values."""
        config = WakeWordDetectionConfig(sensitivity=sensitivity)
        
        assert config.sensitivity == sensitivity
        assert 0.0 <= config.sensitivity <= 1.0

    @pytest.mark.parametrize("cooldown", [0.0, 30.0, 60.0, 120.0])
    def test_config_cooldown_values(self, cooldown):
        """Test config with different cooldown values."""
        config = WakeWordDetectionConfig(cooldown_seconds=cooldown)
        
        assert config.cooldown_seconds == cooldown
        assert config.cooldown_seconds >= 0.0

    def test_wake_word_result_with_various_paths(self):
        """Test WakeWordResult with different path types."""
        paths = [
            Path("/tmp/audio.wav"),
            Path("/usr/local/data/segment_001.wav"),
            Path("relative/path/audio.wav")
        ]
        
        for path in paths:
            result = WakeWordResult(
                word="test",
                confidence=0.8,
                timestamp=30.0,
                audio_segment_path=path
            )
            
            assert result.audio_segment_path == path
            
            # Test to_dict preserves path as string
            result_dict = result.to_dict()
            assert result_dict["audio_segment_path"] == str(path)

    def test_wake_word_result_confidence_bounds(self):
        """Test WakeWordResult with various confidence values."""
        confidences = [0.0, 0.5, 0.9, 1.0]
        
        for conf in confidences:
            result = WakeWordResult(
                word="test",
                confidence=conf,
                timestamp=30.0,
                audio_segment_path=Path("/tmp/test.wav")
            )
            
            assert result.confidence == conf
            assert 0.0 <= result.confidence <= 1.0