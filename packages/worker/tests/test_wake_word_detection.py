"""Tests for wake word detection task."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4
import tempfile

from worker.tasks.wake_word_detection import (
    detect_wake_words_task,
    _detect_wake_words_async,
    _check_wake_word_in_transcript,
    WakeWordDetectionTask,
)
from shared.domain.models.wake_word import WakeWord


class TestWakeWordDetection:
    """Test wake word detection functionality."""
    
    @pytest.fixture
    def wake_word_task(self):
        """Create wake word detection task instance."""
        task = WakeWordDetectionTask()
        task.request = Mock()
        task.request.id = "test-task-id"
        task.request.retries = 0
        return task
    
    @pytest.fixture
    def sample_wake_words(self):
        """Create sample wake words for testing."""
        return [
            WakeWord(
                id=uuid4(),
                organization_id=uuid4(),
                phrase="hey assistant",
                max_edit_distance=2,
                similarity_threshold=0.8,
            ),
            WakeWord(
                id=uuid4(),
                organization_id=uuid4(),
                phrase="start recording",
                max_edit_distance=1,
                similarity_threshold=0.9,
            ),
            WakeWord(
                id=uuid4(),
                organization_id=uuid4(),
                phrase="ok bot",
                case_sensitive=True,
                exact_match=True,
            ),
        ]
    
    @pytest.fixture
    def audio_chunk(self, tmp_path):
        """Create test audio chunk data."""
        audio_file = tmp_path / "test_audio.wav"
        audio_file.write_bytes(b"fake audio data")
        
        return {
            "id": str(uuid4()),
            "path": str(audio_file),
            "start_time": 100.0,
            "end_time": 130.0,
            "video_segment_number": 2,
        }
    
    def test_wake_word_matching_exact(self):
        """Test exact wake word matching."""
        wake_word = WakeWord(
            phrase="activate",
            exact_match=True,
        )
        
        transcript = "please activate the system"
        word_timings = [
            {"word": "please", "start": 0.0, "end": 0.5},
            {"word": "activate", "start": 0.6, "end": 1.2},
            {"word": "the", "start": 1.3, "end": 1.5},
            {"word": "system", "start": 1.6, "end": 2.0},
        ]
        
        result = _check_wake_word_in_transcript(
            wake_word, transcript, word_timings, 0.0
        )
        
        assert result is not None
        assert result["detected_phrase"] == "activate"
        assert result["timestamp"] == 0.6
        assert result["similarity_score"] == 1.0
        assert result["edit_distance"] == 0
    
    def test_wake_word_matching_fuzzy(self):
        """Test fuzzy wake word matching."""
        wake_word = WakeWord(
            phrase="hey assistant",
            max_edit_distance=2,
            similarity_threshold=0.8,
        )
        
        # Test with slight misspelling
        transcript = "hey assistent how are you"
        word_timings = [
            {"word": "hey", "start": 0.0, "end": 0.3},
            {"word": "assistent", "start": 0.4, "end": 1.0},  # Misspelled
            {"word": "how", "start": 1.1, "end": 1.3},
            {"word": "are", "start": 1.4, "end": 1.6},
            {"word": "you", "start": 1.7, "end": 2.0},
        ]
        
        result = _check_wake_word_in_transcript(
            wake_word, transcript, word_timings, 0.0
        )
        
        assert result is not None
        assert result["detected_phrase"] == "hey assistent"
        assert result["timestamp"] == 0.0
        assert result["edit_distance"] <= 2
        assert result["similarity_score"] >= 0.8
    
    def test_wake_word_no_match(self):
        """Test when wake word is not found."""
        wake_word = WakeWord(
            phrase="activate system",
            exact_match=True,
        )
        
        transcript = "please start the program"
        word_timings = []
        
        result = _check_wake_word_in_transcript(
            wake_word, transcript, word_timings, 0.0
        )
        
        assert result is None
    
    def test_case_sensitive_matching(self):
        """Test case sensitive wake word matching."""
        wake_word = WakeWord(
            phrase="JARVIS",
            case_sensitive=True,
            exact_match=True,
        )
        
        # Should not match lowercase
        transcript_lower = "hey jarvis"
        result = _check_wake_word_in_transcript(
            wake_word, transcript_lower, [], 0.0
        )
        assert result is None
        
        # Should match uppercase
        transcript_upper = "hey JARVIS"
        result = _check_wake_word_in_transcript(
            wake_word, transcript_upper, [], 0.0
        )
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_detect_wake_words_task_success(self, wake_word_task, audio_chunk):
        """Test successful wake word detection task."""
        # Mock database and repositories
        with patch("worker.tasks.wake_word_detection.Database") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
            
            # Mock wake word repository
            mock_wake_word_repo = AsyncMock()
            mock_wake_word_repo.get_active_by_organization.return_value = [
                WakeWord(
                    id=uuid4(),
                    phrase="test phrase",
                    organization_id=uuid4(),
                )
            ]
            
            with patch("worker.tasks.wake_word_detection.WakeWordRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_wake_word_repo
                
                # Mock WhisperModel
                with patch.object(wake_word_task, "_whisper_model") as mock_whisper:
                    # Mock transcription result
                    mock_segments = [
                        Mock(
                            text=" test phrase detected ",
                            words=[
                                Mock(word="test", start=0.0, end=0.5),
                                Mock(word="phrase", start=0.6, end=1.2),
                                Mock(word="detected", start=1.3, end=2.0),
                            ]
                        )
                    ]
                    mock_info = Mock(language="en", language_probability=0.99)
                    mock_whisper.transcribe.return_value = (mock_segments, mock_info)
                    
                    # Run task
                    result = await _detect_wake_words_async(
                        wake_word_task,
                        str(uuid4()),
                        audio_chunk,
                        str(uuid4()),
                    )
                    
                    assert len(result["detected"]) == 1
                    assert result["transcript"] == "test phrase detected"
                    assert result["chunk_id"] == audio_chunk["id"]
    
    @pytest.mark.asyncio
    async def test_detect_wake_words_no_active_words(self, wake_word_task, audio_chunk):
        """Test detection when no active wake words."""
        with patch("worker.tasks.wake_word_detection.Database") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
            
            # Mock empty wake word list
            mock_wake_word_repo = AsyncMock()
            mock_wake_word_repo.get_active_by_organization.return_value = []
            
            with patch("worker.tasks.wake_word_detection.WakeWordRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_wake_word_repo
                
                result = await _detect_wake_words_async(
                    wake_word_task,
                    str(uuid4()),
                    audio_chunk,
                    str(uuid4()),
                )
                
                assert result["detected"] == []
                assert result["chunk_id"] == audio_chunk["id"]
    
    @pytest.mark.asyncio
    async def test_detect_wake_words_audio_file_missing(self, wake_word_task):
        """Test handling missing audio file."""
        audio_chunk = {
            "id": str(uuid4()),
            "path": "/nonexistent/audio.wav",
            "start_time": 0.0,
            "end_time": 30.0,
            "video_segment_number": 0,
        }
        
        with patch("worker.tasks.wake_word_detection.Database") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(FileNotFoundError):
                await _detect_wake_words_async(
                    wake_word_task,
                    str(uuid4()),
                    audio_chunk,
                    str(uuid4()),
                )
    
    @pytest.mark.asyncio
    async def test_cooldown_enforcement(self, wake_word_task, audio_chunk):
        """Test wake word cooldown enforcement."""
        wake_word = WakeWord(
            id=uuid4(),
            phrase="trigger",
            cooldown_seconds=60,
            last_triggered_at=datetime.now(timezone.utc) - timedelta(seconds=30),
        )
        
        # Wake word should not trigger due to cooldown
        assert wake_word.can_trigger is False
        
        with patch("worker.tasks.wake_word_detection.Database") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
            
            mock_wake_word_repo = AsyncMock()
            mock_wake_word_repo.get_active_by_organization.return_value = [wake_word]
            mock_wake_word_repo.get_by_id.return_value = wake_word
            
            with patch("worker.tasks.wake_word_detection.WakeWordRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_wake_word_repo
                
                with patch.object(wake_word_task, "_whisper_model") as mock_whisper:
                    mock_segments = [Mock(text=" trigger ", words=[])]
                    mock_info = Mock(language="en", language_probability=0.99)
                    mock_whisper.transcribe.return_value = (mock_segments, mock_info)
                    
                    result = await _detect_wake_words_async(
                        wake_word_task,
                        str(uuid4()),
                        audio_chunk,
                        str(uuid4()),
                    )
                    
                    # Wake word detected but not triggered due to cooldown
                    mock_wake_word_repo.update.assert_not_called()
    
    def test_multi_word_phrase_matching(self):
        """Test matching multi-word phrases."""
        wake_word = WakeWord(
            phrase="hey there assistant",
            max_edit_distance=2,
            similarity_threshold=0.8,
        )
        
        # Test exact match
        transcript = "say hey there assistant please"
        result = _check_wake_word_in_transcript(wake_word, transcript, [], 0.0)
        assert result is not None
        assert result["detected_phrase"] == "hey there assistant"
        
        # Test with word in middle
        transcript = "hey there my assistant"
        result = _check_wake_word_in_transcript(wake_word, transcript, [], 0.0)
        assert result is None  # Should not match with extra word
        
        # Test partial phrase
        transcript = "hey there"
        result = _check_wake_word_in_transcript(wake_word, transcript, [], 0.0)
        assert result is None  # Should not match incomplete phrase
    
    def test_edge_distance_calculation(self):
        """Test edit distance edge cases."""
        wake_word = WakeWord(
            phrase="activate",
            max_edit_distance=0,  # No edits allowed
            similarity_threshold=1.0,  # Perfect match required
        )
        
        # Test exact match
        transcript = "activate"
        result = _check_wake_word_in_transcript(wake_word, transcript, [], 0.0)
        assert result is not None
        
        # Test with one character difference
        transcript = "activete"  # One substitution
        result = _check_wake_word_in_transcript(wake_word, transcript, [], 0.0)
        assert result is None  # Should not match with edit distance 0
    
    @pytest.mark.asyncio
    async def test_whisper_model_initialization(self, wake_word_task):
        """Test lazy initialization of Whisper model."""
        assert wake_word_task._whisper_model is None
        
        with patch("worker.tasks.wake_word_detection.WhisperModel") as mock_whisper_class:
            mock_model = Mock()
            mock_whisper_class.return_value = mock_model
            
            wake_word_task._initialize_models()
            
            assert wake_word_task._whisper_model == mock_model
            mock_whisper_class.assert_called_once_with(
                "base",
                device="auto",
                compute_type="auto",
            )
    
    def test_timestamp_adjustment_with_chunk_offset(self):
        """Test timestamp adjustment with audio chunk offset."""
        wake_word = WakeWord(phrase="test")
        
        word_timings = [
            {"word": "test", "start": 5.0, "end": 5.5},  # Local chunk time
        ]
        chunk_start_time = 100.0  # Absolute stream time
        
        result = _check_wake_word_in_transcript(
            wake_word, "test", word_timings, chunk_start_time
        )
        
        assert result is not None
        # Timestamp should be adjusted to absolute time
        assert result["timestamp"] == 105.0  # 100.0 + 5.0
    
    @pytest.mark.asyncio
    async def test_clip_generation_queuing(self, wake_word_task, audio_chunk):
        """Test that clip generation is queued on wake word detection."""
        with patch("worker.tasks.wake_word_detection.Database") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
            
            wake_word = WakeWord(
                id=uuid4(),
                phrase="record this",
                organization_id=uuid4(),
            )
            
            mock_wake_word_repo = AsyncMock()
            mock_wake_word_repo.get_active_by_organization.return_value = [wake_word]
            mock_wake_word_repo.get_by_id.return_value = wake_word
            
            with patch("worker.tasks.wake_word_detection.WakeWordRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_wake_word_repo
                
                with patch.object(wake_word_task, "_whisper_model") as mock_whisper:
                    mock_segments = [
                        Mock(
                            text=" record this now ",
                            words=[
                                Mock(word="record", start=0.0, end=0.5),
                                Mock(word="this", start=0.6, end=1.0),
                                Mock(word="now", start=1.1, end=1.5),
                            ]
                        )
                    ]
                    mock_info = Mock(language="en", language_probability=0.99)
                    mock_whisper.transcribe.return_value = (mock_segments, mock_info)
                    
                    with patch("worker.tasks.wake_word_detection.generate_wake_word_clip_task") as mock_clip_task:
                        result = await _detect_wake_words_async(
                            wake_word_task,
                            str(uuid4()),
                            audio_chunk,
                            str(uuid4()),
                        )
                        
                        # Verify clip generation was queued
                        mock_clip_task.delay.assert_called_once()
                        call_args = mock_clip_task.delay.call_args
                        assert "wake_word_detection" in call_args.kwargs
                        assert call_args.kwargs["video_segment_number"] == audio_chunk["video_segment_number"]


class TestWakeWordDetectionEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_empty_transcript(self):
        """Test handling empty transcript."""
        wake_word = WakeWord(phrase="test")
        result = _check_wake_word_in_transcript(wake_word, "", [], 0.0)
        assert result is None
    
    def test_transcript_with_only_whitespace(self):
        """Test transcript with only whitespace."""
        wake_word = WakeWord(phrase="test")
        result = _check_wake_word_in_transcript(wake_word, "   \t\n   ", [], 0.0)
        assert result is None
    
    def test_very_long_wake_phrase(self):
        """Test handling very long wake phrases."""
        long_phrase = " ".join(["word"] * 20)
        wake_word = WakeWord(phrase=long_phrase)
        
        transcript = f"start {long_phrase} end"
        result = _check_wake_word_in_transcript(wake_word, transcript, [], 0.0)
        assert result is not None
    
    def test_special_characters_in_transcript(self):
        """Test handling special characters."""
        wake_word = WakeWord(phrase="hey bot")
        
        transcript = "hey! bot?"  # Punctuation between words
        result = _check_wake_word_in_transcript(wake_word, transcript, [], 0.0)
        # Should handle punctuation gracefully
        assert result is not None
    
    def test_unicode_in_transcript(self):
        """Test handling unicode characters."""
        wake_word = WakeWord(phrase="héy bøt", case_sensitive=False)
        
        transcript = "say héy bøt please"
        result = _check_wake_word_in_transcript(wake_word, transcript, [], 0.0)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_transcription_failure(self, wake_word_task, audio_chunk):
        """Test handling transcription failures."""
        with patch("worker.tasks.wake_word_detection.Database") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.session.return_value.__aenter__.return_value = mock_session
            
            mock_wake_word_repo = AsyncMock()
            mock_wake_word_repo.get_active_by_organization.return_value = [
                WakeWord(phrase="test")
            ]
            
            with patch("worker.tasks.wake_word_detection.WakeWordRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_wake_word_repo
                
                with patch.object(wake_word_task, "_whisper_model") as mock_whisper:
                    # Mock transcription failure
                    mock_whisper.transcribe.side_effect = Exception("Transcription failed")
                    
                    with pytest.raises(Exception, match="Transcription failed"):
                        await _detect_wake_words_async(
                            wake_word_task,
                            str(uuid4()),
                            audio_chunk,
                            str(uuid4()),
                        )
    
    def test_multiple_wake_words_same_position(self):
        """Test multiple wake words matching at same position."""
        wake_words = [
            WakeWord(id=uuid4(), phrase="activate"),
            WakeWord(id=uuid4(), phrase="activate system"),
        ]
        
        transcript = "please activate system now"
        
        results = []
        for wake_word in wake_words:
            result = _check_wake_word_in_transcript(wake_word, transcript, [], 0.0)
            if result:
                results.append(result)
        
        # Both should match
        assert len(results) == 2
        # Longer phrase should have detected the full phrase
        assert any(r["detected_phrase"] == "activate system" for r in results)
        assert any(r["detected_phrase"] == "activate" for r in results)