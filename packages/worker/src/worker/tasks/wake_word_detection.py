"""Wake word detection task using faster-whisper transcription."""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from uuid import UUID
from pathlib import Path

from celery import Task
from structlog import get_logger
from faster_whisper import WhisperModel
from rapidfuzz import fuzz
from rapidfuzz.distance import DamerauLevenshtein

from worker.app import celery_app
from shared.infrastructure.storage.repositories import WakeWordRepository, HighlightRepository
from shared.infrastructure.database.database import Database
from shared.infrastructure.config.config import get_settings
from shared.domain.models.wake_word import WakeWord

logger = get_logger()


class WakeWordDetectionTask(Task):
    """Base task for wake word detection with faster-whisper."""
    
    def __init__(self):
        super().__init__()
        self._whisper_model = None
    
    def _initialize_models(self):
        """Initialize faster-whisper model lazily."""
        if self._whisper_model is None:
            # Load faster-whisper model
            logger.info("Loading faster-whisper model")
            self._whisper_model = WhisperModel(
                "base",  # Use base model for speed/accuracy balance
                device="auto",  # Automatically select device
                compute_type="auto",  # Automatically select compute type
            )
            
            logger.info("Faster-whisper model loaded successfully")


@celery_app.task(
    bind=True,
    base=WakeWordDetectionTask,
    name="detect_wake_words",
    max_retries=3,
    default_retry_delay=60,
)
def detect_wake_words_task(
    self,
    stream_id: str,
    audio_chunk: Dict,
    organization_id: str,
) -> Dict:
    """Detect wake words in audio chunk using WhisperX.
    
    Args:
        stream_id: UUID of the stream
        audio_chunk: Audio chunk information with path and timestamps
        organization_id: Organization ID to fetch wake words
        
    Returns:
        Dictionary with detected wake words and their timestamps
    """
    try:
        # Initialize models if needed
        self._initialize_models()
        
        # Run async detection
        result = asyncio.run(
            _detect_wake_words_async(
                self,
                stream_id,
                video_segment,
                organization_id,
            )
        )
        return result
        
    except Exception as exc:
        logger.error(
            "Wake word detection error",
            stream_id=stream_id,
            segment_id=video_segment.get("id"),
            error=str(exc),
            exc_info=True,
        )
        
        # Retry with exponential backoff
        countdown = self.default_retry_delay * (2 ** self.request.retries)
        raise self.retry(exc=exc, countdown=countdown)


async def _detect_wake_words_async(
    task: WakeWordDetectionTask,
    stream_id: str,
    video_segment: Dict,
    organization_id: str,
) -> Dict:
    """Async implementation of wake word detection."""
    settings = get_settings()
    database = Database(settings.database_url)
    
    detected_wake_words = []
    
    try:
        async with database.session() as session:
            # Get active wake words for organization
            wake_word_repo = WakeWordRepository(session)
            wake_words = await wake_word_repo.get_active_by_organization(
                UUID(organization_id)
            )
            
            if not wake_words:
                logger.info(
                    "No active wake words for organization",
                    organization_id=organization_id,
                )
                return {"detected": [], "segment_id": video_segment["id"]}
            
            logger.info(
                f"Checking {len(wake_words)} wake words for video segment",
                segment_id=video_segment["id"],
                wake_words=[w.phrase for w in wake_words],
            )
        
        # Extract audio from video segment using FFmpeg with proper context management
        video_path = Path(video_segment["video_path"])
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Use context manager for audio extraction and cleanup
        async with _extract_audio_from_video(video_path) as audio_path:
            # Transcribe with faster-whisper
            logger.debug(f"Transcribing audio segment: {audio_path}")
            
            # Transcribe with word-level timestamps
            segments, info = task._whisper_model.transcribe(
                str(audio_path),
                word_timestamps=True,  # Enable word-level timestamps
                language=None,  # Auto-detect language
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0,  # Deterministic transcription
            )
            
            logger.info(
                f"Detected language: {info.language} with probability {info.language_probability:.2f}"
            )
            
            # Extract full transcript and word timings
            full_transcript = ""
            word_timings = []
        
        for segment in segments:
            segment_text = segment.text.strip()
            full_transcript += " " + segment_text
            
            # Get word-level timings if available
            if segment.words:
                for word_info in segment.words:
                    word_timings.append({
                        "word": word_info.word.strip(),
                        "start": word_info.start + audio_chunk["start_time"],
                        "end": word_info.end + audio_chunk["start_time"],
                    })
        
        full_transcript = full_transcript.strip()
        logger.info(f"Transcript: {full_transcript}")
        
        # Check each wake word against the transcript
        for wake_word in wake_words:
            detection = _check_wake_word_in_transcript(
                wake_word,
                full_transcript,
                word_timings,
                video_segment["start_time"],
            )
            
            if detection:
                detected_wake_words.append(detection)
                logger.info(
                    f"Wake word detected: '{wake_word.phrase}' at {detection['timestamp']}s",
                    similarity_score=detection["similarity_score"],
                )
        
        # Update wake word usage in database if any detected
        if detected_wake_words:
            async with database.session() as session:
                wake_word_repo = WakeWordRepository(session)
                
                for detection in detected_wake_words:
                    wake_word = await wake_word_repo.get_by_id(
                        UUID(detection["wake_word_id"])
                    )
                    if wake_word and wake_word.can_trigger:
                        wake_word.record_trigger()
                        await wake_word_repo.update(wake_word)
                        
                        # Queue clip generation
                        generate_wake_word_clip_task.delay(
                            stream_id=stream_id,
                            wake_word_detection=detection,
                            video_segment_number=video_segment["segment_number"],
                        )
        
        return {
            "detected": detected_wake_words,
            "chunk_id": audio_chunk["id"],
            "transcript": full_transcript,
        }
        
    finally:
        # Database context manager handles cleanup automatically
        pass


def _check_wake_word_in_transcript(
    wake_word: WakeWord,
    transcript: str,
    word_timings: List[Dict],
    chunk_start_time: float,
) -> Optional[Dict]:
    """Check if wake word appears in transcript using fuzzy matching."""
    phrase = wake_word.phrase.lower() if not wake_word.case_sensitive else wake_word.phrase
    text = transcript.lower() if not wake_word.case_sensitive else transcript
    
    # Split into words for multi-word phrase matching
    phrase_words = phrase.split()
    text_words = text.split()
    
    if not phrase_words or not text_words:
        return None
    
    # Sliding window search for multi-word phrases
    for i in range(len(text_words) - len(phrase_words) + 1):
        window = " ".join(text_words[i:i + len(phrase_words)])
        
        # Calculate similarity using Damerau-Levenshtein distance
        distance = DamerauLevenshtein.distance(phrase, window)
        max_distance = max(len(phrase), len(window))
        similarity = 1.0 - (distance / max_distance) if max_distance > 0 else 0.0
        
        # Also try fuzzy ratio for additional matching
        fuzzy_score = fuzz.ratio(phrase, window) / 100.0
        
        # Use the better score
        best_score = max(similarity, fuzzy_score)
        
        # Check if match meets criteria
        if (distance <= wake_word.max_edit_distance and 
            best_score >= wake_word.similarity_threshold):
            
            # Find timestamp for this position
            timestamp = chunk_start_time
            if i < len(word_timings):
                timestamp = word_timings[i]["start"]
            
            return {
                "wake_word_id": str(wake_word.id),
                "wake_word_phrase": wake_word.phrase,
                "detected_phrase": window,
                "timestamp": timestamp,
                "similarity_score": best_score,
                "edit_distance": distance,
                "position_in_transcript": i,
            }
    
    return None


@celery_app.task(
    bind=True,
    name="generate_wake_word_clip",
    max_retries=3,
    default_retry_delay=60,
)
def generate_wake_word_clip_task(
    self,
    stream_id: str,
    wake_word_detection: Dict,
    video_segment_number: int,
) -> Dict:
    """Generate a clip for detected wake word.
    
    Args:
        stream_id: UUID of the stream
        wake_word_detection: Detection information including timestamp
        video_segment_number: Video segment number containing the wake word
        
    Returns:
        Dictionary with clip information
    """
    try:
        result = asyncio.run(
            _generate_clip_async(
                stream_id,
                wake_word_detection,
                video_segment_number,
            )
        )
        return result
        
    except Exception as exc:
        logger.error(
            "Clip generation error",
            stream_id=stream_id,
            wake_word=wake_word_detection.get("wake_word_phrase"),
            error=str(exc),
            exc_info=True,
        )
        
        countdown = self.default_retry_delay * (2 ** self.request.retries)
        raise self.retry(exc=exc, countdown=countdown)


async def _generate_clip_async(
    stream_id: str,
    wake_word_detection: Dict,
    video_segment_number: int,
) -> Dict:
    """Generate clip around wake word detection."""
    # This will be implemented when we do clip generation
    # For now, just log the detection
    logger.info(
        "Wake word clip generation queued",
        stream_id=stream_id,
        wake_word=wake_word_detection["wake_word_phrase"],
        timestamp=wake_word_detection["timestamp"],
        video_segment=video_segment_number,
    )
    
    return {
        "status": "queued",
        "wake_word_detection": wake_word_detection,
        "video_segment_number": video_segment_number,
    }


@asynccontextmanager
async def _extract_audio_from_video(video_path: Path):
    """Context manager for extracting audio from video with automatic cleanup.
    
    Args:
        video_path: Path to the video file
        
    Yields:
        Path to the extracted audio file
        
    Ensures:
        Temporary audio file is always cleaned up
    """
    import tempfile
    
    # Create temporary audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        audio_path = Path(temp_audio.name)
    
    try:
        # Extract audio using FFmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "16000",  # 16kHz sample rate for WhisperX
            "-ac", "1",  # Mono
            "-y",  # Overwrite output file
            str(audio_path)
        ]
        
        logger.debug(f"Extracting audio from video: {video_path}")
        result = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")
        
        logger.debug(f"Audio extracted to: {audio_path}")
        
        # Yield the audio path for use
        yield audio_path
        
    finally:
        # Clean up temporary audio file
        try:
            if audio_path.exists():
                audio_path.unlink()
                logger.debug(f"Cleaned up temporary audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup audio file {audio_path}: {e}")