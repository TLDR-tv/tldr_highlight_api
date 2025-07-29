"""
Audio processing service for transcription and analysis.

This module provides audio transcription using OpenAI Whisper API,
audio quality analysis, and real-time audio processing capabilities.
"""

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import numpy as np

from src.utils.media_utils import AudioChunk, media_processor
from src.core.config import settings

logger = logging.getLogger(__name__)


class AudioProcessorConfig(BaseModel):
    """Configuration for audio processor."""

    # Whisper API settings
    whisper_model: str = Field(
        default="whisper-1", description="OpenAI Whisper model to use"
    )
    whisper_language: Optional[str] = Field(
        default=None, description="Language for transcription (auto-detect if None)"
    )
    whisper_prompt: Optional[str] = Field(
        default=None, description="Optional prompt to guide transcription"
    )

    # Audio processing settings
    chunk_duration: float = Field(
        default=30.0, description="Duration of audio chunks in seconds"
    )
    sample_rate: int = Field(
        default=16000, description="Target sample rate for audio processing"
    )
    channels: int = Field(
        default=1, description="Number of audio channels (1=mono, 2=stereo)"
    )

    # Quality settings
    min_audio_duration: float = Field(
        default=1.0, description="Minimum audio duration for transcription"
    )
    silence_threshold: float = Field(
        default=0.01, description="Threshold for silence detection"
    )

    # Processing settings
    max_concurrent_requests: int = Field(
        default=5, description="Maximum concurrent Whisper API requests"
    )
    request_timeout: int = Field(default=60, description="Request timeout in seconds")
    retry_attempts: int = Field(
        default=3, description="Number of retry attempts for failed requests"
    )

    # Buffer settings
    buffer_size: int = Field(
        default=100, description="Maximum number of transcriptions to keep in buffer"
    )

    # Quality analysis
    enable_quality_analysis: bool = Field(
        default=True, description="Enable audio quality analysis"
    )
    enable_speaker_detection: bool = Field(
        default=False, description="Enable speaker detection (experimental)"
    )


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    timestamp: float
    duration: float
    confidence: float
    language: Optional[str] = None
    words: Optional[List[Dict[str, Union[str, float]]]] = None
    segments: Optional[List[Dict[str, Union[str, float]]]] = None


@dataclass
class AudioAnalysis:
    """Audio quality and content analysis."""

    volume_level: float
    silence_ratio: float
    speech_ratio: float
    energy: float
    spectral_centroid: float
    zero_crossing_rate: float
    is_speech: bool
    quality_score: float


@dataclass
class ProcessedAudio:
    """Processed audio chunk with transcription and analysis."""

    chunk: AudioChunk
    transcription: Optional[TranscriptionResult]
    analysis: AudioAnalysis
    processing_time: float
    error: Optional[str] = None


@dataclass
class AudioProcessingResult:
    """Result of audio processing operation."""

    transcriptions: List[ProcessedAudio]
    total_duration: float
    processing_time: float
    success_rate: float
    metadata: Dict[str, Union[str, float, int]]


class AudioProcessor:
    """
    Advanced audio processor with OpenAI Whisper integration.

    Features:
    - OpenAI Whisper API integration for transcription
    - Real-time audio processing
    - Audio quality analysis
    - Silence detection and filtering
    - Configurable chunk processing
    - Concurrent processing with rate limiting
    """

    def __init__(self, config: Optional[AudioProcessorConfig] = None):
        self.config = config or AudioProcessorConfig()

        # Initialize OpenAI client
        if not settings.ai_api_key:
            logger.warning(
                "OpenAI API key not configured - transcription will not work"
            )
            self.openai_client = None
        else:
            self.openai_client = AsyncOpenAI(api_key=settings.ai_api_key)

        # Processing state
        self.transcription_buffer: List[ProcessedAudio] = []
        self.processing_stats = {
            "total_chunks_processed": 0,
            "total_transcription_time": 0.0,
            "successful_transcriptions": 0,
            "failed_transcriptions": 0,
            "total_audio_duration": 0.0,
        }

        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._processing_lock = asyncio.Lock()

        # Temporary file management
        self.temp_dir = Path(tempfile.gettempdir()) / "tldr_audio"
        self.temp_dir.mkdir(exist_ok=True)

    async def process_audio_file(
        self,
        source: Union[str, Path],
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> AudioProcessingResult:
        """
        Process audio file and generate transcriptions.

        Args:
            source: Path to audio/video file
            start_time: Start time in seconds
            duration: Duration to process in seconds

        Returns:
            AudioProcessingResult with transcriptions
        """
        start_processing = time.time()
        processed_chunks = []

        try:
            logger.info(f"Starting audio processing for {source}")

            # Extract audio chunks
            chunks_processed = 0
            total_duration = 0.0

            async for audio_chunk in media_processor.extract_audio_chunks(
                source=source,
                chunk_duration=self.config.chunk_duration,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
            ):
                # Skip chunks before start time
                if audio_chunk.timestamp < start_time:
                    continue

                # Stop if we've exceeded duration
                if duration and audio_chunk.timestamp > start_time + duration:
                    break

                # Process chunk
                processed_chunk = await self._process_audio_chunk(audio_chunk)
                processed_chunks.append(processed_chunk)

                chunks_processed += 1
                total_duration += audio_chunk.duration

                # Update buffer
                async with self._processing_lock:
                    self.transcription_buffer.append(processed_chunk)
                    if len(self.transcription_buffer) > self.config.buffer_size:
                        self.transcription_buffer.pop(0)

                # Yield control to event loop
                await asyncio.sleep(0)

            # Calculate results
            processing_time = time.time() - start_processing
            successful_transcriptions = sum(
                1 for chunk in processed_chunks if chunk.transcription is not None
            )
            success_rate = successful_transcriptions / max(len(processed_chunks), 1)

            # Update stats
            self.processing_stats["total_chunks_processed"] += len(processed_chunks)
            self.processing_stats["total_transcription_time"] += processing_time
            self.processing_stats["successful_transcriptions"] += (
                successful_transcriptions
            )
            self.processing_stats["failed_transcriptions"] += (
                len(processed_chunks) - successful_transcriptions
            )
            self.processing_stats["total_audio_duration"] += total_duration

            result = AudioProcessingResult(
                transcriptions=processed_chunks,
                total_duration=total_duration,
                processing_time=processing_time,
                success_rate=success_rate,
                metadata={
                    "source": str(source),
                    "chunks_processed": chunks_processed,
                    "start_time": start_time,
                    "whisper_model": self.config.whisper_model,
                    "sample_rate": self.config.sample_rate,
                    "chunk_duration": self.config.chunk_duration,
                },
            )

            logger.info(
                f"Audio processing completed: {chunks_processed} chunks, "
                f"{processing_time:.2f}s, success rate: {success_rate:.2%}"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing audio {source}: {e}")
            raise

    async def process_audio_stream(
        self, audio_chunks: AsyncGenerator[AudioChunk, None]
    ) -> AsyncGenerator[ProcessedAudio, None]:
        """
        Process audio chunks from a stream.

        Args:
            audio_chunks: Async generator of audio chunks

        Yields:
            ProcessedAudio results
        """
        try:
            async for audio_chunk in audio_chunks:
                processed_chunk = await self._process_audio_chunk(audio_chunk)

                # Update buffer
                async with self._processing_lock:
                    self.transcription_buffer.append(processed_chunk)
                    if len(self.transcription_buffer) > self.config.buffer_size:
                        self.transcription_buffer.pop(0)

                yield processed_chunk

        except Exception as e:
            logger.error(f"Error processing audio stream: {e}")

    async def _process_audio_chunk(self, audio_chunk: AudioChunk) -> ProcessedAudio:
        """
        Process a single audio chunk.

        Args:
            audio_chunk: Audio chunk to process

        Returns:
            ProcessedAudio with transcription and analysis
        """
        start_time = time.time()

        try:
            # Analyze audio quality
            analysis = await self._analyze_audio_quality(audio_chunk)

            # Skip transcription for silent or very short audio
            transcription = None
            error = None

            if (
                audio_chunk.duration >= self.config.min_audio_duration
                and analysis.speech_ratio > 0.1
                and not self._is_mostly_silent(audio_chunk)
            ):
                # Transcribe audio
                transcription = await self._transcribe_audio(audio_chunk)
                if transcription is None:
                    error = "Transcription failed"
            else:
                error = "Audio too short or silent"

            processing_time = time.time() - start_time

            return ProcessedAudio(
                chunk=audio_chunk,
                transcription=transcription,
                analysis=analysis,
                processing_time=processing_time,
                error=error,
            )

        except Exception as e:
            logger.error(
                f"Error processing audio chunk at {audio_chunk.timestamp}: {e}"
            )
            processing_time = time.time() - start_time

            return ProcessedAudio(
                chunk=audio_chunk,
                transcription=None,
                analysis=AudioAnalysis(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, False, 0.0),
                processing_time=processing_time,
                error=str(e),
            )

    async def _transcribe_audio(
        self, audio_chunk: AudioChunk
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe audio chunk using OpenAI Whisper.

        Args:
            audio_chunk: Audio chunk to transcribe

        Returns:
            TranscriptionResult or None if failed
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available for transcription")
            return None

        async with self.semaphore:
            for attempt in range(self.config.retry_attempts):
                try:
                    # Create temporary audio file
                    temp_file = self.temp_dir / f"audio_{time.time()}.wav"

                    # Write audio data to file
                    with open(temp_file, "wb") as f:
                        # Write minimal WAV header
                        f.write(
                            self._create_wav_header(
                                len(audio_chunk.data),
                                audio_chunk.sample_rate,
                                audio_chunk.channels,
                            )
                        )
                        f.write(audio_chunk.data)

                    # Transcribe using OpenAI Whisper
                    with open(temp_file, "rb") as audio_file:
                        transcript = (
                            await self.openai_client.audio.transcriptions.create(
                                model=self.config.whisper_model,
                                file=audio_file,
                                language=self.config.whisper_language,
                                prompt=self.config.whisper_prompt,
                                response_format="verbose_json",
                                timestamp_granularities=["word", "segment"],
                            )
                        )

                    # Clean up temp file
                    temp_file.unlink(missing_ok=True)

                    # Extract results
                    text = transcript.text.strip()
                    confidence = getattr(
                        transcript, "confidence", 0.8
                    )  # Default confidence
                    language = getattr(transcript, "language", None)

                    # Extract words and segments if available
                    words = None
                    segments = None

                    if hasattr(transcript, "words") and transcript.words:
                        words = [
                            {
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "confidence": getattr(word, "confidence", 0.8),
                            }
                            for word in transcript.words
                        ]

                    if hasattr(transcript, "segments") and transcript.segments:
                        segments = [
                            {
                                "text": segment.text,
                                "start": segment.start,
                                "end": segment.end,
                                "confidence": getattr(segment, "confidence", 0.8),
                            }
                            for segment in transcript.segments
                        ]

                    if text:
                        logger.debug(f"Transcribed: {text[:50]}...")
                        return TranscriptionResult(
                            text=text,
                            timestamp=audio_chunk.timestamp,
                            duration=audio_chunk.duration,
                            confidence=confidence,
                            language=language,
                            words=words,
                            segments=segments,
                        )
                    else:
                        logger.debug("Empty transcription result")
                        return None

                except Exception as e:
                    logger.error(f"Transcription attempt {attempt + 1} failed: {e}")
                    if attempt == self.config.retry_attempts - 1:
                        return None
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

        return None

    async def _analyze_audio_quality(self, audio_chunk: AudioChunk) -> AudioAnalysis:
        """
        Analyze audio quality and characteristics.

        Args:
            audio_chunk: Audio chunk to analyze

        Returns:
            AudioAnalysis with quality metrics
        """
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(audio_chunk.data, dtype=np.int16)

            # Normalize to [-1, 1]
            audio_data = audio_data.astype(np.float32) / 32768.0

            # Calculate volume level (RMS)
            volume_level = np.sqrt(np.mean(audio_data**2))

            # Calculate silence ratio
            silence_threshold = self.config.silence_threshold
            silent_samples = np.sum(np.abs(audio_data) < silence_threshold)
            silence_ratio = silent_samples / len(audio_data)

            # Calculate speech ratio (inverse of silence ratio, with some filtering)
            speech_ratio = max(0.0, 1.0 - silence_ratio)

            # Calculate energy
            energy = np.sum(audio_data**2)

            # Calculate spectral centroid (approximate)
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft[: len(fft) // 2])
            freqs = np.fft.fftfreq(len(audio_data), 1 / audio_chunk.sample_rate)[
                : len(fft) // 2
            ]

            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                spectral_centroid = 0.0

            # Calculate zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
            zero_crossing_rate = zero_crossings / len(audio_data)

            # Determine if this is likely speech
            is_speech = (
                speech_ratio > 0.1
                and volume_level > 0.01
                and 100 < spectral_centroid < 8000
                and 0.01 < zero_crossing_rate < 0.5
            )

            # Calculate overall quality score
            quality_score = self._calculate_audio_quality_score(
                volume_level,
                silence_ratio,
                speech_ratio,
                spectral_centroid,
                zero_crossing_rate,
            )

            return AudioAnalysis(
                volume_level=volume_level,
                silence_ratio=silence_ratio,
                speech_ratio=speech_ratio,
                energy=energy,
                spectral_centroid=spectral_centroid,
                zero_crossing_rate=zero_crossing_rate,
                is_speech=is_speech,
                quality_score=quality_score,
            )

        except Exception as e:
            logger.error(f"Error analyzing audio quality: {e}")
            return AudioAnalysis(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, False, 0.0)

    def _calculate_audio_quality_score(
        self,
        volume_level: float,
        silence_ratio: float,
        speech_ratio: float,
        spectral_centroid: float,
        zero_crossing_rate: float,
    ) -> float:
        """Calculate overall audio quality score."""
        try:
            # Volume score (prefer moderate volume)
            volume_score = min(volume_level * 10, 1.0)  # Scale volume

            # Speech score (prefer more speech)
            speech_score = min(speech_ratio * 2, 1.0)

            # Spectral score (prefer speech frequency range)
            if 500 <= spectral_centroid <= 4000:  # Typical speech range
                spectral_score = 1.0
            else:
                spectral_score = max(0.0, 1.0 - abs(spectral_centroid - 2000) / 2000)

            # Zero crossing score (prefer moderate ZCR for speech)
            if 0.05 <= zero_crossing_rate <= 0.3:
                zcr_score = 1.0
            else:
                zcr_score = max(0.0, 1.0 - abs(zero_crossing_rate - 0.15) / 0.15)

            # Weighted combination
            quality_score = (
                volume_score * 0.3
                + speech_score * 0.3
                + spectral_score * 0.2
                + zcr_score * 0.2
            )

            return max(0.0, min(quality_score, 1.0))

        except Exception as e:
            logger.error(f"Error calculating audio quality score: {e}")
            return 0.0

    def _is_mostly_silent(self, audio_chunk: AudioChunk) -> bool:
        """Check if audio chunk is mostly silent."""
        try:
            audio_data = np.frombuffer(audio_chunk.data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0

            silent_samples = np.sum(np.abs(audio_data) < self.config.silence_threshold)
            silence_ratio = silent_samples / len(audio_data)

            return silence_ratio > 0.8

        except Exception as e:
            logger.error(f"Error checking silence: {e}")
            return True

    def _create_wav_header(
        self, data_size: int, sample_rate: int, channels: int
    ) -> bytes:
        """Create minimal WAV file header."""
        # WAV header structure
        header = bytearray(44)

        # RIFF header
        header[0:4] = b"RIFF"
        header[4:8] = (data_size + 36).to_bytes(4, "little")
        header[8:12] = b"WAVE"

        # fmt subchunk
        header[12:16] = b"fmt "
        header[16:20] = (16).to_bytes(4, "little")  # Subchunk size
        header[20:22] = (1).to_bytes(2, "little")  # PCM format
        header[22:24] = channels.to_bytes(2, "little")
        header[24:28] = sample_rate.to_bytes(4, "little")
        header[28:32] = (sample_rate * channels * 2).to_bytes(4, "little")  # Byte rate
        header[32:34] = (channels * 2).to_bytes(2, "little")  # Block align
        header[34:36] = (16).to_bytes(2, "little")  # Bits per sample

        # data subchunk
        header[36:40] = b"data"
        header[40:44] = data_size.to_bytes(4, "little")

        return bytes(header)

    async def get_recent_transcriptions(
        self, limit: int = 10, include_failed: bool = False
    ) -> List[ProcessedAudio]:
        """
        Get recent transcriptions from buffer.

        Args:
            limit: Maximum number of transcriptions to return
            include_failed: Include failed transcriptions

        Returns:
            List of ProcessedAudio objects
        """
        async with self._processing_lock:
            transcriptions = self.transcription_buffer.copy()

        if not include_failed:
            transcriptions = [t for t in transcriptions if t.transcription is not None]

        return transcriptions[-limit:] if transcriptions else []

    async def get_processing_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get processing statistics.

        Returns:
            Dictionary of processing statistics
        """
        stats = self.processing_stats.copy()

        # Calculate derived metrics
        if stats["total_chunks_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_transcription_time"] / stats["total_chunks_processed"]
            )
            stats["success_rate"] = (
                stats["successful_transcriptions"] / stats["total_chunks_processed"]
            )
        else:
            stats["average_processing_time"] = 0.0
            stats["success_rate"] = 0.0

        stats["buffer_size"] = len(self.transcription_buffer)

        return stats

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up audio processor")

        # Clear buffer
        self.transcription_buffer.clear()

        # Clean up temporary files
        try:
            for temp_file in self.temp_dir.glob("audio_*.wav"):
                temp_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")

        logger.info("Audio processor cleanup completed")


# Global audio processor instance
audio_processor = AudioProcessor()
