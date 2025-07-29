"""
Google Gemini integration for unified video, audio, and text processing.

This module provides a unified processor that leverages Google Gemini's
native multimodal capabilities for video understanding, audio transcription,
and content analysis without requiring separate frame extraction or audio processing.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union, Any
import json
import tempfile
from enum import Enum

from pydantic import BaseModel, Field

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logging.warning(
        "Google Generative AI SDK not installed. Gemini integration will not work."
    )

from src.utils.media_utils import AudioChunk, VideoFrame, media_processor
from src.core.config import settings

logger = logging.getLogger(__name__)


class GeminiModel(str, Enum):
    """Available Gemini models for different use cases."""

    FLASH_2_0 = "gemini-2.0-flash-exp"
    PRO_1_5 = "gemini-1.5-pro"
    FLASH_1_5 = "gemini-1.5-flash"


class ProcessingMode(str, Enum):
    """Processing modes for different content types."""

    FILE_API = "file_api"  # For batch processing
    LIVE_API = "live_api"  # For real-time streaming
    DIRECT_URL = "direct_url"  # For YouTube URLs


class GeminiProcessorConfig(BaseModel):
    """Configuration for Gemini processor."""

    # Model settings
    model_name: GeminiModel = Field(
        default=GeminiModel.FLASH_2_0, description="Gemini model to use"
    )

    # Processing settings
    default_mode: ProcessingMode = Field(
        default=ProcessingMode.FILE_API, description="Default processing mode"
    )
    video_fps: float = Field(
        default=1.0, description="Video sampling rate (max 1 FPS for Gemini)"
    )
    audio_bitrate: int = Field(
        default=1000, description="Audio processing bitrate (1Kbps for Gemini)"
    )

    # Context window settings
    max_video_duration_seconds: int = Field(
        default=3600,  # 1 hour for 1M context
        description="Maximum video duration to process",
    )
    chunk_duration_seconds: int = Field(
        default=300,  # 5-minute chunks
        description="Duration of video chunks for processing",
    )

    # Prompt settings
    system_prompt: str = Field(
        default="""You are an AI assistant specialized in identifying highlight moments in video content. 
        Analyze the provided video/audio content and identify exciting, important, or memorable moments.
        Consider visual activity, audio cues, speech content, and overall engagement level.
        Provide timestamps and confidence scores for each highlight moment.""",
        description="System prompt for highlight detection",
    )

    # Highlight detection parameters
    highlight_prompt_template: str = Field(
        default="""Analyze this video segment and identify highlight moments. For each highlight:
        1. Provide the start and end timestamps (in seconds)
        2. Rate the highlight quality (0.0-1.0)
        3. Explain why this is a highlight
        4. Categorize the highlight type (action, emotional, informative, etc.)
        
        Format your response as JSON:
        {
            "highlights": [
                {
                    "start_time": float,
                    "end_time": float,
                    "score": float,
                    "confidence": float,
                    "reason": string,
                    "category": string,
                    "key_moments": [string],
                    "transcription": string (if applicable)
                }
            ],
            "overall_quality": float,
            "content_summary": string
        }""",
        description="Prompt template for highlight extraction",
    )

    # API settings
    api_timeout_seconds: int = Field(default=300, description="API request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay_seconds: float = Field(default=1.0, description="Delay between retries")

    # File API settings
    max_file_size_mb: int = Field(
        default=2048,  # 2GB limit
        description="Maximum file size for File API",
    )
    file_retention_hours: int = Field(
        default=48, description="File retention period in File API"
    )

    # Response parsing
    temperature: float = Field(
        default=0.3, description="Temperature for response generation"
    )
    max_output_tokens: int = Field(
        default=8192, description="Maximum tokens in response"
    )


@dataclass
class GeminiHighlight:
    """Represents a highlight detected by Gemini."""

    start_time: float
    end_time: float
    score: float
    confidence: float
    reason: str
    category: str
    key_moments: List[str]
    transcription: Optional[str] = None
    visual_description: Optional[str] = None
    audio_description: Optional[str] = None

    @property
    def duration(self) -> float:
        """Get highlight duration in seconds."""
        return self.end_time - self.start_time

    @property
    def midpoint(self) -> float:
        """Get highlight midpoint timestamp."""
        return (self.start_time + self.end_time) / 2


@dataclass
class GeminiProcessingResult:
    """Result of Gemini processing operation."""

    highlights: List[GeminiHighlight]
    overall_quality: float
    content_summary: str
    processing_time: float
    mode_used: ProcessingMode
    model_used: str
    total_duration: float
    metadata: Dict[str, Any]
    error: Optional[str] = None


class GeminiProcessor:
    """
    Unified processor using Google Gemini for video understanding.

    Features:
    - Native video understanding without frame extraction
    - Integrated audio transcription
    - Real-time streaming support via Live API
    - Batch processing via File API
    - Direct YouTube URL processing
    - Multimodal highlight detection
    """

    def __init__(self, config: Optional[GeminiProcessorConfig] = None):
        self.config = config or GeminiProcessorConfig()

        # Initialize Gemini
        if not genai:
            raise ImportError(
                "google-generativeai package is required for Gemini integration"
            )

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not configured in settings")

        genai.configure(api_key=settings.gemini_api_key)

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
                "response_mime_type": "application/json",
            },
        )

        # Processing state
        self.processing_stats = {
            "total_videos_processed": 0,
            "total_highlights_found": 0,
            "total_processing_time": 0.0,
            "successful_processes": 0,
            "failed_processes": 0,
            "total_video_duration": 0.0,
        }

        # Temporary file management
        self.temp_dir = Path(tempfile.gettempdir()) / "tldr_gemini"
        self.temp_dir.mkdir(exist_ok=True)

        logger.info(
            f"Initialized Gemini processor with model: {self.config.model_name}"
        )

    async def process_video_file(
        self,
        source: Union[str, Path],
        mode: Optional[ProcessingMode] = None,
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> GeminiProcessingResult:
        """
        Process video file using Gemini's native video understanding.

        Args:
            source: Path to video file or YouTube URL
            mode: Processing mode to use (auto-detected if None)
            start_time: Start time in seconds
            duration: Duration to process in seconds

        Returns:
            GeminiProcessingResult with detected highlights
        """
        start_processing = time.time()
        mode = mode or self._detect_processing_mode(str(source))

        try:
            logger.info(f"Processing video {source} with mode: {mode}")

            if mode == ProcessingMode.DIRECT_URL:
                result = await self._process_youtube_url(
                    str(source), start_time, duration
                )
            elif mode == ProcessingMode.FILE_API:
                result = await self._process_with_file_api(source, start_time, duration)
            elif mode == ProcessingMode.LIVE_API:
                result = await self._process_with_live_api(source, start_time, duration)
            else:
                raise ValueError(f"Unsupported processing mode: {mode}")

            # Update statistics
            processing_time = time.time() - start_processing
            self.processing_stats["total_videos_processed"] += 1
            self.processing_stats["total_highlights_found"] += len(result.highlights)
            self.processing_stats["total_processing_time"] += processing_time
            self.processing_stats["successful_processes"] += 1
            self.processing_stats["total_video_duration"] += result.total_duration

            result.processing_time = processing_time

            logger.info(
                f"Video processing completed: {len(result.highlights)} highlights found, "
                f"{processing_time:.2f}s processing time"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing video {source}: {e}")
            self.processing_stats["failed_processes"] += 1

            return GeminiProcessingResult(
                highlights=[],
                overall_quality=0.0,
                content_summary="Processing failed",
                processing_time=time.time() - start_processing,
                mode_used=mode,
                model_used=self.config.model_name,
                total_duration=0.0,
                metadata={"error": str(e)},
                error=str(e),
            )

    async def process_video_stream(
        self,
        video_chunks: AsyncGenerator[VideoFrame, None],
        audio_chunks: Optional[AsyncGenerator[AudioChunk, None]] = None,
    ) -> AsyncGenerator[GeminiProcessingResult, None]:
        """
        Process video stream in real-time using Live API.

        Args:
            video_chunks: Async generator of video frames
            audio_chunks: Optional async generator of audio chunks

        Yields:
            GeminiProcessingResult for each processed segment
        """
        try:
            # Initialize Live API session
            session = self.model.start_live_session()

            buffer = []
            buffer_duration = 0.0
            last_timestamp = 0.0

            async for frame in video_chunks:
                buffer.append(frame)
                buffer_duration = frame.timestamp - last_timestamp

                # Process when buffer reaches chunk duration
                if buffer_duration >= self.config.chunk_duration_seconds:
                    result = await self._process_stream_buffer(session, buffer)
                    yield result

                    # Clear buffer
                    buffer = []
                    last_timestamp = frame.timestamp

            # Process remaining buffer
            if buffer:
                result = await self._process_stream_buffer(session, buffer)
                yield result

            # Close session
            session.close()

        except Exception as e:
            logger.error(f"Error processing video stream: {e}")

    async def _process_youtube_url(
        self, url: str, start_time: float, duration: Optional[float]
    ) -> GeminiProcessingResult:
        """Process YouTube video directly via URL."""
        try:
            # Prepare prompt with timestamp constraints
            prompt_parts = [self.config.system_prompt]

            if start_time > 0 or duration:
                time_constraint = "Analyze the video"
                if start_time > 0:
                    time_constraint += f" starting from {start_time} seconds"
                if duration:
                    time_constraint += f" for {duration} seconds"
                prompt_parts.append(time_constraint)

            prompt_parts.append(self.config.highlight_prompt_template)

            # Generate content with YouTube URL
            response = await asyncio.to_thread(
                self.model.generate_content, [url] + prompt_parts
            )

            # Parse response
            result_data = json.loads(response.text)
            highlights = self._parse_highlights(result_data, start_time)

            return GeminiProcessingResult(
                highlights=highlights,
                overall_quality=result_data.get("overall_quality", 0.5),
                content_summary=result_data.get("content_summary", ""),
                processing_time=0.0,  # Will be set by caller
                mode_used=ProcessingMode.DIRECT_URL,
                model_used=self.config.model_name,
                total_duration=duration or 0.0,
                metadata={"url": url},
            )

        except Exception as e:
            logger.error(f"Error processing YouTube URL {url}: {e}")
            raise

    async def _process_with_file_api(
        self, source: Union[str, Path], start_time: float, duration: Optional[float]
    ) -> GeminiProcessingResult:
        """Process video using File API for batch processing."""
        try:
            # Upload file to Gemini
            logger.debug(f"Uploading file {source} to Gemini File API")

            video_file = genai.upload_file(str(source))

            # Wait for file to be processed
            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(1)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise Exception(f"File upload failed: {video_file.state.name}")

            logger.debug(f"File uploaded successfully: {video_file.uri}")

            # Prepare prompt
            prompt_parts = [
                video_file,
                self.config.system_prompt,
                self.config.highlight_prompt_template,
            ]

            if start_time > 0 or duration:
                time_constraint = "Focus on the video segment"
                if start_time > 0:
                    time_constraint += f" starting from {start_time} seconds"
                if duration:
                    time_constraint += f" for {duration} seconds"
                prompt_parts.insert(1, time_constraint)

            # Generate content
            response = await asyncio.to_thread(
                self.model.generate_content, prompt_parts
            )

            # Parse response
            result_data = json.loads(response.text)
            highlights = self._parse_highlights(result_data, start_time)

            # Get video metadata
            video_metadata = await self._get_video_metadata(source)

            # Clean up uploaded file
            genai.delete_file(video_file.name)

            return GeminiProcessingResult(
                highlights=highlights,
                overall_quality=result_data.get("overall_quality", 0.5),
                content_summary=result_data.get("content_summary", ""),
                processing_time=0.0,  # Will be set by caller
                mode_used=ProcessingMode.FILE_API,
                model_used=self.config.model_name,
                total_duration=video_metadata.get("duration", duration or 0.0),
                metadata={"file_uri": video_file.uri, "video_metadata": video_metadata},
            )

        except Exception as e:
            logger.error(f"Error processing with File API: {e}")
            raise

    async def _process_with_live_api(
        self, source: Union[str, Path], start_time: float, duration: Optional[float]
    ) -> GeminiProcessingResult:
        """Process video using Live API for streaming."""
        try:
            # For file-based Live API processing, we'll simulate streaming
            # In production, this would connect to actual live streams

            # Initialize Live session
            session = await asyncio.to_thread(self.model.start_chat, history=[])

            # Send initial prompt
            await asyncio.to_thread(session.send_message, self.config.system_prompt)

            # Process video in chunks
            highlights = []
            chunk_size = self.config.chunk_duration_seconds
            current_time = start_time
            end_time = start_time + duration if duration else float("inf")

            # Extract video chunks and process
            async for chunk_data in self._extract_video_chunks(
                source, start_time, duration
            ):
                if current_time >= end_time:
                    break

                # Send chunk for analysis
                chunk_prompt = (
                    f"Analyze this video segment from {current_time} to {current_time + chunk_size} seconds. "
                    + self.config.highlight_prompt_template
                )

                response = await asyncio.to_thread(
                    session.send_message, [chunk_data, chunk_prompt]
                )

                # Parse chunk highlights
                try:
                    chunk_result = json.loads(response.text)
                    chunk_highlights = self._parse_highlights(
                        chunk_result, current_time
                    )
                    highlights.extend(chunk_highlights)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse response for chunk at {current_time}s"
                    )

                current_time += chunk_size

            # Get final summary
            summary_response = await asyncio.to_thread(
                session.send_message,
                "Provide an overall summary of the content and rate its quality (0.0-1.0).",
            )

            try:
                summary_data = json.loads(summary_response.text)
                overall_quality = summary_data.get("quality", 0.5)
                content_summary = summary_data.get("summary", "")
            except Exception:
                overall_quality = 0.5
                content_summary = "Video processed successfully"

            return GeminiProcessingResult(
                highlights=highlights,
                overall_quality=overall_quality,
                content_summary=content_summary,
                processing_time=0.0,  # Will be set by caller
                mode_used=ProcessingMode.LIVE_API,
                model_used=self.config.model_name,
                total_duration=current_time - start_time,
                metadata={
                    "chunks_processed": int((current_time - start_time) / chunk_size)
                },
            )

        except Exception as e:
            logger.error(f"Error processing with Live API: {e}")
            raise

    async def _extract_video_chunks(
        self, source: Union[str, Path], start_time: float, duration: Optional[float]
    ) -> AsyncGenerator[bytes, None]:
        """Extract video chunks for Live API processing."""
        # This is a placeholder - in production, this would extract actual video chunks
        # For now, we'll use the media processor to extract frames

        chunk_duration = self.config.chunk_duration_seconds
        current_time = start_time
        end_time = start_time + duration if duration else float("inf")

        while current_time < end_time:
            # Extract chunk using ffmpeg or similar
            chunk_file = self.temp_dir / f"chunk_{current_time}.mp4"

            # Use media processor to extract chunk
            await media_processor.extract_video_segment(
                source=source,
                output=chunk_file,
                start_time=current_time,
                duration=min(chunk_duration, end_time - current_time),
            )

            # Read chunk data
            with open(chunk_file, "rb") as f:
                chunk_data = f.read()

            yield chunk_data

            # Clean up
            chunk_file.unlink(missing_ok=True)

            current_time += chunk_duration

    async def _process_stream_buffer(
        self, session: Any, buffer: List[VideoFrame]
    ) -> GeminiProcessingResult:
        """Process buffered frames in a live session."""
        if not buffer:
            return GeminiProcessingResult(
                highlights=[],
                overall_quality=0.0,
                content_summary="Empty buffer",
                processing_time=0.0,
                mode_used=ProcessingMode.LIVE_API,
                model_used=self.config.model_name,
                total_duration=0.0,
                metadata={},
            )

        try:
            # Convert frames to video
            start_time = buffer[0].timestamp
            end_time = buffer[-1].timestamp
            duration = end_time - start_time

            # Create temporary video from frames
            temp_video = self.temp_dir / f"buffer_{start_time}.mp4"
            await self._frames_to_video(buffer, temp_video)

            # Send to Live API
            with open(temp_video, "rb") as f:
                video_data = f.read()

            prompt = (
                f"Analyze this video segment from {start_time} to {end_time} seconds. "
                + self.config.highlight_prompt_template
            )

            response = await asyncio.to_thread(
                session.send_message, [video_data, prompt]
            )

            # Parse response
            result_data = json.loads(response.text)
            highlights = self._parse_highlights(result_data, start_time)

            # Clean up
            temp_video.unlink(missing_ok=True)

            return GeminiProcessingResult(
                highlights=highlights,
                overall_quality=result_data.get("overall_quality", 0.5),
                content_summary=result_data.get("content_summary", ""),
                processing_time=0.0,
                mode_used=ProcessingMode.LIVE_API,
                model_used=self.config.model_name,
                total_duration=duration,
                metadata={"frame_count": len(buffer)},
            )

        except Exception as e:
            logger.error(f"Error processing stream buffer: {e}")
            return GeminiProcessingResult(
                highlights=[],
                overall_quality=0.0,
                content_summary="Processing failed",
                processing_time=0.0,
                mode_used=ProcessingMode.LIVE_API,
                model_used=self.config.model_name,
                total_duration=0.0,
                metadata={"error": str(e)},
                error=str(e),
            )

    def _parse_highlights(
        self, result_data: Dict[str, Any], time_offset: float = 0.0
    ) -> List[GeminiHighlight]:
        """Parse highlights from Gemini response."""
        highlights = []

        for highlight_data in result_data.get("highlights", []):
            try:
                highlight = GeminiHighlight(
                    start_time=highlight_data["start_time"] + time_offset,
                    end_time=highlight_data["end_time"] + time_offset,
                    score=highlight_data.get("score", 0.5),
                    confidence=highlight_data.get("confidence", 0.7),
                    reason=highlight_data.get("reason", ""),
                    category=highlight_data.get("category", "general"),
                    key_moments=highlight_data.get("key_moments", []),
                    transcription=highlight_data.get("transcription"),
                    visual_description=highlight_data.get("visual_description"),
                    audio_description=highlight_data.get("audio_description"),
                )
                highlights.append(highlight)
            except Exception as e:
                logger.warning(f"Failed to parse highlight: {e}")

        return highlights

    def _detect_processing_mode(self, source: str) -> ProcessingMode:
        """Auto-detect the best processing mode based on source."""
        if "youtube.com" in source or "youtu.be" in source:
            return ProcessingMode.DIRECT_URL
        elif source.startswith("rtmp://") or source.startswith("rtsp://"):
            return ProcessingMode.LIVE_API
        else:
            return ProcessingMode.FILE_API

    async def _get_video_metadata(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Get video metadata using media processor."""
        try:
            metadata = await media_processor.get_media_info(source)
            return metadata
        except Exception as e:
            logger.warning(f"Failed to get video metadata: {e}")
            return {}

    async def _frames_to_video(
        self, frames: List[VideoFrame], output_path: Path
    ) -> None:
        """Convert frames to video file.
        
        Args:
            frames: List of video frames to convert
            output_path: Path to save the output video
            
        Raises:
            NotImplementedError: This method requires proper video encoding implementation
        """
        raise NotImplementedError(
            "Video encoding from frames is not yet implemented. "
            "Consider using ffmpeg-python or opencv for production use."
        )

    async def get_processing_stats(self) -> Dict[str, Union[int, float]]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()

        # Calculate derived metrics
        if stats["total_videos_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["total_videos_processed"]
            )
            stats["average_highlights_per_video"] = (
                stats["total_highlights_found"] / stats["total_videos_processed"]
            )
            stats["success_rate"] = (
                stats["successful_processes"] / stats["total_videos_processed"]
            )
        else:
            stats["average_processing_time"] = 0.0
            stats["average_highlights_per_video"] = 0.0
            stats["success_rate"] = 0.0

        return stats

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up Gemini processor")

        # Clean up temporary files
        try:
            for temp_file in self.temp_dir.glob("*"):
                temp_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")

        logger.info("Gemini processor cleanup completed")


# Global Gemini processor instance
gemini_processor = None


def initialize_gemini_processor(config: Optional[GeminiProcessorConfig] = None):
    """Initialize the global Gemini processor instance."""
    global gemini_processor
    if settings.gemini_api_key:
        try:
            gemini_processor = GeminiProcessor(config)
            logger.info("Gemini processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini processor: {e}")
            gemini_processor = None
    else:
        logger.warning(
            "GEMINI_API_KEY not configured - Gemini processor not initialized"
        )
