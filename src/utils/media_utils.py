"""
Media processing utilities for video and audio content.

This module provides utilities for video frame extraction, audio processing,
and media format handling optimized for real-time streaming scenarios.
"""

import asyncio
import io
import logging
import tempfile
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import ffmpeg
import aiofiles

logger = logging.getLogger(__name__)

try:
    import magic

    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    logger.warning("python-magic not available, using fallback MIME type detection")


@dataclass
class MediaInfo:
    """Media file information."""

    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: Optional[int] = None
    format: Optional[str] = None
    has_audio: bool = False
    audio_codec: Optional[str] = None
    audio_bitrate: Optional[int] = None
    audio_sample_rate: Optional[int] = None


@dataclass
class VideoFrame:
    """Video frame with metadata."""

    frame: np.ndarray
    timestamp: float
    frame_number: int
    width: int
    height: int
    quality_score: float = 0.0


@dataclass
class AudioChunk:
    """Audio chunk with metadata."""

    data: bytes
    timestamp: float
    duration: float
    sample_rate: int
    channels: int
    format: str


class MediaProcessor:
    """Efficient media processing with memory optimization."""

    def __init__(self, max_memory_mb: int = 500):
        self.max_memory_mb = max_memory_mb
        self.temp_dir = Path(tempfile.gettempdir()) / "tldr_media"
        self.temp_dir.mkdir(exist_ok=True)
        self._cleanup_tasks = set()

    async def get_media_info(self, source: Union[str, Path]) -> Optional[MediaInfo]:
        """
        Get media file information asynchronously.

        Args:
            source: Path to media file or stream URL

        Returns:
            MediaInfo object or None if failed
        """
        try:
            # Use ffprobe to get media information
            probe = ffmpeg.probe(str(source))

            # Get video stream info
            video_stream = None
            audio_stream = None

            for stream in probe.get("streams", []):
                if stream["codec_type"] == "video" and not video_stream:
                    video_stream = stream
                elif stream["codec_type"] == "audio" and not audio_stream:
                    audio_stream = stream

            if not video_stream:
                logger.warning(f"No video stream found in {source}")
                return None

            # Extract video information
            duration = float(video_stream.get("duration", 0))
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            fps = eval(video_stream.get("r_frame_rate", "25/1"))
            codec = video_stream.get("codec_name", "unknown")
            bitrate = (
                int(video_stream.get("bit_rate", 0))
                if video_stream.get("bit_rate")
                else None
            )
            format_name = probe.get("format", {}).get("format_name", "unknown")

            # Extract audio information
            has_audio = audio_stream is not None
            audio_codec = audio_stream.get("codec_name") if audio_stream else None
            audio_bitrate = (
                int(audio_stream.get("bit_rate", 0))
                if audio_stream and audio_stream.get("bit_rate")
                else None
            )
            audio_sample_rate = (
                int(audio_stream.get("sample_rate", 0))
                if audio_stream and audio_stream.get("sample_rate")
                else None
            )

            return MediaInfo(
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                codec=codec,
                bitrate=bitrate,
                format=format_name,
                has_audio=has_audio,
                audio_codec=audio_codec,
                audio_bitrate=audio_bitrate,
                audio_sample_rate=audio_sample_rate,
            )

        except Exception as e:
            logger.error(f"Failed to get media info for {source}: {e}")
            return None

    async def extract_frames(
        self,
        source: Union[str, Path],
        interval_seconds: float = 1.0,
        max_frames: Optional[int] = None,
        quality_threshold: float = 0.3,
        resize_width: Optional[int] = None,
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        Extract video frames at specified intervals.

        Args:
            source: Video source (file path or URL)
            interval_seconds: Interval between frames in seconds
            max_frames: Maximum number of frames to extract
            quality_threshold: Minimum quality score for frames
            resize_width: Resize frames to this width (maintains aspect ratio)

        Yields:
            VideoFrame objects
        """
        cap = None
        try:
            # Open video capture
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                return

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames"
            )

            # Calculate frame interval
            frame_interval = int(fps * interval_seconds)
            frames_extracted = 0

            for frame_number in range(0, total_frames, frame_interval):
                if max_frames and frames_extracted >= max_frames:
                    break

                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Failed to read frame {frame_number}")
                    continue

                # Calculate timestamp
                timestamp = frame_number / fps

                # Calculate quality score
                quality_score = await self._calculate_frame_quality(frame)

                if quality_score < quality_threshold:
                    logger.debug(
                        f"Skipping low quality frame {frame_number} (score: {quality_score:.2f})"
                    )
                    continue

                # Resize if requested
                if resize_width:
                    aspect_ratio = width / height
                    resize_height = int(resize_width / aspect_ratio)
                    frame = cv2.resize(frame, (resize_width, resize_height))
                    width, height = resize_width, resize_height

                # Create VideoFrame object
                video_frame = VideoFrame(
                    frame=frame,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    width=width,
                    height=height,
                    quality_score=quality_score,
                )

                yield video_frame
                frames_extracted += 1

                # Yield control to event loop
                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Error extracting frames from {source}: {e}")
        finally:
            if cap is not None:
                cap.release()

    async def extract_audio_chunks(
        self,
        source: Union[str, Path],
        chunk_duration: float = 30.0,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Extract audio chunks from video/audio source.

        Args:
            source: Audio/video source
            chunk_duration: Duration of each chunk in seconds
            sample_rate: Target sample rate
            channels: Number of audio channels

        Yields:
            AudioChunk objects
        """
        try:
            # Get media info
            media_info = await self.get_media_info(source)
            if not media_info or not media_info.has_audio:
                logger.warning(f"No audio found in {source}")
                return

            # Create temporary file for processed audio
            temp_audio_file = self.temp_dir / f"audio_{datetime.now().timestamp()}.wav"

            # Extract audio using ffmpeg
            stream = ffmpeg.input(str(source))
            stream = ffmpeg.output(
                stream,
                str(temp_audio_file),
                acodec="pcm_s16le",
                ar=sample_rate,
                ac=channels,
                format="wav",
            )

            # Run ffmpeg
            await asyncio.create_subprocess_exec(
                *ffmpeg.compile(stream),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Process audio chunks
            chunk_size = int(
                sample_rate * chunk_duration * channels * 2
            )  # 2 bytes per sample
            chunk_number = 0

            async with aiofiles.open(temp_audio_file, "rb") as f:
                # Skip WAV header (44 bytes)
                await f.seek(44)

                while True:
                    chunk_data = await f.read(chunk_size)
                    if not chunk_data:
                        break

                    timestamp = chunk_number * chunk_duration
                    actual_duration = len(chunk_data) / (sample_rate * channels * 2)

                    audio_chunk = AudioChunk(
                        data=chunk_data,
                        timestamp=timestamp,
                        duration=actual_duration,
                        sample_rate=sample_rate,
                        channels=channels,
                        format="pcm_s16le",
                    )

                    yield audio_chunk
                    chunk_number += 1

                    # Yield control to event loop
                    await asyncio.sleep(0)

            # Cleanup
            self._schedule_cleanup(temp_audio_file)

        except Exception as e:
            logger.error(f"Error extracting audio chunks from {source}: {e}")

    async def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """
        Calculate frame quality score using multiple metrics.

        Args:
            frame: OpenCV frame

        Returns:
            Quality score between 0 and 1
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate Laplacian variance (sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize

            # Calculate brightness (avoid very dark or bright frames)
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Prefer mid-brightness

            # Calculate contrast
            contrast = gray.std() / 255.0
            contrast_score = min(contrast * 2, 1.0)  # Normalize

            # Calculate histogram entropy (information content)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist / hist.sum()  # Normalize
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            entropy_score = min(entropy / 8.0, 1.0)  # Normalize

            # Weighted combination
            quality_score = (
                sharpness_score * 0.4
                + brightness_score * 0.2
                + contrast_score * 0.2
                + entropy_score * 0.2
            )

            return quality_score

        except Exception as e:
            logger.error(f"Error calculating frame quality: {e}")
            return 0.0

    def _schedule_cleanup(self, file_path: Path):
        """Schedule file cleanup."""

        async def cleanup():
            await asyncio.sleep(3600)  # Cleanup after 1 hour
            try:
                file_path.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Failed to cleanup {file_path}: {e}")

        task = asyncio.create_task(cleanup())
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def frame_to_bytes(
        self, frame: VideoFrame, format: str = "JPEG", quality: int = 85
    ) -> bytes:
        """
        Convert video frame to bytes.

        Args:
            frame: VideoFrame object
            format: Image format (JPEG, PNG, etc.)
            quality: Compression quality (1-100)

        Returns:
            Image bytes
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)

            # Save to bytes
            buffer = io.BytesIO()
            if format.upper() == "JPEG":
                pil_image.save(buffer, format=format, quality=quality, optimize=True)
            else:
                pil_image.save(buffer, format=format)

            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error converting frame to bytes: {e}")
            return b""

    async def detect_scene_changes(
        self, frames: List[VideoFrame], threshold: float = 0.3
    ) -> List[int]:
        """
        Detect scene changes in a sequence of frames.

        Args:
            frames: List of VideoFrame objects
            threshold: Scene change threshold (0-1)

        Returns:
            List of frame indices where scene changes occur
        """
        scene_changes = []

        if len(frames) < 2:
            return scene_changes

        try:
            prev_hist = None

            for i, frame in enumerate(frames):
                # Convert to grayscale and calculate histogram
                gray = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

                if prev_hist is not None:
                    # Calculate histogram correlation
                    correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

                    # If correlation is low, it's likely a scene change
                    if correlation < (1.0 - threshold):
                        scene_changes.append(i)
                        logger.debug(
                            f"Scene change detected at frame {i} (correlation: {correlation:.3f})"
                        )

                prev_hist = hist

            return scene_changes

        except Exception as e:
            logger.error(f"Error detecting scene changes: {e}")
            return []

    async def get_file_mime_type(self, file_path: Union[str, Path]) -> str:
        """
        Get MIME type of a file.

        Args:
            file_path: Path to file

        Returns:
            MIME type string
        """
        try:
            if HAS_MAGIC:
                return magic.from_file(str(file_path), mime=True)
            else:
                # Fallback to mimetypes module
                import mimetypes

                mime_type, _ = mimetypes.guess_type(str(file_path))
                return mime_type or "application/octet-stream"
        except Exception as e:
            logger.error(f"Error getting MIME type for {file_path}: {e}")
            return "application/octet-stream"

    async def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file format is supported for processing.

        Args:
            file_path: Path to file

        Returns:
            True if format is supported
        """
        supported_formats = {
            "video/mp4",
            "video/avi",
            "video/mov",
            "video/mkv",
            "video/webm",
            "video/flv",
            "video/wmv",
            "video/m4v",
            "audio/mp3",
            "audio/wav",
            "audio/aac",
            "audio/m4a",
            "audio/ogg",
            "audio/flac",
        }

        mime_type = await self.get_file_mime_type(file_path)
        return mime_type in supported_formats

    def __del__(self):
        """Cleanup on deletion."""
        # Cancel all cleanup tasks
        for task in self._cleanup_tasks:
            if not task.done():
                task.cancel()


# Global media processor instance
media_processor = MediaProcessor()


class StreamCapture:
    """Optimized stream capture for real-time processing."""

    def __init__(self, stream_url: str, buffer_size: int = 10):
        self.stream_url = stream_url
        self.buffer_size = buffer_size
        self.frame_buffer = asyncio.Queue(maxsize=buffer_size)
        self.capture_task = None
        self.is_running = False

    async def start_capture(self):
        """Start capturing frames from stream."""
        if self.is_running:
            return

        self.is_running = True
        self.capture_task = asyncio.create_task(self._capture_loop())

    async def stop_capture(self):
        """Stop capturing frames."""
        if not self.is_running:
            return

        self.is_running = False
        if self.capture_task:
            self.capture_task.cancel()
            try:
                await self.capture_task
            except asyncio.CancelledError:
                pass

    async def get_frame(self, timeout: float = 1.0) -> Optional[VideoFrame]:
        """
        Get next frame from buffer.

        Args:
            timeout: Timeout in seconds

        Returns:
            VideoFrame or None if timeout
        """
        try:
            return await asyncio.wait_for(self.frame_buffer.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def _capture_loop(self):
        """Main capture loop."""
        cap = None
        try:
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                logger.error(f"Failed to open stream: {self.stream_url}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_number = 0

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from stream")
                    await asyncio.sleep(0.1)
                    continue

                # Calculate timestamp
                timestamp = frame_number / fps

                # Create VideoFrame
                video_frame = VideoFrame(
                    frame=frame,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    quality_score=1.0,  # Assume good quality for live streams
                )

                # Add to buffer (non-blocking)
                try:
                    self.frame_buffer.put_nowait(video_frame)
                except asyncio.QueueFull:
                    # Remove oldest frame to make room
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait(video_frame)
                    except asyncio.QueueEmpty:
                        pass

                frame_number += 1

                # Yield control to event loop
                await asyncio.sleep(1 / fps)

        except Exception as e:
            logger.error(f"Error in capture loop: {e}")
        finally:
            if cap is not None:
                cap.release()
