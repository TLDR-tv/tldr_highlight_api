"""FFmpeg-based stream processor with robust error handling and reconnection."""

import asyncio
import csv
import logging
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional, Protocol, List
from uuid import UUID

logger = logging.getLogger(__name__)


class StreamFormat(Enum):
    """Supported stream formats."""

    RTMP = "rtmp"
    HLS = "hls"
    DASH = "dash"
    HTTP = "http"
    FILE = "file"


@dataclass
class FFmpegConfig:
    """FFmpeg processing configuration."""

    # Input settings
    reconnect: bool = True
    reconnect_attempts: int = 10
    reconnect_delay: int = 5
    timeout: int = 30
    rtsp_transport: str = "tcp"

    # Output settings
    segment_duration: int = 120  # 2 minutes for video segments
    audio_segment_duration: int = 30  # 30 seconds for audio chunks
    audio_overlap: int = 5  # 5 seconds overlap for wake word detection
    force_keyframes: bool = True
    keyframe_interval: int = 2  # seconds

    # Video settings
    video_codec: str = "copy"  # copy for efficiency
    preset: str = "fast"
    crf: int = 23

    # Audio settings
    audio_codec: str = "copy"
    audio_sample_rate: int = 16000  # 16kHz for WhisperX
    audio_channels: int = 1  # Mono for WhisperX

    # Ring buffer settings for memory efficiency
    video_ring_buffer_size: int = 10  # Keep last 10 video segments
    audio_ring_buffer_size: int = 50  # Keep last 50 audio chunks

    # Performance settings
    threads: int = 0  # auto-detect
    buffer_size: int = 8192

    # Debug settings
    log_level: str = "error"
    stats: bool = True


@dataclass
class StreamSegment:
    """Represents a video segment."""

    segment_id: UUID
    path: Path
    start_time: float
    duration: float
    segment_number: int
    size_bytes: int
    is_complete: bool = False
    error: Optional[str] = None
    # Associated audio chunks for this video segment
    audio_chunks: list["AudioChunk"] = None
    
    def __post_init__(self):
        if self.audio_chunks is None:
            self.audio_chunks = []


@dataclass
class AudioChunk:
    """Represents an audio chunk for transcription."""
    
    chunk_id: UUID
    path: Path
    start_time: float
    end_time: float
    duration: float
    chunk_number: int
    size_bytes: int
    is_complete: bool = False
    error: Optional[str] = None
    # Which video segments this audio chunk overlaps with
    overlapping_segments: list[int] = None
    
    def __post_init__(self):
        if self.overlapping_segments is None:
            self.overlapping_segments = []


class SegmentHandler(Protocol):
    """Protocol for handling completed segments."""

    async def handle_segment(self, segment: StreamSegment) -> None:
        """Handle a completed segment."""
        ...


class FFmpegProcessor:
    """Robust FFmpeg stream processor with error recovery."""

    def __init__(
        self,
        stream_url: str,
        output_dir: Path,
        config: Optional[FFmpegConfig] = None,
    ):
        """Initialize FFmpeg processor.

        Args:
            stream_url: URL of the stream to process
            output_dir: Directory for output segments
            config: FFmpeg configuration

        """
        self.stream_url = stream_url
        self.output_dir = output_dir
        self.config = config or FFmpegConfig()

        # Process management
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._segment_counter = 0
        self._audio_chunk_counter = 0
        self._retry_count = 0
        self._last_error_time = 0.0

        # Create subdirectories
        self.video_dir = self.output_dir / "video"
        self.audio_dir = self.output_dir / "audio"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Ring buffers for segments
        self._video_ring_buffer = []
        self._audio_ring_buffer = []

    @property
    def format(self) -> StreamFormat:
        """Detect stream format from URL."""
        url_lower = self.stream_url.lower()

        if url_lower.startswith("rtmp://"):
            return StreamFormat.RTMP
        elif ".m3u8" in url_lower or "hls" in url_lower:
            return StreamFormat.HLS
        elif ".mpd" in url_lower or "dash" in url_lower:
            return StreamFormat.DASH
        elif url_lower.startswith(("http://", "https://")):
            return StreamFormat.HTTP
        else:
            return StreamFormat.FILE

    def _build_input_args(self) -> list[str]:
        """Build FFmpeg input arguments based on stream format."""
        args = []

        # Common reconnection flags
        if self.config.reconnect and self.format != StreamFormat.FILE:
            args.extend(
                [
                    "-reconnect",
                    "1",
                    "-reconnect_at_eof",
                    "1",
                    "-reconnect_streamed",
                    "1",
                    "-reconnect_delay_max",
                    str(self.config.reconnect_delay),
                ]
            )

        # Format-specific settings
        if self.format == StreamFormat.RTMP:
            args.extend(
                [
                    "-rtmp_live",
                    "live",
                    "-rtmp_buffer",
                    str(self.config.buffer_size),
                ]
            )
        elif self.format == StreamFormat.HLS:
            args.extend(
                [
                    "-http_persistent",
                    "1",
                    "-http_multiple",
                    "1",
                ]
            )

        # Timeout settings
        if self.format != StreamFormat.FILE:
            args.extend(
                [
                    "-timeout",
                    str(self.config.timeout * 1000000),  # microseconds
                    "-stimeout",
                    str(self.config.timeout * 1000000),
                ]
            )

        # RTSP transport (if applicable)
        if "rtsp://" in self.stream_url.lower():
            args.extend(["-rtsp_transport", self.config.rtsp_transport])

        # Input URL
        args.extend(["-i", self.stream_url])

        return args

    def _build_output_args(self, segment_pattern: str) -> list[str]:
        """Build FFmpeg output arguments."""
        args = []

        # Video codec settings
        if self.config.video_codec == "copy":
            args.extend(["-c:v", "copy"])
        else:
            args.extend(
                [
                    "-c:v",
                    self.config.video_codec,
                    "-preset",
                    self.config.preset,
                    "-crf",
                    str(self.config.crf),
                ]
            )

        # Audio codec settings
        args.extend(["-c:a", self.config.audio_codec])

        # Force keyframes for consistent segmentation
        if self.config.force_keyframes and self.config.video_codec != "copy":
            keyframe_expr = f"expr:gte(t,n_forced*{self.config.keyframe_interval})"
            args.extend(["-force_key_frames", keyframe_expr])

        # Segment settings
        args.extend(
            [
                "-f",
                "segment",
                "-segment_time",
                str(self.config.segment_duration),
                "-segment_format",
                "mp4",
                "-segment_atclocktime",
                "1",
                "-reset_timestamps",
                "1",
                "-movflags",
                "+faststart",
            ]
        )

        # Force keyframe split for segments
        if self.config.force_keyframes:
            args.extend(["-segment_format_options", "movflags=+frag_keyframe"])

        # Add segment list in CSV format
        segment_list_path = str(self.output_dir / "segments.csv")
        args.extend(
            [
                "-segment_list",
                segment_list_path,
                "-segment_list_type",
                "csv",
            ]
        )

        # Output pattern
        args.append(segment_pattern)

        return args

    def _build_ffmpeg_command(self) -> list[str]:
        """Build complete FFmpeg command."""
        # Segment file pattern
        segment_pattern = str(self.video_dir / "segment_%05d.mp4")

        # Build command
        cmd = ["ffmpeg", "-y"]  # overwrite output files

        # Log level
        cmd.extend(["-loglevel", self.config.log_level])

        # Stats
        if self.config.stats:
            cmd.append("-stats")

        # Thread settings
        if self.config.threads:
            cmd.extend(["-threads", str(self.config.threads)])

        # Input arguments
        cmd.extend(self._build_input_args())

        # Output arguments
        cmd.extend(self._build_output_args(segment_pattern))

        return cmd

    async def start(self) -> None:
        """Start FFmpeg process."""
        if self._running:
            logger.warning("FFmpeg processor already running")
            return

        self._running = True
        cmd = self._build_ffmpeg_command()

        logger.info(f"Starting FFmpeg with command: {shlex.join(cmd)}")

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )
            logger.info(f"FFmpeg process started with PID: {self._process.pid}")

        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            self._running = False
            raise

    async def stop(self) -> None:
        """Stop FFmpeg process gracefully."""
        if not self._running or not self._process:
            return

        self._running = False
        logger.info("Stopping FFmpeg processor")

        try:
            # Send SIGTERM for graceful shutdown
            if os.name != "nt":
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            else:
                self._process.terminate()

            # Wait for process to exit (with timeout)
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except asyncio.TimeoutError:
                logger.warning("FFmpeg didn't exit gracefully, forcing kill")
                if os.name != "nt":
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                else:
                    self._process.kill()
                await self._process.wait()

        except Exception as e:
            logger.error(f"Error stopping FFmpeg: {e}")
        finally:
            self._process = None

    async def process_stream(
        self, segment_handler: SegmentHandler
    ) -> AsyncIterator[StreamSegment]:
        """Process stream and yield segments.

        Args:
            segment_handler: Handler for completed segments

        Yields:
            Completed stream segments

        """
        while self._running:
            try:
                # Start or restart FFmpeg
                if not self._process or self._process.returncode is not None:
                    await self._handle_restart()

                # Monitor for new segments
                async for segment in self._monitor_segments():
                    # Handle segment
                    await segment_handler.handle_segment(segment)
                    yield segment

            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                await self._handle_error(e)

    async def _extract_audio_chunks(self, video_segment: StreamSegment) -> list[AudioChunk]:
        """Extract overlapping audio chunks from a video segment."""
        audio_chunks = []
        segment_start = video_segment.start_time
        segment_duration = video_segment.duration
        
        # Calculate audio chunk positions with overlap
        chunk_start = segment_start
        
        while chunk_start < segment_start + segment_duration:
            # Calculate chunk duration
            chunk_duration = min(
                self.config.audio_segment_duration,
                (segment_start + segment_duration) - chunk_start
            )
            
            # Skip if chunk is too short
            if chunk_duration < 5:
                break
            
            # Extract audio chunk
            chunk = await self._extract_single_audio_chunk(
                video_path=video_segment.path,
                start_time=chunk_start - segment_start,  # Relative to video start
                duration=chunk_duration,
                chunk_number=self._audio_chunk_counter,
                absolute_start_time=chunk_start,
            )
            
            if chunk:
                audio_chunks.append(chunk)
                self._audio_chunk_counter += 1
                
                # Add to ring buffer and clean old chunks
                self._audio_ring_buffer.append(chunk)
                if len(self._audio_ring_buffer) > self.config.audio_ring_buffer_size:
                    old_chunk = self._audio_ring_buffer.pop(0)
                    try:
                        old_chunk.path.unlink()
                        logger.debug(f"Deleted old audio chunk: {old_chunk.path}")
                    except Exception as e:
                        logger.error(f"Failed to delete audio chunk: {e}")
            
            # Move to next chunk with overlap
            chunk_start += self.config.audio_segment_duration - self.config.audio_overlap
        
        return audio_chunks
    
    async def _extract_single_audio_chunk(
        self,
        video_path: Path,
        start_time: float,
        duration: float,
        chunk_number: int,
        absolute_start_time: float,
    ) -> Optional[AudioChunk]:
        """Extract a single audio chunk from video."""
        chunk_id = UUID(int=chunk_number)  # Deterministic UUID
        output_path = self.audio_dir / f"chunk_{chunk_number:06d}.wav"
        
        # Use segment muxer with forced segment times for consistency
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM for WhisperX
            "-ar", str(self.config.audio_sample_rate),
            "-ac", str(self.config.audio_channels),
            str(output_path),
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and output_path.exists():
                stats = output_path.stat()
                if stats.st_size > 0:
                    return AudioChunk(
                        chunk_id=chunk_id,
                        path=output_path,
                        start_time=absolute_start_time,
                        end_time=absolute_start_time + duration,
                        duration=duration,
                        chunk_number=chunk_number,
                        size_bytes=stats.st_size,
                        is_complete=True,
                    )
            
            logger.error(f"Audio extraction failed: {stderr.decode()}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting audio chunk: {e}")
            return None

    async def _monitor_segments(self) -> AsyncIterator[StreamSegment]:
        """Monitor output directory for new segments using CSV file."""
        processed_segments = set()
        csv_path = self.video_dir / "segments.csv"
        last_csv_size = 0

        while self._running and self._process and self._process.returncode is None:
            # Check if CSV file exists and has been updated
            try:
                if csv_path.exists():
                    current_csv_size = csv_path.stat().st_size

                    # Only read CSV if it has changed
                    if current_csv_size > last_csv_size:
                        # Read new segment entries from CSV
                        segments_info = await self._read_segment_csv(csv_path)

                        for segment_info in segments_info:
                            filename = segment_info["filename"]
                            if filename in processed_segments:
                                continue

                            segment_file = self.video_dir / filename

                            # Wait for segment file to be complete
                            if (
                                segment_file.exists()
                                and await self._is_segment_complete(segment_file)
                            ):
                                segment = await self._create_segment_from_csv(
                                    segment_info, segment_file
                                )
                                
                                # Extract audio chunks from this video segment
                                segment.audio_chunks = await self._extract_audio_chunks(segment)
                                
                                # Add to ring buffer and clean old segments
                                self._video_ring_buffer.append(segment)
                                if len(self._video_ring_buffer) > self.config.video_ring_buffer_size:
                                    old_segment = self._video_ring_buffer.pop(0)
                                    try:
                                        old_segment.path.unlink()
                                        logger.debug(f"Deleted old video segment: {old_segment.path}")
                                    except Exception as e:
                                        logger.error(f"Failed to delete video segment: {e}")
                                
                                processed_segments.add(filename)
                                yield segment

                        last_csv_size = current_csv_size

            except Exception as e:
                logger.warning(f"Error reading segment CSV: {e}")

            # Check process health
            if self._process.returncode is not None:
                stderr = await self._process.stderr.read()
                raise RuntimeError(
                    f"FFmpeg exited with code {self._process.returncode}: {stderr.decode()}"
                )

            # Small delay to prevent busy waiting
            await asyncio.sleep(0.5)

    async def _read_segment_csv(self, csv_path: Path) -> list[dict]:
        """Read segment information from CSV file."""
        segments = []

        def read_csv():
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        segments.append(
                            {
                                "filename": row[0],
                                "start_time": float(row[1]),
                                "end_time": float(row[2]),
                            }
                        )
            return segments

        # Run in thread to avoid blocking
        return await asyncio.to_thread(read_csv)

    async def _create_segment_from_csv(
        self, segment_info: dict, segment_file: Path
    ) -> StreamSegment:
        """Create segment object from CSV info and file."""
        # Extract segment number from filename
        segment_num = int(segment_file.stem.split("_")[-1])

        # Use timing from CSV
        start_time = segment_info["start_time"]
        end_time = segment_info["end_time"]
        duration = end_time - start_time

        # Get file stats
        stats = segment_file.stat()

        return StreamSegment(
            segment_id=UUID(int=segment_num),  # Deterministic UUID
            path=segment_file,
            start_time=start_time,
            duration=duration,
            segment_number=segment_num,
            size_bytes=stats.st_size,
            is_complete=True,
        )

    async def _is_segment_complete(self, segment_file: Path) -> bool:
        """Check if a segment file is complete."""
        # Simple check: file size hasn't changed in 2 seconds
        try:
            size1 = segment_file.stat().st_size
            await asyncio.sleep(2)
            size2 = segment_file.stat().st_size
            return size1 == size2 and size1 > 0
        except FileNotFoundError:
            return False

    async def _create_segment(self, segment_file: Path) -> StreamSegment:
        """Create segment object from file."""
        # Extract segment number from filename
        segment_num = int(segment_file.stem.split("_")[-1])

        # Calculate timing
        start_time = segment_num * self.config.segment_duration

        # Get file stats
        stats = segment_file.stat()

        return StreamSegment(
            segment_id=UUID(int=segment_num),  # Deterministic UUID
            path=segment_file,
            start_time=start_time,
            duration=self.config.segment_duration,
            segment_number=segment_num,
            size_bytes=stats.st_size,
            is_complete=True,
        )

    async def _handle_restart(self) -> None:
        """Handle FFmpeg restart with exponential backoff."""
        if self._retry_count >= self.config.reconnect_attempts:
            raise RuntimeError(
                f"Max reconnection attempts ({self.config.reconnect_attempts}) reached"
            )

        # Exponential backoff
        delay = min(self.config.reconnect_delay * (2**self._retry_count), 60)

        logger.info(
            f"Restarting FFmpeg (attempt {self._retry_count + 1}) after {delay}s delay"
        )
        await asyncio.sleep(delay)

        await self.start()
        self._retry_count += 1

    async def _handle_error(self, error: Exception) -> None:
        """Handle processing errors."""
        current_time = time.time()

        # Reset retry count if enough time has passed
        if current_time - self._last_error_time > 300:  # 5 minutes
            self._retry_count = 0

        self._last_error_time = current_time

        # Stop current process
        await self.stop()

        # Restart will be handled in main loop
        logger.info("Will attempt restart in next iteration")

    def cleanup_all_segments(self) -> None:
        """Clean up all segments when stopping the processor."""
        # Clean video segments
        for segment in self._video_ring_buffer:
            try:
                segment.path.unlink()
                logger.debug(f"Deleted video segment: {segment.path}")
            except Exception as e:
                logger.error(f"Failed to delete video segment: {e}")
        
        # Clean audio chunks
        for chunk in self._audio_ring_buffer:
            try:
                chunk.path.unlink()
                logger.debug(f"Deleted audio chunk: {chunk.path}")
            except Exception as e:
                logger.error(f"Failed to delete audio chunk: {e}")
        
        # Clear ring buffers
        self._video_ring_buffer.clear()
        self._audio_ring_buffer.clear()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        try:
            await self.stop()
        finally:
            # Always cleanup segments
            self.cleanup_all_segments()
