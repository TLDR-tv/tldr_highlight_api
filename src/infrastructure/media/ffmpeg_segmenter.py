"""FFmpeg-based stream segmenter using the segment muxer.

This module provides a streamlined approach to stream segmentation using
FFmpeg's built-in segment muxer, replacing the complex in-memory frame
buffering approach with efficient file-based segmentation.
"""

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Any, Callable
import aiofiles
import json

logger = logging.getLogger(__name__)


class SegmentFormat(str, Enum):
    """Supported segment output formats."""
    
    MPEGTS = "mpegts"  # .ts files (most compatible)
    MP4 = "mp4"        # Fragmented MP4
    WEBM = "webm"      # WebM format
    MKV = "matroska"   # Matroska format


@dataclass
class SegmentConfig:
    """Configuration for FFmpeg segmentation."""
    
    # Basic settings
    segment_duration: int = 30  # seconds
    segment_format: SegmentFormat = SegmentFormat.MPEGTS
    
    # Keyframe alignment
    force_keyframe_every: int = 2  # Force keyframe every N seconds
    min_seg_duration: float = 0.5  # Minimum segment duration
    
    # File management
    segment_wrap: Optional[int] = None  # Number of segments to keep (None = keep all)
    delete_threshold: Optional[int] = None  # Delete segments older than N segments
    
    # Quality settings
    video_codec: str = "copy"  # "copy" to avoid re-encoding, or specific codec
    audio_codec: str = "copy"
    video_bitrate: Optional[str] = None  # e.g., "2M"
    audio_bitrate: Optional[str] = None  # e.g., "128k"
    
    # Performance
    preset: str = "veryfast"  # FFmpeg preset (ultrafast, veryfast, fast, medium, slow)
    threads: int = 0  # 0 = auto
    
    # Reconnection
    reconnect: bool = True
    reconnect_delay_max: int = 4
    timeout: int = 30000000  # microseconds
    
    # Metadata
    write_metadata: bool = True
    metadata_file: bool = True  # Write segment metadata to JSON


@dataclass
class SegmentInfo:
    """Information about a created segment."""
    
    index: int
    filename: str
    path: Path
    start_time: float
    end_time: float
    duration: float
    size: int
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def exists(self) -> bool:
        """Check if segment file exists."""
        return self.path.exists()


class FFmpegSegmenter:
    """Stream segmenter using FFmpeg's segment muxer.
    
    This class provides a clean interface for segmenting live streams
    using FFmpeg's built-in segmentation capabilities, with support for
    various formats, keyframe alignment, and metadata tracking.
    """
    
    def __init__(
        self,
        output_dir: Path,
        config: Optional[SegmentConfig] = None,
        segment_callback: Optional[Callable[[SegmentInfo], None]] = None
    ):
        """Initialize the segmenter.
        
        Args:
            output_dir: Directory to write segments
            config: Segmentation configuration
            segment_callback: Callback for completed segments
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or SegmentConfig()
        self.segment_callback = segment_callback
        
        # Process management
        self._process: Optional[asyncio.subprocess.Process] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._segment_queue: asyncio.Queue = asyncio.Queue()
        
        # Segment tracking
        self._segments: List[SegmentInfo] = []
        self._current_index = 0
        self._start_time: Optional[float] = None
        
        # File patterns
        self._segment_pattern = "segment_%05d"
        self._extension = self._get_extension()
        
        logger.info(f"Initialized FFmpegSegmenter: output_dir={output_dir}")
    
    def _get_extension(self) -> str:
        """Get file extension for segment format."""
        extensions = {
            SegmentFormat.MPEGTS: "ts",
            SegmentFormat.MP4: "mp4",
            SegmentFormat.WEBM: "webm",
            SegmentFormat.MKV: "mkv"
        }
        return extensions.get(self.config.segment_format, "ts")
    
    def _build_ffmpeg_command(self, input_url: str) -> List[str]:
        """Build FFmpeg command for segmentation."""
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info"]
        
        # Reconnection options for live streams
        if self.config.reconnect:
            cmd.extend([
                "-reconnect", "1",
                "-reconnect_at_eof", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", str(self.config.reconnect_delay_max),
                "-timeout", str(self.config.timeout)
            ])
        
        # Input
        cmd.extend(["-i", input_url])
        
        # Video codec
        if self.config.video_codec == "copy":
            cmd.extend(["-c:v", "copy"])
        else:
            cmd.extend([
                "-c:v", self.config.video_codec,
                "-preset", self.config.preset
            ])
            if self.config.video_bitrate:
                cmd.extend(["-b:v", self.config.video_bitrate])
        
        # Audio codec
        if self.config.audio_codec == "copy":
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-c:a", self.config.audio_codec])
            if self.config.audio_bitrate:
                cmd.extend(["-b:a", self.config.audio_bitrate])
        
        # Force keyframes for better segmentation
        if self.config.force_keyframe_every and self.config.video_codec != "copy":
            cmd.extend([
                "-g", str(self.config.force_keyframe_every * 25),  # Assuming 25fps
                "-keyint_min", str(self.config.force_keyframe_every * 25),
                "-force_key_frames", f"expr:gte(t,n_forced*{self.config.force_keyframe_every})"
            ])
        
        # Segment muxer settings
        cmd.extend([
            "-f", "segment",
            "-segment_time", str(self.config.segment_duration),
            "-segment_format", self.config.segment_format.value,
            "-segment_start_number", "0",
            "-reset_timestamps", "1",
            "-avoid_negative_ts", "make_zero"
        ])
        
        # Segment list for tracking
        segment_list_path = self.output_dir / "segments.txt"
        cmd.extend([
            "-segment_list", str(segment_list_path),
            "-segment_list_flags", "+live"
        ])
        
        # Segment wrap (circular buffer)
        if self.config.segment_wrap:
            cmd.extend(["-segment_wrap", str(self.config.segment_wrap)])
        
        # Threading
        if self.config.threads:
            cmd.extend(["-threads", str(self.config.threads)])
        
        # Output pattern
        output_pattern = self.output_dir / f"{self._segment_pattern}.{self._extension}"
        cmd.append(str(output_pattern))
        
        return cmd
    
    async def start(self, input_url: str) -> None:
        """Start segmenting the input stream.
        
        Args:
            input_url: URL of the input stream (RTMP, HLS, HTTP, etc.)
        """
        if self._process:
            raise RuntimeError("Segmenter already running")
        
        self._start_time = datetime.now().timestamp()
        
        # Clean up any existing segments
        await self._cleanup_output_dir()
        
        # Build and start FFmpeg process
        cmd = self._build_ffmpeg_command(input_url)
        logger.info(f"Starting FFmpeg segmentation: {' '.join(cmd)}")
        
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Start monitoring
        self._monitor_task = asyncio.create_task(self._monitor_segments())
        
        # Start stderr reader for FFmpeg logs
        asyncio.create_task(self._read_ffmpeg_logs())
    
    async def stop(self) -> None:
        """Stop the segmentation process."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("FFmpeg didn't terminate gracefully, killing...")
                self._process.kill()
                await self._process.wait()
            
            self._process = None
        
        logger.info("FFmpeg segmenter stopped")
    
    async def get_segments(self) -> AsyncIterator[SegmentInfo]:
        """Yield segments as they become available."""
        while True:
            try:
                segment = await self._segment_queue.get()
                if segment is None:  # Sentinel for shutdown
                    break
                yield segment
            except asyncio.CancelledError:
                break
    
    async def _monitor_segments(self) -> None:
        """Monitor for new segments and process them."""
        segment_list_path = self.output_dir / "segments.txt"
        processed_segments = set()
        
        while True:
            try:
                # Wait for segment list to be updated
                await asyncio.sleep(0.5)
                
                if not segment_list_path.exists():
                    continue
                
                # Read segment list
                async with aiofiles.open(segment_list_path, 'r') as f:
                    lines = await f.readlines()
                
                for line in lines:
                    segment_file = line.strip()
                    if not segment_file or segment_file in processed_segments:
                        continue
                    
                    # Process new segment
                    segment_path = self.output_dir / segment_file
                    if segment_path.exists():
                        segment_info = await self._process_segment(segment_path)
                        if segment_info:
                            self._segments.append(segment_info)
                            await self._segment_queue.put(segment_info)
                            
                            # Callback
                            if self.segment_callback:
                                try:
                                    if asyncio.iscoroutinefunction(self.segment_callback):
                                        await self.segment_callback(segment_info)
                                    else:
                                        self.segment_callback(segment_info)
                                except Exception as e:
                                    logger.error(f"Segment callback error: {e}")
                            
                            processed_segments.add(segment_file)
                            
                            # Cleanup old segments if configured
                            await self._cleanup_old_segments()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring segments: {e}")
                await asyncio.sleep(1)
    
    async def _process_segment(self, segment_path: Path) -> Optional[SegmentInfo]:
        """Process a completed segment file."""
        try:
            # Get file stats
            stat = segment_path.stat()
            
            # Extract index from filename
            filename = segment_path.name
            index = int(filename.split('_')[1].split('.')[0])
            
            # Calculate timing (approximate)
            start_time = index * self.config.segment_duration
            end_time = (index + 1) * self.config.segment_duration
            
            segment_info = SegmentInfo(
                index=index,
                filename=filename,
                path=segment_path,
                start_time=start_time,
                end_time=end_time,
                duration=self.config.segment_duration,
                size=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime)
            )
            
            # Write metadata if configured
            if self.config.metadata_file:
                await self._write_segment_metadata(segment_info)
            
            logger.debug(f"Processed segment: {filename} (size: {stat.st_size} bytes)")
            return segment_info
            
        except Exception as e:
            logger.error(f"Error processing segment {segment_path}: {e}")
            return None
    
    async def _write_segment_metadata(self, segment: SegmentInfo) -> None:
        """Write segment metadata to JSON file."""
        metadata_path = segment.path.with_suffix('.json')
        
        metadata = {
            "index": segment.index,
            "filename": segment.filename,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "duration": segment.duration,
            "size": segment.size,
            "created_at": segment.created_at.isoformat(),
            "stream_start_time": self._start_time
        }
        
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
    
    async def _cleanup_old_segments(self) -> None:
        """Clean up old segments based on configuration."""
        if not self.config.delete_threshold:
            return
        
        if len(self._segments) > self.config.delete_threshold:
            # Remove oldest segments
            segments_to_remove = len(self._segments) - self.config.delete_threshold
            
            for i in range(segments_to_remove):
                segment = self._segments[i]
                try:
                    # Delete segment file
                    if segment.path.exists():
                        segment.path.unlink()
                    
                    # Delete metadata file
                    metadata_path = segment.path.with_suffix('.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    logger.debug(f"Deleted old segment: {segment.filename}")
                except Exception as e:
                    logger.error(f"Error deleting segment {segment.filename}: {e}")
            
            # Update segment list
            self._segments = self._segments[segments_to_remove:]
    
    async def _read_ffmpeg_logs(self) -> None:
        """Read and log FFmpeg stderr output."""
        if not self._process or not self._process.stderr:
            return
        
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                
                log_line = line.decode().strip()
                if log_line:
                    # Parse FFmpeg output for important events
                    if "error" in log_line.lower():
                        logger.error(f"FFmpeg: {log_line}")
                    elif "warning" in log_line.lower():
                        logger.warning(f"FFmpeg: {log_line}")
                    else:
                        logger.debug(f"FFmpeg: {log_line}")
        except Exception as e:
            logger.error(f"Error reading FFmpeg logs: {e}")
    
    async def _cleanup_output_dir(self) -> None:
        """Clean up output directory."""
        try:
            # Remove only segment files, not the directory
            for file in self.output_dir.glob(f"*.{self._extension}"):
                file.unlink()
            for file in self.output_dir.glob("*.json"):
                file.unlink()
            
            # Remove segment list
            segment_list = self.output_dir / "segments.txt"
            if segment_list.exists():
                segment_list.unlink()
                
        except Exception as e:
            logger.error(f"Error cleaning output directory: {e}")
    
    @property
    def is_running(self) -> bool:
        """Check if segmenter is running."""
        return self._process is not None and self._process.returncode is None
    
    def get_segment_info(self, index: int) -> Optional[SegmentInfo]:
        """Get information about a specific segment by index."""
        for segment in self._segments:
            if segment.index == index:
                return segment
        return None
    
    def get_latest_segments(self, count: int = 10) -> List[SegmentInfo]:
        """Get the latest N segments."""
        return self._segments[-count:] if self._segments else []