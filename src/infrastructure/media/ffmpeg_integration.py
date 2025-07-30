"""FFmpeg integration utilities for video processing and format conversion.

This module provides comprehensive FFmpeg integration for the TL;DR Highlight API,
supporting real-time video processing, format conversion, frame extraction,
and transcoding operations for RTMP streams and other video sources.
"""

import asyncio
import logging
import subprocess
import tempfile
import os
import json
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator, Union
from dataclasses import dataclass
from enum import Enum
import io
import ffmpeg
import av

logger = logging.getLogger(__name__)


class VideoCodec(Enum):
    """Supported video codecs."""
    H264 = "h264"
    H265 = "h265"
    VP8 = "vp8"
    VP9 = "vp9"
    AV1 = "av1"
    MPEG4 = "mpeg4"
    MJPEG = "mjpeg"


class AudioCodec(Enum):
    """Supported audio codecs."""
    AAC = "aac"
    MP3 = "mp3"
    OPUS = "opus"
    VORBIS = "vorbis"
    AC3 = "ac3"
    PCM = "pcm_s16le"


class ContainerFormat(Enum):
    """Supported container formats."""
    MP4 = "mp4"
    FLV = "flv"
    MKV = "mkv"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"
    TS = "ts"
    HLS = "hls"


@dataclass
class VideoInfo:
    """Video stream information."""
    width: int
    height: int
    fps: float
    duration: Optional[float]
    codec: str
    bitrate: Optional[int]
    pixel_format: str
    
    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 1.0


@dataclass
class AudioInfo:
    """Audio stream information."""
    sample_rate: int
    channels: int
    codec: str
    bitrate: Optional[int]
    sample_format: str
    duration: Optional[float]


@dataclass
class MediaInfo:
    """Complete media information."""
    format_name: str
    duration: Optional[float]
    bitrate: Optional[int]
    size: Optional[int]
    video_streams: List[VideoInfo]
    audio_streams: List[AudioInfo]
    metadata: Dict[str, Any]


@dataclass
class TranscodeOptions:
    """Transcoding configuration options."""
    video_codec: Optional[VideoCodec] = None
    audio_codec: Optional[AudioCodec] = None
    video_bitrate: Optional[int] = None
    audio_bitrate: Optional[int] = None
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[float] = None
    quality: Optional[str] = None  # "ultrafast", "fast", "medium", "slow", "veryslow"
    container: Optional[ContainerFormat] = None
    hardware_acceleration: bool = False
    two_pass: bool = False


class FFmpegError(Exception):
    """FFmpeg operation error."""
    pass


class FFmpegProbe:
    """FFmpeg probe utility for analyzing media files and streams."""
    
    @staticmethod
    async def probe_file(file_path: str) -> MediaInfo:
        """Probe a media file and return detailed information."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise FFmpegError(f"FFprobe failed: {stderr.decode()}")
            
            data = json.loads(stdout.decode())
            return FFmpegProbe._parse_probe_data(data)
            
        except Exception as e:
            logger.error(f"Error probing file {file_path}: {e}")
            raise FFmpegError(f"Failed to probe file: {e}")
    
    @staticmethod
    async def probe_stream(stream_url: str, timeout: int = 10) -> MediaInfo:
        """Probe a stream URL and return information."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                '-analyzeduration', '5000000',
                '-probesize', '5000000',
                stream_url
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise FFmpegError("Stream probe timeout")
            
            if process.returncode != 0:
                raise FFmpegError(f"FFprobe failed: {stderr.decode()}")
            
            data = json.loads(stdout.decode())
            return FFmpegProbe._parse_probe_data(data)
            
        except Exception as e:
            logger.error(f"Error probing stream {stream_url}: {e}")
            raise FFmpegError(f"Failed to probe stream: {e}")
    
    @staticmethod
    def _parse_probe_data(data: Dict[str, Any]) -> MediaInfo:
        """Parse FFprobe JSON output into MediaInfo."""
        format_info = data.get('format', {})
        streams = data.get('streams', [])
        
        video_streams = []
        audio_streams = []
        
        for stream in streams:
            if stream.get('codec_type') == 'video':
                video_info = VideoInfo(
                    width=stream.get('width', 0),
                    height=stream.get('height', 0),
                    fps=FFmpegProbe._parse_fps(stream.get('r_frame_rate', '0/1')),
                    duration=float(stream.get('duration', 0)) if stream.get('duration') else None,
                    codec=stream.get('codec_name', 'unknown'),
                    bitrate=int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else None,
                    pixel_format=stream.get('pix_fmt', 'unknown')
                )
                video_streams.append(video_info)
                
            elif stream.get('codec_type') == 'audio':
                audio_info = AudioInfo(
                    sample_rate=int(stream.get('sample_rate', 0)),
                    channels=int(stream.get('channels', 0)),
                    codec=stream.get('codec_name', 'unknown'),
                    bitrate=int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else None,
                    sample_format=stream.get('sample_fmt', 'unknown'),
                    duration=float(stream.get('duration', 0)) if stream.get('duration') else None
                )
                audio_streams.append(audio_info)
        
        return MediaInfo(
            format_name=format_info.get('format_name', 'unknown'),
            duration=float(format_info.get('duration', 0)) if format_info.get('duration') else None,
            bitrate=int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None,
            size=int(format_info.get('size', 0)) if format_info.get('size') else None,
            video_streams=video_streams,
            audio_streams=audio_streams,
            metadata=format_info.get('tags', {})
        )
    
    @staticmethod
    def _parse_fps(fps_string: str) -> float:
        """Parse FPS from FFmpeg format (e.g., '30/1')."""
        try:
            if '/' in fps_string:
                num, den = fps_string.split('/')
                return float(num) / float(den)
            return float(fps_string)
        except (ValueError, ZeroDivisionError):
            return 0.0


class FFmpegProcessor:
    """FFmpeg processor for video operations."""
    
    def __init__(self, hardware_acceleration: bool = False):
        self.hardware_acceleration = hardware_acceleration
        self.hw_accel_method = self._detect_hardware_acceleration()
    
    def _detect_hardware_acceleration(self) -> Optional[str]:
        """Detect available hardware acceleration."""
        if not self.hardware_acceleration:
            return None
        
        # Check for NVIDIA NVENC
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True, text=True, timeout=10
            )
            if 'h264_nvenc' in result.stdout:
                return 'nvenc'
            elif 'h264_videotoolbox' in result.stdout:
                return 'videotoolbox'  # macOS
            elif 'h264_vaapi' in result.stdout:
                return 'vaapi'  # Linux VA-API
        except Exception:
            pass
        
        return None
    
    async def transcode_stream(
        self,
        input_source: str,
        output_path: str,
        options: TranscodeOptions
    ) -> bool:
        """Transcode a stream with specified options."""
        try:
            # Build FFmpeg command
            cmd = self._build_transcode_command(input_source, output_path, options)
            
            logger.info(f"Starting transcode: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise FFmpegError(f"Transcode failed: {error_msg}")
            
            logger.info(f"Transcode completed successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Transcode error: {e}")
            raise FFmpegError(f"Transcode failed: {e}")
    
    def _build_transcode_command(
        self,
        input_source: str,
        output_path: str,
        options: TranscodeOptions
    ) -> List[str]:
        """Build FFmpeg command for transcoding."""
        cmd = ['ffmpeg', '-y']  # -y to overwrite output
        
        # Hardware acceleration input
        if self.hw_accel_method:
            if self.hw_accel_method == 'nvenc':
                cmd.extend(['-hwaccel', 'cuda'])
            elif self.hw_accel_method == 'videotoolbox':
                cmd.extend(['-hwaccel', 'videotoolbox'])
            elif self.hw_accel_method == 'vaapi':
                cmd.extend(['-hwaccel', 'vaapi'])
        
        # Input source
        cmd.extend(['-i', input_source])
        
        # Video codec
        if options.video_codec:
            if self.hw_accel_method == 'nvenc' and options.video_codec == VideoCodec.H264:
                cmd.extend(['-c:v', 'h264_nvenc'])
            elif self.hw_accel_method == 'videotoolbox' and options.video_codec == VideoCodec.H264:
                cmd.extend(['-c:v', 'h264_videotoolbox'])
            else:
                cmd.extend(['-c:v', options.video_codec.value])
        
        # Audio codec
        if options.audio_codec:
            cmd.extend(['-c:a', options.audio_codec.value])
        
        # Video bitrate
        if options.video_bitrate:
            cmd.extend(['-b:v', f'{options.video_bitrate}k'])
        
        # Audio bitrate
        if options.audio_bitrate:
            cmd.extend(['-b:a', f'{options.audio_bitrate}k'])
        
        # Resolution
        if options.resolution:
            width, height = options.resolution
            cmd.extend(['-s', f'{width}x{height}'])
        
        # Frame rate
        if options.fps:
            cmd.extend(['-r', str(options.fps)])
        
        # Quality preset
        if options.quality:
            cmd.extend(['-preset', options.quality])
        
        # Container-specific options
        if options.container == ContainerFormat.HLS:
            cmd.extend([
                '-f', 'hls',
                '-hls_time', '4',
                '-hls_list_size', '0',
                '-hls_segment_filename', f'{output_path}_%03d.ts'
            ])
        
        cmd.append(output_path)
        return cmd
    
    async def extract_frames(
        self,
        input_source: str,
        output_pattern: str,
        interval: float = 1.0,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """Extract frames from video at specified interval."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', input_source,
                '-vf', f'fps=1/{interval}',
                '-q:v', '2'  # High quality
            ]
            
            if max_frames:
                cmd.extend(['-frames:v', str(max_frames)])
            
            cmd.append(output_pattern)
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise FFmpegError(f"Frame extraction failed: {error_msg}")
            
            # Find generated frame files
            import glob
            frame_files = glob.glob(output_pattern.replace('%d', '*'))
            frame_files.sort()
            
            logger.info(f"Extracted {len(frame_files)} frames")
            return frame_files
            
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            raise FFmpegError(f"Frame extraction failed: {e}")
    
    async def create_thumbnail(
        self,
        input_source: str,
        output_path: str,
        timestamp: float = 0.0,
        width: int = 320,
        height: int = 180
    ) -> bool:
        """Create thumbnail from video at specified timestamp."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(timestamp),
                '-i', input_source,
                '-vframes', '1',
                '-vf', f'scale={width}:{height}',
                '-q:v', '2',
                output_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise FFmpegError(f"Thumbnail creation failed: {error_msg}")
            
            logger.info(f"Created thumbnail: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Thumbnail creation error: {e}")
            raise FFmpegError(f"Thumbnail creation failed: {e}")


class FFmpegStreamProcessor:
    """Real-time FFmpeg stream processor for live content."""
    
    def __init__(self, buffer_size: int = 65536):
        self.buffer_size = buffer_size
        self.process: Optional[asyncio.subprocess.Process] = None
        self.output_queue: asyncio.Queue = asyncio.Queue()
    
    async def start_stream_processing(
        self,
        input_source: str,
        output_format: str = 'rawvideo',
        video_codec: str = 'rawvideo',
        pixel_format: str = 'rgb24'
    ) -> AsyncGenerator[bytes, None]:
        """Start processing stream and yield raw video data."""
        try:
            cmd = [
                'ffmpeg',
                '-i', input_source,
                '-f', output_format,
                '-vcodec', video_codec,
                '-pix_fmt', pixel_format,
                '-'
            ]
            
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            logger.info(f"Started FFmpeg stream processing: {' '.join(cmd)}")
            
            while True:
                data = await self.process.stdout.read(self.buffer_size)
                if not data:
                    break
                yield data
            
            await self.process.wait()
            
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            raise FFmpegError(f"Stream processing failed: {e}")
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop stream processing."""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.error(f"Error stopping FFmpeg process: {e}")
            finally:
                self.process = None


class PyAVProcessor:
    """PyAV-based video processor for more direct control."""
    
    def __init__(self):
        self.container: Optional[av.InputContainer] = None
    
    async def open_stream(self, stream_url: str, timeout: int = 10) -> bool:
        """Open stream with PyAV."""
        try:
            # PyAV is synchronous, so we run it in an executor
            loop = asyncio.get_event_loop()
            
            def _open_container():
                options = {'rtmp_live': 'live', 'analyzeduration': '5000000'}
                return av.open(stream_url, timeout=timeout, options=options)
            
            self.container = await loop.run_in_executor(None, _open_container)
            
            logger.info(f"Opened stream with PyAV: {stream_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening stream with PyAV: {e}")
            return False
    
    async def read_frames(self) -> AsyncGenerator[Tuple[av.VideoFrame, float], None]:
        """Read video frames from the stream."""
        if not self.container:
            raise ValueError("No container opened")
        
        try:
            loop = asyncio.get_event_loop()
            
            video_stream = None
            for stream in self.container.streams.video:
                video_stream = stream
                break
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            def _read_frame():
                try:
                    for packet in self.container.demux(video_stream):
                        for frame in packet.decode():
                            timestamp = float(frame.pts * frame.time_base)
                            return frame, timestamp
                except av.EOFError:
                    return None, None
                except Exception as e:
                    logger.error(f"Frame reading error: {e}")
                    return None, None
                return None, None
            
            while True:
                frame, timestamp = await loop.run_in_executor(None, _read_frame)
                if frame is None:
                    break
                yield frame, timestamp
                
        except Exception as e:
            logger.error(f"Error reading frames: {e}")
            raise
    
    async def extract_frame_data(self, frame: av.VideoFrame) -> bytes:
        """Extract raw frame data as bytes."""
        try:
            loop = asyncio.get_event_loop()
            
            def _convert_frame():
                # Convert to RGB format
                rgb_frame = frame.reformat(format='rgb24')
                return rgb_frame.to_ndarray().tobytes()
            
            return await loop.run_in_executor(None, _convert_frame)
            
        except Exception as e:
            logger.error(f"Error extracting frame data: {e}")
            raise
    
    def close(self) -> None:
        """Close the container."""
        if self.container:
            self.container.close()
            self.container = None


class VideoFrameExtractor:
    """High-level video frame extractor for highlight analysis."""
    
    def __init__(self, use_hardware_acceleration: bool = False):
        self.ffmpeg_processor = FFmpegProcessor(use_hardware_acceleration)
        self.pyav_processor = PyAVProcessor()
    
    async def extract_keyframes(
        self,
        stream_url: str,
        output_dir: str,
        duration: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """Extract keyframes from stream for analysis.
        
        Returns:
            List of (frame_path, timestamp) tuples
        """
        try:
            # Create temporary directory for frames
            os.makedirs(output_dir, exist_ok=True)
            
            frame_pattern = os.path.join(output_dir, 'keyframe_%05d.jpg')
            
            cmd = [
                'ffmpeg', '-y',
                '-i', stream_url,
                '-vf', 'select=eq(pict_type\\,I)',  # Extract only I-frames (keyframes)
                '-vsync', 'vfr',  # Variable frame rate
                '-q:v', '2'  # High quality
            ]
            
            if duration:
                cmd.extend(['-t', str(duration)])
            
            cmd.append(frame_pattern)
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise FFmpegError(f"Keyframe extraction failed: {error_msg}")
            
            # Find generated frames and estimate timestamps
            import glob
            frame_files = glob.glob(frame_pattern.replace('%05d', '*'))
            frame_files.sort()
            
            # Estimate timestamps (simplified approach)
            frames_with_timestamps = []
            for i, frame_path in enumerate(frame_files):
                # This is a rough estimate - for precise timestamps,
                # you'd need to parse FFmpeg output or use different approach
                timestamp = i * 2.0  # Assume keyframes every 2 seconds
                frames_with_timestamps.append((frame_path, timestamp))
            
            logger.info(f"Extracted {len(frames_with_timestamps)} keyframes")
            return frames_with_timestamps
            
        except Exception as e:
            logger.error(f"Keyframe extraction error: {e}")
            raise FFmpegError(f"Keyframe extraction failed: {e}")
    
    async def stream_frames_realtime(
        self,
        stream_url: str,
        frame_callback: callable,
        max_duration: Optional[float] = None
    ) -> None:
        """Stream frames in real-time with callback processing."""
        try:
            if await self.pyav_processor.open_stream(stream_url):
                start_time = asyncio.get_event_loop().time()
                
                async for frame, timestamp in self.pyav_processor.read_frames():
                    # Check duration limit
                    if max_duration and (asyncio.get_event_loop().time() - start_time) > max_duration:
                        break
                    
                    # Extract frame data
                    frame_data = await self.pyav_processor.extract_frame_data(frame)
                    
                    # Call the callback with frame info
                    await frame_callback({
                        'data': frame_data,
                        'timestamp': timestamp,
                        'width': frame.width,
                        'height': frame.height,
                        'format': 'rgb24'
                    })
                
                self.pyav_processor.close()
            else:
                raise FFmpegError("Failed to open stream with PyAV")
                
        except Exception as e:
            logger.error(f"Real-time frame streaming error: {e}")
            raise FFmpegError(f"Real-time frame streaming failed: {e}")


def get_ffmpeg_version() -> str:
    """Get FFmpeg version information."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.split('\\n')
            for line in lines:
                if line.startswith('ffmpeg version'):
                    return line.split(' ')[2]
        return "Unknown"
    except Exception:
        return "Not available"


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system."""
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True, timeout=10, check=True
        )
        return True
    except Exception:
        return False