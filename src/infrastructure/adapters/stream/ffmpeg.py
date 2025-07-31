"""Generic FFmpeg stream adapter.

This adapter can handle any stream format that FFmpeg supports,
making it a universal solution for stream ingestion.
"""

from typing import Optional, AsyncIterator
import asyncio
from datetime import datetime

from .base import (
    StreamAdapter,
    StreamConnection,
    StreamMetadata,
    StreamHealth,
    ConnectionStatus,
    ConnectionError,
    StreamNotFoundError,
)


class FFmpegStreamAdapter(StreamAdapter):
    """Generic stream adapter using FFmpeg for any supported format.
    
    This adapter can handle:
    - RTMP/RTMPS streams
    - HLS (m3u8) streams
    - DASH streams
    - Direct HTTP/HTTPS streams
    - Local files
    - UDP/RTP/RTSP streams
    - SRT streams
    - Any other format FFmpeg supports
    """

    def __init__(self, stream_url: str, **kwargs):
        """Initialize FFmpeg stream adapter.

        Args:
            stream_url: URL of the stream (any FFmpeg-supported format)
            **kwargs: Additional configuration options
        """
        super().__init__(stream_url, **kwargs)
        self._connection: Optional[StreamConnection] = None
        self._ffmpeg_process: Optional[asyncio.subprocess.Process] = None

    async def connect(self) -> StreamConnection:
        """Connect to the stream using FFmpeg probe.

        Returns:
            StreamConnection with stream information

        Raises:
            ConnectionError: If connection fails
            StreamNotFoundError: If stream URL is invalid
        """
        try:
            # Use FFmpeg to probe the stream
            from src.infrastructure.media.ffmpeg_integration import FFmpegProbe
            
            # Probe with a reasonable timeout
            probe_info = await FFmpegProbe.probe_stream(self.stream_url, timeout=15)
            
            # Extract metadata from probe
            metadata = StreamMetadata(
                width=probe_info.video_streams[0].width if probe_info.video_streams else None,
                height=probe_info.video_streams[0].height if probe_info.video_streams else None,
                fps=probe_info.video_streams[0].fps if probe_info.video_streams else None,
                video_codec=probe_info.video_streams[0].codec if probe_info.video_streams else None,
                audio_codec=probe_info.audio_streams[0].codec if probe_info.audio_streams else None,
                audio_sample_rate=probe_info.audio_streams[0].sample_rate if probe_info.audio_streams else None,
                audio_channels=probe_info.audio_streams[0].channels if probe_info.audio_streams else None,
                format_name=probe_info.format_name,
                duration_seconds=probe_info.duration,
                bitrate=probe_info.bitrate,
                stream_type="live" if probe_info.is_live else "vod",
            )

            # Create connection object
            self._connection = StreamConnection(
                adapter_type="ffmpeg",
                stream_url=self.stream_url,
                connected_at=datetime.utcnow(),
                status=ConnectionStatus.CONNECTED,
                metadata=metadata,
            )

            return self._connection

        except asyncio.TimeoutError:
            raise ConnectionError(f"Timeout connecting to stream: {self.stream_url}")
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise StreamNotFoundError(f"Stream not found: {self.stream_url}")
            raise ConnectionError(f"Failed to connect to stream: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from the stream."""
        if self._ffmpeg_process and self._ffmpeg_process.returncode is None:
            self._ffmpeg_process.terminate()
            try:
                await asyncio.wait_for(self._ffmpeg_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._ffmpeg_process.kill()
                await self._ffmpeg_process.wait()

        if self._connection:
            self._connection.status = ConnectionStatus.DISCONNECTED
            self._connection.disconnected_at = datetime.utcnow()

    async def get_stream_data(self) -> AsyncIterator[bytes]:
        """Get raw stream data.

        This is typically not used directly as FFmpeg handles
        the data extraction in the processing pipeline.

        Yields:
            Raw stream data chunks
        """
        if not self._connection or self._connection.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("Not connected to stream")

        # For the generic adapter, we don't yield data directly
        # The FFmpeg processor in the pipeline handles data extraction
        yield b""

    async def get_stream_health(self) -> StreamHealth:
        """Get current stream health metrics.

        Returns:
            Stream health information
        """
        if not self._connection:
            return StreamHealth(
                status=ConnectionStatus.DISCONNECTED,
                latency_ms=0,
                dropped_frames=0,
                connection_quality=0.0,
                bandwidth_mbps=0.0,
            )

        # Basic health check - verify stream is still accessible
        try:
            from src.infrastructure.media.ffmpeg_integration import FFmpegProbe
            
            # Quick probe to check if stream is still alive
            await FFmpegProbe.probe_stream(self.stream_url, timeout=5)
            
            return StreamHealth(
                status=self._connection.status,
                latency_ms=50,  # Estimated
                dropped_frames=0,
                connection_quality=1.0,
                bandwidth_mbps=self._connection.metadata.bitrate / 1_000_000 if self._connection.metadata.bitrate else 0.0,
            )
        except Exception:
            return StreamHealth(
                status=ConnectionStatus.ERROR,
                latency_ms=0,
                dropped_frames=0,
                connection_quality=0.0,
                bandwidth_mbps=0.0,
            )

    async def get_metadata(self) -> StreamMetadata:
        """Get current stream metadata.

        Returns:
            Stream metadata

        Raises:
            ConnectionError: If not connected
        """
        if not self._connection:
            raise ConnectionError("Not connected to stream")

        return self._connection.metadata

    async def is_live(self) -> bool:
        """Check if the stream is currently live.

        Returns:
            True if stream is live, False otherwise
        """
        try:
            health = await self.get_stream_health()
            return health.status == ConnectionStatus.CONNECTED
        except Exception:
            return False

    async def __aenter__(self) -> "FFmpegStreamAdapter":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()