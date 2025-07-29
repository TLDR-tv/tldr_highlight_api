"""Enhanced RTMP stream adapter with complete protocol implementation.

This module provides a comprehensive RTMP adapter with full protocol support,
FLV parsing, and FFmpeg integration for real video data access in the
TL;DR Highlight API.
"""

import asyncio
import logging
import socket
import tempfile
import os
import json
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional, Any, Tuple, List

from aiohttp import ClientSession

from src.core.config import get_settings
from src.utils.stream_validation import validate_stream_url
from src.infrastructure.persistence.models.stream import StreamPlatform
from src.utils.rtmp_protocol import (
    RTMPProtocol,
    RTMPMessage,
    RTMPMessageType,
    RTMPHandshake,
    AMFDecoder,
)
from src.utils.flv_parser import (
    FLVStreamProcessor,
    FLVTag,
    FLVTagType,
    FLVVideoTag,
    FLVAudioTag,
    FLVVideoCodec,
    FLVVideoFrameType,
)
from src.utils.ffmpeg_integration import (
    FFmpegProcessor,
    FFmpegProbe,
    VideoFrameExtractor,
    TranscodeOptions,
    VideoCodec,
    AudioCodec,
    check_ffmpeg_available,
)
from .base import (
    BaseStreamAdapter,
    StreamMetadata,
    ConnectionStatus,
    StreamHealth,
    ConnectionError,
    StreamAdapterError,
)


logger = logging.getLogger(__name__)
settings = get_settings()


class RTMPConnectionError(ConnectionError):
    """RTMP-specific connection error."""
    pass


class RTMPProtocolError(StreamAdapterError):
    """RTMP protocol-specific error."""
    pass


class EnhancedRTMPAdapter(BaseStreamAdapter):
    """Enhanced RTMP stream adapter with complete protocol implementation.

    This adapter provides:
    - Full RTMP protocol implementation (handshake, messaging, chunking)
    - FLV format parsing for video/audio extraction
    - FFmpeg integration for robust video processing
    - Real-time stream data access for highlight analysis
    - Support for both RTMP and RTMPS protocols
    """

    def __init__(
        self,
        url: str,
        buffer_size: Optional[int] = None,
        session: Optional[ClientSession] = None,
        **kwargs,
    ):
        """Initialize the enhanced RTMP adapter.

        Args:
            url: RTMP stream URL
            buffer_size: Buffer size for RTMP data (optional)
            session: Optional aiohttp ClientSession (not used for RTMP)
            **kwargs: Additional configuration options:
                - app_name: RTMP application name
                - stream_key: RTMP stream key
                - playpath: RTMP playpath
                - hardware_acceleration: Enable hardware acceleration
                - enable_recording: Enable stream recording
                - output_directory: Directory for recorded files
                - transcode_options: Transcoding configuration
        """
        super().__init__(url, session, **kwargs)

        # Validate URL and extract information
        self.validation_result = validate_stream_url(url, StreamPlatform.RTMP)
        self.hostname = self.validation_result["hostname"]
        self.port = self.validation_result["port"]
        self.path = self.validation_result["path"]
        self.scheme = self.validation_result["scheme"]

        # RTMP configuration
        self.buffer_size = buffer_size or settings.rtmp_buffer_size
        self.connection_timeout = settings.rtmp_connection_timeout
        self.read_timeout = settings.rtmp_read_timeout

        # Connection state
        self._socket: Optional[socket.socket] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._is_secure = self.scheme == "rtmps"

        # RTMP protocol handler
        self._rtmp_protocol = RTMPProtocol()
        
        # FLV stream processor
        self._flv_processor = FLVStreamProcessor(buffer_size=self.buffer_size)
        
        # FFmpeg integration
        self._ffmpeg_processor = FFmpegProcessor(hardware_acceleration=kwargs.get('hardware_acceleration', False))
        self._frame_extractor = VideoFrameExtractor(use_hardware_acceleration=kwargs.get('hardware_acceleration', False))
        
        # Stream data and state
        self._stream_info: Dict[str, Any] = {}
        self._bytes_buffer = bytearray()
        self._app_name = kwargs.get('app_name', 'live')
        self._stream_key = kwargs.get('stream_key', '')
        self._playpath = kwargs.get('playpath', '')
        
        # Video processing state
        self._video_frames: List[Tuple[bytes, float, bool]] = []  # (data, timestamp, is_keyframe)
        self._audio_samples: List[Tuple[bytes, float]] = []  # (data, timestamp)
        self._last_frame_time = 0.0
        self._frame_count = 0
        
        # Output options
        self._enable_recording = kwargs.get('enable_recording', False)
        self._output_directory = kwargs.get('output_directory', tempfile.gettempdir())
        self._transcode_options = kwargs.get('transcode_options', TranscodeOptions())
        
        # Background tasks
        self._message_processing_task: Optional[asyncio.Task] = None
        self._frame_extraction_task: Optional[asyncio.Task] = None
        
        # Check FFmpeg availability
        self._ffmpeg_available = check_ffmpeg_available()
        if not self._ffmpeg_available:
            logger.warning("FFmpeg not available - video processing will be limited")

        logger.info(
            f"Initialized Enhanced RTMP adapter for {self.hostname}:{self.port}{self.path}"
        )

    async def authenticate(self) -> bool:
        """Authenticate with RTMP server.

        For basic RTMP streams, no authentication is typically required.
        This method can be overridden for streams that require authentication.

        Returns:
            bool: True (RTMP typically doesn't require separate authentication)
        """
        logger.info("RTMP authentication not required for basic streams")
        return True

    async def _create_connection(
        self,
    ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Create an RTMP connection.

        Returns:
            Tuple of StreamReader and StreamWriter

        Raises:
            RTMPConnectionError: If connection fails
        """
        try:
            if self._is_secure:
                # For RTMPS, we would need SSL context
                import ssl

                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.hostname, self.port, ssl=ssl_context),
                    timeout=self.connection_timeout,
                )
            else:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.hostname, self.port),
                    timeout=self.connection_timeout,
                )

            return reader, writer

        except asyncio.TimeoutError:
            raise RTMPConnectionError(
                f"RTMP connection timeout to {self.hostname}:{self.port}"
            )
        except OSError as e:
            raise RTMPConnectionError(
                f"RTMP connection failed to {self.hostname}:{self.port}: {e}"
            )

    async def _perform_rtmp_handshake(self) -> bool:
        """Perform complete RTMP handshake using protocol implementation.

        Returns:
            bool: True if handshake successful

        Raises:
            RTMPProtocolError: If handshake fails
        """
        if not self._reader or not self._writer:
            raise RTMPProtocolError("No RTMP connection available for handshake")

        try:
            success = await self._rtmp_protocol.perform_handshake(self._reader, self._writer)
            if success:
                logger.debug("RTMP handshake completed successfully")
            return success
        except Exception as e:
            raise RTMPProtocolError(f"RTMP handshake failed: {e}")

    async def _send_connect_request(self) -> bool:
        """Send RTMP connect command and handle response.

        Returns:
            bool: True if connect request successful

        Raises:
            RTMPProtocolError: If connect request fails
        """
        if not self._writer:
            raise RTMPProtocolError("No RTMP connection available for connect request")

        try:
            # Determine app name from path
            app_name = self._app_name
            if self.path.startswith('/'):
                path_parts = self.path[1:].split('/')
                if path_parts:
                    app_name = path_parts[0]
                    if len(path_parts) > 1:
                        self._stream_key = '/'.join(path_parts[1:])
            
            # Build connection parameters
            tcUrl = f"{self.scheme}://{self.hostname}:{self.port}/{app_name}"
            connection_params = {
                "app": app_name,
                "tcUrl": tcUrl,
                "fpad": False,
                "capabilities": 15.0,
                "audioCodecs": 3575.0,
                "videoCodecs": 252.0,
                "videoFunction": 1.0,
                "objectEncoding": 0.0
            }
            
            # Add stream key if present
            if self._stream_key:
                connection_params["playpath"] = self._stream_key
            
            # Send connect command
            await self._rtmp_protocol.send_connect_command(
                self._writer, app_name, connection_params
            )
            
            # Wait for connect response
            response_received = False
            timeout_time = asyncio.get_event_loop().time() + self.read_timeout
            
            while not response_received and asyncio.get_event_loop().time() < timeout_time:
                message = await self._rtmp_protocol.parse_message(self._reader)
                if message:
                    if message.message_type == RTMPMessageType.AMF0_COMMAND:
                        result = self._rtmp_protocol.process_command_message(message)
                        if result:
                            if result.get('command') == '_result':
                                logger.info("RTMP connect successful")
                                response_received = True
                                break
                            elif result.get('command') == '_error':
                                error_info = result.get('error_object', {})
                                raise RTMPProtocolError(f"RTMP connect failed: {error_info}")
                    
                    elif message.message_type == RTMPMessageType.WINDOW_ACKNOWLEDGEMENT_SIZE:
                        # Handle window ack size
                        if len(message.data) >= 4:
                            import struct
                            window_size = struct.unpack('>I', message.data[0:4])[0]
                            logger.debug(f"Received window ack size: {window_size}")
                    
                    elif message.message_type == RTMPMessageType.SET_PEER_BANDWIDTH:
                        # Handle peer bandwidth
                        if len(message.data) >= 5:
                            import struct
                            bandwidth, limit_type = struct.unpack('>IB', message.data[0:5])
                            logger.debug(f"Received peer bandwidth: {bandwidth}, type: {limit_type}")
                    
                    elif message.message_type == RTMPMessageType.USER_CONTROL_MESSAGE:
                        # Handle user control messages
                        self._rtmp_protocol.handle_user_control_message(message)
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
            
            if not response_received:
                raise RTMPProtocolError("RTMP connect timeout - no response received")
            
            return True
            
        except Exception as e:
            logger.error(f"RTMP connect request failed: {e}")
            raise RTMPProtocolError(f"RTMP connect request failed: {e}")

    async def connect(self) -> bool:
        """Connect to the RTMP stream.

        Returns:
            bool: True if connection was successful

        Raises:
            ConnectionError: If connection fails
        """
        logger.info(f"Connecting to RTMP stream: {self.url}")

        try:
            self.connection.status = ConnectionStatus.CONNECTING

            # Create connection
            self._reader, self._writer = await self._create_connection()

            # Perform RTMP handshake
            if not await self._perform_rtmp_handshake():
                raise RTMPConnectionError("RTMP handshake failed")

            # Send connect request
            if not await self._send_connect_request():
                raise RTMPConnectionError("RTMP connect request failed")

            # Update connection status
            self.connection.status = ConnectionStatus.CONNECTED
            self.connection.connected_at = datetime.utcnow()
            self.connection.health = StreamHealth.HEALTHY

            # Start reading stream info
            await self._read_stream_info()

            # Start background processing tasks
            await self._start_background_processing()

            # Notify connection
            await self._notify_connect()

            logger.info(f"Successfully connected to RTMP stream: {self.url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to RTMP stream: {e}")
            self.connection.status = ConnectionStatus.FAILED
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.utcnow()
            await self._cleanup_connection()
            await self._notify_error(e)
            raise ConnectionError(f"Failed to connect to RTMP stream: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the RTMP stream."""
        logger.info(f"Disconnecting from RTMP stream: {self.url}")

        # Stop background tasks
        await self._stop_background_processing()
        
        await self._cleanup_connection()

        self.connection.status = ConnectionStatus.DISCONNECTED
        self.connection.health = StreamHealth.UNKNOWN

        await self._notify_disconnect()

        logger.info("Disconnected from RTMP stream")

    async def _cleanup_connection(self) -> None:
        """Clean up RTMP connection resources."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as e:
                logger.debug(f"Error closing RTMP writer: {e}")
            finally:
                self._writer = None

        self._reader = None
        self._socket = None

    async def _read_stream_info(self) -> None:
        """Read and parse stream information from RTMP metadata and FLV data."""
        try:
            # Initialize basic stream info
            self._stream_info = {
                "connected_at": datetime.utcnow().isoformat(),
                "url": self.url,
                "hostname": self.hostname,
                "port": self.port,
                "path": self.path,
                "scheme": self.scheme,
                "app_name": self._app_name,
                "stream_key": self._stream_key,
                "rtmp_protocol_version": "1.0",
                "flv_processor_ready": True,
                "ffmpeg_available": self._ffmpeg_available,
            }
            
            # Try to probe stream if FFmpeg is available
            if self._ffmpeg_available:
                try:
                    probe_url = f"rtmp://{self.hostname}:{self.port}{self.path}"
                    media_info = await FFmpegProbe.probe_stream(probe_url, timeout=5)
                    
                    # Extract video stream info
                    if media_info.video_streams:
                        video_stream = media_info.video_streams[0]
                        self._stream_info.update({
                            "video_width": video_stream.width,
                            "video_height": video_stream.height,
                            "video_fps": video_stream.fps,
                            "video_codec": video_stream.codec,
                            "video_bitrate": video_stream.bitrate,
                            "pixel_format": video_stream.pixel_format,
                        })
                    
                    # Extract audio stream info
                    if media_info.audio_streams:
                        audio_stream = media_info.audio_streams[0]
                        self._stream_info.update({
                            "audio_sample_rate": audio_stream.sample_rate,
                            "audio_channels": audio_stream.channels,
                            "audio_codec": audio_stream.codec,
                            "audio_bitrate": audio_stream.bitrate,
                        })
                    
                    # General format info
                    self._stream_info.update({
                        "format_name": media_info.format_name,
                        "total_bitrate": media_info.bitrate,
                        "metadata": media_info.metadata,
                    })
                    
                    logger.info(f"Probed RTMP stream: {video_stream.resolution if media_info.video_streams else 'audio-only'}")
                    
                except Exception as e:
                    logger.warning(f"Could not probe RTMP stream: {e}")
            
            logger.debug(f"RTMP stream info: {self._stream_info}")
            
        except Exception as e:
            logger.error(f"Error reading stream info: {e}")
            # Ensure basic info is always available
            if not self._stream_info:
                self._stream_info = {
                    "connected_at": datetime.utcnow().isoformat(),
                    "url": self.url,
                    "error": str(e)
                }

    async def _start_background_processing(self) -> None:
        """Start background processing tasks."""
        if not self._message_processing_task:
            self._message_processing_task = asyncio.create_task(
                self._process_rtmp_messages()
            )
        
        if self._ffmpeg_available and not self._frame_extraction_task:
            self._frame_extraction_task = asyncio.create_task(
                self._extract_frames_continuously()
            )

    async def _stop_background_processing(self) -> None:
        """Stop background processing tasks."""
        tasks = []
        
        if self._message_processing_task:
            self._message_processing_task.cancel()
            tasks.append(self._message_processing_task)
            self._message_processing_task = None
        
        if self._frame_extraction_task:
            self._frame_extraction_task.cancel()
            tasks.append(self._frame_extraction_task)
            self._frame_extraction_task = None
        
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _process_rtmp_messages(self) -> None:
        """Background task to process RTMP messages."""
        try:
            while self.is_connected and not self._shutdown:
                if self._reader:
                    try:
                        message = await asyncio.wait_for(
                            self._rtmp_protocol.parse_message(self._reader),
                            timeout=1.0
                        )
                        
                        if message:
                            await self._handle_rtmp_message(message)
                    
                    except asyncio.TimeoutError:
                        continue  # Normal timeout, continue processing
                    except Exception as e:
                        logger.error(f"Error processing RTMP message: {e}")
                        break
                else:
                    await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            logger.debug("RTMP message processing task cancelled")
        except Exception as e:
            logger.error(f"RTMP message processing error: {e}")

    async def _handle_rtmp_message(self, message: RTMPMessage) -> None:
        """Handle individual RTMP messages."""
        try:
            if message.message_type == RTMPMessageType.AMF0_COMMAND:
                result = self._rtmp_protocol.process_command_message(message)
                if result:
                    logger.debug(f"Processed RTMP command: {result.get('command')}")
            
            elif message.message_type == RTMPMessageType.AMF0_DATA:
                # Handle metadata (onMetaData)
                try:
                    decoder = AMFDecoder(message.data)
                    event_name = decoder.decode_value()
                    if event_name == "onMetaData":
                        metadata = decoder.decode_value()
                        if isinstance(metadata, dict):
                            self._stream_info.update(metadata)
                            logger.info(f"Updated stream metadata: {metadata}")
                except Exception as e:
                    logger.warning(f"Error parsing metadata: {e}")
            
            elif message.message_type in [RTMPMessageType.VIDEO, RTMPMessageType.AUDIO]:
                # Video/Audio data will be handled by FLV processor
                pass
            
            elif message.message_type == RTMPMessageType.USER_CONTROL_MESSAGE:
                self._rtmp_protocol.handle_user_control_message(message)
                
        except Exception as e:
            logger.error(f"Error handling RTMP message: {e}")

    async def _extract_frames_continuously(self) -> None:
        """Background task to extract frames using FFmpeg."""
        if not self._ffmpeg_available:
            return
        
        try:
            stream_url = f"rtmp://{self.hostname}:{self.port}{self.path}"
            
            async def frame_callback(frame_info):
                """Callback for processed frames."""
                timestamp = frame_info['timestamp']
                frame_data = frame_info['data']
                width = frame_info['width']
                height = frame_info['height']
                
                # Store frame for analysis
                is_keyframe = self._frame_count % 30 == 0  # Simplified keyframe detection
                self._video_frames.append((frame_data, timestamp, is_keyframe))
                
                # Keep only recent frames (last 10 seconds worth)
                max_frames = 300  # Assuming ~30fps
                if len(self._video_frames) > max_frames:
                    self._video_frames = self._video_frames[-max_frames:]
                
                self._frame_count += 1
                self._last_frame_time = timestamp
                
                # Update stream info with frame information
                self._stream_info.update({
                    "current_frame_count": self._frame_count,
                    "last_frame_time": timestamp,
                    "frame_width": width,
                    "frame_height": height,
                })
            
            # Start real-time frame streaming
            await self._frame_extractor.stream_frames_realtime(
                stream_url, frame_callback, max_duration=None
            )
            
        except asyncio.CancelledError:
            logger.debug("Frame extraction task cancelled")
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")

    async def _process_flv_tag(self, tag: FLVTag) -> Optional[bytes]:
        """Process FLV tag and return processed data."""
        try:
            if isinstance(tag.data, FLVVideoTag):
                # Process video tag
                if (tag.data.codec_id == FLVVideoCodec.AVC and 
                    tag.data.frame_type == FLVVideoFrameType.KEY_FRAME):
                    
                    # This is a key frame - important for highlight detection
                    timestamp = tag.header.full_timestamp / 1000.0  # Convert to seconds
                    
                    # Create processed video data packet
                    video_packet = {
                        "type": "video",
                        "timestamp": timestamp,
                        "codec": tag.data.codec_id.name,
                        "frame_type": tag.data.frame_type.name,
                        "data_size": len(tag.data.data),
                        "is_keyframe": tag.data.frame_type == FLVVideoFrameType.KEY_FRAME,
                    }
                    
                    return json.dumps(video_packet).encode('utf-8')
            
            elif isinstance(tag.data, FLVAudioTag):
                # Process audio tag
                timestamp = tag.header.full_timestamp / 1000.0
                
                audio_packet = {
                    "type": "audio",
                    "timestamp": timestamp,
                    "codec": tag.data.sound_format.name,
                    "channels": 2 if tag.data.sound_type.name == "STEREO" else 1,
                    "data_size": len(tag.data.data),
                }
                
                return json.dumps(audio_packet).encode('utf-8')
            
            # Return None for tags we don't specifically process
            return None
            
        except Exception as e:
            logger.error(f"Error processing FLV tag: {e}")
            return None

    async def get_metadata(self) -> StreamMetadata:
        """Get current stream metadata.

        Returns:
            StreamMetadata: Current stream metadata
        """
        # Get FLV processor stream info
        flv_info = self._flv_processor.get_stream_info()
        
        # Basic metadata from connection info
        self.metadata = StreamMetadata(
            title=f"RTMP Stream - {self.hostname}{self.path}",
            description=f"RTMP stream from {self.hostname}:{self.port}",
            is_live=self.is_connected,
            platform_id=f"{self.hostname}:{self.port}{self.path}",
            platform_url=self.url,
            resolution=f"{self._stream_info.get('video_width', 0)}x{self._stream_info.get('video_height', 0)}",
            framerate=self._stream_info.get('video_fps', 0),
            bitrate=self._stream_info.get('total_bitrate', 0),
            platform_data={
                **self._stream_info,
                "adapter_type": "enhanced_rtmp",
                "connection_info": {
                    "hostname": self.hostname,
                    "port": self.port,
                    "path": self.path,
                    "scheme": self.scheme,
                    "is_secure": self._is_secure,
                    "app_name": self._app_name,
                    "stream_key": self._stream_key,
                },
                "flv_info": flv_info,
                "processing_stats": {
                    "frame_count": self._frame_count,
                    "last_frame_time": self._last_frame_time,
                    "video_frames_buffered": len(self._video_frames),
                    "audio_samples_buffered": len(self._audio_samples),
                },
                "ffmpeg_available": self._ffmpeg_available,
            },
            updated_at=datetime.utcnow(),
        )

        return self.metadata

    async def is_stream_live(self) -> bool:
        """Check if the RTMP stream is currently live.

        Returns:
            bool: True if stream is live (connected)
        """
        return self.is_connected and self._rtmp_protocol.connected

    async def get_stream_data(self) -> AsyncGenerator[bytes, None]:
        """Get processed stream data as an async generator.

        Yields:
            bytes: Processed video/audio data or metadata

        Raises:
            StreamAdapterError: If stream data cannot be retrieved
        """
        if not self._reader:
            raise StreamAdapterError("No RTMP connection available for reading data")

        logger.info("Starting RTMP stream data processing")

        try:
            while self.is_connected and not self._shutdown:
                try:
                    # Read raw RTMP data
                    data = await asyncio.wait_for(
                        self._reader.read(self.buffer_size), timeout=self.read_timeout
                    )

                    if not data:
                        logger.info("RTMP stream ended")
                        break

                    # Update connection stats
                    self.connection.bytes_received += len(data)
                    self.connection.packets_received += 1
                    self.connection.last_data_at = datetime.utcnow()

                    # Process FLV data
                    flv_tags = self._flv_processor.process_data(data)
                    
                    for tag in flv_tags:
                        processed_data = await self._process_flv_tag(tag)
                        if processed_data:
                            # Notify data callback
                            await self._notify_data(processed_data)
                            yield processed_data

                    # If no FLV tags were parsed, yield stream metadata periodically
                    if not flv_tags and len(data) > 0:
                        # Yield stream info as JSON every few packets
                        if self.connection.packets_received % 100 == 0:
                            stream_status = {
                                "type": "stream_status",
                                "timestamp": datetime.utcnow().isoformat(),
                                "bytes_received": self.connection.bytes_received,
                                "packets_received": self.connection.packets_received,
                                "frame_count": self._frame_count,
                                "connection_status": self.connection.status.value,
                                "stream_info": self._stream_info
                            }
                            status_data = json.dumps(stream_status).encode('utf-8')
                            await self._notify_data(status_data)
                            yield status_data

                except asyncio.TimeoutError:
                    logger.warning("RTMP read timeout, checking connection health")
                    if await self.check_health() == StreamHealth.UNHEALTHY:
                        break
                    continue

        except Exception as e:
            logger.error(f"Error reading RTMP stream data: {e}")
            self.connection.error_count += 1
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.utcnow()
            await self._notify_error(e)
            raise StreamAdapterError(f"Failed to read RTMP stream data: {e}")

        logger.info("Finished reading RTMP stream data")

    async def check_health(self) -> StreamHealth:
        """Check the health of the RTMP connection.

        Returns:
            StreamHealth: Current health status
        """
        if not self.is_connected:
            self.connection.health = StreamHealth.UNHEALTHY
            return self.connection.health

        try:
            # Check if connection is still alive
            if self._writer and self._writer.is_closing():
                self.connection.health = StreamHealth.UNHEALTHY
                return self.connection.health

            # Check RTMP protocol state
            if not self._rtmp_protocol.handshake_complete:
                self.connection.health = StreamHealth.UNHEALTHY
                return self.connection.health

            # Check data freshness
            now = datetime.utcnow()
            if self.connection.last_data_at:
                time_since_data = (now - self.connection.last_data_at).total_seconds()
                if time_since_data > 60:  # No data for 1 minute
                    self.connection.health = StreamHealth.DEGRADED
                    return self.connection.health

            # Check frame processing if FFmpeg is available
            if self._ffmpeg_available and self._frame_count > 0:
                current_time = asyncio.get_event_loop().time()
                if current_time - self._last_frame_time > 30:  # No frames for 30 seconds
                    self.connection.health = StreamHealth.DEGRADED
                    return self.connection.health

            self.connection.health = StreamHealth.HEALTHY
            self.connection.last_health_check_at = now

        except Exception as e:
            logger.error(f"RTMP health check failed: {e}")
            self.connection.health = StreamHealth.UNHEALTHY
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.utcnow()

        return self.connection.health

    async def get_stream_analytics(self) -> Dict[str, Any]:
        """Get stream analytics and statistics.

        Returns:
            Dict: Stream analytics data
        """
        flv_info = self._flv_processor.get_stream_info()
        
        analytics = {
            "platform": "enhanced_rtmp",
            "url": self.url,
            "hostname": self.hostname,
            "port": self.port,
            "path": self.path,
            "scheme": self.scheme,
            "is_secure": self._is_secure,
            "buffer_size": self.buffer_size,
            "app_name": self._app_name,
            "stream_key": self._stream_key,
            "connection_status": self.connection.status.value,
            "health_status": self.connection.health.value,
            "reconnect_count": self.connection.reconnect_count,
            "error_count": self.connection.error_count,
            "bytes_received": self.connection.bytes_received,
            "packets_received": self.connection.packets_received,
            "last_data_at": self.connection.last_data_at,
            "rtmp_protocol": {
                "handshake_complete": self._rtmp_protocol.handshake_complete,
                "connected": self._rtmp_protocol.connected,
                "transaction_id": self._rtmp_protocol.transaction_id,
            },
            "flv_processing": flv_info,
            "video_processing": {
                "frame_count": self._frame_count,
                "last_frame_time": self._last_frame_time,
                "frames_buffered": len(self._video_frames),
                "audio_samples_buffered": len(self._audio_samples),
            },
            "ffmpeg": {
                "available": self._ffmpeg_available,
                "hardware_acceleration": self._ffmpeg_processor.hardware_acceleration,
                "hw_accel_method": self._ffmpeg_processor.hw_accel_method,
            },
            "stream_info": self._stream_info,
            "metadata": self.metadata.__dict__ if self.metadata else None,
        }

        return analytics

    async def get_recent_frames(self, count: int = 10) -> List[Tuple[bytes, float, bool]]:
        """Get recent video frames for analysis.
        
        Args:
            count: Number of recent frames to return
            
        Returns:
            List of (frame_data, timestamp, is_keyframe) tuples
        """
        return self._video_frames[-count:] if self._video_frames else []

    async def get_frame_at_timestamp(self, timestamp: float) -> Optional[Tuple[bytes, float, bool]]:
        """Get frame closest to specified timestamp.
        
        Args:
            timestamp: Target timestamp in seconds
            
        Returns:
            (frame_data, timestamp, is_keyframe) tuple or None
        """
        if not self._video_frames:
            return None
        
        # Find frame closest to timestamp
        closest_frame = None
        min_diff = float('inf')
        
        for frame_data, frame_timestamp, is_keyframe in self._video_frames:
            diff = abs(frame_timestamp - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_frame = (frame_data, frame_timestamp, is_keyframe)
        
        return closest_frame

    async def start_recording(self, output_path: str) -> bool:
        """Start recording the RTMP stream to file.
        
        Args:
            output_path: Path to output file
            
        Returns:
            bool: True if recording started successfully
        """
        if not self._ffmpeg_available:
            logger.error("Cannot start recording: FFmpeg not available")
            return False
        
        try:
            stream_url = f"rtmp://{self.hostname}:{self.port}{self.path}"
            
            # Use FFmpeg to record the stream
            success = await self._ffmpeg_processor.transcode_stream(
                stream_url, output_path, self._transcode_options
            )
            
            if success:
                self._stream_info["recording"] = {
                    "active": True,
                    "output_path": output_path,
                    "started_at": datetime.utcnow().isoformat()
                }
                logger.info(f"Started recording RTMP stream to {output_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False

    def __repr__(self) -> str:
        """String representation of the enhanced RTMP adapter."""
        return (
            f"EnhancedRTMPAdapter(hostname='{self.hostname}', port={self.port}, "
            f"path='{self.path}', status='{self.connection.status}', "
            f"health='{self.connection.health}', ffmpeg={self._ffmpeg_available})"
        )


# Alias for backward compatibility
RTMPAdapter = EnhancedRTMPAdapter