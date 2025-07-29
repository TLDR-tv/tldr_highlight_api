"""RTMP stream adapter for generic RTMP stream processing.

This module provides the RTMPAdapter class for connecting to and processing
generic RTMP streams. It handles RTMP connection management, stream metadata
extraction, and stream data access.
"""

import asyncio
import logging
import socket
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional, Any, Tuple

from aiohttp import ClientSession

from src.core.config import get_settings
from src.utils.stream_validation import validate_stream_url
from src.infrastructure.persistence.models.stream import StreamPlatform
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


class RTMPAdapter(BaseStreamAdapter):
    """RTMP stream adapter for generic RTMP stream processing.

    This adapter handles RTMP connections, extracts basic stream metadata,
    and provides access to stream data for processing. It supports both
    RTMP and RTMPS protocols.
    """

    def __init__(
        self,
        url: str,
        buffer_size: Optional[int] = None,
        session: Optional[ClientSession] = None,
        **kwargs,
    ):
        """Initialize the RTMP adapter.

        Args:
            url: RTMP stream URL
            buffer_size: Buffer size for RTMP data (optional)
            session: Optional aiohttp ClientSession (not used for RTMP)
            **kwargs: Additional configuration options
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

        # Stream data
        self._stream_info: Dict[str, Any] = {}
        self._bytes_buffer = bytearray()

        logger.info(
            f"Initialized RTMP adapter for {self.hostname}:{self.port}{self.path}"
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
        """Perform RTMP handshake.

        This is a simplified handshake implementation.
        A full RTMP implementation would require proper handshaking protocol.

        Returns:
            bool: True if handshake successful

        Raises:
            RTMPProtocolError: If handshake fails
        """
        if not self._reader or not self._writer:
            raise RTMPProtocolError("No RTMP connection available for handshake")

        try:
            # Simplified RTMP handshake
            # In a real implementation, this would follow the full RTMP specification

            # C0: Version byte (0x03)
            self._writer.write(b"\\x03")

            # C1: 1536 bytes of timestamp and random data
            import time
            import random

            timestamp = int(time.time()).to_bytes(4, "big")
            zero_bytes = b"\\x00" * 4
            random_bytes = bytes([random.randint(0, 255) for _ in range(1528)])
            c1 = timestamp + zero_bytes + random_bytes
            self._writer.write(c1)

            await self._writer.drain()

            # Read S0 and S1
            s0 = await asyncio.wait_for(self._reader.read(1), timeout=self.read_timeout)

            if s0 != b"\\x03":
                raise RTMPProtocolError(f"Invalid RTMP version in S0: {s0}")

            s1 = await asyncio.wait_for(
                self._reader.read(1536), timeout=self.read_timeout
            )

            if len(s1) != 1536:
                raise RTMPProtocolError(f"Invalid S1 length: {len(s1)}")

            # C2: Echo S1
            self._writer.write(s1)
            await self._writer.drain()

            # Read S2
            s2 = await asyncio.wait_for(
                self._reader.read(1536), timeout=self.read_timeout
            )

            if len(s2) != 1536:
                raise RTMPProtocolError(f"Invalid S2 length: {len(s2)}")

            logger.debug("RTMP handshake completed successfully")
            return True

        except asyncio.TimeoutError:
            raise RTMPProtocolError("RTMP handshake timeout")
        except Exception as e:
            raise RTMPProtocolError(f"RTMP handshake failed: {e}")

    async def _send_connect_request(self) -> bool:
        """Send RTMP connect request.

        Returns:
            bool: True if connect request successful

        Raises:
            RTMPProtocolError: If connect request fails
        """
        # This is a placeholder for RTMP connect request
        # A full implementation would construct proper RTMP messages
        logger.debug("Sending RTMP connect request (placeholder)")

        # In a real implementation, you would:
        # 1. Construct RTMP connect command
        # 2. Send it over the connection
        # 3. Wait for and parse the response
        # 4. Handle any errors or redirects

        return True

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
        """Read and parse stream information from RTMP metadata.

        This is a placeholder implementation. A full RTMP implementation
        would parse RTMP metadata messages to extract stream information.
        """
        # Placeholder for reading RTMP metadata
        # In a real implementation, you would:
        # 1. Read RTMP messages
        # 2. Parse metadata messages (onMetaData)
        # 3. Extract stream information like resolution, bitrate, etc.

        self._stream_info = {
            "connected_at": datetime.utcnow().isoformat(),
            "url": self.url,
            "hostname": self.hostname,
            "port": self.port,
            "path": self.path,
            "scheme": self.scheme,
        }

        logger.debug(f"Stream info placeholder: {self._stream_info}")

    async def get_metadata(self) -> StreamMetadata:
        """Get current stream metadata.

        Returns:
            StreamMetadata: Current stream metadata
        """
        # Basic metadata from connection info
        self.metadata = StreamMetadata(
            title=f"RTMP Stream - {self.hostname}{self.path}",
            description=f"RTMP stream from {self.hostname}:{self.port}",
            is_live=self.is_connected,
            platform_id=f"{self.hostname}:{self.port}{self.path}",
            platform_url=self.url,
            platform_data={
                **self._stream_info,
                "adapter_type": "rtmp",
                "connection_info": {
                    "hostname": self.hostname,
                    "port": self.port,
                    "path": self.path,
                    "scheme": self.scheme,
                    "is_secure": self._is_secure,
                },
            },
            updated_at=datetime.utcnow(),
        )

        return self.metadata

    async def is_stream_live(self) -> bool:
        """Check if the RTMP stream is currently live.

        Returns:
            bool: True if stream is live (connected)
        """
        return self.is_connected

    async def get_stream_data(self) -> AsyncGenerator[bytes, None]:
        """Get stream data as an async generator.

        Yields:
            bytes: Stream data chunks

        Raises:
            StreamAdapterError: If stream data cannot be retrieved
        """
        if not self._reader:
            raise StreamAdapterError("No RTMP connection available for reading data")

        logger.info("Starting RTMP stream data reading")

        try:
            while self.is_connected and not self._shutdown:
                try:
                    # Read data chunk
                    data = await asyncio.wait_for(
                        self._reader.read(self.buffer_size), timeout=self.read_timeout
                    )

                    if not data:
                        # End of stream
                        logger.info("RTMP stream ended")
                        break

                    # Update connection stats
                    self.connection.bytes_received += len(data)
                    self.connection.packets_received += 1
                    self.connection.last_data_at = datetime.utcnow()

                    # Notify data callback
                    await self._notify_data(data)

                    yield data

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

            # Check data freshness
            now = datetime.utcnow()
            if self.connection.last_data_at:
                time_since_data = (now - self.connection.last_data_at).total_seconds()
                if time_since_data > 60:  # No data for 1 minute
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
        analytics = {
            "platform": "rtmp",
            "url": self.url,
            "hostname": self.hostname,
            "port": self.port,
            "path": self.path,
            "scheme": self.scheme,
            "is_secure": self._is_secure,
            "buffer_size": self.buffer_size,
            "connection_status": self.connection.status.value,
            "health_status": self.connection.health.value,
            "reconnect_count": self.connection.reconnect_count,
            "error_count": self.connection.error_count,
            "bytes_received": self.connection.bytes_received,
            "packets_received": self.connection.packets_received,
            "last_data_at": self.connection.last_data_at,
            "stream_info": self._stream_info,
            "metadata": self.metadata.__dict__ if self.metadata else None,
        }

        return analytics

    def __repr__(self) -> str:
        """String representation of the RTMP adapter."""
        return (
            f"RTMPAdapter(hostname='{self.hostname}', port={self.port}, "
            f"path='{self.path}', status='{self.connection.status}', "
            f"health='{self.connection.health}')"
        )
