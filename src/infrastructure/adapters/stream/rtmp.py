"""RTMP stream adapter infrastructure implementation.

This module provides the RTMP streaming protocol adapter using
Pythonic patterns and async/await throughout.
"""

import asyncio
import logging
import socket
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Optional, Any, Tuple
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from aiohttp import ClientSession

from src.core.config import get_settings
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


class RTMPStreamAdapter(BaseStreamAdapter):
    """RTMP stream adapter for RTMP protocol support.
    
    Provides basic RTMP connectivity and metadata access.
    Full RTMP protocol implementation would require additional
    infrastructure components.
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
        
        # Parse URL
        parsed = urlparse(url)
        self.hostname = parsed.hostname or "localhost"
        self.port = parsed.port or (443 if parsed.scheme == "rtmps" else 1935)
        self.path = parsed.path or "/"
        self.scheme = parsed.scheme or "rtmp"
        self.is_secure = self.scheme == "rtmps"
        
        # RTMP configuration
        self.buffer_size = buffer_size or settings.rtmp_buffer_size
        self.connection_timeout = settings.rtmp_connection_timeout
        self.read_timeout = settings.rtmp_read_timeout
        
        # Connection state
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        
        # Stream configuration from URL path
        path_parts = self.path.strip("/").split("/")
        self.app_name = path_parts[0] if path_parts else "live"
        self.stream_key = "/".join(path_parts[1:]) if len(path_parts) > 1 else ""
        
        # Stream state
        self._stream_info: Dict[str, Any] = {}
        self._bytes_buffer = bytearray()
        
        logger.info(
            f"Initialized RTMP adapter for {self.hostname}:{self.port}{self.path}"
        )
    
    async def authenticate(self) -> bool:
        """Authenticate with RTMP server.
        
        For basic RTMP streams, no authentication is typically required.
        
        Returns:
            bool: True (RTMP typically doesn't require separate authentication)
        """
        logger.info("RTMP authentication not required for basic streams")
        return True
    
    async def _create_connection(self) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Create an RTMP connection.
        
        Returns:
            Tuple of StreamReader and StreamWriter
            
        Raises:
            RTMPConnectionError: If connection fails
        """
        try:
            if self.is_secure:
                # For RTMPS, we would need SSL context
                import ssl
                
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(
                        self.hostname, self.port, ssl=ssl_context
                    ),
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
                f"Connection timeout to {self.hostname}:{self.port}"
            )
        except Exception as e:
            raise RTMPConnectionError(f"Failed to connect: {e}")
    
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
            
            # Create TCP connection
            self._reader, self._writer = await self._create_connection()
            
            # In a full implementation, we would:
            # 1. Perform RTMP handshake
            # 2. Send connect command
            # 3. Create/play stream
            
            logger.warning(
                "Full RTMP protocol implementation not included. "
                "Connection established but protocol handshake not performed."
            )
            
            # Update connection status
            self.connection.status = ConnectionStatus.CONNECTED
            self.connection.connected_at = datetime.now(timezone.utc)
            self.connection.health = StreamHealth.HEALTHY
            
            # Notify connection
            await self._notify_connect()
            
            logger.info(f"Successfully connected to RTMP stream: {self.url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to RTMP stream: {e}")
            self.connection.status = ConnectionStatus.FAILED
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.now(timezone.utc)
            await self._notify_error(e)
            raise ConnectionError(f"Failed to connect: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the RTMP stream."""
        logger.info(f"Disconnecting from RTMP stream: {self.url}")
        
        # Close connection
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as e:
                logger.warning(f"Error closing RTMP connection: {e}")
        
        self._reader = None
        self._writer = None
        
        self.connection.status = ConnectionStatus.DISCONNECTED
        self.connection.health = StreamHealth.UNKNOWN
        
        await self._notify_disconnect()
        
        logger.info("Disconnected from RTMP stream")
    
    async def get_metadata(self) -> StreamMetadata:
        """Get current stream metadata.
        
        For RTMP, metadata is typically received in the stream data.
        This returns basic metadata based on connection info.
        
        Returns:
            StreamMetadata: Current stream metadata
        """
        # Update basic metadata
        self.metadata = StreamMetadata(
            title=f"RTMP Stream: {self.stream_key or self.app_name}",
            description=f"RTMP stream from {self.hostname}:{self.port}",
            is_live=self.is_connected,
            platform_id=self.stream_key or self.app_name,
            platform_url=self.url,
            platform_data={
                "hostname": self.hostname,
                "port": self.port,
                "app_name": self.app_name,
                "stream_key": self.stream_key,
                "scheme": self.scheme,
                "is_secure": self.is_secure,
            },
            updated_at=datetime.now(timezone.utc),
        )
        
        return self.metadata
    
    async def is_stream_live(self) -> bool:
        """Check if the RTMP stream is currently live.
        
        For RTMP, we consider it live if we're connected.
        
        Returns:
            bool: True if stream is live
        """
        is_live = self.is_connected and self._reader is not None
        
        logger.debug(f"RTMP stream {self.url} live status: {is_live}")
        return is_live
    
    async def get_stream_data(self) -> AsyncGenerator[bytes, None]:
        """Get stream data as an async generator.
        
        Currently returns empty generator - full RTMP implementation
        would require protocol handling and FLV parsing.
        
        Yields:
            bytes: Stream data chunks
        """
        if not self.is_connected or not self._reader:
            raise ConnectionError("Not connected to RTMP stream")
        
        logger.warning(
            "Full RTMP data streaming not implemented in infrastructure layer. "
            "This would require RTMP protocol and FLV parser components."
        )
        
        # Placeholder implementation - read some data to show connectivity
        try:
            # Read a small chunk to demonstrate connection
            data = await asyncio.wait_for(
                self._reader.read(1024),
                timeout=self.read_timeout
            )
            
            if data:
                self.connection.bytes_received += len(data)
                self.connection.packets_received += 1
                self.connection.last_data_at = datetime.now(timezone.utc)
                
                await self._notify_data(data)
                yield data
                
        except asyncio.TimeoutError:
            logger.warning("RTMP read timeout")
        except Exception as e:
            logger.error(f"Error reading RTMP data: {e}")
            self.connection.error_count += 1
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.now(timezone.utc)
            raise
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return (
            f"RTMPStreamAdapter(url='{self.url}', "
            f"status='{self.connection.status}', "
            f"health='{self.connection.health}')"
        )


@asynccontextmanager
async def rtmp_stream(url: str, **kwargs):
    """Context manager for RTMP streams.
    
    Args:
        url: RTMP stream URL
        **kwargs: Additional adapter configuration
        
    Yields:
        Connected RTMPStreamAdapter
    """
    adapter = RTMPStreamAdapter(url, **kwargs)
    try:
        await adapter.start()
        yield adapter
    finally:
        await adapter.stop()