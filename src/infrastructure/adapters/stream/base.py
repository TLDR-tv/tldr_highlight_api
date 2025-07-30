"""Base stream adapter protocol and common functionality.

This module defines the stream adapter protocol and common functionality
using Pythonic patterns with Protocol instead of ABC.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable, Protocol, runtime_checkable
from contextlib import asynccontextmanager

from aiohttp import ClientSession, ClientTimeout

logger = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """Stream connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    CLOSED = "closed"


class StreamHealth(str, Enum):
    """Stream health status."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class StreamMetadata:
    """Metadata about a stream."""
    # Basic stream information
    title: Optional[str] = None
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    
    # Stream details
    is_live: bool = False
    viewer_count: Optional[int] = None
    duration_seconds: Optional[int] = None
    started_at: Optional[datetime] = None
    
    # Platform-specific information
    platform_id: Optional[str] = None
    platform_url: Optional[str] = None
    platform_data: Dict[str, Any] = field(default_factory=dict)
    
    # Stream quality information
    resolution: Optional[str] = None
    framerate: Optional[int] = None
    bitrate: Optional[int] = None
    
    # Categories and tags
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Language and region
    language: Optional[str] = None
    region: Optional[str] = None
    
    # Last update timestamp
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class StreamConnection:
    """Information about a stream connection."""
    # Connection details
    url: str
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    health: StreamHealth = StreamHealth.UNKNOWN
    
    # Connection timing
    connected_at: Optional[datetime] = None
    last_data_at: Optional[datetime] = None
    last_health_check_at: Optional[datetime] = None
    
    # Connection metrics
    bytes_received: int = 0
    packets_received: int = 0
    reconnect_count: int = 0
    error_count: int = 0
    
    # Error information
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None
    
    # Configuration
    reconnect_attempts: int = 3
    reconnect_delay: float = 5.0
    health_check_interval: float = 30.0


class StreamAdapterError(Exception):
    """Base exception for stream adapter errors."""
    pass


class ConnectionError(StreamAdapterError):
    """Exception raised for connection-related errors."""
    pass


class AuthenticationError(StreamAdapterError):
    """Exception raised for authentication-related errors."""
    pass


class RateLimitError(StreamAdapterError):
    """Exception raised when rate limits are exceeded."""
    pass


class StreamNotFoundError(StreamAdapterError):
    """Exception raised when a stream is not found."""
    pass


class StreamOfflineError(StreamAdapterError):
    """Exception raised when a stream is offline."""
    pass


@runtime_checkable
class StreamAdapter(Protocol):
    """Protocol for stream adapters.
    
    This protocol defines the interface that all stream adapters must implement.
    Uses Python Protocol for structural subtyping instead of ABC.
    """
    
    @property
    def url(self) -> str:
        """Stream URL."""
        ...
    
    @property
    def connection(self) -> StreamConnection:
        """Connection information."""
        ...
    
    @property
    def metadata(self) -> StreamMetadata:
        """Stream metadata."""
        ...
    
    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        ...
    
    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        ...
    
    @property
    def platform_name(self) -> str:
        """Get the platform name."""
        ...
    
    async def authenticate(self) -> bool:
        """Authenticate with the platform.
        
        Returns:
            bool: True if authentication was successful
            
        Raises:
            AuthenticationError: If authentication fails
        """
        ...
    
    async def connect(self) -> bool:
        """Connect to the stream.
        
        Returns:
            bool: True if connection was successful
            
        Raises:
            ConnectionError: If connection fails
        """
        ...
    
    async def disconnect(self) -> None:
        """Disconnect from the stream."""
        ...
    
    async def get_metadata(self) -> StreamMetadata:
        """Get current stream metadata.
        
        Returns:
            StreamMetadata: Current stream metadata
            
        Raises:
            StreamAdapterError: If metadata cannot be retrieved
        """
        ...
    
    async def is_stream_live(self) -> bool:
        """Check if the stream is currently live.
        
        Returns:
            bool: True if stream is live
            
        Raises:
            StreamAdapterError: If status cannot be determined
        """
        ...
    
    async def get_stream_data(self) -> AsyncGenerator[bytes, None]:
        """Get stream data as an async generator.
        
        Yields:
            bytes: Stream data chunks
            
        Raises:
            StreamAdapterError: If stream data cannot be retrieved
        """
        ...
    
    async def start(self) -> None:
        """Start the stream adapter."""
        ...
    
    async def stop(self) -> None:
        """Stop the stream adapter."""
        ...
    
    async def check_health(self) -> StreamHealth:
        """Check the health of the stream connection.
        
        Returns:
            StreamHealth: Current health status
        """
        ...


class BaseStreamAdapter:
    """Base implementation with common functionality for stream adapters.
    
    This class provides common functionality that can be shared by concrete
    adapter implementations. It's not abstract - subclasses just extend it.
    """
    
    def __init__(
        self,
        url: str,
        session: Optional[ClientSession] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        health_check_interval: float = 30.0,
        **kwargs,
    ):
        """Initialize the stream adapter.
        
        Args:
            url: The stream URL
            session: Optional aiohttp ClientSession
            timeout: Connection timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
            health_check_interval: Health check interval in seconds
            **kwargs: Additional platform-specific configuration
        """
        self.url = url
        self._session = session
        self._session_owned = session is None
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.health_check_interval = health_check_interval
        
        # Connection state
        self.connection = StreamConnection(
            url=url,
            reconnect_attempts=max_retries,
            reconnect_delay=retry_delay,
            health_check_interval=health_check_interval,
        )
        
        # Metadata
        self.metadata = StreamMetadata()
        
        # Event callbacks
        self._on_connect_callbacks: List[Callable] = []
        self._on_disconnect_callbacks: List[Callable] = []
        self._on_data_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable] = []
        self._on_metadata_update_callbacks: List[Callable] = []
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._connection_task: Optional[asyncio.Task] = None
        
        # Shutdown flag
        self._shutdown = False
        
        logger.info(f"Initialized {self.__class__.__name__} for URL: {url}")
    
    @property
    def session(self) -> ClientSession:
        """Get the HTTP session, creating one if necessary."""
        if self._session is None:
            timeout = ClientTimeout(total=self.timeout)
            self._session = ClientSession(timeout=timeout)
            self._session_owned = True
        return self._session
    
    @property
    def is_connected(self) -> bool:
        """Check if the adapter is connected."""
        return self.connection.status == ConnectionStatus.CONNECTED
    
    @property
    def is_healthy(self) -> bool:
        """Check if the connection is healthy."""
        return self.connection.health == StreamHealth.HEALTHY
    
    @property
    def platform_name(self) -> str:
        """Get the platform name."""
        return self.__class__.__name__.replace("StreamAdapter", "").lower()
    
    async def start(self) -> None:
        """Start the stream adapter.
        
        This method authenticates, connects, and starts background tasks.
        """
        logger.info(f"Starting {self.platform_name} adapter for URL: {self.url}")
        
        try:
            # Authenticate
            if not await self.authenticate():
                raise AuthenticationError("Authentication failed")
            
            # Connect
            if not await self.connect():
                raise ConnectionError("Connection failed")
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info(f"Started {self.platform_name} adapter successfully")
            
        except Exception as e:
            logger.error(f"Failed to start {self.platform_name} adapter: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the stream adapter.
        
        This method stops background tasks and disconnects from the stream.
        """
        logger.info(f"Stopping {self.platform_name} adapter")
        
        self._shutdown = True
        
        # Stop background tasks
        await self._stop_background_tasks()
        
        # Disconnect
        await self.disconnect()
        
        # Close session if we own it
        if self._session_owned and self._session:
            await self._session.close()
            self._session = None
        
        logger.info(f"Stopped {self.platform_name} adapter")
    
    async def reconnect(self) -> bool:
        """Attempt to reconnect to the stream.
        
        Returns:
            bool: True if reconnection was successful
        """
        logger.info(f"Attempting to reconnect {self.platform_name} adapter")
        
        self.connection.status = ConnectionStatus.RECONNECTING
        self.connection.reconnect_count += 1
        
        try:
            # Disconnect first
            await self.disconnect()
            
            # Wait before reconnecting
            await asyncio.sleep(self.connection.reconnect_delay)
            
            # Reconnect
            if await self.connect():
                logger.info(f"Reconnected {self.platform_name} adapter successfully")
                return True
            else:
                logger.error(f"Failed to reconnect {self.platform_name} adapter")
                return False
                
        except Exception as e:
            logger.error(f"Error during reconnection: {e}")
            self.connection.status = ConnectionStatus.FAILED
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.now(timezone.utc)
            return False
    
    async def check_health(self) -> StreamHealth:
        """Check the health of the stream connection.
        
        Returns:
            StreamHealth: Current health status
        """
        try:
            # Check if stream is still live
            if not await self.is_stream_live():
                self.connection.health = StreamHealth.UNHEALTHY
                return self.connection.health
            
            # Check connection freshness
            now = datetime.now(timezone.utc)
            if self.connection.last_data_at:
                time_since_data = (now - self.connection.last_data_at).total_seconds()
                if time_since_data > 60:  # No data for 1 minute
                    self.connection.health = StreamHealth.DEGRADED
                    return self.connection.health
            
            # Update metadata
            self.metadata = await self.get_metadata()
            await self._notify_metadata_update()
            
            self.connection.health = StreamHealth.HEALTHY
            self.connection.last_health_check_at = now
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.connection.health = StreamHealth.UNHEALTHY
            self.connection.last_error = str(e)
            self.connection.last_error_at = datetime.now(timezone.utc)
        
        return self.connection.health
    
    # Event handling
    
    def on_connect(self, callback: Callable) -> None:
        """Register a callback for connection events."""
        self._on_connect_callbacks.append(callback)
    
    def on_disconnect(self, callback: Callable) -> None:
        """Register a callback for disconnection events."""
        self._on_disconnect_callbacks.append(callback)
    
    def on_data(self, callback: Callable) -> None:
        """Register a callback for data events."""
        self._on_data_callbacks.append(callback)
    
    def on_error(self, callback: Callable) -> None:
        """Register a callback for error events."""
        self._on_error_callbacks.append(callback)
    
    def on_metadata_update(self, callback: Callable) -> None:
        """Register a callback for metadata update events."""
        self._on_metadata_update_callbacks.append(callback)
    
    # Protected methods
    
    async def _notify_connect(self) -> None:
        """Notify connection callbacks."""
        for callback in self._on_connect_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self)
                else:
                    callback(self)
            except Exception as e:
                logger.error(f"Error in connect callback: {e}")
    
    async def _notify_disconnect(self) -> None:
        """Notify disconnection callbacks."""
        for callback in self._on_disconnect_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self)
                else:
                    callback(self)
            except Exception as e:
                logger.error(f"Error in disconnect callback: {e}")
    
    async def _notify_data(self, data: bytes) -> None:
        """Notify data callbacks."""
        for callback in self._on_data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self, data)
                else:
                    callback(self, data)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    async def _notify_error(self, error: Exception) -> None:
        """Notify error callbacks."""
        for callback in self._on_error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self, error)
                else:
                    callback(self, error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    async def _notify_metadata_update(self) -> None:
        """Notify metadata update callbacks."""
        for callback in self._on_metadata_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self, self.metadata)
                else:
                    callback(self, self.metadata)
            except Exception as e:
                logger.error(f"Error in metadata update callback: {e}")
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _stop_background_tasks(self) -> None:
        """Stop background tasks."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown:
            try:
                await self.check_health()
                
                # If unhealthy, attempt reconnection
                if (
                    self.connection.health == StreamHealth.UNHEALTHY
                    and self.connection.reconnect_count < self.connection.reconnect_attempts
                ):
                    await self.reconnect()
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of adapter metrics."""
        return {
            "platform": self.platform_name,
            "url": self.url,
            "connection_status": self.connection.status.value,
            "health_status": self.connection.health.value,
            "bytes_received": self.connection.bytes_received,
            "packets_received": self.connection.packets_received,
            "reconnect_count": self.connection.reconnect_count,
            "error_count": self.connection.error_count,
            "last_data_at": self.connection.last_data_at.isoformat()
            if self.connection.last_data_at
            else None,
            "connected_at": self.connection.connected_at.isoformat()
            if self.connection.connected_at
            else None,
        }
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return (
            f"{self.__class__.__name__}(url='{self.url}', "
            f"status='{self.connection.status}', "
            f"health='{self.connection.health}')"
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


@asynccontextmanager
async def stream_adapter_context(adapter: StreamAdapter):
    """Context manager for stream adapters.
    
    This provides a convenient way to ensure proper cleanup of stream adapters.
    
    Args:
        adapter: The stream adapter to manage
        
    Yields:
        The started stream adapter
    """
    try:
        await adapter.start()
        yield adapter
    finally:
        await adapter.stop()