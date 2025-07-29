"""Stream and chat synchronization module.

This module provides synchronization between stream adapters and chat adapters
to ensure timeline alignment for highlight detection.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import uuid

from src.services.stream_adapters.base import BaseStreamAdapter
from .base import BaseChatAdapter, ChatEvent, ChatEventType


logger = logging.getLogger(__name__)


@dataclass
class SyncEvent:
    """Synchronized event with timeline information."""
    
    id: str
    timestamp: datetime
    stream_time: float  # Seconds since stream start
    chat_event: Optional[ChatEvent] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamChatSynchronizer:
    """Synchronizes chat events with stream timeline.
    
    This class coordinates between a stream adapter and chat adapter to provide
    timeline-synchronized events for highlight detection.
    """
    
    def __init__(
        self,
        stream_adapter: BaseStreamAdapter,
        chat_adapter: BaseChatAdapter,
        buffer_seconds: float = 30.0,
        sync_interval: float = 1.0
    ):
        """Initialize the synchronizer.
        
        Args:
            stream_adapter: The stream adapter instance
            chat_adapter: The chat adapter instance
            buffer_seconds: Buffer duration for event correlation
            sync_interval: Synchronization check interval
        """
        self.stream_adapter = stream_adapter
        self.chat_adapter = chat_adapter
        self.buffer_seconds = buffer_seconds
        self.sync_interval = sync_interval
        
        # Stream timing
        self.stream_start_time: Optional[datetime] = None
        self.stream_current_time: float = 0.0
        
        # Event buffer for correlation
        self._event_buffer: List[SyncEvent] = []
        self._buffer_lock = asyncio.Lock()
        
        # Callbacks
        self._sync_callbacks: List[Callable] = []
        
        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._chat_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._events_synced = 0
        self._events_dropped = 0
        
        # Shutdown flag
        self._shutdown = False
        
        logger.info(
            f"Initialized stream-chat synchronizer for "
            f"{stream_adapter.platform_name} / {chat_adapter.platform_name}"
        )
    
    async def start(self) -> None:
        """Start the synchronizer."""
        logger.info("Starting stream-chat synchronizer")
        
        # Get stream start time from metadata
        if self.stream_adapter.metadata and self.stream_adapter.metadata.started_at:
            self.stream_start_time = self.stream_adapter.metadata.started_at
        else:
            # Use current time as fallback
            self.stream_start_time = datetime.now(timezone.utc)
            logger.warning("Stream start time not available, using current time")
        
        # Register chat event handlers
        self.chat_adapter.on_any_event(self._handle_chat_event)
        
        # Start background tasks
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._chat_task = asyncio.create_task(self._process_chat_events())
        
        logger.info("Stream-chat synchronizer started")
    
    async def stop(self) -> None:
        """Stop the synchronizer."""
        logger.info("Stopping stream-chat synchronizer")
        
        self._shutdown = True
        
        # Cancel background tasks
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        if self._chat_task:
            self._chat_task.cancel()
            try:
                await self._chat_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Synchronizer stopped - synced: {self._events_synced}, dropped: {self._events_dropped}")
    
    def on_sync_event(self, callback: Callable) -> None:
        """Register a callback for synchronized events.
        
        Args:
            callback: Function to call with SyncEvent
        """
        self._sync_callbacks.append(callback)
    
    async def _handle_chat_event(self, event: ChatEvent) -> None:
        """Handle incoming chat events.
        
        Args:
            event: The chat event to handle
        """
        if self._shutdown:
            return
        
        # Calculate stream time
        if self.stream_start_time:
            stream_time = (event.timestamp - self.stream_start_time).total_seconds()
        else:
            stream_time = self.stream_current_time
        
        # Create synchronized event
        sync_event = SyncEvent(
            id=str(uuid.uuid4()),
            timestamp=event.timestamp,
            stream_time=stream_time,
            chat_event=event,
            metadata={
                "platform": self.chat_adapter.platform_name,
                "channel_id": self.chat_adapter.channel_id,
            }
        )
        
        # Add to buffer
        async with self._buffer_lock:
            self._event_buffer.append(sync_event)
            
            # Clean old events from buffer
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.buffer_seconds)
            self._event_buffer = [e for e in self._event_buffer if e.timestamp > cutoff_time]
    
    async def _sync_loop(self) -> None:
        """Background task to maintain timeline synchronization."""
        while not self._shutdown:
            try:
                # Update stream current time
                if self.stream_adapter.is_connected and self.stream_adapter.metadata:
                    metadata = self.stream_adapter.metadata
                    if metadata.started_at:
                        self.stream_current_time = (
                            datetime.now(timezone.utc) - metadata.started_at
                        ).total_seconds()
                
                # Process buffered events
                await self._process_buffer()
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(self.sync_interval)
    
    async def _process_chat_events(self) -> None:
        """Process chat events from the adapter."""
        try:
            async for event in self.chat_adapter.get_events():
                if self._shutdown:
                    break
                
                # Events are handled via callback registered in start()
                # This loop just keeps the generator running
                pass
                
        except Exception as e:
            logger.error(f"Error processing chat events: {e}")
    
    async def _process_buffer(self) -> None:
        """Process events in the buffer."""
        async with self._buffer_lock:
            current_time = datetime.now(timezone.utc)
            
            # Find events ready for processing (slight delay for alignment)
            ready_events = []
            remaining_events = []
            
            for event in self._event_buffer:
                age = (current_time - event.timestamp).total_seconds()
                if age > 2.0:  # 2 second delay for alignment
                    ready_events.append(event)
                else:
                    remaining_events.append(event)
            
            self._event_buffer = remaining_events
            
            # Notify callbacks for ready events
            for event in ready_events:
                await self._notify_sync_event(event)
                self._events_synced += 1
    
    async def _notify_sync_event(self, event: SyncEvent) -> None:
        """Notify callbacks of a synchronized event.
        
        Args:
            event: The synchronized event
        """
        for callback in self._sync_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in sync callback: {e}")
    
    def get_events_in_window(
        self,
        start_time: float,
        end_time: float,
        event_types: Optional[List[ChatEventType]] = None
    ) -> List[SyncEvent]:
        """Get events within a time window.
        
        Args:
            start_time: Start time in seconds since stream start
            end_time: End time in seconds since stream start
            event_types: Optional list of event types to filter
            
        Returns:
            List of synchronized events in the window
        """
        events = []
        
        for event in self._event_buffer:
            if start_time <= event.stream_time <= end_time:
                if event_types is None or (
                    event.chat_event and event.chat_event.type in event_types
                ):
                    events.append(event)
        
        return sorted(events, key=lambda e: e.stream_time)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get synchronizer metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "stream_start_time": self.stream_start_time.isoformat() if self.stream_start_time else None,
            "stream_current_time": self.stream_current_time,
            "buffer_size": len(self._event_buffer),
            "events_synced": self._events_synced,
            "events_dropped": self._events_dropped,
            "stream_connected": self.stream_adapter.is_connected,
            "chat_connected": self.chat_adapter.is_connected,
        }


class MultiStreamSynchronizer:
    """Manages multiple stream-chat synchronizers for different platforms."""
    
    def __init__(self):
        """Initialize the multi-stream synchronizer."""
        self._synchronizers: Dict[str, StreamChatSynchronizer] = {}
        self._global_callbacks: List[Callable] = []
        
        logger.info("Initialized multi-stream synchronizer")
    
    async def add_stream(
        self,
        stream_id: str,
        stream_adapter: BaseStreamAdapter,
        chat_adapter: BaseChatAdapter,
        **kwargs
    ) -> bool:
        """Add a stream with chat synchronization.
        
        Args:
            stream_id: Unique identifier for the stream
            stream_adapter: Stream adapter instance
            chat_adapter: Chat adapter instance
            **kwargs: Additional synchronizer configuration
            
        Returns:
            bool: True if added successfully
        """
        if stream_id in self._synchronizers:
            logger.warning(f"Stream {stream_id} already exists")
            return False
        
        try:
            # Create synchronizer
            sync = StreamChatSynchronizer(
                stream_adapter,
                chat_adapter,
                **kwargs
            )
            
            # Register global callbacks
            for callback in self._global_callbacks:
                sync.on_sync_event(callback)
            
            # Start synchronizer
            await sync.start()
            
            self._synchronizers[stream_id] = sync
            logger.info(f"Added stream {stream_id} with chat synchronization")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add stream {stream_id}: {e}")
            return False
    
    async def remove_stream(self, stream_id: str) -> bool:
        """Remove a stream and its synchronizer.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            bool: True if removed successfully
        """
        if stream_id not in self._synchronizers:
            return False
        
        try:
            sync = self._synchronizers[stream_id]
            await sync.stop()
            del self._synchronizers[stream_id]
            logger.info(f"Removed stream {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing stream {stream_id}: {e}")
            return False
    
    def on_sync_event(self, callback: Callable) -> None:
        """Register a global callback for all synchronized events.
        
        Args:
            callback: Function to call with SyncEvent
        """
        self._global_callbacks.append(callback)
        
        # Register with existing synchronizers
        for sync in self._synchronizers.values():
            sync.on_sync_event(callback)
    
    def get_synchronizer(self, stream_id: str) -> Optional[StreamChatSynchronizer]:
        """Get a specific stream synchronizer.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            StreamChatSynchronizer or None
        """
        return self._synchronizers.get(stream_id)
    
    async def stop_all(self) -> None:
        """Stop all synchronizers."""
        logger.info("Stopping all stream synchronizers")
        
        tasks = []
        for stream_id, sync in self._synchronizers.items():
            tasks.append(sync.stop())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self._synchronizers.clear()
        logger.info("All synchronizers stopped")