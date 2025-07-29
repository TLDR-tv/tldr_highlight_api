"""
Comprehensive chat-video timeline synchronization system for the TL;DR Highlight API.

This module provides timeline synchronization between chat events and video segments,
handling platform-specific timestamp differences, event buffering, and highlight generation
based on synchronized chat sentiment analysis.
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Deque, AsyncIterator, Callable, TYPE_CHECKING
import numpy as np
from scipy import signal, interpolate

from .base import ChatEvent, ChatEventType, ChatMessage
from .sentiment_analyzer import (
    SentimentAnalyzer, 
    WindowSentiment, 
    VelocityMetrics,
    SentimentCategory
)
from ...utils.frame_synchronizer import FrameSynchronizer, TimestampFormat
from ...utils.video_buffer import VideoFrame
from ...utils.scoring_utils import normalize_score, calculate_confidence

if TYPE_CHECKING:
    from ..content_processing.stream_buffer_manager import StreamBufferManager

logger = logging.getLogger(__name__)


class StreamType(str, Enum):
    """Stream type identifiers (copied from stream_buffer_manager to avoid circular import)."""
    
    YOUTUBE_HLS = "youtube_hls"
    TWITCH_HLS = "twitch_hls"
    RTMP_FLV = "rtmp_flv"
    GENERIC_HLS = "generic_hls"
    RAW_VIDEO = "raw_video"


class SyncStrategy(str, Enum):
    """Timeline synchronization strategies."""
    EXACT = "exact"                    # Exact timestamp matching
    INTERPOLATED = "interpolated"      # Interpolate between known sync points
    OFFSET_BASED = "offset_based"      # Fixed offset between chat and video
    ADAPTIVE = "adaptive"              # Adaptive sync based on correlation
    HYBRID = "hybrid"                  # Combination of strategies


class ChatSourceType(str, Enum):
    """Types of chat sources."""
    TWITCH_EVENTSUB = "twitch_eventsub"
    YOUTUBE_LIVE = "youtube_live"
    YOUTUBE_VOD = "youtube_vod"
    GENERIC_WEBSOCKET = "generic_websocket"
    REPLAY_FILE = "replay_file"


@dataclass
class TimestampOffset:
    """Represents offset between chat and video timestamps."""
    offset_seconds: float              # Offset in seconds (chat_time = video_time + offset)
    confidence: float                  # Confidence in the offset (0.0 to 1.0)
    drift_rate: float = 0.0           # Drift rate in seconds per second
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    samples: int = 0                   # Number of samples used to calculate offset
    
    def apply(self, timestamp: float, current_time: Optional[datetime] = None) -> float:
        """Apply offset to a timestamp."""
        adjusted = timestamp + self.offset_seconds
        
        # Apply drift correction if available
        if self.drift_rate != 0.0 and current_time:
            elapsed = (current_time - self.last_update).total_seconds()
            drift_correction = self.drift_rate * elapsed
            adjusted += drift_correction
        
        return adjusted


@dataclass
class ChatEventBuffer:
    """Buffer for chat events with timing information."""
    event: ChatEvent
    video_timestamp: Optional[float] = None  # Corresponding video timestamp
    sync_confidence: float = 0.0              # Confidence in synchronization
    processed: bool = False                   # Whether event has been processed
    
    @property
    def is_synchronized(self) -> bool:
        """Check if event is synchronized with video."""
        return self.video_timestamp is not None and self.sync_confidence > 0.5


@dataclass
class SyncPoint:
    """A synchronization point between chat and video."""
    chat_timestamp: float
    video_timestamp: float
    confidence: float
    event_type: Optional[ChatEventType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HighlightCandidate:
    """A potential highlight detected from chat sentiment."""
    start_timestamp: float            # Video timestamp start
    end_timestamp: float              # Video timestamp end
    confidence: float                 # Confidence score (0.0 to 1.0)
    sentiment_score: float            # Average sentiment score
    intensity: float                  # Sentiment intensity
    category: SentimentCategory       # Dominant sentiment category
    chat_events: List[ChatEvent]      # Associated chat events
    velocity_metrics: VelocityMetrics # Chat velocity during highlight
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get highlight duration in seconds."""
        return self.end_timestamp - self.start_timestamp


class TimelineSynchronizer:
    """
    Synchronizes chat events with video timeline, handling platform-specific
    timestamp differences and generating highlight candidates.
    """
    
    def __init__(self,
                 stream_id: str,
                 chat_source: ChatSourceType,
                 video_type: StreamType,
                 buffer_manager: Optional["StreamBufferManager"] = None,
                 frame_synchronizer: Optional[FrameSynchronizer] = None,
                 sentiment_analyzer: Optional[SentimentAnalyzer] = None,
                 sync_strategy: SyncStrategy = SyncStrategy.HYBRID):
        """
        Initialize the timeline synchronizer.
        
        Args:
            stream_id: Unique stream identifier
            chat_source: Type of chat source
            video_type: Type of video stream
            buffer_manager: Stream buffer manager instance
            frame_synchronizer: Frame synchronizer instance
            sentiment_analyzer: Sentiment analyzer instance
            sync_strategy: Synchronization strategy to use
        """
        self.stream_id = stream_id
        self.chat_source = chat_source
        self.video_type = video_type
        self.sync_strategy = sync_strategy
        
        # Use provided instances or defaults
        self.buffer_manager = buffer_manager
        if self.buffer_manager is None:
            # Lazy import to avoid circular dependency
            from ..content_processing.stream_buffer_manager import StreamBufferManager
            self.buffer_manager = StreamBufferManager()
        
        self.frame_synchronizer = frame_synchronizer or FrameSynchronizer()
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        
        # Event buffering
        self.event_buffer: Deque[ChatEventBuffer] = deque(maxlen=10000)
        self.message_buffer: Deque[ChatMessage] = deque(maxlen=5000)
        
        # Synchronization state
        self.sync_points: List[SyncPoint] = []
        self.timestamp_offset = TimestampOffset(0.0, 0.0)
        self.is_synchronized = False
        
        # Highlight detection
        self.highlight_candidates: List[HighlightCandidate] = []
        self.active_highlight: Optional[Dict[str, Any]] = None
        
        # Configuration
        self.config = self._load_config()
        
        # Callbacks
        self.highlight_callbacks: List[Callable[[HighlightCandidate], None]] = []
        
        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._highlight_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized TimelineSynchronizer for stream {stream_id}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration for timeline synchronization."""
        return {
            # Synchronization settings
            "sync_window_ms": 500,           # Window for timestamp matching
            "min_sync_confidence": 0.7,      # Minimum confidence for sync
            "sync_point_interval": 30.0,     # Seconds between sync points
            "max_drift_rate": 0.01,          # Max acceptable drift (s/s)
            
            # Buffer settings
            "event_buffer_window": 60.0,     # Seconds of events to buffer
            "correlation_window": 10.0,      # Window for correlation analysis
            
            # Highlight detection
            "highlight_min_duration": 5.0,   # Minimum highlight duration
            "highlight_max_duration": 60.0,  # Maximum highlight duration
            "highlight_merge_gap": 3.0,      # Gap to merge highlights
            "highlight_confidence_threshold": 0.6,
            
            # Sentiment thresholds
            "sentiment_spike_threshold": 0.7,
            "velocity_spike_multiplier": 2.0,
            "event_impact_threshold": 0.5,
        }
    
    async def start(self):
        """Start the timeline synchronizer."""
        logger.info(f"Starting timeline synchronizer for stream {self.stream_id}")
        
        # Register stream with frame synchronizer
        timestamp_format = self._get_timestamp_format()
        self.frame_synchronizer.register_stream(
            f"{self.stream_id}_chat",
            self.video_type,
            timestamp_format,
            is_reference=False
        )
        
        # Start background tasks
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._highlight_task = asyncio.create_task(self._highlight_detection_loop())
        
        # Perform initial synchronization
        await self._initial_sync()
    
    async def stop(self):
        """Stop the timeline synchronizer."""
        logger.info(f"Stopping timeline synchronizer for stream {self.stream_id}")
        
        # Cancel background tasks
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        if self._highlight_task:
            self._highlight_task.cancel()
            try:
                await self._highlight_task
            except asyncio.CancelledError:
                pass
    
    async def add_chat_event(self, event: ChatEvent):
        """
        Add a chat event for synchronization.
        
        Args:
            event: Chat event to add
        """
        # Create buffered event
        buffered_event = ChatEventBuffer(event=event)
        
        # Apply initial timestamp conversion
        if self.is_synchronized:
            video_timestamp = await self._convert_to_video_timestamp(event.timestamp)
            buffered_event.video_timestamp = video_timestamp
            buffered_event.sync_confidence = self.timestamp_offset.confidence
        
        # Add to buffer
        self.event_buffer.append(buffered_event)
        
        # If it's a message, add to message buffer
        if event.type == ChatEventType.MESSAGE and event.message:
            self.message_buffer.append(event.message)
        
        # Process special events for sync points
        if event.type in [ChatEventType.STREAM_ONLINE, ChatEventType.STREAM_OFFLINE]:
            await self._create_sync_point_from_event(event)
        
        # Process event for sentiment analysis
        await self.sentiment_analyzer.process_event(event)
    
    async def _convert_to_video_timestamp(self, chat_timestamp: datetime) -> float:
        """Convert chat timestamp to video timestamp."""
        # Convert datetime to float if needed
        if isinstance(chat_timestamp, datetime):
            chat_ts = chat_timestamp.timestamp()
        else:
            chat_ts = float(chat_timestamp)
        
        # Apply offset
        video_ts = self.timestamp_offset.apply(chat_ts, datetime.now(timezone.utc))
        
        # Normalize through frame synchronizer
        normalized_ts = self.frame_synchronizer.normalize_timestamp(
            f"{self.stream_id}_chat",
            video_ts,
            TimestampFormat.EPOCH_SECONDS
        )
        
        return normalized_ts
    
    def _get_timestamp_format(self) -> TimestampFormat:
        """Get timestamp format for chat source."""
        format_map = {
            ChatSourceType.TWITCH_EVENTSUB: TimestampFormat.EPOCH_SECONDS,
            ChatSourceType.YOUTUBE_LIVE: TimestampFormat.EPOCH_SECONDS,
            ChatSourceType.YOUTUBE_VOD: TimestampFormat.RELATIVE_SECONDS,
            ChatSourceType.GENERIC_WEBSOCKET: TimestampFormat.EPOCH_SECONDS,
            ChatSourceType.REPLAY_FILE: TimestampFormat.RELATIVE_SECONDS,
        }
        return format_map.get(self.chat_source, TimestampFormat.EPOCH_SECONDS)
    
    async def _initial_sync(self):
        """Perform initial synchronization between chat and video."""
        logger.info("Performing initial synchronization")
        
        if self.sync_strategy == SyncStrategy.OFFSET_BASED:
            # Use a fixed offset based on platform
            offset = self._get_platform_offset()
            self.timestamp_offset = TimestampOffset(offset, 0.8)
            self.is_synchronized = True
            
        elif self.sync_strategy in [SyncStrategy.ADAPTIVE, SyncStrategy.HYBRID]:
            # Try to detect offset automatically
            await self._detect_offset()
    
    def _get_platform_offset(self) -> float:
        """Get platform-specific timestamp offset."""
        # Platform-specific offsets (in seconds)
        offsets = {
            (ChatSourceType.TWITCH_EVENTSUB, StreamType.TWITCH_HLS): -2.0,  # Twitch chat leads
            (ChatSourceType.YOUTUBE_LIVE, StreamType.YOUTUBE_HLS): -1.5,    # YouTube chat leads
            (ChatSourceType.YOUTUBE_VOD, StreamType.YOUTUBE_HLS): 0.0,      # VOD synced
        }
        
        key = (self.chat_source, self.video_type)
        return offsets.get(key, 0.0)
    
    async def _detect_offset(self):
        """Automatically detect timestamp offset using correlation."""
        logger.info("Detecting timestamp offset through correlation")
        
        # Wait for sufficient data
        await asyncio.sleep(10.0)
        
        # Get recent chat velocity
        chat_signal = await self._get_chat_velocity_signal()
        if len(chat_signal) < 10:
            logger.warning("Insufficient chat data for offset detection")
            return
        
        # Get video activity signal (placeholder - would analyze video frames)
        video_signal = await self._get_video_activity_signal()
        if len(video_signal) < 10:
            logger.warning("Insufficient video data for offset detection")
            return
        
        # Compute cross-correlation
        correlation = signal.correlate(chat_signal, video_signal, mode='same')
        
        # Find peak correlation
        peak_idx = np.argmax(correlation)
        offset_samples = peak_idx - len(correlation) // 2
        offset_seconds = offset_samples * self.config["correlation_window"] / len(chat_signal)
        
        # Calculate confidence based on correlation strength
        max_corr = correlation[peak_idx]
        avg_corr = np.mean(np.abs(correlation))
        confidence = min(max_corr / (avg_corr * 3), 1.0) if avg_corr > 0 else 0.0
        
        if confidence > self.config["min_sync_confidence"]:
            self.timestamp_offset = TimestampOffset(offset_seconds, confidence)
            self.is_synchronized = True
            logger.info(f"Detected offset: {offset_seconds:.2f}s (confidence: {confidence:.2f})")
        else:
            logger.warning(f"Low confidence offset detection: {confidence:.2f}")
    
    async def _get_chat_velocity_signal(self) -> np.ndarray:
        """Get chat velocity signal for correlation."""
        # Count messages in time bins
        bin_size = 1.0  # 1 second bins
        current_time = datetime.now(timezone.utc)
        
        # Create time bins
        num_bins = int(self.config["correlation_window"] / bin_size)
        bins = np.zeros(num_bins)
        
        for msg in self.message_buffer:
            time_diff = (current_time - msg.timestamp).total_seconds()
            if 0 <= time_diff < self.config["correlation_window"]:
                bin_idx = int(time_diff / bin_size)
                if 0 <= bin_idx < num_bins:
                    bins[num_bins - 1 - bin_idx] += 1
        
        # Smooth signal
        if len(bins) > 5:
            bins = signal.savgol_filter(bins, 5, 2)
        
        return bins
    
    async def _get_video_activity_signal(self) -> np.ndarray:
        """Get video activity signal for correlation.
        
        This method should analyze video frames for activity/motion detection
        to correlate with chat activity patterns.
        
        Returns:
            np.ndarray: Video activity signal array
            
        Raises:
            NotImplementedError: Video activity analysis not yet implemented
        """
        raise NotImplementedError(
            "Video activity signal extraction is not yet implemented. "
            "This requires frame-by-frame analysis for motion/scene detection."
        )
    
    async def _create_sync_point_from_event(self, event: ChatEvent):
        """Create a synchronization point from a special event."""
        # Convert chat timestamp
        chat_ts = event.timestamp.timestamp() if isinstance(event.timestamp, datetime) else event.timestamp
        
        # Get current video timestamp (if available)
        video_ts = None
        confidence = 0.0
        
        # For stream start/end events, we can be more confident
        if event.type in [ChatEventType.STREAM_ONLINE, ChatEventType.STREAM_OFFLINE]:
            # These events usually align well with video
            video_ts = chat_ts + self._get_platform_offset()
            confidence = 0.9
        
        if video_ts is not None:
            sync_point = SyncPoint(
                chat_timestamp=chat_ts,
                video_timestamp=video_ts,
                confidence=confidence,
                event_type=event.type,
                metadata={"event_id": event.id}
            )
            self.sync_points.append(sync_point)
            
            # Update offset if this is a high-confidence sync point
            if confidence > self.config["min_sync_confidence"]:
                await self._update_offset_from_sync_points()
    
    async def _update_offset_from_sync_points(self):
        """Update timestamp offset based on sync points."""
        if len(self.sync_points) < 2:
            return
        
        # Get recent high-confidence sync points
        recent_points = [
            sp for sp in self.sync_points[-20:]
            if sp.confidence > self.config["min_sync_confidence"]
        ]
        
        if len(recent_points) < 2:
            return
        
        # Calculate offsets
        offsets = [sp.video_timestamp - sp.chat_timestamp for sp in recent_points]
        weights = [sp.confidence for sp in recent_points]
        
        # Weighted average offset
        avg_offset = np.average(offsets, weights=weights)
        
        # Calculate drift if we have enough time span
        if len(recent_points) > 5:
            times = [sp.chat_timestamp for sp in recent_points]
            # Linear regression for drift
            coeffs = np.polyfit(times, offsets, 1)
            drift_rate = coeffs[0]
        else:
            drift_rate = 0.0
        
        # Update offset
        confidence = np.mean(weights)
        self.timestamp_offset = TimestampOffset(
            offset_seconds=avg_offset,
            confidence=confidence,
            drift_rate=drift_rate,
            samples=len(recent_points)
        )
        
        self.is_synchronized = True
    
    async def _sync_loop(self):
        """Background task for continuous synchronization."""
        while True:
            try:
                await asyncio.sleep(self.config["sync_point_interval"])
                
                if self.sync_strategy in [SyncStrategy.ADAPTIVE, SyncStrategy.HYBRID]:
                    # Update synchronization
                    await self._update_offset_from_sync_points()
                    
                    # Check for drift
                    if abs(self.timestamp_offset.drift_rate) > self.config["max_drift_rate"]:
                        logger.warning(f"High drift rate detected: {self.timestamp_offset.drift_rate:.6f} s/s")
                
                # Clean old events
                await self._cleanup_old_events()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
    
    async def _cleanup_old_events(self):
        """Remove old events from buffers."""
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(seconds=self.config["event_buffer_window"])
        
        # Clean event buffer
        while self.event_buffer and self.event_buffer[0].event.timestamp < cutoff_time:
            self.event_buffer.popleft()
        
        # Clean message buffer
        while self.message_buffer and self.message_buffer[0].timestamp < cutoff_time:
            self.message_buffer.popleft()
        
        # Clean old sync points (keep last 100)
        if len(self.sync_points) > 100:
            self.sync_points = self.sync_points[-50:]
        
        # Clean old highlight candidates (keep last 50)
        if len(self.highlight_candidates) > 50:
            self.highlight_candidates = self.highlight_candidates[-25:]
    
    async def _highlight_detection_loop(self):
        """Background task for highlight detection."""
        while True:
            try:
                await asyncio.sleep(1.0)  # Check every second
                
                if not self.is_synchronized:
                    continue
                
                # Analyze recent chat window
                await self._analyze_chat_window()
                
                # Check for highlight end
                if self.active_highlight:
                    await self._check_highlight_end()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in highlight detection: {e}")
    
    async def _analyze_chat_window(self):
        """Analyze recent chat for highlight potential."""
        current_time = datetime.now(timezone.utc)
        window_start = current_time - timedelta(seconds=self.config["correlation_window"])
        
        # Get messages in window
        window_messages = [
            msg for msg in self.message_buffer
            if window_start <= msg.timestamp <= current_time
        ]
        
        if len(window_messages) < 5:  # Need minimum messages
            return
        
        # Convert to ChatMessage objects if needed
        chat_messages = []
        for msg in window_messages:
            if hasattr(msg, 'text'):  # Already a ChatMessage
                chat_messages.append(msg)
        
        if not chat_messages:
            return
        
        # Analyze sentiment
        window_sentiment = await self.sentiment_analyzer.analyze_window(
            chat_messages,
            window_start,
            current_time,
            self.chat_source.value
        )
        
        # Get velocity metrics
        velocity = self.sentiment_analyzer.temporal_analyzer.calculate_velocity_metrics(current_time)
        
        # Get event impact
        event_impact = self.sentiment_analyzer.event_tracker.get_current_impact(current_time)
        
        # Calculate highlight confidence
        highlight_confidence = self.sentiment_analyzer.get_highlight_confidence(
            window_sentiment,
            velocity,
            event_impact
        )
        
        # Check if we should start a highlight
        if highlight_confidence > self.config["highlight_confidence_threshold"]:
            if not self.active_highlight:
                await self._start_highlight(
                    window_sentiment,
                    velocity,
                    highlight_confidence,
                    current_time
                )
            else:
                # Update active highlight
                self.active_highlight["end_time"] = current_time
                self.active_highlight["events"].extend(chat_messages)
                self.active_highlight["confidence"] = max(
                    self.active_highlight["confidence"],
                    highlight_confidence
                )
    
    async def _start_highlight(self, 
                              sentiment: WindowSentiment,
                              velocity: VelocityMetrics,
                              confidence: float,
                              timestamp: datetime):
        """Start tracking a new highlight."""
        logger.info(f"Starting highlight tracking (confidence: {confidence:.2f})")
        
        # Convert to video timestamp
        video_timestamp = await self._convert_to_video_timestamp(timestamp)
        
        self.active_highlight = {
            "start_time": timestamp,
            "end_time": timestamp,
            "video_start": video_timestamp,
            "sentiment": sentiment,
            "velocity": velocity,
            "confidence": confidence,
            "events": list(self.message_buffer)[-20:],  # Recent messages
            "peak_confidence": confidence,
        }
    
    async def _check_highlight_end(self):
        """Check if active highlight should end."""
        if not self.active_highlight:
            return
        
        current_time = datetime.now(timezone.utc)
        duration = (current_time - self.active_highlight["start_time"]).total_seconds()
        
        # Force end if too long
        if duration > self.config["highlight_max_duration"]:
            await self._end_highlight()
            return
        
        # Check if activity has decreased
        recent_confidence = 0.0
        if len(self.sentiment_analyzer.window_history) > 0:
            recent_window = self.sentiment_analyzer.window_history[-1]
            recent_velocity = self.sentiment_analyzer.temporal_analyzer.velocity_history[-1] if hasattr(self.sentiment_analyzer.temporal_analyzer, 'velocity_history') and self.sentiment_analyzer.temporal_analyzer.velocity_history else None
            
            if recent_velocity:
                recent_confidence = self.sentiment_analyzer.get_highlight_confidence(
                    recent_window,
                    recent_velocity,
                    0.0
                )
        
        # End if confidence dropped significantly
        if recent_confidence < self.config["highlight_confidence_threshold"] * 0.5:
            time_since_peak = (current_time - self.active_highlight["end_time"]).total_seconds()
            if time_since_peak > self.config["highlight_merge_gap"]:
                await self._end_highlight()
    
    async def _end_highlight(self):
        """End active highlight and create candidate."""
        if not self.active_highlight:
            return
        
        logger.info("Ending highlight tracking")
        
        # Calculate video timestamps
        video_start = self.active_highlight["video_start"]
        video_end = await self._convert_to_video_timestamp(self.active_highlight["end_time"])
        
        # Create highlight candidate
        candidate = HighlightCandidate(
            start_timestamp=video_start,
            end_timestamp=video_end,
            confidence=self.active_highlight["peak_confidence"],
            sentiment_score=self.active_highlight["sentiment"].avg_sentiment,
            intensity=self.active_highlight["sentiment"].intensity,
            category=self.active_highlight["sentiment"].dominant_category,
            chat_events=self.active_highlight["events"],
            velocity_metrics=self.active_highlight["velocity"],
            metadata={
                "message_count": len(self.active_highlight["events"]),
                "unique_users": self.active_highlight["sentiment"].unique_users,
                "momentum": self.active_highlight["sentiment"].momentum,
            }
        )
        
        # Check minimum duration
        if candidate.duration >= self.config["highlight_min_duration"]:
            # Try to merge with recent highlights
            merged = await self._try_merge_highlight(candidate)
            
            if not merged:
                self.highlight_candidates.append(candidate)
                
                # Notify callbacks
                for callback in self.highlight_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(candidate)
                        else:
                            callback(candidate)
                    except Exception as e:
                        logger.error(f"Error in highlight callback: {e}")
        
        self.active_highlight = None
    
    async def _try_merge_highlight(self, candidate: HighlightCandidate) -> bool:
        """Try to merge candidate with recent highlights."""
        if not self.highlight_candidates:
            return False
        
        # Check last highlight
        last_highlight = self.highlight_candidates[-1]
        gap = candidate.start_timestamp - last_highlight.end_timestamp
        
        if 0 < gap <= self.config["highlight_merge_gap"]:
            # Merge highlights
            logger.info("Merging highlights")
            
            last_highlight.end_timestamp = candidate.end_timestamp
            last_highlight.confidence = max(last_highlight.confidence, candidate.confidence)
            last_highlight.chat_events.extend(candidate.chat_events)
            
            # Update metadata
            last_highlight.metadata["message_count"] += candidate.metadata["message_count"]
            last_highlight.metadata["unique_users"] = max(
                last_highlight.metadata["unique_users"],
                candidate.metadata["unique_users"]
            )
            
            return True
        
        return False
    
    def add_highlight_callback(self, callback: Callable[[HighlightCandidate], None]):
        """Add a callback for highlight detection."""
        self.highlight_callbacks.append(callback)
    
    async def get_synchronized_events(self, 
                                    start_timestamp: float,
                                    end_timestamp: float) -> List[ChatEventBuffer]:
        """
        Get chat events synchronized to video timestamps.
        
        Args:
            start_timestamp: Start video timestamp
            end_timestamp: End video timestamp
            
        Returns:
            List of synchronized chat events
        """
        synchronized_events = []
        
        for buffered_event in self.event_buffer:
            if buffered_event.is_synchronized:
                if start_timestamp <= buffered_event.video_timestamp <= end_timestamp:
                    synchronized_events.append(buffered_event)
        
        return synchronized_events
    
    async def get_highlights(self, 
                           min_confidence: Optional[float] = None,
                           min_duration: Optional[float] = None) -> List[HighlightCandidate]:
        """
        Get detected highlights with optional filtering.
        
        Args:
            min_confidence: Minimum confidence threshold
            min_duration: Minimum duration in seconds
            
        Returns:
            Filtered list of highlight candidates
        """
        highlights = self.highlight_candidates.copy()
        
        if min_confidence is not None:
            highlights = [h for h in highlights if h.confidence >= min_confidence]
        
        if min_duration is not None:
            highlights = [h for h in highlights if h.duration >= min_duration]
        
        # Sort by confidence
        highlights.sort(key=lambda h: h.confidence, reverse=True)
        
        return highlights
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status."""
        return {
            "stream_id": self.stream_id,
            "is_synchronized": self.is_synchronized,
            "sync_strategy": self.sync_strategy.value,
            "timestamp_offset": {
                "offset_seconds": self.timestamp_offset.offset_seconds,
                "confidence": self.timestamp_offset.confidence,
                "drift_rate": self.timestamp_offset.drift_rate,
                "samples": self.timestamp_offset.samples,
            },
            "sync_points": len(self.sync_points),
            "buffered_events": len(self.event_buffer),
            "buffered_messages": len(self.message_buffer),
            "active_highlight": self.active_highlight is not None,
            "detected_highlights": len(self.highlight_candidates),
        }


class MultiStreamTimelineSynchronizer:
    """
    Manages timeline synchronization across multiple streams,
    enabling multi-perspective highlight detection.
    """
    
    def __init__(self,
                 buffer_manager: Optional["StreamBufferManager"] = None,
                 frame_synchronizer: Optional[FrameSynchronizer] = None):
        """Initialize multi-stream timeline synchronizer."""
        self.buffer_manager = buffer_manager
        if self.buffer_manager is None:
            # Lazy import to avoid circular dependency
            from ..content_processing.stream_buffer_manager import StreamBufferManager
            self.buffer_manager = StreamBufferManager()
        
        self.frame_synchronizer = frame_synchronizer or FrameSynchronizer()
        
        # Stream synchronizers
        self.stream_synchronizers: Dict[str, TimelineSynchronizer] = {}
        
        # Multi-stream highlights
        self.multi_stream_highlights: List[Dict[str, Any]] = []
        
        # Configuration
        self.config = {
            "correlation_threshold": 0.7,    # Correlation threshold for multi-stream
            "time_tolerance": 5.0,           # Time tolerance for matching highlights
            "min_streams": 2,                # Minimum streams for multi-stream highlight
        }
        
        logger.info("Initialized MultiStreamTimelineSynchronizer")
    
    async def add_stream(self,
                        stream_id: str,
                        chat_source: ChatSourceType,
                        video_type: StreamType,
                        sync_strategy: SyncStrategy = SyncStrategy.HYBRID) -> TimelineSynchronizer:
        """Add a stream for synchronization."""
        if stream_id in self.stream_synchronizers:
            logger.warning(f"Stream {stream_id} already exists")
            return self.stream_synchronizers[stream_id]
        
        # Create synchronizer
        synchronizer = TimelineSynchronizer(
            stream_id=stream_id,
            chat_source=chat_source,
            video_type=video_type,
            buffer_manager=self.buffer_manager,
            frame_synchronizer=self.frame_synchronizer,
            sync_strategy=sync_strategy
        )
        
        # Add highlight callback
        synchronizer.add_highlight_callback(
            lambda h: asyncio.create_task(self._on_stream_highlight(stream_id, h))
        )
        
        # Start synchronizer
        await synchronizer.start()
        
        self.stream_synchronizers[stream_id] = synchronizer
        logger.info(f"Added stream {stream_id} to multi-stream synchronizer")
        
        return synchronizer
    
    async def remove_stream(self, stream_id: str):
        """Remove a stream from synchronization."""
        if stream_id not in self.stream_synchronizers:
            return
        
        synchronizer = self.stream_synchronizers[stream_id]
        await synchronizer.stop()
        
        del self.stream_synchronizers[stream_id]
        logger.info(f"Removed stream {stream_id} from multi-stream synchronizer")
    
    async def _on_stream_highlight(self, stream_id: str, highlight: HighlightCandidate):
        """Handle highlight detection from a stream."""
        # Check for correlating highlights in other streams
        correlating_highlights = await self._find_correlating_highlights(
            stream_id,
            highlight
        )
        
        if len(correlating_highlights) >= self.config["min_streams"] - 1:
            # Create multi-stream highlight
            multi_highlight = {
                "id": f"multi_{len(self.multi_stream_highlights)}",
                "primary_stream": stream_id,
                "streams": {stream_id: highlight},
                "start_timestamp": highlight.start_timestamp,
                "end_timestamp": highlight.end_timestamp,
                "confidence": highlight.confidence,
                "created_at": datetime.now(timezone.utc),
            }
            
            # Add correlating highlights
            for other_stream_id, other_highlight in correlating_highlights.items():
                multi_highlight["streams"][other_stream_id] = other_highlight
                multi_highlight["start_timestamp"] = min(
                    multi_highlight["start_timestamp"],
                    other_highlight.start_timestamp
                )
                multi_highlight["end_timestamp"] = max(
                    multi_highlight["end_timestamp"],
                    other_highlight.end_timestamp
                )
                multi_highlight["confidence"] = max(
                    multi_highlight["confidence"],
                    other_highlight.confidence
                )
            
            self.multi_stream_highlights.append(multi_highlight)
            logger.info(
                f"Created multi-stream highlight with {len(multi_highlight['streams'])} streams"
            )
    
    async def _find_correlating_highlights(self,
                                         source_stream: str,
                                         source_highlight: HighlightCandidate) -> Dict[str, HighlightCandidate]:
        """Find highlights in other streams that correlate with source."""
        correlating = {}
        
        for stream_id, synchronizer in self.stream_synchronizers.items():
            if stream_id == source_stream:
                continue
            
            # Get highlights from other stream
            other_highlights = await synchronizer.get_highlights()
            
            for other_highlight in other_highlights:
                # Check temporal overlap
                overlap_start = max(
                    source_highlight.start_timestamp,
                    other_highlight.start_timestamp
                )
                overlap_end = min(
                    source_highlight.end_timestamp,
                    other_highlight.end_timestamp
                )
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    source_duration = source_highlight.duration
                    other_duration = other_highlight.duration
                    
                    # Calculate overlap ratio
                    overlap_ratio = overlap_duration / min(source_duration, other_duration)
                    
                    if overlap_ratio > self.config["correlation_threshold"]:
                        correlating[stream_id] = other_highlight
                        break
        
        return correlating
    
    async def get_multi_stream_highlights(self) -> List[Dict[str, Any]]:
        """Get all multi-stream highlights."""
        return self.multi_stream_highlights.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all stream synchronizers."""
        return {
            "active_streams": len(self.stream_synchronizers),
            "stream_status": {
                stream_id: sync.get_sync_status()
                for stream_id, sync in self.stream_synchronizers.items()
            },
            "multi_stream_highlights": len(self.multi_stream_highlights),
        }


# Global instances
timeline_synchronizer = None
multi_stream_synchronizer = None  # Will be initialized when first used