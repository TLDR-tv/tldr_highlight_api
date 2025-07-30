"""Frame synchronization and timestamp normalization for multi-format streams.

This module provides timestamp synchronization across different stream sources,
handling clock drift, format differences, and multi-modal alignment.
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Deque, Any

import numpy as np
from scipy import interpolate

from .video_buffer import VideoFrame, BufferFormat

logger = logging.getLogger(__name__)


class TimestampFormat(str, Enum):
    """Supported timestamp formats."""

    EPOCH_SECONDS = "epoch_seconds"  # Unix timestamp in seconds
    EPOCH_MILLIS = "epoch_millis"  # Unix timestamp in milliseconds
    PTS = "pts"  # Presentation timestamp
    DTS = "dts"  # Decode timestamp
    RELATIVE_SECONDS = "relative_seconds"  # Seconds from stream start
    HLS_TIMESTAMP = "hls_timestamp"  # HLS segment timestamp
    RTMP_TIMESTAMP = "rtmp_timestamp"  # RTMP message timestamp


@dataclass
class SyncConfig:
    """Configuration for frame synchronization."""

    # Synchronization parameters
    max_clock_drift_ms: float = 500.0  # Maximum acceptable clock drift
    sync_window_ms: float = 100.0  # Synchronization window size
    interpolation_method: str = "linear"  # Interpolation method for timestamps

    # Buffer sizes
    timestamp_history_size: int = (
        1000  # Number of timestamps to keep for drift calculation
    )
    sync_buffer_size: int = 100  # Number of frames to buffer for synchronization

    # Alignment settings
    enable_drift_correction: bool = True  # Enable automatic drift correction
    enable_interpolation: bool = True  # Enable timestamp interpolation
    enable_outlier_detection: bool = True  # Enable outlier detection and correction

    # Quality thresholds
    min_sync_confidence: float = 0.8  # Minimum confidence for sync decision
    outlier_threshold_sigma: float = 3.0  # Standard deviations for outlier detection

    # Performance tuning
    drift_check_interval_seconds: float = 10.0
    sync_update_interval_seconds: float = 1.0


@dataclass
class TimestampMapping:
    """Maps between different timestamp formats."""

    source_format: TimestampFormat
    target_format: TimestampFormat
    offset: float = 0.0  # Offset in target format units
    scale: float = 1.0  # Scale factor
    drift_rate: float = 0.0  # Drift rate (units per second)
    last_update: float = field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )

    def convert(self, timestamp: float, current_time: Optional[float] = None) -> float:
        """Convert timestamp from source to target format."""
        # Apply scale
        converted = timestamp * self.scale

        # Apply offset
        converted += self.offset

        # Apply drift correction if enabled
        if self.drift_rate != 0.0 and current_time is not None:
            time_elapsed = current_time - self.last_update
            drift_correction = self.drift_rate * time_elapsed
            converted += drift_correction

        return converted


@dataclass
class SyncPoint:
    """A synchronization point between multiple streams."""

    timestamp: float  # Reference timestamp
    stream_timestamps: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0  # Confidence in this sync point
    is_keyframe: bool = False  # Whether this is a keyframe sync point
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClockDriftDetector:
    """Detects and corrects clock drift between streams."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.timestamp_pairs: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self.drift_rate = 0.0
        self.last_calculation = 0.0

    def add_timestamp_pair(self, reference_time: float, stream_time: float) -> None:
        """Add a timestamp pair for drift calculation."""
        self.timestamp_pairs.append((reference_time, stream_time))

    def calculate_drift(self) -> float:
        """Calculate current drift rate."""
        if len(self.timestamp_pairs) < 10:
            return 0.0

        # Extract arrays
        ref_times = np.array([p[0] for p in self.timestamp_pairs])
        stream_times = np.array([p[1] for p in self.timestamp_pairs])

        # Calculate drift using linear regression
        coeffs = np.polyfit(ref_times, stream_times - ref_times, 1)
        self.drift_rate = coeffs[0]  # Slope represents drift rate

        return self.drift_rate

    def correct_timestamp(self, timestamp: float, elapsed_time: float) -> float:
        """Apply drift correction to a timestamp."""
        return timestamp - (self.drift_rate * elapsed_time)


class FrameSynchronizer:
    """Synchronizes frames across multiple stream sources.

    Features:
    - Multi-format timestamp normalization
    - Clock drift detection and correction
    - Frame alignment across streams
    - Interpolation for missing timestamps
    - Outlier detection and handling
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        """Initialize the frame synchronizer."""
        self.config = config or SyncConfig()

        # Stream tracking
        self._streams: Dict[str, Dict[str, Any]] = {}
        self._reference_stream: Optional[str] = None

        # Timestamp mappings
        self._mappings: Dict[Tuple[str, str], TimestampMapping] = {}

        # Synchronization state
        self._sync_points: List[SyncPoint] = []
        self._sync_buffer: Dict[str, Deque[VideoFrame]] = defaultdict(
            lambda: deque(maxlen=self.config.sync_buffer_size)
        )

        # Clock drift detectors
        self._drift_detectors: Dict[str, ClockDriftDetector] = {}

        # Timestamp history for analysis
        self._timestamp_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.timestamp_history_size)
        )

        # Performance metrics
        self._sync_stats = {
            "frames_synchronized": 0,
            "drift_corrections": 0,
            "outliers_detected": 0,
            "interpolations": 0,
            "sync_points_created": 0,
        }

        # Background tasks
        self._drift_check_task: Optional[asyncio.Task] = None
        self._sync_update_task: Optional[asyncio.Task] = None

        # Start background tasks
        self._start_background_tasks()

        logger.info("Initialized FrameSynchronizer")

    def register_stream(
        self,
        stream_id: str,
        format: BufferFormat,
        timestamp_format: TimestampFormat,
        base_timestamp: Optional[float] = None,
        is_reference: bool = False,
    ) -> None:
        """Register a stream for synchronization.

        Args:
            stream_id: Unique stream identifier
            format: Stream format (HLS, FLV, etc.)
            timestamp_format: Timestamp format used by the stream
            base_timestamp: Base timestamp for relative timestamps
            is_reference: Whether this is the reference stream for sync
        """
        self._streams[stream_id] = {
            "format": format,
            "timestamp_format": timestamp_format,
            "base_timestamp": base_timestamp or datetime.now(timezone.utc).timestamp(),
            "registered_at": datetime.now(timezone.utc).timestamp(),
            "frame_count": 0,
            "last_timestamp": None,
        }

        if is_reference or self._reference_stream is None:
            self._reference_stream = stream_id

        # Initialize drift detector
        self._drift_detectors[stream_id] = ClockDriftDetector(
            window_size=self.config.timestamp_history_size
        )

        logger.info(
            f"Registered stream {stream_id} (format: {format}, "
            f"timestamp: {timestamp_format}, reference: {is_reference})"
        )

    def normalize_timestamp(
        self,
        stream_id: str,
        timestamp: float,
        target_format: TimestampFormat = TimestampFormat.EPOCH_SECONDS,
    ) -> float:
        """Normalize a timestamp to the target format.

        Args:
            stream_id: Stream identifier
            timestamp: Original timestamp
            target_format: Target timestamp format

        Returns:
            Normalized timestamp
        """
        if stream_id not in self._streams:
            logger.warning(f"Unknown stream {stream_id}, returning original timestamp")
            return timestamp

        stream_info = self._streams[stream_id]
        source_format = stream_info["timestamp_format"]

        # Get or create mapping
        mapping_key = (source_format, target_format)
        if mapping_key not in self._mappings:
            self._create_mapping(source_format, target_format, stream_info)

        mapping = self._mappings[mapping_key]

        # Apply drift correction if enabled
        current_time = None
        if self.config.enable_drift_correction and stream_id in self._drift_detectors:
            current_time = datetime.now(timezone.utc).timestamp()

        normalized = mapping.convert(timestamp, current_time)

        # Update history
        self._timestamp_history[stream_id].append(normalized)
        stream_info["last_timestamp"] = normalized

        return normalized

    def _create_mapping(
        self,
        source: TimestampFormat,
        target: TimestampFormat,
        stream_info: Dict[str, Any],
    ) -> None:
        """Create a timestamp format mapping."""
        mapping = TimestampMapping(source_format=source, target_format=target)

        # Set scale based on format conversion
        if (
            source == TimestampFormat.EPOCH_MILLIS
            and target == TimestampFormat.EPOCH_SECONDS
        ):
            mapping.scale = 0.001
        elif (
            source == TimestampFormat.EPOCH_SECONDS
            and target == TimestampFormat.EPOCH_MILLIS
        ):
            mapping.scale = 1000.0
        elif (
            source == TimestampFormat.RELATIVE_SECONDS
            and target == TimestampFormat.EPOCH_SECONDS
        ):
            mapping.offset = stream_info["base_timestamp"]
        elif (
            source == TimestampFormat.RTMP_TIMESTAMP
            and target == TimestampFormat.EPOCH_SECONDS
        ):
            mapping.scale = 0.001  # RTMP timestamps are typically in milliseconds
            mapping.offset = stream_info["base_timestamp"]

        self._mappings[(source, target)] = mapping

    async def synchronize_frame(
        self, stream_id: str, frame: VideoFrame
    ) -> Tuple[VideoFrame, float]:
        """Synchronize a frame with other streams.

        Args:
            stream_id: Stream identifier
            frame: Frame to synchronize

        Returns:
            Tuple of (synchronized frame, sync confidence)
        """
        # Normalize timestamp
        normalized_timestamp = self.normalize_timestamp(
            stream_id, frame.timestamp, TimestampFormat.EPOCH_SECONDS
        )

        # Create synchronized frame
        sync_frame = VideoFrame(
            data=frame.data,
            timestamp=normalized_timestamp,
            duration=frame.duration,
            format=frame.format,
            frame_type=frame.frame_type,
            is_keyframe=frame.is_keyframe,
            width=frame.width,
            height=frame.height,
            fps=frame.fps,
            bitrate=frame.bitrate,
            codec=frame.codec,
            sequence_number=frame.sequence_number,
            pts=frame.pts,
            dts=frame.dts,
            quality_score=frame.quality_score,
            importance_score=frame.importance_score,
        )

        # Add to sync buffer
        self._sync_buffer[stream_id].append(sync_frame)

        # Update stream info
        stream_info = self._streams[stream_id]
        stream_info["frame_count"] += 1

        # Check for sync opportunities
        sync_confidence = await self._check_sync_opportunity(stream_id, sync_frame)

        # Update stats
        self._sync_stats["frames_synchronized"] += 1

        return sync_frame, sync_confidence

    async def _check_sync_opportunity(self, stream_id: str, frame: VideoFrame) -> float:
        """Check if this frame presents a synchronization opportunity.

        Args:
            stream_id: Stream identifier
            frame: Current frame

        Returns:
            Synchronization confidence (0.0 to 1.0)
        """
        confidence = 0.0

        # Keyframes are good sync points
        if frame.is_keyframe:
            confidence += 0.5

        # Check temporal alignment with other streams
        aligned_streams = 0
        total_streams = len(self._streams) - 1  # Exclude current stream

        if total_streams > 0:
            for other_stream_id, buffer in self._sync_buffer.items():
                if other_stream_id == stream_id or not buffer:
                    continue

                # Find temporally close frames
                close_frames = [
                    f
                    for f in buffer
                    if abs(f.timestamp - frame.timestamp)
                    < self.config.sync_window_ms / 1000.0
                ]

                if close_frames:
                    aligned_streams += 1

            alignment_ratio = aligned_streams / total_streams
            confidence += 0.5 * alignment_ratio

        # Create sync point if confidence is high enough
        if confidence >= self.config.min_sync_confidence:
            await self._create_sync_point(stream_id, frame, confidence)

        return confidence

    async def _create_sync_point(
        self, stream_id: str, frame: VideoFrame, confidence: float
    ) -> None:
        """Create a synchronization point."""
        sync_point = SyncPoint(
            timestamp=frame.timestamp,
            stream_timestamps={stream_id: frame.timestamp},
            confidence=confidence,
            is_keyframe=frame.is_keyframe,
        )

        # Add timestamps from other streams
        for other_stream_id, buffer in self._sync_buffer.items():
            if other_stream_id == stream_id or not buffer:
                continue

            # Find closest frame
            closest_frame = min(
                buffer, key=lambda f: abs(f.timestamp - frame.timestamp), default=None
            )

            if closest_frame:
                time_diff = abs(closest_frame.timestamp - frame.timestamp)
                if time_diff < self.config.sync_window_ms / 1000.0:
                    sync_point.stream_timestamps[other_stream_id] = (
                        closest_frame.timestamp
                    )

        self._sync_points.append(sync_point)
        self._sync_stats["sync_points_created"] += 1

        # Keep sync points list manageable
        if len(self._sync_points) > 1000:
            self._sync_points = self._sync_points[-500:]

        logger.debug(
            f"Created sync point at {frame.timestamp:.3f}s with "
            f"{len(sync_point.stream_timestamps)} streams (confidence: {confidence:.2f})"
        )

    def detect_outliers(self, stream_id: str) -> List[float]:
        """Detect timestamp outliers for a stream.

        Args:
            stream_id: Stream identifier

        Returns:
            List of outlier timestamps
        """
        if stream_id not in self._timestamp_history:
            return []

        timestamps = list(self._timestamp_history[stream_id])
        if len(timestamps) < 10:
            return []

        # Calculate timestamp differences
        diffs = np.diff(timestamps)

        # Detect outliers using z-score
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        outliers = []
        for i, diff in enumerate(diffs):
            z_score = abs((diff - mean_diff) / std_diff)
            if z_score > self.config.outlier_threshold_sigma:
                outliers.append(timestamps[i + 1])
                self._sync_stats["outliers_detected"] += 1

        return outliers

    def interpolate_timestamp(
        self, stream_id: str, target_time: float, method: Optional[str] = None
    ) -> Optional[float]:
        """Interpolate a timestamp for a stream.

        Args:
            stream_id: Stream identifier
            target_time: Target time to interpolate
            method: Interpolation method (default: config method)

        Returns:
            Interpolated timestamp or None if not possible
        """
        if not self.config.enable_interpolation:
            return None

        if stream_id not in self._timestamp_history:
            return None

        timestamps = list(self._timestamp_history[stream_id])
        if len(timestamps) < 2:
            return None

        # Create interpolation function
        indices = np.arange(len(timestamps))
        method = method or self.config.interpolation_method

        try:
            if method == "linear":
                f = interpolate.interp1d(
                    timestamps,
                    indices,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            elif method == "cubic":
                if len(timestamps) >= 4:
                    f = interpolate.interp1d(
                        timestamps,
                        indices,
                        kind="cubic",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                else:
                    f = interpolate.interp1d(
                        timestamps,
                        indices,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
            else:
                return None

            # Interpolate
            interpolated_index = f(target_time)

            # Reverse interpolate to get timestamp
            f_inverse = interpolate.interp1d(
                indices,
                timestamps,
                kind=method if method != "cubic" or len(timestamps) >= 4 else "linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            result = float(f_inverse(interpolated_index))
            self._sync_stats["interpolations"] += 1

            return result

        except Exception as e:
            logger.warning(f"Interpolation failed: {e}")
            return None

    async def align_frames(
        self, target_time: float, tolerance: float = 0.1
    ) -> Dict[str, Optional[VideoFrame]]:
        """Get aligned frames from all streams at a target time.

        Args:
            target_time: Target timestamp
            tolerance: Time tolerance in seconds

        Returns:
            Dictionary mapping stream IDs to frames (or None)
        """
        aligned_frames = {}

        for stream_id, buffer in self._sync_buffer.items():
            if not buffer:
                aligned_frames[stream_id] = None
                continue

            # Find closest frame
            closest_frame = None
            min_diff = float("inf")

            for frame in buffer:
                diff = abs(frame.timestamp - target_time)
                if diff < min_diff and diff <= tolerance:
                    min_diff = diff
                    closest_frame = frame

            aligned_frames[stream_id] = closest_frame

        return aligned_frames

    def _start_background_tasks(self) -> None:
        """Start background synchronization tasks."""
        try:
            loop = asyncio.get_running_loop()

            # Drift check task
            self._drift_check_task = loop.create_task(self._drift_check_loop())

            # Sync update task
            self._sync_update_task = loop.create_task(self._sync_update_loop())

        except RuntimeError:
            logger.debug("No event loop available for background tasks")

    async def _drift_check_loop(self) -> None:
        """Background task for drift checking."""
        while True:
            try:
                await asyncio.sleep(self.config.drift_check_interval_seconds)

                if not self.config.enable_drift_correction:
                    continue

                # Check drift for each stream
                for stream_id, detector in self._drift_detectors.items():
                    if stream_id == self._reference_stream:
                        continue

                    drift_rate = detector.calculate_drift()
                    if abs(drift_rate) > 0.001:  # Significant drift
                        logger.info(
                            f"Detected drift for stream {stream_id}: "
                            f"{drift_rate:.6f} seconds/second"
                        )
                        self._sync_stats["drift_corrections"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in drift check loop: {e}")

    async def _sync_update_loop(self) -> None:
        """Background task for sync updates."""
        while True:
            try:
                await asyncio.sleep(self.config.sync_update_interval_seconds)

                # Clean old sync points
                current_time = datetime.now(timezone.utc).timestamp()
                retention_time = 300.0  # 5 minutes

                self._sync_points = [
                    sp
                    for sp in self._sync_points
                    if current_time - sp.timestamp < retention_time
                ]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync update loop: {e}")

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics.

        Returns:
            Dictionary of statistics
        """
        stats = self._sync_stats.copy()
        stats.update(
            {
                "registered_streams": len(self._streams),
                "reference_stream": self._reference_stream,
                "active_sync_points": len(self._sync_points),
                "stream_details": {
                    stream_id: {
                        "format": info["format"].value,
                        "timestamp_format": info["timestamp_format"].value,
                        "frame_count": info["frame_count"],
                        "last_timestamp": info["last_timestamp"],
                    }
                    for stream_id, info in self._streams.items()
                },
            }
        )
        return stats

    async def close(self) -> None:
        """Clean up resources."""
        # Cancel background tasks
        if self._drift_check_task:
            self._drift_check_task.cancel()
            try:
                await self._drift_check_task
            except asyncio.CancelledError:
                pass

        if self._sync_update_task:
            self._sync_update_task.cancel()
            try:
                await self._sync_update_task
            except asyncio.CancelledError:
                pass

        logger.info("Closed FrameSynchronizer")


# Global synchronizer instance
frame_synchronizer = FrameSynchronizer()
