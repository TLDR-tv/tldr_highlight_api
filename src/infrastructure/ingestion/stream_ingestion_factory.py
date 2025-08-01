"""Stream ingestion factory for automatic format detection and processing.

This module provides factory methods for creating appropriate stream processors
based on stream URLs and formats. All streams are processed through FFmpeg.
"""

import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse

from .unified_ingestion_pipeline import StreamIngestionPipeline, IngestionConfig
from ..content_processing.gemini_processor import GeminiProcessorConfig

logger = logging.getLogger(__name__)


class StreamIngestionFactory:
    """Factory for creating stream ingestion pipelines."""

    @staticmethod
    def create_pipeline(
        stream_url: str, config_overrides: Optional[Dict[str, Any]] = None, **kwargs
    ) -> StreamIngestionPipeline:
        """Create an ingestion pipeline for a stream URL.

        All streams are processed through FFmpeg regardless of source.

        Args:
            stream_url: URL of the stream to process
            config_overrides: Optional configuration overrides
            **kwargs: Additional configuration parameters

        Returns:
            Configured stream ingestion pipeline
        """
        # Detect stream type from URL for logging purposes
        stream_type = StreamIngestionFactory.detect_stream_type(stream_url)

        # Create base configuration
        config = IngestionConfig(stream_url=stream_url, **kwargs)

        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Apply stream type optimizations
        StreamIngestionFactory._apply_stream_type_optimizations(config, stream_type)

        logger.info(f"Created ingestion pipeline for {stream_type} stream: {stream_url}")

        return StreamIngestionPipeline(config)

    @staticmethod
    def detect_stream_type(stream_url: str) -> str:
        """Detect stream type from URL.

        This is used only for logging and optimization hints,
        not for platform-specific processing.

        Args:
            stream_url: Stream URL to analyze

        Returns:
            Detected stream type
        """
        parsed = urlparse(stream_url)
        scheme = parsed.scheme.lower()
        path = parsed.path.lower()

        # Detect by protocol
        if scheme in ["rtmp", "rtmps"]:
            return "rtmp"
        elif ".m3u8" in path or "hls" in path:
            return "hls"
        elif ".mpd" in path or "dash" in path:
            return "dash"
        elif scheme in ["http", "https"]:
            # Check file extensions
            if any(ext in path for ext in [".mp4", ".webm", ".mov", ".avi"]):
                return "video_file"
            else:
                return "http_stream"
        elif scheme == "file":
            return "local_file"
        else:
            return "unknown"

    @staticmethod
    def _apply_stream_type_optimizations(
        config: IngestionConfig, stream_type: str
    ) -> None:
        """Apply stream type-specific optimizations.

        These are generic optimizations based on stream type,
        not platform-specific settings.

        Args:
            config: Configuration to optimize
            stream_type: Detected stream type
        """
        if stream_type == "rtmp":
            # RTMP streams often benefit from real-time processing
            config.enable_real_time = True
            config.segment_duration_seconds = 10.0
            config.frame_extraction_interval = 0.5

        elif stream_type == "hls":
            # HLS streams have their own segmentation
            config.segment_duration_seconds = 12.0
            config.frame_extraction_interval = 1.0
            config.hardware_acceleration = True

        elif stream_type == "dash":
            # DASH streams similar to HLS
            config.segment_duration_seconds = 12.0
            config.frame_extraction_interval = 1.0
            config.hardware_acceleration = True

        elif stream_type == "video_file":
            # Video files can be processed more aggressively
            config.enable_real_time = False
            config.segment_duration_seconds = 15.0
            config.frame_extraction_interval = 1.0
            config.hardware_acceleration = True

        elif stream_type == "local_file":
            # Local files have no network constraints
            config.enable_real_time = False
            config.segment_duration_seconds = 20.0
            config.frame_extraction_interval = 0.5
            config.hardware_acceleration = True

        # Log optimizations applied
        logger.debug(
            f"Applied {stream_type} optimizations: "
            f"real_time={config.enable_real_time}, "
            f"segment_duration={config.segment_duration_seconds}s, "
            f"frame_interval={config.frame_extraction_interval}s"
        )

    @classmethod
    def create_for_gaming_content(
        cls, stream_url: str, **kwargs
    ) -> StreamIngestionPipeline:
        """Create pipeline optimized for gaming content.

        Gaming content typically has:
        - Fast motion and scene changes
        - Important UI elements
        - Quick actions that matter

        Args:
            stream_url: URL of the gaming stream
            **kwargs: Additional configuration

        Returns:
            Pipeline optimized for gaming content
        """
        config_overrides = {
            "frame_extraction_interval": 0.5,  # Higher frequency for fast action
            "segment_duration_seconds": 8.0,  # Shorter segments
            "enable_real_time": True,
            "processing_priority": "low_latency",
        }

        return cls.create_pipeline(stream_url, config_overrides, **kwargs)

    @classmethod
    def create_for_educational_content(
        cls, stream_url: str, **kwargs
    ) -> StreamIngestionPipeline:
        """Create pipeline optimized for educational content.

        Educational content typically has:
        - Slower pace with important visual information
        - Text and diagrams that need to be readable
        - Longer segments of related content

        Args:
            stream_url: URL of the educational stream
            **kwargs: Additional configuration

        Returns:
            Pipeline optimized for educational content
        """
        config_overrides = {
            "frame_extraction_interval": 2.0,  # Lower frequency
            "segment_duration_seconds": 20.0,  # Longer segments
            "enable_real_time": False,
            "processing_priority": "quality",
        }

        return cls.create_pipeline(stream_url, config_overrides, **kwargs)

    @classmethod
    def create_for_sports_content(
        cls, stream_url: str, **kwargs
    ) -> StreamIngestionPipeline:
        """Create pipeline optimized for sports content.

        Sports content typically has:
        - Bursts of high action
        - Important moments that are brief
        - Need for real-time processing

        Args:
            stream_url: URL of the sports stream
            **kwargs: Additional configuration

        Returns:
            Pipeline optimized for sports content
        """
        config_overrides = {
            "frame_extraction_interval": 0.3,  # Very high frequency
            "segment_duration_seconds": 5.0,  # Short segments
            "enable_real_time": True,
            "processing_priority": "low_latency",
        }

        return cls.create_pipeline(stream_url, config_overrides, **kwargs)