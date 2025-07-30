"""Stream ingestion factory for automatic format detection and processing.

This module provides factory methods for creating appropriate stream processors
based on stream URLs and formats.
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
        stream_url: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> StreamIngestionPipeline:
        """Create an ingestion pipeline for a stream URL.
        
        Args:
            stream_url: URL of the stream to process
            config_overrides: Optional configuration overrides
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured stream ingestion pipeline
        """
        # Detect platform from URL
        platform = StreamIngestionFactory.detect_platform(stream_url)
        
        # Create base configuration
        config = IngestionConfig(
            stream_url=stream_url,
            platform=platform,
            **kwargs
        )
        
        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Platform-specific optimizations
        StreamIngestionFactory._apply_platform_optimizations(config, platform)
        
        logger.info(f"Created ingestion pipeline for {platform} stream: {stream_url}")
        
        return StreamIngestionPipeline(config)
    
    @staticmethod
    def detect_platform(stream_url: str) -> Optional[str]:
        """Detect streaming platform from URL.
        
        Args:
            stream_url: Stream URL to analyze
            
        Returns:
            Detected platform name or None
        """
        parsed = urlparse(stream_url)
        domain = parsed.netloc.lower()
        
        # Platform detection patterns
        if "twitch.tv" in domain:
            return "twitch"
        elif "youtube.com" in domain or "youtu.be" in domain:
            return "youtube"
        elif "rtmp://" in stream_url.lower():
            return "rtmp"
        elif ".m3u8" in stream_url.lower() or "hls" in stream_url.lower():
            return "hls"
        elif stream_url.startswith("http"):
            return "http"
        
        return None
    
    @staticmethod
    def _apply_platform_optimizations(config: IngestionConfig, platform: Optional[str]) -> None:
        """Apply platform-specific optimizations.
        
        Args:
            config: Configuration to optimize
            platform: Detected platform
        """
        if platform == "twitch":
            # Twitch-specific optimizations
            config.frame_extraction_interval = 0.5  # Higher frequency for gaming
            config.segment_duration_seconds = 8.0  # Shorter segments for action
            config.enable_real_time = True
            
        elif platform == "youtube":
            # YouTube-specific optimizations
            config.frame_extraction_interval = 1.0
            config.segment_duration_seconds = 12.0
            config.hardware_acceleration = True  # YouTube often has high quality
            
        elif platform == "rtmp":
            # RTMP-specific optimizations
            config.frame_extraction_interval = 0.33  # 3 FPS for low latency
            config.segment_duration_seconds = 6.0
            config.enable_real_time = True
            config.buffer_duration_seconds = 120.0  # Smaller buffer
            
        elif platform in ["hls", "http"]:
            # HLS/HTTP stream optimizations
            config.frame_extraction_interval = 1.0
            config.segment_duration_seconds = 10.0
            config.buffer_duration_seconds = 180.0
    
    @staticmethod
    def create_for_gaming(
        stream_url: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> StreamIngestionPipeline:
        """Create pipeline optimized for gaming content.
        
        Args:
            stream_url: Gaming stream URL
            api_key: Gemini API key
            **kwargs: Additional configuration
            
        Returns:
            Gaming-optimized pipeline
        """
        config_overrides = {
            "frame_extraction_interval": 0.33,  # 3 FPS for action detection
            "segment_duration_seconds": 6.0,    # Short segments for highlights
            "enable_real_time": True,
            "min_video_quality": 0.4,           # Accept lower quality for action
        }
        
        if api_key:
            gemini_config = GeminiProcessorConfig(
                api_key=api_key,
                model_name="gemini-2.0-flash-001",
                temperature=0.3,  # Lower temperature for consistent gaming analysis
            )
            config_overrides["gemini_config"] = gemini_config
        
        config_overrides.update(kwargs)
        
        return StreamIngestionFactory.create_pipeline(
            stream_url, config_overrides
        )
    
    @staticmethod
    def create_for_educational(
        stream_url: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> StreamIngestionPipeline:
        """Create pipeline optimized for educational content.
        
        Args:
            stream_url: Educational stream URL
            api_key: Gemini API key
            **kwargs: Additional configuration
            
        Returns:
            Education-optimized pipeline
        """
        config_overrides = {
            "frame_extraction_interval": 2.0,   # Lower frequency for educational content
            "segment_duration_seconds": 15.0,   # Longer segments for concepts
            "enable_real_time": True,
            "min_video_quality": 0.6,           # Higher quality for text/diagrams
        }
        
        if api_key:
            gemini_config = GeminiProcessorConfig(
                api_key=api_key,
                model_name="gemini-2.0-flash-001",
                temperature=0.7,  # Higher temperature for creative educational analysis
            )
            config_overrides["gemini_config"] = gemini_config
        
        config_overrides.update(kwargs)
        
        return StreamIngestionFactory.create_pipeline(
            stream_url, config_overrides
        )
    
    @staticmethod
    def create_for_sports(
        stream_url: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> StreamIngestionPipeline:
        """Create pipeline optimized for sports content.
        
        Args:
            stream_url: Sports stream URL
            api_key: Gemini API key
            **kwargs: Additional configuration
            
        Returns:
            Sports-optimized pipeline
        """
        config_overrides = {
            "frame_extraction_interval": 0.5,   # High frequency for sports action
            "segment_duration_seconds": 10.0,   # Medium segments for plays
            "enable_real_time": True,
            "min_video_quality": 0.5,           # Balance quality and speed
        }
        
        if api_key:
            gemini_config = GeminiProcessorConfig(
                api_key=api_key,
                model_name="gemini-2.0-flash-001",
                temperature=0.4,  # Moderate temperature for sports analysis
            )
            config_overrides["gemini_config"] = gemini_config
        
        config_overrides.update(kwargs)
        
        return StreamIngestionFactory.create_pipeline(
            stream_url, config_overrides
        )