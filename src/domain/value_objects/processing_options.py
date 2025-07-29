"""Processing options value object."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class ProcessingOptions:
    """Value object representing stream processing configuration options.
    
    This is an immutable value object that encapsulates all processing
    options for highlight detection including thresholds, filters, and
    analysis parameters.
    """
    
    # Core processing options
    confidence_threshold: float = 0.7
    min_highlight_duration: float = 10.0  # seconds
    max_highlight_duration: float = 300.0  # seconds
    
    # Content analysis options
    analyze_video: bool = True
    analyze_audio: bool = True
    analyze_chat: bool = True
    analyze_metadata: bool = True
    
    # Platform-specific options
    include_chat_sentiment: bool = True
    include_viewer_metrics: bool = True
    
    # Quality options
    video_quality: str = "high"  # low, medium, high
    audio_quality: str = "medium"  # low, medium, high
    
    # Custom filters and options
    custom_filters: Dict[str, Any] = field(default_factory=dict)
    excluded_categories: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate processing options after initialization."""
        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise InvalidValueError(
                f"Confidence threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )
        
        # Validate duration constraints
        if self.min_highlight_duration < 0:
            raise InvalidValueError(
                f"Minimum highlight duration cannot be negative, got {self.min_highlight_duration}"
            )
        
        if self.max_highlight_duration < self.min_highlight_duration:
            raise InvalidValueError(
                f"Maximum highlight duration ({self.max_highlight_duration}) must be "
                f"greater than minimum ({self.min_highlight_duration})"
            )
        
        # Validate quality options
        valid_qualities = {"low", "medium", "high"}
        if self.video_quality not in valid_qualities:
            raise InvalidValueError(
                f"Video quality must be one of {valid_qualities}, got {self.video_quality}"
            )
        
        if self.audio_quality not in valid_qualities:
            raise InvalidValueError(
                f"Audio quality must be one of {valid_qualities}, got {self.audio_quality}"
            )
        
        # Ensure at least one analysis type is enabled
        if not any([self.analyze_video, self.analyze_audio, 
                   self.analyze_chat, self.analyze_metadata]):
            raise InvalidValueError(
                "At least one analysis type must be enabled"
            )
    
    def with_threshold(self, threshold: float) -> "ProcessingOptions":
        """Create new options with different confidence threshold."""
        return ProcessingOptions(
            confidence_threshold=threshold,
            min_highlight_duration=self.min_highlight_duration,
            max_highlight_duration=self.max_highlight_duration,
            analyze_video=self.analyze_video,
            analyze_audio=self.analyze_audio,
            analyze_chat=self.analyze_chat,
            analyze_metadata=self.analyze_metadata,
            include_chat_sentiment=self.include_chat_sentiment,
            include_viewer_metrics=self.include_viewer_metrics,
            video_quality=self.video_quality,
            audio_quality=self.audio_quality,
            custom_filters=self.custom_filters.copy(),
            excluded_categories=self.excluded_categories.copy()
        )
    
    def with_custom_filter(self, key: str, value: Any) -> "ProcessingOptions":
        """Create new options with additional custom filter."""
        new_filters = self.custom_filters.copy()
        new_filters[key] = value
        
        return ProcessingOptions(
            confidence_threshold=self.confidence_threshold,
            min_highlight_duration=self.min_highlight_duration,
            max_highlight_duration=self.max_highlight_duration,
            analyze_video=self.analyze_video,
            analyze_audio=self.analyze_audio,
            analyze_chat=self.analyze_chat,
            analyze_metadata=self.analyze_metadata,
            include_chat_sentiment=self.include_chat_sentiment,
            include_viewer_metrics=self.include_viewer_metrics,
            video_quality=self.video_quality,
            audio_quality=self.audio_quality,
            custom_filters=new_filters,
            excluded_categories=self.excluded_categories.copy()
        )
    
    @property
    def is_high_quality(self) -> bool:
        """Check if processing is configured for high quality."""
        return self.video_quality == "high" and self.audio_quality in ["medium", "high"]
    
    @property
    def enabled_analyzers(self) -> List[str]:
        """Get list of enabled analyzer types."""
        analyzers = []
        if self.analyze_video:
            analyzers.append("video")
        if self.analyze_audio:
            analyzers.append("audio")
        if self.analyze_chat:
            analyzers.append("chat")
        if self.analyze_metadata:
            analyzers.append("metadata")
        return analyzers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "min_highlight_duration": self.min_highlight_duration,
            "max_highlight_duration": self.max_highlight_duration,
            "analyze_video": self.analyze_video,
            "analyze_audio": self.analyze_audio,
            "analyze_chat": self.analyze_chat,
            "analyze_metadata": self.analyze_metadata,
            "include_chat_sentiment": self.include_chat_sentiment,
            "include_viewer_metrics": self.include_viewer_metrics,
            "video_quality": self.video_quality,
            "audio_quality": self.audio_quality,
            "custom_filters": self.custom_filters.copy(),
            "excluded_categories": self.excluded_categories.copy()
        }