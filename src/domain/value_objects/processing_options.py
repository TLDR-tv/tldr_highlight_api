"""Processing options value object for highlight detection using Gemini's native video understanding."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from src.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class ProcessingOptions:
    """Processing configuration for highlight detection.

    This immutable value object encapsulates all processing options
    for highlight detection using Gemini's native video understanding capabilities.
    """

    # Core processing configuration
    dimension_set_id: Optional[int] = None  # ID of dimension set to use

    # Timing constraints
    min_highlight_duration: float = 10.0  # seconds
    max_highlight_duration: float = 300.0  # seconds
    typical_highlight_duration: float = 60.0  # preferred duration

    # Quality thresholds
    min_confidence_threshold: float = 0.5  # Absolute minimum
    target_confidence_threshold: float = 0.7  # Target quality
    exceptional_threshold: float = 0.85  # High-quality threshold

    # Analysis windows
    analysis_window_seconds: float = 30.0  # Size of analysis chunks
    context_window_seconds: float = 120.0  # Historical context to consider
    lookahead_seconds: float = 15.0  # Future context for better boundaries

    # Detection features (for prompting Gemini)
    enable_wake_words: bool = False  # Enable wake word detection
    wake_words: List[str] = field(default_factory=list)
    enable_scene_detection: bool = True  # Detect scene changes
    enable_silence_detection: bool = True  # Detect significant silences
    enable_motion_detection: bool = True  # Detect motion intensity

    # Post-processing options
    merge_nearby_highlights: bool = True  # Merge close highlights
    merge_threshold_seconds: float = 10.0  # Max gap to merge
    remove_duplicates: bool = True  # Remove similar highlights
    similarity_threshold: float = 0.8  # Similarity threshold for duplicates

    # Output configuration
    include_context_before: float = 5.0  # Seconds before highlight
    include_context_after: float = 5.0  # Seconds after highlight
    generate_thumbnails: bool = True  # Generate preview thumbnails
    generate_previews: bool = True  # Generate video previews

    # Performance options
    parallel_processing: bool = True  # Enable parallel analysis
    max_parallel_tasks: int = 4  # Maximum parallel tasks
    processing_priority: str = "balanced"  # speed, quality, balanced

    # Custom configuration
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)
    custom_filters: Dict[str, Any] = field(default_factory=dict)
    metadata_extractors: List[str] = field(default_factory=list)

    # Industry/domain specific
    industry_preset: Optional[str] = None  # gaming, sports, education, etc.
    content_rating: Optional[str] = None  # G, PG, PG-13, R, etc.

    def __post_init__(self) -> None:
        """Validate processing options after initialization."""
        # Validate thresholds
        thresholds = [
            self.min_confidence_threshold,
            self.target_confidence_threshold,
            self.exceptional_threshold,
        ]
        for threshold in thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise InvalidValueError(
                    "Confidence thresholds must be between 0.0 and 1.0"
                )

        if not (
            self.min_confidence_threshold
            <= self.target_confidence_threshold
            <= self.exceptional_threshold
        ):
            raise InvalidValueError(
                "Thresholds must be in ascending order: min <= target <= exceptional"
            )

        # Validate duration constraints
        if self.min_highlight_duration < 0:
            raise InvalidValueError("Minimum highlight duration cannot be negative")

        if self.max_highlight_duration < self.min_highlight_duration:
            raise InvalidValueError(
                f"Maximum duration ({self.max_highlight_duration}) must be >= minimum ({self.min_highlight_duration})"
            )

        if (
            not self.min_highlight_duration
            <= self.typical_highlight_duration
            <= self.max_highlight_duration
        ):
            raise InvalidValueError("Typical duration must be between min and max")

        # Validate windows
        if self.analysis_window_seconds <= 0:
            raise InvalidValueError("Analysis window must be positive")

        # Validate performance options
        if self.max_parallel_tasks < 1:
            raise InvalidValueError("Must have at least 1 parallel task")

        valid_priorities = {"speed", "quality", "balanced"}
        if self.processing_priority not in valid_priorities:
            raise InvalidValueError(f"Priority must be one of {valid_priorities}")

    def with_dimension_set(self, dimension_set_id: int) -> "ProcessingOptions":
        """Create new options with a different dimension set."""
        return ProcessingOptions(
            dimension_set_id=dimension_set_id,
            min_highlight_duration=self.min_highlight_duration,
            max_highlight_duration=self.max_highlight_duration,
            typical_highlight_duration=self.typical_highlight_duration,
            min_confidence_threshold=self.min_confidence_threshold,
            target_confidence_threshold=self.target_confidence_threshold,
            exceptional_threshold=self.exceptional_threshold,
            analysis_window_seconds=self.analysis_window_seconds,
            context_window_seconds=self.context_window_seconds,
            lookahead_seconds=self.lookahead_seconds,
            enable_wake_words=self.enable_wake_words,
            wake_words=self.wake_words.copy(),
            enable_scene_detection=self.enable_scene_detection,
            enable_silence_detection=self.enable_silence_detection,
            enable_motion_detection=self.enable_motion_detection,
            merge_nearby_highlights=self.merge_nearby_highlights,
            merge_threshold_seconds=self.merge_threshold_seconds,
            remove_duplicates=self.remove_duplicates,
            similarity_threshold=self.similarity_threshold,
            include_context_before=self.include_context_before,
            include_context_after=self.include_context_after,
            generate_thumbnails=self.generate_thumbnails,
            generate_previews=self.generate_previews,
            parallel_processing=self.parallel_processing,
            max_parallel_tasks=self.max_parallel_tasks,
            processing_priority=self.processing_priority,
            custom_rules=self.custom_rules.copy(),
            custom_filters=self.custom_filters.copy(),
            metadata_extractors=self.metadata_extractors.copy(),
            industry_preset=self.industry_preset,
            content_rating=self.content_rating,
        )

    def add_custom_rule(self, rule: Dict[str, Any]) -> "ProcessingOptions":
        """Add a custom detection rule."""
        new_rules = self.custom_rules.copy()
        new_rules.append(rule)

        return ProcessingOptions(
            dimension_set_id=self.dimension_set_id,
            min_highlight_duration=self.min_highlight_duration,
            max_highlight_duration=self.max_highlight_duration,
            typical_highlight_duration=self.typical_highlight_duration,
            min_confidence_threshold=self.min_confidence_threshold,
            target_confidence_threshold=self.target_confidence_threshold,
            exceptional_threshold=self.exceptional_threshold,
            analysis_window_seconds=self.analysis_window_seconds,
            context_window_seconds=self.context_window_seconds,
            lookahead_seconds=self.lookahead_seconds,
            enable_wake_words=self.enable_wake_words,
            wake_words=self.wake_words.copy(),
            enable_scene_detection=self.enable_scene_detection,
            enable_silence_detection=self.enable_silence_detection,
            enable_motion_detection=self.enable_motion_detection,
            merge_nearby_highlights=self.merge_nearby_highlights,
            merge_threshold_seconds=self.merge_threshold_seconds,
            remove_duplicates=self.remove_duplicates,
            similarity_threshold=self.similarity_threshold,
            include_context_before=self.include_context_before,
            include_context_after=self.include_context_after,
            generate_thumbnails=self.generate_thumbnails,
            generate_previews=self.generate_previews,
            parallel_processing=self.parallel_processing,
            max_parallel_tasks=self.max_parallel_tasks,
            processing_priority=self.processing_priority,
            custom_rules=new_rules,
            custom_filters=self.custom_filters.copy(),
            metadata_extractors=self.metadata_extractors.copy(),
            industry_preset=self.industry_preset,
            content_rating=self.content_rating,
        )

    @property
    def is_high_quality(self) -> bool:
        """Check if processing is configured for high quality."""
        return (
            self.processing_priority == "quality"
            and self.target_confidence_threshold >= 0.8
        )

    @property
    def requires_dimension_set(self) -> bool:
        """Check if this configuration requires a dimension set."""
        # Always requires dimension set for AI-based detection
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dimension_set_id": self.dimension_set_id,
            "min_highlight_duration": self.min_highlight_duration,
            "max_highlight_duration": self.max_highlight_duration,
            "typical_highlight_duration": self.typical_highlight_duration,
            "min_confidence_threshold": self.min_confidence_threshold,
            "target_confidence_threshold": self.target_confidence_threshold,
            "exceptional_threshold": self.exceptional_threshold,
            "analysis_window_seconds": self.analysis_window_seconds,
            "context_window_seconds": self.context_window_seconds,
            "lookahead_seconds": self.lookahead_seconds,
            "enable_wake_words": self.enable_wake_words,
            "wake_words": self.wake_words.copy(),
            "enable_scene_detection": self.enable_scene_detection,
            "enable_silence_detection": self.enable_silence_detection,
            "enable_motion_detection": self.enable_motion_detection,
            "merge_nearby_highlights": self.merge_nearby_highlights,
            "merge_threshold_seconds": self.merge_threshold_seconds,
            "remove_duplicates": self.remove_duplicates,
            "similarity_threshold": self.similarity_threshold,
            "include_context_before": self.include_context_before,
            "include_context_after": self.include_context_after,
            "generate_thumbnails": self.generate_thumbnails,
            "generate_previews": self.generate_previews,
            "parallel_processing": self.parallel_processing,
            "max_parallel_tasks": self.max_parallel_tasks,
            "processing_priority": self.processing_priority,
            "custom_rules": self.custom_rules.copy(),
            "custom_filters": self.custom_filters.copy(),
            "metadata_extractors": self.metadata_extractors.copy(),
            "industry_preset": self.industry_preset,
            "content_rating": self.content_rating,
        }

    @classmethod
    def for_gaming(cls) -> "ProcessingOptions":
        """Create processing options optimized for gaming content."""
        return cls(
            min_highlight_duration=15.0,
            max_highlight_duration=90.0,
            typical_highlight_duration=45.0,
            target_confidence_threshold=0.75,
            enable_motion_detection=True,
            enable_scene_detection=True,
            processing_priority="balanced",
            industry_preset="gaming",
        )

    @classmethod
    def for_education(cls) -> "ProcessingOptions":
        """Create processing options optimized for educational content."""
        return cls(
            min_highlight_duration=30.0,
            max_highlight_duration=300.0,
            typical_highlight_duration=120.0,
            min_confidence_threshold=0.6,
            target_confidence_threshold=0.8,
            enable_silence_detection=True,
            processing_priority="quality",
            include_context_before=10.0,
            include_context_after=10.0,
            industry_preset="education",
        )

    @classmethod
    def for_sports(cls) -> "ProcessingOptions":
        """Create processing options optimized for sports content."""
        return cls(
            min_highlight_duration=10.0,
            max_highlight_duration=60.0,
            typical_highlight_duration=30.0,
            target_confidence_threshold=0.8,
            enable_motion_detection=True,
            enable_scene_detection=True,
            processing_priority="speed",
            merge_threshold_seconds=5.0,
            industry_preset="sports",
        )

    @classmethod
    def for_corporate(cls) -> "ProcessingOptions":
        """Create processing options optimized for corporate/meeting content."""
        return cls(
            min_highlight_duration=20.0,
            max_highlight_duration=180.0,
            typical_highlight_duration=60.0,
            min_confidence_threshold=0.7,
            target_confidence_threshold=0.85,
            enable_silence_detection=True,
            enable_wake_words=True,
            processing_priority="quality",
            include_context_before=10.0,
            include_context_after=10.0,
            industry_preset="corporate",
        )
