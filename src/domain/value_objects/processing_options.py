"""Processing options value object for flexible highlight detection."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from enum import Enum

from src.domain.exceptions import InvalidValueError


class DetectionStrategy(str, Enum):
    """Available detection strategies."""

    AI_ONLY = "ai_only"  # Pure AI-based detection
    RULE_BASED = "rule_based"  # Rule-based detection only
    HYBRID = "hybrid"  # Combination of AI and rules
    CUSTOM = "custom"  # Custom detection pipeline


class FusionStrategy(str, Enum):
    """Multi-modal fusion strategies."""

    WEIGHTED = "weighted"  # Weighted average (current approach)
    CONSENSUS = "consensus"  # Majority voting
    CASCADE = "cascade"  # Sequential validation
    MAX_CONFIDENCE = "max_confidence"  # Take highest confidence
    CUSTOM = "custom"  # Custom fusion logic


@dataclass(frozen=True)
class ProcessingOptions:
    """Flexible processing configuration for highlight detection.

    This immutable value object encapsulates all processing options
    for highlight detection, supporting both generic and domain-specific
    configurations.
    """

    # Core processing configuration
    dimension_set_id: Optional[int] = None  # ID of dimension set to use
    detection_strategy: DetectionStrategy = DetectionStrategy.AI_ONLY
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED

    # Timing constraints
    min_highlight_duration: float = 10.0  # seconds
    max_highlight_duration: float = 300.0  # seconds
    typical_highlight_duration: float = 60.0  # preferred duration

    # Quality thresholds
    min_confidence_threshold: float = 0.5  # Absolute minimum
    target_confidence_threshold: float = 0.7  # Target quality
    exceptional_threshold: float = 0.85  # High-quality threshold

    # Multi-modal analysis configuration
    enabled_modalities: Set[str] = field(
        default_factory=lambda: {"video", "audio", "text"}
    )
    modality_weights: Dict[str, float] = field(
        default_factory=lambda: {"video": 0.4, "audio": 0.3, "text": 0.3}
    )

    # Analysis windows
    analysis_window_seconds: float = 30.0  # Size of analysis chunks
    context_window_seconds: float = 120.0  # Historical context to consider
    lookahead_seconds: float = 15.0  # Future context for better boundaries

    # Detection features
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

    def __post_init__(self):
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

        # Validate modalities
        if not self.enabled_modalities:
            raise InvalidValueError("At least one modality must be enabled")

        valid_modalities = {"video", "audio", "text", "metadata", "social"}
        invalid = self.enabled_modalities - valid_modalities
        if invalid:
            raise InvalidValueError(f"Invalid modalities: {invalid}")

        # Validate modality weights
        if self.fusion_strategy == FusionStrategy.WEIGHTED:
            total_weight = sum(
                self.modality_weights.get(m, 0) for m in self.enabled_modalities
            )
            if abs(total_weight - 1.0) > 0.01:
                raise InvalidValueError(
                    f"Modality weights must sum to 1.0, got {total_weight}"
                )

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
            detection_strategy=self.detection_strategy,
            fusion_strategy=self.fusion_strategy,
            min_highlight_duration=self.min_highlight_duration,
            max_highlight_duration=self.max_highlight_duration,
            typical_highlight_duration=self.typical_highlight_duration,
            min_confidence_threshold=self.min_confidence_threshold,
            target_confidence_threshold=self.target_confidence_threshold,
            exceptional_threshold=self.exceptional_threshold,
            enabled_modalities=self.enabled_modalities.copy(),
            modality_weights=self.modality_weights.copy(),
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

    def with_strategy(
        self,
        detection: Optional[DetectionStrategy] = None,
        fusion: Optional[FusionStrategy] = None,
    ) -> "ProcessingOptions":
        """Create new options with different strategies."""
        return ProcessingOptions(
            dimension_set_id=self.dimension_set_id,
            detection_strategy=detection or self.detection_strategy,
            fusion_strategy=fusion or self.fusion_strategy,
            min_highlight_duration=self.min_highlight_duration,
            max_highlight_duration=self.max_highlight_duration,
            typical_highlight_duration=self.typical_highlight_duration,
            min_confidence_threshold=self.min_confidence_threshold,
            target_confidence_threshold=self.target_confidence_threshold,
            exceptional_threshold=self.exceptional_threshold,
            enabled_modalities=self.enabled_modalities.copy(),
            modality_weights=self.modality_weights.copy(),
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
            detection_strategy=self.detection_strategy,
            fusion_strategy=self.fusion_strategy,
            min_highlight_duration=self.min_highlight_duration,
            max_highlight_duration=self.max_highlight_duration,
            typical_highlight_duration=self.typical_highlight_duration,
            min_confidence_threshold=self.min_confidence_threshold,
            target_confidence_threshold=self.target_confidence_threshold,
            exceptional_threshold=self.exceptional_threshold,
            enabled_modalities=self.enabled_modalities.copy(),
            modality_weights=self.modality_weights.copy(),
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
        return self.detection_strategy in [
            DetectionStrategy.AI_ONLY,
            DetectionStrategy.HYBRID,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dimension_set_id": self.dimension_set_id,
            "detection_strategy": self.detection_strategy.value,
            "fusion_strategy": self.fusion_strategy.value,
            "min_highlight_duration": self.min_highlight_duration,
            "max_highlight_duration": self.max_highlight_duration,
            "typical_highlight_duration": self.typical_highlight_duration,
            "min_confidence_threshold": self.min_confidence_threshold,
            "target_confidence_threshold": self.target_confidence_threshold,
            "exceptional_threshold": self.exceptional_threshold,
            "enabled_modalities": list(self.enabled_modalities),
            "modality_weights": self.modality_weights.copy(),
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
            detection_strategy=DetectionStrategy.AI_ONLY,
            fusion_strategy=FusionStrategy.WEIGHTED,
            min_highlight_duration=15.0,
            max_highlight_duration=90.0,
            typical_highlight_duration=45.0,
            target_confidence_threshold=0.75,
            enabled_modalities={"video", "audio", "text"},
            modality_weights={"video": 0.5, "audio": 0.3, "text": 0.2},
            enable_motion_detection=True,
            enable_scene_detection=True,
            processing_priority="balanced",
            industry_preset="gaming",
        )

    @classmethod
    def for_education(cls) -> "ProcessingOptions":
        """Create processing options optimized for educational content."""
        return cls(
            detection_strategy=DetectionStrategy.HYBRID,
            fusion_strategy=FusionStrategy.CONSENSUS,
            min_highlight_duration=30.0,
            max_highlight_duration=300.0,
            typical_highlight_duration=120.0,
            min_confidence_threshold=0.6,
            target_confidence_threshold=0.8,
            enabled_modalities={"video", "audio", "text"},
            modality_weights={"video": 0.3, "audio": 0.4, "text": 0.3},
            enable_silence_detection=True,
            processing_priority="quality",
            include_context_before=10.0,
            include_context_after=10.0,
            industry_preset="education",
        )
