"""Highlight agent configuration domain entity.

DEPRECATED: This complex configuration is being replaced by StreamProcessingConfig
for the streamlined highlight detection flow. This file is kept for backward compatibility
and will be removed in a future version.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..value_objects.scoring_config import DimensionWeights, ScoreThresholds
from ..value_objects.prompt_template import ConfigurablePromptTemplate
from .base import Entity


@dataclass
class KeywordConfig:
    """Configuration for keyword-based highlight detection."""

    high_priority: List[str] = field(default_factory=list)
    medium_priority: List[str] = field(default_factory=list)
    low_priority: List[str] = field(default_factory=list)

    def get_priority_score(self, keyword: str) -> float:
        """Get priority score for a keyword."""
        if keyword.lower() in [k.lower() for k in self.high_priority]:
            return 1.0
        elif keyword.lower() in [k.lower() for k in self.medium_priority]:
            return 0.6
        elif keyword.lower() in [k.lower() for k in self.low_priority]:
            return 0.3
        return 0.0


@dataclass
class ContextModifiers:
    """Configuration for context-based score modifications."""

    modifiers: Dict[str, float] = field(default_factory=dict)

    def apply_modifier(self, context_type: str, base_score: float) -> float:
        """Apply context modifier to base score."""
        modifier = self.modifiers.get(context_type, 0.0)
        return min(1.0, base_score + modifier)


@dataclass
class TimingConfig:
    """Configuration for highlight timing and spacing."""

    min_highlight_duration: int = 30  # seconds
    max_highlight_duration: int = 90  # seconds
    typical_highlight_duration: int = 60  # seconds
    min_spacing_seconds: int = 30  # minimum time between highlights
    max_per_5min_window: int = 3  # maximum highlights per 5-minute window
    pending_timeout_seconds: int = 30  # timeout for pending highlights


@dataclass
class HighlightAgentConfig(Entity[int]):
    """Configuration for highlight detection agents.

    This entity allows B2B consumers to customize how highlights are detected
    and scored for their specific content and audience.
    """

    # Basic identification
    name: str
    description: str
    organization_id: int
    user_id: int

    # Content configuration (with defaults)
    content_type: str = "gaming"  # gaming, sports, general, etc.
    game_name: Optional[str] = None

    # AI Analysis configuration
    prompt_template: ConfigurablePromptTemplate = field(
        default_factory=lambda: ConfigurablePromptTemplate.default()
    )

    # Scoring configuration
    dimension_weights: DimensionWeights = field(
        default_factory=lambda: DimensionWeights.default()
    )
    score_thresholds: ScoreThresholds = field(
        default_factory=lambda: ScoreThresholds.default()
    )

    # Keyword and context configuration
    keyword_config: KeywordConfig = field(default_factory=KeywordConfig)
    context_modifiers: ContextModifiers = field(default_factory=ContextModifiers)

    # Timing and quality configuration
    timing_config: TimingConfig = field(default_factory=TimingConfig)
    min_confidence_threshold: float = 0.7
    similarity_threshold: float = 0.6  # for avoiding duplicate highlights

    # Operational settings
    is_active: bool = True
    version: int = 1

    # Usage tracking
    highlights_generated: int = 0
    last_used_at: Optional[datetime] = None

    def update_configuration(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        prompt_template: Optional[ConfigurablePromptTemplate] = None,
        dimension_weights: Optional[DimensionWeights] = None,
        score_thresholds: Optional[ScoreThresholds] = None,
        keyword_config: Optional[KeywordConfig] = None,
        context_modifiers: Optional[ContextModifiers] = None,
        timing_config: Optional[TimingConfig] = None,
        min_confidence_threshold: Optional[float] = None,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        """Update configuration settings."""
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if prompt_template is not None:
            self.prompt_template = prompt_template
        if dimension_weights is not None:
            self.dimension_weights = dimension_weights
        if score_thresholds is not None:
            self.score_thresholds = score_thresholds
        if keyword_config is not None:
            self.keyword_config = keyword_config
        if context_modifiers is not None:
            self.context_modifiers = context_modifiers
        if timing_config is not None:
            self.timing_config = timing_config
        if min_confidence_threshold is not None:
            self.min_confidence_threshold = min_confidence_threshold
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold

        self.version += 1
        self.updated_at = datetime.utcnow()

    def record_usage(self) -> None:
        """Record that this configuration was used to generate a highlight."""
        self.highlights_generated += 1
        self.last_used_at = datetime.utcnow()

    def clone_for_organization(
        self, new_organization_id: int, new_user_id: int
    ) -> "HighlightAgentConfig":
        """Create a copy of this configuration for another organization."""
        return HighlightAgentConfig(
            id=None,  # New entity
            name=f"{self.name} (Copy)",
            description=f"Copy of {self.name}",
            organization_id=new_organization_id,
            user_id=new_user_id,
            content_type=self.content_type,
            game_name=self.game_name,
            prompt_template=self.prompt_template,
            dimension_weights=self.dimension_weights,
            score_thresholds=self.score_thresholds,
            keyword_config=self.keyword_config,
            context_modifiers=self.context_modifiers,
            timing_config=self.timing_config,
            min_confidence_threshold=self.min_confidence_threshold,
            similarity_threshold=self.similarity_threshold,
            version=1,
            highlights_generated=0,
            last_used_at=None,
        )

    def validate_configuration(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []

        # Validate thresholds
        if not (0.0 <= self.min_confidence_threshold <= 1.0):
            errors.append("min_confidence_threshold must be between 0.0 and 1.0")

        if not (0.0 <= self.similarity_threshold <= 1.0):
            errors.append("similarity_threshold must be between 0.0 and 1.0")

        # Validate timing
        if (
            self.timing_config.min_highlight_duration
            >= self.timing_config.max_highlight_duration
        ):
            errors.append(
                "min_highlight_duration must be less than max_highlight_duration"
            )

        if self.timing_config.min_spacing_seconds < 0:
            errors.append("min_spacing_seconds must be non-negative")

        if self.timing_config.max_per_5min_window <= 0:
            errors.append("max_per_5min_window must be positive")

        # Validate dimension weights sum to 1.0
        total_weight = sum(
            [
                self.dimension_weights.skill_execution,
                self.dimension_weights.game_impact,
                self.dimension_weights.rarity,
                self.dimension_weights.visual_spectacle,
                self.dimension_weights.emotional_intensity,
                self.dimension_weights.narrative_value,
                self.dimension_weights.timing_importance,
                self.dimension_weights.momentum_shift,
            ]
        )

        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            errors.append(
                f"Dimension weights must sum to 1.0 (current sum: {total_weight:.3f})"
            )

        return errors

    def get_effective_prompt(self, context: Dict[str, Any]) -> str:
        """Get the effective prompt with context substitutions."""
        return self.prompt_template.render(context)

    def calculate_highlight_score(
        self,
        dimensions: Dict[str, float],
        keywords: List[str] = None,
        context_type: Optional[str] = None,
    ) -> float:
        """Calculate a highlight score using this configuration.

        Args:
            dimensions: Dictionary of dimension scores (0.0-1.0)
            keywords: Optional list of detected keywords
            context_type: Optional context for applying modifiers

        Returns:
            Final highlight score (0.0-1.0)
        """
        # Calculate weighted dimension score
        weighted_score = (
            dimensions.get("skill_execution", 0)
            * self.dimension_weights.skill_execution
            + dimensions.get("game_impact", 0) * self.dimension_weights.game_impact
            + dimensions.get("rarity", 0) * self.dimension_weights.rarity
            + dimensions.get("visual_spectacle", 0)
            * self.dimension_weights.visual_spectacle
            + dimensions.get("emotional_intensity", 0)
            * self.dimension_weights.emotional_intensity
            + dimensions.get("narrative_value", 0)
            * self.dimension_weights.narrative_value
            + dimensions.get("timing_importance", 0)
            * self.dimension_weights.timing_importance
            + dimensions.get("momentum_shift", 0)
            * self.dimension_weights.momentum_shift
        )

        # Apply keyword boosts
        if keywords:
            keyword_boost = max(
                [self.keyword_config.get_priority_score(kw) for kw in keywords]
            )
            weighted_score = min(1.0, weighted_score + (keyword_boost * 0.1))

        # Apply context modifiers
        if context_type:
            weighted_score = self.context_modifiers.apply_modifier(
                context_type, weighted_score
            )

        return min(1.0, max(0.0, weighted_score))

    @staticmethod
    def create_default_gaming_config(
        organization_id: int, user_id: int
    ) -> "HighlightAgentConfig":
        """Create a default configuration optimized for gaming content."""
        return HighlightAgentConfig(
            id=None,
            name="Default Gaming Configuration",
            description="General gaming highlight detection with balanced scoring",
            organization_id=organization_id,
            user_id=user_id,
            content_type="gaming",
            keyword_config=KeywordConfig(
                high_priority=["ace", "clutch", "epic", "insane", "amazing", "perfect"],
                medium_priority=["nice", "good", "win", "kill", "combo", "streak"],
                low_priority=["play", "move", "shot", "hit", "score"],
            ),
            context_modifiers=ContextModifiers(
                modifiers={
                    "overtime": 0.15,
                    "final_round": 0.2,
                    "comeback": 0.1,
                    "tournament": 0.05,
                }
            ),
        )

    @staticmethod
    def create_valorant_config(
        organization_id: int, user_id: int
    ) -> "HighlightAgentConfig":
        """Create a configuration optimized for Valorant content."""
        return HighlightAgentConfig(
            id=None,
            name="Valorant Highlights",
            description="Optimized for Valorant tactical shooter highlights",
            organization_id=organization_id,
            user_id=user_id,
            content_type="gaming",
            game_name="Valorant",
            dimension_weights=DimensionWeights(
                skill_execution=0.25,
                game_impact=0.20,
                rarity=0.15,
                visual_spectacle=0.15,
                emotional_intensity=0.10,
                narrative_value=0.05,
                timing_importance=0.05,
                momentum_shift=0.05,
            ),
            keyword_config=KeywordConfig(
                high_priority=["ace", "1v4", "1v5", "flawless", "ninja defuse"],
                medium_priority=[
                    "clutch",
                    "triple kill",
                    "headshot",
                    "wallbang",
                    "one tap",
                ],
                low_priority=["double kill", "first blood", "save round"],
            ),
            timing_config=TimingConfig(
                min_highlight_duration=30,
                max_highlight_duration=60,
                typical_highlight_duration=45,
            ),
        )


# Backward compatibility alias - will be removed
# Use StreamProcessingConfig for new code
