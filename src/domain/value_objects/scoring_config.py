"""Scoring configuration value objects for highlight detection.

These value objects define how highlights are scored across multiple dimensions
and what thresholds must be met for different highlight types.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DimensionWeights:
    """Weights for the 8 dimensions of highlight scoring.

    All weights should sum to 1.0 for proper normalization.
    """

    skill_execution: float = 0.20  # Technical skill and precision
    game_impact: float = 0.20  # Effect on game outcome
    rarity: float = 0.15  # How uncommon the event is
    visual_spectacle: float = 0.15  # Visual appeal and clarity
    emotional_intensity: float = 0.10  # Emotional reaction intensity
    narrative_value: float = 0.10  # Story/context importance
    timing_importance: float = 0.05  # Clutch timing factors
    momentum_shift: float = 0.05  # Game momentum changes

    def __post_init__(self) -> None:
        """Validate that weights sum to approximately 1.0."""
        total = (
            self.skill_execution
            + self.game_impact
            + self.rarity
            + self.visual_spectacle
            + self.emotional_intensity
            + self.narrative_value
            + self.timing_importance
            + self.momentum_shift
        )

        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Dimension weights must sum to 1.0, got {total:.3f}")

    @classmethod
    def default(cls) -> "DimensionWeights":
        """Create default balanced weights."""
        return cls()

    @classmethod
    def skill_focused(cls) -> "DimensionWeights":
        """Create weights focused on skill execution."""
        return cls(
            skill_execution=0.35,
            game_impact=0.20,
            rarity=0.15,
            visual_spectacle=0.15,
            emotional_intensity=0.05,
            narrative_value=0.05,
            timing_importance=0.03,
            momentum_shift=0.02,
        )

    @classmethod
    def impact_focused(cls) -> "DimensionWeights":
        """Create weights focused on game impact."""
        return cls(
            skill_execution=0.15,
            game_impact=0.35,
            rarity=0.15,
            visual_spectacle=0.10,
            emotional_intensity=0.10,
            narrative_value=0.05,
            timing_importance=0.05,
            momentum_shift=0.05,
        )

    @classmethod
    def entertainment_focused(cls) -> "DimensionWeights":
        """Create weights focused on entertainment value."""
        return cls(
            skill_execution=0.15,
            game_impact=0.15,
            rarity=0.20,
            visual_spectacle=0.20,
            emotional_intensity=0.15,
            narrative_value=0.10,
            timing_importance=0.03,
            momentum_shift=0.02,
        )


@dataclass(frozen=True)
class ScoreThresholds:
    """Minimum score thresholds for different highlight types and quality levels."""

    # General quality thresholds
    minimum_viable: float = 0.5  # Absolute minimum for any highlight
    good_quality: float = 0.7  # Good quality highlight
    exceptional: float = 0.85  # Exceptional highlight
    legendary: float = 0.95  # Legendary/perfect highlight

    # Content-type specific thresholds
    skill_play: float = 0.65  # Pure skill demonstrations
    clutch_moment: float = 0.75  # Clutch/pressure situations
    comeback: float = 0.70  # Comeback moments
    rare_event: float = 0.80  # Rare game events
    perfect_execution: float = 0.90  # Perfect/flawless execution

    # Wake word and manual triggers
    wake_word_boost: float = 0.15  # Boost for wake word triggers
    manual_trigger: float = 0.90  # Score for manual highlights

    def __post_init__(self) -> None:
        """Validate threshold ordering."""
        thresholds = [
            self.minimum_viable,
            self.good_quality,
            self.exceptional,
            self.legendary,
        ]

        if not all(
            thresholds[i] <= thresholds[i + 1] for i in range(len(thresholds) - 1)
        ):
            raise ValueError("Thresholds must be in ascending order")

        # Validate all thresholds are in valid range
        all_thresholds = [
            self.minimum_viable,
            self.good_quality,
            self.exceptional,
            self.legendary,
            self.skill_play,
            self.clutch_moment,
            self.comeback,
            self.rare_event,
            self.perfect_execution,
            self.wake_word_boost,
            self.manual_trigger,
        ]

        for threshold in all_thresholds:
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(
                    f"All thresholds must be between 0.0 and 1.0, got {threshold}"
                )

    @classmethod
    def default(cls) -> "ScoreThresholds":
        """Create default balanced thresholds."""
        return cls()

    @classmethod
    def strict(cls) -> "ScoreThresholds":
        """Create strict thresholds for high-quality content only."""
        return cls(
            minimum_viable=0.65,
            good_quality=0.75,
            exceptional=0.88,
            legendary=0.95,
            skill_play=0.70,
            clutch_moment=0.80,
            comeback=0.75,
            rare_event=0.85,
            perfect_execution=0.92,
        )

    @classmethod
    def lenient(cls) -> "ScoreThresholds":
        """Create lenient thresholds for more content capture."""
        return cls(
            minimum_viable=0.40,
            good_quality=0.60,
            exceptional=0.80,
            legendary=0.92,
            skill_play=0.55,
            clutch_moment=0.65,
            comeback=0.60,
            rare_event=0.75,
            perfect_execution=0.85,
        )

    def get_quality_level(self, score: float) -> str:
        """Get the quality level for a given score."""
        if score >= self.legendary:
            return "legendary"
        elif score >= self.exceptional:
            return "exceptional"
        elif score >= self.good_quality:
            return "good"
        elif score >= self.minimum_viable:
            return "viable"
        else:
            return "below_threshold"

    def meets_threshold(self, score: float, highlight_type: str = "general") -> bool:
        """Check if a score meets the threshold for a specific highlight type."""
        type_thresholds = {
            "skill_play": self.skill_play,
            "clutch_moment": self.clutch_moment,
            "comeback": self.comeback,
            "rare_event": self.rare_event,
            "perfect_execution": self.perfect_execution,
            "manual": self.manual_trigger,
            "general": self.minimum_viable,
        }

        threshold = type_thresholds.get(highlight_type, self.minimum_viable)
        return score >= threshold


@dataclass(frozen=True)
class ScoringDimensions:
    """Individual dimension scores for a highlight candidate."""

    skill_execution: float = 0.0
    game_impact: float = 0.0
    rarity: float = 0.0
    visual_spectacle: float = 0.0
    emotional_intensity: float = 0.0
    narrative_value: float = 0.0
    timing_importance: float = 0.0
    momentum_shift: float = 0.0

    def __post_init__(self) -> None:
        """Validate all dimensions are in valid range."""
        dimensions = [
            self.skill_execution,
            self.game_impact,
            self.rarity,
            self.visual_spectacle,
            self.emotional_intensity,
            self.narrative_value,
            self.timing_importance,
            self.momentum_shift,
        ]

        for dim in dimensions:
            if not (0.0 <= dim <= 1.0):
                raise ValueError(
                    f"All dimension scores must be between 0.0 and 1.0, got {dim}"
                )

    def calculate_weighted_score(self, weights: DimensionWeights) -> float:
        """Calculate the final weighted score using the provided weights."""
        return (
            self.skill_execution * weights.skill_execution
            + self.game_impact * weights.game_impact
            + self.rarity * weights.rarity
            + self.visual_spectacle * weights.visual_spectacle
            + self.emotional_intensity * weights.emotional_intensity
            + self.narrative_value * weights.narrative_value
            + self.timing_importance * weights.timing_importance
            + self.momentum_shift * weights.momentum_shift
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "skill_execution": self.skill_execution,
            "game_impact": self.game_impact,
            "rarity": self.rarity,
            "visual_spectacle": self.visual_spectacle,
            "emotional_intensity": self.emotional_intensity,
            "narrative_value": self.narrative_value,
            "timing_importance": self.timing_importance,
            "momentum_shift": self.momentum_shift,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ScoringDimensions":
        """Create from dictionary."""
        return cls(
            skill_execution=data.get("skill_execution", 0.0),
            game_impact=data.get("game_impact", 0.0),
            rarity=data.get("rarity", 0.0),
            visual_spectacle=data.get("visual_spectacle", 0.0),
            emotional_intensity=data.get("emotional_intensity", 0.0),
            narrative_value=data.get("narrative_value", 0.0),
            timing_importance=data.get("timing_importance", 0.0),
            momentum_shift=data.get("momentum_shift", 0.0),
        )
