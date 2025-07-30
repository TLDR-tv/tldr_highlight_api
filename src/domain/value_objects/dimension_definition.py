"""Flexible dimension definition value object for customizable highlight detection.

This value object allows clients to define their own scoring dimensions
for highlight detection, moving beyond fixed gaming-oriented dimensions.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from src.domain.exceptions import InvalidValueError


class DimensionType(str, Enum):
    """Types of dimension scoring methods."""

    NUMERIC = "numeric"  # 0.0 to 1.0 scale
    BINARY = "binary"  # True/False (0 or 1)
    CATEGORICAL = "categorical"  # Multiple discrete values
    SEMANTIC = "semantic"  # Natural language evaluation


class AggregationMethod(str, Enum):
    """How to aggregate multiple signals for this dimension."""

    MAX = "max"  # Take highest signal
    AVERAGE = "average"  # Average all signals
    WEIGHTED = "weighted"  # Weighted average
    CONSENSUS = "consensus"  # Majority agreement
    CUSTOM = "custom"  # Custom aggregation function


@dataclass(frozen=True)
class DimensionDefinition:
    """Defines a single scoring dimension for highlight detection.

    This immutable value object encapsulates all information needed
    to score content along a specific dimension, allowing complete
    customization of what aspects matter for highlight detection.
    """

    # Core identification
    id: str  # Unique identifier (e.g., "skill_execution", "humor_level")
    name: str  # Human-readable name
    description: str  # Detailed description for AI and humans

    # Scoring configuration
    dimension_type: DimensionType = DimensionType.NUMERIC
    default_weight: float = 0.1  # Default importance weight

    # Value constraints
    min_value: float = 0.0  # Minimum possible value
    max_value: float = 1.0  # Maximum possible value
    threshold: float = 0.5  # Minimum value to be considered significant

    # AI instructions
    scoring_prompt: str = ""  # Instructions for AI to score this dimension
    examples: List[Dict[str, Any]] = field(default_factory=list)  # Example scores

    # Multi-modal configuration
    applicable_modalities: List[str] = field(
        default_factory=lambda: ["video", "audio", "text"]
    )
    aggregation_method: AggregationMethod = AggregationMethod.MAX

    # Optional metadata
    category: Optional[str] = None  # Grouping category (e.g., "technical", "emotional")
    tags: List[str] = field(default_factory=list)  # Searchable tags
    industry: Optional[str] = None  # Industry this dimension is designed for

    def __post_init__(self) -> None:
        """Validate dimension definition."""
        # Validate ID format
        if not self.id or not self.id.replace("_", "").isalnum():
            raise InvalidValueError(
                f"Dimension ID must be alphanumeric with underscores, got '{self.id}'"
            )

        # Validate weight
        if not 0.0 <= self.default_weight <= 1.0:
            raise InvalidValueError(
                f"Default weight must be between 0.0 and 1.0, got {self.default_weight}"
            )

        # Validate value range
        if self.min_value >= self.max_value:
            raise InvalidValueError(
                f"Min value ({self.min_value}) must be less than max value ({self.max_value})"
            )

        if not self.min_value <= self.threshold <= self.max_value:
            raise InvalidValueError(
                f"Threshold ({self.threshold}) must be between min ({self.min_value}) and max ({self.max_value})"
            )

        # Validate modalities
        valid_modalities = {"video", "audio", "text", "metadata", "social"}
        invalid_modalities = set(self.applicable_modalities) - valid_modalities
        if invalid_modalities:
            raise InvalidValueError(f"Invalid modalities: {invalid_modalities}")

        # Validate examples structure
        for example in self.examples:
            if not isinstance(example, dict):
                raise InvalidValueError("Examples must be dictionaries")
            if "value" not in example or "description" not in example:
                raise InvalidValueError(
                    "Examples must have 'value' and 'description' keys"
                )

    def normalize_value(self, raw_value: float) -> float:
        """Normalize a raw value to 0.0-1.0 range."""
        if self.dimension_type == DimensionType.BINARY:
            return 1.0 if raw_value else 0.0

        # Clamp to valid range
        clamped = max(self.min_value, min(self.max_value, raw_value))

        # Normalize to 0-1
        if self.max_value - self.min_value > 0:
            return (clamped - self.min_value) / (self.max_value - self.min_value)
        return 0.0

    def meets_threshold(self, value: float) -> bool:
        """Check if a value meets the significance threshold."""
        normalized = self.normalize_value(value)
        normalized_threshold = self.normalize_value(self.threshold)
        return normalized >= normalized_threshold

    def generate_ai_instruction(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate AI scoring instructions for this dimension."""
        instruction = f"Score the '{self.name}' dimension:\n"
        instruction += f"Description: {self.description}\n"

        if self.scoring_prompt:
            instruction += f"Scoring guidance: {self.scoring_prompt}\n"

        if self.examples:
            instruction += "\nExamples:\n"
            for example in self.examples[:3]:  # Limit to 3 examples
                instruction += f"- {example['description']}: {example['value']}\n"

        instruction += (
            f"\nProvide a score between {self.min_value} and {self.max_value}"
        )

        if self.dimension_type == DimensionType.BINARY:
            instruction += " (0 for false, 1 for true)"
        elif self.dimension_type == DimensionType.CATEGORICAL:
            instruction += " based on the category that best fits"

        return instruction

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "dimension_type": self.dimension_type.value,
            "default_weight": self.default_weight,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "threshold": self.threshold,
            "scoring_prompt": self.scoring_prompt,
            "examples": self.examples,
            "applicable_modalities": self.applicable_modalities,
            "aggregation_method": self.aggregation_method.value,
            "category": self.category,
            "tags": self.tags,
            "industry": self.industry,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DimensionDefinition":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            dimension_type=DimensionType(data.get("dimension_type", "numeric")),
            default_weight=data.get("default_weight", 0.1),
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 1.0),
            threshold=data.get("threshold", 0.5),
            scoring_prompt=data.get("scoring_prompt", ""),
            examples=data.get("examples", []),
            applicable_modalities=data.get(
                "applicable_modalities", ["video", "audio", "text"]
            ),
            aggregation_method=AggregationMethod(data.get("aggregation_method", "max")),
            category=data.get("category"),
            tags=data.get("tags", []),
            industry=data.get("industry"),
        )

    # Factory methods for common dimension types

    @classmethod
    def create_skill_dimension(
        cls, id: str, name: str, description: str, skill_prompt: str = ""
    ) -> "DimensionDefinition":
        """Create a skill/performance-based dimension."""
        return cls(
            id=id,
            name=name,
            description=description,
            dimension_type=DimensionType.NUMERIC,
            default_weight=0.2,
            scoring_prompt=skill_prompt
            or f"Evaluate the level of {name.lower()} demonstrated. Consider technical difficulty, precision, and execution quality.",
            category="technical",
            examples=[
                {
                    "value": 0.9,
                    "description": "Expert-level execution with perfect precision",
                },
                {
                    "value": 0.5,
                    "description": "Competent performance with minor imperfections",
                },
                {"value": 0.1, "description": "Basic execution with noticeable errors"},
            ],
        )

    @classmethod
    def create_emotional_dimension(
        cls, id: str, name: str, description: str, emotion_prompt: str = ""
    ) -> "DimensionDefinition":
        """Create an emotion/reaction-based dimension."""
        return cls(
            id=id,
            name=name,
            description=description,
            dimension_type=DimensionType.NUMERIC,
            default_weight=0.15,
            scoring_prompt=emotion_prompt
            or f"Assess the {name.lower()} of this moment. Consider audience reaction potential and emotional resonance.",
            category="emotional",
            applicable_modalities=["video", "audio", "text"],
            examples=[
                {"value": 1.0, "description": "Extremely intense emotional response"},
                {"value": 0.6, "description": "Noticeable emotional engagement"},
                {"value": 0.2, "description": "Mild emotional reaction"},
            ],
        )

    @classmethod
    def create_contextual_dimension(
        cls, id: str, name: str, description: str, context_prompt: str = ""
    ) -> "DimensionDefinition":
        """Create a context/situation-based dimension."""
        return cls(
            id=id,
            name=name,
            description=description,
            dimension_type=DimensionType.NUMERIC,
            default_weight=0.1,
            scoring_prompt=context_prompt
            or f"Evaluate the {name.lower()} given the current context and situation.",
            category="contextual",
            aggregation_method=AggregationMethod.WEIGHTED,
            examples=[
                {"value": 0.8, "description": "Highly significant in current context"},
                {"value": 0.4, "description": "Moderate contextual importance"},
                {"value": 0.0, "description": "No contextual relevance"},
            ],
        )

    @classmethod
    def create_binary_dimension(
        cls, id: str, name: str, description: str, detection_prompt: str = ""
    ) -> "DimensionDefinition":
        """Create a binary (yes/no) dimension."""
        return cls(
            id=id,
            name=name,
            description=description,
            dimension_type=DimensionType.BINARY,
            default_weight=0.2,
            min_value=0.0,
            max_value=1.0,
            threshold=0.5,
            scoring_prompt=detection_prompt
            or f"Determine if this moment exhibits {name.lower()} (1 for yes, 0 for no).",
            category="binary",
            examples=[
                {"value": 1.0, "description": f"Clear presence of {name.lower()}"},
                {"value": 0.0, "description": f"No {name.lower()} detected"},
            ],
        )
