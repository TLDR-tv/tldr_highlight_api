"""Dimension set entity for grouping related scoring dimensions.

This entity allows clients to create and manage sets of dimensions
for different content types, industries, or use cases.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

from ..entities.base import Entity
from ..value_objects.dimension_definition import DimensionDefinition, DimensionType
from ..exceptions import BusinessRuleViolation, InvalidValueError


@dataclass
class DimensionSet(Entity[int]):
    """A curated set of dimensions for highlight detection.
    
    This entity groups related dimensions together and manages their
    weights and relationships for specific use cases.
    """
    
    # Basic identification
    name: str
    description: str
    organization_id: int
    created_by_user_id: int
    
    # Dimension configuration
    dimensions: Dict[str, DimensionDefinition] = field(default_factory=dict)
    dimension_weights: Dict[str, float] = field(default_factory=dict)
    
    # Set metadata
    industry: Optional[str] = None
    content_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Configuration
    allow_partial_scoring: bool = True  # Allow highlights even if some dimensions can't be scored
    minimum_dimensions_required: int = 3  # Minimum dimensions that must be scored
    weight_normalization: bool = True  # Auto-normalize weights to sum to 1.0
    
    # Usage tracking
    is_active: bool = True
    is_public: bool = False  # Can other organizations use this set?
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Initialize and validate dimension set."""
        super().__post_init__()
        
        # Initialize weights if not provided
        if not self.dimension_weights and self.dimensions:
            self.dimension_weights = {
                dim_id: dim.default_weight 
                for dim_id, dim in self.dimensions.items()
            }
        
        # Normalize weights if enabled
        if self.weight_normalization:
            self._normalize_weights()
        
        # Validate the set
        self._validate()
    
    def add_dimension(
        self,
        dimension: DimensionDefinition,
        weight: Optional[float] = None
    ) -> None:
        """Add a dimension to the set.
        
        Args:
            dimension: The dimension definition to add
            weight: Optional weight override (uses dimension default if not provided)
            
        Raises:
            BusinessRuleViolation: If dimension ID already exists
        """
        if dimension.id in self.dimensions:
            raise BusinessRuleViolation(
                f"Dimension '{dimension.id}' already exists in this set"
            )
        
        self.dimensions[dimension.id] = dimension
        self.dimension_weights[dimension.id] = weight or dimension.default_weight
        
        if self.weight_normalization:
            self._normalize_weights()
        
        self.updated_at = datetime.utcnow()
    
    def remove_dimension(self, dimension_id: str) -> None:
        """Remove a dimension from the set.
        
        Args:
            dimension_id: ID of the dimension to remove
            
        Raises:
            BusinessRuleViolation: If dimension doesn't exist or removal would violate minimum
        """
        if dimension_id not in self.dimensions:
            raise BusinessRuleViolation(
                f"Dimension '{dimension_id}' not found in this set"
            )
        
        if len(self.dimensions) - 1 < self.minimum_dimensions_required:
            raise BusinessRuleViolation(
                f"Cannot remove dimension. Set requires at least {self.minimum_dimensions_required} dimensions"
            )
        
        del self.dimensions[dimension_id]
        del self.dimension_weights[dimension_id]
        
        if self.weight_normalization and self.dimensions:
            self._normalize_weights()
        
        self.updated_at = datetime.utcnow()
    
    def update_weight(self, dimension_id: str, new_weight: float) -> None:
        """Update the weight for a specific dimension.
        
        Args:
            dimension_id: ID of the dimension to update
            new_weight: New weight value
            
        Raises:
            InvalidValueError: If weight is invalid
            BusinessRuleViolation: If dimension doesn't exist
        """
        if not 0.0 <= new_weight <= 1.0:
            raise InvalidValueError(f"Weight must be between 0.0 and 1.0, got {new_weight}")
        
        if dimension_id not in self.dimensions:
            raise BusinessRuleViolation(
                f"Dimension '{dimension_id}' not found in this set"
            )
        
        self.dimension_weights[dimension_id] = new_weight
        
        if self.weight_normalization:
            self._normalize_weights()
        
        self.updated_at = datetime.utcnow()
    
    def update_all_weights(self, weights: Dict[str, float]) -> None:
        """Update multiple dimension weights at once.
        
        Args:
            weights: Dictionary mapping dimension IDs to weights
            
        Raises:
            BusinessRuleViolation: If any dimension ID doesn't exist
        """
        # Validate all dimensions exist
        invalid_dims = set(weights.keys()) - set(self.dimensions.keys())
        if invalid_dims:
            raise BusinessRuleViolation(
                f"Unknown dimensions: {invalid_dims}"
            )
        
        # Update weights
        self.dimension_weights.update(weights)
        
        if self.weight_normalization:
            self._normalize_weights()
        
        self.updated_at = datetime.utcnow()
    
    def get_scoring_dimensions(self) -> List[Tuple[DimensionDefinition, float]]:
        """Get all dimensions with their weights for scoring.
        
        Returns:
            List of (dimension, weight) tuples sorted by weight
        """
        result = [
            (self.dimensions[dim_id], self.dimension_weights[dim_id])
            for dim_id in self.dimensions
            if self.dimension_weights.get(dim_id, 0) > 0
        ]
        
        return sorted(result, key=lambda x: x[1], reverse=True)
    
    def get_required_modalities(self) -> Set[str]:
        """Get all modalities required by dimensions in this set.
        
        Returns:
            Set of modality names (e.g., {"video", "audio", "text"})
        """
        modalities = set()
        for dimension in self.dimensions.values():
            modalities.update(dimension.applicable_modalities)
        return modalities
    
    def calculate_score(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate weighted score from individual dimension scores.
        
        Args:
            dimension_scores: Dictionary mapping dimension IDs to scores
            
        Returns:
            Weighted score between 0.0 and 1.0
            
        Raises:
            BusinessRuleViolation: If insufficient dimensions are scored
        """
        # Check minimum dimensions
        scored_dimensions = set(dimension_scores.keys()) & set(self.dimensions.keys())
        if len(scored_dimensions) < self.minimum_dimensions_required and not self.allow_partial_scoring:
            raise BusinessRuleViolation(
                f"Insufficient dimensions scored. Required: {self.minimum_dimensions_required}, got: {len(scored_dimensions)}"
            )
        
        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0
        
        for dim_id in scored_dimensions:
            dimension = self.dimensions[dim_id]
            score = dimension.normalize_value(dimension_scores[dim_id])
            weight = self.dimension_weights[dim_id]
            
            total_score += score * weight
            total_weight += weight
        
        # Return normalized score
        if total_weight > 0:
            return min(1.0, total_score / total_weight)
        return 0.0
    
    def validate_configuration(self) -> List[str]:
        """Validate the dimension set configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check minimum dimensions
        if len(self.dimensions) < self.minimum_dimensions_required:
            errors.append(
                f"Set has {len(self.dimensions)} dimensions but requires at least {self.minimum_dimensions_required}"
            )
        
        # Check weight configuration
        if self.weight_normalization:
            total_weight = sum(self.dimension_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                errors.append(f"Weights sum to {total_weight:.3f} but should sum to 1.0")
        
        # Check for zero weights
        zero_weight_dims = [
            dim_id for dim_id, weight in self.dimension_weights.items()
            if weight == 0.0
        ]
        if zero_weight_dims:
            errors.append(f"Dimensions with zero weight: {zero_weight_dims}")
        
        # Validate individual dimensions
        for dim_id, dimension in self.dimensions.items():
            try:
                # Trigger dimension validation
                _ = dimension.to_dict()
            except Exception as e:
                errors.append(f"Invalid dimension '{dim_id}': {str(e)}")
        
        return errors
    
    def clone(self, new_organization_id: int, new_user_id: int) -> "DimensionSet":
        """Create a copy of this dimension set for another organization.
        
        Args:
            new_organization_id: ID of the new organization
            new_user_id: ID of the user creating the clone
            
        Returns:
            New dimension set instance
        """
        return DimensionSet(
            id=None,
            name=f"{self.name} (Copy)",
            description=f"Copy of {self.name}",
            organization_id=new_organization_id,
            created_by_user_id=new_user_id,
            dimensions=self.dimensions.copy(),
            dimension_weights=self.dimension_weights.copy(),
            industry=self.industry,
            content_type=self.content_type,
            tags=self.tags.copy(),
            allow_partial_scoring=self.allow_partial_scoring,
            minimum_dimensions_required=self.minimum_dimensions_required,
            weight_normalization=self.weight_normalization,
            is_active=True,
            is_public=False,
            usage_count=0,
            last_used_at=None
        )
    
    def record_usage(self) -> None:
        """Record that this dimension set was used."""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        if not self.dimension_weights:
            return
        
        total_weight = sum(self.dimension_weights.values())
        if total_weight > 0:
            for dim_id in self.dimension_weights:
                self.dimension_weights[dim_id] /= total_weight
    
    def _validate(self) -> None:
        """Validate the dimension set."""
        errors = self.validate_configuration()
        if errors:
            raise BusinessRuleViolation(
                f"Invalid dimension set configuration: {'; '.join(errors)}"
            )
    
    # Factory methods for common dimension sets
    
    @classmethod
    def create_gaming_set(cls, organization_id: int, user_id: int) -> "DimensionSet":
        """Create a dimension set optimized for gaming content."""
        dimensions = {
            "skill_execution": DimensionDefinition.create_skill_dimension(
                "skill_execution",
                "Skill Execution",
                "Technical skill and precision demonstrated in gameplay"
            ),
            "game_impact": DimensionDefinition(
                id="game_impact",
                name="Game Impact",
                description="Effect on match outcome or game state",
                default_weight=0.2,
                category="outcome"
            ),
            "rarity": DimensionDefinition(
                id="rarity",
                name="Rarity",
                description="How uncommon or special this moment is",
                default_weight=0.15,
                category="uniqueness"
            ),
            "excitement": DimensionDefinition.create_emotional_dimension(
                "excitement",
                "Excitement Level",
                "Hype and excitement generated by this moment"
            ),
            "clutch_factor": DimensionDefinition.create_contextual_dimension(
                "clutch_factor",
                "Clutch Factor",
                "Performance under pressure in critical moments"
            )
        }
        
        return cls(
            id=None,
            name="Gaming Highlights",
            description="Optimized for detecting highlights in gaming content",
            organization_id=organization_id,
            created_by_user_id=user_id,
            dimensions=dimensions,
            industry="gaming",
            content_type="gaming",
            tags=["gaming", "esports", "gameplay"],
            minimum_dimensions_required=3
        )
    
    @classmethod
    def create_educational_set(cls, organization_id: int, user_id: int) -> "DimensionSet":
        """Create a dimension set for educational content."""
        dimensions = {
            "concept_clarity": DimensionDefinition(
                id="concept_clarity",
                name="Concept Clarity",
                description="How clearly a concept is explained or demonstrated",
                default_weight=0.25,
                category="educational"
            ),
            "engagement_level": DimensionDefinition.create_emotional_dimension(
                "engagement_level",
                "Student Engagement",
                "Level of audience engagement and participation"
            ),
            "key_moment": DimensionDefinition.create_binary_dimension(
                "key_moment",
                "Key Learning Moment",
                "Whether this is a crucial learning point"
            ),
            "question_answered": DimensionDefinition.create_binary_dimension(
                "question_answered",
                "Question Answered",
                "Whether an important question was addressed"
            ),
            "visual_demonstration": DimensionDefinition(
                id="visual_demonstration",
                name="Visual Demonstration",
                description="Quality of visual explanation or demonstration",
                default_weight=0.15,
                applicable_modalities=["video"],
                category="presentation"
            )
        }
        
        return cls(
            id=None,
            name="Educational Highlights",
            description="Optimized for educational and instructional content",
            organization_id=organization_id,
            created_by_user_id=user_id,
            dimensions=dimensions,
            industry="education",
            content_type="educational",
            tags=["education", "learning", "instruction"],
            minimum_dimensions_required=2
        )