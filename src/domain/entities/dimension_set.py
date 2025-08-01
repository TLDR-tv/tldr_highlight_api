"""Dimension Set entity - simplified and Pythonic.

This module implements the Dimension Set as a proper aggregate root
with clean, Pythonic naming and structure.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..entities.base import AggregateRoot
from ..value_objects.timestamp import Timestamp
from ..value_objects.dimension_definition import DimensionDefinition
from ..value_objects.dimension_weight import DimensionWeight
from ..value_objects.dimension_score import DimensionScore
from ..exceptions import BusinessRuleViolation


@dataclass(kw_only=True)
class DimensionSet(AggregateRoot[int]):
    """Manages a set of dimensions for highlight detection.
    
    This is an aggregate root that ensures consistency of dimensions
    and their weights within a set.
    """
    
    # Identity
    organization_id: int
    name: str
    description: str = ""
    
    # Domain data
    dimensions: Dict[str, DimensionDefinition] = field(default_factory=dict)
    weights: Dict[str, DimensionWeight] = field(default_factory=dict)
    
    # Configuration
    industry: Optional[str] = None
    is_active: bool = True
    is_default: bool = False
    version: int = 1
    
    # Constraints
    minimum_dimensions_required: int = 1
    normalize_weights: bool = True
    
    def add_dimension(
        self,
        dimension: DimensionDefinition,
        weight: float = 1.0
    ) -> None:
        """Add a dimension to this set.
        
        Enforces business rules:
        - No duplicate dimensions
        - Maximum 20 dimensions
        - Valid weight range
        """
        if dimension.id in self.dimensions:
            raise BusinessRuleViolation(
                f"Dimension {dimension.id} already exists in this set"
            )
        
        if len(self.dimensions) >= 20:
            raise BusinessRuleViolation(
                "Cannot add more than 20 dimensions to a set"
            )
        
        # Add dimension and weight
        self.dimensions[dimension.id] = dimension
        self.weights[dimension.id] = DimensionWeight(dimension.id, weight)
        
        # Normalize weights if configured
        if self.normalize_weights:
            self._normalize_weights()
        
        self.updated_at = Timestamp.now()
        self.version += 1
    
    def remove_dimension(self, dimension_id: str) -> None:
        """Remove a dimension from this set."""
        if dimension_id not in self.dimensions:
            raise BusinessRuleViolation(
                f"Dimension {dimension_id} not found in this set"
            )
        
        # Check minimum dimensions
        if len(self.dimensions) <= self.minimum_dimensions_required:
            raise BusinessRuleViolation(
                f"Cannot remove dimension - need at least "
                f"{self.minimum_dimensions_required} dimensions"
            )
        
        # Remove dimension and weight
        del self.dimensions[dimension_id]
        del self.weights[dimension_id]
        
        # Re-normalize if needed
        if self.normalize_weights and self.dimensions:
            self._normalize_weights()
        
        self.updated_at = Timestamp.now()
        self.version += 1
    
    def update_weight(self, dimension_id: str, new_weight: float) -> None:
        """Update the weight of a dimension."""
        if dimension_id not in self.dimensions:
            raise BusinessRuleViolation(
                f"Dimension {dimension_id} not found in this set"
            )
        
        self.weights[dimension_id] = DimensionWeight(dimension_id, new_weight)
        
        if self.normalize_weights:
            self._normalize_weights()
        
        self.updated_at = Timestamp.now()
    
    def calculate_score(
        self,
        dimension_scores: Dict[str, float]
    ) -> float:
        """Calculate weighted score from dimension scores.
        
        Args:
            dimension_scores: Map of dimension ID to score value
            
        Returns:
            Weighted score between 0.0 and 1.0
        """
        if not dimension_scores:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for dim_id, score in dimension_scores.items():
            if dim_id in self.weights:
                weight = self.weights[dim_id].value
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def evaluate_scores(
        self,
        dimension_scores: Dict[str, DimensionScore]
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate dimension scores and return final score with breakdown.
        
        Returns:
            Tuple of (final_score, weighted_scores_by_dimension)
        """
        # Convert to simple scores
        simple_scores = {
            dim_id: score.value
            for dim_id, score in dimension_scores.items()
        }
        
        # Calculate weighted scores
        weighted_scores = {}
        for dim_id, score in simple_scores.items():
            if dim_id in self.weights:
                weighted_scores[dim_id] = score * self.weights[dim_id].value
        
        final_score = self.calculate_score(simple_scores)
        
        return final_score, weighted_scores
    
    @property
    def dimension_count(self) -> int:
        """Number of dimensions in this set."""
        return len(self.dimensions)
    
    @property
    def required_modalities(self) -> Set[str]:
        """Get modalities required by dimensions in this set."""
        modalities = set()
        for dimension in self.dimensions.values():
            if hasattr(dimension, 'required_modalities'):
                modalities.update(dimension.required_modalities)
        return modalities
    
    @property
    def scoring_dimensions(self) -> List[Tuple[DimensionDefinition, float]]:
        """Get dimensions sorted by weight (highest first)."""
        result = []
        for dim_id, dimension in self.dimensions.items():
            weight = self.weights.get(dim_id, DimensionWeight(dim_id, 1.0)).value
            result.append((dimension, weight))
        
        return sorted(result, key=lambda x: x[1], reverse=True)
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        if not self.weights:
            return
        
        total = sum(w.value for w in self.weights.values())
        if total == 0:
            return
        
        for dim_id in self.weights:
            current = self.weights[dim_id].value
            normalized = current / total
            self.weights[dim_id] = DimensionWeight(dim_id, normalized)
    
    def is_valid(self) -> bool:
        """Check if this dimension set is valid for use."""
        if not self.is_active:
            return False
        
        if len(self.dimensions) < self.minimum_dimensions_required:
            return False
        
        # All dimensions must have weights
        for dim_id in self.dimensions:
            if dim_id not in self.weights:
                return False
        
        return True
    
    @classmethod
    def create_default(
        cls,
        organization_id: int,
        industry: str = "gaming"
    ) -> "DimensionSet":
        """Create a default dimension set for an industry."""
        dimension_set = cls(
            id=None,
            organization_id=organization_id,
            name=f"Default {industry.title()} Set",
            description=f"Default dimension set for {industry}",
            industry=industry,
            is_default=True,
        )
        
        # Add default dimensions based on industry
        if industry == "gaming":
            dimension_set.add_dimension(
                DimensionDefinition(
                    id="action_intensity",
                    name="Action Intensity",
                    type="numeric",
                    description="Level of action and excitement",
                    weight=0.3,
                )
            )
            dimension_set.add_dimension(
                DimensionDefinition(
                    id="skill_display",
                    name="Skill Display",
                    type="numeric",
                    description="Demonstration of player skill",
                    weight=0.3,
                )
            )
            dimension_set.add_dimension(
                DimensionDefinition(
                    id="emotional_peak",
                    name="Emotional Peak",
                    type="numeric",
                    description="Emotional intensity of the moment",
                    weight=0.2,
                )
            )
        
        return dimension_set


# Backward compatibility alias
DimensionSetAggregate = DimensionSet