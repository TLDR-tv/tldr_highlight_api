"""Value object representing dimension scores."""

from dataclasses import dataclass, field
from typing import Dict

from ..exceptions import InvalidValueError


@dataclass(frozen=True)
class DimensionScores:
    """Value object representing scores for multiple dimensions.
    
    This encapsulates the scoring of content across various dimensions
    defined in a DimensionSet.
    """
    
    scores: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate dimension scores."""
        for dimension_id, score in self.scores.items():
            if not isinstance(dimension_id, str):
                raise InvalidValueError(f"Dimension ID must be string, got {type(dimension_id)}")
            
            if not isinstance(score, (int, float)):
                raise InvalidValueError(f"Score must be numeric, got {type(score)}")
            
            if not 0.0 <= score <= 1.0:
                raise InvalidValueError(f"Score must be between 0.0 and 1.0, got {score}")
    
    def get_score(self, dimension_id: str) -> float:
        """Get score for a specific dimension.
        
        Args:
            dimension_id: ID of the dimension
            
        Returns:
            Score value, or 0.0 if dimension not scored
        """
        return self.scores.get(dimension_id, 0.0)
    
    def has_dimension(self, dimension_id: str) -> bool:
        """Check if a dimension has been scored.
        
        Args:
            dimension_id: ID of the dimension
            
        Returns:
            True if the dimension has a score
        """
        return dimension_id in self.scores
    
    @property
    def average_score(self) -> float:
        """Calculate the average score across all dimensions."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)
    
    @property
    def max_score(self) -> float:
        """Get the maximum score across all dimensions."""
        if not self.scores:
            return 0.0
        return max(self.scores.values())
    
    @property
    def min_score(self) -> float:
        """Get the minimum score across all dimensions."""
        if not self.scores:
            return 0.0
        return min(self.scores.values())
    
    def weighted_average(self, weights: Dict[str, float]) -> float:
        """Calculate weighted average score.
        
        Args:
            weights: Dictionary mapping dimension IDs to weights
            
        Returns:
            Weighted average score
        """
        total_weight = 0.0
        weighted_sum = 0.0
        
        for dimension_id, weight in weights.items():
            if dimension_id in self.scores:
                weighted_sum += self.scores[dimension_id] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return dict(self.scores)
    
    def __str__(self) -> str:
        """String representation."""
        if not self.scores:
            return "DimensionScores(empty)"
        
        parts = []
        for dim_id, score in sorted(self.scores.items()):
            parts.append(f"{dim_id}={score:.2f}")
        
        return f"DimensionScores({', '.join(parts)})"