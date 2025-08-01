"""Dimension Set Aggregate Root following DDD principles.

This module implements the Dimension Set as a proper aggregate root,
ensuring consistency and encapsulating all business rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable

from ..entities.base import AggregateRoot
from ..value_objects.timestamp import Timestamp
from ..value_objects.dimension_definition import DimensionDefinition
from ..value_objects.dimension_weight import DimensionWeight
from ..value_objects.dimension_score import DimensionScore
from ..value_objects.dimension_set_version import DimensionSetVersion
from ..events import (
    DimensionAddedToDimensionSetEvent,
    DimensionRemovedFromDimensionSetEvent,
    DimensionWeightUpdatedEvent,
    DimensionSetUsedForEvaluationEvent,
    DimensionSetVersionUpdatedEvent,
)
from ..exceptions import BusinessRuleViolation
from ..policies.dimension_set_policy import (
    can_add_dimension,
    can_remove_dimension,
    create_dimension_set_validators,
)


# Configuration for dimension set constraints
@dataclass
class DimensionSetConfig:
    """Configuration for dimension set business rules."""

    min_dimensions: int = 3
    max_dimensions: int = 20
    require_normalized_weights: bool = True


@dataclass
class DimensionSetAggregate(AggregateRoot):
    """Aggregate root for managing dimension sets.

    This aggregate ensures all business rules are enforced and maintains
    consistency across dimension definitions and their weights.
    """

    # Identity and ownership
    organization_id: int
    created_by_user_id: int

    # Core attributes
    name: str
    description: str

    # Version tracking
    _version: DimensionSetVersion = field(
        default_factory=lambda: DimensionSetVersion(
            major=1, minor=0, patch=0, effective_date=Timestamp.now().value
        )
    )
    _version_history: List[DimensionSetVersion] = field(default_factory=list)

    # Dimensions and weights (internal state)
    _dimensions: Dict[str, DimensionDefinition] = field(default_factory=dict)
    _weights: Dict[str, DimensionWeight] = field(default_factory=dict)

    # Metadata
    industry: Optional[str] = None
    content_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Configuration
    is_active: bool = True
    is_public: bool = False
    _config: DimensionSetConfig = field(default_factory=DimensionSetConfig)

    # Validators (created from config)
    _validators: Dict[str, Callable] = field(init=False)

    # Timestamps
    created_at: Timestamp = field(default_factory=Timestamp.now)
    updated_at: Timestamp = field(default_factory=Timestamp.now)
    last_used_at: Optional[Timestamp] = None

    def __post_init__(self) -> None:
        """Initialize validators based on configuration."""
        self._validators = create_dimension_set_validators(
            min_dimensions=self._config.min_dimensions,
            max_dimensions=self._config.max_dimensions,
            require_normalized_weights=self._config.require_normalized_weights,
        )

    @property
    def dimensions(self) -> Dict[str, DimensionDefinition]:
        """Get immutable view of dimensions."""
        return self._dimensions.copy()

    @property
    def weights(self) -> Dict[str, DimensionWeight]:
        """Get immutable view of weights."""
        return self._weights.copy()

    @property
    def version(self) -> DimensionSetVersion:
        """Get current version."""
        return self._version

    @property
    def version_history(self) -> List[DimensionSetVersion]:
        """Get version history."""
        return self._version_history.copy()

    def add_dimension(
        self, dimension: DimensionDefinition, weight: Optional[float] = None
    ) -> None:
        """Add a dimension to the set.

        This method enforces business rules and raises domain events.
        """
        # Check business rules
        can_add_dimension(self, dimension, self._config.max_dimensions)

        # Add dimension
        self._dimensions[dimension.id] = dimension

        # Set weight
        weight_value = weight if weight is not None else dimension.default_weight
        self._weights[dimension.id] = DimensionWeight(dimension.id, weight_value)

        # Update timestamp
        self.updated_at = Timestamp.now()

        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                DimensionAddedToDimensionSetEvent(
                    dimension_set_id=self.id,
                    dimension_id=dimension.id,
                    dimension_name=dimension.name,
                    weight=weight_value,
                    version=self._version.version_string,
                )
            )

    def remove_dimension(self, dimension_id: str) -> None:
        """Remove a dimension from the set."""
        # Check business rules
        can_remove_dimension(self, dimension_id, self._config.min_dimensions)

        # Remove dimension and weight
        del self._dimensions[dimension_id]
        del self._weights[dimension_id]

        # Update timestamp
        self.updated_at = Timestamp.now()

        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                DimensionRemovedFromDimensionSetEvent(
                    dimension_set_id=self.id, dimension_id=dimension_id
                )
            )

    def update_weight(self, dimension_id: str, new_weight: float) -> None:
        """Update the weight for a specific dimension."""
        if dimension_id not in self._dimensions:
            raise BusinessRuleViolation(
                f"Dimension '{dimension_id}' not found in this set"
            )

        old_weight = self._weights[dimension_id].value
        self._weights[dimension_id] = DimensionWeight(dimension_id, new_weight)

        # Validate new weight configuration
        errors = self._validators["validate_weights"](self._weights)
        if errors:
            # Rollback
            self._weights[dimension_id] = DimensionWeight(dimension_id, old_weight)
            raise BusinessRuleViolation(
                f"Invalid weight configuration: {'; '.join(errors)}"
            )

        # Update timestamp
        self.updated_at = Timestamp.now()

        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                DimensionWeightUpdatedEvent(
                    dimension_set_id=self.id,
                    dimension_id=dimension_id,
                    old_weight=old_weight,
                    new_weight=new_weight,
                )
            )

    def calculate_highlight_score(
        self, dimension_scores: Dict[str, DimensionScore]
    ) -> float:
        """Calculate weighted highlight score from dimension scores.

        This is a core domain operation that encapsulates the scoring logic.
        """
        total_score = 0.0
        total_weight = 0.0

        for dim_id, score in dimension_scores.items():
            if dim_id in self._dimensions and dim_id in self._weights:
                weight = self._weights[dim_id]
                total_score += score.value * weight.value
                total_weight += weight.value

        # Return normalized score
        if total_weight > 0:
            return min(1.0, total_score / total_weight)
        return 0.0

    def meets_evaluation_criteria(
        self,
        dimension_scores: Dict[str, DimensionScore],
        min_dimensions_required: int = 3,
    ) -> bool:
        """Check if evaluation meets minimum criteria."""
        scored_dimensions = set(dimension_scores.keys()) & set(self._dimensions.keys())

        if len(scored_dimensions) < min_dimensions_required:
            return False

        # Check each dimension meets its threshold
        for dim_id in scored_dimensions:
            dimension = self._dimensions[dim_id]
            score = dimension_scores[dim_id]
            if not dimension.meets_threshold(score.value):
                return False

        return True

    def record_usage(self, usage_context: str = "evaluation") -> None:
        """Record that this dimension set was used."""
        self.last_used_at = Timestamp.now()

        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                DimensionSetUsedForEvaluationEvent(
                    dimension_set_id=self.id, organization_id=self.organization_id
                )
            )

    def get_required_modalities(self) -> Set[str]:
        """Get all modalities required by dimensions in this set."""
        modalities = set()
        for dimension in self._dimensions.values():
            modalities.update(dimension.applicable_modalities)
        return modalities

    def validate(self) -> List[str]:
        """Validate the entire aggregate state."""
        errors = []

        # Validate weights
        errors.extend(self._validators["validate_weights"](self._weights))

        # Validate dimension consistency
        weight_dims = set(self._weights.keys())
        dimension_dims = set(self._dimensions.keys())

        if weight_dims != dimension_dims:
            missing_weights = dimension_dims - weight_dims
            extra_weights = weight_dims - dimension_dims
            if missing_weights:
                errors.append(f"Missing weights for dimensions: {missing_weights}")
            if extra_weights:
                errors.append(f"Weights for non-existent dimensions: {extra_weights}")

        return errors

    def increment_version(
        self,
        version_type: str = "patch",
        changelog: Optional[str] = None,
        migration_strategy: Optional[str] = None,
    ) -> None:
        """Increment version based on type of change.

        Args:
            version_type: 'major', 'minor', or 'patch'
            changelog: Description of changes
            migration_strategy: How to migrate from previous version
        """
        # Archive current version
        self._version_history.append(self._version)

        # Create new version
        effective_date = Timestamp.now().value

        if version_type == "major":
            new_version = self._version.increment_major(effective_date)
        elif version_type == "minor":
            new_version = self._version.increment_minor(effective_date)
        else:
            new_version = self._version.increment_patch(effective_date)

        # Update changelog and migration strategy
        if changelog or migration_strategy:
            new_version = DimensionSetVersion(
                major=new_version.major,
                minor=new_version.minor,
                patch=new_version.patch,
                effective_date=new_version.effective_date,
                migration_strategy=migration_strategy,
                changelog=changelog,
            )

        self._version = new_version
        self.updated_at = Timestamp.from_datetime(effective_date)

        # Raise domain event
        if self.id is not None:
            self.add_domain_event(
                DimensionSetVersionUpdatedEvent(
                    dimension_set_id=self.id,
                    old_version=self._version_history[-1].version_string,
                    new_version=self._version.version_string,
                    changelog=changelog,
                )
            )

    def is_compatible_with_version(self, required_version: str) -> bool:
        """Check if current version is compatible with a required version."""
        parts = required_version.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {required_version}")

        required = DimensionSetVersion(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
            effective_date=Timestamp.now().value,
        )

        return self._version.is_compatible_with(required)

    @classmethod
    def create_with_config(
        cls,
        organization_id: int,
        user_id: int,
        name: str,
        description: str,
        config: Optional[DimensionSetConfig] = None,
        **kwargs,
    ) -> "DimensionSetAggregate":
        """Factory method to create dimension set with specific configuration."""
        return cls(
            id=None,
            organization_id=organization_id,
            created_by_user_id=user_id,
            name=name,
            description=description,
            _config=config or DimensionSetConfig(),
            **kwargs,
        )
