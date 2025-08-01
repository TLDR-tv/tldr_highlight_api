"""Highlight type registry for flexible highlight categorization.

This entity replaces the hardcoded HighlightType enum with a flexible
registry that allows clients to define their own highlight types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from ..entities.base import Entity
from ..exceptions import BusinessRuleViolation, EntityNotFoundError
from ..value_objects.timestamp import Timestamp


class BuiltInHighlightType(str, Enum):
    """Built-in highlight types available to all organizations."""

    GENERIC = "generic"  # Default/unspecified
    HIGH_MOMENT = "high_moment"  # High-scoring moment
    KEY_MOMENT = "key_moment"  # Important moment
    CUSTOM = "custom"  # User-defined


@dataclass
class HighlightTypeDefinition:
    """Definition of a single highlight type."""

    id: str  # Unique identifier
    name: str  # Display name
    description: str  # Detailed description

    # Detection criteria
    min_score_threshold: float = 0.5  # Minimum score to qualify
    required_dimensions: List[str] = field(
        default_factory=list
    )  # Must have these dimensions
    required_dimension_scores: Dict[str, float] = field(
        default_factory=dict
    )  # Min scores per dimension

    # Behavior configuration
    priority: int = 0  # Higher priority types override lower ones
    is_exclusive: bool = False  # If true, prevents other types from being assigned
    auto_assign: bool = True  # Automatically assign based on criteria

    # Visual/UI configuration
    color: Optional[str] = None  # Hex color for UI display
    icon: Optional[str] = None  # Icon identifier
    tags: List[str] = field(default_factory=list)  # Searchable tags

    # Metadata
    created_at: Timestamp = field(default_factory=Timestamp.now)
    is_active: bool = True

    def matches_criteria(
        self, score: float, dimension_scores: Dict[str, float]
    ) -> bool:
        """Check if a highlight matches this type's criteria.

        Args:
            score: Overall highlight score
            dimension_scores: Individual dimension scores

        Returns:
            True if the highlight qualifies for this type
        """
        # Check overall score threshold
        if score < self.min_score_threshold:
            return False

        # Check required dimensions exist
        if self.required_dimensions:
            if not all(dim in dimension_scores for dim in self.required_dimensions):
                return False

        # Check dimension score requirements
        for dim, min_score in self.required_dimension_scores.items():
            if dimension_scores.get(dim, 0) < min_score:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "min_score_threshold": self.min_score_threshold,
            "required_dimensions": self.required_dimensions,
            "required_dimension_scores": self.required_dimension_scores,
            "priority": self.priority,
            "is_exclusive": self.is_exclusive,
            "auto_assign": self.auto_assign,
            "color": self.color,
            "icon": self.icon,
            "tags": self.tags,
            "created_at": self.created_at.value.isoformat(),
            "is_active": self.is_active,
        }


@dataclass(kw_only=True)
class HighlightTypeRegistry(Entity[int]):
    """Registry of highlight types for an organization.

    This entity manages the available highlight types that can be
    assigned to highlights, allowing complete customization while
    maintaining consistency within an organization.
    """

    # Basic identification
    name: str
    description: str
    organization_id: int
    created_by_user_id: int

    # Type definitions
    types: Dict[str, HighlightTypeDefinition] = field(default_factory=dict)

    # Configuration
    allow_multiple_types: bool = True  # Can highlights have multiple types?
    max_types_per_highlight: int = 3  # Maximum types per highlight
    include_built_in_types: bool = True  # Include system default types

    # Default handling
    default_type_id: str = BuiltInHighlightType.GENERIC.value
    fallback_on_no_match: bool = True  # Use default if no types match

    # Usage tracking
    is_active: bool = True
    is_public: bool = False  # Can other organizations use this registry?
    usage_count: int = 0

    # Timestamps
    created_at: Timestamp = field(default_factory=Timestamp.now)
    updated_at: Timestamp = field(default_factory=Timestamp.now)

    def __post_init__(self):
        """Initialize the registry with built-in types if configured."""
        super().__post_init__()

        if self.include_built_in_types and not self.types:
            self._add_built_in_types()

    def _add_built_in_types(self):
        """Add the built-in highlight types."""
        built_in_types = [
            HighlightTypeDefinition(
                id=BuiltInHighlightType.GENERIC.value,
                name="Generic Highlight",
                description="A general highlight without specific categorization",
                min_score_threshold=0.5,
                priority=0,
            ),
            HighlightTypeDefinition(
                id=BuiltInHighlightType.HIGH_MOMENT.value,
                name="High Moment",
                description="A moment with exceptionally high score",
                min_score_threshold=0.85,
                priority=10,
            ),
            HighlightTypeDefinition(
                id=BuiltInHighlightType.KEY_MOMENT.value,
                name="Key Moment",
                description="An important or pivotal moment",
                min_score_threshold=0.7,
                priority=5,
            ),
            HighlightTypeDefinition(
                id=BuiltInHighlightType.CUSTOM.value,
                name="Custom",
                description="User-defined highlight type",
                min_score_threshold=0.0,
                priority=1,
                auto_assign=False,
            ),
        ]

        for type_def in built_in_types:
            self.types[type_def.id] = type_def

    def add_type(self, type_definition: HighlightTypeDefinition) -> None:
        """Add a new highlight type to the registry.

        Args:
            type_definition: The highlight type to add

        Raises:
            BusinessRuleViolation: If type ID already exists
        """
        if type_definition.id in self.types:
            raise BusinessRuleViolation(
                f"Highlight type '{type_definition.id}' already exists"
            )

        # Validate against built-in types
        if type_definition.id in [t.value for t in BuiltInHighlightType]:
            raise BusinessRuleViolation(
                f"Cannot override built-in type '{type_definition.id}'"
            )

        self.types[type_definition.id] = type_definition
        self.updated_at = Timestamp.now()

    def update_type(self, type_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing highlight type.

        Args:
            type_id: ID of the type to update
            updates: Dictionary of fields to update

        Raises:
            EntityNotFoundError: If type doesn't exist
            BusinessRuleViolation: If trying to update built-in type
        """
        if type_id not in self.types:
            raise EntityNotFoundError(f"Highlight type '{type_id}' not found")

        # Prevent modification of built-in types
        if type_id in [t.value for t in BuiltInHighlightType]:
            raise BusinessRuleViolation("Cannot modify built-in highlight types")

        type_def = self.types[type_id]

        # Update allowed fields
        allowed_fields = {
            "name",
            "description",
            "min_score_threshold",
            "required_dimensions",
            "required_dimension_scores",
            "priority",
            "is_exclusive",
            "auto_assign",
            "color",
            "icon",
            "tags",
            "is_active",
        }

        for field_name, value in updates.items():
            if field_name in allowed_fields:
                setattr(type_def, field_name, value)

        self.updated_at = Timestamp.now()

    def remove_type(self, type_id: str) -> None:
        """Remove a highlight type from the registry.

        Args:
            type_id: ID of the type to remove

        Raises:
            EntityNotFoundError: If type doesn't exist
            BusinessRuleViolation: If trying to remove built-in or default type
        """
        if type_id not in self.types:
            raise EntityNotFoundError(f"Highlight type '{type_id}' not found")

        if type_id in [t.value for t in BuiltInHighlightType]:
            raise BusinessRuleViolation("Cannot remove built-in highlight types")

        if type_id == self.default_type_id:
            raise BusinessRuleViolation("Cannot remove the default highlight type")

        del self.types[type_id]
        self.updated_at = Timestamp.now()

    def determine_types(
        self,
        score: float,
        dimension_scores: Dict[str, float],
        force_type: Optional[str] = None,
    ) -> List[str]:
        """Determine which highlight types apply to a highlight.

        Args:
            score: Overall highlight score
            dimension_scores: Individual dimension scores
            force_type: Optional type to force assign

        Returns:
            List of applicable type IDs
        """
        if force_type:
            if force_type in self.types:
                return [force_type]
            elif self.fallback_on_no_match:
                return [self.default_type_id]
            else:
                return []

        # Find all matching types
        matching_types = []
        for type_id, type_def in self.types.items():
            if not type_def.is_active or not type_def.auto_assign:
                continue

            if type_def.matches_criteria(score, dimension_scores):
                matching_types.append(
                    (type_def.priority, type_id, type_def.is_exclusive)
                )

        if not matching_types:
            if self.fallback_on_no_match:
                return [self.default_type_id]
            return []

        # Sort by priority (highest first)
        matching_types.sort(reverse=True)

        # Check for exclusive types
        for priority, type_id, is_exclusive in matching_types:
            if is_exclusive:
                return [type_id]

        # Return types based on configuration
        type_ids = [type_id for _, type_id, _ in matching_types]

        if self.allow_multiple_types:
            return type_ids[: self.max_types_per_highlight]
        else:
            return type_ids[:1]

    def get_type(self, type_id: str) -> Optional[HighlightTypeDefinition]:
        """Get a specific highlight type definition.

        Args:
            type_id: ID of the type to retrieve

        Returns:
            HighlightTypeDefinition or None if not found
        """
        return self.types.get(type_id)

    def get_active_types(self) -> List[HighlightTypeDefinition]:
        """Get all active highlight types sorted by priority.

        Returns:
            List of active type definitions
        """
        active_types = [
            type_def for type_def in self.types.values() if type_def.is_active
        ]
        return sorted(active_types, key=lambda t: t.priority, reverse=True)

    def validate_configuration(self) -> List[str]:
        """Validate the registry configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check default type exists
        if self.default_type_id not in self.types:
            errors.append(
                f"Default type '{self.default_type_id}' not found in registry"
            )

        # Check for duplicate priorities on exclusive types
        exclusive_priorities = set()
        for type_def in self.types.values():
            if type_def.is_exclusive:
                if type_def.priority in exclusive_priorities:
                    errors.append(
                        f"Multiple exclusive types have priority {type_def.priority}"
                    )
                exclusive_priorities.add(type_def.priority)

        # Validate max types setting
        if self.max_types_per_highlight < 1:
            errors.append("max_types_per_highlight must be at least 1")

        return errors

    def clone(
        self, new_organization_id: int, new_user_id: int
    ) -> "HighlightTypeRegistry":
        """Create a copy of this registry for another organization.

        Args:
            new_organization_id: ID of the new organization
            new_user_id: ID of the user creating the clone

        Returns:
            New registry instance
        """
        # Clone all non-built-in types
        cloned_types = {}
        for type_id, type_def in self.types.items():
            if type_id not in [t.value for t in BuiltInHighlightType]:
                cloned_types[type_id] = HighlightTypeDefinition(
                    id=type_def.id,
                    name=type_def.name,
                    description=type_def.description,
                    min_score_threshold=type_def.min_score_threshold,
                    required_dimensions=type_def.required_dimensions.copy(),
                    required_dimension_scores=type_def.required_dimension_scores.copy(),
                    priority=type_def.priority,
                    is_exclusive=type_def.is_exclusive,
                    auto_assign=type_def.auto_assign,
                    color=type_def.color,
                    icon=type_def.icon,
                    tags=type_def.tags.copy(),
                )

        return HighlightTypeRegistry(
            id=None,
            name=f"{self.name} (Copy)",
            description=f"Copy of {self.name}",
            organization_id=new_organization_id,
            created_by_user_id=new_user_id,
            types=cloned_types,
            allow_multiple_types=self.allow_multiple_types,
            max_types_per_highlight=self.max_types_per_highlight,
            include_built_in_types=self.include_built_in_types,
            default_type_id=self.default_type_id,
            fallback_on_no_match=self.fallback_on_no_match,
        )

    # Factory methods for common registries

    @classmethod
    def create_gaming_registry(
        cls, organization_id: int, user_id: int
    ) -> "HighlightTypeRegistry":
        """Create a registry optimized for gaming content."""
        registry = cls(
            id=None,
            name="Gaming Highlights",
            description="Highlight types for gaming content",
            organization_id=organization_id,
            created_by_user_id=user_id,
            allow_multiple_types=True,
            max_types_per_highlight=2,
        )

        # Add gaming-specific types
        gaming_types = [
            HighlightTypeDefinition(
                id="ace",
                name="Ace",
                description="Player eliminates entire enemy team",
                min_score_threshold=0.9,
                required_dimension_scores={"skill_execution": 0.8, "rarity": 0.7},
                priority=20,
                color="#FF0000",
                icon="fire",
                tags=["gaming", "skill", "rare"],
            ),
            HighlightTypeDefinition(
                id="clutch",
                name="Clutch",
                description="Win against overwhelming odds",
                min_score_threshold=0.85,
                required_dimension_scores={"clutch_factor": 0.8},
                priority=15,
                color="#FFA500",
                icon="trophy",
                tags=["gaming", "pressure", "skill"],
            ),
            HighlightTypeDefinition(
                id="funny",
                name="Funny Moment",
                description="Humorous or unexpected gameplay",
                min_score_threshold=0.6,
                required_dimensions=["humor_level"],
                priority=5,
                color="#FFFF00",
                icon="laugh",
                tags=["gaming", "humor", "entertainment"],
            ),
            HighlightTypeDefinition(
                id="fail",
                name="Epic Fail",
                description="Spectacular failure or mistake",
                min_score_threshold=0.5,
                priority=3,
                color="#8B4513",
                icon="face-palm",
                tags=["gaming", "humor", "fail"],
            ),
        ]

        for type_def in gaming_types:
            registry.add_type(type_def)

        return registry

    @classmethod
    def create_educational_registry(
        cls, organization_id: int, user_id: int
    ) -> "HighlightTypeRegistry":
        """Create a registry for educational content."""
        registry = cls(
            id=None,
            name="Educational Highlights",
            description="Highlight types for educational content",
            organization_id=organization_id,
            created_by_user_id=user_id,
            allow_multiple_types=True,
            max_types_per_highlight=3,
            default_type_id="key_concept",
        )

        # Add education-specific types
        edu_types = [
            HighlightTypeDefinition(
                id="key_concept",
                name="Key Concept",
                description="Important concept explanation",
                min_score_threshold=0.7,
                required_dimensions=["concept_clarity"],
                priority=15,
                color="#0000FF",
                icon="lightbulb",
                tags=["education", "concept", "learning"],
            ),
            HighlightTypeDefinition(
                id="example",
                name="Example/Demo",
                description="Practical example or demonstration",
                min_score_threshold=0.6,
                required_dimensions=["visual_demonstration"],
                priority=10,
                color="#00FF00",
                icon="play-circle",
                tags=["education", "demo", "practical"],
            ),
            HighlightTypeDefinition(
                id="question_answered",
                name="Q&A Moment",
                description="Important question answered",
                min_score_threshold=0.65,
                required_dimensions=["question_answered"],
                priority=8,
                color="#FF00FF",
                icon="question-circle",
                tags=["education", "qa", "interaction"],
            ),
            HighlightTypeDefinition(
                id="summary",
                name="Summary/Recap",
                description="Summary of key points",
                min_score_threshold=0.7,
                priority=12,
                color="#00FFFF",
                icon="list",
                tags=["education", "summary", "recap"],
            ),
        ]

        for type_def in edu_types:
            registry.add_type(type_def)

        return registry
