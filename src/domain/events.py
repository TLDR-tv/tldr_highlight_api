"""Domain events for the dimension framework (fixed version).

Domain events capture important state changes and business occurrences
in the domain model, following DDD principles.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Dict, List
from uuid import uuid4


class DomainEvent:
    """Base class for all domain events.

    Domain events represent something that has happened in the domain
    that domain experts care about.
    """

    def __init__(self, event_type_name: str, **kwargs):
        self.event_type_name = event_type_name
        self.aggregate_id = kwargs.get("aggregate_id")
        self.aggregate_type = kwargs.get("aggregate_type")
        self.event_id = kwargs.get("event_id", str(uuid4()))
        self.occurred_at = kwargs.get("occurred_at", datetime.utcnow())
        self.metadata = kwargs.get("metadata", {})

    @property
    def event_type(self) -> str:
        """Get the event type name."""
        return self.event_type_name


# Dimension Set Events


@dataclass
class DimensionSetCreatedEvent(DomainEvent):
    """Raised when a new dimension set is created."""

    organization_id: int
    created_by_user_id: int
    name: str
    industry: Optional[str] = None
    content_type: Optional[str] = None

    def __post_init__(self):
        super().__init__("DimensionSetCreatedEvent")


@dataclass
class DimensionSetActivatedEvent(DomainEvent):
    """Raised when a dimension set is activated."""

    dimension_set_id: int
    activated_by_user_id: int

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class DimensionSetDeactivatedEvent(DomainEvent):
    """Raised when a dimension set is deactivated."""

    dimension_set_id: int
    deactivated_by_user_id: int
    reason: Optional[str] = None

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class DimensionSetPublishedEvent(DomainEvent):
    """Raised when a dimension set is made public."""

    dimension_set_id: int
    organization_id: int
    published_by_user_id: int

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


# Dimension Management Events


@dataclass
class DimensionAddedToDimensionSetEvent(DomainEvent):
    """Raised when a dimension is added to a set."""

    dimension_set_id: int
    dimension_id: str
    dimension_name: str
    weight: float
    version: Optional[str] = None

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class DimensionRemovedFromDimensionSetEvent(DomainEvent):
    """Raised when a dimension is removed from a set."""

    dimension_set_id: int
    dimension_id: str

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class DimensionWeightUpdatedEvent(DomainEvent):
    """Raised when dimension weights are updated."""

    dimension_set_id: int
    dimension_id: str
    old_weight: float
    new_weight: float

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class DimensionWeightsNormalizedEvent(DomainEvent):
    """Raised when dimension weights are normalized."""

    dimension_set_id: int
    normalization_factor: float

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class DimensionSetVersionUpdatedEvent(DomainEvent):
    """Raised when a dimension set version is updated."""

    dimension_set_id: int
    old_version: str
    new_version: str
    changelog: Optional[str] = None
    migration_strategy: Optional[str] = None

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


# Usage Events


@dataclass
class DimensionSetUsedForEvaluationEvent(DomainEvent):
    """Raised when a dimension set is used for evaluation."""

    dimension_set_id: int
    organization_id: int
    stream_id: Optional[str] = None
    evaluation_count: int = 1

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class DimensionSetClonedEvent(DomainEvent):
    """Raised when a dimension set is cloned."""

    source_dimension_set_id: int
    new_dimension_set_id: int
    source_organization_id: int
    target_organization_id: int
    cloned_by_user_id: int

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


# Scoring Events


@dataclass
class HighlightScoredEvent(DomainEvent):
    """Raised when a highlight is scored using dimensions."""

    highlight_id: str
    dimension_set_id: int
    weighted_score: float
    quality_level: str
    confidence_level: str
    dimension_count: int

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class DimensionEvaluatedEvent(DomainEvent):
    """Raised when a single dimension is evaluated."""

    dimension_id: str
    dimension_set_id: int
    score: float
    confidence: str
    evaluation_strategy: str

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class ScoringFailedEvent(DomainEvent):
    """Raised when scoring fails."""

    dimension_set_id: int
    reason: str
    failed_dimensions: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


# Calibration Events


@dataclass
class DimensionCalibrationUpdatedEvent(DomainEvent):
    """Raised when dimension calibration is updated."""

    dimension_set_id: int
    dimension_id: str
    sample_count: int
    new_scale_factor: float
    new_offset: float

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class CalibrationProfileCreatedEvent(DomainEvent):
    """Raised when a new calibration profile is created."""

    dimension_set_id: int
    organization_id: int
    target_distribution: str

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


# Validation Events


@dataclass
class DimensionSetValidationFailedEvent(DomainEvent):
    """Raised when dimension set validation fails."""

    dimension_set_id: int
    error_count: int
    warning_count: int
    validation_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class DimensionSetValidationPassedEvent(DomainEvent):
    """Raised when dimension set validation passes."""

    dimension_set_id: int
    validated_dimensions: int
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


# Organization Events


@dataclass
class OrganizationCreatedEvent(DomainEvent):
    """Event raised when a new organization is created."""

    organization_id: int
    name: str
    owner_id: int
    plan_type: str

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class MemberAddedEvent(DomainEvent):
    """Event raised when a member is added to an organization."""

    organization_id: int
    user_id: int
    added_by_user_id: int

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class MemberRemovedEvent(DomainEvent):
    """Event raised when a member is removed from an organization."""

    organization_id: int
    user_id: int
    removed_by_user_id: int

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class PlanUpgradedEvent(DomainEvent):
    """Event raised when an organization upgrades their plan."""

    organization_id: int
    old_plan: str
    new_plan: str
    upgraded_by_user_id: int

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class OrganizationDeactivatedEvent(DomainEvent):
    """Event raised when an organization is deactivated."""

    organization_id: int
    reason: Optional[str] = None

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


# Stream Events


@dataclass
class StreamCreatedEvent(DomainEvent):
    """Event raised when a new stream is created."""

    stream_id: int
    url: str
    platform: str
    user_id: int

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class StreamProcessingStartedEvent(DomainEvent):
    """Event raised when stream processing starts."""

    stream_id: int
    processing_options: Dict[str, Any]

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class StreamProcessingCompletedEvent(DomainEvent):
    """Event raised when stream processing completes."""

    stream_id: int
    duration_seconds: float
    highlight_count: int

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class StreamProcessingFailedEvent(DomainEvent):
    """Event raised when stream processing fails."""

    stream_id: int
    error_message: str

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__


@dataclass
class HighlightAddedToStreamEvent(DomainEvent):
    """Event raised when a highlight is added to a stream."""

    stream_id: int
    highlight_id: int
    confidence_score: float

    def __post_init__(self):
        self.event_type_name = self.__class__.__name__
