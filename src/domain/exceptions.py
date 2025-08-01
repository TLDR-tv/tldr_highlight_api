"""Domain-specific exceptions.

These exceptions represent business rule violations and domain-specific
error conditions that can occur within the domain layer.

Following Pythonic principles:
- Rich exception messages with context
- Using dataclasses for structured error data
- Leveraging Python's exception chaining
- Simple, flat hierarchy
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class ErrorContext:
    """Structured error context using dataclass."""

    entity_type: Optional[str] = None
    entity_id: Optional[Any] = None
    field_name: Optional[str] = None
    invalid_value: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        if self.entity_type:
            result["entity_type"] = self.entity_type
        if self.entity_id is not None:
            result["entity_id"] = self.entity_id
        if self.field_name:
            result["field_name"] = self.field_name
        if self.invalid_value is not None:
            result["invalid_value"] = self.invalid_value
        if self.extra:
            result.update(self.extra)
        return result


@dataclass(frozen=True)
class QuotaInfo:
    """Structured quota information."""

    resource: str
    limit: int
    current: int
    organization_id: Optional[int] = None

    @property
    def exceeded_by(self) -> int:
        """Calculate how much the quota was exceeded by."""
        return max(0, self.current - self.limit)

    @property
    def percentage_used(self) -> float:
        """Calculate percentage of quota used."""
        return (self.current / self.limit) * 100 if self.limit > 0 else 0


@dataclass(frozen=True)
class StateTransitionInfo:
    """Structured state transition information."""

    from_state: str
    to_state: str
    allowed_states: list[str] = field(default_factory=list)
    entity_type: Optional[str] = None
    entity_id: Optional[Any] = None


@dataclass(frozen=True)
class ValidationErrors:
    """Structured validation error information."""

    errors: Dict[str, str] = field(default_factory=dict)

    def add_error(self, field: str, message: str) -> None:
        """Add a validation error (creates new instance due to frozen)."""
        # Note: Since this is frozen, this would typically be used during construction
        pass

    @property
    def field_count(self) -> int:
        """Get number of fields with errors."""
        return len(self.errors)

    @property
    def is_single_field(self) -> bool:
        """Check if only one field has errors."""
        return self.field_count == 1


class DomainException(Exception):
    """Base exception for all domain-specific errors.

    Provides rich context about what went wrong and where.
    """

    def __init__(
        self,
        message: str,
        *,  # Force keyword-only arguments
        context: Optional[ErrorContext] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[Any] = None,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize domain exception with rich context.

        Args:
            message: Human-readable error message
            context: Structured error context (preferred)
            entity_type: Type of entity involved (legacy, will be moved to context)
            entity_id: ID of the entity involved (legacy, will be moved to context)
            field_name: Field that caused the error (legacy, will be moved to context)
            invalid_value: The invalid value that was provided (legacy, will be moved to context)
            extra_context: Additional context information (legacy, will be moved to context)
        """
        super().__init__(message)
        self.message = message

        # Use provided ErrorContext or create one from legacy parameters
        if context is not None:
            self.context = context
        else:
            self.context = ErrorContext(
                entity_type=entity_type,
                entity_id=entity_id,
                field_name=field_name,
                invalid_value=invalid_value,
                extra=extra_context or {},
            )

        # Legacy properties for backward compatibility
        self.entity_type = self.context.entity_type
        self.entity_id = self.context.entity_id
        self.field_name = self.context.field_name
        self.invalid_value = self.context.invalid_value

    def __str__(self) -> str:
        """Provide detailed string representation."""
        parts = [self.message]

        if self.context.entity_type:
            if self.context.entity_id:
                parts.append(f"[{self.context.entity_type}:{self.context.entity_id}]")
            else:
                parts.append(f"[{self.context.entity_type}]")

        if self.context.field_name:
            parts.append(f"Field: {self.context.field_name}")

        if self.context.invalid_value is not None:
            parts.append(f"Invalid value: {self.context.invalid_value!r}")

        return " - ".join(parts)

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"context={self.context!r})"
        )


class InvalidValueError(DomainException):
    """Raised when a value object receives invalid data.

    This is typically used in value object validation.
    """

    pass


class BusinessRuleViolation(DomainException):
    """Raised when a business rule is violated.

    This represents violations of domain invariants.
    """

    pass


class EntityNotFoundError(DomainException):
    """Raised when an entity cannot be found.

    This is used when repository lookups fail.
    """

    def __init__(self, entity_type: str, entity_id: Any, message: Optional[str] = None):
        """Initialize with entity information."""
        message = message or f"{entity_type} with ID {entity_id!r} not found"

        context = ErrorContext(entity_type=entity_type, entity_id=entity_id)

        super().__init__(message, context=context)


class DuplicateEntityError(DomainException):
    """Raised when attempting to create a duplicate entity.

    This protects uniqueness constraints.
    """

    def __init__(self, entity_type: str, duplicate_field: str, duplicate_value: Any):
        """Initialize with duplication information."""
        message = (
            f"{entity_type} with {duplicate_field}={duplicate_value!r} already exists"
        )

        context = ErrorContext(
            entity_type=entity_type,
            field_name=duplicate_field,
            invalid_value=duplicate_value,
        )

        super().__init__(message, context=context)


class UnauthorizedOperation(DomainException):
    """Raised when an operation is not authorized.

    This represents authorization failures at the domain level.
    """

    def __init__(self, operation: str, reason: str, user_id: Optional[int] = None):
        """Initialize with operation details."""
        message = f"Unauthorized operation '{operation}': {reason}"

        context = ErrorContext(
            entity_type="User",
            entity_id=user_id,
            extra={"operation": operation, "reason": reason},
        )

        super().__init__(message, context=context)


class QuotaExceeded(BusinessRuleViolation):
    """Raised when a usage quota is exceeded.

    This protects resource limits.
    """

    def __init__(self, quota_info: QuotaInfo):
        """Initialize with structured quota information."""
        message = f"{quota_info.resource} quota exceeded: {quota_info.current}/{quota_info.limit}"

        context = ErrorContext(
            entity_type="Organization",
            entity_id=quota_info.organization_id,
            extra={
                "resource": quota_info.resource,
                "limit": quota_info.limit,
                "current": quota_info.current,
                "exceeded_by": quota_info.exceeded_by,
                "percentage_used": quota_info.percentage_used,
            },
        )

        super().__init__(message, context=context)
        self.quota_info = quota_info


class InvalidStateTransition(BusinessRuleViolation):
    """Raised when an invalid state transition is attempted.

    This protects state machine invariants.
    """

    def __init__(self, transition_info: StateTransitionInfo):
        """Initialize with structured state transition information."""
        message = f"Cannot transition from {transition_info.from_state} to {transition_info.to_state}"
        if transition_info.allowed_states:
            message += f". Allowed: {', '.join(transition_info.allowed_states)}"

        context = ErrorContext(
            entity_type=transition_info.entity_type,
            entity_id=transition_info.entity_id,
            extra={
                "from_state": transition_info.from_state,
                "to_state": transition_info.to_state,
                "allowed_states": transition_info.allowed_states,
            },
        )

        super().__init__(message, context=context)
        self.transition_info = transition_info


class ValidationError(DomainException):
    """Raised when validation fails.

    This can contain multiple validation errors.
    """

    def __init__(
        self, validation_errors: ValidationErrors, entity_type: Optional[str] = None
    ):
        """Initialize with structured validation errors."""
        if validation_errors.is_single_field:
            field, error = next(iter(validation_errors.errors.items()))
            message = f"{field}: {error}"
        else:
            message = f"Multiple validation errors: {', '.join(validation_errors.errors.keys())}"

        context = ErrorContext(
            entity_type=entity_type,
            extra={
                "errors": validation_errors.errors,
                "field_count": validation_errors.field_count,
            },
        )

        super().__init__(message, context=context)
        self.validation_errors = validation_errors

    @property
    def errors(self) -> Dict[str, str]:
        """Get validation errors."""
        return self.validation_errors.errors
