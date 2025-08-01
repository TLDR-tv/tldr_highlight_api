"""Domain-specific exceptions.

These exceptions represent business rule violations and domain-specific
error conditions that can occur within the domain layer.

Following Pythonic principles:
- Rich exception messages with context
- Using properties for computed values
- Leveraging Python's exception chaining
- Simple, flat hierarchy
"""

from typing import Optional, Dict, Any


class DomainException(Exception):
    """Base exception for all domain-specific errors.

    Provides rich context about what went wrong and where.
    """

    def __init__(
        self,
        message: str,
        *,  # Force keyword-only arguments
        entity_type: Optional[str] = None,
        entity_id: Optional[Any] = None,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize domain exception with rich context.

        Args:
            message: Human-readable error message
            entity_type: Type of entity involved (e.g., "User", "Stream")
            entity_id: ID of the entity involved
            field_name: Field that caused the error
            invalid_value: The invalid value that was provided
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.context = context or {}

    def __str__(self) -> str:
        """Provide detailed string representation."""
        parts = [self.message]

        if self.entity_type:
            if self.entity_id:
                parts.append(f"[{self.entity_type}:{self.entity_id}]")
            else:
                parts.append(f"[{self.entity_type}]")

        if self.field_name:
            parts.append(f"Field: {self.field_name}")

        if self.invalid_value is not None:
            parts.append(f"Invalid value: {self.invalid_value!r}")

        return " - ".join(parts)

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"entity_type={self.entity_type!r}, "
            f"entity_id={self.entity_id!r})"
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
        super().__init__(message, entity_type=entity_type, entity_id=entity_id)


class DuplicateEntityError(DomainException):
    """Raised when attempting to create a duplicate entity.

    This protects uniqueness constraints.
    """

    def __init__(self, entity_type: str, duplicate_field: str, duplicate_value: Any):
        """Initialize with duplication information."""
        message = (
            f"{entity_type} with {duplicate_field}={duplicate_value!r} already exists"
        )
        super().__init__(
            message,
            entity_type=entity_type,
            field_name=duplicate_field,
            invalid_value=duplicate_value,
        )


class UnauthorizedOperation(DomainException):
    """Raised when an operation is not authorized.

    This represents authorization failures at the domain level.
    """

    def __init__(self, operation: str, reason: str, user_id: Optional[int] = None):
        """Initialize with operation details."""
        message = f"Unauthorized operation '{operation}': {reason}"
        super().__init__(message, context={"operation": operation, "user_id": user_id})


class QuotaExceeded(BusinessRuleViolation):
    """Raised when a usage quota is exceeded.

    This protects resource limits.
    """

    def __init__(
        self,
        resource: str,
        limit: int,
        current: int,
        organization_id: Optional[int] = None,
    ):
        """Initialize with quota information."""
        message = f"{resource} quota exceeded: {current}/{limit}"
        super().__init__(
            message,
            entity_type="Organization",
            entity_id=organization_id,
            context={
                "resource": resource,
                "limit": limit,
                "current": current,
                "exceeded_by": current - limit,
            },
        )


class InvalidStateTransition(BusinessRuleViolation):
    """Raised when an invalid state transition is attempted.

    This protects state machine invariants.
    """

    def __init__(
        self,
        entity_type: str,
        entity_id: Any,
        from_state: str,
        to_state: str,
        allowed_states: Optional[list] = None,
    ):
        """Initialize with state transition information."""
        message = f"Cannot transition from {from_state} to {to_state}"
        if allowed_states:
            message += f". Allowed: {', '.join(allowed_states)}"

        super().__init__(
            message,
            entity_type=entity_type,
            entity_id=entity_id,
            context={
                "from_state": from_state,
                "to_state": to_state,
                "allowed_states": allowed_states or [],
            },
        )


class ValidationError(DomainException):
    """Raised when validation fails.

    This can contain multiple validation errors.
    """

    def __init__(self, errors: Dict[str, str], entity_type: Optional[str] = None):
        """Initialize with validation errors."""
        message = "Validation failed"
        if len(errors) == 1:
            field, error = next(iter(errors.items()))
            message = f"{field}: {error}"
        else:
            message = f"Multiple validation errors: {', '.join(errors.keys())}"

        super().__init__(message, entity_type=entity_type, context={"errors": errors})

    @property
    def errors(self) -> Dict[str, str]:
        """Get validation errors."""
        return self.context.get("errors", {})
