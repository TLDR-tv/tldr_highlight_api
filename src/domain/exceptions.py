"""Domain-specific exceptions.

These exceptions represent business rule violations and domain-specific
error conditions that can occur within the domain layer.
"""


class DomainException(Exception):
    """Base exception for all domain-specific errors."""

    pass


class InvalidValueError(DomainException):
    """Raised when a value object receives invalid data."""

    pass


class BusinessRuleViolation(DomainException):
    """Raised when a business rule is violated."""

    pass


class BusinessRuleViolationError(DomainException):
    """Raised when a business rule is violated."""

    pass


class EntityNotFoundError(DomainException):
    """Raised when an entity cannot be found."""

    pass


class DuplicateEntityError(DomainException):
    """Raised when attempting to create a duplicate entity."""

    pass


class UnauthorizedAccessError(DomainException):
    """Raised when access to a resource is unauthorized."""

    pass


class QuotaExceededError(BusinessRuleViolation):
    """Raised when a usage quota is exceeded."""

    pass


class InvalidStateTransition(BusinessRuleViolation):
    """Raised when an invalid state transition is attempted."""

    pass


class InvalidResourceStateError(DomainException):
    """Raised when a resource is in an invalid state for the requested operation."""

    pass


class ValidationError(DomainException):
    """Raised when validation fails."""

    pass


class AuthenticationError(DomainException):
    """Raised when authentication fails."""

    pass
