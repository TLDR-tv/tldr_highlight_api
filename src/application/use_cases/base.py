"""Base use case interfaces and result types."""

from dataclasses import dataclass, field
from typing import Protocol, TypeVar, Optional, List, Any, runtime_checkable
from enum import Enum


class ResultStatus(Enum):
    """Status of use case execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    VALIDATION_ERROR = "validation_error"
    UNAUTHORIZED = "unauthorized"
    NOT_FOUND = "not_found"
    QUOTA_EXCEEDED = "quota_exceeded"


@dataclass
class UseCaseResult:
    """Base result for all use cases."""

    status: ResultStatus
    data: Optional[Any] = None
    errors: List[str] = field(default_factory=list)
    message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Check if the result is successful."""
        return self.status == ResultStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        """Check if the result is a failure."""
        return self.status != ResultStatus.SUCCESS


TInput = TypeVar("TInput")
TResult = TypeVar("TResult", bound=UseCaseResult)


@runtime_checkable
class UseCase(Protocol[TInput, TResult]):
    """Protocol for all use cases.

    Use cases orchestrate domain services to implement
    application workflows. They handle:
    - Input validation
    - Authorization
    - Transaction boundaries
    - Error handling
    - Result mapping
    """

    async def execute(self, request: TInput) -> TResult:
        """Execute the use case with the given request.

        Args:
            request: Input request object

        Returns:
            Result object with status and data
        """
        ...


@runtime_checkable
class SyncUseCase(Protocol[TInput, TResult]):
    """Protocol for synchronous use cases."""

    def execute(self, request: TInput) -> TResult:
        """Execute the use case synchronously.

        Args:
            request: Input request object

        Returns:
            Result object with status and data
        """
        ...
