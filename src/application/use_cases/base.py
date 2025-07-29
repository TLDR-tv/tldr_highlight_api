"""Base use case interfaces and result types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Optional, List, Any
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
    errors: Optional[List[str]] = None
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
TOutput = TypeVar("TOutput", bound=UseCaseResult)


class UseCase(ABC, Generic[TInput, TOutput]):
    """Base class for all use cases.
    
    Use cases orchestrate domain services to implement
    application workflows. They handle:
    - Input validation
    - Authorization
    - Transaction boundaries
    - Error handling
    - Result mapping
    """
    
    @abstractmethod
    async def execute(self, request: TInput) -> TOutput:
        """Execute the use case with the given request.
        
        Args:
            request: Input request object
            
        Returns:
            Result object with status and data
        """
        pass