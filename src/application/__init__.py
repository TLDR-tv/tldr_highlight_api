"""Application layer for the TL;DR Highlight API.

The application layer contains use cases that orchestrate domain services
to implement application workflows. Use cases handle:

- Input validation and request/response mapping
- Transaction boundaries
- Authorization checks
- Error handling and result formatting
- Coordination between multiple domain services
"""

from .use_cases import (
    # Base classes
    UseCase,
    UseCaseResult,
    # Authentication
    AuthenticationUseCase,
    LoginResult,
    RegisterResult,
    # Stream Processing
    StreamProcessingUseCase,
    StreamStartResult,
    StreamStopResult,
)

__all__ = [
    # Base classes
    "UseCase",
    "UseCaseResult",
    # Authentication
    "AuthenticationUseCase",
    "LoginResult",
    "RegisterResult",
    # Stream Processing
    "StreamProcessingUseCase",
    "StreamStartResult",
    "StreamStopResult",
]
