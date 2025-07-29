"""Application use cases for the TL;DR Highlight API.

Use cases orchestrate domain services to implement application workflows.
They handle transaction boundaries, authorization, and coordinate between
multiple domain services.
"""

from .base import UseCase, UseCaseResult
from .authentication import AuthenticationUseCase, LoginResult, RegisterResult
from .stream_processing import StreamProcessingUseCase, StreamStartResult, StreamStopResult
from .batch_processing import BatchProcessingUseCase, BatchCreateResult, BatchStatusResult
from .webhook_processing import WebhookProcessingUseCase, ProcessWebhookRequest, ProcessWebhookResult

__all__ = [
    "UseCase",
    "UseCaseResult",
    "AuthenticationUseCase",
    "LoginResult",
    "RegisterResult",
    "StreamProcessingUseCase",
    "StreamStartResult",
    "StreamStopResult",
    "BatchProcessingUseCase",
    "BatchCreateResult",
    "BatchStatusResult",
    "WebhookProcessingUseCase",
    "ProcessWebhookRequest",
    "ProcessWebhookResult",
]