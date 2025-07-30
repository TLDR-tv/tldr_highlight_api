"""Domain layer for TL;DR Highlight API.

This layer contains the core business logic and rules, independent of
infrastructure concerns like databases or web frameworks.

The domain layer uses dataclasses for entities and value objects,
providing a clean, Pythonic approach to domain modeling.
"""

from src.domain.entities import *  # noqa: F403, F401
from src.domain.value_objects import *  # noqa: F403, F401

__all__ = [
    # Re-export entities
    "User",
    "Stream",
    "Highlight",
    "Batch",
    "Organization",
    "APIKey",
    "Webhook",
    # Re-export value objects
    "Email",
    "Url",
    "ConfidenceScore",
    "Duration",
    "Timestamp",
    "ProcessingOptions",
    "CompanyName",
]
