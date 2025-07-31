"""Domain layer for TL;DR Highlight API.

This layer contains the core business logic and rules, independent of
infrastructure concerns like databases or web frameworks.

The domain layer uses dataclasses for entities and value objects,
providing a clean, Pythonic approach to domain modeling.
"""

# Explicit imports from entities
from src.domain.entities import (
    User,
    Stream,
    Highlight,
    Organization,
    APIKey,
    Webhook,
)

# Explicit imports from value objects
from src.domain.value_objects import (
    Email,
    Url,
    ConfidenceScore,
    Duration,
    Timestamp,
    ProcessingOptions,
    CompanyName,
)

__all__ = [
    # Re-export entities
    "User",
    "Stream",
    "Highlight",
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
