"""Base domain service protocol.

This module defines the base protocol for all domain services,
establishing common patterns and interfaces.
"""

from typing import Protocol, runtime_checkable
from abc import abstractmethod
import logging


@runtime_checkable
class DomainService(Protocol):
    """Base protocol for domain services.

    Domain services encapsulate business logic that doesn't naturally
    fit within a single entity or value object. They orchestrate
    operations across multiple aggregates and repositories.
    """

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """Service-specific logger."""
        ...


class BaseDomainService:
    """Base implementation for domain services.

    Provides common functionality like logging setup.
    """

    def __init__(self):
        """Initialize base domain service."""
        self._logger = logging.getLogger(self.__class__.__module__)

    @property
    def logger(self) -> logging.Logger:
        """Service-specific logger."""
        return self._logger
