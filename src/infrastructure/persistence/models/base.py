"""Base model class for all database models.

This module provides the base declarative class and common mixins
for all SQLAlchemy models in the application.
"""

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all database models.

    Provides common configuration and type annotation support
    for SQLAlchemy 2.0+ models.
    """

    # Allow any unmapped attributes (for type hints, etc.)
    __allow_unmapped__ = True

    # Type annotation map for common types
    type_annotation_map = {
        datetime: DateTime(timezone=True),
    }


class TimestampMixin:
    """Mixin that adds created_at and updated_at timestamps to models.

    Automatically manages timestamp fields with proper defaults
    and update triggers.
    """

    @declared_attr
    def created_at(cls) -> Mapped[datetime]:
        """Timestamp when the record was created."""
        return mapped_column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            comment="Timestamp when the record was created",
        )

    @declared_attr
    def updated_at(cls) -> Mapped[datetime]:
        """Timestamp when the record was last updated."""
        return mapped_column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
            comment="Timestamp when the record was last updated",
        )
