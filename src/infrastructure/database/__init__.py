"""Database infrastructure for the TL;DR Highlight API.

This module provides database connectivity and session management
as infrastructure concerns, following DDD principles.
"""

from .connection import (
    get_async_session,
    get_sync_session,
    get_db,
    get_db_context,
    init_db,
    close_db,
)

__all__ = [
    "get_async_session",
    "get_sync_session",
    "get_db",
    "get_db_context",
    "init_db",
    "close_db",
]
