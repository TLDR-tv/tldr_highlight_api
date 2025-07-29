"""Core application components.

This package contains core components shared across the application,
including database connections, configuration, and utilities.
"""

# Import functions are available but not automatically imported to avoid
# database connection issues during import time

__all__ = [
    "get_db",
    "get_db_context",
    "init_db",
    "close_db",
]
