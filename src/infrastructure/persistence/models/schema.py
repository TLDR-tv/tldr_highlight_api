"""Database schema documentation and utilities.

This module provides utilities for working with the database schema,
including relationship mappings and constraints documentation.
"""

from typing import Any, Dict, List


def get_schema_summary() -> Dict[str, Dict[str, Any]]:
    """Get a summary of the database schema.

    Returns:
        Dict containing schema information for all tables
    """
    return {
        "users": {
            "description": "Enterprise customers using the API",
            "primary_key": "id",
            "indexes": ["email"],
            "unique_constraints": ["email"],
            "relationships": {
                "api_keys": "one-to-many",
                "owned_organizations": "one-to-many",
                "streams": "one-to-many",
                "batches": "one-to-many",
                "webhooks": "one-to-many",
                "usage_records": "one-to-many",
            },
        },
        "api_keys": {
            "description": "API keys for authentication with scoped permissions",
            "primary_key": "id",
            "indexes": ["key", "user_id", "active", "expires_at"],
            "unique_constraints": ["key"],
            "foreign_keys": ["user_id -> users.id"],
            "relationships": {"user": "many-to-one"},
        },
        "organizations": {
            "description": "Enterprise organizations with subscription plans",
            "primary_key": "id",
            "indexes": ["name", "owner_id"],
            "foreign_keys": ["owner_id -> users.id"],
            "relationships": {"owner": "many-to-one"},
        },
        "streams": {
            "description": "Livestream processing jobs",
            "primary_key": "id",
            "indexes": ["platform", "status", "user_id", "completed_at"],
            "foreign_keys": ["user_id -> users.id"],
            "relationships": {"user": "many-to-one", "highlights": "one-to-many"},
        },
        "batches": {
            "description": "Batch video processing jobs",
            "primary_key": "id",
            "indexes": ["status", "user_id"],
            "foreign_keys": ["user_id -> users.id"],
            "relationships": {"user": "many-to-one", "highlights": "one-to-many"},
        },
        "highlights": {
            "description": "Extracted video highlights",
            "primary_key": "id",
            "indexes": ["stream_id", "batch_id", "timestamp", "confidence_score"],
            "foreign_keys": ["stream_id -> streams.id", "batch_id -> batches.id"],
            "constraints": ["Either stream_id OR batch_id must be set (not both)"],
            "relationships": {"stream": "many-to-one", "batch": "many-to-one"},
        },
        "webhooks": {
            "description": "Webhook endpoints for event notifications",
            "primary_key": "id",
            "indexes": ["user_id", "active"],
            "foreign_keys": ["user_id -> users.id"],
            "relationships": {"user": "many-to-one"},
        },
        "usage_records": {
            "description": "Usage tracking for billing and analytics",
            "primary_key": "id",
            "indexes": ["user_id", "record_type", "created_at"],
            "foreign_keys": ["user_id -> users.id"],
            "relationships": {"user": "many-to-one"},
        },
    }


def get_table_dependencies() -> List[str]:
    """Get the order in which tables should be created.

    Returns:
        List of table names in dependency order
    """
    return [
        "users",  # No dependencies
        "organizations",  # Depends on users
        "api_keys",  # Depends on users
        "streams",  # Depends on users
        "batches",  # Depends on users
        "webhooks",  # Depends on users
        "usage_records",  # Depends on users
        "highlights",  # Depends on streams and batches
    ]


def get_cascade_relationships() -> Dict[str, List[str]]:
    """Get cascade delete relationships.

    Returns:
        Dict mapping parent tables to their cascade children
    """
    return {
        "users": [
            "api_keys",
            "organizations",
            "streams",
            "batches",
            "webhooks",
            "usage_records",
        ],
        "streams": ["highlights"],
        "batches": ["highlights"],
    }
