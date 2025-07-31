"""Simplified stream processing configuration.

This replaces the complex HighlightAgentConfig with a much simpler
configuration focused on the essential parameters for streamlined processing.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from .base import Entity


@dataclass
class StreamProcessingConfig(Entity[int]):
    """Simplified configuration for stream processing.

    This focuses on the core parameters needed for the streamlined
    highlight detection flow:
    - Which dimension set to use for scoring
    - Minimum confidence threshold for accepting highlights
    - Maximum number of highlights to generate
    """

    # Basic identification
    name: str
    description: str
    organization_id: int
    created_by_user_id: int

    # Core processing parameters
    dimension_set_id: int  # Which dimension set to use for scoring
    confidence_threshold: float = 0.7  # Minimum confidence (0.0-1.0)
    max_highlights: Optional[int] = (
        None  # Maximum highlights per stream (None = unlimited)
    )

    # Optional parameters
    is_active: bool = True

    # Compatibility properties
    version: int = 1
    config_type: str = "simplified"
    content_type: str = "general"

    # Usage tracking
    times_used: int = 0
    last_used_at: Optional[datetime] = None

    def record_usage(self) -> None:
        """Record that this configuration was used."""
        self.times_used += 1
        self.last_used_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not (0.0 <= self.confidence_threshold <= 1.0):
            return False

        if self.max_highlights is not None and self.max_highlights <= 0:
            return False

        return True

    @staticmethod
    def create_default(
        organization_id: int,
        user_id: int,
        dimension_set_id: int,
        name: str = "Default Configuration",
    ) -> "StreamProcessingConfig":
        """Create a default processing configuration."""
        return StreamProcessingConfig(
            id=None,
            name=name,
            description="Default stream processing configuration",
            organization_id=organization_id,
            created_by_user_id=user_id,
            dimension_set_id=dimension_set_id,
            confidence_threshold=0.7,
            max_highlights=None,  # Unlimited
        )
