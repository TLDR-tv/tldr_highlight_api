"""Stream ingestion infrastructure module.

This module provides simplified stream ingestion using FFmpeg's segment muxer
for efficient and reliable stream processing.
"""

from .simplified_ingestion_pipeline import (
    SimplifiedIngestionPipeline,
    SimplifiedIngestionConfig,
    ProcessingResult,
)

__all__ = [
    "SimplifiedIngestionPipeline",
    "SimplifiedIngestionConfig",
    "ProcessingResult",
]
