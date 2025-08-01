"""Stream ingestion infrastructure module.

This module provides stream ingestion using FFmpeg's segment muxer
for efficient and reliable stream processing.
"""

from .stream_ingestion_pipeline import (
    StreamIngestionPipeline,
    StreamIngestionConfig,
    ProcessingResult,
)

__all__ = [
    "StreamIngestionPipeline",
    "StreamIngestionConfig",
    "ProcessingResult",
]
