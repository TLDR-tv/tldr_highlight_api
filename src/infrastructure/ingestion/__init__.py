"""Stream ingestion infrastructure module.

This module provides unified stream ingestion capabilities that integrate
FFmpeg processing with stream adapters for comprehensive content analysis.
"""

from .unified_ingestion_pipeline import (
    StreamIngestionPipeline,
    IngestionConfig,
    IngestionResult,
    IngestionStatus,
)
from .stream_ingestion_factory import StreamIngestionFactory
from .multi_modal_processor import MultiModalProcessor, MultiModalResult

__all__ = [
    "StreamIngestionPipeline",
    "IngestionConfig", 
    "IngestionResult",
    "IngestionStatus",
    "StreamIngestionFactory",
    "MultiModalProcessor",
    "MultiModalResult",
]