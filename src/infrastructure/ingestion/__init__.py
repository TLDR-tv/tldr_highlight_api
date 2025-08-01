"""Stream ingestion infrastructure module.

This module provides stream ingestion using FFmpeg's segment muxer
with separated concerns for segmentation and processing.
"""

from .stream_ingestion_pipeline import (
    StreamIngestionPipeline,
    StreamIngestionConfig,
)
from .stream_segmenter import (
    StreamSegmenter,
    StreamSegmenterConfig,
)
from .stream_processor import (
    StreamProcessor,
    StreamProcessorConfig,
    ProcessingResult,
)

__all__ = [
    # Pipeline (orchestrator)
    "StreamIngestionPipeline",
    "StreamIngestionConfig",
    # Segmenter (ingestion only)
    "StreamSegmenter", 
    "StreamSegmenterConfig",
    # Processor (analysis only)
    "StreamProcessor",
    "StreamProcessorConfig",
    "ProcessingResult",
]
