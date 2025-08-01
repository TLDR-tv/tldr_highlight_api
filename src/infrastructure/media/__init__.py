"""Media processing infrastructure components.

This module provides utilities for video and audio processing,
with FFmpeg-based segmentation for stream handling.
"""

from .ffmpeg_segmenter import (
    FFmpegSegmenter,
    SegmentConfig,
    SegmentInfo,
    SegmentFormat,
)

from .ffmpeg_integration import (
    FFmpegProcessor,
    FFmpegProbe,
    MediaInfo,
    VideoInfo,
    AudioInfo,
    TranscodeOptions,
)

__all__ = [
    # Segmentation
    "FFmpegSegmenter",
    "SegmentConfig",
    "SegmentInfo",
    "SegmentFormat",
    # FFmpeg utilities
    "FFmpegProcessor",
    "FFmpegProbe",
    "MediaInfo",
    "VideoInfo",
    "AudioInfo",
    "TranscodeOptions",
]
