"""Media processing infrastructure components.

This module provides utilities for video and audio processing,
frame extraction, and media format handling.
"""

from .media_utils import (
    MediaProcessor,
    VideoFrame,
    AudioSegment,
    StreamCapture,
    FrameExtractor,
    AudioExtractor,
    MediaValidator
)

__all__ = [
    "MediaProcessor",
    "VideoFrame",
    "AudioSegment",
    "StreamCapture",
    "FrameExtractor",
    "AudioExtractor",
    "MediaValidator",
]