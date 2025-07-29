"""
Multi-Modal Content Processing Pipeline.

This module provides a comprehensive content processing pipeline for extracting
and analyzing video frames, audio transcription, and chat/comment sentiment
in real-time for the TL;DR Highlight API.

Key Components:
- VideoProcessor: Extracts and processes video frames
- AudioProcessor: Handles audio transcription using OpenAI Whisper
- ChatProcessor: Analyzes chat/comment sentiment and engagement
- ContentSynchronizer: Manages timestamp synchronization across modalities
- BufferManager: Handles processing windows and buffering
"""

from .video_processor import VideoProcessor, VideoProcessorConfig
from .audio_processor import AudioProcessor, AudioProcessorConfig
from .chat_processor import ChatProcessor as ContentChatProcessor, ChatProcessorConfig
from .synchronizer import ContentSynchronizer, SynchronizationConfig
from .buffer_manager import (
    BufferManager,
    BufferConfig,
    ProcessingWindow,
    BufferPriority,
    WindowType,
)
from .gemini_processor import (
    GeminiProcessor,
    GeminiProcessorConfig,
    ProcessingMode,
    GeminiModel,
)

__all__ = [
    "VideoProcessor",
    "VideoProcessorConfig",
    "AudioProcessor",
    "AudioProcessorConfig",
    "ContentChatProcessor",
    "ChatProcessorConfig",
    "ContentSynchronizer",
    "SynchronizationConfig",
    "BufferManager",
    "BufferConfig",
    "ProcessingWindow",
    "BufferPriority",
    "WindowType",
    "GeminiProcessor",
    "GeminiProcessorConfig",
    "ProcessingMode",
    "GeminiModel",
]
