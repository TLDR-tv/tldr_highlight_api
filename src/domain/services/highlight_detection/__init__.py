"""
AI-powered highlight detection engine for the TL;DR Highlight API.

This package provides sophisticated multi-modal analysis for identifying
and scoring potential highlights from livestreams and video content.

Key Components:
- BaseDetector: Abstract interface for detection algorithms
- VideoDetector: Motion detection and scene change analysis
- AudioDetector: Keyword matching and excitement detection
- ChatDetector: Sentiment spike and community engagement analysis
- FusionScorer: Multi-modal scoring combination
- Ranker: Highlight ranking and selection
- PostProcessor: Refinement and quality filtering

Architecture:
- Strategy pattern for pluggable detection algorithms
- Async/await for real-time processing
- Configurable thresholds and weights
- ML model integration capabilities
- Performance optimization for enterprise scale
"""

from .base_detector import (
    BaseDetector,
    DetectionResult,
    DetectionConfig,
    ContentSegment,
    ModalityType,
    HighlightCandidate,
)
from .video_detector import VideoDetector, VideoDetectionConfig
from .audio_detector import AudioDetector, AudioDetectionConfig
from .chat_detector import ChatDetector, ChatDetectionConfig
from .fusion_scorer import FusionScorer, FusionConfig
from .ranker import HighlightRanker, RankingConfig
from .post_processor import HighlightPostProcessor, PostProcessorConfig
from .gemini_detector import GeminiDetector, GeminiDetectionConfig

__all__ = [
    # Base classes
    "BaseDetector",
    "DetectionResult",
    "DetectionConfig",
    "ContentSegment",
    "ModalityType",
    "HighlightCandidate",
    # Detectors
    "VideoDetector",
    "VideoDetectionConfig",
    "AudioDetector",
    "AudioDetectionConfig",
    "ChatDetector",
    "ChatDetectionConfig",
    "GeminiDetector",
    "GeminiDetectionConfig",
    # Fusion and ranking
    "FusionScorer",
    "FusionConfig",
    "HighlightRanker",
    "RankingConfig",
    "HighlightPostProcessor",
    "PostProcessorConfig",
]

__version__ = "1.0.0"
