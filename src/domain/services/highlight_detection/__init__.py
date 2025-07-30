"""
AI-powered highlight detection engine for the TL;DR Highlight API.

This package provides Gemini-based dimension analysis for identifying
and scoring potential highlights from livestreams and video content.

Key Components:
- BaseDetector: Abstract interface for detection algorithms
- GeminiDetector: Gemini video understanding with dimension framework
- FusionScorer: Multi-modal scoring combination
- Ranker: Highlight ranking and selection
- PostProcessor: Refinement and quality filtering

Architecture:
- Dimension-based scoring for industry-agnostic detection
- Gemini video understanding API integration
- Structured outputs with JSON schema validation
- Async/await for real-time processing
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
