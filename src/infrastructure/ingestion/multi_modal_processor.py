"""Multi-modal content processor for unified analysis.

This module combines video, audio, and text analysis into a cohesive
highlight detection system using multiple AI models and processing techniques.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from ..content_processing.video_processor import VideoProcessor, VideoFrameData
from ..content_processing.audio_processor import AudioProcessor, AudioAnalysisResult
from ..content_processing.gemini_processor import GeminiProcessor, GeminiAnalysisResult
from ..streaming.segment_processor import ProcessedSegment

logger = logging.getLogger(__name__)


@dataclass
class MultiModalResult:
    """Result from multi-modal content analysis."""
    
    timestamp: float
    segment_id: str
    
    # Individual modality results
    video_analysis: Optional[Dict[str, Any]] = None
    audio_analysis: Optional[AudioAnalysisResult] = None
    ai_analysis: Optional[GeminiAnalysisResult] = None
    
    # Fused results
    highlight_score: float = 0.0
    confidence: float = 0.0
    detected_elements: List[str] = field(default_factory=list)
    suggested_tags: List[str] = field(default_factory=list)
    
    # Analysis metadata
    processing_time: float = 0.0
    modalities_used: List[str] = field(default_factory=list)
    fusion_method: str = "weighted_average"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiModalProcessor:
    """Processor that combines multiple modalities for comprehensive analysis.
    
    Integrates video processing, audio analysis, and AI-powered understanding
    to provide comprehensive highlight detection and content analysis.
    """
    
    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        gemini_processor: Optional[GeminiProcessor] = None,
        fusion_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize multi-modal processor.
        
        Args:
            video_processor: Video content processor
            audio_processor: Audio content processor  
            gemini_processor: AI-powered content processor
            fusion_weights: Weights for combining modality scores
        """
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.gemini_processor = gemini_processor
        
        # Default fusion weights
        self.fusion_weights = fusion_weights or {
            "video": 0.4,
            "audio": 0.3,
            "ai": 0.5,  # AI gets higher weight as it's multi-modal
        }
        
        # Normalize weights
        total_weight = sum(self.fusion_weights.values())
        if total_weight > 0:
            self.fusion_weights = {
                k: v / total_weight for k, v in self.fusion_weights.items()
            }
        
        # Processing statistics
        self._stats = {
            "segments_processed": 0,
            "total_processing_time": 0.0,
            "video_analyses": 0,
            "audio_analyses": 0,
            "ai_analyses": 0,
            "highlights_detected": 0,
        }
        
        logger.info("Initialized MultiModalProcessor")
    
    async def process_segment(self, segment: ProcessedSegment) -> MultiModalResult:
        """Process a video segment using all available modalities.
        
        Args:
            segment: Video segment to analyze
            
        Returns:
            Multi-modal analysis result
        """
        start_time = asyncio.get_event_loop().time()
        
        result = MultiModalResult(
            timestamp=segment.start_time,
            segment_id=segment.segment_id,
        )
        
        # Process each modality concurrently
        tasks = []
        
        if self.video_processor:
            tasks.append(self._process_video_modality(segment))
        if self.audio_processor:
            tasks.append(self._process_audio_modality(segment))
        if self.gemini_processor:
            tasks.append(self._process_ai_modality(segment))
        
        # Execute all modality processing concurrently
        modality_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results from each modality
        for i, modality_result in enumerate(modality_results):
            if isinstance(modality_result, Exception):
                logger.error(f"Modality {i} processing failed: {modality_result}")
                continue
                
            modality_name, analysis_result = modality_result
            
            if modality_name == "video":
                result.video_analysis = analysis_result
                result.modalities_used.append("video")
                self._stats["video_analyses"] += 1
                
            elif modality_name == "audio":
                result.audio_analysis = analysis_result
                result.modalities_used.append("audio")
                self._stats["audio_analyses"] += 1
                
            elif modality_name == "ai":
                result.ai_analysis = analysis_result
                result.modalities_used.append("ai")
                self._stats["ai_analyses"] += 1
        
        # Fuse results from all modalities
        await self._fuse_modality_results(result)
        
        # Update processing statistics
        processing_time = asyncio.get_event_loop().time() - start_time
        result.processing_time = processing_time
        
        self._stats["segments_processed"] += 1
        self._stats["total_processing_time"] += processing_time
        
        if result.highlight_score > 0.7:
            self._stats["highlights_detected"] += 1
        
        logger.debug(
            f"Processed segment {segment.segment_id}: "
            f"score={result.highlight_score:.3f}, "
            f"modalities={result.modalities_used}, "
            f"time={processing_time:.3f}s"
        )
        
        return result
    
    async def _process_video_modality(self, segment: ProcessedSegment) -> Tuple[str, Dict[str, Any]]:
        """Process video modality."""
        try:
            # Extract representative frames
            keyframes = segment.get_keyframes()
            if not keyframes:
                return "video", {"frame_count": 0, "quality_score": 0.0}
            
            # Convert to video processor format (simplified)
            frame_data_list = []
            for i, frame in enumerate(keyframes[:5]):  # Limit to 5 frames
                frame_data = VideoFrameData(
                    timestamp=frame.timestamp,
                    data=frame.data,
                    width=frame.width,
                    height=frame.height,
                    frame_number=i,
                    metadata=frame.metadata or {},
                )
                frame_data_list.append(frame_data)
            
            # Detect scene changes
            scene_changes = await self.video_processor.detect_scene_changes(frame_data_list)
            
            # Calculate motion and visual complexity (simplified)
            motion_score = sum(f.metadata.get("motion_score", 0.5) for f in keyframes) / len(keyframes)
            visual_complexity = len(scene_changes) / len(keyframes) if keyframes else 0
            
            return "video", {
                "frame_count": len(keyframes),
                "scene_changes": len(scene_changes),
                "motion_score": motion_score,
                "visual_complexity": visual_complexity,
                "quality_score": segment.quality_score,
                "keyframe_ratio": len(segment.keyframe_indices) / segment.frame_count,
            }
            
        except Exception as e:
            logger.error(f"Video modality processing failed: {e}")
            return "video", {"error": str(e), "quality_score": 0.0}
    
    async def _process_audio_modality(self, segment: ProcessedSegment) -> Tuple[str, AudioAnalysisResult]:
        """Process audio modality."""
        try:
            # For real implementation, this would extract audio from the segment
            # For now, create a mock audio analysis correlated with the segment
            
            # Simulate audio processing delay
            await asyncio.sleep(0.1)
            
            # Create mock audio analysis result
            volume_level = 0.5 + (segment.start_time % 10) / 20.0
            
            # Generate transcription based on segment characteristics
            if segment.motion_score > 0.7:
                transcription = "Intense action happening! The players are making crucial moves!"
            elif segment.quality_score > 0.8:
                transcription = "High quality moment with clear visual elements"
            else:
                transcription = f"Audio content for segment at {segment.start_time:.1f}s"
            
            # Detect audio events based on segment properties
            detected_events = []
            if volume_level > 0.7:
                detected_events.append("high_volume")
            if "action" in transcription.lower():
                detected_events.append("action_commentary")
            if segment.motion_score > 0.6:
                detected_events.append("excitement")
            
            return "audio", AudioAnalysisResult(
                timestamp=segment.start_time,
                duration=segment.duration,
                volume_level=volume_level,
                transcription=transcription,
                detected_events=detected_events,
                language_confidence=0.9,
                metadata={"segment_id": segment.segment_id},
            )
            
        except Exception as e:
            logger.error(f"Audio modality processing failed: {e}")
            # Return empty result on error
            return "audio", AudioAnalysisResult(
                timestamp=segment.start_time,
                duration=segment.duration,
                volume_level=0.0,
                transcription="",
                detected_events=[],
                language_confidence=0.0,
                metadata={"error": str(e)},
            )
    
    async def _process_ai_modality(self, segment: ProcessedSegment) -> Tuple[str, GeminiAnalysisResult]:
        """Process AI modality using Gemini."""
        try:
            # Get representative frame and audio data
            keyframes = segment.get_keyframes()
            frame_data = keyframes[0].data if keyframes else b""
            
            # Mock audio transcript (in real implementation, would be from audio processor)
            audio_transcript = f"Audio analysis for segment {segment.segment_id}"
            
            # Mock chat messages (in real implementation, would be from chat adapter)
            chat_messages = ["Great moment!", "Epic play!", "Amazing!"]
            
            # Perform multimodal analysis
            ai_result = await self.gemini_processor.analyze_multimodal(
                video_frame=frame_data,
                audio_transcript=audio_transcript,
                chat_messages=chat_messages,
                timestamp=segment.start_time,
            )
            
            return "ai", ai_result
            
        except Exception as e:
            logger.error(f"AI modality processing failed: {e}")
            # Return mock result on error
            from ..content_processing.gemini_processor import GeminiAnalysisResult
            return "ai", GeminiAnalysisResult(
                request_id=f"error_{segment.segment_id}",
                content_type="multimodal",
                analysis_text=f"AI analysis failed: {str(e)}",
                highlight_score=0.0,
                detected_elements=[],
                suggested_tags=[],
                confidence=0.0,
                processing_time=0.0,
                metadata={"error": str(e)},
            )
    
    async def _fuse_modality_results(self, result: MultiModalResult) -> None:
        """Fuse results from all modalities into final scores."""
        
        scores = []
        confidences = []
        all_elements = []
        all_tags = []
        
        # Video modality contribution
        if result.video_analysis and "video" in result.modalities_used:
            video_score = self._calculate_video_highlight_score(result.video_analysis)
            scores.append(("video", video_score))
            confidences.append(0.8)  # High confidence for visual analysis
            
            # Extract video-based elements
            if result.video_analysis.get("motion_score", 0) > 0.6:
                all_elements.append("high_motion")
            if result.video_analysis.get("scene_changes", 0) > 2:
                all_elements.append("scene_changes")
        
        # Audio modality contribution
        if result.audio_analysis and "audio" in result.modalities_used:
            audio_score = self._calculate_audio_highlight_score(result.audio_analysis)
            scores.append(("audio", audio_score))
            confidences.append(result.audio_analysis.language_confidence)
            
            # Add audio elements and tags
            all_elements.extend(result.audio_analysis.detected_events)
            if result.audio_analysis.volume_level > 0.7:
                all_tags.append("high_energy")
        
        # AI modality contribution
        if result.ai_analysis and "ai" in result.modalities_used:
            ai_score = result.ai_analysis.highlight_score
            scores.append(("ai", ai_score))
            confidences.append(result.ai_analysis.confidence)
            
            # Add AI-detected elements and tags
            all_elements.extend(result.ai_analysis.detected_elements)
            all_tags.extend(result.ai_analysis.suggested_tags)
        
        # Fuse scores using weighted average
        if scores:
            weighted_score = 0.0
            total_weight = 0.0
            
            for modality, score in scores:
                weight = self.fusion_weights.get(modality, 0.33)
                weighted_score += score * weight
                total_weight += weight
            
            result.highlight_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Calculate overall confidence
        if confidences:
            result.confidence = sum(confidences) / len(confidences)
        
        # Deduplicate and limit elements and tags
        result.detected_elements = list(set(all_elements))[:7]
        result.suggested_tags = list(set(all_tags))[:5]
        
        # Add fusion metadata
        result.metadata.update({
            "fusion_scores": {modality: score for modality, score in scores},
            "fusion_weights": self.fusion_weights,
            "modality_count": len(result.modalities_used),
        })
    
    def _calculate_video_highlight_score(self, video_analysis: Dict[str, Any]) -> float:
        """Calculate highlight score from video analysis."""
        score = 0.0
        
        # Motion score contribution
        motion_score = video_analysis.get("motion_score", 0.0)
        score += motion_score * 0.4
        
        # Scene changes contribution
        scene_changes = video_analysis.get("scene_changes", 0)
        if scene_changes > 0:
            score += min(scene_changes / 5.0, 0.3)  # Normalize to 0.3 max
        
        # Visual complexity contribution
        complexity = video_analysis.get("visual_complexity", 0.0)
        score += complexity * 0.2
        
        # Quality score contribution
        quality = video_analysis.get("quality_score", 0.5)
        score += quality * 0.1
        
        return min(score, 1.0)
    
    def _calculate_audio_highlight_score(self, audio_analysis: AudioAnalysisResult) -> float:
        """Calculate highlight score from audio analysis."""
        score = 0.0
        
        # Volume level contribution
        score += audio_analysis.volume_level * 0.4
        
        # Event detection contribution
        event_score = len(audio_analysis.detected_events) / 5.0  # Normalize
        score += min(event_score, 0.3)
        
        # Transcription analysis (simplified)
        if audio_analysis.transcription:
            excitement_keywords = ["amazing", "incredible", "wow", "epic", "great", "awesome"]
            keyword_count = sum(
                1 for keyword in excitement_keywords 
                if keyword in audio_analysis.transcription.lower()
            )
            score += min(keyword_count / 3.0, 0.3)
        
        return min(score, 1.0)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self._stats.copy()
        
        if stats["segments_processed"] > 0:
            stats["avg_processing_time"] = (
                stats["total_processing_time"] / stats["segments_processed"]
            )
            stats["highlight_rate"] = (
                stats["highlights_detected"] / stats["segments_processed"]
            )
        else:
            stats["avg_processing_time"] = 0.0
            stats["highlight_rate"] = 0.0
        
        return stats
    
    async def cleanup(self) -> None:
        """Clean up processor resources."""
        if self.video_processor:
            await self.video_processor.cleanup()
        if self.audio_processor:
            await self.audio_processor.cleanup()
        if self.gemini_processor:
            await self.gemini_processor.cleanup()
        
        logger.info("MultiModalProcessor cleanup completed")