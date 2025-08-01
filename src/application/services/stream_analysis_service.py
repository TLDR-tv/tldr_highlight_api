"""Application service for stream analysis orchestration.

This service coordinates between the infrastructure stream segmentation
and the domain highlight analysis service.
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator
from pathlib import Path

from ...domain.entities.stream import Stream
from ...domain.entities.video_segment import VideoSegment
from ...domain.entities.dimension_set_aggregate import DimensionSetAggregate
from ...domain.entities.highlight_agent_config import HighlightAgentConfig
from ...domain.entities.detected_highlight import DetectedHighlight
from ...domain.services.highlight_analyzer import HighlightAnalyzer
from ...domain.repositories.stream_repository import StreamRepository
from ...domain.repositories.highlight_repository import HighlightRepository
from ...infrastructure.ingestion.stream_segmenter import StreamSegmenter, SegmentInfo
from ...infrastructure.media.ffmpeg_segmenter import SegmentInfo as FFmpegSegmentInfo


class StreamAnalysisService:
    """Application service that orchestrates stream segmentation and analysis.
    
    This service acts as the coordinator between:
    - Infrastructure layer (StreamSegmenter for FFmpeg operations)
    - Domain layer (HighlightAnalyzer for business logic)
    - Repositories for persistence
    """
    
    def __init__(
        self,
        segmenter: StreamSegmenter,
        highlight_service: HighlightAnalyzer,
        stream_repo: StreamRepository,
        highlight_repo: HighlightRepository
    ):
        """Initialize the stream analysis service.
        
        Args:
            segmenter: Infrastructure component for stream segmentation
            highlight_service: Domain service for highlight analysis
            stream_repo: Repository for stream persistence
            highlight_repo: Repository for highlight persistence
        """
        self.segmenter = segmenter
        self.highlight_service = highlight_service
        self.stream_repo = stream_repo
        self.highlight_repo = highlight_repo
    
    async def analyze_stream(
        self,
        stream: Stream,
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Analyze a stream by segmenting and processing each segment.
        
        This method orchestrates the complete stream analysis workflow:
        1. Start stream segmentation
        2. Convert infrastructure segments to domain entities
        3. Analyze each segment using domain service
        4. Persist highlights
        5. Yield progress updates
        
        Args:
            stream: The stream to analyze
            dimension_set: Dimension set for analysis
            agent_config: Optional agent configuration
            progress_callback: Optional callback for progress updates
            
        Yields:
            Progress updates with segment and highlight information
        """
        # Start segmentation
        await self.segmenter.start()
        
        try:
            segments_processed = 0
            total_highlights = 0
            
            # Process segments as they arrive
            async for segment_info in self.segmenter.get_segments():
                # Convert infrastructure segment to domain entity
                video_segment = self._create_video_segment(
                    stream, segment_info, segments_processed
                )
                
                # Analyze segment using domain service
                try:
                    highlights = await self.highlight_service.analyze_segment(
                        segment=video_segment,
                        dimension_set=dimension_set,
                        agent_config=agent_config
                    )
                    
                    # Persist highlights
                    for highlight in highlights:
                        await self.highlight_repo.save(highlight)
                    
                    # Update counters
                    segments_processed += 1
                    total_highlights += len(highlights)
                    
                    # Yield progress update
                    result = {
                        "event": "segment_analyzed",
                        "segment_index": video_segment.segment_index,
                        "highlights_found": len(highlights),
                        "total_segments": segments_processed,
                        "total_highlights": total_highlights,
                        "segment_info": {
                            "start_time": video_segment.start_time,
                            "end_time": video_segment.end_time,
                            "duration": video_segment.duration_seconds
                        },
                        "highlights": [self._highlight_to_dict(h) for h in highlights]
                    }
                    
                    if progress_callback:
                        await progress_callback(result)
                    
                    yield result
                    
                except Exception as e:
                    # Handle analysis errors gracefully
                    error_result = {
                        "event": "segment_error",
                        "segment_index": segments_processed,
                        "error": str(e),
                        "total_segments": segments_processed,
                        "total_highlights": total_highlights
                    }
                    
                    if progress_callback:
                        await progress_callback(error_result)
                    
                    yield error_result
            
            # Final summary
            yield {
                "event": "analysis_complete",
                "total_segments": segments_processed,
                "total_highlights": total_highlights,
                "stream_id": stream.id
            }
            
        finally:
            # Ensure segmenter is stopped
            await self.segmenter.stop()
    
    def _create_video_segment(
        self,
        stream: Stream,
        segment_info: SegmentInfo,
        segment_index: int
    ) -> VideoSegment:
        """Convert infrastructure segment info to domain video segment.
        
        Args:
            stream: The parent stream
            segment_info: Segment info from infrastructure
            segment_index: Index of this segment
            
        Returns:
            Domain VideoSegment entity
        """
        # Handle both types of segment info
        if isinstance(segment_info, FFmpegSegmentInfo):
            path = segment_info.path
            start_time = segment_info.start_time
            end_time = segment_info.end_time
        else:
            # Handle dictionary format
            path = Path(segment_info.get("path", segment_info.get("file_path")))
            start_time = segment_info.get("start_time", 0)
            end_time = segment_info.get("end_time", start_time + segment_info.get("duration", 30))
        
        return VideoSegment.create(
            stream_id=stream.id,
            segment_index=segment_index,
            file_path=path,
            start_time=start_time,
            end_time=end_time,
            metadata={
                "stream_platform": stream.platform.value,
                "stream_title": stream.title,
                "original_segment_info": str(segment_info)
            }
        )
    
    def _highlight_to_dict(self, highlight: DetectedHighlight) -> Dict[str, Any]:
        """Convert highlight entity to dictionary for output.
        
        Args:
            highlight: The highlight entity
            
        Returns:
            Dictionary representation
        """
        return {
            "id": highlight.id,
            "start_time": highlight.start_time.value,
            "end_time": highlight.end_time.value,
            "duration": highlight.duration.value,
            "confidence_score": highlight.confidence_score.value,
            "highlight_types": highlight.highlight_types,
            "title": highlight.title,
            "description": highlight.description,
            "dimension_scores": highlight.dimension_scores
        }


class StreamAnalysisCoordinator:
    """Higher-level coordinator for managing multiple stream analyses.
    
    This can be used by Celery tasks or API endpoints to manage
    the analysis workflow.
    """
    
    def __init__(
        self,
        analysis_service: StreamAnalysisService,
        stream_repo: StreamRepository
    ):
        """Initialize the coordinator.
        
        Args:
            analysis_service: The stream analysis service
            stream_repo: Repository for stream operations
        """
        self.analysis_service = analysis_service
        self.stream_repo = stream_repo
    
    async def process_stream(
        self,
        stream_id: int,
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Process a stream by ID.
        
        Args:
            stream_id: ID of the stream to process
            dimension_set: Dimension set for analysis
            agent_config: Optional agent configuration
            progress_callback: Optional progress callback
            
        Returns:
            Summary of the analysis results
        """
        # Get stream
        stream = await self.stream_repo.get(stream_id)
        if not stream:
            raise ValueError(f"Stream {stream_id} not found")
        
        # Update stream status
        stream.start_processing()
        await self.stream_repo.save(stream)
        
        # Collect results
        results = {
            "segments_processed": 0,
            "highlights_found": 0,
            "errors": []
        }
        
        try:
            # Process stream
            async for update in self.analysis_service.analyze_stream(
                stream=stream,
                dimension_set=dimension_set,
                agent_config=agent_config,
                progress_callback=progress_callback
            ):
                if update["event"] == "segment_analyzed":
                    results["segments_processed"] += 1
                    results["highlights_found"] += update["highlights_found"]
                elif update["event"] == "segment_error":
                    results["errors"].append({
                        "segment": update["segment_index"],
                        "error": update["error"]
                    })
            
            # Mark stream as completed
            stream.complete_processing()
            await self.stream_repo.save(stream)
            
        except Exception as e:
            # Mark stream as failed
            stream.fail_processing(str(e))
            await self.stream_repo.save(stream)
            raise
        
        return results