"""Domain service for orchestrating highlight analysis.

This is a simplified, Pythonic version that focuses on orchestration
rather than business logic (which belongs in entities).
"""

import asyncio
from typing import List, Optional, Dict

from ..entities.stream import Stream
from ..entities.video_segment import VideoSegment
from ..entities.dimension_set_aggregate import DimensionSetAggregate
from ..entities.highlight_agent_config import HighlightAgentConfig
from ..entities.highlight import Highlight, HighlightCandidate
from ..exceptions import DomainError


class HighlightAnalyzer:
    """Orchestrates the highlight analysis process.
    
    This service coordinates between domain entities and the AI analyzer,
    but delegates business logic to the entities themselves.
    """
    
    def __init__(self, ai_analyzer):
        """Initialize with an AI analyzer.
        
        Uses duck typing - any object with analyze_video method works.
        """
        self.ai_analyzer = ai_analyzer
    
    async def analyze_segment(
        self,
        stream: Stream,
        segment: VideoSegment,
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig] = None
    ) -> List[Highlight]:
        """Analyze a video segment and create highlights.
        
        This method orchestrates the analysis process but delegates
        business rules to the domain entities.
        """
        # Let the segment handle its own state
        segment.start_analysis()
        
        try:
            # Get AI analysis results
            candidates = await self.ai_analyzer.analyze_video(
                video_path=segment.file_path.path,
                dimension_set=dimension_set,
                segment_info=segment.to_analysis_info(),
                agent_config=agent_config
            )
            
            # Let the stream create highlights (enforcing aggregate boundary)
            highlights = []
            for candidate in candidates:
                try:
                    highlight = stream.create_highlight_from_candidate(candidate)
                    highlights.append(highlight)
                except Exception as e:
                    # Log but continue with other candidates
                    pass
            
            # Update segment state
            highlight_ids = [h.id for h in highlights if h.id]
            segment.complete_analysis(highlight_ids)
            
            return highlights
            
        except Exception as e:
            segment.fail_analysis(str(e))
            raise DomainError(f"Analysis failed: {e}") from e
    
    async def analyze_stream_segments(
        self,
        stream: Stream,
        segments: List[VideoSegment],
        dimension_set: DimensionSetAggregate,
        agent_config: Optional[HighlightAgentConfig] = None,
        max_concurrent: int = 3
    ) -> Dict[int, List[Highlight]]:
        """Analyze multiple segments concurrently.
        
        Returns a mapping of segment index to highlights found.
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_limit(segment: VideoSegment):
            async with semaphore:
                if not segment.can_be_analyzed():
                    return
                
                try:
                    highlights = await self.analyze_segment(
                        stream, segment, dimension_set, agent_config
                    )
                    results[segment.segment_index] = highlights
                except Exception:
                    results[segment.segment_index] = []
        
        # Process all segments
        await asyncio.gather(
            *[analyze_with_limit(segment) for segment in segments],
            return_exceptions=True
        )
        
        return results
    
    @property
    def capabilities(self) -> Dict[str, any]:
        """Get analyzer capabilities."""
        # Duck typing - if analyzer has capabilities, use them
        if hasattr(self.ai_analyzer, 'capabilities'):
            return self.ai_analyzer.capabilities
        return {}