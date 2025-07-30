"""B2B Stream Agent for configurable highlight detection.

This agent adapts the B2C StreamAgent architecture for B2B use cases,
allowing enterprise customers to configure prompts, thresholds, and scoring
according to their specific requirements.
"""

import asyncio
import uuid
# deque removed for simplified processing
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from ..entities.stream import Stream
from ..entities.highlight import Highlight, HighlightCandidate
from ..entities.highlight_agent_config import HighlightAgentConfig
from ..entities.stream_processing_config import StreamProcessingConfig
from ..entities.dimension_set import DimensionSet
from ..value_objects.processing_options import ProcessingOptions
from ..exceptions import BusinessRuleViolation, ProcessingError
from src.infrastructure.observability import traced_service_method, metrics
import logfire


class AgentStatus(str, Enum):
    """B2B Agent operational status."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# AgentMemoryEntry removed for streamlined processing


class B2BStreamAgent:
    """B2B Stream processing agent with configurable highlight detection.

    This agent provides enterprise customers with:
    - Configurable prompts and scoring
    - Consumer-specific thresholds
    - Custom highlight types and logic
    - Advanced memory and context management
    - Detailed analytics and reporting
    """

    def __init__(
        self,
        stream: Stream,
        agent_config: HighlightAgentConfig | StreamProcessingConfig,  # Accept either config type
        gemini_processor: Any,  # Required Gemini processor
        dimension_set: DimensionSet,  # Required dimension set
        processing_options: Optional[ProcessingOptions] = None,
    ):
        """Initialize B2B stream agent.

        Args:
            stream: Stream entity being processed
            agent_config: Consumer's highlight detection configuration
            gemini_processor: Required Gemini video processor for dimension-based analysis
            dimension_set: Required dimension set for scoring
            processing_options: Optional processing configuration
        """
        self.stream = stream
        self.agent_config = agent_config
        self.gemini_processor = gemini_processor
        self.dimension_set = dimension_set
        self.processing_options = processing_options or ProcessingOptions()

        # Validate required components
        if not self.gemini_processor:
            raise ValueError("Gemini processor is required for highlight detection")
        if not self.dimension_set:
            raise ValueError("Dimension set is required for highlight detection")

        # Agent identification
        self.agent_id = str(uuid.uuid4())
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.utcnow()
        self.started_at = datetime.utcnow()

        # Processing statistics
        self.segments_processed = 0
        self.highlights_created = 0
        self.candidates_evaluated = 0
        self.last_activity = datetime.utcnow()

        # Simplified tracking
        self.recent_highlights: List[HighlightCandidate] = []  # Keep last few for metrics
        self.stream_start_time = datetime.utcnow()

        # Control flags
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Consumer-specific state
        self.consumer_metrics: Dict[str, Any] = self._extract_consumer_metrics(agent_config)

    # ============================================================================
    # CONFIG COMPATIBILITY HELPERS
    # ============================================================================
    
    def _extract_consumer_metrics(self, config) -> Dict[str, Any]:
        """Extract consumer metrics from either config type."""
        if isinstance(config, StreamProcessingConfig):
            return {
                "config_type": "simplified",
                "organization_id": config.organization_id,
                "content_type": "general",  # Simplified config doesn't specify
                "confidence_threshold": config.confidence_threshold,
                "max_highlights": config.max_highlights,
            }
        else:  # HighlightAgentConfig
            return {
                "config_version": config.version,
                "organization_id": config.organization_id,
                "content_type": config.content_type,
                "game_name": config.game_name,
            }
    
    def _get_confidence_threshold(self) -> float:
        """Get confidence threshold from either config type."""
        if isinstance(self.agent_config, StreamProcessingConfig):
            return self.agent_config.confidence_threshold
        else:
            return self.agent_config.min_confidence_threshold
    
    def _get_organization_id(self) -> int:
        """Get organization ID from either config type."""
        return self.agent_config.organization_id
    
    def _record_config_usage(self) -> None:
        """Record usage on either config type."""
        self.agent_config.record_usage()

    # ============================================================================
    # SIMPLIFIED CONTEXT MANAGEMENT  
    # ============================================================================

    def get_context_for_prompt(self) -> Dict[str, Any]:
        """Build basic context dictionary for prompt rendering."""
        recent_highlights = [h.description for h in self.recent_highlights[-3:]] if self.recent_highlights else []

        return {
            "context": f"Stream: {self.stream.title}, Recent highlights: {'; '.join(recent_highlights) if recent_highlights else 'None'}",
            "content_type": self.consumer_metrics.get("content_type", "general"),
            "organization": self._get_organization_id(),
            "stream_duration": str(datetime.utcnow() - self.stream_start_time),
            "highlights_so_far": len(self.recent_highlights),
        }

    # ============================================================================
    # HIGHLIGHT ANALYSIS AND SCORING
    # ============================================================================

    @traced_service_method(name="analyze_content_segment")
    async def analyze_content_segment(
        self,
        segment_data: Dict[str, Any],  # Generic segment data
    ) -> List[HighlightCandidate]:
        """Analyze a content segment for highlights using Gemini dimension-based analysis.

        This method delegates to analyze_video_segment_with_gemini which is now
        the primary analysis method using the dimension framework.

        Args:
            segment_data: Dictionary containing segment information

        Returns:
            List of highlight candidates
        """
        # Delegate to Gemini dimension-based analysis
        return await self.analyze_video_segment_with_gemini(segment_data)

    # Complex candidate creation removed - handled by GeminiVideoProcessor

    @traced_service_method(name="should_create_highlight")
    async def should_create_highlight(self, candidate: HighlightCandidate) -> bool:
        """Simplified highlight validation - just check basic threshold."""
        try:
            with logfire.span("evaluate_highlight_candidate") as span:
                span.set_attribute("candidate.id", candidate.id)
                span.set_attribute("candidate.score", candidate.final_score)
                span.set_attribute("candidate.confidence", candidate.confidence)

                # Simple threshold check
                if candidate.final_score < self._get_confidence_threshold():
                    span.set_attribute("decision", "rejected")
                    span.set_attribute("reason", "below_threshold")
                    return False

                span.set_attribute("decision", "approved")
                return True

        except Exception as e:
            logfire.error("Highlight validation failed", error=str(e))
            return False

    # Similarity calculation removed for streamlined processing

    @traced_service_method(name="create_highlight")
    async def create_highlight(
        self, candidate: HighlightCandidate
    ) -> Optional[Highlight]:
        """Create a highlight from an approved candidate."""
        try:
            with logfire.span("create_highlight_from_candidate") as span:
                span.set_attribute("candidate.id", candidate.id)
                span.set_attribute("candidate.score", candidate.final_score)
                span.set_attribute("organization.id", self._get_organization_id())

                # Record usage in configuration
                self._record_config_usage()

            # Create highlight entity
            highlight = Highlight(
                id=None,  # Will be assigned by repository
                stream_id=self.stream.id,
                start_time=candidate.start_time,
                end_time=candidate.end_time,
                title=candidate.description[:100],  # Truncate for title
                description=candidate.description,
                confidence_score=candidate.confidence,
                metadata={
                    "agent_id": self.agent_id,
                    "config_version": self.agent_config.version,
                    "final_score": candidate.final_score,
                    "dimensions": candidate.dimensions.to_dict() if hasattr(candidate.dimensions, 'to_dict') else candidate.dimensions,
                    "keywords": candidate.detected_keywords,
                    "context_type": candidate.context_type,
                    "trigger_type": candidate.trigger_type,
                },
            )

            # Add to recent highlights tracking (keep last 10 for metrics)
            self.recent_highlights.append(candidate)
            if len(self.recent_highlights) > 10:
                self.recent_highlights.pop(0)
            self.highlights_created += 1

            # Track highlight creation metrics
            metrics.increment_highlights_detected(
                count=1,
                platform=self.stream.platform.value,
                organization_id=str(self._get_organization_id()),
                detection_method="b2b_agent",
            )

            # Track confidence distribution
            metrics.record_highlight_confidence(
                confidence=candidate.confidence,
                detection_method="b2b_agent",
                platform=self.stream.platform.value,
            )

            span.set_attribute("highlight.created", True)
            span.set_attribute("highlights.total", self.highlights_created)

            # Log significant highlights
            if candidate.final_score > 0.9:
                logfire.info(
                    "highlight.created.high_value",
                    highlight_id=highlight.id,
                    score=candidate.final_score,
                    confidence=candidate.confidence,
                    description=highlight.description[:100],
                    organization_id=self._get_organization_id(),
                    agent_id=self.agent_id,
                )

            return highlight

        except Exception as e:
            error_msg = f"Highlight creation failed: {str(e)}"

            logfire.error(
                "agent.highlight.creation_failed",
                error=str(e),
                candidate_id=candidate.id,
                agent_id=self.agent_id,
                organization_id=self._get_organization_id(),
            )

            raise ProcessingError(error_msg) from e

    @traced_service_method(name="analyze_video_segment_with_gemini")
    async def analyze_video_segment_with_gemini(
        self,
        segment_data: Dict[str, Any],
    ) -> List[HighlightCandidate]:
        """
        Analyze a video segment using Gemini's video understanding with dimension framework.

        This is the primary and only method for highlight detection, using:
        1. Dimension set for structured scoring
        2. Gemini's video understanding API
        3. Dimension-aware prompts
        4. Automated refinement and validation
        5. Ranked highlight candidates
        """

        try:
            # Get video file path
            video_path = segment_data.get("path")
            if not video_path:
                raise ValueError("No video path in segment data")

            # Prepare segment info
            segment_info = {
                "id": segment_data.get("id"),
                "start_time": segment_data.get("start_time", 0),
                "end_time": segment_data.get("end_time", 0),
                "duration": segment_data.get("duration", 0),
                "context": self.get_context_for_prompt(),
            }

            # Analyze with Gemini using dimension framework
            with logfire.span("gemini_dimension_analysis") as span:
                span.set_attribute("segment.id", segment_data.get("id"))
                span.set_attribute("dimension_set.name", self.dimension_set.name)
                span.set_attribute(
                    "dimension_count", len(self.dimension_set.dimensions)
                )
                span.set_attribute("agent.config_version", self.agent_config.version)

                analysis = await self.gemini_processor.analyze_video_with_dimensions(
                    video_path=video_path,
                    segment_info=segment_info,
                    dimension_set=self.dimension_set,
                    agent_config=self.agent_config,
                )

                span.set_attribute("highlights.found", len(analysis.highlights))
                span.set_attribute("processing.refinement_enabled", False)  # Refinement removed

            # Convert to highlight candidates using dimension framework
            candidates = self.gemini_processor.convert_to_highlight_candidates(
                analysis,
                self.dimension_set,
                segment_info,
                min_confidence=self._get_confidence_threshold(),
            )

            # Update statistics
            self.segments_processed += 1
            self.candidates_evaluated += len(candidates)

            # Track high-scoring candidates
            for candidate in candidates:
                if candidate.final_score > 0.8:
                    logfire.info(
                        "highlight.candidate.high_score_gemini",
                        candidate_id=candidate.id,
                        score=candidate.final_score,
                        description=candidate.description[:100],
                        organization_id=self._get_organization_id(),
                        agent_id=self.agent_id,
                    )

            return candidates

        except Exception as e:
            error_msg = f"Gemini video analysis failed: {str(e)}"

            logfire.error(
                "agent.gemini_analysis.failed",
                error=str(e),
                segment_id=segment_data.get("id"),
                agent_id=self.agent_id,
                organization_id=self._get_organization_id(),
            )

            # Re-raise error as Gemini is now the required method
            raise ProcessingError(error_msg) from e

    # ============================================================================
    # AGENT LIFECYCLE
    # ============================================================================

    @traced_service_method(name="start_b2b_agent")
    async def start(self) -> None:
        """Start the B2B stream agent."""
        try:
            with logfire.span("agent.start") as span:
                span.set_attribute("agent.id", self.agent_id)
                span.set_attribute("stream.id", self.stream.id)
                span.set_attribute("organization.id", self.agent_config.organization_id)
                span.set_attribute("config.version", self.agent_config.version)

                self.status = AgentStatus.ACTIVE
                self._running = True
                self.started_at = datetime.utcnow()

                # Simple validation for streamlined config
                if isinstance(self.agent_config, StreamProcessingConfig):
                    if not self.agent_config.validate():
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", "Invalid configuration parameters")
                        raise BusinessRuleViolation("Invalid configuration parameters")

                span.set_attribute("status", "active")

                # Log agent start event
                logfire.info(
                    "agent.started",
                    agent_id=self.agent_id,
                    stream_id=self.stream.id,
                    organization_id=self._get_organization_id(),
                    config_type=self.agent_config.config_type,
                    content_type=self.agent_config.content_type,
                )

            # Initialize tasks based on configuration
            # Note: This is a simplified version - full implementation would
            # include segment processing, wake word handling, etc.

        except Exception as e:
            self.status = AgentStatus.ERROR

            logfire.error(
                "agent.start.failed",
                error=str(e),
                agent_id=self.agent_id,
                stream_id=self.stream.id,
                organization_id=self._get_organization_id(),
            )

            raise

    @traced_service_method(name="stop_b2b_agent")
    async def stop(self) -> None:
        """Stop the B2B stream agent."""
        with logfire.span("agent.stop") as span:
            span.set_attribute("agent.id", self.agent_id)
            span.set_attribute("highlights.created", self.highlights_created)
            span.set_attribute("segments.processed", self.segments_processed)

            self._running = False
            self.status = AgentStatus.STOPPING

            # Cancel all tasks
            for task in self._tasks:
                task.cancel()

            # Wait for cleanup
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

            self.status = AgentStatus.STOPPED

            # Log agent stop event with final metrics
            uptime = (datetime.utcnow() - self.started_at).total_seconds()
            logfire.info(
                "agent.stopped",
                agent_id=self.agent_id,
                stream_id=self.stream.id,
                organization_id=self._get_organization_id(),
                uptime_seconds=uptime,
                highlights_created=self.highlights_created,
                segments_processed=self.segments_processed,
                candidates_evaluated=self.candidates_evaluated,
                success_rate=self.highlights_created
                / max(1, self.candidates_evaluated),
            )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for B2B reporting."""
        uptime = (datetime.utcnow() - self.started_at).total_seconds()

        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "uptime_seconds": uptime,
            "configuration": {
                "organization_id": self._get_organization_id(),
                "config_type": "simplified" if isinstance(self.agent_config, StreamProcessingConfig) else "legacy",
                "confidence_threshold": self._get_confidence_threshold(),
            },
            "processing_stats": {
                "segments_processed": self.segments_processed,
                "candidates_evaluated": self.candidates_evaluated,
                "highlights_created": self.highlights_created,
                "success_rate": self.highlights_created
                / max(1, self.candidates_evaluated),
                "processing_rate": self.segments_processed
                / max(1, uptime / 60),  # per minute
            },
            "quality_metrics": {
                "avg_confidence": sum(h.confidence for h in self.recent_highlights)
                / max(1, len(self.recent_highlights)),
                "avg_score": sum(h.final_score for h in self.recent_highlights)
                / max(1, len(self.recent_highlights)),
                "score_distribution": self._get_score_distribution(),
            },
            "recent_highlights_count": len(self.recent_highlights),
            "consumer_metrics": self.consumer_metrics,
        }

    def _get_score_distribution(self) -> Dict[str, int]:
        """Get distribution of highlight scores for analytics."""
        if not self.recent_highlights:
            return {}

        distribution = {
            "0.0-0.5": 0,
            "0.5-0.7": 0,
            "0.7-0.8": 0,
            "0.8-0.9": 0,
            "0.9-1.0": 0,
        }

        for highlight in self.recent_highlights:
            score = highlight.final_score
            if score < 0.5:
                distribution["0.0-0.5"] += 1
            elif score < 0.7:
                distribution["0.5-0.7"] += 1
            elif score < 0.8:
                distribution["0.7-0.8"] += 1
            elif score < 0.9:
                distribution["0.8-0.9"] += 1
            else:
                distribution["0.9-1.0"] += 1

        return distribution
