"""B2B Stream Agent for configurable highlight detection.

This agent adapts the B2C StreamAgent architecture for B2B use cases,
allowing enterprise customers to configure prompts, thresholds, and scoring
according to their specific requirements.
"""

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Deque

from ..entities.stream import Stream
from ..entities.highlight import Highlight, HighlightCandidate
from ..entities.highlight_agent_config import HighlightAgentConfig
from ..value_objects.scoring_config import ScoringDimensions
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


@dataclass
class AgentMemoryEntry:
    """A single entry in agent memory for B2B context."""

    timestamp: float  # Stream time in seconds
    entry_type: str  # highlight, context, analysis, etc.
    content: Dict[str, Any]
    importance: float  # 0.0 to 1.0
    consumer_context: Optional[Dict[str, Any]] = None




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
        agent_config: HighlightAgentConfig,
        content_analyzer: Optional[Any] = None,  # Injected analyzer
    ):
        """Initialize B2B stream agent.

        Args:
            stream: Stream entity being processed
            agent_config: Consumer's highlight detection configuration
            content_analyzer: Optional content analysis service
        """
        self.stream = stream
        self.agent_config = agent_config
        self.content_analyzer = content_analyzer

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

        # Memory system (bounded for B2B efficiency)
        self.memory_entries: Deque[AgentMemoryEntry] = deque(maxlen=200)
        self.recent_highlights: Deque[HighlightCandidate] = deque(maxlen=50)
        self.error_history: Deque[str] = deque(maxlen=20)

        # Context tracking
        self.current_context: Dict[str, Any] = {}
        self.stream_start_time = datetime.utcnow()

        # Control flags
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Consumer-specific state
        self.consumer_metrics: Dict[str, Any] = {
            "config_version": agent_config.version,
            "organization_id": agent_config.organization_id,
            "content_type": agent_config.content_type,
            "game_name": agent_config.game_name,
        }

    # ============================================================================
    # MEMORY AND CONTEXT MANAGEMENT
    # ============================================================================

    def add_memory_entry(self, entry: AgentMemoryEntry) -> None:
        """Add entry to agent memory with automatic cleanup."""
        self.memory_entries.append(entry)

    def get_recent_context(self, seconds: float = 300) -> List[AgentMemoryEntry]:
        """Get memory entries from the last N seconds."""
        if not self.memory_entries:
            return []

        current_time = (datetime.utcnow() - self.stream_start_time).total_seconds()
        cutoff_time = current_time - seconds

        return [
            entry for entry in self.memory_entries if entry.timestamp >= cutoff_time
        ]

    def update_context(self, context_type: str, data: Dict[str, Any]) -> None:
        """Update the current processing context."""
        self.current_context[context_type] = data
        self.last_activity = datetime.utcnow()

        # Add to memory
        self.add_memory_entry(
            AgentMemoryEntry(
                timestamp=(datetime.utcnow() - self.stream_start_time).total_seconds(),
                entry_type="context_update",
                content={"type": context_type, "data": data},
                importance=0.3,
                consumer_context={"config_version": self.agent_config.version},
            )
        )

    def get_context_for_prompt(self) -> Dict[str, Any]:
        """Build context dictionary for prompt rendering."""
        recent_highlights = [h.description for h in list(self.recent_highlights)[-3:]]

        return {
            "context": f"Stream: {self.stream.title}, Recent highlights: {'; '.join(recent_highlights) if recent_highlights else 'None'}",
            "content_type": self.agent_config.content_type,
            "game_name": self.agent_config.game_name or "Unknown",
            "organization": self.agent_config.organization_id,
            "stream_duration": str(datetime.utcnow() - self.stream_start_time),
            "highlights_so_far": len(self.recent_highlights),
            "current_state": self.current_context.get("stream_state", "active"),
        }

    # ============================================================================
    # HIGHLIGHT ANALYSIS AND SCORING
    # ============================================================================

    @traced_service_method(name="analyze_content_segment")
    async def analyze_content_segment(
        self,
        segment_data: Dict[str, Any],  # Generic segment data
    ) -> List[HighlightCandidate]:
        """Analyze a content segment for highlights using consumer configuration.

        Args:
            segment_data: Dictionary containing segment information

        Returns:
            List of highlight candidates
        """
        try:
            self.segments_processed += 1

            with logfire.span("segment_analysis.start") as span:
                span.set_attribute("segment.id", segment_data.get("id", "unknown"))
                span.set_attribute("agent.id", self.agent_id)
                span.set_attribute("organization.id", self.agent_config.organization_id)

                if not self.content_analyzer:
                    span.set_attribute("skipped", True)
                    span.set_attribute("reason", "no_content_analyzer")
                    return []

            # Get effective prompt with consumer customization
            with logfire.span("prepare_analysis_prompt") as span:
                context = self.get_context_for_prompt()
                analysis_prompt = self.agent_config.get_effective_prompt(context)
                span.set_attribute("prompt.length", len(analysis_prompt))
                span.set_attribute(
                    "context.highlights_count", len(self.recent_highlights)
                )

            # Analyze using configured prompt
            with logfire.span("content_analyzer.analyze") as span:
                analysis_start = datetime.utcnow()
                analysis_result = await self.content_analyzer.analyze_with_prompt(
                    segment_data=segment_data,
                    prompt=analysis_prompt,
                    scoring_config=self.agent_config.dimension_weights,
                )
                analysis_duration = (datetime.utcnow() - analysis_start).total_seconds()
                span.set_attribute("analysis.duration_seconds", analysis_duration)
                span.set_attribute(
                    "analysis.candidates_found",
                    len(analysis_result.get("candidates", [])),
                )

                # Track AI analysis metrics
                metrics.record_highlight_processing_time(
                    duration_seconds=analysis_duration,
                    stage="ai_analysis",
                    platform=self.stream.platform.value,
                )

            # Convert analysis to highlight candidates
            candidates = []
            with logfire.span("create_candidates") as span:
                for i, result in enumerate(analysis_result.get("candidates", [])):
                    candidate = self._create_highlight_candidate(result, segment_data)
                    if candidate:
                        candidates.append(candidate)
                        self.candidates_evaluated += 1

                        # Track high-scoring candidates
                        if candidate.final_score > 0.8:
                            logfire.info(
                                "highlight.candidate.high_score",
                                candidate_id=candidate.id,
                                score=candidate.final_score,
                                description=candidate.description[:100],
                                organization_id=self.agent_config.organization_id,
                            )

                span.set_attribute("candidates.created", len(candidates))
                span.set_attribute(
                    "candidates.evaluated_total", self.candidates_evaluated
                )

            # Add analysis to memory
            self.add_memory_entry(
                AgentMemoryEntry(
                    timestamp=(
                        datetime.utcnow() - self.stream_start_time
                    ).total_seconds(),
                    entry_type="content_analysis",
                    content={
                        "segment_id": segment_data.get("id", "unknown"),
                        "candidates_found": len(candidates),
                        "analysis_duration": analysis_result.get("processing_time", 0),
                    },
                    importance=0.6 if candidates else 0.3,
                    consumer_context={
                        "config_version": self.agent_config.version,
                        "prompt_length": len(analysis_prompt),
                    },
                )
            )

            return candidates

        except Exception as e:
            error_msg = f"Content analysis failed: {str(e)}"
            self.error_history.append(error_msg)

            logfire.error(
                "agent.analysis.failed",
                error=str(e),
                segment_id=segment_data.get("id", "unknown"),
                agent_id=self.agent_id,
                organization_id=self.agent_config.organization_id,
            )

            raise ProcessingError(error_msg) from e

    def _create_highlight_candidate(
        self, analysis_result: Dict[str, Any], segment_data: Dict[str, Any]
    ) -> Optional[HighlightCandidate]:
        """Create a highlight candidate from analysis results."""
        try:
            # Extract dimensions from analysis
            dimensions_data = analysis_result.get("dimensions", {})
            dimensions = ScoringDimensions.from_dict(dimensions_data)

            # Calculate final score using consumer configuration
            detected_keywords = analysis_result.get("keywords", [])
            context_type = analysis_result.get("context_type")

            final_score = self.agent_config.calculate_highlight_score(
                dimensions=dimensions.to_dict(),
                keywords=detected_keywords,
                context_type=context_type,
            )

            # Check if score meets consumer thresholds
            highlight_type = analysis_result.get("type", "general")
            if not self.agent_config.score_thresholds.meets_threshold(
                final_score, highlight_type
            ):
                return None

            return HighlightCandidate(
                id=str(uuid.uuid4()),
                start_time=analysis_result.get("start_time", 0),
                end_time=analysis_result.get("end_time", 0),
                peak_time=analysis_result.get("peak_time", 0),
                description=analysis_result.get("description", ""),
                confidence=analysis_result.get("confidence", 0.0),
                dimensions=dimensions,
                final_score=final_score,
                detected_keywords=detected_keywords,
                context_type=context_type,
                metadata={
                    "segment_id": segment_data.get("id"),
                    "analysis_version": self.agent_config.version,
                },
            )

        except Exception as e:
            self.error_history.append(f"Candidate creation failed: {str(e)}")
            return None

    @traced_service_method(name="should_create_highlight")
    async def should_create_highlight(self, candidate: HighlightCandidate) -> bool:
        """Determine if a highlight should be created using consumer rules."""
        try:
            with logfire.span("evaluate_highlight_candidate") as span:
                span.set_attribute("candidate.id", candidate.id)
                span.set_attribute("candidate.score", candidate.final_score)
                span.set_attribute("candidate.confidence", candidate.confidence)

                # Check basic score threshold
                if candidate.final_score < self.agent_config.min_confidence_threshold:
                    span.set_attribute("decision", "rejected")
                    span.set_attribute("reason", "below_threshold")
                    return False

            # Check timing constraints
            timing = self.agent_config.timing_config

            with logfire.span("check_timing_constraints") as timing_span:
                # Check minimum spacing
                if self.recent_highlights:
                    last_highlight = self.recent_highlights[-1]
                    time_since_last = candidate.start_time - last_highlight.end_time
                    timing_span.set_attribute("time_since_last", time_since_last)
                    timing_span.set_attribute("min_spacing", timing.min_spacing_seconds)

                    if time_since_last < timing.min_spacing_seconds:
                        span.set_attribute("decision", "rejected")
                        span.set_attribute("reason", "too_close_to_previous")
                        return False

            # Check highlights per window limit
            recent_window = [
                h
                for h in self.recent_highlights
                if candidate.start_time - h.start_time <= 300  # 5 minutes
            ]

            if len(recent_window) >= timing.max_per_5min_window:
                # Only allow if this is significantly better
                if recent_window:
                    best_recent_score = max(h.final_score for h in recent_window)
                    if candidate.final_score <= best_recent_score + 0.1:
                        return False

            # Check similarity to recent highlights
            for recent in list(self.recent_highlights)[-10:]:  # Check last 10
                similarity = self._calculate_similarity(candidate, recent)
                if similarity > self.agent_config.similarity_threshold:
                    # Similar highlight exists, only create if significantly better
                    if candidate.final_score <= recent.final_score + 0.05:
                        return False

            span.set_attribute("decision", "approved")
            return True

        except Exception as e:
            self.error_history.append(f"Highlight decision failed: {str(e)}")
            return False

    def _calculate_similarity(
        self, candidate1: HighlightCandidate, candidate2: HighlightCandidate
    ) -> float:
        """Calculate similarity between two highlight candidates."""
        # Time proximity (closer = more similar)
        time_diff = abs(candidate1.start_time - candidate2.start_time)
        time_similarity = max(0, 1 - (time_diff / 120))  # 2 minute window

        # Description similarity (simple keyword overlap)
        desc1_words = set(candidate1.description.lower().split())
        desc2_words = set(candidate2.description.lower().split())

        if desc1_words or desc2_words:
            word_similarity = len(desc1_words & desc2_words) / len(
                desc1_words | desc2_words
            )
        else:
            word_similarity = 0

        # Combine similarities
        return (time_similarity * 0.6) + (word_similarity * 0.4)

    @traced_service_method(name="create_highlight")
    async def create_highlight(
        self, candidate: HighlightCandidate
    ) -> Optional[Highlight]:
        """Create a highlight from an approved candidate."""
        try:
            with logfire.span("create_highlight_from_candidate") as span:
                span.set_attribute("candidate.id", candidate.id)
                span.set_attribute("candidate.score", candidate.final_score)
                span.set_attribute("organization.id", self.agent_config.organization_id)

                # Record usage in configuration
                self.agent_config.record_usage()

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
                    "dimensions": candidate.dimensions.to_dict(),
                    "keywords": candidate.detected_keywords,
                    "context_type": candidate.context_type,
                    "trigger_type": candidate.trigger_type,
                },
            )

            # Add to recent highlights tracking
            self.recent_highlights.append(candidate)
            self.highlights_created += 1

            # Track highlight creation metrics
            metrics.increment_highlights_detected(
                count=1,
                platform=self.stream.platform.value,
                organization_id=str(self.agent_config.organization_id),
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

            # Add to memory
            self.add_memory_entry(
                AgentMemoryEntry(
                    timestamp=candidate.start_time,
                    entry_type="highlight_created",
                    content={
                        "candidate_id": candidate.id,
                        "score": candidate.final_score,
                        "description": candidate.description,
                    },
                    importance=0.9,
                    consumer_context={
                        "config_version": self.agent_config.version,
                        "organization_id": self.agent_config.organization_id,
                    },
                )
            )

            # Log significant highlights
            if candidate.final_score > 0.9:
                logfire.info(
                    "highlight.created.high_value",
                    highlight_id=highlight.id,
                    score=candidate.final_score,
                    confidence=candidate.confidence,
                    description=highlight.description[:100],
                    organization_id=self.agent_config.organization_id,
                    agent_id=self.agent_id,
                )

            return highlight

        except Exception as e:
            error_msg = f"Highlight creation failed: {str(e)}"
            self.error_history.append(error_msg)

            logfire.error(
                "agent.highlight.creation_failed",
                error=str(e),
                candidate_id=candidate.id,
                agent_id=self.agent_id,
                organization_id=self.agent_config.organization_id,
            )

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

                # Validate configuration
                config_errors = self.agent_config.validate_configuration()
                if config_errors:
                    span.set_attribute("error", True)
                    span.set_attribute(
                        "error.message", f"Invalid configuration: {config_errors}"
                    )
                    raise BusinessRuleViolation(
                        f"Invalid configuration: {config_errors}"
                    )

                span.set_attribute("status", "active")

                # Log agent start event
                logfire.info(
                    "agent.started",
                    agent_id=self.agent_id,
                    stream_id=self.stream.id,
                    organization_id=self.agent_config.organization_id,
                    config_type=self.agent_config.config_type,
                    content_type=self.agent_config.content_type,
                )

            # Initialize tasks based on configuration
            # Note: This is a simplified version - full implementation would
            # include segment processing, wake word handling, etc.

        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error_history.append(f"Agent start failed: {str(e)}")

            logfire.error(
                "agent.start.failed",
                error=str(e),
                agent_id=self.agent_id,
                stream_id=self.stream.id,
                organization_id=self.agent_config.organization_id,
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
                organization_id=self.agent_config.organization_id,
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
                "version": self.agent_config.version,
                "organization_id": self.agent_config.organization_id,
                "content_type": self.agent_config.content_type,
                "game_name": self.agent_config.game_name,
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
            "memory_stats": {
                "memory_entries": len(self.memory_entries),
                "recent_highlights": len(self.recent_highlights),
                "error_count": len(self.error_history),
            },
            "consumer_metrics": self.consumer_metrics,
            "recent_errors": list(self.error_history)[-5:]
            if self.error_history
            else [],
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
