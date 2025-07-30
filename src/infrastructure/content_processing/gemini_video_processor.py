"""Enhanced Gemini video processor with dimension framework integration.

This module provides advanced video analysis using Google's Gemini API,
integrated with the flexible dimension framework for customizable highlight detection.
"""

import asyncio
from typing import List, Dict, Any
import uuid
import backoff

from google import genai
from google.genai import types
import logfire
from pydantic import BaseModel

from src.domain.entities.highlight import HighlightCandidate
from src.domain.entities.highlight_agent_config import HighlightAgentConfig
from src.domain.entities.dimension_set import DimensionSet
from src.domain.value_objects.scoring_config import ScoringDimensions
from src.infrastructure.content_processing.schemas.gemini_schemas import (
    GeminiVideoAnalysis,
    GeminiHighlight,
)
from src.infrastructure.observability import traced_service_method


class GeminiVideoProcessor:
    """
    Enhanced Gemini processor that uses the dimension framework
    for flexible, industry-agnostic highlight detection.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash-exp",
        max_retries: int = 3,
    ):
        """Initialize streamlined Gemini processor with dimension support.

        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model to use
            max_retries: Maximum retries for API calls
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries

        self.uploaded_files: Dict[str, Any] = {}
        self.logger = logfire.get_logger(__name__)

    @traced_service_method(name="analyze_with_dimensions")
    async def analyze_video_with_dimensions(
        self,
        video_path: str,
        segment_info: Dict[str, Any],
        dimension_set: DimensionSet,
        agent_config: HighlightAgentConfig,
    ) -> GeminiVideoAnalysis:
        """
        Analyze video using a specific dimension set.

        Args:
            video_path: Path to video file
            segment_info: Segment metadata
            dimension_set: The dimension set to use for scoring
            agent_config: Agent configuration for prompts

        Returns:
            GeminiVideoAnalysis with dimension-based scoring
        """
        # Removed caching for streamlined processing

        uploaded_file = None

        try:
            # Upload video
            with logfire.span("upload_video_for_dimension_analysis") as span:
                span.set_attribute("segment.id", segment_info.get("id"))
                span.set_attribute("dimension_set.name", dimension_set.name)
                span.set_attribute("dimension_count", len(dimension_set.dimensions))

                uploaded_file = await self._upload_video_with_retry(
                    video_path, segment_info.get("id", "unknown")
                )

                self.uploaded_files[uploaded_file.name] = uploaded_file
                self.logger.info(
                    f"Video uploaded for dimension analysis: {uploaded_file.name}",
                    extra={
                        "dimension_set": dimension_set.name,
                        "dimensions": list(dimension_set.dimensions.keys()),
                    },
                )

            # Build dimension-aware prompt
            analysis_prompt = self._build_dimension_aware_prompt(
                dimension_set, agent_config, segment_info
            )

            # Generate structured response
            with logfire.span("gemini_dimension_analysis") as span:
                span.set_attribute("prompt_length", len(analysis_prompt))

                response = await self._generate_with_structured_output(
                    uploaded_file.uri,
                    analysis_prompt,
                    response_schema=GeminiVideoAnalysis,
                )

                # Validate dimension scores match our dimension set
                response = self._validate_dimension_scores(response, dimension_set)

                span.set_attribute("highlights_found", len(response.highlights))
                self.logger.info(
                    f"Dimension analysis complete: {len(response.highlights)} highlights found"
                )

            # Removed refinement pipeline and caching for streamlined processing

            return response

        except Exception as e:
            self.logger.error(
                f"Dimension-based video analysis failed: {str(e)}",
                extra={"segment_id": segment_info.get("id")},
                exc_info=True,
            )
            raise

        finally:
            if uploaded_file:
                await self._cleanup_file(uploaded_file)

    def _build_dimension_aware_prompt(
        self,
        dimension_set: DimensionSet,
        agent_config: HighlightAgentConfig,
        segment_info: Dict[str, Any],
    ) -> str:
        """Build a prompt that incorporates dimension definitions."""
        # Get base prompt from agent config
        context = {
            "content_type": agent_config.content_type,
            "game_name": agent_config.game_name or "content",
            "industry": dimension_set.industry or "general",
            "dimension_set_name": dimension_set.name,
        }

        base_prompt = agent_config.get_effective_prompt(context)

        # Build dimension-specific instructions
        dimension_instructions = self._build_dimension_instructions(dimension_set)

        # Combine into comprehensive prompt
        return f"""
{base_prompt}

## Dimension-Based Scoring Framework

You must analyze this video segment using the following specific dimensions:

{dimension_instructions}

## Analysis Requirements

1. For each potential highlight found:
   - Score EVERY dimension listed above (provide 0.0 if not applicable)
   - Ensure scores are between 0.0 and 1.0
   - Consider the specific scoring guidance for each dimension
   - Use the examples provided as reference points

2. Calculate overall ranking score:
   - Use the dimension weights provided
   - Weighted score = sum(dimension_score * dimension_weight)
   - Ensure final score is normalized to 0.0-1.0

3. Segment Information:
   - Start time in stream: {segment_info.get("start_time", 0)} seconds
   - Duration: {segment_info.get("duration", 0)} seconds
   - Context: {segment_info.get("context", "N/A")}

4. Output Format:
   - Follow the exact schema structure
   - Include all required fields
   - Provide dimension_scores as a dictionary with ALL dimension IDs

Remember: Focus on moments that score highly across multiple dimensions,
especially those with higher weights. Quality over quantity - only identify
truly noteworthy moments.
"""

    def _build_dimension_instructions(self, dimension_set: DimensionSet) -> str:
        """Build detailed instructions for each dimension."""
        instructions = []

        # Get sorted dimensions by weight
        sorted_dims = dimension_set.get_scoring_dimensions()

        for dimension, weight in sorted_dims:
            dim_instruction = f"""
### {dimension.name} (ID: {dimension.id})
- **Weight**: {weight:.2f} ({"High" if weight > 0.15 else "Medium" if weight > 0.05 else "Low"} importance)
- **Description**: {dimension.description}
- **Type**: {dimension.dimension_type.value}
- **Threshold**: {dimension.threshold} (minimum significant value)
"""

            if dimension.scoring_prompt:
                dim_instruction += (
                    f"- **Scoring Guidance**: {dimension.scoring_prompt}\n"
                )

            if dimension.examples:
                dim_instruction += "- **Examples**:\n"
                for example in dimension.examples[:3]:
                    dim_instruction += (
                        f"  - {example['description']}: {example['value']}\n"
                    )

            instructions.append(dim_instruction)

        return "\n".join(instructions)

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60,
    )
    async def _generate_with_structured_output(
        self,
        file_uri: str,
        prompt: str,
        response_schema: type[BaseModel],
    ) -> GeminiVideoAnalysis:
        """Generate response with structured output and retry logic."""
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(file_data=types.FileData(file_uri=file_uri)),
                            types.Part(text=prompt),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,  # Lower temperature for consistency
                    top_p=0.8,
                    top_k=40,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )

            # Parse structured response
            return response_schema.model_validate_json(response.text)

        except Exception as e:
            self.logger.error(f"Structured generation failed: {str(e)}", exc_info=True)
            raise

    def _validate_dimension_scores(
        self,
        analysis: GeminiVideoAnalysis,
        dimension_set: DimensionSet,
    ) -> GeminiVideoAnalysis:
        """Ensure all highlights have scores for all dimensions."""
        dimension_ids = set(dimension_set.dimensions.keys())

        for highlight in analysis.highlights:
            # Ensure all dimensions are present
            for dim_id in dimension_ids:
                if dim_id not in highlight.dimension_scores:
                    # Add missing dimension with 0 score
                    highlight.dimension_scores[dim_id] = 0.0

            # Remove any extra dimensions not in our set
            extra_dims = set(highlight.dimension_scores.keys()) - dimension_ids
            for extra_dim in extra_dims:
                del highlight.dimension_scores[extra_dim]

            # Recalculate ranking score using dimension set
            highlight.ranking_score = dimension_set.calculate_score(
                highlight.dimension_scores
            )

        return analysis

    # Removed complex refinement pipeline for streamlined processing

    def convert_to_highlight_candidates(
        self,
        analysis: GeminiVideoAnalysis,
        dimension_set: DimensionSet,
        segment_info: Dict[str, Any],
        min_confidence: float = 0.7,
    ) -> List[HighlightCandidate]:
        """Convert Gemini analysis to highlight candidates using dimension framework."""
        candidates = []

        for highlight in analysis.highlights:
            # Check confidence threshold
            if highlight.confidence < min_confidence:
                continue

            # Check if meets dimension thresholds
            meets_thresholds = sum(
                1
                for dim_id, score in highlight.dimension_scores.items()
                if dim_id in dimension_set.dimensions
                and dimension_set.dimensions[dim_id].meets_threshold(score)
            )

            if meets_thresholds < dimension_set.minimum_dimensions_required:
                continue

            # Convert timestamps
            start_seconds = self._timestamp_to_seconds(highlight.start_time)
            end_seconds = self._timestamp_to_seconds(highlight.end_time)

            # Adjust to stream time
            start_time = segment_info["start_time"] + start_seconds
            end_time = segment_info["start_time"] + end_seconds

            # Create scoring dimensions object
            dimensions = ScoringDimensions()
            for dim_id, score in highlight.dimension_scores.items():
                setattr(dimensions, dim_id, score)

            candidate = HighlightCandidate(
                id=str(uuid.uuid4()),
                start_time=start_time,
                end_time=end_time,
                peak_time=(start_time + end_time) / 2,
                description=highlight.description,
                confidence=highlight.confidence,
                dimensions=dimensions,
                final_score=highlight.ranking_score,
                detected_keywords=self._extract_keywords(highlight, analysis),
                context_type=highlight.type.value,
                metadata={
                    "gemini_analysis": True,
                    "dimension_set": dimension_set.name,
                    "segment_id": segment_info.get("id"),
                    "dimension_scores": highlight.dimension_scores,
                    "meets_thresholds": meets_thresholds,
                    "transcript_excerpt": highlight.transcript_excerpt,
                    "key_moments": highlight.key_moments,
                    "viewer_impact": highlight.viewer_impact,
                },
            )

            candidates.append(candidate)

        # Sort by final score
        candidates.sort(key=lambda c: c.final_score, reverse=True)

        return candidates

    # Utility methods

    # Removed complex caching system for streamlined processing

    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert MM:SS or seconds timestamp to float seconds."""
        try:
            if ":" in timestamp:
                parts = timestamp.split(":")
                if len(parts) == 2:
                    minutes, seconds = map(float, parts)
                    return minutes * 60 + seconds
            return float(timestamp)
        except (ValueError, TypeError):
            return 0.0

    def _extract_keywords(
        self,
        highlight: GeminiHighlight,
        analysis: GeminiVideoAnalysis,
    ) -> List[str]:
        """Extract relevant keywords from highlight and transcript."""
        keywords = []

        # Extract from description
        if highlight.description:
            # Simple keyword extraction (could be enhanced with NLP)
            words = highlight.description.lower().split()
            keywords.extend([w for w in words if len(w) > 4])

        # Extract from transcript
        if highlight.transcript_excerpt:
            words = highlight.transcript_excerpt.lower().split()
            keywords.extend([w for w in words if len(w) > 4])

        # Deduplicate and limit
        return list(set(keywords))[:10]

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
    )
    async def _upload_video_with_retry(self, video_path: str, segment_id: str) -> Any:
        """Upload video with retry logic."""
        return await asyncio.to_thread(
            self.client.files.upload,
            file=video_path,
            config={"display_name": f"segment_{segment_id}"},
        )

    async def _cleanup_file(self, file: Any) -> None:
        """Clean up uploaded file."""
        try:
            await asyncio.to_thread(self.client.files.delete, name=file.name)
            self.uploaded_files.pop(file.name, None)
            self.logger.info(f"Cleaned up Gemini file: {file.name}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup file: {str(e)}")

    async def cleanup_all_files(self) -> None:
        """Clean up all tracked files."""
        for file_name, file in list(self.uploaded_files.items()):
            await self._cleanup_file(file)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup all files."""
        await self.cleanup_all_files()
