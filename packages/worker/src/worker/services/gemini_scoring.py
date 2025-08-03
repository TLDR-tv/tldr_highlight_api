"""Gemini-based scoring strategy for highlight detection."""

import logging
from pathlib import Path
from typing import Any, Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from shared.domain.services.dimension_framework import (
    DimensionDefinition,
    ScoringRubric,
    ScoringStrategy,
)
from shared.infrastructure.config.config import Settings

logger = logging.getLogger(__name__)


class GeminiScoringStrategy(ScoringStrategy):
    """Scoring strategy using Google's Gemini AI model."""

    def __init__(self, settings: Settings):
        """Initialize Gemini scoring strategy.

        Args:
            settings: Application settings containing API key

        """
        self.api_key = settings.gemini_api_key
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 1.5 Flash for efficient video analysis
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.3,  # Lower temperature for consistent scoring
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Cache for uploaded files
        self._file_cache: dict[str, Any] = {}

    async def score(
        self,
        content: Path,
        rubric: ScoringRubric,
        context: Optional[list[Any]] = None,
    ) -> dict[str, tuple[float, float]]:
        """Score video content using Gemini.

        Args:
            content: Path to video file
            rubric: Scoring rubric with dimensions
            context: Optional context from previous segments

        Returns:
            Dictionary mapping dimension names to (score, confidence) tuples

        """
        try:
            # Upload video file to Gemini
            video_file = await self._upload_video(content)
            
            # Build scoring prompt
            prompt = self._build_prompt(rubric, context)
            
            # Generate content
            response = await self._generate_response(video_file, prompt)
            
            # Parse scores from response
            scores = self._parse_scores(response.text, rubric)
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to score video with Gemini: {e}")
            # Return zero scores on failure
            return {
                dim.name: (0.0, 0.0) 
                for dim in rubric.dimensions
            }

    async def _upload_video(self, video_path: Path) -> Any:
        """Upload video to Gemini File API.

        Args:
            video_path: Path to video file

        Returns:
            Uploaded file object

        """
        # Check cache first
        path_str = str(video_path)
        if path_str in self._file_cache:
            return self._file_cache[path_str]
        
        # Upload new file
        logger.info(f"Uploading video to Gemini: {video_path}")
        video_file = genai.upload_file(path=str(video_path))
        
        # Cache for reuse
        self._file_cache[path_str] = video_file
        
        return video_file

    def _build_prompt(
        self, rubric: ScoringRubric, context: Optional[list[Any]] = None
    ) -> str:
        """Build scoring prompt for Gemini.

        Args:
            rubric: Scoring rubric with dimensions
            context: Optional context from previous segments

        Returns:
            Formatted prompt string

        """
        prompt_parts = [
            "You are an expert video analyst. Score this video segment on the following dimensions.",
            "Provide scores as decimal numbers between 0.0 and 10.0.",
            "Also provide a confidence score (0.0-1.0) for each dimension.",
            "",
            "DIMENSIONS TO SCORE:",
        ]
        
        # Add each dimension
        for dim in rubric.dimensions:
            prompt_parts.append(f"\n{dim.name}:")
            prompt_parts.append(f"  Description: {dim.description}")
            if dim.scoring_prompt:
                prompt_parts.append(f"  Scoring guidance: {dim.scoring_prompt}")
            if dim.examples:
                prompt_parts.append("  Examples:")
                for example in dim.examples[:3]:  # Limit examples
                    prompt_parts.append(
                        f"    - {example.get('context', 'N/A')}: "
                        f"score={example.get('score', 0)}"
                    )
        
        # Add context if available
        if context:
            prompt_parts.extend([
                "",
                "CONTEXT FROM PREVIOUS SEGMENTS:",
                "Consider the following context when scoring this segment:",
            ])
            # Summarize recent context
            recent_context = context[-3:]  # Last 3 segments
            for i, ctx in enumerate(recent_context):
                if hasattr(ctx, 'dimension_scores'):
                    prompt_parts.append(f"  Segment {i+1}: {ctx.dimension_scores}")
        
        # Add output format
        prompt_parts.extend([
            "",
            "OUTPUT FORMAT:",
            "Provide your response as a JSON object with this structure:",
            "{",
            '  "dimension_name": {"score": 0.0, "confidence": 0.0},',
            '  ...',
            "}",
            "",
            "Analyze the video and provide your scores:",
        ])
        
        return "\n".join(prompt_parts)

    async def _generate_response(self, video_file: Any, prompt: str) -> Any:
        """Generate response from Gemini.

        Args:
            video_file: Uploaded video file
            prompt: Scoring prompt

        Returns:
            Gemini response object

        """
        logger.debug("Generating Gemini response for video scoring")
        
        response = self.model.generate_content(
            [video_file, prompt],
            request_options={"timeout": 30}
        )
        
        return response

    def _parse_scores(
        self, response_text: str, rubric: ScoringRubric
    ) -> dict[str, tuple[float, float]]:
        """Parse scores from Gemini response.

        Args:
            response_text: Raw response text from Gemini
            rubric: Scoring rubric for validation

        Returns:
            Dictionary mapping dimension names to (score, confidence) tuples

        """
        import json
        import re
        
        scores = {}
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # Extract scores for each dimension
                for dim in rubric.dimensions:
                    if dim.name in parsed:
                        dim_data = parsed[dim.name]
                        if isinstance(dim_data, dict):
                            score = float(dim_data.get("score", 0))
                            confidence = float(dim_data.get("confidence", 0.5))
                        else:
                            # Handle simple numeric response
                            score = float(dim_data)
                            confidence = 0.7  # Default confidence
                        
                        # Clamp values to valid ranges
                        score = max(0.0, min(10.0, score))
                        confidence = max(0.0, min(1.0, confidence))
                        
                        scores[dim.name] = (score, confidence)
                    else:
                        # Missing dimension
                        scores[dim.name] = (0.0, 0.0)
            else:
                logger.warning("No JSON found in Gemini response")
                # Return zero scores
                for dim in rubric.dimensions:
                    scores[dim.name] = (0.0, 0.0)
                    
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            # Return zero scores on parse error
            for dim in rubric.dimensions:
                scores[dim.name] = (0.0, 0.0)
        
        return scores

    def cleanup(self):
        """Clean up cached files."""
        self._file_cache.clear()