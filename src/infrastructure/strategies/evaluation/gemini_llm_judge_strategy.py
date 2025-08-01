"""Gemini-based implementation of LLM-as-Judge evaluation strategy.

This module provides a concrete implementation of the LLMAsJudgeStrategy
using Google's Gemini API for dimension evaluation with structured outputs.
"""

import asyncio
from typing import Dict, Optional
import backoff

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import logfire

from src.domain.services.dimension_evaluation_strategy import (
    LLMAsJudgeStrategy,
    DimensionEvaluationResult,
    EvaluationConfidence,
    EvaluationContext,
)
from src.domain.value_objects.dimension_definition import DimensionDefinition
from src.infrastructure.observability import traced_service_method


class GeminiDimensionEvaluation(BaseModel):
    """Structured output schema for Gemini dimension evaluation."""

    score: float = Field(ge=0.0, le=1.0, description="Dimension score from 0.0 to 1.0")
    confidence: str = Field(
        description="Confidence level: high, medium, low, or uncertain",
        pattern="^(high|medium|low|uncertain)$",
    )
    reasoning: str = Field(description="Explanation of the scoring decision")
    evidence: list[str] = Field(
        default_factory=list, description="Specific evidence supporting the score"
    )


class GeminiLLMJudgeStrategy(LLMAsJudgeStrategy):
    """Gemini-specific implementation of LLM-as-Judge strategy.

    Uses Gemini's multimodal capabilities and structured output
    to evaluate dimensions with chain-of-thought reasoning.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.3,
        use_few_shot: bool = True,
        max_retries: int = 2,
        use_video_context: bool = True,
    ):
        # Initialize base class without llm_client
        super().__init__(
            llm_client=None,  # We'll manage the client here
            temperature=temperature,
            use_few_shot=use_few_shot,
            max_retries=max_retries,
        )

        # Create Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.use_video_context = use_video_context
        self.logger = logfire.get_logger(__name__)

    @traced_service_method(name="gemini_evaluate_dimension")
    async def evaluate_dimension(
        self, dimension: DimensionDefinition, context: EvaluationContext
    ) -> DimensionEvaluationResult:
        """Evaluate dimension using Gemini with structured output."""
        with logfire.span("gemini_dimension_evaluation") as span:
            span.set_attribute("dimension.id", dimension.id)
            span.set_attribute("dimension.name", dimension.name)
            span.set_attribute("dimension.type", dimension.dimension_type.value)

            # Build evaluation prompt
            prompt = self._build_enhanced_prompt(dimension, context)

            # Prepare content parts
            content_parts = []

            # Add video if available and enabled
            if self.use_video_context and context.modalities.get("video_uri"):
                content_parts.append(
                    types.Part(
                        file_data=types.FileData(
                            file_uri=context.modalities["video_uri"]
                        )
                    )
                )
                span.set_attribute("includes_video", True)

            # Add prompt text
            content_parts.append(types.Part(text=prompt))

            # Try evaluation with retries
            for attempt in range(self.max_retries):
                try:
                    response = await self._call_gemini_structured(
                        content_parts, response_schema=GeminiDimensionEvaluation
                    )

                    # Convert to evaluation result
                    result = DimensionEvaluationResult(
                        dimension_id=dimension.id,
                        score=response.score,
                        confidence=EvaluationConfidence(response.confidence),
                        reasoning=response.reasoning,
                        evidence=response.evidence,
                    )

                    # Validate result
                    if self.validate_evaluation_result(result, dimension):
                        span.set_attribute("evaluation.score", result.score)
                        span.set_attribute(
                            "evaluation.confidence", result.confidence.value
                        )
                        return result

                except Exception as e:
                    self.logger.warning(
                        f"Gemini evaluation attempt {attempt + 1} failed",
                        error=str(e),
                        dimension_id=dimension.id,
                        attempt=attempt + 1,
                    )
                    span.set_attribute(f"attempt_{attempt + 1}_error", str(e))

            # Fallback result
            span.set_attribute("evaluation.fallback", True)
            return DimensionEvaluationResult(
                dimension_id=dimension.id,
                score=dimension.threshold,
                confidence=EvaluationConfidence.UNCERTAIN,
                reasoning="Failed to get reliable Gemini evaluation after retries",
            )

    def _build_enhanced_prompt(
        self, dimension: DimensionDefinition, context: EvaluationContext
    ) -> str:
        """Build enhanced prompt with multimodal context."""
        # Start with base prompt
        prompt = super()._build_evaluation_prompt(dimension, context)

        # Add multimodal context instructions
        if self.use_video_context and context.modalities.get("video_uri"):
            prompt += """
## Video Analysis Instructions
You have access to the video segment. Please analyze:
1. Visual elements relevant to this dimension
2. Audio cues and speech content
3. Overall context and atmosphere
4. Specific moments that support your evaluation

"""

        # Add modality-specific guidance
        if dimension.applicable_modalities:
            prompt += "\n## Focus Modalities\n"
            prompt += f"This dimension primarily considers: {', '.join(dimension.applicable_modalities)}\n"

            if "video" in dimension.applicable_modalities:
                prompt += "- Pay special attention to visual elements and actions\n"
            if "audio" in dimension.applicable_modalities:
                prompt += "- Focus on audio cues, speech, and sound effects\n"
            if "text" in dimension.applicable_modalities:
                prompt += "- Analyze any text, captions, or chat messages\n"

        # Add aggregation method guidance
        if dimension.aggregation_method:
            prompt += "\n## Aggregation Guidance\n"
            prompt += f"When multiple signals are present, use '{dimension.aggregation_method}' method:\n"

            if dimension.aggregation_method == "max":
                prompt += "- Take the highest signal across all observations\n"
            elif dimension.aggregation_method == "average":
                prompt += "- Average all relevant signals\n"
            elif dimension.aggregation_method == "consensus":
                prompt += "- Require agreement between multiple signals\n"

        return prompt

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=60)
    async def _call_gemini_structured(
        self, content_parts: list, response_schema: type[BaseModel]
    ) -> GeminiDimensionEvaluation:
        """Call Gemini with structured output and retry logic."""
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[types.Content(parts=content_parts)],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    top_p=0.8,
                    top_k=40,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )

            # Parse structured response
            return response_schema.model_validate_json(response.text)

        except Exception as e:
            self.logger.error(
                f"Gemini structured generation failed: {str(e)}", exc_info=True
            )
            raise

    async def evaluate_all_dimensions_with_context(
        self, context: EvaluationContext, video_file_uri: Optional[str] = None
    ) -> Dict[str, DimensionEvaluationResult]:
        """Evaluate all dimensions with shared video context.

        This method is more efficient when evaluating multiple dimensions
        for the same video segment, as it can reuse the video analysis.

        Args:
            context: Evaluation context
            video_file_uri: Optional Gemini file URI for video

        Returns:
            Dictionary mapping dimension IDs to evaluation results
        """
        # Add video URI to context if provided
        if video_file_uri:
            context.modalities["video_uri"] = video_file_uri

        results = {}

        # Group dimensions by evaluation approach
        subjective_dims = []
        objective_dims = []

        for dim_id, dimension in context.dimension_set.dimensions.items():
            if dimension.dimension_type in [
                "semantic",
                "numeric",
            ] and dimension.category in ["emotional", "contextual", "entertainment"]:
                subjective_dims.append((dim_id, dimension))
            else:
                objective_dims.append((dim_id, dimension))

        # Evaluate subjective dimensions with full context
        for dim_id, dimension in subjective_dims:
            try:
                result = await self.evaluate_dimension(dimension, context)
                results[dim_id] = result
            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate dimension {dim_id}",
                    error=str(e),
                    dimension_id=dim_id,
                )
                # Use fallback
                results[dim_id] = DimensionEvaluationResult(
                    dimension_id=dim_id,
                    score=dimension.threshold,
                    confidence=EvaluationConfidence.UNCERTAIN,
                    reasoning=f"Evaluation failed: {str(e)}",
                )

        # For objective dimensions, use AI evaluation
        for dim_id, dimension in objective_dims:
            try:
                result = await self.evaluate_dimension(dimension, context)
                results[dim_id] = result
            except Exception:
                results[dim_id] = DimensionEvaluationResult(
                    dimension_id=dim_id,
                    score=dimension.threshold,
                    confidence=EvaluationConfidence.LOW,
                    reasoning="Using default threshold due to evaluation error",
                )

        return results

    async def batch_evaluate_dimensions(
        self,
        dimensions: list[DimensionDefinition],
        context: EvaluationContext,
        batch_size: int = 5,
    ) -> Dict[str, DimensionEvaluationResult]:
        """Evaluate multiple dimensions in batches for efficiency.

        Args:
            dimensions: List of dimensions to evaluate
            context: Shared evaluation context
            batch_size: Number of dimensions to evaluate per batch

        Returns:
            Dictionary mapping dimension IDs to results
        """
        results = {}

        # Process in batches
        for i in range(0, len(dimensions), batch_size):
            batch = dimensions[i : i + batch_size]

            # Create batch evaluation prompt
            batch_prompt = self._build_batch_prompt(batch, context)

            try:
                # Call Gemini with batch prompt
                batch_results = await self._evaluate_batch(batch_prompt, batch, context)
                results.update(batch_results)

            except Exception as e:
                self.logger.error(
                    f"Batch evaluation failed for dimensions {[d.id for d in batch]}",
                    error=str(e),
                )

                # Fall back to individual evaluation
                for dimension in batch:
                    try:
                        result = await self.evaluate_dimension(dimension, context)
                        results[dimension.id] = result
                    except Exception:
                        results[dimension.id] = DimensionEvaluationResult(
                            dimension_id=dimension.id,
                            score=dimension.threshold,
                            confidence=EvaluationConfidence.UNCERTAIN,
                            reasoning="Batch and individual evaluation failed",
                        )

        return results

    def _build_batch_prompt(
        self, dimensions: list[DimensionDefinition], context: EvaluationContext
    ) -> str:
        """Build prompt for evaluating multiple dimensions at once."""
        prompt = """You are evaluating content across multiple dimensions simultaneously.

## Dimensions to Evaluate
"""

        for i, dimension in enumerate(dimensions, 1):
            prompt += f"\n### Dimension {i}: {dimension.name} (ID: {dimension.id})\n"
            prompt += f"- Description: {dimension.description}\n"
            prompt += f"- Type: {dimension.dimension_type}\n"
            prompt += f"- Threshold: {dimension.threshold}\n"
            if dimension.scoring_prompt:
                prompt += f"- Instructions: {dimension.scoring_prompt}\n"

        prompt += """
## Evaluation Instructions
1. Analyze the content once, considering all dimensions
2. For each dimension, provide a score and reasoning
3. Share evidence across dimensions where relevant
4. Maintain consistency in your analysis

## Output Format
Provide evaluations in the following JSON format:
{
    "evaluations": [
        {
            "dimension_id": "<dimension_id>",
            "score": <0.0 to 1.0>,
            "confidence": "<high|medium|low|uncertain>",
            "reasoning": "<explanation>",
            "evidence": ["<evidence1>", "<evidence2>"]
        },
        ...
    ]
}
"""

        return prompt

    async def _evaluate_batch(
        self,
        batch_prompt: str,
        dimensions: list[DimensionDefinition],
        context: EvaluationContext,
    ) -> Dict[str, DimensionEvaluationResult]:
        """Evaluate a batch of dimensions with single LLM call."""
        # Implementation would parse batch response
        # This is a placeholder for the actual batch implementation
        raise NotImplementedError("Batch evaluation requires custom response parsing")
