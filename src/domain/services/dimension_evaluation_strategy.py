"""Dimension evaluation strategies for flexible highlight detection.

This module implements the Strategy pattern for evaluating content against
dimensions, supporting multiple evaluation approaches including LLM-as-Judge,
rule-based, and hybrid strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Protocol
from enum import Enum
import asyncio

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
from src.domain.value_objects.dimension_definition import DimensionDefinition
from src.domain.value_objects.dimension_score import DimensionScore
import logfire


class EvaluationConfidence(str, Enum):
    """Confidence levels for dimension evaluations."""

    HIGH = "high"  # Very confident in the evaluation
    MEDIUM = "medium"  # Reasonably confident
    LOW = "low"  # Low confidence, might need review
    UNCERTAIN = "uncertain"  # Very uncertain, needs human review


@dataclass
class DimensionEvaluationResult:
    """Result of evaluating a single dimension."""

    dimension_id: str
    score: float  # 0.0 to 1.0
    confidence: EvaluationConfidence
    reasoning: Optional[str] = None  # Why this score was given
    evidence: List[str] = field(default_factory=list)  # Supporting evidence
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationContext:
    """Context information for dimension evaluation."""

    segment_data: Dict[str, Any]  # Video segment information
    dimension_set: Any  # DimensionSetAggregate (using Any to avoid circular import)
    modalities: Dict[str, Any] = field(default_factory=dict)  # Available modalities
    previous_evaluations: List[DimensionEvaluationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DimensionEvaluationStrategy(Protocol):
    """Protocol for dimension evaluation strategies.

    Different strategies can be used to evaluate content against dimensions,
    from simple rule-based approaches to sophisticated LLM-based evaluation.
    """

    name: str

    async def evaluate_dimension(
        self, dimension: DimensionDefinition, context: EvaluationContext
    ) -> DimensionEvaluationResult:
        """Evaluate a single dimension for the given context.

        Args:
            dimension: The dimension to evaluate
            context: Evaluation context with segment data

        Returns:
            DimensionEvaluationResult with score and confidence
        """
        ...

    async def evaluate_all_dimensions(
        self, context: EvaluationContext
    ) -> Dict[str, DimensionScore]:
        """Evaluate all dimensions in the dimension set.

        Args:
            context: Evaluation context

        Returns:
            Dictionary of dimension IDs to DimensionScore objects
        """
        results = {}
        tasks = []

        # Create tasks for parallel evaluation
        for dim_id, dimension in context.dimension_set.dimensions.items():
            task = self.evaluate_dimension(dimension, context)
            tasks.append((dim_id, task))

        # Execute evaluations in parallel
        for dim_id, task in tasks:
            try:
                result = await task
                results[dim_id] = result.score

                # Store detailed result in context for reference
                context.previous_evaluations.append(result)

            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate dimension {dim_id}",
                    error=str(e),
                    dimension_id=dim_id,
                )
                # Use default threshold as fallback
                results[dim_id] = context.dimension_set.dimensions[dim_id].threshold

        # Convert to proper DimensionScore objects
        dimension_scores = {}
        for dim_id, score_value in results.items():
            dimension_scores[dim_id] = DimensionScore(
                dimension_id=dim_id,
                value=score_value,
                confidence="medium",  # Default confidence
                evidence="Evaluated using dimension evaluation strategy"
            )
        return dimension_scores

    def validate_evaluation_result(
        self, result: DimensionEvaluationResult, dimension: DimensionDefinition
    ) -> bool:
        """Validate an evaluation result against dimension constraints.

        Args:
            result: The evaluation result to validate
            dimension: The dimension definition

        Returns:
            True if valid, False otherwise
        """
        # Check score is in valid range
        if not 0.0 <= result.score <= 1.0:
            return False

        # Check dimension type constraints
        if dimension.dimension_type == "binary":
            if result.score not in [0.0, 1.0]:
                return False

        return True

    async def calibrate_scores(
        self,
        raw_scores: Dict[str, float],
        calibration_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Calibrate raw scores based on historical data or reference points.

        Args:
            raw_scores: Raw dimension scores
            calibration_data: Optional calibration information

        Returns:
            Calibrated scores
        """
        # Default implementation: no calibration
        return raw_scores


class LLMAsJudgeStrategy:
    """LLM-based evaluation strategy using structured prompts.

    Implements the G-Eval approach with chain-of-thought reasoning
    for dimension evaluation.
    """

    def __init__(
        self,
        llm_client: Any,  # Gemini or other LLM client
        temperature: float = 0.3,
        use_few_shot: bool = True,
        max_retries: int = 2,
    ):
        self.name = "llm_as_judge"
        self.logger = logfire.get_logger(f"{__name__}.{self.name}")
        self.llm_client = llm_client
        self.temperature = temperature
        self.use_few_shot = use_few_shot
        self.max_retries = max_retries

    async def evaluate_dimension(
        self, dimension: DimensionDefinition, context: EvaluationContext
    ) -> DimensionEvaluationResult:
        """Evaluate using LLM with structured prompt."""
        prompt = self._build_evaluation_prompt(dimension, context)

        for attempt in range(self.max_retries):
            try:
                # Call LLM with structured output
                response = await self._call_llm(prompt, dimension)

                # Parse and validate response
                result = self._parse_llm_response(response, dimension)

                if self._validate_evaluation_result(result, dimension):
                    return result

            except Exception as e:
                self.logger.warning(
                    f"LLM evaluation attempt {attempt + 1} failed",
                    error=str(e),
                    dimension_id=dimension.id,
                )

        # Fallback to uncertain evaluation
        return DimensionEvaluationResult(
            dimension_id=dimension.id,
            score=dimension.threshold,
            confidence=EvaluationConfidence.UNCERTAIN,
            reasoning="Failed to get reliable LLM evaluation",
        )

    def _build_evaluation_prompt(
        self, dimension: DimensionDefinition, context: EvaluationContext
    ) -> str:
        """Build structured evaluation prompt with chain-of-thought."""
        prompt = f"""You are evaluating content for the dimension: {dimension.name}

## Dimension Details
- Description: {dimension.description}
- Type: {dimension.dimension_type}
- Scoring Range: {dimension.min_value} to {dimension.max_value}
- Significance Threshold: {dimension.threshold}

## Evaluation Instructions
{dimension.scoring_prompt or dimension.generate_ai_instruction(context.metadata)}

## Evaluation Steps
1. First, identify relevant evidence from the content
2. Consider how strongly this evidence supports the dimension
3. Compare against the examples provided
4. Assign a score based on your analysis
5. Explain your reasoning

"""

        # Add few-shot examples if enabled
        if self.use_few_shot and dimension.examples:
            prompt += "\n## Examples\n"
            for example in dimension.examples[:3]:
                prompt += f"- Score {example['value']}: {example['description']}\n"

        # Add context information
        prompt += "\n## Content Context\n"
        prompt += (
            f"Duration: {context.segment_data.get('duration', 'unknown')} seconds\n"
        )
        prompt += f"Content Type: {context.metadata.get('content_type', 'general')}\n"

        if context.modalities.get("transcript"):
            prompt += (
                f"\nTranscript excerpt: {context.modalities['transcript'][:500]}...\n"
            )

        prompt += """
## Output Format
Provide your evaluation in the following JSON format:
{
    "score": <float between 0.0 and 1.0>,
    "confidence": <"high", "medium", "low", or "uncertain">,
    "reasoning": "<explanation of your scoring decision>",
    "evidence": ["<specific evidence 1>", "<specific evidence 2>", ...]
}
"""

        return prompt

    async def _call_llm(
        self, prompt: str, dimension: DimensionDefinition
    ) -> Dict[str, Any]:
        """Call LLM and get structured response."""
        # This is a placeholder - actual implementation depends on LLM client
        # For Gemini, it would use the structured output feature
        raise NotImplementedError("LLM client integration required")

    def _parse_llm_response(
        self, response: Dict[str, Any], dimension: DimensionDefinition
    ) -> DimensionEvaluationResult:
        """Parse LLM response into evaluation result."""
        return DimensionEvaluationResult(
            dimension_id=dimension.id,
            score=float(response.get("score", dimension.threshold)),
            confidence=EvaluationConfidence(response.get("confidence", "uncertain")),
            reasoning=response.get("reasoning"),
            evidence=response.get("evidence", []),
        )

    def _validate_evaluation_result(
        self, result: DimensionEvaluationResult, dimension: DimensionDefinition
    ) -> bool:
        """Validate an evaluation result against dimension constraints."""
        # Check score is in valid range
        if not 0.0 <= result.score <= 1.0:
            return False

        # Check dimension type constraints
        if dimension.dimension_type == "binary":
            if result.score not in [0.0, 1.0]:
                return False

        return True


class RuleBasedStrategy:
    """Rule-based evaluation strategy for objective dimensions.

    Uses deterministic rules to evaluate dimensions based on
    measurable criteria.
    """

    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        self.name = "rule_based"
        self.logger = logfire.get_logger(f"{__name__}.{self.name}")
        self.rules = rules or {}

    async def evaluate_dimension(
        self, dimension: DimensionDefinition, context: EvaluationContext
    ) -> DimensionEvaluationResult:
        """Evaluate using predefined rules."""
        # Check if we have a specific rule for this dimension
        if dimension.id in self.rules:
            rule = self.rules[dimension.id]
            score = await self._apply_rule(rule, context)
        else:
            # Try to infer from dimension metadata
            score = await self._infer_score(dimension, context)

        # Rule-based evaluations typically have high confidence
        confidence = (
            EvaluationConfidence.HIGH
            if score != dimension.threshold
            else EvaluationConfidence.MEDIUM
        )

        return DimensionEvaluationResult(
            dimension_id=dimension.id,
            score=score,
            confidence=confidence,
            reasoning=f"Rule-based evaluation for {dimension.name}",
        )

    async def _apply_rule(
        self, rule: Dict[str, Any], context: EvaluationContext
    ) -> float:
        """Apply a specific rule to get score."""
        rule_type = rule.get("type", "threshold")

        if rule_type == "threshold":
            # Simple threshold rule
            value = self._extract_value(rule["field"], context)
            threshold = rule["threshold"]
            return 1.0 if value >= threshold else 0.0

        elif rule_type == "range":
            # Range-based scoring
            value = self._extract_value(rule["field"], context)
            min_val, max_val = rule["range"]
            if value <= min_val:
                return 0.0
            elif value >= max_val:
                return 1.0
            else:
                return (value - min_val) / (max_val - min_val)

        elif rule_type == "pattern":
            # Pattern matching
            text = self._extract_value(rule["field"], context)
            pattern = rule["pattern"]
            matches = len([1 for p in pattern if p in text])
            return min(1.0, matches / len(pattern))

        else:
            return 0.5  # Default middle score

    async def _infer_score(
        self, dimension: DimensionDefinition, context: EvaluationContext
    ) -> float:
        """Infer score based on dimension properties."""
        # Simple inference based on modalities
        if "audio_level" in dimension.id.lower():
            audio_data = context.modalities.get("audio", {})
            level = audio_data.get("average_level", 0.5)
            return min(1.0, level)

        elif "motion" in dimension.id.lower():
            video_data = context.modalities.get("video", {})
            motion = video_data.get("motion_score", 0.5)
            return min(1.0, motion)

        # Default to threshold
        return dimension.threshold

    def _extract_value(self, field_path: str, context: EvaluationContext) -> Any:
        """Extract value from context using dot notation."""
        parts = field_path.split(".")
        value = context

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict):
                value = value.get(part, 0)
            else:
                return 0

        return value


class HybridStrategy:
    """Hybrid strategy combining LLM and rule-based approaches.

    Uses rules for objective dimensions and LLM for subjective ones,
    with the ability to cross-validate results.
    """

    def __init__(
        self,
        llm_strategy: LLMAsJudgeStrategy,
        rule_strategy: RuleBasedStrategy,
        subjective_dimensions: Optional[List[str]] = None,
        confidence_threshold: float = 0.8,
    ):
        self.name = "hybrid"
        self.logger = logfire.get_logger(f"{__name__}.{self.name}")
        self.llm_strategy = llm_strategy
        self.rule_strategy = rule_strategy
        self.subjective_dimensions = subjective_dimensions or []
        self.confidence_threshold = confidence_threshold

    async def evaluate_dimension(
        self, dimension: DimensionDefinition, context: EvaluationContext
    ) -> DimensionEvaluationResult:
        """Evaluate using appropriate strategy based on dimension type."""
        # Determine primary strategy
        use_llm = (
            dimension.id in self.subjective_dimensions
            or dimension.dimension_type == "semantic"
            or dimension.category in ["emotional", "contextual"]
        )

        if use_llm:
            # Use LLM as primary, optionally validate with rules
            primary_result = await self.llm_strategy.evaluate_dimension(
                dimension, context
            )

            # If low confidence, try rule-based validation
            if primary_result.confidence in [
                EvaluationConfidence.LOW,
                EvaluationConfidence.UNCERTAIN,
            ]:
                rule_result = await self.rule_strategy.evaluate_dimension(
                    dimension, context
                )
                return self._reconcile_results(primary_result, rule_result)

            return primary_result
        else:
            # Use rules as primary
            return await self.rule_strategy.evaluate_dimension(dimension, context)

    def _reconcile_results(
        self,
        llm_result: DimensionEvaluationResult,
        rule_result: DimensionEvaluationResult,
    ) -> DimensionEvaluationResult:
        """Reconcile differences between LLM and rule evaluations."""
        # If results are close, average them
        if abs(llm_result.score - rule_result.score) < 0.2:
            return DimensionEvaluationResult(
                dimension_id=llm_result.dimension_id,
                score=(llm_result.score + rule_result.score) / 2,
                confidence=EvaluationConfidence.MEDIUM,
                reasoning=f"Hybrid evaluation: LLM ({llm_result.score:.2f}) and rules ({rule_result.score:.2f}) agree",
            )

        # If very different, prefer the one with higher confidence
        if rule_result.confidence == EvaluationConfidence.HIGH:
            return rule_result
        else:
            return llm_result


class ConsensusStrategy:
    """Consensus strategy using multiple evaluation methods.

    Combines results from multiple strategies to achieve
    higher confidence through agreement.
    """

    def __init__(
        self,
        strategies: List[DimensionEvaluationStrategy],
        min_agreement: int = 2,
        agreement_threshold: float = 0.2,
    ):
        self.name = "consensus"
        self.logger = logfire.get_logger(f"{__name__}.{self.name}")
        self.strategies = strategies
        self.min_agreement = min_agreement
        self.agreement_threshold = agreement_threshold

    async def evaluate_dimension(
        self, dimension: DimensionDefinition, context: EvaluationContext
    ) -> DimensionEvaluationResult:
        """Evaluate using multiple strategies and find consensus."""
        # Run all strategies in parallel
        tasks = [
            strategy.evaluate_dimension(dimension, context)
            for strategy in self.strategies
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed evaluations
        valid_results = [
            r
            for r in results
            if isinstance(r, DimensionEvaluationResult) and not isinstance(r, Exception)
        ]

        if not valid_results:
            return DimensionEvaluationResult(
                dimension_id=dimension.id,
                score=dimension.threshold,
                confidence=EvaluationConfidence.UNCERTAIN,
                reasoning="All evaluation strategies failed",
            )

        # Calculate consensus
        scores = [r.score for r in valid_results]
        avg_score = sum(scores) / len(scores)

        # Check agreement
        agreements = sum(
            1 for score in scores if abs(score - avg_score) <= self.agreement_threshold
        )

        if agreements >= self.min_agreement:
            confidence = EvaluationConfidence.HIGH
            reasoning = f"Strong consensus among {agreements}/{len(scores)} evaluators"
        elif agreements >= 2:
            confidence = EvaluationConfidence.MEDIUM
            reasoning = "Moderate consensus among evaluators"
        else:
            confidence = EvaluationConfidence.LOW
            reasoning = f"Low consensus among evaluators (scores: {scores})"

        return DimensionEvaluationResult(
            dimension_id=dimension.id,
            score=avg_score,
            confidence=confidence,
            reasoning=reasoning,
            evidence=[
                f"{s.name}: {r.score:.2f}"
                for s, r in zip(self.strategies, valid_results)
            ],
            metadata={"individual_scores": scores},
        )
