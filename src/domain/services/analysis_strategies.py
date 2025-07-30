"""Analysis strategy interfaces and implementations for flexible highlight detection.

This module provides different strategies for analyzing content and detecting
highlights, supporting AI-based, rule-based, and hybrid approaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..entities.dimension_set import DimensionSet
from ..entities.highlight_type_registry import HighlightTypeRegistry
from ..value_objects.processing_options import ProcessingOptions
from ..exceptions import ProcessingError


@dataclass
class AnalysisSegment:
    """A segment of content to be analyzed."""

    id: str
    start_time: float
    end_time: float
    modality_data: Dict[str, Any]  # video, audio, text, metadata
    context: Dict[str, Any]  # Additional context


@dataclass
class AnalysisResult:
    """Result from analyzing a content segment."""

    segment_id: str
    candidates: List["HighlightCandidate"]
    metadata: Dict[str, Any]
    processing_time: float
    strategy_used: str


@dataclass
class HighlightCandidate:
    """A potential highlight identified by analysis."""

    id: str
    start_time: float
    end_time: float
    score: float
    dimension_scores: Dict[str, float]
    highlight_types: List[str]
    confidence: float
    description: str
    metadata: Dict[str, Any]


class AnalysisStrategy(ABC):
    """Abstract base class for highlight analysis strategies."""

    def __init__(
        self,
        dimension_set: Optional[DimensionSet] = None,
        type_registry: Optional[HighlightTypeRegistry] = None,
        options: Optional[ProcessingOptions] = None,
    ):
        """Initialize the analysis strategy.

        Args:
            dimension_set: Set of dimensions to score against
            type_registry: Registry of highlight types
            options: Processing options
        """
        self.dimension_set = dimension_set
        self.type_registry = type_registry
        self.options = options or ProcessingOptions()

    @abstractmethod
    async def analyze(self, segment: AnalysisSegment) -> AnalysisResult:
        """Analyze a content segment for highlights.

        Args:
            segment: The segment to analyze

        Returns:
            Analysis result with highlight candidates
        """
        pass

    @abstractmethod
    def supports_modality(self, modality: str) -> bool:
        """Check if this strategy supports a given modality.

        Args:
            modality: The modality to check (video, audio, text, etc.)

        Returns:
            True if the modality is supported
        """
        pass

    def validate_configuration(self) -> List[str]:
        """Validate the strategy configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.options.requires_dimension_set and not self.dimension_set:
            errors.append("This strategy requires a dimension set")

        if not self.type_registry:
            errors.append("Type registry is required")

        return errors


class AIAnalysisStrategy(AnalysisStrategy):
    """AI-based analysis using configurable dimensions and prompts."""

    def __init__(
        self,
        ai_client: Any,  # AI client interface
        prompt_template: Any,  # Prompt template
        dimension_set: Optional[DimensionSet] = None,
        type_registry: Optional[HighlightTypeRegistry] = None,
        options: Optional[ProcessingOptions] = None,
    ):
        """Initialize AI analysis strategy.

        Args:
            ai_client: Client for AI model interaction
            prompt_template: Template for generating AI prompts
            dimension_set: Set of dimensions to score
            type_registry: Registry of highlight types
            options: Processing options
        """
        super().__init__(dimension_set, type_registry, options)
        self.ai_client = ai_client
        self.prompt_template = prompt_template

    async def analyze(self, segment: AnalysisSegment) -> AnalysisResult:
        """Analyze segment using AI model.

        Args:
            segment: The segment to analyze

        Returns:
            Analysis result with AI-detected highlights
        """
        start_time = datetime.utcnow()

        try:
            # Build prompt with dimension instructions
            prompt = self._build_analysis_prompt(segment)

            # Call AI model
            ai_response = await self.ai_client.analyze(
                prompt=prompt,
                modality_data=segment.modality_data,
                options={"temperature": 0.7, "max_tokens": 2000},
            )

            # Parse AI response into candidates
            candidates = self._parse_ai_response(ai_response, segment)

            # Filter by confidence and assign types
            filtered_candidates = []
            for candidate in candidates:
                if candidate.confidence >= self.options.min_confidence_threshold:
                    # Determine highlight types
                    candidate.highlight_types = self.type_registry.determine_types(
                        candidate.score, candidate.dimension_scores
                    )
                    filtered_candidates.append(candidate)

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            return AnalysisResult(
                segment_id=segment.id,
                candidates=filtered_candidates,
                metadata={
                    "ai_model": self.ai_client.model_name,
                    "prompt_length": len(prompt),
                    "dimensions_evaluated": len(self.dimension_set.dimensions),
                },
                processing_time=processing_time,
                strategy_used="ai_analysis",
            )

        except Exception as e:
            raise ProcessingError(f"AI analysis failed: {str(e)}")

    def supports_modality(self, modality: str) -> bool:
        """AI strategy supports all modalities."""
        return True

    def _build_analysis_prompt(self, segment: AnalysisSegment) -> str:
        """Build AI prompt with dimension instructions."""
        # Get dimension instructions
        dimension_instructions = []
        for dim_id, dimension in self.dimension_set.dimensions.items():
            instruction = dimension.generate_ai_instruction(segment.context)
            dimension_instructions.append(instruction)

        # Build context
        context = {
            "segment_duration": segment.end_time - segment.start_time,
            "dimensions": "\n".join(dimension_instructions),
            "modalities": list(segment.modality_data.keys()),
            **segment.context,
        }

        # Render prompt template
        return self.prompt_template.render(context)

    def _parse_ai_response(
        self, ai_response: Dict[str, Any], segment: AnalysisSegment
    ) -> List[HighlightCandidate]:
        """Parse AI response into highlight candidates."""
        candidates = []

        for idx, detection in enumerate(ai_response.get("detections", [])):
            # Extract dimension scores
            dimension_scores = {}
            for dim_id in self.dimension_set.dimensions:
                score = detection.get("dimensions", {}).get(dim_id, 0.0)
                dimension = self.dimension_set.dimensions[dim_id]
                dimension_scores[dim_id] = dimension.normalize_value(score)

            # Calculate overall score
            overall_score = self.dimension_set.calculate_score(dimension_scores)

            candidate = HighlightCandidate(
                id=f"{segment.id}_ai_{idx}",
                start_time=detection.get("start_time", segment.start_time),
                end_time=detection.get("end_time", segment.end_time),
                score=overall_score,
                dimension_scores=dimension_scores,
                highlight_types=[],  # Will be assigned later
                confidence=detection.get("confidence", overall_score),
                description=detection.get("description", "AI-detected highlight"),
                metadata={
                    "ai_reasoning": detection.get("reasoning", ""),
                    "detected_features": detection.get("features", []),
                },
            )

            candidates.append(candidate)

        return candidates


class RuleBasedAnalysisStrategy(AnalysisStrategy):
    """Rule-based analysis using predefined detection rules."""

    def __init__(
        self,
        rules: List[Dict[str, Any]],
        type_registry: Optional[HighlightTypeRegistry] = None,
        options: Optional[ProcessingOptions] = None,
    ):
        """Initialize rule-based strategy.

        Args:
            rules: List of detection rules
            type_registry: Registry of highlight types
            options: Processing options
        """
        super().__init__(None, type_registry, options)
        self.rules = rules
        self._compiled_rules = self._compile_rules(rules)

    async def analyze(self, segment: AnalysisSegment) -> AnalysisResult:
        """Analyze segment using predefined rules.

        Args:
            segment: The segment to analyze

        Returns:
            Analysis result with rule-detected highlights
        """
        start_time = datetime.utcnow()
        candidates = []

        # Apply each rule
        for rule_idx, rule in enumerate(self._compiled_rules):
            if self._evaluate_rule(rule, segment):
                candidate = self._create_candidate_from_rule(rule, segment, rule_idx)
                candidates.append(candidate)

        # Merge overlapping candidates
        merged_candidates = self._merge_overlapping_candidates(candidates)

        # Assign highlight types
        for candidate in merged_candidates:
            candidate.highlight_types = self.type_registry.determine_types(
                candidate.score,
                candidate.dimension_scores,
                force_type=candidate.metadata.get("force_type"),
            )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return AnalysisResult(
            segment_id=segment.id,
            candidates=merged_candidates,
            metadata={
                "rules_evaluated": len(self.rules),
                "rules_matched": len(candidates),
                "candidates_merged": len(candidates) - len(merged_candidates),
            },
            processing_time=processing_time,
            strategy_used="rule_based",
        )

    def supports_modality(self, modality: str) -> bool:
        """Check if any rule supports the modality."""
        for rule in self.rules:
            if modality in rule.get("required_modalities", []):
                return True
        return False

    def _compile_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compile rules for efficient evaluation."""
        compiled = []

        for rule in rules:
            compiled_rule = {
                "id": rule.get("id", f"rule_{len(compiled)}"),
                "name": rule.get("name", "Unnamed rule"),
                "conditions": rule.get("conditions", []),
                "actions": rule.get("actions", {}),
                "priority": rule.get("priority", 0),
                "required_modalities": rule.get("required_modalities", []),
            }
            compiled.append(compiled_rule)

        # Sort by priority
        return sorted(compiled, key=lambda r: r["priority"], reverse=True)

    def _evaluate_rule(self, rule: Dict[str, Any], segment: AnalysisSegment) -> bool:
        """Evaluate if a rule matches the segment."""
        # Check required modalities
        for modality in rule["required_modalities"]:
            if modality not in segment.modality_data:
                return False

        # Evaluate all conditions
        for condition in rule["conditions"]:
            if not self._evaluate_condition(condition, segment):
                return False

        return True

    def _evaluate_condition(
        self, condition: Dict[str, Any], segment: AnalysisSegment
    ) -> bool:
        """Evaluate a single rule condition."""
        cond_type = condition.get("type")

        if cond_type == "metadata":
            # Check metadata value
            key = condition["key"]
            expected = condition["value"]
            actual = segment.modality_data.get("metadata", {}).get(key)
            return self._compare_values(
                actual, expected, condition.get("operator", "==")
            )

        elif cond_type == "duration":
            # Check segment duration
            duration = segment.end_time - segment.start_time
            threshold = condition["threshold"]
            return self._compare_values(
                duration, threshold, condition.get("operator", ">=")
            )

        elif cond_type == "time_range":
            # Check if segment is within time range
            start = condition.get("start", 0)
            end = condition.get("end", float("inf"))
            return start <= segment.start_time <= end

        elif cond_type == "custom":
            # Custom condition function
            func = condition.get("function")
            if callable(func):
                return func(segment)

        return False

    def _compare_values(self, actual: Any, expected: Any, operator: str) -> bool:
        """Compare values using the specified operator."""
        if operator == "==":
            return actual == expected
        elif operator == "!=":
            return actual != expected
        elif operator == ">":
            return actual > expected
        elif operator == ">=":
            return actual >= expected
        elif operator == "<":
            return actual < expected
        elif operator == "<=":
            return actual <= expected
        elif operator == "in":
            return actual in expected
        elif operator == "contains":
            return expected in actual
        else:
            return False

    def _create_candidate_from_rule(
        self, rule: Dict[str, Any], segment: AnalysisSegment, rule_idx: int
    ) -> HighlightCandidate:
        """Create a highlight candidate from a matched rule."""
        actions = rule["actions"]

        return HighlightCandidate(
            id=f"{segment.id}_rule_{rule_idx}",
            start_time=actions.get("start_time", segment.start_time),
            end_time=actions.get("end_time", segment.end_time),
            score=actions.get("score", 0.7),
            dimension_scores=actions.get("dimension_scores", {}),
            highlight_types=[],  # Will be assigned later
            confidence=actions.get("confidence", 1.0),
            description=actions.get("description", f"Matched rule: {rule['name']}"),
            metadata={
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "force_type": actions.get("force_type"),
            },
        )

    def _merge_overlapping_candidates(
        self, candidates: List[HighlightCandidate]
    ) -> List[HighlightCandidate]:
        """Merge overlapping highlight candidates."""
        if not candidates:
            return []

        # Sort by start time
        sorted_candidates = sorted(candidates, key=lambda c: c.start_time)
        merged = [sorted_candidates[0]]

        for candidate in sorted_candidates[1:]:
            last_merged = merged[-1]

            # Check for overlap
            if candidate.start_time <= last_merged.end_time:
                # Merge candidates
                last_merged.end_time = max(last_merged.end_time, candidate.end_time)
                last_merged.score = max(last_merged.score, candidate.score)

                # Merge metadata
                if "merged_rules" not in last_merged.metadata:
                    last_merged.metadata["merged_rules"] = [
                        last_merged.metadata.get("rule_id")
                    ]
                last_merged.metadata["merged_rules"].append(
                    candidate.metadata.get("rule_id")
                )
            else:
                merged.append(candidate)

        return merged


class HybridAnalysisStrategy(AnalysisStrategy):
    """Hybrid strategy combining AI and rule-based analysis."""

    def __init__(
        self,
        ai_strategy: AIAnalysisStrategy,
        rule_strategy: RuleBasedAnalysisStrategy,
        combination_mode: str = "union",  # union, intersection, ai_validated
        options: Optional[ProcessingOptions] = None,
    ):
        """Initialize hybrid strategy.

        Args:
            ai_strategy: AI analysis strategy
            rule_strategy: Rule-based strategy
            combination_mode: How to combine results
            options: Processing options
        """
        super().__init__(ai_strategy.dimension_set, ai_strategy.type_registry, options)
        self.ai_strategy = ai_strategy
        self.rule_strategy = rule_strategy
        self.combination_mode = combination_mode

    async def analyze(self, segment: AnalysisSegment) -> AnalysisResult:
        """Analyze using both AI and rules.

        Args:
            segment: The segment to analyze

        Returns:
            Combined analysis result
        """
        start_time = datetime.utcnow()

        # Run both strategies in parallel
        ai_result = await self.ai_strategy.analyze(segment)
        rule_result = await self.rule_strategy.analyze(segment)

        # Combine results based on mode
        if self.combination_mode == "union":
            combined_candidates = self._union_candidates(
                ai_result.candidates, rule_result.candidates
            )
        elif self.combination_mode == "intersection":
            combined_candidates = self._intersect_candidates(
                ai_result.candidates, rule_result.candidates
            )
        elif self.combination_mode == "ai_validated":
            combined_candidates = self._ai_validate_rules(
                ai_result.candidates, rule_result.candidates
            )
        else:
            combined_candidates = ai_result.candidates + rule_result.candidates

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return AnalysisResult(
            segment_id=segment.id,
            candidates=combined_candidates,
            metadata={
                "combination_mode": self.combination_mode,
                "ai_candidates": len(ai_result.candidates),
                "rule_candidates": len(rule_result.candidates),
                "combined_candidates": len(combined_candidates),
                "ai_metadata": ai_result.metadata,
                "rule_metadata": rule_result.metadata,
            },
            processing_time=processing_time,
            strategy_used="hybrid",
        )

    def supports_modality(self, modality: str) -> bool:
        """Supports modalities supported by either strategy."""
        return self.ai_strategy.supports_modality(
            modality
        ) or self.rule_strategy.supports_modality(modality)

    def _union_candidates(
        self,
        ai_candidates: List[HighlightCandidate],
        rule_candidates: List[HighlightCandidate],
    ) -> List[HighlightCandidate]:
        """Union of AI and rule candidates."""
        # Combine all candidates
        all_candidates = ai_candidates + rule_candidates

        # Remove duplicates based on time overlap
        unique_candidates = []
        for candidate in all_candidates:
            is_duplicate = False
            for existing in unique_candidates:
                overlap = self._calculate_overlap(candidate, existing)
                if overlap > 0.8:  # 80% overlap threshold
                    # Keep the one with higher score
                    if candidate.score > existing.score:
                        unique_candidates.remove(existing)
                        unique_candidates.append(candidate)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_candidates.append(candidate)

        return unique_candidates

    def _intersect_candidates(
        self,
        ai_candidates: List[HighlightCandidate],
        rule_candidates: List[HighlightCandidate],
    ) -> List[HighlightCandidate]:
        """Intersection of AI and rule candidates."""
        intersected = []

        for ai_candidate in ai_candidates:
            for rule_candidate in rule_candidates:
                overlap = self._calculate_overlap(ai_candidate, rule_candidate)
                if overlap > 0.5:  # 50% overlap required
                    # Create merged candidate
                    merged = HighlightCandidate(
                        id=f"{ai_candidate.id}_intersect",
                        start_time=max(
                            ai_candidate.start_time, rule_candidate.start_time
                        ),
                        end_time=min(ai_candidate.end_time, rule_candidate.end_time),
                        score=(ai_candidate.score + rule_candidate.score) / 2,
                        dimension_scores=ai_candidate.dimension_scores,
                        highlight_types=ai_candidate.highlight_types,
                        confidence=min(
                            ai_candidate.confidence, rule_candidate.confidence
                        ),
                        description=ai_candidate.description,
                        metadata={
                            **ai_candidate.metadata,
                            "validated_by_rule": rule_candidate.metadata.get("rule_id"),
                        },
                    )
                    intersected.append(merged)

        return intersected

    def _ai_validate_rules(
        self,
        ai_candidates: List[HighlightCandidate],
        rule_candidates: List[HighlightCandidate],
    ) -> List[HighlightCandidate]:
        """Use AI to validate rule-based candidates."""
        validated = []

        # Include all AI candidates
        validated.extend(ai_candidates)

        # Include rule candidates that overlap with AI detections
        for rule_candidate in rule_candidates:
            for ai_candidate in ai_candidates:
                if self._calculate_overlap(rule_candidate, ai_candidate) > 0.3:
                    rule_candidate.metadata["ai_validated"] = True
                    rule_candidate.confidence *= 1.2  # Boost confidence
                    validated.append(rule_candidate)
                    break

        return validated

    def _calculate_overlap(
        self, candidate1: HighlightCandidate, candidate2: HighlightCandidate
    ) -> float:
        """Calculate temporal overlap between two candidates."""
        overlap_start = max(candidate1.start_time, candidate2.start_time)
        overlap_end = min(candidate1.end_time, candidate2.end_time)

        if overlap_start >= overlap_end:
            return 0.0

        overlap_duration = overlap_end - overlap_start
        total_duration = max(
            candidate1.end_time - candidate1.start_time,
            candidate2.end_time - candidate2.start_time,
        )

        return overlap_duration / total_duration if total_duration > 0 else 0.0
