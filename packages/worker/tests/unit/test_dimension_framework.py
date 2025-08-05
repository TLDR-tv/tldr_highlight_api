"""Unit tests for dimension framework."""

import pytest
from worker.services.dimension_framework import (
    DimensionType,
    AggregationMethod,
    DimensionExample,
    DimensionDefinition,
    ScoringRubric,
)


class TestDimensionType:
    """Test DimensionType enum."""

    def test_dimension_types(self):
        """Test all dimension types are defined."""
        assert DimensionType.NUMERIC.value == "numeric"
        assert DimensionType.BINARY.value == "binary"
        assert DimensionType.CATEGORICAL.value == "categorical"
        assert DimensionType.ORDINAL.value == "ordinal"
        assert DimensionType.SCALE_1_4.value == "scale_1_4"


class TestAggregationMethod:
    """Test AggregationMethod enum."""

    def test_aggregation_methods(self):
        """Test all aggregation methods are defined."""
        assert AggregationMethod.MEAN.value == "mean"
        assert AggregationMethod.MAX.value == "max"
        assert AggregationMethod.MIN.value == "min"
        assert AggregationMethod.WEIGHTED_MEAN.value == "weighted_mean"
        assert AggregationMethod.PERCENTILE.value == "percentile"


class TestDimensionExample:
    """Test DimensionExample dataclass."""

    def test_dimension_example_creation(self):
        """Test creating a dimension example."""
        example = DimensionExample(
            input_description="Fast-paced combat scene",
            expected_score=0.9,
            reasoning="High action intensity with multiple combatants",
        )
        
        assert example.input_description == "Fast-paced combat scene"
        assert example.expected_score == 0.9
        assert example.reasoning == "High action intensity with multiple combatants"


class TestDimensionDefinition:
    """Test DimensionDefinition dataclass."""

    def test_dimension_definition_creation(self):
        """Test creating a dimension definition with defaults."""
        dim = DimensionDefinition(
            name="action_intensity",
            description="Level of action and excitement",
        )
        
        assert dim.name == "action_intensity"
        assert dim.description == "Level of action and excitement"
        assert dim.type == DimensionType.NUMERIC
        assert dim.weight == 1.0
        assert dim.scoring_prompt == ""
        assert dim.evaluation_criteria == []
        assert dim.examples == []
        assert dim.aggregation_method == AggregationMethod.MEAN
        assert dim.temporal_weight_decay == 0.0
        assert dim.min_score == 0.0
        assert dim.max_score == 1.0
        assert dim.requires_context is False

    def test_dimension_definition_with_all_params(self):
        """Test creating a dimension definition with all parameters."""
        example = DimensionExample(
            input_description="Combat scene",
            expected_score=0.9,
            reasoning="High action",
        )
        
        dim = DimensionDefinition(
            name="action_intensity",
            description="Level of action and excitement",
            type=DimensionType.SCALE_1_4,
            weight=2.0,
            scoring_prompt="Focus on combat and movement",
            evaluation_criteria=["Combat presence", "Movement speed"],
            examples=[example],
            aggregation_method=AggregationMethod.MAX,
            temporal_weight_decay=0.1,
            min_score=1.0,
            max_score=4.0,
            requires_context=True,
        )
        
        assert dim.type == DimensionType.SCALE_1_4
        assert dim.weight == 2.0
        assert dim.scoring_prompt == "Focus on combat and movement"
        assert len(dim.evaluation_criteria) == 2
        assert len(dim.examples) == 1
        assert dim.aggregation_method == AggregationMethod.MAX
        assert dim.temporal_weight_decay == 0.1
        assert dim.min_score == 1.0
        assert dim.max_score == 4.0
        assert dim.requires_context is True

    def test_dimension_definition_invalid_weight(self):
        """Test dimension definition with invalid weight."""
        with pytest.raises(ValueError, match="Weight must be between 0 and 10"):
            DimensionDefinition(
                name="test",
                description="Test dimension",
                weight=11.0,
            )
        
        with pytest.raises(ValueError, match="Weight must be between 0 and 10"):
            DimensionDefinition(
                name="test",
                description="Test dimension",
                weight=-1.0,
            )

    def test_dimension_definition_binary_type(self):
        """Test dimension definition with binary type sets correct bounds."""
        dim = DimensionDefinition(
            name="has_speech",
            description="Whether speech is present",
            type=DimensionType.BINARY,
            min_score=5.0,  # Should be overridden
            max_score=10.0,  # Should be overridden
        )
        
        assert dim.min_score == 0.0
        assert dim.max_score == 1.0

    def test_build_evaluation_prompt_numeric(self):
        """Test building evaluation prompt for numeric dimension."""
        dim = DimensionDefinition(
            name="action_intensity",
            description="Level of action and excitement",
            type=DimensionType.NUMERIC,
            evaluation_criteria=["Combat presence", "Movement speed"],
            scoring_prompt="Additional instructions here",
        )
        
        prompt = dim.build_evaluation_prompt()
        
        assert "Task: Evaluate this video segment for the dimension 'action_intensity'" in prompt
        assert "Definition: Level of action and excitement" in prompt
        assert "Evaluation Rubric:" in prompt
        assert "1. Combat presence" in prompt
        assert "2. Movement speed" in prompt
        assert "Score Range: 0.0 to 1.0 (decimal values allowed)" in prompt
        assert "Additional Instructions:" in prompt
        assert "Additional instructions here" in prompt
        assert '"dimension": "action_intensity"' in prompt

    def test_build_evaluation_prompt_scale_1_4(self):
        """Test building evaluation prompt for 1-4 scale dimension."""
        dim = DimensionDefinition(
            name="quality",
            description="Overall quality rating",
            type=DimensionType.SCALE_1_4,
        )
        
        prompt = dim.build_evaluation_prompt()
        
        assert "Scoring Scale:" in prompt
        assert "1 = Poor/None" in prompt
        assert "2 = Fair" in prompt
        assert "3 = Good" in prompt
        assert "4 = Excellent" in prompt

    def test_build_evaluation_prompt_binary(self):
        """Test building evaluation prompt for binary dimension."""
        dim = DimensionDefinition(
            name="has_speech",
            description="Whether speech is present",
            type=DimensionType.BINARY,
        )
        
        prompt = dim.build_evaluation_prompt()
        
        assert "Binary Scoring:" in prompt
        assert "0 = No/Absent" in prompt
        assert "1 = Yes/Present" in prompt

    def test_build_evaluation_prompt_with_examples(self):
        """Test building evaluation prompt with examples."""
        example = DimensionExample(
            input_description="Fast combat scene",
            expected_score=0.9,
            reasoning="Intense fighting sequence",
        )
        
        dim = DimensionDefinition(
            name="action",
            description="Action level",
            examples=[example],
        )
        
        prompt = dim.build_evaluation_prompt()
        
        assert "Reference Examples:" in prompt
        assert "Example 1:" in prompt
        assert "Scenario: Fast combat scene" in prompt
        assert "Score: 0.9" in prompt
        assert "Reasoning: Intense fighting sequence" in prompt

    def test_build_evaluation_prompt_with_timestamp(self):
        """Test building evaluation prompt with video timestamp."""
        dim = DimensionDefinition(
            name="action",
            description="Action level",
        )
        
        prompt = dim.build_evaluation_prompt(video_timestamp="02:45")
        
        assert "Focus on the moment at timestamp 02:45" in prompt

    def test_build_evaluation_prompt_without_timestamp(self):
        """Test building evaluation prompt without video timestamp."""
        dim = DimensionDefinition(
            name="action",
            description="Action level",
        )
        
        prompt = dim.build_evaluation_prompt()
        
        assert "Analyze the entire video segment provided" in prompt


class TestScoringRubric:
    """Test ScoringRubric dataclass."""

    def test_scoring_rubric_creation(self):
        """Test creating a scoring rubric with defaults."""
        rubric = ScoringRubric(
            name="Gaming Highlights",
            description="Rubric for gaming content",
        )
        
        assert rubric.name == "Gaming Highlights"
        assert rubric.description == "Rubric for gaming content"
        assert rubric.dimensions == []
        assert rubric.requires_all_dimensions is False
        assert rubric.min_confidence_threshold == 0.5
        assert rubric.normalization_enabled is True
        assert rubric.highlight_threshold == 0.7
        assert rubric.highlight_confidence_threshold == 0.6

    def test_scoring_rubric_with_dimensions(self):
        """Test creating a scoring rubric with dimensions."""
        dim1 = DimensionDefinition(name="action", description="Action level")
        dim2 = DimensionDefinition(name="humor", description="Humor level")
        
        rubric = ScoringRubric(
            name="Gaming Highlights",
            description="Rubric for gaming content",
            dimensions=[dim1, dim2],
        )
        
        assert len(rubric.dimensions) == 2
        assert rubric.dimensions[0].name == "action"
        assert rubric.dimensions[1].name == "humor"

    def test_scoring_rubric_duplicate_dimensions(self):
        """Test scoring rubric with duplicate dimension names."""
        dim1 = DimensionDefinition(name="action", description="Action level")
        dim2 = DimensionDefinition(name="action", description="Another action")
        
        with pytest.raises(ValueError, match="Dimension names must be unique"):
            ScoringRubric(
                name="Test",
                description="Test rubric",
                dimensions=[dim1, dim2],
            )

    def test_total_weight(self):
        """Test calculating total weight."""
        dim1 = DimensionDefinition(name="action", description="Action", weight=2.0)
        dim2 = DimensionDefinition(name="humor", description="Humor", weight=1.5)
        dim3 = DimensionDefinition(name="skill", description="Skill", weight=3.0)
        
        rubric = ScoringRubric(
            name="Test",
            description="Test rubric",
            dimensions=[dim1, dim2, dim3],
        )
        
        assert rubric.total_weight == 6.5

    def test_get_normalized_weights(self):
        """Test getting normalized weights."""
        dim1 = DimensionDefinition(name="action", description="Action", weight=2.0)
        dim2 = DimensionDefinition(name="humor", description="Humor", weight=1.0)
        dim3 = DimensionDefinition(name="skill", description="Skill", weight=1.0)
        
        rubric = ScoringRubric(
            name="Test",
            description="Test rubric",
            dimensions=[dim1, dim2, dim3],
        )
        
        weights = rubric.get_normalized_weights()
        
        assert weights["action"] == 0.5  # 2.0 / 4.0
        assert weights["humor"] == 0.25  # 1.0 / 4.0
        assert weights["skill"] == 0.25  # 1.0 / 4.0
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_get_normalized_weights_disabled(self):
        """Test getting weights when normalization is disabled."""
        dim1 = DimensionDefinition(name="action", description="Action", weight=2.0)
        dim2 = DimensionDefinition(name="humor", description="Humor", weight=1.5)
        
        rubric = ScoringRubric(
            name="Test",
            description="Test rubric",
            dimensions=[dim1, dim2],
            normalization_enabled=False,
        )
        
        weights = rubric.get_normalized_weights()
        
        assert weights["action"] == 2.0
        assert weights["humor"] == 1.5

    def test_get_normalized_weights_zero_total(self):
        """Test getting normalized weights when total weight is zero."""
        dim1 = DimensionDefinition(name="action", description="Action", weight=0.0)
        dim2 = DimensionDefinition(name="humor", description="Humor", weight=0.0)
        
        rubric = ScoringRubric(
            name="Test",
            description="Test rubric",
            dimensions=[dim1, dim2],
        )
        
        weights = rubric.get_normalized_weights()
        
        # Should distribute equally when total is zero
        assert weights["action"] == 0.5
        assert weights["humor"] == 0.5

    def test_add_dimension(self):
        """Test adding a dimension to rubric."""
        rubric = ScoringRubric(
            name="Test",
            description="Test rubric",
        )
        
        dim = DimensionDefinition(name="action", description="Action")
        rubric.add_dimension(dim)
        
        assert len(rubric.dimensions) == 1
        assert rubric.dimensions[0].name == "action"

    def test_add_duplicate_dimension(self):
        """Test adding duplicate dimension to rubric."""
        dim1 = DimensionDefinition(name="action", description="Action")
        rubric = ScoringRubric(
            name="Test",
            description="Test rubric",
            dimensions=[dim1],
        )
        
        dim2 = DimensionDefinition(name="action", description="Another action")
        with pytest.raises(ValueError, match="Dimension 'action' already exists"):
            rubric.add_dimension(dim2)

    def test_remove_dimension(self):
        """Test removing a dimension from rubric."""
        dim1 = DimensionDefinition(name="action", description="Action")
        dim2 = DimensionDefinition(name="humor", description="Humor")
        
        rubric = ScoringRubric(
            name="Test",
            description="Test rubric",
            dimensions=[dim1, dim2],
        )
        
        rubric.remove_dimension("action")
        
        assert len(rubric.dimensions) == 1
        assert rubric.dimensions[0].name == "humor"

    def test_remove_nonexistent_dimension(self):
        """Test removing nonexistent dimension from rubric."""
        rubric = ScoringRubric(
            name="Test",
            description="Test rubric",
        )
        
        # Should not raise an error
        rubric.remove_dimension("nonexistent")
        assert len(rubric.dimensions) == 0

    def test_build_multi_dimensional_prompt(self):
        """Test building multi-dimensional prompt."""
        dim1 = DimensionDefinition(
            name="action",
            description="Action level",
            evaluation_criteria=["Combat", "Movement"],
        )
        dim2 = DimensionDefinition(
            name="humor",
            description="Humor level",
            evaluation_criteria=["Funny moments"],
        )
        
        rubric = ScoringRubric(
            name="Gaming Highlights",
            description="Rubric for gaming content",
            dimensions=[dim1, dim2],
        )
        
        prompt = rubric.build_multi_dimensional_prompt()
        
        assert "Task: Evaluate this video segment using the 'Gaming Highlights' scoring rubric" in prompt
        assert "Description: Rubric for gaming content" in prompt
        assert "DIMENSIONS TO EVALUATE:" in prompt
        assert "### Dimension 1: action" in prompt
        assert "Description: Action level" in prompt
        assert "### Dimension 2: humor" in prompt
        assert "Description: Humor level" in prompt
        assert "Criteria:" in prompt
        assert "Combat" in prompt
        assert "Movement" in prompt
        assert "Funny moments" in prompt