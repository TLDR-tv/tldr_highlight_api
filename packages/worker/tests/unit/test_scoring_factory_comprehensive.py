"""Comprehensive tests for scoring factory."""

import pytest
from worker.services.scoring_factory import ScoringRubricFactory
from worker.services.dimension_framework import (
    DimensionDefinition,
    DimensionType,
    AggregationMethod,
    ScoringRubric
)


class TestScoringRubricFactory:
    """Test ScoringRubricFactory methods comprehensively."""

    def test_create_gaming_rubric(self):
        """Test creating gaming rubric."""
        rubric = ScoringRubricFactory.create_gaming_rubric()
        
        assert isinstance(rubric, ScoringRubric)
        assert rubric.name == "Gaming Highlights"
        assert rubric.description == "Scoring rubric for gaming content highlights"
        assert rubric.highlight_threshold == 0.7
        assert rubric.highlight_confidence_threshold == 0.6
        assert len(rubric.dimensions) > 0
        
        # Check for expected gaming dimensions
        dimension_names = [d.name for d in rubric.dimensions]
        assert "action_intensity" in dimension_names
        assert "skill_display" in dimension_names
        assert "emotional_moment" in dimension_names
        assert "humor" in dimension_names

    def test_create_gaming_rubric_with_custom_params(self):
        """Test creating gaming rubric with custom parameters."""
        custom_name = "Custom Gaming Rubric"
        custom_description = "Custom description for gaming content"
        
        rubric = ScoringRubricFactory.create_gaming_rubric(
            name=custom_name,
            description=custom_description
        )
        
        assert rubric.name == custom_name
        assert rubric.description == custom_description
        assert len(rubric.dimensions) > 0

    def test_create_sports_rubric(self):
        """Test creating sports rubric."""
        rubric = ScoringRubricFactory.create_sports_rubric()
        
        assert isinstance(rubric, ScoringRubric)
        assert rubric.name == "Sports Highlights"
        assert rubric.description == "Scoring rubric for sports content highlights"
        assert rubric.highlight_threshold == 0.65
        assert rubric.highlight_confidence_threshold == 0.7
        assert len(rubric.dimensions) > 0
        
        # Check for expected sports dimensions
        dimension_names = [d.name for d in rubric.dimensions]
        assert "scoring_play" in dimension_names
        assert "momentum_shift" in dimension_names
        assert "action_intensity" in dimension_names
        assert "emotional_moment" in dimension_names

    def test_create_education_rubric(self):
        """Test creating education rubric."""
        rubric = ScoringRubricFactory.create_education_rubric()
        
        assert isinstance(rubric, ScoringRubric)
        assert rubric.name == "Educational Highlights"
        assert rubric.description == "Scoring rubric for educational content highlights"
        assert rubric.highlight_threshold == 0.75
        assert rubric.highlight_confidence_threshold == 0.65
        assert len(rubric.dimensions) > 0
        
        # Check for expected education dimensions
        dimension_names = [d.name for d in rubric.dimensions]
        assert "concept_clarity" in dimension_names
        assert "educational_value" in dimension_names
        assert "engagement_level" in dimension_names
        assert "visual_interest" in dimension_names

    def test_create_corporate_rubric(self):
        """Test creating corporate rubric."""
        rubric = ScoringRubricFactory.create_corporate_rubric()
        
        assert isinstance(rubric, ScoringRubric)
        assert rubric.name == "Corporate Highlights"
        assert rubric.description == "Scoring rubric for corporate/business content highlights"
        assert rubric.highlight_threshold == 0.8
        assert rubric.highlight_confidence_threshold == 0.7
        assert len(rubric.dimensions) > 0
        
        # Check for expected corporate dimensions
        dimension_names = [d.name for d in rubric.dimensions]
        assert "key_decision" in dimension_names
        assert "technical_demo" in dimension_names
        assert "concept_clarity" in dimension_names
        assert "engagement_level" in dimension_names

    def test_create_general_rubric(self):
        """Test creating general rubric."""
        rubric = ScoringRubricFactory.create_general_rubric()
        
        assert isinstance(rubric, ScoringRubric)
        assert rubric.name == "General Highlights"
        assert rubric.description == "General-purpose scoring rubric for diverse content"
        assert rubric.highlight_threshold == 0.7
        assert rubric.highlight_confidence_threshold == 0.6
        assert len(rubric.dimensions) > 0
        
        # Check for expected general dimensions
        dimension_names = [d.name for d in rubric.dimensions]
        assert "action_intensity" in dimension_names
        assert "emotional_moment" in dimension_names
        assert "visual_interest" in dimension_names
        assert "narrative_importance" in dimension_names
        assert "humor" in dimension_names

    def test_create_dyli_rubric(self):
        """Test creating DYLI rubric."""
        rubric = ScoringRubricFactory.create_dyli_rubric()
        
        assert isinstance(rubric, ScoringRubric)
        assert rubric.name == "DYLI Trading Card & Collectibles"
        assert "trading card pack openings" in rubric.description
        assert rubric.highlight_threshold == 0.65
        assert rubric.highlight_confidence_threshold == 0.7
        assert len(rubric.dimensions) > 0
        
        # Check for expected DYLI-specific dimensions
        dimension_names = [d.name for d in rubric.dimensions]
        assert "rare_pull_moment" in dimension_names
        assert "product_showcase" in dimension_names
        assert "excitement_level" in dimension_names
        assert "collector_reaction" in dimension_names
        assert "value_discussion" in dimension_names
        assert "community_engagement" in dimension_names

    def test_dyli_rubric_dimension_weights(self):
        """Test DYLI rubric has correct dimension weights."""
        rubric = ScoringRubricFactory.create_dyli_rubric()
        
        dimensions_by_name = {d.name: d for d in rubric.dimensions}
        
        # Check specific weights for DYLI rubric
        assert dimensions_by_name["rare_pull_moment"].weight == 3.0
        assert dimensions_by_name["product_showcase"].weight == 2.5
        assert dimensions_by_name["excitement_level"].weight == 2.0
        assert dimensions_by_name["collector_reaction"].weight == 1.8
        assert dimensions_by_name["value_discussion"].weight == 1.5
        assert dimensions_by_name["community_engagement"].weight == 1.2

    def test_dyli_rubric_dimension_types(self):
        """Test DYLI rubric has correct dimension types."""
        rubric = ScoringRubricFactory.create_dyli_rubric()
        
        dimensions_by_name = {d.name: d for d in rubric.dimensions}
        
        assert dimensions_by_name["rare_pull_moment"].type == DimensionType.BINARY
        assert dimensions_by_name["product_showcase"].type == DimensionType.SCALE_1_4
        assert dimensions_by_name["excitement_level"].type == DimensionType.SCALE_1_4
        assert dimensions_by_name["collector_reaction"].type == DimensionType.SCALE_1_4
        assert dimensions_by_name["value_discussion"].type == DimensionType.NUMERIC
        assert dimensions_by_name["community_engagement"].type == DimensionType.SCALE_1_4

    def test_dyli_rubric_aggregation_methods(self):
        """Test DYLI rubric has correct aggregation methods."""
        rubric = ScoringRubricFactory.create_dyli_rubric()
        
        dimensions_by_name = {d.name: d for d in rubric.dimensions}
        
        assert dimensions_by_name["rare_pull_moment"].aggregation_method == AggregationMethod.MAX
        assert dimensions_by_name["product_showcase"].aggregation_method == AggregationMethod.MEAN
        assert dimensions_by_name["excitement_level"].aggregation_method == AggregationMethod.MAX
        assert dimensions_by_name["collector_reaction"].aggregation_method == AggregationMethod.MAX
        assert dimensions_by_name["value_discussion"].aggregation_method == AggregationMethod.MAX
        assert dimensions_by_name["community_engagement"].aggregation_method == AggregationMethod.MEAN

    def test_create_custom_rubric_basic(self):
        """Test creating basic custom rubric."""
        dimensions = [
            DimensionDefinition(
                name="custom_dimension",
                description="A custom dimension",
                type=DimensionType.SCALE_1_4,
                weight=1.0
            )
        ]
        
        rubric = ScoringRubricFactory.create_custom_rubric(
            name="Custom Rubric",
            description="A custom rubric",
            dimensions=dimensions
        )
        
        assert rubric.name == "Custom Rubric"
        assert rubric.description == "A custom rubric"
        assert len(rubric.dimensions) == 1
        assert rubric.dimensions[0].name == "custom_dimension"
        assert rubric.highlight_threshold == 0.7  # default
        assert rubric.highlight_confidence_threshold == 0.6  # default
        assert rubric.requires_all_dimensions is False  # default
        assert rubric.normalization_enabled is True  # default

    def test_create_custom_rubric_with_all_params(self):
        """Test creating custom rubric with all parameters."""
        dimensions = [
            DimensionDefinition(
                name="dim1",
                description="Dimension 1",
                type=DimensionType.BINARY,
                weight=2.0
            ),
            DimensionDefinition(
                name="dim2",
                description="Dimension 2",
                type=DimensionType.NUMERIC,
                weight=1.5
            )
        ]
        
        rubric = ScoringRubricFactory.create_custom_rubric(
            name="Advanced Custom Rubric",
            description="An advanced custom rubric",
            dimensions=dimensions,
            highlight_threshold=0.85,
            highlight_confidence_threshold=0.75,
            requires_all_dimensions=True,
            normalization_enabled=False
        )
        
        assert rubric.name == "Advanced Custom Rubric"
        assert rubric.description == "An advanced custom rubric"
        assert len(rubric.dimensions) == 2
        assert rubric.highlight_threshold == 0.85
        assert rubric.highlight_confidence_threshold == 0.75
        assert rubric.requires_all_dimensions is True
        assert rubric.normalization_enabled is False

    def test_create_from_template_names_success(self):
        """Test creating rubric from template names."""
        template_names = [
            "action_intensity",
            "educational_value",
            "humor"
        ]
        
        rubric = ScoringRubricFactory.create_from_template_names(
            name="Template Rubric",
            description="Rubric from templates",
            template_names=template_names
        )
        
        assert rubric.name == "Template Rubric"
        assert rubric.description == "Rubric from templates"
        assert len(rubric.dimensions) == 3
        
        dimension_names = [d.name for d in rubric.dimensions]
        assert "action_intensity" in dimension_names
        assert "educational_value" in dimension_names
        assert "humor" in dimension_names

    def test_create_from_template_names_with_kwargs(self):
        """Test creating rubric from template names with additional kwargs."""
        template_names = ["action_intensity", "visual_interest"]
        
        rubric = ScoringRubricFactory.create_from_template_names(
            name="Template Rubric",
            description="Rubric from templates",
            template_names=template_names,
            highlight_threshold=0.9,
            highlight_confidence_threshold=0.85,
            requires_all_dimensions=True
        )
        
        assert len(rubric.dimensions) == 2
        assert rubric.highlight_threshold == 0.9
        assert rubric.highlight_confidence_threshold == 0.85
        assert rubric.requires_all_dimensions is True

    def test_create_from_template_names_invalid_template(self):
        """Test creating rubric with invalid template name."""
        template_names = [
            "action_intensity",
            "nonexistent_template",
            "humor"
        ]
        
        with pytest.raises(ValueError) as exc_info:
            ScoringRubricFactory.create_from_template_names(
                name="Template Rubric",
                description="Rubric from templates",
                template_names=template_names
            )
        
        assert "Unknown template: nonexistent_template" in str(exc_info.value)
        assert "Available templates:" in str(exc_info.value)

    def test_create_from_empty_template_names(self):
        """Test creating rubric with empty template names list."""
        rubric = ScoringRubricFactory.create_from_template_names(
            name="Empty Rubric",
            description="Rubric with no templates",
            template_names=[]
        )
        
        assert rubric.name == "Empty Rubric"
        assert len(rubric.dimensions) == 0

    def test_all_rubric_types_have_valid_dimensions(self):
        """Test that all factory methods create valid rubrics with dimensions."""
        factory_methods = [
            ScoringRubricFactory.create_gaming_rubric,
            ScoringRubricFactory.create_sports_rubric,
            ScoringRubricFactory.create_education_rubric,
            ScoringRubricFactory.create_corporate_rubric,
            ScoringRubricFactory.create_general_rubric,
            ScoringRubricFactory.create_dyli_rubric
        ]
        
        for factory_method in factory_methods:
            rubric = factory_method()
            assert isinstance(rubric, ScoringRubric)
            assert len(rubric.dimensions) > 0
            assert all(isinstance(d, DimensionDefinition) for d in rubric.dimensions)
            assert rubric.highlight_threshold > 0
            assert rubric.highlight_confidence_threshold > 0

    def test_rubric_dimension_uniqueness(self):
        """Test that rubrics don't have duplicate dimensions."""
        factory_methods = [
            ScoringRubricFactory.create_gaming_rubric,
            ScoringRubricFactory.create_sports_rubric,
            ScoringRubricFactory.create_education_rubric,
            ScoringRubricFactory.create_corporate_rubric,
            ScoringRubricFactory.create_general_rubric,
            ScoringRubricFactory.create_dyli_rubric
        ]
        
        for factory_method in factory_methods:
            rubric = factory_method()
            dimension_names = [d.name for d in rubric.dimensions]
            assert len(dimension_names) == len(set(dimension_names)), \
                f"Duplicate dimensions found in {factory_method.__name__}"

    def test_dimension_weights_are_valid(self):
        """Test that all dimensions have valid weights."""
        factory_methods = [
            ScoringRubricFactory.create_gaming_rubric,
            ScoringRubricFactory.create_sports_rubric,
            ScoringRubricFactory.create_education_rubric,
            ScoringRubricFactory.create_corporate_rubric,
            ScoringRubricFactory.create_general_rubric,
            ScoringRubricFactory.create_dyli_rubric
        ]
        
        for factory_method in factory_methods:
            rubric = factory_method()
            for dimension in rubric.dimensions:
                assert 0 <= dimension.weight <= 10, \
                    f"Invalid weight {dimension.weight} in {dimension.name}"

    def test_rubric_threshold_consistency(self):
        """Test that rubric thresholds are reasonable."""
        factory_methods = [
            ScoringRubricFactory.create_gaming_rubric,
            ScoringRubricFactory.create_sports_rubric,
            ScoringRubricFactory.create_education_rubric,
            ScoringRubricFactory.create_corporate_rubric,
            ScoringRubricFactory.create_general_rubric,
            ScoringRubricFactory.create_dyli_rubric
        ]
        
        for factory_method in factory_methods:
            rubric = factory_method()
            assert 0 < rubric.highlight_threshold <= 1.0
            assert 0 < rubric.highlight_confidence_threshold <= 1.0
            # Generally confidence threshold should be higher than score threshold
            # but this isn't a hard rule, so we don't enforce it