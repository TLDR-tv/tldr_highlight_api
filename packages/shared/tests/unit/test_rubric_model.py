"""Unit tests for rubric domain model."""

import pytest
from uuid import uuid4
from datetime import datetime, timezone

from shared.domain.models.rubric import Rubric, RubricVisibility


class TestRubric:
    """Test Rubric domain model."""

    def test_rubric_creation(self):
        """Test basic rubric creation."""
        rubric_id = uuid4()
        org_id = uuid4()
        
        rubric = Rubric(
            id=rubric_id,
            organization_id=org_id,
            name="Test Rubric",
            description="Test description"
        )
        
        assert rubric.id == rubric_id
        assert rubric.organization_id == org_id
        assert rubric.name == "Test Rubric"
        assert rubric.description == "Test description"
        assert rubric.visibility == RubricVisibility.PRIVATE
        assert rubric.is_active is True
        assert rubric.version == 1
        assert rubric.usage_count == 0
        assert rubric.last_used_at is None

    def test_rubric_name_required(self):
        """Test that rubric name is required."""
        with pytest.raises(ValueError, match="Rubric name is required"):
            Rubric(organization_id=uuid4(), name="")

    def test_system_rubric_validation(self):
        """Test system rubric validation."""
        # System rubric should not have organization_id
        with pytest.raises(ValueError, match="System rubrics cannot belong to an organization"):
            Rubric(
                name="System Rubric",
                organization_id=uuid4(),
                visibility=RubricVisibility.SYSTEM
            )
        
        # Valid system rubric
        rubric = Rubric(
            name="System Rubric",
            visibility=RubricVisibility.SYSTEM
        )
        assert rubric.organization_id is None
        assert rubric.is_system_rubric is True

    def test_private_rubric_validation(self):
        """Test private rubric validation."""
        # Private rubric must have organization_id
        with pytest.raises(ValueError, match="Private rubrics must belong to an organization"):
            Rubric(
                name="Private Rubric",
                visibility=RubricVisibility.PRIVATE
            )
        
        # Valid private rubric
        org_id = uuid4()
        rubric = Rubric(
            name="Private Rubric",
            organization_id=org_id,
            visibility=RubricVisibility.PRIVATE
        )
        assert rubric.organization_id == org_id
        assert rubric.is_system_rubric is False

    def test_rubric_properties(self):
        """Test rubric properties."""
        org_id = uuid4()
        
        # System rubric
        system_rubric = Rubric(
            name="System Rubric",
            visibility=RubricVisibility.SYSTEM
        )
        assert system_rubric.is_system_rubric is True
        assert system_rubric.is_template is True
        
        # Public rubric
        public_rubric = Rubric(
            name="Public Rubric",
            organization_id=org_id,
            visibility=RubricVisibility.PUBLIC
        )
        assert public_rubric.is_system_rubric is False
        assert public_rubric.is_template is True
        
        # Private rubric
        private_rubric = Rubric(
            name="Private Rubric",
            organization_id=org_id,
            visibility=RubricVisibility.PRIVATE
        )
        assert private_rubric.is_system_rubric is False
        assert private_rubric.is_template is False

    def test_can_be_edited_by(self):
        """Test rubric editing permissions."""
        org_id = uuid4()
        other_org_id = uuid4()
        
        # System rubric cannot be edited
        system_rubric = Rubric(
            name="System Rubric",
            visibility=RubricVisibility.SYSTEM
        )
        assert system_rubric.can_be_edited_by(org_id) is False
        
        # Organization can edit its own rubrics
        org_rubric = Rubric(
            name="Org Rubric",
            organization_id=org_id,
            visibility=RubricVisibility.PRIVATE
        )
        assert org_rubric.can_be_edited_by(org_id) is True
        assert org_rubric.can_be_edited_by(other_org_id) is False

    def test_can_be_used_by(self):
        """Test rubric usage permissions."""
        org_id = uuid4()
        other_org_id = uuid4()
        
        # System rubric can be used by anyone
        system_rubric = Rubric(
            name="System Rubric",
            visibility=RubricVisibility.SYSTEM
        )
        assert system_rubric.can_be_used_by(org_id) is True
        assert system_rubric.can_be_used_by(other_org_id) is True
        
        # Public rubric can be used by anyone
        public_rubric = Rubric(
            name="Public Rubric",
            organization_id=org_id,
            visibility=RubricVisibility.PUBLIC
        )
        assert public_rubric.can_be_used_by(org_id) is True
        assert public_rubric.can_be_used_by(other_org_id) is True
        
        # Private rubric can only be used by owner
        private_rubric = Rubric(
            name="Private Rubric",
            organization_id=org_id,
            visibility=RubricVisibility.PRIVATE
        )
        assert private_rubric.can_be_used_by(org_id) is True
        assert private_rubric.can_be_used_by(other_org_id) is False

    def test_increment_usage(self):
        """Test usage tracking."""
        rubric = Rubric(
            name="Test Rubric",
            organization_id=uuid4()
        )
        
        assert rubric.usage_count == 0
        assert rubric.last_used_at is None
        
        rubric.increment_usage()
        
        assert rubric.usage_count == 1
        assert rubric.last_used_at is not None
        assert isinstance(rubric.last_used_at, datetime)
        assert rubric.updated_at is not None

    def test_clone_for_organization(self):
        """Test rubric cloning."""
        original_org_id = uuid4()
        target_org_id = uuid4()
        
        original_rubric = Rubric(
            name="Original Rubric",
            organization_id=original_org_id,
            description="Original description",
            config={"dimension": "action", "weight": 1.0},
            visibility=RubricVisibility.PUBLIC
        )
        
        # Clone with default name
        cloned_rubric = original_rubric.clone_for_organization(target_org_id)
        
        assert cloned_rubric.id != original_rubric.id  # New ID
        assert cloned_rubric.organization_id == target_org_id
        assert cloned_rubric.name == "Original Rubric (Copy)"
        assert cloned_rubric.description == "Original description"
        assert cloned_rubric.config == {"dimension": "action", "weight": 1.0}
        assert cloned_rubric.visibility == RubricVisibility.PRIVATE
        assert cloned_rubric.created_at != original_rubric.created_at
        
        # Clone with custom name
        custom_clone = original_rubric.clone_for_organization(
            target_org_id, 
            name="Custom Clone Name"
        )
        assert custom_clone.name == "Custom Clone Name"

    def test_rubric_config_handling(self):
        """Test rubric configuration handling."""
        config = {
            "dimensions": [
                {"name": "action", "weight": 1.0},
                {"name": "emotion", "weight": 0.8}
            ],
            "threshold": 0.7
        }
        
        rubric = Rubric(
            name="Test Rubric",
            organization_id=uuid4(),
            config=config
        )
        
        assert rubric.config == config
        
        # Test config is independent when cloning
        cloned = rubric.clone_for_organization(uuid4())
        cloned.config["threshold"] = 0.9
        
        assert rubric.config["threshold"] == 0.7
        assert cloned.config["threshold"] == 0.9


class TestRubricVisibility:
    """Test RubricVisibility enum."""

    def test_rubric_visibility_values(self):
        """Test enum values."""
        assert RubricVisibility.PRIVATE.value == "private"
        assert RubricVisibility.PUBLIC.value == "public"
        assert RubricVisibility.SYSTEM.value == "system"

    def test_rubric_visibility_comparison(self):
        """Test enum comparison."""
        assert RubricVisibility.PRIVATE == RubricVisibility.PRIVATE
        assert RubricVisibility.PRIVATE != RubricVisibility.PUBLIC
        assert RubricVisibility.PRIVATE != RubricVisibility.SYSTEM