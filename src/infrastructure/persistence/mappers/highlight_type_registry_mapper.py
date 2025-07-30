"""Mapper for HighlightTypeRegistry domain entity to persistence model conversion."""


from src.infrastructure.persistence.mappers.base import Mapper
from src.domain.entities.highlight_type_registry import (
    HighlightTypeRegistry as DomainHighlightTypeRegistry,
    HighlightTypeDefinition,
)
from src.domain.value_objects.timestamp import Timestamp
from src.infrastructure.persistence.models.highlight_type_registry import (
    HighlightTypeRegistry as PersistenceHighlightTypeRegistry,
)


class HighlightTypeRegistryMapper(
    Mapper[DomainHighlightTypeRegistry, PersistenceHighlightTypeRegistry]
):
    """Maps between HighlightTypeRegistry domain entity and persistence model."""

    def to_domain(
        self, model: PersistenceHighlightTypeRegistry
    ) -> DomainHighlightTypeRegistry:
        """Convert HighlightTypeRegistry persistence model to domain entity."""
        # Parse type definitions from JSON
        types = {}
        for type_id, type_data in model.types.items():
            type_def = HighlightTypeDefinition(
                id=type_data["id"],
                name=type_data["name"],
                description=type_data["description"],
                criteria=type_data.get("criteria", {}),
                priority=type_data.get("priority", 0),
                auto_assign=type_data.get("auto_assign", True),
                icon=type_data.get("icon"),
                color=type_data.get("color"),
            )
            types[type_id] = type_def

        return DomainHighlightTypeRegistry(
            id=model.id,
            organization_id=model.organization_id,
            types=types,
            allow_multiple_types=model.allow_multiple_types,
            max_types_per_highlight=model.max_types_per_highlight,
            include_built_in_types=model.include_built_in_types,
            created_at=Timestamp(model.created_at),
            updated_at=Timestamp(model.updated_at),
        )

    def to_persistence(
        self, entity: DomainHighlightTypeRegistry
    ) -> PersistenceHighlightTypeRegistry:
        """Convert HighlightTypeRegistry domain entity to persistence model."""
        model = PersistenceHighlightTypeRegistry()

        # Set basic attributes
        if entity.id is not None:
            model.id = entity.id

        model.organization_id = entity.organization_id

        # Serialize type definitions to JSON
        types_data = {}
        for type_id, type_def in entity.types.items():
            type_data = {
                "id": type_def.id,
                "name": type_def.name,
                "description": type_def.description,
                "criteria": type_def.criteria,
                "priority": type_def.priority,
                "auto_assign": type_def.auto_assign,
            }
            if type_def.icon:
                type_data["icon"] = type_def.icon
            if type_def.color:
                type_data["color"] = type_def.color

            types_data[type_id] = type_data

        model.types = types_data

        # Set configuration
        model.allow_multiple_types = entity.allow_multiple_types
        model.max_types_per_highlight = entity.max_types_per_highlight
        model.include_built_in_types = entity.include_built_in_types

        # Set audit timestamps for existing entities
        if entity.id is not None:
            model.created_at = entity.created_at.value
            model.updated_at = entity.updated_at.value

        return model
