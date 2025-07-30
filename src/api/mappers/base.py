"""Base mapper protocol for API DTOs to domain entities."""

from typing import Protocol, TypeVar, runtime_checkable

# Type variables for DTOs and domain entities
TDto = TypeVar("TDto")  # API DTO (Pydantic model)
TDomain = TypeVar("TDomain")  # Domain entity (dataclass)


@runtime_checkable
class APIMapper(Protocol[TDto, TDomain]):
    """Protocol for mapping between API DTOs and domain entities.

    This follows the DDD pattern of keeping domain entities separate
    from API representations. Uses Protocol for structural subtyping
    instead of ABC, following Pythonic patterns.
    """

    def to_domain(self, dto: TDto) -> TDomain:
        """Convert API DTO to domain entity.

        Args:
            dto: API data transfer object (Pydantic model)

        Returns:
            Domain entity (dataclass)
        """
        ...

    def to_dto(self, entity: TDomain) -> TDto:
        """Convert domain entity to API DTO.

        Args:
            entity: Domain entity (dataclass)

        Returns:
            API data transfer object (Pydantic model)
        """
        ...
