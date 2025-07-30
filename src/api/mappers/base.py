"""Base mapper for API DTOs to domain entities."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

# Type variables for DTOs and domain entities
TDto = TypeVar("TDto")  # API DTO (Pydantic model)
TDomain = TypeVar("TDomain")  # Domain entity (dataclass)


class BaseAPIMapper(ABC, Generic[TDto, TDomain]):
    """Base class for mapping between API DTOs and domain entities.
    
    This follows the DDD pattern of keeping domain entities separate
    from API representations.
    """
    
    @abstractmethod
    def to_domain(self, dto: TDto) -> TDomain:
        """Convert API DTO to domain entity.
        
        Args:
            dto: API data transfer object (Pydantic model)
            
        Returns:
            Domain entity (dataclass)
        """
        pass
    
    @abstractmethod
    def to_dto(self, entity: TDomain) -> TDto:
        """Convert domain entity to API DTO.
        
        Args:
            entity: Domain entity (dataclass)
            
        Returns:
            API data transfer object (Pydantic model)
        """
        pass