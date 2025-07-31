"""Highlight management use cases."""

from dataclasses import dataclass
from typing import Optional, List

from src.application.use_cases.base import UseCase, UseCaseResult, ResultStatus
from src.domain.entities.highlight import Highlight
from src.domain.entities.highlight_type_registry import (
    BuiltInHighlightType as HighlightType,
)
from src.domain.repositories.highlight_repository import HighlightRepository
from src.domain.repositories.stream_repository import StreamRepository
from src.domain.repositories.user_repository import UserRepository


@dataclass
class GetHighlightRequest:
    """Request to get a highlight."""

    user_id: int
    highlight_id: int


@dataclass
class GetHighlightResult(UseCaseResult):
    """Result of getting a highlight."""

    highlight: Optional[Highlight] = None
    stream_info: Optional[dict] = None


@dataclass
class ListHighlightsRequest:
    """Request to list highlights."""

    user_id: int
    stream_id: Optional[int] = None
    highlight_type: Optional[str] = None
    min_confidence: Optional[float] = None
    page: int = 1
    per_page: int = 20


@dataclass
class ListHighlightsResult(UseCaseResult):
    """Result of listing highlights."""

    highlights: List[Highlight] = None
    total: Optional[int] = None
    page: Optional[int] = None
    per_page: Optional[int] = None


@dataclass
class UpdateHighlightRequest:
    """Request to update highlight metadata."""

    user_id: int
    highlight_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class UpdateHighlightResult(UseCaseResult):
    """Result of updating highlight."""

    highlight: Optional[Highlight] = None


@dataclass
class DeleteHighlightRequest:
    """Request to delete a highlight."""

    user_id: int
    highlight_id: int


@dataclass
class DeleteHighlightResult(UseCaseResult):
    """Result of deleting highlight."""

    pass


@dataclass
class ExportHighlightRequest:
    """Request to export/download a highlight."""

    user_id: int
    highlight_id: int
    format: str = "mp4"
    quality: str = "high"


@dataclass
class ExportHighlightResult(UseCaseResult):
    """Result of exporting highlight."""

    download_url: Optional[str] = None
    expires_at: Optional[str] = None


class HighlightManagementUseCase(UseCase[GetHighlightRequest, GetHighlightResult]):
    """Use case for highlight management operations."""

    def __init__(
        self,
        highlight_repo: HighlightRepository,
        stream_repo: StreamRepository,
        user_repo: UserRepository,
    ):
        """Initialize highlight management use case.

        Args:
            highlight_repo: Repository for highlight operations
            stream_repo: Repository for stream operations
            user_repo: Repository for user operations
        """
        self.highlight_repo = highlight_repo
        self.stream_repo = stream_repo
        self.user_repo = user_repo

    async def get_highlight(self, request: GetHighlightRequest) -> GetHighlightResult:
        """Get a specific highlight.

        Args:
            request: Get highlight request

        Returns:
            Highlight details
        """
        try:
            # Get highlight
            highlight = await self.highlight_repo.get(request.highlight_id)
            if not highlight:
                return GetHighlightResult(
                    status=ResultStatus.NOT_FOUND, errors=["Highlight not found"]
                )

            # Check access permission via stream ownership
            stream = await self.stream_repo.get(highlight.stream_id)
            if not stream or stream.user_id != request.user_id:
                return GetHighlightResult(
                    status=ResultStatus.UNAUTHORIZED, errors=["Access denied"]
                )

            # Get stream info for context
            stream_info = {
                "stream_id": stream.id,
                "stream_url": str(stream.url),
                "stream_title": stream.title or "Untitled Stream",
                "platform": stream.platform.value,
            }

            return GetHighlightResult(
                status=ResultStatus.SUCCESS,
                highlight=highlight,
                stream_info=stream_info,
                message="Highlight retrieved successfully",
            )

        except Exception as e:
            return GetHighlightResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to get highlight: {str(e)}"],
            )

    async def list_highlights(
        self, request: ListHighlightsRequest
    ) -> ListHighlightsResult:
        """List highlights for a user.

        Args:
            request: List highlights request

        Returns:
            Paginated list of highlights
        """
        try:
            # If specific stream is requested, check ownership
            if request.stream_id:
                stream = await self.stream_repo.get(request.stream_id)
                if not stream or stream.user_id != request.user_id:
                    return ListHighlightsResult(
                        status=ResultStatus.UNAUTHORIZED, errors=["Access denied"]
                    )

                # Get highlights for specific stream
                highlights = await self.highlight_repo.get_by_stream(request.stream_id)
            else:
                # Get all user's streams first
                # Note: This is simplified - in production we'd have a more efficient query
                highlights = []
                # TODO: Implement efficient user highlights query
                return ListHighlightsResult(
                    status=ResultStatus.SUCCESS,
                    highlights=[],
                    total=0,
                    page=request.page,
                    per_page=request.per_page,
                    message="Highlight listing across all streams not yet implemented",
                )

            # Apply filters
            if request.highlight_type:
                try:
                    filter_type = HighlightType(request.highlight_type)
                    highlights = [
                        h for h in highlights if h.highlight_type == filter_type
                    ]
                except ValueError:
                    pass  # Invalid type, skip filter

            if request.min_confidence:
                highlights = [
                    h
                    for h in highlights
                    if h.confidence_score.value >= request.min_confidence
                ]

            # Sort by confidence score (highest first)
            highlights.sort(key=lambda h: h.confidence_score.value, reverse=True)

            # Apply pagination
            total = len(highlights)
            start = (request.page - 1) * request.per_page
            end = start + request.per_page
            paginated_highlights = highlights[start:end]

            return ListHighlightsResult(
                status=ResultStatus.SUCCESS,
                highlights=paginated_highlights,
                total=total,
                page=request.page,
                per_page=request.per_page,
                message=f"Found {total} highlights",
            )

        except Exception as e:
            return ListHighlightsResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to list highlights: {str(e)}"],
            )

    async def update_highlight(
        self, request: UpdateHighlightRequest
    ) -> UpdateHighlightResult:
        """Update highlight metadata.

        Args:
            request: Update highlight request

        Returns:
            Updated highlight
        """
        try:
            # Get highlight
            highlight = await self.highlight_repo.get(request.highlight_id)
            if not highlight:
                return UpdateHighlightResult(
                    status=ResultStatus.NOT_FOUND, errors=["Highlight not found"]
                )

            # Check access permission
            stream = await self.stream_repo.get(highlight.stream_id)
            if not stream or stream.user_id != request.user_id:
                return UpdateHighlightResult(
                    status=ResultStatus.UNAUTHORIZED, errors=["Access denied"]
                )

            # For now, return an error as highlight entities are immutable
            # In a full implementation, we might create new highlight versions
            return UpdateHighlightResult(
                status=ResultStatus.VALIDATION_ERROR,
                errors=["Highlight updates are not currently supported"],
            )

        except Exception as e:
            return UpdateHighlightResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to update highlight: {str(e)}"],
            )

    async def delete_highlight(
        self, request: DeleteHighlightRequest
    ) -> DeleteHighlightResult:
        """Delete a highlight.

        Args:
            request: Delete highlight request

        Returns:
            Deletion result
        """
        try:
            # Get highlight
            highlight = await self.highlight_repo.get(request.highlight_id)
            if not highlight:
                return DeleteHighlightResult(
                    status=ResultStatus.NOT_FOUND, errors=["Highlight not found"]
                )

            # Check access permission
            stream = await self.stream_repo.get(highlight.stream_id)
            if not stream or stream.user_id != request.user_id:
                return DeleteHighlightResult(
                    status=ResultStatus.UNAUTHORIZED, errors=["Access denied"]
                )

            # Delete highlight
            await self.highlight_repo.delete(request.highlight_id)

            return DeleteHighlightResult(
                status=ResultStatus.SUCCESS, message="Highlight deleted successfully"
            )

        except Exception as e:
            return DeleteHighlightResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to delete highlight: {str(e)}"],
            )

    async def export_highlight(
        self, request: ExportHighlightRequest
    ) -> ExportHighlightResult:
        """Export/download a highlight.

        Args:
            request: Export highlight request

        Returns:
            Download URL and expiration
        """
        try:
            # Get highlight
            highlight = await self.highlight_repo.get(request.highlight_id)
            if not highlight:
                return ExportHighlightResult(
                    status=ResultStatus.NOT_FOUND, errors=["Highlight not found"]
                )

            # Check access permission
            stream = await self.stream_repo.get(highlight.stream_id)
            if not stream or stream.user_id != request.user_id:
                return ExportHighlightResult(
                    status=ResultStatus.UNAUTHORIZED, errors=["Access denied"]
                )

            # Get user and organization info
            user = await self.user_repo.get(request.user_id)
            if not user or not user.organization_id:
                return ExportHighlightResult(
                    status=ResultStatus.FAILURE, 
                    errors=["User or organization not found"]
                )

            # Generate signed download URL
            from src.infrastructure.security.url_signer import url_signer
            from src.infrastructure.config import settings
            from datetime import datetime, timedelta

            # Determine expiration time based on request or default
            expiry_hours = 1  # Default 1 hour
            if request.format == "embed":
                expiry_hours = 24  # 24 hours for embeds
            
            # Generate the signed URL
            download_url = url_signer.generate_signed_url(
                base_url=settings.api_base_url,
                highlight_id=highlight.id,
                stream_id=stream.id,
                org_id=user.organization_id,
                expiry_hours=expiry_hours,
                additional_claims={"format": request.format, "quality": request.quality}
            )

            # Calculate expiration time
            expires_at = (datetime.utcnow() + timedelta(hours=expiry_hours)).isoformat()

            return ExportHighlightResult(
                status=ResultStatus.SUCCESS,
                download_url=download_url,
                expires_at=expires_at,
                message="Signed URL generated successfully",
            )

        except Exception as e:
            return ExportHighlightResult(
                status=ResultStatus.FAILURE,
                errors=[f"Failed to export highlight: {str(e)}"],
            )

    async def execute(self, request: GetHighlightRequest) -> GetHighlightResult:
        """Execute get highlight (default use case method).

        Args:
            request: Get highlight request

        Returns:
            Highlight result
        """
        return await self.get_highlight(request)
