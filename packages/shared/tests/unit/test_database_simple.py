"""Simple unit tests for database module focusing on coverage."""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infrastructure.database.database import Database, Base


class TestDatabaseSimple:
    """Simple tests to cover database session context manager."""

    @pytest.mark.asyncio
    async def test_session_context_success_path(self):
        """Test session context manager success path."""
        # Mock everything to focus on the context manager logic
        mock_engine = AsyncMock()
        mock_session = AsyncMock(spec=AsyncSession)
        
        # Create a proper async context manager mock
        @asynccontextmanager
        async def mock_session_cm():
            yield mock_session
        
        mock_sessionmaker = Mock()
        mock_sessionmaker.return_value = mock_session_cm
        
        with patch("shared.infrastructure.database.database.create_async_engine", return_value=mock_engine), \
             patch("shared.infrastructure.database.database.async_sessionmaker", return_value=mock_sessionmaker):
            
            db = Database("postgresql://user:pass@localhost:5432/testdb")
            
            # This should trigger the context manager success path (lines 39-41)
            async with db.session() as session:
                assert session == mock_session
                # Do some work in the session
                pass
            
            # Verify session was committed and closed
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()
            mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio 
    async def test_session_context_exception_path(self):
        """Test session context manager exception handling path."""
        # Mock everything to focus on the context manager logic
        mock_engine = AsyncMock()
        mock_session = AsyncMock(spec=AsyncSession)
        
        # Create a proper async context manager mock
        @asynccontextmanager
        async def mock_session_cm():
            yield mock_session
        
        mock_sessionmaker = Mock()
        mock_sessionmaker.return_value = mock_session_cm
        
        with patch("shared.infrastructure.database.database.create_async_engine", return_value=mock_engine), \
             patch("shared.infrastructure.database.database.async_sessionmaker", return_value=mock_sessionmaker):
            
            db = Database("postgresql://user:pass@localhost:5432/testdb")
            
            # This should trigger the exception path (lines 42-44)
            test_exception = ValueError("Test database error")
            with pytest.raises(ValueError, match="Test database error"):
                async with db.session() as session:
                    assert session == mock_session
                    # Simulate an error occurring during session use
                    raise test_exception
            
            # Verify session was rolled back and closed, but not committed
            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()
            mock_session.commit.assert_not_called()