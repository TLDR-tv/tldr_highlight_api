"""Unit tests for database configuration."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from shared.infrastructure.database.database import Database, Base


class TestDatabase:
    """Test Database class."""

    @patch('shared.infrastructure.database.database.create_async_engine')
    @patch('shared.infrastructure.database.database.async_sessionmaker')
    def test_database_initialization_postgresql_url(self, mock_sessionmaker, mock_create_engine):
        """Test database initialization with PostgreSQL URL conversion."""
        # Mock engine
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="postgresql+asyncpg://user:pass@host:5432/db")
        mock_create_engine.return_value = mock_engine
        
        # Test URL conversion
        db = Database("postgresql://user:pass@host:5432/db")
        
        # Verify URL was converted before passing to create_async_engine
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0]
        assert "postgresql+asyncpg://" in call_args[0]

    @patch('shared.infrastructure.database.database.create_async_engine')
    @patch('shared.infrastructure.database.database.async_sessionmaker')
    def test_database_initialization_asyncpg_url(self, mock_sessionmaker, mock_create_engine):
        """Test database initialization with asyncpg URL (no conversion)."""
        # Mock engine
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Test no conversion needed
        url = "postgresql+asyncpg://user:pass@host:5432/db"
        db = Database(url)
        
        # Verify URL was passed unchanged
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0]
        assert call_args[0] == url

    @patch('shared.infrastructure.database.database.create_async_engine')
    @patch('shared.infrastructure.database.database.async_sessionmaker')
    def test_database_initialization_sqlite_url(self, mock_sessionmaker, mock_create_engine):
        """Test database initialization with SQLite URL (no conversion)."""
        # Mock engine
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Test other database types don't get converted
        url = "sqlite+aiosqlite:///test.db"
        db = Database(url)
        
        # Verify URL was passed unchanged
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args[0]
        assert call_args[0] == url

    @patch('shared.infrastructure.database.database.create_async_engine')
    @patch('shared.infrastructure.database.database.async_sessionmaker')
    @pytest.mark.asyncio
    async def test_session_context_manager_success(self, mock_sessionmaker, mock_create_engine):
        """Test successful session context manager."""
        # Mock engine
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Mock session maker and session
        mock_session = AsyncMock(spec=AsyncSession)
        mock_async_session = AsyncMock()
        mock_async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_async_session.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_sessionmaker.return_value = mock_async_session
        
        db = Database("postgresql://user:pass@host:5432/db")
        
        async with db.session() as session:
            # Session should be our mock
            assert session == mock_session
        
        # Verify session lifecycle
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.rollback.assert_not_called()

    @patch('shared.infrastructure.database.database.create_async_engine')
    @patch('shared.infrastructure.database.database.async_sessionmaker')
    @pytest.mark.asyncio
    async def test_session_context_manager_exception(self, mock_sessionmaker, mock_create_engine):
        """Test session context manager with exception."""
        # Mock engine
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Mock session maker and session
        mock_session = AsyncMock(spec=AsyncSession)
        mock_async_session = AsyncMock()
        mock_async_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_async_session.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_sessionmaker.return_value = mock_async_session
        
        db = Database("postgresql://user:pass@host:5432/db")
        
        # Test exception handling
        with pytest.raises(ValueError):
            async with db.session() as session:
                # Simulate an error
                raise ValueError("Test error")
        
        # Verify rollback was called, not commit
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.commit.assert_not_called()

    @patch('shared.infrastructure.database.database.create_async_engine')
    @patch('shared.infrastructure.database.database.async_sessionmaker')
    @pytest.mark.asyncio
    async def test_create_tables(self, mock_sessionmaker, mock_create_engine):
        """Test table creation."""
        # Mock engine and connection
        mock_conn = AsyncMock()
        mock_engine = Mock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_engine.return_value = mock_engine
        
        db = Database("postgresql://user:pass@host:5432/db")
        
        await db.create_tables()
        
        # Verify create_all was called
        mock_conn.run_sync.assert_called_once_with(Base.metadata.create_all)

    @patch('shared.infrastructure.database.database.create_async_engine')
    @patch('shared.infrastructure.database.database.async_sessionmaker')
    @pytest.mark.asyncio
    async def test_drop_tables(self, mock_sessionmaker, mock_create_engine):
        """Test table dropping."""
        # Mock engine and connection
        mock_conn = AsyncMock()
        mock_engine = Mock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_create_engine.return_value = mock_engine
        
        db = Database("postgresql://user:pass@host:5432/db")
        
        await db.drop_tables()
        
        # Verify drop_all was called
        mock_conn.run_sync.assert_called_once_with(Base.metadata.drop_all)

    @patch('shared.infrastructure.database.database.create_async_engine')
    @patch('shared.infrastructure.database.database.async_sessionmaker')
    @pytest.mark.asyncio
    async def test_close(self, mock_sessionmaker, mock_create_engine):
        """Test database connection closing."""
        # Mock engine
        mock_engine = Mock()
        mock_engine.dispose = AsyncMock()
        mock_create_engine.return_value = mock_engine
        
        db = Database("postgresql://user:pass@host:5432/db")
        
        await db.close()
        
        # Verify dispose was called
        mock_engine.dispose.assert_called_once()

    def test_base_class_exists(self):
        """Test that Base class is available."""
        # Verify Base is the SQLAlchemy declarative base
        assert hasattr(Base, 'metadata')
        assert hasattr(Base, 'registry')

    @patch('shared.infrastructure.database.database.create_async_engine')
    @patch('shared.infrastructure.database.database.async_sessionmaker')
    def test_database_engine_configuration(self, mock_sessionmaker, mock_create_engine):
        """Test database engine configuration."""
        # Mock engine
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Mock session maker
        mock_async_session = Mock()
        mock_sessionmaker.return_value = mock_async_session
        
        db = Database("postgresql://user:pass@host:5432/db")
        
        # Verify engine and session maker are set
        assert db.engine is mock_engine
        assert db.async_session is mock_async_session
        
        # Verify create_async_engine was called with proper parameters
        mock_create_engine.assert_called_once()
        call_kwargs = mock_create_engine.call_args[1]
        assert call_kwargs['echo'] is False
        assert call_kwargs['pool_pre_ping'] is True
        assert call_kwargs['pool_size'] == 10
        assert call_kwargs['max_overflow'] == 20

    @patch('shared.infrastructure.database.database.create_async_engine')  
    @patch('shared.infrastructure.database.database.async_sessionmaker')
    def test_session_maker_configuration(self, mock_sessionmaker, mock_create_engine):
        """Test session maker configuration."""
        # Mock engine
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Mock session maker
        mock_async_session = Mock()
        mock_sessionmaker.return_value = mock_async_session
        
        db = Database("postgresql://user:pass@host:5432/db")
        
        # Verify session maker was configured properly
        mock_sessionmaker.assert_called_once()
        call_args = mock_sessionmaker.call_args
        assert call_args[0][0] is mock_engine  # First argument is engine
        assert call_args[1]['class_'] == AsyncSession
        assert call_args[1]['expire_on_commit'] is False