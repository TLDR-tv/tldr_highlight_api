"""Basic import tests for infrastructure modules."""

import pytest


class TestBasicImports:
    """Test basic imports to get coverage on module loading."""

    def test_import_storage_repositories(self):
        """Test importing storage repository modules."""
        # Import modules to trigger coverage
        from shared.infrastructure.storage.repositories import (
            api_key,
            highlight, 
            organization,
            stream,
            user,
            wake_word,
        )
        
        # Basic assertions to ensure imports worked
        assert hasattr(api_key, 'APIKeyRepository')
        assert hasattr(highlight, 'HighlightRepository')
        assert hasattr(organization, 'OrganizationRepository')
        assert hasattr(stream, 'StreamRepository')
        assert hasattr(user, 'UserRepository')
        assert hasattr(wake_word, 'WakeWordRepository')

    def test_import_database_module(self):
        """Test importing database module."""
        from shared.infrastructure.database import database
        
        assert hasattr(database, 'Database')
        assert hasattr(database, 'Base')

    def test_import_protocols(self):
        """Test importing protocols."""
        from shared.domain.protocols import protocols
        
        assert hasattr(protocols, 'Repository')
        assert hasattr(protocols, 'OrganizationRepository')
        assert hasattr(protocols, 'UserRepository')
        assert hasattr(protocols, 'StreamRepository')
        assert hasattr(protocols, 'HighlightRepository')
        assert hasattr(protocols, 'APIKeyRepository')
        assert hasattr(protocols, 'WakeWordRepository')
        assert hasattr(protocols, 'StorageService')
        assert hasattr(protocols, 'VideoAnalyzer')
        assert hasattr(protocols, 'StreamProcessor')
        assert hasattr(protocols, 'WebhookService')
        assert hasattr(protocols, 'AuthenticationService')

    def test_repository_classes_instantiable(self):
        """Test that repository classes can be imported and checked."""
        from unittest.mock import AsyncMock
        from sqlalchemy.ext.asyncio import AsyncSession
        
        from shared.infrastructure.storage.repositories.api_key import APIKeyRepository
        from shared.infrastructure.storage.repositories.highlight import HighlightRepository
        from shared.infrastructure.storage.repositories.organization import OrganizationRepository
        from shared.infrastructure.storage.repositories.stream import StreamRepository
        from shared.infrastructure.storage.repositories.user import UserRepository
        from shared.infrastructure.storage.repositories.wake_word import WakeWordRepository
        
        # Mock session
        mock_session = AsyncMock(spec=AsyncSession)
        
        # Test that classes can be instantiated
        api_key_repo = APIKeyRepository(mock_session)
        assert api_key_repo.session == mock_session
        
        highlight_repo = HighlightRepository(mock_session)
        assert highlight_repo.session == mock_session
        
        org_repo = OrganizationRepository(mock_session)
        assert org_repo.session == mock_session
        
        stream_repo = StreamRepository(mock_session)
        assert stream_repo.session == mock_session
        
        user_repo = UserRepository(mock_session)
        assert user_repo.session == mock_session
        
        wake_word_repo = WakeWordRepository(mock_session)
        assert wake_word_repo.session == mock_session

    def test_database_class_instantiable(self):
        """Test that Database class is instantiable."""
        from unittest.mock import patch
        
        with patch("shared.infrastructure.database.database.create_async_engine"), \
             patch("shared.infrastructure.database.database.async_sessionmaker"):
            
            from shared.infrastructure.database.database import Database
            
            db = Database("postgresql://test:test@localhost:5432/test")
            assert hasattr(db, 'engine')
            assert hasattr(db, 'async_session')
            assert hasattr(db, 'session')
            assert hasattr(db, 'create_tables')
            assert hasattr(db, 'drop_tables')
            assert hasattr(db, 'close')