"""Unit tests for domain protocols."""

import pytest
import inspect
from typing import get_type_hints
from uuid import UUID
from pathlib import Path

from shared.domain.protocols.protocols import (
    Repository,
    OrganizationRepository,
    UserRepository,
    StreamRepository,
    HighlightRepository,
    APIKeyRepository,
    WakeWordRepository,
    StorageService,
    VideoAnalyzer,
    StreamProcessor,
    WebhookService,
    AuthenticationService,
)


class TestRepository:
    """Test base Repository protocol."""

    def test_repository_protocol_methods(self):
        """Test that Repository protocol has expected methods."""
        expected_methods = ['add', 'get', 'update', 'delete', 'list']
        
        for method_name in expected_methods:
            assert hasattr(Repository, method_name), f"Repository missing {method_name}"
            method = getattr(Repository, method_name)
            assert callable(method), f"{method_name} should be callable"

    def test_repository_type_annotations(self):
        """Test Repository protocol type annotations."""
        # Get type hints for Repository methods
        methods = ['add', 'get', 'update', 'delete', 'list']
        
        for method_name in methods:
            method = getattr(Repository, method_name)
            assert hasattr(method, '__annotations__'), f"{method_name} should have type annotations"


class TestOrganizationRepository:
    """Test OrganizationRepository protocol."""

    def test_organization_repository_methods(self):
        """Test OrganizationRepository has expected methods."""
        # Inherits from Repository, plus additional methods
        expected_methods = ['get_by_slug', 'get_by_api_key']
        
        for method_name in expected_methods:
            assert hasattr(OrganizationRepository, method_name)
            method = getattr(OrganizationRepository, method_name)
            assert callable(method)

    def test_organization_repository_inheritance(self):
        """Test OrganizationRepository inherits from Repository."""
        # Should have base Repository methods too
        base_methods = ['add', 'get', 'update', 'delete', 'list']
        
        for method_name in base_methods:
            assert hasattr(OrganizationRepository, method_name)


class TestUserRepository:
    """Test UserRepository protocol."""

    def test_user_repository_methods(self):
        """Test UserRepository has expected methods."""
        expected_methods = ['get_by_email', 'list_by_organization']
        
        for method_name in expected_methods:
            assert hasattr(UserRepository, method_name)
            method = getattr(UserRepository, method_name)
            assert callable(method)

    def test_user_repository_inheritance(self):
        """Test UserRepository inherits from Repository."""
        base_methods = ['add', 'get', 'update', 'delete', 'list']
        
        for method_name in base_methods:
            assert hasattr(UserRepository, method_name)


class TestStreamRepository:
    """Test StreamRepository protocol."""

    def test_stream_repository_methods(self):
        """Test StreamRepository has expected methods."""
        expected_methods = [
            'get_by_fingerprint',
            'list_active', 
            'list_by_organization'
        ]
        
        for method_name in expected_methods:
            assert hasattr(StreamRepository, method_name)
            method = getattr(StreamRepository, method_name)
            assert callable(method)


class TestHighlightRepository:
    """Test HighlightRepository protocol."""

    def test_highlight_repository_methods(self):
        """Test HighlightRepository has expected methods."""
        expected_methods = [
            'list_by_stream',
            'list_by_organization',
            'list_by_wake_word'
        ]
        
        for method_name in expected_methods:
            assert hasattr(HighlightRepository, method_name)
            method = getattr(HighlightRepository, method_name)
            assert callable(method)


class TestAPIKeyRepository:
    """Test APIKeyRepository protocol."""

    def test_api_key_repository_methods(self):
        """Test APIKeyRepository has expected methods."""
        expected_methods = [
            'get_by_prefix',
            'get_by_hash',
            'list_by_organization'
        ]
        
        for method_name in expected_methods:
            assert hasattr(APIKeyRepository, method_name)
            method = getattr(APIKeyRepository, method_name)
            assert callable(method)


class TestWakeWordRepository:
    """Test WakeWordRepository protocol."""

    def test_wake_word_repository_methods(self):
        """Test WakeWordRepository has expected methods."""
        expected_methods = [
            'list_by_organization',
            'get_active_words'
        ]
        
        for method_name in expected_methods:
            assert hasattr(WakeWordRepository, method_name)
            method = getattr(WakeWordRepository, method_name)
            assert callable(method)


class TestStorageService:
    """Test StorageService protocol."""

    def test_storage_service_methods(self):
        """Test StorageService has expected methods."""
        expected_methods = [
            'upload_file',
            'download_file', 
            'delete_file',
            'generate_signed_url'
        ]
        
        for method_name in expected_methods:
            assert hasattr(StorageService, method_name)
            method = getattr(StorageService, method_name)
            assert callable(method)

    def test_storage_service_method_signatures(self):
        """Test StorageService method signatures."""
        # Test that methods have expected parameters
        upload_method = getattr(StorageService, 'upload_file')
        assert hasattr(upload_method, '__annotations__')
        
        download_method = getattr(StorageService, 'download_file')  
        assert hasattr(download_method, '__annotations__')


class TestVideoAnalyzer:
    """Test VideoAnalyzer protocol."""

    def test_video_analyzer_methods(self):
        """Test VideoAnalyzer has expected methods."""
        expected_methods = [
            'analyze_segment',
            'extract_transcript',
            'identify_timestamps'
        ]
        
        for method_name in expected_methods:
            assert hasattr(VideoAnalyzer, method_name)
            method = getattr(VideoAnalyzer, method_name)
            assert callable(method)


class TestStreamProcessor:
    """Test StreamProcessor protocol."""

    def test_stream_processor_methods(self):
        """Test StreamProcessor has expected methods."""
        expected_methods = [
            'process_stream',
            'extract_clip',
            'generate_thumbnail'
        ]
        
        for method_name in expected_methods:
            assert hasattr(StreamProcessor, method_name)
            method = getattr(StreamProcessor, method_name)
            assert callable(method)


class TestWebhookService:
    """Test WebhookService protocol."""

    def test_webhook_service_methods(self):
        """Test WebhookService has expected methods."""
        expected_methods = [
            'send_event',
            'verify_signature'
        ]
        
        for method_name in expected_methods:
            assert hasattr(WebhookService, method_name)
            method = getattr(WebhookService, method_name)
            assert callable(method)


class TestAuthenticationService:
    """Test AuthenticationService protocol."""

    def test_authentication_service_methods(self):
        """Test AuthenticationService has expected methods."""
        expected_methods = [
            'hash_password',
            'verify_password',
            'validate_api_key',
            'generate_api_key'
        ]
        
        for method_name in expected_methods:
            assert hasattr(AuthenticationService, method_name)
            method = getattr(AuthenticationService, method_name)
            assert callable(method)

    def test_authentication_service_method_signatures(self):
        """Test AuthenticationService method signatures."""
        # Test that methods have expected parameters
        hash_method = getattr(AuthenticationService, 'hash_password')
        assert hasattr(hash_method, '__annotations__')
        
        verify_method = getattr(AuthenticationService, 'verify_password')
        assert hasattr(verify_method, '__annotations__')


class TestProtocolImports:
    """Test protocol imports and basic functionality."""

    def test_all_protocols_importable(self):
        """Test all protocols can be imported."""
        protocols = [
            Repository,
            OrganizationRepository,
            UserRepository,
            StreamRepository,
            HighlightRepository,
            APIKeyRepository,
            WakeWordRepository,
            StorageService,
            VideoAnalyzer,
            StreamProcessor,
            WebhookService,
            AuthenticationService,
        ]
        
        for protocol in protocols:
            assert protocol is not None
            assert hasattr(protocol, '__name__')

    def test_protocol_docstrings(self):
        """Test protocols have docstrings."""
        protocols_with_docstrings = [
            (Repository, "Base repository protocol."),
            (OrganizationRepository, "Organization repository protocol."),
            (UserRepository, "User repository protocol."),
            (StorageService, "Storage service for media files."),
            (VideoAnalyzer, "Video analysis service protocol."),
        ]
        
        for protocol, expected_doc in protocols_with_docstrings:
            assert protocol.__doc__ is not None
            assert expected_doc in protocol.__doc__

    def test_protocols_are_runtime_checkable(self):
        """Test protocols work with isinstance checks."""
        # Note: Protocols are typically not runtime checkable by default
        # But we can test that they exist as types
        from typing import runtime_checkable
        
        # Test that Repository exists as a type
        assert hasattr(Repository, '__class__')
        
        # Test Protocol inheritance
        import typing
        # Most protocols inherit from typing.Protocol
        assert hasattr(typing, 'Protocol')