"""Unit tests for email templates."""

import pytest
from shared.infrastructure.email.email_templates import EmailTemplates


class TestEmailTemplates:
    """Test EmailTemplates implementation."""

    @pytest.fixture
    def password_reset_data(self):
        """Sample data for password reset email."""
        return {
            "user_name": "John Doe",
            "reset_url": "https://example.com/reset?token=abc123",
            "expiry_hours": 24,
            "support_email": "support@example.com",
        }

    @pytest.fixture
    def welcome_data(self):
        """Sample data for welcome email."""
        return {
            "user_name": "Jane Smith",
            "organization_name": "Acme Corp",
            "login_url": "https://example.com/login",
            "docs_url": "https://example.com/docs",
            "support_email": "support@example.com",
        }

    def test_password_reset_template_constants(self):
        """Test that password reset template constants are defined."""
        assert hasattr(EmailTemplates, 'PASSWORD_RESET_HTML')
        assert hasattr(EmailTemplates, 'PASSWORD_RESET_TEXT')
        
        # Verify they are Template objects
        from string import Template
        assert isinstance(EmailTemplates.PASSWORD_RESET_HTML, Template)
        assert isinstance(EmailTemplates.PASSWORD_RESET_TEXT, Template)

    def test_welcome_template_constants(self):
        """Test that welcome template constants are defined."""
        assert hasattr(EmailTemplates, 'WELCOME_HTML')
        assert hasattr(EmailTemplates, 'WELCOME_TEXT')
        
        # Verify they are Template objects
        from string import Template
        assert isinstance(EmailTemplates.WELCOME_HTML, Template)
        assert isinstance(EmailTemplates.WELCOME_TEXT, Template)

    def test_render_password_reset(self, password_reset_data):
        """Test rendering password reset email templates."""
        html_content, text_content = EmailTemplates.render_password_reset(password_reset_data)
        
        # Verify both contents are strings
        assert isinstance(html_content, str)
        assert isinstance(text_content, str)
        
        # Verify HTML content contains expected elements
        assert "John Doe" in html_content
        assert "https://example.com/reset?token=abc123" in html_content
        assert "24 hours" in html_content
        assert "support@example.com" in html_content
        assert "<!DOCTYPE html>" in html_content
        assert "<html>" in html_content
        assert "TLDR Highlights" in html_content
        
        # Verify text content contains expected elements
        assert "John Doe" in text_content
        assert "https://example.com/reset?token=abc123" in text_content
        assert "24 hours" in text_content
        assert "support@example.com" in text_content
        assert "TLDR Highlights" in text_content

    def test_render_welcome(self, welcome_data):
        """Test rendering welcome email templates."""
        html_content, text_content = EmailTemplates.render_welcome(welcome_data)
        
        # Verify both contents are strings
        assert isinstance(html_content, str)
        assert isinstance(text_content, str)
        
        # Verify HTML content contains expected elements
        assert "Jane Smith" in html_content
        assert "Acme Corp" in html_content
        assert "https://example.com/login" in html_content
        assert "https://example.com/docs" in html_content
        assert "support@example.com" in html_content
        assert "<!DOCTYPE html>" in html_content
        assert "<html>" in html_content
        assert "Welcome to TLDR Highlights" in html_content
        
        # Verify text content contains expected elements
        assert "Jane Smith" in text_content
        assert "Acme Corp" in text_content
        assert "https://example.com/login" in text_content
        assert "https://example.com/docs" in text_content
        assert "support@example.com" in text_content
        assert "TLDR Highlights" in text_content

    def test_password_reset_html_structure(self, password_reset_data):
        """Test password reset HTML template structure."""
        html_content, _ = EmailTemplates.render_password_reset(password_reset_data)
        
        # Verify HTML structure (strip leading whitespace)
        assert html_content.strip().startswith("<!DOCTYPE html>")
        assert "<head>" in html_content
        assert "<body>" in html_content
        assert "<style>" in html_content
        assert "</html>" in html_content
        
        # Verify specific elements
        assert "Reset Password" in html_content
        assert 'class="container"' in html_content
        assert 'class="header"' in html_content
        assert 'class="content"' in html_content
        assert 'class="button"' in html_content
        assert 'class="footer"' in html_content

    def test_welcome_html_structure(self, welcome_data):
        """Test welcome HTML template structure."""
        html_content, _ = EmailTemplates.render_welcome(welcome_data)
        
        # Verify HTML structure (strip leading whitespace)
        assert html_content.strip().startswith("<!DOCTYPE html>")
        assert "<head>" in html_content
        assert "<body>" in html_content
        assert "<style>" in html_content
        assert "</html>" in html_content
        
        # Verify specific elements
        assert "Welcome to TLDR Highlights" in html_content
        assert 'class="container"' in html_content
        assert 'class="header"' in html_content
        assert 'class="content"' in html_content
        assert 'class="button"' in html_content
        assert 'class="footer"' in html_content
        assert 'class="feature"' in html_content

    def test_password_reset_text_structure(self, password_reset_data):
        """Test password reset text template structure."""
        _, text_content = EmailTemplates.render_password_reset(password_reset_data)
        
        # Verify text structure (no HTML tags)
        assert "<!DOCTYPE" not in text_content
        assert "<html>" not in text_content
        assert "<div>" not in text_content
        
        # Verify content structure
        assert "Hello John Doe," in text_content
        assert "We received a request" in text_content
        assert "To reset your password" in text_content
        assert "This link will expire" in text_content
        assert "Best regards," in text_content
        assert "The TLDR Highlights Team" in text_content

    def test_welcome_text_structure(self, welcome_data):
        """Test welcome text template structure."""
        _, text_content = EmailTemplates.render_welcome(welcome_data)
        
        # Verify text structure (no HTML tags)
        assert "<!DOCTYPE" not in text_content
        assert "<html>" not in text_content
        assert "<div>" not in text_content
        
        # Verify content structure
        assert "Hello Jane Smith," in text_content
        assert "Welcome to Acme Corp" in text_content
        assert "Your account has been successfully created" in text_content
        assert "• Extract highlights" in text_content
        assert "• Use AI-powered detection" in text_content
        assert "• Integrate with your existing workflow" in text_content
        assert "Best regards," in text_content
        assert "The TLDR Highlights Team" in text_content

    def test_render_password_reset_with_missing_data(self):
        """Test rendering password reset with missing template variables."""
        incomplete_data = {
            "user_name": "John Doe",
            "reset_url": "https://example.com/reset",
            # Missing expiry_hours and support_email
        }
        
        # Should use safe_substitute, so missing variables become empty strings or remain as placeholders
        html_content, text_content = EmailTemplates.render_password_reset(incomplete_data)
        
        # Should still contain the provided data
        assert "John Doe" in html_content
        assert "John Doe" in text_content
        assert "https://example.com/reset" in html_content
        assert "https://example.com/reset" in text_content
        
        # Missing variables should be handled gracefully (either empty or as placeholders)
        assert isinstance(html_content, str)
        assert isinstance(text_content, str)
        assert len(html_content) > 0
        assert len(text_content) > 0

    def test_render_welcome_with_missing_data(self):
        """Test rendering welcome with missing template variables."""
        incomplete_data = {
            "user_name": "Jane Smith",
            "organization_name": "Acme Corp",
            # Missing other fields
        }
        
        # Should use safe_substitute, so missing variables are handled gracefully
        html_content, text_content = EmailTemplates.render_welcome(incomplete_data)
        
        # Should still contain the provided data
        assert "Jane Smith" in html_content
        assert "Jane Smith" in text_content
        assert "Acme Corp" in html_content
        assert "Acme Corp" in text_content
        
        # Should handle missing variables gracefully
        assert isinstance(html_content, str)
        assert isinstance(text_content, str)
        assert len(html_content) > 0
        assert len(text_content) > 0

    def test_render_methods_are_static(self):
        """Test that render methods are static methods."""
        # Should be able to call without instantiating the class
        data = {
            "user_name": "Test User",
            "reset_url": "https://test.com",
            "expiry_hours": 1,
            "support_email": "test@test.com",
        }
        
        # This should work without creating an instance
        html, text = EmailTemplates.render_password_reset(data)
        assert isinstance(html, str)
        assert isinstance(text, str)
        
        # Same for welcome
        welcome_data = {
            "user_name": "Test User",
            "organization_name": "Test Org",
            "login_url": "https://test.com/login",
            "docs_url": "https://test.com/docs",
            "support_email": "test@test.com",
        }
        
        html, text = EmailTemplates.render_welcome(welcome_data)
        assert isinstance(html, str)
        assert isinstance(text, str)

    def test_template_consistency(self, password_reset_data, welcome_data):
        """Test that HTML and text templates contain consistent information."""
        # Test password reset templates
        html_content, text_content = EmailTemplates.render_password_reset(password_reset_data)
        
        # Key information should be in both versions
        for key_info in ["John Doe", "reset", "24 hours", "support@example.com"]:
            assert key_info in html_content
            assert key_info in text_content
        
        # Test welcome templates
        html_content, text_content = EmailTemplates.render_welcome(welcome_data)
        
        # Key information should be in both versions
        for key_info in ["Jane Smith", "Acme Corp", "login", "docs", "support@example.com"]:
            assert key_info in html_content
            assert key_info in text_content