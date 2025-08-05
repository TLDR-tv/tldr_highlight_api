"""Unit tests for email service."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart

from api.services.email_service import EmailService
from shared.infrastructure.config.config import Settings


class TestEmailService:
    """Test suite for EmailService."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock(spec=Settings)
        settings.email_enabled = True
        settings.email_host = "smtp.example.com"
        settings.email_port = 587
        settings.email_use_ssl = False
        settings.email_use_tls = True
        settings.email_username = "smtp_user"
        settings.email_password = "smtp_password"
        settings.email_from_name = "TLDR Highlights"
        settings.email_from_address = "noreply@example.com"
        settings.frontend_url = "https://app.example.com"
        settings.password_reset_url_path = "/reset-password"
        settings.password_reset_token_expiry_hours = 24
        return settings

    @pytest.fixture
    def email_service(self, mock_settings):
        """Create email service with mocked settings."""
        with patch("api.services.email_service.Environment") as mock_env:
            mock_jinja_env = Mock()
            mock_env.return_value = mock_jinja_env
            service = EmailService(mock_settings)
            service.jinja_env = mock_jinja_env
            return service

    @pytest.mark.asyncio
    async def test_send_email_success(self, email_service):
        """Test successful email sending."""
        # Arrange
        mock_template = Mock()
        mock_template.render.side_effect = ["HTML content", "Text content"]
        email_service.jinja_env.get_template.return_value = mock_template

        with patch.object(email_service, "_send_smtp_email") as mock_smtp:
            # Act
            result = await email_service.send_email(
                to_email="user@example.com",
                subject="Test Subject",
                template_name="test_template",
                template_data={"key": "value"},
                to_name="Test User",
            )

        # Assert
        assert result is True
        assert email_service.jinja_env.get_template.call_count == 2
        email_service.jinja_env.get_template.assert_any_call("test_template.html")
        email_service.jinja_env.get_template.assert_any_call("test_template.txt")
        mock_smtp.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_email_disabled(self, email_service, mock_settings):
        """Test email sending when disabled."""
        # Arrange
        mock_settings.email_enabled = False
        email_service.enabled = False

        # Act
        result = await email_service.send_email(
            to_email="user@example.com",
            subject="Test Subject",
            template_name="test_template",
            template_data={"key": "value"},
        )

        # Assert
        assert result is False
        email_service.jinja_env.get_template.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_email_template_error(self, email_service):
        """Test email sending with template rendering error."""
        # Arrange
        email_service.jinja_env.get_template.side_effect = Exception("Template not found")

        # Act
        result = await email_service.send_email(
            to_email="user@example.com",
            subject="Test Subject",
            template_name="missing_template",
            template_data={"key": "value"},
        )

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_send_email_smtp_error(self, email_service):
        """Test email sending with SMTP error."""
        # Arrange
        mock_template = Mock()
        mock_template.render.return_value = "Content"
        email_service.jinja_env.get_template.return_value = mock_template

        with patch.object(email_service, "_send_smtp_email") as mock_smtp:
            mock_smtp.side_effect = Exception("SMTP connection failed")

            # Act
            result = await email_service.send_email(
                to_email="user@example.com",
                subject="Test Subject",
                template_name="test_template",
                template_data={"key": "value"},
            )

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_send_email_without_recipient_name(self, email_service):
        """Test email sending without recipient name."""
        # Arrange
        mock_template = Mock()
        mock_template.render.return_value = "Content"
        email_service.jinja_env.get_template.return_value = mock_template

        with patch.object(email_service, "_send_smtp_email") as mock_smtp:
            # Act
            result = await email_service.send_email(
                to_email="user@example.com",
                subject="Test Subject",
                template_name="test_template",
                template_data={"key": "value"},
            )

        # Assert
        assert result is True
        mock_smtp.assert_called_once()
        # Check that the message was created properly
        call_args = mock_smtp.call_args[0]
        msg = call_args[1]
        assert msg["To"] == "user@example.com"

    def test_render_template_success(self, email_service):
        """Test successful template rendering."""
        # Arrange
        mock_template = Mock()
        mock_template.render.return_value = "Rendered content"
        email_service.jinja_env.get_template.return_value = mock_template

        # Act
        result = email_service._render_template("test.html", {"key": "value"})

        # Assert
        assert result == "Rendered content"
        mock_template.render.assert_called_once_with(key="value")

    def test_render_template_error_html(self, email_service):
        """Test template rendering error for HTML."""
        # Arrange
        email_service.jinja_env.get_template.side_effect = Exception("Template error")

        # Act
        result = email_service._render_template("test.html", {})

        # Assert
        assert result == "<p>Template rendering failed: Template error</p>"

    def test_render_template_error_text(self, email_service):
        """Test template rendering error for text."""
        # Arrange
        email_service.jinja_env.get_template.side_effect = Exception("Template error")

        # Act
        result = email_service._render_template("test.txt", {})

        # Assert
        assert result == "Template rendering failed: Template error"

    def test_send_smtp_email_with_ssl(self, email_service, mock_settings):
        """Test SMTP email sending with SSL."""
        # Arrange
        mock_settings.email_use_ssl = True
        message = MIMEMultipart()

        with patch("smtplib.SMTP_SSL") as mock_smtp_ssl:
            mock_server = MagicMock()
            mock_smtp_ssl.return_value.__enter__.return_value = mock_server

            # Act
            email_service._send_smtp_email("user@example.com", message)

        # Assert
        mock_smtp_ssl.assert_called_once_with("smtp.example.com", 587)
        mock_server.login.assert_called_once_with("smtp_user", "smtp_password")
        mock_server.send_message.assert_called_once_with(message)

    def test_send_smtp_email_with_tls(self, email_service, mock_settings):
        """Test SMTP email sending with TLS."""
        # Arrange
        mock_settings.email_use_ssl = False
        mock_settings.email_use_tls = True
        message = MIMEMultipart()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            # Act
            email_service._send_smtp_email("user@example.com", message)

        # Assert
        mock_smtp.assert_called_once_with("smtp.example.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("smtp_user", "smtp_password")
        mock_server.send_message.assert_called_once_with(message)

    def test_send_smtp_email_no_auth(self, email_service, mock_settings):
        """Test SMTP email sending without authentication."""
        # Arrange
        mock_settings.email_username = None
        mock_settings.email_password = None
        message = MIMEMultipart()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            # Act
            email_service._send_smtp_email("user@example.com", message)

        # Assert
        mock_server.login.assert_not_called()
        mock_server.send_message.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_send_password_reset_email(self, email_service):
        """Test sending password reset email."""
        # Arrange
        with patch.object(email_service, "send_email") as mock_send:
            mock_send.return_value = True

            # Act
            result = await email_service.send_password_reset_email(
                to_email="user@example.com",
                to_name="Test User",
                reset_token="reset_token_123",
            )

        # Assert
        assert result is True
        mock_send.assert_called_once()
        call_args = mock_send.call_args[1]
        assert call_args["to_email"] == "user@example.com"
        assert call_args["to_name"] == "Test User"
        assert call_args["subject"] == "Reset Your Password - TLDR Highlights"
        assert call_args["template_name"] == "password_reset"
        
        template_data = call_args["template_data"]
        assert template_data["user_name"] == "Test User"
        assert "reset_token_123" in template_data["reset_url"]
        assert template_data["expiry_hours"] == 24

    @pytest.mark.asyncio
    async def test_send_welcome_email(self, email_service):
        """Test sending welcome email."""
        # Arrange
        with patch.object(email_service, "send_email") as mock_send:
            mock_send.return_value = True

            # Act
            result = await email_service.send_welcome_email(
                to_email="user@example.com",
                to_name="Test User",
                organization_name="Test Org",
            )

        # Assert
        assert result is True
        mock_send.assert_called_once()
        call_args = mock_send.call_args[1]
        assert call_args["to_email"] == "user@example.com"
        assert call_args["to_name"] == "Test User"
        assert call_args["subject"] == "Welcome to TLDR Highlights - Test Org"
        assert call_args["template_name"] == "welcome"
        
        template_data = call_args["template_data"]
        assert template_data["user_name"] == "Test User"
        assert template_data["organization_name"] == "Test Org"
        assert template_data["login_url"] == "https://app.example.com/login"
        assert template_data["docs_url"] == "https://app.example.com/docs"

    def test_init_with_template_directory(self, mock_settings):
        """Test initialization with template directory setup."""
        # Arrange
        with patch("api.services.email_service.Environment") as mock_env_class:
            with patch("api.services.email_service.FileSystemLoader") as mock_loader_class:
                mock_jinja_env = Mock()
                mock_env_class.return_value = mock_jinja_env

                # Act
                service = EmailService(mock_settings)

                # Assert
                assert service.settings == mock_settings
                assert service.enabled is True
                assert service.jinja_env == mock_jinja_env
                
                # Check that FileSystemLoader was called with proper path
                mock_loader_class.assert_called_once()
                loader_path = mock_loader_class.call_args[0][0]
                assert "templates/emails" in loader_path
                
                # Check Environment was created with loader and autoescape
                mock_env_class.assert_called_once()
                env_kwargs = mock_env_class.call_args[1]
                assert env_kwargs["autoescape"] is True

    @pytest.mark.asyncio
    async def test_send_email_full_message_construction(self, email_service):
        """Test full email message construction."""
        # Arrange
        mock_template = Mock()
        mock_template.render.side_effect = ["<h1>HTML</h1>", "Plain text"]
        email_service.jinja_env.get_template.return_value = mock_template

        captured_message = None
        def capture_message(to_email, message):
            nonlocal captured_message
            captured_message = message

        with patch.object(email_service, "_send_smtp_email", side_effect=capture_message):
            # Act
            result = await email_service.send_email(
                to_email="user@example.com",
                subject="Test Subject",
                template_name="test",
                template_data={"data": "value"},
                to_name="Test User",
            )

        # Assert
        assert result is True
        assert captured_message is not None
        assert captured_message["Subject"] == "Test Subject"
        assert captured_message["From"] == "TLDR Highlights <noreply@example.com>"
        assert captured_message["To"] == "Test User <user@example.com>"
        
        # Check that both text and HTML parts were added
        parts = list(captured_message.walk())
        assert len(parts) == 3  # Multipart + text + html
        assert parts[1].get_content_type() == "text/plain"
        assert parts[2].get_content_type() == "text/html"