"""Unit tests for email client."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from email.mime.multipart import MIMEMultipart
import smtplib

from shared.infrastructure.email.email_client import EmailClient
from shared.infrastructure.config.config import Settings


class TestEmailClient:
    """Test EmailClient implementation."""

    @pytest.fixture
    def email_settings(self):
        """Create email settings for testing."""
        return Settings(
            email_enabled=True,
            email_host="smtp.test.com",
            email_port=587,
            email_username="test@example.com",
            email_password="test_password",
            email_from_address="noreply@test.com",
            email_from_name="Test App",
            email_use_tls=True,
            email_use_ssl=False,
        )

    @pytest.fixture
    def disabled_email_settings(self):
        """Create disabled email settings for testing."""
        return Settings(email_enabled=False)

    @pytest.fixture
    def ssl_email_settings(self):
        """Create SSL email settings for testing."""
        return Settings(
            email_enabled=True,
            email_host="smtp.ssl.com",
            email_port=465,
            email_username="test@example.com",
            email_password="test_password",
            email_from_address="noreply@ssl.com",
            email_from_name="SSL App",
            email_use_tls=False,
            email_use_ssl=True,
        )

    @pytest.fixture
    def client(self, email_settings):
        """Create email client instance."""
        return EmailClient(email_settings)

    def test_email_client_initialization(self, email_settings):
        """Test email client initialization."""
        client = EmailClient(email_settings)
        
        assert client.settings == email_settings
        assert client.enabled is True

    def test_email_client_initialization_disabled(self, disabled_email_settings):
        """Test email client initialization when disabled."""
        client = EmailClient(disabled_email_settings)
        
        assert client.settings == disabled_email_settings
        assert client.enabled is False

    @patch('shared.infrastructure.email.email_client.logger')
    def test_send_email_disabled(self, mock_logger, disabled_email_settings):
        """Test sending email when email is disabled."""
        client = EmailClient(disabled_email_settings)
        
        result = client.send_email(
            to_email="user@example.com",
            subject="Test Subject",
            html_content="<p>HTML content</p>",
            text_content="Text content",
        )
        
        assert result is False
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "Email sending disabled" in call_args[0][0]

    @patch('shared.infrastructure.email.email_client.smtplib.SMTP')
    @patch('shared.infrastructure.email.email_client.logger')
    def test_send_email_success(self, mock_logger, mock_smtp_class, client):
        """Test successful email sending."""
        # Mock SMTP server
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_server
        
        result = client.send_email(
            to_email="user@example.com",
            subject="Test Subject",
            html_content="<p>HTML content</p>",
            text_content="Text content",
        )
        
        # Verify result
        assert result is True
        
        # Verify SMTP calls
        mock_smtp_class.assert_called_once_with("smtp.test.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("test@example.com", "test_password")
        mock_server.send_message.assert_called_once()
        
        # Verify logging
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Email sent successfully" in call_args[0][0]

    @patch('shared.infrastructure.email.email_client.smtplib.SMTP')
    @patch('shared.infrastructure.email.email_client.logger')
    def test_send_email_with_recipient_name(self, mock_logger, mock_smtp_class, client):
        """Test sending email with recipient name."""
        # Mock SMTP server
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_server
        
        result = client.send_email(
            to_email="user@example.com",
            subject="Test Subject",
            html_content="<p>HTML content</p>",
            text_content="Text content",
            to_name="John Doe",
        )
        
        assert result is True
        
        # Verify message was created with recipient name
        mock_server.send_message.assert_called_once()
        message = mock_server.send_message.call_args[0][0]
        assert "John Doe <user@example.com>" in message["To"]

    @patch('shared.infrastructure.email.email_client.smtplib.SMTP_SSL')
    @patch('shared.infrastructure.email.email_client.logger')
    def test_send_email_ssl(self, mock_logger, mock_smtp_ssl_class, ssl_email_settings):
        """Test sending email with SSL."""
        client = EmailClient(ssl_email_settings)
        
        # Mock SMTP server
        mock_server = MagicMock()
        mock_smtp_ssl_class.return_value.__enter__.return_value = mock_server
        
        result = client.send_email(
            to_email="user@example.com",
            subject="Test Subject",
            html_content="<p>HTML content</p>",
            text_content="Text content",
        )
        
        assert result is True
        
        # Verify SSL SMTP was used
        mock_smtp_ssl_class.assert_called_once_with("smtp.ssl.com", 465)
        # TLS should not be called with SSL
        mock_server.starttls.assert_not_called()
        mock_server.login.assert_called_once()

    @patch('shared.infrastructure.email.email_client.smtplib.SMTP')
    @patch('shared.infrastructure.email.email_client.logger')
    def test_send_email_no_auth(self, mock_logger, mock_smtp_class):
        """Test sending email without authentication."""
        # Settings without username/password
        settings = Settings(
            email_enabled=True,
            email_host="smtp.test.com",
            email_port=25,
            email_username=None,
            email_password=None,
            email_use_tls=False,
            email_use_ssl=False,
        )
        client = EmailClient(settings)
        
        # Mock SMTP server
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_server
        
        result = client.send_email(
            to_email="user@example.com",
            subject="Test Subject",
            html_content="<p>HTML content</p>",
            text_content="Text content",
        )
        
        assert result is True
        
        # Verify no authentication occurred
        mock_server.login.assert_not_called()
        mock_server.starttls.assert_not_called()

    @patch('shared.infrastructure.email.email_client.smtplib.SMTP')
    @patch('shared.infrastructure.email.email_client.logger')
    def test_send_email_smtp_exception(self, mock_logger, mock_smtp_class, client):
        """Test email sending with SMTP exception."""
        # Mock SMTP server to raise exception
        mock_smtp_class.return_value.__enter__.side_effect = smtplib.SMTPException("Connection failed")
        
        result = client.send_email(
            to_email="user@example.com",
            subject="Test Subject",
            html_content="<p>HTML content</p>",
            text_content="Text content",
        )
        
        # Verify failure
        assert result is False
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Failed to send email" in call_args[0][0]

    @patch('shared.infrastructure.email.email_client.smtplib.SMTP')
    def test_send_smtp_email_tls_and_ssl_false(self, mock_smtp_class):
        """Test SMTP email sending with both TLS and SSL false."""
        settings = Settings(
            email_enabled=True,
            email_host="smtp.test.com",
            email_port=25,
            email_use_tls=False,
            email_use_ssl=False,
        )
        client = EmailClient(settings)
        
        # Mock SMTP server
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_server
        
        # Create test message
        message = MIMEMultipart()
        message["To"] = "test@example.com"
        
        # Execute
        client._send_smtp_email("test@example.com", message)
        
        # Verify regular SMTP was used (not SSL)
        mock_smtp_class.assert_called_once_with("smtp.test.com", 25)
        # Neither TLS nor authentication should be called
        mock_server.starttls.assert_not_called()
        mock_server.send_message.assert_called_once_with(message)

    def test_message_formatting(self, client):
        """Test email message formatting."""
        # We'll test this by mocking the _send_smtp_email method
        with patch.object(client, '_send_smtp_email') as mock_send:
            client.send_email(
                to_email="user@example.com",
                subject="Test Subject",
                html_content="<p>HTML content</p>",
                text_content="Text content",
                to_name="John Doe",
            )
            
            # Verify _send_smtp_email was called
            mock_send.assert_called_once()
            
            # Get the message that was passed
            args = mock_send.call_args[0]
            to_email = args[0]
            message = args[1]
            
            assert to_email == "user@example.com"
            assert message["Subject"] == "Test Subject"
            assert "Test App <noreply@test.com>" in message["From"]
            assert "John Doe <user@example.com>" in message["To"]
            
            # Verify message has both HTML and text parts
            parts = message.get_payload()
            assert len(parts) == 2
            
            # Check text part
            text_part = parts[0]
            assert text_part.get_content_type() == "text/plain"
            assert text_part.get_payload() == "Text content"
            
            # Check HTML part
            html_part = parts[1]
            assert html_part.get_content_type() == "text/html"
            assert html_part.get_payload() == "<p>HTML content</p>"

    def test_message_formatting_no_recipient_name(self, client):
        """Test email message formatting without recipient name."""
        with patch.object(client, '_send_smtp_email') as mock_send:
            client.send_email(
                to_email="user@example.com",
                subject="Test Subject",
                html_content="<p>HTML content</p>",
                text_content="Text content",
            )
            
            # Get the message
            message = mock_send.call_args[0][1]
            
            # Should just have email address, no name
            assert message["To"] == "user@example.com"

    @patch('shared.infrastructure.email.email_client.smtplib.SMTP')
    def test_tls_only_when_not_ssl(self, mock_smtp_class):
        """Test TLS is only used when SSL is not enabled."""
        settings = Settings(
            email_enabled=True,
            email_host="smtp.test.com",
            email_port=587,
            email_use_tls=True,
            email_use_ssl=True,  # SSL takes precedence
        )
        client = EmailClient(settings)
        
        # This should use SSL, not regular SMTP with TLS
        with patch('shared.infrastructure.email.email_client.smtplib.SMTP_SSL') as mock_smtp_ssl_class:
            mock_server = MagicMock()
            mock_smtp_ssl_class.return_value.__enter__.return_value = mock_server
            
            message = MIMEMultipart()
            client._send_smtp_email("test@example.com", message)
            
            # Should use SSL class
            mock_smtp_ssl_class.assert_called_once()
            # Should not call regular SMTP
            mock_smtp_class.assert_not_called()
            # Should not call starttls since SSL is used
            mock_server.starttls.assert_not_called()