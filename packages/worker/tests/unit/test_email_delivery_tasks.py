"""Tests for email delivery tasks."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from worker.tasks.email_delivery import (
    send_email,
    send_password_reset_email
)
from shared.domain.models.user import User


class TestSendEmail:
    """Test the send_email task."""

    @pytest.fixture
    def mock_task(self):
        """Create a mock task instance."""
        task = Mock()
        task.retry = Mock(side_effect=Exception("Retry called"))
        return task

    @patch('worker.tasks.email_delivery.get_settings')
    @patch('worker.tasks.email_delivery.EmailClient')
    def test_send_email_success(self, mock_email_client_class, mock_get_settings, mock_task):
        """Test successful email sending."""
        # Setup mocks
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        
        mock_email_client = Mock()
        mock_email_client.send_email.return_value = True
        mock_email_client_class.return_value = mock_email_client
        
        # Test data
        to_email = "user@example.com"
        subject = "Test Subject"
        html_content = "<h1>Test HTML</h1>"
        text_content = "Test Plain Text"
        to_name = "Test User"
        
        # Test
        result = send_email(
            mock_task,
            to_email,
            subject, 
            html_content,
            text_content,
            to_name
        )
        
        # Verify
        assert result is True
        mock_get_settings.assert_called_once()
        mock_email_client_class.assert_called_once_with(mock_settings)
        mock_email_client.send_email.assert_called_once_with(
            to_email=to_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            to_name=to_name
        )

    @patch('worker.tasks.email_delivery.get_settings')
    @patch('worker.tasks.email_delivery.EmailClient')
    def test_send_email_without_name(self, mock_email_client_class, mock_get_settings, mock_task):
        """Test email sending without recipient name."""
        # Setup mocks
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        
        mock_email_client = Mock()
        mock_email_client.send_email.return_value = True
        mock_email_client_class.return_value = mock_email_client
        
        # Test data
        to_email = "user@example.com"
        subject = "Test Subject"
        html_content = "<h1>Test HTML</h1>"
        text_content = "Test Plain Text"
        
        # Test
        result = send_email(
            mock_task,
            to_email,
            subject,
            html_content,
            text_content
        )
        
        # Verify
        assert result is True
        mock_email_client.send_email.assert_called_once_with(
            to_email=to_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            to_name=None
        )

    @patch('worker.tasks.email_delivery.get_settings')
    @patch('worker.tasks.email_delivery.EmailClient')
    @patch('worker.tasks.email_delivery.logger')
    def test_send_email_client_returns_false(self, mock_logger, mock_email_client_class, 
                                             mock_get_settings, mock_task):
        """Test email sending when client returns False."""
        # Setup mocks
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        
        mock_email_client = Mock()
        mock_email_client.send_email.return_value = False
        mock_email_client_class.return_value = mock_email_client
        
        # Test data
        to_email = "user@example.com"
        subject = "Test Subject"
        html_content = "<h1>Test HTML</h1>"
        text_content = "Test Plain Text"
        
        # Test
        with pytest.raises(Exception, match="Retry called"):
            send_email(
                mock_task,
                to_email,
                subject,
                html_content,
                text_content
            )
        
        # Verify retry was called
        mock_task.retry.assert_called_once()
        retry_args = mock_task.retry.call_args[1]
        assert "Email sending failed" in str(retry_args["exc"])

    @patch('worker.tasks.email_delivery.get_settings')
    @patch('worker.tasks.email_delivery.EmailClient')
    @patch('worker.tasks.email_delivery.logger')
    def test_send_email_exception(self, mock_logger, mock_email_client_class, 
                                  mock_get_settings, mock_task):
        """Test email sending with exception."""
        # Setup mocks
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        
        mock_email_client = Mock()
        test_exception = Exception("SMTP connection failed")
        mock_email_client.send_email.side_effect = test_exception
        mock_email_client_class.return_value = mock_email_client
        
        # Test data
        to_email = "user@example.com"
        subject = "Test Subject"
        html_content = "<h1>Test HTML</h1>"
        text_content = "Test Plain Text"
        
        # Test
        with pytest.raises(Exception, match="Retry called"):
            send_email(
                mock_task,
                to_email,
                subject,
                html_content,
                text_content
            )
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        log_call = mock_logger.error.call_args
        assert "Email sending failed" in log_call[0][0]
        assert log_call[1]["extra"]["to_email"] == to_email
        assert log_call[1]["extra"]["subject"] == subject
        
        # Verify retry
        mock_task.retry.assert_called_once()
        retry_args = mock_task.retry.call_args[1]
        assert retry_args["exc"] == test_exception


class TestSendPasswordResetEmail:
    """Test the send_password_reset_email task."""

    @pytest.fixture
    def mock_user(self):
        """Create a mock user."""
        user = Mock(spec=User)
        user.id = uuid4()
        user.email = "user@example.com"
        user.full_name = "Test User"
        return user

    @pytest.fixture
    def mock_task(self):
        """Create a mock task instance."""
        task = Mock()
        task.retry = Mock(side_effect=Exception("Retry called"))
        return task

    @patch('worker.tasks.email_delivery.get_settings')
    @patch('worker.tasks.email_delivery.Database')
    @patch('worker.tasks.email_delivery.UserRepository')
    @patch('worker.tasks.email_delivery.EmailClient')
    @patch('worker.tasks.email_delivery.EmailTemplates')
    @patch('worker.tasks.email_delivery.asyncio')
    def test_send_password_reset_email_success(self, mock_asyncio, mock_email_templates,
                                               mock_email_client_class, mock_user_repo_class,
                                               mock_database_class, mock_get_settings,
                                               mock_task, mock_user):
        """Test successful password reset email sending."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.database_url = "sqlite:///test.db"
        mock_get_settings.return_value = mock_settings
        
        # Setup database and repository mocks
        mock_database = Mock()
        mock_database_class.return_value = mock_database
        
        mock_session = AsyncMock()
        mock_database.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_database.session.return_value.__aexit__ = AsyncMock()
        
        mock_user_repo = AsyncMock()
        mock_user_repo.get.return_value = mock_user
        mock_user_repo_class.return_value = mock_user_repo
        
        # Setup email client
        mock_email_client = Mock()
        mock_email_client.send_email.return_value = True
        mock_email_client_class.return_value = mock_email_client
        
        # Setup email templates
        mock_templates = Mock()
        mock_templates.password_reset.return_value = {
            "subject": "Reset Your Password",
            "html_content": "<h1>Reset Password</h1>",
            "text_content": "Reset your password"
        }
        mock_email_templates.return_value = mock_templates
        
        # Setup asyncio
        mock_loop = Mock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_asyncio.set_event_loop = Mock()
        
        # Mock the async function execution
        async def mock_get_user_details():
            return mock_user
        
        mock_loop.run_until_complete.return_value = mock_user
        
        # Test data
        user_id = str(mock_user.id)
        reset_token = "test_reset_token_123"
        
        # Test
        result = send_password_reset_email(mock_task, user_id, reset_token)
        
        # Verify
        assert result is True
        mock_get_settings.assert_called_once()
        mock_asyncio.new_event_loop.assert_called_once()
        mock_asyncio.set_event_loop.assert_called_once_with(mock_loop)

    @patch('worker.tasks.email_delivery.get_settings')
    @patch('worker.tasks.email_delivery.Database')
    @patch('worker.tasks.email_delivery.UserRepository')
    @patch('worker.tasks.email_delivery.asyncio')
    @patch('worker.tasks.email_delivery.logger')
    def test_send_password_reset_email_user_not_found(self, mock_logger, mock_asyncio,
                                                       mock_user_repo_class, mock_database_class,
                                                       mock_get_settings, mock_task):
        """Test password reset email when user not found."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.database_url = "sqlite:///test.db"
        mock_get_settings.return_value = mock_settings
        
        # Setup database and repository mocks
        mock_database = Mock()
        mock_database_class.return_value = mock_database
        
        mock_session = AsyncMock()
        mock_database.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_database.session.return_value.__aexit__ = AsyncMock()
        
        mock_user_repo = AsyncMock()
        mock_user_repo.get.return_value = None  # User not found
        mock_user_repo_class.return_value = mock_user_repo
        
        # Setup asyncio
        mock_loop = Mock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_asyncio.set_event_loop = Mock()
        mock_loop.run_until_complete.return_value = None
        
        # Test data
        user_id = str(uuid4())
        reset_token = "test_reset_token_123"
        
        # Test
        with pytest.raises(Exception, match="Retry called"):
            send_password_reset_email(mock_task, user_id, reset_token)
        
        # Verify error handling
        mock_task.retry.assert_called_once()

    @patch('worker.tasks.email_delivery.get_settings')
    @patch('worker.tasks.email_delivery.asyncio')
    @patch('worker.tasks.email_delivery.logger')
    def test_send_password_reset_email_database_error(self, mock_logger, mock_asyncio,
                                                      mock_get_settings, mock_task):
        """Test password reset email with database error."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.database_url = "sqlite:///test.db"
        mock_get_settings.return_value = mock_settings
        
        # Setup asyncio to raise exception
        test_exception = Exception("Database connection failed")
        mock_loop = Mock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_asyncio.set_event_loop = Mock()
        mock_loop.run_until_complete.side_effect = test_exception
        
        # Test data
        user_id = str(uuid4())
        reset_token = "test_reset_token_123"
        
        # Test
        with pytest.raises(Exception, match="Retry called"):
            send_password_reset_email(mock_task, user_id, reset_token)
        
        # Verify error logging and retry
        mock_logger.error.assert_called_once()
        mock_task.retry.assert_called_once()
        retry_args = mock_task.retry.call_args[1]
        assert retry_args["exc"] == test_exception

    @patch('worker.tasks.email_delivery.get_settings')
    @patch('worker.tasks.email_delivery.Database')
    @patch('worker.tasks.email_delivery.UserRepository')
    @patch('worker.tasks.email_delivery.EmailClient')
    @patch('worker.tasks.email_delivery.EmailTemplates')
    @patch('worker.tasks.email_delivery.asyncio')
    @patch('worker.tasks.email_delivery.logger')
    def test_send_password_reset_email_send_failure(self, mock_logger, mock_asyncio,
                                                     mock_email_templates, mock_email_client_class,
                                                     mock_user_repo_class, mock_database_class,
                                                     mock_get_settings, mock_task, mock_user):
        """Test password reset email when email sending fails."""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.database_url = "sqlite:///test.db"
        mock_get_settings.return_value = mock_settings
        
        # Setup database and repository mocks
        mock_database = Mock()
        mock_database_class.return_value = mock_database
        
        mock_session = AsyncMock()
        mock_database.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_database.session.return_value.__aexit__ = AsyncMock()
        
        mock_user_repo = AsyncMock()
        mock_user_repo.get.return_value = mock_user
        mock_user_repo_class.return_value = mock_user_repo
        
        # Setup email client to fail
        mock_email_client = Mock()
        mock_email_client.send_email.return_value = False
        mock_email_client_class.return_value = mock_email_client
        
        # Setup email templates
        mock_templates = Mock()
        mock_templates.password_reset.return_value = {
            "subject": "Reset Your Password",
            "html_content": "<h1>Reset Password</h1>",
            "text_content": "Reset your password"
        }
        mock_email_templates.return_value = mock_templates
        
        # Setup asyncio
        mock_loop = Mock()
        mock_asyncio.new_event_loop.return_value = mock_loop
        mock_asyncio.set_event_loop = Mock()
        mock_loop.run_until_complete.return_value = mock_user
        
        # Test data
        user_id = str(mock_user.id)
        reset_token = "test_reset_token_123"
        
        # Test
        with pytest.raises(Exception, match="Retry called"):
            send_password_reset_email(mock_task, user_id, reset_token)
        
        # Verify retry was called
        mock_task.retry.assert_called_once()

    def test_send_password_reset_email_with_invalid_uuid(self, mock_task):
        """Test password reset email with invalid UUID format."""
        invalid_user_id = "not-a-uuid"
        reset_token = "test_reset_token_123"
        
        # This should raise an exception during UUID conversion
        with pytest.raises(Exception, match="Retry called"):
            send_password_reset_email(mock_task, invalid_user_id, reset_token)