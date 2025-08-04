"""Celery tasks for email delivery."""

from typing import Dict, Any, Optional
from uuid import UUID
from celery import shared_task
from celery.utils.log import get_task_logger

from shared.infrastructure.config.config import get_settings
from shared.infrastructure.email import EmailClient, EmailTemplates
from shared.infrastructure.database.database import Database
from shared.infrastructure.storage.repositories import UserRepository

logger = get_task_logger(__name__)


@shared_task(
    bind=True,
    name="worker.tasks.email_delivery.send_email",
    max_retries=3,
    default_retry_delay=60,  # 1 minute
)
def send_email(
    self,
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str,
    to_name: Optional[str] = None,
) -> bool:
    """Send an email asynchronously.

    Args:
        to_email: Recipient email address
        subject: Email subject
        html_content: HTML content of the email
        text_content: Plain text content of the email
        to_name: Recipient name (optional)

    Returns:
        True if email was sent successfully
    """
    try:
        settings = get_settings()
        email_client = EmailClient(settings)
        
        result = email_client.send_email(
            to_email=to_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            to_name=to_name,
        )
        
        if not result:
            # Retry if email sending failed
            raise self.retry(exc=Exception("Email sending failed"))
        
        return result
        
    except Exception as exc:
        logger.error(
            f"Email sending failed: {str(exc)}",
            extra={
                "to_email": to_email,
                "subject": subject,
            }
        )
        raise self.retry(exc=exc)


@shared_task(
    bind=True,
    name="worker.tasks.email_delivery.send_password_reset_email",
    max_retries=3,
    default_retry_delay=60,
)
def send_password_reset_email(
    self,
    user_id: str,
    reset_token: str,
) -> bool:
    """Send password reset email to user.

    Args:
        user_id: User ID
        reset_token: Password reset token

    Returns:
        True if email was sent successfully
    """
    try:
        settings = get_settings()
        
        # Get user details
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_user_details():
            db = Database(settings.database_url)
            async with db.session() as session:
                user_repo = UserRepository(session)
                user = await user_repo.get(UUID(user_id))
                return user
        
        user = loop.run_until_complete(get_user_details())
        
        if not user:
            logger.error(f"User not found: {user_id}")
            return False
        
        # Generate reset URL
        reset_url = (
            f"{settings.frontend_url}{settings.password_reset_url_path}"
            f"?token={reset_token}"
        )
        
        # Prepare template data
        template_data = {
            "user_name": user.name,
            "reset_url": reset_url,
            "expiry_hours": settings.password_reset_token_expiry_hours,
            "support_email": settings.email_from_address,
        }
        
        # Render templates
        html_content, text_content = EmailTemplates.render_password_reset(template_data)
        
        # Send email
        email_client = EmailClient(settings)
        result = email_client.send_email(
            to_email=user.email,
            to_name=user.name,
            subject="Reset Your Password - TLDR Highlights",
            html_content=html_content,
            text_content=text_content,
        )
        
        if not result:
            raise self.retry(exc=Exception("Password reset email sending failed"))
        
        logger.info(f"Password reset email sent to user {user_id}")
        return result
        
    except Exception as exc:
        logger.error(
            f"Failed to send password reset email: {str(exc)}",
            extra={"user_id": user_id}
        )
        raise self.retry(exc=exc)


@shared_task(
    bind=True,
    name="worker.tasks.email_delivery.send_welcome_email",
    max_retries=3,
    default_retry_delay=60,
)
def send_welcome_email(
    self,
    user_id: str,
    organization_name: str,
) -> bool:
    """Send welcome email to new user.

    Args:
        user_id: User ID
        organization_name: Name of the organization

    Returns:
        True if email was sent successfully
    """
    try:
        settings = get_settings()
        
        # Get user details
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_user_details():
            db = Database(settings.database_url)
            async with db.session() as session:
                user_repo = UserRepository(session)
                user = await user_repo.get(UUID(user_id))
                return user
        
        user = loop.run_until_complete(get_user_details())
        
        if not user:
            logger.error(f"User not found: {user_id}")
            return False
        
        # Prepare template data
        template_data = {
            "user_name": user.name,
            "organization_name": organization_name,
            "login_url": f"{settings.frontend_url}/login",
            "docs_url": f"{settings.frontend_url}/docs",
            "support_email": settings.email_from_address,
        }
        
        # Render templates
        html_content, text_content = EmailTemplates.render_welcome(template_data)
        
        # Send email
        email_client = EmailClient(settings)
        result = email_client.send_email(
            to_email=user.email,
            to_name=user.name,
            subject=f"Welcome to TLDR Highlights - {organization_name}",
            html_content=html_content,
            text_content=text_content,
        )
        
        if not result:
            raise self.retry(exc=Exception("Welcome email sending failed"))
        
        logger.info(f"Welcome email sent to user {user_id}")
        return result
        
    except Exception as exc:
        logger.error(
            f"Failed to send welcome email: {str(exc)}",
            extra={"user_id": user_id, "organization_name": organization_name}
        )
        raise self.retry(exc=exc)