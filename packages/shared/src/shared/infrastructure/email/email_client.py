"""Shared email client for SMTP operations."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import structlog

from shared.infrastructure.config.config import Settings

logger = structlog.get_logger()


class EmailClient:
    """Low-level SMTP email client."""

    def __init__(self, settings: Settings):
        """Initialize email client with settings."""
        self.settings = settings
        self.enabled = settings.email_enabled

    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str,
        to_name: Optional[str] = None,
    ) -> bool:
        """Send an email via SMTP.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML content of the email
            text_content: Plain text content of the email
            to_name: Recipient name (optional)

        Returns:
            True if email was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning(
                "Email sending disabled, skipping email",
                to_email=to_email,
                subject=subject,
            )
            return False

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.settings.email_from_name} <{self.settings.email_from_address}>"
            msg["To"] = f"{to_name} <{to_email}>" if to_name else to_email

            # Add text and HTML parts
            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")
            msg.attach(text_part)
            msg.attach(html_part)

            # Send email
            self._send_smtp_email(to_email, msg)

            logger.info(
                "Email sent successfully",
                to_email=to_email,
                subject=subject,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to send email",
                to_email=to_email,
                subject=subject,
                error=str(e),
            )
            return False

    def _send_smtp_email(self, to_email: str, message: MIMEMultipart) -> None:
        """Send email via SMTP.

        Args:
            to_email: Recipient email address
            message: Email message to send

        Raises:
            Exception: If email sending fails
        """
        smtp_class = smtplib.SMTP_SSL if self.settings.email_use_ssl else smtplib.SMTP
        
        with smtp_class(self.settings.email_host, self.settings.email_port) as server:
            if self.settings.email_use_tls and not self.settings.email_use_ssl:
                server.starttls()
            
            if self.settings.email_username and self.settings.email_password:
                server.login(self.settings.email_username, self.settings.email_password)
            
            server.send_message(message)