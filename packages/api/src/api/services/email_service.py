"""Email service for sending transactional emails."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List, Dict, Any
from pathlib import Path
import structlog
from jinja2 import Template, Environment, FileSystemLoader

from shared.infrastructure.config.config import Settings

logger = structlog.get_logger()


class EmailService:
    """Service for sending emails via SMTP."""

    def __init__(self, settings: Settings):
        """Initialize email service with settings."""
        self.settings = settings
        self.enabled = settings.email_enabled
        
        # Set up Jinja2 for email templates
        template_dir = Path(__file__).parent.parent / "templates" / "emails"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )

    async def send_email(
        self,
        to_email: str,
        subject: str,
        template_name: str,
        template_data: Dict[str, Any],
        to_name: Optional[str] = None,
    ) -> bool:
        """Send an email using a template.

        Args:
            to_email: Recipient email address
            subject: Email subject
            template_name: Name of the template file (without extension)
            template_data: Data to pass to the template
            to_name: Recipient name (optional)

        Returns:
            True if email was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning(
                "Email sending disabled, skipping email",
                to_email=to_email,
                subject=subject,
                template_name=template_name,
            )
            return False

        try:
            # Render email templates
            html_content = self._render_template(f"{template_name}.html", template_data)
            text_content = self._render_template(f"{template_name}.txt", template_data)

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
                template_name=template_name,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to send email",
                to_email=to_email,
                subject=subject,
                template_name=template_name,
                error=str(e),
            )
            return False

    def _render_template(self, template_path: str, data: Dict[str, Any]) -> str:
        """Render an email template with data.

        Args:
            template_path: Path to template file
            data: Data to pass to template

        Returns:
            Rendered template string
        """
        try:
            template = self.jinja_env.get_template(template_path)
            return template.render(**data)
        except Exception as e:
            logger.error(
                "Failed to render email template",
                template_path=template_path,
                error=str(e),
            )
            # Fall back to simple text if template fails
            if template_path.endswith(".txt"):
                return f"Template rendering failed: {str(e)}"
            else:
                return f"<p>Template rendering failed: {str(e)}</p>"

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

    async def send_password_reset_email(
        self,
        to_email: str,
        to_name: str,
        reset_token: str,
    ) -> bool:
        """Send password reset email.

        Args:
            to_email: Recipient email address
            to_name: Recipient name
            reset_token: Password reset token

        Returns:
            True if email was sent successfully
        """
        # Generate reset URL
        reset_url = (
            f"{self.settings.frontend_url}{self.settings.password_reset_url_path}"
            f"?token={reset_token}"
        )

        template_data = {
            "user_name": to_name,
            "reset_url": reset_url,
            "expiry_hours": self.settings.password_reset_token_expiry_hours,
            "support_email": self.settings.email_from_address,
        }

        return await self.send_email(
            to_email=to_email,
            to_name=to_name,
            subject="Reset Your Password - TLDR Highlights",
            template_name="password_reset",
            template_data=template_data,
        )

    async def send_welcome_email(
        self,
        to_email: str,
        to_name: str,
        organization_name: str,
    ) -> bool:
        """Send welcome email to new user.

        Args:
            to_email: Recipient email address
            to_name: Recipient name
            organization_name: Name of the organization

        Returns:
            True if email was sent successfully
        """
        template_data = {
            "user_name": to_name,
            "organization_name": organization_name,
            "login_url": f"{self.settings.frontend_url}/login",
            "docs_url": f"{self.settings.frontend_url}/docs",
            "support_email": self.settings.email_from_address,
        }

        return await self.send_email(
            to_email=to_email,
            to_name=to_name,
            subject=f"Welcome to TLDR Highlights - {organization_name}",
            template_name="welcome",
            template_data=template_data,
        )