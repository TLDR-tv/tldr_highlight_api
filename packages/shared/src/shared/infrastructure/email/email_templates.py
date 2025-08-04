"""Email template definitions and rendering."""

from typing import Dict, Any
from string import Template


class EmailTemplates:
    """Email template definitions."""

    PASSWORD_RESET_HTML = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Reset Your Password</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background-color: #f9f9f9; }
        .button { display: inline-block; padding: 12px 24px; background-color: #3498db; color: white; text-decoration: none; border-radius: 4px; margin: 20px 0; }
        .footer { padding: 20px; text-align: center; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>TLDR Highlights</h1>
        </div>
        <div class="content">
            <h2>Hello ${user_name},</h2>
            <p>We received a request to reset your password. Click the button below to create a new password:</p>
            <p style="text-align: center;">
                <a href="${reset_url}" class="button">Reset Password</a>
            </p>
            <p>This link will expire in ${expiry_hours} hours.</p>
            <p>If you didn't request this password reset, please ignore this email.</p>
            <p>Best regards,<br>The TLDR Highlights Team</p>
        </div>
        <div class="footer">
            <p>If you're having trouble clicking the button, copy and paste this URL into your browser:</p>
            <p>${reset_url}</p>
            <p>Need help? Contact us at ${support_email}</p>
        </div>
    </div>
</body>
</html>
""")

    PASSWORD_RESET_TEXT = Template("""
Hello ${user_name},

We received a request to reset your password for TLDR Highlights.

To reset your password, visit this link:
${reset_url}

This link will expire in ${expiry_hours} hours.

If you didn't request this password reset, please ignore this email.

Best regards,
The TLDR Highlights Team

Need help? Contact us at ${support_email}
""")

    WELCOME_HTML = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Welcome to TLDR Highlights</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background-color: #f9f9f9; }
        .button { display: inline-block; padding: 12px 24px; background-color: #3498db; color: white; text-decoration: none; border-radius: 4px; margin: 20px 0; }
        .footer { padding: 20px; text-align: center; font-size: 12px; color: #666; }
        .feature { margin: 15px 0; padding-left: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Welcome to TLDR Highlights</h1>
        </div>
        <div class="content">
            <h2>Hello ${user_name},</h2>
            <p>Welcome to <strong>${organization_name}</strong> on TLDR Highlights!</p>
            <p>Your account has been successfully created. You can now:</p>
            <div class="feature">• Extract highlights from livestreams and videos</div>
            <div class="feature">• Use AI-powered detection for key moments</div>
            <div class="feature">• Integrate with your existing workflow via our API</div>
            <p style="text-align: center;">
                <a href="${login_url}" class="button">Login to Your Account</a>
            </p>
            <p>To get started, check out our <a href="${docs_url}">documentation</a>.</p>
            <p>Best regards,<br>The TLDR Highlights Team</p>
        </div>
        <div class="footer">
            <p>Need help? Contact us at ${support_email}</p>
        </div>
    </div>
</body>
</html>
""")

    WELCOME_TEXT = Template("""
Hello ${user_name},

Welcome to ${organization_name} on TLDR Highlights!

Your account has been successfully created. You can now:
• Extract highlights from livestreams and videos
• Use AI-powered detection for key moments
• Integrate with your existing workflow via our API

Login to your account: ${login_url}

To get started, check out our documentation: ${docs_url}

Best regards,
The TLDR Highlights Team

Need help? Contact us at ${support_email}
""")

    @staticmethod
    def render_password_reset(data: Dict[str, Any]) -> tuple[str, str]:
        """Render password reset email templates.

        Args:
            data: Template data containing user_name, reset_url, expiry_hours, support_email

        Returns:
            Tuple of (html_content, text_content)
        """
        html = EmailTemplates.PASSWORD_RESET_HTML.safe_substitute(**data)
        text = EmailTemplates.PASSWORD_RESET_TEXT.safe_substitute(**data)
        return html, text

    @staticmethod
    def render_welcome(data: Dict[str, Any]) -> tuple[str, str]:
        """Render welcome email templates.

        Args:
            data: Template data containing user_name, organization_name, login_url, docs_url, support_email

        Returns:
            Tuple of (html_content, text_content)
        """
        html = EmailTemplates.WELCOME_HTML.safe_substitute(**data)
        text = EmailTemplates.WELCOME_TEXT.safe_substitute(**data)
        return html, text