"""Webhook security validation."""

import hmac
import hashlib
import time
from typing import Optional, Dict
from abc import ABC, abstractmethod


class WebhookValidator(ABC):
    """Base class for webhook validators."""

    @abstractmethod
    def validate_signature(
        self, payload: bytes, signature: str, timestamp: Optional[str] = None
    ) -> bool:
        """Validate webhook signature.

        Args:
            payload: Raw webhook payload
            signature: Signature to validate
            timestamp: Optional timestamp for replay protection

        Returns:
            True if valid, False otherwise
        """
        pass


class HMACWebhookValidator(WebhookValidator):
    """HMAC-based webhook validator."""

    def __init__(
        self, secret: str, header_prefix: str = "sha256=", max_age_seconds: int = 300
    ):
        """Initialize HMAC validator.

        Args:
            secret: Shared secret for HMAC
            header_prefix: Prefix for signature header
            max_age_seconds: Maximum age for timestamp validation
        """
        self.secret = secret
        self.header_prefix = header_prefix
        self.max_age_seconds = max_age_seconds

    def validate_signature(
        self, payload: bytes, signature: str, timestamp: Optional[str] = None
    ) -> bool:
        """Validate HMAC signature."""
        # Check timestamp if provided
        if timestamp and not self._validate_timestamp(timestamp):
            return False

        # Remove prefix if present
        if signature.startswith(self.header_prefix):
            signature = signature[len(self.header_prefix) :]

        # Calculate expected signature
        expected_signature = hmac.new(
            self.secret.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()

        # Compare signatures (constant time)
        return hmac.compare_digest(expected_signature, signature)

    def _validate_timestamp(self, timestamp: str) -> bool:
        """Validate timestamp is within acceptable range."""
        try:
            request_time = float(timestamp)
            current_time = time.time()
            age = abs(current_time - request_time)
            return age <= self.max_age_seconds
        except (ValueError, TypeError):
            return False


class HundredMSWebhookValidator(HMACWebhookValidator):
    """100ms-specific webhook validator."""

    def __init__(self, webhook_secret: str):
        """Initialize 100ms validator."""
        super().__init__(
            secret=webhook_secret,
            header_prefix="",  # 100ms doesn't use prefix
            max_age_seconds=300,
        )


class TwitchWebhookValidator(WebhookValidator):
    """Twitch EventSub webhook validator."""

    def __init__(self, webhook_secret: str):
        """Initialize Twitch validator."""
        self.secret = webhook_secret

    def validate_signature(
        self, payload: bytes, signature: str, timestamp: Optional[str] = None
    ) -> bool:
        """Validate Twitch EventSub signature."""
        if not timestamp:
            return False

        # Twitch uses: HMAC(secret, headers + body)
        message = f"twitch-eventsub-message-id={timestamp}".encode()
        message += b"twitch-eventsub-message-timestamp=" + timestamp.encode()
        message += b"twitch-eventsub-message-signature=" + signature.encode()
        message += payload

        expected_signature = hmac.new(
            self.secret.encode("utf-8"), message, hashlib.sha256
        ).hexdigest()

        expected_signature = f"sha256={expected_signature}"

        return hmac.compare_digest(expected_signature, signature)


class WebhookValidatorFactory:
    """Factory for creating platform-specific validators."""

    def __init__(self, webhook_secrets: Dict[str, str]):
        """Initialize validator factory.

        Args:
            webhook_secrets: Platform -> secret mapping
        """
        self.webhook_secrets = webhook_secrets

    def get_validator(self, platform: str) -> Optional[WebhookValidator]:
        """Get validator for platform.

        Args:
            platform: Platform name

        Returns:
            Validator instance or None
        """
        secret = self.webhook_secrets.get(platform)
        if not secret:
            return None

        platform_lower = platform.lower()

        if platform_lower == "100ms":
            return HundredMSWebhookValidator(secret)
        elif platform_lower == "twitch":
            return TwitchWebhookValidator(secret)
        else:
            # Default HMAC validator
            return HMACWebhookValidator(secret)


class WebhookAuthenticationError(Exception):
    """Raised when webhook authentication fails."""

    pass
