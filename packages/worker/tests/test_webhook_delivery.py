"""Tests for webhook delivery tasks."""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import hmac
import hashlib
import json

from worker.tasks.webhook_delivery import send_generic_webhook as send_webhook, _send_webhook_async


class TestWebhookDelivery:
    """Test webhook delivery functionality."""
    
    @pytest.mark.asyncio
    async def test_send_webhook_success(self):
        """Test successful webhook delivery."""
        webhook_url = "https://example.com/webhook"
        webhook_secret = "test-secret"
        payload = {
            "event": "stream.processed",
            "stream_id": "123",
            "highlights_count": 5
        }
        
        with patch("worker.tasks.webhook_delivery.httpx.AsyncClient") as mock_client_class:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "OK"
            
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            # Send webhook
            result = await _send_webhook_async(webhook_url, webhook_secret, payload)
            
            # Verify result
            assert result["status"] == "success"
            assert result["status_code"] == 200
            assert result["attempts"] == 1
            
            # Verify request was made correctly
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            
            # Check URL
            assert call_args[0][0] == webhook_url
            
            # Check headers
            headers = call_args[1]["headers"]
            assert headers["Content-Type"] == "application/json"
            assert "X-Webhook-Signature" in headers
            
            # Verify signature
            payload_json = json.dumps(payload, separators=(",", ":"))
            expected_signature = hmac.new(
                webhook_secret.encode(),
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
            assert headers["X-Webhook-Signature"] == f"sha256={expected_signature}"
    
    @pytest.mark.asyncio
    async def test_send_webhook_retry_on_failure(self):
        """Test webhook retry on temporary failure."""
        webhook_url = "https://example.com/webhook"
        webhook_secret = "test-secret"
        payload = {"event": "test"}
        
        with patch("worker.tasks.webhook_delivery.httpx.AsyncClient") as mock_client_class:
            # Mock responses: fail twice, then succeed
            mock_responses = [
                MagicMock(status_code=500, text="Server Error"),
                MagicMock(status_code=502, text="Bad Gateway"),
                MagicMock(status_code=200, text="OK")
            ]
            
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=mock_responses)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            # Mock sleep to speed up test
            with patch("worker.tasks.webhook_delivery.asyncio.sleep", new_callable=AsyncMock):
                # Send webhook
                result = await _send_webhook_async(webhook_url, webhook_secret, payload)
                
                # Should succeed after retries
                assert result["status"] == "success"
                assert result["status_code"] == 200
                assert result["attempts"] == 3
                
                # Verify retries
                assert mock_client.post.call_count == 3
    
    @pytest.mark.asyncio
    async def test_send_webhook_max_retries_exceeded(self):
        """Test webhook delivery when max retries exceeded."""
        webhook_url = "https://example.com/webhook"
        webhook_secret = "test-secret"
        payload = {"event": "test"}
        
        with patch("worker.tasks.webhook_delivery.httpx.AsyncClient") as mock_client_class:
            # Mock all responses as failures
            mock_response = MagicMock(status_code=500, text="Server Error")
            
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            # Mock sleep to speed up test
            with patch("worker.tasks.webhook_delivery.asyncio.sleep", new_callable=AsyncMock):
                # Send webhook
                result = await _send_webhook_async(webhook_url, webhook_secret, payload)
                
                # Should fail after max retries
                assert result["status"] == "failed"
                assert result["status_code"] == 500
                assert result["attempts"] == 3  # Default max retries
                assert "Max retries exceeded" in result["error"]
    
    @pytest.mark.asyncio
    async def test_send_webhook_network_error(self):
        """Test webhook delivery with network error."""
        webhook_url = "https://example.com/webhook"
        webhook_secret = "test-secret"
        payload = {"event": "test"}
        
        with patch("worker.tasks.webhook_delivery.httpx.AsyncClient") as mock_client_class:
            # Mock network error
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            # Mock sleep to speed up test
            with patch("worker.tasks.webhook_delivery.asyncio.sleep", new_callable=AsyncMock):
                # Send webhook
                result = await _send_webhook_async(webhook_url, webhook_secret, payload)
                
                # Should fail with error
                assert result["status"] == "failed"
                assert result["attempts"] == 3
                assert "Connection refused" in result["error"]
    
    @pytest.mark.asyncio
    async def test_send_webhook_timeout(self):
        """Test webhook delivery with timeout."""
        webhook_url = "https://example.com/webhook"
        webhook_secret = "test-secret"
        payload = {"event": "test"}
        
        with patch("worker.tasks.webhook_delivery.httpx.AsyncClient") as mock_client_class:
            # Mock timeout error
            import httpx
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Request timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            # Mock sleep to speed up test
            with patch("worker.tasks.webhook_delivery.asyncio.sleep", new_callable=AsyncMock):
                # Send webhook
                result = await _send_webhook_async(webhook_url, webhook_secret, payload)
                
                # Should retry and eventually fail
                assert result["status"] == "failed"
                assert result["attempts"] == 3
                assert "Request timeout" in result["error"]
    
    @pytest.mark.asyncio
    async def test_send_webhook_non_retryable_error(self):
        """Test webhook delivery with non-retryable error (4xx)."""
        webhook_url = "https://example.com/webhook"
        webhook_secret = "test-secret"
        payload = {"event": "test"}
        
        with patch("worker.tasks.webhook_delivery.httpx.AsyncClient") as mock_client_class:
            # Mock 4xx response (should not retry)
            mock_response = MagicMock(status_code=400, text="Bad Request")
            
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            # Send webhook
            result = await _send_webhook_async(webhook_url, webhook_secret, payload)
            
            # Should fail immediately without retries
            assert result["status"] == "failed"
            assert result["status_code"] == 400
            assert result["attempts"] == 1  # No retries for 4xx
            
            # Verify only called once
            mock_client.post.assert_called_once()
    
    def test_send_webhook_celery_task(self):
        """Test Celery task wrapper."""
        webhook_url = "https://example.com/webhook"
        webhook_secret = "test-secret"
        payload = {"event": "test"}
        
        with patch("worker.tasks.webhook_delivery.asyncio.run") as mock_run:
            mock_run.return_value = {
                "status": "success",
                "status_code": 200,
                "attempts": 1
            }
            
            # Call the Celery task
            result = send_webhook(webhook_url, webhook_secret, payload)
            
            # Verify asyncio.run was called
            mock_run.assert_called_once()
            
            # Verify result
            assert result["status"] == "success"
            assert result["status_code"] == 200