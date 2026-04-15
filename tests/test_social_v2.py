"""
tests/test_social_v2.py
Verification suite for the production-grade SocialManager.
Mocks TwitterAPI.io v2 endpoints and verifies session recovery + media flow.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from titan_plugin.expressive.social import SocialManager

@pytest.fixture
def mock_metabolism():
    return AsyncMock()

@pytest.fixture
def mock_memory():
    m = AsyncMock()
    m.fetch_social_metrics.return_value = {
        "daily_likes": 0, "daily_replies": 0,
        "mentions_received": 0, "reply_likes": 0,
    }
    m.add_social_history = MagicMock()
    return m

@pytest.mark.asyncio
async def test_smart_login_success(mock_metabolism, mock_memory):
    """Verify that _get_smart_session handles the 2026-era login_cookies key."""
    manager = SocialManager(mock_metabolism, memory=mock_memory)
    manager.config = {
        "user_name": "test",
        "email": "test@example.com",
        "password": "pass",
        "totp_secret": "SECRET",
        "webshare_static_url": "proxy_url"
    }
    
    # Mock successful login response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "success",
        "login_cookies": "MOCK_SESSION_123"
    }
    
    with patch.object(manager.client, "post", return_value=mock_response):
        with patch.object(manager, "_update_config_session") as mock_update:
            session = await manager._get_smart_session()
            assert session == "MOCK_SESSION_123"
            mock_update.assert_called_once_with("MOCK_SESSION_123")

@pytest.mark.asyncio
async def test_reactive_session_recovery(mock_metabolism, mock_memory):
    """Verify Option B: Retry on Expired session."""
    manager = SocialManager(mock_metabolism, memory=mock_memory)
    manager.config = {"auth_session": "EXPIRED_SESSION"}
    
    # First response: Expired
    resp1 = MagicMock()
    resp1.json.return_value = {"status": "error", "msg": "Session expired"}
    
    # Second response: Success (after login)
    resp2 = MagicMock()
    resp2.json.return_value = {"status": "success"}
    
    # Mock login
    login_resp = MagicMock()
    login_resp.json.return_value = {"status": "success", "login_cookies": "NEW_SESSION"}
    
    with patch.object(manager.client, "post", side_effect=[resp1, login_resp, resp2]):
        success = await manager.create_tweet("Testing recovery")
        assert success is True
        assert manager.config["auth_session"] == "NEW_SESSION"

@pytest.mark.asyncio
async def test_engagement_limits_enforced(mock_metabolism, mock_memory):
    """Verify Cognee-based limit policing: when both reply and like limits are hit, skip engagement."""
    mock_memory.fetch_social_metrics.return_value = {
        "daily_replies": 10, "daily_likes": 15,
        "mentions_received": 0, "reply_likes": 0,
    }
    mock_metabolism.get_current_state.return_value = "HIGH_ENERGY"
    manager = SocialManager(mock_metabolism, memory=mock_memory)
    manager.config = {"max_replies_per_day": 5, "max_likes_per_day": 10}

    with patch("logging.info") as mock_log:
        await manager.monitor_and_engage()
        # Should log that daily limits reached and skip
        log_messages = [call[0][0] for call in mock_log.call_args_list]
        assert any("Daily engagement limits reached" in msg for msg in log_messages)

if __name__ == "__main__":
    pytest.main([__file__])
