"""
tests/verify_social_standalone.py
Standalone verification script for SocialManager logic.
Does NOT require pytest.
"""
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Ensure the project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from titan_plugin.expressive.social import SocialManager

async def test_smart_login_logic():
    print("Testing Smart Login logic...")
    mock_metabolism = AsyncMock()
    mock_memory = AsyncMock()
    manager = SocialManager(mock_metabolism, memory=mock_memory)
    manager.config = {
        "user_name": "test",
        "email": "test@example.com",
        "password": "pass",
        "totp_secret": "SECRET",
        "webshare_static_url": "proxy_url"
    }
    
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "success",
        "login_cookies": "MOCK_SESSION_123"
    }
    
    with patch.object(manager.client, "post", return_value=mock_response):
        with patch.object(manager, "_update_config_session") as mock_update:
            session = await manager._get_smart_session()
            assert session == "MOCK_SESSION_123", f"Expected MOCK_SESSION_123, got {session}"
            mock_update.assert_called_once_with("MOCK_SESSION_123")
    print("✅ Smart Login logic passed.")

async def test_reactive_recovery_logic():
    print("Testing Reactive Recovery logic...")
    mock_metabolism = AsyncMock()
    mock_memory = AsyncMock()
    manager = SocialManager(mock_metabolism, memory=mock_memory)
    manager.config = {"auth_session": "EXPIRED_SESSION"}
    
    # Step 1: Expired response
    resp1 = MagicMock()
    resp1.json.return_value = {"status": "error", "msg": "Session expired"}
    
    # Step 2: Login success
    login_resp = MagicMock()
    login_resp.json.return_value = {"status": "success", "login_cookies": "NEW_SESSION"}
    
    # Step 3: Action retry success
    resp3 = MagicMock()
    resp3.json.return_value = {"status": "success"}
    
    with patch.object(manager.client, "post", side_effect=[resp1, login_resp, resp3]):
        success = await manager.create_tweet("Testing recovery")
        assert success is True
        assert manager.config["auth_session"] == "NEW_SESSION"
    print("✅ Reactive Recovery logic passed.")

async def main():
    try:
        await test_smart_login_logic()
        await test_reactive_recovery_logic()
        print("\n🎉 All standalone logic checks PASSED.")
    except Exception as e:
        print(f"\n❌ Logic check FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
