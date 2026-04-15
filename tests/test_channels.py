"""
tests/test_channels.py — Unit tests for channel adapters.

All external channel libraries and the Titan API are mocked — no live
services required.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from titan_plugin.channels import (
    forward_to_titan,
    format_response,
    get_channel_config,
    split_message,
)


# =========================================================================
# forward_to_titan
# =========================================================================

class TestForwardToTitan:
    """Test the core HTTP bridge to POST /chat."""

    @pytest.mark.asyncio
    async def test_success(self):
        mock_response = httpx.Response(
            200,
            json={
                "response": "Hello from Titan!",
                "session_id": "test_sess",
                "mode": "Sovereign",
                "mood": "Curious",
            },
            request=httpx.Request("POST", "http://test/chat"),
        )
        with patch("titan_plugin.channels.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await forward_to_titan("hello", "sess1", "user1", "http://test")

        assert result["response"] == "Hello from Titan!"
        assert result["mode"] == "Sovereign"
        assert result["mood"] == "Curious"

    @pytest.mark.asyncio
    async def test_guardian_block_403(self):
        mock_response = httpx.Response(
            403,
            json={
                "error": "Sovereignty Violation: blocked",
                "blocked": True,
                "mode": "Guardian",
            },
            request=httpx.Request("POST", "http://test/chat"),
        )
        with patch("titan_plugin.channels.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await forward_to_titan("bad query", "sess1", "user1", "http://test")

        assert result["blocked"] is True
        assert result["mode"] == "Guardian"
        assert "Sovereignty Violation" in result["error"]

    @pytest.mark.asyncio
    async def test_limbo_503(self):
        mock_response = httpx.Response(
            503,
            json={"error": "Titan is in Limbo state."},
            request=httpx.Request("POST", "http://test/chat"),
        )
        with patch("titan_plugin.channels.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await forward_to_titan("hello", "sess1", "user1", "http://test")

        assert result.get("limbo") is True
        assert "Limbo" in result["error"]

    @pytest.mark.asyncio
    async def test_timeout(self):
        with patch("titan_plugin.channels.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("timed out")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await forward_to_titan("hello", "sess1", "user1", "http://test")

        assert "error" in result
        assert "not respond in time" in result["error"]

    @pytest.mark.asyncio
    async def test_connection_error(self):
        with patch("titan_plugin.channels.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await forward_to_titan("hello", "sess1", "user1", "http://test")

        assert "error" in result
        assert "Cannot reach Titan" in result["error"]

    @pytest.mark.asyncio
    async def test_unexpected_status(self):
        mock_response = httpx.Response(
            500,
            json={"error": "Internal server error"},
            request=httpx.Request("POST", "http://test/chat"),
        )
        with patch("titan_plugin.channels.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await forward_to_titan("hello", "sess1", "user1", "http://test")

        assert "error" in result
        assert "HTTP 500" in result["error"]


# =========================================================================
# format_response
# =========================================================================

class TestFormatResponse:
    """Test response formatting for chat display."""

    def test_success_format(self):
        result = {
            "response": "Hello! I'm Titan.",
            "mode": "Sovereign",
            "mood": "Curious",
        }
        assert format_response(result) == "[Sovereign|Curious] Hello! I'm Titan."

    def test_error_format_with_mode(self):
        result = {
            "error": "Sovereignty Violation: blocked",
            "mode": "Guardian",
        }
        assert format_response(result) == "[Guardian] Sovereignty Violation: blocked"

    def test_error_format_without_mode(self):
        result = {"error": "Connection failed."}
        assert format_response(result) == "[Error] Connection failed."

    def test_missing_fields_defaults(self):
        result = {"response": "Hi"}
        out = format_response(result)
        assert "[Unknown|Unknown]" in out
        assert "Hi" in out


# =========================================================================
# split_message
# =========================================================================

class TestSplitMessage:
    """Test message splitting for channel character limits."""

    def test_short_message_no_split(self):
        assert split_message("hello", 100) == ["hello"]

    def test_exact_limit(self):
        text = "a" * 100
        assert split_message(text, 100) == [text]

    def test_split_on_paragraph(self):
        text = "paragraph one\n\nparagraph two"
        chunks = split_message(text, 20)
        assert len(chunks) == 2
        assert "paragraph one" in chunks[0]
        assert "paragraph two" in chunks[1]

    def test_split_on_newline(self):
        text = "line one\nline two is longer"
        chunks = split_message(text, 15)
        assert len(chunks) >= 2

    def test_split_on_space(self):
        text = "word " * 20  # 100 chars
        chunks = split_message(text, 30)
        assert all(len(c) <= 30 for c in chunks)

    def test_telegram_limit(self):
        text = "x" * 8000
        chunks = split_message(text, 4096)
        assert all(len(c) <= 4096 for c in chunks)
        assert "".join(chunks) == text

    def test_discord_limit(self):
        text = "word " * 500  # 2500 chars
        chunks = split_message(text, 2000)
        assert all(len(c) <= 2000 for c in chunks)

    def test_empty_string(self):
        assert split_message("", 100) == [""]


# =========================================================================
# get_channel_config
# =========================================================================

class TestGetChannelConfig:
    """Test config.toml channel config loading."""

    def test_loads_telegram_config(self):
        config = get_channel_config("telegram")
        # Should have at least the enabled and bot_token keys
        assert "enabled" in config
        assert "bot_token" in config

    def test_loads_discord_config(self):
        config = get_channel_config("discord")
        assert "enabled" in config
        assert "bot_token" in config

    def test_loads_slack_config(self):
        config = get_channel_config("slack")
        assert "enabled" in config
        assert "bot_token" in config
        assert "app_token" in config

    def test_loads_whatsapp_config(self):
        config = get_channel_config("whatsapp")
        assert "enabled" in config
        assert "phone_id" in config
        assert "token" in config

    def test_unknown_channel_returns_empty(self):
        config = get_channel_config("nonexistent_channel")
        assert config == {}


# =========================================================================
# Session ID generation
# =========================================================================

class TestSessionIds:
    """Verify session ID conventions per channel."""

    def test_telegram_session_format(self):
        sid = f"telegram_{12345678}"
        assert sid.startswith("telegram_")
        assert "12345678" in sid

    def test_discord_session_format(self):
        sid = f"discord_{987654321}"
        assert sid.startswith("discord_")

    def test_slack_session_format(self):
        sid = f"slack_C01ABCDEF"
        assert sid.startswith("slack_")

    def test_whatsapp_session_format(self):
        sid = f"whatsapp_15551234567"
        assert sid.startswith("whatsapp_")

    def test_user_id_prefixes(self):
        assert f"tg_{123}".startswith("tg_")
        assert f"dc_{456}".startswith("dc_")
        assert f"sl_{789}".startswith("sl_")
        assert f"wa_{111}".startswith("wa_")


# =========================================================================
# Telegram adapter
# =========================================================================

class TestTelegramAdapter:
    """Test TitanTelegramBot message handling with mocked telegram lib."""

    def _make_bot(self):
        """Create a TitanTelegramBot without importing python-telegram-bot."""
        from titan_plugin.channels.telegram import TitanTelegramBot
        bot = object.__new__(TitanTelegramBot)
        bot.token = "test"
        bot.titan_url = "http://test"
        bot._app = None
        return bot

    @pytest.mark.asyncio
    async def test_handle_message_success(self):
        """Simulate a Telegram text message forwarded to Titan."""
        mock_update = MagicMock()
        mock_update.message.text = "Hello Titan"
        mock_update.effective_chat.id = 12345
        mock_update.effective_user.id = 67890
        mock_update.effective_chat.send_action = AsyncMock()
        mock_update.message.reply_text = AsyncMock()

        titan_response = {
            "response": "Hello from Titan!",
            "session_id": "telegram_12345",
            "mode": "Collaborative",
            "mood": "Curious",
        }

        with patch("titan_plugin.channels.telegram.forward_to_titan", new_callable=AsyncMock) as mock_fwd:
            mock_fwd.return_value = titan_response
            bot = self._make_bot()
            await bot.handle_message(mock_update, MagicMock())

            mock_fwd.assert_called_once_with(
                message="Hello Titan",
                session_id="telegram_12345",
                user_id="tg_67890",
                base_url="http://test",
            )
        mock_update.message.reply_text.assert_called_once()
        reply_text = mock_update.message.reply_text.call_args[0][0]
        assert "Hello from Titan!" in reply_text

    @pytest.mark.asyncio
    async def test_handle_message_guardian_block(self):
        """Guardian block should reply with error, not crash."""
        mock_update = MagicMock()
        mock_update.message.text = "bad stuff"
        mock_update.effective_chat.id = 12345
        mock_update.effective_user.id = 67890
        mock_update.effective_chat.send_action = AsyncMock()
        mock_update.message.reply_text = AsyncMock()

        blocked_response = {
            "error": "Sovereignty Violation: blocked",
            "blocked": True,
            "mode": "Guardian",
        }

        with patch("titan_plugin.channels.telegram.forward_to_titan", new_callable=AsyncMock) as mock_fwd:
            mock_fwd.return_value = blocked_response
            bot = self._make_bot()
            await bot.handle_message(mock_update, MagicMock())

        mock_update.message.reply_text.assert_called_once()
        reply_text = mock_update.message.reply_text.call_args[0][0]
        assert "Guardian" in reply_text

    @pytest.mark.asyncio
    async def test_long_message_splitting(self):
        """Messages exceeding 4096 chars should be split."""
        mock_update = MagicMock()
        mock_update.message.text = "hi"
        mock_update.effective_chat.id = 1
        mock_update.effective_user.id = 2
        mock_update.effective_chat.send_action = AsyncMock()
        mock_update.message.reply_text = AsyncMock()

        long_text = "word " * 1000  # ~5000 chars
        titan_response = {
            "response": long_text,
            "session_id": "telegram_1",
            "mode": "Sovereign",
            "mood": "Content",
        }

        with patch("titan_plugin.channels.telegram.forward_to_titan", new_callable=AsyncMock) as mock_fwd:
            mock_fwd.return_value = titan_response
            bot = self._make_bot()
            await bot.handle_message(mock_update, MagicMock())

        # Should have been called multiple times due to splitting
        assert mock_update.message.reply_text.call_count >= 2


# =========================================================================
# Discord adapter
# =========================================================================

class TestDiscordAdapter:
    """Test Discord message handling with mocked discord.py."""

    @pytest.mark.asyncio
    async def test_handle_chat_success(self):
        mock_message = MagicMock()
        mock_message.content = "Hello Titan"
        mock_message.channel.id = 111222333
        mock_message.author.id = 444555666
        mock_message.channel.send = AsyncMock()
        mock_message.channel.typing = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        titan_response = {
            "response": "Discord reply!",
            "session_id": "discord_111222333",
            "mode": "Collaborative",
            "mood": "Happy",
        }

        with patch("titan_plugin.channels.discord_bot.forward_to_titan", new_callable=AsyncMock) as mock_fwd:
            mock_fwd.return_value = titan_response

            from titan_plugin.channels.discord_bot import TitanDiscordBot
            bot = object.__new__(TitanDiscordBot)
            bot.titan_url = "http://test"
            bot.token = "test"

            await bot._handle_chat(mock_message)

            mock_fwd.assert_called_once_with(
                message="Hello Titan",
                session_id="discord_111222333",
                user_id="dc_444555666",
                base_url="http://test",
            )
        mock_message.channel.send.assert_called_once()
        assert "Discord reply!" in mock_message.channel.send.call_args[0][0]


# =========================================================================
# Slack adapter
# =========================================================================

class TestSlackAdapter:
    """Test Slack message handling with mocked slack-bolt."""

    def _make_bot(self):
        from titan_plugin.channels.slack_bot import TitanSlackBot
        bot = object.__new__(TitanSlackBot)
        bot.titan_url = "http://test"
        return bot

    @pytest.mark.asyncio
    async def test_on_message_success(self):
        event = {
            "text": "Hello Titan",
            "channel": "C01GENERAL",
            "user": "U01ADMIN",
        }
        say = AsyncMock()

        titan_response = {
            "response": "Slack reply!",
            "session_id": "slack_C01GENERAL",
            "mode": "Sovereign",
            "mood": "Focused",
        }

        with patch("titan_plugin.channels.slack_bot.forward_to_titan", new_callable=AsyncMock) as mock_fwd:
            mock_fwd.return_value = titan_response
            bot = self._make_bot()
            await bot._on_message(event, say)

            mock_fwd.assert_called_once_with(
                message="Hello Titan",
                session_id="slack_C01GENERAL",
                user_id="sl_U01ADMIN",
                base_url="http://test",
            )
        say.assert_called_once()
        assert "Slack reply!" in say.call_args[0][0]

    @pytest.mark.asyncio
    async def test_ignores_bot_messages(self):
        event = {
            "text": "bot message",
            "channel": "C01",
            "user": "U01",
            "bot_id": "B01",
        }
        say = AsyncMock()

        bot = self._make_bot()
        await bot._on_message(event, say)

        say.assert_not_called()

    @pytest.mark.asyncio
    async def test_strips_mention_prefix(self):
        event = {
            "text": "<@U012ABC> what is the meaning of life",
            "channel": "C01",
            "user": "U02",
        }
        say = AsyncMock()

        titan_response = {
            "response": "42",
            "session_id": "slack_C01",
            "mode": "Collaborative",
            "mood": "Philosophical",
        }

        with patch("titan_plugin.channels.slack_bot.forward_to_titan", new_callable=AsyncMock) as mock_fwd:
            mock_fwd.return_value = titan_response
            bot = self._make_bot()
            await bot._on_message(event, say)

            # The forwarded message should have the mention stripped
            call_args = mock_fwd.call_args
            assert call_args.kwargs["message"] == "what is the meaning of life"


# =========================================================================
# Start function guards
# =========================================================================

class TestStartGuards:
    """Verify that start_* functions gracefully skip when libs are missing."""

    @pytest.mark.asyncio
    async def test_telegram_start_no_lib(self):
        with patch("titan_plugin.channels.telegram._HAS_TELEGRAM", False):
            from titan_plugin.channels.telegram import start_telegram
            # Should not raise — just log and return
            await start_telegram({"bot_token": "test"})

    @pytest.mark.asyncio
    async def test_discord_start_no_lib(self):
        with patch("titan_plugin.channels.discord_bot._HAS_DISCORD", False):
            from titan_plugin.channels.discord_bot import start_discord
            await start_discord({"bot_token": "test"})

    @pytest.mark.asyncio
    async def test_slack_start_no_lib(self):
        with patch("titan_plugin.channels.slack_bot._HAS_SLACK", False):
            from titan_plugin.channels.slack_bot import start_slack
            await start_slack({"bot_token": "test", "app_token": "test"})

    @pytest.mark.asyncio
    async def test_whatsapp_start_no_lib(self):
        with patch("titan_plugin.channels.whatsapp._HAS_PYWA", False):
            from titan_plugin.channels.whatsapp import start_whatsapp
            await start_whatsapp({"phone_id": "123", "token": "test"})

    @pytest.mark.asyncio
    async def test_telegram_start_no_token(self):
        with patch("titan_plugin.channels.telegram._HAS_TELEGRAM", True):
            from titan_plugin.channels.telegram import start_telegram
            # Empty token — should skip gracefully
            await start_telegram({})

    @pytest.mark.asyncio
    async def test_discord_start_no_token(self):
        with patch("titan_plugin.channels.discord_bot._HAS_DISCORD", True):
            from titan_plugin.channels.discord_bot import start_discord
            await start_discord({})

    @pytest.mark.asyncio
    async def test_slack_start_missing_tokens(self):
        with patch("titan_plugin.channels.slack_bot._HAS_SLACK", True):
            from titan_plugin.channels.slack_bot import start_slack
            await start_slack({"bot_token": "yes"})  # missing app_token

    @pytest.mark.asyncio
    async def test_whatsapp_start_missing_creds(self):
        with patch("titan_plugin.channels.whatsapp._HAS_PYWA", True):
            from titan_plugin.channels.whatsapp import start_whatsapp
            await start_whatsapp({})  # missing phone_id and token
