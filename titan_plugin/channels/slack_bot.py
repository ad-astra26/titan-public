"""
channels/slack_bot.py — Slack channel adapter for Titan.

Uses ``slack-bolt`` (v1.x, async mode with Socket Mode).  The library is
an optional dependency — if not installed the module logs a warning and
``start_slack()`` becomes a no-op.

Features:
  - Unified command registry (shared with all channels)
  - "Thinking..." placeholder while processing
  - Auto-retry on transient errors (via forward_to_titan)
  - User-friendly error messages
"""
import logging
import re

from titan_plugin.channels import (
    forward_to_titan,
    format_response,
    split_message,
)
from titan_plugin.channels.commands import CommandRegistry

logger = logging.getLogger(__name__)

_SLACK_CHAR_LIMIT = 4000  # Slack block text limit

try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    _HAS_SLACK = True
except ImportError:
    _HAS_SLACK = False
    logger.info("[Slack] slack-bolt not installed — adapter disabled.")


class TitanSlackBot:
    """Thin Slack ↔ Titan bridge with unified command support."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        titan_base_url: str = "http://127.0.0.1:7777",
    ):
        if not _HAS_SLACK:
            raise RuntimeError("slack-bolt is not installed.")

        self.titan_url = titan_base_url
        self._cmd_registry = CommandRegistry(titan_base_url=titan_base_url)

        self.app = AsyncApp(token=bot_token)
        self.handler = AsyncSocketModeHandler(self.app, app_token)
        self._bot_token = bot_token

        self._register_listeners()

    # ----- Listeners ------------------------------------------------------

    def _register_listeners(self) -> None:
        @self.app.event("message")
        async def handle_message(event, say, client):
            await self._on_message(event, say, client)

        @self.app.event("app_mention")
        async def handle_mention(event, say, client):
            await self._on_message(event, say, client)

    async def _on_message(self, event: dict, say, client) -> None:
        """Process an incoming message or @mention."""
        # Ignore bot messages and message_changed subtypes
        if event.get("bot_id") or event.get("subtype"):
            return

        text = event.get("text", "").strip()
        if not text:
            return

        # Strip the bot mention prefix if present (e.g. "<@U12345> hello")
        text = re.sub(r"^<@[A-Z0-9]+>\s*", "", text).strip()
        if not text:
            return

        channel = event.get("channel", "unknown")
        user = event.get("user", "unknown")
        user_id = f"sl_{user}"

        # Handle slash commands
        if text.startswith("/") and self._cmd_registry.is_command(text):
            response = await self._cmd_registry.execute(text, user_id)
            # Convert markdown bold to Slack mrkdwn bold
            response = response.replace("**", "*")
            chunks = split_message(response, _SLACK_CHAR_LIMIT)
            for chunk in chunks:
                await say(chunk)
            return

        session_id = f"slack_{channel}"

        # Post "Thinking..." placeholder, update with response
        thinking_msg = await say("Thinking...")
        thinking_ts = thinking_msg.get("ts", "") if isinstance(thinking_msg, dict) else ""

        result = await forward_to_titan(
            message=text,
            session_id=session_id,
            user_id=user_id,
            base_url=self.titan_url,
        )

        reply = format_response(result)
        # Convert markdown bold to Slack mrkdwn bold
        reply = reply.replace("**", "*")

        chunks = split_message(reply, _SLACK_CHAR_LIMIT)

        # Update the "Thinking..." message with the first chunk
        if thinking_ts and chunks:
            try:
                await client.chat_update(
                    channel=channel,
                    ts=thinking_ts,
                    text=chunks[0],
                )
                # Send remaining chunks as new messages
                for chunk in chunks[1:]:
                    await say(chunk)
            except Exception:
                # Fallback: just send all chunks
                for chunk in chunks:
                    await say(chunk)
        else:
            for chunk in chunks:
                await say(chunk)

    # ----- Lifecycle ------------------------------------------------------

    async def start(self) -> None:
        """Start Socket Mode connection (runs until cancelled)."""
        logger.info("[Slack] Starting Socket Mode...")
        await self.handler.start_async()


async def start_slack(config: dict, titan_base_url: str = "http://127.0.0.1:7777") -> None:
    """Boot the Slack adapter from a config dict."""
    if not _HAS_SLACK:
        logger.warning("[Slack] slack-bolt not installed — skipping.")
        return

    bot_token = config.get("bot_token", "")
    app_token = config.get("app_token", "")

    if not bot_token or not app_token:
        logger.warning("[Slack] Missing bot_token or app_token — skipping.")
        return

    bot = TitanSlackBot(
        bot_token=bot_token,
        app_token=app_token,
        titan_base_url=titan_base_url,
    )
    await bot.start()
