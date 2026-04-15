"""
channels/discord_bot.py — Discord channel adapter for Titan.

Uses ``discord.py`` (v2.x, async-native).  The library is an optional
dependency — if not installed the module logs a warning and
``start_discord()`` becomes a no-op.

Features:
  - Unified command registry (shared with all channels)
  - Typing indicator while processing
  - Auto-retry on transient errors (via forward_to_titan)
  - User-friendly error messages
"""
import logging

from titan_plugin.channels import (
    forward_to_titan,
    format_response,
    split_message,
)
from titan_plugin.channels.commands import CommandRegistry

logger = logging.getLogger(__name__)

_DISCORD_CHAR_LIMIT = 2000

try:
    import discord
    from discord.ext import commands
    _HAS_DISCORD = True
except ImportError:
    _HAS_DISCORD = False
    logger.info("[Discord] discord.py not installed — adapter disabled.")


class TitanDiscordBot:
    """Thin Discord ↔ Titan bridge with unified command support."""

    def __init__(
        self,
        token: str,
        titan_base_url: str = "http://127.0.0.1:7777",
        command_prefix: str = "/",
    ):
        if not _HAS_DISCORD:
            raise RuntimeError("discord.py is not installed.")

        self.token = token
        self.titan_url = titan_base_url
        self._cmd_registry = CommandRegistry(titan_base_url=titan_base_url)

        intents = discord.Intents.default()
        intents.message_content = True

        self.bot = commands.Bot(command_prefix=command_prefix, intents=intents)
        self._register_events()
        self._register_commands()

    # ----- Events ---------------------------------------------------------

    def _register_events(self) -> None:
        @self.bot.event
        async def on_ready():
            logger.info("[Discord] Logged in as %s (ID: %s)", self.bot.user, self.bot.user.id)

        @self.bot.event
        async def on_message(message: "discord.Message"):
            # Ignore own messages and other bots
            if message.author == self.bot.user or message.author.bot:
                return

            # Process commands first
            await self.bot.process_commands(message)

            # If the message was a command, don't also treat it as chat
            ctx = await self.bot.get_context(message)
            if ctx.valid:
                return

            # Check for slash commands in text (Discord text commands)
            text = message.content.strip()
            if text.startswith("/") and self._cmd_registry.is_command(text):
                user_id = f"dc_{message.author.id}"
                response = await self._cmd_registry.execute(text, user_id)
                chunks = split_message(response, _DISCORD_CHAR_LIMIT)
                for chunk in chunks:
                    await message.channel.send(chunk)
                return

            await self._handle_chat(message)

    async def _handle_chat(self, message: "discord.Message") -> None:
        """Forward a regular text message to POST /chat."""
        session_id = f"discord_{message.channel.id}"
        user_id = f"dc_{message.author.id}"

        # Show typing indicator during processing
        async with message.channel.typing():
            result = await forward_to_titan(
                message=message.content,
                session_id=session_id,
                user_id=user_id,
                base_url=self.titan_url,
            )

        reply = format_response(result)
        chunks = split_message(reply, _DISCORD_CHAR_LIMIT)
        for chunk in chunks:
            await message.channel.send(chunk)

    # ----- Commands -------------------------------------------------------

    def _register_commands(self) -> None:
        """Register bot framework commands for backward compat (!commands)."""
        @self.bot.command(name="commands")
        async def commands_cmd(ctx: "commands.Context"):
            """List all available Titan commands."""
            user_id = f"dc_{ctx.author.id}"
            response = await self._cmd_registry.execute("/commands", user_id)
            await ctx.send(response)

    # ----- Lifecycle ------------------------------------------------------

    def run(self) -> None:
        """Start the bot (blocking)."""
        logger.info("[Discord] Starting bot...")
        self.bot.run(self.token, log_handler=None)


async def start_discord(config: dict, titan_base_url: str = "http://127.0.0.1:7777") -> None:
    """Boot the Discord adapter from a config dict."""
    if not _HAS_DISCORD:
        logger.warning("[Discord] discord.py not installed — skipping.")
        return

    token = config.get("bot_token", "")
    if not token:
        logger.warning("[Discord] No bot_token configured — skipping.")
        return

    adapter = TitanDiscordBot(token=token, titan_base_url=titan_base_url)
    logger.info("[Discord] Connecting...")
    await adapter.bot.start(token)
