"""
channels/telegram.py — Telegram channel adapter for Titan.

Uses ``python-telegram-bot`` (v22+, async-native).  The library is an
optional dependency — if not installed the module logs a warning and
``start_telegram()`` becomes a no-op.

Features:
  - Unified command registry (shared with all channels)
  - Typing indicator while processing
  - Auto-retry on transient errors (via forward_to_titan)
  - User-friendly error messages (never raw tracebacks)
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from titan_plugin.channels import (
    fetch_banner,
    forward_to_titan,
    format_response,
    split_message,
)
from titan_plugin.channels.commands import CommandRegistry

logger = logging.getLogger(__name__)

_TELEGRAM_CHAR_LIMIT = 4096

try:
    from telegram import BotCommand, Update
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        filters,
    )
    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False
    logger.info("[Telegram] python-telegram-bot not installed — adapter disabled.")


class TitanTelegramBot:
    """Thin Telegram ↔ Titan bridge with unified command support."""

    def __init__(self, token: str, titan_base_url: str = "http://127.0.0.1:7777"):
        if not _HAS_TELEGRAM:
            raise RuntimeError("python-telegram-bot is not installed.")
        self.token = token
        self.titan_url = titan_base_url
        self._app: Optional[Application] = None  # type: ignore[assignment]
        self._commands = CommandRegistry(titan_base_url=titan_base_url)

    # ----- Commands -------------------------------------------------------

    async def _handle_command(self, update: "Update", context) -> None:
        """Route any slash command through the unified registry."""
        if not update.message or not update.message.text:
            return

        text = update.message.text.strip()
        user_id = f"tg_{update.effective_user.id}" if update.effective_user else ""

        # Handle /start specially (Telegram convention)
        if text == "/start":
            cmd_list = self._commands.get_command_list()
            lines = [
                "Titan online. I am a sovereign AI agent with persistent memory, "
                "on-chain identity, and autonomous research capabilities.\n",
                "Send me any message to begin a conversation.\n",
                "Commands:",
            ]
            for name, desc in cmd_list:
                lines.append(f"  /{name} — {desc}")
            await update.message.reply_text("\n".join(lines))
            return

        response = await self._commands.execute(text, user_id)
        chunks = split_message(response, _TELEGRAM_CHAR_LIMIT)
        for chunk in chunks:
            await update.message.reply_text(chunk)

        # If restart was triggered, wait for health and send follow-up
        if text.startswith("/restart") and hasattr(self._commands, '_restart_pending') and self._commands._restart_pending:
            async def _send_restart_followup():
                for _ in range(13):
                    await asyncio.sleep(5)
                    if not self._commands._restart_pending:
                        result_msg = getattr(self._commands, '_restart_result', 'Titan status unknown.')
                        await update.message.reply_text(result_msg)
                        return
            asyncio.create_task(_send_restart_followup())

    # ----- Message handler ------------------------------------------------

    async def handle_message(self, update: "Update", context) -> None:
        """Forward every text message to POST /chat."""
        if not update.message or not update.message.text:
            return

        chat_id = update.effective_chat.id
        user_id = update.effective_user.id if update.effective_user else 0

        session_id = f"telegram_{chat_id}"
        uid = f"tg_{user_id}"

        # Show typing indicator immediately
        await update.effective_chat.send_action("typing")

        # Keep typing indicator alive during long processing
        typing_task = asyncio.create_task(
            self._keep_typing(update.effective_chat)
        )

        try:
            result = await forward_to_titan(
                message=update.message.text,
                session_id=session_id,
                user_id=uid,
                base_url=self.titan_url,
            )
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        banner = await fetch_banner(self.titan_url)
        reply = format_response(result, banner=banner)

        # Check for file paths in the response (art/audio generation)
        await self._send_reply_with_files(update, reply)

    async def _send_reply_with_files(self, update: "Update", reply: str) -> None:
        """Send reply text, and if it contains file paths, send those as media too."""
        import re

        # Find file paths in the response
        file_pattern = re.compile(r'(?:Art generated|Audio generated|File): (/[^\s\n"]+\.(?:png|jpg|jpeg|wav|mp3|gif))', re.IGNORECASE)
        matches = file_pattern.findall(reply)

        # Send text reply (strip raw file paths for cleaner display)
        text_reply = reply
        for match in matches:
            text_reply = text_reply.replace(match, os.path.basename(match))

        chunks = split_message(text_reply, _TELEGRAM_CHAR_LIMIT)
        for chunk in chunks:
            await update.message.reply_text(chunk)

        # Send any found files as media
        for fpath in matches:
            try:
                if not os.path.exists(fpath):
                    continue
                ext = os.path.splitext(fpath)[1].lower()
                if ext in ('.png', '.jpg', '.jpeg', '.gif'):
                    with open(fpath, 'rb') as f:
                        await update.message.reply_photo(
                            photo=f,
                            caption=f"Generated by Titan's inner vision"
                        )
                    logger.info("[Telegram] Sent image: %s", fpath)
                elif ext in ('.wav', '.mp3'):
                    with open(fpath, 'rb') as f:
                        await update.message.reply_audio(
                            audio=f,
                            caption=f"Sonified by Titan's inner hearing"
                        )
                    logger.info("[Telegram] Sent audio: %s", fpath)
            except Exception as e:
                logger.warning("[Telegram] Failed to send file %s: %s", fpath, e)

    async def handle_photo(self, update: "Update", context) -> None:
        """
        Download photo to media_queue for digestion by MediaWorker.

        Flow: TG photo → download → data/media_queue/ → MediaWorker digests
        → SENSE_VISUAL → Mind tensor → Spirit scalar enrichment.

        If a caption is provided, also forward it as a chat message so Titan
        gets the human-level context ("This is the Eiffel Tower at sunset").
        """
        if not update.message or not update.message.photo:
            return

        chat_id = update.effective_chat.id
        user_id = update.effective_user.id if update.effective_user else 0
        session_id = f"telegram_{chat_id}"
        uid = f"tg_{user_id}"

        # Get highest resolution photo
        photo = update.message.photo[-1]

        # Ensure media queue exists
        queue_dir = Path(__file__).resolve().parent.parent.parent / "data" / "media_queue"
        queue_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download the photo
            file = await context.bot.get_file(photo.file_id)
            filename = f"tg_{user_id}_{int(asyncio.get_event_loop().time())}_{photo.file_unique_id}.jpg"
            dest = queue_dir / filename
            await file.download_to_drive(str(dest))
            logger.info("[Telegram] Photo saved to media queue: %s (%dx%d)",
                       filename, photo.width, photo.height)

            # Acknowledge receipt
            await update.message.reply_text(
                "I can see your image. Let me perceive it through my senses..."
            )

            # If there's a caption, forward it as a chat message too
            caption = update.message.caption
            if caption:
                await update.effective_chat.send_action("typing")
                typing_task = asyncio.create_task(
                    self._keep_typing(update.effective_chat)
                )
                try:
                    # Include context that an image was shared
                    enriched_msg = f"[Maker shared an image with caption: \"{caption}\"]"
                    result = await forward_to_titan(
                        message=enriched_msg,
                        session_id=session_id,
                        user_id=uid,
                        base_url=self.titan_url,
                    )
                finally:
                    typing_task.cancel()
                    try:
                        await typing_task
                    except asyncio.CancelledError:
                        pass

                banner = await fetch_banner(self.titan_url)
                reply = format_response(result, banner=banner)
                chunks = split_message(reply, _TELEGRAM_CHAR_LIMIT)
                for chunk in chunks:
                    await update.message.reply_text(chunk)

        except Exception as e:
            logger.error("[Telegram] Photo download failed: %s", e)
            await update.message.reply_text("I couldn't process that image. Please try again.")

    async def _keep_typing(self, chat) -> None:
        """Resend typing action every 4s (Telegram clears it after 5s)."""
        try:
            while True:
                await asyncio.sleep(4)
                await chat.send_action("typing")
        except asyncio.CancelledError:
            pass

    # ----- Lifecycle ------------------------------------------------------

    def build(self) -> "Application":
        """Build (but do not start) the Application."""
        builder = Application.builder().token(self.token)
        app = builder.build()

        # Register all commands from the unified registry
        cmd_names = [name for name, _ in self._commands.get_all_commands()]
        for name in cmd_names:
            app.add_handler(CommandHandler(name, self._handle_command))

        # Also handle /start
        if "start" not in cmd_names:
            app.add_handler(CommandHandler("start", self._handle_command))

        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self._app = app
        return app

    async def _register_bot_commands(self, app: "Application") -> None:
        """Register command menu in Telegram UI (autocomplete)."""
        try:
            cmds = [
                BotCommand(name, desc)
                for name, desc in self._commands.get_command_list()
            ]
            await app.bot.set_my_commands(cmds)
            logger.info("[Telegram] Registered %d bot commands for autocomplete.", len(cmds))
        except Exception as e:
            logger.warning("[Telegram] Failed to register bot commands: %s", e)

    def run(self) -> None:
        """Start polling (blocking)."""
        app = self.build()
        logger.info("[Telegram] Starting polling...")
        app.run_polling(drop_pending_updates=True)


async def start_telegram(config: dict, titan_base_url: str = "http://127.0.0.1:7777") -> None:
    """Boot the Telegram adapter from a config dict."""
    if not _HAS_TELEGRAM:
        logger.warning("[Telegram] python-telegram-bot not installed — skipping.")
        return

    token = config.get("bot_token", "")
    if not token:
        logger.warning("[Telegram] No bot_token configured — skipping.")
        return

    bot = TitanTelegramBot(token=token, titan_base_url=titan_base_url)
    app = bot.build()
    logger.info("[Telegram] Initialising and starting polling...")
    await app.initialize()
    await app.start()

    # Register command autocomplete in Telegram UI
    await bot._register_bot_commands(app)

    await app.updater.start_polling(drop_pending_updates=False)
    logger.info("[Telegram] Polling active — waiting for messages.")

    # Block forever so the polling loop stays alive
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        logger.info("[Telegram] Shutting down...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
