"""
channels/whatsapp.py — WhatsApp channel adapter for Titan.

Uses ``pywa`` (WhatsApp Cloud API wrapper).  The library is an optional
dependency — if not installed the module logs a warning and
``start_whatsapp()`` becomes a no-op.

Features:
  - Unified command registry (shared with all channels)
  - Read receipt on message receive (immediate feedback)
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

_WHATSAPP_CHAR_LIMIT = 4096

try:
    from pywa import WhatsApp
    from pywa.types import Message
    _HAS_PYWA = True
except ImportError:
    _HAS_PYWA = False
    logger.info("[WhatsApp] pywa not installed — adapter disabled.")


class TitanWhatsAppBot:
    """Thin WhatsApp ↔ Titan bridge with unified command support."""

    def __init__(
        self,
        phone_id: str,
        token: str,
        verify_token: str = "",
        titan_base_url: str = "http://127.0.0.1:7777",
    ):
        if not _HAS_PYWA:
            raise RuntimeError("pywa is not installed.")

        self.titan_url = titan_base_url
        self._cmd_registry = CommandRegistry(titan_base_url=titan_base_url)

        server_args = {}
        if verify_token:
            server_args["verify_token"] = verify_token

        self.wa = WhatsApp(
            phone_id=phone_id,
            token=token,
            **server_args,
        )

        self._setup_handlers()

    # ----- Handlers -------------------------------------------------------

    def _setup_handlers(self) -> None:
        @self.wa.on_message()
        async def on_message(client: "WhatsApp", msg: "Message"):
            await self._handle_message(client, msg)

    async def _handle_message(self, client: "WhatsApp", msg: "Message") -> None:
        """Process an incoming WhatsApp text message."""
        # Only handle text messages — ignore media, location, etc.
        if not msg.text:
            return

        # Mark as read immediately (visual feedback)
        try:
            await msg.mark_as_read()
        except Exception:
            pass  # Non-critical

        wa_id = msg.from_user.wa_id
        user_id = f"wa_{wa_id}"
        text = msg.text.strip()

        # Handle slash commands
        if text.startswith("/") and self._cmd_registry.is_command(text):
            response = await self._cmd_registry.execute(text, user_id)
            chunks = split_message(response, _WHATSAPP_CHAR_LIMIT)
            for chunk in chunks:
                await msg.reply_text(chunk)
            return

        session_id = f"whatsapp_{wa_id}"

        result = await forward_to_titan(
            message=text,
            session_id=session_id,
            user_id=user_id,
            base_url=self.titan_url,
        )

        reply = format_response(result)
        chunks = split_message(reply, _WHATSAPP_CHAR_LIMIT)
        for chunk in chunks:
            await msg.reply_text(chunk)


async def start_whatsapp(config: dict, titan_base_url: str = "http://127.0.0.1:7777") -> None:
    """Boot the WhatsApp adapter from a config dict."""
    if not _HAS_PYWA:
        logger.warning("[WhatsApp] pywa not installed — skipping.")
        return

    phone_id = config.get("phone_id", "")
    token = config.get("token", "")

    if not phone_id or not token:
        logger.warning("[WhatsApp] Missing phone_id or token — skipping.")
        return

    verify_token = config.get("verify_token", "")

    TitanWhatsAppBot(
        phone_id=phone_id,
        token=token,
        verify_token=verify_token,
        titan_base_url=titan_base_url,
    )
    logger.info("[WhatsApp] Handler registered. Waiting for webhook callbacks...")
