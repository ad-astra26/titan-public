"""
channels/terminal.py — Terminal chat adapter for Titan.

Bridges stdin/stdout to Titan's /chat API using the unified command
registry.  Allows makers to chat with Titan directly from SSH sessions
without needing Telegram or any external client.

Features:
  - Unified command registry (shared with all channels)
  - "Thinking..." indicator while waiting for response
  - Colored ANSI output (gold Titan name, cyan banner)
  - Graceful Ctrl+C / /quit / /exit handling
  - Configurable user ID via --user-id flag
"""
import argparse
import asyncio
import logging
import os
import sys

from titan_plugin.channels import (
    fetch_banner,
    forward_to_titan,
    format_response,
)
from titan_plugin.channels.commands import CommandRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------
_GOLD = "\033[33m"       # Titan brand — gold/yellow
_CYAN = "\033[36m"       # Banner / info
_DIM = "\033[2m"         # Thinking indicator
_BOLD = "\033[1m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Terminal adapter
# ---------------------------------------------------------------------------
class TitanTerminalChat:
    """Interactive terminal ↔ Titan bridge with unified command support."""

    def __init__(
        self,
        user_id: str = "terminal_maker",
        titan_base_url: str = "http://127.0.0.1:7777",
    ):
        self.user_id = user_id
        self.titan_url = titan_base_url
        self.session_id = f"terminal_{os.getpid()}"
        self._commands = CommandRegistry(titan_base_url=titan_base_url)
        self._running = True

    # ----- Display helpers ------------------------------------------------

    @staticmethod
    def _print_header() -> None:
        print(f"\n{_BOLD}{_GOLD}╔══════════════════════════════════════╗{_RESET}")
        print(f"{_BOLD}{_GOLD}║          T I T A N   C H A T        ║{_RESET}")
        print(f"{_BOLD}{_GOLD}╚══════════════════════════════════════╝{_RESET}")
        print(f"{_DIM}  Type /commands for help, /quit to exit{_RESET}")
        print()

    @staticmethod
    def _print_titan(text: str, banner: str = "") -> None:
        if banner:
            print(f"\n{_CYAN}{banner}{_RESET}")
        print(f"\n{_BOLD}{_GOLD}Titan > {_RESET}{text}\n")

    @staticmethod
    def _print_error(text: str) -> None:
        print(f"\n{_BOLD}{_GOLD}Titan > {_RESET}{text}\n")

    # ----- Command handling -----------------------------------------------

    async def _handle_input(self, text: str) -> None:
        """Process a single line of user input."""
        stripped = text.strip()
        if not stripped:
            return

        # Local exit commands (not part of unified registry)
        if stripped.lower() in ("/quit", "/exit"):
            print(f"\n{_DIM}Goodbye.{_RESET}\n")
            self._running = False
            return

        # Slash commands via unified registry
        if self._commands.is_command(stripped):
            response = await self._commands.execute(stripped, self.user_id)
            self._print_titan(response)
            return

        # Regular message — forward to Titan API
        sys.stdout.write(f"{_DIM}Thinking...{_RESET}")
        sys.stdout.flush()

        result = await forward_to_titan(
            message=stripped,
            session_id=self.session_id,
            user_id=self.user_id,
            base_url=self.titan_url,
        )

        # Clear "Thinking..." line
        sys.stdout.write("\r" + " " * 20 + "\r")
        sys.stdout.flush()

        banner = await fetch_banner(self.titan_url)
        reply = format_response(result, banner=banner)
        self._print_titan(reply, banner="")

    # ----- Main loop ------------------------------------------------------

    async def run(self) -> None:
        """Async main loop — reads stdin, dispatches to handlers."""
        self._print_header()

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                line = await loop.run_in_executor(
                    None, lambda: input(f"{_BOLD}You > {_RESET}"),
                )
            except (EOFError, KeyboardInterrupt):
                print(f"\n{_DIM}Goodbye.{_RESET}\n")
                break

            try:
                await self._handle_input(line)
            except Exception as exc:
                logger.error("[Terminal] Error processing input: %s", exc)
                self._print_error(f"[Error] {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Titan Terminal Chat — talk to Titan from your terminal.",
    )
    parser.add_argument(
        "--user-id",
        default="terminal_maker",
        help="User ID sent with each message (default: terminal_maker)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:7777",
        help="Titan API base URL (default: http://127.0.0.1:7777)",
    )
    args = parser.parse_args()

    chat = TitanTerminalChat(user_id=args.user_id, titan_base_url=args.url)

    try:
        asyncio.run(chat.run())
    except KeyboardInterrupt:
        print(f"\n{_DIM}Goodbye.{_RESET}\n")


if __name__ == "__main__":
    main()
