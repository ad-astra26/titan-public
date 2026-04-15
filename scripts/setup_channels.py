#!/usr/bin/env python3
"""
setup_channels.py — Titan Communication Channel Setup Wizard (TUI + CLI).

Interactive Textual-based wizard for configuring Titan's communication channels:
  - Telegram (via @BotFather)
  - Discord (via Developer Portal)
  - Slack (via Slack API, Socket Mode)
  - WhatsApp (via Meta Business API)

Writes configuration to titan_plugin/config.toml [channels] section.
Tests connectivity before saving credentials.

Usage:
    python scripts/setup_channels.py          # TUI mode (default)
    python scripts/setup_channels.py --cli    # CLI fallback for basic SSH sessions
"""
import argparse
import getpass
import os
import re
import sys
import asyncio
import pathlib

# Ensure project root on path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_PATH = PROJECT_ROOT / "titan_plugin" / "config.toml"

TEXTUAL_AVAILABLE = True
try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Vertical, Horizontal, VerticalScroll
    from textual.widgets import (
        Header, Footer, Static, Button, Input, Switch, Label,
        LoadingIndicator, Rule,
    )
    from textual.binding import Binding
    from textual.screen import Screen
    from textual import on
except ImportError:
    TEXTUAL_AVAILABLE = False

# ─── Brand Colors (from /brand/Titan_brand_style_tailwind_guidelines.md) ───
# Deep Space #0B0E14, Obsidian #1A1D23, Titanium Steel #8E9AAF,
# Diffused Golden Haze #E5C79E, Solana Neon #9945FF, Pastel Teal-Emerald #77CCCC

TITAN_CSS = """
Screen {
    background: #0B0E14;
}

#logo-container {
    height: auto;
    padding: 1 2;
    background: #0B0E14;
}

.logo-text {
    text-align: center;
    color: #E5C79E;
    text-style: bold;
}

.subtitle {
    text-align: center;
    color: #8E9AAF;
    margin-bottom: 1;
}

.channel-card {
    background: #1A1D23;
    border: solid #8E9AAF 30%;
    padding: 1 2;
    margin: 1 2;
    height: auto;
}

.channel-card-active {
    background: #1A1D23;
    border: solid #77CCCC;
    padding: 1 2;
    margin: 1 2;
    height: auto;
}

.channel-title {
    color: #E5C79E;
    text-style: bold;
    margin-bottom: 1;
}

.channel-desc {
    color: #8E9AAF;
    margin-bottom: 1;
}

.field-label {
    color: #8E9AAF;
    margin-top: 1;
}

.status-ok {
    color: #77CCCC;
    text-style: bold;
}

.status-fail {
    color: #ff6b6b;
    text-style: bold;
}

.status-pending {
    color: #E5C79E;
}

Input {
    background: #0B0E14;
    border: solid #8E9AAF 30%;
    color: #E5C79E;
    margin: 0 0 1 0;
}

Input:focus {
    border: solid #9945FF;
}

Button {
    margin: 1 1;
}

Button.primary {
    background: #9945FF;
    color: white;
}

Button.success {
    background: #77CCCC;
    color: #0B0E14;
}

Button.secondary {
    background: #1A1D23;
    border: solid #8E9AAF 40%;
    color: #8E9AAF;
}

Switch {
    margin: 0 1;
}

#footer-bar {
    background: #1A1D23;
    color: #8E9AAF;
    padding: 0 2;
    height: 1;
    dock: bottom;
}

#main-scroll {
    margin: 0 0;
}

Rule {
    color: #8E9AAF 30%;
    margin: 1 2;
}

.test-result {
    padding: 0 2;
    margin: 0 2 1 2;
}

#save-bar {
    height: auto;
    padding: 1 2;
    background: #1A1D23;
    dock: bottom;
}
"""

# ─── ASCII Logo ────────────────────────────────────────────────────────────
TITAN_LOGO = r"""
    ╔══════════════════════════════════════╗
    ║           ⬡  T I T A N  ⬡           ║
    ║     Sovereign AI Cognitive Agent     ║
    ╚══════════════════════════════════════╝
"""

# ─── Channel Definitions ──────────────────────────────────────────────────
CHANNELS = {
    "telegram": {
        "name": "Telegram",
        "icon": "✈",
        "description": "Connect Titan to Telegram via a Bot. Users can chat with Titan in DMs or group chats.",
        "setup_guide": "1. Open Telegram → @BotFather\n2. Send /newbot\n3. Choose a name and username\n4. Copy the bot token",
        "fields": [
            {"key": "bot_token", "label": "Bot Token", "placeholder": "123456:ABC-DEF...", "password": True},
        ],
        "pip_dep": "python-telegram-bot",
    },
    "discord": {
        "name": "Discord",
        "icon": "🎮",
        "description": "Connect Titan to a Discord server. Titan responds to messages and slash commands.",
        "setup_guide": "1. Go to discord.com/developers/applications\n2. Create New Application\n3. Bot → Add Bot → Copy Token\n4. Enable Message Content Intent\n5. Invite bot to your server with chat permissions",
        "fields": [
            {"key": "bot_token", "label": "Bot Token", "placeholder": "MTIz...abc", "password": True},
        ],
        "pip_dep": "discord.py",
    },
    "slack": {
        "name": "Slack",
        "icon": "💼",
        "description": "Connect Titan to a Slack workspace using Socket Mode for real-time messaging.",
        "setup_guide": "1. Go to api.slack.com/apps → Create New App\n2. Enable Socket Mode (get App Token: xapp-...)\n3. Add Bot Token Scopes: chat:write, app_mentions:read\n4. Install to workspace (get Bot Token: xoxb-...)",
        "fields": [
            {"key": "bot_token", "label": "Bot Token (xoxb-...)", "placeholder": "xoxb-...", "password": True},
            {"key": "app_token", "label": "App Token (xapp-...)", "placeholder": "xapp-...", "password": True},
        ],
        "pip_dep": "slack-bolt",
    },
    "whatsapp": {
        "name": "WhatsApp",
        "icon": "📱",
        "description": "Connect Titan to WhatsApp Business API for messaging via Meta or Twilio.",
        "setup_guide": "1. Go to developers.facebook.com → Create App\n2. Add WhatsApp product\n3. Get Phone Number ID and permanent token\n4. Set up webhook URL: https://yourdomain.com/whatsapp/webhook",
        "fields": [
            {"key": "phone_id", "label": "Phone Number ID", "placeholder": "1234567890", "password": False},
            {"key": "token", "label": "Access Token", "placeholder": "EAAx...", "password": True},
            {"key": "verify_token", "label": "Webhook Verify Token", "placeholder": "my_verify_token", "password": False},
        ],
        "pip_dep": "pywa",
    },
}


# ─── Config Helpers ────────────────────────────────────────────────────────

def load_config_value(key: str) -> str:
    """Read a single value from config.toml by key."""
    if not CONFIG_PATH.exists():
        return ""
    content = CONFIG_PATH.read_text()
    match = re.search(rf'^{re.escape(key)}\s*=\s*"([^"]*)"', content, re.MULTILINE)
    return match.group(1) if match else ""


def save_config_value(key: str, value: str):
    """Write a single key = "value" in config.toml."""
    if not CONFIG_PATH.exists():
        return
    content = CONFIG_PATH.read_text()
    pattern = rf'^({re.escape(key)}\s*=\s*)"[^"]*"'
    replacement = rf'\1"{value}"'
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    CONFIG_PATH.write_text(new_content)


def save_config_bool(key: str, value: bool):
    """Write a single key = true/false in config.toml."""
    if not CONFIG_PATH.exists():
        return
    content = CONFIG_PATH.read_text()
    val_str = "true" if value else "false"
    pattern = rf'^({re.escape(key)}\s*=\s*)(true|false)'
    replacement = rf'\1{val_str}'
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    CONFIG_PATH.write_text(new_content)


async def test_channel_connectivity(channel_key: str, config_values: dict) -> tuple[bool, str]:
    """Test if channel credentials are valid. Returns (success, message)."""
    try:
        if channel_key == "telegram":
            import httpx
            token = config_values.get("bot_token", "")
            if not token:
                return False, "No token provided"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"https://api.telegram.org/bot{token}/getMe")
                if resp.status_code == 200:
                    data = resp.json()
                    bot_name = data.get("result", {}).get("username", "unknown")
                    return True, f"Connected as @{bot_name}"
                return False, f"HTTP {resp.status_code}: Invalid token"

        elif channel_key == "discord":
            import httpx
            token = config_values.get("bot_token", "")
            if not token:
                return False, "No token provided"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://discord.com/api/v10/users/@me",
                    headers={"Authorization": f"Bot {token}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return True, f"Connected as {data.get('username', 'unknown')}#{data.get('discriminator', '0')}"
                return False, f"HTTP {resp.status_code}: Invalid token"

        elif channel_key == "slack":
            import httpx
            bot_token = config_values.get("bot_token", "")
            if not bot_token:
                return False, "No bot token provided"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    "https://slack.com/api/auth.test",
                    headers={"Authorization": f"Bearer {bot_token}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        return True, f"Connected to {data.get('team', 'unknown')} as {data.get('user', 'unknown')}"
                    return False, f"Slack error: {data.get('error', 'unknown')}"
                return False, f"HTTP {resp.status_code}"

        elif channel_key == "whatsapp":
            import httpx
            phone_id = config_values.get("phone_id", "")
            token = config_values.get("token", "")
            if not phone_id or not token:
                return False, "Missing phone_id or token"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"https://graph.facebook.com/v18.0/{phone_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    number = data.get("display_phone_number", phone_id)
                    return True, f"Connected: {number}"
                return False, f"HTTP {resp.status_code}: Check credentials"

        return False, "Unknown channel"
    except ImportError as e:
        return False, f"Missing dependency: {e}"
    except Exception as e:
        return False, f"Connection error: {e}"


# ─── TUI Widgets ───────────────────────────────────────────────────────────

class ChannelCard(Vertical):
    """A single channel configuration card."""

    def __init__(self, channel_key: str, **kwargs):
        super().__init__(**kwargs)
        self.channel_key = channel_key
        self.channel = CHANNELS[channel_key]
        self.field_inputs: dict[str, Input] = {}

    def compose(self) -> ComposeResult:
        ch = self.channel
        config_prefix = f"{self.channel_key}_"

        # Check if currently enabled
        enabled = load_config_value(f"{self.channel_key}_enabled") == "" and False
        enabled_raw = load_config_value(f"{self.channel_key}_bot_token") or load_config_value(f"{self.channel_key}_token")

        yield Static(f"{ch['icon']}  {ch['name']}", classes="channel-title")
        yield Static(ch["description"], classes="channel-desc")

        with Horizontal():
            yield Label("Enable:", classes="field-label")
            yield Switch(value=False, id=f"switch-{self.channel_key}")

        yield Static(f"[dim]{ch['setup_guide']}[/dim]", classes="channel-desc")

        for field in ch["fields"]:
            full_key = f"{config_prefix}{field['key']}"
            existing = load_config_value(full_key)
            yield Label(field["label"], classes="field-label")
            inp = Input(
                placeholder=field["placeholder"],
                password=field.get("password", False),
                value=existing,
                id=f"input-{self.channel_key}-{field['key']}",
            )
            self.field_inputs[field["key"]] = inp
            yield inp

        yield Static(f"Requires: pip install {ch['pip_dep']}", classes="channel-desc")

        with Horizontal():
            yield Button("Test Connection", id=f"test-{self.channel_key}", classes="secondary")
            yield Button(f"Save {ch['name']}", id=f"save-{self.channel_key}", classes="primary")

        yield Static("", id=f"result-{self.channel_key}", classes="test-result")

    def get_values(self) -> dict:
        """Collect current input values."""
        values = {}
        for field in self.channel["fields"]:
            inp = self.query_one(f"#input-{self.channel_key}-{field['key']}", Input)
            values[field["key"]] = inp.value
        return values

    def get_enabled(self) -> bool:
        return self.query_one(f"#switch-{self.channel_key}", Switch).value


# ─── Main App ──────────────────────────────────────────────────────────────

class TitanChannelSetup(App):
    """Titan Communication Channel Setup Wizard."""

    CSS = TITAN_CSS
    TITLE = "Titan Channel Setup"
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+s", "save_all", "Save All", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with VerticalScroll(id="main-scroll"):
            yield Static(TITAN_LOGO, classes="logo-text")
            yield Static("Communication Channel Setup Wizard", classes="subtitle")
            yield Rule()

            for ch_key in CHANNELS:
                card = ChannelCard(ch_key, classes="channel-card", id=f"card-{ch_key}")
                yield card
                yield Rule()

            yield Static(
                "\n  After saving, start channels with:  python scripts/start_channels.py\n",
                classes="subtitle",
            )

        yield Footer()

    @on(Button.Pressed)
    async def handle_button(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        if btn_id.startswith("test-"):
            channel_key = btn_id[5:]
            card = self.query_one(f"#card-{channel_key}", ChannelCard)
            result_widget = self.query_one(f"#result-{channel_key}", Static)
            result_widget.update("[dim]Testing connection...[/dim]")

            values = card.get_values()
            success, message = await test_channel_connectivity(channel_key, values)

            if success:
                result_widget.update(f"[green bold]✓ {message}[/green bold]")
                card.classes = "channel-card-active"
            else:
                result_widget.update(f"[red bold]✗ {message}[/red bold]")

        elif btn_id.startswith("save-"):
            channel_key = btn_id[5:]
            card = self.query_one(f"#card-{channel_key}", ChannelCard)
            values = card.get_values()
            enabled = card.get_enabled()

            # Save to config.toml
            prefix = f"{channel_key}_"
            save_config_bool(f"{prefix}enabled", enabled)
            for field_key, field_value in values.items():
                save_config_value(f"{prefix}{field_key}", field_value)

            result_widget = self.query_one(f"#result-{channel_key}", Static)
            ch_name = CHANNELS[channel_key]["name"]
            if enabled:
                result_widget.update(f"[green bold]✓ {ch_name} saved and enabled[/green bold]")
            else:
                result_widget.update(f"[dim]✓ {ch_name} saved (disabled)[/dim]")

    def action_save_all(self) -> None:
        """Save all channels at once."""
        for ch_key in CHANNELS:
            card = self.query_one(f"#card-{ch_key}", ChannelCard)
            values = card.get_values()
            enabled = card.get_enabled()
            prefix = f"{ch_key}_"
            save_config_bool(f"{prefix}enabled", enabled)
            for field_key, field_value in values.items():
                save_config_value(f"{prefix}{field_key}", field_value)

        self.notify("All channels saved to config.toml", title="Saved", severity="information")


# ─── ANSI Color Helpers (CLI mode) ────────────────────────────────────────

class _ANSI:
    """Minimal ANSI color helpers for the CLI fallback."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GOLD    = "\033[38;5;222m"   # approximate #E5C79E
    TEAL    = "\033[38;5;80m"    # approximate #77CCCC
    PURPLE  = "\033[38;5;135m"   # approximate #9945FF
    RED     = "\033[38;5;203m"   # approximate #ff6b6b
    GREY    = "\033[38;5;146m"   # approximate #8E9AAF
    GREEN   = "\033[32m"

    @classmethod
    def gold(cls, text: str) -> str:
        return f"{cls.GOLD}{cls.BOLD}{text}{cls.RESET}"

    @classmethod
    def teal(cls, text: str) -> str:
        return f"{cls.TEAL}{cls.BOLD}{text}{cls.RESET}"

    @classmethod
    def red(cls, text: str) -> str:
        return f"{cls.RED}{cls.BOLD}{text}{cls.RESET}"

    @classmethod
    def dim(cls, text: str) -> str:
        return f"{cls.DIM}{text}{cls.RESET}"

    @classmethod
    def purple(cls, text: str) -> str:
        return f"{cls.PURPLE}{cls.BOLD}{text}{cls.RESET}"

    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.GREEN}{cls.BOLD}{text}{cls.RESET}"

    @classmethod
    def header(cls, text: str) -> str:
        width = 50
        line = "═" * width
        return (
            f"\n{cls.GOLD}╔{line}╗{cls.RESET}\n"
            f"{cls.GOLD}║{text:^{width}}║{cls.RESET}\n"
            f"{cls.GOLD}╚{line}╝{cls.RESET}\n"
        )


# ─── CLI Fallback Mode ───────────────────────────────────────────────────

class CLIChannelSetup:
    """Plain-terminal fallback for channel setup (no Textual dependency)."""

    def __init__(self):
        self.results: dict[str, dict] = {}

    def run(self):
        """Run the interactive CLI setup wizard."""
        C = _ANSI
        print(C.header("T I T A N  —  Channel Setup"))
        print(C.dim("  CLI mode — configure communication channels\n"))
        print(C.dim(f"  Config file: {CONFIG_PATH}\n"))
        print(C.dim("  Channels: Telegram, Discord, Slack, WhatsApp"))
        print(C.dim("  Press Ctrl-C at any time to abort.\n"))

        for ch_key, ch in CHANNELS.items():
            self._setup_channel(ch_key, ch)

        self._print_summary()
        print(C.dim("\n  After saving, start channels with:  python scripts/start_channels.py\n"))

    def _setup_channel(self, ch_key: str, ch: dict):
        C = _ANSI
        separator = C.dim("─" * 54)
        print(separator)
        print(f"\n  {ch['icon']}  {C.gold(ch['name'])}")
        print(f"  {C.dim(ch['description'])}\n")

        # Show setup guide
        print(f"  {C.purple('Setup Guide:')}")
        for guide_line in ch["setup_guide"].split("\n"):
            print(f"    {C.dim(guide_line)}")
        pip_dep = ch["pip_dep"]
        print(f"  {C.dim(f'Requires: pip install {pip_dep}')}\n")

        # Ask enable/disable
        enable_str = self._prompt(f"  Enable {ch['name']}? [y/N]: ").strip().lower()
        enabled = enable_str in ("y", "yes")

        if not enabled:
            ch_name = ch["name"]
            print(f"  {C.dim(f'{ch_name} skipped.')}\n")
            self.results[ch_key] = {"enabled": False, "tested": False, "test_ok": False}
            return

        # Collect field values
        config_prefix = f"{ch_key}_"
        field_values: dict[str, str] = {}

        for field in ch["fields"]:
            full_key = f"{config_prefix}{field['key']}"
            existing = load_config_value(full_key)
            existing_hint = ""
            if existing:
                if field.get("password", False):
                    existing_hint = f" [current: {'*' * min(len(existing), 8)}...{existing[-4:] if len(existing) > 4 else ''}]"
                else:
                    existing_hint = f" [current: {existing}]"

            prompt_text = f"  {field['label']}{existing_hint}: "

            if field.get("password", False):
                value = getpass.getpass(prompt_text)
            else:
                value = self._prompt(prompt_text).strip()

            # Keep existing value if user left it blank
            if not value and existing:
                value = existing
                print(f"    {C.dim('(keeping existing value)')}")

            field_values[field["key"]] = value

        # Test connectivity
        test_str = self._prompt(f"\n  Test {ch['name']} connection? [Y/n]: ").strip().lower()
        test_ok = False
        tested = False

        if test_str not in ("n", "no"):
            tested = True
            print(f"  {C.dim('Testing connection...')}", end="", flush=True)
            try:
                success, message = asyncio.run(
                    test_channel_connectivity(ch_key, field_values)
                )
            except Exception as e:
                success, message = False, f"Error: {e}"

            if success:
                print(f"\r  {C.green('✓')} {C.teal(message)}")
                test_ok = True
            else:
                print(f"\r  {C.red('✗')} {C.red(message)}")

        # Save
        save_str = self._prompt(f"\n  Save {ch['name']} config? [Y/n]: ").strip().lower()
        if save_str not in ("n", "no"):
            save_config_bool(f"{config_prefix}enabled", enabled)
            for field_key, field_value in field_values.items():
                save_config_value(f"{config_prefix}{field_key}", field_value)
            print(f"  {C.green('✓')} {C.gold(ch['name'])} saved to config.toml\n")
            self.results[ch_key] = {"enabled": True, "tested": tested, "test_ok": test_ok, "saved": True}
        else:
            ch_name = ch["name"]
            print(f"  {C.dim(f'{ch_name} not saved.')}\n")
            self.results[ch_key] = {"enabled": True, "tested": tested, "test_ok": test_ok, "saved": False}

    def _print_summary(self):
        C = _ANSI
        print(C.header("Setup Summary"))
        for ch_key, ch in CHANNELS.items():
            r = self.results.get(ch_key, {})
            name = ch["name"]
            if not r.get("enabled"):
                status = C.dim("skipped")
            elif not r.get("saved"):
                status = C.red("not saved")
            elif r.get("tested") and r.get("test_ok"):
                status = C.green("✓ saved & verified")
            elif r.get("tested") and not r.get("test_ok"):
                status = f"{C.gold('saved')} {C.red('(test failed)')}"
            else:
                status = C.gold("saved (not tested)")
            print(f"  {ch['icon']}  {C.gold(name):30s} {status}")

    @staticmethod
    def _prompt(text: str) -> str:
        """Wrapper around input() for testability."""
        try:
            return input(text)
        except EOFError:
            return ""


# ─── Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Titan Communication Channel Setup Wizard",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use plain CLI mode instead of Textual TUI (for basic SSH sessions)",
    )
    args = parser.parse_args()

    if args.cli:
        CLIChannelSetup().run()
        return

    if not TEXTUAL_AVAILABLE:
        print(
            f"{_ANSI.gold('Warning:')} Textual TUI not available "
            f"(pip install textual). Falling back to CLI mode.\n"
        )
        CLIChannelSetup().run()
        return

    app = TitanChannelSetup()
    app.run()


if __name__ == "__main__":
    main()
