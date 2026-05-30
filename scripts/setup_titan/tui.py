"""W1.b.2 — branded Textual front-end for the install wizard.

The TUI is a *guided question collector*, not a second engine: it gathers every
answer the install needs in one branded, full-screen form (live-validated with
the SAME validators the phase bodies use), then hands control back to
``cmd_install`` which runs the proven phase walker with a
:class:`~scripts.setup_titan.prompts.ScriptedPrompter` seeded from those answers.

Why collect-then-execute instead of running the walker inside a Textual log:
the install does a multi-minute ``pip install`` and (on mainnet) a one-time
Shard-1 Genesis ceremony that MUST touch the real terminal. Those belong on a
scrollable, copy-pasteable tty — not trapped in a widget. The TUI shines for the
Q&A; the trusted CLI does the work. This is RFP decision #11.

``run_install_tui()`` returns ``(mode, answers, state_seed)`` or ``None`` if the
user quit. ``answers`` is the ScriptedPrompter dict; ``state_seed`` pre-fills the
install_state (maker wallet / RPC) so Phase 2 short-circuits its prompts.
"""
from __future__ import annotations

from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (Button, Footer, Header, Input, Label, RadioButton,
                             RadioSet, Static, Switch)

from .comms import TELEGRAM_TOKEN_RE, TWITTERAPI_KEY_RE, WEBSHARE_URL_RE
from .inference import is_ollama_cloud_key, is_openrouter_key, ollama_alive, OLLAMA_DEFAULT_HOST
from .modes import Mode, spec_for
from .phases import looks_like_solana_pubkey
from .ui import TITAN_CSS

# Result type: (mode, scripted-answers, state-seed) | None
TuiResult = Optional[tuple[Mode, dict, dict]]

_BANNER = """\
████████╗██╗████████╗ █████╗ ███╗   ██╗
╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║
   ██║   ██║   ██║   ███████║██╔██╗ ██║
   ██║   ██║   ██║   ██║  ██║██║ ╚████║
   ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝"""


class _Field(VerticalScroll):
    """A labelled card grouping one question's widgets (visibility-toggleable)."""


class InstallWizard(App):
    """Single-screen branded install form. Validates on Begin; returns answers."""

    CSS = TITAN_CSS + """
    #banner { color: $text; text-style: bold; content-align: center middle; height: auto; }
    .qcard { background: #1A1D23; border: solid #9945FF; padding: 1 2; margin: 1 2; height: auto; }
    .qtitle { color: #E5C79E; text-style: bold; }
    .qhelp { color: #8E9AAF; }
    .qerror { color: #ff6b6b; text-style: bold; height: auto; }
    Input { margin: 1 0; }
    RadioSet { height: auto; }
    #actions { height: auto; align: center middle; padding: 1; }
    Button { margin: 0 2; }
    """
    BINDINGS = [("ctrl+q", "quit", "Quit"), ("escape", "quit", "Quit")]
    TITLE = "Titan — Sovereign Agent Setup"

    def __init__(self) -> None:
        super().__init__()
        self._result: TuiResult = None
        self._ollama_alive = ollama_alive()

    # ── layout ────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(_BANNER, id="banner")
            yield Static("Stand up your sovereign Titan — answer a few questions, then we install.",
                         classes="qhelp")

            # 1 — Mode
            with _Field(classes="qcard"):
                yield Label("1 · Setup mode", classes="qtitle")
                with RadioSet(id="mode"):
                    for m in Mode:
                        spec = spec_for(m)
                        yield RadioButton(f"{spec.label} — {spec.one_liner}",
                                          value=(m == Mode.DEVNET), id=f"mode_{m.value}")

            # 2 — Wallet + RPC (modes 1/2)
            with _Field(classes="qcard", id="onchain"):
                yield Label("2 · On-chain identity", classes="qtitle")
                yield Static("Mainnet/devnet mint a real on-chain identity — they need your "
                             "wallet + a Solana RPC URL.", classes="qhelp")
                yield Input(placeholder="Maker wallet (Solana pubkey, base58)", id="wallet")
                yield Input(placeholder="Solana RPC URL", id="rpc")

            # 3 — Inference
            with _Field(classes="qcard"):
                yield Label("3 · Inference provider", classes="qtitle")
                with RadioSet(id="inference"):
                    if self._ollama_alive:
                        yield RadioButton(f"Local Ollama (detected at {OLLAMA_DEFAULT_HOST}) — most sovereign, no key",
                                          value=True, id="inf_local")
                    yield RadioButton("Ollama Cloud (ollama.com — hosted, the fleet default)",
                                      value=not self._ollama_alive, id="inf_ollama_cloud")
                    yield RadioButton("OpenRouter (openrouter.ai — many models)", id="inf_openrouter")
                yield Input(placeholder="API key (for the hosted provider)", password=True, id="inf_key")

            # 4 — Telegram (required)
            with _Field(classes="qcard"):
                yield Label("4 · Telegram bot token (required)", classes="qtitle")
                yield Static("Titan is always reachable via /chat on Telegram. Get a token from "
                             "@BotFather → /newbot.", classes="qhelp")
                yield Input(placeholder="<numeric_id>:<alphanumeric secret>", password=True, id="telegram")

            # 5 — X (opt-in)
            with _Field(classes="qcard"):
                with Horizontal():
                    yield Switch(value=False, id="x_on")
                    yield Label("5 · Enable X (Twitter) posting  —  needs twitterapi.io + Webshare",
                                classes="qtitle")
                with _Field(id="x_fields"):
                    yield Input(placeholder="twitterapi.io API key (UUID)", password=True, id="x_key")
                    yield Input(placeholder="Webshare static URL (http://user:pass@host:port/)",
                                password=True, id="x_url")

            yield Static("", id="error", classes="qerror")
            with Horizontal(id="actions"):
                yield Button("Begin install", variant="success", id="begin")
                yield Button("Quit", variant="error", id="quit")
        yield Footer()

    def on_mount(self) -> None:
        self._sync_visibility()

    # ── reactivity ──────────────────────────────────────────────────────────
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self._sync_visibility()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self._sync_visibility()

    def _selected_mode(self) -> Mode:
        rs = self.query_one("#mode", RadioSet)
        idx = rs.pressed_index if rs.pressed_index >= 0 else 0
        return list(Mode)[idx]

    def _inference_choice(self) -> str:
        """'local' | 'ollama_cloud' | 'openrouter'."""
        rs = self.query_one("#inference", RadioSet)
        btn = rs.pressed_button
        if btn is None:
            return "local" if self._ollama_alive else "ollama_cloud"
        return {"inf_local": "local", "inf_ollama_cloud": "ollama_cloud",
                "inf_openrouter": "openrouter"}[btn.id]

    def _sync_visibility(self) -> None:
        on_chain = spec_for(self._selected_mode()).genesis_on_chain
        self.query_one("#onchain").display = on_chain
        # API key only for hosted providers
        self.query_one("#inf_key").display = self._inference_choice() != "local"
        self.query_one("#x_fields").display = self.query_one("#x_on", Switch).value

    # ── submit ────────────────────────────────────────────────────────────
    def _fail(self, msg: str) -> None:
        self.query_one("#error", Static).update(f"✗ {msg}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.exit(None)
            return
        if event.button.id != "begin":
            return

        mode = self._selected_mode()
        answers: dict = {}
        state_seed: dict = {}

        # On-chain creds
        if spec_for(mode).genesis_on_chain:
            wallet = self.query_one("#wallet", Input).value.strip()
            rpc = self.query_one("#rpc", Input).value.strip()
            if not looks_like_solana_pubkey(wallet):
                return self._fail("Maker wallet must be a Solana pubkey (32–44 base58 chars).")
            if not rpc.startswith(("http://", "https://")):
                return self._fail("Solana RPC URL must start with http:// or https://.")
            state_seed["maker_wallet"] = wallet
            state_seed["solana_rpc"] = rpc

        # Inference
        choice = self._inference_choice()
        key = self.query_one("#inf_key", Input).value.strip()
        if choice == "local":
            if self._ollama_alive:
                answers["use_local_ollama"] = True
        else:
            if self._ollama_alive:
                answers["use_local_ollama"] = False
            if choice == "openrouter":
                if not is_openrouter_key(key):
                    return self._fail("OpenRouter key should start with 'sk-or-' (+≥8 more chars).")
                answers["inference_provider_choice"] = "2"
                answers["openrouter_key"] = key
            else:
                if not is_ollama_cloud_key(key):
                    return self._fail("Ollama Cloud key should be a long token (≥20 chars, no spaces).")
                answers["inference_provider_choice"] = "1"
                answers["ollama_cloud_key"] = key

        # Telegram (required)
        token = self.query_one("#telegram", Input).value.strip()
        if not TELEGRAM_TOKEN_RE.match(token):
            return self._fail("Telegram token format: '<numeric_id>:<alphanumeric>' (≥30-char secret).")
        answers["telegram_bot_token"] = token

        # X (opt-in)
        x_on = self.query_one("#x_on", Switch).value
        answers["enable_x"] = x_on
        if x_on:
            x_key = self.query_one("#x_key", Input).value.strip()
            x_url = self.query_one("#x_url", Input).value.strip()
            if not TWITTERAPI_KEY_RE.match(x_key):
                return self._fail("twitterapi.io key must be UUID format (8-4-4-4-12 hex).")
            if not WEBSHARE_URL_RE.match(x_url):
                return self._fail("Webshare URL format: http://user:pass@host:port/")
            answers["twitterapi_key"] = x_key
            answers["webshare_url"] = x_url

        self._result = (mode, answers, state_seed)
        self.exit(self._result)


def run_install_tui() -> TuiResult:
    """Run the branded wizard; return (mode, answers, state_seed) or None if quit."""
    return InstallWizard().run()
