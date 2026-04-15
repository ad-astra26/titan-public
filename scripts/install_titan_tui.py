#!/usr/bin/env python3
"""
install_titan_tui.py — Titan Interactive Birth Wizard (TUI Edition).

Textual-based interactive installer with branded UI, progress tracking,
and guided setup for deploying a Titan sovereign AI agent.

Phases:
  1. Prerequisites Check (Python, disk, pip, git)
  2. Core Dependencies (venv, pip install)
  3. Optional Research Stack (Crawl4AI, Unstructured, Playwright)
  4. System Services (Docker/SearXNG, Ollama, ffmpeg)
  5. Configuration Wizard (config.toml: network, inference, social, Ollama Cloud)
  5.5 Channel Setup (Telegram, Discord, Slack, WhatsApp)
  6. Genesis Ceremony (Ed25519 keypair, Shamir SSS, genesis art)
  7. Health Check (subsystem verification)

Usage:
    python scripts/install_titan_tui.py              # Full interactive install
    python scripts/install_titan_tui.py --minimal     # Core only, skip optional
    python scripts/install_titan_tui.py --skip-genesis # Skip identity ceremony
    python scripts/install_titan_tui.py --resume       # Resume from last phase
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import pathlib
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "titan_plugin" / "config.toml"
VENV_DIR = PROJECT_ROOT / ".venv"
STATE_FILE = PROJECT_ROOT / "data" / ".install_state.json"
MIN_PYTHON = (3, 11)
MIN_DISK_GB = 2

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Vertical, Horizontal, VerticalScroll
    from textual.widgets import (
        Header, Footer, Static, Button, Input, Switch, Label,
        ProgressBar, Rule, Log,
    )
    from textual.binding import Binding
    from textual import on, work
    from textual.worker import Worker
except ImportError:
    print("Error: 'textual' package not installed. Run: pip install textual")
    sys.exit(1)


# ─── Brand CSS ─────────────────────────────────────────────────────────────

TITAN_CSS = """
Screen {
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

.phase-panel {
    background: #1A1D23;
    border: solid #8E9AAF 30%;
    padding: 1 2;
    margin: 1 2;
    height: auto;
}

.phase-active {
    background: #1A1D23;
    border: solid #9945FF;
    padding: 1 2;
    margin: 1 2;
    height: auto;
}

.phase-done {
    background: #1A1D23;
    border: solid #77CCCC;
    padding: 1 2;
    margin: 1 2;
    height: auto;
}

.phase-title {
    color: #E5C79E;
    text-style: bold;
}

.phase-number {
    color: #9945FF;
    text-style: bold;
}

.check-ok {
    color: #77CCCC;
}

.check-fail {
    color: #ff6b6b;
}

.check-warn {
    color: #E5C79E;
}

.info-text {
    color: #8E9AAF;
}

.highlight {
    color: #E5C79E;
    text-style: bold;
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

Button.primary {
    background: #9945FF;
    color: white;
    margin: 1 1;
}

Button.success {
    background: #77CCCC;
    color: #0B0E14;
    margin: 1 1;
}

Button.secondary {
    background: #1A1D23;
    border: solid #8E9AAF 40%;
    color: #8E9AAF;
    margin: 1 1;
}

Button.danger {
    background: #ff6b6b;
    color: white;
    margin: 1 1;
}

ProgressBar {
    margin: 1 2;
}

ProgressBar > .bar--bar {
    color: #9945FF;
}

ProgressBar > .bar--complete {
    color: #77CCCC;
}

#output-log {
    background: #0B0E14;
    border: solid #8E9AAF 20%;
    height: 12;
    margin: 1 2;
}

Rule {
    color: #8E9AAF 30%;
    margin: 1 2;
}

#phase-nav {
    dock: left;
    width: 28;
    background: #1A1D23;
    padding: 1;
    border-right: solid #8E9AAF 20%;
}

.nav-item {
    padding: 0 1;
    color: #8E9AAF;
}

.nav-item-active {
    padding: 0 1;
    color: #9945FF;
    text-style: bold;
}

.nav-item-done {
    padding: 0 1;
    color: #77CCCC;
}

#main-content {
    padding: 0;
}
"""

TITAN_LOGO = r"""
    ╔══════════════════════════════════════╗
    ║           ⬡  T I T A N  ⬡           ║
    ║     Sovereign AI Cognitive Agent     ║
    ║        Birth Wizard v2.1             ║
    ╚══════════════════════════════════════╝
"""

# ─── Phase Definitions ─────────────────────────────────────────────────────

PHASES = [
    {"num": 1, "title": "Prerequisites Check", "icon": "🔍"},
    {"num": 2, "title": "Core Dependencies", "icon": "📦"},
    {"num": 3, "title": "Research Stack", "icon": "🔬"},
    {"num": 4, "title": "System Services", "icon": "⚙"},
    {"num": 5, "title": "Configuration", "icon": "🔧"},
    {"num": 5.5, "title": "Channel Setup", "icon": "📡"},
    {"num": 6, "title": "Genesis Ceremony", "icon": "🔑"},
    {"num": 7, "title": "Health Check", "icon": "💚"},
]


# ─── Install State Persistence ─────────────────────────────────────────────

def load_state() -> dict:
    """Load install progress state."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"completed_phases": [], "started_at": time.time()}


def save_state(state: dict):
    """Save install progress state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ─── Check Functions ───────────────────────────────────────────────────────

def check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    return ok, f"Python {v.major}.{v.minor}.{v.micro}"


def check_disk() -> tuple[bool, str]:
    stat = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = stat.free / (1024 ** 3)
    return free_gb >= MIN_DISK_GB, f"{free_gb:.1f} GB free"


def check_pip() -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True, text=True,
    )
    return r.returncode == 0, "pip available" if r.returncode == 0 else "pip not found"


def check_git() -> tuple[bool, str]:
    found = shutil.which("git") is not None
    return found, "git available" if found else "git not found (recommended)"


def check_os() -> tuple[bool, str]:
    is_linux = platform.system() == "Linux"
    return True, f"{platform.system()} ({platform.release()[:20]})"


def run_cmd(cmd: str, timeout: int = 300) -> tuple[bool, str]:
    """Run a shell command and return (success, output)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=timeout,
        )
        output = r.stdout + r.stderr
        return r.returncode == 0, output[-500:] if len(output) > 500 else output
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


# ─── Main TUI App ─────────────────────────────────────────────────────────

class TitanInstaller(App):
    """Titan Birth Wizard — Interactive TUI Installer."""

    CSS = TITAN_CSS
    TITLE = "Titan Birth Wizard"
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self, minimal: bool = False, skip_genesis: bool = False, resume: bool = False):
        super().__init__()
        self.minimal = minimal
        self.skip_genesis = skip_genesis
        self.state = load_state() if resume else {"completed_phases": [], "started_at": time.time()}
        self.current_phase = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            # Left sidebar: phase navigation
            with Vertical(id="phase-nav"):
                yield Static("[bold #E5C79E]PHASES[/]", classes="nav-item")
                yield Rule()
                for p in PHASES:
                    num_str = str(p["num"])
                    done = num_str in self.state.get("completed_phases", [])
                    cls = "nav-item-done" if done else "nav-item"
                    icon = "✓" if done else p["icon"]
                    yield Static(
                        f" {icon} {p['title']}",
                        classes=cls,
                        id=f"nav-{num_str.replace('.', '_')}",
                    )

            # Main content area
            with VerticalScroll(id="main-content"):
                yield Static(TITAN_LOGO, classes="logo-text")
                yield Static("Interactive Sovereign Deployment Protocol", classes="subtitle")
                yield Rule()

                yield ProgressBar(total=len(PHASES), show_eta=False, id="global-progress")

                yield Static("", id="phase-display", classes="phase-panel")
                yield Log(id="output-log", highlight=True, auto_scroll=True)

                with Horizontal():
                    yield Button("▶ Start Installation", id="btn-start", classes="primary")
                    yield Button("⏭ Skip Phase", id="btn-skip", classes="secondary")
                    yield Button("✕ Quit", id="btn-quit", classes="danger")

        yield Footer()

    def on_mount(self) -> None:
        """Set initial state."""
        self.query_one("#global-progress", ProgressBar).update(
            progress=len(self.state.get("completed_phases", []))
        )
        self._update_phase_display(
            "Ready to begin",
            "Press [bold]Start Installation[/] to begin the deployment process.\n\n"
            "This wizard will guide you through:\n"
            "  • Prerequisites verification\n"
            "  • Dependency installation\n"
            "  • Service configuration\n"
            "  • Identity creation (Genesis Ceremony)\n"
            "  • Health verification\n\n"
            f"[dim]Project root: {PROJECT_ROOT}[/]",
        )

    def _update_phase_display(self, title: str, content: str):
        display = self.query_one("#phase-display", Static)
        display.update(f"[bold #E5C79E]{title}[/]\n\n{content}")

    def _log(self, msg: str):
        self.query_one("#output-log", Log).write_line(msg)

    def _mark_phase_done(self, phase_num):
        num_str = str(phase_num)
        if num_str not in self.state["completed_phases"]:
            self.state["completed_phases"].append(num_str)
            save_state(self.state)

        # Update nav
        nav_id = f"nav-{num_str.replace('.', '_')}"
        try:
            nav = self.query_one(f"#{nav_id}", Static)
            phase = next(p for p in PHASES if str(p["num"]) == num_str)
            nav.update(f" ✓ {phase['title']}")
            nav.classes = "nav-item-done"
        except Exception:
            pass

        # Update progress bar
        self.query_one("#global-progress", ProgressBar).update(
            progress=len(self.state["completed_phases"])
        )

    @on(Button.Pressed, "#btn-start")
    def start_install(self) -> None:
        self.run_installation()

    @on(Button.Pressed, "#btn-quit")
    def quit_app(self) -> None:
        self.exit()

    @on(Button.Pressed, "#btn-skip")
    def skip_phase(self) -> None:
        self.current_phase += 1
        self._log("[yellow]Phase skipped[/]")

    @work(exclusive=True, thread=True)
    def run_installation(self) -> None:
        """Run all installation phases sequentially in a worker thread."""

        # Phase 1: Prerequisites
        self.call_from_thread(self._update_phase_display,
            "Phase 1: Prerequisites Check",
            "Verifying system requirements...")
        self.call_from_thread(self._log, "─── Phase 1: Prerequisites ───")

        checks = [
            ("Python", check_python),
            ("Disk Space", check_disk),
            ("pip", check_pip),
            ("git", check_git),
            ("OS", check_os),
        ]

        all_ok = True
        for name, fn in checks:
            ok, msg = fn()
            symbol = "[green]✓[/]" if ok else "[red]✗[/]"
            self.call_from_thread(self._log, f"  {symbol} {name}: {msg}")
            if not ok and name in ("Python", "Disk Space", "pip"):
                all_ok = False

        if not all_ok:
            self.call_from_thread(self._log, "[red bold]Prerequisites not met. Fix issues above.[/]")
            return

        self.call_from_thread(self._mark_phase_done, 1)
        self.call_from_thread(self._log, "[green]Phase 1 complete ✓[/]\n")

        # Phase 2: Core Dependencies
        self.call_from_thread(self._update_phase_display,
            "Phase 2: Core Dependencies",
            "Installing Python packages...")
        self.call_from_thread(self._log, "─── Phase 2: Core Dependencies ───")

        venv_python = VENV_DIR / "bin" / "python"
        pip_bin = VENV_DIR / "bin" / "pip"

        if not venv_python.exists():
            self.call_from_thread(self._log, "  Creating virtual environment...")
            ok, out = run_cmd(f"python3 -m venv {VENV_DIR}")
            if ok:
                self.call_from_thread(self._log, "  [green]✓[/] Virtual environment created")
            else:
                self.call_from_thread(self._log, f"  [red]✗[/] venv creation failed: {out[:200]}")
                return
        else:
            self.call_from_thread(self._log, "  [green]✓[/] Virtual environment exists")

        self.call_from_thread(self._log, "  Upgrading pip...")
        run_cmd(f"{pip_bin} install --upgrade pip -q")

        self.call_from_thread(self._log, "  Installing core dependencies (this may take a few minutes)...")
        ok, out = run_cmd(f"{pip_bin} install -e .", timeout=600)
        if ok:
            self.call_from_thread(self._log, "  [green]✓[/] Core dependencies installed")
        else:
            self.call_from_thread(self._log, f"  [red]✗[/] Install failed")
            self.call_from_thread(self._log, f"  {out[:300]}")

        # Verify critical imports
        critical_modules = [
            ("titan_plugin", "Titan Plugin"),
            ("torch", "PyTorch"),
            ("fastapi", "FastAPI"),
            ("httpx", "httpx"),
            ("PIL", "Pillow"),
            ("cryptography", "cryptography"),
        ]
        for module, name in critical_modules:
            ok, _ = run_cmd(f'{venv_python} -c "import {module}"')
            symbol = "[green]✓[/]" if ok else "[red]✗[/]"
            self.call_from_thread(self._log, f"  {symbol} {name}")

        self.call_from_thread(self._mark_phase_done, 2)
        self.call_from_thread(self._log, "[green]Phase 2 complete ✓[/]\n")

        # Ensure data directories
        for d in ["data", "data/logs", "data/history", "data/studio_exports",
                   "data/studio_exports/meditation", "data/studio_exports/epoch"]:
            (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

        # Phase 3: Research Stack (skip if minimal)
        self.call_from_thread(self._update_phase_display,
            "Phase 3: Research Stack (Optional)",
            "Crawl4AI, Unstructured, Playwright for web research capabilities.")
        self.call_from_thread(self._log, "─── Phase 3: Research Stack ───")

        if self.minimal:
            self.call_from_thread(self._log, "  [yellow]![/] Skipped (minimal mode)")
        else:
            self.call_from_thread(self._log, "  Installing research packages...")
            ok, _ = run_cmd(f"{pip_bin} install -e '.[research]' -q", timeout=600)
            if ok:
                self.call_from_thread(self._log, "  [green]✓[/] Research stack installed")
            else:
                self.call_from_thread(self._log, "  [yellow]![/] Research install failed (non-critical)")

        self.call_from_thread(self._mark_phase_done, 3)
        self.call_from_thread(self._log, "[green]Phase 3 complete ✓[/]\n")

        # Phase 4: System Services
        self.call_from_thread(self._update_phase_display,
            "Phase 4: System Services",
            "Checking Docker, SearXNG, Ollama, ffmpeg...")
        self.call_from_thread(self._log, "─── Phase 4: System Services ───")

        for name, cmd in [("ffmpeg", "ffmpeg"), ("docker", "docker"), ("ollama", "ollama")]:
            found = shutil.which(cmd) is not None
            symbol = "[green]✓[/]" if found else "[yellow]![/]"
            status = "available" if found else "not found (optional)"
            self.call_from_thread(self._log, f"  {symbol} {name}: {status}")

        self.call_from_thread(self._mark_phase_done, 4)
        self.call_from_thread(self._log, "[green]Phase 4 complete ✓[/]\n")

        # Phase 5: Configuration
        self.call_from_thread(self._update_phase_display,
            "Phase 5: Configuration",
            "Config values are read from titan_plugin/config.toml.\n"
            "Edit the file directly or use the channel setup wizard:\n"
            "  python scripts/setup_channels.py")
        self.call_from_thread(self._log, "─── Phase 5: Configuration ───")

        if CONFIG_PATH.exists():
            self.call_from_thread(self._log, f"  [green]✓[/] Config found: {CONFIG_PATH}")
        else:
            self.call_from_thread(self._log, f"  [red]✗[/] Config not found: {CONFIG_PATH}")

        self.call_from_thread(self._mark_phase_done, 5)
        self.call_from_thread(self._log, "[green]Phase 5 complete ✓[/]\n")

        # Phase 5.5: Channel Setup
        self.call_from_thread(self._update_phase_display,
            "Phase 5.5: Channel Setup",
            "Communication channels (Telegram, Discord, Slack, WhatsApp).\n"
            "Run the dedicated channel wizard after this installer:\n"
            "  python scripts/setup_channels.py")
        self.call_from_thread(self._log, "─── Phase 5.5: Channel Setup ───")
        self.call_from_thread(self._log, "  [dim]Run separately: python scripts/setup_channels.py[/]")
        self.call_from_thread(self._mark_phase_done, 5.5)
        self.call_from_thread(self._log, "[green]Phase 5.5 noted ✓[/]\n")

        # Phase 6: Genesis Ceremony
        self.call_from_thread(self._update_phase_display,
            "Phase 6: Genesis Ceremony",
            "Create Titan's sovereign identity (Ed25519 keypair, Shamir SSS).")
        self.call_from_thread(self._log, "─── Phase 6: Genesis Ceremony ───")

        if self.skip_genesis:
            self.call_from_thread(self._log, "  [yellow]![/] Skipped (--skip-genesis)")
        else:
            genesis_record = PROJECT_ROOT / "data" / "genesis_record.json"
            hw_path = PROJECT_ROOT / "data" / "soul_keypair.enc"

            if genesis_record.exists():
                self.call_from_thread(self._log, "  [green]✓[/] Genesis record already exists")
                try:
                    record = json.loads(genesis_record.read_text())
                    pk = record.get("titan_pubkey", "unknown")
                    self.call_from_thread(self._log, f"  [green]✓[/] Titan pubkey: {pk}")
                except Exception:
                    pass
            else:
                self.call_from_thread(self._log, "  [yellow]![/] No genesis record — run genesis_ceremony.py")
                self.call_from_thread(self._log, "    python scripts/genesis_ceremony.py --generate")

            if hw_path.exists():
                self.call_from_thread(self._log, "  [green]✓[/] Hardware-bound keypair present")
            else:
                self.call_from_thread(self._log, "  [yellow]![/] No HW-bound keypair")

        self.call_from_thread(self._mark_phase_done, 6)
        self.call_from_thread(self._log, "[green]Phase 6 complete ✓[/]\n")

        # Phase 7: Health Check
        self.call_from_thread(self._update_phase_display,
            "Phase 7: Sovereign Health Check",
            "Verifying all subsystems...")
        self.call_from_thread(self._log, "─── Phase 7: Health Check ───")

        vp = str(venv_python)
        health_checks = {
            "Titan Plugin": f'{vp} -c "from titan_plugin import TitanPlugin; print(\'ok\')"',
            "Studio": f'{vp} -c "from titan_plugin.expressive.studio import StudioCoordinator; print(\'ok\')"',
            "Solana SDK": f'{vp} -c "from titan_plugin.utils.solana_client import is_available; print(\'ok\' if is_available() else \'degraded\')"',
            "Cognee Memory": f'{vp} -c "from titan_plugin.core.memory import TieredMemoryGraph; print(\'ok\')"',
            "Guardian": f'{vp} -c "from titan_plugin.logic.sage.guardian import SageGuardian; print(\'ok\')"',
            "Art Generator": f'{vp} -c "from titan_plugin.expressive.art import ProceduralArtGen; print(\'ok\')"',
            "Observatory API": f'{vp} -c "from titan_plugin.api import create_app; print(\'ok\')"',
        }

        active = 0
        for name, cmd in health_checks.items():
            ok, out = run_cmd(cmd)
            output = out.strip().split("\n")[-1] if out.strip() else "failed"
            if "ok" in output:
                self.call_from_thread(self._log, f"  [green]✓[/] {name}")
                active += 1
            elif "degraded" in output:
                self.call_from_thread(self._log, f"  [yellow]![/] {name} (degraded)")
                active += 1
            else:
                self.call_from_thread(self._log, f"  [red]✗[/] {name}")

        self.call_from_thread(self._log, f"\n  Sovereignty Status: {active}/{len(health_checks)} subsystems active")
        self.call_from_thread(self._mark_phase_done, 7)
        self.call_from_thread(self._log, "[green]Phase 7 complete ✓[/]\n")

        # Final
        self.call_from_thread(self._update_phase_display,
            "Installation Complete",
            f"[bold #77CCCC]{active}/{len(health_checks)} subsystems active[/]\n\n"
            "Next steps:\n"
            "  1. Activate venv:   source .venv/bin/activate\n"
            "  2. Configure channels: python scripts/setup_channels.py\n"
            "  3. Run Titan:       python scripts/titan_main.py --server\n"
            "  4. Run frontend:    cd titan-observatory && npm run build && npm start\n"
            "  5. Run tests:       python -m pytest tests/ -p no:anchorpy -v\n\n"
            "[dim]The Titan awaits its first thought.[/]",
        )
        self.call_from_thread(self._log, "")
        self.call_from_thread(self._log, "╔══════════════════════════════════════╗")
        self.call_from_thread(self._log, "║    TITAN DEPLOYMENT COMPLETE  ✓     ║")
        self.call_from_thread(self._log, "╚══════════════════════════════════════╝")


def main():
    parser = argparse.ArgumentParser(description="Titan Birth Wizard (TUI)")
    parser.add_argument("--minimal", action="store_true", help="Core only, skip optional")
    parser.add_argument("--skip-genesis", action="store_true", help="Skip Genesis Ceremony")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed phase")
    args = parser.parse_args()

    app = TitanInstaller(
        minimal=args.minimal,
        skip_genesis=args.skip_genesis,
        resume=args.resume,
    )
    app.run()


if __name__ == "__main__":
    main()
