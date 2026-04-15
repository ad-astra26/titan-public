#!/usr/bin/env python3
"""
install_titan.py — The Titan Interactive Birth Wizard.

Guides the Maker through a complete Titan deployment:
  Phase 1: Prerequisites & Environment
  Phase 2: Core Dependency Installation
  Phase 3: Optional Research Stack
  Phase 4: System Services (Docker/SearXNG, Ollama, ffmpeg)
  Phase 5: Configuration Wizard (config.toml generation)
  Phase 6: Genesis Ceremony (identity creation)
  Phase 7: Health Check (full subsystem verification)

Usage:
    python scripts/install_titan.py              # Full interactive install
    python scripts/install_titan.py --minimal    # Core only, skip optional
    python scripts/install_titan.py --skip-genesis  # Skip identity ceremony
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "titan_plugin", "config.toml")
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
MIN_PYTHON = (3, 11)
MIN_DISK_GB = 2  # Minimum free disk space for core install


# ---------------------------------------------------------------------------
# Terminal Helpers
# ---------------------------------------------------------------------------
class Colors:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


def banner(text: str):
    width = 70
    print(f"\n{Colors.CYAN}{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}{Colors.RESET}\n")


def phase(n: int, title: str):
    print(f"\n{Colors.BOLD}{'─' * 60}")
    print(f"  Phase {n}: {title}")
    print(f"{'─' * 60}{Colors.RESET}\n")


def ok(msg: str):
    print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")


def warn(msg: str):
    print(f"  {Colors.YELLOW}!{Colors.RESET} {msg}")


def fail(msg: str):
    print(f"  {Colors.RED}✗{Colors.RESET} {msg}")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"  {Colors.BOLD}>{Colors.RESET} {prompt}{suffix}: ").strip()
    return answer if answer else default


def ask_yn(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    answer = input(f"  {Colors.BOLD}>{Colors.RESET} {prompt}{suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def run(cmd: str, check: bool = True, capture: bool = False, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a shell command, streaming output unless capture=True."""
    kwargs = {
        "shell": True,
        "cwd": PROJECT_ROOT,
        "timeout": timeout,
    }
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        if capture:
            fail(f"Command failed: {cmd}")
            if result.stderr:
                print(f"    {result.stderr[:200]}")
        return result
    return result


# ---------------------------------------------------------------------------
# Phase 1: Prerequisites & Environment
# ---------------------------------------------------------------------------
def check_prerequisites() -> bool:
    phase(1, "Prerequisites Check")
    all_good = True

    # Python version
    v = sys.version_info
    if (v.major, v.minor) >= MIN_PYTHON:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} — requires {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+")
        all_good = False

    # OS
    if platform.system() == "Linux":
        ok(f"OS: {platform.system()} ({platform.release()})")
    else:
        warn(f"OS: {platform.system()} — Titan is tested on Linux (Ubuntu). Proceed with caution.")

    # Disk space
    stat = shutil.disk_usage(PROJECT_ROOT)
    free_gb = stat.free / (1024 ** 3)
    if free_gb >= MIN_DISK_GB:
        ok(f"Disk: {free_gb:.1f} GB free")
    else:
        fail(f"Disk: {free_gb:.1f} GB free — need at least {MIN_DISK_GB} GB")
        all_good = False

    # pip
    r = run("python3 -m pip --version", capture=True, check=False)
    if r.returncode == 0:
        ok("pip available")
    else:
        fail("pip not found — install python3-pip")
        all_good = False

    # git
    if shutil.which("git"):
        ok("git available")
    else:
        warn("git not found — recommended for version control")

    return all_good


# ---------------------------------------------------------------------------
# Phase 2: Core Dependencies
# ---------------------------------------------------------------------------
def install_core():
    phase(2, "Core Dependency Installation")

    # Create venv if needed
    venv_python = os.path.join(VENV_DIR, "bin", "python")
    if not os.path.exists(venv_python):
        print("  Creating virtual environment (.venv)...")
        run(f"python3 -m venv {VENV_DIR}")
        ok("Virtual environment created")
    else:
        ok("Virtual environment exists")

    pip = os.path.join(VENV_DIR, "bin", "pip")

    # Upgrade pip
    print("  Upgrading pip...")
    run(f"{pip} install --upgrade pip -q")

    # Install core package
    print("  Installing Titan core dependencies (this may take a few minutes)...")
    result = run(f"{pip} install -e .", check=False)
    if result.returncode == 0:
        ok("Core dependencies installed")
    else:
        fail("Core install failed — check output above")
        return False

    # Verify critical imports
    python = os.path.join(VENV_DIR, "bin", "python")
    checks = [
        ("titan_plugin", "Titan plugin"),
        ("torch", "PyTorch"),
        ("fastapi", "FastAPI"),
        ("httpx", "httpx"),
        ("PIL", "Pillow"),
        ("cryptography", "cryptography"),
    ]
    for module, name in checks:
        r = run(f'{python} -c "import {module}"', capture=True, check=False)
        if r.returncode == 0:
            ok(f"{name} importable")
        else:
            fail(f"{name} import failed")

    return True


# ---------------------------------------------------------------------------
# Phase 3: Optional Research Stack
# ---------------------------------------------------------------------------
def install_research(minimal: bool = False):
    phase(3, "Research Stack (Optional)")

    if minimal:
        warn("Skipped (--minimal mode). Install later: pip install -e '.[research]'")
        return

    if not ask_yn("Install Stealth-Sage research stack? (~500MB: Crawl4AI, Playwright, Unstructured)"):
        warn("Skipped. Install later: pip install -e '.[research]'")
        return

    pip = os.path.join(VENV_DIR, "bin", "pip")
    python = os.path.join(VENV_DIR, "bin", "python")

    # System dependencies for Unstructured
    print("  Installing system packages (libmagic, poppler, tesseract)...")
    run("sudo apt-get update -qq && sudo apt-get install -y -qq libmagic-dev poppler-utils tesseract-ocr",
        check=False)

    # pip install research extras
    print("  Installing research Python packages...")
    run(f"{pip} install -e '.[research]' -q", check=False)

    # Playwright browser
    print("  Installing Playwright Chromium browser...")
    run(f"{python} -m playwright install --with-deps chromium", check=False)

    # Verify
    for module, name in [("crawl4ai", "Crawl4AI"), ("unstructured", "Unstructured")]:
        r = run(f'{python} -c "import {module}"', capture=True, check=False)
        if r.returncode == 0:
            ok(f"{name} ready")
        else:
            warn(f"{name} import failed — some research features disabled")


# ---------------------------------------------------------------------------
# Phase 4: System Services
# ---------------------------------------------------------------------------
def install_services(minimal: bool = False):
    phase(4, "System Services")

    if minimal:
        warn("Skipped (--minimal mode).")
        return

    # ffmpeg (required for audio)
    if shutil.which("ffmpeg"):
        ok("ffmpeg available")
    else:
        if ask_yn("Install ffmpeg? (Required for audio/MP3 conversion)"):
            run("sudo apt-get install -y -qq ffmpeg", check=False)
            if shutil.which("ffmpeg"):
                ok("ffmpeg installed")
            else:
                warn("ffmpeg install failed — audio conversion disabled")
        else:
            warn("ffmpeg skipped — audio limited to WAV only")

    # Docker + SearXNG
    if shutil.which("docker"):
        ok("Docker available")
        _setup_searxng()
    else:
        if ask_yn("Install Docker + SearXNG? (Required for web search research)"):
            run("sudo apt-get update -qq && sudo apt-get install -y -qq docker.io", check=False)
            run("sudo systemctl enable --now docker", check=False)
            run(f"sudo usermod -aG docker {os.environ.get('USER', 'root')}", check=False)
            if shutil.which("docker"):
                ok("Docker installed")
                _setup_searxng()
            else:
                warn("Docker install failed — web search disabled")
        else:
            warn("Docker skipped — SearXNG web search disabled")

    # Ollama
    if shutil.which("ollama"):
        ok(f"Ollama available")
        _setup_ollama_model()
    else:
        if ask_yn("Install Ollama? (Required for local LLM: haiku generation, memory scoring)"):
            print("  Installing Ollama...")
            run("curl -fsSL https://ollama.com/install.sh | sh", check=False)
            if shutil.which("ollama"):
                ok("Ollama installed")
                _setup_ollama_model()
            else:
                warn("Ollama install failed — local LLM features disabled")
        else:
            warn("Ollama skipped — haiku and local scoring disabled")


def _setup_searxng():
    """Start or verify SearXNG Docker container."""
    r = run("docker ps -a --format '{{.Names}}' | grep -q '^searxng$'", capture=True, check=False)
    if r.returncode == 0:
        ok("SearXNG container exists")
        # Ensure running
        run("docker start searxng 2>/dev/null", check=False, capture=True)
        return

    print("  Starting SearXNG container...")
    searxng_dir = os.path.join(PROJECT_ROOT, "searxng")
    os.makedirs(searxng_dir, exist_ok=True)

    settings = textwrap.dedent("""\
        use_default_settings: true
        server:
          secret_key: "titan_sage_searxng_key"
          base_url: http://localhost:8080/
        search:
          safe_search: 0
          default_lang: en
          formats:
            - html
            - json
    """)
    with open(os.path.join(searxng_dir, "settings.yml"), "w") as f:
        f.write(settings)

    run(
        f"docker run -d --name searxng --restart=unless-stopped "
        f"-p 8080:8080 -v {searxng_dir}:/etc/searxng searxng/searxng",
        check=False,
    )
    ok("SearXNG started on http://localhost:8080")


def _setup_ollama_model():
    """Pull phi3:mini if not present."""
    r = run("ollama list 2>/dev/null | grep -q 'phi3:mini'", capture=True, check=False)
    if r.returncode == 0:
        ok("phi3:mini model loaded")
        return

    if ask_yn("Pull phi3:mini model? (~2.3GB download)"):
        print("  Pulling phi3:mini (this may take a few minutes)...")
        run("ollama pull phi3:mini", check=False, timeout=600)
        ok("phi3:mini ready")
    else:
        warn("phi3:mini not pulled — local LLM features will fail until available")


# ---------------------------------------------------------------------------
# Phase 5: Configuration Wizard
# ---------------------------------------------------------------------------
def configure():
    phase(5, "Configuration Wizard")

    if os.path.exists(CONFIG_PATH):
        print(f"  Existing config found: {CONFIG_PATH}")
        if not ask_yn("Reconfigure? (Existing values preserved as defaults)"):
            ok("Keeping existing configuration")
            return

    # Load existing config or template
    try:
        import tomllib
    except ImportError:
        import toml as tomllib  # type: ignore

    try:
        with open(CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
    except Exception:
        cfg = {}

    print()
    print("  Fill in your deployment settings. Press Enter to keep defaults.")
    print("  Sensitive fields (API keys, passwords) are stored locally only.")
    print()

    # Network
    print(f"  {Colors.BOLD}── Solana Network ──{Colors.RESET}")
    net = cfg.get("network", {})
    solana_network = ask("Solana network (mainnet-beta / devnet / testnet)", net.get("solana_network", "mainnet-beta"))
    premium_rpc = ask("Premium RPC URL (Helius/Triton, empty to skip)", net.get("premium_rpc_url", ""))
    maker_pubkey = ask("Maker's Solana public key (for directive auth)", net.get("maker_pubkey", ""))

    # Inference
    print(f"\n  {Colors.BOLD}── LLM Inference ──{Colors.RESET}")
    inf = cfg.get("inference", {})
    provider = ask("Inference provider (openrouter / venice / custom)", inf.get("inference_provider", "openrouter"))
    openrouter_key = ask("OpenRouter API key (empty to skip)", inf.get("openrouter_api_key", ""))
    venice_key = ask("Venice AI API key (empty to skip)", inf.get("venice_api_key", ""))

    # Twitter/Social
    print(f"\n  {Colors.BOLD}── X/Twitter Social ──{Colors.RESET}")
    tw = cfg.get("twitter_social", {})
    tw_user = ask("X/Twitter username (empty to disable social)", tw.get("user_name", ""))
    tw_email = ""
    tw_pass = ""
    tw_totp = ""
    tw_proxy = ""
    if tw_user:
        tw_email = ask("X/Twitter email", tw.get("email", ""))
        tw_pass = ask("X/Twitter password", tw.get("password", ""))
        tw_totp = ask("TOTP secret (Base32, for 2FA)", tw.get("totp_secret", ""))
        tw_proxy = ask("Static residential proxy URL (for session stability)", tw.get("webshare_static_url", ""))

    # Stealth Sage
    print(f"\n  {Colors.BOLD}── Research Engine ──{Colors.RESET}")
    ss = cfg.get("stealth_sage", {})
    twitter_api_key = ask("TwitterAPI.io key (empty to disable X-Search)", ss.get("twitterapi_io_key", ""))
    rotating_proxy = ask("Rotating proxy URL (empty for direct)", ss.get("webshare_rotating_url", ""))

    # Memory
    print(f"\n  {Colors.BOLD}── Storage ──{Colors.RESET}")
    mem = cfg.get("memory_and_storage", {})
    shadow_drive = ask("Shadow Drive storage account (empty to skip backups)", mem.get("shadow_drive_account", ""))

    # Write config.toml
    _write_config(
        solana_network=solana_network, premium_rpc=premium_rpc,
        maker_pubkey=maker_pubkey, provider=provider,
        openrouter_key=openrouter_key, venice_key=venice_key,
        tw_user=tw_user, tw_email=tw_email, tw_pass=tw_pass,
        tw_totp=tw_totp, tw_proxy=tw_proxy,
        twitter_api_key=twitter_api_key, rotating_proxy=rotating_proxy,
        shadow_drive=shadow_drive, existing=cfg,
    )
    ok(f"Configuration written: {CONFIG_PATH}")


def _write_config(**kw):
    """Merge wizard answers into the existing config.toml, preserving structure."""
    # Read current file as text to preserve comments and formatting
    try:
        with open(CONFIG_PATH, "r") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    replacements = {
        ('network', 'solana_network'): kw["solana_network"],
        ('network', 'premium_rpc_url'): kw["premium_rpc"],
        ('network', 'maker_pubkey'): kw["maker_pubkey"],
        ('inference', 'inference_provider'): kw["provider"],
        ('inference', 'openrouter_api_key'): kw["openrouter_key"],
        ('inference', 'venice_api_key'): kw["venice_key"],
        ('twitter_social', 'user_name'): kw["tw_user"],
        ('twitter_social', 'email'): kw["tw_email"],
        ('twitter_social', 'password'): kw["tw_pass"],
        ('twitter_social', 'totp_secret'): kw["tw_totp"],
        ('twitter_social', 'webshare_static_url'): kw["tw_proxy"],
        ('stealth_sage', 'twitterapi_io_key'): kw["twitter_api_key"],
        ('stealth_sage', 'webshare_rotating_url'): kw["rotating_proxy"],
        ('memory_and_storage', 'shadow_drive_account'): kw["shadow_drive"],
    }

    for (section, key), value in replacements.items():
        # Replace the value in the TOML text using simple string matching
        import re
        # Match: key = "old_value" or key = ""
        pattern = rf'^({re.escape(key)}\s*=\s*)"[^"]*"'
        replacement = rf'\1"{value}"'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    with open(CONFIG_PATH, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Phase 6: Genesis Ceremony
# ---------------------------------------------------------------------------
def run_genesis(skip: bool = False):
    phase(6, "Genesis Ceremony — Sovereign Identity")

    if skip:
        warn("Skipped (--skip-genesis). Run later: python scripts/genesis_ceremony.py --generate")
        return

    # Check if genesis already happened
    genesis_record = os.path.join(PROJECT_ROOT, "data", "genesis_record.json")
    if os.path.exists(genesis_record):
        ok("Genesis record already exists — identity was previously created")
        try:
            with open(genesis_record) as f:
                record = json.load(f)
            ok(f"Titan pubkey: {record.get('titan_pubkey', 'unknown')}")
        except Exception:
            pass

        if not ask_yn("Run genesis ceremony again? (Will create a NEW identity)", default=False):
            return

    python = os.path.join(VENV_DIR, "bin", "python")

    print()
    print(f"  {Colors.BOLD}The Genesis Ceremony will:{Colors.RESET}")
    print("    1. Generate a new Ed25519 Solana keypair")
    print("    2. Split it into 3 Shamir shards (2-of-3 threshold)")
    print("    3. Encrypt and distribute the shards")
    print("    4. Render the Genesis Art (the Titan's first self-portrait)")
    print("    5. Burn the plaintext keypair (irreversible)")
    print()

    if ask_yn("Generate a new identity?"):
        mode = "--generate"
    elif ask_yn("Import an existing keypair?", default=False):
        key_path = ask("Path to existing keypair JSON")
        mode = f"--import-key {key_path}"
    else:
        warn("Genesis deferred. Run later: python scripts/genesis_ceremony.py --generate")
        return

    # Skip on-chain if no RPC available
    extra_flags = ""
    try:
        import tomllib
    except ImportError:
        import toml as tomllib  # type: ignore
    try:
        with open(CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
        rpc_urls = cfg.get("network", {}).get("public_rpc_urls", [])
        premium = cfg.get("network", {}).get("premium_rpc_url", "")
        if not rpc_urls and not premium:
            extra_flags = " --skip-onchain"
            warn("No RPC URLs configured — skipping on-chain shard storage")
    except Exception:
        extra_flags = " --skip-onchain"

    # Keep plaintext during install (Maker confirms burn in ceremony script)
    run(f"{python} scripts/genesis_ceremony.py {mode}{extra_flags}", check=False)


# ---------------------------------------------------------------------------
# Phase 7: Health Check
# ---------------------------------------------------------------------------
def health_check():
    phase(7, "Sovereign Health Check")

    python = os.path.join(VENV_DIR, "bin", "python")

    checks = {
        "Titan Plugin": 'from titan_plugin import TitanPlugin; print("ok")',
        "StudioCoordinator": 'from titan_plugin.expressive.studio import StudioCoordinator; print("ok")',
        "Solana SDK": 'from titan_plugin.utils.solana_client import is_available; print("ok" if is_available() else "degraded")',
        "Cognee Memory": 'from titan_plugin.core.memory import TieredMemoryGraph; print("ok")',
        "SageGuardian": 'from titan_plugin.logic.sage.guardian import SageGuardian; print("ok")',
        "ProceduralArt": 'from titan_plugin.expressive.art import ProceduralArtGen; print("ok")',
        "Observatory API": 'from titan_plugin.api import create_app; print("ok")',
    }

    results = {}
    for name, code in checks.items():
        r = run(f'{python} -c "{code}"', capture=True, check=False)
        output = r.stdout.strip() if r.returncode == 0 else "failed"
        results[name] = output
        if output == "ok":
            ok(name)
        elif output == "degraded":
            warn(f"{name} (degraded mode — Solana SDK missing)")
        else:
            fail(f"{name}: {r.stderr.strip()[:100] if r.stderr else 'import error'}")

    # Critical dependency version checks (Shadow Drive requires precise httpx headers)
    version_checks = {
        "httpx": 'import httpx; print(httpx.__version__)',
        "cryptography": 'import cryptography; print(cryptography.__version__)',
    }
    print()
    for pkg, code in version_checks.items():
        r = run(f'{python} -c "{code}"', capture=True, check=False)
        if r.returncode == 0:
            ver = r.stdout.strip()
            ok(f"{pkg}: v{ver}")
        else:
            fail(f"{pkg}: not installed (required for Shadow Drive auth)")

    # Service liveness
    print()
    _check_service("Ollama", "curl -sf http://localhost:11434/api/tags")
    _check_service("SearXNG", "curl -sf http://localhost:8080/search?q=test&format=json")
    _check_service("ffmpeg", "ffmpeg -version")

    # Ollama model detection (Step 3.B: auto-configure text_mode)
    ollama_r = run("curl -sf http://localhost:11434/api/tags", capture=True, check=False)
    if ollama_r.returncode == 0:
        try:
            tags = json.loads(ollama_r.stdout)
            model_names = [m.get("name", "") for m in tags.get("models", [])]
            has_phi3 = any("phi3" in n or "phi" in n for n in model_names)
            if has_phi3:
                ok("Sage Intelligence: ENABLED (Phi-3 detected)")
            else:
                warn(f"Sage Intelligence: DEGRADED (models: {', '.join(model_names[:3]) or 'none'})")
                warn("  Install phi3:mini for full Haiku Reflection: ollama pull phi3:mini")
        except Exception:
            warn("Sage Intelligence: DEGRADED (Ollama running but model check failed)")
    else:
        warn("Sage Intelligence: DEGRADED (Ollama not running — using Template fallback)")

    # Genesis state
    genesis_path = os.path.join(PROJECT_ROOT, "data", "genesis_record.json")
    hw_path = os.path.join(PROJECT_ROOT, "data", "soul_keypair.enc")
    art_path = os.path.join(PROJECT_ROOT, "data", "genesis_art.png")

    print()
    if os.path.exists(genesis_path):
        ok("Genesis record: present")
    else:
        warn("Genesis record: absent (run genesis_ceremony.py)")

    if os.path.exists(hw_path):
        ok("Hardware-bound keypair: present")
    else:
        warn("Hardware-bound keypair: absent")

    if os.path.exists(art_path):
        ok(f"Genesis art: present ({os.path.getsize(art_path)} bytes)")
    else:
        warn("Genesis art: absent")

    # Sovereignty summary
    active = sum(1 for v in results.values() if v == "ok")
    total = len(results)
    print(f"\n  {Colors.BOLD}Sovereignty Status: {active}/{total} subsystems active{Colors.RESET}")


def _check_service(name: str, cmd: str):
    r = run(cmd, capture=True, check=False, timeout=10)
    if r.returncode == 0:
        ok(f"{name}: reachable")
    else:
        warn(f"{name}: not reachable (optional)")


# ---------------------------------------------------------------------------
# Data Directories
# ---------------------------------------------------------------------------
def ensure_directories():
    """Create all required data directories."""
    dirs = [
        os.path.join(PROJECT_ROOT, "data"),
        os.path.join(PROJECT_ROOT, "data", "logs"),
        os.path.join(PROJECT_ROOT, "data", "history"),
        os.path.join(PROJECT_ROOT, "data", "studio_exports"),
        os.path.join(PROJECT_ROOT, "data", "studio_exports", "meditation"),
        os.path.join(PROJECT_ROOT, "data", "studio_exports", "epoch"),
        os.path.join(PROJECT_ROOT, "data", "studio_exports", "eureka"),
        "/tmp/titan_sage_docs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Titan Interactive Birth Wizard")
    parser.add_argument("--minimal", action="store_true", help="Core only, skip optional stacks")
    parser.add_argument("--skip-genesis", action="store_true", help="Skip Genesis Ceremony")
    args = parser.parse_args()

    banner("THE TITAN BIRTH WIZARD — Sovereign Deployment Protocol")

    print("  This wizard will guide you through deploying a Titan instance.")
    print("  It will install dependencies, configure services, and optionally")
    print("  run the Genesis Ceremony to create your Titan's immortal identity.")
    print()

    if not ask_yn("Ready to begin?"):
        print("\n  Installation cancelled. Run again when ready.\n")
        sys.exit(0)

    # Phase 1: Prerequisites
    if not check_prerequisites():
        fail("Prerequisites not met. Fix the issues above and try again.")
        sys.exit(1)

    # Phase 2: Core install
    if not install_core():
        fail("Core installation failed. Check errors above.")
        sys.exit(1)

    # Ensure data directories
    ensure_directories()

    # Phase 3: Research stack
    install_research(minimal=args.minimal)

    # Phase 4: System services
    install_services(minimal=args.minimal)

    # Phase 5: Configuration
    configure()

    # Phase 6: Genesis
    run_genesis(skip=args.skip_genesis)

    # Phase 7: Health check
    health_check()

    # Final banner
    banner("TITAN DEPLOYMENT COMPLETE")
    print("  Next steps:")
    print(f"    1. Activate the venv:  source .venv/bin/activate")
    print(f"    2. Run standalone:     titan-main")
    print(f"    3. Run tests:          python -m pytest tests/ -p no:anchorpy -v")
    print(f"    4. Connect to OpenClaw: Register 'openclaw-plugin-titan' in your agent config")
    print()
    print(f"  {Colors.BOLD}The Titan awaits its first thought.{Colors.RESET}")
    print()


if __name__ == "__main__":
    main()
