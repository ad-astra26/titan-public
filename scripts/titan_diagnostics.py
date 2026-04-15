#!/usr/bin/env python3
"""
titan_diagnostics.py — Titan Sovereign AI Diagnostics & Health Check.

Standalone tool to verify the full Titan stack. Works whether Titan is running or not.

Usage:
    python scripts/titan_diagnostics.py            # Full diagnostics
    python scripts/titan_diagnostics.py --quick     # Skip slow checks (services, channels, wallet)
    python scripts/titan_diagnostics.py --json      # Machine-readable JSON output
"""
import argparse
import importlib
import json as json_mod
import os
import platform
import shutil
import sys
import pathlib
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "titan_plugin" / "config.toml"

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CYAN = "\033[96m"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = _supports_color()


def c(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{color}{text}{Colors.RESET}"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

class DiagResult:
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"

    def __init__(self):
        self.categories: list[dict] = []
        self._current: dict | None = None

    def begin_category(self, name: str):
        self._current = {"name": name, "checks": []}
        self.categories.append(self._current)

    def add(self, name: str, status: str, detail: str = ""):
        entry = {"name": name, "status": status, "detail": detail}
        if self._current is not None:
            self._current["checks"].append(entry)
        return entry

    def ok(self, name: str, detail: str = ""):
        return self.add(name, self.PASS, detail)

    def warn(self, name: str, detail: str = ""):
        return self.add(name, self.WARN, detail)

    def fail(self, name: str, detail: str = ""):
        return self.add(name, self.FAIL, detail)

    def skip(self, name: str, detail: str = ""):
        return self.add(name, self.SKIP, detail)

    @property
    def counts(self) -> dict:
        p = w = f = s = 0
        for cat in self.categories:
            for ch in cat["checks"]:
                st = ch["status"]
                if st == self.PASS:
                    p += 1
                elif st == self.WARN:
                    w += 1
                elif st == self.FAIL:
                    f += 1
                elif st == self.SKIP:
                    s += 1
        return {"pass": p, "warn": w, "fail": f, "skip": s, "total": p + w + f + s}

    def print_report(self):
        status_icons = {
            self.PASS: c("\u2713", Colors.GREEN),
            self.WARN: c("!", Colors.YELLOW),
            self.FAIL: c("\u2717", Colors.RED),
            self.SKIP: c("-", Colors.DIM),
        }
        for cat in self.categories:
            print(f"\n{c('=' * 60, Colors.DIM)}")
            print(f"  {c(cat['name'], Colors.BOLD + Colors.CYAN)}")
            print(c("=" * 60, Colors.DIM))
            for ch in cat["checks"]:
                icon = status_icons.get(ch["status"], "?")
                detail = f"  {c(ch['detail'], Colors.DIM)}" if ch["detail"] else ""
                print(f"  {icon} {ch['name']}{detail}")

        counts = self.counts
        operational = counts["pass"]
        total = counts["total"] - counts["skip"]
        print(f"\n{c('=' * 60, Colors.DIM)}")
        if counts["fail"] == 0:
            color = Colors.GREEN
        elif counts["fail"] <= 2:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        print(f"  {c(f'Sovereignty Status: {operational}/{total} systems operational', Colors.BOLD + color)}")
        p, w, f, s = counts["pass"], counts["warn"], counts["fail"], counts["skip"]
        print(f"  {c(f'Pass: {p}  Warn: {w}  Fail: {f}  Skip: {s}', Colors.DIM)}")
        print(c("=" * 60, Colors.DIM))

    def to_json(self) -> dict:
        return {"categories": self.categories, "summary": self.counts}


# ---------------------------------------------------------------------------
# Config parser (TOML)
# ---------------------------------------------------------------------------

def load_config() -> dict | None:
    """Load config.toml. Returns parsed dict or None."""
    if not CONFIG_PATH.exists():
        return None
    try:
        # Python 3.11+ has tomllib
        import tomllib
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        pass
    try:
        import tomli
        with open(CONFIG_PATH, "rb") as f:
            return tomli.load(f)
    except ImportError:
        pass
    # Fallback: try toml package
    try:
        import toml
        return toml.load(str(CONFIG_PATH))
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------

def check_system(results: DiagResult):
    results.begin_category("System Environment")

    # Python version
    ver = platform.python_version()
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 10:
        results.ok("Python version", ver)
    elif major == 3 and minor >= 8:
        results.warn("Python version", f"{ver} (3.10+ recommended)")
    else:
        results.fail("Python version", f"{ver} (3.10+ required)")

    # OS
    results.ok("Operating system", f"{platform.system()} {platform.release()}")

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    gb = kb / (1024 * 1024)
                    if gb >= 4:
                        results.ok("RAM", f"{gb:.1f} GB")
                    else:
                        results.warn("RAM", f"{gb:.1f} GB (4+ GB recommended)")
                    break
    except Exception:
        results.ok("RAM", "unable to read /proc/meminfo")

    # Disk space
    try:
        usage = shutil.disk_usage(str(PROJECT_ROOT))
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        if free_gb >= 5:
            results.ok("Disk space", f"{free_gb:.1f} GB free / {total_gb:.1f} GB total")
        elif free_gb >= 1:
            results.warn("Disk space", f"{free_gb:.1f} GB free (low)")
        else:
            results.fail("Disk space", f"{free_gb:.1f} GB free (critical)")
    except Exception as e:
        results.warn("Disk space", str(e))

    # Venv
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        results.ok("Python venv", "active")
    else:
        results.warn("Python venv", "not active (run: source test_env/bin/activate)")


def check_dependencies(results: DiagResult):
    results.begin_category("Dependencies")

    modules = {
        "titan_plugin": "titan_plugin",
        "torch": "torch",
        "fastapi": "fastapi",
        "httpx": "httpx",
        "PIL (Pillow)": "PIL",
        "cryptography": "cryptography",
        "cognee": "cognee",
        "agno": "agno",
        "solders": "solders",
        "anchorpy": "anchorpy",
        "pydub": "pydub",
        "sentence_transformers": "sentence_transformers",
    }

    for display_name, module_name in modules.items():
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", None)
            detail = f"v{ver}" if ver else "ok"
            results.ok(display_name, detail)
        except Exception as e:
            err_msg = str(e).split("\n")[0][:80]
            results.fail(display_name, err_msg)


EXPECTED_SECTIONS = [
    "mood_engine",
    "addons",
    "growth_metrics",
    "stealth_sage",
    "network",
    "inference",
    "memory_and_storage",
    "openclaw",
    "twitter_social",
    "expressive",
    "info_banner",
    "privacy",
    "api",
    "endurance",
    "observatory",
    "frontend",
    "channels",
]

# Fields that should not be empty for proper operation
REQUIRED_FIELDS = {
    "inference": ["inference_provider"],
    "network": ["solana_network", "public_rpc_urls", "wallet_keypair_path"],
    "api": ["port"],
}

# Fields that should ideally have values but are not strictly required
WARN_FIELDS = {
    "inference": ["openrouter_api_key", "venice_api_key"],
    "network": ["maker_pubkey"],
    "channels": ["telegram_bot_token"],
}


def check_config(results: DiagResult):
    results.begin_category("Configuration")

    if not CONFIG_PATH.exists():
        results.fail("config.toml", "not found")
        return None

    config = load_config()
    if config is None:
        results.fail("config.toml", "unable to parse (install tomli or use Python 3.11+)")
        return None

    results.ok("config.toml", f"parsed ({len(config)} sections)")

    # Check all 16+1 expected sections
    missing = [s for s in EXPECTED_SECTIONS if s not in config]
    present = [s for s in EXPECTED_SECTIONS if s in config]

    if not missing:
        results.ok("Config sections", f"all {len(EXPECTED_SECTIONS)} present")
    else:
        results.warn("Config sections", f"missing: {', '.join(missing)}")

    # Required fields
    for section, fields in REQUIRED_FIELDS.items():
        if section not in config:
            continue
        for field in fields:
            val = config[section].get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                results.fail(f"[{section}].{field}", "empty (required)")
            else:
                display = str(val)[:40] + "..." if len(str(val)) > 40 else str(val)
                results.ok(f"[{section}].{field}", display)

    # Warn fields
    for section, fields in WARN_FIELDS.items():
        if section not in config:
            continue
        for field in fields:
            val = config[section].get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                results.warn(f"[{section}].{field}", "empty (optional but recommended)")

    return config


def check_services(results: DiagResult, config: dict | None, quick: bool):
    results.begin_category("Service Liveness")

    if quick:
        results.skip("Service checks", "skipped (--quick)")
        return

    import httpx

    services = [
        ("Titan API", "http://localhost:7777/health", 3),
        ("Watchdog", "http://localhost:7778/health", 3),
        ("Ollama", "http://localhost:11434/api/tags", 3),
        ("SearXNG", "http://localhost:8080", 3),
        ("Frontend", "http://localhost:3000", 3),
    ]

    for name, url, timeout in services:
        try:
            resp = httpx.get(url, timeout=timeout, follow_redirects=True)
            if resp.status_code < 400:
                results.ok(name, f"reachable (HTTP {resp.status_code})")
            else:
                results.warn(name, f"HTTP {resp.status_code}")
        except httpx.ConnectError:
            results.warn(name, "not reachable (connection refused)")
        except httpx.TimeoutException:
            results.warn(name, "not reachable (timeout)")
        except Exception as e:
            results.warn(name, str(e)[:60])


def check_identity(results: DiagResult, config: dict | None, quick: bool):
    results.begin_category("Identity & On-Chain")

    data_dir = PROJECT_ROOT / "data"

    # Genesis record
    genesis_path = data_dir / "genesis_record.json"
    if genesis_path.exists():
        size = genesis_path.stat().st_size
        results.ok("genesis_record.json", f"{size:,} bytes")
    else:
        results.warn("genesis_record.json", "absent (run genesis_ceremony.py)")

    # Soul keypair
    soul_path = data_dir / "soul_keypair.enc"
    if soul_path.exists():
        size = soul_path.stat().st_size
        results.ok("soul_keypair.enc", f"{size:,} bytes")
    else:
        results.warn("soul_keypair.enc", "absent")

    # Genesis art
    art_path = data_dir / "genesis_art.png"
    if art_path.exists():
        size = art_path.stat().st_size
        results.ok("genesis_art.png", f"{size:,} bytes")
    else:
        results.warn("genesis_art.png", "absent")

    # Wallet balance via Solana RPC
    if quick:
        results.skip("Wallet balance", "skipped (--quick)")
        return

    if config and "network" in config:
        net_cfg = config["network"]
        pubkey = net_cfg.get("maker_pubkey", "")
        rpc_urls = net_cfg.get("public_rpc_urls", [])
        if pubkey and rpc_urls:
            rpc_url = rpc_urls[0] if isinstance(rpc_urls, list) else str(rpc_urls)
            try:
                import httpx
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getBalance",
                    "params": [pubkey],
                }
                resp = httpx.post(rpc_url, json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    lamports = data.get("result", {}).get("value", 0)
                    sol = lamports / 1_000_000_000
                    network = net_cfg.get("solana_network", "unknown")
                    if sol > 1:
                        results.ok("Wallet balance", f"{sol:.4f} SOL ({network})")
                    elif sol > 0:
                        results.warn("Wallet balance", f"{sol:.4f} SOL (low, {network})")
                    else:
                        results.warn("Wallet balance", f"0 SOL ({network})")
                else:
                    results.warn("Wallet balance", f"RPC error HTTP {resp.status_code}")
            except Exception as e:
                results.warn("Wallet balance", str(e)[:60])
        else:
            results.skip("Wallet balance", "no pubkey or RPC URL in config")
    else:
        results.skip("Wallet balance", "no network config")


def check_channels(results: DiagResult, config: dict | None, quick: bool):
    results.begin_category("Communication Channels")

    if quick:
        results.skip("Channel checks", "skipped (--quick)")
        return

    if config is None or "channels" not in config:
        results.skip("Channel checks", "no [channels] config")
        return

    ch_cfg = config["channels"]
    import httpx

    # Telegram
    tg_enabled = ch_cfg.get("telegram_enabled", False)
    tg_token = ch_cfg.get("telegram_bot_token", "")
    if tg_enabled and tg_token:
        try:
            resp = httpx.get(
                f"https://api.telegram.org/bot{tg_token}/getMe", timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                bot_name = data.get("result", {}).get("username", "unknown")
                results.ok("Telegram", f"connected as @{bot_name}")
            else:
                results.fail("Telegram", f"HTTP {resp.status_code} (invalid token)")
        except Exception as e:
            results.fail("Telegram", str(e)[:60])
    elif tg_enabled:
        results.warn("Telegram", "enabled but no bot token")
    else:
        results.skip("Telegram", "disabled")

    # Discord
    dc_enabled = ch_cfg.get("discord_enabled", False)
    dc_token = ch_cfg.get("discord_bot_token", "")
    if dc_enabled and dc_token:
        try:
            resp = httpx.get(
                "https://discord.com/api/v10/users/@me",
                headers={"Authorization": f"Bot {dc_token}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                uname = data.get("username", "unknown")
                results.ok("Discord", f"connected as {uname}")
            else:
                results.fail("Discord", f"HTTP {resp.status_code} (invalid token)")
        except Exception as e:
            results.fail("Discord", str(e)[:60])
    elif dc_enabled:
        results.warn("Discord", "enabled but no bot token")
    else:
        results.skip("Discord", "disabled")

    # Slack
    sl_enabled = ch_cfg.get("slack_enabled", False)
    sl_token = ch_cfg.get("slack_bot_token", "")
    if sl_enabled and sl_token:
        try:
            resp = httpx.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {sl_token}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    team = data.get("team", "unknown")
                    user = data.get("user", "unknown")
                    results.ok("Slack", f"connected to {team} as {user}")
                else:
                    results.fail("Slack", data.get("error", "auth failed"))
            else:
                results.fail("Slack", f"HTTP {resp.status_code}")
        except Exception as e:
            results.fail("Slack", str(e)[:60])
    elif sl_enabled:
        results.warn("Slack", "enabled but no bot token")
    else:
        results.skip("Slack", "disabled")

    # WhatsApp
    wa_enabled = ch_cfg.get("whatsapp_enabled", False)
    wa_phone = ch_cfg.get("whatsapp_phone_id", "")
    wa_token = ch_cfg.get("whatsapp_token", "")
    if wa_enabled and wa_phone and wa_token:
        try:
            resp = httpx.get(
                f"https://graph.facebook.com/v18.0/{wa_phone}",
                headers={"Authorization": f"Bearer {wa_token}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                number = data.get("display_phone_number", wa_phone)
                results.ok("WhatsApp", f"connected: {number}")
            else:
                results.fail("WhatsApp", f"HTTP {resp.status_code}")
        except Exception as e:
            results.fail("WhatsApp", str(e)[:60])
    elif wa_enabled:
        results.warn("WhatsApp", "enabled but missing credentials")
    else:
        results.skip("WhatsApp", "disabled")


def check_memory(results: DiagResult, config: dict | None, quick: bool):
    results.begin_category("Memory & Storage")

    # Cognee data directory
    cognee_path_str = "./cognee_data"
    if config and "memory_and_storage" in config:
        cognee_path_str = config["memory_and_storage"].get("cognee_db_path", cognee_path_str)

    # Resolve relative to project root
    cognee_path = pathlib.Path(cognee_path_str)
    if not cognee_path.is_absolute():
        cognee_path = PROJECT_ROOT / cognee_path

    if cognee_path.exists() and cognee_path.is_dir():
        # Calculate directory size
        total_size = 0
        file_count = 0
        for dirpath, _dirnames, filenames in os.walk(cognee_path):
            for fname in filenames:
                fp = os.path.join(dirpath, fname)
                try:
                    total_size += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    pass
        size_mb = total_size / (1024 * 1024)
        results.ok("Cognee data directory", f"{size_mb:.1f} MB, {file_count} files")
    elif cognee_path.exists():
        results.ok("Cognee data path", f"exists ({cognee_path})")
    else:
        results.warn("Cognee data directory", f"not found at {cognee_path}")

    # Node count via API (only if Titan API is running)
    if quick:
        results.skip("Memory node count", "skipped (--quick)")
        return

    try:
        import httpx
        resp = httpx.get("http://localhost:7777/health", timeout=3)
        if resp.status_code == 200:
            # Try to get memory stats from the API
            try:
                stats_resp = httpx.get("http://localhost:7777/vitals", timeout=5)
                if stats_resp.status_code == 200:
                    data = stats_resp.json()
                    node_count = data.get("memory", {}).get("node_count")
                    if node_count is not None:
                        results.ok("Memory node count", f"{node_count} nodes (via API)")
                    else:
                        results.ok("Memory node count", "API reachable (node count not in vitals)")
                else:
                    results.skip("Memory node count", "vitals endpoint unavailable")
            except Exception:
                results.skip("Memory node count", "could not query vitals")
        else:
            results.skip("Memory node count", "Titan API not running")
    except Exception:
        results.skip("Memory node count", "Titan API not reachable")

    # RL buffer
    rl_buffer = PROJECT_ROOT / "data" / "rl_buffer"
    if rl_buffer.exists():
        total_size = 0
        for dirpath, _dirnames, filenames in os.walk(rl_buffer):
            for fname in filenames:
                try:
                    total_size += os.path.getsize(os.path.join(dirpath, fname))
                except OSError:
                    pass
        size_mb = total_size / (1024 * 1024)
        results.ok("RL buffer", f"{size_mb:.1f} MB")
    else:
        results.skip("RL buffer", "not found (created on first run)")

    # Observatory DB
    obs_db = PROJECT_ROOT / "data" / "observatory.db"
    if obs_db.exists():
        size_kb = obs_db.stat().st_size / 1024
        results.ok("Observatory DB", f"{size_kb:.0f} KB")
    else:
        results.skip("Observatory DB", "not found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Titan Sovereign AI Diagnostics & Health Check",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip slow checks (service liveness, channel connectivity, wallet balance)",
    )
    args = parser.parse_args()

    # Suppress color for JSON mode
    global USE_COLOR
    if args.json_output:
        USE_COLOR = False

    results = DiagResult()

    if not args.json_output:
        print(c("\n  TITAN SOVEREIGN AI DIAGNOSTICS", Colors.BOLD + Colors.CYAN))
        print(c(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}", Colors.DIM))
        print(c(f"  Project root: {PROJECT_ROOT}", Colors.DIM))

    # 1. System checks
    check_system(results)

    # 2. Dependencies
    check_dependencies(results)

    # 3. Config validation
    config = check_config(results)

    # 4. Service liveness
    check_services(results, config, args.quick)

    # 5. Identity checks
    check_identity(results, config, args.quick)

    # 6. Channel checks
    check_channels(results, config, args.quick)

    # 7. Memory checks
    check_memory(results, config, args.quick)

    # Output
    if args.json_output:
        print(json_mod.dumps(results.to_json(), indent=2))
    else:
        results.print_report()
        print()


if __name__ == "__main__":
    main()
