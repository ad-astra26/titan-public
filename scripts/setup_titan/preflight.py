"""Phase 1 — preflight: OS · Python · git · sudo · resources · idempotency.

Returns a structured report (no prints in the check functions — caller renders
through the TUI or the ANSI fallback in ui.py). Every check produces a Result
with a clear severity + remediation hint.

Locked minimums from RFP_Titan_setup_release.md (2026-05-22):
    headless Telegram-only:  2 vCPU / 4 GB RAM   (min)
    with Observatory:        4 vCPU / 8 GB RAM   (recommended)
    disk:                    20 GB free          (binaries + venv + data + headroom)
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import state as install_state
from .modes import Mode, spec_for

Severity = Literal["ok", "warn", "fail"]


@dataclass
class Result:
    name: str
    severity: Severity
    detail: str
    remediation: str = ""


# ── individual checks ───────────────────────────────────────────────────────

def check_os() -> Result:
    if platform.system() != "Linux":
        return Result("os", "fail",
                      f"Titan supports Linux only (found {platform.system()}).",
                      "Run the installer on a Debian/Ubuntu Linux host (your own VPS or localhost).")
    try:
        with open("/etc/os-release") as f:
            osr = dict(line.strip().split("=", 1) for line in f if "=" in line)
        idl = osr.get("ID", "").strip('"').lower()
        like = osr.get("ID_LIKE", "").strip('"').lower()
        pretty = osr.get("PRETTY_NAME", "").strip('"') or idl
    except OSError:
        return Result("os", "warn", "Linux detected but /etc/os-release unreadable.",
                      "Continue at your own risk — Debian/Ubuntu are the supported flavors.")
    if idl in ("debian", "ubuntu") or "debian" in like or "ubuntu" in like:
        return Result("os", "ok", f"{pretty} — supported.")
    return Result("os", "warn", f"{pretty} — not Debian/Ubuntu; supported but untested.",
                  "Debian/Ubuntu are the tested platforms; other distros may need manual fix-ups.")


def check_python() -> Result:
    import sys
    v = sys.version_info
    if (v.major, v.minor) < (3, 11):
        return Result("python", "fail",
                      f"Python {v.major}.{v.minor} — need ≥3.11.",
                      "Install Python 3.11+: sudo apt install python3.11 python3.11-venv")
    return Result("python", "ok", f"Python {v.major}.{v.minor}.{v.micro}")


def check_cmd(name: str, install_hint: str, *, severity_if_missing: Severity = "fail") -> Result:
    path = shutil.which(name)
    if path:
        return Result(name, "ok", f"{name} available ({path})")
    return Result(name, severity_if_missing, f"{name} not found.", install_hint)


def check_sudo() -> Result:
    if os.geteuid() == 0:
        return Result("sudo", "warn",
                      "Running as root — preferred is a non-root user with sudo (so files land owned by you).",
                      "Re-run as your normal user; the wizard will sudo for the specific apt installs only.")
    if not shutil.which("sudo"):
        return Result("sudo", "fail", "sudo not available — the installer needs it for apt + systemd.",
                      "Add your user to sudoers, then re-run.")
    # Non-interactive sudo probe — does NOT prompt; just checks if a session is cached
    r = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    if r.returncode == 0:
        return Result("sudo", "ok", "sudo available + session cached (no prompt needed right now).")
    return Result("sudo", "ok", "sudo available — the wizard will prompt when needed.")


def check_resources(install_root: Path) -> list[Result]:
    out: list[Result] = []

    # RAM
    try:
        with open("/proc/meminfo") as f:
            kv = {ln.split(":")[0]: ln.split(":")[1].strip() for ln in f if ":" in ln}
        ram_kb = int(kv["MemTotal"].split()[0])
        ram_gb = ram_kb / 1024 / 1024
        if ram_gb < 3.5:
            out.append(Result("ram", "fail",
                              f"{ram_gb:.1f} GB RAM — minimum 4 GB (headless, Telegram-only).",
                              "Increase VPS RAM to ≥4 GB (≥8 GB recommended with Observatory)."))
        elif ram_gb < 7.5:
            out.append(Result("ram", "warn",
                              f"{ram_gb:.1f} GB RAM — meets minimum (4 GB) but below recommended (8 GB).",
                              "Observatory + heavy LLM use will be tight. 8 GB recommended."))
        else:
            out.append(Result("ram", "ok", f"{ram_gb:.1f} GB RAM"))
    except (OSError, KeyError, ValueError):
        out.append(Result("ram", "warn", "could not read /proc/meminfo — RAM unverified."))

    # CPU
    try:
        cpus = os.cpu_count() or 0
        if cpus < 2:
            out.append(Result("cpu", "fail", f"{cpus} vCPU — minimum 2.",
                              "Increase VPS to ≥2 vCPU (≥4 recommended)."))
        elif cpus < 4:
            out.append(Result("cpu", "warn", f"{cpus} vCPU — meets minimum (2) but below recommended (4)."))
        else:
            out.append(Result("cpu", "ok", f"{cpus} vCPU"))
    except Exception:
        out.append(Result("cpu", "warn", "could not read CPU count."))

    # disk (free on install_root's filesystem)
    try:
        target = install_root if install_root.exists() else install_root.parent
        free_gb = shutil.disk_usage(target).free / (1024 ** 3)
        if free_gb < 10:
            out.append(Result("disk", "fail",
                              f"{free_gb:.1f} GB free on {target} — minimum 20 GB.",
                              "Free disk or pick a different install root."))
        elif free_gb < 20:
            out.append(Result("disk", "warn",
                              f"{free_gb:.1f} GB free on {target} — below recommended 20 GB.",
                              "Titan grows over time (logs, databases, Arweave cache). 20 GB+ recommended."))
        else:
            out.append(Result("disk", "ok", f"{free_gb:.1f} GB free on {target}"))
    except OSError as e:
        out.append(Result("disk", "warn", f"could not stat disk: {e}"))

    return out


def check_existing_install() -> Result:
    """Idempotency probe — refuse to clobber an existing Titan unless --repair / --upgrade."""
    st = install_state.load()
    if install_state.exists() and st.get("phases"):
        done = [p for p, v in st["phases"].items() if v.get("status") == "done"]
        return Result("existing", "warn",
                      f"Existing install detected ({len(done)} phase(s) complete) at {st.get('install_root') or 'unknown root'}.",
                      "Use `setup_titan repair` to fix detected issues or `setup_titan install --resume` to continue.")
    return Result("existing", "ok", "no prior install state — fresh setup.")


def check_mode_toolchain(mode: Mode) -> list[Result]:
    """For mainnet/devnet, ensure Rust + Anchor + Solana CLI are available (or installable)."""
    spec = spec_for(mode)
    out: list[Result] = []
    if spec.needs_rust:
        out.append(check_cmd("cargo",
                             "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh",
                             severity_if_missing="warn"))
    if spec.needs_anchor:
        out.append(check_cmd("anchor",
                             "Install Anchor: cargo install --git https://github.com/coral-xyz/anchor anchor-cli --locked",
                             severity_if_missing="warn"))
    if spec.needs_solana_cli:
        out.append(check_cmd("solana",
                             "Install Solana CLI: sh -c \"$(curl -sSfL https://release.solana.com/stable/install)\"",
                             severity_if_missing="warn"))
    return out


# ── orchestration ───────────────────────────────────────────────────────────

def run_preflight(install_root: Path, mode: Mode | None) -> list[Result]:
    """Run every preflight check and return the flat list of Results.

    `mode` may be None for the very first wizard screen (before the user picks
    a mode); toolchain checks are skipped in that case.
    """
    results: list[Result] = [
        check_os(),
        check_python(),
        check_cmd("git", "sudo apt install git"),
        check_cmd("rsync", "sudo apt install rsync"),
        check_sudo(),
    ]
    results += check_resources(install_root)
    results.append(check_existing_install())
    if mode is not None and mode != Mode.LOCAL:
        results += check_mode_toolchain(mode)
    return results


def summarize(results: list[Result]) -> tuple[int, int, int]:
    """Return (ok, warn, fail) counts."""
    ok = sum(1 for r in results if r.severity == "ok")
    warn = sum(1 for r in results if r.severity == "warn")
    fail = sum(1 for r in results if r.severity == "fail")
    return ok, warn, fail
