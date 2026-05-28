"""Lifecycle management commands (W1.g) — diagnostic / upgrade / repair / uninstall.

Thin wrappers over machinery that already exists:
  - diagnostic : translate `arch_map health` + systemd state + /health into a
                 tester-friendly report (wrap existing tools, don't rebuild).
  - upgrade    : stop → git update @ tag → pip refresh → refresh binaries →
                 reinstall unit + restart (reuses systemd_runner + binaries).
  - repair     : idempotent heal — regenerate the unit + restart (systemd_runner).
  - uninstall  : disable + remove the unit + cleanup; keeps data/ unless --purge.

All paths resolve install_root from ~/.titan/install_state.json, falling back to
the repo this package lives in. Stdlib + subprocess only.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import state as install_state
from .binaries import build_release_binaries, fetch_release_binaries, run_binaries_phase
from .modes import Mode
from .preflight import Result
from .systemd_runner import (
    UNIT_DEST,
    UNIT_NAME,
    DEFAULT_API_PORT,
    _health_ok,
    _sudo,
    cleanup_script,
    resolve_install_titan_id,
    run_systemd_phase,
)
from .ui import cprint, section


def resolve_install_root() -> Path:
    """Install root from state, else the repo this package lives in."""
    st = install_state.load()
    root = st.get("install_root")
    if root and Path(root).exists():
        return Path(root)
    return Path(__file__).resolve().parents[2]


def _venv_python(root: Path) -> Path:
    return root / "test_env" / "bin" / "python"


def _render(results: list[Result]) -> int:
    from .phases import render
    return render(results)


# ── diagnostic ───────────────────────────────────────────────────────────────


def run_diagnostic(*, api_port: int = DEFAULT_API_PORT) -> int:
    root = resolve_install_root()
    titan_id = resolve_install_titan_id(root)
    section(f"Diagnostic — Titan {titan_id}")

    # 1. systemd unit state (friendly)
    active = subprocess.run(["systemctl", "is-active", UNIT_NAME],
                            capture_output=True, text=True).stdout.strip()
    enabled = subprocess.run(["systemctl", "is-enabled", UNIT_NAME],
                             capture_output=True, text=True).stdout.strip()
    if active == "active":
        cprint(f"  ● Service: running (and {enabled or 'unknown'} on boot)", role="success")
    else:
        cprint(f"  ○ Service: {active or 'not installed'} ({enabled or 'unknown'} on boot)",
               role="warning")

    # 2. liveness (friendly)
    if _health_ok(api_port):
        cprint(f"  ● Titan is awake and responding at :{api_port}/health", role="success")
    else:
        cprint(f"  ○ Titan is not responding at :{api_port}/health", role="warning")
        cprint("    Recent logs:", role="text_muted")
        logs = subprocess.run(["journalctl", "-u", UNIT_NAME, "-n", "15", "--no-pager"],
                              capture_output=True, text=True)
        for line in (logs.stdout or logs.stderr or "(no journal access)").splitlines()[-15:]:
            print(f"      {line}")

    # 3. deep view via the existing dev tool (wrapped, clearly labelled advanced)
    py = _venv_python(root)
    arch_map = root / "scripts" / "arch_map.py"
    if py.exists() and arch_map.exists():
        cprint("\n  Advanced (arch_map health) ─────────────────────────────", role="text_muted")
        subprocess.run([str(py), str(arch_map), "health", "--titan", titan_id], cwd=str(root))
    else:
        cprint("\n  (Install the venv to unlock the advanced `arch_map health` view.)",
               role="text_muted")
    return 0


# ── upgrade ──────────────────────────────────────────────────────────────────


def run_upgrade(*, tag: str | None, build_rust: bool, api_port: int = DEFAULT_API_PORT) -> int:
    root = resolve_install_root()
    st = install_state.load()
    mode = Mode(st["mode"]) if st.get("mode") else Mode.LOCAL
    section(f"Upgrade — {root}" + (f" → {tag}" if tag else " (binaries refresh only)"))

    # 1. stop the running service so files/binaries can be swapped safely
    _sudo(["systemctl", "stop", UNIT_NAME], explain="stop Titan before upgrading")

    # 2. git update to the requested ref (skip if no tag given — binary-only refresh)
    if tag and (root / ".git").exists():
        cprint(f"  Fetching {tag}…", role="text_strong")
        if subprocess.run(["git", "-C", str(root), "fetch", "--depth", "1", "origin", tag]).returncode \
           or subprocess.run(["git", "-C", str(root), "checkout", "-q", "FETCH_HEAD"]).returncode:
            return _render([Result("upgrade", "fail", f"git update to {tag} failed",
                                   "Resolve the git error above (local edits? detached state?).")])

    # 3. refresh Python deps
    pip = root / "test_env" / "bin" / "pip"
    if pip.exists():
        cprint("  Refreshing Python dependencies (pip install -e .)…", role="text_strong")
        if subprocess.run([str(pip), "install", "-e", str(root)]).returncode:
            return _render([Result("upgrade", "fail", "pip install -e . failed",
                                   "Inspect the pip output above.")])

    # 4. refresh the Rust binaries (force re-acquire, not the install-time skip)
    cprint("  Refreshing Rust daemons…", role="text_strong")
    bres = build_release_binaries(root) if build_rust else fetch_release_binaries(root, tag or "main")
    if _render(bres):
        return 1

    # 5. regenerate the unit + restart + health-gate (reuse the install phase)
    rc = _render(run_systemd_phase(st, root, mode, default=False, api_port=api_port))
    return 1 if rc else 0


# ── repair ───────────────────────────────────────────────────────────────────


def run_repair(*, api_port: int = DEFAULT_API_PORT) -> int:
    root = resolve_install_root()
    st = install_state.load()
    mode = Mode(st["mode"]) if st.get("mode") else Mode.LOCAL
    section(f"Repair — {root}")

    # ensure the cleanup script is executable (zip-download loses the bit)
    cs = cleanup_script(root)
    if cs.exists():
        cs.chmod(0o755)
        cprint(f"  ✓ cleanup script executable ({cs})", role="success")
    else:
        cprint(f"  ⚠ cleanup script missing ({cs}) — re-clone the repo.", role="warning")

    # ensure the binaries are present (idempotent — skips if already there)
    bres = run_binaries_phase(root, tag=st.get("tag") or "main", build_rust=False)
    # a fetch-fail here is non-fatal for repair if binaries already exist
    if any(r.severity == "fail" for r in bres) and (root / "bin" / "titan-kernel-rs").exists():
        cprint("  (binaries already present — skipping re-fetch)", role="text_muted")
    else:
        _render(bres)

    # regenerate + re-enable + restart the unit + health-gate
    rc = _render(run_systemd_phase(st, root, mode, default=False, api_port=api_port))
    return 1 if rc else 0


# ── uninstall ────────────────────────────────────────────────────────────────


def run_uninstall(*, purge: bool, assume_yes: bool = False) -> int:
    root = resolve_install_root()
    titan_id = resolve_install_titan_id(root)
    section(f"Uninstall — Titan {titan_id}")

    if purge:
        cprint("  ⚠ --purge will DELETE data/ AND ~/.titan/ — your identity, soul, and",
               role="error", bold=True)
        cprint("    Shamir shard. This is IRREVERSIBLE without your offline backup shard.",
               role="error", bold=True)
        if not assume_yes:
            try:
                ans = input("  Type the Titan id to confirm purge (or anything else to abort): ").strip()
            except EOFError:
                ans = ""
            if ans != titan_id:
                cprint("  Aborted — nothing deleted.", role="success")
                return 1

    # 1. disable + stop the unit, remove it
    _sudo(["systemctl", "disable", "--now", UNIT_NAME], explain="stop + disable Titan")
    _sudo(["rm", "-f", str(UNIT_DEST)], explain=f"remove the unit file {UNIT_DEST}")
    _sudo(["systemctl", "daemon-reload"], explain="reload systemd after removing the unit")

    # 2. runtime cleanup (shm, sockets, pid) via the generic cleanup script
    cs = cleanup_script(root)
    if cs.exists():
        subprocess.run(["bash", str(cs)],
                       env={"TITAN_ID": titan_id, "TITAN_ROOT": str(root), "PATH": "/usr/bin:/bin"})

    # 3. data
    if purge:
        for victim in (root / "data", Path.home() / ".titan"):
            if victim.exists():
                shutil.rmtree(victim, ignore_errors=True)
                cprint(f"  ✗ removed {victim}", role="warning")
        cprint("  Purge complete. The repo checkout itself was left in place.", role="text_strong")
    else:
        cprint(f"  Service removed. Your data/ + ~/.titan/ were KEPT (use --purge to delete).",
               role="success")
    return 0
