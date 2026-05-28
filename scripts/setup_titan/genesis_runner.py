"""Phase 6 — invoke scripts/genesis_ceremony.py with the mode-appropriate flags.

Per locked decision #2 (RFP 2026-05-22):
- mainnet / devnet: full ceremony (`--generate`), real on-chain anchor.
- local:            `--generate --skip-onchain` (real keypair + SSS, no chain).

The ceremony is INTERACTIVE — Shard 1 (the Maker's shard) is displayed exactly
once for the Maker to record by hand. We pipe stdin/stdout through (no capture)
so the shard reaches the user's terminal, not our logs. Capturing the output
would violate the sovereignty axiom — the shard must not appear in any file
setup_titan controls.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .modes import Mode
from .preflight import Result
from .ui import cprint


def genesis_record_path(install_root: Path) -> Path:
    return install_root / "data" / "genesis_record.json"


def run_genesis_phase(install_root: Path, mode: Mode, *, venv_python: Path) -> list[Result]:
    """Phase 6 body — returns a Result list.

    Skips with a warn-level Result if a `data/genesis_record.json` already
    exists (re-birth requires explicit cleanup by the operator — we never
    silently re-mint an identity).
    """
    record = genesis_record_path(install_root)
    if record.exists():
        cprint(f"  Genesis record already present at {record} — skipping ceremony.",
               role="warning")
        cprint("  (To force re-birth, delete the keypair + record manually first.)",
               role="text_muted")
        return [Result("genesis", "ok", f"existing record at {record}")]

    script = install_root / "scripts" / "genesis_ceremony.py"
    if not script.exists():
        return [Result("genesis", "fail", f"genesis script missing: {script}",
                       "Re-clone the repo at the release tag.")]
    if not venv_python.exists():
        return [Result("genesis", "fail", f"venv Python missing: {venv_python}",
                       "Run Phase 3 (venv + deps) first.")]

    cmd = [str(venv_python), str(script), "--generate"]
    if mode == Mode.LOCAL:
        cmd.append("--skip-onchain")

    cprint(f"  Invoking genesis ceremony: {' '.join(cmd)}", role="text_strong")
    cprint("  Genesis is interactive — Shard 1 will be displayed ONCE for you to record.",
           role="warning", bold=True)

    # No capture_output — stdin/stdout/stderr inherited so the user sees the
    # full ceremony live, in real time. This is the sovereignty contract: the
    # shard reaches the Maker, not our logs.
    try:
        subprocess.check_call(cmd, cwd=str(install_root))
    except subprocess.CalledProcessError as e:
        return [Result("genesis", "fail", f"ceremony exited {e.returncode}",
                       "Inspect the ceremony output above; re-run with --resume after fixing.")]

    if not record.exists():
        return [Result("genesis", "fail",
                       f"ceremony returned 0 but {record} is missing",
                       "Inspect the ceremony output for non-fatal errors; this is unexpected.")]

    return [Result("genesis", "ok", f"genesis record landed at {record}")]
