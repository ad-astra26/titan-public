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

import json
import subprocess
import sys
from pathlib import Path

from .modes import Mode
from .preflight import Result
from .ui import cprint

DEFAULT_TITAN_ID = "T1"   # canonical single-node id (matches resolve_titan_id fallback)


def genesis_record_path(install_root: Path) -> Path:
    return install_root / "data" / "genesis_record.json"


def keypair_path(install_root: Path) -> Path:
    """The PLAINTEXT keypair the kernel loads at boot (SPEC G16(8) B1)."""
    return install_root / "data" / "titan_identity_keypair.json"


def identity_path(install_root: Path) -> Path:
    return install_root / "data" / "titan_identity.json"


def _resolve_titan_id(install_root: Path) -> str:
    ident = identity_path(install_root)
    try:
        if ident.exists():
            tid = json.loads(ident.read_text()).get("titan_id")
            if tid:
                return str(tid)
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return DEFAULT_TITAN_ID


def _materialize_bootable_identity(install_root: Path) -> list[Result]:
    """Turn the kept plaintext keypair into the files the kernel boots from.

    A born sovereign Titan has its keypair written by the ceremony as
    `authority.json` (repo root). T2/T3 prove the devnet/local boot model is a
    plain `data/titan_identity_keypair.json` (0600) + `data/titan_identity.json`
    — no burn, no Resurrection. This bridges the ceremony output to that model.
    """
    authority = install_root / "authority.json"
    kp = keypair_path(install_root)
    if kp.exists():
        return [Result("identity", "ok", f"bootable keypair already present: {kp}")]
    if not authority.exists():
        return [Result("identity", "fail",
                       "no plaintext keypair to materialize (authority.json absent)",
                       "Genesis must run with --keep-plaintext for local/devnet (mainnet "
                       "burns the key — boot then needs Resurrection / setup_titan restore).")]
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_bytes(authority.read_bytes())
    kp.chmod(0o600)
    # Wipe the stray repo-root copy — one plaintext keypair, in data/, 0600.
    try:
        authority.unlink()
    except OSError:
        pass

    titan_id = _resolve_titan_id(install_root)
    pubkey = ""
    rec = genesis_record_path(install_root)
    try:
        if rec.exists():
            pubkey = json.loads(rec.read_text()).get("titan_pubkey", "")
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    ident = identity_path(install_root)
    if not ident.exists():
        ident.write_text(json.dumps({"titan_id": titan_id, "titan_pubkey": pubkey}, indent=2))
    return [Result("identity", "ok",
                   f"bootable identity materialized (titan_id={titan_id}) → {kp}")]


def run_genesis_phase(install_root: Path, mode: Mode, *, venv_python: Path) -> list[Result]:
    """Phase 6 body — returns a Result list.

    Idempotent on the BOOTABLE artifact (`data/titan_identity_keypair.json`),
    not `genesis_record.json` (which the ceremony writes early, before the burn
    — so a half-finished ceremony must NOT count as done).
    """
    kp = keypair_path(install_root)
    if kp.exists():
        cprint(f"  Bootable identity already present at {kp} — skipping ceremony.",
               role="warning")
        cprint("  (To force re-birth, delete data/titan_identity_keypair.json + "
               "genesis_record.json first.)", role="text_muted")
        return [Result("genesis", "ok", f"existing identity at {kp}")]

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
    # Local + devnet keep the plaintext keypair (the T2/T3 boot model): the
    # kernel needs it at data/titan_identity_keypair.json. --keep-plaintext also
    # skips the interactive 'SOVEREIGN' burn prompt → the ceremony runs cleanly
    # non-interactively. Only mainnet performs the Burn (→ Resurrection to boot).
    keep_plaintext = mode in (Mode.LOCAL, Mode.DEVNET)
    if keep_plaintext:
        cmd.append("--keep-plaintext")

    cprint(f"  Invoking genesis ceremony: {' '.join(cmd)}", role="text_strong")
    if not keep_plaintext:
        cprint("  Genesis is interactive — Shard 1 will be displayed ONCE for you to record.",
               role="warning", bold=True)

    # No capture_output — stdin/stdout/stderr inherited so the Maker sees the
    # full ceremony live (and, for mainnet, records Shard 1). This is the
    # sovereignty contract: the shard reaches the Maker, never our logs.
    try:
        subprocess.check_call(cmd, cwd=str(install_root))
    except subprocess.CalledProcessError as e:
        return [Result("genesis", "fail", f"ceremony exited {e.returncode}",
                       "Inspect the ceremony output above; re-run with --resume after fixing.")]

    record = genesis_record_path(install_root)
    if not record.exists():
        return [Result("genesis", "fail",
                       f"ceremony returned 0 but {record} is missing",
                       "Inspect the ceremony output for non-fatal errors; this is unexpected.")]

    results = [Result("genesis", "ok", f"genesis ceremony complete ({record.name})")]
    if keep_plaintext:
        # Bridge the ceremony output to the kernel's bootable-identity contract.
        results += _materialize_bootable_identity(install_root)
    else:
        results.append(Result("identity", "warn",
                              "mainnet burn complete — plaintext keypair deleted.",
                              "This Titan boots only after Resurrection (2-of-3 Shamir) via "
                              "`setup_titan restore` (W1.5)."))
    return results
