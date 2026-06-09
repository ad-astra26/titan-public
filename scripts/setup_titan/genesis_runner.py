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

DEFAULT_TITAN_ID = "titan"   # fresh-install default (fleet T1/T2/T3 set TITAN_ID
                             # explicitly via systemd, so this only names a NEW user's Titan)


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


def write_bootable_identity(
    install_root: Path,
    key_bytes: bytes,
    *,
    titan_id: str | None = None,
    titan_pubkey: str = "",
) -> Path:
    """Write the kernel-boot identity artifacts from a raw 64-byte keypair.

    The single primitive shared by genesis (Phase 6 materialization, below)
    and resurrection (`scripts/resurrection.py` Phase 4 first-breath). Both
    paths must produce the SAME two files the kernel loads at boot (SPEC
    G16(8) B1), so the format lives in exactly one place:

      - ``data/titan_identity_keypair.json`` — 0600, a JSON array of 64 ints
        (the ``solders.Keypair.from_bytes`` byte form the kernel reads).
      - ``data/titan_identity.json`` — ``{titan_id, titan_pubkey}``.

    Always (over)writes both from ``key_bytes``; idempotency / existence
    guards are the caller's concern. Returns the keypair Path.
    """
    if len(key_bytes) != 64:
        raise ValueError(
            f"bootable keypair must be 64 bytes, got {len(key_bytes)}"
        )
    kp = keypair_path(install_root)
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_text(json.dumps(list(key_bytes)))
    kp.chmod(0o600)
    tid = titan_id or _resolve_titan_id(install_root)
    identity_path(install_root).write_text(
        json.dumps({"titan_id": tid, "titan_pubkey": titan_pubkey}, indent=2)
    )
    return kp


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
    try:
        key_bytes = bytes(json.loads(authority.read_text()))
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
        return [Result("identity", "fail",
                       f"authority.json is not a valid keypair array: {e}")]

    titan_id = _resolve_titan_id(install_root)
    pubkey = ""
    rec = genesis_record_path(install_root)
    try:
        if rec.exists():
            pubkey = json.loads(rec.read_text()).get("titan_pubkey", "")
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    try:
        write_bootable_identity(install_root, key_bytes,
                                titan_id=titan_id, titan_pubkey=pubkey)
    except ValueError as e:
        return [Result("identity", "fail", f"bootable identity write failed: {e}")]

    # Wipe the stray repo-root copy — one plaintext keypair, in data/, 0600.
    try:
        authority.unlink()
    except OSError:
        pass
    return [Result("identity", "ok",
                   f"bootable identity materialized (titan_id={titan_id}) → {kp}")]


def run_genesis_phase(install_root: Path, mode: Mode, *, venv_python: Path,
                      simulate: bool = False) -> list[Result]:
    """Phase 6 body — returns a Result list.

    Idempotent on the BOOTABLE artifact (`data/titan_identity_keypair.json`),
    not `genesis_record.json` (which the ceremony writes early, before the burn
    — so a half-finished ceremony must NOT count as done).

    ``simulate`` runs the ceremony's mainnet-readiness rehearsal (#34/G7): the
    full on-chain ceremony is walked but every submit is stubbed (0 SOL, nothing
    minted). It always re-runs (no bootable artifact is produced to short-circuit
    on) and never materializes a boot identity.
    """
    kp = keypair_path(install_root)
    if kp.exists() and not simulate:
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
    if simulate:
        # Mainnet-readiness rehearsal (#34/G7): walk the FULL on-chain ceremony,
        # stubbing every submit — 0 SOL, nothing minted. readiness targets mainnet.
        cmd.extend(["--network", "mainnet" if mode == Mode.LOCAL else mode.value, "--simulate"])
    elif mode == Mode.LOCAL:
        cmd.append("--skip-onchain")
    else:
        # devnet / mainnet — the ceremony's only behavioral switch (Arweave +
        # the Burn are mainnet-only). mode.value == "devnet" | "mainnet".
        cmd.extend(["--network", mode.value])
    # Local + devnet keep the plaintext keypair (the T2/T3 boot model): the
    # kernel needs it at data/titan_identity_keypair.json. --keep-plaintext also
    # skips the interactive 'SOVEREIGN' burn prompt → the ceremony runs cleanly
    # non-interactively. Only mainnet performs the Burn (→ Resurrection to boot).
    # A --simulate rehearsal mints/burns nothing, so it keeps no plaintext.
    keep_plaintext = (not simulate) and mode in (Mode.LOCAL, Mode.DEVNET)
    if keep_plaintext:
        cmd.append("--keep-plaintext")

    cprint(f"  Invoking genesis ceremony: {' '.join(cmd)}", role="text_strong")
    if simulate:
        cprint("  --simulate: rehearsing the FULL mainnet ceremony — 0 SOL, nothing minted.",
               role="text_strong")
    elif not keep_plaintext:
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

    if simulate:
        return [Result("genesis", "ok",
                       "mainnet-readiness ceremony SIMULATED — every on-chain step prepared "
                       "and stubbed at submit; 0 SOL spent, nothing minted.")]

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
