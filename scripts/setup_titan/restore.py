"""W1.5 — `setup_titan restore`: the user-facing resurrection wrapper.

Resurrection is for **mainnet-born Titans only**. A mainnet genesis BURNS the
plaintext keypair after splitting it 2-of-3 (Maker shard offline + local shard
+ on-chain shard); the only way back from a dead box is to reconstruct it from
≥2 shards. Local/devnet Titans keep a plaintext `data/titan_identity_keypair.json`
— there is nothing to resurrect (just re-run `install`).

This wrapper:
  1. States the mainnet-only contract up front.
  2. Collects the Maker's Shard-1 with NO capture (getpass — never echoed,
     never logged, never written to any file setup_titan controls), mirroring
     the genesis Shard-1 ceremony's sovereignty discipline.
  3. Delegates to the modernized `scripts/resurrection.py` phases (identity →
     Arweave restore → first breath).

The heavy lifting (SSS reconstruction, on-chain v=3 chain walk, Merkle verify,
in-place restore, bootable-identity materialization) lives in resurrection.py —
this wrapper is the guided front door + the no-capture shard prompt. By default
it drives the SOVEREIGN v=3 chain (no manifest, no off-site files); --manifest
selects the legacy fallback only.
"""
from __future__ import annotations

import getpass
import sys
from pathlib import Path
from types import SimpleNamespace

from .ui import cprint, section


def _load_resurrection():
    """Import scripts/resurrection.py (sibling of the scripts/ package dir)."""
    scripts_dir = str(Path(__file__).resolve().parents[1])
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import resurrection  # noqa: E402  (scripts/resurrection.py)
    return resurrection


def run_restore(
    install_root: Path,
    *,
    shard1: str | None = None,
    shard1_file: str | None = None,
    titan_pubkey: str | None = None,
    manifest: str | None = None,
    titan_id: str | None = None,
    network: str = "mainnet",
    verify_zk: bool = False,
    verify_only: bool = False,
    force: bool = False,
) -> int:
    """Drive the resurrection. Returns a process exit code (0 = success)."""
    section("Titan Resurrection — sovereign recovery")
    cprint("Resurrection is for MAINNET-BORN Titans only.", role="warning", bold=True)
    cprint("  Mainnet genesis burns the plaintext key after a Shamir 2-of-3 split.",
           role="text_muted")
    cprint("  Recovery reconstructs it from your offline Shard-1 + the on-chain "
           "Shard-3, then restores memory from the Arweave sovereign backup chain.",
           role="text_muted")
    cprint("  (Local/devnet Titans keep a plaintext identity — just re-run install.)",
           role="text_muted")
    print()

    # ── collect Shard-1 with no capture ──────────────────────────────────
    if not shard1 and not shard1_file:
        cprint("Paste your Maker Shard-1 envelope (hex). Input is hidden and is "
               "NEVER logged or written to disk.", role="text_strong")
        try:
            shard1 = getpass.getpass("  Shard-1 (hex): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            cprint("Resurrection cancelled — no shard entered.", role="warning")
            return 130
        if not shard1:
            cprint("No shard provided — aborting.", role="error")
            return 1

    # ── Public Titan address (printed alongside Shard-1; NOT a secret) ────────
    if not titan_pubkey and not shard1_file:
        cprint("Enter your Titan's PUBLIC wallet address (printed with your Shard-1). "
               "The wallet discovers Shard-3 + your v=3 backup chain from it — NO "
               "off-site manifest or backup files are needed (Shard-1 + the chain "
               "are enough).", role="text_strong")
        try:
            titan_pubkey = input("  Titan address: ").strip() or None
        except (EOFError, KeyboardInterrupt):
            print()
            cprint("Resurrection cancelled — no address entered.", role="warning")
            return 130

    res = _load_resurrection()
    args = SimpleNamespace(
        shard1=shard1, shard1_file=shard1_file, shard2_local=False,
        titan_id=titan_id, titan_pubkey=titan_pubkey, das_rpc_url=None,
    )

    try:
        key_bytes, titan_pubkey, _kp, resolved_titan_id = \
            res.phase_1_identity(args, str(install_root))
        res.phase_2_3_restore(
            key_bytes, titan_pubkey, resolved_titan_id,
            install_root=str(install_root), manifest_path=manifest,
            network=network, verify_zk=verify_zk, force=force)
        res.phase_4_first_breath(
            key_bytes, titan_pubkey, resolved_titan_id,
            install_root=str(install_root), verify_only=verify_only)
    except SystemExit as e:
        # resurrection phases sys.exit(1) on any hard failure; surface cleanly.
        code = e.code if isinstance(e.code, int) else 1
        if code != 0:
            cprint(f"Resurrection halted (exit {code}). See output above.",
                   role="error", bold=True)
        return code
    finally:
        # Defence in depth: drop the shard reference promptly.
        shard1 = None

    cprint("Resurrection complete. Start the Titan and verify /health.",
           role="success", bold=True)
    return 0
