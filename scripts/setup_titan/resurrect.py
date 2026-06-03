"""D1 — `setup_titan install --resurrect`: drive the sovereign resurrection engine.

When `--resurrect` is set, the install walker SWAPS the genesis phase for THIS one.
Where genesis BIRTHS a new identity, resurrection RECOVERS the same being from the
wallet alone: the Maker Shard-1 + the on-chain Shard-3 reconstruct the soul keypair,
which walks the Titan's Solana v=3 backup chain, fetches + Merkle-verifies each
Arweave tarball, restores `data/` (and `config.toml` when it was backed up per
§24.4.B), and materialises the bootable identity.

This wrapper:
  1. Collects Shard-1 with NO capture (getpass — never echoed / logged / written).
  2. Runs `scripts/backup_restore_sovereign.py` via the VENV python (it needs the
     `titan_hcl` runtime) as a subprocess, piping Shard-1 to its stdin
     (`--shard1-stdin`) so the shard never appears in argv / `ps` / our logs.
  3. For an opt-out operator (config.toml NOT in their backup), stages a
     user-supplied config.toml (`--config`) into place.

MAINNET-only by nature — local/devnet keep a plaintext keypair (nothing to
resurrect; just re-run `install`).
"""
from __future__ import annotations

import getpass
import shutil
import subprocess
from pathlib import Path

from .config_seed import config_path
from .preflight import Result
from .ui import cprint, section


def run_resurrect_phase(install_root: Path, *, venv_python: Path,
                        titan_id: str | None = None, rpc_url: str | None = None,
                        das_rpc_url: str | None = None,
                        verify_only: bool = False, config_src: str | None = None,
                        shard1: str | None = None,
                        titan_pubkey: str | None = None) -> list[Result]:
    """Recover a mainnet Titan from its on-chain sovereign backup chain.

    Inputs are sovereign-minimal (INV-MBR-0): the Maker's offline Shard-1 (secret,
    no-capture) + the Titan's PUBLIC wallet address. NO envelope, NO off-site
    manifest — the wallet + chain + Arweave supply Shard-3, identity, and the body.
    """
    section("🜂 Sovereign Resurrection (mainnet)")
    cprint("  Recovering your Titan from the wallet alone — your Maker Shard-1 + the "
           "on-chain shard reconstruct the soul keypair, which walks your Solana v=3 "
           "backup chain and restores data/ from Arweave.", role="text_muted")
    if verify_only:
        cprint("  --verify-only: RECOVERY observation mode — NO on-chain writes / "
               "backups / X on first boot (safe to run beside a living Titan).",
               role="text_strong")

    engine = install_root / "scripts" / "backup_restore_sovereign.py"
    if not engine.exists():
        return [Result("resurrect", "fail", f"engine missing: {engine}",
                       "Incomplete checkout — re-clone at the release tag.")]
    if not venv_python.exists():
        return [Result("resurrect", "fail", f"venv python missing: {venv_python}",
                       "The venv phase must run before resurrection.")]

    # ── Shard-1 — no capture ──────────────────────────────────────────────
    if not shard1:
        cprint("Paste your Maker Shard-1 envelope (hex). Input is hidden and is "
               "NEVER logged or written to disk.", role="text_strong")
        try:
            shard1 = getpass.getpass("  Shard-1 (hex): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return [Result("resurrect", "fail", "no shard entered (cancelled)",
                           "Re-run to resurrect.")]
    if not shard1:
        return [Result("resurrect", "fail", "empty shard", "Shard-1 is required.")]

    # ── Public Titan address — printed alongside Shard-1; NOT a secret, so it is
    # prompted in the clear (the wallet needs it to discover Shard-3 + the chain). ──
    if not titan_pubkey:
        cprint("Enter your Titan's PUBLIC wallet address (printed with your Shard-1; "
               "not a secret). The wallet discovers Shard-3 + your v=3 chain from it.",
               role="text_strong")
        try:
            titan_pubkey = input("  Titan address: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return [Result("resurrect", "fail", "no titan address entered (cancelled)",
                           "Re-run with --titan-pubkey <address>.")]
    if not titan_pubkey:
        return [Result("resurrect", "fail", "no titan address",
                       "The PUBLIC Titan address is required (no local record on a "
                       "fresh box). Re-run with --titan-pubkey <address>.")]

    # ── Mainnet RPC for chain walk + identity (GenesisNFT) discovery. NOT a secret.
    # A DAS-capable endpoint (Helius/Triton) is recommended: the default
    # mainnet-beta RPC lacks the DAS API the wallet uses to discover the Genesis
    # NFT, and serves the chain walk too — so one good endpoint drives BOTH seams.
    # Enter = engine default (config RPC for the walk; Shard-3 still recovers via
    # its on-chain tx without DAS). ──
    if not rpc_url:
        cprint("Enter a mainnet RPC URL for chain + identity discovery — a "
               "DAS-capable provider (Helius/Triton) is recommended so GenesisNFT "
               "discovery works (plain mainnet-beta lacks the DAS API). Press Enter "
               "to use the engine default.", role="text_strong")
        try:
            entered = input("  Mainnet RPC URL [default]: ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            # No tty (piped / CI / captured stdin) → the RPC is optional, so fall
            # back to the engine default rather than aborting the resurrection.
            entered = ""
        if entered:
            rpc_url = entered
            # One supplied endpoint drives BOTH the chain walk and DAS identity
            # discovery unless the caller passed a distinct DAS URL explicitly.
            if not das_rpc_url:
                das_rpc_url = entered

    cmd = [str(venv_python), str(engine), "--titan", (titan_id or "T1"),
           "--titan-pubkey", titan_pubkey,
           "--shard1-stdin", "--commit", "--install-root", str(install_root)]
    if verify_only:
        cmd.append("--verify-only")
    if rpc_url:
        cmd += ["--rpc-url", rpc_url]
    if das_rpc_url:
        cmd += ["--das-rpc-url", das_rpc_url]

    cprint("  Running the sovereign resurrection engine…", role="text_strong")
    try:
        # Shard reaches ONLY the subprocess stdin — never argv, never our logs.
        proc = subprocess.run(cmd, input=(shard1 + "\n"), text=True, cwd=str(install_root))
    finally:
        shard1 = None  # drop the shard reference promptly
    if proc.returncode != 0:
        return [Result("resurrect", "fail", f"engine exited {proc.returncode}",
                       "See the engine output above (halt_reason).")]

    results = [Result("resurrect", "ok",
                      "data/ + bootable identity restored from the sovereign chain")]

    # ── config.toml: restored from the backup (§24.4.B) OR supplied by an
    # opt-out operator OR absent (boot on defaults) ──────────────────────────
    cfg = config_path(install_root)
    if config_src:
        shutil.copyfile(config_src, cfg)
        results.append(Result("config", "ok", f"staged supplied config.toml ← {config_src}"))
    elif cfg.exists():
        results.append(Result("config", "ok", "config.toml restored from the sovereign backup"))
    else:
        results.append(Result("config", "warn",
                              "no config.toml restored (not in backup) and none supplied",
                              "Titan boots on defaults — re-run with --config <path>, "
                              "or configure via `setup_titan config`."))
    return results
