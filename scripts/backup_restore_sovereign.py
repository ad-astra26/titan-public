#!/usr/bin/env python3
"""🜂 SOVEREIGN TITAN RESURRECTION PROTOCOL v1 (RFP §3B.A chunk 5J-3).

Bring a dead Titan back to life from **the wallet alone** — no local files, no
off-host mirror, no backup_records. The Arweave URLs live on-chain in the v=3
backup chain (chunk 5J-1/5J-2), so given only the reconstructed soul keypair
(Maker Shard-1 + on-chain Shard-3, 2-of-3) the protocol:

  1. Walks the Titan wallet's Solana signature history (newest → oldest).
  2. Decodes every v=3 backup memo; groups per-component memos by event.
  3. Decrypts each Arweave URL (Mode A) or reads it plaintext (Mode B).
  4. Orders events genesis → latest via ts (prev= gives tamper-evident linkage).
  5. Fetches each component tarball from Arweave; verifies sha256 == arc and the
     recomposed event_merkle_root == mrkl. Halts hard on any mismatch.
  6. Applies the chain into a scratch dir using the shipped Phase-6 primitives
     (apply_event_components — the tarballs self-describe per-file diff-mode, so
     genesis → latest application needs no on-chain type markers).
  7. On full success, atomically swaps scratch → data/ (with --commit).

This is the §3B.0(4) sovereignty-flaw closure: the trust root is the wallet
(plus SSS shards for Mode A). Nothing on local infra is required.

Mode A (data plaintext on Arweave, current T1) is fully supported. Mode B
(data ciphertext) additionally needs the AES iv on-chain to decrypt the tarball
during a sovereign restore — tracked as a follow-up; this tool halts with clear
guidance if it meets a Mode-B component.

The core `resurrect_from_chain()` takes injectable seams (sig lister / memo
fetcher / Arweave fetch) so it is fully unit-testable against a mock chain
(tests/test_backup_restore_sovereign.py) with zero network.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

# Repo root must be ahead of scripts/ on sys.path so `import titan_hcl` resolves
# to the package, not scripts/titan_hcl.py (the agent entry). scripts/ is added
# (appended, never ahead of root) only in main() for `import resurrection`.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from titan_hcl.logic.backup_memo_v3 import (  # noqa: E402
    derive_backup_url_key,
    parse_v3_memo,
    read_url,
)
from titan_hcl.logic.backup_crypto import (  # noqa: E402
    decrypt_component_tarball,
    derive_master_key,
)
from titan_hcl.logic.backup_restore import apply_event_components  # noqa: E402
from titan_hcl.logic.backup_zk_commit import compute_event_merkle_root  # noqa: E402

# RFP_backup_redesign_spine §7.C — the sovereign resurrection engine RELOCATED
# to titan_hcl.logic.restore_worker (ONE home). Re-exported here so the CLI +
# resurrection.py's call-time import + the test mocks resolve against this
# module's namespace (patch.object(sov, 'restore_body_from_chain') still works).
from titan_hcl.logic.restore_worker import (  # noqa: E402
    PROTOCOL_BANNER,
    ResurrectionResult,
    SigLister,
    MemoFetcher,
    ArweaveFetch,
    ArcToTarget,
    HALT_NO_CHAIN, HALT_NO_BASELINE, HALT_BROKEN_CHAIN, HALT_ARC_MISMATCH,
    HALT_EVENT_MERKLE_MISMATCH, HALT_MODE_B_UNSUPPORTED,
    HALT_MODE_B_DECRYPT_FAILED, HALT_APPLY_FAILED,
    resurrect_from_chain,
    restore_body_from_chain,
    _atomic_swap_into_data,
    RestoreWorker,
)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        prog="backup_restore_sovereign",
        description=PROTOCOL_BANNER + " — resurrect a Titan from the wallet alone "
                    "(on-chain v=3 backup chain; no local files).")
    p.add_argument("--titan", required=True, help="Titan id (e.g. T1).")
    p.add_argument("--shard1", help="Maker raw Shard-1 (hex). Prefer --shard1-stdin.")
    p.add_argument("--shard1-stdin", action="store_true",
                   help="Read Shard-1 from stdin (ps-safe, never on the command line).")
    p.add_argument("--titan-pubkey", default=None,
                   help="The Titan's PUBLIC wallet address (printed alongside "
                        "Shard-1; not a secret). Required for a fresh-box recovery "
                        "(no local genesis record). NO envelope needed.")
    p.add_argument("--install-root", default=_REPO_ROOT, help="Target install tree.")
    p.add_argument("--scratch-dir", default=None,
                   help="Scratch reconstruction dir (default: <install-root>/data.resurrect).")
    p.add_argument("--rpc-url", default=None, help="Solana RPC URL override (chain "
                   "walk: signatures + memos). Any standard RPC works.")
    p.add_argument("--das-rpc-url", default=None,
                   help="Optional DAS-capable RPC (Helius/Triton) for GenesisNFT "
                        "identity discovery. Shard-3 recovery uses --rpc-url and "
                        "never needs DAS; this only speeds NFT-based discovery.")
    p.add_argument("--network", default="mainnet", choices=["mainnet", "devnet"],
                   help="Arweave/Solana network (default: mainnet).")
    p.add_argument("--commit", action="store_true",
                   help="Atomically swap the reconstructed state into data/ on success "
                        "(default: leave it in the scratch dir for inspection).")
    p.add_argument("--verify-only", action="store_true",
                   help="Materialize the bootable identity in RECOVERY observation mode "
                        "(no on-chain writes / backups / X on first boot) — the live "
                        "restore-test isolation guard. Implies --commit.")
    p.add_argument("--best-effort", action="store_true",
                   help="Recover the MAXIMUM restorable state from a partially-broken "
                        "chain: an unreplayable per-file diff (e.g. a divergent baseline) "
                        "is logged + SKIPPED (file keeps its last-good bytes) instead of "
                        "halting. Status becomes 'resurrected_partial' with the skipped "
                        "list. Use when a clean strict restore halts on chain damage.")
    args = p.parse_args(argv)
    if args.verify_only:
        args.commit = True  # a verify-only test must produce a bootable, comparable box

    print("\n" + "═" * 64)
    print(PROTOCOL_BANNER)
    print("  Recovery from zero local state — trust root: the wallet (+ SSS).")
    print("═" * 64 + "\n")

    install_root = os.path.abspath(args.install_root)
    scratch_dir = args.scratch_dir or os.path.join(install_root, "data.resurrect")

    # Reconstruct the soul keypair from Shard-1 + on-chain Shard-3 (reuse the
    # shipped resurrection Phase-1 identity recovery — same 2-of-3 SSS path).
    shard1 = args.shard1
    if args.shard1_stdin and not shard1:
        shard1 = sys.stdin.readline().strip()
    if not shard1:
        _print("No Shard-1 provided (use --shard1-stdin and pipe it). Aborting.")
        return 2
    try:
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.append(scripts_dir)  # appended — never ahead of repo root
        import resurrection as _res
        from types import SimpleNamespace
        key_bytes, titan_pubkey, _kp, titan_id = _res.phase_1_identity(
            SimpleNamespace(shard1=shard1, shard1_file=None, shard2_local=False,
                            titan_id=args.titan, titan_pubkey=args.titan_pubkey,
                            das_rpc_url=args.das_rpc_url),
            install_root)
    except SystemExit:
        _print("Identity recovery failed (see above).")
        return 1
    except Exception as e:
        _print(f"Identity recovery error: {e}")
        return 1
    finally:
        shard1 = None  # drop the shard reference promptly

    # ── body restore: the SHARED sovereign chain engine (no manifest) ─────────
    try:
        result = restore_body_from_chain(
            key_bytes=key_bytes, titan_pubkey=titan_pubkey, titan_id=titan_id,
            install_root=install_root, scratch_dir=scratch_dir,
            rpc_url=args.rpc_url, network=args.network, commit=args.commit,
            best_effort=args.best_effort)
    except Exception as e:
        _print(f"Sovereign restore error: {e}")
        return 1

    if result.status not in ("resurrected", "resurrected_partial"):
        _print(f"HALTED: {result.halt_reason}")
        for e in result.errors[-5:]:
            _print(f"  · {e}")
        return 1

    if result.status == "resurrected_partial":
        _print(f"PARTIAL recovery: {result.events_applied} events applied; "
               f"{len(result.skipped_files)} file(s) kept at last-good (unreplayable "
               f"diffs — chain damage). This is the MAXIMUM restorable state.")
        for s in result.skipped_files[:12]:
            _print(f"  ⚠ kept-last-good: {s}")
        if len(result.skipped_files) > 12:
            _print(f"  … +{len(result.skipped_files) - 12} more")
    _print(f"Resurrected {result.events_applied} events into {scratch_dir}")
    if not args.commit:
        _print("Re-run with --commit to swap the reconstruction into data/.")
        return 0

    # ── restore_body_from_chain already swapped data/ into place (commit=True);
    # now materialize the bootable identity. The chain restore recovers the BODY
    # (data/); the kernel's B1 boot additionally needs the plaintext 0600 keypair
    # + hardware-bound soul_keypair.enc + RECOVERY flag — none of which live in
    # the backup (mainnet genesis burns the plaintext key). phase_4_first_breath
    # writes them from the already-reconstructed soul keypair (no second Shard-1
    # prompt), leaving a box the kernel can actually boot. ────────────────────────
    try:
        _res.phase_4_first_breath(
            key_bytes, titan_pubkey, titan_id,
            install_root=install_root, verify_only=args.verify_only)
    except SystemExit:
        _print("Bootable-identity materialization failed (see above).")
        return 1
    except Exception as e:
        _print(f"Bootable-identity materialization error: {e}")
        return 1
    finally:
        key_bytes = None  # drop the reconstructed key promptly
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
