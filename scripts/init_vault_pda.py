#!/usr/bin/env python3
"""
init_vault_pda.py — Idempotent initializer for the Titan ZK-Vault PDA.

Calls the Anchor program's `initialize_vault` instruction once for a given
authority wallet on a given Solana cluster, creating the vault PDA so
subsequent `commit_state` instructions land state roots on-chain.

Idempotent: if the PDA already exists, exits cleanly without a TX.

Used to bring up the T2/T3 shared-wallet vault on devnet (companion to
BUG-T2T3-VAULT-RPC-NETWORK-MISMATCH fix). The vault program is deployed
on devnet at the same PID as mainnet
(``52an8WjtfxpkCqZZ1AYFkaDTGb4RyNFFD9VQRVdxcpJw``); only the per-authority
PDA needs initialization.

Usage:
    # Initialize vault PDA for the T2/T3 shared wallet on devnet:
    source test_env/bin/activate
    python scripts/init_vault_pda.py \\
        --keypair /home/antigravity/.config/solana/id.json \\
        --rpc-url https://api.devnet.solana.com

    # Dry-run (derives PDA + checks existence; does NOT submit TX):
    python scripts/init_vault_pda.py \\
        --keypair /home/antigravity/.config/solana/id.json \\
        --rpc-url https://api.devnet.solana.com --dry-run

Exit codes:
    0  — PDA exists (already-initialized OR newly-initialized this run)
    1  — Script error (keypair load, RPC, TX failure, etc.)
    2  — Bad arguments

Safety:
    - Requires --confirm flag for the actual TX submission. Without it,
      runs in dry-run mode regardless of --rpc-url.
    - Refuses to run against mainnet-beta unless --i-know-this-is-mainnet
      is also passed (T1 vault was initialized 2026-04-06 during Genesis;
      re-initializing it would be a programming error caught here).
    - Prints the derived PDA + balance + estimated rent before any TX so
      the operator can verify they're targeting the right wallet/cluster.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Project root for imports (so this script runs as `python scripts/init_vault_pda.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("init_vault_pda")


def _load_keypair(path: Path):
    """Load a Solana keypair from a JSON file (array-of-bytes format)."""
    from solders.keypair import Keypair
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 64:
        raise ValueError(
            f"Keypair file {path} must contain a 64-byte JSON array; "
            f"got len={len(data) if isinstance(data, list) else type(data).__name__}"
        )
    return Keypair.from_bytes(bytes(data))


def _check_pda_exists(client, pda) -> bool:
    """Return True if the on-chain account at PDA already exists."""
    resp = client.get_account_info(pda)
    return resp.value is not None


def _confirm_signature(client, sig_str: str, timeout_s: float = 60.0) -> bool:
    """Poll get_signature_statuses until confirmed/finalized or timeout."""
    from solders.signature import Signature as SoldersSig
    sig = SoldersSig.from_string(sig_str)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        statuses = client.get_signature_statuses([sig]).value
        if statuses and statuses[0] is not None:
            status = statuses[0]
            if status.err is not None:
                log.error("[init_vault_pda] TX %s failed on-chain: %s",
                          sig_str[:24], status.err)
                return False
            confirmation = status.confirmation_status
            if confirmation is not None and str(confirmation).lower() in (
                "confirmed", "finalized", "confirmationstatus.confirmed",
                "confirmationstatus.finalized",
            ):
                log.info("[init_vault_pda] TX confirmed (%s): %s",
                         confirmation, sig_str[:24])
                return True
        time.sleep(2.0)
    log.error("[init_vault_pda] TX confirmation timeout (%.0fs): %s",
              timeout_s, sig_str[:24])
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Idempotent initializer for the Titan ZK-Vault PDA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument(
        "--keypair", required=True,
        help="Path to the authority wallet's JSON keypair file "
             "(64-byte array format).",
    )
    parser.add_argument(
        "--rpc-url", required=True,
        help="Solana RPC URL (e.g. https://api.devnet.solana.com).",
    )
    parser.add_argument(
        "--vault-program-id", default=None,
        help="Override vault program ID. Defaults to the configured "
             "VAULT_PROGRAM_ID in titan_plugin.utils.solana_client.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Derive PDA + check existence + print TX plan, but do NOT submit.",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Required for actual TX submission. Without it, runs as if "
             "--dry-run.",
    )
    parser.add_argument(
        "--i-know-this-is-mainnet", action="store_true",
        help="Required only when --rpc-url contains 'mainnet-beta'. T1 vault "
             "was initialized 2026-04-06; this flag exists to make any "
             "second mainnet init explicit.",
    )
    parser.add_argument(
        "--timeout-s", type=float, default=120.0,
        help="TX confirmation timeout (default 120s — Solana 'Confirmed' "
             "commitment can take 60-90s under network load; 60s caused "
             "a false-negative timeout on T2/T3 init 2026-04-29 11:59 "
             "even though the TX did land at PDA "
             "9T8nd6owTM94xu21hjeDpNcFoDFLAcZsQiMGaZFe9xrR).",
    )
    args = parser.parse_args(argv)

    # ── Sanity guards ─────────────────────────────────────────────────
    if "mainnet" in args.rpc_url and not args.i_know_this_is_mainnet:
        log.error(
            "[init_vault_pda] Refusing mainnet RPC without "
            "--i-know-this-is-mainnet. T1 vault was initialized 2026-04-06; "
            "re-running on mainnet is almost always a mistake."
        )
        return 2

    keypair_path = Path(args.keypair).expanduser()
    if not keypair_path.exists():
        log.error("[init_vault_pda] Keypair file not found: %s", keypair_path)
        return 1

    # ── Load deps + keypair ───────────────────────────────────────────
    try:
        from solana.rpc.api import Client
        from solders.message import Message
        from solders.transaction import Transaction
        from titan_plugin.utils.solana_client import (
            VAULT_PROGRAM_ID,
            build_vault_initialize_instruction,
            decode_vault_state,
            derive_vault_pda,
        )
    except ImportError as e:
        log.error("[init_vault_pda] Missing dependency: %s. "
                  "Run: source test_env/bin/activate", e)
        return 1

    program_id_str = args.vault_program_id or VAULT_PROGRAM_ID

    try:
        kp = _load_keypair(keypair_path)
    except Exception as e:
        log.error("[init_vault_pda] Failed to load keypair: %s", e)
        return 1
    authority_pubkey = kp.pubkey()

    # ── Derive PDA + check existence ─────────────────────────────────
    pda_result = derive_vault_pda(authority_pubkey, program_id_str)
    if pda_result is None:
        log.error("[init_vault_pda] Failed to derive vault PDA")
        return 1
    vault_pda, bump = pda_result

    log.info("[init_vault_pda] Authority wallet: %s", authority_pubkey)
    log.info("[init_vault_pda] Vault program:    %s", program_id_str)
    log.info("[init_vault_pda] Derived PDA:      %s (bump=%d)", vault_pda, bump)
    log.info("[init_vault_pda] RPC URL:          %s", args.rpc_url)

    client = Client(args.rpc_url)

    # Authority balance check (will pay rent ≈ 0.0014 SOL for 123-byte account
    # + ~5,000 lamports priority fee).
    try:
        balance = client.get_balance(authority_pubkey).value
        log.info("[init_vault_pda] Authority balance: %d lamports (%.4f SOL)",
                 balance, balance / 1e9)
        if balance < 5_000_000:  # 0.005 SOL minimum cushion
            log.warning(
                "[init_vault_pda] Authority balance very low — TX may fail. "
                "Consider funding before init."
            )
    except Exception as e:
        log.warning("[init_vault_pda] Could not fetch balance: %s "
                    "(continuing — may indicate RPC issue)", e)

    if _check_pda_exists(client, vault_pda):
        log.info(
            "[init_vault_pda] PDA already exists — vault is initialized. "
            "No TX needed. Decoded state:"
        )
        try:
            resp = client.get_account_info(vault_pda)
            state = decode_vault_state(bytes(resp.value.data))
            log.info("[init_vault_pda]   commit_count=%s sov_index=%s root=%s...",
                     state.get("commit_count"),
                     state.get("sovereignty_index"),
                     (state.get("latest_root") or "")[:32])
        except Exception as e:
            log.warning("[init_vault_pda]   (decode error: %s)", e)
        return 0

    log.info("[init_vault_pda] PDA does NOT exist on-chain — needs initialization.")

    if args.dry_run or not args.confirm:
        log.info(
            "[init_vault_pda] Dry-run mode (or --confirm not passed) — "
            "would submit initialize_vault TX with:"
        )
        log.info("[init_vault_pda]   accounts: vault_pda=%s authority=%s "
                 "system_program=11111111111111111111111111111111",
                 vault_pda, authority_pubkey)
        log.info("[init_vault_pda]   instruction discriminator: "
                 "0x30bfa32c47813fa4 (initialize_vault)")
        log.info(
            "[init_vault_pda] Re-run with --confirm to actually submit the TX."
        )
        return 0

    # ── Build + submit TX ─────────────────────────────────────────────
    ix = build_vault_initialize_instruction(authority_pubkey, program_id_str)
    if ix is None:
        log.error("[init_vault_pda] Failed to build initialize_vault instruction")
        return 1

    try:
        blockhash_resp = client.get_latest_blockhash()
        blockhash = blockhash_resp.value.blockhash
    except Exception as e:
        log.error("[init_vault_pda] Failed to fetch latest blockhash: %s", e)
        return 1

    msg = Message.new_with_blockhash([ix], authority_pubkey, blockhash)
    tx = Transaction.new_unsigned(msg)
    tx.sign([kp], blockhash)

    log.info("[init_vault_pda] Submitting initialize_vault TX...")
    try:
        sig_resp = client.send_raw_transaction(bytes(tx))
        sig_str = str(sig_resp.value)
    except Exception as e:
        log.error("[init_vault_pda] send_raw_transaction failed: %s", e)
        return 1
    log.info("[init_vault_pda] TX submitted: %s", sig_str)

    if not _confirm_signature(client, sig_str, timeout_s=args.timeout_s):
        return 1

    # ── Verify post-init ──────────────────────────────────────────────
    if not _check_pda_exists(client, vault_pda):
        log.error(
            "[init_vault_pda] Post-confirmation check: PDA still does not "
            "exist. This should not happen — investigate."
        )
        return 1
    log.info(
        "[init_vault_pda] ✓ Vault PDA initialized successfully. "
        "TX: %s | PDA: %s",
        sig_str, vault_pda,
    )

    # Decode the fresh state to give the operator a clean confirmation.
    try:
        resp = client.get_account_info(vault_pda)
        state = decode_vault_state(bytes(resp.value.data))
        log.info(
            "[init_vault_pda] Initial state: commit_count=%s sov_index=%s "
            "authority=%s",
            state.get("commit_count"),
            state.get("sovereignty_index"),
            state.get("authority"),
        )
    except Exception as e:
        log.warning("[init_vault_pda] Decode of fresh state failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
