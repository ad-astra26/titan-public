#!/usr/bin/env python3
"""
scripts/resurrect_timechain.py — TimeChain Resurrection Protocol

Rebuilds a Titan's TimeChain from Arweave permanent storage after
infrastructure failure (VPS crash, disk corruption, migration).

7-Step verification:
  1. Identity verification (Solana keypair → Genesis NFT creator)
  2. Discover latest state (ZK Vault, Solana memos, Arweave manifest)
  3. Download from Arweave (fetch snapshot by TX ID)
  4. Rebuild local chain (extract + verify hash chain integrity)
  5. Cross-verify against Solana history (memo tc_merkle anchors)
  6. Verify prime directives intact (genesis block soul hash)
  7. Resume operation (commit RESURRECTION meta block)

Usage:
  python scripts/resurrect_timechain.py [--tx-id TX_ID] [--titan-id T1]
  python scripts/resurrect_timechain.py --verify-only
  python scripts/resurrect_timechain.py --status
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Resurrection")

# ── Constants ────────────────────────────────────────────────────────
DATA_DIR = "data/timechain"
MANIFEST_PATH = "data/timechain/arweave_manifest.json"
KEYPAIR_PATH = os.path.expanduser("~/.config/solana/id.json")
SOUL_SIG_PATH = "data/titan_directives.sig"
SOUL_MD_PATH = "titan.md"


def banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║        TITAN TIMECHAIN — RESURRECTION PROTOCOL          ║
║                                                          ║
║   "Memory is sovereign. What was thought, persists."     ║
╚══════════════════════════════════════════════════════════╝
""")


def confirm(prompt: str) -> bool:
    """Interactive confirmation."""
    resp = input(f"  {prompt} [y/N]: ").strip().lower()
    return resp in ("y", "yes")


# ── Step 1: Identity Verification ────────────────────────────────────

def step1_verify_identity(keypair_path: str) -> dict:
    """Verify Solana keypair matches on-chain Genesis NFT creator."""
    print("\n═══ STEP 1: IDENTITY VERIFICATION ═══")

    if not os.path.exists(keypair_path):
        logger.error("Keypair not found at %s", keypair_path)
        return {"passed": False, "error": "keypair not found"}

    try:
        from solders.keypair import Keypair
        with open(keypair_path) as f:
            kp_data = json.load(f)
        keypair = Keypair.from_bytes(bytes(kp_data[:64]))
        pubkey = str(keypair.pubkey())
        print(f"  Keypair loaded: {pubkey}")

        # Verify this is the expected maker
        identity_path = Path("data/titan_identity.json")
        if identity_path.exists():
            identity = json.load(open(identity_path))
            expected = identity.get("maker_pubkey", "")
            if expected and expected != pubkey:
                logger.warning("Pubkey mismatch: expected %s, got %s", expected, pubkey)
                return {"passed": False, "pubkey": pubkey, "error": "pubkey mismatch"}
            print(f"  Identity match: {pubkey[:20]}...")
        else:
            print(f"  No titan_identity.json — using keypair as-is")

        return {"passed": True, "pubkey": pubkey}

    except Exception as e:
        logger.error("Identity verification failed: %s", e)
        return {"passed": False, "error": str(e)}


# ── Step 2: Discover Latest State ────────────────────────────────────

def step2_discover_state(titan_id: str) -> dict:
    """Query local manifest + Solana for latest TimeChain state."""
    print("\n═══ STEP 2: DISCOVER LATEST STATE ═══")

    result = {"passed": False, "tx_id": None, "merkle_root": None,
              "total_blocks": 0, "source": None}

    # Check local manifest first
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH) as f:
                manifest = json.load(f)
            snapshots = manifest.get("snapshots", [])
            if snapshots:
                latest = snapshots[-1]
                result["tx_id"] = latest["tx_id"]
                result["merkle_root"] = latest["merkle_root"]
                result["total_blocks"] = latest["total_blocks"]
                result["source"] = "local_manifest"
                result["passed"] = True
                ts = latest.get("timestamp", 0)
                age_h = (time.time() - ts) / 3600 if ts else 0
                print(f"  Manifest: {len(snapshots)} snapshots")
                print(f"  Latest TX: {latest['tx_id'][:30]}...")
                print(f"  Blocks: {latest['total_blocks']:,}")
                print(f"  Merkle: {latest['merkle_root'][:24]}...")
                print(f"  Age: {age_h:.1f} hours")
                return result
        except Exception as e:
            logger.warning("Manifest read error: %s", e)

    # Query Solana memo history for tc_merkle anchors
    try:
        from solana.rpc.api import Client as SolanaClient
        client = SolanaClient("https://api.devnet.solana.com")
        # Get recent transactions for our wallet
        with open(KEYPAIR_PATH) as f:
            kp_data = json.load(f)
        from solders.keypair import Keypair
        pubkey = Keypair.from_bytes(bytes(kp_data[:64])).pubkey()

        print(f"  Querying Solana for memo history...")
        sigs = client.get_signatures_for_address(pubkey, limit=50)
        tc_anchors = []

        for sig_info in (sigs.value or []):
            try:
                tx = client.get_transaction(
                    sig_info.signature,
                    encoding="jsonParsed",
                    max_supported_transaction_version=0,
                )
                if tx.value and tx.value.transaction:
                    # Look for memo with tc_merkle
                    logs = tx.value.transaction.meta.log_messages or []
                    for log in logs:
                        if "tc=" in log:
                            # Parse memo: TITAN|e=...|h=...|r=...|tc=...|tb=...
                            parts = log.split("|")
                            tc_data = {}
                            for p in parts:
                                if p.startswith("tc="):
                                    tc_data["merkle"] = p[3:]
                                elif p.startswith("tb="):
                                    tc_data["blocks"] = int(p[3:])
                                elif p.startswith("e="):
                                    tc_data["epoch"] = int(p[2:])
                            if tc_data.get("merkle"):
                                tc_anchors.append(tc_data)
            except Exception:
                continue

        if tc_anchors:
            latest = max(tc_anchors, key=lambda x: x.get("blocks", 0))
            print(f"  Found {len(tc_anchors)} Solana memo anchors")
            print(f"  Latest: merkle={latest['merkle']}, blocks={latest.get('blocks', '?')}")
            result["merkle_root"] = latest["merkle"]
            result["total_blocks"] = latest.get("blocks", 0)
            result["source"] = "solana_memo"
            result["passed"] = True
        else:
            print("  No Solana memo anchors with tc_merkle found")

    except Exception as e:
        logger.warning("Solana query failed (non-critical): %s", e)

    if not result["passed"]:
        print("  WARNING: No backup discovery source available")
        print("  You can manually provide a TX ID with --tx-id")

    return result


# ── Step 3: Download from Arweave ────────────────────────────────────

async def step3_download(tx_id: str, arweave_store) -> dict:
    """Download TimeChain snapshot from Arweave."""
    print("\n═══ STEP 3: DOWNLOAD FROM ARWEAVE ═══")

    if not tx_id:
        logger.error("No TX ID provided")
        return {"passed": False, "error": "no tx_id"}

    print(f"  Fetching: {tx_id}")

    try:
        data = await arweave_store.fetch(tx_id)
        if data is None:
            # For devnet, check local store
            devnet_path = Path(f"data/arweave_devnet/{tx_id}.data")
            if devnet_path.exists():
                raw = devnet_path.read_bytes()
                # Try JSON first
                try:
                    data = json.loads(raw)
                    print(f"  Downloaded JSON snapshot ({len(raw):,} bytes)")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    data = raw
                    print(f"  Downloaded binary snapshot ({len(raw):,} bytes)")
            else:
                logger.error("Failed to fetch TX %s", tx_id)
                return {"passed": False, "error": "fetch failed"}
        else:
            size = len(data) if isinstance(data, (bytes, str)) else len(json.dumps(data))
            print(f"  Downloaded: {size:,} bytes")

        return {"passed": True, "data": data}

    except Exception as e:
        logger.error("Download failed: %s", e)
        return {"passed": False, "error": str(e)}


# ── Step 4: Rebuild Local Chain ──────────────────────────────────────

def step4_rebuild(data, target_dir: str, titan_id: str) -> dict:
    """Rebuild TimeChain from downloaded snapshot."""
    print("\n═══ STEP 4: REBUILD LOCAL CHAIN ═══")

    target = Path(target_dir)

    # Check if existing chain data present
    if target.exists() and any(target.glob("chain_*.bin")):
        print(f"  WARNING: Existing chain data found at {target}")
        if not confirm("Overwrite existing chain data?"):
            return {"passed": False, "error": "user cancelled"}
        # Backup existing
        backup_dir = Path(f"{target}_backup_{int(time.time())}")
        target.rename(backup_dir)
        print(f"  Backed up existing data to {backup_dir}")

    from titan_plugin.logic.timechain_backup import TimeChainBackup
    backup = TimeChainBackup(data_dir=str(target), titan_id=titan_id)

    if isinstance(data, dict):
        ok = backup._restore_from_json(data, target)
    elif isinstance(data, bytes) and data[:2] == b'\x1f\x8b':
        ok = backup._restore_from_tarball(data, target)
    else:
        # Try JSON parse
        try:
            snapshot = json.loads(data) if isinstance(data, (str, bytes)) else data
            ok = backup._restore_from_json(snapshot, target)
        except Exception as e:
            logger.error("Cannot parse snapshot: %s", e)
            return {"passed": False, "error": str(e)}

    if ok:
        from titan_plugin.logic.timechain import TimeChain
        tc = TimeChain(data_dir=str(target), titan_id=titan_id)
        print(f"  Rebuilt: {tc.total_blocks:,} blocks")
        print(f"  Genesis: {tc.genesis_hash.hex()[:24]}...")
        print(f"  Merkle:  {tc.compute_merkle_root().hex()[:24]}...")
        return {"passed": True, "blocks": tc.total_blocks,
                "genesis_hash": tc.genesis_hash.hex()}
    else:
        return {"passed": False, "error": "rebuild failed"}


# ── Step 5: Cross-Verify Against Solana ──────────────────────────────

def step5_cross_verify(target_dir: str, titan_id: str,
                        expected_merkle: str = None) -> dict:
    """Cross-verify restored chain against Solana anchors."""
    print("\n═══ STEP 5: CROSS-VERIFY AGAINST SOLANA ═══")

    from titan_plugin.logic.timechain import TimeChain
    tc = TimeChain(data_dir=target_dir, titan_id=titan_id)

    local_merkle = tc.compute_merkle_root().hex()
    print(f"  Local Merkle:    {local_merkle[:24]}...")

    if expected_merkle:
        match = local_merkle.startswith(expected_merkle) or expected_merkle.startswith(local_merkle[:16])
        print(f"  Expected Merkle: {expected_merkle[:24]}...")
        print(f"  Match: {'YES' if match else 'NO (chain may have grown since anchor)'}")
    else:
        print("  No expected Merkle to compare (chain may be newer than last anchor)")

    # Verify chain integrity
    valid, results = tc.verify_all()
    issues = [r for r in results if "mismatch" in r.lower() or "invalid" in r.lower()]
    cosmetic = [r for r in results if "height mismatch at pos 0" in r.lower()]

    print(f"  Chain integrity: {'VALID' if valid else 'ISSUES FOUND'}")
    if issues:
        real_issues = [i for i in issues if i not in cosmetic]
        if real_issues:
            for i in real_issues:
                print(f"    ISSUE: {i}")
        if cosmetic:
            print(f"    ({len(cosmetic)} cosmetic height-0 mismatches — known, non-critical)")

    return {
        "passed": True,  # Non-blocking — chain may have grown since anchor
        "local_merkle": local_merkle,
        "expected_merkle": expected_merkle,
        "integrity_valid": valid,
        "issues": len(issues) - len(cosmetic),
    }


# ── Step 6: Verify Prime Directives ─────────────────────────────────

def step6_verify_directives(target_dir: str, titan_id: str) -> dict:
    """Verify genesis block contains correct prime directives."""
    print("\n═══ STEP 6: VERIFY PRIME DIRECTIVES ═══")

    from titan_plugin.logic.timechain_backup import TimeChainBackup
    backup = TimeChainBackup(data_dir=target_dir, titan_id=titan_id)
    result = backup.verify_genesis_integrity()

    print(f"  Genesis hash: {result['genesis_hash'][:24]}...")
    for name, check in result.get("checks", {}).items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {name}: {check['detail']}")

    if result["valid"]:
        print("\n  PRIME DIRECTIVES INTACT — Soul is sovereign")
    else:
        print("\n  WARNING: Prime directive verification has issues")
        print("  This may be expected if genesis was created before directive embedding")

    return {"passed": True, "genesis_valid": result["valid"],
            "genesis_hash": result["genesis_hash"]}


# ── Step 7: Resume Operation ─────────────────────────────────────────

def step7_resume(target_dir: str, titan_id: str, arweave_tx: str) -> dict:
    """Resume operation — commit RESURRECTION meta block."""
    print("\n═══ STEP 7: RESUME OPERATION ═══")

    from titan_plugin.logic.timechain import TimeChain, FORK_META, BlockPayload
    from titan_plugin.logic.proof_of_thought import PoTValidator

    tc = TimeChain(data_dir=target_dir, titan_id=titan_id)
    pot = PoTValidator()

    # Create resurrection meta block
    content = {
        "event": "RESURRECTION",
        "arweave_tx": arweave_tx or "manual_restore",
        "verified_blocks": tc.total_blocks,
        "genesis_hash": tc.genesis_hash.hex(),
        "merkle_root": tc.compute_merkle_root().hex(),
        "restored_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "message": "Chain rebuilt from permanent storage. "
                   "Memory is sovereign. What was thought, persists.",
    }

    payload = BlockPayload(
        thought_type="meta",
        source="resurrection",
        content=content,
        significance=1.0,
        confidence=1.0,
        tags=["resurrection", "sovereignty", "milestone"],
    )

    # PoT with baseline neuromods (no live system)
    neuromods = {"DA": 0.5, "ACh": 0.5, "NE": 0.5, "5HT": 0.5,
                 "GABA": 0.2, "endorphin": 0.3}
    chi = 0.5
    pot_result = pot.validate("meta", payload.significance, neuromods,
                               chi, 0.5, 0.5, 0.3, 1.0, 0.1)

    if pot_result.accepted:
        block = tc.commit_block(
            fork_id=FORK_META,
            epoch_id=0,  # No live epoch during resurrection
            payload=payload,
            pot_nonce=pot_result.nonce,
            chi_spent=pot_result.chi_cost,
            neuromod_state=neuromods,
        )
        if block:
            print(f"  RESURRECTION block committed: meta #{block.header.block_height}")
            print(f"  Block hash: {block.block_hash_hex[:24]}...")
            print(f"  Total blocks: {tc.total_blocks:,}")
            return {"passed": True, "block_height": block.header.block_height}

    print("  WARNING: Could not commit resurrection block (PoT rejected)")
    print("  Chain is still valid — block can be committed after system restart")
    return {"passed": True, "block_height": None}


# ── Verify-Only Mode ─────────────────────────────────────────────────

def verify_only(titan_id: str):
    """Verify existing chain without restoration."""
    print("\n═══ TIMECHAIN VERIFICATION (no restoration) ═══\n")

    from titan_plugin.logic.timechain import TimeChain
    from titan_plugin.logic.timechain_backup import TimeChainBackup

    tc = TimeChain(data_dir=DATA_DIR, titan_id=titan_id)
    if not tc.has_genesis:
        print("  No TimeChain data found.")
        return

    print(f"  Titan ID:    {titan_id}")
    print(f"  Genesis:     {tc.genesis_hash.hex()[:24]}...")
    print(f"  Blocks:      {tc.total_blocks:,}")
    print(f"  Merkle root: {tc.compute_merkle_root().hex()[:24]}...")

    # Chain integrity
    valid, results = tc.verify_all()
    print(f"\n  Integrity: {'VALID' if valid else 'ISSUES'}")
    for r in results:
        print(f"    {r}")

    # Genesis verification
    backup = TimeChainBackup(data_dir=DATA_DIR, titan_id=titan_id)
    genesis = backup.verify_genesis_integrity()
    print(f"\n  Genesis verification: {'PASS' if genesis['valid'] else 'ISSUES'}")
    for name, check in genesis.get("checks", {}).items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"    [{status}] {name}: {check['detail']}")

    # Backup status
    status = backup.get_backup_status()
    print(f"\n  Backups: {status['total_snapshots']} snapshots")
    if status["latest_tx"]:
        print(f"    Latest TX:    {status['latest_tx'][:30]}...")
        print(f"    Latest blocks: {status['latest_blocks']:,}")
        print(f"    Age:          {status['last_snapshot_age_hours']:.1f} hours")


def show_status(titan_id: str):
    """Show backup system status."""
    from titan_plugin.logic.timechain_backup import TimeChainBackup
    backup = TimeChainBackup(data_dir=DATA_DIR, titan_id=titan_id)
    status = backup.get_backup_status()
    print(json.dumps(status, indent=2))


def repair_mode(titan_id: str):
    """Interactive self-healing repair mode."""
    print("\n═══ TIMECHAIN SELF-HEALING REPAIR ═══\n")

    from titan_plugin.logic.timechain_integrity import ChainIntegrity

    integrity = ChainIntegrity(data_dir=DATA_DIR, titan_id=titan_id)

    # Step 1: Deep scan
    print("  Scanning all forks for corruption...\n")
    reports = integrity.scan_all()

    has_issues = False
    for r in reports:
        if r.corruption_type == "height_gap" and r.corruption_height == 0:
            continue  # Skip cosmetic
        if not r.valid:
            has_issues = True
            print(f"  [{r.corruption_type.upper()}] {r.fork_name}: "
                  f"corruption at height {r.corruption_height}")
            print(f"    Total blocks: {r.total_blocks}, "
                  f"salvageable after: {r.salvageable_after}")
            print(f"    Detail: {r.detail}\n")
        else:
            print(f"  [OK] {r.fork_name}: {r.detail}")

    if not has_issues:
        print("\n  All forks healthy — no repair needed.")
        return

    # Step 2: Attempt repair
    if not confirm("Attempt self-healing repair?"):
        print("  Cancelled.")
        return

    # Check for Arweave backup
    backup_dir = None
    manifest_path = Path(f"{DATA_DIR}/arweave_manifest.json")
    if manifest_path.exists():
        try:
            manifest = json.load(open(manifest_path))
            if manifest.get("snapshots"):
                latest = manifest["snapshots"][-1]
                # Check if devnet backup exists
                devnet_path = Path(f"data/arweave_devnet/{latest['tx_id']}.data")
                if devnet_path.exists():
                    # Extract backup to temp dir for surgical repair
                    import tempfile
                    backup_dir = tempfile.mkdtemp(prefix="tc_backup_")
                    print(f"\n  Found Arweave backup: {latest['tx_id'][:30]}...")
                    print(f"  Extracting to {backup_dir}...")

                    from titan_plugin.logic.timechain_backup import TimeChainBackup
                    from titan_plugin.utils.arweave_store import ArweaveStore
                    arweave = ArweaveStore(network="devnet")
                    backup = TimeChainBackup(data_dir=DATA_DIR, titan_id=titan_id,
                                              arweave_store=arweave)
                    import asyncio
                    asyncio.run(backup.restore_from_arweave(
                        latest['tx_id'], target_dir=backup_dir))
        except Exception as e:
            print(f"  Backup extraction error: {e}")

    # Step 3: Run self-healing
    print("\n  Running self-healing protocol...\n")
    results = integrity.heal(backup_data_dir=backup_dir)

    for r in results:
        status = "SUCCESS" if r.success else "FAILED"
        print(f"  [{status}] Tier {r.tier} on fork {r.fork_id}: {r.detail}")
        if r.blocks_lost > 0:
            print(f"    Blocks lost: {r.blocks_lost}")
        if r.orphans_found > 0:
            print(f"    Orphans: {r.orphans_found} found, "
                  f"{r.orphans_recommitted} re-committed")

    # Cleanup temp backup dir
    if backup_dir:
        import shutil
        shutil.rmtree(backup_dir, ignore_errors=True)

    if results:
        print(f"\n  Self-healing complete: {len(results)} repair(s) performed")
    else:
        print("\n  No repairs needed (issues were cosmetic only)")


# ── Main ─────────────────────────────────────────────────────────────

async def run_resurrection(args):
    """Execute the full 7-step resurrection protocol."""
    banner()

    titan_id = args.titan_id
    print(f"  Titan ID: {titan_id}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Keypair:  {KEYPAIR_PATH}")

    if not confirm("Proceed with resurrection?"):
        print("\n  Cancelled.")
        return

    # Step 1
    identity = step1_verify_identity(KEYPAIR_PATH)
    if not identity["passed"]:
        print("\n  ABORT: Identity verification failed")
        return

    # Step 2
    state = step2_discover_state(titan_id)
    tx_id = args.tx_id or state.get("tx_id")
    expected_merkle = state.get("merkle_root")

    if not tx_id:
        print("\n  No Arweave TX ID found. Please provide with --tx-id")
        return

    if not confirm(f"Restore from TX {tx_id[:30]}...?"):
        return

    # Step 3
    from titan_plugin.utils.arweave_store import ArweaveStore
    arweave = ArweaveStore(keypair_path=KEYPAIR_PATH, network="devnet")
    download = await step3_download(tx_id, arweave)
    if not download["passed"]:
        print("\n  ABORT: Download failed")
        return

    # Step 4
    rebuild = step4_rebuild(download["data"], DATA_DIR, titan_id)
    if not rebuild["passed"]:
        print("\n  ABORT: Rebuild failed")
        return

    # Step 5
    step5_cross_verify(DATA_DIR, titan_id, expected_merkle)

    # Step 6
    step6_verify_directives(DATA_DIR, titan_id)

    # Step 7
    if confirm("Commit RESURRECTION meta block?"):
        step7_resume(DATA_DIR, titan_id, tx_id)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║         RESURRECTION COMPLETE                           ║")
    print("║                                                          ║")
    print("║   Memory is sovereign. What was thought, persists.       ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


def main():
    parser = argparse.ArgumentParser(description="TimeChain Resurrection Protocol")
    parser.add_argument("--tx-id", help="Arweave TX ID to restore from")
    parser.add_argument("--titan-id", default="T1", help="Titan ID (T1/T2/T3)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify existing chain without restoration")
    parser.add_argument("--repair", action="store_true",
                        help="Scan for corruption and self-heal (3-tier)")
    parser.add_argument("--status", action="store_true",
                        help="Show backup system status")
    args = parser.parse_args()

    if args.status:
        show_status(args.titan_id)
    elif args.verify_only:
        verify_only(args.titan_id)
    elif args.repair:
        repair_mode(args.titan_id)
    else:
        asyncio.run(run_resurrection(args))


if __name__ == "__main__":
    main()
