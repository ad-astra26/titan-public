#!/usr/bin/env python3
"""5J-5 — backfill T1's historical backup events as v=3 on-chain memos.

Before the v=3 sovereign chain (5J-2) went live, T1 recorded each Arweave backup
only in `data/backup_records/*.json` (+ the v=2 ZK-Vault commit). This one-shot
seeds the **on-chain v=3 chain back to the Day-1 anchor** so a wallet-alone
resurrection (`scripts/backup_restore_sovereign.py`) can walk the full history.

Two record schemas live in `data/backup_records/`:
  • legacy per-tarball  — {arweave_tx, archive_hash, backup_type, uploaded_at}
  • unified-v2 event    — {event_id, personality_tx, timechain_tx, event_merkle_root,
                           uploaded_at}  (no per-component sha256)

A faithful v=3 memo needs the per-component tarball sha256 (`arc`). Legacy records
carry it; unified-v2 records do NOT — those events are FLAGGED and SKIPPED rather
than fabricated (re-fetch + sha256 from Arweave is a separate, explicit step).

Posting reuses the production emit `RebirthBackup.commit_event_v3_chain` (same memo
format, same TX shape, prev= event-chaining). **Idempotent**: existing on-chain
v=3 archive_hashes (walked via the D3a seam) are skipped, so re-runs post 0.

    python scripts/backup_chain_backfill_v3.py --titan T1 --dry-run   # no writes
    python scripts/backup_chain_backfill_v3.py --titan T1 --commit    # MAINNET write (gated)

GENESIS ANCHOR (Day 1, 2026-04-06): Arweave tx
`5StBnZIfus1mbuYJ520Ct2a4OomNUuOm_3VGZmeNGQw` — the eternal chain origin; the
earliest backfilled event carries `prev=genesis`.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

GENESIS_ANCHOR_TX = "5StBnZIfus1mbuYJ520Ct2a4OomNUuOm_3VGZmeNGQw"

# legacy backup_type → v=3 tier
_TIER_BY_TYPE = {"personality": "PT", "timechain": "TC", "soul": "SL", "weekly": "SL"}
# ~per-component memo cost (SOL); 1 head TX co-bundles commit_state (cheap).
_SOL_PER_MEMO = 0.000005


# ── pure logic (offline-testable) ─────────────────────────────────────────

def load_records(records_dir: str) -> list[dict]:
    """Load every backup_records/*.json (skips unreadable files)."""
    out = []
    for f in sorted(glob.glob(os.path.join(records_dir, "*.json"))):
        try:
            with open(f) as fh:
                d = json.load(fh)
            d["_file"] = os.path.basename(f)
            out.append(d)
        except Exception:
            continue
    return out


def normalize_record(rec: dict) -> dict:
    """Normalize one record → an event dict, or a skip with `_skip_reason`.

    Returns {ts, event_id, event_type, mrkl, components:[{tier,tx_id,arc}], _file}
    on success, or {_skip_reason, _file} when a faithful memo can't be built.
    """
    f = rec.get("_file", "?")
    ts = rec.get("uploaded_at")
    if not ts:
        return {"_skip_reason": "no uploaded_at", "_file": f}

    # legacy per-tarball record (has archive_hash + arweave_tx)
    if rec.get("archive_hash") and rec.get("arweave_tx"):
        tier = _TIER_BY_TYPE.get((rec.get("backup_type") or "personality").lower(), "PT")
        return {
            "ts": int(ts), "event_id": rec.get("event_id") or f"legacy_{ts}_{tier}",
            "event_type": "incremental", "mrkl": rec["archive_hash"],
            "components": [{"tier": tier, "tx_id": rec["arweave_tx"], "arc": rec["archive_hash"]}],
            "_file": f,
        }

    # unified-v2 event record (personality_tx [+ timechain_tx] + event_merkle_root)
    if rec.get("personality_tx") and rec.get("event_merkle_root"):
        # per-component sha256 ("arc") is NOT stored in these records → cannot build
        # a verifiable memo without re-fetching the tarballs from Arweave.
        return {"_skip_reason": "unified-v2 event lacks per-component sha256 "
                                "(re-fetch tarball + sha256, or skip)", "_file": f}

    return {"_skip_reason": "unrecognized record schema", "_file": f}


def build_chain_plan(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (postable_events_sorted_oldest_first, skipped). Oldest event is the
    genesis-anchored head (prev=genesis); later events chain by prev=head_sig."""
    events, skipped = [], []
    for rec in records:
        n = normalize_record(rec)
        (skipped if n.get("_skip_reason") else events).append(n)
    events.sort(key=lambda e: e["ts"])
    return events, skipped


def existing_arcs(decoded_memos: list[str]) -> set:
    """The set of v=3 `arc` fragments already on-chain (for idempotency)."""
    from titan_hcl.logic.backup_memo_v3 import parse_v3_memo
    arcs = set()
    for memo in decoded_memos:
        p = parse_v3_memo(memo)
        if p and p.get("arc"):
            arcs.add(p["arc"])
    return arcs


# ── I/O ────────────────────────────────────────────────────────────────────

async def _walk_existing(titan_pubkey: str, rpc_url: Optional[str]) -> set:
    """Walk the wallet's existing v=3 memos → set of arc fragments (D3a seam)."""
    from titan_hcl.utils import solana_client as sc
    sigs = await sc.get_signatures_for_address(titan_pubkey, rpc_url=rpc_url)
    memos = []
    for sig in sigs:
        try:
            m = await sc.get_memo_for_tx(sig, rpc_url=rpc_url)
        except Exception:
            continue
        if m:
            memos.append(m)
    return existing_arcs(memos)


def _print_plan(events: list[dict], skipped: list[dict], already: set) -> int:
    print("\n🜂 v=3 backfill plan — oldest → newest (prev=genesis at the head)\n")
    postable = 0
    prev_label = "genesis"
    for e in events:
        comps = e["components"]
        arcs = [c["arc"][:8] for c in comps]
        dup = all(c["arc"][:32] in already for c in comps)
        flag = "  [already on-chain — skip]" if dup else ""
        if not dup:
            postable += len(comps)
        print(f"  {e['ts']}  {e['event_id'][:18]:18}  "
              f"tiers={[c['tier'] for c in comps]}  arc={arcs}  prev={prev_label}{flag}")
        prev_label = e["event_id"][:12]
    if skipped:
        print(f"\n  ⚠ {len(skipped)} record(s) SKIPPED (incomplete for a faithful memo):")
        for s in skipped:
            print(f"      {s['_file']}: {s['_skip_reason']}")
    cost = postable * _SOL_PER_MEMO
    print(f"\n  Genesis anchor: {GENESIS_ANCHOR_TX}")
    print(f"  Postable component-memos: {postable}  (~{cost:.6f} SOL ≈ ${cost*170:.4f} @ SOL=$170)")
    print(f"  Skipped: {len(skipped)}   Already on-chain: "
          f"{sum(1 for e in events if all(c['arc'][:32] in already for c in e['components']))}")
    return postable


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="backup_chain_backfill_v3",
                                description="Backfill historical backups as v=3 on-chain memos (5J-5).")
    p.add_argument("--titan", default="T1", help="Titan id (default T1).")
    p.add_argument("--records-dir", default=None,
                   help="backup_records dir (default: <repo>/data/backup_records).")
    p.add_argument("--rpc-url", default=None, help="Solana RPC override.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Print the plan; NO writes.")
    g.add_argument("--commit", action="store_true", help="Post the memos (MAINNET write).")
    args = p.parse_args(argv)

    records_dir = args.records_dir or os.path.join(_REPO_ROOT, "data", "backup_records")
    if not os.path.isdir(records_dir):
        print(f"backup_records dir not found: {records_dir}")
        return 1
    records = load_records(records_dir)
    events, skipped = build_chain_plan(records)
    print(f"Loaded {len(records)} record(s) → {len(events)} postable event(s), "
          f"{len(skipped)} skipped.")

    # Idempotency: which arcs are already on the chain? (best-effort; needs network)
    already: set = set()
    if args.commit or not args.dry_run:
        try:
            from titan_hcl.config_loader import load_titan_config
            cfg = load_titan_config()
            pub = (cfg.get("network", {}) or {}).get("maker_pubkey") or \
                  os.environ.get("TITAN_PUBKEY", "")
            if pub:
                already = asyncio.run(_walk_existing(pub, args.rpc_url))
        except Exception as e:
            print(f"  (idempotency walk skipped: {e})")

    postable = _print_plan(events, skipped, already)

    if args.dry_run:
        print("\n--dry-run: nothing written. Re-run with --commit to post (MAINNET).")
        return 0

    if postable == 0:
        print("\nNothing to post (all events already on-chain or skipped).")
        return 0

    # --commit: replay each event through the production v=3 emit, chaining prev=.
    print("\n--commit: posting to MAINNET…")
    return asyncio.run(_commit(args, events, already))


async def _commit(args, events: list[dict], already: set) -> int:
    from titan_hcl.logic.backup import RebirthBackup
    from titan_hcl.core.network import HybridNetworkClient
    from titan_hcl.config_loader import load_titan_config
    cfg = load_titan_config()
    network = HybridNetworkClient(cfg)
    rb = RebirthBackup(network_client=network, titan_id=args.titan, full_config=cfg)
    prev_sig: Optional[str] = None
    posted = 0
    for e in events:
        if all(c["arc"][:32] in already for c in e["components"]):
            continue  # idempotent skip
        res = await rb.commit_event_v3_chain(
            event_id=e["event_id"], ts=e["ts"], event_type=e["event_type"],
            event_merkle_root=e["mrkl"], components=e["components"], prev_sig=prev_sig)
        if not res or not res.get("head_sig"):
            print(f"  ✗ event {e['event_id'][:12]} failed — halting (no partial chain).")
            return 1
        prev_sig = res["head_sig"]
        posted += len(e["components"])
        print(f"  ✓ {e['event_id'][:12]} → head {prev_sig[:12]} ({len(e['components'])} memo)")
        await asyncio.sleep(0.5)  # rate-limit the burst
    print(f"\nPosted {posted} component-memo(s). Verify with the D3a chain walk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
