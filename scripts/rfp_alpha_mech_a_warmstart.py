#!/usr/bin/env python3
"""
rFP α Mechanism A warm-start — seed sequence_quality.json from chain_archive.

Purpose
-------
Pre-populate SequenceQualityStore with EMA entries derived from real
post-TUNING-015 reasoning chains so Phase 2 Mechanism A has useful signal
from minute-one of activation (instead of cold-starting at 0 signal until
the visit_gate=3 threshold is reached via live chains).

Safety design
-------------
1. Default output path is STAGED (`sequence_quality.json.warmstart`).
   User promotes it to `sequence_quality.json` only after reviewing.
2. Detects running T1 titan_main process — warns if found (don't write
   to main file while engine has it mapped in memory; live engine's
   periodic save_all would clobber our seed).
3. `--dry-run` prints full preview without writing anything.
4. `--deploy` promotes staged → main file; requires T1 to be stopped
   (hard fail if pgrep finds titan_main running).
5. Filters: chain_length >= 4 (excludes pre-TUNING-015 len-3 monoculture);
   created_at cutoff configurable (default: TUNING-015 commit date);
   outcome_score >= 0 (excludes null / invalid rows).

Workflow
--------
Offline seed + deploy:
    $ bash scripts/safe_restart.sh t1 --stop        # (or equivalent stop)
    $ python scripts/rfp_alpha_mech_a_warmstart.py --dry-run
    $ python scripts/rfp_alpha_mech_a_warmstart.py  # writes staged file
    $ python scripts/rfp_alpha_mech_a_warmstart.py --deploy
    $ # restart T1 normally — engine loads the pre-seeded table

Preview-only (safe while T1 is running):
    $ python scripts/rfp_alpha_mech_a_warmstart.py --dry-run

Contract
--------
Output file format mirrors SequenceQualityStore.save(): a JSON object
with `entries: [{seq, ema, n, last_ts}]` plus metadata. Visits get
count-weighted ramped: when multiple archive rows match the same prefix,
the EMA converges to the mean of their outcome_scores.

Arguments
---------
--db-path        SQLite path containing chain_archive (default: data/inner_memory.db)
--out-dir        Output directory (default: data/reasoning)
--min-length     Minimum chain_length to include (default: 4)
--since          ISO timestamp lower bound (default: 2026-04-16 00:00 UTC — TUNING-015 ship)
--max-rows       Limit row count for sanity (default: 10000)
--dry-run        Print preview only, no file writes
--deploy         Promote staged file to main (requires T1 stopped)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

# Make project imports work when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from titan_plugin.logic.reasoning import SequenceQualityStore  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────

# TUNING-015 shipped on 2026-04-16 — use 00:00 UTC as conservative floor.
# Any chain archived before this is pre-015 (100% at length 3, monoculture).
DEFAULT_SINCE = "2026-04-16 00:00:00"

DEFAULT_DB_PATH = "data/inner_memory.db"
DEFAULT_OUT_DIR = "data/reasoning"
STAGED_FILENAME = "sequence_quality.json.warmstart"
MAIN_FILENAME = "sequence_quality.json"


# ── Helpers ───────────────────────────────────────────────────────


def _t1_running() -> bool:
    """Return True if titan_main is running locally."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", "titan_main"],
            capture_output=True, text=True, timeout=3)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def _parse_since(s: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM:SS' (UTC) → unix timestamp."""
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _fetch_rows(db_path: str, since_ts: float, min_length: int,
                max_rows: int, source: str = "main") -> list[dict]:
    """Query chain_archive for qualifying rows.

    CRITICAL: filters source='main' by default. chain_archive stores BOTH
    reasoning chains (source='main') AND meta-reasoning chains (source='meta').
    Their primitive sets are DISJOINT (reasoning: COMPARE/IF_THEN/SEQUENCE/...,
    meta: FORMULATE/RECALL/HYPOTHESIZE/...). Mixing them into Mechanism A's
    table would pollute the reasoning-engine EMA store with entries the
    engine can never query, forcing LRU eviction of useful reasoning prefixes.
    """
    import sqlite3
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        cur = conn.execute(
            """
            SELECT chain_sequence, chain_length, outcome_score, confidence,
                   gut_agreement, created_at
            FROM chain_archive
            WHERE created_at >= ?
              AND chain_length >= ?
              AND source = ?
              AND outcome_score IS NOT NULL
              AND outcome_score >= 0
              AND chain_sequence IS NOT NULL
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (since_ts, min_length, source, max_rows),
        )
        out = []
        for row in cur:
            seq_raw = row[0]
            if isinstance(seq_raw, (bytes, bytearray)):
                try:
                    seq_raw = seq_raw.decode("utf-8")
                except Exception:
                    continue
            try:
                seq = json.loads(seq_raw) if isinstance(seq_raw, str) else seq_raw
            except Exception:
                continue
            if not isinstance(seq, list) or not seq:
                continue
            out.append({
                "chain_sequence": [str(p) for p in seq],
                "chain_length": int(row[1]),
                "outcome_score": float(row[2]),
                "confidence": float(row[3] or 0.0),
                "gut_agreement": float(row[4] or 0.0),
                "created_at": float(row[5]),
            })
        return out
    finally:
        conn.close()


def _summarize_rows(rows: list[dict]) -> dict:
    """Compute stats for dry-run report."""
    if not rows:
        return {"count": 0}
    lens = [r["chain_length"] for r in rows]
    outs = [r["outcome_score"] for r in rows]
    length_hist: dict[int, int] = {}
    for L in lens:
        length_hist[L] = length_hist.get(L, 0) + 1
    return {
        "count": len(rows),
        "length_min": min(lens),
        "length_max": max(lens),
        "length_mean": round(sum(lens) / len(lens), 2),
        "length_hist": dict(sorted(length_hist.items())),
        "outcome_min": round(min(outs), 4),
        "outcome_max": round(max(outs), 4),
        "outcome_mean": round(sum(outs) / len(outs), 4),
        "earliest": datetime.fromtimestamp(min(r["created_at"] for r in rows), tz=timezone.utc)
                     .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "latest": datetime.fromtimestamp(max(r["created_at"] for r in rows), tz=timezone.utc)
                   .strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def _report_store_state(store: SequenceQualityStore, top_n: int = 12) -> None:
    """Print top/bottom entries for sanity audit."""
    s = store.stats()
    print(f"  table size: {s['size']}")
    print(f"  gated entries (n>=visit_gate): {s['gated_size']}")
    print(f"  evictions: {s['evictions']}")
    # Rank entries by ema × n (most-visited × highest-scoring)
    ranked = sorted(
        store._table.items(),
        key=lambda kv: kv[1]["ema"] * min(10, kv[1]["n"]),
        reverse=True,
    )
    print(f"\n  TOP {top_n} prefix EMAs (rank = ema × min(n,10)):")
    for prefix, entry in ranked[:top_n]:
        seq_str = "→".join(prefix)
        print(f"    ema={entry['ema']:.4f}  n={entry['n']:3d}  seq={seq_str}")
    print(f"\n  BOTTOM {top_n // 2} (lowest ema, filtered n≥3):")
    bot = sorted(
        [(k, v) for k, v in store._table.items() if v["n"] >= 3],
        key=lambda kv: kv[1]["ema"],
    )
    for prefix, entry in bot[:top_n // 2]:
        seq_str = "→".join(prefix)
        print(f"    ema={entry['ema']:.4f}  n={entry['n']:3d}  seq={seq_str}")


# ── Main ──────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="rFP α Mechanism A warm-start from chain_archive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Contract\n")[0],
    )
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--min-length", type=int, default=4,
                   help="exclude pre-TUNING-015 len-3 chains (default 4)")
    p.add_argument("--since", default=DEFAULT_SINCE,
                   help=f"UTC timestamp floor (default: {DEFAULT_SINCE})")
    p.add_argument("--max-rows", type=int, default=10000)
    p.add_argument("--source", default="main",
                   help="chain_archive.source filter (default: main — reasoning chains only;"
                        " 'meta' = meta-reasoning (wrong store); any string passes through)")
    p.add_argument("--dry-run", action="store_true",
                   help="preview only — no file writes")
    p.add_argument("--deploy", action="store_true",
                   help="promote staged file to main (requires T1 stopped)")
    p.add_argument("--top-n", type=int, default=12)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    staged_path = os.path.join(args.out_dir, STAGED_FILENAME)
    main_path = os.path.join(args.out_dir, MAIN_FILENAME)

    # ── Deploy mode: promote staged → main ───────────────────────
    if args.deploy:
        if _t1_running():
            print("ERROR: T1 titan_main is running. Stop T1 before --deploy —",
                  "otherwise the live engine's periodic save_all would clobber",
                  "the warm-start seed.", file=sys.stderr)
            return 2
        if not os.path.exists(staged_path):
            print(f"ERROR: no staged file at {staged_path}. Run without --deploy",
                  "first to generate the seed.", file=sys.stderr)
            return 2
        # Backup existing main file if present
        if os.path.exists(main_path):
            backup = main_path + f".pre_warmstart.{int(time.time())}"
            shutil.copy2(main_path, backup)
            print(f"  ✓ backed up existing main file → {backup}")
        shutil.move(staged_path, main_path)
        print(f"  ✓ promoted {staged_path} → {main_path}")
        print(f"  → next T1 start will load the warm-started table via SequenceQualityStore.load()")
        return 0

    # ── Preview / write mode ─────────────────────────────────────
    if _t1_running():
        if args.dry_run:
            print("  ⓘ T1 is running — dry-run mode is safe (no writes)")
        else:
            print("  ⚠ T1 is running. Staged file will be written safely (not main).",
                  "T1 must be stopped before --deploy promotes staged → main.")

    since_ts = _parse_since(args.since)
    print("=" * 72)
    print("rFP α Mechanism A — warm-start from chain_archive")
    print("=" * 72)
    print(f"  db_path:    {args.db_path}")
    print(f"  out_dir:    {args.out_dir}")
    print(f"  min_length: {args.min_length}")
    print(f"  source:     {args.source}  (must be 'main' for reasoning — 'meta' would poison table)")
    print(f"  since:      {args.since} UTC (ts={since_ts:.0f})")
    print(f"  max_rows:   {args.max_rows}")
    print(f"  dry_run:    {args.dry_run}")
    print()

    if not os.path.exists(args.db_path):
        print(f"ERROR: DB not found at {args.db_path}", file=sys.stderr)
        return 2

    # Fetch
    t0 = time.time()
    rows = _fetch_rows(args.db_path, since_ts, args.min_length, args.max_rows,
                      source=args.source)
    dt_fetch = time.time() - t0
    print(f"  fetched {len(rows)} rows from chain_archive "
          f"(source='{args.source}') in {dt_fetch:.2f}s")

    if not rows:
        print("  ⚠ no qualifying rows — filters too strict, or no post-TUNING-015 chains yet")
        return 1

    stats_in = _summarize_rows(rows)
    print(f"  source stats: length_hist={stats_in['length_hist']}")
    print(f"                outcome_mean={stats_in['outcome_mean']}  "
          f"min={stats_in['outcome_min']}  max={stats_in['outcome_max']}")
    print(f"                earliest={stats_in['earliest']}")
    print(f"                latest={stats_in['latest']}")
    print()

    # Seed
    store = SequenceQualityStore(cap=10000, visit_gate=3, ema_alpha=0.1,
                                 ramp_cutoff=20)
    t0 = time.time()
    seeded = store.seed_from_archive(rows)
    dt_seed = time.time() - t0
    print(f"  seeded {seeded} prefix updates across {len(store._table)} unique "
          f"sequences in {dt_seed:.2f}s")
    _report_store_state(store, top_n=args.top_n)
    print()

    # Write
    if args.dry_run:
        print("  (dry-run — no file written)")
        return 0

    store.save(staged_path)
    size = os.path.getsize(staged_path)
    print(f"  ✓ wrote staged seed → {staged_path}  ({size / 1024:.1f} KB)")
    print()
    print("  Next steps:")
    print(f"    1. Review staged file: less {staged_path}")
    print(f"    2. Stop T1: bash scripts/safe_restart.sh t1 --stop  (or kill)")
    print(f"    3. Deploy: python {__file__} --deploy")
    print(f"    4. Start T1: bash scripts/safe_restart.sh t1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
