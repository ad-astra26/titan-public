#!/usr/bin/env python3
"""
restore_pi_heartbeat.py — rebuild pi_heartbeat_state.json from consciousness.db.

The pi_heartbeat feature was added after Titans were already running
(2026-04-06 on T1, 2026-04-08 on T2 + T3). Titans had weeks of consciousness
epoch data in consciousness.db BEFORE pi_heartbeat started tracking, so the
current cluster_count undercounts the full developmental_age.

This script replays all historical epochs (epoch_id + curvature) through
PiHeartbeatMonitor to reconstruct the accurate state file.

USAGE:
    # Always dry-run first — reports projected stats, writes nothing
    python scripts/restore_pi_heartbeat.py --dry-run

    # After reviewing dry-run, apply — backs up existing state file then writes
    python scripts/restore_pi_heartbeat.py --apply

    # Remote paths (T2 / T3 over ssh-mounted FS or run via ssh root@10.135.0.6)
    python scripts/restore_pi_heartbeat.py --apply \\
        --db /home/antigravity/projects/titan/data/consciousness.db \\
        --state /home/antigravity/projects/titan/data/pi_heartbeat_state.json

SAFETY:
  - --dry-run is the default (no args → dry-run behavior)
  - --apply ALWAYS backs up the target state file first (*.bak_YYYYMMDD_HHMMSS)
  - Exit non-zero on any error before a write
  - Requires target Titan to be STOPPED to avoid overwrite race
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="./data/consciousness.db",
                        help="Path to consciousness.db (default: ./data/consciousness.db)")
    parser.add_argument("--state", default="./data/pi_heartbeat_state.json",
                        help="Path to pi_heartbeat_state.json (default: ./data/pi_heartbeat_state.json)")
    parser.add_argument("--apply", action="store_true",
                        help="Write restored state (default: dry-run only)")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-gap-size", type=int, default=2)
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    state_path = os.path.abspath(args.state)

    print(f"Restore pi_heartbeat state")
    print(f"  DB:     {db_path}")
    print(f"  State:  {state_path}")
    print(f"  Mode:   {'APPLY' if args.apply else 'DRY-RUN'}")
    print()

    if not os.path.exists(db_path):
        print(f"ERROR: consciousness.db not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    # Import the monitor lazily (requires titan venv on PYTHONPATH)
    # Use a nonexistent init path so the monitor starts CLEAN (no load).
    try:
        sys.path.insert(0, os.path.abspath("."))
        from titan_plugin.logic.pi_heartbeat import PiHeartbeatMonitor
    except ImportError as e:
        print(f"ERROR: cannot import PiHeartbeatMonitor ({e})", file=sys.stderr)
        print("Run from the project root with test_env activated.", file=sys.stderr)
        sys.exit(1)

    init_path = f"/tmp/pi_heartbeat_init_{os.getpid()}.nonexistent"
    monitor = PiHeartbeatMonitor(
        min_cluster_size=args.min_cluster_size,
        min_gap_size=args.min_gap_size,
        state_path=init_path,
    )
    # Rebind to the real target so _save_state() writes there in --apply mode
    monitor._state_path = state_path

    # Stream rows in ascending epoch_id order (the canonical event sequence)
    t0 = time.time()
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        total_rows = conn.execute("SELECT COUNT(*) FROM epochs").fetchone()[0]
        print(f"Source DB: {total_rows:,} epoch records to replay")
        first_ts = conn.execute("SELECT timestamp FROM epochs ORDER BY epoch_id ASC LIMIT 1").fetchone()
        last_ts = conn.execute("SELECT timestamp FROM epochs ORDER BY epoch_id DESC LIMIT 1").fetchone()
        if first_ts and last_ts:
            span_days = (last_ts[0] - first_ts[0]) / 86400.0
            first_dt = datetime.fromtimestamp(first_ts[0]).isoformat(sep=' ', timespec='seconds')
            last_dt = datetime.fromtimestamp(last_ts[0]).isoformat(sep=' ', timespec='seconds')
            print(f"  Span: {first_dt}  →  {last_dt}  ({span_days:.1f} days)")
        print()

        cur = conn.execute("SELECT epoch_id, curvature FROM epochs ORDER BY epoch_id ASC")
        replayed = 0
        skipped_null = 0
        for epoch_id, curvature in cur:
            if curvature is None:
                skipped_null += 1
                continue
            monitor.observe(float(curvature), int(epoch_id))
            replayed += 1
            if replayed % 100000 == 0:
                print(f"  Replayed {replayed:,}/{total_rows:,} epochs "
                      f"(cluster_count={monitor._cluster_count})...")
    finally:
        conn.close()
    elapsed = time.time() - t0

    print()
    print(f"=== Replay complete ({elapsed:.1f}s) ===")
    print(f"  Epochs replayed:             {replayed:,}")
    print(f"  Epochs skipped (null curv):  {skipped_null:,}")
    print(f"  Total π-epochs:              {monitor._total_pi_epochs:,}")
    print(f"  Total zero-epochs:           {monitor._total_zero_epochs:,}")
    print(f"  Cluster count:               {monitor._cluster_count:,}")
    print(f"  Heartbeat ratio:             {monitor.heartbeat_ratio:.4f}")
    print(f"  Last cluster end epoch:      {monitor._last_cluster_end_epoch}")
    print(f"  Current in_cluster:          {monitor._in_cluster}")
    print()

    if args.apply:
        # Backup existing state file if present
        if os.path.exists(state_path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{state_path}.bak_{ts}"
            shutil.copy2(state_path, backup_path)
            print(f"  ✓ Existing state backed up to: {backup_path}")
        # Write new state
        monitor._save_state()
        # Verify it was written
        if os.path.exists(state_path):
            size = os.path.getsize(state_path)
            print(f"  ✓ Wrote restored state ({size} bytes) to {state_path}")
        else:
            print(f"  ✗ State file not written!", file=sys.stderr)
            sys.exit(2)
    else:
        print("  (dry-run — no file written. Pass --apply to write.)")


if __name__ == "__main__":
    main()
