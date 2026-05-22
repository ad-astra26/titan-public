#!/usr/bin/env python3
"""Timechain-v2 usage report (Synthesis Engine Phase 0 / 0A step 1).

READ-ONLY instrumentation over data/timechain/{index.db, chain_*.bin}. Produces the
baseline that grounds the 0A "episodic new-block rate ↓ ≥10×" gate and reveals which
TX classes (thought_type / source) dominate each fork — refining where tiered
Merkle-batching pays off.

No writes. Opens index.db with SQLite URI mode=ro so it is safe to run against a
live Titan. Bytes-per-block are derived from consecutive file_offset deltas within a
fork (the last block of each fork is sized via the on-disk chain_*.bin length).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

DAY = 86400.0


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro&immutable=0"
    return sqlite3.connect(uri, uri=True, timeout=10.0)


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f}{unit}"
        n /= 1024.0
    return f"{n:,.1f}TB"


def _fork_file(data_dir: Path, fork_name: str) -> Path | None:
    # Primary forks map to chain_<name>.bin; topic sidechains live under sidechains/.
    cand = data_dir / f"chain_{fork_name}.bin"
    if cand.exists():
        return cand
    sc = data_dir / "sidechains" / f"chain_{fork_name}.bin"
    return sc if sc.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        default="/home/antigravity/projects/titan/data/timechain",
        help="Live timechain data dir (default: T1 local).",
    )
    ap.add_argument("--top", type=int, default=12, help="Top-N TX classes per fork.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    index_db = data_dir / "index.db"
    if not index_db.exists():
        print(f"ERROR: {index_db} not found", file=sys.stderr)
        return 1

    now = time.time()
    con = _connect_ro(index_db)

    forks = {
        fid: (name, ftype)
        for fid, name, ftype in con.execute(
            "SELECT fork_id, fork_name, fork_type FROM fork_registry"
        )
    }

    # Pull every block ordered by (fork, height) so file_offset deltas give per-block size.
    rows = con.execute(
        "SELECT fork_id, block_height, timestamp, thought_type, source, tags, "
        "chi_spent, file_offset FROM block_index ORDER BY fork_id, block_height"
    ).fetchall()
    con.close()

    by_fork: dict[int, list] = defaultdict(list)
    for r in rows:
        by_fork[r[0]].append(r)

    print("=" * 92)
    print(f"TIMECHAIN-V2 USAGE REPORT  ·  {data_dir}  ·  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 92)

    grand_blocks = 0
    grand_bytes = 0
    fork_summaries = []

    for fid in sorted(by_fork, key=lambda f: -len(by_fork[f])):
        name, ftype = forks.get(fid, (f"fork{fid}", "?"))
        blks = by_fork[fid]
        n = len(blks)
        grand_blocks += n

        fpath = _fork_file(data_dir, name)
        fsize = fpath.stat().st_size if fpath else 0
        grand_bytes += fsize

        # Per-block byte size from offset deltas; last block sized to EOF.
        offs = [b[7] for b in blks]
        sizes = []
        for i in range(n):
            if i + 1 < n:
                sizes.append(max(0, offs[i + 1] - offs[i]))
            else:
                sizes.append(max(0, fsize - offs[i]) if fsize else 0)

        ts = [b[2] for b in blks]
        t_first, t_last = (min(ts), max(ts)) if ts else (now, now)
        span_days = max((t_last - t_first) / DAY, 1e-9)
        b_24h = sum(1 for t in ts if now - t <= DAY)
        b_7d = sum(1 for t in ts if now - t <= 7 * DAY)
        rate_overall = n / span_days

        # TX-class histogram: (thought_type, source) — counts + bytes.
        cls_count: dict = defaultdict(int)
        cls_bytes: dict = defaultdict(int)
        cls_24h: dict = defaultdict(int)
        for b, sz in zip(blks, sizes):
            key = (b[3] or "∅", b[4] or "∅")
            cls_count[key] += 1
            cls_bytes[key] += sz
            if now - b[2] <= DAY:
                cls_24h[key] += 1

        fork_summaries.append((name, ftype, n, fsize, b_24h, rate_overall))

        print(f"\n── fork {fid}: {name}  [{ftype}]")
        print(
            f"   blocks={n:,}  file={_human(fsize)}  "
            f"avg/block={_human(fsize / n) if n else '0'}  "
            f"span={span_days:.1f}d"
        )
        print(
            f"   block-rate: overall={rate_overall:,.1f}/d  "
            f"last_24h={b_24h:,}  last_7d={b_7d:,}"
        )
        top = sorted(cls_count.items(), key=lambda kv: -cls_bytes[kv[0]])[: args.top]
        print(f"   top TX-classes by bytes (thought_type / source):")
        for (tt, src), cnt in top:
            print(
                f"     {tt:<28.28} {src:<22.22} "
                f"n={cnt:>8,}  bytes={_human(cls_bytes[(tt, src)]):>10}  "
                f"24h={cls_24h[(tt, src)]:>6,}  "
                f"avg={_human(cls_bytes[(tt, src)] / cnt) if cnt else '0':>9}"
            )

    print("\n" + "=" * 92)
    print(
        f"TOTALS: blocks={grand_blocks:,}  on-disk={_human(grand_bytes)}  "
        f"forks={len(by_fork)}"
    )
    print("Fork ranking by 24h block volume (0A tiering targets the top):")
    for name, ftype, n, fsize, b24, rate in sorted(
        fork_summaries, key=lambda s: -s[4]
    ):
        print(
            f"   {name:<16.16} 24h={b24:>7,}  overall={rate:>9,.1f}/d  "
            f"total={n:>9,}  disk={_human(fsize):>10}"
        )
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
