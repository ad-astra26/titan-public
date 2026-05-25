#!/usr/bin/env python3
"""dim_breath_audit.py — Trinity P0.6-A variance audit (PLAN §6.6.1).

Per-dim rolling-24h variance health card for the **40 BMS dims** (inner+outer
× body 5D + mind 15D). Reads `consciousness.db` epochs (132D state_vector
column) and computes per-dim:

  - mean, std, min, max
  - saturation_pct_0:  % rows with value ≤ 0.01 (stuck-dark)
  - saturation_pct_1:  % rows with value ≥ 0.99 (saturated-bright)
  - cluster_pct:       % rows within ±0.05 of the mean (stuck-at-single-level)
  - flat_score:        1.0 − (std / 0.25); 1.0 = totally flat, 0.0 = max breath
                       (max possible std for [0,1] uniform ≈ 0.289; we use 0.25
                       as the "healthy variance" reference per `middle_path.py`
                       coherence formula `1 − var/0.25`)

Per-layer intra-sibling Pearson correlation (avg pairwise within the layer)
captures the "all dims moving together → kills variance" trap.

Output:
  - Per-Titan summary table grouped by layer
  - Re-grounding hit list sorted by flat_score descending (worst variance first)

Usage:
  # Single Titan, local DB
  python scripts/dim_breath_audit.py --db data/consciousness.db --titan-id T1

  # Fleet (T1 local + T2/T3 via SSH; assumes ~/.titan-fleet/ remote-db-fetch
  # is staged, OR pass --remote=root@10.135.0.6:/path/to/db with ssh access)
  python scripts/dim_breath_audit.py --fleet

  # JSON output for downstream tooling (P0.6-B tuning input)
  python scripts/dim_breath_audit.py --db data/consciousness.db --titan-id T1 --json hit_list.json

Audit per Trinity Homeostasis P0.6-A (PLAN_trinity_homeostasis_p0.md §6.6.1).
"""
import argparse
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# 40 BMS dims layout in 132D state_vector (per consciousness.py + spirit_loop.py:1090-1167)
BMS_LAYERS = [
    ("inner_body",  range(0, 5)),
    ("inner_mind",  range(5, 20)),
    ("outer_body",  range(65, 70)),
    ("outer_mind",  range(70, 85)),
]
SATURATION_DARK_THRESHOLD = 0.01
SATURATION_BRIGHT_THRESHOLD = 0.99
CLUSTER_HALFWIDTH = 0.05
HEALTHY_VARIANCE_REF = 0.25  # max var of uniform on [0,1] is ≈0.0833; 0.25 = balanced bimodal


def fetch_window(db_path: str, window_seconds: int) -> list[list[float]]:
    """Pull state_vectors from the last `window_seconds` of epochs."""
    cutoff = time.time() - window_seconds
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = []
    for (sv,) in db.execute("SELECT state_vector FROM epochs WHERE timestamp >= ?", (cutoff,)):
        try:
            v = json.loads(sv or "[]")
            if len(v) >= 85:  # need at least up to outer_mind end
                rows.append(v)
        except json.JSONDecodeError:
            continue
    db.close()
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 1e-12 and dy > 1e-12 else 0.0


def analyze_dim(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    sat_dark = sum(1 for v in values if v <= SATURATION_DARK_THRESHOLD) / n
    sat_bright = sum(1 for v in values if v >= SATURATION_BRIGHT_THRESHOLD) / n
    cluster = sum(1 for v in values if abs(v - mean) <= CLUSTER_HALFWIDTH) / n
    flat = max(0.0, 1.0 - std / math.sqrt(HEALTHY_VARIANCE_REF))
    return {
        "n": n, "mean": mean, "std": std, "min": min(values), "max": max(values),
        "sat_0_pct": sat_dark * 100, "sat_1_pct": sat_bright * 100,
        "cluster_pct": cluster * 100, "flat_score": flat,
    }


def health_tag(stats: dict) -> str:
    """One-glance tag from the stats."""
    if stats["n"] == 0:
        return "NO_DATA"
    if stats["flat_score"] >= 0.95:
        return "🚨 DEAD"  # essentially no breath
    if stats["flat_score"] >= 0.80:
        return "⚠ FLAT"
    if stats["sat_0_pct"] >= 80:
        return "⚫ DARK"
    if stats["sat_1_pct"] >= 80:
        return "⬜ SAT"
    if stats["cluster_pct"] >= 90:
        return "🔘 STUCK"
    if stats["flat_score"] >= 0.55:
        return "🟡 LOW"
    return "✓ OK"


def render_titan(titan_id: str, rows: list[list[float]], window_h: float) -> dict:
    """Print per-Titan table + return structured hit list."""
    header = f"=== {titan_id} — {len(rows):,} rows over last {window_h:.1f}h ==="
    print()
    print(header)
    print("=" * len(header))
    if not rows:
        print("  NO DATA in window — skipping")
        return {"titan": titan_id, "n_rows": 0, "dims": []}

    structured = []
    for layer_name, idx_range in BMS_LAYERS:
        print(f"\n  {layer_name}")
        print(f"  {'dim':>3} | {'tag':>8} | {'mean':>6} | {'std':>6} | {'sat0%':>6} | {'sat1%':>6} | {'clust%':>6} | flat")
        print(f"  {'-'*3} + {'-'*8} + {'-'*6} + {'-'*6} + {'-'*6} + {'-'*6} + {'-'*6} + {'-'*4}")
        layer_vals = []
        for dim in idx_range:
            col = [r[dim] for r in rows if dim < len(r)]
            stats = analyze_dim(col)
            layer_vals.append(col)
            tag = health_tag(stats)
            structured.append({
                "titan": titan_id, "layer": layer_name, "dim_index": dim,
                "tag": tag, **{k: v for k, v in stats.items() if k != "n"},
            })
            print(f"  {dim:>3} | {tag:>8} | {stats['mean']:>6.3f} | {stats['std']:>6.3f} | "
                  f"{stats['sat_0_pct']:>6.1f} | {stats['sat_1_pct']:>6.1f} | "
                  f"{stats['cluster_pct']:>6.1f} | {stats['flat_score']:.3f}")
        # Intra-layer sibling correlation: average pairwise |Pearson|
        if len(layer_vals) >= 2:
            pearsons = []
            for i in range(len(layer_vals)):
                for j in range(i + 1, len(layer_vals)):
                    pearsons.append(abs(pearson(layer_vals[i], layer_vals[j])))
            avg_r = sum(pearsons) / len(pearsons) if pearsons else 0.0
            print(f"  └─ avg |intra-layer sibling Pearson|: {avg_r:.3f}"
                  f" {'⚠ HIGH (all sibling dims co-moving)' if avg_r > 0.85 else ''}")

    return {"titan": titan_id, "n_rows": len(rows), "dims": structured}


def render_hit_list(per_titan: list[dict]) -> list[dict]:
    """Aggregate fleet-wide and sort by worst flat_score."""
    all_dims = [d for t in per_titan for d in t.get("dims", [])]
    all_dims.sort(key=lambda d: d.get("flat_score", 0), reverse=True)
    print()
    print("=" * 78)
    print("FLEET RE-GROUNDING HIT LIST (sorted by flat_score, worst first)")
    print("=" * 78)
    print(f"{'titan':>5} | {'layer':>11} | {'dim':>3} | {'tag':>8} | {'flat':>5} | "
          f"{'sat0%':>6} | {'sat1%':>6} | {'clust%':>6}")
    print(f"{'-'*5} + {'-'*11} + {'-'*3} + {'-'*8} + {'-'*5} + {'-'*6} + {'-'*6} + {'-'*6}")
    for d in all_dims[:30]:  # top 30 worst
        print(f"{d['titan']:>5} | {d['layer']:>11} | {d['dim_index']:>3} | "
              f"{d['tag']:>8} | {d.get('flat_score', 0):>5.3f} | "
              f"{d.get('sat_0_pct', 0):>6.1f} | {d.get('sat_1_pct', 0):>6.1f} | "
              f"{d.get('cluster_pct', 0):>6.1f}")
    print()
    print(f"Total dims audited fleet-wide: {len(all_dims)}")
    dead = [d for d in all_dims if d.get("flat_score", 0) >= 0.95]
    flat = [d for d in all_dims if 0.80 <= d.get("flat_score", 0) < 0.95]
    print(f"  🚨 DEAD  (flat ≥ 0.95): {len(dead):>3}")
    print(f"  ⚠  FLAT  (flat 0.80–0.95): {len(flat):>3}")
    return all_dims


def fetch_remote(spec: str) -> str:
    """spec = 'host:/path/to/consciousness.db' — scp into a temp file, return local path."""
    if ":" not in spec:
        return spec
    host, remote = spec.split(":", 1)
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    subprocess.run(["scp", "-q", f"{host}:{remote}", tmp.name], check=True)
    return tmp.name


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", help="Path to consciousness.db (or host:path for remote scp)")
    p.add_argument("--titan-id", default="T1")
    p.add_argument("--window-hours", type=float, default=24.0)
    p.add_argument("--fleet", action="store_true",
                   help="Audit T1 (local data/consciousness.db) + T2 + T3 (via ssh)")
    p.add_argument("--json", help="Also write structured hit list to this JSON path")
    args = p.parse_args()

    window_s = int(args.window_hours * 3600)
    per_titan = []

    if args.fleet:
        fleet = [
            ("T1", "data/consciousness.db"),
            ("T2", "root@10.135.0.6:/home/antigravity/projects/titan/data/consciousness.db"),
            ("T3", "root@10.135.0.6:/home/antigravity/projects/titan3/data/consciousness.db"),
        ]
    elif args.db:
        fleet = [(args.titan_id, args.db)]
    else:
        p.error("provide --db <path> or --fleet")

    for tid, spec in fleet:
        try:
            local = fetch_remote(spec)
            rows = fetch_window(local, window_s)
            per_titan.append(render_titan(tid, rows, args.window_hours))
        except Exception as e:
            print(f"\n[{tid}] FAILED: {e}", file=sys.stderr)
            per_titan.append({"titan": tid, "n_rows": 0, "dims": [], "error": str(e)})

    hit_list = render_hit_list(per_titan)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "audit_window_hours": args.window_hours,
                "generated_ts": time.time(),
                "per_titan_rows": {t["titan"]: t["n_rows"] for t in per_titan},
                "hit_list": hit_list,
            }, f, indent=2)
        print(f"\nJSON hit list → {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
