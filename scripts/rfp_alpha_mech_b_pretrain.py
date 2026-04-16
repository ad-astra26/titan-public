#!/usr/bin/env python3
"""
rFP α Mechanism B offline pre-train — train StepValueNet from action_chains_step.

Purpose
-------
Before activating Phase 3 (Mech B at 70% weight), pre-train the value net
offline on accumulated step snapshots from Phase 2. This avoids Phase 3
activating with a random-init net that would inject noise at high weight.

Data source
-----------
`action_chains_step` SQLite table in `data/inner_memory.db`, populated
during Phase 1+2 by `_rr_persist_step_snapshots`. Each row:
  - chain_id, step_idx, created_at
  - terminal_reward (target for TD regression)
  - outcome_action (COMMIT / ABANDON / HOLD)
  - confidence, gut_agreement (at that step)
  - chain_prefix (primitive list up to step_idx+1)
  - last_primitive
  - policy_input_json (the 115D state at step time)

Training
--------
1. Group rows by chain_id. Sort each chain's steps by step_idx.
2. For each step, construct StepValueNet input via build_input()
3. Target = chain's terminal_reward (all steps in chain share target)
4. Shuffle, split 80/20 train/val, train for N epochs on train, eval on val
5. Z-score normalization runs via engine's EMA (same as online)

Output
------
`data/reasoning/value_head.json.pretrained` — staged file. Deploy with
--deploy flag (requires T1 stopped, same safety as warm-start script).

Safety
------
- Read-only on inner_memory.db
- Default output staged, not main
- Detects T1 running, warns/blocks as appropriate
- --dry-run for preview without writes
- Target row count >= 500 (below that, training is unreliable)

Arguments
---------
--db-path        SQLite path (default: data/inner_memory.db)
--out-dir        Output dir (default: data/reasoning)
--epochs         Training epochs (default: 10)
--batch-size     Minibatch size (default: 32)
--min-rows       Refuse to train if fewer rows available (default: 500)
--since          ISO timestamp floor on created_at (default: no floor)
--dry-run        Stats preview only
--deploy         Promote staged file to main (requires T1 stopped)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from titan_plugin.logic.reasoning import StepValueNet  # noqa: E402


STAGED_FILENAME = "value_head.json.pretrained"
MAIN_FILENAME = "value_head.json"


def _t1_running() -> bool:
    try:
        r = subprocess.run(
            ["pgrep", "-f", "titan_main"],
            capture_output=True, text=True, timeout=3)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def _load_rows(db_path: str, since_ts: float, max_rows: int = 50000,
               required_dim: int = 115) -> tuple[list[dict], dict]:
    """Load step_snapshots from SQLite, filtering to rows where policy_input
    length matches the current engine's policy_input_dim.

    Older rows may have been written with a different dim (99D observed);
    including them would cause matmul shape errors in the net. Returns
    (rows, stats_dict) where stats_dict tracks skip counts by dim.
    """
    conn = sqlite3.connect(db_path, timeout=5.0)
    skip_stats: dict = {"skipped_dim_mismatch": 0, "dim_histogram": {}}
    try:
        cur = conn.execute(
            """
            SELECT chain_id, step_idx, created_at, terminal_reward,
                   outcome_action, confidence, gut_agreement,
                   chain_prefix, last_primitive, policy_input_json
            FROM action_chains_step
            WHERE created_at >= ?
            ORDER BY chain_id ASC, step_idx ASC
            LIMIT ?
            """,
            (since_ts, max_rows),
        )
        out = []
        for r in cur:
            try:
                pi_list = json.loads(r[9]) if r[9] else []
                pi_len = len(pi_list)
                skip_stats["dim_histogram"][pi_len] = \
                    skip_stats["dim_histogram"].get(pi_len, 0) + 1
                if pi_len != required_dim:
                    skip_stats["skipped_dim_mismatch"] += 1
                    continue
                out.append({
                    "chain_id": int(r[0]),
                    "step_idx": int(r[1]),
                    "created_at": float(r[2]),
                    "terminal_reward": float(r[3]),
                    "outcome_action": str(r[4]),
                    "confidence": float(r[5]),
                    "gut_agreement": float(r[6]),
                    "chain_prefix": json.loads(r[7]) if r[7] else [],
                    "last_primitive": str(r[8]) if r[8] else "",
                    "policy_input": pi_list,
                })
            except Exception:
                continue
        return out, skip_stats
    finally:
        conn.close()


def _load_engine_cfg() -> dict:
    """Read policy_input_dim + Mech B params from titan_params.toml."""
    import tomllib
    with open("titan_plugin/titan_params.toml", "rb") as f:
        params = tomllib.load(f)
    reasoning = params.get("reasoning", {})
    rr = params.get("reasoning_rewards", {})
    return {
        "policy_input_dim": int(reasoning.get("policy_input_dim", 115)),
        "max_chain_length": int(reasoning.get("max_chain_length", 10)),
        "mech_b_h1": int(rr.get("mech_b_h1", 64)),
        "mech_b_h2": int(rr.get("mech_b_h2", 32)),
        "mech_b_lr": float(rr.get("mech_b_learning_rate", 0.001)),
        "mech_b_vema": float(rr.get("mech_b_vtarget_ema_alpha", 0.01)),
    }


def _train(net: StepValueNet, rows: list[dict], max_chain: int,
           epochs: int, batch_size: int, val_frac: float = 0.2) -> dict:
    """Offline training loop. Returns final train/val loss stats."""
    n = len(rows)
    rng = np.random.default_rng(42)
    # Shuffle + split
    indices = np.arange(n)
    rng.shuffle(indices)
    val_n = int(n * val_frac)
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    def _iter_batches(idx_arr):
        for i in range(0, len(idx_arr), batch_size):
            yield idx_arr[i:i + batch_size]

    # Pre-build all inputs once (avoid re-computing each epoch)
    def _build_input_for_row(r):
        pi = np.array(r["policy_input"], dtype=np.float64)
        return net.build_input(
            policy_input=pi,
            confidence=r["confidence"],
            gut_agreement=r["gut_agreement"],
            chain_len=r["step_idx"] + 1,
            max_chain=max_chain,
            last_primitive=r["last_primitive"],
        )

    inputs = [_build_input_for_row(r) for r in rows]
    targets = [r["terminal_reward"] for r in rows]

    stats = {"epochs": [], "n_train": len(train_idx), "n_val": len(val_idx)}
    for epoch in range(epochs):
        rng.shuffle(train_idx)
        train_loss_sum = 0.0
        train_n = 0
        for batch in _iter_batches(train_idx):
            for i in batch:
                loss = net.train_step(inputs[i], targets[i])
                train_loss_sum += loss
                train_n += 1
        train_loss_avg = train_loss_sum / max(1, train_n)

        # Validation — forward only, no training
        val_sum = 0.0
        val_n_cnt = 0
        for i in val_idx:
            pred_norm = net.forward(inputs[i])
            target_norm = (targets[i] - net._vtarget_mean) / (net._vtarget_std + 1e-5)
            val_sum += 0.5 * (pred_norm - target_norm) ** 2
            val_n_cnt += 1
        val_loss_avg = val_sum / max(1, val_n_cnt)
        stats["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss_avg, 5),
            "val_loss": round(val_loss_avg, 5),
            "vtarget_mean": round(net._vtarget_mean, 4),
            "vtarget_std": round(net._vtarget_std, 4),
        })
        print(f"  epoch {epoch+1:2d}/{epochs}  train={train_loss_avg:.5f}  "
              f"val={val_loss_avg:.5f}  μ={net._vtarget_mean:.4f}  "
              f"σ={net._vtarget_std:.4f}")
    return stats


def main() -> int:
    p = argparse.ArgumentParser(
        description="rFP α Mechanism B offline pre-train from action_chains_step")
    p.add_argument("--db-path", default="data/inner_memory.db")
    p.add_argument("--out-dir", default="data/reasoning")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--min-rows", type=int, default=500,
                   help="refuse to train if fewer rows available")
    p.add_argument("--since", default=None,
                   help="ISO 'YYYY-MM-DD HH:MM:SS' UTC floor (default: all data)")
    p.add_argument("--max-rows", type=int, default=50000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--deploy", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    staged_path = os.path.join(args.out_dir, STAGED_FILENAME)
    main_path = os.path.join(args.out_dir, MAIN_FILENAME)

    # Deploy mode
    if args.deploy:
        if _t1_running():
            print("ERROR: T1 running. Stop T1 before --deploy.", file=sys.stderr)
            return 2
        if not os.path.exists(staged_path):
            print(f"ERROR: no staged file at {staged_path}", file=sys.stderr)
            return 2
        if os.path.exists(main_path):
            backup = main_path + f".pre_pretrain.{int(time.time())}"
            shutil.copy2(main_path, backup)
            print(f"  ✓ backed up {main_path} → {backup}")
        shutil.move(staged_path, main_path)
        print(f"  ✓ promoted {staged_path} → {main_path}")
        return 0

    # Training mode
    if _t1_running() and not args.dry_run:
        print("  ⚠ T1 is running. Staged file will be written safely. "
              "Stop T1 before --deploy.")

    since_ts = 0.0
    if args.since:
        dt = datetime.strptime(args.since, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        since_ts = dt.timestamp()

    print("=" * 72)
    print("rFP α Mechanism B — offline pre-train from action_chains_step")
    print("=" * 72)
    print(f"  db_path:    {args.db_path}")
    print(f"  out_dir:    {args.out_dir}")
    print(f"  epochs:     {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  min_rows:   {args.min_rows}")
    print(f"  since:      {args.since or '(all)'}")
    print(f"  dry_run:    {args.dry_run}")
    print()

    cfg = _load_engine_cfg()
    print(f"  Engine config from toml:")
    print(f"    policy_input_dim = {cfg['policy_input_dim']}")
    print(f"    max_chain = {cfg['max_chain_length']}")
    print(f"    B dims = {cfg['mech_b_h1']}→{cfg['mech_b_h2']}→1")
    print()

    rows, skip_stats = _load_rows(args.db_path, since_ts, args.max_rows,
                                  required_dim=cfg["policy_input_dim"])
    print(f"  loaded {len(rows)} step rows (filter: policy_input len == {cfg['policy_input_dim']})")
    if skip_stats["skipped_dim_mismatch"] > 0:
        print(f"  skipped {skip_stats['skipped_dim_mismatch']} rows due to dim mismatch")
        print(f"  dim histogram: {sorted(skip_stats['dim_histogram'].items())}")
    if len(rows) < args.min_rows:
        print(f"  ⚠ below min_rows={args.min_rows} — insufficient data")
        return 1

    # Quick stats
    unique_chains = len(set(r["chain_id"] for r in rows))
    t_rewards = [r["terminal_reward"] for r in rows]
    print(f"  unique chains: {unique_chains}")
    print(f"  terminal_reward: min={min(t_rewards):.4f} "
          f"max={max(t_rewards):.4f} mean={sum(t_rewards)/len(t_rewards):.4f}")
    print(f"  earliest row: {datetime.fromtimestamp(min(r['created_at'] for r in rows), tz=timezone.utc)}")
    print(f"  latest row:   {datetime.fromtimestamp(max(r['created_at'] for r in rows), tz=timezone.utc)}")
    print()

    if args.dry_run:
        print("  (dry-run — no training)")
        return 0

    # Instantiate fresh net (no load — we want to train from scratch on this data)
    net = StepValueNet(
        policy_input_dim=cfg["policy_input_dim"],
        hidden_1=cfg["mech_b_h1"],
        hidden_2=cfg["mech_b_h2"],
        learning_rate=cfg["mech_b_lr"],
        vtarget_ema_alpha=cfg["mech_b_vema"],
    )
    print(f"  instantiated StepValueNet: input_dim={net.input_dim} "
          f"(policy_input_dim+11 = {cfg['policy_input_dim']+11})")
    print()
    print("  training...")
    stats = _train(net, rows, cfg["max_chain_length"], args.epochs, args.batch_size)

    # Save staged
    net.save(staged_path)
    size = os.path.getsize(staged_path)
    print()
    print(f"  ✓ wrote staged value_head → {staged_path} ({size/1024:.1f} KB)")
    print(f"    final train_loss={stats['epochs'][-1]['train_loss']:.5f}")
    print(f"    final val_loss={stats['epochs'][-1]['val_loss']:.5f}")
    print(f"    vtarget μ={net._vtarget_mean:.4f} σ={net._vtarget_std:.4f}")
    print()
    print("  Next steps:")
    print(f"    1. Review: less {staged_path}")
    print(f"    2. Stop T1: pkill -9 -f scripts/titan_main")
    print(f"    3. Deploy: python {__file__} --deploy")
    print(f"    4. Start T1 — engine loads pre-trained weights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
