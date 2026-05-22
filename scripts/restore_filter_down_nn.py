#!/usr/bin/env python3
"""Restore a Titan's persisted FilterDownV5 NN checkpoint from a backup.

The Phase C migration (T1: `*.pre_phase_c_migration.20260514`; T3: the
`_pre_c_s7_corrupt_state_20260505_*` quarantine) sidelined each Titan's
trained unified-spirit filter_down brain and reset the live engine to a
fresh (0-step) network — a `directive_memory_preservation` violation. The
backups are otherwise valid v1 data; they predate the `schema_version`
field the Rust loader was later given (`filter_down.rs` load/load_state
reject any file whose `schema_version != 1`). This tool injects that field
and reinstalls the matched weights+state(+buffer) checkpoint, after
validating architecture shapes (162x128 / 128x64 / 64x1) and finiteness.

Dry-run by default. `--apply` backs up the current files to
`*.pre_nn_restore_<ts>` then atomically installs the restored checkpoint.

The Titan MUST be stopped before `--apply` (the engine persists on
shutdown and would overwrite the restore). Restart after, then verify via
`ShmReaderBank.read_filter_down_state()` that `total_train_steps` matches
the backup and `multipliers_mean` is no longer all-1.0.

Usage:
  python scripts/restore_filter_down_nn.py \
      --data-dir /home/antigravity/projects/titan/data \
      --weights-backup data/filter_down_v5_weights.json.pre_phase_c_migration.20260514 \
      --state-backup   data/filter_down_v5_state.json.pre_phase_c_migration.20260514 \
      [--buffer-backup data/filter_down_v5_buffer.json] \
      [--apply]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

WEIGHTS_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
# Rust TrinityValueNet: 162 -> 128 -> 64 -> 1 (SPEC §G7).
EXPECTED_W = {"w1": (162, 128), "w2": (128, 64), "w3": (64, 1)}
EXPECTED_B = {"b1": 128, "b2": 64, "b3": 1}
STATE_FIELDS = (
    "total_train_steps",
    "last_loss",
    "recent_losses",
    "phase8_snapshot_taken",
    "multipliers_ema",
)


def _count_nonfinite(arr) -> int:
    bad = 0
    stack = [arr]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, (int, float)):
            if math.isnan(x) or math.isinf(x):
                bad += 1
    return bad


def validate_weights(w: dict) -> list[str]:
    errs: list[str] = []
    for k, (rows, cols) in EXPECTED_W.items():
        if k not in w:
            errs.append(f"missing {k}")
            continue
        if len(w[k]) != rows or not w[k] or len(w[k][0]) != cols:
            got = f"{len(w[k])}x{len(w[k][0]) if w[k] else '?'}"
            errs.append(f"{k} shape {got} != {rows}x{cols}")
    for k, n in EXPECTED_B.items():
        if k not in w:
            errs.append(f"missing {k}")
        elif len(w[k]) != n:
            errs.append(f"{k} len {len(w[k])} != {n}")
    nonfinite = sum(_count_nonfinite(w.get(k, [])) for k in (*EXPECTED_W, *EXPECTED_B))
    if nonfinite:
        errs.append(f"{nonfinite} NaN/Inf weight values")
    return errs


def validate_state(s: dict) -> list[str]:
    errs = [f"missing {k}" for k in STATE_FIELDS if k not in s]
    if not isinstance(s.get("total_train_steps"), int):
        errs.append("total_train_steps not an int")
    return errs


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--weights-backup", required=True, type=Path)
    ap.add_argument("--state-backup", required=True, type=Path)
    ap.add_argument("--buffer-backup", type=Path, default=None)
    ap.add_argument("--apply", action="store_true", help="perform the restore (default: dry-run)")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    weights_dst = data_dir / "filter_down_v5_weights.json"
    state_dst = data_dir / "filter_down_v5_state.json"
    buffer_dst = data_dir / "filter_down_v5_buffer.json"

    for p in (args.weights_backup, args.state_backup):
        if not p.exists():
            print(f"ERROR: backup not found: {p}", file=sys.stderr)
            return 2

    weights = json.loads(args.weights_backup.read_text())
    state = json.loads(args.state_backup.read_text())

    w_errs = validate_weights(weights)
    s_errs = validate_state(state)
    if w_errs or s_errs:
        print("VALIDATION FAILED — refusing to restore:")
        for e in w_errs:
            print(f"  weights: {e}")
        for e in s_errs:
            print(f"  state:   {e}")
        return 3

    weights["schema_version"] = WEIGHTS_SCHEMA_VERSION
    state["schema_version"] = STATE_SCHEMA_VERSION

    print("=== restore plan ===")
    print(f"  data-dir            : {data_dir}")
    print(f"  weights backup      : {args.weights_backup}  (schema_version injected -> {WEIGHTS_SCHEMA_VERSION})")
    print(f"  state backup        : {args.state_backup}    (total_train_steps={state['total_train_steps']}, schema_version injected -> {STATE_SCHEMA_VERSION})")
    if args.buffer_backup:
        n = len(json.loads(args.buffer_backup.read_text())) if args.buffer_backup.exists() else "MISSING"
        print(f"  buffer backup       : {args.buffer_backup}  (transitions={n})")
    print("  validation          : PASS (shapes 162x128/128x64/64x1, finite)")

    if not args.apply:
        print("\nDRY-RUN — re-run with --apply to install (Titan must be STOPPED first).")
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    for dst in (weights_dst, state_dst, buffer_dst):
        if dst.exists():
            bak = dst.with_name(dst.name + f".pre_nn_restore_{ts}")
            shutil.copy2(dst, bak)
            print(f"  backed up current  : {dst.name} -> {bak.name}")

    _atomic_write(weights_dst, weights)
    _atomic_write(state_dst, state)
    print(f"  installed          : {weights_dst.name} + {state_dst.name}")
    if args.buffer_backup and args.buffer_backup.exists():
        shutil.copy2(args.buffer_backup, buffer_dst)
        print(f"  installed          : {buffer_dst.name}")
    print("\nRESTORE COMPLETE — restart the Titan, then verify total_train_steps + multipliers_mean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
