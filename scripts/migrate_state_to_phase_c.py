#!/usr/bin/env python3
"""migrate_state_to_phase_c.py — one-shot Phase A+B → Phase C state file migrator.

Run BEFORE flipping `l0_rust_enabled=true` on T1 or T2. Translates the 3
JSON state files that the Rust kernel-rs reads on boot into the Phase C
schema by:

1. **`unified_spirit_state.json`** — Add `schema_version: 1` at the top
   level. For each entry in `epochs`, ensure `resonance_snapshot.pair_big_pulse_counts`
   exists as an empty dict (matches Rust `BTreeMap<String, u64>` default
   when no per-pair counter snapshot is available from Phase A+B history).
   The 130D `spirit_tensor` + `magnitude` + `velocity` + `anchor_hash` per
   epoch are preserved verbatim — 24+ days of subjective-time history kept.

2. **`filter_down_v5_state.json`** — Add `schema_version: 1` at the top
   level. All other fields (`total_train_steps`, `last_loss`,
   `recent_losses`, `phase8_snapshot_taken`, `multipliers_ema`) carry
   forward verbatim with correct shapes (5/15/40 per EMA group).

3. **`resonance_state.json`** — Add `schema_version: 1` at the top level.
   For each entry in `pairs`, strip the leading underscore from
   Python's `_consecutive_resonant` / `_total_*` / `_big_pulse_count` etc.
   field names so the Rust serde deserializer can find them. Top-level
   `great_pulse_count` + `last_great_pulse_ts` carry forward.

The script ALWAYS writes a `.pre_phase_c_migration.YYYYMMDD` backup of
each original file before mutating. Originals are NEVER deleted (per
`directive_memory_preservation.md`). Idempotent: if a file already has
`schema_version` set, it is skipped (no-op).

Per Phase C T1+T2 migration session 2026-05-14 (Task #2).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

# Rust schema version constants (mirror `titan-rust/crates/.../constants.rs`).
UNIFIED_SPIRIT_STATE_SCHEMA_VERSION = 1
FILTER_DOWN_V5_STATE_SCHEMA_VERSION = 1
FILTER_DOWN_WEIGHTS_SCHEMA_VERSION = 1
RESONANCE_STATE_SCHEMA_VERSION = 1


def _backup(path: Path) -> Path:
    """Copy path → path.pre_phase_c_migration.YYYYMMDD. Idempotent —
    if backup already exists with today's date, suffix with -N."""
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    base = path.with_suffix(
        path.suffix + f".pre_phase_c_migration.{today}"
    )
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = base.with_suffix(base.suffix + f"-{counter}")
        counter += 1
    shutil.copy2(path, candidate)
    return candidate


def _atomic_write_json(path: Path, data: dict) -> None:
    """Atomic write — temp file + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


def migrate_unified_spirit_state(path: Path) -> Tuple[bool, str]:
    """Migrate `unified_spirit_state.json` to Phase C schema.

    Returns (changed: bool, summary: str).
    """
    if not path.exists():
        return False, "MISSING (not present — Rust will bootstrap fresh)"
    with open(path) as f:
        data = json.load(f)
    existing_version = data.get("schema_version")
    # Re-run defense: schema_version alone is insufficient — a partial
    # prior migration may have set it before fixing all epoch[].resonance_snapshot
    # entries. Sample the FIRST epoch's snapshot to confirm it has the
    # minimal Phase C shape (great_pulse_count + pair_big_pulse_counts + ts,
    # NOT the legacy nested `pairs` map).
    sample_epoch = (data.get("epochs") or [{}])[0]
    sample_rs = sample_epoch.get("resonance_snapshot") if sample_epoch else None
    fully_migrated = (
        existing_version == UNIFIED_SPIRIT_STATE_SCHEMA_VERSION
        and isinstance(sample_rs, dict)
        and "ts" in sample_rs
        and "great_pulse_count" in sample_rs
        and "pair_big_pulse_counts" in sample_rs
        and "pairs" not in sample_rs
    )
    if fully_migrated:
        return False, (
            f"ALREADY MIGRATED (schema_version={existing_version}, "
            f"sample epoch resonance_snapshot in minimal Phase C shape)"
        )

    backup_path = _backup(path)
    n_epochs = len(data.get("epochs", []))
    n_resonance_fixed = 0
    for epoch in data.get("epochs", []):
        rs = epoch.get("resonance_snapshot")
        if not isinstance(rs, dict):
            continue
        # Phase A+B stored the FULL ResonanceDetector.to_dict() in each
        # epoch's resonance_snapshot — a nested {pairs: {body/mind/spirit: ...}}
        # structure. Phase C's Rust ResonanceSnapshot is the minimal trio
        # {great_pulse_count, pair_big_pulse_counts, ts}. Translate by:
        #   - great_pulse_count := epoch.epoch_id (each GreatEpoch IS a
        #     GREAT PULSE coincidence by definition, so epoch_id is the
        #     best historical estimate of the running great_pulse_count)
        #   - pair_big_pulse_counts := {pair_name: pairs[pair_name].big_pulse_count}
        #   - ts := epoch.timestamp (closest historical proxy for the
        #     wall-clock instant of the GREAT PULSE)
        # Idempotent re-migration: detect the Phase A+B "pairs" key as
        # the marker that this epoch's resonance_snapshot is the OLD
        # nested format; presence of `ts` + absence of `pairs` means
        # it's already minimal.
        is_phase_ab_format = ("pairs" in rs and "ts" not in rs)
        if is_phase_ab_format:
            pair_big_pulse_counts: dict[str, int] = {}
            for pair_name, pair_state in (rs.get("pairs") or {}).items():
                if isinstance(pair_state, dict):
                    val = pair_state.get("big_pulse_count")
                    if isinstance(val, int):
                        pair_big_pulse_counts[pair_name] = val
            new_rs = {
                "great_pulse_count": int(epoch.get("epoch_id", 0) or 0),
                "pair_big_pulse_counts": pair_big_pulse_counts,
                "ts": float(epoch.get("timestamp", 0.0) or 0.0),
            }
            epoch["resonance_snapshot"] = new_rs
            n_resonance_fixed += 1
        else:
            # Already minimal (e.g., a previous partial migration run added
            # empty pair_big_pulse_counts but left other fields wrong).
            # Defensive: ensure ALL three required keys exist.
            if "pair_big_pulse_counts" not in rs:
                rs["pair_big_pulse_counts"] = {}
                n_resonance_fixed += 1
            if "ts" not in rs:
                rs["ts"] = float(epoch.get("timestamp", 0.0) or 0.0)
                n_resonance_fixed += 1
            if "great_pulse_count" not in rs:
                rs["great_pulse_count"] = int(epoch.get("epoch_id", 0) or 0)
                n_resonance_fixed += 1
    data["schema_version"] = UNIFIED_SPIRIT_STATE_SCHEMA_VERSION

    _atomic_write_json(path, data)
    return True, (
        f"OK — schema_version={UNIFIED_SPIRIT_STATE_SCHEMA_VERSION} added, "
        f"{n_resonance_fixed}/{n_epochs} epochs gained empty pair_big_pulse_counts; "
        f"backup={backup_path.name}"
    )


def migrate_filter_down_v5_state(path: Path) -> Tuple[bool, str]:
    """Migrate `filter_down_v5_state.json` to Phase C schema.

    Returns (changed: bool, summary: str).
    """
    if not path.exists():
        return False, "MISSING (not present — Rust will bootstrap fresh)"
    with open(path) as f:
        data = json.load(f)
    existing_version = data.get("schema_version")
    if existing_version == FILTER_DOWN_V5_STATE_SCHEMA_VERSION:
        return False, f"ALREADY MIGRATED (schema_version={existing_version})"

    backup_path = _backup(path)
    data["schema_version"] = FILTER_DOWN_V5_STATE_SCHEMA_VERSION
    # Sanity: confirm shapes of multipliers_ema match Rust struct
    # (5/15/40/5/15/40). The Rust deserializer accepts Vec<f64> of any
    # length, but downstream consumers assume the canonical shapes.
    ema = data.get("multipliers_ema") or {}
    shape_hints = []
    for group, expected_len in [
        ("inner_body", 5), ("inner_mind", 15), ("inner_spirit_content", 40),
        ("outer_body", 5), ("outer_mind", 15), ("outer_spirit_content", 40),
    ]:
        actual = len(ema.get(group, [])) if isinstance(ema.get(group), list) else 0
        if actual != expected_len:
            shape_hints.append(f"{group}={actual}!={expected_len}")

    _atomic_write_json(path, data)
    summary = (
        f"OK — schema_version={FILTER_DOWN_V5_STATE_SCHEMA_VERSION} added; "
        f"backup={backup_path.name}"
    )
    if shape_hints:
        summary += f"; ⚠ shape drift: {','.join(shape_hints)}"
    return True, summary


def migrate_filter_down_v5_weights(path: Path) -> Tuple[bool, str]:
    """Migrate `filter_down_v5_weights.json` to Phase C schema.

    Same as filter_down_v5_state.json — just add `schema_version: 1`.
    Weights matrices (w1/b1/w2/b2 layers) carry forward verbatim.
    Phase C deserializer's shape-check verifies V5 162D matrices on load.
    """
    if not path.exists():
        return False, "MISSING (not present — Rust will bootstrap fresh)"
    with open(path) as f:
        data = json.load(f)
    existing_version = data.get("schema_version")
    if existing_version == FILTER_DOWN_WEIGHTS_SCHEMA_VERSION:
        return False, f"ALREADY MIGRATED (schema_version={existing_version})"

    backup_path = _backup(path)
    data["schema_version"] = FILTER_DOWN_WEIGHTS_SCHEMA_VERSION
    _atomic_write_json(path, data)
    return True, (
        f"OK — schema_version={FILTER_DOWN_WEIGHTS_SCHEMA_VERSION} added; "
        f"backup={backup_path.name}"
    )


def migrate_resonance_state(path: Path) -> Tuple[bool, str]:
    """Migrate `resonance_state.json` to Phase C schema.

    Strips leading underscore from per-pair field names (Python pickled
    them as `_consecutive_resonant` etc. but Rust serde expects the
    bare names per `ResonancePairState`).

    Returns (changed: bool, summary: str).
    """
    if not path.exists():
        return False, "MISSING (not present — Rust will bootstrap fresh)"
    with open(path) as f:
        data = json.load(f)
    existing_version = data.get("schema_version")
    if existing_version == RESONANCE_STATE_SCHEMA_VERSION:
        # Also verify no _underscore fields remain (defensive — schema
        # version alone could lie if someone mutated mid-flight)
        bad = []
        for pair_name, pair_state in (data.get("pairs") or {}).items():
            for k in pair_state:
                if k.startswith("_"):
                    bad.append(f"{pair_name}.{k}")
        if bad:
            return False, (
                f"⚠ schema_version says migrated but underscore fields remain: "
                f"{bad[:5]}"
            )
        return False, f"ALREADY MIGRATED (schema_version={existing_version})"

    backup_path = _backup(path)
    n_pairs = 0
    n_fields_renamed = 0
    fields_to_rename = {
        "_consecutive_resonant": "consecutive_resonant",
        "_total_resonant_cycles": "total_resonant_cycles",
        "_total_checks": "total_checks",
        "_big_pulse_count": "big_pulse_count",
        "_last_big_pulse_ts": "last_big_pulse_ts",
        "_inner_pulse_count": "inner_pulse_count",
        "_outer_pulse_count": "outer_pulse_count",
        "_is_resonant": "is_resonant",
    }
    for pair_name, pair_state in (data.get("pairs") or {}).items():
        n_pairs += 1
        for old_key, new_key in fields_to_rename.items():
            if old_key in pair_state:
                pair_state[new_key] = pair_state.pop(old_key)
                n_fields_renamed += 1
    data["schema_version"] = RESONANCE_STATE_SCHEMA_VERSION

    _atomic_write_json(path, data)
    return True, (
        f"OK — schema_version={RESONANCE_STATE_SCHEMA_VERSION} added, "
        f"{n_fields_renamed} field-rename operations across {n_pairs} pairs; "
        f"backup={backup_path.name}"
    )


def main(data_dir: str) -> int:
    base = Path(data_dir)
    if not base.is_dir():
        print(f"ERROR: data_dir {data_dir!r} does not exist or is not a directory",
              file=sys.stderr)
        return 2

    print(f"Phase C state-schema migration for: {base}")
    print()

    targets = [
        ("unified_spirit_state.json", migrate_unified_spirit_state),
        ("filter_down_v5_state.json", migrate_filter_down_v5_state),
        ("filter_down_v5_weights.json", migrate_filter_down_v5_weights),
        ("resonance_state.json", migrate_resonance_state),
    ]
    any_failed = False
    for name, fn in targets:
        path = base / name
        try:
            changed, summary = fn(path)
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {name}: ERROR {type(e).__name__}: {e}")
            any_failed = True
            continue
        prefix = "  ✓" if changed else "  ·"
        print(f"{prefix} {name}: {summary}")

    print()
    if any_failed:
        print("⚠ Some files failed migration — review and rerun. "
              "Backups untouched if a write step did not start.")
        return 1
    print("✓ All targets processed. Safe to flip `l0_rust_enabled=true` "
          "in the per-Titan TOML override and restart via systemd.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: migrate_state_to_phase_c.py <data_dir>")
        print()
        print("Example: migrate_state_to_phase_c.py /home/antigravity/projects/titan/data")
        print()
        print("Idempotent — safe to rerun. Originals always preserved as")
        print(".pre_phase_c_migration.YYYYMMDD backups (never deleted).")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
