#!/usr/bin/env python3
"""migrate_consciousness_db_blob_quantize.py — D-SPEC-127 (planned)

One-shot migration of `data/consciousness.db` `epochs` table from TEXT-JSON
vector storage → BLOB quantized storage. Cuts row data ~8x; cuts DB file
~75% after VACUUM (~4.4GB → ~600MB on T1 1M-row scale).

Per Maker design call 2026-05-25 (Task #25-B), refined 2026-05-25 after live audit:
  - state_vector:      TEXT-JSON 130D float64 → BLOB f32 LOSSLESS (520B/row)
                       Originally proposed u8 quantization, but live audit
                       revealed 81% of rows have values OUTSIDE [0,1]
                       (max overshoot 2.14) — SPEC §G3 says "bounded [0,1]"
                       but writer doesn't clamp historically. f32 preserves
                       every value exactly (no clamping, no quantization).
                       Separate bug-file required for the §G3 bounds violation.
  - drift_vector:      TEXT-JSON 9D float (CAN be negative) → BLOB f32 (36B/row)
  - trajectory_vector: TEXT-JSON 9D float                   → BLOB f32 (36B/row)
  - distillation:      TEXT (sparse — 132/1M rows non-empty) → kept as-is
                       (sparse migration to separate table = optional second pass)
  - anchored_tx:       TEXT 88-char Solana sig → kept as-is

Algorithm:
  1. ATTACH original DB read-only.
  2. CREATE TABLE epochs_migrated (... same schema + new BLOB columns).
  3. Stream rows in batches of 10_000:
     - parse state_vector JSON → pad/clip to 130D → u8 quantize → BLOB
     - parse drift_vector / trajectory_vector → 9× f32 LE → BLOB
     - keep distillation / anchored_tx as-is
  4. CREATE INDEX on (epoch_id), (timestamp).
  5. VACUUM, ANALYZE.
  6. Atomic rename: epochs → epochs_legacy_TEXT, epochs_migrated → epochs.

Safety:
  - DRY-RUN by default — count rows + size estimate, no writes.
  - Run on a COPY of consciousness.db first (per Maker `directive_memory_preservation`).
  - Old TEXT columns kept (renamed) until verified — drop in second pass.
  - State_vector dim varies historically (9D legacy → 130D modern) — handled
    by JSON-parse-then-pad-or-clip-to-130D.

Usage:
  # DRY-RUN (default) — shows row count + projected sizes, no changes
  python3 scripts/migrate_consciousness_db_blob_quantize.py

  # ACTUAL MIGRATION — on a copy first, then verify, then swap
  cp data/consciousness.db data/consciousness.db.pre_blob_migration
  python3 scripts/migrate_consciousness_db_blob_quantize.py --apply \
      --source data/consciousness.db.pre_blob_migration \
      --target data/consciousness.db.migrated

  # Verify counts + sample rows roundtrip then atomic swap:
  mv data/consciousness.db data/consciousness.db.OLD_pre_blob
  mv data/consciousness.db.migrated data/consciousness.db
"""
import argparse
import json
import sqlite3
import struct
import sys
import time
from pathlib import Path


STATE_VECTOR_DIM = 130   # modern unified-spirit + observer count
DRIFT_VECTOR_DIM = 9     # 9D drift
TRAJ_VECTOR_DIM = 9      # 9D trajectory


def pack_f32(vals: list[float], target_dim: int = None) -> bytes:
    """Pack signed floats as f32 LE BLOB at the row's EXACT length — pure
    TEXT→BLOB encoding, NO pad/clip (SPEC §11.H.1.bis + consciousness.pack_vector).

    Preserving exact dims is load-bearing: the live writer packs the full
    `state.to_list()` (currently 132D = 130D trinity + meta tail at [130:132]
    which MSL attention reads), so migrated rows MUST keep the same arity or
    old rows (clipped to 130D) would mismatch new writes + drop the meta tail.
    Legacy 9D/67D-era rows are preserved as-is; the consciousness reader
    unpacks `len(blob)//4` floats per row and downstream consumers pad/slice.
    `target_dim` is accepted but ignored (kept for call-site compatibility)."""
    return struct.pack(f"<{len(vals)}f", *(float(v) for v in vals))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="data/consciousness.db",
                   help="source DB path (read-only)")
    p.add_argument("--target", default="data/consciousness.db.migrated",
                   help="target DB path (created)")
    p.add_argument("--apply", action="store_true",
                   help="actually perform migration (default: dry-run)")
    p.add_argument("--batch", type=int, default=10_000,
                   help="rows per batch (default 10k)")
    args = p.parse_args()

    src_path = Path(args.source)
    if not src_path.exists():
        print(f"ERROR: source not found: {src_path}", file=sys.stderr)
        return 2

    src_size = src_path.stat().st_size
    print(f"=== consciousness.db BLOB quantization migration ===")
    print(f"  source: {src_path} ({src_size / 1e9:.2f} GB)")
    print(f"  target: {args.target}")
    print(f"  mode:   {'APPLY' if args.apply else 'DRY-RUN'}")
    print()

    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row

    n_rows = src.execute("SELECT COUNT(*) FROM epochs").fetchone()[0]
    print(f"  total rows: {n_rows:,}")

    # Size projection (independent of dry-run vs apply)
    row_text_avg = src.execute(
        "SELECT AVG(LENGTH(state_vector) + LENGTH(drift_vector) + "
        "LENGTH(trajectory_vector) + LENGTH(distillation) + "
        "LENGTH(COALESCE(anchored_tx,'')) + LENGTH(COALESCE(block_hash,''))) "
        "FROM epochs"
    ).fetchone()[0] or 0
    row_blob_projected = (
        STATE_VECTOR_DIM * 4 +           # f32 state (lossless — was u8, refined 2026-05-25)
        DRIFT_VECTOR_DIM * 4 +           # f32 drift
        TRAJ_VECTOR_DIM * 4 +            # f32 traj
        4 +                               # ~avg distillation (sparse)
        88                                # anchored_tx
    )
    projected_data = n_rows * row_blob_projected
    projected_file_estimate = projected_data * 1.5  # ~50% overhead for indexes/pages
    print(f"  avg row text size:        {row_text_avg:.0f} bytes")
    print(f"  projected row blob size:  {row_blob_projected} bytes")
    print(f"  projected row reduction:  {row_text_avg / row_blob_projected:.1f}x")
    print(f"  projected target db size: ~{projected_file_estimate / 1e9:.2f} GB")
    print(f"                            (~{(src_size - projected_file_estimate) / src_size * 100:.0f}% reduction from {src_size / 1e9:.2f} GB)")
    print()

    if not args.apply:
        # Dry-run: sample 5 rows to demonstrate transform
        print("  === SAMPLE TRANSFORM (5 rows, dry-run) ===")
        for row in src.execute("SELECT * FROM epochs ORDER BY epoch_id LIMIT 5"):
            sv = json.loads(row["state_vector"] or "[]")
            dv = json.loads(row["drift_vector"] or "[]")
            tv = json.loads(row["trajectory_vector"] or "[]")
            sv_blob = pack_f32(sv, STATE_VECTOR_DIM)
            dv_blob = pack_f32(dv, DRIFT_VECTOR_DIM)
            tv_blob = pack_f32(tv, TRAJ_VECTOR_DIM)
            text_size = (len(row["state_vector"] or "") + len(row["drift_vector"] or "")
                         + len(row["trajectory_vector"] or ""))
            blob_size = len(sv_blob) + len(dv_blob) + len(tv_blob)
            print(f"  epoch_id={row['epoch_id']:>9}  text={text_size:>5}B  blob={blob_size:>3}B  ratio={text_size/max(1,blob_size):.1f}x  "
                  f"sv_dim_in={len(sv)}→{STATE_VECTOR_DIM}")
        print()
        print("  DRY-RUN complete. Re-run with --apply to migrate.")
        src.close()
        return 0

    # APPLY MODE — actually migrate
    if Path(args.target).exists():
        print(f"ERROR: target exists, refusing to overwrite: {args.target}", file=sys.stderr)
        return 3

    dst = sqlite3.connect(args.target)
    dst.execute("""
        CREATE TABLE epochs (
            epoch_id          INTEGER PRIMARY KEY,
            timestamp         REAL NOT NULL,
            block_hash        TEXT NOT NULL DEFAULT '',
            state_vector      BLOB NOT NULL,      -- 520B (SPEC §11.H.1.bis, lossless 130D × f32)
            drift_vector      BLOB NOT NULL,      -- 36B
            trajectory_vector BLOB NOT NULL,      -- 36B
            journey_x         REAL NOT NULL,
            journey_y         REAL NOT NULL,
            journey_z         REAL NOT NULL,
            curvature         REAL NOT NULL DEFAULT 0.0,
            density           REAL NOT NULL DEFAULT 0.0,
            distillation      TEXT NOT NULL DEFAULT '',
            anchored_tx       TEXT NOT NULL DEFAULT ''
        )
    """)
    dst.execute("CREATE INDEX idx_epochs_ts ON epochs(timestamp)")

    t0 = time.time()
    written = 0
    batch = []
    for row in src.execute("SELECT * FROM epochs ORDER BY epoch_id"):
        try:
            sv = json.loads(row["state_vector"] or "[]")
            dv = json.loads(row["drift_vector"] or "[]")
            tv = json.loads(row["trajectory_vector"] or "[]")
        except json.JSONDecodeError as e:
            print(f"WARN: row {row['epoch_id']} JSON decode error: {e}; using zeros")
            sv, dv, tv = [], [], []
        batch.append((
            row["epoch_id"], row["timestamp"], row["block_hash"] or "",
            pack_f32(sv, STATE_VECTOR_DIM),
            pack_f32(dv, DRIFT_VECTOR_DIM),
            pack_f32(tv, TRAJ_VECTOR_DIM),
            row["journey_x"], row["journey_y"], row["journey_z"],
            row["curvature"], row["density"],
            row["distillation"] or "", row["anchored_tx"] or "",
        ))
        if len(batch) >= args.batch:
            dst.executemany(
                "INSERT INTO epochs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", batch
            )
            dst.commit()
            written += len(batch)
            elapsed = time.time() - t0
            rate = written / max(1e-6, elapsed)
            eta_s = (n_rows - written) / max(1, rate)
            print(f"  ... {written:>9,}/{n_rows:,}  {rate:>8,.0f} rows/s  eta {eta_s:>5.0f}s")
            batch = []
    if batch:
        dst.executemany(
            "INSERT INTO epochs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", batch
        )
        dst.commit()
        written += len(batch)

    print(f"  ... migrated {written:,} rows in {time.time()-t0:.0f}s")

    print("  VACUUM + ANALYZE ...")
    dst.execute("VACUUM")
    dst.execute("ANALYZE")

    src.close()
    dst.close()

    final_size = Path(args.target).stat().st_size
    print()
    print(f"  === DONE ===")
    print(f"  source: {src_size / 1e9:.2f} GB → target: {final_size / 1e9:.2f} GB")
    print(f"  reduction: {(src_size - final_size) / src_size * 100:.1f}%")
    print()
    print("  NEXT: verify with `sqlite3 {target} 'PRAGMA integrity_check;'`")
    print("        + roundtrip a few rows (quantized BLOB → float → compare to source).")
    print("        Then atomic swap (after Maker greenlight).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
