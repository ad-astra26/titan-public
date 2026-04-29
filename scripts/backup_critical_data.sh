#!/bin/bash
# backup_critical_data.sh — operator-discretion backup of at-risk Titan data.
#
# 2026-04-27 PM (T2-shadow-swap-fix session): created during the recovery
# from yesterday's shadow-swap data corruption (timechain/index.db).
# Use this BEFORE any risky operation (shadow swap, large migration,
# manual surgery) to give yourself an instant restore point.
#
# Strategy:
#   - SQLite/DuckDB: real copy (NOT hardlink — those are the corruption
#     vector we're protecting against). Cost: ~5-10 GB on T1.
#   - Top-level critical JSONs: real copy (cheap).
#   - Skips the rest of data/ — most files use atomic-rename which is
#     already corruption-safe + would multiply backup cost.
#
# Default destination: data.backup.<UTC_timestamp>/ alongside data/.
# Override with: bash scripts/backup_critical_data.sh /path/to/dst/
#
# Restore:
#   1. Stop T1: bash scripts/safe_restart.sh t1 (then Ctrl-C after STOP)
#      OR for T1 specifically, use the kill+wait pattern in the script.
#   2. Replace each affected file from the backup dir:
#      cp -a data.backup.<ts>/timechain/index.db data/timechain/index.db
#   3. Start T1: bash scripts/safe_restart.sh t1
#
# This script is fast (~30s for ~10GB on local SSD) and idempotent (each
# invocation creates a NEW timestamped backup; old backups are untouched).

set -e

PROJECT_DIR="${PROJECT_DIR:-/home/antigravity/projects/titan}"
SRC="${PROJECT_DIR}/data"

if [ ! -d "$SRC" ]; then
    echo "ERROR: $SRC not found" >&2
    exit 1
fi

if [ -n "$1" ]; then
    DST="$1"
else
    TS=$(date -u '+%Y%m%d_%H%M%S')
    DST="${PROJECT_DIR}/data.backup.${TS}"
fi

if [ -e "$DST" ]; then
    echo "ERROR: destination $DST already exists — refusing to overwrite" >&2
    exit 1
fi

echo "=== Backing up critical Titan data ==="
echo "  src: $SRC"
echo "  dst: $DST"
echo

mkdir -p "$DST"
mkdir -p "$DST/timechain"
mkdir -p "$DST/run"

# 1. Top-level SQLite + DuckDB (real copy — break hardlink chain)
echo "[1/4] SQLite + DuckDB files (top-level)..."
SQLITE_COUNT=0
for f in "$SRC"/*.db "$SRC"/*.duckdb; do
    if [ -f "$f" ]; then
        # Real copy (NOT cp -al; we want a sole-link backup)
        cp -a "$f" "$DST/"
        SQLITE_COUNT=$((SQLITE_COUNT + 1))
    fi
done
# Also -wal and -shm sidecars (SQLite WAL state)
for f in "$SRC"/*.db-wal "$SRC"/*.db-shm; do
    if [ -f "$f" ]; then
        cp -a "$f" "$DST/"
    fi
done
echo "  copied $SQLITE_COUNT DB files"

# 2. timechain — chain_*.bin files are source of truth, index.db is rebuildable
echo "[2/4] timechain/ (chain_*.bin + index.db)..."
TC_COUNT=0
for f in "$SRC"/timechain/chain_*.bin "$SRC"/timechain/index.db \
         "$SRC"/timechain/index.db-wal "$SRC"/timechain/index.db-shm \
         "$SRC"/timechain/contract_stats.json \
         "$SRC"/timechain/.birth_block_created \
         "$SRC"/timechain/arweave_manifest_T1.json; do
    if [ -f "$f" ]; then
        cp -a "$f" "$DST/timechain/"
        TC_COUNT=$((TC_COUNT + 1))
    fi
done
# sidechains/ subdir
if [ -d "$SRC/timechain/sidechains" ]; then
    cp -a "$SRC/timechain/sidechains" "$DST/timechain/"
fi
echo "  copied $TC_COUNT timechain files"

# 3. Critical state JSONs at top-level
echo "[3/4] Critical state JSONs..."
JSON_COUNT=0
for jsonfile in \
    dreaming_state.json \
    birth_dna_snapshot.json \
    unified_spirit_state.json \
    backup_state.json \
    neuromodulator_state.json \
    filter_down_weights.json \
    filter_down_v5_buffer.json \
    titan_identity_keypair.json \
    reasoning_totals.json \
    chi_state.json \
    consciousness_v6_state.json \
    expression_state.json \
    msl_state.json; do
    f="$SRC/$jsonfile"
    if [ -f "$f" ]; then
        cp -a "$f" "$DST/"
        JSON_COUNT=$((JSON_COUNT + 1))
    fi
done
echo "  copied $JSON_COUNT critical JSON files"

# 4. Worker metric files (run/*.json) — small, useful for forensics
echo "[4/4] run/ metric files..."
RUN_COUNT=0
if [ -d "$SRC/run" ]; then
    for f in "$SRC"/run/*.json; do
        if [ -f "$f" ]; then
            cp -a "$f" "$DST/run/"
            RUN_COUNT=$((RUN_COUNT + 1))
        fi
    done
fi
echo "  copied $RUN_COUNT metric files"

# Manifest
DST_SIZE=$(du -sh "$DST" 2>/dev/null | awk '{print $1}')
cat > "$DST/MANIFEST.txt" <<EOF
Titan critical data backup
==========================
Created: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
Source:  $SRC
Backup:  $DST
Size:    $DST_SIZE

Contents:
  - Top-level *.db / *.duckdb / *.db-wal / *.db-shm  ($SQLITE_COUNT files)
  - timechain/ chain_*.bin + index.db + sidechains/  ($TC_COUNT files)
  - Critical state JSONs (dreaming, neuromods, ...) ($JSON_COUNT files)
  - run/ metric files                               ($RUN_COUNT files)

Restore:
  1. Stop T1 (bash scripts/safe_restart.sh t1, Ctrl-C after STOP phase)
  2. Replace affected file(s) from this backup dir
  3. Restart T1 (bash scripts/safe_restart.sh t1)

Created by: scripts/backup_critical_data.sh
Trigger:    operator discretion (no scheduling)
EOF

echo
echo "=== Backup complete ==="
echo "  Total: $DST_SIZE  →  $DST"
echo "  Manifest: $DST/MANIFEST.txt"
exit 0
