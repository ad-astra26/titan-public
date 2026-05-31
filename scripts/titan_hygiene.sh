#!/usr/bin/env bash
# titan_hygiene.sh — daily disk hygiene. Turns the recurring MANUAL cleanup
# (memory project-t1-mainnet-disk-recurring-fill: "Maker keeps forgetting") into
# a scheduled job. Disjoint from titan-cargo-sweep (that owns Rust target/); this
# owns /tmp transients + data/backups pruning + migration leftovers.
#
# Safe by construction: only -mtime-guarded transient artifacts and a
# keep-newest-N prune of data/backups that NEVER touches the canonical baseline,
# baseline_active, or archives (directive_memory_preservation). Runs on T1; the
# same script + unit is mirrored to T2/T3 (same /tmp churn).
set -euo pipefail
REPO="/home/antigravity/projects/titan"
[ -d "$REPO" ] || REPO="$(cd "$(dirname "$0")/.." && pwd)"   # T3 path = titan3/
cd "$REPO"
echo "[hygiene] $(date -u +%FT%TZ) start ($REPO)"

# 1) /tmp transient backup/export/migration artifacts older than 3 days.
#    (These are written and consumed within minutes; 3d is a wide safety margin.)
find /tmp -maxdepth 1 -mtime +3 \( \
      -name 'backup_dry_run_*.tar.gz' -o \
      -name 'titan_personality_arweave_*.tar.gz' -o \
      -name 'titan_unified_*.tar.gz' -o \
      -name 'titan_soul_*.tar.gz' -o \
      -name 'tmp*.bin' -o \
      -name 'tmp*.db' -o \
      -name 'titan_brain.log.*.gz' \) -delete 2>/dev/null || true

# 2) data/backups: keep the newest 10 personality_*.tar.gz; prune older.
#    Explicitly NEVER touches unified_baseline_T1 / baseline_active / archives.
if [ -d data/backups ]; then
    ls -t data/backups/personality_*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm -f || true
fi

# 3) consciousness.db migration leftovers (post-verify), older than 7 days.
find data -maxdepth 1 -mtime +7 \( \
      -name 'consciousness.db.legacy_*' -o \
      -name 'consciousness.db.f32_*' \) -delete 2>/dev/null || true

echo "[hygiene] $(date -u +%FT%TZ) done — $(df -h / | awk 'NR==2{print $5" used, "$4" free"}')"
