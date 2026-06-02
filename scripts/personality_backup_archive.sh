#!/bin/bash
# personality_backup_archive.sh — gzip-9 monthly archive of old personality backups.
#
# Per Maker directive 2026-04-30: keep last 7 days of personality_*.tar.gz files
# loose; archive older into monthly tarballs (gzip-9, matches Arweave pipeline
# compression). Frees inode pressure + reduces backup size.
#
# SAFETY (per directive_memory_preservation.md):
#   - Files NEVER deleted before archive verified (tar -tzf count == input count)
#   - Skips files actively held open by any process (lsof check)
#   - Skips current month's files (preserves rolling-window safety)
#   - Idempotent: skips year-month buckets that already have an archive
#
# Maker action: run manually OR schedule via cron (suggest weekly):
#   crontab -e
#   0 5 * * 0 bash /home/youruser/projects/titan/scripts/personality_backup_archive.sh \
#     >> /tmp/personality_archive.log 2>&1
#
# Closes session 2026-04-30 task #7. Companion script to studio_exports archive.

set -euo pipefail

BACKUPS_DIR=/home/youruser/projects/titan/data/backups
ARCHIVE_DIR="$BACKUPS_DIR/archives"
RETENTION_DAYS=7
LOG=/tmp/personality_archive_$(date -u +%Y%m%d_%H%M%S).log

mkdir -p "$ARCHIVE_DIR"

echo "=== personality backup archive run: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$LOG"
echo "Backups dir: $BACKUPS_DIR" | tee -a "$LOG"
echo "Archive dir: $ARCHIVE_DIR" | tee -a "$LOG"
echo "Retention: $RETENTION_DAYS days loose" | tee -a "$LOG"
echo "" | tee -a "$LOG"

cd "$BACKUPS_DIR"

# Step 1: enumerate eligible files (older than RETENTION_DAYS, no live open handle)
EFILE=/tmp/_personality_archive_eligible.txt
rm -f "$EFILE"

find . -maxdepth 1 -name "personality_*.tar.gz" -mtime +$RETENTION_DAYS \
    -printf "%T@ %f\n" 2>/dev/null > "$EFILE"

TOTAL=$(wc -l < "$EFILE")
echo "Files older than ${RETENTION_DAYS}d: $TOTAL" | tee -a "$LOG"

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to archive. Exiting." | tee -a "$LOG"
    rm -f "$EFILE"
    exit 0
fi

# Step 2: verify no eligible file is currently held open
OPEN_COUNT=$(lsof 2>/dev/null | awk -v dir="$BACKUPS_DIR" '$0 ~ dir"/personality_"' | wc -l)
if [ "$OPEN_COUNT" -gt 0 ]; then
    echo "WARN: $OPEN_COUNT personality_* files held open by live processes; aborting for safety." | tee -a "$LOG"
    lsof 2>/dev/null | grep "$BACKUPS_DIR/personality_" | tee -a "$LOG"
    rm -f "$EFILE"
    exit 1
fi

# Step 3: bucket by year-month (UTC) from mtime
declare -A buckets
while IFS= read -r line; do
    ts="${line%% *}"
    fname="${line#* }"
    yyyymm=$(date -u -d @"${ts%.*}" +%Y%m)
    buckets[$yyyymm]+="$fname"$'\n'
done < "$EFILE"

# Step 4: process each bucket
for ym in "${!buckets[@]}"; do
    archive_path="$ARCHIVE_DIR/personality_archive_${ym}.tar.gz"
    file_list="$ARCHIVE_DIR/.personality_${ym}_files.txt"

    # Idempotent: skip if archive already exists
    if [ -f "$archive_path" ]; then
        echo "SKIP $ym: archive already exists at $archive_path" | tee -a "$LOG"
        continue
    fi

    # Write file list
    echo "${buckets[$ym]}" | grep -v '^$' > "$file_list"
    count=$(wc -l < "$file_list")
    echo "Processing $ym: $count files..." | tee -a "$LOG"

    # Create tarball with gzip-9 (max compression)
    if tar --use-compress-program="gzip -9" \
           -cf "$archive_path" \
           --files-from="$file_list" 2>>"$LOG"; then
        archive_size=$(stat -c%s "$archive_path")
        echo "  archive created: $(numfmt --to=iec $archive_size)" | tee -a "$LOG"

        # Verify: list contents must match input count
        verify_count=$(tar -tzf "$archive_path" 2>/dev/null | wc -l)
        if [ "$verify_count" -eq "$count" ]; then
            echo "  ✓ verified ($verify_count == $count) — removing originals" | tee -a "$LOG"
            xargs -a "$file_list" rm -f
            echo "  ✓ originals removed" | tee -a "$LOG"
        else
            echo "  ✗ VERIFY FAILED (in_archive=$verify_count expected=$count) — archive + originals kept for inspection" | tee -a "$LOG"
        fi
    else
        echo "  ✗ tar FAILED for $ym — removing partial archive, originals untouched" | tee -a "$LOG"
        rm -f "$archive_path"
    fi
done

rm -f "$EFILE"

echo "" | tee -a "$LOG"
echo "=== Archive run complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
df -h / | head -2 | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Archives in $ARCHIVE_DIR:" | tee -a "$LOG"
ls -lh "$ARCHIVE_DIR/" 2>/dev/null | tail -10 | tee -a "$LOG"
