#!/bin/bash
# studio_exports_archive.sh — gzip-9 monthly archive of old studio export files.
#
# Per Maker directive 2026-04-30: keep last 30 days of studio_exports loose;
# tar+gzip-9 older files into monthly buckets to drop inode pressure.
# Studio exports are creative artifacts (jpg/wav/etc) — NEVER deleted before
# archive verified. Per directive_memory_preservation.md.
#
# Maker action: run manually OR schedule via cron (suggest weekly):
#   crontab -e
#   30 5 * * 0 bash /home/antigravity/projects/titan/scripts/studio_exports_archive.sh \
#     >> /tmp/studio_archive.log 2>&1
#
# Closes session 2026-04-30 task #7. Companion to personality_backup_archive.sh.

set -euo pipefail

STUDIO_DIR=/home/antigravity/projects/titan/data/studio_exports
ARCHIVE_DIR="$STUDIO_DIR/archives"
RETENTION_DAYS=30
LOG=/tmp/studio_archive_$(date -u +%Y%m%d_%H%M%S).log

mkdir -p "$ARCHIVE_DIR"

echo "=== studio_exports archive run: $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$LOG"
echo "Studio dir: $STUDIO_DIR" | tee -a "$LOG"
echo "Retention: $RETENTION_DAYS days loose" | tee -a "$LOG"
echo "" | tee -a "$LOG"

cd "$STUDIO_DIR"

# Step 1: enumerate top-level files older than RETENTION_DAYS (subdirs untouched)
EFILE=/tmp/_studio_archive_eligible.txt
rm -f "$EFILE"

find . -maxdepth 1 -type f -mtime +$RETENTION_DAYS \
    -printf "%T@ %f\n" 2>/dev/null > "$EFILE"

TOTAL=$(wc -l < "$EFILE")
echo "Top-level files older than ${RETENTION_DAYS}d: $TOTAL" | tee -a "$LOG"

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to archive. Exiting." | tee -a "$LOG"
    rm -f "$EFILE"
    exit 0
fi

# Step 2: bucket by year-month (UTC) from mtime
declare -A buckets
while IFS= read -r line; do
    ts="${line%% *}"
    fname="${line#* }"
    yyyymm=$(date -u -d @"${ts%.*}" +%Y%m)
    buckets[$yyyymm]+="$fname"$'\n'
done < "$EFILE"

# Step 3: process each bucket
for ym in "${!buckets[@]}"; do
    archive_path="$ARCHIVE_DIR/studio_archive_${ym}.tar.gz"
    file_list="$ARCHIVE_DIR/.studio_${ym}_files.txt"

    # Idempotent: skip if archive already exists
    if [ -f "$archive_path" ]; then
        echo "SKIP $ym: archive already exists at $archive_path" | tee -a "$LOG"
        continue
    fi

    echo "${buckets[$ym]}" | grep -v '^$' > "$file_list"
    count=$(wc -l < "$file_list")
    echo "Processing $ym: $count files..." | tee -a "$LOG"

    if tar --use-compress-program="gzip -9" \
           -cf "$archive_path" \
           --files-from="$file_list" 2>>"$LOG"; then
        archive_size=$(stat -c%s "$archive_path")
        echo "  archive created: $(numfmt --to=iec $archive_size)" | tee -a "$LOG"

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
