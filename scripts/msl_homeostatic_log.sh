#!/bin/bash
# msl_homeostatic_log.sh — Hourly snapshot of MSL homeostatic state for
# trend analysis post-foundational-healing-rFP (2026-04-13).
#
# Run via cron every hour:
#   0 * * * * bash /home/antigravity/projects/titan/scripts/msl_homeostatic_log.sh >> /tmp/msl_log_cron.log 2>&1
#
# Output: /tmp/msl_homeostatic_history.tsv (append-only)
# Columns: timestamp_utc | titan | attn_entropy | setpoint_entropy_norm | drift_guard_active_count | update_count
#
# Used to answer: after MSL reset to baseline + foundational fixes, does
# setpoint_entropy stay healthy (>0.95) or drift back toward collapse?
# If it stays healthy → drift_guard + foundation is sufficient.
# If it re-collapses → HOMEO-REDESIGN is genuinely needed.

LOG_FILE="/tmp/msl_homeostatic_history.tsv"
TS_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

# Write header on first run
if [ ! -f "$LOG_FILE" ]; then
    echo -e "timestamp_utc\ttitan\tattn_entropy\tsetpoint_entropy_norm\tdrift_guard_active_count\tupdate_count" > "$LOG_FILE"
fi

# Sample each Titan
for entry in "T1:localhost:7777" "T2:10.135.0.6:7777" "T3:10.135.0.6:7778"; do
    NAME=${entry%%:*}
    HOST=${entry#*:}
    OUT=$(curl -s --max-time 30 "http://${HOST}/v4/inner-trinity" 2>/dev/null)
    if [ -z "$OUT" ]; then
        echo -e "${TS_UTC}\t${NAME}\tERR_TIMEOUT\t-\t-\t-" >> "$LOG_FILE"
        continue
    fi
    # Parse with python (already in venv-friendly path)
    PARSED=$(echo "$OUT" | /home/antigravity/projects/titan/test_env/bin/python -c "
import json, sys
try:
    d = json.load(sys.stdin).get('data', {})
    msl = d.get('msl', {})
    homeo = msl.get('homeostatic', {})
    print(f'{msl.get(\"attention_entropy\",\"-\"):>6}\t{homeo.get(\"setpoint_entropy_normalized\",\"-\"):>6}\t{homeo.get(\"drift_guard_active_count\",\"-\")}\t{homeo.get(\"update_count\",\"-\")}')
except Exception as e:
    print(f'PARSE_ERR\t-\t-\t-')
")
    echo -e "${TS_UTC}\t${NAME}\t${PARSED}" >> "$LOG_FILE"
done
