#!/bin/bash
# Auto-commit telemetry data to git every hour + ensure telemetry is running
# Cron: 0 * * * * /home/antigravity/scripts/telemetry_autocommit.sh
#
# 1. Health check: restart twin_telemetry if not running (prevents data gaps)
# 2. Commits: telemetry JSON, phase status, neuromodulator state, pipeline reports
# Pushes to remote for safety

TITAN_DIR="/home/antigravity/projects/titan"
PYTHON="${TITAN_DIR}/test_env/bin/python"
T2_HOST="root@10.135.0.6"

cd "$TITAN_DIR"

# ── Telemetry health check ──────────────────────────────────────
# If twin_telemetry.py is not running AND at least one Titan is alive, start it.
# Also restarts if telemetry log hasn't been updated in 5+ minutes (hung process).
TELEMETRY_RUNNING=0
if pgrep -f "twin_telemetry.py" > /dev/null 2>&1; then
    TELEMETRY_RUNNING=1
    # Check if it's actually working (log updated in last 5 min)
    if [ -f /tmp/twin_telemetry.log ]; then
        LAST_MOD=$(stat -c %Y /tmp/twin_telemetry.log 2>/dev/null || echo 0)
        NOW=$(date +%s)
        AGE=$(( NOW - LAST_MOD ))
        if [ "$AGE" -gt 300 ]; then
            echo "[$(date +%H:%M)] Telemetry hung (log ${AGE}s old) — killing and restarting"
            pkill -f "twin_telemetry.py" 2>/dev/null
            sleep 2
            TELEMETRY_RUNNING=0
        fi
    fi
fi

if [ "$TELEMETRY_RUNNING" -eq 0 ]; then
    if pgrep -f "titan_main" > /dev/null 2>&1; then
        nohup "$PYTHON" scripts/twin_telemetry.py --duration 600 > /tmp/twin_telemetry.log 2>&1 &
        echo "[$(date +%H:%M)] Telemetry auto-started (PID: $!)"
    fi
fi

# ── Learning TestSuite RETIRED ──────────────────────────────────
# The external learning_testsuite.py has been replaced by the internal
# language_worker module (CGN-connected, Phase 5 word consolidation).
# Kill any remaining testsuite processes from previous sessions.
if pgrep -f "learning_testsuite" > /dev/null 2>&1; then
    echo "[$(date +%H:%M)] Stopping retired learning_testsuite processes"
    pkill -f "learning_testsuite" 2>/dev/null
fi

# ── Auto-parse latest conversation JSONL ──────────────────────
"$PYTHON" scripts/parse_session_conversation.py --auto 2>/dev/null

# Collect telemetry files that changed
CHANGED=0

# Twin telemetry snapshots — DO NOT COMMIT (2026-04-14 incident fix).
# These files are operational observability: keeping them local-only on T1
# prevents two documented failure modes:
#   1. The original .gitignore had `data/` but these files were already
#      tracked — each hourly `git add -f` here defeated the ignore and kept
#      pushing them to remote.
#   2. T2/T3 pulls then re-populated their data/ with 2000+ files (~19 GB
#      on 2026-04-14), filling their shared disk to 100% and causing FAISS
#      0-byte corruption → memory worker crash loop → T2+T3 outage.
# Local retention (500 files) is now enforced inside twin_telemetry.py.
# To analyse a specific snapshot, ssh to T1 and read the file directly.
#
# Per-Titan state files (pi_heartbeat, neuromodulator, hormonal,
# phase_status) were already not committed for the same reason.

# Pipeline reports
for f in titan-docs/REPORT_language_pipeline_*.md; do
    [ -f "$f" ] && git add -f "$f" 2>/dev/null && CHANGED=1
done

# Pipeline telemetry
for f in data/language_pipeline_*.json; do
    [ -f "$f" ] && git add -f "$f" 2>/dev/null && CHANGED=1
done

# Learning TestSuite checkpoints
for f in data/testsuite_checkpoint_*.json; do
    [ -f "$f" ] && git add -f "$f" 2>/dev/null && CHANGED=1
done

# Conversation transcripts (auto-parsed from JSONL)
for f in titan-docs/conversations/CONVERSATION_*.md; do
    [ -f "$f" ] && git add -f "$f" 2>/dev/null && CHANGED=1
done

# Only commit if there are staged changes
if [ "$CHANGED" -eq 1 ] && ! git diff --cached --quiet 2>/dev/null; then
    TIMESTAMP=$(date +%Y%m%d_%H%M)
    EPOCH_T1=$(grep "Epoch.*complete" /tmp/titan_brain.log 2>/dev/null | tail -1 | grep -oP 'Epoch \K\d+' || echo "?")
    EMOTION=$(grep "emotion=" /tmp/titan_brain.log 2>/dev/null | tail -1 | grep -oP 'emotion=\K[^(]+' || echo "?")

    git commit -m "data: Telemetry snapshot ${TIMESTAMP} — T1 epoch ${EPOCH_T1}, emotion=${EMOTION}" --no-gpg-sign 2>/dev/null

    echo "[${TIMESTAMP}] Telemetry committed: epoch=${EPOCH_T1} emotion=${EMOTION}"
else
    echo "[$(date +%H:%M)] No telemetry changes to commit"
fi
