#!/bin/bash
# Overnight Phase Runner — Runs Phases 3-6 sequentially with 30-min rests
# Usage: nohup bash scripts/overnight_phase_runner.sh titan1 > /tmp/overnight_t1.log 2>&1 &
#        nohup bash scripts/overnight_phase_runner.sh titan2 > /tmp/overnight_t2.log 2>&1 &

set -e

INSTANCE="${1:-titan1}"
REST_SECONDS=1800  # 30 minutes between phases
PROJECT_DIR="/home/antigravity/projects/titan"

cd "$PROJECT_DIR"
source test_env/bin/activate

if [ "$INSTANCE" = "titan2" ]; then
    export TITAN_API_BASE="http://10.135.0.6:7777"
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Overnight Phase Runner — Instance: $INSTANCE"
echo "  Started: $(date)"
echo "  API: ${TITAN_API_BASE:-http://localhost:7777}"
echo "  Rest between phases: ${REST_SECONDS}s (30 min)"
echo "═══════════════════════════════════════════════════════════"

# Wait for Phase 2 to complete (check every 60s)
echo "[$(date +%H:%M:%S)] Waiting for Phase 2 to complete..."
while pgrep -f "autonomous_language_pipeline.*--phase 2.*--instance $INSTANCE" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] Phase 2 complete (or not running). Proceeding."

# ── Phase 3: Composition + Grammar Learning ──
echo ""
echo "[$(date +%H:%M:%S)] ═══ Starting Phase 3: Composition + Grammar Learning ═══"
python -u scripts/autonomous_language_pipeline.py --phase 3 --instance "$INSTANCE" 2>&1
echo "[$(date +%H:%M:%S)] Phase 3 complete. Resting ${REST_SECONDS}s..."
sleep $REST_SECONDS

# ── Phase 4: Autonomous Expression ──
echo ""
echo "[$(date +%H:%M:%S)] ═══ Starting Phase 4: Autonomous Expression ═══"
python -u scripts/autonomous_language_pipeline.py --phase 4 --instance "$INSTANCE" 2>&1
echo "[$(date +%H:%M:%S)] Phase 4 complete. Resting ${REST_SECONDS}s..."
sleep $REST_SECONDS

# ── Phase 5: Dialogue Test ──
echo ""
echo "[$(date +%H:%M:%S)] ═══ Starting Phase 5: Simulated Dialogue Test ═══"
python -u scripts/autonomous_language_pipeline.py --phase 5 --instance "$INSTANCE" 2>&1
echo "[$(date +%H:%M:%S)] Phase 5 complete. Resting ${REST_SECONDS}s..."
sleep $REST_SECONDS

# ── Phase 6: Narrative Test ──
echo ""
echo "[$(date +%H:%M:%S)] ═══ Starting Phase 6: Narrative Composition Test ═══"
python -u scripts/autonomous_language_pipeline.py --phase 6 --instance "$INSTANCE" 2>&1
echo "[$(date +%H:%M:%S)] Phase 6 complete."

# ── Summary ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Overnight Run Complete — Instance: $INSTANCE"
echo "  Finished: $(date)"
echo "  Reports saved to: titan-docs/REPORT_language_pipeline_*.md"
echo "  Telemetry in: data/language_pipeline_*.json"
echo "═══════════════════════════════════════════════════════════"
