#!/bin/bash
# arc_cron_train.sh — Periodic ARC training with game rotation
#
# ⚠️  DISABLED FLEET-WIDE since 2026-04-26 (CPU starvation).
#     See memory/project_arc_disabled_cpu.md.
#     Cron entries on T1 + T2 + T3 are commented out and the auto-trigger
#     in services_watchdog.sh CHECK 4 was removed 2026-05-09 (commit
#     e1611d81) after a stale-log auto-launch on T2 drove VPS swap to
#     100% and hung T3's Python child for 22h.
#     Do NOT manually invoke without confirming CPU/swap budget on the
#     target host. ARC training will be revived in a dedicated rFP once
#     the CPU budget is resolved.
#
# Original docstring (kept for reference):
# Run via cron every 3 hours on T1:
#   0 */3 * * * bash /home/antigravity/projects/titan/scripts/arc_cron_train.sh >> /tmp/arc_training.log 2>&1
#
# Host-scoped non-overlap guard (2026-04-20): when T2 and T3 share a host, a
# retry from one Titan could overlap the next cron fire of the other,
# saturating the 4-CPU box. flock -n ensures only one ARC run per host at a
# time — the second invocation exits silently, next cron fires in 3h.

# Hard exit guard — refuse to run silently if accidentally invoked.
if [ "${ARC_FORCE_RUN:-0}" != "1" ]; then
    echo "ERROR: ARC training is disabled fleet-wide since 2026-04-26."
    echo "       To force a manual run after confirming CPU/swap budget:"
    echo "         ARC_FORCE_RUN=1 bash $0 $@"
    exit 1
fi

TITAN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="/tmp/arc_cron_host.lock"

# Re-exec under flock if not already holding the lock.
if [ "${ARC_CRON_LOCKED:-0}" != "1" ]; then
    exec env ARC_CRON_LOCKED=1 flock -n "$LOCK_FILE" "$0" "$@" || {
        NOW=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
        echo "[$NOW] ARC training: another run on this host holds $LOCK_FILE — skipping ($TITAN_DIR)"
        exit 0
    }
fi

cd "$TITAN_DIR" || exit 1
source test_env/bin/activate
export OPENROUTER_API_KEY=

# Self-identification for cross-Titan goal broadcast (rFP Step C, 2026-04-20).
# T1=10.135.0.3, T2/T3 share 10.135.0.6 (T3 dir = titan3/). Best-effort — if
# detection fails, falls back to "unknown" which kin will ignore.
if [[ "$TITAN_DIR" == */titan3 ]]; then
    export TITAN_KIN_SOURCE="T3"
elif hostname -I 2>/dev/null | grep -q "10.135.0.3"; then
    export TITAN_KIN_SOURCE="T1"
else
    export TITAN_KIN_SOURCE="T2"
fi

# 4/2/2 weighted rotation (2026-04-20): ls20 gets 4 of 8 daily slots, ft09 and
# vc33 get 2 each. Drop internal --cycle so each run is pure single-game. Slot
# index from hour (T2/T3/T1 all fire every 3h, different :00/:30). Same
# slot-to-game mapping across all Titans for consistency.
HOUR=$(date -u +%H)
SLOT=$(( HOUR / 3 ))
case $SLOT in
    0|2|4|6) GAME="ls20" ;;
    1|5)     GAME="ft09" ;;
    3|7)     GAME="vc33" ;;
    *)       GAME="ls20" ;;
esac

NOW=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
echo "[$NOW] ARC training: game=$GAME (slot=$SLOT from hour=$HOUR)  dir=$TITAN_DIR"

# Check if titan_main is running (use API health check, not just process check)
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://127.0.0.1:7777/health" 2>/dev/null || echo "000")
if [ "$API_STATUS" != "200" ]; then
    echo "[$NOW] Titan API not healthy (HTTP $API_STATUS) — skipping ARC training"
    exit 0
fi

# 2026-04-20: episodes 50 → 200 (give Titans a real chance at first win, per
# rFP_arc_training_fix.md iter-3). Drop --cycle (was inflating primary to
# ~94%/secondary ~3%/3% via 2-ep breaks — we now rotate at cron level). Timeout
# bumped 2700 → 4500 (75min). Expected wall time 200 ls20 eps × ~14s ≈ 47min;
# 75min timeout gives 60% safety margin. MAX_RETRIES dropped 1 → 0 because
# flock + timeout already bound things; a retry could push a run past the
# 90-min T2/T3 gap and flock-skip the next Titan's slot entirely.
#
# 2026-04-21: shared-host budget. T2/T3 share a 4-vCPU box; an active
# arc_competition.py + 2 running Titans pushed load avg to ~14, starving
# media-module heartbeats and triggering Guardian restart loops every 3 min.
# T1 keeps full 200 eps + 4500s (own VPS, no contention). Shared-host runs
# (T2/T3) get half: 100 eps + 2700s. Halves the sustained-CPU window per
# slot from ~47 min to ~24 min, well inside media's 180s heartbeat tolerance
# AND inside the 90-min cron gap to the other Titan's next slot.
if [ "${TITAN_KIN_SOURCE:-}" = "T1" ]; then
    EPISODES=200
    TIMEOUT=4500
else
    EPISODES=100
    TIMEOUT=2700
fi

timeout "$TIMEOUT" python scripts/arc_competition.py \
    --game "$GAME" \
    --episodes "$EPISODES" \
    --save-results \
    --force \
    --reasoning \
    2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$NOW] ARC training complete: game=$GAME ($EPISODES episodes)"
    exit 0
fi
echo "[$NOW] ARC training FAILED (exit=$EXIT_CODE) game=$GAME — no retry; next cron in 3h"
exit 1
