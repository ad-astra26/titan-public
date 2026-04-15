#!/bin/bash
# arc_cron_train.sh — Periodic ARC training with game rotation
# Run via cron every 3 hours on T1:
#   0 */3 * * * bash /home/antigravity/projects/titan/scripts/arc_cron_train.sh >> /tmp/arc_training.log 2>&1

TITAN_DIR="/home/antigravity/projects/titan"
cd "$TITAN_DIR" || exit 1
source test_env/bin/activate
export OPENROUTER_API_KEY=

# Rotate primary game based on hour (0,3,6... = ls20; 1,4,7... = ft09; 2,5,8... = vc33)
HOUR=$(date -u +%H)
GAME_IDX=$(( (HOUR / 3) % 3 ))
GAMES=("ls20" "ft09" "vc33")
GAME="${GAMES[$GAME_IDX]}"

NOW=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
echo "[$NOW] ARC training: game=$GAME (idx=$GAME_IDX from hour=$HOUR)"

# Check if titan_main is running (use API health check, not just process check)
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://127.0.0.1:7777/health" 2>/dev/null || echo "000")
if [ "$API_STATUS" != "200" ]; then
    echo "[$NOW] Titan API not healthy (HTTP $API_STATUS) — skipping ARC training"
    exit 0
fi

# Run 50 episodes (primary) + cycling breaks, reasoning, save results.
# 2026-04-15: bumped 10 → 50 after Phase A reward rebalance exposed exploration
# lift but scorer convergence needs more data — 10 eps/cron × 8 crons/day × ⅓
# rotation = ~25 ls20/day (too slow). 50 eps/cron gives ~130 ls20/day.
# Timeout bumped 1800 → 2700 to accommodate larger session (50 eps × 3 games
# × cycle-break ≈ 150 max-step episodes × ~14s = ~35 min worst case).
# Retry once on failure after 60s wait
MAX_RETRIES=1
RETRY=0
while [ $RETRY -le $MAX_RETRIES ]; do
    timeout 2700 python scripts/arc_competition.py \
        --game "$GAME" \
        --episodes 50 \
        --cycle \
        --cycle-break 2 \
        --save-results \
        --force \
        --reasoning \
        2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$NOW] ARC training complete: game=$GAME"
        exit 0
    fi

    if [ $RETRY -lt $MAX_RETRIES ]; then
        echo "[$NOW] ARC training FAILED (exit=$EXIT_CODE) — retrying in 60s (attempt $((RETRY+2))/$((MAX_RETRIES+1)))"
        sleep 60
    else
        echo "[$NOW] ARC training FAILED (exit=$EXIT_CODE) after $((MAX_RETRIES+1)) attempts: game=$GAME"
    fi
    RETRY=$((RETRY + 1))
done
exit 1
