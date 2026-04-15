#!/bin/bash
# safe_restart.sh — Check dreaming state before restarting Titan
# Usage: bash scripts/safe_restart.sh [t1|t2] [--force]
#
# Checks if Titan is dreaming before restart. If sleeping:
#   - Without --force: refuses to restart, shows warning
#   - With --force: warns but proceeds

TARGET="${1:-t1}"
FORCE="${2}"

check_dreaming_t1() {
    local state=$(python3 -c "
import json
try:
    with open('data/dreaming_state.json') as f:
        d = json.load(f)
    # Also check API if available
    import urllib.request
    r = urllib.request.urlopen('http://localhost:7777/v4/inner-trinity', timeout=3)
    api = json.loads(r.read())
    is_dreaming = api.get('dreaming', {}).get('is_dreaming', False)
    drain = d.get('metabolic_drain', 0)
    cycles = d.get('cycle_count', 0)
    print(f'dreaming={is_dreaming} drain={drain:.4f} cycles={cycles}')
except Exception:
    # Fallback to file only
    try:
        with open('data/dreaming_state.json') as f:
            d = json.load(f)
        drain = d.get('metabolic_drain', 0)
        epochs = d.get('epochs_since_dream', 999)
        print(f'dreaming=unknown drain={drain:.4f} epochs_since={epochs}')
    except:
        print('dreaming=unknown drain=0 status=no_state_file')
" 2>/dev/null)
    echo "$state"
}

check_dreaming_t2() {
    local state=$(ssh -o ConnectTimeout=3 root@10.135.0.6 "python3 -c \"
import json
try:
    with open('/home/antigravity/projects/titan/data/dreaming_state.json') as f:
        d = json.load(f)
    import urllib.request
    r = urllib.request.urlopen('http://localhost:7777/v4/inner-trinity', timeout=3)
    api = json.loads(r.read())
    is_dreaming = api.get('dreaming', {}).get('is_dreaming', False)
    drain = d.get('metabolic_drain', 0)
    cycles = d.get('cycle_count', 0)
    print(f'dreaming={is_dreaming} drain={drain:.4f} cycles={cycles}')
except Exception:
    try:
        with open('/home/antigravity/projects/titan/data/dreaming_state.json') as f:
            d = json.load(f)
        drain = d.get('metabolic_drain', 0)
        epochs = d.get('epochs_since_dream', 999)
        print(f'dreaming=unknown drain={drain:.4f} epochs_since={epochs}')
    except:
        print('dreaming=unknown drain=0 status=no_state_file')
\"" 2>/dev/null)
    echo "$state"
}

echo "=== Checking ${TARGET} dreaming state before restart ==="

if [[ "$TARGET" == "t1" ]]; then
    STATE=$(check_dreaming_t1)
elif [[ "$TARGET" == "t2" ]]; then
    STATE=$(check_dreaming_t2)
else
    echo "Usage: bash scripts/safe_restart.sh [t1|t2] [--force]"
    exit 1
fi

echo "  State: $STATE"

if echo "$STATE" | grep -q "dreaming=True"; then
    echo ""
    echo "  ⚠  TITAN IS DREAMING — restart would disrupt sleep cycle!"
    echo "     Restarting during dreams causes neuromod imbalance (DA/NE suppressed, GABA elevated)"
    echo "     and may trigger a dream loop on reboot."
    if [[ "$FORCE" == "--force" ]]; then
        echo ""
        echo "  --force specified, proceeding with restart despite dreaming state..."
    else
        echo ""
        echo "  Refusing to restart. Options:"
        echo "    1. Wait for dream to end (check: curl -s localhost:7777/v4/inner-trinity | jq .dreaming)"
        echo "    2. Override: bash scripts/safe_restart.sh ${TARGET} --force"
        exit 1
    fi
else
    echo "  ✓ Titan is awake — safe to restart"
fi

# Perform the restart
if [[ "$TARGET" == "t1" ]]; then
    echo ""
    echo "=== Restarting T1 ==="
    ps aux | grep titan_main | grep python | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
    sleep 3
    source test_env/bin/activate
    # Append (>>) so pre-restart forensics are preserved. Size is bounded by
    # t1_watchdog.sh log rotation (>100MB → gz archive).
    echo "" >> /tmp/titan_brain.log
    echo "=== [$(date -u '+%Y-%m-%d %H:%M:%S UTC')] RESTART boundary — safe_restart.sh ===" >> /tmp/titan_brain.log
    OPENROUTER_API_KEY='' nohup python -u scripts/titan_main.py --server >> /tmp/titan_brain.log 2>&1 &
    NEW_PID=$!
    echo "$NEW_PID" > /tmp/titan1.pid   # keep watchdog in sync; otherwise it kills us on next cron tick
    echo "T1 restarted (PID=$NEW_PID)"
elif [[ "$TARGET" == "t2" ]]; then
    echo ""
    echo "=== Restarting T2 ==="
    ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh restart"
fi
