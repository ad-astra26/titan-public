#!/bin/bash
# t2_manage.sh — Titan2 VPS management script
# Lives ON T2 at /home/antigravity/projects/titan/scripts/t2_manage.sh
# All commands run with proper cd + venv activation.
#
# Usage from T1:
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh status"
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh stop"
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh start"
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh restart"
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh health"
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh log"
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh log-errors"
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh wallet"
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh vocab"
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh pyexec 'print(1+1)'"
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh connections"

TITAN_DIR="/home/antigravity/projects/titan"
VENV="${TITAN_DIR}/test_env/bin/activate"
BRAIN_LOG="/tmp/titan2_brain.log"
WALLET="/home/antigravity/.config/solana/id.json"
PIDFILE="/tmp/titan2.pid"

cd "$TITAN_DIR" || { echo "ERROR: Cannot cd to $TITAN_DIR"; exit 1; }
source "$VENV" 2>/dev/null || { echo "ERROR: Cannot activate venv at $VENV"; exit 1; }
export OPENROUTER_API_KEY=
# Match T1's ulimit (VS Code raises it; SSH/cron default is 1024 which causes fd exhaustion)
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true
# T2 connects to T1 via nginx reverse proxy (VPC port 7777 is blocked)
export TITAN_KIN_ADDRESSES="https://iamtitan.tech"

# Helper: check if T2 is actually running via PID file
t2_is_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null
}

CMD="${1:-status}"
shift 2>/dev/null || true

case "$CMD" in
    status)
        echo "=== T2 Process ==="
        if t2_is_running; then
            echo "RUNNING (PID=$(cat "$PIDFILE"))"
        else
            echo "NOT RUNNING"
            rm -f "$PIDFILE"
        fi
        echo ""
        echo "=== API Health ==="
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7777/health 2>/dev/null)
        echo "HTTP: $HTTP"
        echo ""
        echo "=== Connections on 7777 ==="
        ss -tnp 2>/dev/null | grep -c ':7777' || echo "0"
        echo ""
        echo "=== Last 5 log lines ==="
        tail -5 "$BRAIN_LOG" 2>/dev/null || echo "No log"
        ;;

    stop)
        echo "=== Stopping T2 ==="
        # Kill entire process group (main + all children) via PGID
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            # Get process group ID (PGID) and kill the whole group
            PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
            if [ -n "$PGID" ] && [ "$PGID" != "0" ]; then
                kill -- -"$PGID" 2>/dev/null
                sleep 2
                kill -9 -- -"$PGID" 2>/dev/null
            else
                kill "$PID" 2>/dev/null
                sleep 2
                kill -9 "$PID" 2>/dev/null
            fi
            rm -f "$PIDFILE"
        fi
        # Kill orphaned T2 processes (from manual starts without PID file)
        for p in $(pgrep -f "titan_main.*--server"); do
            PCWD=$(readlink -f /proc/$p/cwd 2>/dev/null)
            # Only kill if cwd is T2's dir (not titan3)
            if echo "$PCWD" | grep -q "/titan$"; then
                kill -9 "$p" 2>/dev/null
            fi
        done
        # Clean up port 7777 (T3 is on 7778, safe to kill)
        sleep 1
        fuser -k 7777/tcp 2>/dev/null
        echo "Stopped"
        ;;

    start)
        echo "=== Starting T2 ==="
        if t2_is_running; then
            echo "Already running! (PID=$(cat "$PIDFILE"))"
            exit 1
        fi
        rm -f "$PIDFILE"
        setsid nohup python -u scripts/titan_main.py --server >> "$BRAIN_LOG" 2>&1 &
        echo "$!" > "$PIDFILE"
        echo "PID: $!"
        echo "Waiting 15s for boot..."
        sleep 15
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7777/health 2>/dev/null)
        if [ "$HTTP" = "200" ]; then
            echo "Health: OK (200)"
        else
            echo "Health: PENDING ($HTTP) — may need more boot time"
        fi
        ;;

    restart)
        # Safe restart: check is_dreaming before stopping (matches T1's
        # safe_restart.sh philosophy — never wake a Titan mid-dream).
        # If mid-dream, waits up to WAIT_S seconds for natural wake before
        # giving up. Pass --force as second arg to skip the check entirely.
        if [ "$1" != "--force" ]; then
            # Check dream state via HTTP API. Helper: echoes "True"/"False"/"unknown"
            check_dreaming() {
                local dj
                dj=$(curl -s --max-time 5 http://localhost:7777/v4/dreaming 2>/dev/null)
                echo "$dj" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin).get('data', {})
    print(d.get('is_dreaming', 'unknown'))
except: print('unknown')
" 2>/dev/null
            }
            IS_DREAMING=$(check_dreaming)
            if [ "$IS_DREAMING" = "True" ]; then
                # Wait-for-wake: poll every 10s up to 300s (5min). T2 dream
                # history shows durations 50-220s, so 5min covers the worst case.
                WAIT_S=300
                POLL_S=10
                WAITED=0
                echo "=== T2 is DREAMING — waiting up to ${WAIT_S}s for natural wake (poll every ${POLL_S}s) ==="
                while [ $WAITED -lt $WAIT_S ]; do
                    sleep $POLL_S
                    WAITED=$((WAITED + POLL_S))
                    IS_DREAMING=$(check_dreaming)
                    if [ "$IS_DREAMING" != "True" ]; then
                        echo "  ✓ T2 woke after ${WAITED}s (is_dreaming=$IS_DREAMING) — proceeding with restart"
                        break
                    fi
                    echo "  [t+${WAITED}s] still dreaming..."
                done
                if [ "$IS_DREAMING" = "True" ]; then
                    echo "=== T2 still dreaming after ${WAIT_S}s — restart skipped ==="
                    echo "Code is on disk; next natural restart picks it up."
                    echo "Pass --force to override (will wake mid-dream)."
                    exit 1
                fi
            else
                echo "  ✓ T2 dream check: is_dreaming=$IS_DREAMING — safe to restart"
            fi
        fi
        echo "=== Restarting T2 ==="
        bash "$0" stop
        sleep 2
        bash "$0" start
        ;;

    health)
        curl -s http://localhost:7777/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "API not responding"
        ;;

    log)
        N="${1:-30}"
        tail -"$N" "$BRAIN_LOG" 2>/dev/null || echo "No log"
        ;;

    log-errors)
        echo "=== Errors (last 20) ==="
        grep -E "ERROR|CRITICAL" "$BRAIN_LOG" 2>/dev/null | tail -20
        echo ""
        echo "=== Queue full ==="
        grep -c "Queue full" "$BRAIN_LOG" 2>/dev/null || echo "0"
        echo ""
        echo "=== Timeouts ==="
        grep -c "Request timed out" "$BRAIN_LOG" 2>/dev/null || echo "0"
        ;;

    wallet)
        python3 -c "
import json
from solders.keypair import Keypair
with open('$WALLET') as f:
    secret = json.load(f)
kp = Keypair.from_bytes(bytes(secret))
print(f'Address: {str(kp.pubkey())}')
" 2>/dev/null || echo "Wallet error"
        solana balance -k "$WALLET" --url devnet 2>/dev/null || echo "Balance check failed"
        ;;

    vocab)
        python3 -c "
import sqlite3
conn = sqlite3.connect('data/inner_memory.db', timeout=10.0)
try:
    conn.execute('PRAGMA journal_mode=WAL')
except: pass
rows = conn.execute('SELECT word, confidence, times_encountered FROM vocabulary WHERE confidence > 0 ORDER BY confidence DESC').fetchall()
print(f'T2 Vocabulary: {len(rows)} words')
for r in rows[:15]:
    print(f'  {r[0]:>15s}  conf={r[1]:.3f}  seen={r[2]}')
if len(rows) > 15:
    print(f'  ... and {len(rows)-15} more')
conn.close()
" 2>/dev/null || echo "DB error"
        ;;

    connections)
        echo "=== Active connections on 7777 ==="
        ss -tnp 2>/dev/null | grep 7777 | awk '{print $1}' | sort | uniq -c | sort -rn
        echo ""
        echo "Total: $(ss -tnp 2>/dev/null | grep 7777 | wc -l)"
        ;;

    pyexec)
        # Run arbitrary Python with venv active
        # Usage: t2_manage.sh pyexec "print('hello')"
        python3 -c "$*"
        ;;

    *)
        echo "Usage: $0 {status|stop|start|restart|health|log [N]|log-errors|wallet|vocab|connections|pyexec 'code'}"
        exit 1
        ;;
esac
