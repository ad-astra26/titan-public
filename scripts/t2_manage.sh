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
# Raise FD limit (cron default 1024 causes fd exhaustion ~2h into uptime,
# 2026-04-22 T2 incident). `ulimit -n N` can return 0 in cron/env-i context
# WITHOUT actually raising the soft limit, so the old `|| fallback` chain
# short-circuited silently. Verify after setting + fall back explicitly.
ulimit -n 1048576 2>/dev/null
if [ "$(ulimit -n)" -lt 65536 ]; then
    ulimit -n 65536 2>/dev/null || true
fi
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
        # Phase C C-S2 (PLAN §17.3 / BUG-DUPLICATE-KERNELS-FRAGMENT-BUS-20260428):
        # `stop` kills ALL titan_main process groups whose cwd matches T2's
        # project dir, NOT just the PIDFILE PID. Pre-fix: services_watchdog
        # could have spawned a fresh titan_main between our stop call and
        # PIDFILE write — old process survived, leading to two parents
        # competing for the bus socket.
        #
        # Algorithm:
        #   1. Collect orphan PIDs whose /proc/<pid>/cwd is exactly TITAN_DIR.
        #      Use `pgrep -f` to find titan_main candidates, then exact-match
        #      cwd resolution (NOT substring, per feedback_shared_vps_pkill_trap.md).
        #   2. Add the PIDFILE PID if present.
        #   3. Resolve each PID's PGID; SIGTERM the group; wait + SIGKILL.
        #   4. Wait until /proc/<pid> is gone for every known PID before
        #      returning so callers can rely on the post-stop state.

        TITAN_PIDS=""
        if [ -f "$PIDFILE" ]; then
            PFPID=$(cat "$PIDFILE" 2>/dev/null | tr -d '[:space:]')
            if [ -n "$PFPID" ] && kill -0 "$PFPID" 2>/dev/null; then
                TITAN_PIDS="$PFPID"
            fi
            rm -f "$PIDFILE"
        fi
        for p in $(pgrep -f "titan_main.*--server" 2>/dev/null); do
            PCWD=$(readlink -f "/proc/$p/cwd" 2>/dev/null)
            if [ "$PCWD" = "$TITAN_DIR" ]; then
                TITAN_PIDS="$TITAN_PIDS $p"
            fi
        done
        # De-dupe + drop empty
        TITAN_PIDS=$(echo "$TITAN_PIDS" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')

        if [ -n "$TITAN_PIDS" ]; then
            # Resolve unique PGIDs
            TITAN_PGIDS=""
            for p in $TITAN_PIDS; do
                PGID=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
                if [ -n "$PGID" ] && [ "$PGID" != "0" ] && [ "$PGID" != "1" ]; then
                    TITAN_PGIDS="$TITAN_PGIDS $PGID"
                fi
            done
            TITAN_PGIDS=$(echo "$TITAN_PGIDS" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')

            for pgid in $TITAN_PGIDS; do
                kill -- -"$pgid" 2>/dev/null
            done
            # Wait up to ~6s for graceful exit
            for _ in 1 2 3 4 5 6; do
                ALIVE=0
                for p in $TITAN_PIDS; do
                    if [ -e "/proc/$p" ]; then ALIVE=$((ALIVE+1)); fi
                done
                [ "$ALIVE" -eq 0 ] && break
                sleep 1
            done
            # Force-kill anything still alive
            for pgid in $TITAN_PGIDS; do
                kill -9 -- -"$pgid" 2>/dev/null
            done
            # Final wait — every /proc/<pid> must disappear before we return
            for _ in 1 2 3; do
                ALIVE=0
                for p in $TITAN_PIDS; do
                    if [ -e "/proc/$p" ]; then ALIVE=$((ALIVE+1)); fi
                done
                [ "$ALIVE" -eq 0 ] && break
                sleep 1
            done
        fi

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
        # MALLOC_ARENA_MAX=2 — limits glibc malloc arenas (see safe_restart.sh
        # for full rationale). Closes C-level RssAnon fragmentation 2026-04-27.
        MALLOC_ARENA_MAX=2 setsid nohup python -u scripts/titan_main.py --server >> "$BRAIN_LOG" 2>&1 &
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
            # 2026-04-23 fix: longer curl timeout (5s → 10s) + file-based
            # epochs_since_dream fallback when API fails. Previous
            # implementation false-negatived under Observatory API latency
            # (routinely 1+ min under load). epochs_since_dream > 5 is
            # definitive awake evidence (dreams are 30-50+ epochs).
            check_dreaming() {
                local dj
                dj=$(curl -s --max-time 10 http://localhost:7777/v4/dreaming 2>/dev/null)
                local api_result
                api_result=$(echo "$dj" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin).get('data', {})
    v = d.get('is_dreaming')
    if v is None:
        print('unknown')
    else:
        print(v)
except: print('unknown')
" 2>/dev/null)
                if [ "$api_result" = "True" ] || [ "$api_result" = "False" ]; then
                    echo "$api_result"
                    return
                fi
                # API failed — fall back to local state file
                python3 -c "
import json
try:
    with open('/home/antigravity/projects/titan/data/dreaming_state.json') as f:
        d = json.load(f)
    epochs = d.get('epochs_since_dream', 0)
    print('False' if epochs > 5 else 'unknown')
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
                    if [ "$IS_DREAMING" = "False" ]; then
                        echo "  ✓ T2 woke after ${WAITED}s (is_dreaming=False) — proceeding with restart"
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
            elif [ "$IS_DREAMING" = "False" ]; then
                echo "  ✓ T2 dream check: is_dreaming=False — safe to restart"
            else
                # 2026-04-20 fix: "unknown" = API unreachable or parse error.
                # Previously this fell through as "safe to restart" which is
                # wrong — refuse unless --force so operator sees the real state.
                echo "=== T2 dream state could not be verified (is_dreaming=$IS_DREAMING) ==="
                echo "  API at localhost:7777/v4/dreaming returned no/bad response."
                echo "  Refusing to restart without verified awake state."
                echo "  Pass --force to override (will proceed regardless of dream state)."
                exit 1
            fi
        fi
        echo "=== Restarting T2 ==="
        # Acquire restart-coordination lockfile so services_watchdog.sh skips
        # its zombie/duplicate-group check during the kill-then-spawn window.
        # Closes BUG-RESTART-WATCHDOG-RACE (2026-04-27).
        RESTART_LOCK="/tmp/titan2_restart.lock"
        date +%s > "$RESTART_LOCK"
        trap 'rm -f "$RESTART_LOCK"' INT TERM
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
