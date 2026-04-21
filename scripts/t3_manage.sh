#!/bin/bash
# t3_manage.sh — Titan3 VPS management script
# T3 runs alongside T2 on the same VPS (10.135.0.6) on port 7778
# For clean birth experiment: NO accumulated state, fresh DNA
#
# Usage from T1:
#   ssh root@10.135.0.6 "bash /home/antigravity/projects/titan3/scripts/t3_manage.sh {status|start|stop|restart|log|health|vocab|telemetry}"

TITAN_DIR="/home/antigravity/projects/titan3"
VENV="/home/antigravity/projects/titan3/test_env/bin/activate"  # dedicated T3 venv
BRAIN_LOG="/tmp/titan3_brain.log"
TELEM_LOG="/tmp/titan3_telemetry.log"
PIDFILE="/tmp/titan3.pid"

cd "$TITAN_DIR" || { echo "ERROR: Cannot cd to $TITAN_DIR"; exit 1; }
source "$VENV" 2>/dev/null || { echo "ERROR: Cannot activate venv at $VENV"; exit 1; }
export OPENROUTER_API_KEY=
# Match T1's ulimit (VS Code raises it; SSH/cron default is 1024 which causes fd exhaustion)
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true
# T3 can sense T2 as kin on localhost
export TITAN_KIN_ADDRESSES="http://localhost:7777"

# Helper: check if T3 is actually running via PID file
t3_is_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null
}

CMD="${1:-status}"
shift 2>/dev/null || true

case "$CMD" in
    status)
        echo "=== T3 Process ==="
        if t3_is_running; then
            echo "RUNNING (PID=$(cat "$PIDFILE"))"
        else
            echo "NOT RUNNING"
            rm -f "$PIDFILE"
        fi
        echo ""
        echo "=== API Health ==="
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7778/health 2>/dev/null)
        echo "HTTP: $HTTP"
        echo ""
        echo "=== Connections on 7778 ==="
        ss -tnp | grep -c ':7778' 2>/dev/null || echo "0"
        echo ""
        echo "=== Last 5 log lines ==="
        tail -5 "$BRAIN_LOG" 2>/dev/null || echo "(no log)"
        ;;

    start)
        echo "=== Starting T3 ==="
        if t3_is_running; then
            echo "Already running! (PID=$(cat "$PIDFILE"))"
            exit 1
        fi
        # Guard: T3 MUST run on 7778 (T2 owns 7777 on this shared VPS).
        # `git pull` restores the repo default `port = 7777`, clobbering the
        # local T3-only override. Idempotent sed here — no-op if already 7778,
        # fixes it otherwise. Documented in memory/feedback_t3_deploy_port.md.
        if grep -q '^port = 7777' titan_plugin/config.toml 2>/dev/null; then
            sed -i 's/^port = 7777/port = 7778/' titan_plugin/config.toml
            echo "Port fixed 7777 → 7778 (was clobbered by git pull)"
        fi
        rm -f "$PIDFILE"
        cd "$TITAN_DIR"
        # setsid: ensure the new process becomes its own session leader so
        # the PGID-based kill in `stop` works correctly when called from
        # SSH/cron without a controlling terminal. Matches t2_manage.sh.
        setsid nohup python -u scripts/titan_main.py --server >> "$BRAIN_LOG" 2>&1 &
        echo "$!" > "$PIDFILE"
        echo "PID: $!"
        echo "Waiting 15s for boot..."
        sleep 15
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7778/health 2>/dev/null)
        echo "Health: ${HTTP:-PENDING}"
        ;;

    stop)
        echo "=== Stopping T3 ==="
        # 1. Kill process group via PID file (catches main + all children)
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
            if [ -n "$PGID" ] && [ "$PGID" != "0" ] && [ "$PGID" != "1" ]; then
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
        # 2. Kill any process with titan3 in its cwd (catches orphans from manual starts)
        for p in $(pgrep -f "titan_main.*--server"); do
            PCWD=$(readlink -f /proc/$p/cwd 2>/dev/null)
            if echo "$PCWD" | grep -q "titan3"; then
                kill -9 "$p" 2>/dev/null
            fi
        done
        # 3. Kill anything still on port 7778
        sleep 1
        fuser -k 7778/tcp 2>/dev/null
        echo "Stopped"
        ;;

    restart)
        # Safe restart: check is_dreaming before stopping (matches T1's
        # safe_restart.sh philosophy — never wake a Titan mid-dream).
        # If mid-dream, waits up to WAIT_S seconds for natural wake before
        # giving up. Pass --force as second arg to skip the check entirely.
        if [ "$1" != "--force" ]; then
            check_dreaming() {
                local dj
                dj=$(curl -s --max-time 5 http://localhost:7778/v4/dreaming 2>/dev/null)
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
                # Wait-for-wake: poll every 10s up to 300s (5min).
                WAIT_S=300
                POLL_S=10
                WAITED=0
                echo "=== T3 is DREAMING — waiting up to ${WAIT_S}s for natural wake (poll every ${POLL_S}s) ==="
                while [ $WAITED -lt $WAIT_S ]; do
                    sleep $POLL_S
                    WAITED=$((WAITED + POLL_S))
                    IS_DREAMING=$(check_dreaming)
                    if [ "$IS_DREAMING" = "False" ]; then
                        echo "  ✓ T3 woke after ${WAITED}s (is_dreaming=False) — proceeding with restart"
                        break
                    fi
                    echo "  [t+${WAITED}s] still dreaming..."
                done
                if [ "$IS_DREAMING" = "True" ]; then
                    echo "=== T3 still dreaming after ${WAIT_S}s — restart skipped ==="
                    echo "Code is on disk; next natural restart picks it up."
                    echo "Pass --force to override (will wake mid-dream)."
                    exit 1
                fi
            elif [ "$IS_DREAMING" = "False" ]; then
                echo "  ✓ T3 dream check: is_dreaming=False — safe to restart"
            else
                # 2026-04-20 fix: "unknown" = API unreachable or parse error.
                # Previously this fell through as "safe to restart" which is
                # wrong — refuse unless --force so operator sees the real state.
                echo "=== T3 dream state could not be verified (is_dreaming=$IS_DREAMING) ==="
                echo "  API at localhost:7778/v4/dreaming returned no/bad response."
                echo "  Refusing to restart without verified awake state."
                echo "  Pass --force to override (will proceed regardless of dream state)."
                exit 1
            fi
        fi
        $0 stop
        sleep 2
        $0 start
        ;;

    log)
        N="${1:-30}"
        tail -"$N" "$BRAIN_LOG" 2>/dev/null
        ;;

    log-errors)
        grep -a "ERROR\|WARNING\|Traceback\|Exception" "$BRAIN_LOG" | tail -20
        ;;

    health)
        curl -s http://localhost:7778/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "unavailable"
        ;;

    vocab)
        python3 -c "
import sqlite3
db = sqlite3.connect('${TITAN_DIR}/data/inner_memory.db', timeout=5)
total = db.execute('SELECT COUNT(*) FROM vocabulary').fetchone()[0]
by_phase = db.execute('SELECT learning_phase, COUNT(*) as c FROM vocabulary GROUP BY learning_phase').fetchall()
comps = db.execute('SELECT COUNT(*) FROM composition_history').fetchone()[0]
print(f'Vocabulary: {total} words')
for p, c in by_phase: print(f'  {p}: {c}')
print(f'Compositions: {comps}')
db.close()
" 2>/dev/null || echo "(no vocabulary yet)"
        ;;

    telemetry)
        # One-shot developmental telemetry snapshot
        python3 -c "
import json, sqlite3, time, os, urllib.request

snap = {'timestamp': time.time(), 'utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}

# API data
try:
    r = urllib.request.urlopen('http://localhost:7778/health', timeout=5)
    snap['health'] = json.loads(r.read().decode())
except: snap['health'] = 'unavailable'

try:
    r = urllib.request.urlopen('http://localhost:7778/v4/neuromodulators', timeout=5)
    d = json.loads(r.read().decode())
    mods = d.get('modulators', d) if isinstance(d, dict) else {}
    snap['neuromods'] = {n: {'level': m.get('level',0), 'setpoint': m.get('setpoint',0)} for n,m in mods.items() if isinstance(m, dict)}
    snap['emotion'] = d.get('current_emotion', 'unknown')
except: snap['neuromods'] = {}

try:
    r = urllib.request.urlopen('http://localhost:7778/v4/inner-trinity', timeout=5)
    d = json.loads(r.read().decode())
    snap['epoch'] = d.get('epoch', 0)
    snap['ns_steps'] = d.get('ns', {}).get('total_train_steps', 0)
    snap['dreaming'] = d.get('dreaming', {})
except: pass

# DB stats
db_path = '${TITAN_DIR}/data/inner_memory.db'
if os.path.exists(db_path):
    try:
        db = sqlite3.connect(db_path, timeout=5)
        snap['vocabulary'] = db.execute('SELECT COUNT(*) FROM vocabulary').fetchone()[0]
        snap['vocab_by_phase'] = dict(db.execute('SELECT learning_phase, COUNT(*) FROM vocabulary GROUP BY learning_phase').fetchall())
        snap['compositions'] = db.execute('SELECT COUNT(*) FROM composition_history').fetchone()[0]
        snap['comp_by_level'] = dict(db.execute('SELECT level, COUNT(*) FROM composition_history GROUP BY level').fetchall())
        try: snap['teacher_sessions'] = db.execute('SELECT COUNT(*) FROM teacher_sessions').fetchone()[0]
        except: snap['teacher_sessions'] = 0
        try: snap['grammar_patterns'] = db.execute('SELECT COUNT(*) FROM grammar_patterns').fetchone()[0]
        except: snap['grammar_patterns'] = 0
        db.close()
    except: pass

# Log-derived stats
try:
    import subprocess
    log = '${BRAIN_LOG}'
    snap['commits'] = int(subprocess.check_output(['grep', '-ac', 'Reasoning.*COMMIT', log]).strip())
    snap['abandons'] = int(subprocess.check_output(['grep', '-ac', 'Reasoning.*ABANDON', log]).strip())
    snap['interpreter_fires'] = int(subprocess.check_output(['grep', '-ac', 'INTERPRET', log]).strip())
    snap['expression_fires'] = {}
    for comp in ['SPEAK', 'ART', 'MUSIC', 'SOCIAL', 'KIN_SENSE', 'LONGING']:
        snap['expression_fires'][comp] = int(subprocess.check_output(['grep', '-ac', f'EXPRESSION.{comp}.*FIRED', log]).strip())
    snap['teacher_errors'] = int(subprocess.check_output(['grep', '-ac', 'TEACHER.*error', log]).strip())
    snap['bus_timeouts'] = int(subprocess.check_output(['grep', '-ac', 'timed out', log]).strip())
except: pass

print(json.dumps(snap, indent=2, default=str))
" 2>/dev/null
        ;;

    *)
        echo "Usage: $0 {status|start|stop|restart|log|log-errors|health|vocab|telemetry}"
        ;;
esac
