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
# 2026-04-23 fix: use /v4/dreaming (lightweight endpoint), longer timeout
# (10s — API latency under load can exceed old 3s timeout), and fall back
# to state file's epochs_since_dream as authoritative on API failure.
# Previous implementation returned 'unknown' on any exception, causing
# false-negatives when Observatory/coordinator cache was warm-lagging.
try:
    import urllib.request
    r = urllib.request.urlopen('http://localhost:7777/v4/dreaming', timeout=10)
    api = json.loads(r.read())
    is_dreaming = api.get('data', {}).get('is_dreaming', None)
    if is_dreaming is None:
        raise ValueError('missing is_dreaming in response')
    with open('data/dreaming_state.json') as f:
        d = json.load(f)
    drain = d.get('metabolic_drain', 0)
    cycles = d.get('cycle_count', 0)
    print(f'dreaming={is_dreaming} drain={drain:.4f} cycles={cycles} src=api')
except Exception:
    # Fallback: file-based inference from epochs_since_dream.
    # >5 epochs since last dream = definitely awake (dreams are 30-50+
    # epoch cycles, so a gap >5 proves we're not mid-dream). State file
    # is updated every epoch, stale <1s.
    try:
        with open('data/dreaming_state.json') as f:
            d = json.load(f)
        drain = d.get('metabolic_drain', 0)
        epochs = d.get('epochs_since_dream', 0)
        if epochs > 5:
            print(f'dreaming=False drain={drain:.4f} epochs_since={epochs} src=file_fallback')
        else:
            # Too recent — can't be sure
            print(f'dreaming=unknown drain={drain:.4f} epochs_since={epochs} src=file_uncertain')
    except Exception:
        print('dreaming=unknown drain=0 status=no_state_file')
" 2>/dev/null)
    echo "$state"
}

check_dreaming_t2() {
    local state=$(ssh -o ConnectTimeout=10 root@10.135.0.6 "python3 -c \"
import json
try:
    import urllib.request
    r = urllib.request.urlopen('http://localhost:7777/v4/dreaming', timeout=10)
    api = json.loads(r.read())
    is_dreaming = api.get('data', {}).get('is_dreaming', None)
    if is_dreaming is None:
        raise ValueError('missing is_dreaming in response')
    with open('/home/antigravity/projects/titan/data/dreaming_state.json') as f:
        d = json.load(f)
    drain = d.get('metabolic_drain', 0)
    cycles = d.get('cycle_count', 0)
    print(f'dreaming={is_dreaming} drain={drain:.4f} cycles={cycles} src=api')
except Exception:
    try:
        with open('/home/antigravity/projects/titan/data/dreaming_state.json') as f:
            d = json.load(f)
        drain = d.get('metabolic_drain', 0)
        epochs = d.get('epochs_since_dream', 0)
        if epochs > 5:
            print(f'dreaming=False drain={drain:.4f} epochs_since={epochs} src=file_fallback')
        else:
            print(f'dreaming=unknown drain={drain:.4f} epochs_since={epochs} src=file_uncertain')
    except Exception:
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
elif echo "$STATE" | grep -q "dreaming=False"; then
    echo "  ✓ Titan is awake — safe to restart"
else
    # 2026-04-20 fix: "dreaming=unknown" = API unreachable or state file
    # parse error. Previously this fell through as "safe to restart" which
    # is wrong — refuse unless --force so operator sees the real state.
    echo ""
    echo "  ⚠  DREAM STATE COULD NOT BE VERIFIED — refusing to restart"
    echo "     State output: $STATE"
    if [[ "$FORCE" == "--force" ]]; then
        echo ""
        echo "  --force specified, proceeding with restart despite unknown state..."
    else
        echo ""
        echo "  Refusing to restart. Options:"
        echo "    1. Check API: curl -s localhost:7777/health"
        echo "    2. Check state file: cat data/dreaming_state.json"
        echo "    3. Override: bash scripts/safe_restart.sh ${TARGET} --force"
        exit 1
    fi
fi

# Perform the restart
if [[ "$TARGET" == "t1" ]]; then
    echo ""
    echo "=== Restarting T1 ==="
    # 0. Acquire the restart-coordination lockfile so services_watchdog.sh
    #    skips its zombie/duplicate-group check while we're tearing down +
    #    spawning. Without this, the kill-then-spawn window leaves both old
    #    children (PPID=1, in graceful shutdown) and new parent's group
    #    visible to services_watchdog's CHECK 6 — which can race-kill the
    #    fresh parent. Closes BUG-RESTART-WATCHDOG-RACE (2026-04-27).
    #
    #    The lockfile records start time. services_watchdog treats it as
    #    expired after 90s (covers ~75s SIGTERM/SIGKILL grace + spawn) so
    #    a crashed restart can't permanently disable the watchdog.
    RESTART_LOCK="/tmp/titan1_restart.lock"
    date +%s > "$RESTART_LOCK"
    # Only clean up on abnormal exit (Ctrl-C, terminate). On normal end
    # (we just spawned a fresh process), leave the file — services_watchdog
    # treats it as expired after 90s, which covers the orphan-children
    # graceful-shutdown window. Self-cleaning prevents a crashed restart
    # from permanently disabling the watchdog.
    trap 'rm -f "$RESTART_LOCK"' INT TERM
    # 1. Identify titan_main parent process group and signal the WHOLE GROUP.
    #    Microkernel v2 amendment 2026-04-26: previously only matched
    #    "titan_main" in cmdline, but Guardian's spawned workers (api_subprocess,
    #    body_worker, mind_worker, spirit_worker etc.) have generic cmdlines
    #    like `python -c from multiprocessing.spawn import spawn_main` that
    #    DON'T mention titan_main. SIGKILL hit only the parent, leaving 5-10
    #    worker orphans (PPID=1) per restart. Across this session that
    #    accumulated to 50+ orphans eating ~2GB RSS and pushing system to
    #    9GB swap. Process-group kill (kill -SIG -PGID) terminates parent
    #    + every descendant atomically.
    PARENT_PIDS=$(ps aux | grep titan_main | grep python | grep -v grep | awk '{print $2}')
    PGIDS=""
    for pid in $PARENT_PIDS; do
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
        if [[ -n "$pgid" && "$pgid" != "0" ]]; then
            PGIDS="$PGIDS $pgid"
        fi
    done
    PGIDS=$(echo "$PGIDS" | tr ' ' '\n' | sort -u | grep -v '^$')
    if [[ -n "$PGIDS" ]]; then
        for pgid in $PGIDS; do
            kill -TERM -$pgid 2>/dev/null
        done
    else
        # Fallback to legacy single-PID kill if pgid lookup failed.
        echo "$PARENT_PIDS" | xargs -r kill -TERM 2>/dev/null
    fi
    # 2. Wait up to 15s for graceful shutdown — workers must release flocks.
    for i in $(seq 1 50); do
        if ! pgrep -f "python.*titan_main" >/dev/null 2>&1; then
            break
        fi
        sleep 0.3
    done
    # 3. Escalate: SIGKILL the process groups (kills all descendants too).
    if pgrep -f "python.*titan_main" >/dev/null 2>&1; then
        echo "  ⚠  titan_main processes still alive after 15s SIGTERM — escalating to SIGKILL"
        if [[ -n "$PGIDS" ]]; then
            for pgid in $PGIDS; do
                kill -KILL -$pgid 2>/dev/null
            done
        fi
        # Also kill any titan_main parent that survived (paranoid)
        ps aux | grep titan_main | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
        for i in $(seq 1 17); do
            if ! pgrep -f "python.*titan_main" >/dev/null 2>&1; then
                break
            fi
            sleep 0.3
        done
    fi
    # 3b. Sweep any orphan multiprocessing spawn workers from titan's venv
    #     (PPID=1 = re-parented to init = orphan from prior restart). These
    #     don't show up in `pgrep titan_main` but consume RAM until reaped.
    ORPHAN_PIDS=$(ps -eo pid,ppid,cmd 2>/dev/null | awk '$2 == 1 && /titan\/test_env\/bin\/python -c.*multiprocessing.spawn/ {print $1}')
    if [[ -n "$ORPHAN_PIDS" ]]; then
        ORPHAN_COUNT=$(echo "$ORPHAN_PIDS" | wc -l)
        echo "  ⚠  Sweeping $ORPHAN_COUNT orphan multiprocessing worker(s) from prior restarts"
        echo "$ORPHAN_PIDS" | xargs -r kill -9 2>/dev/null
        sleep 1
    fi
    # 4. Clean any stale PID file. Even with the 2026-04-21 _acquire_pid_lock
    #    fix (truncate AFTER flock), removing the stale file here makes the
    #    abort-message branch's "PID ???" display honest — no more referencing
    #    a dead parent.
    rm -f data/titan_main.pid 2>/dev/null
    source test_env/bin/activate
    # Append (>>) so pre-restart forensics are preserved. Size is bounded by
    # titan_watchdog.sh log rotation (>100MB → gz archive).
    echo "" >> /tmp/titan_brain.log
    echo "=== [$(date -u '+%Y-%m-%d %H:%M:%S UTC')] RESTART boundary — safe_restart.sh ===" >> /tmp/titan_brain.log
    # ── Microkernel v2 Phase B fix (2026-04-27 PM) ───────────────────────
    # Read the active API port from data/active_api_port. This file is the
    # source-of-truth for which port the running kernel is bound to + which
    # port nginx is proxying to. shadow_orchestrator writes it after each
    # successful nginx_swap (line ~1270 in shadow_orchestrator.py).
    #
    # Without this read, post-shadow-swap restarts default to port 7777 while
    # nginx points at the swap-target port (e.g. 7779) → all dashboard
    # endpoints return HTTP 502 (full public observability outage).
    # Concrete incident: 2026-04-27 timechain index.db corruption recovery
    # — yesterday's swap left nginx on 7779; today's watchdog restart bound
    # 7777 → ~30-90min observatory dark.
    # See BUG-SHADOW-SWAP-NGINX-PORT-NOT-REVERTED in BUGS.md.
    ACTIVE_PORT=$(cat data/active_api_port 2>/dev/null | tr -d '[:space:]')
    if [[ -z "$ACTIVE_PORT" || ! "$ACTIVE_PORT" =~ ^[0-9]+$ ]]; then
        ACTIVE_PORT=7777
        echo "[safe_restart] no valid data/active_api_port — using default port $ACTIVE_PORT" >> /tmp/titan_brain.log
    else
        echo "[safe_restart] honoring data/active_api_port=$ACTIVE_PORT (matches nginx upstream)" >> /tmp/titan_brain.log
    fi
    # MALLOC_ARENA_MAX=2 limits glibc to 2 malloc arenas (default = 8 ×
    # num_cpus = 32 on a 4-CPU host). Reduces RssAnon fragmentation in
    # heavily-threaded Python+numpy+torch workloads. Standard pattern for
    # memory-conscious Python services. Closes the C-level "leak"
    # observed 2026-04-27 (T1 700MB/30min RssAnon growth not in Python
    # heap — was per-arena retention, not real leak).
    # TITAN_API_PORT honored by api_subprocess at boot (titan_main.py:267-269).
    MALLOC_ARENA_MAX=2 OPENROUTER_API_KEY='' TITAN_API_PORT="$ACTIVE_PORT" nohup python -u scripts/titan_main.py --server >> /tmp/titan_brain.log 2>&1 &
    NEW_PID=$!
    echo "$NEW_PID" > /tmp/titan1.pid   # keep watchdog in sync; otherwise it kills us on next cron tick
    echo "T1 restarted (PID=$NEW_PID)"
elif [[ "$TARGET" == "t2" ]]; then
    echo ""
    echo "=== Restarting T2 ==="
    ssh root@10.135.0.6 "bash /home/antigravity/projects/titan/scripts/t2_manage.sh restart"
fi
