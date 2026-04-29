#!/bin/bash
# services_watchdog.sh — Unified service health monitor for Titan
# Checks: Teacher, Persona, ARC, Data integrity, Zombie processes
# Alerts via Telegram on state changes (healthy→error, error→recovered)
#
# Usage:
#   ./scripts/services_watchdog.sh T1              # Run for T1 (local)
#   ./scripts/services_watchdog.sh T2              # Run for T2 (local on VPS)
#   ./scripts/services_watchdog.sh T3              # Run for T3 (local on VPS)
#   ./scripts/services_watchdog.sh T1 --check-kin  # T1 also checks T2/T3
#
# Cron (T1):
#   */5 * * * * bash /home/antigravity/projects/titan/scripts/services_watchdog.sh T1 --check-kin >> /tmp/services_watchdog_t1.log 2>&1
# Cron (T2, on VPS):
#   */5 * * * * bash /home/antigravity/projects/titan/scripts/services_watchdog.sh T2 >> /tmp/services_watchdog_t2.log 2>&1
# Cron (T3, on VPS):
#   */5 * * * * bash /home/antigravity/projects/titan3/scripts/services_watchdog.sh T3 >> /tmp/services_watchdog_t3.log 2>&1

set -euo pipefail

TITAN_ID="${1:-T1}"
CHECK_KIN="${2:-}"
NOW=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
TIMESTAMP=$(date +%s)

# ── Host memory-pressure thresholds (Layer 1 safety net 2026-04-28) ──
# Triggered by 2026-04-28 04:53 host OOM crash: parent-resident state owners
# (Sage, L2 caches, bus history) accumulated until host RAM exhausted, swap
# thrashed, DigitalOcean hypervisor force-rebooted the VM. Without a
# system-level pre-OOM trigger, ANY future leak in the parent process kills
# the host. This watchdog fires safe_restart.sh BEFORE OOM lands, capping
# blast radius at one Titan restart vs whole-host reboot + filesystem damage.
#
# Trigger: MemAvailable% < threshold OR Swap% > threshold, sustained for the
# grace window (3 consecutive 5-min cron ticks). Sustained-only — single
# spike from a one-shot tool (npm build, pytest) won't fire.
HOST_MEM_AVAILABLE_PCT_FLOOR=10   # MemAvailable < 10% of MemTotal = pre-OOM
HOST_SWAP_USED_PCT_CEIL=80        # Swap usage > 80% = thrashing imminent
HOST_PRESSURE_GRACE_S=900         # 15 min sustained (3 cron ticks @ 5min)
HOST_PRESSURE_FLAG_FILE="/tmp/titan_host_memory_pressure.flag"

# ── Configuration per Titan ──
case "$TITAN_ID" in
    T1)
        PROJECT_DIR="/home/antigravity/projects/titan"
        API_PORT=7777
        API_HOST="127.0.0.1"
        BRAIN_LOG="/tmp/titan_brain.log"
        PERSONA_LOG="/tmp/persona_social_t1.log"
        TELEMETRY_FILE="${PROJECT_DIR}/data/persona_telemetry.jsonl"
        ARC_LOG="/tmp/arc_training.log"
        PIDFILE="/tmp/titan1.pid"
        ;;
    T2)
        PROJECT_DIR="/home/antigravity/projects/titan"
        API_PORT=7777
        API_HOST="127.0.0.1"
        BRAIN_LOG="/tmp/titan2_brain.log"
        PERSONA_LOG="/tmp/persona_social_t2.log"
        TELEMETRY_FILE="${PROJECT_DIR}/data/persona_telemetry.jsonl"
        ARC_LOG="/tmp/t2_arc_training.log"
        PIDFILE="/tmp/titan2.pid"
        ;;
    T3)
        PROJECT_DIR="/home/antigravity/projects/titan3"
        API_PORT=7778
        API_HOST="127.0.0.1"
        BRAIN_LOG="/tmp/titan3_brain.log"
        PERSONA_LOG="/tmp/persona_social_t3.log"
        TELEMETRY_FILE="${PROJECT_DIR}/data/persona_telemetry.jsonl"
        ARC_LOG="/tmp/t3_arc_training.log"
        PIDFILE="/tmp/titan3.pid"
        ;;
    *)
        echo "Usage: $0 {T1|T2|T3} [--check-kin]"
        exit 1
        ;;
esac

# ── Telegram ──
TELEGRAM_BOT_TOKEN="8531091229:AAElGsqbsLDvDfxaCwMwG1qGdBzyVlksI0c"
TELEGRAM_CHAT_ID="6345894322"
STATE_FILE="/tmp/services_watchdog_${TITAN_ID}_state"
TELEMETRY_OUT="${PROJECT_DIR}/data/service_telemetry.jsonl"

send_telegram() {
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="$TELEGRAM_CHAT_ID" \
        -d text="$msg" \
        -d parse_mode="Markdown" > /dev/null 2>&1
}

# Load previous state (to detect transitions)
declare -A PREV_STATE
if [ -f "$STATE_FILE" ]; then
    while IFS='=' read -r key val; do
        PREV_STATE["$key"]="$val"
    done < "$STATE_FILE"
fi

declare -A CURR_STATE
ERRORS=""
RECOVERIES=""

log_telemetry() {
    local service="$1" event="$2"
    shift 2
    local extra="$*"
    echo "{\"ts\":${TIMESTAMP},\"titan\":\"${TITAN_ID}\",\"service\":\"${service}\",\"event\":\"${event}\"${extra:+,$extra}}" >> "$TELEMETRY_OUT" 2>/dev/null || true
}

# ═══════════════════════════════════════════════════════════════════════
# CHECK 0: Host memory pressure pre-OOM safety net (Layer 1, 2026-04-28)
# ═══════════════════════════════════════════════════════════════════════
# Reads /proc/meminfo, evaluates two pre-OOM conditions, requires sustained
# pressure across the grace window before firing safe_restart. Acts only
# on T1 (the Titan running on this host). T2/T3 cron paths skip — they run
# on a different host, and that host has its own watchdog instance.
check_host_memory_pressure() {
    if [ "$TITAN_ID" != "T1" ]; then
        # T2/T3 watchdog runs on a different host; trigger logic applies
        # locally only. Skip silently — no per-Titan flag noise.
        return 0
    fi

    local mem_total mem_available swap_total swap_free
    mem_total=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    mem_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    swap_total=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
    swap_free=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)

    if [ -z "$mem_total" ] || [ "$mem_total" = "0" ]; then
        return 0  # /proc/meminfo unreadable; skip silently
    fi

    local mem_avail_pct=$((mem_available * 100 / mem_total))
    local swap_used_pct=0
    if [ -n "$swap_total" ] && [ "$swap_total" != "0" ]; then
        swap_used_pct=$(((swap_total - swap_free) * 100 / swap_total))
    fi

    CURR_STATE[host_mem]="ok"
    local pressure_reason=""

    if [ "$mem_avail_pct" -lt "$HOST_MEM_AVAILABLE_PCT_FLOOR" ]; then
        pressure_reason="MemAvailable=${mem_avail_pct}% < ${HOST_MEM_AVAILABLE_PCT_FLOOR}% floor"
    fi
    if [ "$swap_used_pct" -gt "$HOST_SWAP_USED_PCT_CEIL" ]; then
        if [ -n "$pressure_reason" ]; then
            pressure_reason="${pressure_reason}; Swap=${swap_used_pct}% > ${HOST_SWAP_USED_PCT_CEIL}% ceiling"
        else
            pressure_reason="Swap=${swap_used_pct}% > ${HOST_SWAP_USED_PCT_CEIL}% ceiling"
        fi
    fi

    if [ -z "$pressure_reason" ]; then
        if [ -f "$HOST_PRESSURE_FLAG_FILE" ]; then
            rm -f "$HOST_PRESSURE_FLAG_FILE"
            log_telemetry "host_mem" "pressure_cleared" "\"mem_avail_pct\":${mem_avail_pct},\"swap_used_pct\":${swap_used_pct}"
        fi
        return 0
    fi

    CURR_STATE[host_mem]="pressure"
    log_telemetry "host_mem" "pressure_detected" "\"mem_avail_pct\":${mem_avail_pct},\"swap_used_pct\":${swap_used_pct},\"reason\":\"${pressure_reason}\""

    local first_seen sustained_s
    if [ -f "$HOST_PRESSURE_FLAG_FILE" ]; then
        first_seen=$(cat "$HOST_PRESSURE_FLAG_FILE" 2>/dev/null || echo "$TIMESTAMP")
    else
        first_seen="$TIMESTAMP"
        echo "$first_seen" > "$HOST_PRESSURE_FLAG_FILE"
        send_telegram "🟡 *T1 HOST MEMORY PRESSURE*
${pressure_reason}
Sustained-trigger countdown started (need ${HOST_PRESSURE_GRACE_S}s sustained → safe_restart).
_${NOW}_"
        return 0
    fi

    sustained_s=$((TIMESTAMP - first_seen))
    if [ "$sustained_s" -lt "$HOST_PRESSURE_GRACE_S" ]; then
        # Pressure still on but grace window not exhausted.
        return 0
    fi

    # Sustained pressure exhausted grace — fire safe_restart.sh BEFORE host OOMs.
    # Check restart lockfile first (set by safe_restart while running).
    if [ -f "/tmp/titan1_restart.lock" ]; then
        local lock_age=$(( TIMESTAMP - $(stat -c %Y /tmp/titan1_restart.lock 2>/dev/null || echo $TIMESTAMP) ))
        if [ "$lock_age" -lt 90 ]; then
            return 0  # restart already in flight
        fi
    fi

    send_telegram "🔴 *T1 HOST MEMORY — FIRING PRE-OOM RESTART*
${pressure_reason} sustained ${sustained_s}s (grace=${HOST_PRESSURE_GRACE_S}s).
Triggering safe_restart.sh t1 BEFORE host OOM.
_${NOW}_"
    log_telemetry "host_mem" "pre_oom_restart_fired" "\"sustained_s\":${sustained_s},\"reason\":\"${pressure_reason}\""

    rm -f "$HOST_PRESSURE_FLAG_FILE"
    cd "$PROJECT_DIR" 2>/dev/null
    nohup bash "${PROJECT_DIR}/scripts/safe_restart.sh" t1 --force \
        >> /tmp/titan_pre_oom_restart.log 2>&1 &
    return 0
}

check_host_memory_pressure

# ═══════════════════════════════════════════════════════════════════════
# CHECK 1: Core API health
# ═══════════════════════════════════════════════════════════════════════
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://${API_HOST}:${API_PORT}/health" 2>/dev/null || echo "000")
if [ "$API_STATUS" = "200" ]; then
    CURR_STATE[api]="ok"
    # Get guardian details for worker health
    HEALTH_JSON=$(curl -s --max-time 10 "http://${API_HOST}:${API_PORT}/health" 2>/dev/null || echo "{}")
else
    CURR_STATE[api]="down"
    ERRORS="${ERRORS}API: HTTP ${API_STATUS}\n"
    HEALTH_JSON="{}"
fi

# ═══════════════════════════════════════════════════════════════════════
# CHECK 2: Teacher activity (runs inside titan_main)
# ═══════════════════════════════════════════════════════════════════════
if [ -f "$BRAIN_LOG" ]; then
    # Check if teacher produced output in last 30 minutes
    TEACHER_RECENT=$(grep -c "\[TEACHER\].*Session complete\|TEACHER.*modeling\|TEACHER.*conversation" "$BRAIN_LOG" 2>/dev/null | tail -1 || echo "0")
    # More precise: check last teacher log timestamp
    TEACHER_LAST=$(grep "\[TEACHER\]" "$BRAIN_LOG" 2>/dev/null | tail -1 | grep -oP '^\d{2}:\d{2}:\d{2}' || echo "")
    if [ -n "$TEACHER_LAST" ]; then
        CURR_STATE[teacher]="ok"
        log_telemetry "teacher" "check_ok" "\"last_seen\":\"${TEACHER_LAST}\""
    else
        # Teacher may not have fired yet this boot — check if API is up
        if [ "$API_STATUS" = "200" ]; then
            CURR_STATE[teacher]="waiting"
        else
            CURR_STATE[teacher]="down"
            ERRORS="${ERRORS}Teacher: no activity in log\n"
        fi
    fi
else
    CURR_STATE[teacher]="no_log"
fi

# ═══════════════════════════════════════════════════════════════════════
# CHECK 3: Persona Social v2 (cron job)
# ═══════════════════════════════════════════════════════════════════════
if [ -f "$TELEMETRY_FILE" ]; then
    PERSONA_AGE=$(( TIMESTAMP - $(stat -c %Y "$TELEMETRY_FILE" 2>/dev/null || echo 0) ))
    if [ "$PERSONA_AGE" -lt 5400 ]; then  # Updated in last 90 minutes
        CURR_STATE[persona]="ok"
        PERSONA_ENTRIES=$(wc -l < "$TELEMETRY_FILE" 2>/dev/null || echo 0)
        log_telemetry "persona" "check_ok" "\"entries\":${PERSONA_ENTRIES},\"age_sec\":${PERSONA_AGE}"
    else
        CURR_STATE[persona]="stale"
        ERRORS="${ERRORS}Persona: telemetry stale (${PERSONA_AGE}s old)\n"
        # Try to trigger a manual run if API is up
        if [ "$API_STATUS" = "200" ] && [ "$PERSONA_AGE" -gt 7200 ]; then
            cd "$PROJECT_DIR" 2>/dev/null
            if [ -f "test_env/bin/activate" ]; then
                . test_env/bin/activate 2>/dev/null
                timeout 300 python scripts/persona_social_v2.py --titan "$TITAN_ID" --once >> "$PERSONA_LOG" 2>&1 &
                log_telemetry "persona" "auto_triggered" "\"reason\":\"stale_telemetry\""
                echo "[$NOW] Persona auto-triggered (telemetry ${PERSONA_AGE}s old)"
            fi
        fi
    fi
else
    CURR_STATE[persona]="no_data"
    # Not an error on first run — persona needs at least one cron cycle
    if [ -n "${PREV_STATE[persona]:-}" ] && [ "${PREV_STATE[persona]}" != "no_data" ]; then
        ERRORS="${ERRORS}Persona: telemetry file missing\n"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# CHECK 4: ARC-AGI training (cron job, every 3 hours)
# ═══════════════════════════════════════════════════════════════════════
if [ -f "$ARC_LOG" ]; then
    ARC_AGE=$(( TIMESTAMP - $(stat -c %Y "$ARC_LOG" 2>/dev/null || echo 0) ))
    if [ "$ARC_AGE" -lt 14400 ]; then  # Updated in last 4 hours
        CURR_STATE[arc]="ok"
        log_telemetry "arc" "check_ok" "\"age_sec\":${ARC_AGE}"
    else
        CURR_STATE[arc]="stale"
        ERRORS="${ERRORS}ARC: training log stale (${ARC_AGE}s old)\n"
        # Auto-trigger ARC if stale > 5 hours and API is up
        if [ "$ARC_AGE" -gt 18000 ] && [ "$API_STATUS" = "200" ]; then
            cd "$PROJECT_DIR" 2>/dev/null
            if [ -f "test_env/bin/activate" ]; then
                . test_env/bin/activate 2>/dev/null
                nohup bash scripts/arc_cron_train.sh >> "$ARC_LOG" 2>&1 &
                log_telemetry "arc" "auto_triggered" "\"reason\":\"stale_log\""
                echo "[$NOW] ARC auto-triggered (log ${ARC_AGE}s old)"
            fi
        fi
    fi
else
    CURR_STATE[arc]="no_log"
    # Not critical on first check
fi

# ═══════════════════════════════════════════════════════════════════════
# CHECK 4b: Events Teacher (cron job, every 15 minutes)
# ═══════════════════════════════════════════════════════════════════════
EVENTS_LOG="/tmp/events_teacher_$(echo "$TITAN_ID" | tr '[:upper:]' '[:lower:]').log"
if [ -f "$EVENTS_LOG" ]; then
    EVENTS_AGE=$(( TIMESTAMP - $(stat -c %Y "$EVENTS_LOG" 2>/dev/null || echo 0) ))
    if [ "$EVENTS_AGE" -lt 1800 ]; then  # Updated in last 30 minutes
        CURR_STATE[events_teacher]="ok"
        log_telemetry "events_teacher" "check_ok" "\"age_sec\":${EVENTS_AGE}"
    elif [ "$EVENTS_AGE" -lt 7200 ]; then
        CURR_STATE[events_teacher]="stale"
        ERRORS="${ERRORS}Events Teacher: log stale (${EVENTS_AGE}s old)\n"
    else
        CURR_STATE[events_teacher]="down"
        ERRORS="${ERRORS}Events Teacher: DEAD — no activity for $(( EVENTS_AGE / 3600 ))h\n"
    fi
else
    CURR_STATE[events_teacher]="no_log"
    # Not an error on first check
    if [ -n "${PREV_STATE[events_teacher]:-}" ] && [ "${PREV_STATE[events_teacher]}" != "no_log" ]; then
        ERRORS="${ERRORS}Events Teacher: log file missing\n"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# CHECK 4c: CGN (Concept Grounding Network) — language consumer
# ═══════════════════════════════════════════════════════════════════════
CGN_RESP=$(curl -s --max-time 5 "http://${API_HOST}:${API_PORT}/v4/language-grounding" 2>/dev/null)
if [ -n "$CGN_RESP" ]; then
    CGN_GROUNDED=$(echo "$CGN_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin).get('data',{}); print(d.get('grounded',0))" 2>/dev/null || echo "0")
    CGN_RATE=$(echo "$CGN_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin).get('data',{}); print(d.get('grounding_rate',0))" 2>/dev/null || echo "0")
    if [ "$CGN_GROUNDED" != "0" ] && [ "$CGN_GROUNDED" != "" ]; then
        CURR_STATE[cgn]="ok"
        log_telemetry "cgn" "check_ok" "\"grounded\":${CGN_GROUNDED},\"rate\":${CGN_RATE}"
    else
        CURR_STATE[cgn]="no_data"
        ERRORS="${ERRORS}CGN: no grounded words (rate=${CGN_RATE})\n"
    fi
else
    CURR_STATE[cgn]="down"
    ERRORS="${ERRORS}CGN: API unreachable\n"
fi

# ═══════════════════════════════════════════════════════════════════════
# CHECK 5: Data file integrity
# ═══════════════════════════════════════════════════════════════════════
DATA_DIR="${PROJECT_DIR}/data"
CRITICAL_FILES=(
    "msl/msl_policy.json"
    "msl/msl_identity.json"
    "msl/msl_stats.json"
    "msl/msl_buffer.json"
    "filter_down_weights.json"
    "pi_heartbeat_state.json"
)
MISSING_FILES=""
for cf in "${CRITICAL_FILES[@]}"; do
    if [ ! -f "${DATA_DIR}/${cf}" ]; then
        MISSING_FILES="${MISSING_FILES}${cf} "
    elif [ ! -s "${DATA_DIR}/${cf}" ]; then
        MISSING_FILES="${MISSING_FILES}${cf}(empty) "
    fi
done

# Check inner_memory.db (vocabulary) and consciousness.db
for db in "inner_memory.db" "consciousness.db"; do
    if [ ! -f "${DATA_DIR}/${db}" ]; then
        MISSING_FILES="${MISSING_FILES}${db} "
    fi
done

# Active DB health: try a quick query on critical SQLite databases
DB_ISSUES=""
for db in "inner_memory.db" "consciousness.db" "social_x.db"; do
    DB_FILE="${DATA_DIR}/${db}"
    if [ -f "$DB_FILE" ]; then
        RESULT=$(timeout 5 python3 -c "
import sqlite3, sys
try:
    c = sqlite3.connect('${DB_FILE}', timeout=2)
    c.execute('BEGIN IMMEDIATE')  # acquire write lock — detects locked DBs
    c.execute('ROLLBACK')
    c.close()
except Exception as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)
" 2>&1)
        if [ "$?" -ne 0 ]; then
            DB_ISSUES="${DB_ISSUES}${db}(${RESULT}) "
        fi
    fi
done

if [ -z "$MISSING_FILES" ] && [ -z "$DB_ISSUES" ]; then
    CURR_STATE[data]="ok"
elif [ -n "$DB_ISSUES" ]; then
    CURR_STATE[data]="db_error"
    ERRORS="${ERRORS}Data: DB issues: ${DB_ISSUES}\n"
    [ -n "$MISSING_FILES" ] && ERRORS="${ERRORS}Data: missing/empty: ${MISSING_FILES}\n"
else
    CURR_STATE[data]="missing"
    ERRORS="${ERRORS}Data: missing/empty: ${MISSING_FILES}\n"
fi

# ═══════════════════════════════════════════════════════════════════════
# CHECK 6: Zombie + duplicate process detection (auto-kill)
# ═══════════════════════════════════════════════════════════════════════
# Skip this check if a coordinated restart is in flight. Two writers
# may produce this file:
#   1. safe_restart.sh / t{2,3}_manage.sh restart — write a plain epoch
#      seconds integer; honor it for 90s.
#   2. shadow_orchestrator (Phase C C-S2 PLAN §17.1) — writes JSON
#      payload with structured `started_at` + `expected_end_at` +
#      `heartbeat_at` + `swap_id` + heartbeats every 10s during swap.
#      Honor while now < expected_end_at; if expired, force-clean both
#      PIDs (orchestrator crashed mid-swap).
# Closes BUG-RESTART-WATCHDOG-RACE (2026-04-27) +
# BUG-SERVICES-WATCHDOG-SHADOW-SWAP-RACE-20260428.
RESTART_LOCK_FILE="/tmp/titan$(echo "$TITAN_ID" | tr 'A-Z' 'a-z' | sed 's/t//')_restart.lock"
RESTART_IN_FLIGHT=false
LOCK_EXPIRED=false
if [ -f "$RESTART_LOCK_FILE" ]; then
    LOCK_RAW=$(cat "$RESTART_LOCK_FILE" 2>/dev/null || echo "")
    NOW_TS=$(date +%s)
    if [ -n "$LOCK_RAW" ] && [ "${LOCK_RAW:0:1}" = "{" ]; then
        # JSON form (shadow_orchestrator). Parse expected_end_at.
        EXPECTED_END=$(echo "$LOCK_RAW" | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(int(float(d.get('expected_end_at', 0))))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
        if [ "$NOW_TS" -lt "$EXPECTED_END" ]; then
            RESTART_IN_FLIGHT=true
        else
            LOCK_EXPIRED=true
        fi
    else
        # Plain-epoch form (safe_restart.sh / *_manage.sh) — 90s window
        LOCK_TS=${LOCK_RAW:-0}
        LOCK_AGE=$((NOW_TS - LOCK_TS))
        if [ "$LOCK_AGE" -lt 90 ]; then
            RESTART_IN_FLIGHT=true
        else
            LOCK_EXPIRED=true
        fi
    fi
fi

ZOMBIES=$(ps aux 2>/dev/null | grep -c ' Z ') || ZOMBIES=0
ZOMBIES=$((ZOMBIES > 0 ? ZOMBIES : 0))

# Count titan_main parent processes (should be exactly 1 per Titan)
# IMPORTANT: Filter by CWD matching PROJECT_DIR so T2/T3 on shared VPS
# don't kill each other's processes
_own_pids() {
    pgrep -f "titan_main.*--server" 2>/dev/null | while read pid; do
        PID_CWD=$(readlink /proc/$pid/cwd 2>/dev/null || echo "")
        if [ "$PID_CWD" = "$PROJECT_DIR" ]; then
            echo "$pid"
        fi
    done
}

TITAN_PARENTS=$(_own_pids | while read pid; do
    ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' '
done | sort -u | wc -l)

if [ "$TITAN_PARENTS" -gt 1 ] && [ "$RESTART_IN_FLIGHT" = "true" ]; then
    # Coordinated restart in flight — duplicates are expected (old children
    # still in graceful shutdown). Don't intervene; just note and move on.
    CURR_STATE[zombies]="ok"
elif [ "$TITAN_PARENTS" -gt 1 ] && [ "$LOCK_EXPIRED" = "true" ]; then
    # Phase C C-S2 (PLAN §17.1): orchestrator wrote a lock and then
    # crashed (heartbeat stopped, expected_end_at passed). Force-clean
    # ALL titan_main groups + the stale lock so the next watchdog cycle
    # restarts cleanly via the dead-titan branch.
    CURR_STATE[zombies]="orchestrator_crashed"
    ERRORS="${ERRORS}Zombies: stale shadow-swap lock + ${TITAN_PARENTS} duplicate titan_main groups — force-cleaning\n"
    _own_pids | while read pid; do
        kill -9 "$pid" 2>/dev/null
    done
    rm -f "$RESTART_LOCK_FILE"
elif [ "$TITAN_PARENTS" -gt 1 ]; then
    CURR_STATE[zombies]="duplicates"
    ERRORS="${ERRORS}Zombies: ${TITAN_PARENTS} duplicate titan_main groups detected — killing extras\n"
    # Keep the process group matching the PID file (the legitimate instance).
    # Old logic kept "oldest" which is WRONG — orphans are older than the
    # freshly restarted working instance.
    KEEPER_PGID=""
    if [ -f "$PIDFILE" ]; then
        KEEPER_PID=$(cat "$PIDFILE" 2>/dev/null)
        if kill -0 "$KEEPER_PID" 2>/dev/null; then
            KEEPER_PGID=$(ps -o pgid= -p "$KEEPER_PID" 2>/dev/null | tr -d ' ')
        fi
    fi
    # Fallback: if PID file is stale/missing, keep the newest group (most
    # likely the last watchdog restart) instead of the oldest (orphans)
    if [ -z "$KEEPER_PGID" ]; then
        KEEPER_PGID=$(_own_pids | while read pid; do
            echo "$(ps -o lstart= -p "$pid" 2>/dev/null | xargs -I{} date -d '{}' +%s 2>/dev/null) $(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
        done | sort -rn | head -1 | awk '{print $2}')
    fi
    if [ -n "$KEEPER_PGID" ]; then
        _own_pids | while read pid; do
            PID_PGID=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
            if [ "$PID_PGID" != "$KEEPER_PGID" ]; then
                kill -9 "$pid" 2>/dev/null
            fi
        done
    fi
elif [ "$ZOMBIES" -gt 2 ]; then
    CURR_STATE[zombies]="found"
    ERRORS="${ERRORS}Zombies: ${ZOMBIES} zombie processes detected\n"
else
    CURR_STATE[zombies]="ok"
fi

# ═══════════════════════════════════════════════════════════════════════
# CHECK 7: Guardian worker health (from API response)
# ═══════════════════════════════════════════════════════════════════════
if [ "$API_STATUS" = "200" ]; then
    # Check for disabled or crash-looping modules
    DISABLED_MODULES=$(echo "$HEALTH_JSON" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    g = d.get('data', d).get('guardian', {}).get('modules', {})
    issues = []
    for name, info in g.items():
        state = info.get('state', '')
        restarts = info.get('restart_count', 0)
        if state == 'disabled':
            issues.append(f'{name}=DISABLED')
        elif restarts >= 5:
            issues.append(f'{name}=restarts:{restarts}')
    print(' '.join(issues))
except: pass
" 2>/dev/null)
    if [ -n "$DISABLED_MODULES" ]; then
        CURR_STATE[guardian]="issues"
        ERRORS="${ERRORS}Guardian: ${DISABLED_MODULES}\n"
    else
        CURR_STATE[guardian]="ok"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# STATE TRANSITIONS: detect changes and alert
# ═══════════════════════════════════════════════════════════════════════

# Save current state
> "$STATE_FILE"
for key in "${!CURR_STATE[@]}"; do
    echo "${key}=${CURR_STATE[$key]}" >> "$STATE_FILE"
done

# Detect transitions
for key in "${!CURR_STATE[@]}"; do
    prev="${PREV_STATE[$key]:-unknown}"
    curr="${CURR_STATE[$key]}"
    if [ "$prev" = "ok" ] && [ "$curr" != "ok" ]; then
        # Healthy → Error
        ERRORS="${ERRORS}[${key}] degraded: ${prev} → ${curr}\n"
    elif [ "$prev" != "ok" ] && [ "$prev" != "unknown" ] && [ "$curr" = "ok" ]; then
        # Error → Recovered
        RECOVERIES="${RECOVERIES}[${key}] recovered: ${prev} → ${curr}\n"
    fi
done

# ═══════════════════════════════════════════════════════════════════════
# TELEGRAM ALERTS (only on state changes)
# ═══════════════════════════════════════════════════════════════════════

if [ -n "$ERRORS" ]; then
    MSG="🔴 *${TITAN_ID} Service Alert*
$(echo -e "$ERRORS")
_${NOW}_"
    send_telegram "$MSG"
    log_telemetry "watchdog" "alert_sent" "\"errors\":\"$(echo -e "$ERRORS" | tr '\n' ' ')\""
    echo "[$NOW] ALERT: $ERRORS"
fi

if [ -n "$RECOVERIES" ]; then
    MSG="✅ *${TITAN_ID} Recovered*
$(echo -e "$RECOVERIES")
_${NOW}_"
    send_telegram "$MSG"
    log_telemetry "watchdog" "recovery_sent" "\"recoveries\":\"$(echo -e "$RECOVERIES" | tr '\n' ' ')\""
    echo "[$NOW] RECOVERED: $RECOVERIES"
fi

# ═══════════════════════════════════════════════════════════════════════
# T1 EXTRA: Check T2/T3 watchdog health over local network
# ═══════════════════════════════════════════════════════════════════════

if [ "$CHECK_KIN" = "--check-kin" ] && [ "$TITAN_ID" = "T1" ]; then
    KIN_VPS="10.135.0.6"

    # Check T2 API
    T2_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://${KIN_VPS}:7777/health" 2>/dev/null || echo "000")
    if [ "$T2_STATUS" != "200" ]; then
        # Check if T2 watchdog process exists
        T2_WD_RUNNING=$(ssh -o ConnectTimeout=5 root@${KIN_VPS} 'pgrep -f "t2_watchdog" | head -1' 2>/dev/null || echo "")
        if [ -z "$T2_WD_RUNNING" ]; then
            send_telegram "🔴 *T1→T2 KIN CHECK*
T2 API: ❌ (HTTP ${T2_STATUS})
T2 watchdog: ❌ NOT RUNNING
_Watchdog should auto-recover T2 on next cron cycle_
_${NOW}_"
        else
            send_telegram "🟡 *T1→T2 KIN CHECK*
T2 API: ❌ (HTTP ${T2_STATUS})
T2 watchdog: ✅ running (PID ${T2_WD_RUNNING})
_Watchdog should auto-recover T2_
_${NOW}_"
        fi
    fi

    # Check T3 API
    T3_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://${KIN_VPS}:7778/health" 2>/dev/null || echo "000")
    if [ "$T3_STATUS" != "200" ]; then
        T3_WD_RUNNING=$(ssh -o ConnectTimeout=5 root@${KIN_VPS} 'pgrep -f "t3_watchdog\|t3_manage" | head -1' 2>/dev/null || echo "")
        if [ -z "$T3_WD_RUNNING" ]; then
            send_telegram "🔴 *T1→T3 KIN CHECK*
T3 API: ❌ (HTTP ${T3_STATUS})
T3 watchdog: ❌ NOT RUNNING
_${NOW}_"
        else
            send_telegram "🟡 *T1→T3 KIN CHECK*
T3 API: ❌ (HTTP ${T3_STATUS})
T3 watchdog: ✅ running
_${NOW}_"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# Summary log line
# ═══════════════════════════════════════════════════════════════════════
STATUS_LINE="[$NOW] ${TITAN_ID}: api=${CURR_STATE[api]:-?} teacher=${CURR_STATE[teacher]:-?} persona=${CURR_STATE[persona]:-?} arc=${CURR_STATE[arc]:-?} events=${CURR_STATE[events_teacher]:-?} cgn=${CURR_STATE[cgn]:-?} data=${CURR_STATE[data]:-?} zombies=${CURR_STATE[zombies]:-?} guardian=${CURR_STATE[guardian]:-?}"
echo "$STATUS_LINE"

# Trim log to last 500 lines
LOG_FILE="/tmp/services_watchdog_${TITAN_ID}.log"
if [ -f "$LOG_FILE" ]; then
    tail -500 "$LOG_FILE" > "${LOG_FILE}.tmp" 2>/dev/null && mv "${LOG_FILE}.tmp" "$LOG_FILE" 2>/dev/null
fi
