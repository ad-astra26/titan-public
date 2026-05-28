#!/bin/bash
# titan_common.sh — Shared fleet-management helpers, sourced by t{1,2,3}_manage.sh.
#
# Variables the caller MUST set BEFORE sourcing:
#   TITAN_ID         T1 / T2 / T3
#   SYSTEMD_UNIT     titan-t1.service / titan-t2.service / titan-t3.service
#   API_URL          http://localhost:7777 / http://10.135.0.6:7777 / http://10.135.0.6:7778
#   REMOTE_HOST      empty for local (T1) — "root@10.135.0.6" for T2/T3
#   TITAN_DIR        /home/antigravity/projects/titan (T1/T2) or /home/antigravity/projects/titan3 (T3)
#   STATE_FILE       absolute path to data/dreaming_state.json
#
# Optional vars (have defaults):
#   SUDO_PREFIX      "sudo" for T1 local. Empty for T2/T3 (already root over SSH).
#   POST_RESTART_WAIT_S    seconds to wait after restart before health check (default 12)
#   DREAMING_WAIT_TIMEOUT_S  seconds to wait for natural wake (default 300)
#   DREAMING_POLL_S         polling interval during wait (default 10)
#
# Exit codes (used by cmd_* funcs and propagated by dispatch):
#   0  success
#   1  systemctl / SSH command failed
#   2  post-restart health check failed
#   3  dreaming-wait timed out without natural wake (and no --force)
#   4  bad CLI arguments

set -u

POST_RESTART_WAIT_S="${POST_RESTART_WAIT_S:-12}"
DREAMING_WAIT_TIMEOUT_S="${DREAMING_WAIT_TIMEOUT_S:-300}"
DREAMING_POLL_S="${DREAMING_POLL_S:-10}"
SUDO_PREFIX="${SUDO_PREFIX:-}"

# ── Internal: run a shell command locally or via SSH ────────────────────
# Use this whenever the command needs to execute on the Titan's host.
# Local (T1): runs directly. Remote (T2/T3): wraps in ssh.
_titan_run() {
    if [ -z "${REMOTE_HOST}" ]; then
        bash -c "$*"
    else
        ssh -o ConnectTimeout=10 -o BatchMode=yes "${REMOTE_HOST}" "$*"
    fi
}

# ── Internal: run systemctl with appropriate auth ───────────────────────
# T1 local: `sudo` is needed only for STATE-CHANGING commands (start, stop,
# restart, reload, daemon-reload). Read-only commands (status, show, is-active,
# is-enabled) work without sudo. T2/T3 SSH as root — never need sudo.
_titan_systemctl() {
    local action="$1"
    local needs_sudo=0
    if [ -z "${REMOTE_HOST}" ]; then
        case "${action}" in
            start|stop|restart|reload|daemon-reload|enable|disable|kill|reset-failed)
                needs_sudo=1 ;;
        esac
    fi
    if [ "${needs_sudo}" = "1" ]; then
        ${SUDO_PREFIX} systemctl "$@"
    elif [ -z "${REMOTE_HOST}" ]; then
        systemctl "$@"
    else
        ssh -o ConnectTimeout=10 -o BatchMode=yes "${REMOTE_HOST}" "systemctl $*"
    fi
}

# ── Internal: curl Titan's API ──────────────────────────────────────────
_titan_curl() {
    local path="$1"
    local timeout="${2:-5}"
    curl -sS --max-time "${timeout}" "${API_URL}${path}" 2>&1
}

# ── Public: check if Titan is currently dreaming ────────────────────────
# Prints one of: "True", "False", "unknown"
# Authoritative source: /v4/dreaming endpoint. Falls back to dreaming_state.json
# file's epochs_since_dream (>5 epochs since last dream = awake).
check_dreaming() {
    local resp
    resp=$(_titan_curl /v4/dreaming 10 2>/dev/null)
    if [ -n "${resp}" ]; then
        local api_result
        api_result=$(echo "${resp}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin).get('data', {})
    v = d.get('is_dreaming')
    print('unknown' if v is None else str(v))
except Exception:
    print('unknown')
" 2>/dev/null)
        if [ "${api_result}" = "True" ] || [ "${api_result}" = "False" ]; then
            echo "${api_result}"
            return
        fi
    fi
    # API failed — fall back to state file via the run channel (local for T1,
    # SSH cat for T2/T3) so file-based inference works regardless of locality.
    _titan_run "python3 -c \"
import json
try:
    with open('${STATE_FILE}') as f:
        d = json.load(f)
    epochs = d.get('epochs_since_dream', 0)
    print('False' if epochs > 5 else 'unknown')
except Exception:
    print('unknown')
\"" 2>/dev/null
}

# ── Public: verify sovereign-state invariants (pubkey + sacred files) ──
# Runs scripts/verify_sovereign_state.sh locally (T1) or via SSH (T2/T3).
# Returns the verifier's exit code: 0 OK, 2 pubkey mismatch, 1 missing
# manifest. Callers (cmd_start, cmd_restart, cmd_deploy) should respect
# non-zero and refuse the operation — this catches identity divergence
# BEFORE we touch the bus or push code.
verify_sovereign_state() {
    if [ -z "${REMOTE_HOST}" ]; then
        bash "${TITAN_DIR}/scripts/verify_sovereign_state.sh" --titan "${TITAN_ID}" 2>&1
    else
        ssh -o ConnectTimeout=10 "${REMOTE_HOST}" \
            "bash ${TITAN_DIR}/scripts/verify_sovereign_state.sh --titan ${TITAN_ID}" 2>&1
    fi
}

# ── Public: print VPS resource snapshot for the Titan's host ───────────
# CPU load, memory, swap, disk, fd usage. Runs locally for T1, via SSH
# for T2/T3. Called at start/restart boundaries so resource pressure
# surfaces immediately. ⚠ markers when usage crosses warning thresholds:
#   memory >90%, swap >50%, disk /  >85%, load >cores
print_vps_resources() {
    local label="${1:-snapshot}"
    echo "  ── ${TITAN_ID} VPS resources [${label}] ──"
    _titan_run "
        # CPU
        load=\$(awk '{print \$1\"/\"\$2\"/\"\$3}' /proc/loadavg)
        cores=\$(nproc)
        load1=\$(awk '{print \$1}' /proc/loadavg)
        load1_int=\$(printf '%.0f' \"\$load1\")
        cpu_warn=''
        [ \"\$load1_int\" -gt \"\$cores\" ] && cpu_warn=' ⚠'
        echo \"    CPU load (1/5/15min): \$load (cores=\$cores)\$cpu_warn\"

        # Memory + swap
        free -h | awk '
            /^Mem:/  {used=\$3; tot=\$2; pct=int((\$3*100)/(\$2+0.001)); warn=(pct>90?\" ⚠\":\"\"); printf \"    Memory: %s/%s used (%d%%)%s\n\", used, tot, pct, warn}
            /^Swap:/ {used=\$3; tot=\$2; if(\$2==\"0B\"){printf \"    Swap:   none\n\"} else {pct=int((\$3*100)/(\$2+0.001)); warn=(pct>50?\" ⚠\":\"\"); printf \"    Swap:   %s/%s used (%d%%)%s\n\", used, tot, pct, warn}}
        '

        # Disk on /
        df -h / | awk 'NR==2 {pct=int(\$5); warn=(pct>85?\" ⚠\":\"\"); printf \"    Disk /: %s/%s used (%s)%s\n\", \$3, \$2, \$5, warn}'

        # Total open files across host (kernel-rs heavy fd usage is a Phase C concern)
        nfiles=\$(awk '{print \$1}' /proc/sys/fs/file-nr 2>/dev/null)
        nfiles_max=\$(awk '{print \$3}' /proc/sys/fs/file-nr 2>/dev/null)
        [ -n \"\$nfiles\" ] && echo \"    Open files (host): \$nfiles / \$nfiles_max\"

        # Per-process fd usage for kernel-rs + api_subprocess — warn at >70%, crit at >90%.
        # Phase C added uvicorn api_subprocess as biggest fd consumer; if it hits
        # the systemd LimitNOFILE soft cap, accept() returns EMFILE (the 2026-05-14
        # T2 burst pattern). LimitNOFILE was bumped to 65536 same day.
        main_pid=\$(systemctl show ${SYSTEMD_UNIT} -p MainPID --value 2>/dev/null)
        if [ -n \"\$main_pid\" ] && [ \"\$main_pid\" != \"0\" ]; then
            for pname in main api; do
                if [ \"\$pname\" = \"main\" ]; then
                    pid=\$main_pid
                else
                    # api_subprocess: find python child of main listening on api port
                    pid=\$(pgrep -P \$main_pid python | head -1)
                    # Or the python listening on the API TCP port
                    api_port=\$(echo ${API_URL} | grep -oE '[0-9]+\$')
                    pid_tcp=\$(ss -tlnp 2>/dev/null | grep \":\$api_port \" | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
                    [ -n \"\$pid_tcp\" ] && pid=\$pid_tcp
                fi
                [ -z \"\$pid\" ] || [ ! -d /proc/\$pid ] && continue
                nfds=\$(ls /proc/\$pid/fd 2>/dev/null | wc -l)
                soft=\$(awk '/Max open files/{print \$4}' /proc/\$pid/limits 2>/dev/null)
                if [ -n \"\$soft\" ] && [ \"\$soft\" -gt 0 ]; then
                    pct=\$((nfds * 100 / soft))
                    warn=''
                    if [ \"\$pct\" -gt 90 ]; then warn=' 🚨 critical'
                    elif [ \"\$pct\" -gt 70 ]; then warn=' ⚠ approaching limit'
                    fi
                    printf '    fds (%s pid=%s): %d / %s (%d%%)%s\n' \"\$pname\" \"\$pid\" \"\$nfds\" \"\$soft\" \"\$pct\" \"\$warn\"
                fi
            done
        fi
    " 2>/dev/null
}

# ── Public: poll /health until it returns 200 OR timeout ────────────────
# Returns 0 if healthy within timeout, 2 otherwise.
wait_for_health() {
    local timeout="${1:-30}"
    local elapsed=0
    while [ "${elapsed}" -lt "${timeout}" ]; do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "${API_URL}/health" 2>/dev/null || echo "000")
        if [ "${code}" = "200" ]; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 2
}

# ────────────────────────────────────────────────────────────────────────
# Commands (cmd_*)
# ────────────────────────────────────────────────────────────────────────

cmd_status() {
    echo "=== ${TITAN_ID} systemd status (${SYSTEMD_UNIT}) ==="
    _titan_systemctl status "${SYSTEMD_UNIT}" --no-pager 2>&1 | head -10 || return 1
    echo
    echo "=== ${TITAN_ID} API probe ==="
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${API_URL}/health" 2>/dev/null || echo "000")
    if [ "${code}" = "200" ]; then
        echo "  ✓ ${API_URL}/health → 200"
    else
        echo "  ✗ ${API_URL}/health → ${code}"
        return 1
    fi
}

cmd_health() {
    local resp readiness
    resp=$(_titan_curl /health 8)
    # Module-running count comes from /v6/readiness (Phase 11 §11.I.5 — reads
    # the authoritative module_<name>_state.bin slots + orchestrator roster).
    # The legacy /health v3.guardian_status field is the vestigial monolith-era
    # guardian view (reports every module 'stopped' under Phase 11) — NOT used.
    readiness=$(_titan_curl /v6/readiness 8)
    READINESS_JSON="${readiness}" python3 -c "
import sys, json, os
try:
    d = json.load(sys.stdin).get('data', {})
except Exception as e:
    print(f'  ✗ Failed to parse /health: {e}')
    sys.exit(1)
v3 = d.get('v3', {})
sub = d.get('subsystems', {})
active = sum(1 for _,s in sub.items() if s == 'ACTIVE')
bus = v3.get('bus_stats', {})
v = d.get('vault', {})
# Authoritative module count from /v6/readiness.
mod_line = '  Modules: (readiness unavailable)'
try:
    rj = json.loads(os.environ.get('READINESS_JSON') or '{}')
    rd = rj.get('data', rj)
    summ = rd.get('module_state_summary', {}) or {}
    running = rd.get('module_running_count', summ.get('running', 0))
    fleet = rd.get('fleet') or {}
    expected = (fleet.get('mandatory_total', 0) or 0) + (fleet.get('post_boot_total', 0) or 0)
    extra = []
    for k in ('booted','starting','unhealthy_or_crashed','not_booted'):
        if summ.get(k):
            extra.append(f'{k}={summ[k]}')
    suffix = ('  [' + ', '.join(extra) + ']') if extra else ''
    mod_line = f'  Modules: {running}/{expected or running} running (boot_phase={fleet.get(\"boot_phase\",\"?\")}){suffix}'
except Exception as e:
    mod_line = f'  Modules: (readiness parse error: {e})'
print(f'  Status: {d.get(\"status\")}')
print(f'  Boot time: {v3.get(\"boot_time\")}s')
print(mod_line)
print(f'  Subsystems: {active}/{len(sub)} ACTIVE')
print(f'  Bus: pub={bus.get(\"published\",\"?\")} routed={bus.get(\"routed\",\"?\")} dropped={bus.get(\"dropped\",\"?\")}')
print(f'  Vault: commits={v.get(\"commit_count\",\"?\")}, sovereignty={v.get(\"sovereignty_pct\",\"?\")}%')
print(f'  SOL: {d.get(\"sol_balance\",\"?\")}')
" <<< "${resp}"
}

cmd_start() {
    echo "=== Starting ${TITAN_ID} ==="
    echo
    echo "=== Sovereign-state pre-check ==="
    if ! verify_sovereign_state | tail -20; then
        echo "  🚨 Sovereign-state check FAILED — refusing to start ${TITAN_ID}"
        echo "     Run 'bash scripts/verify_sovereign_state.sh --titan ${TITAN_ID}' for full report."
        return 2
    fi
    echo
    print_vps_resources "pre-start"
    _titan_systemctl start "${SYSTEMD_UNIT}" || return 1
    echo "  systemctl start issued — waiting up to ${POST_RESTART_WAIT_S}s for /health..."
    if wait_for_health "${POST_RESTART_WAIT_S}"; then
        echo "  ✓ ${TITAN_ID} responsive"
        print_vps_resources "post-start"
    else
        echo "  ✗ ${TITAN_ID} did not respond within ${POST_RESTART_WAIT_S}s"
        print_vps_resources "post-start (DEGRADED)"
        return 2
    fi
}

cmd_stop() {
    echo "=== Stopping ${TITAN_ID} (graceful) ==="
    _titan_systemctl stop "${SYSTEMD_UNIT}" || return 1
    echo "  ✓ ${TITAN_ID} stopped"
}

cmd_restart() {
    local force="${1:-}"
    if [ "${force}" != "--force" ]; then
        echo "=== Pre-restart check: is ${TITAN_ID} dreaming? ==="
        local is_dreaming
        is_dreaming=$(check_dreaming)
        echo "  is_dreaming = ${is_dreaming}"
        if [ "${is_dreaming}" = "True" ]; then
            echo "  ⚠ ${TITAN_ID} is DREAMING — waiting up to ${DREAMING_WAIT_TIMEOUT_S}s for natural wake"
            local waited=0
            while [ "${waited}" -lt "${DREAMING_WAIT_TIMEOUT_S}" ]; do
                sleep "${DREAMING_POLL_S}"
                waited=$((waited + DREAMING_POLL_S))
                is_dreaming=$(check_dreaming)
                if [ "${is_dreaming}" = "False" ]; then
                    echo "  ✓ woke after ${waited}s — proceeding with restart"
                    break
                fi
                echo "  [t+${waited}s] still dreaming..."
            done
            if [ "${is_dreaming}" = "True" ]; then
                echo "  ✗ still dreaming after ${DREAMING_WAIT_TIMEOUT_S}s — restart skipped (pass --force to override)"
                return 3
            fi
        fi
    else
        echo "=== --force flag set: skipping dreaming check ==="
    fi
    echo "=== Sovereign-state pre-check ==="
    if ! verify_sovereign_state | tail -20; then
        echo "  🚨 Sovereign-state check FAILED — refusing to restart ${TITAN_ID}"
        echo "     Run 'bash scripts/verify_sovereign_state.sh --titan ${TITAN_ID}' for full report."
        return 2
    fi
    echo
    echo "=== Restarting ${TITAN_ID} (systemctl) ==="
    print_vps_resources "pre-restart"
    _titan_systemctl restart "${SYSTEMD_UNIT}" || return 1
    echo "  systemctl restart issued — waiting up to ${POST_RESTART_WAIT_S}s for /health..."
    if wait_for_health "${POST_RESTART_WAIT_S}"; then
        echo "  ✓ ${TITAN_ID} responsive after restart"
        print_vps_resources "post-restart"
        cmd_health
    else
        echo "  ✗ ${TITAN_ID} did not respond within ${POST_RESTART_WAIT_S}s — check logs"
        print_vps_resources "post-restart (DEGRADED)"
        return 2
    fi
}

cmd_resources() {
    print_vps_resources "snapshot"
}

cmd_verify() {
    verify_sovereign_state
}

cmd_logs() {
    echo "=== ${TITAN_ID} live logs (Ctrl-C to exit) ==="
    # journalctl -u for unit-scoped logs is readable without sudo for any
    # user with appropriate group membership (systemd-journal / adm).
    # On both T1 (antigravity in adm group) and T2/T3 (root) this is fine.
    if [ -z "${REMOTE_HOST}" ]; then
        journalctl -u "${SYSTEMD_UNIT}" -f --no-pager
    else
        ssh -t "${REMOTE_HOST}" "journalctl -u ${SYSTEMD_UNIT} -f --no-pager"
    fi
}

cmd_pid() {
    local pid
    pid=$(_titan_systemctl show "${SYSTEMD_UNIT}" -p MainPID --value 2>/dev/null | tr -d '[:space:]')
    if [ -n "${pid}" ] && [ "${pid}" != "0" ]; then
        echo "${pid}"
    else
        echo "  (not running)" >&2
        return 1
    fi
}

# ── cmd_deploy is overridden in t2_manage.sh / t3_manage.sh ─────────────
# Default stub for T1 (which is the source of truth — no deploy needed).
cmd_deploy() {
    echo "  ✗ deploy not supported for ${TITAN_ID} (this Titan IS the source of truth)" >&2
    return 4
}

# ────────────────────────────────────────────────────────────────────────
# Dispatch
# ────────────────────────────────────────────────────────────────────────

print_help() {
    cat <<EOF
Usage: bash $(basename "$0") {status|health|resources|verify|start|stop|restart|logs|pid|deploy|help}

Commands:
  status     systemd status + API probe
  health     /health summary (modules running, bus, vault, SOL)
  resources  VPS resource snapshot (CPU load, mem, swap, disk, fd count)
  verify     run sovereign-state verifier (pubkey + sacred-file invariants)
  start      systemctl start ${SYSTEMD_UNIT} (+ sovereign pre-check + resource snapshot)
  stop       systemctl stop ${SYSTEMD_UNIT}
  restart    sovereign pre-check + dreaming-aware restart + /health verify
             (pass '--force' to skip the dreaming wait)
  logs       journalctl -f
  pid        print kernel-rs main PID
  deploy     (T2/T3 only) sovereign pre-check + git pull + optional Rust binary push + restart
  help       this help

Exit codes:
  0  success    1  systemctl/SSH failed    2  health check failed
  3  dreaming-wait timed out    4  bad CLI args
EOF
}

dispatch() {
    local cmd="${1:-help}"
    shift || true
    case "${cmd}" in
        status)    cmd_status "$@" ;;
        health)    cmd_health "$@" ;;
        resources) cmd_resources "$@" ;;
        verify)    cmd_verify "$@" ;;
        start)   cmd_start "$@" ;;
        stop)    cmd_stop "$@" ;;
        restart) cmd_restart "$@" ;;
        logs)    cmd_logs "$@" ;;
        pid)     cmd_pid "$@" ;;
        deploy)  cmd_deploy "$@" ;;
        help|-h|--help) print_help ;;
        *)
            echo "  ✗ unknown command: ${cmd}" >&2
            print_help >&2
            return 4
            ;;
    esac
}
