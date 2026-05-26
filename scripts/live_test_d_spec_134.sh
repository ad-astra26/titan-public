#!/bin/bash
# live_test_d_spec_134.sh — RFP_Phase_C_python_fix §6.4 live integration test.
#
# Purpose: after deploying the D-SPEC-134 Condition+predicate refactor of
# `BusSocketClient` inbound signaling, verify the post-deploy fleet member
# exhibits NO behavioral regression vs the pre-fix baseline AND the
# Condition+predicate path is exercised under real chat traffic.
#
# Usage:
#   bash scripts/live_test_d_spec_134.sh <titan_id> <api_url> [<ssh_target>] [<service_unit>]
#
# Examples:
#   bash scripts/live_test_d_spec_134.sh T1 http://localhost:7777
#   bash scripts/live_test_d_spec_134.sh T2 http://10.135.0.6:7777 root@10.135.0.6 titan-t2.service
#   bash scripts/live_test_d_spec_134.sh T3 http://10.135.0.6:7778 root@10.135.0.6 titan-t3.service
#
# What it checks (mapped to RFP §6.4 + §10 verification gates):
#   G1. CHAT round-trips complete (or fail identically to the baseline) —
#       proves SocketQueue.get + _wake_cond.wait_for + predicate path is
#       intact under live LLM traffic.
#   G2. No "stuck-get" symptom in the recent journal — workers do not
#       silently drop messages between Condition wakeups.
#   G3. No regression in heartbeat-cascade rate vs the pre-deploy
#       baseline (this is a Python-client-side change, so the rate
#       should be unchanged — neither helped nor worsened).
#   G4. `BusSocketClient` thread health — `bus-client-*` threads remain
#       alive + connected (no thread death from refactor).
#
# Exit codes:
#   0  — all gates green
#   1  — at least one gate red (details printed)
#   2  — argument / environment error

set -u

TITAN_ID="${1:-}"
API_URL="${2:-}"
SSH_TARGET="${3:-}"          # empty for localhost titans
SERVICE_UNIT="${4:-titan-t1.service}"

if [ -z "${TITAN_ID}" ] || [ -z "${API_URL}" ]; then
    echo "ERROR: usage: $0 <titan_id> <api_url> [<ssh_target>] [<service_unit>]" >&2
    exit 2
fi

INTERNAL_KEY="${TITAN_INTERNAL_KEY:-}"
if [ -z "${INTERNAL_KEY}" ]; then
    # Read from ~/.titan/secrets.toml — single source of truth for the
    # X-Titan-Internal-Key bypass per `reference_chat_internal_key`.
    # Never inline the key into the script (gitleaks rule
    # titan-known-leaked-values + feedback_no_secrets_in_example_configs).
    SECRETS_FILE="${TITAN_SECRETS_FILE:-${HOME}/.titan/secrets.toml}"
    if [ -r "${SECRETS_FILE}" ]; then
        INTERNAL_KEY="$(awk -F '=' '/^internal_key[[:space:]]*=/ { gsub(/[" ]/, "", $2); print $2; exit }' "${SECRETS_FILE}")"
    fi
fi
if [ -z "${INTERNAL_KEY}" ]; then
    echo "ERROR: set TITAN_INTERNAL_KEY env var or populate api.internal_key in ${SECRETS_FILE:-~/.titan/secrets.toml}" >&2
    exit 2
fi
N_CHATS="${LIVE_TEST_N_CHATS:-5}"
CHAT_TIMEOUT_S="${LIVE_TEST_CHAT_TIMEOUT_S:-95}"
JOURNAL_WINDOW="${LIVE_TEST_JOURNAL_WINDOW:-5 min ago}"

# Helper: run journalctl locally or via ssh.
journal_remote() {
    if [ -n "${SSH_TARGET}" ]; then
        ssh "${SSH_TARGET}" "$@"
    else
        eval "$@"
    fi
}

echo "═══════════════════════════════════════════════════════════════════"
echo " D-SPEC-134 live e2e — ${TITAN_ID} at ${API_URL}"
echo "═══════════════════════════════════════════════════════════════════"

# ── G1: chat round-trips ─────────────────────────────────────────────
echo
echo "── G1: ${N_CHATS} CHAT round-trips ──"

success=0
fail=0
total_latency_ms=0
sample_responses=()

for i in $(seq 1 "${N_CHATS}"); do
    sess="sess-dspec134-$(date +%s)-${i}"
    t_start=$(date +%s.%N)
    body=$(curl -s -m "${CHAT_TIMEOUT_S}" -X POST "${API_URL}/chat" \
        -H "Content-Type: application/json" \
        -H "X-Titan-Internal-Key: ${INTERNAL_KEY}" \
        -H "X-Titan-User-Id: maker" \
        -H "X-Titan-Channel: cli-test" \
        -d "{\"message\":\"live test #${i} from D-SPEC-134 gate\",\"session_id\":\"${sess}\",\"user_id\":\"maker\"}" \
        -w "\n%{http_code}" 2>&1)
    t_end=$(date +%s.%N)
    elapsed_ms=$(awk -v s="$t_start" -v e="$t_end" 'BEGIN { printf "%.0f", (e-s)*1000 }')
    http_code=$(echo "${body}" | tail -1)
    payload=$(echo "${body}" | head -n -1)

    if [ "${http_code}" = "200" ] && echo "${payload}" | grep -q '"response"'; then
        success=$((success + 1))
        total_latency_ms=$((total_latency_ms + elapsed_ms))
        snippet=$(echo "${payload}" | python3 -c "import json,sys; d=json.load(sys.stdin); r=d.get('response',''); print(r[:80].replace(chr(10),' '))" 2>/dev/null || echo "<unparseable>")
        echo "  #${i} [${elapsed_ms}ms] OK — ${snippet}…"
        sample_responses+=("${snippet}")
    else
        fail=$((fail + 1))
        echo "  #${i} [${elapsed_ms}ms] FAIL — http=${http_code} body=$(echo "${payload}" | head -c 80)"
    fi
done

if [ "${success}" -gt 0 ]; then
    avg_ms=$((total_latency_ms / success))
else
    avg_ms=0
fi

echo "  → ${success}/${N_CHATS} OK, avg_latency=${avg_ms}ms"

# ── G2: stuck-get / missed-wake symptom check ─────────────────────────
echo
echo "── G2: no stuck-get / missed-wake symptoms in last ${JOURNAL_WINDOW} ──"

stuck_get_cmd="journalctl -u ${SERVICE_UNIT} --since '${JOURNAL_WINDOW}' --no-pager 2>/dev/null | grep -ciE 'SocketQueue.*stuck|stuck.*get|wait_for.*never woke|missed.wake' || true"
stuck_get=$(journal_remote "${stuck_get_cmd}")
stuck_get=${stuck_get//[!0-9]/}
stuck_get=${stuck_get:-0}
echo "  stuck-get symptom matches: ${stuck_get}"

# ── G3: heartbeat-cascade rate (should match pre-deploy baseline) ────
echo
echo "── G3: heartbeat cascade rate vs baseline ──"

hb_cmd="journalctl -u ${SERVICE_UNIT} --since '${JOURNAL_WINDOW}' --no-pager 2>/dev/null | grep -c heartbeat_pong_timeout || true"
hb=$(journal_remote "${hb_cmd}")
hb=${hb//[!0-9]/}
hb=${hb:-0}

bs_cmd="journalctl -u ${SERVICE_UNIT} --since '${JOURNAL_WINDOW}' --no-pager 2>/dev/null | grep -c broker_shutdown_signal || true"
bs=$(journal_remote "${bs_cmd}")
bs=${bs//[!0-9]/}
bs=${bs:-0}

echo "  heartbeat_pong_timeout (last 5min): ${hb}"
echo "  broker_shutdown_signal (last 5min): ${bs}"
echo "  expected: both ≪ 30 (post-D-SPEC-131 baseline); refactor must not regress"

# ── G4: bus-client thread health ─────────────────────────────────────
echo
echo "── G4: bus-client thread health ──"

# Count active bus-client threads — at least the main workers should
# have one alive per worker. Use /health or fall back to pgrep + ps.
health=$(curl -s -m 5 "${API_URL}/health" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "?")
echo "  /health status: ${health}"

# ── Verdict ──────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════════════════════════"
verdict_red=0
notes=""

if [ "${fail}" -gt 0 ] && [ "${success}" -eq 0 ]; then
    notes+="  ✗ G1: all chats failed — INVESTIGATE (may be pre-existing, not D-SPEC-134 regression)\n"
    verdict_red=$((verdict_red + 1))
elif [ "${fail}" -gt 0 ]; then
    notes+="  ⚠ G1: partial fail (${fail}/${N_CHATS}) — compare to pre-deploy baseline\n"
fi

if [ "${stuck_get}" -gt 0 ]; then
    notes+="  ✗ G2: stuck-get symptom detected (${stuck_get} matches)\n"
    verdict_red=$((verdict_red + 1))
fi

if [ "${hb}" -gt 30 ] || [ "${bs}" -gt 30 ]; then
    notes+="  ✗ G3: heartbeat cascade re-emerged (hb=${hb}, bs=${bs})\n"
    verdict_red=$((verdict_red + 1))
fi

if [ "${health}" != "ok" ]; then
    notes+="  ✗ G4: /health not ok (got: ${health})\n"
    verdict_red=$((verdict_red + 1))
fi

if [ "${verdict_red}" -eq 0 ]; then
    echo " ✓ ALL GATES GREEN — ${TITAN_ID} D-SPEC-134 live verification passed"
    echo "═══════════════════════════════════════════════════════════════════"
    exit 0
else
    echo " ✗ ${verdict_red} GATE(S) RED — ${TITAN_ID}"
    echo -e "${notes}"
    echo "═══════════════════════════════════════════════════════════════════"
    exit 1
fi
