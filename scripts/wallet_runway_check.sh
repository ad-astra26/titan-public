#!/usr/bin/env bash
# rFP_backup_worker I6 — Daily Irys runway check, independent of backup execution.
#
# Runs via cron (once daily). Queries /v4/backup/wallet-runway, writes result
# to data/telemetry/wallet_runway_{date}.json for audit trail, and fires a
# Telegram alert if the tier transitioned into yellow/orange/red since the
# prior day.
#
# Cron (T1 only — T2/T3 devnet have no Irys deposit):
#   0 9 * * * cd /home/antigravity/projects/titan && . test_env/bin/activate \\
#             && bash scripts/wallet_runway_check.sh \\
#             >> /tmp/titan_wallet_runway.log 2>&1

set -euo pipefail

cd "$(dirname "$0")/.."

DATE=$(date -u +%Y-%m-%d)
OUT="data/telemetry/wallet_runway_${DATE}.json"
mkdir -p data/telemetry

# Query the running Titan's endpoint (T1 by default; port 7777)
PORT="${TITAN_PORT:-7777}"
URL="http://127.0.0.1:${PORT}/v4/backup/wallet-runway"

echo "[$(date -u +%H:%M:%S)] wallet_runway_check: querying $URL"

RESULT="$(curl -s -m 30 "$URL" || echo '{"ok":false,"error":"curl_failed"}')"
echo "$RESULT" > "$OUT"

# Parse tier + previous tier for transition detection
TIER=$(echo "$RESULT" | python3 -c 'import sys, json
try:
    d = json.load(sys.stdin).get("data", {})
    print(d.get("tier") or "unknown")
except Exception:
    print("parse_error")
')
DAYS=$(echo "$RESULT" | python3 -c 'import sys, json
try:
    d = json.load(sys.stdin).get("data", {})
    print(f"{d.get(\"days_runway\", 0):.1f}")
except Exception:
    print("?")
')
SOL=$(echo "$RESULT" | python3 -c 'import sys, json
try:
    d = json.load(sys.stdin).get("data", {})
    print(f"{d.get(\"irys_sol\", 0):.4f}")
except Exception:
    print("?")
')

echo "[$(date -u +%H:%M:%S)] tier=$TIER irys=$SOL SOL runway=${DAYS}d"

# Find prior day's tier to detect transitions
PRIOR_FILE=$(ls -t data/telemetry/wallet_runway_*.json 2>/dev/null | grep -v "${DATE}" | head -n 1 || echo "")
PRIOR_TIER="none"
if [ -n "$PRIOR_FILE" ] && [ -f "$PRIOR_FILE" ]; then
    PRIOR_TIER=$(python3 -c '
import sys, json
try:
    with open(sys.argv[1]) as f:
        d = json.load(f).get("data", {})
    print(d.get("tier") or "unknown")
except Exception:
    print("unknown")
' "$PRIOR_FILE")
fi

echo "[$(date -u +%H:%M:%S)] prior_tier=$PRIOR_TIER"

# Fire Telegram alert on tier transition (or any non-green standing daily reminder)
if [ "$TIER" != "green" ] && [ "$TIER" != "unknown" ] && [ "$TIER" != "parse_error" ]; then
    # Transition or daily standing alert
    if [ "$TIER" != "$PRIOR_TIER" ] || [ "$TIER" = "red" ]; then
        echo "[$(date -u +%H:%M:%S)] Firing maker_notify (transition: $PRIOR_TIER → $TIER)"
        python3 -c "
from titan_plugin.utils.maker_notify import notify_maker, format_runway_alert
import os, json
titan_id = 'T1'
if os.path.exists('data/titan_identity.json'):
    with open('data/titan_identity.json') as f:
        titan_id = json.load(f).get('titan_id', 'T1')
ok = notify_maker(
    'runway_daily_${TIER}', titan_id,
    format_runway_alert('${TIER}', float('${SOL}'), float('${DAYS}')),
    cooldown_s=43200,  # 12h — independent daily check
)
print('notify_sent=' + str(ok))
"
    fi
fi

echo "[$(date -u +%H:%M:%S)] wallet_runway_check: done"
