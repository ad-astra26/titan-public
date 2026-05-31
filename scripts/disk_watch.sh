#!/usr/bin/env bash
# disk_watch.sh — page via Telegram when any fleet host crosses 80% / 90% disk.
# Converts the manual "df -h at session start" ritual (which depends on the human
# remembering — T1 hit 90% on 2026-05-29) into an automatic alert. Reuses the
# EXISTING Telegram bot + cooldown pattern from observatory_watchdog.sh; the token
# is sourced from that script (not duplicated) so no new secret enters the tree.
# Wire as cron (*/30) next to the observatory_watchdog line.
set -euo pipefail
REPO="/home/antigravity/projects/titan"
# Pull the already-committed Telegram credentials without re-hardcoding them.
eval "$(grep -E '^TELEGRAM_(BOT_TOKEN|CHAT_ID)=' "$REPO/scripts/observatory_watchdog.sh")"
COOLDOWN_SECONDS=3600

send_telegram() {
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="$TELEGRAM_CHAT_ID" -d text="$1" >/dev/null 2>&1 || true
}

check_host() {  # $1=name  $2=use_pct
    local name="$1"
    local use="$2"
    local cd="/tmp/disk_watch_alerted_${name}"
    [ "${use:-0}" -ge 80 ] || return 0
    local tier="80%"; [ "$use" -ge 90 ] && tier="90% 🔴"
    local now last=0; now=$(date +%s); [ -f "$cd" ] && last=$(cat "$cd" 2>/dev/null || echo 0)
    if [ $((now - last)) -ge "$COOLDOWN_SECONDS" ]; then
        send_telegram "🚨 Titan disk — ${name} at ${use}% (>=${tier}). Run scripts/titan_hygiene.sh / see project_t1_mainnet_disk_recurring_fill."
        echo "$now" > "$cd"
    fi
}

# T1 (local)
check_host T1 "$(df -P / | awk 'NR==2{print +$5}')"
# VPS (hosts T2+T3) — bounded so a network blip can't hang cron
VPS=$(ssh -o ConnectTimeout=10 -o BatchMode=yes root@10.135.0.6 "df -P / | awk 'NR==2{print +\$5}'" 2>/dev/null || echo 0)
[ "${VPS:-0}" -gt 0 ] && check_host VPS "$VPS"
