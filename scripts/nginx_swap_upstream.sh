#!/bin/bash
# B.1 §9 — nginx upstream port swap helper.
#
# Used by titan_plugin/core/shadow_orchestrator.py during phase 4 to
# update /etc/nginx/sites-enabled/iamtitan.tech upstream from OLD_PORT
# to NEW_PORT, validate via `nginx -t`, then reload via `nginx -s reload`.
#
# Usage: sudo bash scripts/nginx_swap_upstream.sh OLD_PORT NEW_PORT
#
# Returns 0 on success, non-zero on:
#   - missing args / bad ports
#   - nginx config not found
#   - sed failed
#   - nginx -t failed (config invalid → revert + abort)
#   - nginx -s reload failed
#
# Safety: backs up current nginx config to /tmp/nginx_iamtitan_pre_swap.conf
# before any mutation. On `nginx -t` failure, revert via cp.

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: sudo bash $0 OLD_PORT NEW_PORT" >&2
    exit 64
fi

OLD_PORT="$1"
NEW_PORT="$2"

if [ -z "$OLD_PORT" ] || [ -z "$NEW_PORT" ]; then
    echo "ERROR: OLD_PORT and NEW_PORT required" >&2
    exit 64
fi

if [ "$OLD_PORT" = "$NEW_PORT" ]; then
    echo "ERROR: OLD_PORT == NEW_PORT (no swap needed)" >&2
    exit 64
fi

CONFIG="/etc/nginx/sites-enabled/iamtitan.tech"
BACKUP="/tmp/nginx_iamtitan_pre_swap.conf"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: $CONFIG not found" >&2
    exit 1
fi

# Backup
cp "$CONFIG" "$BACKUP"

# Replace ONLY local upstream port references for 127.0.0.1 listener.
# We're careful not to mutate the T2/T3 upstream lines (10.135.0.6:7777,
# 10.135.0.6:7778) — only the local 127.0.0.1:OLD_PORT on T1.
# Pattern: proxy_pass http://127.0.0.1:OLD_PORT
sed -i "s|proxy_pass http://127.0.0.1:${OLD_PORT}|proxy_pass http://127.0.0.1:${NEW_PORT}|g" "$CONFIG"

# Validate
if ! nginx -t 2>&1 | grep -qE "syntax is ok|test is successful"; then
    echo "ERROR: nginx -t failed after sed — reverting" >&2
    cp "$BACKUP" "$CONFIG"
    nginx -t  # surface the error to caller
    exit 2
fi

# Reload
if ! nginx -s reload; then
    echo "ERROR: nginx -s reload failed — reverting" >&2
    cp "$BACKUP" "$CONFIG"
    nginx -s reload || true
    exit 3
fi

echo "OK: nginx upstream swapped 127.0.0.1:${OLD_PORT} → 127.0.0.1:${NEW_PORT}"
exit 0
