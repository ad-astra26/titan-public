#!/bin/bash
# deploy_t2.sh — Git-based code deployment to T2/T3 VPS
# T2 and T3 are git clones of titan-dev with config.toml --assume-unchanged
# data/ is gitignored — never touched by git pull
#
# Robust config.toml handling:
#   The --assume-unchanged flag can drift (e.g. after a manual git checkout
#   or git update-index reset). When that happens, git pull would fail with
#   a merge conflict on config.toml. This script ALWAYS does the safe dance:
#     1. backup config.toml to /tmp
#     2. clear --assume-unchanged so git can take upstream cleanly
#     3. checkout the HEAD copy of config.toml (drops local edits)
#     4. git pull --ff-only origin titan-v6
#     5. restore config.toml from the /tmp backup (preserves runtime tokens)
#     6. re-set --assume-unchanged
#     7. verify the restored byte size matches the backup
#
# This is idempotent and safe even when --assume-unchanged is already lost.

set -e

T2_HOST="root@10.135.0.6"
TITAN_DIR="/home/antigravity/projects/titan"
T3_DIR="/home/antigravity/projects/titan3"

# ── Helper: deploy one Titan dir on the VPS ───────────────────────
# Uses a single SSH session per Titan so the backup→pull→restore is atomic.
# Args: $1=ssh-host  $2=remote-titan-dir  $3=label
deploy_one() {
    local host="$1"
    local dir="$2"
    local label="$3"
    local backup="/tmp/${label}_config_backup_$(date +%s).toml"
    echo ""
    echo "=== Deploying to ${label} (git pull) ==="
    ssh "${host}" bash -s -- "${dir}" "${backup}" "${label}" <<'REMOTE_SCRIPT'
set -e
REMOTE_DIR="$1"
BACKUP="$2"
LABEL="$3"
cd "${REMOTE_DIR}"

# 1. Backup local config.toml (preserves runtime tokens like Twitter auth_session)
if [ ! -f titan_plugin/config.toml ]; then
    echo "  ✗ ${LABEL}: titan_plugin/config.toml missing — aborting"
    exit 1
fi
cp titan_plugin/config.toml "${BACKUP}"
BACKUP_SIZE=$(stat -c%s "${BACKUP}")
echo "  ✓ ${LABEL}: backed up config.toml (${BACKUP_SIZE} bytes) → ${BACKUP}"

# 2. Clear --assume-unchanged so git can resolve config.toml on its own
git update-index --no-assume-unchanged titan_plugin/config.toml 2>/dev/null || true

# 3. Drop any local config.toml edits (we'll restore from backup after pull)
git checkout -- titan_plugin/config.toml 2>/dev/null || true

# 4. Reset any other local code edits, then fast-forward pull
# (filter out config.toml from the reset list as a belt-and-suspenders measure)
git diff --name-only | grep -v "^titan_plugin/config.toml$" | xargs -r git checkout -- 2>/dev/null || true
git pull --ff-only origin titan-v6 2>&1 | tail -10

# 5. Restore config.toml from the /tmp backup
cp "${BACKUP}" titan_plugin/config.toml
RESTORED_SIZE=$(stat -c%s titan_plugin/config.toml)

# 6. Re-set --assume-unchanged so future pulls don't see config.toml as modified
git update-index --assume-unchanged titan_plugin/config.toml

# 7. Sanity check: backup size must match restored size
if [ "${BACKUP_SIZE}" != "${RESTORED_SIZE}" ]; then
    echo "  ✗ ${LABEL}: SIZE MISMATCH after restore (backup=${BACKUP_SIZE}, restored=${RESTORED_SIZE})"
    echo "  ✗ ${LABEL}: backup preserved at ${BACKUP} for manual recovery"
    exit 1
fi

echo "  ✓ ${LABEL}: config.toml restored (${RESTORED_SIZE} bytes) + assume-unchanged re-set"
echo "  ✓ ${LABEL}: now at $(git log --oneline -1)"
REMOTE_SCRIPT
    echo "✓ ${label} code updated"
}

# ── Ensure local changes are pushed to titan-dev ──────────────────
echo "=== Pushing local changes to titan-dev ==="
git push origin titan-v6 2>/dev/null && echo "✓ Pushed to titan-dev" || echo "⚠ Push failed or nothing to push"

# ── Deploy ────────────────────────────────────────────────────────
deploy_one "${T2_HOST}" "${TITAN_DIR}" "T2"
deploy_one "${T2_HOST}" "${T3_DIR}" "T3"

echo ""
echo "=== Deploy complete (config.toml preserved via backup→pull→restore) ==="

# ── T2/T3 Management ──────────────────────────────────────────────
T2_MANAGE="${TITAN_DIR}/scripts/t2_manage.sh"
T3_MANAGE="${T3_DIR}/scripts/t3_manage.sh"

if [[ "$1" == "--restart" ]]; then
    echo "Restarting T2..."
    ssh ${T2_HOST} "bash ${T2_MANAGE} restart"
    echo "Restarting T3..."
    ssh ${T2_HOST} "bash ${T3_MANAGE} restart"
elif [[ "$1" == "--restart-t2" ]]; then
    ssh ${T2_HOST} "bash ${T2_MANAGE} restart"
elif [[ "$1" == "--restart-t3" ]]; then
    ssh ${T2_HOST} "bash ${T3_MANAGE} restart"
elif [[ "$1" == "--stop" ]]; then
    ssh ${T2_HOST} "bash ${T2_MANAGE} stop"
    ssh ${T2_HOST} "bash ${T3_MANAGE} stop"
elif [[ "$1" == "--start" ]]; then
    ssh ${T2_HOST} "bash ${T2_MANAGE} start"
    ssh ${T2_HOST} "bash ${T3_MANAGE} start"
elif [[ "$1" == "--status" ]]; then
    echo "=== T2 ===" && ssh ${T2_HOST} "bash ${T2_MANAGE} status"
    echo "=== T3 ===" && ssh ${T2_HOST} "bash ${T3_MANAGE} status"
elif [[ "$1" == "--log" ]]; then
    echo "=== T2 ===" && ssh ${T2_HOST} "bash ${T2_MANAGE} log ${2:-30}"
    echo "=== T3 ===" && ssh ${T2_HOST} "bash ${T3_MANAGE} log ${2:-30}"
else
    echo "Deploy + management commands:"
    echo "  bash scripts/deploy_t2.sh               Deploy code to T2+T3 (git pull)"
    echo "  bash scripts/deploy_t2.sh --restart      Deploy + restart T2+T3"
    echo "  bash scripts/deploy_t2.sh --restart-t2   Deploy + restart T2 only"
    echo "  bash scripts/deploy_t2.sh --restart-t3   Deploy + restart T3 only"
    echo "  bash scripts/deploy_t2.sh --stop         Stop T2+T3"
    echo "  bash scripts/deploy_t2.sh --start        Start T2+T3"
    echo "  bash scripts/deploy_t2.sh --status       Status of T2+T3"
    echo "  bash scripts/deploy_t2.sh --log [N]      Last N log lines"
fi
