#!/bin/bash
# deploy_t3.sh — T3 deployment script
#
# Two modes:
#   (default)         Birth experiment bootstrap — creates a CLEAN T3 instance
#                     at /home/antigravity/projects/titan3 with NO accumulated
#                     state. Uses rsync. Use --start to also start T3.
#   --restart         T3-only code update via git pull + restart. Uses the
#                     same atomic config.toml backup→pull→restore dance as
#                     deploy_t2.sh, robust to --assume-unchanged drift.
#                     This is the day-to-day code update path for T3 only;
#                     when you need to update both T2 and T3 use
#                     `deploy_t2.sh --restart` instead (it handles both).
#   --update-only     T3 code update without restart (rare — most updates
#                     should also restart so the new code takes effect).
set -e

T2_HOST="root@10.135.0.6"
T3_DIR="/home/antigravity/projects/titan3"
T1_DIR="/home/antigravity/projects/titan"

# ── Phase C C-S2 (PLAN §15.2): --include-rust-binaries flag ─────────
# When set, scp titan-rust musl static binaries (titan-kernel-rs +
# titan-trinity-rs-placeholder) to T3's bin/ directory after the git
# pull, with SHA verification. Filter out the flag so positional
# commands ($1) like --restart still work for the rest of the script.
INCLUDE_RUST=0
FILTERED_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --include-rust-binaries) INCLUDE_RUST=1 ;;
        *) FILTERED_ARGS+=("$arg") ;;
    esac
done
set -- "${FILTERED_ARGS[@]}"

deploy_rust_binaries_t3() {
    local bins_local
    bins_local="$(dirname "$0")/../titan-rust/target/x86_64-unknown-linux-musl/release"

    if [ ! -x "${bins_local}/titan-kernel-rs" ] || [ ! -x "${bins_local}/titan-trinity-rs-placeholder" ]; then
        echo "  [T3] building Rust binaries (musl static)..."
        bash "$(dirname "$0")/build_titan_rust.sh" musl
    fi

    echo "  [T3] copying Rust binaries to ${T2_HOST}:${T3_DIR}/bin/"
    ssh "${T2_HOST}" "mkdir -p \"${T3_DIR}/bin\""
    scp -q \
        "${bins_local}/titan-kernel-rs" \
        "${bins_local}/titan-trinity-rs-placeholder" \
        "${T2_HOST}:${T3_DIR}/bin/"

    for bin_name in titan-kernel-rs titan-trinity-rs-placeholder; do
        local local_sha remote_sha
        local_sha=$(sha256sum "${bins_local}/${bin_name}" | awk '{print $1}')
        remote_sha=$(ssh "${T2_HOST}" "sha256sum \"${T3_DIR}/bin/${bin_name}\"" 2>/dev/null | awk '{print $1}')
        if [ "${local_sha}" != "${remote_sha}" ]; then
            echo "  ✗ T3: ${bin_name} SHA MISMATCH (local=${local_sha:0:12} remote=${remote_sha:0:12})" >&2
            exit 1
        fi
        echo "  ✓ T3: ${bin_name} sha256=${local_sha:0:12}…"
    done
}

# ── Helper: T3 git-based code update with config.toml safety dance ──
# Mirrors deploy_t2.sh's deploy_one() but T3-only. Uses a single SSH session
# so backup→pull→restore is atomic. Verifies byte-size match after restore.
deploy_t3_update() {
    local backup="/tmp/T3_config_backup_$(date +%s).toml"
    echo "=== Pushing local changes to titan-dev ==="
    git push origin titan-v6 2>/dev/null && echo "✓ Pushed to titan-dev" || echo "⚠ Push failed or nothing to push"
    echo ""
    echo "=== Deploying to T3 (git pull) ==="
    ssh "${T2_HOST}" bash -s -- "${T3_DIR}" "${backup}" "T3" <<'REMOTE_SCRIPT'
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

# Phase C C-S2 (PLAN §17.2 / BUG-DEPLOY-T3-WIPES-LOCAL-EDITS-20260428):
# protect titan_params.toml + config.toml from being wiped by deploy.
# `--skip-worktree` tells git to ignore worktree changes; both `git
# checkout --` and `git pull` will leave the file alone. Idempotent —
# safe to re-run on every deploy.
git update-index --skip-worktree titan_plugin/titan_params.toml 2>/dev/null || true

# 4. Reset other local code edits, then fast-forward pull. Files marked
# --skip-worktree (or --assume-unchanged "h") are filtered out so
# legitimate Maker edits (titan_params.toml flag flips, etc.) survive
# deploys. config.toml is also filtered (already handled by step 5
# backup→restore).
SKIP_WORKTREE_FILES=$(git ls-files -v 2>/dev/null \
    | awk '$1 == "S" || $1 == "h" {sub(/^[a-zA-Z] /,""); print}')
{
    echo "titan_plugin/config.toml"
    [ -n "$SKIP_WORKTREE_FILES" ] && echo "$SKIP_WORKTREE_FILES"
} | sort -u > /tmp/${LABEL}_skip_reset.lst
git diff --name-only \
    | grep -vxFf /tmp/${LABEL}_skip_reset.lst \
    | xargs -r git checkout -- 2>/dev/null || true
rm -f /tmp/${LABEL}_skip_reset.lst
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
    echo "✓ T3 code updated"
}

# ── --restart: T3-only update + restart ──
if [[ "$1" == "--restart" ]]; then
    deploy_t3_update
    if [ "$INCLUDE_RUST" -eq 1 ]; then
        echo ""
        echo "=== Shipping Rust binaries (--include-rust-binaries) ==="
        deploy_rust_binaries_t3
    fi
    echo ""
    echo "=== Restarting T3 ==="
    ssh "${T2_HOST}" "bash ${T3_DIR}/scripts/t3_manage.sh restart"
    exit 0
fi

# ── --update-only: T3-only code update WITHOUT restart ──
if [[ "$1" == "--update-only" ]]; then
    deploy_t3_update
    if [ "$INCLUDE_RUST" -eq 1 ]; then
        echo ""
        echo "=== Shipping Rust binaries (--include-rust-binaries) ==="
        deploy_rust_binaries_t3
    fi
    echo ""
    echo "ℹ Code updated but NOT restarted — new code takes effect on next restart."
    exit 0
fi

# ── Default: birth experiment bootstrap ──
echo "════════════════════════════════════════"
echo "  T3 BIRTH EXPERIMENT DEPLOYMENT"
echo "════════════════════════════════════════"
echo ""
echo "ℹ This is the BIRTH bootstrap path (rsync, fresh state)."
echo "ℹ For day-to-day code updates use: bash scripts/deploy_t3.sh --restart"
echo ""

# ── 1. Create T3 directory structure on T2 VPS ──
echo "=== Creating T3 directory structure ==="
ssh $T2_HOST "mkdir -p ${T3_DIR}/{data,scripts}"

# ── 2. Sync code (plugin + scripts, NO data) ──
echo "=== Syncing code ==="
rsync -avz --delete \
    --exclude='data/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='cognee_data/' \
    --exclude='node_modules/' \
    --exclude='test_env/' \
    --exclude='.git/' \
    --exclude='.next/' \
    --exclude='titan-docs/' \
    --exclude='titan-observatory/' \
    --exclude='titan-docs-site/' \
    --exclude='programs/' \
    --exclude='*.jpg' \
    --exclude='*.png' \
    "${T1_DIR}/titan_plugin/" \
    "${T2_HOST}:${T3_DIR}/titan_plugin/"

rsync -avz \
    --exclude='__pycache__/' \
    "${T1_DIR}/scripts/titan_main.py" \
    "${T2_HOST}:${T3_DIR}/scripts/"

rsync -avz \
    "${T1_DIR}/scripts/t3_manage.sh" \
    "${T1_DIR}/scripts/t3_watchdog.sh" \
    "${T2_HOST}:${T3_DIR}/scripts/"

# ── 3. Create T3-specific config (port 7778) ──
echo "=== Creating T3 config (port 7778) ==="
# Copy config and modify port
scp "${T1_DIR}/titan_plugin/config.toml" "${T2_HOST}:${T3_DIR}/titan_plugin/config.toml"
ssh $T2_HOST "sed -i 's/^port = 7777/port = 7778/' ${T3_DIR}/titan_plugin/config.toml"

# Copy titan_params.toml (this IS the DNA — identical architecture, fresh start)
scp "${T1_DIR}/titan_plugin/titan_params.toml" "${T2_HOST}:${T3_DIR}/titan_plugin/titan_params.toml"

# ── 4. Create minimal empty data directories ──
echo "=== Creating clean data directories ==="
ssh $T2_HOST "mkdir -p ${T3_DIR}/data/{neural_nervous_system,neuromodulator,mini_reasoning,reasoning,interpreter,logs,telemetry,backups,media_queue,neural_nervous_system}"

# ── 5. Create empty state files that spirit_worker expects ──
echo "=== Creating birth state files ==="
ssh $T2_HOST "cat > ${T3_DIR}/data/dreaming_state.json << 'EOF'
{\"is_dreaming\": false, \"metabolic_drain\": 0.0, \"dream_cycle\": 0}
EOF"

ssh $T2_HOST "cat > ${T3_DIR}/data/neuromodulator/neuromodulator_state.json << 'EOF'
{}
EOF"

ssh $T2_HOST "cat > ${T3_DIR}/data/neural_nervous_system/hormonal_state.json << 'EOF'
{}
EOF"

ssh $T2_HOST "cat > ${T3_DIR}/data/pi_heartbeat_state.json << 'EOF'
{\"total_epochs\": 0, \"pi_events\": 0, \"clusters\": 0}
EOF"

ssh $T2_HOST "cat > ${T3_DIR}/data/anchor_state.json << 'EOF'
{\"total_anchors\": 0}
EOF"

# ── 6. Set up watchdog cron ──
echo "=== Setting up watchdog cron ==="
ssh $T2_HOST "
# Add T3 watchdog cron if not already present
if ! crontab -l 2>/dev/null | grep -q 't3_watchdog'; then
    (crontab -l 2>/dev/null; echo '*/5 * * * * bash ${T3_DIR}/scripts/t3_watchdog.sh >> /tmp/titan3_watchdog.log 2>&1') | crontab -
    echo 'Watchdog cron installed'
else
    echo 'Watchdog cron already exists'
fi
"

echo ""
echo "════════════════════════════════════════"
echo "  T3 DEPLOYMENT COMPLETE"
echo "════════════════════════════════════════"
echo ""
echo "T3 Location: ${T3_DIR}"
echo "T3 Port:     7778"
echo "T3 Log:      /tmp/titan3_brain.log"
echo "T3 Kin:      T2 on localhost:7777"
echo "Watchdog:    Every 5 min (auto-restart + telemetry)"
echo ""

if [[ "$1" == "--start" ]]; then
    echo "=== Starting T3 ==="
    ssh $T2_HOST "bash ${T3_DIR}/scripts/t3_manage.sh start"
fi
