#!/bin/bash
# deploy_t2.sh — Git-based code deployment to T2/T3 VPS
# T2 and T3 are git clones of titan-dev with config.toml --assume-unchanged
# data/ is gitignored — never touched by git pull
#
# Robust config.toml handling:
#   The --assume-unchanged flag can drift (e.g. after a manual git checkout
#   or git update-index reset). When that happens, git pull would fail with
#   a merge conflict on config.toml. This script ALWAYS does the safe dance:
#     1. backup config.toml to /tmp (preserves runtime tokens like auth_session)
#     2. clear --assume-unchanged so git can take upstream cleanly
#     3. checkout the HEAD copy of config.toml (drops local edits)
#     4. git pull --ff-only origin titan-v6 (pulled config.toml now on disk)
#     4b. MERGE new sections from pulled into backup — any section that exists
#         in pulled but NOT in backup is appended to backup, so config
#         additions like new [voice] or [feature.x] make it onto T2/T3
#         naturally. Existing sections + credentials in backup are preserved
#         untouched. (Added 2026-04-20 after Phase 5C deploy missed the new
#         [voice] section; see rFP_phase5_narrator_evolution §9.)
#     5. restore config.toml from the /tmp backup (now merged, preserves
#        credentials AND picks up new upstream sections)
#     6. re-set --assume-unchanged
#     7. verify the restored byte size matches the (possibly-grown) backup
#
# This is idempotent and safe even when --assume-unchanged is already lost.

set -e

T2_HOST="root@10.135.0.6"
TITAN_DIR="/home/antigravity/projects/titan"
T3_DIR="/home/antigravity/projects/titan3"

# ── Phase C C-S2 (PLAN §15.2): --include-rust-binaries flag ─────────
# When set, scp titan-rust musl static binaries (titan-kernel-rs +
# titan-trinity-rs-placeholder) to T2 and T3's bin/ directory after the
# git pull, with SHA verification. Build first if local musl binaries
# are missing. Filter out the flag so positional management commands
# ($1) like --restart still work for the rest of the script.
INCLUDE_RUST=0
FILTERED_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --include-rust-binaries) INCLUDE_RUST=1 ;;
        *) FILTERED_ARGS+=("$arg") ;;
    esac
done
set -- "${FILTERED_ARGS[@]}"

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

# Phase C C-S2 (PLAN §17.2 / BUG-DEPLOY-T3-WIPES-LOCAL-EDITS-20260428):
# protect titan_params.toml + config.toml from being wiped by deploy.
# `--skip-worktree` tells git to ignore worktree changes; both `git
# checkout --` and `git pull` will leave the file alone. Idempotent —
# safe to re-run on every deploy.
git update-index --skip-worktree titan_plugin/titan_params.toml 2>/dev/null || true

# 4. Reset any other local code edits, then fast-forward pull. Files
# marked --skip-worktree (or --assume-unchanged "h") are filtered out so
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

# 4b. MERGE new upstream sections into backup (preserves credentials + picks up
# config additions like a new [voice] or [feature.x] section on next deploy)
# Section detection: any TOML line starting with [...] — matches top-level
# ([voice]) AND nested ([voice.sub]) sections uniformly.
PULLED_CFG="titan_plugin/config.toml"
NEW_SECTIONS=$(comm -23 \
    <(grep -oE '^\[[^]]+\]' "${PULLED_CFG}" | sort -u) \
    <(grep -oE '^\[[^]]+\]' "${BACKUP}" | sort -u))
if [ -n "${NEW_SECTIONS}" ]; then
    NEW_COUNT=$(echo "${NEW_SECTIONS}" | grep -c .)
    echo "  ℹ ${LABEL}: ${NEW_COUNT} new section(s) in upstream config — merging into backup:"
    echo "${NEW_SECTIONS}" | sed "s/^/      /"
    {
        echo ""
        echo "# ─────────────────────────────────────────────────────────────"
        echo "# Auto-merged from upstream titan-v6 by deploy_t2.sh"
        echo "# at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# ─────────────────────────────────────────────────────────────"
        while IFS= read -r section; do
            [ -z "${section}" ] && continue
            # Extract section block: from ^[section] until next ^[...] or EOF
            # Normalize each line by stripping inline comments + trailing WS
            # so `[voice]  # master switch` still matches the section `[voice]`.
            awk -v sec="${section}" '
                { line=$0; sub(/[[:space:]]*#.*$/, "", line); gsub(/[[:space:]]+$/, "", line) }
                line == sec { in_sec=1; print $0; next }
                in_sec && line ~ /^\[[^]]+\]/ { in_sec=0 }
                in_sec { print $0 }
            ' "${PULLED_CFG}"
        done <<< "${NEW_SECTIONS}"
    } >> "${BACKUP}"
    BACKUP_SIZE=$(stat -c%s "${BACKUP}")  # refresh for post-restore size check
    echo "  ✓ ${LABEL}: merged ${NEW_COUNT} new section(s) — backup now ${BACKUP_SIZE} bytes"
else
    echo "  ℹ ${LABEL}: no new upstream sections to merge"
fi

# 5. Restore config.toml from the /tmp backup (now includes any merged sections)
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

# ── Pre-deploy: verify local is clean + pushed ───────────────────
LOCAL_COMMIT=$(git rev-parse HEAD)
LOCAL_SHORT=$(git rev-parse --short HEAD)
UNCOMMITTED=$(git diff --name-only -- ':(exclude)data/' | wc -l)
STAGED=$(git diff --cached --name-only | wc -l)

if [ "$UNCOMMITTED" -gt 0 ] || [ "$STAGED" -gt 0 ]; then
    echo "⚠ WARNING: ${UNCOMMITTED} uncommitted + ${STAGED} staged changes (excluding data/)"
    echo "  T2/T3 will NOT get these changes. Commit first!"
    echo "  Proceeding with deploy of committed code only..."
    echo ""
fi

echo "=== Pushing local changes to titan-dev ==="
git push origin titan-v6 2>/dev/null && echo "✓ Pushed to titan-dev" || echo "⚠ Push failed or nothing to push"

# ── Deploy ────────────────────────────────────────────────────────
deploy_one "${T2_HOST}" "${TITAN_DIR}" "T2"
deploy_one "${T2_HOST}" "${T3_DIR}" "T3"

# ── Optional: ship Phase C Rust binaries (PLAN §15.2) ─────────────
deploy_rust_binaries() {
    local host="$1"
    local remote_dir="$2"
    local label="$3"
    local bins_local
    bins_local="$(dirname "$0")/../titan-rust/target/x86_64-unknown-linux-musl/release"

    # Build first if needed
    if [ ! -x "${bins_local}/titan-kernel-rs" ] || [ ! -x "${bins_local}/titan-trinity-rs-placeholder" ]; then
        echo "  [${label}] building Rust binaries (musl static)..."
        bash "$(dirname "$0")/build_titan_rust.sh" musl
    fi

    echo "  [${label}] copying Rust binaries to ${host}:${remote_dir}/bin/"
    ssh "${host}" "mkdir -p \"${remote_dir}/bin\""
    scp -q \
        "${bins_local}/titan-kernel-rs" \
        "${bins_local}/titan-trinity-rs-placeholder" \
        "${host}:${remote_dir}/bin/"

    # SHA verification — proves no in-flight tamper + correct file landed
    for bin_name in titan-kernel-rs titan-trinity-rs-placeholder; do
        local local_sha remote_sha
        local_sha=$(sha256sum "${bins_local}/${bin_name}" | awk '{print $1}')
        remote_sha=$(ssh "${host}" "sha256sum \"${remote_dir}/bin/${bin_name}\"" 2>/dev/null | awk '{print $1}')
        if [ "${local_sha}" != "${remote_sha}" ]; then
            echo "  ✗ ${label}: ${bin_name} SHA MISMATCH (local=${local_sha:0:12} remote=${remote_sha:0:12})" >&2
            exit 1
        fi
        echo "  ✓ ${label}: ${bin_name} sha256=${local_sha:0:12}…"
    done
}

if [ "$INCLUDE_RUST" -eq 1 ]; then
    echo ""
    echo "=== Shipping Rust binaries (--include-rust-binaries) ==="
    deploy_rust_binaries "${T2_HOST}" "${TITAN_DIR}" "T2"
    deploy_rust_binaries "${T2_HOST}" "${T3_DIR}" "T3"
fi

# ── Post-deploy: verify remote commits match local ────────────────
echo ""
echo "=== Verifying commit alignment ==="
T2_COMMIT=$(ssh "${T2_HOST}" "cd ${TITAN_DIR} && git rev-parse HEAD" 2>/dev/null)
T3_COMMIT=$(ssh "${T2_HOST}" "cd ${T3_DIR} && git rev-parse HEAD" 2>/dev/null)
T2_SHORT=$(echo "$T2_COMMIT" | cut -c1-7)
T3_SHORT=$(echo "$T3_COMMIT" | cut -c1-7)

ALL_MATCH=true
if [ "$T2_COMMIT" = "$LOCAL_COMMIT" ]; then
    echo "  ✓ T2 at ${T2_SHORT} — matches local"
else
    echo "  ✗ T2 at ${T2_SHORT} — LOCAL is ${LOCAL_SHORT} (MISMATCH!)"
    ALL_MATCH=false
fi
if [ "$T3_COMMIT" = "$LOCAL_COMMIT" ]; then
    echo "  ✓ T3 at ${T3_SHORT} — matches local"
else
    echo "  ✗ T3 at ${T3_SHORT} — LOCAL is ${LOCAL_SHORT} (MISMATCH!)"
    ALL_MATCH=false
fi

if [ "$ALL_MATCH" = false ] && [[ "$1" == "--restart"* ]]; then
    echo ""
    echo "  ⚠ COMMIT MISMATCH — restarting anyway, but T2/T3 may run stale code."
    echo "  ⚠ Did you forget to commit? Check: git status"
fi

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
    echo "  bash scripts/deploy_t2.sh --include-rust-binaries"
    echo "                                            Combine with any cmd above to ship"
    echo "                                            titan-kernel-rs + titan-trinity-rs-placeholder"
    echo "                                            (musl static) to T2/T3 bin/ post-pull (Phase C C-S2)"
fi
