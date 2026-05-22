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

# ── Phase C C-S7 (2026-05-05): --include-rust-binaries flag ─────────
# When set, scp ALL 9 titan-rust musl static binaries to T3's bin/
# directory after the git pull, with per-binary SHA verification.
#
# Pre-2026-05-05: only shipped 2 binaries (kernel-rs +
# trinity-rs-placeholder) per the C-S2 era when the daemon binaries
# didn't exist yet. C-S5 + C-S6 + Phase C activation surfaced that
# unified-spirit + 6 trinity daemons (inner/outer × body/mind/spirit)
# also need to reach T3. Now ships the full fleet.
#
# Filter out the flag so positional commands ($1) like --restart still
# work for the rest of the script.
INCLUDE_RUST=0
FILTERED_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --include-rust-binaries) INCLUDE_RUST=1 ;;
        *) FILTERED_ARGS+=("$arg") ;;
    esac
done
set -- "${FILTERED_ARGS[@]}"

# Phase C C-S7 fleet — 9 binaries shipped to each Titan's bin/.
# Per SPEC §9.A naming. titan-trinity-rs replaces the C-S2-era
# titan-trinity-rs-placeholder (real substrate ships in C-S3+).
TITAN_RUST_FLEET=(
    titan-kernel-rs
    titan-trinity-rs
    titan-unified-spirit-rs
    titan-inner-body-rs
    titan-inner-mind-rs
    titan-inner-spirit-rs
    titan-outer-body-rs
    titan-outer-mind-rs
    titan-outer-spirit-rs
)

deploy_rust_binaries_t3() {
    local bins_local
    bins_local="$(dirname "$0")/../titan-rust/target/x86_64-unknown-linux-musl/release"

    # Verify every binary present locally. If any missing, build the
    # whole workspace (cheap incremental — cached crates are skipped).
    local missing=0
    for bin_name in "${TITAN_RUST_FLEET[@]}"; do
        if [ ! -x "${bins_local}/${bin_name}" ]; then
            missing=1
            break
        fi
    done
    if [ "${missing}" = "1" ]; then
        echo "  [T3] building Rust binaries (musl static, full workspace)..."
        bash "$(dirname "$0")/build_titan_rust.sh" musl
    fi

    echo "  [T3] copying ${#TITAN_RUST_FLEET[@]} Rust binaries to ${T2_HOST}:${T3_DIR}/bin/"
    ssh "${T2_HOST}" "mkdir -p \"${T3_DIR}/bin\""

    # Pre-clear ETXTBSY: when titan-t3.service is running, the kernel
    # holds the binaries' text segments mmapped, and `scp` opens with
    # O_TRUNC which Linux refuses on busy text files (silent "dest open
    # Failure"). `rm -f` works regardless — Linux unlinks busy text
    # files cleanly via inode refcount: the running process keeps the
    # OLD binary via its open file descriptor; scp creates the new file
    # at a new inode, ready for the next exec.
    # Codified after 2026-05-08 deploy hit ETXTBSY on all 9 binaries
    # (feedback_deploy_t3_orphan_files_and_etxtbsy.md).
    ssh "${T2_HOST}" "rm -f \"${T3_DIR}/bin/\"titan-*-rs 2>/dev/null || true"

    # Single scp invocation for all binaries (one TCP setup, faster).
    local scp_args=()
    for bin_name in "${TITAN_RUST_FLEET[@]}"; do
        scp_args+=("${bins_local}/${bin_name}")
    done
    scp -q "${scp_args[@]}" "${T2_HOST}:${T3_DIR}/bin/"

    # Per-binary SHA verify — bail loud on any mismatch.
    for bin_name in "${TITAN_RUST_FLEET[@]}"; do
        local local_sha remote_sha
        local_sha=$(sha256sum "${bins_local}/${bin_name}" | awk '{print $1}')
        remote_sha=$(ssh "${T2_HOST}" "sha256sum \"${T3_DIR}/bin/${bin_name}\"" 2>/dev/null | awk '{print $1}')
        if [ "${local_sha}" != "${remote_sha}" ]; then
            echo "  ✗ T3: ${bin_name} SHA MISMATCH (local=${local_sha:0:12} remote=${remote_sha:0:12})" >&2
            exit 1
        fi
        echo "  ✓ T3: ${bin_name} sha256=${local_sha:0:12}…"
    done

    # Permissions: bin owned by antigravity:antigravity (matches T1 convention).
    # Under l0_rust=true T3 systemd unit currently runs as User=root per Gap F
    # operational fix (PLAN §2B); root can read regardless. Future Phase D
    # cleanup standardizes on User=antigravity fleet-wide.
    ssh "${T2_HOST}" "chown antigravity:antigravity \"${T3_DIR}/bin/\"titan-*-rs 2>/dev/null || true"
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
if [ ! -f titan_hcl/config.toml ]; then
    echo "  ✗ ${LABEL}: titan_hcl/config.toml missing — aborting"
    exit 1
fi
cp titan_hcl/config.toml "${BACKUP}"
BACKUP_SIZE=$(stat -c%s "${BACKUP}")
echo "  ✓ ${LABEL}: backed up config.toml (${BACKUP_SIZE} bytes) → ${BACKUP}"

# 2. Clear --assume-unchanged so git can resolve config.toml on its own
git update-index --no-assume-unchanged titan_hcl/config.toml 2>/dev/null || true

# 3. Drop any local config.toml edits (we'll restore from backup after pull)
git checkout -- titan_hcl/config.toml 2>/dev/null || true

# Phase C C-S2 (PLAN §17.2 / BUG-DEPLOY-T3-WIPES-LOCAL-EDITS-20260428):
# protect titan_params.toml + config.toml from being wiped by deploy.
# `--skip-worktree` tells git to ignore worktree changes; both `git
# checkout --` and `git pull` will leave the file alone. Idempotent —
# safe to re-run on every deploy.
git update-index --skip-worktree titan_hcl/titan_params.toml 2>/dev/null || true

# 4. Reset other local code edits, then fast-forward pull. Files marked
# --skip-worktree (or --assume-unchanged "h") are filtered out so
# legitimate Maker edits (titan_params.toml flag flips, etc.) survive
# deploys. config.toml is also filtered (already handled by step 5
# backup→restore).
SKIP_WORKTREE_FILES=$(git ls-files -v 2>/dev/null \
    | awk '$1 == "S" || $1 == "h" {sub(/^[a-zA-Z] /,""); print}')
{
    echo "titan_hcl/config.toml"
    [ -n "$SKIP_WORKTREE_FILES" ] && echo "$SKIP_WORKTREE_FILES"
} | sort -u > /tmp/${LABEL}_skip_reset.lst
git diff --name-only \
    | grep -vxFf /tmp/${LABEL}_skip_reset.lst \
    | xargs -r git checkout -- 2>/dev/null || true

# 4b. Discard STAGED-BUT-NOT-COMMITTED changes (M/A in index) that would
# block ff-pull. These accumulate when prior deploys crashed mid-stream
# or were interrupted. `git checkout origin/titan-v6 --` forces both
# index and working tree to match upstream for these paths. Filter the
# skip list (config.toml + skip-worktree files) so we don't fight that
# dance. Codified after 2026-05-08 deploy hit ROADMAP.md +
# rFP_x_voice_enrichment.md staged blockers.
STAGED_BLOCKERS=$(git diff --name-only --cached 2>/dev/null \
    | grep -vxFf /tmp/${LABEL}_skip_reset.lst 2>/dev/null || true)
if [ -n "${STAGED_BLOCKERS}" ]; then
    echo "  ${LABEL}: discarding ${LABEL} stale staged changes (forcing to upstream titan-v6):"
    echo "${STAGED_BLOCKERS}" | sed 's/^/    /'
    git fetch origin titan-v6 2>/dev/null || true
    echo "${STAGED_BLOCKERS}" | xargs -r git checkout origin/titan-v6 -- 2>/dev/null || true
fi

# 4c. Auto-clean UNTRACKED files that duplicate incoming tracked paths
# (would block pull with "untracked working tree files would be
# overwritten by merge"). Only safe when the local content is
# byte-identical to upstream. Codified after 2026-05-08 hit untracked
# social_x/ duplicates from a prior partial deploy.
git fetch origin titan-v6 2>/dev/null || true
INCOMING=$(git diff --name-only --diff-filter=A HEAD..origin/titan-v6 2>/dev/null || true)
for f in ${INCOMING}; do
    [ -f "$f" ] || continue
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        continue  # already tracked, handled by other steps
    fi
    # Untracked-but-incoming. Compare byte-for-byte against upstream.
    if git show "origin/titan-v6:$f" 2>/dev/null | cmp -s - "$f" 2>/dev/null; then
        rm -f "$f"
    fi
done
# After cleaning byte-identical dupes, drop any leftover empty dirs.
find titan_hcl titan-docs scripts data -type d -empty -delete 2>/dev/null || true

rm -f /tmp/${LABEL}_skip_reset.lst
git pull --ff-only origin titan-v6 2>&1 | tail -10

# 5. Restore config.toml from the /tmp backup
cp "${BACKUP}" titan_hcl/config.toml
RESTORED_SIZE=$(stat -c%s titan_hcl/config.toml)

# 6. Re-set --assume-unchanged so future pulls don't see config.toml as modified
git update-index --assume-unchanged titan_hcl/config.toml

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

# ── Phase C T3 detection (2026-05-06) ──
# When titan-t3.service is installed AND kernel-rs binary is present,
# T3 runs under systemd-managed kernel-rs (NOT legacy `python titan_hcl`).
# In that mode, `t3_manage.sh start/restart` is harmful — it would launch
# a SECOND, non-supervised legacy Python that holds data/titan_hcl.pid,
# causing the kernel-rs-spawned titan_hcl to abort with "Another
# titan_hcl is already running". Use `systemctl restart titan-t3` instead;
# the unit's ExecStopPost+ExecStartPre cleanup hooks handle pid/shm/socket
# hygiene. See feedback_phase_c_crash_diagnosis_chain.md for the failure
# mode this guards against.
is_t3_phase_c() {
    ssh "${T2_HOST}" '
        systemctl list-unit-files titan-t3.service 2>/dev/null | \
            grep -q "titan-t3.service" && \
        [ -x /home/antigravity/projects/titan3/bin/titan-kernel-rs ]
    ' >/dev/null 2>&1
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
    if is_t3_phase_c; then
        echo "=== Restarting T3 (Phase C — systemctl + kernel-rs) ==="
        # `systemctl restart` runs ExecStop → ExecStopPost (cleanup) →
        # ExecStartPre (cleanup again, defensive) → ExecStart (kernel-rs).
        # No legacy Python involvement; kernel-rs spawns titan_hcl itself.
        ssh "${T2_HOST}" 'systemctl restart titan-t3.service && sleep 8 && systemctl is-active titan-t3.service'
    else
        echo "=== Restarting T3 (legacy — t3_manage.sh) ==="
        ssh "${T2_HOST}" "bash ${T3_DIR}/scripts/t3_manage.sh restart"
    fi
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

# ── Default mode: same as --update-only ──
#
# 2026-05-10 cleanup: the legacy rsync-based "BIRTH BOOTSTRAP" path was
# removed. Per `feedback_t2t3_deployment_via_git_pull.md` (memory): T2/T3
# deploy via git pull, NEVER rsync/scp. The bootstrap path was a one-time
# fresh-T3 setup from the T3 birth experiment (Mar 2026); it has been dead
# code since T3 separated into its own repo. Worse, it overwrote
# config.toml + sed-patched the port back to 7778 which fought with the
# update_code_t3() backup→pull→restore dance, AND it kept references to
# t3_watchdog.sh which was retired 2026-05-09 (commit 9f4b5389) — so any
# bare `bash scripts/deploy_t3.sh` invocation was rsync-erroring out
# halfway through and silently leaving T3's port reset to 7777.
#
# Default behaviour now == `--update-only`: safe git-pull update + leave
# restart to the operator. Use `--restart` for atomic update+restart.
echo ""
echo "ℹ No flag given — defaulting to --update-only (safe git-pull mode)."
echo "ℹ Use `--restart` for atomic update+restart, or `--update-only` to be explicit."
echo ""
deploy_t3_update
if [ "$INCLUDE_RUST" -eq 1 ]; then
    echo ""
    echo "=== Shipping Rust binaries (--include-rust-binaries) ==="
    deploy_rust_binaries_t3
fi
echo ""
echo "ℹ Code updated but NOT restarted — new code takes effect on next restart."
echo "ℹ Restart with: ssh root@10.135.0.6 'systemctl restart titan-t3.service'"
exit 0
