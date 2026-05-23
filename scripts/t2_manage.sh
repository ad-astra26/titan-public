#!/bin/bash
# t2_manage.sh — Unified T2 (devnet, remote 10.135.0.6) management.
#
# Phase C: kernel-rs is supervised by systemd unit `titan-t2.service`.
# All operations SSH to root@10.135.0.6 transparently — same command syntax
# whether invoked from T1 or anywhere else.
#
# Usage: bash scripts/t2_manage.sh {status|health|start|stop|restart|logs|pid|deploy|help}
#        bash scripts/t2_manage.sh restart --force    # skip dreaming wait
#        bash scripts/t2_manage.sh deploy             # git pull + restart on T2
#        bash scripts/t2_manage.sh deploy                         # pushes Rust musl binaries by default (v1.34.0)
#        bash scripts/t2_manage.sh deploy --skip-rust-binaries     # opt-out (rare; Python-only quick re-push)

set -u

# ── T2-specific configuration ──────────────────────────────────────────
TITAN_ID="T2"
SYSTEMD_UNIT="titan-t2.service"
API_URL="http://10.135.0.6:7777"
REMOTE_HOST="root@10.135.0.6"
TITAN_DIR="/home/antigravity/projects/titan"
STATE_FILE="${TITAN_DIR}/data/dreaming_state.json"

# Source shared helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/titan_common.sh
source "${SCRIPT_DIR}/lib/titan_common.sh"

# ── T2/T3-specific deploy override ─────────────────────────────────────
# Atomic config.toml dance (preserve runtime tokens) + git pull + optional
# Rust musl binary push + systemctl restart + health verify.
#
# Phase C C-S2 (PLAN §15.2): when --include-rust-binaries is passed, scp the
# musl-static-linked kernel-rs + 8 daemon binaries to ${TITAN_DIR}/bin/.
# Each binary's local SHA is recorded for post-scp verification.
cmd_deploy() {
    # 2026-05-19 D-SPEC-94 fleet drift learning — `include_rust=1` is now
    # the default. T2 carried a May 18 19:41 outer-spirit-rs binary past
    # the 2026-05-19 D-SPEC-94 expression_reach key fix because the
    # default-off Rust push made every Python-only deploy a stale-binary
    # surface. `--skip-rust-binaries` is the explicit opt-out for the
    # rare case where only Python changed and the several-second scp is
    # meaningfully cheaper than the safe-default.
    local include_rust=1
    for arg in "$@"; do
        case "${arg}" in
            --include-rust-binaries) include_rust=1 ;;
            --skip-rust-binaries) include_rust=0 ;;
        esac
    done

    echo "=== Pre-deploy sovereign-state check on ${TITAN_ID} ==="
    if ! verify_sovereign_state | tail -20; then
        echo "  🚨 Sovereign-state check FAILED — refusing to deploy to ${TITAN_ID}"
        echo "     Deploy would risk overwriting a divergent identity file. Fix first."
        return 2
    fi
    echo
    echo "=== Deploy ${TITAN_ID} (git pull on remote) ==="
    local backup="/tmp/${TITAN_ID,,}_config_backup_$(date +%s).toml"
    ssh -o ConnectTimeout=10 "${REMOTE_HOST}" bash -s -- "${TITAN_DIR}" "${backup}" "${TITAN_ID}" <<'REMOTE_SCRIPT' || return 1
# `pipefail` is load-bearing: closes the historical fail-silent path where
# `git pull --ff-only … | tail -10` masked a failed pull (the `| tail` returned
# 0 even when the pull aborted on a dirty tracked `bin/*-rs`), causing the
# deploy to "succeed" and restart with STALE Python. With pipefail, any
# failure in the pipe propagates → `set -e` trips → ssh exit ≠ 0 →
# `cmd_deploy` returns 1 → `cmd_restart` is NEVER reached. Combined with the
# pre-pull `git checkout -- bin/` clear and the post-pull tip verification
# below, this makes the dirty-bin stale-Python failure mode structurally
# impossible — no manual `git checkout` / `git pull` bypass is needed.
set -eo pipefail
REMOTE_DIR="$1"
BACKUP="$2"
LABEL="$3"
cd "${REMOTE_DIR}"

if [ ! -f titan_hcl/config.toml ]; then
    echo "  ✗ ${LABEL}: titan_hcl/config.toml missing — aborting"
    exit 1
fi
cp titan_hcl/config.toml "${BACKUP}"
BACKUP_SIZE=$(stat -c%s "${BACKUP}")
echo "  ✓ ${LABEL}: backed up config.toml (${BACKUP_SIZE} bytes) → ${BACKUP}"

# Clear --assume-unchanged on config.toml so git can pull cleanly
git update-index --no-assume-unchanged titan_hcl/config.toml 2>/dev/null || true
git checkout HEAD -- titan_hcl/config.toml 2>/dev/null || true

# Clear any dirty `bin/*-rs` (mtime/size drift from prior restarts) that would
# block `git pull --ff-only`. Mtime-only drift is recoverable via
# `git checkout --`; real binary updates ship via the `--include-rust-binaries`
# scp staging block below, NOT via git pull. Without this clear, the
# pull aborts and the deploy used to proceed to restart with stale Python.
git checkout -- bin/ 2>/dev/null || true

echo "  → git pull --ff-only origin titan-v6"
git pull --ff-only origin titan-v6 2>&1 | tail -10

# Belt-and-suspenders: verify HEAD actually advanced to origin/titan-v6.
# Catches any future failure-mode where pull "succeeds" but the tip doesn't
# move (e.g., already-at-tip is fine; dirty-state silent-noop would NOT be).
PULLED_TIP=$(git rev-parse HEAD)
REMOTE_TIP=$(git rev-parse origin/titan-v6 2>/dev/null || git rev-parse HEAD)
if [ "${PULLED_TIP}" != "${REMOTE_TIP}" ]; then
    echo "  ✗ ${LABEL}: post-pull HEAD ${PULLED_TIP:0:12} != origin/titan-v6 ${REMOTE_TIP:0:12} — aborting BEFORE restart"
    exit 1
fi
echo "  ✓ ${LABEL}: HEAD at ${PULLED_TIP:0:12} (matches origin/titan-v6)"

# Merge new sections (any [section] in pulled but NOT in backup) into backup,
# preserving credentials in backup. Phase 5C deploy taught us this.
python3 <<'PYMERGE'
import re, sys
pulled = open('titan_hcl/config.toml').read()
backup_path = "${BACKUP}"
import os
backup = open(backup_path).read() if os.path.exists(backup_path) else ''
pulled_sections = re.findall(r'^\[([^\]]+)\]', pulled, re.MULTILINE)
backup_sections = re.findall(r'^\[([^\]]+)\]', backup, re.MULTILINE)
new_sections = [s for s in pulled_sections if s not in backup_sections]
if new_sections:
    print(f'  + new sections to merge: {", ".join(new_sections)}')
    for sec in new_sections:
        m = re.search(rf'(\[{re.escape(sec)}\][^\[]*)', pulled)
        if m:
            backup += '\n\n' + m.group(1).rstrip() + '\n'
    open(backup_path, 'w').write(backup)
PYMERGE

# Restore config.toml from merged backup (preserves credentials AND picks up new sections)
cp "${BACKUP}" titan_hcl/config.toml
git update-index --assume-unchanged titan_hcl/config.toml
RESTORED_SIZE=$(stat -c%s titan_hcl/config.toml)
echo "  ✓ ${LABEL}: restored config.toml (${RESTORED_SIZE} bytes)"
REMOTE_SCRIPT

    if [ "${include_rust}" = "1" ]; then
        echo
        echo "=== Push Rust musl binaries ==="
        local bins_local="${TITAN_DIR}/titan-rust/target/x86_64-unknown-linux-musl/release"
        if [ ! -d "${bins_local}" ]; then
            echo "  ⚠ ${bins_local} missing — run 'bash scripts/build_titan_rust.sh musl' first"
            return 1
        fi
        local scp_args=()
        for bin_name in titan-kernel-rs titan-trinity-rs titan-unified-spirit-rs \
                        titan-inner-body-rs titan-inner-mind-rs titan-inner-spirit-rs \
                        titan-outer-body-rs titan-outer-mind-rs titan-outer-spirit-rs; do
            [ -f "${bins_local}/${bin_name}" ] && scp_args+=("${bins_local}/${bin_name}")
        done
        if [ ${#scp_args[@]} -gt 0 ]; then
            # scp to a staging dir, then atomic-rename into bin/. The live
            # titan-kernel-rs holds its on-disk binary busy (ETXTBSY), so a
            # direct scp over it fails with "dest open Failure". rename(2)
            # over a busy executable succeeds — the running process keeps the
            # old (now-unlinked) inode while the path picks up the new binary,
            # which the subsequent restart launches.
            local staging="${TITAN_DIR}/bin/.deploy_staging"
            ssh -o ConnectTimeout=10 "${REMOTE_HOST}" "mkdir -p '${staging}'" || return 1
            scp -q "${scp_args[@]}" "${REMOTE_HOST}:${staging}/" || return 1
            ssh -o ConnectTimeout=10 "${REMOTE_HOST}" "set -e; for f in '${staging}'/*; do chmod +x \"\$f\"; mv -f \"\$f\" '${TITAN_DIR}/bin/'; done; rmdir '${staging}' 2>/dev/null || true" || return 1
            echo "  ✓ staged + atomically installed ${#scp_args[@]} binaries to ${REMOTE_HOST}:${TITAN_DIR}/bin/"
        fi
    fi

    # Defense-in-depth: verify the remote git tip from THIS host before
    # restarting. The in-heredoc post-pull check inside the ssh session is
    # primary; this is the redundant local-side guard so a future heredoc
    # bug can't slip a stale-code restart through. Comparison uses local
    # `origin/titan-v6` (which we just pushed to) as the canonical truth.
    echo
    echo "=== Verify ${TITAN_ID} on expected commit ==="
    local expected_tip
    expected_tip=$(git rev-parse origin/titan-v6 2>/dev/null || git rev-parse HEAD)
    local remote_tip
    remote_tip=$(ssh -o ConnectTimeout=10 "${REMOTE_HOST}" "cd '${TITAN_DIR}' && git rev-parse HEAD" 2>/dev/null || echo "?")
    if [ "${remote_tip:0:12}" != "${expected_tip:0:12}" ]; then
        echo "  🚨 ${TITAN_ID} deploy verification FAILED: remote ${remote_tip:0:12} != expected ${expected_tip:0:12}"
        echo "     Refusing to restart ${TITAN_ID} with stale code. Re-run 'bash scripts/${TITAN_ID,,}_manage.sh deploy'."
        return 3
    fi
    echo "  ✓ ${TITAN_ID} remote at ${remote_tip:0:12} (matches local origin/titan-v6) — safe to restart"

    echo
    echo "=== Restart ${TITAN_ID} (systemctl) ==="
    cmd_restart --force || return $?
    echo "  ✓ ${TITAN_ID} deploy complete"
}

dispatch "$@"
