#!/bin/bash
# t3_manage.sh — Unified T3 (devnet, remote 10.135.0.6) management.
#
# Phase C: kernel-rs is supervised by systemd unit `titan-t3.service`.
# T3 lives at /home/antigravity/projects/titan3/ on 10.135.0.6 (separate
# repo clone from T2's /home/antigravity/projects/titan/).
#
# Usage: bash scripts/t3_manage.sh {status|health|start|stop|restart|logs|pid|deploy|help}
#        bash scripts/t3_manage.sh restart --force    # skip dreaming wait
#        bash scripts/t3_manage.sh deploy             # git pull + restart on T3
#        bash scripts/t3_manage.sh deploy                         # pushes Rust musl binaries by default (v1.34.0)
#        bash scripts/t3_manage.sh deploy --skip-rust-binaries     # opt-out (rare; Python-only quick re-push)

set -u

# ── T3-specific configuration ──────────────────────────────────────────
TITAN_ID="T3"
SYSTEMD_UNIT="titan-t3.service"
API_URL="http://10.135.0.6:7778"
REMOTE_HOST="root@10.135.0.6"
TITAN_DIR="/home/antigravity/projects/titan3"
STATE_FILE="${TITAN_DIR}/data/dreaming_state.json"

# Source shared helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/titan_common.sh
source "${SCRIPT_DIR}/lib/titan_common.sh"

# ── T3-specific deploy override ────────────────────────────────────────
# Same atomic config.toml dance + Rust binary push as T2's deploy, but
# targets T3's directory (/home/antigravity/projects/titan3/).
cmd_deploy() {
    # 2026-05-19 D-SPEC-94 fleet drift learning — `include_rust=1` is now
    # the default. T2 binary drift (May 18 19:41) persisted past the
    # D-SPEC-94 expression_reach Rust fix on 2026-05-19 because the
    # default-off Rust push left every Python-only deploy carrying a
    # stale Rust daemon. `--skip-rust-binaries` is the explicit opt-out
    # for the rare case where only Python code changed and a
    # several-second scp is meaningfully cheaper than the safe-default.
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
set -e
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

git update-index --no-assume-unchanged titan_hcl/config.toml 2>/dev/null || true
git checkout HEAD -- titan_hcl/config.toml 2>/dev/null || true

echo "  → git pull --ff-only origin titan-v6"
git pull --ff-only origin titan-v6 2>&1 | tail -10

python3 <<'PYMERGE'
import re, sys, os
pulled = open('titan_hcl/config.toml').read()
backup_path = "${BACKUP}"
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

cp "${BACKUP}" titan_hcl/config.toml
git update-index --assume-unchanged titan_hcl/config.toml
RESTORED_SIZE=$(stat -c%s titan_hcl/config.toml)
echo "  ✓ ${LABEL}: restored config.toml (${RESTORED_SIZE} bytes)"
REMOTE_SCRIPT

    if [ "${include_rust}" = "1" ]; then
        echo
        echo "=== Push Rust musl binaries ==="
        # T3 builds from /home/antigravity/projects/titan/titan-rust/ (T2 repo
        # is the shared source-of-truth; T3 is a parallel clone that pulls).
        local bins_local="/home/antigravity/projects/titan/titan-rust/target/x86_64-unknown-linux-musl/release"
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

    echo
    echo "=== Restart ${TITAN_ID} (systemctl) ==="
    cmd_restart --force || return $?
    echo "  ✓ ${TITAN_ID} deploy complete"
}

dispatch "$@"
