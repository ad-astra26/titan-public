#!/bin/bash
# t1_manage.sh — Unified T1 (mainnet, localhost) management.
#
# Phase C: kernel-rs is supervised by systemd unit `titan-t1.service`.
# Python plugin + 8 Rust daemons + L2/L3 workers are kernel-rs children.
# All start/stop/restart go through systemctl (NOT pkill).
#
# Usage: bash scripts/t1_manage.sh {status|health|start|stop|restart|logs|pid|help}
#        bash scripts/t1_manage.sh restart --force    # skip dreaming wait
#
# T1 has no `deploy` command — T1 IS the source of truth (this repo lives here).
# To deploy code to T2/T3, use `bash scripts/t2_manage.sh deploy` and
# `bash scripts/t3_manage.sh deploy`.

set -u

# ── T1-specific configuration ──────────────────────────────────────────
TITAN_ID="T1"
SYSTEMD_UNIT="titan-t1.service"
API_URL="http://localhost:7777"
REMOTE_HOST=""    # local
TITAN_DIR="/home/antigravity/projects/titan"
STATE_FILE="${TITAN_DIR}/data/dreaming_state.json"
SUDO_PREFIX="sudo"  # T1 systemd unit runs as antigravity → systemctl needs sudo

# ── Source shared helpers + dispatch ───────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/titan_common.sh
source "${SCRIPT_DIR}/lib/titan_common.sh"

dispatch "$@"
