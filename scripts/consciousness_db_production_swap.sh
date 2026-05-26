#!/usr/bin/env bash
# consciousness_db_production_swap.sh — promote the migrated f32 consciousness.db
# (per `migrate_consciousness_db_blob_quantize.py`, commit `5906cdd1`) into
# production with full safety net: backup → migrate → smoke-verify → atomic
# swap → restart Titan → post-swap smoke.
#
# Per Maker call 2026-05-26: ship the long-standing P0.5/P0.6 deferred
# consciousness.db production swap.
#
# Per `directive_memory_preservation`: NEVER delete Titan data. This script
# moves the legacy DB to `data/consciousness.db.legacy_TEXT_<ts>` so it
# remains recoverable even after the swap completes.
#
# Usage (per-Titan, T3→T2→T1 cascade per `feedback_cascade_order_inverted...`):
#   bash scripts/consciousness_db_production_swap.sh --titan T1
#   bash scripts/consciousness_db_production_swap.sh --titan T2 --remote root@10.135.0.6:/home/antigravity/projects/titan
#   bash scripts/consciousness_db_production_swap.sh --titan T3 --remote root@10.135.0.6:/home/antigravity/projects/titan3

set -euo pipefail

TITAN=""
REMOTE=""
DRY=0
SKIP_RESTART=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --titan) TITAN="$2"; shift 2 ;;
        --remote) REMOTE="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --skip-restart) SKIP_RESTART=1; shift ;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TITAN" ]]; then
    echo "ERROR: --titan T1|T2|T3 required" >&2
    exit 1
fi

if [[ -n "$REMOTE" ]]; then
    HOST="${REMOTE%%:*}"
    REPO="${REMOTE#*:}"
    echo "[swap] Copying script to $HOST:$REPO/scripts/consciousness_db_production_swap.sh"
    if [[ $DRY -eq 0 ]]; then
        scp -q "$0" "$HOST:$REPO/scripts/consciousness_db_production_swap.sh"
    fi
    ARGS=(--titan "$TITAN")
    [[ $DRY -eq 1 ]] && ARGS+=(--dry-run)
    [[ $SKIP_RESTART -eq 1 ]] && ARGS+=(--skip-restart)
    ssh "$HOST" "cd $REPO && bash scripts/consciousness_db_production_swap.sh ${ARGS[*]}"
    exit $?
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
DB_PATH="$REPO_ROOT/data/consciousness.db"
TS=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$REPO_ROOT/data/consciousness.db.legacy_TEXT_${TS}"
MIGRATED_PATH="$REPO_ROOT/data/consciousness.db.f32_${TS}"
MIGRATION_SCRIPT="$REPO_ROOT/scripts/migrate_consciousness_db_blob_quantize.py"

echo "==========================================================="
echo "consciousness.db production swap — Titan $TITAN"
echo "  repo:        $REPO_ROOT"
echo "  source DB:   $DB_PATH"
echo "  backup to:   $BACKUP_PATH"
echo "  migrated to: $MIGRATED_PATH"
echo "  dry-run:     $DRY"
echo "  skip-restart:$SKIP_RESTART"
echo "==========================================================="

if [[ ! -f "$DB_PATH" ]]; then
    echo "ERROR: source DB not found at $DB_PATH" >&2
    exit 2
fi
if [[ ! -f "$MIGRATION_SCRIPT" ]]; then
    echo "ERROR: migration script not found at $MIGRATION_SCRIPT" >&2
    exit 3
fi

PRE_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM epochs")
PRE_BYTES=$(stat -c%s "$DB_PATH")
echo "[swap] PRE-swap: $PRE_COUNT epochs, $PRE_BYTES bytes"

echo "[swap] Step 2: copy source -> $MIGRATED_PATH"
if [[ $DRY -eq 0 ]]; then cp "$DB_PATH" "$MIGRATED_PATH"; else echo "[swap] (dry-run) skipping copy"; fi

echo "[swap] Step 3: run migrate_consciousness_db_blob_quantize.py --apply"
if [[ $DRY -eq 0 ]]; then
    if [[ -f "$REPO_ROOT/test_env/bin/python" ]]; then
        PY="$REPO_ROOT/test_env/bin/python"
    else
        PY=$(command -v python3)
    fi
    "$PY" "$MIGRATION_SCRIPT" --db "$MIGRATED_PATH" --apply
else
    echo "[swap] (dry-run) skipping migration"
fi

echo "[swap] Step 4: verify migrated DB integrity"
if [[ $DRY -eq 0 ]]; then
    MIG_COUNT=$(sqlite3 "$MIGRATED_PATH" "SELECT COUNT(*) FROM epochs")
    if [[ "$MIG_COUNT" -ne "$PRE_COUNT" ]]; then
        echo "ERROR: migrated DB has $MIG_COUNT epochs, expected $PRE_COUNT" >&2
        exit 4
    fi
    echo "[swap] verify OK: $MIG_COUNT epochs match pre-swap baseline"
else
    echo "[swap] (dry-run) skipping verify"
fi

echo "[swap] Step 5: stop Titan to drain consciousness writes"
if [[ $SKIP_RESTART -eq 1 ]]; then
    echo "[swap] --skip-restart set; leaving Titan running."
elif [[ $DRY -eq 0 ]]; then
    case "$TITAN" in
        T1) bash "$REPO_ROOT/scripts/t1_manage.sh" stop || true ;;
        T2) systemctl stop titan-t2.service || true ;;
        T3) systemctl stop titan-t3.service || true ;;
    esac
    sleep 2
else
    echo "[swap] (dry-run) skipping stop"
fi

echo "[swap] Step 6: atomic swap legacy -> f32"
if [[ $DRY -eq 0 ]]; then
    cp "$DB_PATH" "$BACKUP_PATH"
    mv "$MIGRATED_PATH" "$DB_PATH"
    echo "[swap] legacy preserved at $BACKUP_PATH"
else
    echo "[swap] (dry-run) skipping mv"
fi

if [[ $SKIP_RESTART -ne 1 ]]; then
    echo "[swap] Step 7: restart Titan + post-swap smoke"
    if [[ $DRY -eq 0 ]]; then
        case "$TITAN" in
            T1) bash "$REPO_ROOT/scripts/t1_manage.sh" start || true ;;
            T2) systemctl start titan-t2.service || true ;;
            T3) systemctl start titan-t3.service || true ;;
        esac
        sleep 20
        POST_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM epochs")
        echo "[swap] POST-swap: $POST_COUNT epochs (expected >= $PRE_COUNT)"
        if [[ "$POST_COUNT" -lt "$PRE_COUNT" ]]; then
            echo "ERROR: post-swap row count regressed" >&2
            exit 5
        fi
    else
        echo "[swap] (dry-run) skipping restart"
    fi
fi

echo "==========================================================="
echo "[swap] DONE — Titan $TITAN consciousness.db on f32 schema."
echo "  legacy backup: $BACKUP_PATH"
echo "==========================================================="
