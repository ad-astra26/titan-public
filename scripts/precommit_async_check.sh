#!/bin/bash
# precommit_async_check.sh — Block commits that introduce CRITICAL async-blocks.
#
# Invoked by .git/hooks/pre-commit. Runs the async-blocks scanner against
# the entire titan_plugin/ tree (fast — ~10s) only when .py files under
# titan_plugin/ are staged. Fails if CRITICAL > 0.
#
# Enable:   ln -sf ../../scripts/precommit_async_check.sh .git/hooks/pre-commit
# Disable:  rm .git/hooks/pre-commit   (or: git commit --no-verify)
#
# Scanner lives in scripts/arch_map.py. See `async-blocks` subcommand.
# False-positive suppression: add `# noqa: async-block` on the offending line.

set -u

# Only run when titan_plugin/ Python files are staged.
STAGED_PY=$(git diff --cached --name-only --diff-filter=ACM | grep -E '^titan_plugin/.*\.py$' || true)
if [ -z "$STAGED_PY" ]; then
    exit 0
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT" || exit 1

PY="${REPO_ROOT}/test_env/bin/python"
if [ ! -x "$PY" ]; then
    echo "[precommit] test_env/bin/python not found — skipping async-blocks check"
    exit 0
fi

OUT=$("$PY" scripts/arch_map.py async-blocks 2>&1)
CRITICAL=$(echo "$OUT" | grep -oE 'CRITICAL=[0-9]+' | head -1 | sed 's/CRITICAL=//')

if [ -z "$CRITICAL" ]; then
    echo "[precommit] ⚠ async-blocks scanner output unparseable — allowing commit"
    exit 0
fi

if [ "$CRITICAL" -gt 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  🛑 PRE-COMMIT BLOCKED — $CRITICAL CRITICAL async-block site(s) detected"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "$OUT" | sed -n '/🔴 CRITICAL/,/🟡 HIGH\|🟠 MEDIUM\|TOTALS:/p' | head -40
    echo ""
    echo 'CRITICAL = sync I/O directly inside an "async def". These block the'
    echo "FastAPI event loop and degrade all endpoint latency."
    echo ""
    echo "Fix options:"
    echo "  1. Wrap the sync call at the call site:"
    echo "       result = await asyncio.to_thread(sync_fn, arg1, arg2)"
    echo "       result = await asyncio.to_thread(lambda: obj.method(x=1))"
    echo ""
    echo "  2. If verified-false-positive, suppress the specific line:"
    echo "       sync_call()  # noqa: async-block"
    echo ""
    echo "To override (not recommended): git commit --no-verify"
    echo "════════════════════════════════════════════════════════════════════════"
    exit 1
fi

echo "[precommit] ✓ async-blocks: CRITICAL=0 (scan took $(echo "$OUT" | grep -oE 'Scanned [0-9]+ files'))"
exit 0
