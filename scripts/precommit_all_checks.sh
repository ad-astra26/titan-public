#!/usr/bin/env bash
# precommit_all_checks.sh — composite pre-commit hook.
#
# Runs in order:
#   1. scripts/precommit_async_check.sh — blocks async-boundary violations
#      in titan_plugin/*.py (existing, established 2026-04-14)
#   2. gitleaks on staged diffs — blocks commits that introduce secrets
#      (added 2026-04-15 after public-repo leak incident)
#
# Enable:   ln -sf ../../scripts/precommit_all_checks.sh .git/hooks/pre-commit
# Bypass:   git commit --no-verify  (use sparingly; each check has a reason)

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── 1. Async-block scanner (existing) ────────────────────────────
if [ -x "$REPO_ROOT/scripts/precommit_async_check.sh" ]; then
    if ! bash "$REPO_ROOT/scripts/precommit_async_check.sh"; then
        echo >&2 "⛔ pre-commit: async-block check failed"
        exit 1
    fi
fi

# ── 2. Gitleaks on staged changes ────────────────────────────────
GITLEAKS="$REPO_ROOT/scripts/bin/gitleaks"
CONFIG="$REPO_ROOT/scripts/public_sync/gitleaks.toml"

if [ -x "$GITLEAKS" ] && [ -f "$CONFIG" ]; then
    # gitleaks `protect --staged` scans only what's about to be committed
    if ! "$GITLEAKS" protect --staged --config="$CONFIG" --redact --no-banner 2>&1; then
        cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  Gitleaks found potential secrets in your staged changes.        │
│                                                                      │
│  Review the findings above. Either:                                  │
│    • Remove/scrub the secret from your staged files, OR              │
│    • If it's a genuine false-positive, extend the allowlist at       │
│      scripts/public_sync/gitleaks.toml                               │
│                                                                      │
│  Bypass (only after confirming NO real secret is present):           │
│      git commit --no-verify                                          │
│                                                                      │
│  Never bypass for speed. A bypassed commit + a future public sync    │
│  = a leak that cannot be unpublished.                                │
└──────────────────────────────────────────────────────────────────────┘

BANNER
        exit 1
    fi
fi

exit 0
