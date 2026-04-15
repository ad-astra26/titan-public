#!/usr/bin/env bash
# ─────────────────────────────────────────────
#  Titan Terminal Chat — launcher script
#  Talk to Titan directly from your terminal.
# ─────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate venv
if [ -f "test_env/bin/activate" ]; then
    source test_env/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Warning: No virtualenv found (tried test_env, .venv). Using system Python."
fi

exec python -m titan_plugin.channels.terminal "$@"
