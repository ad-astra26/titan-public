#!/bin/bash
# Git merge driver — regenerate an auto-generated tracker index file
# from its source rather than 3-way-merging the generated output.
#
# Called by git on conflict via .gitattributes mapping:
#   %A = path to the version in OUR branch (the working file — overwrite this)
#   %O = path to the common ancestor version (unused)
#   %B = path to the THEIRS version (unused — we just regen)
#   $4 = which index to regenerate (mapped in install.sh)
#
# After this script runs, git takes %A as the final merged result.
# We discard both sides' generated output and re-emit from source.
#
# Exit 0 = merge succeeded (no conflict markers).
# Exit non-zero = fall back to conflict markers (manual resolution).

set -euo pipefail

OURS_PATH="$1"
COMMON_PATH="$2"   # unused
THEIRS_PATH="$3"   # unused
INDEX_KIND="$4"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Activate venv if present (most regen scripts need project deps).
if [ -f "test_env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source test_env/bin/activate
fi

case "$INDEX_KIND" in
    SPEC_index)
        # spec_index.py reads titan-docs/specs/SPEC_titan_architecture.md and
        # writes titan-docs/specs/SPEC_index.md. Since git places the working
        # version at OURS_PATH, we redirect the output there.
        python scripts/spec_index.py > /dev/null 2>&1
        cp titan-docs/specs/SPEC_index.md "$OURS_PATH"
        ;;
    OBSERVABLES_index)
        python scripts/tracker_indexer.py --emit-index titan-docs/OBSERVABLES.md > /dev/null 2>&1
        cp titan-docs/OBSERVABLES_index.md "$OURS_PATH"
        ;;
    BUGS_index)
        python scripts/tracker_indexer.py --emit-index titan-docs/BUGS.md > /dev/null 2>&1
        cp titan-docs/BUGS_index.md "$OURS_PATH"
        ;;
    DEFERRED_index)
        python scripts/tracker_indexer.py --emit-index titan-docs/DEFERRED_ITEMS.md > /dev/null 2>&1
        cp titan-docs/DEFERRED_index.md "$OURS_PATH"
        ;;
    ARCHITECTURE_cgn_family_index)
        # architecture_cgn_family_index.py reads titan-docs/specs/ARCHITECTURE_cgn_family.md
        # and writes titan-docs/specs/ARCHITECTURE_cgn_family_index.md.
        python scripts/architecture_cgn_family_index.py > /dev/null 2>&1
        cp titan-docs/specs/ARCHITECTURE_cgn_family_index.md "$OURS_PATH"
        ;;
    ARCHITECTURE_synthesis_engine_index)
        # architecture_synthesis_engine_index.py reads titan-docs/specs/ARCHITECTURE_synthesis_engine.md
        # and writes titan-docs/specs/ARCHITECTURE_synthesis_engine_index.md.
        python scripts/architecture_synthesis_engine_index.py > /dev/null 2>&1
        cp titan-docs/specs/ARCHITECTURE_synthesis_engine_index.md "$OURS_PATH"
        ;;
    ARCHITECTURE_api_family_index)
        # architecture_api_family_index.py reads titan-docs/specs/ARCHITECTURE_api_family.md
        # and writes titan-docs/specs/ARCHITECTURE_api_family_index.md.
        python scripts/architecture_api_family_index.py > /dev/null 2>&1
        cp titan-docs/specs/ARCHITECTURE_api_family_index.md "$OURS_PATH"
        ;;
    ARCHITECTURE_trinity_index)
        # architecture_trinity_index.py reads titan-docs/specs/ARCHITECTURE_trinity.md
        # and writes titan-docs/specs/ARCHITECTURE_trinity_index.md.
        python scripts/architecture_trinity_index.py > /dev/null 2>&1
        cp titan-docs/specs/ARCHITECTURE_trinity_index.md "$OURS_PATH"
        ;;
    *)
        echo "regen.sh: unknown index kind '$INDEX_KIND'" >&2
        exit 1
        ;;
esac

exit 0
