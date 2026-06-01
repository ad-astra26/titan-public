#!/bin/bash
# Install per-repo git merge drivers for auto-generated tracker index files.
# Run ONCE per clone (or rerun after any update to this file).
#
# Purpose: when two branches both regenerate e.g. SPEC_index.md from
# different source revisions, git's default 3-way merge produces a
# spurious conflict on every line. Our merge drivers just take "ours"
# (HEAD's tree state of the source file is what will land in the merge
# commit) and rerun the regen script — output is deterministic from
# source so there's nothing to merge.
#
# See .gitattributes for the file→driver mapping.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "Installing git merge drivers for auto-gen tracker indexes..."

git config merge.regen-spec-index.name "Regenerate SPEC_index.md from SPEC body"
git config merge.regen-spec-index.driver "bash scripts/git_merge_drivers/regen.sh %A %O %B SPEC_index"

git config merge.regen-observables-index.name "Regenerate OBSERVABLES_index.md from OBSERVABLES.md"
git config merge.regen-observables-index.driver "bash scripts/git_merge_drivers/regen.sh %A %O %B OBSERVABLES_index"

git config merge.regen-bugs-index.name "Regenerate BUGS_index.md from BUGS.md"
git config merge.regen-bugs-index.driver "bash scripts/git_merge_drivers/regen.sh %A %O %B BUGS_index"

git config merge.regen-deferred-index.name "Regenerate DEFERRED_index.md from DEFERRED_ITEMS.md"
git config merge.regen-deferred-index.driver "bash scripts/git_merge_drivers/regen.sh %A %O %B DEFERRED_index"

git config merge.regen-architecture-cgn-family-index.name "Regenerate ARCHITECTURE_cgn_family_index.md from ARCHITECTURE_cgn_family.md"
git config merge.regen-architecture-cgn-family-index.driver "bash scripts/git_merge_drivers/regen.sh %A %O %B ARCHITECTURE_cgn_family_index"

git config merge.regen-architecture-synthesis-engine-index.name "Regenerate ARCHITECTURE_synthesis_engine_index.md from ARCHITECTURE_synthesis_engine.md"
git config merge.regen-architecture-synthesis-engine-index.driver "bash scripts/git_merge_drivers/regen.sh %A %O %B ARCHITECTURE_synthesis_engine_index"

git config merge.regen-architecture-api-family-index.name "Regenerate ARCHITECTURE_api_family_index.md from ARCHITECTURE_api_family.md"
git config merge.regen-architecture-api-family-index.driver "bash scripts/git_merge_drivers/regen.sh %A %O %B ARCHITECTURE_api_family_index"

git config merge.regen-architecture-trinity-index.name "Regenerate ARCHITECTURE_trinity_index.md from ARCHITECTURE_trinity.md"
git config merge.regen-architecture-trinity-index.driver "bash scripts/git_merge_drivers/regen.sh %A %O %B ARCHITECTURE_trinity_index"

git config merge.regen-architecture-backup-restore-index.name "Regenerate ARCHITECTURE_backup_restore_index.md from ARCHITECTURE_backup_restore.md"
git config merge.regen-architecture-backup-restore-index.driver "bash scripts/git_merge_drivers/regen.sh %A %O %B ARCHITECTURE_backup_restore_index"

# `union` driver is built-in to git; .gitattributes references it directly
# (titan-docs/conversations/INDEX.md, sessions/meta_cgn_trajectory.tsv).

echo "Done. Drivers active in this clone:"
git config --get-regexp 'merge\.regen-.*\.driver' | sed 's/^/  /'
echo ""
echo "Note: .gitattributes is committed to the repo so every clone sees the"
echo "mapping, but the driver scripts must be local — run this script after"
echo "every fresh clone."
