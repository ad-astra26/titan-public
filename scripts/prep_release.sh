#!/usr/bin/env bash
# prep_release.sh — draft a CHANGELOG.md section for the next release.
#
# Usage:
#   bash scripts/prep_release.sh vX.Y.Z
#
# What it does:
#   1. Verifies the tag isn't already in CHANGELOG.md
#   2. Generates a draft section from `git log <prev-tag>..HEAD --oneline`
#      on titan-v6, grouped by conventional-commit prefix (feat/fix/docs/…)
#   3. Inserts the draft section at the top of CHANGELOG.md (above [Unreleased]
#      if present; otherwise above any "## v..." line)
#   4. Opens $EDITOR so the maintainer curates the prose
#   5. Stops there — no commit, no tag, no push (deliberate; maintainer decides)
#
# After curating, conventional next steps:
#   git add CHANGELOG.md
#   git commit -m "chore(release): prep vX.Y.Z"
#   bash scripts/git_publish.sh                    # push private + sync public
#   bash scripts/cut_release.sh vX.Y.Z             # tag on public → trigger release.yml

set -euo pipefail

DEV_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DEV_ROOT"

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
    echo "usage: $0 vX.Y.Z" >&2
    exit 2
fi

# semver-ish sanity check (loose — we allow pre-release suffixes)
if ! [[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-].+)?$ ]]; then
    echo "error: tag '$TAG' doesn't look like semver (e.g. v0.0.1, v1.2.3-rc1)" >&2
    exit 2
fi

CHANGELOG=CHANGELOG.md
[[ -f "$CHANGELOG" ]] || { echo "error: $CHANGELOG not found" >&2; exit 1; }

# already in CHANGELOG?
if grep -qE "^## $TAG([[:space:]]|$)" "$CHANGELOG"; then
    echo "error: $TAG already has a section in $CHANGELOG — refusing to overwrite" >&2
    exit 1
fi

# find the most recent v-prefixed git tag matching semver
PREV=$(git tag --list 'v*' --sort=-version:refname \
       | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+([.-].+)?$' \
       | head -1 || true)

DATE=$(date -u +%Y-%m-%d)
LOG_RANGE=""
if [[ -n "$PREV" ]]; then
    LOG_RANGE="$PREV..HEAD"
    echo "==> Drafting from $LOG_RANGE on titan-v6 ($(git rev-list --count "$LOG_RANGE") commits)"
else
    LOG_RANGE="HEAD"
    echo "==> No prior semver tag found — drafting from start of history"
fi

DRAFT=$(mktemp)
{
    echo "## $TAG — $DATE"
    echo
    echo "_$(if [[ -n "$PREV" ]]; then echo "Changes since $PREV."; else echo "Initial public release."; fi)_"
    echo

    # Group commit subjects by conventional-commit prefix.
    # `git log --no-merges` skips merge commits; `--pretty=%s` = subject only.
    SUBJECTS=$(git log --no-merges --pretty='%s' $LOG_RANGE)

    add_section() {
        local heading="$1" pattern="$2"
        local lines
        lines=$(echo "$SUBJECTS" | grep -iE "^($pattern)(\(|:|!)" || true)
        if [[ -n "$lines" ]]; then
            echo "### $heading"
            echo
            echo "$lines" | sed 's/^/- /'
            echo
        fi
    }

    add_section "Added"      "feat"
    add_section "Fixed"      "fix|hotfix"
    add_section "Changed"    "refactor|chore|perf|build|style"
    add_section "Docs"       "docs?"
    add_section "Tests"      "test"
    add_section "Other"      "spec|ci"

    # anything that didn't match any prefix
    UNCLASSIFIED=$(echo "$SUBJECTS" | grep -vE '^(feat|fix|hotfix|refactor|chore|perf|build|style|docs?|test|spec|ci)(\(|:|!)' || true)
    if [[ -n "$UNCLASSIFIED" ]]; then
        echo "### Uncategorized (review me — no conventional-commit prefix)"
        echo
        echo "$UNCLASSIFIED" | sed 's/^/- /'
        echo
    fi

    echo "### SPEC version embodied"
    echo
    SPEC_VER=$(grep -E '^spec_version:' titan-docs/specs/SPEC_titan_architecture.md 2>/dev/null | head -1 | awk '{print $2}' || echo "UNKNOWN")
    echo "\`titan-docs/specs/SPEC_titan_architecture.md $SPEC_VER\` (internal)."
    echo
    echo "---"
    echo
} > "$DRAFT"

# Insert the draft above the existing "## [Unreleased]" or "## v..." block
TMP=$(mktemp)
awk -v draft="$DRAFT" '
    /^## (\[Unreleased\]|v[0-9])/ && !inserted {
        while ((getline line < draft) > 0) print line
        close(draft)
        inserted = 1
    }
    { print }
' "$CHANGELOG" > "$TMP"
mv "$TMP" "$CHANGELOG"
rm -f "$DRAFT"

echo "==> Draft section inserted in $CHANGELOG."
echo "==> Opening \$EDITOR (${EDITOR:-vi}) for curation. Tighten the prose, drop noise, group logically."

${EDITOR:-vi} "$CHANGELOG"

echo
echo "==> Done. Next steps (when you're happy with $CHANGELOG):"
echo "      git add $CHANGELOG"
echo "      git commit -m 'chore(release): prep $TAG'"
echo "      bash scripts/git_publish.sh             # push private + sync public"
echo "      bash scripts/cut_release.sh $TAG        # tag on public → trigger release.yml"
