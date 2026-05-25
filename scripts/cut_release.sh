#!/usr/bin/env bash
# cut_release.sh — tag a release on the PUBLIC repo (ad-astra26/titan-public).
#
# Usage:
#   bash scripts/cut_release.sh vX.Y.Z
#
# Why this script exists:
#   Public-repo commit SHAs are NOT the same as dev-repo SHAs — the public
#   sync re-creates each commit with the filtered tree + same author/date,
#   producing different SHAs. So a tag created locally on dev cannot be
#   pushed to public (the SHA doesn't exist there). This script clones the
#   public repo into a temp dir, tags `<TAG>` at its current HEAD, and pushes
#   the tag to public — which fires `.github/workflows/release.yml`.
#
# Safety:
#   - Refuses to tag if the local `CHANGELOG.md` has no section for $TAG.
#   - Refuses to tag if the tag already exists on public.
#   - Uses an ANNOTATED tag (`-a`) so `git log` shows the release note.
#   - All operations happen in a temp dir; nothing on dev is mutated.

set -euo pipefail

DEV_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DEV_ROOT"

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
    echo "usage: $0 vX.Y.Z" >&2
    exit 2
fi

if ! [[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-].+)?$ ]]; then
    echo "error: tag '$TAG' doesn't look like semver" >&2
    exit 2
fi

CHANGELOG=CHANGELOG.md
[[ -f "$CHANGELOG" ]] || { echo "error: $CHANGELOG not found" >&2; exit 1; }

if ! grep -qE "^## $TAG([[:space:]]|$)" "$CHANGELOG"; then
    echo "error: $CHANGELOG has no '## $TAG' section — run prep_release.sh first" >&2
    exit 1
fi

# Extract the CHANGELOG section for the tag (this becomes the annotated-tag
# message AND the release body — release.yml reads CHANGELOG.md the same way)
NOTES=$(mktemp)
awk -v tag="$TAG" '
    /^## / {
        if (in_section) exit
        if ($2 == tag) { in_section = 1; print; next }
    }
    in_section { print }
' "$CHANGELOG" > "$NOTES"

if [[ ! -s "$NOTES" ]]; then
    echo "error: failed to extract '## $TAG' section from $CHANGELOG" >&2
    exit 1
fi

PUBLIC_REMOTE="public"
PUBLIC_BRANCH="main"
git remote | grep -qx "$PUBLIC_REMOTE" \
    || { echo "error: git remote '$PUBLIC_REMOTE' not configured" >&2; exit 1; }
PUBLIC_URL=$(git remote get-url "$PUBLIC_REMOTE")

# Verify the tag isn't already on public
if git ls-remote --tags "$PUBLIC_REMOTE" "refs/tags/$TAG" | grep -q .; then
    echo "error: tag '$TAG' already exists on $PUBLIC_REMOTE — refusing to overwrite" >&2
    exit 1
fi

# Clone, tag, push, cleanup
TMP=$(mktemp -d)
trap 'rm -rf "$TMP" "$NOTES"' EXIT

echo "==> Cloning $PUBLIC_REMOTE/$PUBLIC_BRANCH into temp dir..."
git clone --quiet --branch "$PUBLIC_BRANCH" --depth 1 "$PUBLIC_URL" "$TMP/public"

cd "$TMP/public"
HEAD_SHA=$(git rev-parse HEAD)
echo "==> Tagging $TAG at public HEAD ($HEAD_SHA)"
git tag -a "$TAG" -F "$NOTES"

echo "==> Pushing tag to $PUBLIC_REMOTE..."
git push origin "refs/tags/$TAG"

cd "$DEV_ROOT"
echo
echo "==> Tag $TAG pushed to $PUBLIC_REMOTE. The release workflow should fire."
echo "==> Watch: https://github.com/ad-astra26/titan-public/actions"
echo "==> When it completes, the GitHub Release appears at:"
echo "      https://github.com/ad-astra26/titan-public/releases/tag/$TAG"
