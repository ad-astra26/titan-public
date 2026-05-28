#!/usr/bin/env bash
# upload_release_binaries.sh — fast path: publish localhost-built musl daemons
# to an existing GitHub Release on the PUBLIC repo.
#
# WHY: localhost already builds the 9 x86_64-linux-musl daemons for the fleet
# (build once, scp to T2/T3). Re-building them in CI (release.yml) is the
# CANONICAL, transparent path — a tester can verify those binaries were built
# from the public source. This helper is the CONVENIENCE path for when CI is
# slow/unavailable: it uploads the binaries you already have, with the SAME
# SHA256SUMS manifest the installer verifies against.
#
# TRADE-OFF (be honest with yourself): localhost-uploaded binaries ask the
# tester to TRUST that they match the public source; CI-built ones let them
# VERIFY it. Prefer cut_release.sh + release.yml for public alphas; use this
# when you knowingly accept that trade-off.
#
# Usage:
#   bash scripts/upload_release_binaries.sh vX.Y.Z            # upload existing bin/
#   bash scripts/upload_release_binaries.sh vX.Y.Z --build    # cargo build first
#
# Requires: gh (authenticated), the release tag to already exist on public
# (create it with cut_release.sh). Uploads with --clobber (replaces any CI
# assets of the same name).
set -euo pipefail

DEV_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PUBLIC_REPO="ad-astra26/titan-public"
MUSL_TARGET="x86_64-unknown-linux-musl"
DAEMONS=(
    titan-inner-body-rs titan-inner-mind-rs titan-inner-spirit-rs
    titan-kernel-rs titan-outer-body-rs titan-outer-mind-rs
    titan-outer-spirit-rs titan-trinity-rs titan-unified-spirit-rs
)

TAG="${1:-}"
DO_BUILD="${2:-}"
[[ -n "$TAG" ]] || { echo "usage: $0 vX.Y.Z [--build]" >&2; exit 2; }
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-].+)?$ ]] || { echo "error: '$TAG' not semver" >&2; exit 2; }
command -v gh >/dev/null 2>&1 || { echo "error: gh CLI not found" >&2; exit 1; }

BIN_DIR="$DEV_ROOT/bin"

if [[ "$DO_BUILD" == "--build" ]]; then
    echo "==> Building 9 musl daemons from titan-rust/ …"
    ( cd "$DEV_ROOT/titan-rust" && cargo build --release --bins --target "$MUSL_TARGET" )
    SRC="$DEV_ROOT/titan-rust/target/$MUSL_TARGET/release"
    mkdir -p "$BIN_DIR"
    for d in "${DAEMONS[@]}"; do cp "$SRC/$d" "$BIN_DIR/$d"; done
fi

# Verify all 9 are present
for d in "${DAEMONS[@]}"; do
    [[ -f "$BIN_DIR/$d" ]] || { echo "error: missing binary $BIN_DIR/$d (build first with --build)" >&2; exit 1; }
done

# Confirm the release exists (cut_release.sh / a pushed tag creates it)
gh release view "$TAG" --repo "$PUBLIC_REPO" >/dev/null 2>&1 \
    || { echo "error: release '$TAG' not found on $PUBLIC_REPO — run cut_release.sh $TAG first" >&2; exit 1; }

# Stage + SHA256SUMS (identical recipe to release.yml: names only, no path)
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
for d in "${DAEMONS[@]}"; do cp "$BIN_DIR/$d" "$STAGE/$d"; done
( cd "$STAGE" && sha256sum "${DAEMONS[@]}" > SHA256SUMS )
echo "==> SHA256SUMS:"; sed 's/^/      /' "$STAGE/SHA256SUMS"

echo "==> Uploading 9 daemons + SHA256SUMS to $PUBLIC_REPO release $TAG (--clobber)…"
( cd "$STAGE" && gh release upload "$TAG" "${DAEMONS[@]}" SHA256SUMS \
    --repo "$PUBLIC_REPO" --clobber )

echo
echo "==> Done. The installer fetches + verifies these at:"
echo "      https://github.com/$PUBLIC_REPO/releases/download/$TAG/<daemon>"
