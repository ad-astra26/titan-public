#!/usr/bin/env bash
# setup_titan.sh — thin bootstrap for a sovereign Titan install (W1.h).
#
# Philosophy: THIN bootstrap, FAT audited wizard. The only thing a stranger
# runs sight-unseen is this ~100-line script; it merely (1) checks the host can
# run Titan, (2) clones the PUBLIC repo at a pinned ref, and (3) hands off to
# the reviewed, versioned in-repo wizard (`python3 -m scripts.setup_titan`).
# Everything consequential lives in code you can read on GitHub before trusting.
#
#   curl -fsSL https://raw.githubusercontent.com/ad-astra26/titan-public/main/setup_titan.sh | bash
#   # pass wizard args after `--`:
#   curl -fsSL .../setup_titan.sh | bash -s -- --default
#
# Flags (consumed here): --tag <ref>  --dir <path>  --help
# All other args are forwarded verbatim to `setup_titan install`.
set -euo pipefail

PUBLIC_REPO="https://github.com/ad-astra26/titan-public.git"
# Newest release first (incl. pre-releases) — GitHub's `/releases/latest` EXCLUDES
# pre-releases, so we query the full list and take element 0.
RELEASES_API="https://api.github.com/repos/ad-astra26/titan-public/releases?per_page=1"
DEFAULT_REF=""               # empty ⇒ auto-resolve the latest published (pre)release
DEFAULT_DIR="${HOME}/titan"
MIN_PY_MINOR=11               # require Python 3.11+

# ── brand-ish output (no deps) ──────────────────────────────────────────────
_haze()  { printf '\033[38;2;229;199;158m%s\033[0m\n' "$*"; }
_grow()  { printf '\033[38;2;119;204;204m%s\033[0m\n' "$*"; }
_warn()  { printf '\033[38;2;229;199;158m⚠ %s\033[0m\n' "$*" >&2; }
_die()   { printf '\033[38;2;255;107;107m✗ %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
setup_titan.sh — bootstrap a sovereign Titan.

  --tag <ref>   git ref to clone (default: the latest published release; a
                release uses vX.Y.Z — its binaries are fetched to match)
  --dir <path>  install directory (default: ${DEFAULT_DIR})
  --help        this message

Any other flags are passed to the wizard, e.g.:
  setup_titan.sh --default
  setup_titan.sh --tag v0.0.1 --dir ~/mytitan --mode local
  setup_titan.sh --resurrect                 # 🜂 recover a mainnet Titan from its
                                             #    on-chain sovereign backup (needs Shard-1)
  setup_titan.sh --resurrect --verify-only   #    recovery observation mode (live restore-test)
EOF
}

REF="$DEFAULT_REF"
DIR="$DEFAULT_DIR"
WIZARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) REF="${2:?--tag needs a value}"; shift 2 ;;
        --dir) DIR="${2:?--dir needs a value}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) WIZARD_ARGS+=("$1"); shift ;;
    esac
done

# ── resolve the install ref ─────────────────────────────────────────────────
# No --tag ⇒ pin to the LATEST published release so the bare one-liner
# (`curl … | bash`) Just Works: the matching release ALSO carries the verified
# Rust daemon binaries the wizard fetches. (Cloning `main` has no release →
# the binaries phase would fail; a user must NOT need to know a version tag.)
if [[ -z "$REF" ]]; then
    if command -v curl >/dev/null 2>&1; then
        REF="$(curl -fsSL "$RELEASES_API" 2>/dev/null \
                | grep -m1 '"tag_name"' | cut -d'"' -f4 || true)"
    fi
    if [[ -n "$REF" ]]; then
        _grow "Pinned to latest release: ${REF}"
    else
        REF="main"
        _warn "Could not resolve the latest release (offline / API limit). Falling back to 'main' — pass --tag vX.Y.Z, or --build-rust, so the binaries phase can complete."
    fi
fi

# ── self-integrity hint (checksum) ──────────────────────────────────────────
# When run from a file we can show our own sha256 so it can be matched against
# the value published in the GitHub Release notes. Piped via curl, $0 is bash,
# so we print the verify recipe instead of a misleading hash.
if [[ -f "${BASH_SOURCE[0]}" ]] && command -v sha256sum >/dev/null 2>&1; then
    _haze "setup_titan.sh sha256: $(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
else
    _haze "To verify this bootstrap before trusting it:"
    echo "  curl -fsSLO https://raw.githubusercontent.com/ad-astra26/titan-public/${REF}/setup_titan.sh"
    echo "  sha256sum setup_titan.sh   # compare against the release notes, then: bash setup_titan.sh"
fi

# ── 1. host preflight + OS prerequisites ────────────────────────────────────
# A truly fresh cloud image has neither git nor python3-venv nor a C toolchain,
# so the bootstrap installs them — the wizard's deeper preflight then runs on a
# box that can actually build the venv. (Surfaced by real-world testing on a
# stock Ubuntu box, 2026-05-29.)
[[ "$(uname -s)" == "Linux" ]] || _die "Titan requires Linux (got $(uname -s))."
IS_APT=""
if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    case "${ID:-}:${ID_LIKE:-}" in
        *debian*|*ubuntu*) IS_APT=1 ;;
        *) _warn "Tested on Debian/Ubuntu; '${ID:-unknown}' may need manual deps." ;;
    esac
fi

# Run a command as root: directly if already root, else via sudo.
_root() { if [[ "$(id -u)" -eq 0 ]]; then "$@"; else sudo "$@"; fi; }

if [[ -n "$IS_APT" ]]; then
    if [[ "$(id -u)" -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
        _die "Need root or sudo to install prerequisites (git, python3-venv, build tools)."
    fi
    # Base deps the wizard + provisioner need: git/python to run the wizard;
    # build-essential/pkg-config/libssl-dev for the rust+avm compiles (Phase B);
    # xdelta3 for sovereign restore; nftables for the resurrection-test netjail.
    _grow "Installing OS prerequisites (git, python3-venv/dev, build-essential, pkg-config, libssl-dev, xdelta3, nftables)…"
    _root apt-get update -y >/dev/null 2>&1 || _warn "apt-get update had warnings (continuing)."
    _root apt-get install -y git python3 python3-venv python3-dev build-essential ca-certificates \
        pkg-config libssl-dev xdelta3 nftables \
        || _die "Failed to install prerequisites. Install manually then re-run: git python3-venv python3-dev build-essential pkg-config libssl-dev xdelta3 nftables"
fi

command -v git     >/dev/null 2>&1 || _die "git not found (prerequisite install failed)."
command -v python3 >/dev/null 2>&1 || _die "python3 not found (prerequisite install failed)."
[[ "$(id -u)" -eq 0 ]] || command -v sudo >/dev/null 2>&1 || \
    _warn "Not root and sudo missing — the systemd install phase will fail."

PY_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)')"
PY_MAJOR="$(python3 -c 'import sys; print(sys.version_info.major)')"
[[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -ge "$MIN_PY_MINOR" ]] || _die \
"Python 3.${MIN_PY_MINOR}+ required (found ${PY_MAJOR}.${PY_MINOR}). \
Ubuntu 24.04+ ships 3.12 natively; on 22.04 (3.10) add a Python 3.11+ backport (deadsnakes) first."

# ── 2. clone (or update) the PUBLIC repo at the pinned ref ───────────────────
if [[ -d "$DIR/.git" ]]; then
    _warn "Existing checkout at $DIR — fetching '${REF}' (your data/ + identity are never touched by git)."
    git -C "$DIR" fetch --depth 1 origin "$REF"
    git -C "$DIR" checkout -q FETCH_HEAD
elif [[ -e "$DIR" ]]; then
    _die "$DIR exists but is not a git checkout. Move it aside or pass --dir."
else
    _grow "Cloning Titan ($REF) → $DIR"
    git clone --depth 1 --branch "$REF" "$PUBLIC_REPO" "$DIR" 2>/dev/null \
        || git clone --depth 1 "$PUBLIC_REPO" "$DIR"   # fallback: default branch if ref is a SHA
fi

# ── 3. hand off to the audited, versioned wizard ────────────────────────────
cd "$DIR"
_grow "Handing off to the Titan setup wizard…"
# Forward the resolved ref as --tag so the binary-fetch phase pulls the matching
# release assets (a vX.Y.Z release). The wizard still accepts --build-rust to
# compile from source instead.
#
# When invoked via `curl … | bash`, this script's stdin IS the piped script, so
# the wizard's interactive prompts (inference key, Telegram token, mainnet burn)
# would hit EOF. Reattach stdin to the controlling terminal so prompts work;
# fall back to inherited stdin when there's no tty (headless/automated runs,
# e.g. `bash setup_titan.sh … < answers.txt`).
#
# NOTE: test that /dev/tty can actually be OPENED, not just that the node is
# readable (`[[ -r /dev/tty ]]`). Under nohup / setsid / a piped CI runner the
# device node exists and is mode-readable, but there is no controlling terminal,
# so `< /dev/tty` fails with ENXIO ("No such device or address") and — under
# `set -e` — aborts the whole install. `{ : < /dev/tty; }` performs the real open.
if { : < /dev/tty; } 2>/dev/null; then
    exec python3 -m scripts.setup_titan install --tag "$REF" "${WIZARD_ARGS[@]}" < /dev/tty
else
    exec python3 -m scripts.setup_titan install --tag "$REF" "${WIZARD_ARGS[@]}"
fi
