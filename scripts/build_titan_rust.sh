#!/usr/bin/env bash
# build_titan_rust.sh — build Phase C Rust binaries
#
# Per PLAN_microkernel_phase_c_s2_kernel.md §15.1 + SPEC §13.
#
# Modes:
#   debug   — fast iteration, glibc-dynamic, target/debug/
#   release — production, glibc-dynamic, target/release/
#   musl    — production, musl-static, target/x86_64-unknown-linux-musl/release/
#
# Phase C C-S7 (2026-05-05): builds the full 9-binary Rust fleet —
# titan-kernel-rs, titan-trinity-rs, titan-unified-spirit-rs, plus 6
# trinity daemons (titan-{inner,outer}-{body,mind,spirit}-rs). All
# compiled by `cargo build --workspace` regardless of mode.
#
# Usage: bash scripts/build_titan_rust.sh [debug|release|musl]

set -euo pipefail

MODE="${1:-musl}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}/../titan-rust"

if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "ERROR: workspace dir not found at $WORKSPACE_DIR" >&2
    exit 1
fi

cd "$WORKSPACE_DIR"

# Static-linkage verifier — supports both `ldd` "not a dynamic executable" and
# "statically linked" outputs, plus `file` heuristic as fallback for ldd-less
# musl hosts.
_is_static() {
    local bin="$1"
    if ldd "$bin" 2>&1 | grep -qE "not a dynamic|statically linked"; then
        return 0
    fi
    if command -v file >/dev/null 2>&1; then
        if file "$bin" 2>/dev/null | grep -q "statically linked"; then
            return 0
        fi
    fi
    return 1
}

case "$MODE" in
    debug)
        cargo build
        echo "Built debug binaries: target/debug/titan-*"
        ;;
    release)
        cargo build --release
        echo "Built release binaries: target/release/titan-*"
        ;;
    musl)
        # Default linker for musl is musl-gcc; let cargo handle it via PATH.
        cargo build --release --target x86_64-unknown-linux-musl
        echo "Built musl static binaries: target/x86_64-unknown-linux-musl/release/titan-*"
        echo
        # Verify static linkage on every shippable binary
        any_failed=0
        # The 9 shippable fleet binaries. (`titan-trinity-rs-placeholder` was
        # renamed to `titan-trinity-rs` in C-S3 chunk C3-3 — the stale name made
        # this loop always exit 1; restored to the real fleet list 2026-06-02.)
        for bin in \
            target/x86_64-unknown-linux-musl/release/titan-kernel-rs \
            target/x86_64-unknown-linux-musl/release/titan-trinity-rs \
            target/x86_64-unknown-linux-musl/release/titan-unified-spirit-rs \
            target/x86_64-unknown-linux-musl/release/titan-inner-body-rs \
            target/x86_64-unknown-linux-musl/release/titan-inner-mind-rs \
            target/x86_64-unknown-linux-musl/release/titan-inner-spirit-rs \
            target/x86_64-unknown-linux-musl/release/titan-outer-body-rs \
            target/x86_64-unknown-linux-musl/release/titan-outer-mind-rs \
            target/x86_64-unknown-linux-musl/release/titan-outer-spirit-rs
        do
            if [ -x "$bin" ]; then
                if _is_static "$bin"; then
                    sz=$(du -h "$bin" | awk '{print $1}')
                    sha=$(sha256sum "$bin" | awk '{print $1}' | cut -c1-12)
                    echo "  ✓ $(basename "$bin") — static (size=$sz sha256=${sha}…)"
                else
                    echo "  ✗ $(basename "$bin") — NOT static (build error)" >&2
                    any_failed=1
                fi
            else
                echo "  ✗ $(basename "$bin") — missing or not executable" >&2
                any_failed=1
            fi
        done
        if [ "$any_failed" -ne 0 ]; then
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [debug|release|musl]" >&2
        exit 1
        ;;
esac
