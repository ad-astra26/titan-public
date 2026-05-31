#!/usr/bin/env bash
# titan_build_cache_sweep.sh — weekly Rust build-cache hygiene (T1 localhost).
#
# Driven by the user systemd timer titan-cargo-sweep.timer. Keeps the local
# Rust build caches bounded so target/debug never silently regrows toward the
# 17G that filled the mainnet disk on 2026-05-31. Complements the in-repo
# [profile.dev] slimming + shared worktree target-dir (see
# titan-rust/Cargo.toml + .claude/worktrees/.cargo/config.toml).
#
# Two caches, two strategies:
#   1) Main repo + anchor programs — cargo-sweep is artifact-aware (knows which
#      .rlib/.o are still referenced), so age-based --time pruning is safe and
#      incremental. This is the deploy source (titan-rust/target/...) + the
#      local debug build dir; recent artifacts are always kept.
#   2) Shared worktree target-dir — cargo-sweep v0.8 resolves <project>/target by
#      filesystem heuristic and does NOT honor the .cargo target-dir redirect, so
#      it can't prune the shared dir. It's pure build cache for ephemeral session
#      worktrees, so we cap it by size: over the cap → wipe (worktrees rebuild on
#      demand). Under the cap → leave the incremental cache intact.
#
# NEVER touches data/, sovereign state, or the musl release binaries.
set -euo pipefail

export PATH="/home/antigravity/.cargo/bin:${PATH}"
REPO="/home/antigravity/projects/titan"
SWEEP="/home/antigravity/.cargo/bin/cargo-sweep"
SHARED="/home/antigravity/.cache/titan-cargo-target-worktrees"
AGE_DAYS=14
SHARED_CAP_MB=6000

echo "[build-cache-sweep] $(date -u +%FT%TZ) start"

# 1) Age-prune the main repo + anchor program caches (artifact-aware).
if [ -x "$SWEEP" ]; then
    "$SWEEP" sweep --time "$AGE_DAYS" --recursive \
        "$REPO/titan-rust" "$REPO/programs" || \
        echo "[build-cache-sweep] WARN: cargo-sweep exited non-zero (continuing)"
else
    echo "[build-cache-sweep] WARN: cargo-sweep not installed at $SWEEP"
fi

# 2) Size-cap the shared worktree target-dir.
if [ -d "$SHARED" ]; then
    sz=$(du -sm "$SHARED" 2>/dev/null | awk '{print $1}')
    sz=${sz:-0}
    if [ "$sz" -gt "$SHARED_CAP_MB" ]; then
        echo "[build-cache-sweep] shared worktree cache ${sz}MB > ${SHARED_CAP_MB}MB cap — wiping (pure build cache)"
        rm -rf "${SHARED:?}/"* 2>/dev/null || true
    else
        echo "[build-cache-sweep] shared worktree cache ${sz}MB (under ${SHARED_CAP_MB}MB cap) — kept"
    fi
fi

echo "[build-cache-sweep] $(date -u +%FT%TZ) done"
