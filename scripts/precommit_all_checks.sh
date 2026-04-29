#!/usr/bin/env bash
# precommit_all_checks.sh — composite pre-commit hook.
#
# Runs in order:
#   1. scripts/precommit_async_check.sh — blocks async-boundary violations
#      in titan_plugin/*.py (existing, established 2026-04-14)
#   2. gitleaks on staged diffs — blocks commits that introduce secrets
#      (added 2026-04-15 after public-repo leak incident)
#   3. arch_map cache-keys --audit — blocks commits that drift the
#      observatory data contract registry (added 2026-04-26 per
#      rFP_observatory_data_loading_v1 Phase 1). Skipped if no
#      titan_plugin/api/* or titan_plugin/{modules,core}/* files staged.
#   4. tests/test_lazy_imports.py — blocks commits that re-introduce
#      eager module-level imports of torch/transformers/faiss/etc into
#      worker modules (added 2026-04-27 closing DEFERRED I-002). Skipped
#      if no titan_plugin/*.py files staged. ~5s wall when active.
#      Reason: a single eager torch import in a hot-path file costs
#      ~860MB × 9 workers ≈ 2.3GB at boot (per commit 7f01125 history).
#   5. arch_map phase-c verify --strict — blocks SPEC drift per
#      titan-docs/SPEC_titan_architecture.md §20 + Rule 2 of
#      feedback_phase_c_spec_enforcement.md. Runs when SPEC TOML, generated
#      constants files, generator script, arch_map.py, or any titan_plugin/
#      / titan-rust/ source is staged. Catches:
#        • Hand-edits to _phase_c_constants.py / constants.rs (regen drift)
#        • TOML constant added without regen
#        • Domain used by constant but missing from [domains.X] block
#        • Missing G1-G16 ground truth entry
#   6. cargo fmt --check + cargo clippy -D warnings on titan-rust/ —
#      blocks Rust formatting drift + clippy warnings per Phase C C-S8
#      PLAN §15.4. Skipped if no titan-rust/ files staged.
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

# ── 3. Cache key registry audit ──────────────────────────────────
# Only runs when staged changes touch files that could affect the
# producer / cache-key / consumer chain.
STAGED_RELEVANT=$(git diff --cached --name-only --diff-filter=ACM \
    -- 'titan_plugin/api/**' 'titan_plugin/modules/**' \
       'titan_plugin/core/**' 'titan_plugin/bus.py' 2>/dev/null || true)

if [ -n "$STAGED_RELEVANT" ]; then
    if [ -x "$REPO_ROOT/test_env/bin/python" ]; then
        if ! "$REPO_ROOT/test_env/bin/python" "$REPO_ROOT/scripts/arch_map.py" \
                cache-keys --audit > /tmp/precommit_cache_keys.log 2>&1; then
            cat >&2 <<BANNER

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  cache_key_registry audit failed.                                │
│                                                                      │
│  The observatory data contract has drifted — a producer was renamed, │
│  a new cache.get() call was added without registering, a consumer    │
│  endpoint moved, or a bus constant was renamed.                      │
│                                                                      │
│  See full audit:                                                     │
│      cat /tmp/precommit_cache_keys.log                               │
│                                                                      │
│  Re-run manually:                                                    │
│      python scripts/arch_map.py cache-keys --audit                   │
│                                                                      │
│  Fix by editing titan_plugin/api/cache_key_registry.py.              │
│                                                                      │
│  Bypass (only when registry is intentionally being updated):         │
│      git commit --no-verify                                          │
└──────────────────────────────────────────────────────────────────────┘

BANNER
            tail -40 /tmp/precommit_cache_keys.log >&2
            exit 1
        fi
    fi
fi

# ── 4. Lazy-imports enforcement (I-002 closure 2026-04-27) ───────
# Block re-introduction of eager torch/transformers/faiss imports in
# worker modules. Test runs only when titan_plugin/*.py is staged.
STAGED_TP_PY=$(git diff --cached --name-only --diff-filter=ACM \
    -- 'titan_plugin/**/*.py' 2>/dev/null || true)

if [ -n "$STAGED_TP_PY" ]; then
    if [ -x "$REPO_ROOT/test_env/bin/python" ]; then
        if ! "$REPO_ROOT/test_env/bin/python" -m pytest \
                "$REPO_ROOT/tests/test_lazy_imports.py" \
                -p no:anchorpy -q --no-header --tb=short \
                > /tmp/precommit_lazy_imports.log 2>&1; then
            cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  Lazy-imports check failed.                                      │
│                                                                      │
│  A staged file in titan_plugin/ introduced (or re-introduced) an     │
│  eager module-level import of a heavy ML library — torch,            │
│  transformers, faiss, sentence_transformers, triton, or torchvision. │
│                                                                      │
│  Each Guardian-managed worker is its own Python process — eager      │
│  heavy imports cost RAM × N workers. History (commit 7f01125):       │
│  one eager torch import = ~860 MB × 9 workers ≈ 2.3 GB at boot.      │
│                                                                      │
│  Fix: move the import inside the function/method that uses it,       │
│  OR use PEP 562 `__getattr__` for module-level exports.              │
│                                                                      │
│  See: memory/feedback_lazy_imports_titan_plugin.md                   │
│       tests/test_lazy_imports.py (the canonical enforcement)         │
│                                                                      │
│  Full output:                                                        │
│      cat /tmp/precommit_lazy_imports.log                             │
│                                                                      │
│  Bypass (only when intentionally adding a heavy import to a non-     │
│  worker entry-point): git commit --no-verify                         │
└──────────────────────────────────────────────────────────────────────┘

BANNER
            tail -40 /tmp/precommit_lazy_imports.log >&2
            exit 1
        fi
    fi
fi

# ── 5. Phase C SPEC enforcer (titan-docs/SPEC_titan_architecture.md §20) ───
# Runs when SPEC scope is staged: TOML / generated files / generator /
# arch_map / any titan_plugin or titan-rust source. ~1s wall when active.
STAGED_SPEC_SCOPE=$(git diff --cached --name-only --diff-filter=ACM -- \
    'titan-docs/SPEC_titan_architecture*' \
    'titan_plugin/_phase_c_constants.py' \
    'titan-rust/crates/titan-core/src/constants.rs' \
    'scripts/generate_phase_c_constants.py' \
    'scripts/arch_map.py' \
    'titan_plugin/**/*.py' \
    'titan-rust/**' \
    2>/dev/null || true)

if [ -n "$STAGED_SPEC_SCOPE" ]; then
    if [ -x "$REPO_ROOT/test_env/bin/python" ]; then
        if ! "$REPO_ROOT/test_env/bin/python" "$REPO_ROOT/scripts/arch_map.py" \
                phase-c verify --strict > /tmp/precommit_phase_c.log 2>&1; then
            cat >&2 <<BANNER

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  phase-c verify --strict failed.                                 │
│                                                                      │
│  The SPEC has drifted from code, OR generated constants files were   │
│  hand-edited, OR the SPEC TOML was changed without regen.            │
│                                                                      │
│  See full output:                                                    │
│      cat /tmp/precommit_phase_c.log                                  │
│                                                                      │
│  Re-run manually:                                                    │
│      python scripts/arch_map.py phase-c verify                       │
│                                                                      │
│  Common fix:                                                         │
│      python scripts/arch_map.py phase-c regen                        │
│      git add titan_plugin/_phase_c_constants.py                      │
│      git add titan-rust/crates/titan-core/src/constants.rs           │
│                                                                      │
│  Per feedback_phase_c_spec_enforcement.md Rule 2:                    │
│  bypassing this check is a SPEC violation. Fix root cause.           │
│                                                                      │
│  Bypass (only after Maker greenlight + spec_version bump):           │
│      git commit --no-verify                                          │
└──────────────────────────────────────────────────────────────────────┘

BANNER
            tail -40 /tmp/precommit_phase_c.log >&2
            exit 1
        fi
    fi
fi

# ── 6. cargo fmt + clippy on Rust workspace (Phase C C-S8 PLAN §15.4) ─
# Runs only when staged changes touch titan-rust/. Drift discipline:
# lint warnings = build failure for Rust, mirroring SPEC strictness.
STAGED_RUST=$(git diff --cached --name-only --diff-filter=ACM -- \
    'titan-rust/**' 2>/dev/null || true)

if [ -n "$STAGED_RUST" ]; then
    if [ -d "$REPO_ROOT/titan-rust" ] && command -v cargo >/dev/null 2>&1; then
        cd "$REPO_ROOT/titan-rust"
        if ! cargo fmt --check > /tmp/precommit_cargo_fmt.log 2>&1; then
            cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  cargo fmt --check failed.                                       │
│                                                                      │
│  Rust source is not in canonical formatting. Run:                    │
│      cd titan-rust && cargo fmt --all                                │
│      git add -u                                                      │
│                                                                      │
│  See:  cat /tmp/precommit_cargo_fmt.log                              │
└──────────────────────────────────────────────────────────────────────┘

BANNER
            tail -20 /tmp/precommit_cargo_fmt.log >&2
            cd "$REPO_ROOT"
            exit 1
        fi
        if ! cargo clippy --workspace --all-targets -- -D warnings \
                > /tmp/precommit_cargo_clippy.log 2>&1; then
            cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  cargo clippy emitted warnings (treated as errors per SPEC).     │
│                                                                      │
│  Fix the warnings above OR add a justified #[allow(...)] at the     │
│  call site. Drift discipline applies: warnings = build failure.      │
│                                                                      │
│  Re-run manually:                                                    │
│      cd titan-rust && cargo clippy --workspace --all-targets         │
│                                                                      │
│  See:  cat /tmp/precommit_cargo_clippy.log                           │
└──────────────────────────────────────────────────────────────────────┘

BANNER
            tail -40 /tmp/precommit_cargo_clippy.log >&2
            cd "$REPO_ROOT"
            exit 1
        fi
        cd "$REPO_ROOT"
    fi
fi

exit 0
