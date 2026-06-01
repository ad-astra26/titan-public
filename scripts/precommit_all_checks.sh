#!/usr/bin/env bash
# precommit_all_checks.sh — composite pre-commit hook.
#
# Runs in order:
#   1. scripts/precommit_async_check.sh — blocks async-boundary violations
#      in titan_hcl/*.py (existing, established 2026-04-14)
#   2. gitleaks on staged diffs — blocks commits that introduce secrets
#      (added 2026-04-15 after public-repo leak incident)
#   3. tests/test_lazy_imports.py — blocks commits that re-introduce
#      eager module-level imports of torch/transformers/faiss/etc into
#      worker modules (added 2026-04-27 closing DEFERRED I-002). Skipped
#      if no titan_hcl/*.py files staged. ~5s wall when active.
#      Reason: a single eager torch import in a hot-path file costs
#      ~860MB × 9 workers ≈ 2.3GB at boot (per commit 7f01125 history).
#   4. arch_map phase-c verify --strict — blocks SPEC drift + G-RPC
#      violations per titan-docs/specs/SPEC_titan_architecture.md §20 + Rule
#      2 of feedback_phase_c_spec_enforcement.md. Runs when SPEC TOML,
#      generated constants files, generator script, arch_map.py, or any
#      titan_hcl/ / titan-rust/ source is staged. Catches:
#        • Hand-edits to _phase_c_constants.py / constants.rs (regen drift)
#        • TOML constant added without regen
#        • Domain used by constant but missing from [domains.X] block
#        • Missing G1-G16 ground truth entry
#        • G-RPC-1 sync bus.request outside phase_c_rpc_exemptions.yaml
#        • G-RPC-2 bus.request* without explicit timeout kwarg
#        • G-RPC-3 proxy `def get_*` not reading SHM (Preamble G18)
#        • G-RPC-4 orphan `if action == "get_*"` handler (caller graph)
#        • G-RPC-5 forbidden _cache.get(state_key) read OR Python-side
#          StateRegistryWriter for Rust-canonical L0+L1 slot (D-SPEC-81
#          Phase E enforcement gate — blocks bus-cache drift recurrence)
#      G-RPC-1..4 added 2026-05-07 by Phase C Session 5 rFP §4.E.
#      G-RPC-5 added 2026-05-18 by Phase E (D-SPEC-83 v1.22.0).
#   5. cargo fmt --check + cargo clippy -D warnings on titan-rust/ —
#      blocks Rust formatting drift + clippy warnings per Phase C C-S8
#      PLAN §15.4. Skipped if no titan-rust/ files staged.
#
# RETIRED 2026-05-18 (D-SPEC-80 Phase D):
#   - arch_map cache-keys --audit (former step 3) — cache_key_registry
#     deleted along with the bus-cache → CachedState pipeline. G-RPC-5
#     in step 4 supersedes the audit role.
#
# Enable:   ln -sf ../../scripts/precommit_all_checks.sh .git/hooks/pre-commit
# Bypass:   git commit --no-verify  (use sparingly; each check has a reason)

set -u

# 2026-05-14 — use git's worktree-aware resolver instead of
# `$(cd "$(dirname "$0")/.." && pwd)`. The dirname-based form was broken
# in git worktrees because `.git/hooks/pre-commit` is a SHARED symlink
# pointing to this script in the MAIN repo (lives at
# `<main-repo>/.git/hooks/pre-commit` regardless of which worktree
# triggers the commit), so dirname/.. always resolved to the main repo's
# `.git/` dir — meaning `$REPO_ROOT/scripts/tracker_indexer.py` tried to
# read `/path/to/main/.git/scripts/tracker_indexer.py` (doesn't exist)
# and the hook bailed with a misleading "tracker drift" banner.
# `git rev-parse --show-toplevel` returns the ACTIVE worktree path, so
# `scripts/*` refs resolve correctly from any worktree. Discovered
# 2026-05-14 during BUG-CHRONICLE-WRITER-DEAD-POST-A87 filing — first
# commit in the session to touch a tracker body from a worktree.
REPO_ROOT="$(git rev-parse --show-toplevel)"
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

# ── 3. Lazy-imports enforcement (I-002 closure 2026-04-27) ───────
# Block re-introduction of eager torch/transformers/faiss imports in
# worker modules. Test runs only when titan_hcl/*.py is staged.
STAGED_TP_PY=$(git diff --cached --name-only --diff-filter=ACM \
    -- 'titan_hcl/**/*.py' 2>/dev/null || true)

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
│  A staged file in titan_hcl/ introduced (or re-introduced) an     │
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
│  See: memory/feedback_lazy_imports_titan_hcl.md                   │
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

# ── 4. Phase C SPEC enforcer (titan-docs/specs/SPEC_titan_architecture.md §20) ───
# Runs when SPEC scope is staged: TOML / generated files / generator /
# arch_map / any titan_hcl or titan-rust source. ~1s wall when active.
STAGED_SPEC_SCOPE=$(git diff --cached --name-only --diff-filter=ACM -- \
    'titan-docs/specs/SPEC_titan_architecture*' \
    'titan_hcl/_phase_c_constants.py' \
    'titan-rust/crates/titan-core/src/constants.rs' \
    'scripts/generate_phase_c_constants.py' \
    'scripts/arch_map.py' \
    'titan_hcl/**/*.py' \
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
│      git add titan_hcl/_phase_c_constants.py                      │
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
        # G17 ownership scanner — closes DEFERRED CRATE-OWNERSHIP-PRECOMMIT-INTEGRATION
        # per C-S6 chunk C6-12. Verifies each architectural concept lives in
        # EXACTLY ONE crate per master plan §7 + SPEC Preamble G17. Soaked
        # through C-S6 (no false positives surfaced) before mandatory activation.
        if ! "$REPO_ROOT/test_env/bin/python" "$REPO_ROOT/scripts/arch_map.py" \
                phase-c crate-ownership --strict > /tmp/precommit_phase_c_ownership.log 2>&1; then
            cat >&2 <<BANNER

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  phase-c crate-ownership --strict failed (G17 violation).        │
│                                                                      │
│  An architectural concept is defined in MULTIPLE crates — duplicate  │
│  ownership violates SPEC Preamble G17 + master plan §7. The sibling  │
│  byte-layout / wiring scanners CANNOT catch this; it's exactly the   │
│  C-S5 SchumannTicker duplication incident that motivated this gate.  │
│                                                                      │
│  See full output:                                                    │
│      cat /tmp/precommit_phase_c_ownership.log                        │
│                                                                      │
│  Re-run manually:                                                    │
│      python scripts/arch_map.py phase-c crate-ownership              │
│                                                                      │
│  Fix: move the duplicate into the canonical crate listed in          │
│  scripts/arch_map.py::_ARCH_OWNERSHIP_REGISTRY, or update the        │
│  registry IF the concept genuinely moved (rare — needs SPEC bump).   │
│                                                                      │
│  Bypass (only after Maker greenlight):                               │
│      git commit --no-verify                                          │
└──────────────────────────────────────────────────────────────────────┘

BANNER
            tail -40 /tmp/precommit_phase_c_ownership.log >&2
            exit 1
        fi
    fi
fi

# ── 5. cargo fmt + clippy on Rust workspace (Phase C C-S8 PLAN §15.4) ─
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

# ─────────────────────────────────────────────────────────────────────────────
# 7. Tracker index drift gate — added 2026-05-13.
#
# When BUGS.md / OBSERVABLES.md / DEFERRED_ITEMS.md is edited, the matching
# *_index.md must be regenerated in the same commit. Post-commit hook
# auto-regenerates for the NEXT commit; this gate prevents shipping a body
# change without its synced index. Skipped if no tracker body is staged.
# Discipline rule: memory/feedback_tracker_index_discipline.md
# ─────────────────────────────────────────────────────────────────────────────
STAGED_TRACKERS=$(git diff --cached --name-only --diff-filter=ACM -- \
    titan-docs/BUGS.md \
    titan-docs/OBSERVABLES.md \
    titan-docs/DEFERRED_ITEMS.md \
    2>/dev/null)

# Architecture doc drift gate — added 2026-05-19.
# When ARCHITECTURE_cgn_family.md is edited, ARCHITECTURE_cgn_family_index.md
# must be regenerated in the same commit. Same discipline as SPEC_index.
STAGED_ARCH=$(git diff --cached --name-only --diff-filter=ACM -- \
    titan-docs/specs/ARCHITECTURE_cgn_family.md \
    2>/dev/null)

if [ -n "$STAGED_ARCH" ]; then
    PY="$REPO_ROOT/test_env/bin/python"
    [ -x "$PY" ] || PY=python3
    if ! "$PY" "$REPO_ROOT/scripts/architecture_cgn_family_index.py" --check 2>/tmp/precommit_arch_drift.log; then
        cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  Architecture index drift detected.                              │
│                                                                      │
│  ARCHITECTURE_cgn_family.md was edited but its _index.md is stale.   │
│                                                                      │
│  Fix:                                                                │
│      python scripts/architecture_cgn_family_index.py                 │
│      git add titan-docs/specs/ARCHITECTURE_cgn_family_index.md             │
│                                                                      │
│  See: cat /tmp/precommit_arch_drift.log                              │
└──────────────────────────────────────────────────────────────────────┘

BANNER
        cat /tmp/precommit_arch_drift.log >&2
        exit 1
    fi
fi

# Synthesis-engine architecture doc drift gate — added 2026-05-20.
# When ARCHITECTURE_synthesis_engine.md is edited, its _index.md must be
# regenerated in the same commit. Same discipline as the CGN-family gate above.
STAGED_SYNTH_ARCH=$(git diff --cached --name-only --diff-filter=ACM -- \
    titan-docs/specs/ARCHITECTURE_synthesis_engine.md \
    2>/dev/null)

if [ -n "$STAGED_SYNTH_ARCH" ]; then
    PY="$REPO_ROOT/test_env/bin/python"
    [ -x "$PY" ] || PY=python3
    if ! "$PY" "$REPO_ROOT/scripts/architecture_synthesis_engine_index.py" --check 2>/tmp/precommit_synth_arch_drift.log; then
        cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  Synthesis-engine architecture index drift detected.             │
│                                                                      │
│  ARCHITECTURE_synthesis_engine.md was edited but its _index.md is    │
│  stale.                                                              │
│                                                                      │
│  Fix:                                                                │
│      python scripts/architecture_synthesis_engine_index.py           │
│      git add titan-docs/specs/ARCHITECTURE_synthesis_engine_index.md       │
│                                                                      │
│  See: cat /tmp/precommit_synth_arch_drift.log                        │
└──────────────────────────────────────────────────────────────────────┘

BANNER
        cat /tmp/precommit_synth_arch_drift.log >&2
        exit 1
    fi
fi

# When ARCHITECTURE_api_family.md is edited, its _index.md must be
# regenerated in the same commit. Same discipline as the CGN-family +
# synthesis-engine gates above.
STAGED_API_ARCH=$(git diff --cached --name-only --diff-filter=ACM -- \
    titan-docs/specs/ARCHITECTURE_api_family.md \
    2>/dev/null)

if [ -n "$STAGED_API_ARCH" ]; then
    PY="$REPO_ROOT/test_env/bin/python"
    [ -x "$PY" ] || PY=python3
    if ! "$PY" "$REPO_ROOT/scripts/architecture_api_family_index.py" --check 2>/tmp/precommit_api_arch_drift.log; then
        cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  API-family architecture index drift detected.                   │
│                                                                      │
│  ARCHITECTURE_api_family.md was edited but its _index.md is stale.   │
│                                                                      │
│  Fix:                                                                │
│      python scripts/architecture_api_family_index.py                 │
│      git add titan-docs/specs/ARCHITECTURE_api_family_index.md             │
│                                                                      │
│  See: cat /tmp/precommit_api_arch_drift.log                          │
└──────────────────────────────────────────────────────────────────────┘

BANNER
        cat /tmp/precommit_api_arch_drift.log >&2
        exit 1
    fi
fi

# When ARCHITECTURE_trinity.md is edited, its _index.md must be
# regenerated in the same commit. Same discipline as the CGN-family +
# synthesis-engine + API-family gates above.
STAGED_TRINITY_ARCH=$(git diff --cached --name-only --diff-filter=ACM -- \
    titan-docs/specs/ARCHITECTURE_trinity.md \
    2>/dev/null)

if [ -n "$STAGED_TRINITY_ARCH" ]; then
    PY="$REPO_ROOT/test_env/bin/python"
    [ -x "$PY" ] || PY=python3
    if ! "$PY" "$REPO_ROOT/scripts/architecture_trinity_index.py" --check 2>/tmp/precommit_trinity_arch_drift.log; then
        cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  Trinity architecture index drift detected.                      │
│                                                                      │
│  ARCHITECTURE_trinity.md was edited but its _index.md is stale.      │
│                                                                      │
│  Fix:                                                                │
│      python scripts/architecture_trinity_index.py                    │
│      git add titan-docs/specs/ARCHITECTURE_trinity_index.md               │
│                                                                      │
│  See: cat /tmp/precommit_trinity_arch_drift.log                      │
└──────────────────────────────────────────────────────────────────────┘

BANNER
        cat /tmp/precommit_trinity_arch_drift.log >&2
        exit 1
    fi
fi

# Backup/Restore architecture doc drift gate — added 2026-06-01.
# When ARCHITECTURE_backup_restore.md is edited, its _index.md must be
# regenerated in the same commit. Same discipline as the gates above.
STAGED_BACKUP_RESTORE_ARCH=$(git diff --cached --name-only --diff-filter=ACM -- \
    titan-docs/specs/ARCHITECTURE_backup_restore.md \
    2>/dev/null)

if [ -n "$STAGED_BACKUP_RESTORE_ARCH" ]; then
    PY="$REPO_ROOT/test_env/bin/python"
    [ -x "$PY" ] || PY=python3
    if ! "$PY" "$REPO_ROOT/scripts/architecture_backup_restore_index.py" --check 2>/tmp/precommit_backup_restore_arch_drift.log; then
        cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  Backup/Restore architecture index drift detected.               │
│                                                                      │
│  ARCHITECTURE_backup_restore.md was edited but its _index.md is stale.│
│                                                                      │
│  Fix:                                                                │
│      python scripts/architecture_backup_restore_index.py             │
│      git add titan-docs/specs/ARCHITECTURE_backup_restore_index.md         │
│                                                                      │
│  See: cat /tmp/precommit_backup_restore_arch_drift.log               │
└──────────────────────────────────────────────────────────────────────┘

BANNER
        cat /tmp/precommit_backup_restore_arch_drift.log >&2
        exit 1
    fi
fi

# Mainnet-birth/resurrection architecture doc drift gate — added 2026-06-01.
STAGED_MBR_ARCH=$(git diff --cached --name-only --diff-filter=ACM -- \
    titan-docs/specs/ARCHITECTURE_mainnet_birth_resurrection.md \
    2>/dev/null)

if [ -n "$STAGED_MBR_ARCH" ]; then
    PY="$REPO_ROOT/test_env/bin/python"
    [ -x "$PY" ] || PY=python3
    if ! "$PY" "$REPO_ROOT/scripts/architecture_mainnet_birth_resurrection_index.py" --check 2>/tmp/precommit_mbr_arch_drift.log; then
        cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  Mainnet-birth/resurrection architecture index drift detected.   │
│                                                                      │
│  ARCHITECTURE_mainnet_birth_resurrection.md edited but _index stale. │
│                                                                      │
│  Fix:                                                                │
│      python scripts/architecture_mainnet_birth_resurrection_index.py │
│      git add titan-docs/specs/ARCHITECTURE_mainnet_birth_resurrection_index.md │
│                                                                      │
│  See: cat /tmp/precommit_mbr_arch_drift.log                         │
└──────────────────────────────────────────────────────────────────────┘

BANNER
        cat /tmp/precommit_mbr_arch_drift.log >&2
        exit 1
    fi
fi

if [ -n "$STAGED_TRACKERS" ]; then
    PY="$REPO_ROOT/test_env/bin/python"
    [ -x "$PY" ] || PY=python3
    if ! "$PY" "$REPO_ROOT/scripts/tracker_indexer.py" --check-all 2>/tmp/precommit_tracker_drift.log; then
        cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  Tracker index drift detected.                                   │
│                                                                      │
│  A staged tracker body was edited but its *_index.md is stale.       │
│                                                                      │
│  Fix:                                                                │
│      python scripts/tracker_indexer.py --emit-all                    │
│      git add titan-docs/BUGS_index.md \                              │
│              titan-docs/OBSERVABLES_index.md \                       │
│              titan-docs/DEFERRED_index.md                            │
│                                                                      │
│  See: cat /tmp/precommit_tracker_drift.log                           │
│  Discipline: memory/feedback_tracker_index_discipline.md             │
└──────────────────────────────────────────────────────────────────────┘

BANNER
        cat /tmp/precommit_tracker_drift.log >&2
        exit 1
    fi

    # ── 6b. Graveyard-split drift gate (2026-05-26) ─────────────────────
    # When OBSERVABLES.md or BUGS.md is edited and a previously-active
    # entry gets marked closed (strikethrough table row OR closure marker
    # in detail-block header), it MUST be physically moved to the
    # corresponding *_graveyard.md so the active file stays lean.
    #
    # `--check` mode exits 1 if any closed entry still lives in the
    # active body file. Fix: run the split script + git add both files.
    #
    # Closes the "active file bloats to 500KB" class (OBSERVABLES.md
    # reached 467KB / 4870 lines pre-split). Discipline mirrors the
    # tracker_indexer drift gate above — CHECK in hook, MANUAL regen.
    if ! "$PY" "$REPO_ROOT/scripts/split_observables_graveyard.py" --check 2>/tmp/precommit_obs_graveyard_drift.log; then
        cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  OBSERVABLES_graveyard split is STALE.                           │
│                                                                      │
│  Closed entries (~~OBS-X~~ in table OR ✅/🔁/SUPERSEDED in detail    │
│  header) are still in OBSERVABLES.md — should be in graveyard file.  │
│                                                                      │
│  Fix:                                                                │
│      python scripts/split_observables_graveyard.py                   │
│      git add titan-docs/OBSERVABLES.md \                             │
│              titan-docs/OBSERVABLES_graveyard.md                     │
│                                                                      │
│  See: cat /tmp/precommit_obs_graveyard_drift.log                     │
└──────────────────────────────────────────────────────────────────────┘

BANNER
        cat /tmp/precommit_obs_graveyard_drift.log >&2
        exit 1
    fi
    if ! "$PY" "$REPO_ROOT/scripts/split_bugs_graveyard.py" --check 2>/tmp/precommit_bugs_graveyard_drift.log; then
        cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  BUGS_graveyard split is STALE.                                  │
│                                                                      │
│  Closed entries (~~BUG-X~~ or ✅ FIXED in detail header, or          │
│  **Status:** FIXED in body) are still in BUGS.md — should be in      │
│  graveyard file.                                                     │
│                                                                      │
│  Fix:                                                                │
│      python scripts/split_bugs_graveyard.py                          │
│      git add titan-docs/BUGS.md titan-docs/BUGS_graveyard.md         │
│                                                                      │
│  See: cat /tmp/precommit_bugs_graveyard_drift.log                    │
└──────────────────────────────────────────────────────────────────────┘

BANNER
        cat /tmp/precommit_bugs_graveyard_drift.log >&2
        exit 1
    fi
fi

# ── 7. Trinity SPEC-conformance gate ─────────────────────────────────────────
# Blocks any commit that REGRESSES a LOCKED §G5.1/§G5.2/§G10/§G11/§9.5 clause, or
# marks a clause LOCKED without a passing test, per the clause→test manifest
# titan-docs/specs/TRINITY_CONFORMANCE.md (checker: scripts/trinity_conformance.py).
# PENDING (not-yet-conforming) clauses are reported LOUDLY but do NOT block routine
# commits (so P0 implementation can proceed). Runs only when trinity tensor/pulse
# code or the trinity spec/manifest is staged. Discipline:
# feedback_spec_bound_work_zero_simplification_clause_by_clause.
STAGED_TRINITY=$(git diff --cached --name-only --diff-filter=ACM -- \
    'titan-rust/crates/titan-trinity-daemon/**' \
    'titan-rust/crates/titan-trinity-rs/**' \
    'titan-rust/crates/titan-unified-spirit-rs/**' \
    'titan-rust/crates/titan-inner-body-rs/**' \
    'titan-rust/crates/titan-inner-mind-rs/**' \
    'titan-rust/crates/titan-inner-spirit-rs/**' \
    'titan-rust/crates/titan-outer-body-rs/**' \
    'titan-rust/crates/titan-outer-mind-rs/**' \
    'titan-rust/crates/titan-outer-spirit-rs/**' \
    'titan-docs/specs/TRINITY_CONFORMANCE.md' \
    'titan-docs/specs/ARCHITECTURE_trinity.md' 2>/dev/null || true)
if [ -n "$STAGED_TRINITY" ]; then
    PY="$REPO_ROOT/test_env/bin/python"
    [ -x "$PY" ] || PY=python3
    if ! "$PY" "$REPO_ROOT/scripts/trinity_conformance.py" > /tmp/precommit_trinity_conformance.log 2>&1; then
        cat >&2 <<'BANNER'

┌──────────────────────────────────────────────────────────────────────┐
│  ⛔  Trinity SPEC-conformance gate FAILED.                           │
│                                                                      │
│  A LOCKED §G5.x/§9 clause regressed, a clause is marked LOCKED        │
│  without a passing test, or a clause has no mapped test. The trinity  │
│  must be 100% per spec — no dropped or simplified terms.             │
│                                                                      │
│  Run:  python scripts/trinity_conformance.py                         │
│  Map:  titan-docs/specs/TRINITY_CONFORMANCE.md                       │
└──────────────────────────────────────────────────────────────────────┘

BANNER
        cat /tmp/precommit_trinity_conformance.log >&2
        exit 1
    fi
    # Surface the PENDING (not-yet-conforming) clause list even on a pass.
    sed -n '/TRINITY NOT YET 100%/,/promote each/p' /tmp/precommit_trinity_conformance.log 2>/dev/null || true
fi

exit 0
