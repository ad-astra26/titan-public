"""
Per-shadow data directory isolation (BUG-B1-SHARED-LOCKS fix).

When the shadow kernel boots in Phase 3, the original kernel is still
alive (we don't kill it until Phase 5, after Maker confirms the swap).
Both kernels would then try to bind the same lock-protected files —
DuckDB exclusive lock on `data/titan_memory.duckdb`, SQLite WAL on
`data/inner_memory.db`, FAISS index files. Even with PR_SET_PDEATHSIG
ensuring orphans don't accumulate, ACTIVE original-kernel workers hold
the same locks the shadow needs.

This module shapes a per-shadow data directory `data_shadow_<port>/`
that the shadow kernel uses instead of `data/`. Implementation:

1. **copy_data_dir(src, dst)** — hardlink-copy src/* → dst/*. Hardlinks
   share inodes so the copy is cheap (no extra disk for read-only files)
   AND DuckDB / SQLite / FAISS write semantics break the hardlink on
   first write to a file, giving each kernel its own copy on demand.

2. **swap_data_dirs(canonical, shadow)** — after a successful swap
   (Phase 5: original kernel gone), atomically rename:
      canonical → canonical.OLD.<ts>/
      shadow    → canonical/
   Future boots see the post-swap state as `data/` automatically — no
   env-var persistence needed.

3. **cleanup_shadow_dir(shadow)** — on swap failure, remove the shadow
   dir and any backing shadow files (lock files, WAL, etc.).

4. **resolve_data_path(rel)** — small helper for code paths that need
   to read TITAN_DATA_DIR-aware paths. Most code uses Path("data/X")
   relative-to-cwd; this helper redirects through env when set.

The shadow process is spawned with TITAN_DATA_DIR=<shadow_dir> in env;
worker entry points (IMW, observatory_writer, memory) call
resolve_data_path() at config-load time to pick up the env var.

rFP: BUG-B1-SHARED-LOCKS in BUGS.md (recommended fix path 1).
Composes with BUG-B1-WEAK-HEALTH-CHECK + worker_lifecycle PDEATHSIG.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_data_path(rel: str) -> str:
    """Resolve a 'data/...' relative path through TITAN_DATA_DIR env var.

    When TITAN_DATA_DIR is set (shadow process), `data/foo.db` resolves
    to `<TITAN_DATA_DIR>/foo.db`. When unset (original kernel), returns
    `data/foo.db` unchanged — fully backward compatible.

    Accepts paths with or without leading `data/`:
      resolve_data_path("data/foo.db") → "data/foo.db" or "<override>/foo.db"
      resolve_data_path("foo.db")       → "data/foo.db" or "<override>/foo.db"
    """
    base = os.environ.get("TITAN_DATA_DIR", "data")
    if rel.startswith("data/"):
        rel = rel[len("data/"):]
    elif rel == "data":
        return base
    return os.path.join(base, rel)


def shadow_data_dir_for_port(port: int, root: Path | str = ".") -> Path:
    """Canonical per-shadow data dir name. One per port avoids collisions
    across overlapping swap attempts."""
    return Path(root) / f"data_shadow_{int(port)}"


def _break_db_hardlinks(dst: Path) -> int:
    """Replace EVERY hardlinked top-level file in `dst` with a real copy.

    Defense-in-depth (2026-04-28 PM late, BUG-T1-INNER-MEMORY-CORRUPTION):
      Earlier versions of this function used pattern-matching to decide
      which files to un-hardlink (`*.db`, `*.duckdb`, `*.kuzu`, `*.kuzu.wal`).
      That left a corruption hole for SQLite auxiliary files (`*.db-wal`,
      `*.db-shm`) which are NOT a separate `.db` file but share an inode
      with the canonical's WAL state. When the shadow process opened its
      copied `inner_memory.db` and ran SQLite WAL recovery, recovery wrote
      to the (still hardlinked) `-wal` file — corrupting both kernels'
      view of the database. This is exactly what happened to T1's
      `inner_memory.db` and `observatory.db` during the 2026-04-28 15:00:34
      shadow_swap attempt.

    Root-cause-fix: walk ALL top-level files in `dst` and break any
    hardlink (`nlink > 1`) regardless of extension. The pattern-match
    safety floor was leaky; this removes the extension dependency.
    Auxiliary files SQLite/DuckDB/Kuzu may use without us knowing
    (`-journal`, `-wal`, `-shm`, `.tmp`, `.lock`, etc.) are now all
    covered by construction. So are append-only logs, FAISS indices,
    and anything else a future contributor adds at the top level.

    Background — original (T2 swap, 2026-04-27 PM):
      `cp -al` (hardlink) shares inodes between src and dst. The earlier
      docstring claimed "SQLite/DuckDB break the hardlink on first write"
      — FALSE for SQLite WAL mode: writes go to a separate `-wal` file,
      leaving the main `.db` inode untouched until checkpoint, so the
      hardlink persists. Two kernels with the same `.db` inode open can
      race during checkpoint, corrupting pages.

    Top-level only — subdirs (backups/, history/, meta_teacher/, etc.)
    stay hardlinked because they aren't concurrently written during the
    swap window. Append-only files in subdirs (`.jsonl` logs) would be
    a corruption risk if they are concurrently written during a swap;
    none are today, but a future audit should re-check before adding
    any concurrently-written subdir.

    Trade-off: breaking every top-level hardlink costs disk I/O during
    swap (each file becomes a real copy). On T1 this is ~5–10 GB of
    SQLite/DuckDB DBs plus ~50 MB of small files. Acceptable. The cost
    of pattern-leak corruption is *all of inner_memory.db's history*,
    which is incomparably worse.

    Returns the number of files that had their hardlink broken.
    """
    broken = 0
    for f in dst.iterdir():
        if not f.is_file():
            continue
        try:
            if f.stat().st_nlink <= 1:
                continue  # already a separate inode
            tmp = f.with_suffix(f.suffix + ".unlink.tmp")
            shutil.copy2(f, tmp)
            os.replace(tmp, f)
            broken += 1
        except Exception as e:
            logger.warning(
                "[shadow_data_dir] failed to break hardlink for %s: %s — "
                "shadow may contend with old kernel on this file", f, e)
    return broken


def break_canonical_db_hardlinks(canonical: Path | str) -> int:
    """Phase B.3 Layer 3 — break leftover hardlinks on canonical data dir's
    SQLite/DuckDB DBs before a swap.

    Background: every successful shadow swap calls swap_data_dirs() which
    rotates data/ → data.OLD.<ts>/ via filesystem rename (inode-preserving).
    Result: the new data/<file> and data.OLD.<ts>/<file> share the same
    inode for any file that was hardlinked during cp -al (which is most of
    them when reflink is unavailable, i.e., ext4 VPS). Concrete state on T1
    after 2026-04-26 swap: `data/timechain/index.db` and
    `data.OLD.20260427_184938/timechain/index.db` are inode 5548048 with
    link_count=2.

    On the NEXT swap, cp -al into data_shadow_<new_port>/ creates yet
    another hardlink — link_count=3. This compounds the corruption surface
    across multiple swaps: any process that touches data.OLD/<file> can
    affect data/<file>.

    This function walks the canonical data dir + breaks the hardlink on
    every SQLite/DuckDB file that has link_count > 1. For each:
      1. shutil.copy2 to a .unlink.tmp sibling
      2. os.replace to atomically swap into place
    After this, each at-risk DB has link_count=1 (sole link, no inode
    sharing with any prior data.OLD/). Next swap's cp -al starts fresh.

    Symmetric to _break_db_hardlinks(dst) which operates on the shadow
    dir post-cp; this one operates on the canonical data dir pre-cp.
    Together they ensure no inode-sharing path exists at any point in
    the swap lifecycle.

    Returns the number of hardlinks broken. Files matching the SQLite/
    DuckDB extensions but already at link_count=1 are skipped silently.
    Top-level only (subdirs left alone — those rarely hold concurrent-
    write DBs and breaking links there would be expensive).
    """
    canonical = Path(canonical)
    if not canonical.exists():
        return 0
    broken = 0
    # Defense-in-depth (2026-04-28 PM late, BUG-T1-INNER-MEMORY-CORRUPTION) —
    # break ALL top-level hardlinks regardless of extension. Pattern-match
    # previously missed *.db-wal / *.db-shm and corrupted T1's inner_memory.db.
    # Symmetric to _break_db_hardlinks(dst); see that function's docstring
    # for full incident analysis.
    for f in canonical.iterdir():
        if not f.is_file():
            continue
        try:
            if f.stat().st_nlink <= 1:
                continue
            tmp = f.with_suffix(f.suffix + ".unlink.tmp")
            shutil.copy2(f, tmp)
            os.replace(tmp, f)
            broken += 1
        except Exception as e:
            logger.warning(
                "[shadow_data_dir] Layer 3 hardlink-break failed for "
                "%s: %s — file remains link>1 (corruption surface "
                "until next successful break)", f, e)
    # Also handle the timechain/index.db (subdir, but critical — it's the
    # one that actually corrupted on 2026-04-26).
    tc_index = canonical / "timechain" / "index.db"
    if tc_index.exists() and tc_index.is_file():
        try:
            if tc_index.stat().st_nlink > 1:
                tmp = tc_index.with_suffix(".db.unlink.tmp")
                shutil.copy2(tc_index, tmp)
                os.replace(tmp, tc_index)
                broken += 1
        except Exception as e:
            logger.warning(
                "[shadow_data_dir] Layer 3 hardlink-break failed for "
                "timechain/index.db: %s", e)
    return broken


def copy_data_dir(src: Path | str, dst: Path | str,
                  use_reflink: bool = True) -> tuple[bool, str]:
    """Replicate src into dst as a per-shadow snapshot.

    Strategy preference (in order):
      1. reflink (`cp --reflink=always`) — copy-on-write at the FS level
         (btrfs/xfs/zfs); zero extra disk until shadow writes, then
         per-block divergence. Fastest + cheapest. Inherently safe for
         SQLite/DuckDB because each writer's first page-write triggers
         a per-block COW.
      2. hardlink (`cp -al`) — same inode for unchanged files. UNSAFE
         for SQLite WAL-mode mainfiles (inode shared, writes don't
         break the link). After this branch we call _break_db_hardlinks
         to replace top-level `*.db` + `*.duckdb` with real copies, so
         the shadow's IMW/DuckDB writers see independent inodes. Other
         files (JSON caches, FAISS, etc.) stay hardlinked — they either
         tolerate inode sharing or follow atomic-rename patterns that
         break the link naturally.
      3. plain copy (`cp -a`) — fallback when neither works (e.g. cross-
         filesystem). Slower but always works.

    Returns (ok, method) where method ∈ {"reflink", "hardlink", "copy"}.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False, "src_missing"
    # If dst exists, refuse — caller should cleanup_shadow_dir first
    if dst.exists():
        return False, "dst_exists"

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Each attempted method must produce: dst/foo, dst/sub/bar, ...
    # (src CONTENTS in dst, not src ITSELF inside dst). The `src/.` form
    # forces this regardless of dst's pre-existence. Between attempts we
    # also remove dst if a partial-failure left an empty dir behind.
    src_arg = f"{src}/."

    def _try(extra_flags: list[str]) -> int:
        # Recreate dst fresh for each attempt — partial state from a
        # failed reflink would otherwise corrupt the next attempt.
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True, exist_ok=True)
        return subprocess.run(
            ["cp"] + extra_flags + ["-a", src_arg, str(dst)],
            capture_output=True,
        ).returncode

    if use_reflink:
        if _try(["--reflink=always"]) == 0:
            return True, "reflink"

    if _try(["-l"]) == 0:
        broken = _break_db_hardlinks(dst)
        if broken:
            logger.info(
                "[shadow_data_dir] broke %d SQLite/DuckDB hardlinks in %s "
                "(safety: shadow gets independent inodes for DB writers)",
                broken, dst)
        return True, "hardlink"

    # Last resort: real copy
    rc = subprocess.run(
        ["cp", "-a", src_arg, str(dst)], capture_output=True,
    )
    if rc.returncode == 0:
        return True, "copy"

    return False, f"all_methods_failed: {rc.stderr.decode()[:200]}"


def cleanup_shadow_dir(shadow_dir: Path | str) -> bool:
    """Remove a shadow dir + all its contents. Used on swap failure or
    after a successful canonical-swap (data → data.OLD; shadow → data).

    Returns True if the directory was removed (or was already gone).
    """
    p = Path(shadow_dir)
    if not p.exists():
        return True
    try:
        shutil.rmtree(p)
        return True
    except Exception as e:
        logger.warning("[shadow_data_dir] cleanup failed for %s: %s", p, e)
        return False


def swap_data_dirs(canonical: Path | str, shadow: Path | str) -> tuple[bool, str]:
    """After successful swap: rotate canonical → backup; shadow → canonical.

    Steps (all atomic at the rename level):
      1. Move canonical → canonical.OLD.<ts>
      2. Move shadow → canonical
      3. Schedule canonical.OLD.<ts> for cleanup (caller's choice — kept
         by default for one swap so failure can recover, removed on next
         successful swap)

    Returns (ok, message). On any failure we attempt to roll back and
    leave the world in its starting state.
    """
    canonical = Path(canonical)
    shadow = Path(shadow)
    if not shadow.exists():
        return False, "shadow_dir_missing"

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    backup = Path(f"{canonical}.OLD.{ts}")

    # Step 1: canonical → backup
    if canonical.exists():
        try:
            canonical.rename(backup)
        except Exception as e:
            return False, f"backup_rename_failed: {e}"

    # Step 2: shadow → canonical
    try:
        shadow.rename(canonical)
    except Exception as e:
        # Roll back step 1 if needed
        if backup.exists() and not canonical.exists():
            try:
                backup.rename(canonical)
            except Exception:
                pass
        return False, f"shadow_promotion_failed: {e}"

    return True, f"backup={backup}"


def cleanup_old_backups(canonical: Path | str, keep_count: int = 2) -> int:
    """Remove canonical.OLD.<ts> dirs beyond `keep_count` most-recent.

    Called after a successful swap to bound disk usage. Default keeps
    2 most-recent backups (current + 1 prior).
    """
    canonical = Path(canonical)
    parent = canonical.parent
    name = canonical.name
    candidates = sorted(parent.glob(f"{name}.OLD.*"), reverse=True)
    removed = 0
    for old in candidates[keep_count:]:
        if cleanup_shadow_dir(old):
            removed += 1
    return removed
