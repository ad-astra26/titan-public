"""
titan_hcl/logic/local_diff.py — L5 (2026-05-14) local diff/baseline engine.

Replaces the "ship a full ~30MB personality tarball every meditation" pattern
with: weekly full SNAPSHOT (baseline) on Sunday + daily INCREMENTAL tarballs
(xdelta3 patches against baseline) Mon-Sat.

Expected savings: ~4× disk reduction per Titan over the 30-day rolling window
  before: 30 days × 30MB = 900MB
  after:  4 baselines × 30MB + 26 incrementals × ~2MB ≈ 172MB

Manifest: data/backups/local_diff_manifest_{titan_id}.json — tracks the
baseline/incremental chain. Restore walks the manifest backward to the
latest baseline + applies incrementals in order.

Design properties:
  - Feature-gated by [backup].local_diff_enabled (default false — opt-in).
    When off, code path is unchanged (full daily tarballs).
  - LOCAL-ONLY scope. Arweave path is separate (see rFP_backup_diff_baseline_unified_v1).
  - Per-format diff: chains/DBs/JSONs all go through xdelta3 (uniform tool).
    rFP's per-format split (tail-bytes for chains) is an optimization deferred
    until measurements show it matters; v1 uses one tool for simplicity.
  - Skip-if-unchanged: per-file SHA256 compare; unchanged files marked
    "skipped" in incremental manifest with a pointer to the baseline.
  - Restore is its own dedicated path — does NOT touch the full-tarball
    restore logic in backup.py.
  - Atomic-write semantics: manifest written via .tmp + rename, tarballs
    likewise. Failed events leave the prior chain intact.
"""

import json
import logging
import os
import subprocess
import tarfile
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────

MANIFEST_NAME_FMT = "local_diff_manifest_{titan_id}.json"
MANIFEST_SCHEMA_VERSION = 1
BASELINE_TARBALL_FMT = "personality_baseline_{date}_{hash8}.tar.gz"
INCREMENTAL_TARBALL_FMT = "personality_incremental_{date}_{hash8}.tar.gz"
# Baseline cadence: weekly on Sunday (UTC weekday 6, but Python's
# datetime.weekday() returns Monday=0 / Sunday=6).
BASELINE_WEEKDAY = 6  # Sunday
XDELTA3_BIN = "xdelta3"


# ── Manifest dataclasses ─────────────────────────────────────────────


@dataclass
class FileEntry:
    """One file's record within a baseline or incremental event."""
    arc_name: str  # archive member name (matches PERSONALITY_PATHS mapping)
    sha256: str    # hex digest of the file's bytes
    size_bytes: int
    diff_mode: str  # "full" (in baseline) | "patch" (in incremental) | "skipped"
    # Only set for incrementals with diff_mode="skipped" or "patch":
    baseline_sha256: Optional[str] = None  # the baseline's hash of this file


@dataclass
class DiffEvent:
    """One backup event in the local diff chain."""
    event_id: str         # uuid4
    ts_unix: float
    iso_date: str         # YYYY-MM-DD UTC
    type: str             # "baseline" | "incremental"
    tarball_name: str     # basename in local_dir
    tarball_sha256: str
    tarball_size_bytes: int
    files: list[FileEntry] = field(default_factory=list)
    # For incrementals: pointer to the baseline they diff against.
    baseline_event_id: Optional[str] = None
    # For incrementals: count of files marked patched / skipped / removed.
    patched_count: int = 0
    skipped_count: int = 0
    removed_count: int = 0


# ── Diff primitives (xdelta3 wrapper) ─────────────────────────────────


def _xdelta3_encode(baseline_path: str, current_path: str) -> Optional[str]:
    """Produce an xdelta3 patch FILE (current - baseline), STREAMED to disk.

    2026-06-04 fix (BUG-BACKUP-RSS-FLAP): previously `capture_output=True`
    returned the patch BYTES — for a multi-GB DB whose daily diff is GB-scale
    that buffered the whole patch in this process's RSS → guardian rss_limit
    kill → backup flap. Now xdelta3 writes the patch to a temp file (trailing
    output arg, no stdout capture) so the patch never enters RAM here; we return
    its PATH (the caller streams it into the tar via addfile(fileobj) + unlinks
    it). The patch CONTENT is byte-identical — only its location changes.
    Returns the patch path or None on error.
    """
    fd, patch_path = tempfile.mkstemp(suffix=".vcdiff")
    os.close(fd)
    try:
        result = subprocess.run(
            [XDELTA3_BIN, "-e", "-s", baseline_path, current_path, patch_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120,
        )
        if result.returncode != 0:
            logger.warning(
                "[local_diff] xdelta3 encode failed (%s vs %s): rc=%d stderr=%s",
                baseline_path, current_path, result.returncode,
                result.stderr[:200].decode("utf-8", "replace"))
            try:
                os.unlink(patch_path)
            except OSError:
                pass
            return None
        return patch_path
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("[local_diff] xdelta3 encode error: %s", e)
        try:
            os.unlink(patch_path)
        except OSError:
            pass
        return None


def _xdelta3_decode(baseline_path: str, patch_bytes: bytes) -> Optional[bytes]:
    """Apply xdelta3 patch to baseline, returning reconstructed bytes.
    Returns None on error."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".vcdiff", delete=False) as tf:
            tf.write(patch_bytes)
            patch_path = tf.name
        try:
            result = subprocess.run(
                [XDELTA3_BIN, "-d", "-s", baseline_path, "-c", patch_path],
                capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning(
                    "[local_diff] xdelta3 decode failed: rc=%d stderr=%s",
                    result.returncode,
                    result.stderr[:200].decode("utf-8", "replace"))
                return None
            return result.stdout
        finally:
            try:
                os.unlink(patch_path)
            except OSError:
                pass
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("[local_diff] xdelta3 decode error: %s", e)
        return None


def _sha256_bytes(data: bytes) -> str:
    h = sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: str) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Manifest I/O ─────────────────────────────────────────────────────


def manifest_path(local_dir: str, titan_id: str) -> Path:
    return Path(local_dir) / MANIFEST_NAME_FMT.format(titan_id=titan_id)


def load_manifest(local_dir: str, titan_id: str) -> dict:
    """Load manifest or return fresh empty one. Never raises."""
    p = manifest_path(local_dir, titan_id)
    if not p.exists():
        return {
            "titan_id": titan_id,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "current_baseline_event_id": None,
            "events": [],
        }
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning("[local_diff] manifest load failed (%s): %s — using empty", p, e)
        return {
            "titan_id": titan_id,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "current_baseline_event_id": None,
            "events": [],
        }


def save_manifest(manifest: dict, local_dir: str, titan_id: str) -> None:
    """Atomic write: tmp → fsync → rename. Never raises (logs on failure)."""
    p = manifest_path(local_dir, titan_id)
    tmp = p.with_suffix(".json.tmp")
    try:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        # fsync the tmp file
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        tmp.rename(p)
        # fsync parent dir
        dir_fd = os.open(str(p.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception as e:
        logger.error("[local_diff] manifest save failed: %s", e)
        with suppress_unlink(tmp):
            pass


class suppress_unlink:
    """Context manager that best-effort deletes a path on exit."""
    def __init__(self, p):
        self.p = p
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        try:
            Path(self.p).unlink()
        except OSError:
            pass


# ── Cadence decision ─────────────────────────────────────────────────


def should_create_baseline(manifest: dict, today_iso: str) -> bool:
    """Determine whether today's backup should be a baseline (full) or
    an incremental.

    Baseline triggers:
      1. No prior baseline ever (cold start)
      2. Today is Sunday (BASELINE_WEEKDAY)
      3. Latest baseline is more than 7 days old (e.g., we missed a Sunday)
    """
    baseline_id = manifest.get("current_baseline_event_id")
    if not baseline_id:
        return True  # cold start
    # Find current baseline event
    events = manifest.get("events", [])
    baseline_event = None
    for ev in reversed(events):
        if ev.get("event_id") == baseline_id:
            baseline_event = ev
            break
    if baseline_event is None:
        return True  # manifest drift — treat as cold start

    # Today is Sunday → new baseline
    today_dt = datetime.fromisoformat(today_iso)
    if today_dt.weekday() == BASELINE_WEEKDAY:
        # But don't re-baseline if today's already in the manifest as baseline
        if baseline_event.get("iso_date") == today_iso:
            return False
        return True

    # Baseline older than 7 days → new baseline (failsafe for missed Sundays)
    baseline_ts = baseline_event.get("ts_unix", 0)
    age_days = (time.time() - baseline_ts) / 86400
    if age_days > 7.0:
        return True

    return False


# ── Build incremental tarball ────────────────────────────────────────


def build_incremental_tarball(
    file_specs: list[tuple[str, str]],  # [(source_path, arc_name), ...]
    baseline_event: dict,
    baseline_files_dir: str,  # extracted baseline directory (for diff source)
    output_path: str,
    skip_patterns: tuple = (),
) -> Optional[DiffEvent]:
    """Build incremental tarball — contains xdelta3 patches for files that
    changed since baseline, plus a manifest declaring skipped/removed entries.

    `baseline_files_dir`: directory containing the EXTRACTED baseline's files
    (one-shot extraction handled by caller). Paths inside follow the same
    arcname mapping. Patches are computed against these.

    Returns DiffEvent on success, None on error.
    """
    files_entries: list[FileEntry] = []
    patched = skipped = removed = 0

    # Build baseline arc_name → sha256 map for skip-detection.
    baseline_by_arc = {fe["arc_name"]: fe for fe in baseline_event.get("files", [])}

    try:
        with tarfile.open(output_path, "w:gz", compresslevel=9) as tar:
            tar_added_arcs: set = set()
            for source_path, arc_name in file_specs:
                source = Path(source_path)
                if not source.exists():
                    # Source missing — if baseline had it, mark removed.
                    if arc_name in baseline_by_arc:
                        files_entries.append(FileEntry(
                            arc_name=arc_name,
                            sha256="",
                            size_bytes=0,
                            diff_mode="removed",
                            baseline_sha256=baseline_by_arc[arc_name].get("sha256"),
                        ))
                        removed += 1
                    continue
                # Skip-pattern filter (legacy backup, .bak files, etc.)
                if any(p in source.name for p in skip_patterns):
                    continue

                # Handle directory: recurse into the dir tree. For v1, treat
                # each file inside a directory as its own arc_name suffix.
                if source.is_dir():
                    for sub in source.rglob("*"):
                        if not sub.is_file():
                            continue
                        if any(p in sub.name for p in skip_patterns):
                            continue
                        rel = sub.relative_to(source)
                        sub_arc = f"{arc_name}/{rel}"
                        if sub_arc in tar_added_arcs:
                            continue
                        _add_file_to_incremental(
                            tar, sub_arc, str(sub),
                            baseline_files_dir, baseline_by_arc,
                            files_entries)
                        tar_added_arcs.add(sub_arc)
                        # accounting handled inside helper via files_entries
                else:
                    if arc_name in tar_added_arcs:
                        continue
                    _add_file_to_incremental(
                        tar, arc_name, str(source),
                        baseline_files_dir, baseline_by_arc,
                        files_entries)
                    tar_added_arcs.add(arc_name)

            # Accounting from files_entries (single source of truth)
            for fe in files_entries:
                if fe.diff_mode == "patch":
                    patched += 1
                elif fe.diff_mode == "skipped":
                    skipped += 1
                elif fe.diff_mode == "removed":
                    removed += 1

            # Write incremental manifest as a JSON member inside the tarball.
            inc_manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "baseline_event_id": baseline_event["event_id"],
                "files": [asdict(fe) for fe in files_entries],
                "patched_count": patched,
                "skipped_count": skipped,
                "removed_count": removed,
            }
            inc_bytes = json.dumps(inc_manifest, indent=2, sort_keys=True).encode("utf-8")
            info = tarfile.TarInfo(name="__local_diff_manifest.json")
            info.size = len(inc_bytes)
            info.mtime = int(time.time())
            import io as _io
            tar.addfile(info, _io.BytesIO(inc_bytes))

        # Compute final tarball hash + size
        tar_size = os.path.getsize(output_path)
        tar_hash = _sha256_file(output_path)
        ev_uuid = str(uuid.uuid4())
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return DiffEvent(
            event_id=ev_uuid,
            ts_unix=time.time(),
            iso_date=today,
            type="incremental",
            tarball_name=os.path.basename(output_path),
            tarball_sha256=tar_hash,
            tarball_size_bytes=tar_size,
            files=files_entries,
            baseline_event_id=baseline_event["event_id"],
            patched_count=patched,
            skipped_count=skipped,
            removed_count=removed,
        )

    except Exception as e:
        logger.error("[local_diff] build_incremental_tarball failed: %s", e, exc_info=True)
        with suppress_unlink(output_path):
            pass
        return None


def _add_file_to_incremental(
    tar: tarfile.TarFile,
    arc_name: str,
    current_path: str,
    baseline_files_dir: str,
    baseline_by_arc: dict,
    files_entries: list,
) -> None:
    """Helper: compute diff for one file and add patch/skip/full to incremental tarball."""
    import io as _io
    try:
        current_hash = _sha256_file(current_path)
        current_size = os.path.getsize(current_path)
    except OSError as e:
        logger.warning("[local_diff] stat failed for %s: %s", current_path, e)
        return

    baseline_entry = baseline_by_arc.get(arc_name)
    if baseline_entry and baseline_entry.get("sha256") == current_hash:
        # Skip-if-unchanged: file identical to baseline → record skip only.
        files_entries.append(FileEntry(
            arc_name=arc_name,
            sha256=current_hash,
            size_bytes=current_size,
            diff_mode="skipped",
            baseline_sha256=current_hash,
        ))
        return

    if baseline_entry:
        # File changed — produce xdelta3 patch.
        baseline_path = os.path.join(baseline_files_dir, arc_name)
        if os.path.exists(baseline_path):
            patch_path = _xdelta3_encode(baseline_path, current_path)
            if patch_path is not None:
                patch_size = os.path.getsize(patch_path)
                if patch_size < current_size:
                    # Patch smaller than the file — ship it, STREAMED from disk
                    # via addfile(fileobj) (the patch is never read into RAM).
                    # TarInfo (name/size/mtime) is identical to the prior path →
                    # byte-identical tar member; only the source is a file now.
                    info = tarfile.TarInfo(name=arc_name + ".vcdiff")
                    info.size = patch_size
                    info.mtime = int(time.time())
                    with open(patch_path, "rb") as _pf:
                        tar.addfile(info, _pf)
                    files_entries.append(FileEntry(
                        arc_name=arc_name,
                        sha256=current_hash,
                        size_bytes=current_size,
                        diff_mode="patch",
                        baseline_sha256=baseline_entry.get("sha256"),
                    ))
                    try:
                        os.unlink(patch_path)
                    except OSError:
                        pass
                    return
                # Patch not smaller than full-ship → discard it + full-ship below.
                try:
                    os.unlink(patch_path)
                except OSError:
                    pass
        # Patch failed or larger than full — fall through to full-ship.

    # New file (not in baseline) or patch produced no gain → ship full.
    tar.add(current_path, arcname=arc_name)
    files_entries.append(FileEntry(
        arc_name=arc_name,
        sha256=current_hash,
        size_bytes=current_size,
        diff_mode="full",
    ))


# ── Baseline-active directory ─────────────────────────────────────────
#
# After each baseline build, extract its contents to `data/backups/baseline_active/`
# so subsequent incremental builds can diff against the baseline files at known
# arc_name paths. Adds ~30MB disk for ~10× speedup on each incremental.


def baseline_active_dir(local_dir: str) -> Path:
    return Path(local_dir) / "baseline_active"


def refresh_baseline_active(
    baseline_tarball_path: str,
    local_dir: str,
) -> bool:
    """Extract baseline tarball to baseline_active/ (replacing any prior contents).
    Returns True on success. Atomic: builds new dir alongside, then swap-rename."""
    active = baseline_active_dir(local_dir)
    tmp_active = active.with_suffix(".extracting")
    bak_active = active.with_suffix(".old")
    try:
        # Clean tmp
        if tmp_active.exists():
            import shutil
            shutil.rmtree(tmp_active)
        tmp_active.mkdir(parents=True)
        with tarfile.open(baseline_tarball_path, "r:gz") as tar:
            tar.extractall(path=tmp_active)
        # Swap: old → bak, tmp → active, then remove bak
        if active.exists():
            if bak_active.exists():
                import shutil
                shutil.rmtree(bak_active)
            active.rename(bak_active)
        tmp_active.rename(active)
        if bak_active.exists():
            import shutil
            shutil.rmtree(bak_active)
        return True
    except Exception as e:
        logger.error("[local_diff] refresh_baseline_active failed: %s", e, exc_info=True)
        # Best-effort restore old dir
        if bak_active.exists() and not active.exists():
            try:
                bak_active.rename(active)
            except OSError:
                pass
        return False


# ── Build baseline tarball ────────────────────────────────────────────


def build_baseline_tarball(
    file_specs: list[tuple[str, str]],
    output_path: str,
    skip_patterns: tuple = (),
) -> Optional[DiffEvent]:
    """Build a baseline (full snapshot) tarball. Same shape as the existing
    create_personality_archive, but emits a DiffEvent for the manifest."""
    files_entries: list[FileEntry] = []
    try:
        with tarfile.open(output_path, "w:gz", compresslevel=9) as tar:
            tar_added_arcs: set = set()
            for source_path, arc_name in file_specs:
                source = Path(source_path)
                if not source.exists():
                    continue
                if source.is_dir():
                    for sub in source.rglob("*"):
                        if not sub.is_file():
                            continue
                        if any(p in sub.name for p in skip_patterns):
                            continue
                        rel = sub.relative_to(source)
                        sub_arc = f"{arc_name}/{rel}"
                        if sub_arc in tar_added_arcs:
                            continue
                        tar.add(str(sub), arcname=sub_arc)
                        files_entries.append(FileEntry(
                            arc_name=sub_arc,
                            sha256=_sha256_file(str(sub)),
                            size_bytes=sub.stat().st_size,
                            diff_mode="full",
                        ))
                        tar_added_arcs.add(sub_arc)
                else:
                    if any(p in source.name for p in skip_patterns):
                        continue
                    if arc_name in tar_added_arcs:
                        continue
                    tar.add(source_path, arcname=arc_name)
                    files_entries.append(FileEntry(
                        arc_name=arc_name,
                        sha256=_sha256_file(source_path),
                        size_bytes=source.stat().st_size,
                        diff_mode="full",
                    ))
                    tar_added_arcs.add(arc_name)

        tar_size = os.path.getsize(output_path)
        tar_hash = _sha256_file(output_path)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return DiffEvent(
            event_id=str(uuid.uuid4()),
            ts_unix=time.time(),
            iso_date=today,
            type="baseline",
            tarball_name=os.path.basename(output_path),
            tarball_sha256=tar_hash,
            tarball_size_bytes=tar_size,
            files=files_entries,
        )
    except Exception as e:
        logger.error("[local_diff] build_baseline_tarball failed: %s", e, exc_info=True)
        with suppress_unlink(output_path):
            pass
        return None


# ── Restore from incremental chain ────────────────────────────────────


def restore_from_local_chain(
    manifest: dict,
    local_dir: str,
    target_dir: str,
    arc_to_source: dict,  # arc_name → real source path mapping (PERSONALITY_PATHS reverse)
) -> dict:
    """Walk the manifest from latest event back to baseline, then forward-apply
    incrementals to reconstruct full state at `target_dir`.

    Returns dict with: restored_files, errors, baseline_event_id.
    """
    result: dict = {"restored_files": 0, "errors": [], "baseline_event_id": None}
    events = manifest.get("events", [])
    if not events:
        result["errors"].append("manifest empty")
        return result

    # Walk backward to find the most recent baseline.
    latest_event = events[-1]
    if latest_event["type"] == "baseline":
        chain: list[dict] = [latest_event]
    else:
        baseline_id = latest_event.get("baseline_event_id")
        baseline = next((ev for ev in events if ev["event_id"] == baseline_id), None)
        if baseline is None:
            result["errors"].append(f"baseline {baseline_id} not in manifest")
            return result
        # Chain = baseline followed by all incrementals after it up to + including latest
        chain = [baseline]
        baseline_idx = events.index(baseline)
        for ev in events[baseline_idx + 1:]:
            if ev.get("baseline_event_id") == baseline_id:
                chain.append(ev)

    result["baseline_event_id"] = chain[0]["event_id"]
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    local = Path(local_dir)

    # Phase 1: extract baseline.
    baseline = chain[0]
    baseline_tarball = local / baseline["tarball_name"]
    if not baseline_tarball.exists():
        result["errors"].append(f"baseline tarball missing: {baseline_tarball}")
        return result
    try:
        with tarfile.open(baseline_tarball, "r:gz") as tar:
            tar.extractall(path=target)
        result["restored_files"] += len(baseline.get("files", []))
    except Exception as e:
        result["errors"].append(f"extract baseline: {e}")
        return result

    # Phase 2: apply each incremental in order.
    for inc in chain[1:]:
        inc_tarball = local / inc["tarball_name"]
        if not inc_tarball.exists():
            result["errors"].append(f"incremental tarball missing: {inc_tarball}")
            continue
        try:
            with tarfile.open(inc_tarball, "r:gz") as tar:
                # Read inner manifest
                try:
                    inner_member = tar.getmember("__local_diff_manifest.json")
                    inner_bytes = tar.extractfile(inner_member).read()
                    inner = json.loads(inner_bytes)
                except KeyError:
                    result["errors"].append(f"{inc_tarball.name}: missing __local_diff_manifest.json")
                    continue

                for fe in inner["files"]:
                    arc = fe["arc_name"]
                    mode = fe["diff_mode"]
                    target_file = target / arc
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    if mode == "skipped":
                        # No change — file already in target from baseline
                        continue
                    elif mode == "removed":
                        if target_file.exists():
                            try:
                                target_file.unlink()
                            except OSError as e:
                                result["errors"].append(f"{arc}: unlink {e}")
                        continue
                    elif mode == "full":
                        try:
                            member = tar.getmember(arc)
                            with tar.extractfile(member) as src:
                                target_file.write_bytes(src.read())
                            result["restored_files"] += 1
                        except KeyError:
                            result["errors"].append(f"{arc}: full file missing from tarball")
                    elif mode == "patch":
                        try:
                            patch_member = tar.getmember(arc + ".vcdiff")
                            patch_bytes = tar.extractfile(patch_member).read()
                            reconstructed = _xdelta3_decode(str(target_file), patch_bytes)
                            if reconstructed is None:
                                result["errors"].append(f"{arc}: xdelta3 decode failed")
                                continue
                            # Verify hash matches manifest expectation
                            if fe.get("sha256") and _sha256_bytes(reconstructed) != fe["sha256"]:
                                result["errors"].append(f"{arc}: post-patch hash mismatch")
                                continue
                            target_file.write_bytes(reconstructed)
                            result["restored_files"] += 1
                        except KeyError:
                            result["errors"].append(f"{arc}: patch missing")
        except Exception as e:
            result["errors"].append(f"{inc_tarball.name}: {e}")

    return result
