"""Phase 5 chunk 5H — closure regression gates.

rFP_phase_c_enhancements §3B.3 collects the acceptance gates spelled out
across 5A–5F. The other test files cover their own slice (5D in
test_arch_map_backup_restore_test.py, 5E in test_irys_persistent_client.py,
5F in test_backup_event_tarball.py). This file holds the cross-chunk
property gates that don't fit cleanly under any one module's suite:

  - 5A streaming determinism: pack_event_tarball produces byte-identical
    output (and tarball_sha256) whether the encoder supplies patch_path
    (streaming) or patch_bytes (legacy, in-memory).
  - 5G regression sentinel: backup_worker ModuleSpec declares
    rss_limit_mb=500 (was 1200 before 5A; bumping it again would mask
    the underlying memory amplification bug per
    feedback_no_rss_band_aid_understand_root_cause).
  - 5C bootstrap CLI smoke: scripts/backup_force_baseline.py is
    importable, exposes a dry-run path, and refuses to commit without
    explicit Maker confirmation.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from titan_hcl.logic.backup_event_tarball import (
    EVENT_TARBALL_EXT,
    FileDiffSpec,
    pack_event_tarball,
)


# ── 5A: streaming vs legacy patch-bytes produces identical output ───────


def _legacy_full_diff(content: bytes) -> dict:
    return {
        "diff_mode": "full",
        "patch_bytes": content,
        "merkle_root": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "encoder": "full_ship",
    }


def _streaming_full_diff(content: bytes, patch_path: str) -> dict:
    Path(patch_path).write_bytes(content)
    return {
        "diff_mode": "full",
        "patch_path": patch_path,
        "patch_size_bytes": len(content),
        "patch_owned": False,  # caller owns; pack must not unlink
        "merkle_root": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "encoder": "full_ship",
    }


def test_5a_streaming_and_legacy_paths_produce_identical_tarball(tmp_path):
    """The two encoder branches must round-trip to the same on-disk bytes
    so 5A's RSS savings don't accidentally diverge the Merkle anchor."""
    payload = b"abc" * 4096  # 12 KiB — large enough to span multiple blocks
    legacy_dd = _legacy_full_diff(payload)
    stream_patch = tmp_path / "patch_a.bin"
    stream_dd = _streaming_full_diff(payload, str(stream_patch))

    legacy_out = tmp_path / f"legacy{EVENT_TARBALL_EXT}"
    stream_out = tmp_path / f"stream{EVENT_TARBALL_EXT}"

    info_legacy = pack_event_tarball(
        event_id="evt_eq", event_type="baseline", component="personality",
        file_specs=[FileDiffSpec("inner_memory.db", legacy_dd)],
        output_path=str(legacy_out),
        ts_unix=1779000000.0,  # pin so timestamps don't diverge
    )
    info_stream = pack_event_tarball(
        event_id="evt_eq", event_type="baseline", component="personality",
        file_specs=[FileDiffSpec("inner_memory.db", stream_dd)],
        output_path=str(stream_out),
        ts_unix=1779000000.0,
    )

    # Same recorded sha256 …
    assert info_legacy["tarball_sha256"] == info_stream["tarball_sha256"]
    # … and same on-disk file size (the zstd frame is deterministic for
    # identical bytes + same level).
    assert info_legacy["size_bytes"] == info_stream["size_bytes"]
    # And byte-identical on disk.
    assert legacy_out.read_bytes() == stream_out.read_bytes()
    # patch_owned=False → input file survives pack
    assert stream_patch.exists()


def test_5a_streaming_unlinks_when_patch_owned(tmp_path):
    """patch_owned=True signals encoder ownership; pack_event_tarball MUST
    unlink the temp on success so vcdiff / tail tempfiles don't leak."""
    payload = b"x" * 1024
    p = tmp_path / "owned.bin"
    p.write_bytes(payload)
    dd = {
        "diff_mode": "full",
        "patch_path": str(p),
        "patch_size_bytes": len(payload),
        "patch_owned": True,
        "merkle_root": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "encoder": "full_ship",
    }
    out = tmp_path / f"evt{EVENT_TARBALL_EXT}"
    pack_event_tarball(
        event_id="evt_owned", event_type="baseline", component="personality",
        file_specs=[FileDiffSpec("inner_memory.db", dd)],
        output_path=str(out),
    )
    assert out.exists()
    assert not p.exists(), "patch_owned=True path must be unlinked after pack"


# ── 5G: backup ModuleSpec rss_limit_mb regression sentinel ──────────────


def test_5g_backup_module_spec_rss_limit_is_500():
    """Phase 5 chunk 5G fully reverted 1200 → 500. Anyone bumping it
    again without commensurate streaming-encoder work re-introduces the
    1.7 GB-RSS amplification bug. This test must fail loudly on any
    such bump until the underlying memory profile is changed.
    """
    plugin_src = (Path(__file__).resolve().parents[1]
                  / "titan_hcl" / "core" / "plugin.py")
    text = plugin_src.read_text()
    # Find the backup ModuleSpec block and confirm rss_limit_mb=500
    # appears in the same registration call.
    needle_start = text.find('name="backup"')
    assert needle_start != -1, "backup ModuleSpec missing"
    block = text[needle_start:needle_start + 2000]
    assert "rss_limit_mb=500" in block, (
        "Phase 5 5G regression: backup ModuleSpec rss_limit_mb must be 500.\n"
        "Found block:\n" + block[:500]
    )


# ── 5C: backup_force_baseline.py CLI smoke ──────────────────────────────


def test_5c_bootstrap_cli_dry_run_help_succeeds():
    """The bootstrap CLI must be invokable with --help so cron/Maker
    scripts can verify the binary is in a sane state without running
    a real cascade."""
    script = (Path(__file__).resolve().parents[1]
              / "scripts" / "backup_force_baseline.py")
    assert script.exists()
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--titan-id" in proc.stdout
    assert "--dry-run" in proc.stdout
    assert "--commit" in proc.stdout


def test_5c_bootstrap_cli_module_importable():
    """scripts/ is not on sys.path by default but the file must remain a
    valid Python module so future test scaffolding can drive it."""
    script_path = (Path(__file__).resolve().parents[1]
                   / "scripts" / "backup_force_baseline.py")
    spec = importlib.util.spec_from_file_location(
        "backup_force_baseline", script_path,
    )
    module = importlib.util.module_from_spec(spec)
    # Importing must NOT trigger any wallet/Solana network calls. The
    # script gates all of those behind argparse + Maker confirmation.
    spec.loader.exec_module(module)
    assert hasattr(module, "main"), "backup_force_baseline.main missing"
