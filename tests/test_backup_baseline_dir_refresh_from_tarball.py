"""§24 #3 (2026-06-01): the baseline working dir is refreshed from the FULL-mode
files INSIDE the just-shipped tarball — NOT a re-copy of live source.

Why it matters: the staged path packs off-loop (possibly ~20 min before the
ship), and live sqlite re-reads are torn. The baseline-dir bytes MUST equal the
bytes uploaded to Arweave, else a future xdelta3 incremental reconstructs wrong
at restore. Extracting the tarball's full-mode payloads guarantees the diff
base == shipped bytes.
"""
from __future__ import annotations

import os

from titan_hcl.logic.backup import RebirthBackup
from titan_hcl.logic.diff_encoders import full_ship
from titan_hcl.logic.backup_event_tarball import FileDiffSpec, pack_event_tarball


class _FakeBackup:
    """Minimal carrier so we can exercise the unbound helper without a full
    RebirthBackup init (no DB/Arweave)."""
    def __init__(self, base_dir):
        self._base = base_dir

    def _baseline_working_dir(self):
        return self._base


def _build_baseline_tarball(tmp_path, files: dict[str, bytes]) -> str:
    """full_ship-encode each file (diff_mode=full) and pack a baseline tarball."""
    specs = []
    for arc, content in files.items():
        src = tmp_path / arc
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(content)
        dd = full_ship.encode_diff(str(src))
        assert dd["diff_mode"] == "full"
        specs.append(FileDiffSpec(arc_name=arc, diff_dict=dd))
    out = str(tmp_path / "event_baseline.tar.zst")
    pack_event_tarball(
        event_id="evt-base-1", event_type="baseline", component="soul",
        file_specs=specs, output_path=out)
    return out


def test_refresh_writes_shipped_bytes_not_source(tmp_path):
    files = {
        "consciousness.db": b"\x00\x01CONSCIOUSNESS-BASELINE-BYTES" * 1000,
        "sub/nested.json": b'{"k": "v"}',
    }
    tarball = _build_baseline_tarball(tmp_path / "src", files)

    base = tmp_path / "baseline_dir"
    fake = _FakeBackup(str(base))
    written = RebirthBackup._refresh_baseline_dir_from_tarballs(fake, [tarball])

    assert written == 2
    # base_dir now holds EXACTLY the shipped bytes for each arc_name.
    assert (base / "consciousness.db").read_bytes() == files["consciousness.db"]
    assert (base / "sub" / "nested.json").read_bytes() == files["sub/nested.json"]


def test_refresh_skips_missing_tarball_gracefully(tmp_path):
    base = tmp_path / "baseline_dir"
    fake = _FakeBackup(str(base))
    # nonexistent path → skipped, no raise, 0 written
    assert RebirthBackup._refresh_baseline_dir_from_tarballs(
        fake, ["/tmp/does_not_exist_zzz.tar.zst"]) == 0


def test_refresh_overwrites_stale_baseline_file(tmp_path):
    """A pre-existing (stale) baseline-dir file is replaced by the shipped bytes."""
    base = tmp_path / "baseline_dir"
    base.mkdir()
    (base / "consciousness.db").write_bytes(b"STALE-OLD-BASELINE")
    files = {"consciousness.db": b"FRESH-SHIPPED-BASELINE" * 500}
    tarball = _build_baseline_tarball(tmp_path / "src", files)
    fake = _FakeBackup(str(base))
    RebirthBackup._refresh_baseline_dir_from_tarballs(fake, [tarball])
    assert (base / "consciousness.db").read_bytes() == files["consciousness.db"]
