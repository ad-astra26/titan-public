"""Tests for AUXILIARY_BACKUP_PATHS relocation in TimeChainBackup (restore side).

Verifies that maker_proposals.db (and any future auxiliary databases) are
relocated to their disk paths during restore. Critical for governance state
durability — without this, R8 signatures and Maker dialogue history would not
survive infrastructure failure. (The create-time backup of these files now
rides the unified `timechain` tier — data/timechain/auxiliary/maker_proposals.db
is in RebirthBackup.TIMECHAIN_PATHS — not the retired create_snapshot_tarball.)
"""
import io
import os
import tarfile
import tempfile

import pytest


def test_auxiliary_backup_paths_constant_includes_maker_proposals():
    """Sanity check: the constant has the expected entry."""
    from titan_hcl.logic.timechain_backup import AUXILIARY_BACKUP_PATHS
    assert "auxiliary/maker_proposals.db" in AUXILIARY_BACKUP_PATHS
    assert AUXILIARY_BACKUP_PATHS["auxiliary/maker_proposals.db"] == \
        "data/maker_proposals.db"


def test_auxiliary_files_relocated_during_restore(tmp_path, monkeypatch):
    """Verify _restore_from_tarball moves auxiliary/* to disk paths."""
    from titan_hcl.logic.timechain_backup import TimeChainBackup
    # Stub Path for the relocation target — keep within tmp_path
    aux_target = tmp_path / "data_dir" / "maker_proposals.db"
    monkeypatch.setattr(
        "titan_hcl.logic.timechain_backup.AUXILIARY_BACKUP_PATHS",
        {"auxiliary/maker_proposals.db": str(aux_target)},
    )

    # Build a synthetic tarball that has only the auxiliary file
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        # Create a fake aux file
        fake_aux = tmp_path / "_src.db"
        fake_aux.write_bytes(b"fake-sqlite-content")
        tar.add(str(fake_aux), arcname="auxiliary/maker_proposals.db")
    raw_tar = buf.getvalue()

    # Compress with gzip (simpler than zstd for the test)
    import gzip
    tarball_bytes = gzip.compress(raw_tar)

    # Run _restore_from_tarball
    backup = TimeChainBackup(data_dir=str(tmp_path / "tc_target"), titan_id="T1")
    # Patch _verify_restored_chain to skip integrity check (no real chain)
    backup._verify_restored_chain = lambda target: True
    target_dir = tmp_path / "tc_target"
    ok = backup._restore_from_tarball(tarball_bytes, target_dir)
    assert ok is True

    # The auxiliary file should now be at its disk path, not at <target>/auxiliary/
    assert aux_target.exists(), \
        f"Expected {aux_target} after restore — relocation failed"
    assert aux_target.read_bytes() == b"fake-sqlite-content"
    # The auxiliary/ subdir should be cleaned up
    assert not (target_dir / "auxiliary" / "maker_proposals.db").exists()


def test_backward_compat_v2_tarball_has_no_auxiliary(tmp_path):
    """v2 tarballs (no auxiliary section) should still restore cleanly."""
    from titan_hcl.logic.timechain_backup import TimeChainBackup
    # Build a v2-style tarball (only chain data, no auxiliary/)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        fake_chain = tmp_path / "chain_main.bin"
        fake_chain.write_bytes(b"fake-chain-blocks")
        tar.add(str(fake_chain), arcname="chain_main.bin")
    import gzip
    tarball_bytes = gzip.compress(buf.getvalue())

    backup = TimeChainBackup(data_dir=str(tmp_path / "tc_target"), titan_id="T1")
    backup._verify_restored_chain = lambda target: True
    target_dir = tmp_path / "tc_target"
    ok = backup._restore_from_tarball(tarball_bytes, target_dir)
    assert ok is True
    assert (target_dir / "chain_main.bin").exists()
