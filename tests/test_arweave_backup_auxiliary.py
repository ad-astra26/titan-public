"""Tests for AUXILIARY_BACKUP_PATHS wiring in TimeChainBackup.

Verifies that maker_proposals.db (and any future auxiliary databases)
get included in the snapshot tarball at create time and relocated to
their disk paths during restore. Critical for governance state
durability — without this wiring, R8 signatures and Maker dialogue
history would not survive infrastructure failure.
"""
import io
import os
import tarfile
import tempfile

import pytest


def test_auxiliary_backup_paths_constant_includes_maker_proposals():
    """Sanity check: the constant has the expected entry."""
    from titan_plugin.logic.timechain_backup import AUXILIARY_BACKUP_PATHS
    assert "auxiliary/maker_proposals.db" in AUXILIARY_BACKUP_PATHS
    assert AUXILIARY_BACKUP_PATHS["auxiliary/maker_proposals.db"] == \
        "data/maker_proposals.db"


def test_auxiliary_files_included_in_tarball(tmp_path, monkeypatch):
    """Verify create_snapshot_tarball() includes auxiliary files when present."""
    from titan_plugin.logic.timechain_backup import (
        AUXILIARY_BACKUP_PATHS, TimeChainBackup,
    )
    # Build a real ProposalStore at a tmp path so the file exists
    from titan_plugin.maker import ProposalStore, ProposalType
    fake_aux_path = tmp_path / "fake_maker_proposals.db"
    store = ProposalStore(db_path=str(fake_aux_path))
    store.create(
        proposal_type=ProposalType.CONTRACT_BUNDLE,
        title="Test bundle",
        description="A test bundle for backup wiring",
        payload={"x": 1}, requires_signature=True)
    assert fake_aux_path.exists()

    # Monkey-patch AUXILIARY_BACKUP_PATHS to point to our tmp file
    monkeypatch.setattr(
        "titan_plugin.logic.timechain_backup.AUXILIARY_BACKUP_PATHS",
        {"auxiliary/maker_proposals.db": str(fake_aux_path)},
    )

    # Build a minimal fake TimeChain dir so create_snapshot_tarball doesn't bail
    fake_tc_dir = tmp_path / "timechain"
    fake_tc_dir.mkdir()
    # Skip the "no genesis" early-return by stubbing TimeChain
    from unittest.mock import patch, MagicMock
    fake_tc = MagicMock()
    fake_tc.has_genesis = True
    fake_tc.genesis_hash = b"\x00" * 32
    fake_tc.compute_merkle_root = lambda: b"\x01" * 32
    fake_tc.total_blocks = 0
    with patch("titan_plugin.logic.timechain.TimeChain", return_value=fake_tc):
        backup = TimeChainBackup(data_dir=str(fake_tc_dir), titan_id="T1")
        tarball, metadata = backup.create_snapshot_tarball()

    assert metadata["version"] == 3
    assert "auxiliary/maker_proposals.db" in metadata["auxiliary_paths"]

    # Decompress + inspect tarball members
    if metadata["compression"].startswith("zstd"):
        import zstandard
        raw = zstandard.ZstdDecompressor().decompress(tarball, max_output_size=50_000_000)
        tar_buf = io.BytesIO(raw)
        with tarfile.open(fileobj=tar_buf, mode="r") as tar:
            names = [m.name for m in tar.getmembers()]
    else:
        import gzip
        raw = gzip.decompress(tarball)
        tar_buf = io.BytesIO(raw)
        with tarfile.open(fileobj=tar_buf, mode="r") as tar:
            names = [m.name for m in tar.getmembers()]

    assert "auxiliary/maker_proposals.db" in names, \
        f"Expected auxiliary/maker_proposals.db in tarball, got: {names}"


def test_auxiliary_files_relocated_during_restore(tmp_path, monkeypatch):
    """Verify _restore_from_tarball moves auxiliary/* to disk paths."""
    from titan_plugin.logic.timechain_backup import TimeChainBackup
    # Stub Path for the relocation target — keep within tmp_path
    aux_target = tmp_path / "data_dir" / "maker_proposals.db"
    monkeypatch.setattr(
        "titan_plugin.logic.timechain_backup.AUXILIARY_BACKUP_PATHS",
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
    from titan_plugin.logic.timechain_backup import TimeChainBackup
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
