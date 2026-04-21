"""Tests for rFP_backup_worker Phase 2 extension (2026-04-20) —
TimeChainBackup.snapshot_to_arweave uses the shared BackupCascade.

Covers: BackupCascade class (class-level), run_bytes variant, and
TimeChainBackup integration with cascade passthrough.

Does NOT test live Arweave upload (S5) — that's covered by the live smoke test.
"""
import asyncio
import hashlib
import io
import os
import tarfile
import time

import pytest

from titan_plugin.logic.backup_cascade import BackupCascade


def _make_tarball_bytes(payload: bytes = b"hello world") -> tuple[bytes, str]:
    """Build a minimal valid tar.gz, return (bytes, sha256_hex)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="test.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    data = buf.getvalue()
    return data, hashlib.sha256(data).hexdigest()


def _make_cascade(tmp_path, mode=None, arweave_store=object()):
    cfg = {
        "backup": {"mode": mode, "local_rolling_days": 30} if mode else {},
        "network": {"wallet_keypair_path": "data/titan_identity_keypair.json"},
    }
    return BackupCascade(full_config=cfg, arweave_store=arweave_store,
                         local_dir=str(tmp_path))


# ── BackupCascade class tests ────────────────────────────────────────────

def test_cascade_class_validate_ok(tmp_path):
    c = _make_cascade(tmp_path)
    data, _ = _make_tarball_bytes()
    p = tmp_path / "ok.tar.gz"
    p.write_bytes(data)
    assert c.validate_tarball(str(p)) is True


def test_cascade_class_validate_corrupt(tmp_path):
    c = _make_cascade(tmp_path)
    p = tmp_path / "bad.tar.gz"
    p.write_bytes(b"not a tarball")
    assert c.validate_tarball(str(p)) is False


def test_cascade_save_local_bytes_for_timechain(tmp_path):
    c = _make_cascade(tmp_path)
    data, h = _make_tarball_bytes()
    local = c.save_local_bytes(data, "timechain", h, ext="tar.zst")
    assert local is not None
    assert os.path.exists(local)
    assert h[:12] in local
    assert local.endswith(".tar.zst")


def test_cascade_is_local_only_explicit(tmp_path):
    c = _make_cascade(tmp_path, mode="local_only")
    assert c.is_local_only() is True


def test_cascade_is_local_only_no_store_fallback(tmp_path):
    c = _make_cascade(tmp_path, mode=None, arweave_store=None)
    assert c.is_local_only() is True


def test_cascade_is_local_only_store_present(tmp_path):
    c = _make_cascade(tmp_path, mode=None, arweave_store=object())
    assert c.is_local_only() is False


def test_cascade_cleanup_timechain(tmp_path):
    c = _make_cascade(tmp_path)
    old = tmp_path / "timechain_20251201_aaaaaaaaaaaa.tar.zst"
    new = tmp_path / "timechain_20260420_bbbbbbbbbbbb.tar.zst"
    old.write_bytes(b"o"); new.write_bytes(b"n")
    old_ts = time.time() - 60 * 86400
    os.utime(old, (old_ts, old_ts))
    os.utime(new, (time.time() - 86400, time.time() - 86400))
    n = c.cleanup_local("timechain", retention_days=30)
    assert n == 1
    assert not old.exists()
    assert new.exists()


# ── run_bytes orchestrator (TimeChain's path) ────────────────────────────

def test_run_bytes_local_only_persists_but_skips_upload(tmp_path):
    c = _make_cascade(tmp_path, mode="local_only")
    data, h = _make_tarball_bytes()

    async def _upload_fn(_bytes):
        pytest.fail("upload_fn must not be called in local_only mode")

    result = asyncio.run(c.run_bytes(
        tarball_bytes=data, archive_hash=h, backup_type="timechain",
        upload_fn=_upload_fn, retention_days=30, ext="tar.gz"))
    assert result is not None
    assert result["mode"] == "local_only"
    assert os.path.exists(result["local_path"])
    assert "arweave_tx" not in result


def test_run_bytes_s2_fail_on_corrupt(tmp_path):
    c = _make_cascade(tmp_path, mode="local_only")
    bad_bytes = b"not a tarball"

    async def _upload_fn(_bytes):
        pytest.fail("should not reach S5")

    # _make_tarball_bytes returns real hash, but we'll fake one for this test
    fake_hash = hashlib.sha256(bad_bytes).hexdigest()
    result = asyncio.run(c.run_bytes(
        tarball_bytes=bad_bytes, archive_hash=fake_hash,
        backup_type="timechain", upload_fn=_upload_fn, ext="tar.gz"))
    assert result is not None
    assert result.get("cascade_fail") == "S2_validate"


def test_run_bytes_s5_success_devnet_skips_verify(tmp_path):
    c = _make_cascade(tmp_path, mode="mainnet_arweave")
    # Monkeypatch balance check for test isolation
    c.check_irys_balance = lambda size: (True, 1.0, "fake")
    data, h = _make_tarball_bytes()

    async def _upload_fn(_bytes):
        return {"arweave_tx": "devnet_fake_tx", "size_mb": 0.001}

    result = asyncio.run(c.run_bytes(
        tarball_bytes=data, archive_hash=h, backup_type="timechain",
        upload_fn=_upload_fn, ext="tar.gz"))
    assert result is not None
    assert result["arweave_tx"] == "devnet_fake_tx"
    assert result["verified"] == "skipped_devnet"
    assert os.path.exists(result["local_path"])
    # Confirms S3 persisted even though S6 is devnet-skipped


def test_run_bytes_low_balance_falls_back_to_local(tmp_path):
    c = _make_cascade(tmp_path, mode="mainnet_arweave")
    # Force S4 FAIL
    c.check_irys_balance = lambda size: (False, 0.0, "need=0.01 have=0.0")
    data, h = _make_tarball_bytes()

    async def _upload_fn(_bytes):
        pytest.fail("upload_fn must not run when balance insufficient")

    result = asyncio.run(c.run_bytes(
        tarball_bytes=data, archive_hash=h, backup_type="timechain",
        upload_fn=_upload_fn, ext="tar.gz"))
    assert result is not None
    assert result["mode"] == "local_fallback_low_balance"
    assert os.path.exists(result["local_path"])
    assert "arweave_tx" not in result


def test_run_bytes_s5_none_surfaces(tmp_path):
    c = _make_cascade(tmp_path, mode="mainnet_arweave")
    c.check_irys_balance = lambda size: (True, 1.0, "fake")
    data, h = _make_tarball_bytes()

    async def _upload_fn(_bytes):
        return None  # simulated upload failure

    result = asyncio.run(c.run_bytes(
        tarball_bytes=data, archive_hash=h, backup_type="timechain",
        upload_fn=_upload_fn, ext="tar.gz"))
    assert result is not None
    assert result.get("cascade_fail") == "S5_upload_none"
    # S3 still ran — local path exists
    assert os.path.exists(result["local_path"])


def test_run_bytes_diff_audit_flags_size_delta(tmp_path):
    c = _make_cascade(tmp_path, mode="local_only")
    data, h = _make_tarball_bytes(payload=b"x" * 10)

    def fake_latest(backup_type):
        return {"size_mb": 100.0}  # big drop from 100MB → <1MB

    async def _upload_fn(_bytes):
        return None

    result = asyncio.run(c.run_bytes(
        tarball_bytes=data, archive_hash=h, backup_type="timechain",
        upload_fn=_upload_fn, get_latest_record_fn=fake_latest, ext="tar.gz"))
    assert result is not None
    assert "diff_alert" in result
    assert result["diff_alert"]["severity"] == "ERROR"  # >50% drop → ERROR


# ── TimeChainBackup integration ──────────────────────────────────────────

def test_timechain_snapshot_to_arweave_uses_cascade(tmp_path, monkeypatch):
    """End-to-end: TimeChainBackup calls cascade under the hood.

    Covers the wiring: TimeChainBackup constructs BackupCascade, passes
    full_config through, and the cascade handles S2-S10 around the upload.
    """
    from titan_plugin.logic.timechain_backup import TimeChainBackup

    class _FakeStore:
        async def upload_file_bytes(self, data, tags, ct):
            return "test_tx_123456789abcdef"

    # We won't call create_snapshot_tarball (requires real TimeChain data);
    # instead we patch it to return fake tarball bytes.
    tcb = TimeChainBackup(data_dir="data/timechain", titan_id="T1",
                          arweave_store=_FakeStore())

    # TimeChain tarballs are zstd-compressed — build real zstd bytes so S2 validate passes
    import zstandard
    raw_tar_buf = io.BytesIO()
    with tarfile.open(fileobj=raw_tar_buf, mode="w") as tf:
        info = tarfile.TarInfo(name="chain_0.bin")
        info.size = 8
        tf.addfile(info, io.BytesIO(b"gen_hash"))
    cctx = zstandard.ZstdCompressor(level=3)
    fake_tarball = cctx.compress(raw_tar_buf.getvalue())
    fake_hash = hashlib.sha256(fake_tarball).hexdigest()
    fake_metadata = {
        "version": 3,
        "titan_id": "T1",
        "genesis_hash": "fake_genesis",
        "merkle_root": "fake_merkle",
        "total_blocks": 100,
        "tarball_sha256": fake_hash,
        "tarball_size_bytes": len(fake_tarball),
        "raw_size_bytes": len(fake_tarball),
        "compression": "zstd-19",
        "timestamp": time.time(),
        "auxiliary_paths": [],
    }
    monkeypatch.setattr(tcb, "create_snapshot_tarball",
                        lambda: (fake_tarball, fake_metadata))
    # Bypass S4 (no real Irys query in tests)
    from titan_plugin.logic.backup_cascade import BackupCascade
    monkeypatch.setattr(BackupCascade, "check_irys_balance",
                        lambda self, size_mb: (True, 1.0, "test"))
    # Don't let _record_arweave_anchor touch real manifest
    monkeypatch.setattr(tcb, "_record_arweave_anchor", lambda tx, md: None)

    # Redirect local_dir
    full_cfg = {
        "backup": {"mode": "mainnet_arweave", "local_dir": str(tmp_path)},
        "network": {"wallet_keypair_path": "fake"},
    }

    tx_id = asyncio.run(tcb.snapshot_to_arweave(
        full_config=full_cfg,
        local_dir=str(tmp_path),
        retention_days=30,
    ))
    assert tx_id == "test_tx_123456789abcdef"
    # Verify S3 local snapshot WAS created
    local_files = list(tmp_path.glob("timechain_*.tar.zst"))
    assert len(local_files) == 1, \
        f"Expected 1 timechain local snapshot, got {len(local_files)}"
