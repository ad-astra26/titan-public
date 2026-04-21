"""Tests for rFP_backup_worker Phase 2 cascade on path-based uploads (personality, soul).

Cascade logic lives in titan_plugin.logic.backup_cascade.BackupCascade — exercised
directly here. For TimeChain (bytes-based) variant see test_backup_cascade_timechain.py.
Does NOT test live Arweave upload (S5) or gateway HEAD verify (S6).
"""
import asyncio
import hashlib
import io
import os
import tarfile
import time

import pytest

from titan_plugin.logic.backup_cascade import BackupCascade


def _make_tarball_file(path: str, payload: bytes = b"hello world") -> str:
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(name="test.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return path


def _make_cascade(tmp_path, mode=None, arweave_store=object()):
    cfg = {
        "backup": {"mode": mode, "local_rolling_days": 30} if mode else {},
        "network": {"wallet_keypair_path": "data/titan_identity_keypair.json"},
    }
    return BackupCascade(full_config=cfg, arweave_store=arweave_store,
                         local_dir=str(tmp_path))


# ── S2 validate ──────────────────────────────────────────────────────────

def test_s2_validate_accepts_valid_tarball(tmp_path):
    p = _make_tarball_file(str(tmp_path / "ok.tar.gz"))
    assert BackupCascade.validate_tarball(p) is True


def test_s2_validate_rejects_truncated_tarball(tmp_path):
    p = _make_tarball_file(str(tmp_path / "trunc.tar.gz"))
    with open(p, "r+b") as f:
        f.truncate(20)
    assert BackupCascade.validate_tarball(p) is False


def test_s2_validate_rejects_nonexistent(tmp_path):
    assert BackupCascade.validate_tarball(str(tmp_path / "nope.tar.gz")) is False


# ── S3 save local ────────────────────────────────────────────────────────

def test_s3_save_local_copies_to_backup_dir(tmp_path):
    c = _make_cascade(tmp_path)
    src = _make_tarball_file(str(tmp_path / "src.tar.gz"))
    h = hashlib.sha256(open(src, "rb").read()).hexdigest()
    local = c.save_local(src, "personality", h)
    assert local is not None
    assert os.path.exists(local)
    assert "personality_" in local
    assert h[:12] in local


def test_s3_save_local_idempotent(tmp_path):
    c = _make_cascade(tmp_path)
    src = _make_tarball_file(str(tmp_path / "src.tar.gz"))
    h = hashlib.sha256(open(src, "rb").read()).hexdigest()
    local1 = c.save_local(src, "soul", h)
    mtime1 = os.path.getmtime(local1)
    time.sleep(0.01)
    local2 = c.save_local(src, "soul", h)
    assert local1 == local2
    assert os.path.getmtime(local2) == mtime1


# ── S10 cleanup ──────────────────────────────────────────────────────────

def test_s10_cleanup_removes_old(tmp_path):
    c = _make_cascade(tmp_path)
    old = tmp_path / "personality_20251201_aaaaaaaaaaaa.tar.gz"
    new = tmp_path / "personality_20260420_bbbbbbbbbbbb.tar.gz"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    os.utime(old, (time.time() - 60 * 86400, time.time() - 60 * 86400))
    os.utime(new, (time.time() - 1 * 86400, time.time() - 1 * 86400))
    n = c.cleanup_local("personality", retention_days=30)
    assert n == 1
    assert not old.exists()
    assert new.exists()


def test_s10_cleanup_separates_by_type(tmp_path):
    c = _make_cascade(tmp_path)
    p = tmp_path / "personality_20251201_aaaa.tar.gz"
    s = tmp_path / "soul_20251201_bbbb.tar.gz"
    p.write_bytes(b"p"); s.write_bytes(b"s")
    old_t = time.time() - 60 * 86400
    os.utime(p, (old_t, old_t)); os.utime(s, (old_t, old_t))
    n = c.cleanup_local("personality", retention_days=30)
    assert n == 1
    assert not p.exists()
    assert s.exists()


# ── I5 diff audit ────────────────────────────────────────────────────────

def test_i5_diff_audit_flags_large_size_drop(tmp_path):
    c = _make_cascade(tmp_path)
    fake_latest = lambda bt: {"size_mb": 100.0}
    alert = c.diff_audit("personality", 40.0, fake_latest)
    assert alert is not None
    assert alert["severity"] == "ERROR"
    assert alert["delta_pct"] == 60.0


def test_i5_diff_audit_silent_on_small_delta(tmp_path):
    c = _make_cascade(tmp_path)
    assert c.diff_audit("personality", 110.0, lambda bt: {"size_mb": 100.0}) is None


def test_i5_diff_audit_no_prior_record(tmp_path):
    c = _make_cascade(tmp_path)
    assert c.diff_audit("personality", 100.0, lambda bt: None) is None


def test_i5_diff_audit_no_getter(tmp_path):
    c = _make_cascade(tmp_path)
    assert c.diff_audit("personality", 100.0, None) is None


# ── Mode dispatch ────────────────────────────────────────────────────────

def test_local_only_mode_explicit(tmp_path):
    c = _make_cascade(tmp_path, mode="local_only")
    assert c.is_local_only() is True


def test_mainnet_arweave_mode_explicit(tmp_path):
    c = _make_cascade(tmp_path, mode="mainnet_arweave")
    assert c.is_local_only() is False


def test_mode_fallback_from_arweave_store_none(tmp_path):
    c = _make_cascade(tmp_path, mode=None, arweave_store=None)
    assert c.is_local_only() is True


def test_mode_fallback_from_arweave_store_set(tmp_path):
    c = _make_cascade(tmp_path, mode=None, arweave_store=object())
    assert c.is_local_only() is False


# ── End-to-end cascade (local_only — no network) ─────────────────────────

def test_cascade_local_only_runs_s1_s2_s3_skips_s4_s5(tmp_path):
    c = _make_cascade(tmp_path, mode="local_only")
    src = _make_tarball_file(str(tmp_path / "pers.tar.gz"))

    async def _upload_fn(_path):
        pytest.fail("upload_fn must not be called in local_only mode")

    result = asyncio.run(c.run(src, "personality", _upload_fn, retention_days=30))
    assert result is not None
    assert result["mode"] == "local_only"
    assert "local_path" in result
    assert os.path.exists(result["local_path"])
    assert "arweave_tx" not in result


def test_cascade_s2_fail_returns_cascade_fail(tmp_path):
    c = _make_cascade(tmp_path, mode="local_only")
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"not a tarball")

    async def _upload_fn(_path):
        pytest.fail("should not reach S5")

    result = asyncio.run(c.run(str(bad), "personality", _upload_fn))
    assert result is not None
    assert result.get("cascade_fail") == "S2_validate"


def test_cascade_s5_returns_none_surfaces(tmp_path):
    c = _make_cascade(tmp_path, mode="mainnet_arweave")
    c.check_irys_balance = lambda size: (True, 1.0, "fake")
    src = _make_tarball_file(str(tmp_path / "pers.tar.gz"))

    async def _upload_fn(_path):
        return None

    result = asyncio.run(c.run(src, "personality", _upload_fn))
    assert result is not None
    assert result.get("cascade_fail") == "S5_upload_none"


def test_cascade_s5_success_without_verify_for_devnet(tmp_path):
    c = _make_cascade(tmp_path, mode="mainnet_arweave")
    c.check_irys_balance = lambda size: (True, 1.0, "fake")
    src = _make_tarball_file(str(tmp_path / "pers.tar.gz"))

    async def _upload_fn(_path):
        return {"arweave_tx": "devnet_fake_tx", "size_mb": 0.001}

    result = asyncio.run(c.run(src, "personality", _upload_fn))
    assert result is not None
    assert result["arweave_tx"] == "devnet_fake_tx"
    assert result["verified"] == "skipped_devnet"
    assert result["local_path"] is not None


def test_cascade_low_balance_falls_back_to_local(tmp_path):
    c = _make_cascade(tmp_path, mode="mainnet_arweave")
    c.check_irys_balance = lambda size: (False, 0.0, "need=0.01 have=0.0")
    src = _make_tarball_file(str(tmp_path / "pers.tar.gz"))

    async def _upload_fn(_path):
        pytest.fail("upload_fn must not run when balance insufficient")

    result = asyncio.run(c.run(src, "personality", _upload_fn))
    assert result is not None
    assert result["mode"] == "local_fallback_low_balance"
    assert os.path.exists(result["local_path"])
