"""RestoreWorker — RFP_backup_redesign_spine Phase C (§7.C).

The unified cold+warm restore entry over the relocated sovereign engine. The
engine itself (resurrect_from_chain / restore_body_from_chain) is exercised
byte-identically by test_backup_restore_sovereign / test_resurrection /
test_resurrection_routing (all green post-relocation). Here we test the NEW
wrapper behaviour: the single-flight guard (INV-BRS-6) + mode → commit mapping.
"""
from unittest.mock import patch

import pytest

from titan_hcl.logic.restore_worker import RestoreWorker, ResurrectionResult


def test_single_flight_guard():
    """INV-BRS-6 — a second restore while one is in flight is refused (closes the
    audit H-bug: the weekly restore-test wasn't lock-guarded vs a meditation cascade)."""
    rw = RestoreWorker(titan_id="T1")
    # Simulate an in-flight restore by holding the process-wide lock.
    assert RestoreWorker._restore_lock.acquire(blocking=False)
    try:
        out = rw.restore(key_bytes=b"\x00" * 64, titan_pubkey="PUB",
                         install_root="/tmp/should_not_be_touched", mode="full")
        assert out.status == "halted"
        assert out.halt_reason == "restore_already_in_flight"
    finally:
        RestoreWorker._restore_lock.release()
    # lock released → a subsequent restore is allowed to proceed (reaches the engine)
    assert RestoreWorker._restore_lock.acquire(blocking=False)
    RestoreWorker._restore_lock.release()


def test_mode_validation():
    rw = RestoreWorker(titan_id="T1")
    with pytest.raises(ValueError):
        rw.restore(key_bytes=b"\x00" * 64, titan_pubkey="P",
                   install_root="/tmp/x", mode="bogus")


def test_mode_commit_mapping():
    """full → commit (swap into data/); verify_test → no commit (scratch only)."""
    rw = RestoreWorker(titan_id="T1")
    captured = {}

    def _fake(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return ResurrectionResult(status="resurrected")

    with patch("titan_hcl.logic.restore_worker.restore_body_from_chain",
               side_effect=_fake):
        rw.restore(key_bytes=b"\x00" * 64, titan_pubkey="PUB",
                   install_root="/tmp/x", mode="full")
        assert captured["commit"] is True
        assert captured["titan_pubkey"] == "PUB"
        rw.restore(key_bytes=b"\x00" * 64, titan_pubkey="PUB",
                   install_root="/tmp/x", mode="verify_test")
        assert captured["commit"] is False
