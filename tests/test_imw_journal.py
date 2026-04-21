"""Tests for IMW CallerJournal — append, fsync, replay, rotate, truncate."""
import os
from pathlib import Path

import pytest

from titan_plugin.persistence.journal import (
    HEADER_SIZE,
    CallerJournal,
    JournalError,
    scan_orphan_journals,
)


def test_fresh_journal_writes_header(tmp_path):
    p = tmp_path / "imw_9999.jrn"
    j = CallerJournal(str(p), pid=9999)
    assert p.stat().st_size == HEADER_SIZE
    j.close()


def test_append_and_replay(tmp_path):
    p = tmp_path / "imw_1.jrn"
    j = CallerJournal(str(p), pid=1)
    offs = []
    for i in range(5):
        off = j.append(f"req{i}", "INSERT INTO t VALUES (?)", [i])
        offs.append(off)
    # Don't close — simulate crash. Reopen and iterate unacked
    j2 = CallerJournal(str(p), pid=1)
    unacked = list(j2.iter_unacked())
    assert len(unacked) == 5
    assert unacked[0][2]["req_id"] == "req0"
    assert unacked[4][2]["params"] == [4]
    j2.close()


def test_ack_advances_last_acked_offset(tmp_path):
    p = tmp_path / "imw_2.jrn"
    j = CallerJournal(str(p), pid=2)
    # Append many so header flush triggers
    offs = []
    for i in range(150):
        off = j.append(f"req{i}", "x", [i])
        offs.append(off)
    # Ack all — triggers header flush at 100
    for off in offs:
        # compute frame length by reading directly
        import struct
        with open(p, "rb") as f:
            f.seek(off)
            n = struct.unpack(">I", f.read(4))[0]
        j.ack(off, 4 + n)
    j.close()
    # Reopen: unacked should be empty
    j2 = CallerJournal(str(p), pid=2)
    assert list(j2.iter_unacked()) == []
    j2.close()


def test_close_truncates_acked_region(tmp_path):
    p = tmp_path / "imw_3.jrn"
    j = CallerJournal(str(p), pid=3)
    offs = []
    for i in range(3):
        offs.append(j.append(f"r{i}", "x", [i]))
    for off in offs:
        import struct
        with open(p, "rb") as f:
            f.seek(off)
            n = struct.unpack(">I", f.read(4))[0]
        j.ack(off, 4 + n)
    size_before = p.stat().st_size
    j.close()
    size_after = p.stat().st_size
    assert size_after <= size_before
    assert size_after == HEADER_SIZE  # fully acked → trimmed to header


def test_scan_orphan_journals_skips_live_pid(tmp_path):
    own_pid = os.getpid()
    p_alive = tmp_path / f"imw_{own_pid}.jrn"
    p_dead = tmp_path / "imw_1.jrn"
    CallerJournal(str(p_alive), pid=own_pid).close()
    CallerJournal(str(p_dead), pid=1).close()
    orphans = scan_orphan_journals(str(tmp_path), exclude_pid=own_pid)
    # our pid is alive, skipped; pid=1 should be dead on this system
    names = [str(o[0]) for o in orphans]
    assert str(p_alive) not in names


def test_rotate_on_size_cap(tmp_path):
    p = tmp_path / "imw_7.jrn"
    # Use a very tiny cap to force rotation
    j = CallerJournal(str(p), pid=7, max_mb=1)
    # At 1MB cap, each record ~50 bytes — need 20000 to trigger
    # Instead, use large payloads
    big = "x" * 100_000
    for i in range(15):
        j.append(f"r{i}", "INSERT INTO t VALUES (?)", [big])
    old = p.with_suffix(p.suffix + ".old")
    assert old.exists()  # rotation happened at some point
    j.close()
