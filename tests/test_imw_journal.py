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


def test_scan_orphan_journals_filters_by_instance_prefix(tmp_path):
    """Per-instance journal naming: each daemon's scan only picks up
    journals belonging to its own writer instance.

    Pre-fix bug: both inner_memory_client and observatory_writer_client
    wrote to `imw_<pid>.jrn` and the inner_memory daemon's
    `scan_orphan_journals` globbed `imw_*.jrn`, so observatory writes
    were replayed against `inner_memory.db` → "no such table" failures.
    Fixed by deriving the prefix from `Path(cfg.socket_path).stem`.
    """
    own_pid = os.getpid()
    # Create journals from BOTH writer instances for a guaranteed-dead pid.
    # 999999 is well above the pid_max on Linux (typically 32768 or 4194304)
    # but `os.kill(pid, 0)` will reliably raise ProcessLookupError for it.
    DEAD_PID = 999999
    inner_journal = tmp_path / f"imw_{DEAD_PID}.jrn"
    obs_journal = tmp_path / f"observatory_writer_{DEAD_PID}.jrn"
    CallerJournal(str(inner_journal), pid=DEAD_PID).close()
    CallerJournal(str(obs_journal), pid=DEAD_PID).close()

    # inner_memory daemon (default prefix "imw") sees ONLY its own
    inner_orphans = scan_orphan_journals(
        str(tmp_path), instance_prefix="imw", exclude_pid=own_pid)
    inner_names = [str(o[0]) for o in inner_orphans]
    assert str(inner_journal) in inner_names, "inner_memory daemon must see imw_*.jrn"
    assert str(obs_journal) not in inner_names, \
        "inner_memory daemon must NOT see observatory_writer_*.jrn"

    # observatory_writer daemon sees ONLY its own
    obs_orphans = scan_orphan_journals(
        str(tmp_path), instance_prefix="observatory_writer", exclude_pid=own_pid)
    obs_names = [str(o[0]) for o in obs_orphans]
    assert str(obs_journal) in obs_names, \
        "observatory_writer daemon must see observatory_writer_*.jrn"
    assert str(inner_journal) not in obs_names, \
        "observatory_writer daemon must NOT see imw_*.jrn"


def test_scan_orphan_journals_default_prefix_is_backward_compat(tmp_path):
    """Existing callers (none in-repo, but to avoid surprises) that don't
    pass instance_prefix get the legacy "imw" default — same behavior as
    before the multi-instance fix."""
    own_pid = os.getpid()
    DEAD_PID = 999998
    p = tmp_path / f"imw_{DEAD_PID}.jrn"
    CallerJournal(str(p), pid=DEAD_PID).close()
    orphans = scan_orphan_journals(str(tmp_path), exclude_pid=own_pid)
    assert str(p) in [str(o[0]) for o in orphans]


def test_scan_orphan_journals_parses_pid_from_multiword_prefix(tmp_path):
    """Underscore-containing instance prefixes (e.g. observatory_writer)
    must still parse the trailing pid correctly — uses rsplit(_,1)."""
    own_pid = os.getpid()
    p = tmp_path / "observatory_writer_42.jrn"
    CallerJournal(str(p), pid=42).close()
    orphans = scan_orphan_journals(
        str(tmp_path), instance_prefix="observatory_writer", exclude_pid=own_pid)
    assert len(orphans) == 1
    assert orphans[0][1] == 42, f"expected pid=42, got {orphans[0][1]}"


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
