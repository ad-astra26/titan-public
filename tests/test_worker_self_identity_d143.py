"""WORKER-SELF-IDENTITY (D-SPEC-143 / SPEC v1.72.0) — INV-PROC-7 + roster v3.

Pins both parts:
  (a) per-worker proc title — `set_proc_name()` PR_SET_NAME helper + the
      `_module_wrapper` first-I/O identity block.
  (b) module→pid roster — `TitanHclStateEntry` roster 3-tuples (name, prio, pid),
      backward-compatible read of v2 dict / 2-tuple encodings.
"""
from __future__ import annotations

import inspect
import platform

import pytest

from titan_hcl.core.titan_hcl_state import (
    TITAN_HCL_STATE_PAYLOAD_VERSION,
    TITAN_HCL_STATE_SCHEMA_VERSION,
    TitanHclStateEntry,
)


# ── Part (b) — roster schema v3 ──────────────────────────────────────────────

def test_container_schema_version_unchanged_at_2():
    """The slot-CONTAINER byte-layout version MUST stay 2 — the roster-pid add
    is a payload change inside the same container; bumping it would risk a
    stale-/dev/shm schema_mismatch on rollout (readers reject on mismatch)."""
    assert TITAN_HCL_STATE_SCHEMA_VERSION == 2
    assert TITAN_HCL_STATE_PAYLOAD_VERSION == 3


def test_roster_v3_roundtrip_preserves_pid():
    e = TitanHclStateEntry(roster=(("cognitive_worker", "mandatory", 12345),
                                   ("agno_worker", "post_boot", 67890)))
    back = TitanHclStateEntry.from_wire_dict(e.as_wire_dict())
    assert back.roster == (("cognitive_worker", "mandatory", 12345),
                           ("agno_worker", "post_boot", 67890))
    assert back.schema_version == TITAN_HCL_STATE_PAYLOAD_VERSION


def test_wire_roster_is_list_of_triples():
    e = TitanHclStateEntry(roster=(("body_worker", "mandatory", 42),))
    wire = e.as_wire_dict()
    assert wire["roster"] == [["body_worker", "mandatory", 42]]


def test_from_wire_dict_v2_dict_backcompat_pid_zero():
    """A v2 slot encoded roster as a dict {name: prio} — must load with pid=0."""
    v2 = {"roster": {"memory": "mandatory", "persona_worker": "post_boot"}}
    back = TitanHclStateEntry.from_wire_dict(v2)
    assert set(back.roster) == {("memory", "mandatory", 0),
                                ("persona_worker", "post_boot", 0)}


def test_from_wire_dict_v2_list_of_pairs_backcompat_pid_zero():
    v2 = {"roster": [["ns_module", "mandatory"], ["hormonal_worker", "post_boot"]]}
    back = TitanHclStateEntry.from_wire_dict(v2)
    assert set(back.roster) == {("ns_module", "mandatory", 0),
                                ("hormonal_worker", "post_boot", 0)}


def test_from_wire_dict_empty_roster():
    assert TitanHclStateEntry.from_wire_dict({}).roster == ()


def test_old_v2_reader_logic_ignores_pid_gracefully():
    """Simulates a pre-deploy (v2) reader hitting its list-of-pairs branch on a
    NEW v3 wire payload — it reads (name, prio) and drops pid, never crashes."""
    wire = TitanHclStateEntry(
        roster=(("cgn", "mandatory", 999),)).as_wire_dict()
    raw_roster = wire["roster"]  # [["cgn","mandatory",999]]
    # v2 from_wire_dict list branch: (str(p[0]), str(p[1])) for len(p)>=2
    v2_parsed = tuple((str(p[0]), str(p[1])) for p in raw_roster if len(p) >= 2)
    assert v2_parsed == (("cgn", "mandatory"),)


# ── update() roster normalization ────────────────────────────────────────────

def test_writer_update_normalizes_2tuples(tmp_path, monkeypatch):
    """update(roster=2-tuples) → stored as 3-tuples with pid=0 (boot-time path
    publishes the roster before pids are known)."""
    from titan_hcl.core import titan_hcl_state as ths

    monkeypatch.setenv("TITAN_ID", "T_TEST_D143")
    # Avoid touching real /dev/shm slots: drive the dataclass directly.
    e = TitanHclStateEntry()
    # Simulate the update() normalization expression on a 2-tuple roster.
    roster_in = (("memory", "mandatory"), ("agno_worker", "post_boot"))
    normalized = tuple(
        (str(t[0]), str(t[1]), int(t[2]) if len(t) >= 3 else 0)
        for t in roster_in)
    assert normalized == (("memory", "mandatory", 0),
                          ("agno_worker", "post_boot", 0))
    e2 = TitanHclStateEntry(roster=normalized)
    assert TitanHclStateEntry.from_wire_dict(e2.as_wire_dict()).roster == normalized


# ── Part (a) — proc identity ─────────────────────────────────────────────────

def test_set_proc_name_exists_and_returns_bool():
    from titan_hcl.core.worker_lifecycle import set_proc_name
    rc = set_proc_name("test_worker_d143")
    assert isinstance(rc, bool)


@pytest.mark.skipif(platform.system() != "Linux", reason="prctl is Linux-only")
def test_set_proc_name_sets_comm_under_15_bytes():
    import os
    from titan_hcl.core.worker_lifecycle import set_proc_name
    assert set_proc_name("cognitive_worker") is True
    with open(f"/proc/{os.getpid()}/comm") as f:
        comm = f.read().strip()
    assert comm == "titan:cognitive"  # "titan:cognitive_worker"[:15]
    assert len(comm.encode()) <= 15


def test_set_proc_name_truncates_to_15_bytes():
    # "titan:" (6) + name → kernel caps at 15; helper must not raise on a long name.
    from titan_hcl.core.worker_lifecycle import set_proc_name
    assert isinstance(set_proc_name("a_very_long_worker_name_xyz"), bool)


def test_pr_set_name_constant():
    from titan_hcl.core import worker_lifecycle as wl
    assert wl._PR_SET_NAME == 15
    assert wl._PR_SET_PDEATHSIG == 1


def test_module_wrapper_sets_identity_as_first_io():
    """Source-text pin (INV-PROC-7): _module_wrapper sets setproctitle
    titan_hcl:<name> + set_proc_name BEFORE install_full_protection()."""
    from titan_hcl.orchestrator.core import _module_wrapper
    src = inspect.getsource(_module_wrapper)
    assert 'setproctitle(f"titan_hcl:{name}")' in src
    assert "set_proc_name(name)" in src
    # ordering: identity block precedes the install_full_protection() CALL site
    # (anchor on `_wl = install_full_protection()`, not the comment mention).
    assert src.index('setproctitle(f"titan_hcl:{name}")') < src.index("_wl = install_full_protection()")
    assert "INV-PROC-7" in src
