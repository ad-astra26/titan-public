"""
Tests for B.1 §10 — TimeChain system fork registration + dual-fork
write semantics.

PLAN: titan-docs/PLAN_microkernel_phase_b1_shadow_swap.md §10
"""
from __future__ import annotations

import sqlite3
import time

import pytest

from titan_plugin.logic.timechain import (
    BlockPayload, FORK_EPISODIC, FORK_NAMES, FORK_SYSTEM, TimeChain,
)


class TestForkSystemConstant:
    def test_fork_system_id_is_100(self):
        assert FORK_SYSTEM == 100

    def test_fork_system_in_names(self):
        assert FORK_SYSTEM in FORK_NAMES
        assert FORK_NAMES[FORK_SYSTEM] == "system"


class TestSystemForkRegistration:
    def test_new_chain_genesis_registers_system_fork(self, tmp_path):
        tc = TimeChain(data_dir=str(tmp_path), titan_id="T1")
        # Before genesis, fork_tips might be empty
        # After we trigger genesis OR call _ensure...
        # _ensure is called by __init__ unconditionally; verify
        assert FORK_SYSTEM in tc._fork_tips

    def test_existing_chain_migration_idempotent(self, tmp_path):
        tc1 = TimeChain(data_dir=str(tmp_path), titan_id="T1")
        # Re-instantiate (simulates restart)
        del tc1
        tc2 = TimeChain(data_dir=str(tmp_path), titan_id="T1")
        assert FORK_SYSTEM in tc2._fork_tips
        # Verify no double-insert in DB
        conn = sqlite3.connect(str(tc2._index_db_path))
        cur = conn.execute(
            "SELECT COUNT(*) FROM fork_registry WHERE fork_id = ?",
            (FORK_SYSTEM,),
        )
        count = cur.fetchone()[0]
        conn.close()
        assert count == 1, "FORK_SYSTEM registered exactly once"


class TestDualForkCommit:
    def _build_tc(self, tmp_path):
        tc = TimeChain(data_dir=str(tmp_path), titan_id="T1")
        tc.create_genesis(genesis_content={
            "titan_id": "T1", "test": True, "purpose": "B.1 §10 test",
        })
        return tc

    def test_commit_to_system_fork(self, tmp_path):
        tc = self._build_tc(tmp_path)
        event_id = "abc123def456"
        payload = BlockPayload(
            thought_type="system", source="shadow_swap",
            content={
                "event_id": event_id,
                "event_type": "SYSTEM_UPGRADE_QUEUED",
                "reason": "test",
            },
            tags=["b1", "shadow_swap"],
            significance=1.0, confidence=1.0,
        )
        block = tc.commit_block(
            fork_id=FORK_SYSTEM, epoch_id=1,
            payload=payload, pot_nonce=0, chi_spent=0.0,
            neuromod_state={"DA": 0.5, "5HT": 0.5, "NE": 0.5},
        )
        assert block is not None
        assert block.header.fork_id == FORK_SYSTEM
        assert block.payload.content["event_id"] == event_id

    def test_dual_write_linked_by_event_id(self, tmp_path):
        """Verify Option C: one event_id appears in both system + episodic blocks."""
        tc = self._build_tc(tmp_path)
        event_id = "linked_event_id_xyz"
        # System block
        sys_block = tc.commit_block(
            fork_id=FORK_SYSTEM, epoch_id=1,
            payload=BlockPayload(
                thought_type="system", source="shadow_swap",
                content={"event_id": event_id, "event_type": "SYSTEM_UPGRADE_QUEUED"},
                significance=1.0,
            ),
            pot_nonce=0, chi_spent=0.0,
            neuromod_state={"DA": 0.5, "5HT": 0.5, "NE": 0.5},
        )
        # Episodic block (the experiential thought)
        ep_block = tc.commit_block(
            fork_id=FORK_EPISODIC, epoch_id=1,
            payload=BlockPayload(
                thought_type="episodic", source="spirit",
                content={
                    "event_id": event_id,
                    "system_event_type": "queued",
                    "self_thought": "I sense an upgrade approaching...",
                },
                significance=1.0,
            ),
            pot_nonce=0, chi_spent=0.0,
            neuromod_state={"DA": 0.5, "5HT": 0.5, "NE": 0.5},
        )
        assert sys_block.payload.content["event_id"] == ep_block.payload.content["event_id"]
        assert sys_block.header.fork_id != ep_block.header.fork_id


class TestNoBackwardsBreak:
    def test_existing_forks_unchanged(self):
        """Adding FORK_SYSTEM=100 doesn't shift existing fork IDs."""
        from titan_plugin.logic.timechain import (
            FORK_MAIN, FORK_DECLARATIVE, FORK_PROCEDURAL,
            FORK_EPISODIC as _EPI, FORK_META, FORK_CONVERSATION,
            FORK_SIDECHAIN_START,
        )
        assert FORK_MAIN == 0
        assert FORK_DECLARATIVE == 1
        assert FORK_PROCEDURAL == 2
        assert _EPI == 3
        assert FORK_META == 4
        assert FORK_CONVERSATION == 5
        assert FORK_SIDECHAIN_START == 6
        # FORK_SYSTEM uses high reservation, doesn't conflict with sidechains
        assert FORK_SYSTEM > FORK_SIDECHAIN_START
        assert FORK_SYSTEM == 100
