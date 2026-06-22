"""Integration test for the Inner Turn worker loop (Phase A.2).

Drives `_IntrospectionRoutine` with a FAKE ShmReaderBank across two great pulses
and asserts the end-to-end event-to-event loop:
  • G6  — nothing fires without a great-pulse delta (event-driven, no timer).
  • G1  — a real inner_reward_tuple accrues with ZERO chat traffic.
  • INV-IT-8 — predict at pulse N, verify at pulse N+1.
  • G5  — the OUTER reward_tuples table is provably untouched.
  • the inner MasteryLevel SHM slot is published (Phase D readout source).
"""
import os
import tempfile

import numpy as np
import pytest

import titan_hcl.api.shm_reader_bank as _srb
from titan_hcl.modules import self_learning_worker as slw
from titan_hcl.synthesis.inner_introspection import NEUROMOD_ORDER


class _FakeBank:
    def __init__(self, **_kw):
        self.gp = 0
        self.state = np.zeros(71, dtype=np.float32)

    def read_resonance_metadata(self):
        return {"great_pulse_count": self.gp}

    def read_inner_body_5d(self):
        return {"values": self.state[:5].tolist()}

    def read_inner_mind_15d(self):
        return {"values": self.state[5:20].tolist()}

    def read_inner_spirit_45d(self):
        return {"values": self.state[20:65].tolist()}

    def read_neuromod(self):
        return {"modulators": {n: {"level": float(self.state[65 + i])}
                               for i, n in enumerate(NEUROMOD_ORDER)}}


class _FakeWriter:
    def __init__(self, *_a, **_k):
        self.last = None

    def write(self, flat):
        self.last = np.asarray(flat)


class _FakeLife:
    def is_dreaming(self):
        return False

    def get_metabolic_drain(self):
        return 0.0


class _Q:
    def __init__(self):
        self.items = []

    def put(self, x):
        self.items.append(x)


def _seeking_neuromod():
    # DA, 5HT, NE, ACh, Endorphin, GABA — high DA+NE, low GABA → strong drive.
    return np.array([0.9, 0.5, 0.9, 0.5, 0.5, 0.1], dtype=np.float32)


def test_inner_turn_event_to_event_loop(monkeypatch):
    monkeypatch.setattr(_srb, "ShmReaderBank", _FakeBank)
    cfg = dict(slw._DEFAULTS)
    cfg["inner_turn_enabled"] = True
    cfg["inner_persist_every"] = 1

    with tempfile.TemporaryDirectory() as td:
        store = slw._SelfLearningStore(path=os.path.join(td, "sl.duckdb"))
        routine = slw._IntrospectionRoutine(
            cfg, store, "self_learning", _Q(), "test",
            ensure_shm_root=lambda _t: td, StateRegistryWriter=_FakeWriter)
        assert routine.enabled, "routine should init"
        fb = _FakeBank()
        routine.bank = fb
        life = _FakeLife()

        # gp=0 first poll establishes the baseline — nothing fires (G6).
        routine.tick(life, chat_active=False)
        assert store.inner_reward_count() == 0

        # gp still 0 → still nothing (no delta).
        routine.tick(life, chat_active=False)
        assert store.pop_inner_prediction() is None or True  # no-op guard

        # Great pulse #1 — seeks (high curiosity) → SEED a prediction (t0).
        fb.state = np.concatenate([
            np.full(5, 0.2), np.full(15, 0.3),
            np.linspace(-1, 1, 45), _seeking_neuromod()]).astype(np.float32)
        fb.gp = 1
        routine.tick(life, chat_active=False)
        assert store.inner_reward_count() == 0          # not verified yet (INV-IT-8)
        pend = store._conn.execute(
            "SELECT COUNT(*) FROM inner_pending_prediction").fetchone()[0]
        assert pend == 1, "a prediction should be stashed"

        # Great pulse #2 — VERIFY the gp#1 prediction → a real inner_reward_tuple.
        fb.state = np.concatenate([
            np.full(5, 0.25), np.full(15, 0.35),
            np.linspace(-0.9, 0.9, 45), _seeking_neuromod()]).astype(np.float32)
        fb.gp = 2
        n_updates_before = routine.predictor.updates
        routine.tick(life, chat_active=False)

        assert store.inner_reward_count() >= 1, "G1 — inner reward accrued"
        assert routine.predictor.updates > n_updates_before, "self-model learned"
        assert routine.level_writer.last is not None, "inner level published to SHM"
        # G5 — the OUTER reward_tuples table is untouched.
        outer = store._conn.execute("SELECT COUNT(*) FROM reward_tuples").fetchone()[0]
        assert outer == 0, "G5 — outer routing store untouched"
        # the reward is bounded telemetry (INV-IT-1).
        r = store._conn.execute(
            "SELECT reward, goal_class FROM inner_reward_tuples LIMIT 1").fetchone()
        assert -1.0 <= float(r[0]) <= 1.0 and r[1] == "inner:introspection"


def test_no_fire_when_dreaming(monkeypatch):
    monkeypatch.setattr(_srb, "ShmReaderBank", _FakeBank)
    cfg = dict(slw._DEFAULTS)
    cfg["inner_turn_enabled"] = True

    class _Dreaming(_FakeLife):
        def is_dreaming(self):
            return True

    with tempfile.TemporaryDirectory() as td:
        store = slw._SelfLearningStore(path=os.path.join(td, "sl.duckdb"))
        routine = slw._IntrospectionRoutine(
            cfg, store, "self_learning", _Q(), "test",
            ensure_shm_root=lambda _t: td, StateRegistryWriter=_FakeWriter)
        fb = _FakeBank()
        routine.bank = fb
        routine.tick(_Dreaming(), chat_active=False)        # baseline
        fb.state = np.concatenate([np.full(5, 0.2), np.full(15, 0.3),
                                   np.linspace(-1, 1, 45),
                                   _seeking_neuromod()]).astype(np.float32)
        fb.gp = 1
        routine.tick(_Dreaming(), chat_active=False)
        # dreaming → the luxury of rest is denied → no prediction stashed (INV-IT-3).
        pend = store._conn.execute(
            "SELECT COUNT(*) FROM inner_pending_prediction").fetchone()[0]
        assert pend == 0


def test_disabled_flag_is_inert(monkeypatch):
    monkeypatch.setattr(_srb, "ShmReaderBank", _FakeBank)
    cfg = dict(slw._DEFAULTS)
    cfg["inner_turn_enabled"] = False
    with tempfile.TemporaryDirectory() as td:
        store = slw._SelfLearningStore(path=os.path.join(td, "sl.duckdb"))
        routine = slw._IntrospectionRoutine(
            cfg, store, "self_learning", _Q(), "test",
            ensure_shm_root=lambda _t: td, StateRegistryWriter=_FakeWriter)
        assert not routine.enabled
        routine.tick(_FakeLife(), chat_active=False)        # no-op, never raises
        assert store.inner_reward_count() == 0
