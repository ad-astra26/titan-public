"""Phase D (D-SPEC-116) — spirit_worker retirement + orphaned-flow restoration.

Guards the retirement of spirit_worker.py and the faithful re-homing of its 6
orphaned dst="spirit" flows. Static-source checks (no worker boot) — fast and
deterministic.

  REPOINT  MEMORY_RECALL_PERTURBATION → neuromod (nudge) + cognitive (i_depth/wm)
  REPOINT  REFLEX_REWARD → NS_REWARD (firehose) → cognitive
  RESTORE  TEACHER_SIGNALS → cognitive (msl.concept_grounder + neuromod nudge)
  RESTORE  OUTER_OBSERVATION → cognitive (msl.signal_engagement)
  RETIRE   STATE_SNAPSHOT publisher + observe_topology chain + RATE_LIMIT notify

Run: python -m pytest tests/test_phase_d_spirit_retirement.py -v -p no:anchorpy
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Module retirement ─────────────────────────────────────────────────

def test_spirit_worker_module_deleted():
    assert importlib.util.find_spec("titan_hcl.modules.spirit_worker") is None, \
        "spirit_worker.py must stay deleted (D-SPEC-116)."


def test_spirit_loop_fully_retired():
    """spirit_loop.py is DELETED (rFP §3G Phase 10I) — the orphan helper module
    left by the deleted spirit_worker is gone. Its live functions were relocated
    to proper Phase C homes; the Rust/L2-superseded duplicates + orphaned/dead
    functions were deleted.
    """
    # The module no longer exists.
    assert importlib.util.find_spec("titan_hcl.modules.spirit_loop") is None, \
        "spirit_loop.py must be deleted (Phase 10I)."

    # Relocated LIVE logic lives in its Phase C homes:
    from titan_hcl.logic.spirit_helpers import _compute_spirit_reflex_intuition
    from titan_hcl.logic.consciousness_epoch import (
        _init_consciousness, _run_consciousness_epoch,
    )
    assert callable(_compute_spirit_reflex_intuition)   # 10C
    assert callable(_init_consciousness)                # 10D
    assert callable(_run_consciousness_epoch)           # 10D (+ Rust-SHM consumer-fix)
    # agno_hooks consumes the spirit reflex via the reflex_intuition surface.
    from titan_hcl.logic.reflex_intuition import _compute_spirit_reflex_intuition as _via_reflex
    assert _via_reflex is _compute_spirit_reflex_intuition


# ── cognitive_worker now subscribes to the 3 re-homed flows ────────────

def test_cognitive_subscribes_rehomed_flows():
    from titan_hcl.modules.cognitive_worker import _COGNITIVE_WORKER_SUBSCRIBE_TOPICS
    from titan_hcl import bus
    for t in (bus.MEMORY_RECALL_PERTURBATION, bus.TEACHER_SIGNALS,
              bus.OUTER_OBSERVATION):
        assert t in _COGNITIVE_WORKER_SUBSCRIBE_TOPICS, \
            f"{t} must be in cognitive_worker broadcast_topics (D-SPEC-116)"


def test_cognitive_handles_rehomed_flows():
    """The dispatcher must have a branch for each re-homed type + call the
    restored capability functions."""
    from titan_hcl.modules import cognitive_worker
    src = inspect.getsource(cognitive_worker.cognitive_worker_main)
    assert "bus.MEMORY_RECALL_PERTURBATION" in src
    assert "record_recall_perturbation" in src        # i_depth leg
    assert "bus.TEACHER_SIGNALS" in src
    assert "signal_yes" in src and "signal_you" in src  # MSL grounding leg
    assert "bus.OUTER_OBSERVATION" in src
    assert "signal_engagement" in src                  # X-engagement leg


# ── Emitters repointed away from dst="spirit" ──────────────────────────

def test_agno_hooks_emitters_repointed():
    from titan_hcl.modules import agno_hooks
    src = inspect.getsource(agno_hooks)
    # Recall bridge: nudge → neuromod (target-shaped), legs → cognitive_worker
    assert 'bus.NEUROMOD_EXTERNAL_NUDGE, "interface", "neuromod"' in src
    assert 'bus.MEMORY_RECALL_PERTURBATION, "interface", "cognitive_worker"' in src
    # Reflex reward → NS_REWARD (no dst="spirit", no REFLEX_REWARD emit)
    assert 'NS_REWARD, "titan_vm", "all"' in src
    assert 'REFLEX_REWARD, "titan_vm", "spirit"' not in src


def test_outer_observation_repointed_in_plugin():
    from titan_hcl.core import plugin
    src = inspect.getsource(plugin)
    assert 'OUTER_OBSERVATION, "core", "cognitive_worker"' in src
    assert 'OUTER_OBSERVATION, "core", "spirit"' not in src
    # Spirit ModuleSpec gone.
    assert 'name="spirit"' not in src


def test_teacher_signals_repointed_in_language_worker():
    from titan_hcl.modules import language_worker
    src = inspect.getsource(language_worker)
    assert 'bus.TEACHER_SIGNALS, name, "cognitive_worker"' in src
    assert 'bus.TEACHER_SIGNALS, name, "spirit"' not in src


# ── Retired emitters gone ──────────────────────────────────────────────

def test_state_snapshot_publisher_retired():
    from titan_hcl.logic import state_register
    src = inspect.getsource(state_register)
    assert "_snapshot_publish_loop" not in src
    assert 'bus.STATE_SNAPSHOT, "state_register", "spirit"' not in src


def test_rate_limit_spirit_notify_retired():
    from titan_hcl.core import plugin
    src = inspect.getsource(plugin)
    assert 'make_msg(bus.RATE_LIMIT, "core", "spirit"' not in src


def test_neuromod_nudge_payload_is_target_shaped():
    """The recall-bridge nudge must convert deltas→targets (apply_external_nudge
    pulls toward target). Verify the agno_hooks conversion reads current levels."""
    from titan_hcl.modules import agno_hooks
    src = inspect.getsource(agno_hooks)
    assert "read_neuromod" in src           # reads current levels for the conversion
    assert "_nm_targets" in src             # builds target map


# ── DreamingMeta consult — RUNTIME drive (forced-dream path) ───────────

class _FakeQueue:
    def __init__(self):
        self.items = []

    def put_nowait(self, m):
        self.items.append(m)

    put = put_nowait


def _meta_reqs(q):
    from titan_hcl import bus
    return [m for m in q.items
            if m.get("type") == bus.META_REASON_REQUEST
            and (m.get("payload") or {}).get("consumer_id") == "dreaming"]


def _drive(state_refs, coordinator, send_queue):
    """Drive one waking epoch through the REAL _drive_one_epoch. Downstream
    engine work needs full wiring (out of scope); the DreamingMeta consult
    fires near the top (right after is_dreaming resolution), so we tolerate any
    later raise and inspect what was emitted by then."""
    from titan_hcl.modules import cognitive_worker
    try:
        cognitive_worker._drive_one_epoch(
            state_refs, {}, consciousness=None, coordinator=coordinator,
            pi_monitor=None, reasoning_engine=None, meta_engine=None,
            neuromod_reader=lambda: {"DA": 0.6, "5HT": 0.5},
            shm_bank=None, send_queue=send_queue, name="cognitive_worker")
    except Exception:
        pass  # later engine couplings need full boot — not under test here


def test_dreaming_meta_consult_fires_on_forced_dream_entry():
    """RUNTIME: when the coordinator reports is_dreaming=True (the FORCE_DREAM
    path sets this), _drive_one_epoch emits the DreamingMeta consult — a
    META_REASON_REQUEST(consumer_id='dreaming', question_type='synthesize_insight').
    This is the restored A.2 loop (D-SPEC-116); 'synthesize_insight' is used
    because the original 'consolidate_themes' was never a valid question_type."""
    class _Inner:
        is_dreaming = True

    class _Coord:
        inner = _Inner()

    q = _FakeQueue()
    state_refs = {}
    _drive(state_refs, _Coord(), q)

    reqs = _meta_reqs(q)
    assert len(reqs) == 1, f"expected 1 DreamingMeta consult, got {len(reqs)}"
    payload = reqs[0]["payload"]
    assert payload["question_type"] == "synthesize_insight"
    assert len(payload["context_vector"]) == 30   # build_dreaming_meta_context_30d
    assert state_refs.get("_dreaming_meta_was_dreaming") is True


def test_dreaming_meta_consult_only_on_rising_edge():
    """RUNTIME: a sustained dream (is_dreaming stays True across epochs) must
    consult ONCE — on the waking→dreaming rising edge — not every epoch."""
    class _Inner:
        is_dreaming = True

    class _Coord:
        inner = _Inner()

    q = _FakeQueue()
    state_refs = {}
    coord = _Coord()
    _drive(state_refs, coord, q)   # rising edge → 1 consult
    _drive(state_refs, coord, q)   # still dreaming → no new consult
    _drive(state_refs, coord, q)
    assert len(_meta_reqs(q)) == 1, "DreamingMeta must fire only on the rising edge"


def test_no_dreaming_meta_consult_while_awake():
    """RUNTIME: awake (is_dreaming=False) → no DreamingMeta consult."""
    class _Inner:
        is_dreaming = False

    class _Coord:
        inner = _Inner()

    q = _FakeQueue()
    _drive({}, _Coord(), q)
    assert len(_meta_reqs(q)) == 0
