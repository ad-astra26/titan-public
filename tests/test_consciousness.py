"""Tests for consciousness.py — rFP #1.5 state vector tail meta cleanup.

Covers:
  - _meta_indices helper: 9D / 67D / 132D / unsupported sizes
  - state[7:8] pollution cleanup — for 67D and 132D, writes must go to tail
  - backward compat — 9D legacy path still uses [7:8] (STATE_DIMS canonical)
"""
import pytest


# ── _meta_indices helper ────────────────────────────────────────────

def test_meta_indices_9d_legacy_returns_7_8():
    """9D legacy state has curvature/density canonically at [7:8]
    (STATE_DIMS[7]=curvature, [8]=density). Helper preserves this."""
    from titan_plugin.logic.consciousness import _meta_indices
    assert _meta_indices(9) == (7, 8)


def test_meta_indices_67d_inner_returns_65_66():
    """67D inner-only vector: tail positions."""
    from titan_plugin.logic.consciousness import _meta_indices
    assert _meta_indices(67) == (65, 66)


def test_meta_indices_132d_full_returns_130_131():
    """132D full vector: tail positions match msl.py attention masks."""
    from titan_plugin.logic.consciousness import _meta_indices
    assert _meta_indices(132) == (130, 131)


def test_meta_indices_unsupported_raises():
    """Non-canonical dimension raises — prevents silent dim drift."""
    from titan_plugin.logic.consciousness import _meta_indices
    with pytest.raises(ValueError):
        _meta_indices(50)
    with pytest.raises(ValueError):
        _meta_indices(100)
    with pytest.raises(ValueError):
        _meta_indices(0)


# ── state[7:8] pollution cleanup ────────────────────────────────────

def _make_ctx_stub():
    """Minimal StateVector + ConsciousnessDB mock for isolated tests."""
    from titan_plugin.logic.consciousness import StateVector, EpochRecord

    class _MockDB:
        def __init__(self):
            self._epochs = []

        def get_recent_epochs(self, n=1):
            return self._epochs[-n:] if self._epochs else []

        def get_epoch_count(self):
            return len(self._epochs)

        def push_epoch(self, curvature, density):
            """Simulate a prior epoch with known curvature/density."""
            rec = EpochRecord(
                epoch_id=len(self._epochs) + 1,
                timestamp=0.0,
                block_hash="",
                state_vector=[0.5] * 9,  # dummy
                drift_vector=[0.0] * 9,
                trajectory_vector=[0.0] * 9,
                journey_point=(0.5, 0.5, 0.5),
                curvature=curvature,
                density=density,
                distillation="",
                anchored_tx="",
            )
            self._epochs.append(rec)

    return _MockDB()


def test_snapshot_state_9d_preserves_legacy_meta_at_7_8():
    """For 9D vectors, meta writes land at [7:8] — backward compat unchanged."""
    from titan_plugin.logic.consciousness import StateVector
    sv = StateVector(values=[0.5] * 9)
    db = _make_ctx_stub()
    db.push_epoch(curvature=0.42, density=0.77)

    # Simulate the meta-write block directly (bypasses full snapshot_state
    # which needs a full ConsciousnessLoop — here we isolate the logic).
    from titan_plugin.logic.consciousness import _meta_indices
    ci, di = _meta_indices(len(sv))
    recent = db.get_recent_epochs(1)
    sv[ci] = recent[-1].curvature
    sv[di] = recent[-1].density

    assert ci == 7 and di == 8
    assert sv[7] == pytest.approx(0.42)
    assert sv[8] == pytest.approx(0.77)


def test_snapshot_state_67d_writes_meta_to_tail_not_7_8():
    """For 67D vectors, meta writes MUST go to [65:67], NOT [7:8]."""
    from titan_plugin.logic.consciousness import StateVector, _meta_indices
    sv = StateVector(values=[0.5] * 67)
    db = _make_ctx_stub()
    db.push_epoch(curvature=1.23, density=0.33)

    ci, di = _meta_indices(len(sv))
    recent = db.get_recent_epochs(1)
    sv[ci] = recent[-1].curvature
    sv[di] = recent[-1].density

    assert ci == 65 and di == 66
    # Tail has the meta values
    assert sv[65] == pytest.approx(1.23)
    assert sv[66] == pytest.approx(0.33)
    # CRITICAL: [7:8] unchanged — remains pure felt (inner_mind[2:4])
    assert sv[7] == pytest.approx(0.5)
    assert sv[8] == pytest.approx(0.5)


def test_snapshot_state_132d_writes_meta_to_tail_not_7_8():
    """For 132D vectors, meta writes MUST go to [130:132], NOT [7:8]."""
    from titan_plugin.logic.consciousness import StateVector, _meta_indices
    sv = StateVector(values=[0.5] * 132)
    db = _make_ctx_stub()
    db.push_epoch(curvature=2.71, density=0.11)

    ci, di = _meta_indices(len(sv))
    recent = db.get_recent_epochs(1)
    sv[ci] = recent[-1].curvature
    sv[di] = recent[-1].density

    assert ci == 130 and di == 131
    # Tail has the meta — matches msl.py attention masks
    assert sv[130] == pytest.approx(2.71)
    assert sv[131] == pytest.approx(0.11)
    # CRITICAL: [7:8] unchanged — remains pure felt
    assert sv[7] == pytest.approx(0.5)
    assert sv[8] == pytest.approx(0.5)


def test_first_epoch_initialises_meta_tail_to_zero():
    """Empty DB → meta at tail initialised to 0.0."""
    from titan_plugin.logic.consciousness import StateVector, _meta_indices
    sv = StateVector(values=[0.5] * 132)
    db = _make_ctx_stub()
    # No epochs pushed

    ci, di = _meta_indices(len(sv))
    recent = db.get_recent_epochs(1)
    if recent:
        sv[ci] = recent[-1].curvature
        sv[di] = recent[-1].density
    else:
        sv[ci] = 0.0
        sv[di] = 0.0

    assert sv[130] == 0.0
    assert sv[131] == 0.0
    # Felt dims untouched
    assert sv[7] == pytest.approx(0.5)
    assert sv[8] == pytest.approx(0.5)


def test_run_write_back_writes_meta_to_tail_132d():
    """The run()'s feedback write (state[ci]=curvature, state[di]=density)
    lands at tail positions for 132D state — simulates the post-distill write."""
    from titan_plugin.logic.consciousness import StateVector, _meta_indices
    state = StateVector(values=[0.3] * 132)
    curvature = 1.57  # π/2
    density = 0.5

    ci, di = _meta_indices(len(state))
    state[ci] = curvature
    state[di] = density

    # Tail contains meta
    assert state[130] == pytest.approx(1.57)
    assert state[131] == pytest.approx(0.5)
    # Felt dims [7:8] untouched
    assert state[7] == pytest.approx(0.3)
    assert state[8] == pytest.approx(0.3)
    # All other dims also unchanged
    for i in range(130):
        assert state[i] == pytest.approx(0.3)
