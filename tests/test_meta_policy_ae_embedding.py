"""Tests for nn_iql_rl audit C1 fix — meta_autoencoder embedding wired
into MetaPolicy input (replaces hash-based 16D stub when AE is trained).

Per titan-docs/audits/COMPONENT_1_MetaPolicy.md and
titan-docs/audits/COMPONENT_14_Meta_Autoencoder.md.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from titan_plugin.logic.meta_autoencoder import MetaAutoencoder
from titan_plugin.logic.meta_reasoning import MetaChainState, MetaReasoningEngine


class _FakeMetaReasoning:
    """Just enough of MetaReasoning to exercise _build_meta_input."""
    def __init__(self, formulate_output=None):
        self._real_build = MetaReasoningEngine._build_meta_input
        self.state = MetaChainState()
        if formulate_output is not None:
            self.state.formulate_output = formulate_output
        else:
            self.state.formulate_output = {"problem_template": "foo-bar"}
        self._ema_state = np.zeros(132, dtype=np.float32)
        self._strategy_history = np.zeros(12, dtype=np.float32)
        # M7/M8/M9 fields + counters referenced in the [74:80] slice
        self._spirit_self_cooldown_max = 100
        self._total_eurekas = 0
        self._total_meta_chains = 0
        self._spirit_self_gate = 50

    def build(self, sv, nm, meta_autoencoder=None, chain_archive=None):
        return self._real_build(
            self, sv, nm, chain_archive, meta_autoencoder=meta_autoencoder)


@pytest.fixture
def trained_ae():
    """Build a minimally-trained meta_autoencoder so is_trained=True."""
    with tempfile.TemporaryDirectory() as td:
        ae = MetaAutoencoder(save_dir=td)
        # Force is_trained=True by setting _training_steps above the threshold
        # (full training loop is out of scope; encode() itself works from init).
        ae._training_steps = 150
        yield ae


@pytest.fixture
def untrained_ae():
    """Autoencoder fresh from __init__ — encode() works, is_trained=False."""
    with tempfile.TemporaryDirectory() as td:
        ae = MetaAutoencoder(save_dir=td)
        ae._training_steps = 0
        yield ae


def _sv_132() -> list:
    """Deterministic 132D state vector for reproducible tests."""
    rng = np.random.default_rng(seed=42)
    return rng.uniform(0.0, 1.0, size=132).astype(np.float32).tolist()


def test_trained_ae_fills_slot_20_36(trained_ae):
    """When AE is trained, the 16D embedding slot uses AE.encode output."""
    mr = _FakeMetaReasoning()
    sv = _sv_132()
    nm = {}
    inp = mr.build(sv, nm, meta_autoencoder=trained_ae)
    # MetaPolicy input has problem embedding at slice [20:36]
    slot = inp[20:36]
    assert len(slot) == 16
    # AE output is tanh-bounded [-1, 1]; we rescale to [0, 1]. So slot values
    # should all lie in [0.0, 1.0] and NOT be the hash sentinel 0.5*16 pattern.
    assert all(0.0 <= v <= 1.0 for v in slot)
    # Expected slot = (ae.encode(sv) + 1) / 2 for each element
    expected = [(v + 1.0) * 0.5 for v in trained_ae.encode(sv)]
    assert slot == pytest.approx(expected, abs=1e-6)


def test_untrained_ae_falls_back_to_hash(untrained_ae):
    """When AE is untrained (is_trained=False), fall back to hash-based stub."""
    mr = _FakeMetaReasoning(formulate_output={"problem_template": "foo-bar"})
    sv = _sv_132()
    nm = {}
    inp = mr.build(sv, nm, meta_autoencoder=untrained_ae)
    slot = inp[20:36]
    assert len(slot) == 16
    # Hash stub: values = (hash((template, i)) % 1000) / 1000.0
    expected = [(hash(("foo-bar", i)) % 1000) / 1000.0 for i in range(16)]
    assert slot == pytest.approx(expected, abs=1e-6)


def test_no_ae_and_no_formulate_output_fills_neutral():
    """No AE and no formulate_output → slot is all 0.5."""
    mr = _FakeMetaReasoning(formulate_output={})
    sv = _sv_132()
    nm = {}
    inp = mr.build(sv, nm, meta_autoencoder=None)
    slot = inp[20:36]
    assert slot == [0.5] * 16


def test_ae_exception_falls_back_gracefully(trained_ae):
    """If AE.encode raises, fall back to hash — policy must never crash."""
    class _BrokenAE:
        is_trained = True
        def encode(self, x):
            raise RuntimeError("synthetic AE failure")

    mr = _FakeMetaReasoning(formulate_output={"problem_template": "baz"})
    sv = _sv_132()
    nm = {}
    inp = mr.build(sv, nm, meta_autoencoder=_BrokenAE())
    slot = inp[20:36]
    # Should have fallen back to hash stub, not a stack trace
    expected = [(hash(("baz", i)) % 1000) / 1000.0 for i in range(16)]
    assert slot == pytest.approx(expected, abs=1e-6)
