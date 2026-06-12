"""OML Phase C piece 7a — the live 20-D MSL distilled_context feeds the agno
decision feature vector (Q4 full MSL; closes the "_v5 fetched after the decision"
gap). Verifies _outer_policy_decide reads read_msl_context() into OuterFeatures."""
import numpy as np
import pytest

import titan_hcl.synthesis.outer_meta_policy as omp
from titan_hcl.synthesis.outer_meta_policy import (
    MSL_CONTEXT_DIM,
    OuterMetaPolicy,
    _BASE_FEATURE_NAMES,
)

agno_hooks = pytest.importorskip("titan_hcl.modules.agno_hooks")

_N_BASE = len(_BASE_FEATURE_NAMES)  # 8 — MSL block starts here


class _Reader:
    def __init__(self, flat):
        self._flat = flat

    def read(self):
        return self._flat


class _Plugin:
    pass


class _Readout:
    recall_score = 0.4
    skill_utility = 0.2
    engram_ground = 0.1


def _plugin_with_policy():
    p = _Plugin()
    p._outer_policy_reader = _Reader(OuterMetaPolicy().to_flat())
    return p


def test_live_msl_context_flows_into_feature_vector(monkeypatch):
    msl = np.linspace(-1.0, 1.0, MSL_CONTEXT_DIM).astype(np.float32)
    monkeypatch.setattr(omp, "read_msl_context", lambda *a, **k: msl)

    out = agno_hooks._outer_policy_decide(
        _plugin_with_policy(), _Readout(), True, "compute 8 factorial")
    assert out is not None
    _mode, vec, _action = out
    assert len(vec) == omp.OUTER_POLICY_INPUT_DIM == 30
    msl_block = vec[_N_BASE:_N_BASE + MSL_CONTEXT_DIM]
    assert np.allclose(msl_block, msl, atol=1e-5)
    # retrieval-prior dims (last 2) remain 0.0 until piece 7b
    assert vec[-2] == 0.0 and vec[-1] == 0.0


def test_msl_miss_is_zeros(monkeypatch):
    monkeypatch.setattr(omp, "read_msl_context", lambda *a, **k: None)
    out = agno_hooks._outer_policy_decide(
        _plugin_with_policy(), _Readout(), False, "hello")
    assert out is not None
    _mode, vec, _action = out
    msl_block = vec[_N_BASE:_N_BASE + MSL_CONTEXT_DIM]
    assert all(x == 0.0 for x in msl_block)   # cold-start → zeros, no crash


def test_decide_returns_none_without_published_policy():
    # no SHM policy → grounded_route stands (byte-identical, flag-off safety)
    p = _Plugin()
    p._outer_policy_reader = _Reader(None)
    assert agno_hooks._outer_policy_decide(p, _Readout(), True, "x") is None
