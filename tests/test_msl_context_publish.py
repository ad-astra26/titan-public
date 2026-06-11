"""Tests for the MSL distilled_context[20] SHM publish/read (OML Phase C piece 2).

A dedicated fixed float32(20) slot lets the agno DECIDE path read Titan's FULL
MSL `distilled_context` O(1) at decision-time (Q4 = full MSL). Pure SHM
round-trip — no cognitive_worker / MSL engine needed.
"""
import numpy as np
import pytest

from titan_hcl.synthesis.outer_meta_policy import (
    MSL_CONTEXT_DIM,
    OUTER_MSL_CONTEXT_STATE_SPEC,
    OUTER_MSL_CONTEXT_STATE_SCHEMA_VERSION,
    msl_context_to_fixed,
    read_msl_context,
)
from titan_hcl.core.state_registry import StateRegistryWriter


def test_spec_is_fixed_20_float32():
    assert MSL_CONTEXT_DIM == 20
    assert OUTER_MSL_CONTEXT_STATE_SPEC.shape == (MSL_CONTEXT_DIM,)
    assert OUTER_MSL_CONTEXT_STATE_SPEC.dtype == np.dtype("float32")
    assert OUTER_MSL_CONTEXT_STATE_SPEC.variable_size is False
    assert OUTER_MSL_CONTEXT_STATE_SCHEMA_VERSION == 1


def test_to_fixed_pads_trims_and_nan_guards():
    # exact 20 → preserved
    v = msl_context_to_fixed(list(np.linspace(-1, 1, 20)))
    assert v.shape == (20,) and v.dtype == np.float32
    # short → zero-padded
    s = msl_context_to_fixed([0.5, -0.5])
    assert s[0] == pytest.approx(0.5) and s[1] == pytest.approx(-0.5)
    assert float(np.abs(s[2:]).sum()) == 0.0
    # over-long → trimmed to 20
    assert msl_context_to_fixed([0.1] * 50).shape == (20,)
    # NaN / inf → 0.0 (guard)
    g = msl_context_to_fixed([float("nan"), float("inf"), float("-inf")] + [0.0] * 17)
    assert g[0] == 0.0 and g[1] == 0.0 and g[2] == 0.0
    # None / empty → zeros
    assert float(np.abs(msl_context_to_fixed(None)).sum()) == 0.0
    assert float(np.abs(msl_context_to_fixed([])).sum()) == 0.0


def test_publish_read_roundtrip(tmp_path):
    # representative distilled_context ∈ [-1,1] (tanh-shaped)
    ctx = np.sin(np.linspace(0.0, 3.0, MSL_CONTEXT_DIM)).astype(np.float32)
    writer = StateRegistryWriter(OUTER_MSL_CONTEXT_STATE_SPEC, tmp_path)
    writer.write(msl_context_to_fixed(ctx))
    got = read_msl_context(tmp_path)
    assert got is not None
    assert got.shape == (MSL_CONTEXT_DIM,) and got.dtype == np.float32
    np.testing.assert_allclose(got, ctx, rtol=1e-5, atol=1e-5)


def test_publish_read_roundtrip_short_context_padded(tmp_path):
    # a short context still publishes a valid 20D vector (zero-padded)
    writer = StateRegistryWriter(OUTER_MSL_CONTEXT_STATE_SPEC, tmp_path)
    writer.write(msl_context_to_fixed([0.3, -0.7, 0.1]))
    got = read_msl_context(tmp_path)
    assert got is not None and got.shape == (MSL_CONTEXT_DIM,)
    assert got[0] == pytest.approx(0.3) and got[2] == pytest.approx(0.1)
    assert float(np.abs(got[3:]).sum()) == 0.0


def test_read_unpublished_slot_returns_none(tmp_path):
    # cold-start: nothing published at this root → None (caller treats as zeros)
    assert read_msl_context(tmp_path / "never_published") is None
