"""Tests for `TitanKernel._prime_balance_from_anchor_state`.

Boot-prime path: read `data/anchor_state.json` (SPEC §11.H entry #19),
extract `sol_balance`, publish SOLANA_BALANCE_UPDATED so /status returns
the last-known balance during the BalancePublisher first_delay_s window
instead of the initial in-memory 0.0 (which the metabolism layer would
classify as HIBERNATION → Pitch UI black-and-white "Metabolic Crisis"
banner during T1 cold-boot warmup).

Per Rule 0 (SPEC has 100% precedence): the prime ONLY reads from the
SPEC-recognized critical-data file (§11.H entry #19). No new SPEC entry
needed for this fix.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from titan_hcl.core.kernel import TitanKernel


@pytest.fixture
def kernel_stub():
    """Construct a minimal TitanKernel just enough to call the helper.

    The helper only touches: self.bus.publish, self._config (for the
    boot_delay log), and the on-disk anchor_state.json path. Everything
    else (network client, soul, ...) is irrelevant to the prime path.
    """
    k = TitanKernel.__new__(TitanKernel)  # bypass __init__
    k.bus = MagicMock()
    k._config = {"microkernel": {"balance_publisher_first_delay_s": 30.0}}
    return k


def _set_anchor_path_temp(monkeypatch, tmp_path, content: dict | None):
    """Patch the kernel module so its `os.path.dirname` chain resolves
    `data/anchor_state.json` into tmp_path. content=None → file missing.
    """
    import titan_hcl.core.kernel as kernel_mod

    if content is not None:
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "data" / "anchor_state.json").write_text(json.dumps(content))

    # Patch __file__ resolution: the helper resolves the path by walking
    # 3 levels up from kernel.py. Easiest: patch os.path.join inside the
    # helper to redirect to our tmp_path. Cleaner: monkeypatch a constant.
    original_abspath = kernel_mod.os.path.abspath
    real_kernel_file = kernel_mod.__file__

    def fake_abspath(p):
        if p == real_kernel_file or p.endswith("/kernel.py"):
            # Return a path whose 3-level-up grandparent is tmp_path.
            return str(tmp_path / "titan_hcl" / "core" / "kernel.py")
        return original_abspath(p)

    monkeypatch.setattr(kernel_mod.os.path, "abspath", fake_abspath)


def test_prime_publishes_balance_from_anchor_state(kernel_stub, monkeypatch, tmp_path):
    """Happy path: anchor_state.json has sol_balance → bus.publish called
    with SOLANA_BALANCE_UPDATED + {"balance": <value>}."""
    _set_anchor_path_temp(monkeypatch, tmp_path, {
        "sol_balance": 0.009465,
        "last_anchor_time": 1778610000.0,
        "anchor_count": 138,
    })
    kernel_stub._prime_balance_from_anchor_state()

    assert kernel_stub.bus.publish.called, "bus.publish should have fired"
    msg = kernel_stub.bus.publish.call_args[0][0]
    assert msg["type"] == "SOLANA_BALANCE_UPDATED"
    assert msg["src"] == "kernel"
    assert msg["dst"] == "all"
    assert msg["payload"] == {"balance": 0.009465}


def test_prime_skips_when_file_missing(kernel_stub, monkeypatch, tmp_path, caplog):
    """File missing → no-op, no exception, no bus.publish."""
    import logging
    _set_anchor_path_temp(monkeypatch, tmp_path, content=None)  # file absent
    with caplog.at_level(logging.INFO, logger="titan_hcl.core.kernel"):
        kernel_stub._prime_balance_from_anchor_state()

    assert not kernel_stub.bus.publish.called
    assert any("does not exist" in rec.message for rec in caplog.records)


def test_prime_raises_on_malformed_json(kernel_stub, monkeypatch, tmp_path):
    """Malformed JSON → raises RuntimeError (caller logs at warning level
    + falls through to first fetch). Validates the error path doesn't
    silently swallow corruption."""
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "anchor_state.json").write_text("not-json{")

    import titan_hcl.core.kernel as kernel_mod
    real_kernel_file = kernel_mod.__file__

    def fake_abspath(p):
        if p == real_kernel_file or p.endswith("/kernel.py"):
            return str(tmp_path / "titan_hcl" / "core" / "kernel.py")
        return kernel_mod.os.path.abspath(p)

    monkeypatch.setattr(kernel_mod.os.path, "abspath", fake_abspath)
    with pytest.raises(RuntimeError, match="unreadable/malformed"):
        kernel_stub._prime_balance_from_anchor_state()
    assert not kernel_stub.bus.publish.called


def test_prime_skips_when_sol_balance_field_missing(kernel_stub, monkeypatch, tmp_path, caplog):
    """anchor_state.json present but sol_balance key absent or non-numeric
    → no-op (graceful degradation; an older format file shouldn't crash boot)."""
    import logging
    _set_anchor_path_temp(monkeypatch, tmp_path, {
        "last_anchor_time": 1778610000.0,
        # sol_balance intentionally missing
    })
    with caplog.at_level(logging.INFO, logger="titan_hcl.core.kernel"):
        kernel_stub._prime_balance_from_anchor_state()

    assert not kernel_stub.bus.publish.called
    assert any("not numeric" in rec.message for rec in caplog.records)


def test_prime_publishes_zero_when_anchor_state_says_zero(kernel_stub, monkeypatch, tmp_path):
    """`sol_balance: 0.0` is a VALID wallet state (a wallet can genuinely
    be drained). Prime publishes the zero anyway; metabolism layer
    handles the HIBERNATION classification correctly when the value
    is real (not boot-time uninitialized)."""
    _set_anchor_path_temp(monkeypatch, tmp_path, {
        "sol_balance": 0.0,
        "last_anchor_time": 1778610000.0,
    })
    kernel_stub._prime_balance_from_anchor_state()

    assert kernel_stub.bus.publish.called
    msg = kernel_stub.bus.publish.call_args[0][0]
    assert msg["payload"] == {"balance": 0.0}
