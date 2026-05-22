"""Boot order tests — ModuleSpec registration in plugin.py (TitanHCL).

PLAN §11 acceptance criterion #1 + SPEC §10.A:
  - cognitive_worker registers AFTER body, mind, spirit so guardian's
    autostart sequence boots them first (cognitive_worker reads the
    trinity tensors body/mind/spirit publish).
  - cognitive_worker registers BEFORE social_graph_writer + consciousness_writer
    (those consume cognitive_worker's *_UPDATED snapshot publishers).

Phase C: TitanHCL (core/plugin.py) is the SOLE boot path. The legacy
TitanCore monolith (legacy_core.py) was retired 2026-05-21 (D-SPEC-106);
the old parallel-parity helpers that statically inspected TitanCore are
gone. These tests build a real TitanHCL(kernel) and call
_register_modules() (no subprocess spawn) to capture the registration
sequence from guardian._modules (insertion-ordered dict).

cognitive_worker registration remains gated on microkernel.l0_rust_enabled
(plugin.py) — the production fleet runs l0_rust_enabled=true — so the
fixture forces the flag on. Per PLAN §7.5.
"""
from __future__ import annotations

import pytest

from titan_hcl.core.kernel import TitanKernel
from titan_hcl.core.plugin import TitanHCL


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """Real TitanHCL over a limbo TitanKernel (bogus wallet → no chain).

    _register_modules only builds ModuleSpec entries; it does not spawn
    subprocesses (that happens in boot()), so this is fast + isolated.
    """
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    kernel = TitanKernel(str(tmp_path / "nonexistent_wallet.json"))
    plugin = TitanHCL(kernel)
    # cognitive_worker (L2) registration is gated on microkernel.l0_rust_enabled
    # (plugin.py:936). Production fleet runs l0_rust_enabled=true; force it on
    # so the worker appears in the captured sequence.
    plugin._full_config.setdefault("microkernel", {})["l0_rust_enabled"] = True
    yield plugin
    try:
        kernel.registry_bank.close_all()
    except Exception:
        pass
    try:
        kernel.disk_health.stop()
    except Exception:
        pass


def _capture_registration_sequence(plugin) -> list[str]:
    """Run _register_modules + return the ordered ModuleSpec names.

    guardian._modules is an insertion-ordered dict, so its key order is
    the registration order."""
    plugin._register_modules()
    return list(plugin.guardian._modules.keys())


def test_cognitive_worker_is_registered_under_l0_rust_true(plugin):
    """Per chunk 8E + Maker D3 (b): cognitive_worker registers when
    microkernel.l0_rust_enabled=true. Verify it appears in the sequence."""
    sequence = _capture_registration_sequence(plugin)
    assert "cognitive_worker" in sequence, (
        f"cognitive_worker should register under l0_rust=true. "
        f"Sequence was: {sequence}"
    )


def test_cognitive_worker_registered_after_body_mind_spirit(plugin):
    """Boot ordering invariant — guardian autostart processes
    registrations in order, so cognitive_worker (reads body/mind tensors)
    must register AFTER body, mind. (D-SPEC-116: spirit_worker retired — its
    tensors are Rust-owned + read SHM-direct, so it's no longer in the
    registration sequence.)"""
    sequence = _capture_registration_sequence(plugin)

    assert "cognitive_worker" in sequence, (
        f"cognitive_worker not registered. Sequence was: {sequence}"
    )

    cog_idx = sequence.index("cognitive_worker")
    for required_dep in ("body", "mind"):
        assert required_dep in sequence, (
            f"Required dep '{required_dep}' missing from sequence: {sequence}"
        )
        dep_idx = sequence.index(required_dep)
        assert dep_idx < cog_idx, (
            f"cognitive_worker (idx={cog_idx}) must register AFTER "
            f"'{required_dep}' (idx={dep_idx}) — boot ordering violation."
        )
