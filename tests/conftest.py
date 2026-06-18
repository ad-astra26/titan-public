"""Project-wide pytest configuration.

Auto-disables the IMW (writer-service) production-default config path
for every test. Without this, any test that touches code which calls
`get_client()` / `IMWConfig.from_titan_config()` (e.g. ReasoningEngine,
NeuralNervousSystem, language_worker, spirit_worker) ends up trying to
connect to the IMW Unix socket — which doesn't exist in a fresh test
process — and times out after 5s with `WriterError: IMW client loop
thread failed to start`.

Scope of patch (intentionally narrow):
  * `IMWConfig.from_titan_config()` returns a disabled config.

NOT patched:
  * `IMWConfig.from_dict(...)` — used by IMW tests (test_imw_chaos,
    test_imw_end_to_end, test_observatory_canonical_e2e) that spawn
    their own daemon.
  * `IMWConfig.from_titan_config_section(...)` — used by
    test_observatory_singleton + test_universal_sqlite_writer_bundle
    with explicit config dicts.

Net effect: production callers go through `from_titan_config` and now
get a disabled client (safe no-op writes); IMW tests bypass via the
other constructors and exercise the real client against their own
spawned daemon.

Lifted from the local fixture in test_titan_vm_v2_supervision.py which
codified the pattern after the same WriterError cascade was hit during
NNS Phase 3 testing.

Closes the missing-test-services bucket of BUG-PYTEST-SUITE-HYGIENE-20260427.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_live_shm_config_reads(monkeypatch):
    """Force tests onto the config bootstrap path (RFP_config_as_shm_state §7.C).

    Phase C / C.4 flipped ``params._config_shm_enabled()`` to default-ON: on a
    live box ``get_params``/``load_titan_params`` read the in-kernel daemon's SHM
    slots. But a test process on a box where a Titan is running resolves
    ``titan_id`` to that Titan and would read its LIVE ``/dev/shm/titan_<id>/config``
    slots — non-hermetic (e.g. mainnet T1). Setting ``TITAN_CONFIG_SHM_READ=0``
    routes every test through ``_bootstrap_merge`` (the documented no-daemon path:
    titan_params.toml ⊎ config.toml + secrets), reading repo files, not live SHM.

    Tests that specifically exercise the SHM read path (test_config_phaseb_get_params's
    ``shm`` fixture) re-enable it with an ISOLATED temp ``_shm_root`` — their
    function-scoped ``monkeypatch.setenv(..., "1")`` runs after this autouse and wins."""
    monkeypatch.setenv("TITAN_CONFIG_SHM_READ", "0")
    import titan_hcl.params as _params
    # Drop any SHM reader/root state a prior test may have cached.
    _params._shm_readers.clear()
    _params._shm_root = None
    _params._shm_unavailable = False
    yield


@pytest.fixture(autouse=True)
def _disable_imw_for_tests(monkeypatch):
    from titan_hcl.persistence.config import IMWConfig
    from titan_hcl.persistence import writer_client as _wc

    def _disabled(cls):
        return cls(enabled=False, mode="disabled")

    monkeypatch.setattr(
        IMWConfig, "from_titan_config", classmethod(_disabled)
    )
    _wc.reset_client()
    yield
    _wc.reset_client()
