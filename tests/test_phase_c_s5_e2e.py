"""C5-9 e2e harness — Python cross-module integration tests.

Per master plan §10.5 chunk C5-9 + my C-S5 PLAN §4.9. The Rust-side e2e
lives at `titan-rust/crates/titan-trinity-daemon/tests/inner_trinity_e2e.rs`
(bus round-trip + slot lifecycle + ground_up cascade). This Python file
covers the cross-module Python pieces:

- All 4 new C-S5 modules import cleanly
- All NEW bus message constants are accessible
- Slot byte-size invariants are consistent across module boundaries
- SPEC v0.1.5 generated constants match the locked values

Subprocess-based daemon-spawn integration (where the kernel actually
spawns the 3 daemon binaries + a stub publisher) is deferred to C-S7
flag-flip prep where the full Rust tree boots end-to-end.
"""
from __future__ import annotations


def test_all_new_python_modules_import_cleanly():
    """C5-5 / C5-6 / C5-7 / C5-8 modules must all be importable —
    catches accidental breakage of the spirit_worker triad split."""
    from titan_plugin.modules import ns_worker
    from titan_plugin.modules import neuromod_worker
    from titan_plugin.modules import hormonal_worker
    from titan_plugin.modules import spirit_worker

    # All 4 main entry functions exist + are callable
    assert callable(ns_worker.ns_worker_main)
    assert callable(neuromod_worker.neuromod_worker_main)
    assert callable(hormonal_worker.hormonal_worker_main)
    assert callable(spirit_worker.spirit_worker_main)
    # The C5-8 thin shim
    assert callable(spirit_worker._spirit_worker_shim_loop)


def test_all_new_bus_constants_present():
    """The 3 new XXX_READY constants for the Python L2 worker triad."""
    from titan_plugin import bus
    assert hasattr(bus, "NS_READY")
    assert hasattr(bus, "NEUROMOD_READY")
    assert hasattr(bus, "HORMONAL_READY")
    assert bus.NS_READY == "NS_READY"
    assert bus.NEUROMOD_READY == "NEUROMOD_READY"
    assert bus.HORMONAL_READY == "HORMONAL_READY"


def test_slot_byte_invariants_consistent_across_modules():
    """Sibling-symmetry per master plan §10 D22 + SPEC §7.1 v0.1.5:
    titanvm_registers + hormonal_state both 200B total (11 × 4 × float32
    + 24B header). neuromod_state is the smaller sibling (48B)."""
    from titan_plugin.modules import ns_worker, hormonal_worker, neuromod_worker

    # NS + Hormonal share identical byte layout (11 rows × 4 fields)
    assert ns_worker.TITANVM_REGISTERS_PAYLOAD_BYTES == 176
    assert hormonal_worker.HORMONAL_STATE_PAYLOAD_BYTES == 176
    # Both have 11-element rosters
    assert ns_worker.NS_PROGRAM_COUNT == hormonal_worker.HORMONE_COUNT == 11
    # Both have 4 fields per row
    assert ns_worker.NS_FIELD_COUNT == hormonal_worker.HORMONE_FIELD_COUNT == 4

    # Neuromod is the smaller sibling — 6 mods × 1 field
    assert neuromod_worker.NEUROMOD_COUNT == 6
    assert neuromod_worker.NEUROMOD_STATE_PAYLOAD_BYTES == 24


def test_ns_and_hormonal_share_canonical_program_order():
    """11 NS programs + 11 hormones MUST share the same canonical order
    per SPEC §7.1 — drift here = silent slot-layout mismatch between
    the sibling slots."""
    from titan_plugin.modules import ns_worker, hormonal_worker
    assert ns_worker.NS_PROGRAM_NAMES == hormonal_worker.HORMONE_NAMES
    # Both must start with the 5 inner programs in canonical order
    expected_inner = ("REFLEX", "FOCUS", "INTUITION", "IMPULSE", "METABOLISM")
    assert ns_worker.NS_PROGRAM_NAMES[:5] == expected_inner
    assert hormonal_worker.HORMONE_NAMES[:5] == expected_inner


def test_module_names_match_spec_9b_titan_hcl_row():
    """SPEC §9.B titan_HCL line 982 lists the new modules — drift =
    supervisor cannot route bus traffic to them."""
    from titan_plugin.modules import ns_worker, neuromod_worker, hormonal_worker
    assert ns_worker.MODULE_NAME == "ns_module"
    assert neuromod_worker.MODULE_NAME == "neuromod_module"
    assert hormonal_worker.MODULE_NAME == "hormonal_module"


def test_spec_v0_1_5_generated_constants_match():
    """SPEC v0.1.5 PATCH locked HORMONAL_STATE_SCHEMA_VERSION + canonical
    ADOPTION vectors. The auto-generated constants module must reflect this."""
    from titan_plugin._phase_c_constants import (
        HORMONAL_STATE_SCHEMA_VERSION,
        TITANVM_REGISTERS_SCHEMA_VERSION,
        NEUROMOD_SCHEMA_VERSION,
        INNER_BODY_5D_SCHEMA_VERSION,
        INNER_MIND_15D_SCHEMA_VERSION,
        INNER_SPIRIT_45D_SCHEMA_VERSION,
    )
    # All slot schema versions = 1 at SPEC v0.1.x (per SPEC §3.1 D05)
    assert HORMONAL_STATE_SCHEMA_VERSION == 1
    assert TITANVM_REGISTERS_SCHEMA_VERSION == 1
    assert NEUROMOD_SCHEMA_VERSION == 1
    assert INNER_BODY_5D_SCHEMA_VERSION == 1
    assert INNER_MIND_15D_SCHEMA_VERSION == 1
    assert INNER_SPIRIT_45D_SCHEMA_VERSION == 1


def test_state_registry_specs_for_all_3_python_l2_slots():
    """state_registry.py must expose RegistrySpec for all 3 Python-managed
    slots in the C-S5 triad."""
    from titan_plugin.core.state_registry import HORMONAL_STATE, NEUROMOD_STATE
    import numpy as np

    # HORMONAL_STATE is the new C-S5 v0.1.4 addition
    assert HORMONAL_STATE.name == "hormonal_state"
    assert HORMONAL_STATE.shape == (11, 4)
    assert HORMONAL_STATE.dtype == np.dtype("<f4")
    assert HORMONAL_STATE.payload_bytes == 176

    # NEUROMOD_STATE existed pre-C-S5 but neuromod_worker now owns it
    assert NEUROMOD_STATE.name == "neuromod_state"
    assert NEUROMOD_STATE.shape == (6,)
    assert NEUROMOD_STATE.dtype == np.dtype("<f4")
    assert NEUROMOD_STATE.payload_bytes == 24


def test_spirit_worker_shim_lookup_uses_l0_rust_enabled_flag():
    """C5-8 flag-gate: only `microkernel.l0_rust_enabled` triggers shim mode.
    All 3 sibling Python L2 workers each have their OWN feature flag
    (`microkernel.shm_ns_enabled`, `_neuromod_enabled`, `_hormonal_enabled`)
    independently of the shim flag — drift here = workers + shim out of sync."""
    from titan_plugin.modules import (
        ns_worker, neuromod_worker, hormonal_worker, spirit_worker,
    )
    import inspect

    # Confirm the shim's flag check uses microkernel.l0_rust_enabled
    src = inspect.getsource(spirit_worker.spirit_worker_main)
    assert "l0_rust_enabled" in src
    assert "_spirit_worker_shim_loop" in src

    # Confirm each worker uses its own slot feature flag
    for module, expected_flag in [
        (ns_worker, "shm_ns_enabled"),
        (neuromod_worker, "shm_neuromod_enabled"),
        (hormonal_worker, "shm_hormonal_enabled"),
    ]:
        src = inspect.getsource(module)
        assert expected_flag in src, (
            f"{module.__name__} must reference its own flag {expected_flag}"
        )


def test_canonical_adoption_vectors_loaded_and_locked():
    """C-S5 SPEC v0.1.5 D-SPEC-31 locked the canonical ADOPTION_REQUEST +
    ACK msgpack vectors. vectors.json now has them — verify they decode
    to the canonical inputs."""
    import json
    from pathlib import Path
    import msgpack

    here = Path(__file__).parent
    vectors_path = here / "parity" / "vectors.json"
    data = json.loads(vectors_path.read_text())

    adoption = data["adoption_payload"]["canonical_v1"]
    assert "request" in adoption
    assert "ack_accepted" in adoption

    # Request: decode the locked hex + verify it matches the canonical input
    req_hex = adoption["request"]["msgpack_hex"]
    req_bytes = bytes.fromhex(req_hex)
    assert len(req_bytes) == adoption["request"]["msgpack_bytes"] == 60
    decoded_req = msgpack.unpackb(req_bytes, raw=False)
    expected_req = adoption["request"]["input"]
    assert decoded_req == expected_req

    # ACK: same structure
    ack_hex = adoption["ack_accepted"]["msgpack_hex"]
    ack_bytes = bytes.fromhex(ack_hex)
    assert len(ack_bytes) == adoption["ack_accepted"]["msgpack_bytes"] == 61
    decoded_ack = msgpack.unpackb(ack_bytes, raw=False)
    expected_ack = adoption["ack_accepted"]["input"]
    assert decoded_ack == expected_ack
