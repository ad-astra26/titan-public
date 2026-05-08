"""
tests/test_phase_c_chunk_8m_observatory_pipeline.py

chunk 8M test suite — closes rFP_phase_c_observatory_data_pipeline.md
acceptance gate items via static / unit / integration coverage.

Sections:
  §1 BUG-STATE-REGISTER-BYTES-NO-GET-ATTR-20260505 — bytes-decode regression
  §2 chunk 8M.1 — ns/neuromod/hormonal ModuleSpec registration assertions
  §3 chunk 8M.3 — cognitive_worker.broadcast_topics matches subscribe topics
  §4 chunk 8M.4 — ShmReaderBank extended readers + cognitive_worker bank init
  §5 chunk 8M.5 — build_coordinator_snapshot shm-fallback
  §6 chunk 8M.6+8M.7 — chi (LifeForceEngine) + MSL init helpers importable
  §7 chunk 8M.8 — memory_worker TOPOLOGY_PUBLISH_INTERVAL_S sanity
  §8 chunk 8M.9 — api_subprocess /v4/* fallback paths exist
"""
from __future__ import annotations

import importlib
import os
import sys
import unittest

import msgpack


# ─────────────────────────────────────────────────────────────────────
# §1 StateRegister bytes-decode (BUG-STATE-REGISTER-BYTES-NO-GET-ATTR-20260505)
# ─────────────────────────────────────────────────────────────────────


class TestStateRegisterBytesDecode(unittest.TestCase):
    """Closes BUG-STATE-REGISTER-BYTES-NO-GET-ATTR-20260505."""

    def setUp(self):
        from titan_plugin.logic.state_register import OuterState
        self.state = OuterState()

    def test_decode_msg_dict_passthrough(self):
        from titan_plugin.logic.state_register import _decode_msg
        d = {"type": "BODY_STATE", "payload": {}}
        self.assertEqual(_decode_msg(d), d)

    def test_decode_msg_bytes_unpack(self):
        from titan_plugin.logic.state_register import _decode_msg
        b = msgpack.packb({"type": "BODY_STATE", "payload": {"values": [0.5] * 5}},
                          use_bin_type=True)
        decoded = _decode_msg(b)
        self.assertEqual(decoded["type"], "BODY_STATE")
        self.assertEqual(decoded["payload"]["values"], [0.5] * 5)

    def test_decode_msg_none(self):
        from titan_plugin.logic.state_register import _decode_msg
        self.assertEqual(_decode_msg(None), {})

    def test_decode_msg_garbage_returns_empty_no_raise(self):
        from titan_plugin.logic.state_register import _decode_msg
        self.assertEqual(_decode_msg(b"\xff\xff\xff"), {})

    def test_decode_payload_returns_raw_for_non_dict(self):
        from titan_plugin.logic.state_register import _decode_payload
        b = msgpack.packb([1, 2, 3], use_bin_type=True)
        self.assertEqual(_decode_payload(b), {"_raw": [1, 2, 3]})

    def test_process_bus_message_handles_bytes_msg(self):
        b = msgpack.packb({"type": "BODY_STATE",
                           "payload": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]}},
                          use_bin_type=True)
        # Pre-fix: AttributeError 'bytes' object has no attribute 'get'
        # Post-fix: silently decoded + dispatched.
        self.state._process_bus_message(b)
        self.assertEqual(self.state.body_tensor[0], 0.1)

    def test_process_bus_message_handles_none(self):
        # Should not raise — defensive handling for blackboard-fallback path.
        self.state._process_bus_message(None)

    def test_process_bus_message_handles_garbage(self):
        # Garbage bytes → empty dict → silent skip.
        self.state._process_bus_message(b"\xff\xff")


# ─────────────────────────────────────────────────────────────────────
# §2 chunk 8M.1 — ModuleSpec registration assertions
# ─────────────────────────────────────────────────────────────────────


class TestChunk8M1ModuleRegistration(unittest.TestCase):
    """Closes rFP §1.2 + §3.1 — 3 missing C-S5 modules now register under
    l0_rust_enabled=true."""

    def test_plugin_register_modules_text_contains_expected_names(self):
        # Static AST-style check: confirm the plugin source declares all 3
        # ModuleSpec names. Avoids constructing a full TitanPlugin (which
        # depends on kernel + bus + heavy state).
        from titan_plugin.core import plugin as plugin_mod
        src = open(plugin_mod.__file__).read()
        for name in ('"ns_module"', '"neuromod_module"', '"hormonal_module"'):
            self.assertIn(name, src,
                          f"chunk 8M.1: {name} not registered in plugin.py")

    def test_plugin_imports_three_worker_main_functions(self):
        from titan_plugin.core import plugin as plugin_mod
        src = open(plugin_mod.__file__).read()
        for sym in ("ns_worker_main", "neuromod_worker_main",
                    "hormonal_worker_main"):
            self.assertIn(sym, src,
                          f"chunk 8M.1: {sym} not imported in plugin.py")

    def test_titan_params_has_shm_ns_and_hormonal_flags(self):
        # chunk 8M.1 added shm_ns_enabled / shm_hormonal_enabled aliases
        # so the workers' flag-gating fires the writers under l0_rust=true.
        params_path = os.path.join(
            os.path.dirname(__file__), "..", "titan_plugin", "titan_params.toml")
        params_path = os.path.normpath(params_path)
        text = open(params_path).read()
        self.assertIn("shm_ns_enabled", text)
        self.assertIn("shm_hormonal_enabled", text)


# ─────────────────────────────────────────────────────────────────────
# §3 chunk 8M.3 — cognitive_worker.broadcast_topics
# ─────────────────────────────────────────────────────────────────────


class TestChunk8M3BroadcastTopics(unittest.TestCase):
    """Closes rFP §2.4 + §3.5 — broker filter equals worker subscribe set."""

    def test_subscribe_topic_constant_is_module_level(self):
        from titan_plugin.modules import cognitive_worker as cw
        self.assertTrue(hasattr(cw, "_COGNITIVE_WORKER_SUBSCRIBE_TOPICS"))
        topics = cw._COGNITIVE_WORKER_SUBSCRIBE_TOPICS
        self.assertIsInstance(topics, list)
        self.assertGreaterEqual(len(topics), 10)

    def test_plugin_imports_subscribe_constant_for_broadcast_filter(self):
        # Static check: plugin.py imports _COGNITIVE_WORKER_SUBSCRIBE_TOPICS
        # alongside cognitive_worker_main, and uses it as broadcast_topics=
        # arg to ModuleSpec — ensures broker filter matches worker subscribe.
        from titan_plugin.core import plugin as plugin_mod
        src = open(plugin_mod.__file__).read()
        self.assertIn("_COGNITIVE_WORKER_SUBSCRIBE_TOPICS", src)
        self.assertIn("broadcast_topics=_COGNITIVE_WORKER_SUBSCRIBE_TOPICS", src)


# ─────────────────────────────────────────────────────────────────────
# §4 chunk 8M.4 — ShmReaderBank extensions + cognitive_worker bank init
# ─────────────────────────────────────────────────────────────────────


class TestChunk8M4ShmReaderBank(unittest.TestCase):
    """Closes rFP §2.1 + §3.3 — SPEC §1096 read-back layer."""

    def test_bank_has_all_new_readers(self):
        from titan_plugin.api.shm_reader_bank import ShmReaderBank
        bank = ShmReaderBank(titan_id="T_test")
        for method in (
            "read_topology_30d",
            "read_hormonal",
            "read_inner_body_5d",
            "read_inner_mind_15d",
            "read_outer_body_5d",
            "read_outer_mind_15d",
            "read_outer_spirit_45d",
        ):
            self.assertTrue(hasattr(bank, method),
                            f"chunk 8M.4: missing reader {method}")

    def test_availability_report_lists_extended_set(self):
        from titan_plugin.api.shm_reader_bank import ShmReaderBank
        bank = ShmReaderBank(titan_id="T_test")
        report = bank.availability_report()
        # 8 original + 7 new = 15 entries
        for slot in (
            "trinity", "neuromod", "epoch", "inner_spirit_45d",
            "sphere_clocks", "chi", "titanvm_registers", "identity",
            "topology_30d", "hormonal",
            "inner_body_5d", "inner_mind_15d",
            "outer_body_5d", "outer_mind_15d", "outer_spirit_45d",
        ):
            self.assertIn(slot, report,
                          f"chunk 8M.4: availability_report missing {slot}")

    def test_readers_return_none_when_shm_absent(self):
        # Worktree env has no /dev/shm/titan_T_test/ — every read should
        # return None gracefully (never raise).
        from titan_plugin.api.shm_reader_bank import ShmReaderBank
        bank = ShmReaderBank(titan_id="T_nonexistent_chunk_8m_test")
        for method in (
            "read_topology_30d", "read_hormonal",
            "read_inner_body_5d", "read_inner_mind_15d",
            "read_outer_body_5d", "read_outer_mind_15d", "read_outer_spirit_45d",
        ):
            try:
                result = getattr(bank, method)()
            except Exception as e:
                self.fail(f"chunk 8M.4: {method} raised on missing shm: {e}")
            self.assertIsNone(result,
                              f"chunk 8M.4: {method} should return None when "
                              f"shm slot absent (got {result!r})")

    def test_cognitive_worker_init_helper_returns_bank(self):
        from titan_plugin.modules.cognitive_worker import _init_shm_reader_bank
        bank = _init_shm_reader_bank("T_chunk_8m_test")
        self.assertIsNotNone(bank)
        # Should be a ShmReaderBank instance
        from titan_plugin.api.shm_reader_bank import ShmReaderBank
        self.assertIsInstance(bank, ShmReaderBank)


# ─────────────────────────────────────────────────────────────────────
# §5 chunk 8M.5 — build_coordinator_snapshot shm-fallback
# ─────────────────────────────────────────────────────────────────────


class _StubCoord:
    """Coordinator stub with the chunk 8M.4 snapshot attrs populated."""

    def __init__(self):
        self._sphere_clocks_snapshot = {
            "clocks": {"inner_body": {"phase": 0.5, "radius": 1.0}},
            "age_seconds": 1.0, "seq": 42}
        self._chi_snapshot = {"total": 0.7, "spirit": 0.5,
                              "mind": 0.6, "body": 0.65,
                              "age_seconds": 0.8}
        self._self_162d_snapshot = {
            "full_130dt": [0.5] * 130,
            "full_30d_topology": [0.0] * 30,
            "journey": {"curvature": 0.1, "density": 0.2},
            "age_seconds": 1.0}
        self._hormonal_snapshot = {"hormones": {
            "cortisol": {"level": 0.3, "target": 0.5,
                         "acceleration": 0.0, "decay": 0.99}}}
        self._titanvm_snapshot = {
            "programs": {"REFLEX": {"urgency": 0.4, "fire_count": 10,
                                    "total_updates": 100, "last_loss": 0.05}}}
        self._topology_snapshot = {
            "values": [0.0] * 30,
            "parts": {"head": {"coherence": 0.6,
                               "magnitude": 0.5,
                               "velocity": 0.0,
                               "direction": 0.0,
                               "polarity": 1.0}},
            "age_seconds": 1.0}
        self._inner_spirit_45d_snapshot = None
        self.topology = None
        self.dreaming = None
        self.inner = None
        self._meta_engine = None
        self._meta_service = None
        self.nervous_system = None

    def get_stats(self):
        return {"epoch": 100, "inner_state": "awake"}


class TestChunk8M5SnapshotShmFallback(unittest.TestCase):
    """Closes rFP §2.8 Gap H + §3.4."""

    def test_coordinator_snapshot_includes_shm_keys(self):
        from titan_plugin.modules.spirit_loop import build_coordinator_snapshot
        snap = build_coordinator_snapshot({"coordinator": _StubCoord()})
        self.assertIsNotNone(snap)
        for key in ("sphere_clocks", "chi", "unified_spirit", "self_162d",
                    "hormonal", "titanvm_registers", "topology"):
            self.assertIn(key, snap,
                          f"chunk 8M.5: snapshot missing {key} from shm fallback")
        # Chi from shm wins over absent life_force_engine
        self.assertEqual(snap["chi"]["total"], 0.7)

    def test_topology_block_enriches_from_shm_when_observables_empty(self):
        from titan_plugin.modules.spirit_loop import build_coordinator_snapshot
        snap = build_coordinator_snapshot({"coordinator": _StubCoord()})
        topo = snap.get("topology") or {}
        self.assertEqual(topo.get("observables_30d"), [0.0] * 30)
        self.assertIn("head", topo.get("observables_dict", {}))

    def test_trinity_snapshot_uses_shm_when_engines_none(self):
        from titan_plugin.modules.spirit_loop import build_trinity_snapshot
        snap = build_trinity_snapshot({
            "coordinator": _StubCoord(),
            "body_state": {"values": [0.5] * 5},
            "mind_state": {"values": [0.5] * 5},
            "consciousness": None,
        }, {})
        self.assertIn("sphere_clock", snap)
        self.assertIn("unified_spirit", snap)
        self.assertEqual(snap["self_162d"]["journey"]["curvature"], 0.1)


# ─────────────────────────────────────────────────────────────────────
# §6 chunk 8M.6+8M.7 — chi + MSL init helpers importable
# ─────────────────────────────────────────────────────────────────────


class TestChunk8M67EngineInitImportable(unittest.TestCase):
    """Closes rFP §3.6 + §3.7 — engines importable for cognitive_worker init."""

    def test_life_force_engine_importable(self):
        from titan_plugin.logic.life_force import LifeForceEngine
        engine = LifeForceEngine()
        self.assertEqual(engine._metabolic_drain, 0.0)

    def test_msl_layer_importable(self):
        from titan_plugin.logic.msl import MultisensorySynthesisLayer
        # Light import-only check — full init requires config
        self.assertTrue(callable(MultisensorySynthesisLayer))

    def test_cognitive_worker_init_returns_life_force_and_msl_keys(self):
        # Static check: init function returns dict with both keys.
        # Full execution depends on heavy imports; check source contract.
        from titan_plugin.modules import cognitive_worker as cw
        src = open(cw.__file__).read()
        self.assertIn('"life_force_engine": life_force_engine', src)
        self.assertIn('"msl": msl', src)


# ─────────────────────────────────────────────────────────────────────
# §7 chunk 8M.8 — memory_worker publish cadence
# ─────────────────────────────────────────────────────────────────────


class TestChunk8M8MemoryCadence(unittest.TestCase):
    """Closes rFP §2.5 + §3.8 — cadence ≤10s for acceptance gate item #10."""

    def test_topology_publish_interval_at_or_below_10s(self):
        from titan_plugin.modules import memory_worker as mw_mod
        src = open(mw_mod.__file__).read()
        # Find the canonical assignment line.
        import re
        m = re.search(r"TOPOLOGY_PUBLISH_INTERVAL_S\s*=\s*([\d.]+)", src)
        self.assertIsNotNone(m, "TOPOLOGY_PUBLISH_INTERVAL_S not found")
        self.assertLessEqual(float(m.group(1)), 10.0,
                             f"chunk 8M.8: cadence too slow ({m.group(1)})")


# ─────────────────────────────────────────────────────────────────────
# §8 chunk 8M.9 — api_subprocess fallback paths exist
# ─────────────────────────────────────────────────────────────────────


class TestChunk8M9ApiFallbacks(unittest.TestCase):
    """Closes rFP §3.9 — defense-in-depth shm fallback at endpoint layer."""

    def test_dashboard_endpoints_call_shm_reader_bank(self):
        from titan_plugin.api import dashboard as dashboard_mod
        src = open(dashboard_mod.__file__).read()
        # /v4/sphere-clocks falls back to titan_state.shm.read_sphere_clocks
        self.assertIn("titan_state.shm.read_sphere_clocks", src)
        # /v4/chi falls back to titan_state.shm.read_chi
        self.assertIn("titan_state.shm.read_chi", src)
        # /v4/inner-trinity enriches from multiple shm readers
        self.assertIn("titan_state.shm.read_trinity", src)
        self.assertIn("titan_state.shm.read_hormonal", src)
        self.assertIn("titan_state.shm.read_titanvm_registers", src)
        self.assertIn("titan_state.shm.read_neuromod", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
