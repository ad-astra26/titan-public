"""SPEC parity tests — chunk 8J §7.7.

Per `feedback_specs_need_enforcement_automation.md`: every "use X, not
Y" rule needs a test/scanner that fails on violation. This file is the
canonical SPEC enforcement automation for cognitive_worker — drift
between SPEC §1, §9.B, §8.5, [domains.COGNITIVE] TOML and the live
cognitive_worker.py / legacy_core.py implementation surfaces here.

Drift class examples this guards against:
  - SPEC §1 glossary defines cognitive_worker layer=L2 but ModuleSpec
    registers L3 → drift.
  - PLAN §11(3) lists 10 subscribe topics but cognitive_worker
    subscribes to 9 → drift.
  - SPEC v0.2.0 [domains.COGNITIVE] declares 4 epoch constants but Rust
    constants.rs has 3 → drift (caught by phase-c verify, but this test
    locks the value-level expectations from the Python side).

These tests are pure-Python static — no subprocess, no live engines.
Run as part of every commit via pre-commit hook.
"""
from __future__ import annotations

import pytest


# ── §1 Glossary + §9.B Python tree alignment ────────────────────────


class TestSpecGlossaryAndTreeAlignment:
    """Cognitive_worker must exist at the layer + via the registration
    pattern documented in SPEC §1 + §9.B."""

    def test_module_importable_at_documented_path(self):
        """SPEC §1 + §9.B documents:
            entry: titan_plugin/modules/cognitive_worker.py — cognitive_worker_main"""
        from titan_plugin.modules.cognitive_worker import cognitive_worker_main
        assert callable(cognitive_worker_main)

    def test_module_runs_only_under_l0_rust_enabled(self):
        """SPEC §1: 'Active under: microkernel.l0_rust_enabled=true ONLY'.
        Verified at the entry function — under l0_rust=false it should
        return early after MODULE_READY."""
        # Indirect verification: the source contains a flag-gate reference.
        # We can't easily run the function without spawning + bus setup,
        # but a static contains check catches accidental removal.
        import inspect
        from titan_plugin.modules import cognitive_worker
        src = inspect.getsource(cognitive_worker)
        assert "l0_rust_enabled" in src, (
            "cognitive_worker.py lost the l0_rust_enabled flag-gate — "
            "Maker D3 (b) rollback path would silently break."
        )

    def test_canonical_name_in_legacy_core_registration(self):
        """SPEC §9.B documents 'cognitive_worker' as the ModuleSpec name.
        legacy_core.py registers under that exact name."""
        import inspect
        from titan_plugin import legacy_core
        src = inspect.getsource(legacy_core)
        assert 'name="cognitive_worker"' in src, (
            "legacy_core.py registration under name 'cognitive_worker' "
            "missing — would be invisible to guardian + arch_map."
        )


# ── §8.5 trinity wire contract ──────────────────────────────────────


class TestSpec85WireContract:
    """SPEC §8.5: 3 trinity event types × payload.src ∈ {inner, outer}
    → 6 streams. cognitive_worker subscribes to the 3 event types and
    dispatches by src. NOT 6 separate event names."""

    def test_subscribe_topics_use_canonical_3_event_types(self):
        """Per SPEC §8.5: BODY_STATE, MIND_STATE, SPIRIT_STATE — not the
        legacy A.S8 names OUTER_BODY_STATE / OUTER_MIND_STATE /
        OUTER_SPIRIT_STATE (those are l0_rust=false rollback path).
        See PLAN §3.1 driver table."""
        from titan_plugin.modules.cognitive_worker import _COGNITIVE_WORKER_SUBSCRIBE_TOPICS
        from titan_plugin import bus

        topics = set(_COGNITIVE_WORKER_SUBSCRIBE_TOPICS)
        assert bus.BODY_STATE in topics
        assert bus.MIND_STATE in topics
        assert bus.SPIRIT_STATE in topics
        # Negative: legacy A.S8 wire NOT subscribed (Maker D3 b → only
        # the canonical Phase C wire under l0_rust=true).
        assert bus.OUTER_BODY_STATE not in topics, (
            "cognitive_worker subscribed to OUTER_BODY_STATE — that's the "
            "legacy l0_rust=false A.S8 wire. Phase C uses BODY_STATE "
            "src=outer per SPEC §8.5."
        )
        assert bus.OUTER_MIND_STATE not in topics
        assert bus.OUTER_SPIRIT_STATE not in topics

    def test_six_internal_cache_slots_per_g1_symmetry(self):
        """G1 doctrinal symmetry: Trinity = Inner 65D + Outer 65D = 130D.
        cognitive_worker's dispatcher fans BODY_STATE / MIND_STATE /
        SPIRIT_STATE into 6 first-class internal cache slots indexed by
        payload.src — preserving inner↔outer symmetry at the cognitive
        layer.

        Static check: source contains the 6 cache-slot key strings.
        Functional check: dispatcher tests in test_cognitive_worker.py."""
        import inspect
        from titan_plugin.modules import cognitive_worker
        src = inspect.getsource(cognitive_worker)
        for slot in ("_inner_body_state", "_outer_body_state",
                     "_inner_mind_state", "_outer_mind_state",
                     "_inner_spirit_state", "_outer_spirit_state"):
            assert slot in src, (
                f"cognitive_worker.py missing cache slot '{slot}' — G1 "
                f"inner↔outer doctrinal symmetry violated."
            )


# ── SPEC v0.2.0 [domains.COGNITIVE] constants parity ───────────────


class TestSpecV020CognitiveConstants:
    """Lock-in test: SPEC v0.2.0 introduced [domains.COGNITIVE] with 4
    constants. Drift here = chunk 8B regression OR Maker D4 (a)
    architectural override unrecorded in SPEC."""

    def test_constants_exist_in_python_phase_c_module(self):
        from titan_plugin._phase_c_constants import (
            COGNITIVE_EPOCH_MIN_INTERVAL_S,
            COGNITIVE_EPOCH_DEFAULT_INTERVAL_S,
            COGNITIVE_EPOCH_MAX_INTERVAL_S,
            COGNITIVE_PERSIST_EVERY_N_EPOCHS,
        )
        # Trivial — import means generated file has them.

    def test_schumann_body_multiples_locked(self):
        """Per Maker D4 (a) the constants are integer multiples of
        Schumann body period (1.15s) — preserves harmonic structure at
        the cognitive layer."""
        from titan_plugin._phase_c_constants import (
            COGNITIVE_EPOCH_MIN_INTERVAL_S,
            COGNITIVE_EPOCH_DEFAULT_INTERVAL_S,
            COGNITIVE_EPOCH_MAX_INTERVAL_S,
        )
        # Schumann body period = 1/7.83 Hz ≈ 0.1278s; the constants are
        # 9× Schumann body fundamental frequency cycle = 9× 1.15s =
        # ~10.35s. Verify the integer multiples per Maker D4 (a).
        SCHUMANN_BODY_S = 1.15
        EPS = 0.01
        assert abs(COGNITIVE_EPOCH_MIN_INTERVAL_S - (1 * SCHUMANN_BODY_S)) < EPS
        assert abs(COGNITIVE_EPOCH_DEFAULT_INTERVAL_S - (9 * SCHUMANN_BODY_S)) < EPS
        assert abs(COGNITIVE_EPOCH_MAX_INTERVAL_S - (27 * SCHUMANN_BODY_S)) < EPS

    def test_persist_cadence(self):
        from titan_plugin._phase_c_constants import COGNITIVE_PERSIST_EVERY_N_EPOCHS
        assert COGNITIVE_PERSIST_EVERY_N_EPOCHS == 100


# ── PLAN §11 acceptance criteria — static-checkable subset ─────────


class TestPlanAcceptanceStatic:
    """PLAN §11 has 12 acceptance criteria — most need a live T3 boot
    (chunk 8L verification). The static-checkable subset is captured
    here so commits can't introduce regressions before deploy."""

    def test_criterion_3_subscribe_topics_count(self):
        """PLAN §11(3): journalctl `subscribed to 10 bus topics`."""
        from titan_plugin.modules.cognitive_worker import _COGNITIVE_WORKER_SUBSCRIBE_TOPICS
        assert len(_COGNITIVE_WORKER_SUBSCRIBE_TOPICS) == 10

    def test_criterion_10_no_slim_shim_loop_remains(self):
        """PLAN §11(10): grep -r `_spirit_worker_shim_loop` returns empty.
        Mirrors test_spirit_worker_shim.py::test_no_old_shim_loop_remains
        — defense in depth."""
        from titan_plugin.modules import spirit_worker
        assert not hasattr(spirit_worker, "_spirit_worker_shim_loop"), (
            "Slim-shim 4A code resurrected — chunk 8I deletion regressed."
        )
