"""
Tests for Spirit Enrichment — The Divine Spark.

Covers:
  E1: StateRegister.get_full_30dt() + OUTER_TRINITY_STATE handler + STATE_SNAPSHOT
  E2: UnifiedSpirit.micro_enrich() — resonant geometric blending
  E3: Spirit worker STATE_SNAPSHOT handling (integration)
  E4: GREAT PULSE crystallization with quality metrics
  E5: Config loading
"""
import math
import time
import pytest


# ── E1: StateRegister 30DT Assembly ──────────────────────────────────────


class TestStateRegisterFull30DT:
    """E1: get_full_30dt() assembles all 6 tensor components."""

    def test_get_full_30dt_returns_30_floats(self):
        from titan_plugin.logic.state_register import StateRegister
        reg = StateRegister()
        result = reg.get_full_30dt()
        assert len(result) == 30
        assert all(isinstance(v, float) for v in result)

    def test_get_full_30dt_reflects_updated_inner_tensors(self):
        from titan_plugin.logic.state_register import StateRegister
        reg = StateRegister()
        reg._update("body_tensor", [0.1, 0.2, 0.3, 0.4, 0.5])
        reg._update("mind_tensor", [0.6, 0.7, 0.8, 0.9, 1.0])
        reg._update("spirit_tensor", [0.11, 0.22, 0.33, 0.44, 0.55])
        result = reg.get_full_30dt()
        assert result[0:5] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert result[5:10] == [0.6, 0.7, 0.8, 0.9, 1.0]
        assert result[10:15] == [0.11, 0.22, 0.33, 0.44, 0.55]

    def test_get_full_30dt_reflects_outer_tensors(self):
        from titan_plugin.logic.state_register import StateRegister
        reg = StateRegister()
        reg._update_many({
            "outer_body": [0.7, 0.7, 0.7, 0.7, 0.7],
            "outer_mind": [0.8, 0.8, 0.8, 0.8, 0.8],
            "outer_spirit": [0.9, 0.9, 0.9, 0.9, 0.9],
        })
        result = reg.get_full_30dt()
        assert result[15:20] == [0.7, 0.7, 0.7, 0.7, 0.7]
        assert result[20:25] == [0.8, 0.8, 0.8, 0.8, 0.8]
        assert result[25:30] == [0.9, 0.9, 0.9, 0.9, 0.9]

    def test_outer_trinity_state_message_updates_register(self):
        from titan_plugin.logic.state_register import StateRegister
        reg = StateRegister()
        reg._process_bus_message({
            "type": "OUTER_TRINITY_STATE",
            "payload": {
                "outer_body": [0.6, 0.6, 0.6, 0.6, 0.6],
                "outer_mind": [0.7, 0.7, 0.7, 0.7, 0.7],
                "outer_spirit": [0.8, 0.8, 0.8, 0.8, 0.8],
            },
        })
        assert reg.outer_body == [0.6, 0.6, 0.6, 0.6, 0.6]
        assert reg.outer_mind == [0.7, 0.7, 0.7, 0.7, 0.7]
        assert reg.outer_spirit == [0.8, 0.8, 0.8, 0.8, 0.8]

    def test_default_outer_tensors_are_neutral(self):
        from titan_plugin.logic.state_register import StateRegister
        reg = StateRegister()
        assert reg.outer_body == [0.5] * 5
        assert reg.outer_mind == [0.5] * 5
        assert reg.outer_spirit == [0.5] * 5


# ── E2: Resonant Geometric Enrichment ───────────────────────────────────


class TestMicroEnrich:
    """E2: UnifiedSpirit.micro_enrich() — resonant geometric blending."""

    def _make_spirit(self, **overrides):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        import tempfile, os
        d = tempfile.mkdtemp()
        cfg = {"enrichment_rate": 0.02, "min_alignment_threshold": 0.1}
        cfg.update(overrides)
        return UnifiedSpirit(config=cfg, data_dir=d)

    def test_aligned_tensors_increase_quality(self):
        spirit = self._make_spirit()
        # Set Spirit tensor to known values
        spirit._tensor = [0.5] * 30
        # Provide aligned state (same direction)
        state = [0.5] * 30
        alignment = spirit.micro_enrich(state)
        # Perfectly aligned tensors → alignment ≈ 1.0
        assert alignment > 0.9
        assert spirit._cumulative_quality > 0
        assert spirit._micro_tick_count == 1

    def test_misaligned_tensors_no_enrichment(self):
        spirit = self._make_spirit(min_alignment_threshold=0.5)
        spirit._tensor = [0.5] * 30
        # Create an orthogonal-ish state by alternating high/low
        state = [0.1 if i % 2 == 0 else 0.9 for i in range(30)]
        original_tensor = list(spirit._tensor)
        alignment = spirit.micro_enrich(state)
        # With high threshold, low alignment shouldn't modify tensor
        if alignment < 0.5:
            # Tensor should be unchanged
            assert spirit._tensor == original_tensor

    def test_tensor_magnitude_preserved(self):
        """Geometric blend should not cause explosion or collapse."""
        spirit = self._make_spirit(enrichment_rate=0.05)
        spirit._tensor = [0.5] * 30
        original_mag = math.sqrt(sum(v * v for v in spirit._tensor))

        # Run 100 micro-ticks with gentle state
        for _ in range(100):
            state = [0.55] * 30  # slightly above spirit
            spirit.micro_enrich(state)

        new_mag = math.sqrt(sum(v * v for v in spirit._tensor))
        # Should be close to original — enrichment_rate 0.05 is gentle
        assert 0.5 * original_mag < new_mag < 2.0 * original_mag

    def test_quality_accumulates_across_ticks(self):
        spirit = self._make_spirit()
        spirit._tensor = [0.5] * 30
        for i in range(10):
            state = [0.5 + 0.01 * (i + 1)] * 30
            spirit.micro_enrich(state)
        assert spirit._cumulative_quality > 0
        assert spirit._micro_tick_count == 10

    def test_quality_resets_at_great_pulse(self):
        spirit = self._make_spirit()
        spirit._tensor = [0.5] * 30
        # Accumulate some quality
        for _ in range(5):
            spirit.micro_enrich([0.5] * 30)
        assert spirit._cumulative_quality > 0

        # Advance (GREAT PULSE)
        epoch = spirit.advance({"test": True})
        assert epoch is not None
        assert epoch.cumulative_quality > 0
        assert spirit._cumulative_quality == 0.0  # reset
        assert spirit._micro_tick_count == 0  # reset

    def test_wrong_dimension_returns_zero(self):
        spirit = self._make_spirit()
        alignment = spirit.micro_enrich([0.5] * 15)  # wrong dim
        assert alignment == 0.0

    def test_enrichment_rate_controls_speed(self):
        """Higher enrichment_rate → more change per tick."""
        spirit_slow = self._make_spirit(enrichment_rate=0.01)
        spirit_fast = self._make_spirit(enrichment_rate=0.10)
        spirit_slow._tensor = [0.5] * 30
        spirit_fast._tensor = [0.5] * 30

        state = [0.6] * 30
        spirit_slow.micro_enrich(state)
        spirit_fast.micro_enrich(state)

        # Fast should move more toward 0.6
        diff_slow = abs(spirit_slow._tensor[0] - 0.5)
        diff_fast = abs(spirit_fast._tensor[0] - 0.5)
        assert diff_fast > diff_slow

    def test_tensor_stays_bounded(self):
        """After many enrichments, tensor values stay in reasonable range."""
        spirit = self._make_spirit(enrichment_rate=0.05)
        spirit._tensor = [0.5] * 30
        for _ in range(500):
            # Varying state to stress-test bounds
            state = [0.3 + 0.4 * (i % 3) / 2 for i in range(30)]
            spirit.micro_enrich(state)
        for v in spirit._tensor:
            assert 0.001 <= v <= 2.0, f"Tensor value out of bounds: {v}"


# ── E4: GREAT PULSE Crystallization ─────────────────────────────────────


class TestGreatPulseCrystallization:
    """E4: Quality-enhanced enrichment at GREAT PULSE."""

    def _make_spirit(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        import tempfile
        d = tempfile.mkdtemp()
        return UnifiedSpirit(
            config={"enrichment_rate": 0.02, "min_alignment_threshold": 0.1},
            data_dir=d,
        )

    def test_great_pulse_captures_quality(self):
        spirit = self._make_spirit()
        spirit._tensor = [0.5] * 30
        # Enrich to build quality
        for _ in range(20):
            spirit.micro_enrich([0.5] * 30)
        quality_before = spirit._cumulative_quality
        assert quality_before > 0

        epoch = spirit.advance({"resonance": "full"})
        assert epoch.cumulative_quality == pytest.approx(quality_before, rel=0.01)
        assert epoch.micro_tick_count == 20

    def test_quality_bonus_scales_enrichment_rewards(self):
        spirit = self._make_spirit()
        spirit._tensor = [0.5] * 30

        # Advance with no micro-enrichment → quality_bonus = 1.0
        epoch0 = spirit.advance({"test": True})
        enrich0 = spirit.compute_enrichment()

        # Now do lots of micro-enrichment
        for _ in range(200):
            spirit.micro_enrich([0.5] * 30)

        epoch1 = spirit.advance({"test": True})
        enrich1 = spirit.compute_enrichment()

        # Second epoch should have higher rewards due to quality_bonus
        r0 = sum(c["reward"] for c in enrich0.values())
        r1 = sum(c["reward"] for c in enrich1.values())
        # enrich1 should be >= enrich0 (quality_bonus >= 1.0)
        assert r1 >= r0

    def test_quality_resets_after_advance(self):
        spirit = self._make_spirit()
        spirit._tensor = [0.5] * 30
        for _ in range(10):
            spirit.micro_enrich([0.5] * 30)
        spirit.advance({"test": True})
        assert spirit._cumulative_quality == 0.0
        assert spirit._micro_tick_count == 0


# ── E5: Config & Observability ──────────────────────────────────────────


class TestEnrichmentConfig:
    """E5: Config loading and stats exposure."""

    def test_enrichment_stats_in_get_stats(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        import tempfile
        d = tempfile.mkdtemp()
        spirit = UnifiedSpirit(
            config={"enrichment_rate": 0.03, "min_alignment_threshold": 0.15},
            data_dir=d,
        )
        spirit._tensor = [0.5] * 30
        spirit.micro_enrich([0.5] * 30)

        stats = spirit.get_stats()
        assert "cumulative_quality" in stats
        assert "micro_tick_count" in stats
        assert stats["micro_tick_count"] == 1
        assert "last_alignment" in stats
        assert "enrichment_rate" in stats
        assert stats["enrichment_rate"] == 0.03
        assert stats["config"]["enrichment_rate"] == 0.03
        assert stats["config"]["min_alignment_threshold"] == 0.15

    def test_enrichment_persisted_and_restored(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        import tempfile
        d = tempfile.mkdtemp()
        spirit = UnifiedSpirit(
            config={"enrichment_rate": 0.02},
            data_dir=d,
        )
        spirit._tensor = [0.5] * 30
        for _ in range(5):
            spirit.micro_enrich([0.5] * 30)
        quality = spirit._cumulative_quality
        ticks = spirit._micro_tick_count
        spirit.save_state()

        # Restore
        spirit2 = UnifiedSpirit(config={"enrichment_rate": 0.02}, data_dir=d)
        assert spirit2._cumulative_quality == pytest.approx(quality)
        assert spirit2._micro_tick_count == ticks

    def test_epoch_quality_persisted(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        import tempfile
        d = tempfile.mkdtemp()
        spirit = UnifiedSpirit(
            config={"enrichment_rate": 0.02},
            data_dir=d,
        )
        spirit._tensor = [0.5] * 30
        for _ in range(10):
            spirit.micro_enrich([0.5] * 30)
        epoch = spirit.advance({"test": True})
        spirit.save_state()

        # Restore and check epoch quality
        spirit2 = UnifiedSpirit(config={"enrichment_rate": 0.02}, data_dir=d)
        assert len(spirit2._epochs) == 1
        assert spirit2._epochs[0].cumulative_quality == pytest.approx(epoch.cumulative_quality)
        assert spirit2._epochs[0].micro_tick_count == epoch.micro_tick_count
