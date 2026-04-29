"""
Tests for V4 Time Awareness — UnifiedSpirit (30DT SPIRIT tensor).

Tests tensor composition, GREAT EPOCH advancement, velocity tracking,
STALE detection, enrichment computation, and persistence.
"""
import math
import tempfile
import time
import pytest


# ── GreatEpoch tests ──────────────────────────────────────────────────

class TestGreatEpoch:
    """Tests for the GreatEpoch record."""

    def test_creation(self):
        from titan_plugin.logic.unified_spirit import GreatEpoch
        tensor = [0.5] * 30
        epoch = GreatEpoch(
            epoch_id=1,
            spirit_tensor=tensor,
            velocity=1.0,
            resonance_snapshot={"body": True, "mind": True, "spirit": True},
        )
        assert epoch.epoch_id == 1
        assert len(epoch.spirit_tensor) == 30
        assert epoch.velocity == 1.0
        assert epoch.magnitude > 0
        assert epoch.enrichment_sent is False

    def test_magnitude_computed(self):
        from titan_plugin.logic.unified_spirit import GreatEpoch
        # All at 1.0 → magnitude = sqrt(30 * 1.0) ≈ 5.477
        tensor = [1.0] * 30
        epoch = GreatEpoch(1, tensor, 1.0, {})
        assert epoch.magnitude == pytest.approx(math.sqrt(30), abs=0.01)

    def test_serialization(self):
        from titan_plugin.logic.unified_spirit import GreatEpoch
        tensor = [0.6] * 30
        epoch = GreatEpoch(42, tensor, 1.5, {"body": True})

        data = epoch.to_dict()
        assert data["epoch_id"] == 42
        assert data["velocity"] == 1.5

        epoch2 = GreatEpoch.from_dict(data)
        assert epoch2.epoch_id == 42
        assert epoch2.velocity == pytest.approx(1.5)
        assert len(epoch2.spirit_tensor) == 30


# ── UnifiedSpirit core tests ──────────────────────────────────────────

class TestUnifiedSpirit:
    """Tests for the Unified SPIRIT."""

    def test_init_defaults(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit, SPIRIT_DIMS
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            assert len(spirit.tensor) == SPIRIT_DIMS
            assert spirit.epoch_count == 0
            assert spirit.velocity == 1.0
            assert spirit.is_stale is False
            assert spirit.latest_epoch is None

    def test_update_subconscious(self):
        """Subconscious layer updates Inner Trinity slice (5D body + 15D mind + 45D spirit = 65D)."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            inner_body = [0.1, 0.2, 0.3, 0.4, 0.5]
            inner_mind = [0.05 * i for i in range(15)]      # 15D
            inner_spirit = [0.01 * i for i in range(45)]    # 45D
            spirit.update_subconscious(
                inner_body=inner_body,
                inner_mind=inner_mind,
                inner_spirit=inner_spirit,
            )
            t = spirit.tensor
            assert t[0:5] == inner_body
            assert t[5:20] == inner_mind
            assert t[20:65] == inner_spirit

    def test_update_conscious(self):
        """Conscious layer updates Outer Trinity slice (5D body + 15D mind + 45D spirit = 65D)."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            outer_body = [0.8, 0.7, 0.6, 0.5, 0.4]
            outer_mind = [0.05 * i for i in range(15)]      # 15D
            outer_spirit = [0.02 * i for i in range(45)]    # 45D
            spirit.update_conscious(
                outer_body=outer_body,
                outer_mind=outer_mind,
                outer_spirit=outer_spirit,
            )
            t = spirit.tensor
            assert t[65:70] == outer_body
            assert t[70:85] == outer_mind
            assert t[85:130] == outer_spirit

    def test_inner_outer_tensors(self):
        """inner_tensor and outer_tensor properties return correct 65D slices."""
        from titan_plugin.logic.unified_spirit import (
            UnifiedSpirit, INNER_DIMS, OUTER_DIMS,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            spirit.update_subconscious([0.1]*5, [0.2]*15, [0.3]*45)
            spirit.update_conscious([0.7]*5, [0.8]*15, [0.9]*45)

            assert len(spirit.inner_tensor) == INNER_DIMS == 65
            assert len(spirit.outer_tensor) == OUTER_DIMS == 65
            assert spirit.inner_tensor[0] == 0.1   # inner_body[0]
            assert spirit.inner_tensor[5] == 0.2   # inner_mind[0]
            assert spirit.inner_tensor[20] == 0.3  # inner_spirit[0]
            assert spirit.outer_tensor[0] == 0.7   # outer_body[0]
            assert spirit.outer_tensor[5] == 0.8   # outer_mind[0]
            assert spirit.outer_tensor[20] == 0.9  # outer_spirit[0]

    def test_tensor_padding(self):
        """Short tensors are padded with 0.5."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            spirit.update_subconscious([0.1, 0.2], [0.3], [])
            t = spirit.tensor
            assert t[0:5] == [0.1, 0.2, 0.5, 0.5, 0.5]
            assert t[5:10] == [0.3, 0.5, 0.5, 0.5, 0.5]
            assert t[10:15] == [0.5, 0.5, 0.5, 0.5, 0.5]


# ── GREAT EPOCH advancement ──────────────────────────────────────────

class TestGreatPulseAdvancement:
    """Tests for SPIRIT advancement via GREAT PULSE."""

    def test_advance_creates_epoch(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit, SPIRIT_DIMS
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            spirit.update_subconscious([0.6]*5, [0.5]*15, [0.7]*45)
            spirit.update_conscious([0.4]*5, [0.5]*15, [0.6]*45)

            epoch = spirit.advance({"body": True, "mind": True, "spirit": True})

            assert epoch is not None
            assert epoch.epoch_id == 1
            assert len(epoch.spirit_tensor) == SPIRIT_DIMS == 130
            assert spirit.epoch_count == 1

    def test_advance_increments_epoch_id(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            for i in range(5):
                epoch = spirit.advance({"all": True})
                assert epoch.epoch_id == i + 1
            assert spirit.epoch_count == 5

    def test_advance_captures_tensor_snapshot(self):
        """Each epoch captures the tensor state at that moment."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)

            spirit.update_subconscious([0.3]*5, [0.3]*5, [0.3]*5)
            e1 = spirit.advance({})

            spirit.update_subconscious([0.8]*5, [0.8]*5, [0.8]*5)
            e2 = spirit.advance({})

            # Tensors should be different
            assert e1.spirit_tensor[0] == 0.3
            assert e2.spirit_tensor[0] == 0.8

    def test_advance_cannot_go_backward(self):
        """SPIRIT tensor cannot move backward — epoch IDs always increase."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            spirit.advance({})
            spirit.advance({})
            spirit.advance({})

            # All epochs are stored, IDs increasing
            assert [e.epoch_id for e in spirit._epochs] == [1, 2, 3]

    def test_latest_epoch(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            spirit.advance({})
            spirit.advance({})
            assert spirit.latest_epoch.epoch_id == 2

    def test_get_epoch_by_id(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            spirit.advance({})
            spirit.advance({})
            spirit.advance({})

            e = spirit.get_epoch(2)
            assert e is not None
            assert e.epoch_id == 2

            assert spirit.get_epoch(99) is None


# ── Velocity tracking ─────────────────────────────────────────────────

class TestVelocityTracking:
    """Tests for SPIRIT growth velocity computation."""

    def test_first_epoch_neutral_velocity(self):
        """First epoch has neutral velocity (no history to compare)."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            epoch = spirit.advance({})
            assert epoch.velocity == 1.0

    def test_growing_tensor_high_velocity(self):
        """Tensor magnitude increasing → velocity > 1.0."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)

            # Start small
            spirit.update_subconscious([0.1]*5, [0.1]*5, [0.1]*5)
            spirit.update_conscious([0.1]*5, [0.1]*5, [0.1]*5)
            spirit.advance({})

            # Grow larger
            spirit.update_subconscious([0.8]*5, [0.8]*5, [0.8]*5)
            spirit.update_conscious([0.8]*5, [0.8]*5, [0.8]*5)
            e2 = spirit.advance({})

            assert e2.velocity > 1.0

    def test_shrinking_tensor_low_velocity(self):
        """Tensor magnitude decreasing → velocity < 1.0."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)

            # Start large
            spirit.update_subconscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.update_conscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.advance({})

            # Shrink
            spirit.update_subconscious([0.1]*5, [0.1]*5, [0.1]*5)
            spirit.update_conscious([0.1]*5, [0.1]*5, [0.1]*5)
            e2 = spirit.advance({})

            assert e2.velocity < 1.0

    def test_stable_tensor_neutral_velocity(self):
        """Unchanged tensor → velocity ≈ 1.0."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)

            spirit.update_subconscious([0.5]*5, [0.5]*5, [0.5]*5)
            spirit.update_conscious([0.5]*5, [0.5]*5, [0.5]*5)
            spirit.advance({})
            e2 = spirit.advance({})

            assert e2.velocity == pytest.approx(1.0, abs=0.01)

    def test_velocity_window_limited(self):
        """Velocity only looks back N epochs (velocity_window)."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(
                config={"velocity_window": 3},
                data_dir=tmpdir,
            )

            # Create 5 epochs with same tensor
            for _ in range(5):
                spirit.update_subconscious([0.5]*5, [0.5]*5, [0.5]*5)
                spirit.update_conscious([0.5]*5, [0.5]*5, [0.5]*5)
                spirit.advance({})

            # Velocity should be ≈1.0 (stable over window)
            assert spirit.velocity == pytest.approx(1.0, abs=0.05)


# ── STALE detection ───────────────────────────────────────────────────

class TestStaleDetection:
    """Tests for STALE state detection and FOCUS cascade multiplier."""

    def test_not_stale_initially(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            assert spirit.is_stale is False

    def test_stale_when_shrinking(self):
        """SPIRIT goes STALE when tensor magnitude drops."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(
                config={"stale_threshold": 0.9},
                data_dir=tmpdir,
            )

            # Start large
            spirit.update_subconscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.update_conscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.advance({})

            # Shrink significantly
            spirit.update_subconscious([0.2]*5, [0.2]*5, [0.2]*5)
            spirit.update_conscious([0.2]*5, [0.2]*5, [0.2]*5)
            spirit.advance({})

            assert spirit.is_stale is True

    def test_stale_focus_multiplier_escalates(self):
        """FOCUS cascade multiplier grows with consecutive STALE epochs."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(
                config={"stale_threshold": 0.9},
                data_dir=tmpdir,
            )

            # First epoch (reference)
            spirit.update_subconscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.update_conscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.advance({})

            # Multiple shrinking epochs
            spirit.update_subconscious([0.2]*5, [0.2]*5, [0.2]*5)
            spirit.update_conscious([0.2]*5, [0.2]*5, [0.2]*5)
            spirit.advance({})
            m1 = spirit.stale_focus_multiplier

            spirit.advance({})
            m2 = spirit.stale_focus_multiplier

            assert m2 > m1  # Escalation

    def test_stale_recovery(self):
        """SPIRIT recovers from STALE when growth resumes."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(
                config={"stale_threshold": 0.9},
                data_dir=tmpdir,
            )

            # Large → small (STALE)
            spirit.update_subconscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.update_conscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.advance({})

            spirit.update_subconscious([0.2]*5, [0.2]*5, [0.2]*5)
            spirit.update_conscious([0.2]*5, [0.2]*5, [0.2]*5)
            spirit.advance({})
            assert spirit.is_stale is True

            # Back to large (recovery)
            spirit.update_subconscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.update_conscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.advance({})
            assert spirit.is_stale is False
            assert spirit._consecutive_stale == 0

    def test_focus_multiplier_capped(self):
        """FOCUS cascade multiplier is capped at 3×."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(
                config={"stale_threshold": 0.99},
                data_dir=tmpdir,
            )

            # Force many STALE epochs
            spirit.update_subconscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.update_conscious([0.9]*5, [0.9]*5, [0.9]*5)
            spirit.advance({})

            spirit.update_subconscious([0.1]*5, [0.1]*5, [0.1]*5)
            spirit.update_conscious([0.1]*5, [0.1]*5, [0.1]*5)
            for _ in range(20):
                spirit.advance({})

            assert spirit.stale_focus_multiplier <= 3.0


# ── Enrichment ────────────────────────────────────────────────────────

class TestEnrichment:
    """Tests for GREAT PULSE enrichment rewards."""

    def test_enrichment_after_advance(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            spirit.update_subconscious([0.5]*5, [0.5]*5, [0.5]*5)
            spirit.update_conscious([0.5]*5, [0.5]*5, [0.5]*5)
            spirit.advance({})

            enrichment = spirit.compute_enrichment()
            assert len(enrichment) == 6
            for comp in ["inner_body", "inner_mind", "inner_spirit",
                         "outer_body", "outer_mind", "outer_spirit"]:
                assert comp in enrichment
                assert "reward" in enrichment[comp]
                assert "balance_score" in enrichment[comp]

    def test_balanced_gets_higher_enrichment(self):
        """Components at center (0.5) get higher enrichment than off-center."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            # Inner Body at center, Inner Mind far off
            spirit.update_subconscious([0.5]*5, [1.0]*5, [0.5]*5)
            spirit.update_conscious([0.5]*5, [0.5]*5, [0.5]*5)
            spirit.advance({})

            enrichment = spirit.compute_enrichment()
            # Body (at center) should have higher balance_score than Mind (at 1.0)
            assert enrichment["inner_body"]["balance_score"] > enrichment["inner_mind"]["balance_score"]

    def test_enrichment_marks_sent(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            spirit.advance({})
            spirit.compute_enrichment()
            assert spirit.latest_epoch.enrichment_sent is True

    def test_no_enrichment_without_epochs(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            assert spirit.compute_enrichment() == {}


# ── Persistence ───────────────────────────────────────────────────────

class TestPersistence:
    """Tests for SPIRIT state persistence."""

    def test_save_load_round_trip(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = UnifiedSpirit(data_dir=tmpdir)
            s1.update_subconscious([0.7]*5, [0.6]*5, [0.8]*5)
            s1.update_conscious([0.4]*5, [0.5]*5, [0.3]*5)
            s1.advance({"test": True})
            s1.advance({"test": True})
            s1.save_state()

            s2 = UnifiedSpirit(data_dir=tmpdir)
            assert s2.epoch_count == 2
            assert s2._current_epoch_id == 2
            assert s2.tensor[0] == pytest.approx(0.7)

    def test_auto_save_on_advance(self):
        """advance() auto-saves state."""
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = UnifiedSpirit(data_dir=tmpdir)
            s1.advance({})

            # State file should exist
            assert os.path.exists(os.path.join(tmpdir, "unified_spirit_state.json"))

    def test_stats_structure(self):
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        with tempfile.TemporaryDirectory() as tmpdir:
            spirit = UnifiedSpirit(data_dir=tmpdir)
            spirit.advance({})
            stats = spirit.get_stats()

            assert "epoch_count" in stats
            assert "velocity" in stats
            assert "is_stale" in stats
            assert "tensor_magnitude" in stats
            assert "latest_epoch" in stats
            assert stats["latest_epoch"]["epoch_id"] == 1


# ── Utility tests ─────────────────────────────────────────────────────

class TestUtilities:

    def test_tensor_magnitude(self):
        from titan_plugin.logic.unified_spirit import _tensor_magnitude
        assert _tensor_magnitude([0.0] * 30) == 0.0
        assert _tensor_magnitude([1.0] * 30) == pytest.approx(math.sqrt(30), abs=0.01)

    def test_pad_or_trim(self):
        from titan_plugin.logic.unified_spirit import _pad_or_trim
        assert _pad_or_trim([0.1, 0.2], 5) == [0.1, 0.2, 0.5, 0.5, 0.5]
        assert _pad_or_trim([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 5) == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert _pad_or_trim([], 3) == [0.5, 0.5, 0.5]
