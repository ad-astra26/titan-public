"""
Tests for General-Purpose Spatial Perception module.

Verifies 30D feature extraction, value bounds, delta tracking,
source detection, performance, and backward compatibility.
"""
import time

import numpy as np
import pytest


@pytest.fixture
def sp():
    from titan_plugin.logic.spatial_perception import SpatialPerception
    return SpatialPerception()


def _random_image(h=256, w=256):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8).astype(np.float64)


def _flat_image(h=256, w=256, color=(0, 0, 0)):
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[:, :, 0] = color[0]
    img[:, :, 1] = color[1]
    img[:, :, 2] = color[2]
    return img


def _gradient_image(h=256, w=256):
    img = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(w):
        img[:, c, :] = c * 255.0 / max(1, w - 1)
    return img


class TestSpatialPerceptionOutput:
    """Verify output structure and value bounds."""

    def test_perceive_returns_30d(self, sp):
        result = sp.perceive(_random_image())
        assert "physical" in result
        assert "pattern" in result
        assert "spatial" in result
        assert "semantic" in result
        assert "journey" in result
        assert "resonance" in result
        assert "flat_30d" in result
        assert len(result["flat_30d"]) == 30

    def test_all_groups_have_5_floats(self, sp):
        result = sp.perceive(_random_image())
        for key in ["physical", "pattern", "spatial", "semantic", "journey", "resonance"]:
            vals = result[key]
            assert len(vals) == 5, f"{key} has {len(vals)} values, expected 5"

    def test_all_values_in_0_1(self, sp):
        result = sp.perceive(_random_image())
        for v in result["flat_30d"]:
            assert 0.0 <= v <= 1.0, f"Value {v} out of [0,1] range"

    def test_flat_30d_is_concatenation(self, sp):
        result = sp.perceive(_random_image())
        expected = (result["physical"] + result["pattern"] + result["spatial"]
                    + result["semantic"] + result["journey"] + result["resonance"])
        assert result["flat_30d"] == expected


class TestFeatureQuality:
    """Verify features produce meaningful differences for different inputs."""

    def test_flat_image_low_entropy_edges(self, sp):
        result = sp.perceive(_flat_image(color=(0, 0, 0)))
        # All-black: low entropy, low edges, high symmetry
        assert result["physical"][0] < 0.1, "Black image should have low color entropy"
        assert result["physical"][1] < 0.1, "Black image should have low edge density"

    def test_gradient_high_freq(self, sp):
        result = sp.perceive(_gradient_image())
        # Horizontal gradient: asymmetric, moderate frequency
        assert result["physical"][2] < 0.7, "Gradient should have lower symmetry"

    def test_random_image_moderate_features(self, sp):
        result = sp.perceive(_random_image())
        # Random noise: high entropy, moderate edges
        assert result["physical"][0] > 0.5, "Random image should have high color entropy"

    def test_white_image_high_symmetry(self, sp):
        result = sp.perceive(_flat_image(color=(255, 255, 255)))
        assert result["physical"][2] >= 0.49, "White image should have high symmetry"


class TestDeltaTracking:
    """Verify frame-to-frame change detection."""

    def test_first_frame_neutral_spatial(self, sp):
        result = sp.perceive(_random_image())
        # First frame: no prev → neutral spatial
        assert result["spatial"] == [0.5, 0.5, 0.0, 0.5, 0.0]

    def test_second_frame_has_delta(self, sp):
        sp.perceive(_random_image(64, 64))
        result = sp.perceive(_random_image(64, 64))
        # Second frame should detect changes
        assert result["pattern"][3] > 0.0, "Delta should be non-zero after second frame"
        assert result["spatial"] != [0.5, 0.5, 0.0, 0.5, 0.0], "Spatial should activate"

    def test_identical_frames_zero_delta(self, sp):
        img = _random_image(64, 64)
        sp.perceive(img)
        result = sp.perceive(img.copy())
        assert result["pattern"][3] == 0.0, "Identical frames should have zero delta"
        assert result["spatial"] == [0.5, 0.5, 0.0, 0.5, 0.0], "No change = neutral spatial"


class TestJourneyTracking:
    """Verify visual experience accumulation."""

    def test_exploration_rate_decreases_with_repeats(self, sp):
        img = _random_image(64, 64)
        sp.perceive(img)
        r1 = sp.perceive(img.copy())
        r2 = sp.perceive(img.copy())
        # Seeing same image repeatedly → exploration rate should decrease
        assert r2["journey"][0] <= r1["journey"][0]

    def test_image_count_increments(self, sp):
        sp.perceive(_random_image(64, 64))
        sp.perceive(_random_image(64, 64))
        result = sp.perceive(_random_image(64, 64))
        assert result["journey"][4] > 0.0, "Image count should increment"


class TestSourceDetection:
    """Verify filename-based source tagging logic."""

    def test_telegram_detected_as_external(self):
        assert "tg_123_456_abc.jpg".startswith("tg_")

    def test_art_detected_as_self(self):
        assert not "flow_field_xyz.png".startswith("tg_")
        assert not "l_system_abc.png".startswith("tg_")


class TestPerformance:
    """Verify perception completes within time budget."""

    def test_256x256_under_100ms(self, sp):
        img = _random_image(256, 256)
        sp.perceive(img)  # warmup
        t0 = time.perf_counter()
        sp.perceive(_random_image(256, 256))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"Took {elapsed_ms:.1f}ms, expected <100ms"


class TestSmallImages:
    """Edge cases with small images."""

    def test_tiny_image_no_crash(self, sp):
        result = sp.perceive(_random_image(8, 8))
        assert len(result["flat_30d"]) == 30

    def test_single_pixel_no_crash(self, sp):
        img = np.array([[[128.0, 64.0, 32.0]]])
        result = sp.perceive(img)
        assert len(result["flat_30d"]) == 30


class TestBackwardCompat:
    """Verify old 5D path still works."""

    def test_digest_image_still_returns_5d(self, tmp_path):
        from PIL import Image
        # Create a test image
        img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        fpath = tmp_path / "test.png"
        img.save(str(fpath))

        from titan_plugin.modules.media_worker import _digest_image
        result = _digest_image(fpath)
        assert result is not None
        assert len(result) == 5
        assert all(0.0 <= v <= 1.0 for v in result)
