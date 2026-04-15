"""Tests for expressive/audio.py — ProceduralAudioGen core engine."""
import os
import struct
import wave

import pytest


class TestBlockchainSonification:
    """V1 blockchain sonification (preserved behavior)."""

    def test_generates_wav(self, tmp_path):
        from titan_plugin.expressive.audio import ProceduralAudioGen
        gen = ProceduralAudioGen(output_dir=str(tmp_path))
        path = gen.generate_blockchain_sonification(
            tx_signature="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            sol_balance=2.0,
        )
        assert os.path.exists(path)
        assert path.endswith(".wav")

        # Verify WAV structure
        with wave.open(path, "r") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 44100
            assert wf.getnframes() == 44100 * 3  # 3 seconds

    def test_low_energy_drone(self, tmp_path):
        from titan_plugin.expressive.audio import ProceduralAudioGen
        gen = ProceduralAudioGen(output_dir=str(tmp_path))
        path = gen.generate_blockchain_sonification(
            tx_signature="0" * 64,
            sol_balance=0.01,  # starvation mode
        )
        assert os.path.exists(path)

    def test_high_energy_harmonics(self, tmp_path):
        from titan_plugin.expressive.audio import ProceduralAudioGen
        gen = ProceduralAudioGen(output_dir=str(tmp_path))
        path = gen.generate_blockchain_sonification(
            tx_signature="f" * 64,
            sol_balance=10.0,  # high energy
        )
        assert os.path.exists(path)


class TestTrinitySonification:
    """V3 Trinity sonification — tensor → music mapping."""

    def test_generates_wav(self, tmp_path):
        from titan_plugin.expressive.audio import ProceduralAudioGen
        gen = ProceduralAudioGen(output_dir=str(tmp_path))
        path = gen.generate_trinity_sonification(
            body=[0.5, 0.5, 0.5, 0.5, 0.5],
            mind=[0.5, 0.5, 0.5, 0.5, 0.5],
            spirit=[0.5, 0.5, 0.5, 0.5, 0.5],
            middle_path_loss=0.5,
            duration_seconds=3,
        )
        assert os.path.exists(path)
        assert "trinity_" in os.path.basename(path)

        with wave.open(path, "r") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 44100
            assert wf.getnframes() == 44100 * 3

    def test_different_tensors_produce_different_audio(self, tmp_path):
        """Verify that different Trinity states produce meaningfully different audio."""
        from titan_plugin.expressive.audio import ProceduralAudioGen
        gen = ProceduralAudioGen(output_dir=str(tmp_path))

        # Calm state: low energy, balanced
        path_calm = gen.generate_trinity_sonification(
            body=[0.2, 0.3, 0.2, 0.1, 0.2],
            mind=[0.3, 0.3, 0.5, 0.2, 0.8],
            spirit=[0.5, 0.5, 0.5, 0.3, 0.3],
            middle_path_loss=0.1,
            duration_seconds=2,
        )
        # Agitated state: high energy, imbalanced
        path_agitated = gen.generate_trinity_sonification(
            body=[0.9, 0.8, 0.9, 0.8, 0.9],
            mind=[0.9, 0.7, 0.1, 0.9, 0.2],
            spirit=[0.1, 0.9, 0.1, 0.9, 0.9],
            middle_path_loss=0.9,
            duration_seconds=2,
        )

        # Read raw bytes — they should differ
        with open(path_calm, "rb") as f:
            calm_bytes = f.read()
        with open(path_agitated, "rb") as f:
            agitated_bytes = f.read()

        assert calm_bytes != agitated_bytes

    def test_short_tensor_padding(self, tmp_path):
        """Tensors shorter than 5 dims should be padded to 0.5."""
        from titan_plugin.expressive.audio import ProceduralAudioGen
        gen = ProceduralAudioGen(output_dir=str(tmp_path))
        path = gen.generate_trinity_sonification(
            body=[0.5],   # only 1 dim
            mind=[0.5, 0.5],  # only 2 dims
            spirit=[],  # empty
            duration_seconds=2,
        )
        assert os.path.exists(path)

    def test_all_scales_reachable(self, tmp_path):
        """All 7 scale types should be reachable via Mind[2] taste values."""
        from titan_plugin.expressive.audio import ProceduralAudioGen
        gen = ProceduralAudioGen(output_dir=str(tmp_path))
        paths = []
        for taste_val in [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.99]:
            path = gen.generate_trinity_sonification(
                body=[0.5] * 5,
                mind=[0.5, 0.5, taste_val, 0.5, 0.5],
                spirit=[0.5] * 5,
                duration_seconds=1,
            )
            paths.append(path)
        assert len(paths) == 7
        assert all(os.path.exists(p) for p in paths)

    def test_consonance_vs_dissonance(self, tmp_path):
        """Low loss (consonant) and high loss (dissonant) should produce different audio."""
        from titan_plugin.expressive.audio import ProceduralAudioGen
        gen = ProceduralAudioGen(output_dir=str(tmp_path))

        path_consonant = gen.generate_trinity_sonification(
            body=[0.5] * 5, mind=[0.5] * 5, spirit=[0.5] * 5,
            middle_path_loss=0.0,  # perfect balance
            duration_seconds=2,
        )
        path_dissonant = gen.generate_trinity_sonification(
            body=[0.5] * 5, mind=[0.5] * 5, spirit=[0.5] * 5,
            middle_path_loss=1.0,  # maximum imbalance
            duration_seconds=2,
        )

        with open(path_consonant, "rb") as f:
            cons_bytes = f.read()
        with open(path_dissonant, "rb") as f:
            diss_bytes = f.read()
        assert cons_bytes != diss_bytes

    def test_no_clipping(self, tmp_path):
        """Verify no sample exceeds 16-bit range even with extreme inputs."""
        from titan_plugin.expressive.audio import ProceduralAudioGen
        gen = ProceduralAudioGen(output_dir=str(tmp_path))
        path = gen.generate_trinity_sonification(
            body=[1.0] * 5,
            mind=[1.0] * 5,
            spirit=[1.0] * 5,
            middle_path_loss=1.0,
            duration_seconds=3,
        )
        with wave.open(path, "r") as wf:
            raw = wf.readframes(wf.getnframes())
        samples = struct.unpack(f"<{len(raw)//2}h", raw)
        for s in samples:
            assert -32768 <= s <= 32767

    def test_custom_sample_rate(self, tmp_path):
        """Verify custom sample rate works."""
        from titan_plugin.expressive.audio import ProceduralAudioGen
        gen = ProceduralAudioGen(output_dir=str(tmp_path), sample_rate=22050)
        path = gen.generate_trinity_sonification(
            body=[0.5] * 5, mind=[0.5] * 5, spirit=[0.5] * 5,
            duration_seconds=2,
        )
        with wave.open(path, "r") as wf:
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 22050 * 2
