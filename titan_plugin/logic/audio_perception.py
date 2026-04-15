"""
titan_plugin/logic/audio_perception.py — General-Purpose Audio Perception.

Extracts 15D audio features from sample arrays, mapped to Titan's outer Trinity.
Pure function: no bus access, no state mutation beyond internal tracking.

Feature groups (3 x 5D = 15D):
  Physical (5D) → outer_body[0:5]       — what the audio SOUNDS like
  Pattern  (5D) → oMind Feeling[5:10]   — what musical PATTERNS exist
  Temporal (5D) → modulates Physical     — HOW the sound evolves over time

Wired to Trinity: 10D (Physical + Pattern). Temporal modulates Physical.
"""
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class AudioPerception:
    """General-purpose audio perception: sample array -> 15D features."""

    def __init__(self):
        self._prev_spectrum: np.ndarray | None = None
        self._heard_count: int = 0

    def perceive(self, samples: np.ndarray, sr: int = 16000) -> dict:
        """
        Extract 15D features from audio samples.

        Args:
            samples: numpy array of float64, normalized -1.0 to 1.0
            sr: sample rate (default 16000)

        Returns:
            Dict with keys: physical, pattern, temporal (each list[5 floats]
            in [0,1]), and flat_15d (list[15]).
        """
        if samples is None or len(samples) < sr:
            return self._neutral()

        # Cap at 30 seconds
        max_samples = sr * 30
        if len(samples) > max_samples:
            samples = samples[:max_samples]

        self._heard_count += 1

        # Full spectrum for feature extraction
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / sr)

        physical = self._extract_physical(samples, spectrum, freqs, sr)
        pattern = self._extract_pattern(samples, spectrum, sr)
        temporal = self._extract_temporal(samples, spectrum, sr)

        # Update prev spectrum for delta tracking
        self._prev_spectrum = spectrum.copy()

        def _r(vals):
            return [round(max(0.0, min(1.0, v)), 4) for v in vals]

        phys_r = _r(physical)
        patt_r = _r(pattern)
        temp_r = _r(temporal)

        return {
            "physical": phys_r,
            "pattern": patt_r,
            "temporal": temp_r,
            "flat_15d": phys_r + patt_r + temp_r,
        }

    def _neutral(self) -> dict:
        """Return neutral features when audio is too short."""
        n = [0.5, 0.5, 0.5, 0.5, 0.5]
        return {"physical": n[:], "pattern": n[:], "temporal": n[:],
                "flat_15d": n * 3}

    # ── Physical (5D): What the audio SOUNDS like ────────────────────

    def _extract_physical(self, samples: np.ndarray, spectrum: np.ndarray,
                          freqs: np.ndarray, sr: int) -> list[float]:
        """
        [0] spectral_centroid: frequency center of mass (bright vs dark)
        [1] harmonic_ratio:    consonance/tonality (tonal vs noisy)
        [2] rhythmic_regularity: onset interval regularity
        [3] spectral_symmetry: balance across frequency bands
        [4] harmony:           combined beauty measure
        """
        centroid = self._spectral_centroid(spectrum, freqs, sr)
        harmonic = self._harmonic_ratio(samples, sr)
        rhythm = self._rhythmic_regularity(samples, sr)
        symmetry = self._spectral_symmetry(spectrum)
        harmony = harmonic * 0.4 + symmetry * 0.3 + (1.0 - abs(rhythm - 0.5) * 2) * 0.3
        return [centroid, harmonic, rhythm, symmetry, harmony]

    # ── Pattern (5D): What musical PATTERNS exist ────────────────────

    def _extract_pattern(self, samples: np.ndarray, spectrum: np.ndarray,
                         sr: int) -> list[float]:
        """
        [0] onset_count:      number of detected onsets (normalized)
        [1] tempo_variance:   variability of beat intervals
        [2] key_stability:    how consistent the dominant pitch is
        [3] melodic_contour:  pitch movement direction (rising vs falling)
        [4] timbre_diversity: spectral variation over time
        """
        onset_count = self._onset_count(samples, sr)
        tempo_variance = self._tempo_variance(samples, sr)
        key_stability = self._key_stability(samples, sr)
        melodic_contour = self._melodic_contour(samples, sr)
        timbre_diversity = self._timbre_diversity(samples, sr)
        return [onset_count, tempo_variance, key_stability, melodic_contour, timbre_diversity]

    # ── Temporal (5D): HOW the sound evolves ─────────────────────────

    def _extract_temporal(self, samples: np.ndarray, spectrum: np.ndarray,
                          sr: int) -> list[float]:
        """
        [0] attack_sharpness: how quickly energy rises at onset
        [1] decay_rate:       how quickly energy falls after peak
        [2] dynamic_range:    loudest vs quietest (compressed vs dynamic)
        [3] spectral_flux:    how much the spectrum changes from prev audio
        [4] duration_norm:    audio length normalized (0=1s, 1=30s+)
        """
        attack = self._attack_sharpness(samples, sr)
        decay = self._decay_rate(samples, sr)
        dynamic_range = self._dynamic_range(samples)
        spectral_flux = self._spectral_flux(spectrum)
        duration_norm = min(1.0, len(samples) / (sr * 30.0))
        return [attack, decay, dynamic_range, spectral_flux, duration_norm]

    # ── Helper methods ───────────────────────────────────────────────

    @staticmethod
    def _spectral_centroid(spectrum: np.ndarray, freqs: np.ndarray,
                           sr: int) -> float:
        total = np.sum(spectrum) + 1e-10
        centroid_hz = np.sum(freqs * spectrum) / total
        return min(1.0, max(0.0, centroid_hz / (sr / 2)))

    @staticmethod
    def _harmonic_ratio(samples: np.ndarray, sr: int) -> float:
        window = samples[:min(len(samples), sr)]
        corr = np.correlate(window, window, mode="full")
        corr = corr[len(corr) // 2:]
        min_lag = max(1, sr // 2000)
        max_lag = min(len(corr) - 1, sr // 50)
        if max_lag <= min_lag:
            return 0.5
        segment = corr[min_lag:max_lag + 1]
        return min(1.0, max(0.0, np.max(segment) / (corr[0] + 1e-10)))

    @staticmethod
    def _rhythmic_regularity(samples: np.ndarray, sr: int) -> float:
        hop = 512
        n_frames = len(samples) // hop
        if n_frames < 10:
            return 0.5
        energy = np.array([np.sum(samples[i * hop:(i + 1) * hop] ** 2)
                           for i in range(n_frames)])
        diff = np.diff(energy)
        threshold = np.mean(diff) + np.std(diff)
        onsets = np.where(diff > threshold)[0]
        if len(onsets) < 3:
            return 0.5
        intervals = np.diff(onsets)
        if len(intervals) < 2:
            return 0.5
        bins = min(20, len(intervals))
        hist, _ = np.histogram(intervals, bins=bins)
        hist = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
        max_entropy = math.log2(bins) + 1e-10
        return min(1.0, max(0.0, 1.0 - entropy / max_entropy))

    @staticmethod
    def _spectral_symmetry(spectrum: np.ndarray) -> float:
        n = len(spectrum)
        if n < 4:
            return 0.5
        quarter = n // 4
        bands = [np.sum(spectrum[i * quarter:(i + 1) * quarter] ** 2) for i in range(4)]
        total = sum(bands) + 1e-10
        proportions = [b / total for b in bands]
        deviation = sum(abs(p - 0.25) for p in proportions) / 2.0
        return max(0.0, min(1.0, 1.0 - deviation / 0.75))

    @staticmethod
    def _onset_count(samples: np.ndarray, sr: int) -> float:
        hop = 512
        n_frames = len(samples) // hop
        if n_frames < 4:
            return 0.0
        energy = np.array([np.sum(samples[i * hop:(i + 1) * hop] ** 2)
                           for i in range(n_frames)])
        diff = np.diff(energy)
        threshold = np.mean(diff) + np.std(diff)
        onsets = np.sum(diff > threshold)
        # Normalize: ~30 onsets in 5s audio = 1.0
        duration_s = len(samples) / sr
        return min(1.0, onsets / (duration_s * 6.0 + 1e-10))

    @staticmethod
    def _tempo_variance(samples: np.ndarray, sr: int) -> float:
        hop = 512
        n_frames = len(samples) // hop
        if n_frames < 10:
            return 0.5
        energy = np.array([np.sum(samples[i * hop:(i + 1) * hop] ** 2)
                           for i in range(n_frames)])
        diff = np.diff(energy)
        threshold = np.mean(diff) + np.std(diff)
        onsets = np.where(diff > threshold)[0]
        if len(onsets) < 4:
            return 0.5
        intervals = np.diff(onsets).astype(np.float64)
        mean_int = np.mean(intervals)
        if mean_int < 1e-10:
            return 0.5
        cv = np.std(intervals) / mean_int  # Coefficient of variation
        return min(1.0, cv)  # 0=perfectly regular, 1=highly variable

    @staticmethod
    def _key_stability(samples: np.ndarray, sr: int) -> float:
        """How consistent the dominant pitch is across the audio."""
        chunk_size = sr  # 1-second chunks
        n_chunks = min(10, len(samples) // chunk_size)
        if n_chunks < 2:
            return 0.5
        dominant_freqs = []
        for i in range(n_chunks):
            chunk = samples[i * chunk_size:(i + 1) * chunk_size]
            spectrum = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
            # Find peak in musical range (50-2000Hz)
            mask = (freqs >= 50) & (freqs <= 2000)
            if mask.any():
                peak_idx = np.argmax(spectrum[mask])
                dominant_freqs.append(freqs[mask][peak_idx])
        if len(dominant_freqs) < 2:
            return 0.5
        # Low variance = stable key
        cv = np.std(dominant_freqs) / (np.mean(dominant_freqs) + 1e-10)
        return min(1.0, max(0.0, 1.0 - cv))

    @staticmethod
    def _melodic_contour(samples: np.ndarray, sr: int) -> float:
        """Pitch movement: 0=falling, 0.5=stable, 1=rising."""
        chunk_size = sr // 2  # Half-second chunks
        n_chunks = min(20, len(samples) // chunk_size)
        if n_chunks < 4:
            return 0.5
        centroids = []
        for i in range(n_chunks):
            chunk = samples[i * chunk_size:(i + 1) * chunk_size]
            spectrum = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
            total = np.sum(spectrum) + 1e-10
            centroids.append(np.sum(freqs * spectrum) / total)
        if len(centroids) < 4:
            return 0.5
        # Linear regression slope
        x = np.arange(len(centroids), dtype=np.float64)
        y = np.array(centroids)
        if np.std(x) < 1e-10:
            return 0.5
        slope = np.corrcoef(x, y)[0, 1]
        if np.isnan(slope):
            return 0.5
        return min(1.0, max(0.0, (slope + 1.0) / 2.0))

    @staticmethod
    def _timbre_diversity(samples: np.ndarray, sr: int) -> float:
        """Spectral variation over time (0=constant timbre, 1=highly varied)."""
        chunk_size = sr  # 1-second chunks
        n_chunks = min(10, len(samples) // chunk_size)
        if n_chunks < 2:
            return 0.0
        spectra = []
        for i in range(n_chunks):
            chunk = samples[i * chunk_size:(i + 1) * chunk_size]
            spectrum = np.abs(np.fft.rfft(chunk))
            # Normalize spectrum
            norm = np.sum(spectrum) + 1e-10
            spectra.append(spectrum / norm)
        # Average pairwise cosine distance
        distances = []
        for i in range(len(spectra)):
            for j in range(i + 1, len(spectra)):
                min_len = min(len(spectra[i]), len(spectra[j]))
                a, b = spectra[i][:min_len], spectra[j][:min_len]
                cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                distances.append(1.0 - cos_sim)
        return min(1.0, np.mean(distances) * 5.0) if distances else 0.0

    @staticmethod
    def _attack_sharpness(samples: np.ndarray, sr: int) -> float:
        """How quickly energy rises at onset. Sharp=1, gradual=0."""
        # Analyze first 500ms
        window = samples[:min(len(samples), sr // 2)]
        hop = 64
        n_frames = len(window) // hop
        if n_frames < 4:
            return 0.5
        energy = np.array([np.sum(window[i * hop:(i + 1) * hop] ** 2)
                           for i in range(n_frames)])
        # Find peak energy frame
        peak_idx = np.argmax(energy)
        if peak_idx == 0:
            return 1.0  # Instant attack
        # Attack time = frames to reach peak / total frames
        attack_ratio = peak_idx / n_frames
        return min(1.0, max(0.0, 1.0 - attack_ratio * 2.0))

    @staticmethod
    def _decay_rate(samples: np.ndarray, sr: int) -> float:
        """How quickly energy falls after peak. Fast=1, sustained=0."""
        hop = 256
        n_frames = len(samples) // hop
        if n_frames < 4:
            return 0.5
        energy = np.array([np.sum(samples[i * hop:(i + 1) * hop] ** 2)
                           for i in range(n_frames)])
        peak_idx = np.argmax(energy)
        if peak_idx >= n_frames - 2:
            return 0.0  # Peak at end = no decay
        post_peak = energy[peak_idx:]
        if len(post_peak) < 2:
            return 0.5
        # How quickly does energy drop to 10% of peak?
        peak_val = energy[peak_idx] + 1e-10
        threshold = peak_val * 0.1
        below = np.where(post_peak < threshold)[0]
        if len(below) == 0:
            return 0.0  # Never decays = sustained
        decay_frames = below[0]
        return min(1.0, decay_frames / (len(post_peak) + 1e-10) * 2.0)

    @staticmethod
    def _dynamic_range(samples: np.ndarray) -> float:
        """Loudest vs quietest (0=compressed, 1=highly dynamic)."""
        hop = 512
        n_frames = len(samples) // hop
        if n_frames < 4:
            return 0.5
        rms = np.array([np.sqrt(np.mean(samples[i * hop:(i + 1) * hop] ** 2))
                         for i in range(n_frames)])
        rms = rms[rms > 1e-10]  # Drop silence
        if len(rms) < 2:
            return 0.0
        ratio = np.max(rms) / (np.min(rms) + 1e-10)
        # Normalize: ratio of 10 = 1.0 (20dB range)
        return min(1.0, math.log10(ratio + 1e-10) / 1.0)

    def _spectral_flux(self, spectrum: np.ndarray) -> float:
        """How much the spectrum changed from previous audio."""
        if self._prev_spectrum is None:
            return 0.0
        min_len = min(len(spectrum), len(self._prev_spectrum))
        diff = np.sum(np.abs(spectrum[:min_len] - self._prev_spectrum[:min_len]))
        norm = np.sum(spectrum[:min_len]) + np.sum(self._prev_spectrum[:min_len]) + 1e-10
        return min(1.0, diff / norm * 2.0)
