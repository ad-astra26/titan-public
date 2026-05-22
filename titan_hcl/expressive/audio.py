"""
expressive/audio.py
Pure mathematical sonification — no external audio dependencies.

Two generation modes:
  1. Blockchain sonification — TX hash → pentatonic melody (V1, 3s fixed)
  2. Trinity sonification — Body/Mind/Spirit tensors → layered musical expression

Body → Rhythm layer (tempo, complexity, noise, density)
Mind → Melody layer (register, timbre, scale degree, vibrato, dynamics)
Spirit → Harmony layer (key center, tension, resolution, bass/treble weight)
Middle Path loss → overall consonance vs dissonance

Uses only stdlib: wave, struct, math. Output: 44100Hz 16-bit PCM mono WAV.
"""

import logging
import math
import os
import struct
import time as _time
import wave

logger = logging.getLogger(__name__)

# --- Musical Constants ---

# Chromatic frequencies (C3 to B5 — 3 octaves)
_CHROMATIC = [
    130.81, 138.59, 146.83, 155.56, 164.81, 174.61,  # C3-F3
    185.00, 196.00, 207.65, 220.00, 233.08, 246.94,  # F#3-B3
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23,  # C4-F4
    369.99, 392.00, 415.30, 440.00, 466.16, 493.88,  # F#4-B4
    523.25, 554.37, 587.33, 622.25, 659.26, 698.46,  # C5-F5
    739.99, 783.99, 830.61, 880.00, 932.33, 987.77,  # F#5-B5
]

# Scale patterns (semitone intervals from root)
_SCALES = {
    "major":       [0, 2, 4, 5, 7, 9, 11],
    "minor":       [0, 2, 3, 5, 7, 8, 10],
    "pentatonic":  [0, 2, 4, 7, 9],
    "blues":       [0, 3, 5, 6, 7, 10],
    "dorian":      [0, 2, 3, 5, 7, 9, 10],
    "lydian":      [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":  [0, 2, 4, 5, 7, 9, 10],
}

# A minor pentatonic for blockchain mode
_PENTATONIC_A = [220.00, 261.63, 293.66, 329.63, 392.00, 440.00]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class ProceduralAudioGen:
    """
    Pure mathematical audio generation from cryptographic and tensor data.
    """

    def __init__(self, output_dir: str = "./art_exports", sample_rate: int = 44100):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Mode 1: Blockchain Sonification (V1 — preserved)
    # ------------------------------------------------------------------

    def generate_blockchain_sonification(self, tx_signature: str, sol_balance: float) -> str:
        """
        Translates a Solana TX hash into a pentatonic WAV chime.

        Args:
            tx_signature: Hash string whose bytes dictate the musical sequence.
            sol_balance: Agent energy determining audio overtones.

        Returns:
            Path to the generated .wav file.
        """
        filepath = os.path.join(self.output_dir, f"sonification_{tx_signature[:8]}.wav")

        num_samples = self.sample_rate * 5  # 5 seconds (was 3)
        note_sequence = [
            _PENTATONIC_A[int(c, 16) % len(_PENTATONIC_A)]
            for c in tx_signature[-12:]
        ]

        with wave.open(filepath, "w") as wf:
            wf.setparams((1, 2, self.sample_rate, num_samples, "NONE", "not compressed"))
            samples_per_note = num_samples // len(note_sequence)

            for i in range(num_samples):
                t = i / self.sample_rate
                ni = min(i // samples_per_note, len(note_sequence) - 1)
                freq = note_sequence[ni]

                if sol_balance > 1.0:
                    val = math.sin(2 * math.pi * freq * t) + 0.5 * math.sin(2 * math.pi * freq * 1.5 * t)
                elif sol_balance < 0.1:
                    val = math.sin(2 * math.pi * 40.0 * t) + 0.3 * math.sin(2 * math.pi * freq * t)
                else:
                    val = math.sin(2 * math.pi * freq * t)

                env = math.sin(math.pi * ((i % samples_per_note) / samples_per_note))
                sample_int = int(val * env * 0.5 * 32767)
                wf.writeframesraw(struct.pack("<h", max(-32767, min(32767, sample_int))))

        logger.info("Generated blockchain sonification: %s", filepath)
        return filepath

    # ------------------------------------------------------------------
    # Mode 2: Trinity Sonification (V3)
    # ------------------------------------------------------------------

    def generate_trinity_sonification(
        self,
        body: list[float],
        mind: list[float],
        spirit: list[float],
        middle_path_loss: float = 0.5,
        duration_seconds: int = 15,
    ) -> str:
        """
        Translate Inner Trinity tensor state into a layered musical composition.

        Body[0-4] → rhythm layer (tempo, complexity, noise, density, pulse)
        Mind[0-4] → melody layer (register, timbre, scale, vibrato, dynamics)
        Spirit[0-4] → harmony layer (key, tension, resolution, bass, treble)
        Middle Path loss → consonance (low loss) vs dissonance (high loss)

        Args:
            body: 5-dim Body tensor [interoception, proprioception, somatosensation, entropy, thermal]
            mind: 5-dim Mind tensor [vision, hearing, taste, smell, touch]
            spirit: 5-dim Spirit tensor [who, why, what, body_scalar, mind_scalar]
            middle_path_loss: Combined equilibrium metric (0=perfect balance, 1=severe imbalance)
            duration_seconds: Output duration in seconds (capped by caller config)

        Returns:
            Path to the generated .wav file.
        """
        # Pad tensors to 5 dims if short
        body = (list(body) + [0.5] * 5)[:5]
        mind = (list(mind) + [0.5] * 5)[:5]
        spirit = (list(spirit) + [0.5] * 5)[:5]
        loss = _clamp(middle_path_loss)

        # Unique filename: timestamp + hash of tensor state
        ts = int(_time.time() * 1000)  # millisecond precision
        tensor_hash = hash((tuple(body), tuple(mind), tuple(spirit))) & 0xFFFF
        filepath = os.path.join(self.output_dir, f"trinity_{ts}_{tensor_hash:04x}.wav")
        num_samples = self.sample_rate * duration_seconds

        # --- Derive musical parameters from tensors ---

        # BODY → Rhythm
        tempo_bpm = 60 + _clamp(body[0]) * 120  # 60-180 BPM
        beat_samples = int(self.sample_rate * 60 / max(tempo_bpm, 30))
        rhythm_complexity = _clamp(body[1])  # 0=simple kick, 1=complex polyrhythm
        noise_amount = _clamp(body[3]) * 0.15  # entropy → noise floor
        rhythm_density = _clamp(body[4])  # thermal → how many beats active

        # MIND → Melody
        register_shift = int(_clamp(mind[0]) * 12)  # vision → octave region (0-12 semitones up)
        harmonic_richness = 1 + _clamp(mind[1]) * 4  # hearing → overtone count (1-5)
        scale_name = list(_SCALES.keys())[int(_clamp(mind[2]) * 6.99)]  # taste → scale choice
        vibrato_depth = _clamp(mind[3]) * 8.0  # smell → vibrato Hz range
        dynamics = 0.3 + _clamp(mind[4]) * 0.7  # touch → volume (0.3-1.0)

        # SPIRIT → Harmony
        root_note_idx = int(_clamp(spirit[0]) * 11)  # who → root note (C-B)
        tension = _clamp(spirit[1])  # why → harmonic tension
        resolution = _clamp(spirit[2])  # what → tendency to resolve
        bass_weight = 0.2 + _clamp(spirit[3]) * 0.6  # body_scalar → bass presence
        treble_weight = 0.2 + _clamp(spirit[4]) * 0.6  # mind_scalar → treble presence

        # LOSS → Consonance
        # Low loss = consonant intervals (octave, fifth, fourth)
        # High loss = dissonant intervals (tritone, minor second)
        consonance = 1.0 - loss  # 1=pure consonance, 0=maximum dissonance

        # Build scale frequencies
        scale_intervals = _SCALES[scale_name]
        scale_freqs = []
        for octave_offset in range(3):
            base_idx = root_note_idx + octave_offset * 12
            for interval in scale_intervals:
                idx = base_idx + interval + register_shift
                if 0 <= idx < len(_CHROMATIC):
                    scale_freqs.append(_CHROMATIC[idx])
        if not scale_freqs:
            scale_freqs = _PENTATONIC_A  # fallback

        # Melody note sequence: one note per beat, selected by cycling through scale
        beats_total = max(1, num_samples // beat_samples)
        melody_notes = []
        for b in range(beats_total):
            # Walk through scale with occasional jumps based on tension
            step = 1 if tension < 0.5 else int(1 + tension * 3)
            idx = (b * step) % len(scale_freqs)
            # Resolution: tendency to return to root
            if resolution > 0.6 and b % 4 == 3:
                idx = 0  # resolve to root every 4th beat
            melody_notes.append(scale_freqs[idx])

        # Harmony: chord intervals based on consonance
        def _chord_ratios(cons: float) -> list[float]:
            """Return frequency ratios for chord tones based on consonance level."""
            if cons > 0.8:
                return [1.0, 1.5, 2.0]  # root + fifth + octave (pure)
            elif cons > 0.6:
                return [1.0, 1.25, 1.5]  # root + major third + fifth
            elif cons > 0.4:
                return [1.0, 1.2, 1.5]  # root + minor third + fifth
            elif cons > 0.2:
                return [1.0, 1.26, 1.414]  # root + ~tritone region (tense)
            else:
                return [1.0, 1.06, 1.414]  # minor second + tritone (max dissonance)

        chord_ratios = _chord_ratios(consonance)

        # --- Generate samples ---
        with wave.open(filepath, "w") as wf:
            wf.setparams((1, 2, self.sample_rate, num_samples, "NONE", "not compressed"))

            for i in range(num_samples):
                t = i / self.sample_rate
                beat_idx = min(i // beat_samples, len(melody_notes) - 1)
                beat_phase = (i % beat_samples) / beat_samples

                val = 0.0

                # --- Layer 1: RHYTHM (Body) ---
                # Kick pulse on downbeat
                if beat_phase < 0.1:
                    kick_env = 1.0 - (beat_phase / 0.1)
                    kick = math.sin(2 * math.pi * 55 * t * (1 - beat_phase * 2)) * kick_env
                    val += kick * bass_weight * 0.4

                # Hi-hat on upbeats (complexity controls presence)
                if rhythm_complexity > 0.3:
                    hh_phase = (beat_phase * 2) % 1.0
                    if hh_phase < 0.05 and beat_phase > 0.4:
                        hh_env = 1.0 - (hh_phase / 0.05)
                        # Noise-like hi-hat via high-freq sine cluster
                        hh = sum(
                            math.sin(2 * math.pi * f * t)
                            for f in [5000, 7000, 9000, 11000]
                        ) * 0.25
                        val += hh * hh_env * rhythm_complexity * treble_weight * 0.15

                # Sub-rhythm for density > 0.5 (16th-note ghost pattern)
                if rhythm_density > 0.5:
                    sub_phase = (beat_phase * 4) % 1.0
                    if sub_phase < 0.03:
                        ghost = math.sin(2 * math.pi * 110 * t) * (1.0 - sub_phase / 0.03) * 0.1
                        val += ghost * (rhythm_density - 0.5) * 2

                # Noise floor from entropy
                if noise_amount > 0.01:
                    # Deterministic "noise" via high-freq sine beating
                    noise = math.sin(2 * math.pi * 3141.59 * t) * math.sin(2 * math.pi * 2718.28 * t)
                    val += noise * noise_amount

                # --- Layer 2: MELODY (Mind) ---
                base_freq = melody_notes[beat_idx]

                # Vibrato
                vib = vibrato_depth * math.sin(2 * math.pi * 5.5 * t) if vibrato_depth > 0.5 else 0.0
                mel_freq = base_freq + vib

                # Fundamental + overtones (harmonic richness)
                mel = 0.0
                for h in range(1, int(harmonic_richness) + 1):
                    amplitude = 1.0 / h  # natural harmonic decay
                    mel += amplitude * math.sin(2 * math.pi * mel_freq * h * t)
                mel /= harmonic_richness  # normalize

                # Envelope: smooth attack-sustain-release per beat
                if beat_phase < 0.05:
                    mel_env = beat_phase / 0.05  # attack
                elif beat_phase > 0.85:
                    mel_env = (1.0 - beat_phase) / 0.15  # release
                else:
                    mel_env = 1.0  # sustain
                val += mel * mel_env * dynamics * 0.35

                # --- Layer 3: HARMONY (Spirit) ---
                harm = 0.0
                for ratio in chord_ratios:
                    chord_freq = base_freq * ratio * 0.5  # one octave below melody
                    harm += math.sin(2 * math.pi * chord_freq * t)
                harm /= len(chord_ratios)

                # Pad-like slow envelope for harmony (breathes over 4 beats)
                four_beat_phase = (i % (beat_samples * 4)) / (beat_samples * 4) if beat_samples > 0 else 0
                harm_env = 0.5 + 0.5 * math.sin(2 * math.pi * four_beat_phase)
                val += harm * harm_env * bass_weight * 0.25

                # --- Master mix ---
                # Gentle overall envelope (fade in first 0.5s, fade out last 0.5s)
                fade_samples = int(self.sample_rate * 0.5)
                if i < fade_samples:
                    master_env = i / fade_samples
                elif i > num_samples - fade_samples:
                    master_env = (num_samples - i) / fade_samples
                else:
                    master_env = 1.0

                val *= master_env * 0.6  # headroom
                sample_int = int(_clamp(val, -1.0, 1.0) * 32767)
                wf.writeframesraw(struct.pack("<h", sample_int))

        logger.info(
            "Generated trinity sonification: %s (%.1fs, %s scale, root=%s, loss=%.2f)",
            filepath, duration_seconds, scale_name,
            ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][root_note_idx],
            loss,
        )
        return filepath
