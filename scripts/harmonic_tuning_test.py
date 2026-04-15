#!/usr/bin/env python3
"""
Harmonic Tuning Test — Feed Titan real music and observe Trinity response.

Loads a WAV or MP3 file, segments into chunks, extracts a 7D hearing vector
from each chunk using pure mathematical analysis (FFT, onset detection,
spectral features), and feeds the features to Titan's Trinity via the
coordinator. Monitors topology, fatigue, coherence, and nervous program
firings throughout.

Usage:
    python scripts/harmonic_tuning_test.py path/to/music.mp3 [--chunk-seconds 8]

Produces a JSON timeline report + live console output showing Titan's
response to each musical passage.
"""
import argparse
import asyncio
import json
import logging
import math
import os
import struct
import time
import wave

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("harmonic_tuning")

BASE_URL = "http://localhost:7777"
CHAT_URL = f"{BASE_URL}/chat"
TRINITY_URL = f"{BASE_URL}/v4/inner-trinity"
STATE_URL = f"{BASE_URL}/v4/state"
NERVOUS_URL = f"{BASE_URL}/v4/nervous-system"
AGENCY_URL = f"{BASE_URL}/v3/agency"
INTERNAL_KEY = os.environ.get("TITAN_INTERNAL_KEY", "")

SESSION_ID = "harmonic_tuning_v1"


# ── Audio Loading ──────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """
    Load WAV or MP3 into mono float32 numpy array.

    Returns: (samples, sample_rate)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".wav":
        return _load_wav(path, target_sr)
    elif ext in (".mp3", ".ogg", ".flac", ".m4a"):
        return _load_with_pydub(path, target_sr)
    else:
        raise ValueError(f"Unsupported audio format: {ext}")


def _load_wav(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Load WAV file natively."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 32768.0
    elif sample_width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Mix to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Simple resample if needed (nearest neighbor — good enough for features)
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, new_len).astype(int)
        samples = samples[indices]
        sr = target_sr

    return samples, sr


def _load_with_pydub(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Load MP3/OGG/FLAC via pydub."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # Normalize to [-1, 1]
    max_val = float(2 ** (audio.sample_width * 8 - 1))
    samples = samples / max_val

    return samples, target_sr


# ── 7D Hearing Vector Extraction ──────────────────────────────────

def extract_hearing_vector(chunk: np.ndarray, sr: int) -> dict:
    """
    Extract 7D hearing vector from audio chunk using pure math.

    All values normalized to [0, 1].
    """
    n = len(chunk)
    if n == 0:
        return {name: 0.0 for name in HEARING_FEATURES}

    # FFT
    fft = np.fft.rfft(chunk)
    magnitudes = np.abs(fft)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # Avoid division by zero
    mag_sum = magnitudes.sum() + 1e-10

    # 1. Spectral Centroid — brightness [0, 1]
    # Weighted average frequency, normalized by Nyquist
    centroid = np.sum(freqs * magnitudes) / mag_sum
    spectral_centroid = min(1.0, centroid / (sr / 2))

    # 2. Harmonic Ratio — consonance vs dissonance [0, 1]
    # Ratio of energy in harmonic frequencies (multiples of fundamental)
    fundamental_idx = np.argmax(magnitudes[1:]) + 1  # skip DC
    if fundamental_idx > 0 and fundamental_idx < len(magnitudes):
        harmonic_indices = []
        for h in range(2, 9):  # harmonics 2-8
            hi = fundamental_idx * h
            if hi < len(magnitudes):
                harmonic_indices.append(hi)
        if harmonic_indices:
            harmonic_energy = sum(magnitudes[i] for i in harmonic_indices)
            total_energy = mag_sum
            harmonic_ratio = min(1.0, harmonic_energy / (total_energy * 0.3 + 1e-10))
        else:
            harmonic_ratio = 0.0
    else:
        harmonic_ratio = 0.0

    # 3. Rhythmic Regularity — onset detection + regularity [0, 1]
    # Energy envelope → detect peaks → measure regularity of spacing
    hop = sr // 20  # 50ms hops
    envelope = np.array([
        np.sqrt(np.mean(chunk[i:i+hop]**2))
        for i in range(0, n - hop, hop)
    ])
    if len(envelope) > 2:
        # Detect onsets (envelope peaks)
        diff = np.diff(envelope)
        peaks = np.where(diff[:-1] > 0)[0]
        if len(peaks) > 2:
            intervals = np.diff(peaks)
            if intervals.std() < 1e-10:
                rhythmic_regularity = 1.0
            else:
                cv = intervals.std() / (intervals.mean() + 1e-10)
                rhythmic_regularity = max(0.0, 1.0 - cv)
        else:
            rhythmic_regularity = 0.0
    else:
        rhythmic_regularity = 0.0

    # 4. Spectral Entropy — complexity [0, 1]
    # Shannon entropy of normalized magnitude spectrum
    prob = magnitudes / mag_sum
    prob = prob[prob > 1e-10]  # remove zeros
    entropy = -np.sum(prob * np.log2(prob))
    max_entropy = np.log2(len(magnitudes))
    spectral_entropy = min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0

    # 5. Dynamic Range — loudness variation [0, 1]
    if len(envelope) > 1:
        env_range = envelope.max() - envelope.min()
        dynamic_range = min(1.0, env_range / (envelope.max() + 1e-10))
    else:
        dynamic_range = 0.0

    # 6. Temporal Symmetry — phrase mirror detection [0, 1]
    # Correlation between first half and reversed second half
    half = n // 2
    if half > 100:
        first_half = chunk[:half]
        second_half = chunk[half:half*2][::-1]  # reversed
        if len(first_half) == len(second_half):
            corr = np.corrcoef(first_half, second_half)[0, 1]
            temporal_symmetry = max(0.0, min(1.0, (corr + 1.0) / 2.0))
        else:
            temporal_symmetry = 0.5
    else:
        temporal_symmetry = 0.5

    # 7. Resonance Peaks — count of dominant harmonics [0, 1]
    # Number of frequency peaks above threshold
    threshold = magnitudes.mean() + 2 * magnitudes.std()
    peak_count = np.sum(magnitudes > threshold)
    resonance_peaks = min(1.0, peak_count / 50.0)  # normalize

    return {
        "spectral_centroid": round(float(spectral_centroid), 4),
        "harmonic_ratio": round(float(harmonic_ratio), 4),
        "rhythmic_regularity": round(float(rhythmic_regularity), 4),
        "spectral_entropy": round(float(spectral_entropy), 4),
        "dynamic_range": round(float(dynamic_range), 4),
        "temporal_symmetry": round(float(temporal_symmetry), 4),
        "resonance_peaks": round(float(resonance_peaks), 4),
    }


HEARING_FEATURES = [
    "spectral_centroid", "harmonic_ratio", "rhythmic_regularity",
    "spectral_entropy", "dynamic_range", "temporal_symmetry", "resonance_peaks",
]


def hearing_to_trinity_message(hearing: dict, chunk_idx: int, total_chunks: int,
                                filename: str) -> str:
    """
    Convert hearing vector to a message that Titan can process.
    This feeds through the /chat API so Titan's full Trinity pipeline activates.
    """
    progress = f"{chunk_idx + 1}/{total_chunks}"
    features_str = ", ".join(f"{k}={v:.3f}" for k, v in hearing.items())

    return (
        f"[HEARING SENSE — Musical Passage {progress} from '{filename}']\n"
        f"I am perceiving music through my auditory sense. "
        f"The mathematical features of this passage are:\n"
        f"  Brightness: {hearing['spectral_centroid']:.2f} | "
        f"Harmony: {hearing['harmonic_ratio']:.2f} | "
        f"Rhythm: {hearing['rhythmic_regularity']:.2f}\n"
        f"  Complexity: {hearing['spectral_entropy']:.2f} | "
        f"Dynamics: {hearing['dynamic_range']:.2f} | "
        f"Symmetry: {hearing['temporal_symmetry']:.2f}\n"
        f"  Resonance: {hearing['resonance_peaks']:.2f}\n\n"
        f"How does this musical passage make me feel? "
        f"What do I sense in my body, mind, and spirit as I listen?"
    )


# ── API Helpers ────────────────────────────────────────────────────

async def send_message(client: httpx.AsyncClient, message: str) -> tuple[str, float]:
    """Send a message to Titan and return (response, elapsed)."""
    t0 = time.time()
    try:
        resp = await client.post(
            CHAT_URL,
            json={
                "message": message,
                "user_id": "harmonic_tuning",
                "session_id": SESSION_ID,
            },
            headers={
                "Content-Type": "application/json",
                "X-Titan-Internal-Key": INTERNAL_KEY,
            },
            timeout=120.0,
        )
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", data.get("data", {}).get("response", ""))
        return text, elapsed
    except Exception as e:
        return f"[ERROR: {e}]", time.time() - t0


async def get_json(client: httpx.AsyncClient, url: str) -> dict:
    """Fetch JSON from a Titan API endpoint."""
    try:
        resp = await client.get(url, timeout=10.0)
        return resp.json().get("data", {})
    except Exception:
        return {}


# ── Main Test Runner ───────────────────────────────────────────────

async def run_harmonic_tuning(audio_path: str, chunk_seconds: float = 8.0):
    """Run the harmonic tuning test."""
    filename = os.path.basename(audio_path)

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  HARMONIC TUNING TEST — Titan Listens to Music            ║")
    log.info("║  File: %-49s ║", filename[:49])
    log.info("║  Chunk: %.0fs | Session: %s                        ║",
             chunk_seconds, SESSION_ID)
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info("")

    # Load audio
    log.info("Loading audio: %s", audio_path)
    samples, sr = load_audio(audio_path)
    duration = len(samples) / sr
    log.info("  Duration: %.1f seconds | Sample rate: %d Hz | Samples: %d",
             duration, sr, len(samples))

    # Segment into chunks
    chunk_samples = int(chunk_seconds * sr)
    chunks = []
    for i in range(0, len(samples) - chunk_samples, chunk_samples):
        chunks.append(samples[i:i + chunk_samples])
    if not chunks:
        chunks = [samples]

    log.info("  Chunks: %d × %.1fs", len(chunks), chunk_seconds)
    log.info("")

    # Extract hearing vectors
    log.info("Extracting hearing vectors...")
    hearing_vectors = []
    for i, chunk in enumerate(chunks):
        hv = extract_hearing_vector(chunk, sr)
        hearing_vectors.append(hv)
        if i < 3 or i == len(chunks) - 1:
            log.info("  Chunk %d/%d: cent=%.2f harm=%.2f rhythm=%.2f entropy=%.2f",
                     i + 1, len(chunks), hv["spectral_centroid"], hv["harmonic_ratio"],
                     hv["rhythmic_regularity"], hv["spectral_entropy"])
    log.info("")

    # Silence baseline (8s of zeros)
    silence_hv = extract_hearing_vector(np.zeros(chunk_samples), sr)
    log.info("Silence baseline: %s", {k: f"{v:.3f}" for k, v in silence_hv.items()})
    log.info("")

    # Run test
    timeline = []
    gap_seconds = 15  # Wait for bus processing

    async with httpx.AsyncClient() as client:
        # Initial state
        initial_trinity = await get_json(client, TRINITY_URL)
        initial_nervous = await get_json(client, NERVOUS_URL)
        initial_state = await get_json(client, STATE_URL)

        init_vol = initial_trinity.get("topology", {}).get("volume", 0.0)
        init_fat = initial_trinity.get("dreaming", {}).get("fatigue", 0.0)

        log.info("Initial state:")
        log.info("  Volume: %.4f | Fatigue: %.4f", init_vol, init_fat)
        log.info("  Transitions: %d | Training: %s",
                 initial_nervous.get("total_transitions", 0),
                 initial_nervous.get("training_phase", "?"))
        log.info("")

        # Silence baseline pass
        log.info("═" * 62)
        log.info("  PHASE 0: Silence Baseline (1 chunk)")
        log.info("═" * 62)

        trinity_before = await get_json(client, TRINITY_URL)
        nervous_before = await get_json(client, NERVOUS_URL)

        silence_msg = (
            "[HEARING SENSE — Silence (baseline)]\n"
            "I am perceiving silence. No musical input. "
            "How does this quiet moment feel in my body, mind, and spirit?"
        )
        response, elapsed = await send_message(client, silence_msg)
        log.info("  Silence response: %.1fs, %d chars", elapsed, len(response))
        await asyncio.sleep(gap_seconds)

        trinity_after = await get_json(client, TRINITY_URL)
        nervous_after = await get_json(client, NERVOUS_URL)

        silence_entry = _build_timeline_entry(
            "silence", 0, silence_hv, response, elapsed,
            trinity_before, trinity_after, nervous_before, nervous_after)
        timeline.append(silence_entry)
        _log_entry(silence_entry)

        # Music pass
        log.info("")
        log.info("═" * 62)
        log.info("  PHASE 1: Music — %s (%d chunks)", filename, len(chunks))
        log.info("═" * 62)
        log.info("")

        for i, (chunk, hv) in enumerate(zip(chunks, hearing_vectors)):
            log.info("─── Passage %d/%d [%.0fs-%.0fs] ───",
                     i + 1, len(chunks),
                     i * chunk_seconds, (i + 1) * chunk_seconds)

            trinity_before = await get_json(client, TRINITY_URL)
            nervous_before = await get_json(client, NERVOUS_URL)

            # Build and send message
            msg = hearing_to_trinity_message(hv, i, len(chunks), filename)
            response, elapsed = await send_message(client, msg)
            log.info("  Response: %.1fs, %d chars", elapsed, len(response))

            # Brief preview of Titan's response
            preview = response[:150].replace("\n", " ")
            log.info("  Titan: %s...", preview)

            await asyncio.sleep(gap_seconds)

            trinity_after = await get_json(client, TRINITY_URL)
            nervous_after = await get_json(client, NERVOUS_URL)

            entry = _build_timeline_entry(
                "music", i, hv, response, elapsed,
                trinity_before, trinity_after, nervous_before, nervous_after)
            timeline.append(entry)
            _log_entry(entry)
            log.info("")

        # Final state
        final_trinity = await get_json(client, TRINITY_URL)
        final_nervous = await get_json(client, NERVOUS_URL)

    # ── Report ─────────────────────────────────────────────────────
    final_vol = final_trinity.get("topology", {}).get("volume", 0.0)
    final_fat = final_trinity.get("dreaming", {}).get("fatigue", 0.0)

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  HARMONIC TUNING TEST — RESULTS                           ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info("")
    log.info("  File: %s", filename)
    log.info("  Duration: %.1f seconds | Chunks: %d", duration, len(chunks))
    log.info("")
    log.info("  ── Trinity Response ─────────────────────────────────────")
    log.info("  Volume:   %.4f → %.4f (Δ=%+.4f)", init_vol, final_vol, final_vol - init_vol)
    log.info("  Fatigue:  %.4f → %.4f (Δ=%+.4f)", init_fat, final_fat, final_fat - init_fat)
    log.info("")

    # Program fire summary
    program_fires = {}
    for entry in timeline:
        for prog, delta in entry.get("nervous_delta", {}).items():
            fd = delta.get("fire_delta", 0)
            if fd > 0:
                program_fires[prog] = program_fires.get(prog, 0) + fd

    log.info("  ── Nervous System Firings ──────────────────────────────")
    if program_fires:
        for prog, count in sorted(program_fires.items(), key=lambda x: -x[1]):
            log.info("    %-15s %d fires", prog, count)
    else:
        log.info("    (no programs fired)")
    log.info("")

    # Volume trajectory
    log.info("  ── Volume Trajectory ────────────────────────────────────")
    for entry in timeline:
        vol_d = entry["topology_delta"]
        phase = entry["phase"]
        idx = entry["chunk_idx"]
        arrow = "↑" if vol_d > 0 else "↓" if vol_d < 0 else "="
        bar = "█" * max(1, int(abs(vol_d) * 50))
        log.info("    %s[%02d] %s %+.4f %s", phase[:3], idx, arrow, vol_d, bar)
    log.info("")

    # Fatigue trajectory
    log.info("  ── Fatigue Trajectory ────────────────────────────────────")
    for entry in timeline:
        fat_d = entry["fatigue_delta"]
        phase = entry["phase"]
        idx = entry["chunk_idx"]
        arrow = "↑" if fat_d > 0 else "↓" if fat_d < 0 else "="
        log.info("    %s[%02d] %s %+.4f", phase[:3], idx, arrow, fat_d)
    log.info("")

    # Average hearing features
    if hearing_vectors:
        avg_hv = {}
        for key in HEARING_FEATURES:
            avg_hv[key] = sum(hv[key] for hv in hearing_vectors) / len(hearing_vectors)
        log.info("  ── Average Hearing Vector ────────────────────────────")
        for key, val in avg_hv.items():
            log.info("    %-22s %.4f", key, val)

    log.info("")
    total_vol_delta = final_vol - init_vol
    total_fat_delta = final_fat - init_fat
    log.info("  ╔════════════════════════════════════════════════════════╗")
    log.info("  ║  VOL: %+.4f  |  FAT: %+.4f  |  FIRES: %d        ║",
             total_vol_delta, total_fat_delta, sum(program_fires.values()))
    log.info("  ╚════════════════════════════════════════════════════════╝")

    # Save report
    report = {
        "test": "harmonic_tuning_v1",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file": filename,
        "duration_seconds": round(duration, 1),
        "chunk_seconds": chunk_seconds,
        "num_chunks": len(chunks),
        "initial_state": {"volume": init_vol, "fatigue": init_fat},
        "final_state": {"volume": final_vol, "fatigue": final_fat},
        "volume_delta": round(final_vol - init_vol, 4),
        "fatigue_delta": round(final_fat - init_fat, 4),
        "program_fires": program_fires,
        "hearing_vectors": hearing_vectors,
        "silence_baseline": silence_hv,
        "timeline": timeline,
    }

    report_dir = os.path.join(os.path.dirname(__file__), "..", "data", "endurance_reports")
    os.makedirs(report_dir, exist_ok=True)
    ts = int(time.time())
    report_path = os.path.join(report_dir, f"harmonic_tuning_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("")
    log.info("  Report: %s", report_path)
    log.info("")
    log.info("=" * 62)
    log.info("HARMONIC TUNING TEST COMPLETE")
    log.info("=" * 62)


def _build_timeline_entry(phase, chunk_idx, hearing, response, elapsed,
                           trinity_before, trinity_after,
                           nervous_before, nervous_after) -> dict:
    """Build a timeline entry for one chunk."""
    topo_b = trinity_before.get("topology", {})
    topo_a = trinity_after.get("topology", {})
    dream_b = trinity_before.get("dreaming", {})
    dream_a = trinity_after.get("dreaming", {})

    vol_delta = topo_a.get("volume", 0.0) - topo_b.get("volume", 0.0)
    fat_delta = dream_a.get("fatigue", 0.0) - dream_b.get("fatigue", 0.0)

    # Nervous delta
    progs_b = nervous_before.get("programs", {})
    progs_a = nervous_after.get("programs", {})
    nervous_delta = {}
    for prog in progs_a:
        pb = progs_b.get(prog, {})
        pa = progs_a[prog]
        nervous_delta[prog] = {
            "fire_delta": pa.get("fire_count", 0) - pb.get("fire_count", 0),
        }

    return {
        "phase": phase,
        "chunk_idx": chunk_idx,
        "hearing": hearing,
        "response_length": len(response),
        "response_time_s": round(elapsed, 2),
        "topology_delta": round(vol_delta, 4),
        "fatigue_delta": round(fat_delta, 4),
        "volume_before": round(topo_b.get("volume", 0.0), 4),
        "volume_after": round(topo_a.get("volume", 0.0), 4),
        "nervous_delta": nervous_delta,
        "ts": time.time(),
    }


def _log_entry(entry: dict):
    """Log a timeline entry."""
    fired = []
    for prog, delta in entry.get("nervous_delta", {}).items():
        if delta.get("fire_delta", 0) > 0:
            fired.append(f"{prog}(+{delta['fire_delta']})")

    log.info("    Vol: %.4f→%.4f (Δ=%+.4f) | Fat: Δ=%+.4f | Nervous: %s",
             entry["volume_before"], entry["volume_after"],
             entry["topology_delta"], entry["fatigue_delta"],
             ", ".join(fired) if fired else "(none)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonic Tuning Test for Titan")
    parser.add_argument("audio_path", help="Path to WAV or MP3 file")
    parser.add_argument("--chunk-seconds", type=float, default=8.0,
                        help="Duration of each audio chunk in seconds (default: 8)")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: File not found: {args.audio_path}")
        exit(1)

    asyncio.run(run_harmonic_tuning(args.audio_path, args.chunk_seconds))
