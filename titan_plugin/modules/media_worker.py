"""
Media Module Worker — pure math perception engine for images and audio.

Picks up queued media files from data/media_queue/, extracts mathematical
features (no neural networks), and publishes SENSE_VISUAL / SENSE_AUDIO
to the bus for Mind worker consumption.

Image features (5-dim):
  [0] Color entropy — RGB distribution information density
  [1] Edge density — Sobel gradient magnitude (structure vs flat)
  [2] Symmetry score — left/right pixel correlation
  [3] Spatial frequency — FFT high-freq energy ratio
  [4] Harmony — combined balance metric (entropy * symmetry * freq_balance)

Audio features (5-dim):
  [0] Spectral centroid — frequency center of mass (brightness)
  [1] Harmonic ratio — consonance measure from autocorrelation peaks
  [2] Rhythmic entropy — onset interval regularity
  [3] Spectral symmetry — balance across frequency bands
  [4] Harmony — combined tonal beauty metric

Entry point: media_worker_main(recv_queue, send_queue, name, config)
"""
import logging
import math
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported formats
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# Lazy-init 30D spatial perception engine
_spatial_perception = None


def _get_spatial_perception():
    """Lazy singleton for SpatialPerception (heavy import, only when needed)."""
    global _spatial_perception
    if _spatial_perception is None:
        from titan_plugin.logic.spatial_perception import SpatialPerception
        _spatial_perception = SpatialPerception()
    return _spatial_perception


# Lazy-init 15D audio perception engine
_audio_perception = None


def _get_audio_perception():
    """Lazy singleton for AudioPerception."""
    global _audio_perception
    if _audio_perception is None:
        from titan_plugin.logic.audio_perception import AudioPerception
        _audio_perception = AudioPerception()
    return _audio_perception


def media_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Media module process."""
    from queue import Empty

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    queue_dir = config.get("queue_dir", os.path.join(project_root, "data", "media_queue"))
    os.makedirs(queue_dir, exist_ok=True)

    logger.info("[MediaWorker] Pure math perception engine online, queue: %s", queue_dir)

    # Signal ready
    _send_msg(send_queue, "MODULE_READY", name, "guardian", {})

    last_scan = 0.0
    scan_interval = 10.0  # Check queue every 10s

    while True:
        # Heartbeat fires every iteration (throttled to 3s min inside the helper).
        # MUST be at top of loop — NOT only in the `except Empty` branch — because
        # broadcast messages (e.g. TITAN_SELF_STATE with dst="all") arriving inside
        # the 3s get() window would starve the Empty path and prevent heartbeats
        # from ever reaching Guardian, causing the 180s heartbeat-timeout restart
        # loop observed after rFP #2 landed. 2026-04-15 diagnosis.
        _send_heartbeat(send_queue, name)

        try:
            msg = recv_queue.get(timeout=3.0)
        except Empty:
            now = time.time()
            if now - last_scan >= scan_interval:
                _scan_and_digest(queue_dir, send_queue, name)
                last_scan = now
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        msg_type = msg.get("type", "")

        if msg_type == "MODULE_SHUTDOWN":
            logger.info("[MediaWorker] Shutdown requested")
            break

        if msg_type == "QUERY":
            from titan_plugin.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            payload = msg.get("payload", {})
            action = payload.get("action", "")
            rid = msg.get("rid")
            src = msg.get("src", "")

            if action == "digest_now":
                results = _scan_and_digest(queue_dir, send_queue, name)
                _send_response(send_queue, name, src, {"digested": results}, rid)
            elif action == "status":
                pending = _list_pending(queue_dir)
                _send_response(send_queue, name, src, {
                    "queue_dir": queue_dir,
                    "pending_files": len(pending),
                }, rid)

    logger.info("[MediaWorker] Exiting")


# ── Queue Scanner ──────────────────────────────────────────────────

def _list_pending(queue_dir: str) -> list[Path]:
    """List unprocessed media files in queue."""
    pending = []
    try:
        for f in Path(queue_dir).iterdir():
            if f.suffix.lower() in IMAGE_EXTS | AUDIO_EXTS:
                pending.append(f)
    except Exception:
        pass
    return sorted(pending, key=lambda p: p.stat().st_mtime)


def _scan_and_digest(queue_dir: str, send_queue, name: str) -> int:
    """Scan queue, digest each file, publish to bus, delete original."""
    files = _list_pending(queue_dir)
    if not files:
        return 0

    digested = 0
    for fpath in files[:5]:  # Max 5 per scan to avoid blocking
        # Heartbeat between files — prevents Guardian 180s timeout on T3 where
        # 30D spatial perception + audio FFT on 5 dense files can exceed the
        # window. Before fix, only one HB was sent pre-scan; a 5-file batch at
        # ~35s/file = 175s + 3s recv timeout + jitter = 180.7s, just over limit.
        _send_heartbeat(send_queue, name)
        try:
            ext = fpath.suffix.lower()
            if ext in IMAGE_EXTS:
                features = _digest_image(fpath)
                if features:
                    # 30D spatial perception (additive — 5D always sent for compat)
                    features_30d = _perceive_image_30d(fpath)
                    # Source: tg_ prefix = external user photo, else self-generated
                    source = "external" if fpath.name.startswith("tg_") else "self"
                    _payload = {
                        "features": features,
                        "features_30d": features_30d,
                        "source": source,
                        "filename": fpath.name,
                        "digest_ts": time.time(),
                    }
                    _send_msg(send_queue, "SENSE_VISUAL", name, "mind", _payload)
                    # G1: Also send to spirit for creative perception loop
                    _send_msg(send_queue, "SENSE_VISUAL", name, "spirit", _payload)
                    logger.info("[MediaWorker] Digested image: %s → %s (src=%s, harmony=%.3f)",
                               fpath.name, "30D" if features_30d else "5D",
                               source, features[4])
            elif ext in AUDIO_EXTS:
                features = _digest_audio(fpath)
                if features:
                    # 15D audio perception (additive — 5D always sent for compat)
                    features_15d = _perceive_audio_15d(fpath)
                    source = "external" if fpath.name.startswith("tg_") else "self"
                    _a_payload = {
                        "features": features,
                        "features_15d": features_15d,
                        "source": source,
                        "filename": fpath.name,
                        "digest_ts": time.time(),
                    }
                    _send_msg(send_queue, "SENSE_AUDIO", name, "mind", _a_payload)
                    # G1: Also send to spirit for creative perception loop
                    _send_msg(send_queue, "SENSE_AUDIO", name, "spirit", _a_payload)
                    logger.info("[MediaWorker] Digested audio: %s → %s (src=%s, harmony=%.3f)",
                               fpath.name, "15D" if features_15d else "5D",
                               source, features[4])

            # Delete after digestion (keep patterns, not raw data)
            fpath.unlink()
            digested += 1

        except Exception as e:
            logger.warning("[MediaWorker] Failed to digest %s: %s", fpath.name, e)
            # Move failed file aside so we don't retry forever
            try:
                fpath.rename(fpath.with_suffix(fpath.suffix + ".failed"))
            except Exception:
                pass

    return digested


# ── Image Perception (Pure Math) ───────────────────────────────────

def _digest_image(fpath: Path) -> list[float] | None:
    """
    Extract 5 pure math features from an image.

    Returns [color_entropy, edge_density, symmetry, spatial_freq, harmony]
    All normalized to 0.0-1.0.
    """
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(fpath).convert("RGB")
        # Resize for consistent processing (max 256px on longest side)
        img.thumbnail((256, 256), Image.LANCZOS)
        pixels = np.array(img, dtype=np.float64)

        # [0] Color entropy — information density of RGB histogram
        color_entropy = _image_color_entropy(pixels)

        # [1] Edge density — Sobel gradient magnitude
        edge_density = _image_edge_density(pixels)

        # [2] Symmetry — left/right correlation
        symmetry = _image_symmetry(pixels)

        # [3] Spatial frequency — FFT high-frequency energy ratio
        spatial_freq = _image_spatial_frequency(pixels)

        # [4] Harmony — combined balance (high entropy + high symmetry + balanced freq = beautiful)
        harmony = (color_entropy * 0.3 + symmetry * 0.4 + (1.0 - abs(spatial_freq - 0.5) * 2) * 0.3)

        return [round(v, 4) for v in [color_entropy, edge_density, symmetry, spatial_freq, harmony]]

    except Exception as e:
        logger.warning("[MediaWorker] Image digest error: %s", e)
        return None


def _perceive_image_30d(fpath: Path) -> dict | None:
    """Extract 30D spatial perception features + 7D pattern profile from an image file."""
    try:
        import numpy as np
        from PIL import Image
        img = Image.open(fpath).convert("RGB")
        img.thumbnail((256, 256), Image.LANCZOS)
        arr = np.array(img, dtype=np.float64)
        features = _get_spatial_perception().perceive(arr)

        # Generalized pattern detection: downscale to 16x16 grayscale grid
        # and run the 7 pattern primitives (symmetry, translation, alignment,
        # containment, adjacency, repetition, shape). Originally ARC-only,
        # now applied to ALL visual perception for MSL readiness.
        try:
            from titan_plugin.logic.pattern_primitives import PatternPrimitives
            gray = Image.open(fpath).convert("L").resize((16, 16), Image.LANCZOS)
            # Quantize to ~10 levels for meaningful pattern detection
            grid = (np.array(gray, dtype=np.float64) / 25.5).astype(int)
            pp = PatternPrimitives()
            profile = pp.compute_profile(grid)
            if features and isinstance(features, dict):
                features["pattern_profile_7d"] = pp.profile_to_vector(profile)
        except Exception:
            pass

        return features
    except Exception as e:
        logger.warning("[MediaWorker] 30D perception error: %s", e)
        return None


def _perceive_audio_15d(fpath: Path) -> dict | None:
    """Extract 15D audio perception features from an audio file."""
    try:
        samples, sr = _load_audio_samples(fpath)
        if samples is None or len(samples) < sr:
            return None
        return _get_audio_perception().perceive(samples, sr)
    except Exception as e:
        logger.warning("[MediaWorker] 15D audio perception error: %s", e)
        return None


def _image_color_entropy(pixels: "np.ndarray") -> float:
    """Shannon entropy of RGB color histogram, normalized to 0-1."""
    import numpy as np

    # Quantize to 32 bins per channel
    bins = 32
    entropy_sum = 0.0
    for ch in range(3):
        hist, _ = np.histogram(pixels[:, :, ch], bins=bins, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        # Shannon entropy
        h = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
        entropy_sum += h

    # Max possible entropy per channel = log2(32) = 5
    max_entropy = 3 * math.log2(bins)
    return min(1.0, entropy_sum / max_entropy)


def _image_edge_density(pixels: "np.ndarray") -> float:
    """Fraction of pixels with strong gradient (Sobel-like), 0-1."""
    import numpy as np

    gray = np.mean(pixels, axis=2)
    # Simple Sobel approximation via finite differences
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    # Trim to common shape
    min_h = min(gx.shape[0], gy.shape[0])
    min_w = min(gx.shape[1], gy.shape[1])
    gx = gx[:min_h, :min_w]
    gy = gy[:min_h, :min_w]
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Threshold: edge if gradient > 10% of max possible (255)
    edge_count = np.sum(magnitude > 25.5)
    total = magnitude.size
    return min(1.0, edge_count / (total * 0.5))  # Normalize: 50% edges = 1.0


def _image_symmetry(pixels: "np.ndarray") -> float:
    """Left-right symmetry score via pixel correlation, 0-1."""
    import numpy as np

    gray = np.mean(pixels, axis=2)
    w = gray.shape[1]
    if w < 4:
        return 0.5

    left = gray[:, :w // 2]
    right = gray[:, w // 2:w // 2 + left.shape[1]][:, ::-1]  # Mirror right half

    if left.shape != right.shape:
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]

    # Pearson correlation
    left_flat = left.flatten()
    right_flat = right.flatten()
    corr = np.corrcoef(left_flat, right_flat)[0, 1]
    if np.isnan(corr):
        return 0.5
    # Map correlation (-1 to 1) → (0 to 1)
    return max(0.0, min(1.0, (corr + 1.0) / 2.0))


def _image_spatial_frequency(pixels: "np.ndarray") -> float:
    """Ratio of high-frequency energy in 2D FFT, 0-1."""
    import numpy as np

    gray = np.mean(pixels, axis=2)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    # Define "low frequency" as center 25% of spectrum
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    low_mask = ((y - cy) ** 2 + (x - cx) ** 2) <= r ** 2

    total_energy = np.sum(magnitude ** 2) + 1e-10
    low_energy = np.sum(magnitude[low_mask] ** 2)
    high_ratio = 1.0 - (low_energy / total_energy)

    return min(1.0, max(0.0, high_ratio))


# ── Audio Perception (Pure Math) ───────────────────────────────────

def _digest_audio(fpath: Path) -> list[float] | None:
    """
    Extract 5 pure math features from audio.

    Returns [spectral_centroid, harmonic_ratio, rhythmic_entropy, spectral_symmetry, harmony]
    All normalized to 0.0-1.0.
    """
    try:
        import numpy as np

        samples, sr = _load_audio_samples(fpath)
        if samples is None or len(samples) < sr:
            return None

        # Use at most 30 seconds
        max_samples = sr * 30
        if len(samples) > max_samples:
            samples = samples[:max_samples]

        # [0] Spectral centroid
        centroid = _audio_spectral_centroid(samples, sr)

        # [1] Harmonic ratio
        harmonic = _audio_harmonic_ratio(samples, sr)

        # [2] Rhythmic entropy
        rhythm = _audio_rhythmic_entropy(samples, sr)

        # [3] Spectral symmetry
        spec_sym = _audio_spectral_symmetry(samples, sr)

        # [4] Harmony — combined tonal beauty
        harmony = harmonic * 0.4 + spec_sym * 0.3 + (1.0 - abs(rhythm - 0.5) * 2) * 0.3

        return [round(v, 4) for v in [centroid, harmonic, rhythm, spec_sym, harmony]]

    except Exception as e:
        logger.warning("[MediaWorker] Audio digest error: %s", e)
        return None


def _load_audio_samples(fpath: Path) -> tuple:
    """Load audio file to numpy array using pydub. Returns (samples, sample_rate)."""
    try:
        from pydub import AudioSegment
        import numpy as np

        audio = AudioSegment.from_file(str(fpath))
        # Convert to mono, 16kHz for consistent analysis
        audio = audio.set_channels(1).set_frame_rate(16000)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
        # Normalize to -1.0 to 1.0
        max_val = np.max(np.abs(samples)) + 1e-10
        samples = samples / max_val
        return samples, 16000
    except Exception as e:
        logger.warning("[MediaWorker] Audio load failed: %s", e)
        return None, 0


def _audio_spectral_centroid(samples: "np.ndarray", sr: int) -> float:
    """Frequency center of mass, normalized to 0-1."""
    import numpy as np

    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sr)

    total = np.sum(spectrum) + 1e-10
    centroid_hz = np.sum(freqs * spectrum) / total

    # Normalize: 0Hz=0.0, 8000Hz(Nyquist)=1.0
    return min(1.0, max(0.0, centroid_hz / (sr / 2)))


def _audio_harmonic_ratio(samples: "np.ndarray", sr: int) -> float:
    """Consonance measure via autocorrelation peak strength, 0-1."""
    import numpy as np

    # Autocorrelation of a 1-second window
    window = samples[:min(len(samples), sr)]
    corr = np.correlate(window, window, mode="full")
    corr = corr[len(corr) // 2:]  # Keep positive lags

    # Skip lag 0 (always 1.0), find strongest peak in musical range (50-2000Hz)
    min_lag = max(1, sr // 2000)  # 2000 Hz
    max_lag = min(len(corr) - 1, sr // 50)  # 50 Hz

    if max_lag <= min_lag:
        return 0.5

    segment = corr[min_lag:max_lag + 1]
    peak = np.max(segment)
    base = corr[0] + 1e-10

    ratio = peak / base
    return min(1.0, max(0.0, ratio))


def _audio_rhythmic_entropy(samples: "np.ndarray", sr: int) -> float:
    """Onset interval regularity, 0-1. Low entropy = regular rhythm."""
    import numpy as np

    # Energy envelope (hop = 512 samples)
    hop = 512
    n_frames = len(samples) // hop
    if n_frames < 10:
        return 0.5

    energy = np.array([np.sum(samples[i * hop:(i + 1) * hop] ** 2) for i in range(n_frames)])

    # Onset detection: energy increase
    diff = np.diff(energy)
    threshold = np.mean(diff) + np.std(diff)
    onsets = np.where(diff > threshold)[0]

    if len(onsets) < 3:
        return 0.5

    # Interval histogram
    intervals = np.diff(onsets)
    if len(intervals) < 2:
        return 0.5

    # Normalized entropy of interval distribution
    bins = min(20, len(intervals))
    hist, _ = np.histogram(intervals, bins=bins)
    hist = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
    max_entropy = math.log2(bins) + 1e-10

    # Low entropy = regular rhythm = high score
    regularity = 1.0 - (entropy / max_entropy)
    return min(1.0, max(0.0, regularity))


def _audio_spectral_symmetry(samples: "np.ndarray", sr: int) -> float:
    """Balance across frequency bands, 0-1."""
    import numpy as np

    spectrum = np.abs(np.fft.rfft(samples))
    n = len(spectrum)
    if n < 4:
        return 0.5

    # Split into 4 frequency bands
    quarter = n // 4
    bands = [np.sum(spectrum[i * quarter:(i + 1) * quarter] ** 2) for i in range(4)]
    total = sum(bands) + 1e-10

    # Perfect symmetry = all bands equal (0.25 each)
    proportions = [b / total for b in bands]
    deviation = sum(abs(p - 0.25) for p in proportions) / 2.0  # Max deviation = 1.5
    symmetry = 1.0 - min(1.0, deviation / 0.75)

    return max(0.0, min(1.0, symmetry))


# ── Messaging Helpers ──────────────────────────────────────────────

def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict, rid: str = None) -> None:
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception:
        from titan_plugin.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _send_response(send_queue, src: str, dst: str, payload: dict, rid: str) -> None:
    _send_msg(send_queue, "RESPONSE", src, dst, payload, rid)


# Heartbeat throttle (Phase E Fix 2): 3s min interval per process.
_last_hb_ts: float = 0.0


def _send_heartbeat(send_queue, name: str) -> None:
    global _last_hb_ts
    now = time.time()
    if now - _last_hb_ts < 3.0:
        return
    _last_hb_ts = now
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        rss_mb = 0
    _send_msg(send_queue, "MODULE_HEARTBEAT", name, "guardian", {"rss_mb": round(rss_mb, 1)})
