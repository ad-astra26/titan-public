"""
studio_state_publisher — owns studio_state.bin SHM writer.

Phase C v1.8.3 (D-SPEC-57) per `rFP_titan_hcl_l2_separation_strategy.md §4.K`.

Invoked from studio_worker on dual-trigger cadence per Maker Q3 greenlight
(2026-05-15): on every `KERNEL_EPOCH_TICK` (1.0 Hz adaptive cadence — keeps
counts fresh against external dir scans) AND immediately after every successful
render completion (updates counts + `last_render_ts` / `last_render_type`).
Single-threaded (G21 single-writer-per-slot).

Hot-path stats reads (`/v4/studio/stats` Observatory route, future dashboards)
bypass the bus entirely via this slot per G18+G20.

Cold-boot safe — first publish before any render completes uses defaults
(meditation/epoch/eureka counts read from disk, last_render_ts=0,
last_render_type="none"). Readers see the cold values and proceed.

Failure modes mirror DreamStatePublisher + MemoryStatePublisher +
SocialGraphStatePublisher + MetabolismStatePublisher precedents:
  - encode/oversize/write fails handled per-tick; first WARN with exc_info,
    subsequent throttled to every 60 ticks.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import msgpack

from titan_hcl.core.state_registry import (
    StateRegistryWriter,
    ensure_shm_root,
)
from titan_hcl.logic.studio_state_specs import (
    STUDIO_STATE_SLOT,
    STUDIO_STATE_SPEC,
)

logger = logging.getLogger(__name__)


_WARN_THROTTLE_EVERY = 60
_HEARTBEAT_TICKS = (1, 10, 60, 600, 3600)

# Valid `last_render_type` enum values per SPEC §7.1 studio_state.bin schema.
_VALID_RENDER_TYPES = ("none", "meditation", "epoch", "eureka")


class StudioStatePublisher:
    """Owns studio_state.bin SHM writer (G21 single-writer).

    Stateful: tracks (meditation_count, epoch_count, eureka_count,
    last_render_ts, last_render_type) across calls so the KERNEL_EPOCH_TICK
    republish path always has the freshest values.

    Directory scanning is done lazily — the dirs are scanned on every
    `refresh_counts()` call (cheap: stat-only listdir, no file content read).
    Callers should refresh before every publish OR rely on post-render
    `record_render(...)` to bump counters directly.
    """

    def __init__(
        self,
        titan_id: str,
        output_root: Path,
        meditation_dir: Path,
        epoch_dir: Path,
        eureka_dir: Path,
        default_resolution: int,
        highres_resolution: int,
        nft_composite_enabled: bool,
    ):
        self._titan_id = titan_id
        self._shm_root = ensure_shm_root(titan_id)
        self._writer: Optional[StateRegistryWriter] = None
        self._publish_count = 0
        self._publish_success = 0
        self._encode_fails = 0
        self._oversize_fails = 0
        self._write_fails = 0

        # Config echoed in payload (Observatory display only).
        self._output_root = str(output_root)
        self._default_resolution = int(default_resolution)
        self._highres_resolution = int(highres_resolution)
        self._nft_composite_enabled = bool(nft_composite_enabled)

        # Per-category dir refs for refresh_counts.
        self._meditation_dir = Path(meditation_dir)
        self._epoch_dir = Path(epoch_dir)
        self._eureka_dir = Path(eureka_dir)

        # Sticky state — initialized via initial refresh_counts() below.
        self._meditation_count: int = 0
        self._epoch_count: int = 0
        self._eureka_count: int = 0
        self._last_render_ts: float = 0.0
        self._last_render_type: str = "none"

        # Initial count scan — readers see real values on first publish.
        try:
            self.refresh_counts()
        except Exception as e:
            logger.warning(
                "[StudioStatePublisher] initial refresh_counts failed (%s) — "
                "publishing zeros until next refresh", e, exc_info=True)

        logger.info(
            "[StudioStatePublisher] initialized — titan_id=%s shm_root=%s "
            "(slot=%s — SPEC §7.1 v1.8.3 / Preamble G18 + G21 / D-SPEC-57) "
            "initial counts: meditation=%d epoch=%d eureka=%d",
            titan_id, self._shm_root, STUDIO_STATE_SLOT,
            self._meditation_count, self._epoch_count, self._eureka_count)

    def _writer_attach(self) -> StateRegistryWriter:
        if self._writer is not None:
            return self._writer
        self._writer = StateRegistryWriter(STUDIO_STATE_SPEC, self._shm_root)
        logger.info(
            "[StudioStatePublisher] writer attached — slot=%s "
            "max_bytes=%d schema_version=%d path=%s",
            STUDIO_STATE_SLOT, STUDIO_STATE_SPEC.payload_bytes,
            STUDIO_STATE_SPEC.schema_version,
            self._shm_root / f"{STUDIO_STATE_SLOT}.bin")
        return self._writer

    @staticmethod
    def _count_artifacts(directory: Path) -> int:
        """Count non-sidecar files in a directory. Cheap: listdir + extension
        filter, no content read.
        """
        if not directory.exists():
            return 0
        count = 0
        for entry in directory.iterdir():
            if entry.is_file() and not entry.name.endswith(".json"):
                count += 1
        return count

    def refresh_counts(self) -> None:
        """Re-scan the three output dirs and update sticky counts. Called
        on every KERNEL_EPOCH_TICK so external dir scans / out-of-band
        deletes are reflected.
        """
        self._meditation_count = self._count_artifacts(self._meditation_dir)
        self._epoch_count = self._count_artifacts(self._epoch_dir)
        self._eureka_count = self._count_artifacts(self._eureka_dir)

    def record_render(self, render_type: str) -> None:
        """Record a successful render — bumps the matching counter + updates
        last_render_ts / last_render_type. Called from studio_worker after
        every successful StudioCoordinator.* completion.

        Cheaper than re-scanning the dir; the dir-scan refresh_counts() still
        runs on KERNEL_EPOCH_TICK as the safety net.
        """
        if render_type not in _VALID_RENDER_TYPES or render_type == "none":
            logger.warning(
                "[StudioStatePublisher] record_render: unknown render_type=%r "
                "(expected one of meditation/epoch/eureka) — ignoring",
                render_type)
            return
        if render_type == "meditation":
            self._meditation_count += 1
        elif render_type == "epoch":
            self._epoch_count += 1
        elif render_type == "eureka":
            self._eureka_count += 1
        self._last_render_ts = time.time()
        self._last_render_type = render_type

    def publish(self) -> None:
        """Encode current sticky state + write SHM. Called from worker on
        KERNEL_EPOCH_TICK + immediately after every successful render
        (via record_render then publish).
        """
        self._publish_count += 1
        payload = self._compute_payload()
        self._write(payload)

        if self._publish_count in _HEARTBEAT_TICKS:
            logger.info(
                "[StudioStatePublisher] heartbeat — publish_count=%d "
                "success=%d fails={encode=%d oversize=%d write=%d} "
                "counts={meditation=%d epoch=%d eureka=%d} "
                "last_render_type=%s last_render_age_s=%s",
                self._publish_count, self._publish_success,
                self._encode_fails, self._oversize_fails, self._write_fails,
                self._meditation_count, self._epoch_count, self._eureka_count,
                self._last_render_type,
                f"{time.time() - self._last_render_ts:.1f}"
                if self._last_render_ts > 0 else "never")

    def _compute_payload(self) -> dict[str, Any]:
        return {
            "schema_version": STUDIO_STATE_SPEC.schema_version,
            "meditation_count": int(self._meditation_count),
            "epoch_count": int(self._epoch_count),
            "eureka_count": int(self._eureka_count),
            "last_render_ts": float(self._last_render_ts),
            "last_render_type": self._last_render_type,
            "output_root": self._output_root,
            "default_resolution": self._default_resolution,
            "highres_resolution": self._highres_resolution,
            "nft_composite_enabled": self._nft_composite_enabled,
            "ts": time.time(),
        }

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            encoded = msgpack.packb(payload, use_bin_type=True)
        except (TypeError, ValueError) as e:
            self._encode_fails += 1
            if self._encode_fails == 1 or self._encode_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[StudioStatePublisher] msgpack encode failed (#%d): "
                    "%s — keys=%s",
                    self._encode_fails, e, sorted(payload.keys()),
                    exc_info=True)
            return

        if len(encoded) > STUDIO_STATE_SPEC.payload_bytes:
            self._oversize_fails += 1
            logger.critical(
                "[StudioStatePublisher] payload %dB > MAX %dB (#%d) — "
                "slot retains last-known. Investigate upstream shape drift; "
                "do NOT silently truncate.",
                len(encoded), STUDIO_STATE_SPEC.payload_bytes,
                self._oversize_fails)
            return

        try:
            writer = self._writer_attach()
            writer.write_variable(encoded)
            self._publish_success += 1
            if self._publish_success == 1:
                logger.info(
                    "[StudioStatePublisher] FIRST PUBLISH SUCCESS — "
                    "slot=%s payload_bytes=%d meditation=%d epoch=%d eureka=%d "
                    "(consumers can now read via StudioStateShmReader per "
                    "G18 + D-SPEC-57)",
                    STUDIO_STATE_SLOT, len(encoded),
                    self._meditation_count, self._epoch_count, self._eureka_count)
        except Exception as e:
            self._write_fails += 1
            if self._write_fails == 1 or self._write_fails % _WARN_THROTTLE_EVERY == 0:
                logger.warning(
                    "[StudioStatePublisher] writer.write_variable failed (#%d): %s",
                    self._write_fails, e, exc_info=True)
