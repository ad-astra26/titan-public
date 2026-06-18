"""Titan-time spine — the circadian-cycle counter + the his-time↔human-time translator.

`RFP_verifiable_autobiographical_presence_memory` §7.0 (Phase 0). Two primitives:

  • **CircadianCycleCounter** — a monotonic, persisted, Titan-owned cycle key (Q3).
    Seeded from `consciousness_age`; incremented EXACTLY ONCE per circadian trough
    via an edge-triggered idempotent latch. The `cycle_id` + `age_epoch_range` are
    the pure Titan-time keys the autobiography is sealed under (INV-PAM-TITAN-TIME).
    NOT `dream_state.bin:cycle_count` (that is intra-cycle, many dreams per cycle).

  • **TitanTimeTranslator** — converts a his-time epoch-gap to an approximate human
    phrase ("~3 days ago"), at the NARRATION EDGE ONLY (Q4). The per-Titan epoch
    rate is MEASURED LIVE from the `consciousness_age` SHM `(age_epochs, ts)` pair
    vs a persisted baseline anchor — NEVER hardcoded (BRAIN §248 / BRAIN-INV-20,
    RATIFIED: "reading live per-Titan rates — never hardcoded; the rates drift and
    differ per Titan"). `COGNITIVE_EPOCH_DEFAULT_INTERVAL_S` (10.35s) is only the
    *default* of an adaptive 1.15–31.05s timer — referenced as a labeled nominal,
    never used to fabricate false precision (cold-start hedges honestly to
    "recently").

Scope (Phase 0): primitives ONLY. The latch is CALLED from the MEDITATION_COMPLETE
handler + the Soul Diary's UTC-day latch is swapped to `cycle_id` in **Phase C**.
This module does not seal, recall, or touch the OVG.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable, Optional

from titan_hcl._phase_c_constants import COGNITIVE_EPOCH_DEFAULT_INTERVAL_S
from titan_hcl.logic.consciousness_age_reader import ConsciousnessAgeReader

logger = logging.getLogger(__name__)

# Config-gain defaults (Q1; overridable via `[titan_time]` → get_params). These are
# circadian-phase thresholds on the 0..1 `get_circadian_phase()` overlay (trough≈0.2,
# peak≈0.9): latch fires below `trough_threshold`, re-arms above `rearm_threshold`
# (hysteresis → exactly one latch per night).
DEFAULT_TROUGH_THRESHOLD: float = 0.30
DEFAULT_REARM_THRESHOLD: float = 0.45
DEFAULT_SAVE_DIR: str = "data/titan_time"


def _load_titan_time_config() -> dict:
    """Read the `[titan_time]` config section via the Phase-B SHM path; tolerant of
    being called outside a worker (returns {} → defaults apply)."""
    try:
        from titan_hcl.params import get_params
        return get_params("titan_time") or {}
    except Exception:  # noqa: BLE001 — config not yet seeded / non-worker context
        return {}


class CircadianCycleCounter:
    """The Titan-owned circadian-cycle counter (Q3).

    `cycle_id` names the currently-OPEN cycle; `cycle_start_epoch` is the
    `age_epochs` at which it opened. A trough-latch closes the open cycle (its
    `age_epoch_range = [cycle_start_epoch, now_age_epochs]`) and opens the next.
    Persisted so the count survives restart; edge-triggered so it fires exactly
    once per circadian trough.
    """

    def __init__(
        self,
        *,
        save_dir: str = DEFAULT_SAVE_DIR,
        titan_id: Optional[str] = None,
        age_reader: Optional[ConsciousnessAgeReader] = None,
        config: Optional[dict] = None,
    ) -> None:
        self._save_dir = save_dir
        self._state_path = os.path.join(save_dir, "cycle_state.json")
        self._age_reader = age_reader or ConsciousnessAgeReader(titan_id=titan_id)
        cfg = config if config is not None else _load_titan_time_config()
        self._trough_threshold = float(cfg.get("trough_threshold", DEFAULT_TROUGH_THRESHOLD))
        self._rearm_threshold = float(cfg.get("rearm_threshold", DEFAULT_REARM_THRESHOLD))
        # state
        self._cycle_id: int = 0
        self._cycle_start_epoch: int = 0
        self._armed: bool = False  # seed disarmed → first latch needs a daytime re-arm
        self._last_latch_ts: float = 0.0
        self._load_or_seed()

    # ── properties (read-only view; Phase C reads these BEFORE calling latch) ──
    @property
    def cycle_id(self) -> int:
        return self._cycle_id

    @property
    def cycle_start_epoch(self) -> int:
        return self._cycle_start_epoch

    @property
    def armed(self) -> bool:
        return self._armed

    # ── persistence ───────────────────────────────────────────────────────────
    def _load_or_seed(self) -> None:
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    data = json.load(f)
                self._cycle_id = int(data["cycle_id"])
                self._cycle_start_epoch = int(data["cycle_start_epoch"])
                self._armed = bool(data["armed"])
                self._last_latch_ts = float(data.get("last_latch_ts", 0.0))
                logger.info(
                    "[titan_time] cycle counter restored — cycle_id=%d start_epoch=%d armed=%s",
                    self._cycle_id, self._cycle_start_epoch, self._armed)
                return
            except Exception as e:  # noqa: BLE001 — corrupt/partial → re-seed
                logger.warning("[titan_time] cycle_state.json unreadable (%s) — re-seeding", e)
        # first-ever seed: cycle 0 opens at the CURRENT age (the autobiography
        # accrues from ship; we never claim pre-feature epochs). Disarmed so the
        # inaugural latch is a genuine post-daytime trough, never a boot artifact.
        self._cycle_id = 0
        self._cycle_start_epoch = int(self._age_reader.get_age_epochs())
        self._armed = False
        self._last_latch_ts = 0.0
        self._persist()
        logger.info("[titan_time] cycle counter SEEDED — cycle_id=0 start_epoch=%d (disarmed)",
                    self._cycle_start_epoch)

    def _persist(self) -> None:
        os.makedirs(self._save_dir, exist_ok=True)
        data = {
            "cycle_id": self._cycle_id,
            "cycle_start_epoch": self._cycle_start_epoch,
            "armed": self._armed,
            "last_latch_ts": self._last_latch_ts,
        }
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self._state_path)  # atomic (G16)

    # ── the latch (Q1/Q3) ───────────────────────────────────────────────────────
    def latch_if_trough(self, phase: float, age_epochs: Optional[int] = None) -> Optional[int]:
        """Edge-triggered, idempotent. Returns the NEW `cycle_id` iff a latch fired
        this call (the "this IS the closing meditation" signal), else None.

        • `phase < trough_threshold` and armed → close the open cycle, open the next
          (`cycle_id += 1`, `cycle_start_epoch = age_epochs`), DISARM, persist, return new id.
        • `phase >= rearm_threshold` → re-ARM (for the next night), return None.
        • otherwise (hysteresis band / already-latched night) → no-op, return None.
        """
        if age_epochs is None:
            age_epochs = int(self._age_reader.get_age_epochs())
        if phase < self._trough_threshold:
            if self._armed:
                self._cycle_id += 1
                self._cycle_start_epoch = int(age_epochs)
                self._armed = False
                self._last_latch_ts = time.time()
                self._persist()
                logger.info(
                    "[titan_time] CYCLE LATCH — new cycle_id=%d opened at age_epochs=%d (phase=%.3f)",
                    self._cycle_id, age_epochs, phase)
                return self._cycle_id
            return None
        if phase >= self._rearm_threshold:
            if not self._armed:
                self._armed = True
                self._persist()
                logger.debug("[titan_time] cycle counter RE-ARMED (phase=%.3f)", phase)
            return None
        return None


# ── human-phrase buckets (approximate; "~" honours that this is a translation) ──
_MINUTE = 60.0
_HOUR = 3600.0
_DAY = 86400.0
_WEEK = 7 * _DAY


def _bucket_seconds(seconds: float) -> str:
    if seconds < 90.0:
        return "moments ago"
    if seconds < 90 * _MINUTE:
        return f"~{int(round(seconds / _MINUTE))} minutes ago"
    if seconds < 36 * _HOUR:
        return f"~{int(round(seconds / _HOUR))} hours ago"
    if seconds < 11 * _DAY:
        return f"~{int(round(seconds / _DAY))} days ago"
    return f"~{int(round(seconds / _WEEK))} weeks ago"


class TitanTimeTranslator:
    """Converts a his-time epoch-gap → an approximate human phrase, at the NARRATION
    EDGE ONLY (Q4). The per-Titan epoch rate is MEASURED LIVE from the
    `consciousness_age` SHM `(age_epochs, ts)` pair vs a persisted baseline anchor
    `(epoch0, ts0)` — never hardcoded (BRAIN §248 / BRAIN-INV-20).
    """

    def __init__(
        self,
        *,
        save_dir: str = DEFAULT_SAVE_DIR,
        titan_id: Optional[str] = None,
        age_reader: Optional[ConsciousnessAgeReader] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._save_dir = save_dir
        self._anchor_path = os.path.join(save_dir, "rate_anchor.json")
        self._age_reader = age_reader or ConsciousnessAgeReader(titan_id=titan_id)
        self._clock = clock or time.time
        self._epoch0: Optional[int] = None
        self._ts0: Optional[float] = None
        self._load_or_set_anchor()

    def _load_or_set_anchor(self) -> None:
        if os.path.exists(self._anchor_path):
            try:
                with open(self._anchor_path, "r") as f:
                    data = json.load(f)
                self._epoch0 = int(data["epoch0"])
                self._ts0 = float(data["ts0"])
                return
            except Exception as e:  # noqa: BLE001 — corrupt → re-anchor
                logger.warning("[titan_time] rate_anchor.json unreadable (%s) — re-anchoring", e)
        # capture the baseline from the live SHM snapshot (epoch, ts). The snapshot's
        # own ts is preferred (it pairs with the epoch); fall back to wall-clock.
        epoch_now, ts_snap = self._age_reader.get_age_snapshot()
        self._epoch0 = int(epoch_now)
        self._ts0 = float(ts_snap) if ts_snap > 0 else float(self._clock())
        self._persist_anchor()
        logger.info("[titan_time] rate anchor SET — epoch0=%d ts0=%.1f", self._epoch0, self._ts0)

    def _persist_anchor(self) -> None:
        os.makedirs(self._save_dir, exist_ok=True)
        tmp = self._anchor_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"epoch0": self._epoch0, "ts0": self._ts0}, f)
        os.replace(tmp, self._anchor_path)

    def measured_sec_per_epoch(self) -> Optional[float]:
        """The LIVE per-Titan rate: `(ts_now − ts0) / (epoch_now − epoch0)`. None when
        degenerate (cold-start: not enough epochs/wall-time has elapsed since the
        anchor) — callers must hedge, never hardcode."""
        if self._epoch0 is None or self._ts0 is None:
            return None
        epoch_n, ts_snap = self._age_reader.get_age_snapshot()
        ts_n = float(ts_snap) if ts_snap > 0 else float(self._clock())
        d_epoch = epoch_n - self._epoch0
        d_ts = ts_n - self._ts0
        if d_epoch <= 0 or d_ts <= 0:
            return None
        return d_ts / d_epoch

    def to_human(self, gap_epochs: int) -> str:
        """`gap_epochs` (his-time) → an approximate human phrase. Honest cold-start
        hedge ("recently") when the live rate isn't measurable yet — we never use the
        nominal `COGNITIVE_EPOCH_DEFAULT_INTERVAL_S` (~10.35s) to fabricate precision."""
        if gap_epochs <= 0:
            return "moments ago"
        rate = self.measured_sec_per_epoch()
        if rate is None or rate <= 0:
            return "recently"  # cold-start: no false precision (BRAIN §248)
        return _bucket_seconds(gap_epochs * rate)

    # the nominal is exposed for diagnostics only — NOT a basis for to_human().
    NOMINAL_SEC_PER_EPOCH: float = COGNITIVE_EPOCH_DEFAULT_INTERVAL_S


__all__ = ("CircadianCycleCounter", "TitanTimeTranslator")
