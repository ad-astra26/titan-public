"""
titan_plugin/logic/meditation_watchdog.py — Self-healing meditation cadence watchdog.

Per rFP_self_healing_meditation_cadence.md Phase 1 (MVP scope).

Detects failure modes F1-F8 using only state Titan already exposes:
  F1+F2: meditation overdue (emergent trigger never fires / timer dies)
  F3+F6: stuck in_meditation (crashed mid-flight or no COMPLETE)
  F4:    backup count lag (spirit tracker ahead of backup state)
  F7:    consecutive promoted=0 (dreams not distilling — e.g. T2 I-016)

Design principles:
- Watchdog is a pure class — no spirit_worker internals coupling
- check() accepts tracker dict + now (testable without wall-clock mocking)
- I1 self-test on boot: synthetic F3 injection
- I3 safety floor: never alert before min_alert_hours regardless of avg gap
- Diagnose-first classification for overdue (§5.2: don't force-trigger when Titan is legitimately calm)
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("MeditationWatchdog")


@dataclass
class WatchdogAlert:
    severity: str            # HIGH | MEDIUM | LOW
    failure_mode: str        # e.g. "F1_F2_OVERDUE"
    detail: str
    diagnostic: dict
    ts: float

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "failure_mode": self.failure_mode,
            "detail": self.detail,
            "diagnostic": self.diagnostic,
            "ts": self.ts,
        }


class MeditationWatchdog:
    """Observes _meditation_tracker state. Returns alerts without side effects.

    The owning spirit_worker is responsible for:
      - calling check() at 60s cadence (or multiple of)
      - emitting MEDITATION_HEALTH_ALERT bus events for returned alerts
      - calling record_meditation() on MEDITATION_COMPLETE to update gap history
      - gating Tier-1 force-trigger on classify_overdue() output

    This class has no knowledge of bus or send_msg — keeps it unit-testable.
    """

    def __init__(
        self,
        titan_id: str,
        bootstrap_hours: float = 12.0,
        min_alert_hours: float = 3.0,          # I3 safety floor
        gap_window: int = 50,                  # per rFP §5.3 (wide window for stability)
        stuck_threshold_seconds: float = 600.0,  # F3/F6: 10 min in_meditation = stuck
        backup_lag_threshold: int = 2,         # F4: spirit - backup >= this = alert
        zero_promoted_streak_threshold: int = 3,  # F7
    ):
        self.titan_id = titan_id
        self.bootstrap_hours = bootstrap_hours
        self.min_alert_hours = min_alert_hours
        self.gap_window = gap_window
        self.stuck_threshold = stuck_threshold_seconds
        self.backup_lag_threshold = backup_lag_threshold
        self.zero_promoted_streak_threshold = zero_promoted_streak_threshold

        self._gaps: deque = deque(maxlen=max(100, gap_window))
        self._in_meditation_since: float = 0.0
        self._last_seen_count: int = 0
        self._prev_last_ts: float = 0.0
        self._consecutive_zero_promoted: int = 0
        self._last_check_ts: float = 0.0
        self._selftest_done: bool = False
        self._selftest_pass: bool = False

    # ── Public: gap history ──────────────────────────────────────────

    def record_meditation(self, last_ts: float, promoted: int = 0) -> None:
        """Called externally on MEDITATION_COMPLETE. Updates gap history and
        distillation streak. Keeps watchdog's internal history in sync with
        tracker state without requiring it to observe every tracker mutation.
        """
        if self._prev_last_ts > 0 and last_ts > self._prev_last_ts:
            gap = last_ts - self._prev_last_ts
            if gap > 0:
                self._gaps.append(gap)
        self._prev_last_ts = last_ts

        if promoted == 0:
            self._consecutive_zero_promoted += 1
        else:
            self._consecutive_zero_promoted = 0

    # ── Public: emergent interval computation ────────────────────────

    def expected_interval(self) -> float:
        """Return expected interval in seconds.
        - <5 samples: bootstrap (12h default — transitional, sunsets per rFP §5.3)
        - >=5 samples: 2 × simple mean of last `gap_window` gaps (stability over responsiveness)
        """
        gaps = list(self._gaps)[-self.gap_window:]
        if len(gaps) >= 5:
            return 2.0 * sum(gaps) / len(gaps)
        return self.bootstrap_hours * 3600.0

    # ── Public: primary watchdog check ───────────────────────────────

    def check(
        self,
        tracker: dict,
        now: float,
        backup_state_count: Optional[int] = None,
    ) -> list[WatchdogAlert]:
        """Run watchdog checks against tracker state.

        Args:
          tracker: the module-level _meditation_tracker dict from spirit_worker
          now: current wall-clock (injected for testability)
          backup_state_count: optional, read from data/backup_state.json by caller

        Returns:
          list of WatchdogAlert (possibly empty)
        """
        self._last_check_ts = now
        alerts: list[WatchdogAlert] = []

        last_ts: float = float(tracker.get("last_ts", 0.0) or 0.0)
        count: int = int(tracker.get("count", 0) or 0)
        in_med: bool = bool(tracker.get("in_meditation", False))

        # ── Track in_meditation entry timestamp (fresh-entry detection) ──
        if in_med and self._in_meditation_since == 0.0:
            self._in_meditation_since = now
        elif not in_med:
            self._in_meditation_since = 0.0
        self._last_seen_count = count

        # ── F1+F2: overdue ──
        # Skip if never meditated (last_ts=0 and count=0) — no baseline yet
        if last_ts > 0 and count > 0:
            time_since = now - last_ts
            expected = self.expected_interval()
            # I3 floor: max(min_alert_hours, 2 × expected_gap)
            floor_seconds = self.min_alert_hours * 3600.0
            threshold = max(floor_seconds, expected)
            if time_since > threshold:
                alerts.append(WatchdogAlert(
                    severity="HIGH",
                    failure_mode="F1_F2_OVERDUE",
                    detail=(
                        f"{time_since / 3600:.1f}h since last meditation "
                        f"(threshold {threshold / 3600:.1f}h, expected {expected / 3600:.1f}h)"
                    ),
                    diagnostic={
                        "time_since_hours": round(time_since / 3600, 2),
                        "expected_interval_hours": round(expected / 3600, 2),
                        "threshold_hours": round(threshold / 3600, 2),
                        "count": count,
                        "gap_samples": len(self._gaps),
                    },
                    ts=now,
                ))

        # ── F3+F6: stuck in_meditation ──
        if in_med and self._in_meditation_since > 0:
            stuck_for = now - self._in_meditation_since
            if stuck_for > self.stuck_threshold:
                alerts.append(WatchdogAlert(
                    severity="HIGH",
                    failure_mode="F3_F6_STUCK",
                    detail=(
                        f"in_meditation=True for {stuck_for / 60:.1f}min "
                        f"(threshold {self.stuck_threshold / 60:.0f}min)"
                    ),
                    diagnostic={
                        "stuck_for_minutes": round(stuck_for / 60, 1),
                        "stuck_since_ts": self._in_meditation_since,
                        "count": count,
                    },
                    ts=now,
                ))

        # ── F4: backup count lag ──
        if backup_state_count is not None:
            lag = count - int(backup_state_count)
            if lag >= self.backup_lag_threshold:
                alerts.append(WatchdogAlert(
                    severity="MEDIUM",
                    failure_mode="F4_BACKUP_LAG",
                    detail=(
                        f"spirit_count={count} backup_count={backup_state_count} "
                        f"lag={lag}"
                    ),
                    diagnostic={
                        "spirit_count": count,
                        "backup_count": int(backup_state_count),
                        "lag": lag,
                    },
                    ts=now,
                ))

        # ── F7: not distilling streak ──
        if self._consecutive_zero_promoted >= self.zero_promoted_streak_threshold:
            alerts.append(WatchdogAlert(
                severity="MEDIUM",
                failure_mode="F7_NOT_DISTILLING",
                detail=(
                    f"{self._consecutive_zero_promoted} consecutive meditations "
                    f"with promoted=0"
                ),
                diagnostic={
                    "zero_promoted_streak": self._consecutive_zero_promoted,
                    "count": count,
                },
                ts=now,
            ))

        return alerts

    # ── Diagnose-first classification (rFP §5.2) ─────────────────────

    def classify_overdue(
        self, diagnostic: dict, drain_flat_12h: bool, gaba_flat_12h: bool,
    ) -> str:
        """Classify F1/F2 overdue as 'natural_calm' vs 'stuck'.

        Only 'stuck' triggers Tier-1 force-trigger. 'natural_calm' logs + waits.
        Conservative: both signals flat = stuck; otherwise natural_calm (avoid
        waking a legitimately calm Titan per Maker directive 2026-04-12).
        """
        if drain_flat_12h and gaba_flat_12h:
            return "stuck"
        return "natural_calm"

    # ── I1: self-test on boot ────────────────────────────────────────

    def self_test(self, current_now: Optional[float] = None) -> bool:
        """Fire synthetic F3 (stuck in_meditation) + verify detection.

        Abort boot if this fails — never deploy a safety system we've never seen trigger.
        Writes result to data/meditation_watchdog_selftest.log for audit.
        """
        try:
            now = current_now if current_now is not None else time.time()
            # Preserve real state
            saved_ims = self._in_meditation_since
            saved_last_seen = self._last_seen_count

            # Synthetic tracker: in_meditation=True for longer than stuck threshold
            self._in_meditation_since = now - (self.stuck_threshold + 60.0)
            synth = {
                "last_epoch": 100, "count": 5, "count_since_nft": 1,
                "last_ts": now - 3600.0,  # 1h ago — NOT overdue for default config
                "in_meditation": True,
            }
            alerts = self.check(synth, now)
            f3 = [a for a in alerts if a.failure_mode == "F3_F6_STUCK"]
            passed = (len(f3) == 1 and f3[0].severity == "HIGH")

            # Restore real state (don't pollute with synthetic entry timestamp)
            self._in_meditation_since = saved_ims
            self._last_seen_count = saved_last_seen

            self._selftest_done = True
            self._selftest_pass = passed

            # Audit log
            try:
                os.makedirs("data", exist_ok=True)
                with open("data/meditation_watchdog_selftest.log", "a") as fh:
                    fh.write(json.dumps({
                        "ts": now,
                        "titan_id": self.titan_id,
                        "passed": passed,
                        "alerts_count": len(alerts),
                        "f3_detected": len(f3) > 0,
                        "f3_detail": f3[0].detail if f3 else None,
                    }) + "\n")
            except Exception as log_err:
                logger.warning("[MeditationWatchdog] selftest log write failed: %s", log_err)

            if passed:
                logger.info(
                    "[MeditationWatchdog] self-test PASSED — synthetic F3 detected as HIGH")
            else:
                logger.error(
                    "[MeditationWatchdog] self-test FAILED — F3 not detected (alerts=%d)",
                    len(alerts))
            return passed

        except Exception as e:
            logger.error("[MeditationWatchdog] self_test exception: %s", e, exc_info=True)
            self._selftest_done = True
            self._selftest_pass = False
            return False

    # ── Status for /v4/meditation/health endpoint ────────────────────

    def health_snapshot(self) -> dict:
        """Serialize watchdog state for API / telemetry."""
        return {
            "titan_id": self.titan_id,
            "last_check_ts": self._last_check_ts,
            "gap_samples": len(self._gaps),
            "expected_interval_hours": round(self.expected_interval() / 3600, 2),
            "in_meditation_since_ts": self._in_meditation_since,
            "consecutive_zero_promoted": self._consecutive_zero_promoted,
            "selftest_done": self._selftest_done,
            "selftest_pass": self._selftest_pass,
            "config": {
                "bootstrap_hours": self.bootstrap_hours,
                "min_alert_hours": self.min_alert_hours,
                "gap_window": self.gap_window,
                "stuck_threshold_seconds": self.stuck_threshold,
                "backup_lag_threshold": self.backup_lag_threshold,
                "zero_promoted_streak_threshold": self.zero_promoted_streak_threshold,
            },
        }
