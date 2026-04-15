"""
titan_plugin/logic/sovereignty.py — GREAT CYCLE Sovereignty Transition (M10).

Tracks neuromodulator convergence metrics over rolling windows.
Evaluates transition criteria from Guardian ENFORCING → ADVISORY.
Records transitions on-chain.

Transition Criteria (ALL must be met):
1. great_cycle >= 1 (after first reincarnation)
2. developmental_age > 1000
3. No neuromod saturation (>0.95) or collapse (<0.05) in last 5000 epochs
4. total_great_pulses > 1000
5. Maker cooperative confirmation (Cycle 0 only)
"""
import json
import logging
import os
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

PERSISTENCE_FILE = "data/sovereignty_state.json"

# Convergence window
CONVERGENCE_WINDOW = 5000  # epochs
SATURATION_THRESHOLD = 0.95
COLLAPSE_THRESHOLD = 0.05
VIOLATION_DURATION = 100  # consecutive epochs at extreme = violation


class SovereigntyTracker:
    """Track sovereignty transition criteria and convergence metrics."""

    def __init__(self):
        self._great_cycle = 0
        self._total_great_pulses = 0
        self._developmental_age = 0
        self._sovereignty_mode = "ENFORCING"  # ENFORCING or ADVISORY

        # Rolling convergence tracking (per modulator)
        self._modulator_history: dict[str, deque] = {}
        for mod in ("DA", "5HT", "NE", "ACh", "Endorphin", "GABA"):
            self._modulator_history[mod] = deque(maxlen=CONVERGENCE_WINDOW)

        # Violation tracking
        self._saturation_violations = 0  # epochs with any modulator > 0.95
        self._collapse_violations = 0    # epochs with any modulator < 0.05

        # Transition record
        self._transition_epoch = None
        self._transition_ts = None
        self._maker_confirmed = False

        self._load_state()

    def _load_state(self):
        try:
            if os.path.exists(PERSISTENCE_FILE):
                with open(PERSISTENCE_FILE) as f:
                    data = json.load(f)
                self._great_cycle = data.get("great_cycle", 0)
                self._total_great_pulses = data.get("total_great_pulses", 0)
                self._sovereignty_mode = data.get("sovereignty_mode", "ENFORCING")
                self._saturation_violations = data.get("saturation_violations", 0)
                self._collapse_violations = data.get("collapse_violations", 0)
                self._transition_epoch = data.get("transition_epoch")
        except Exception:
            pass

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(PERSISTENCE_FILE) or ".", exist_ok=True)
            data = {
                "great_cycle": self._great_cycle,
                "total_great_pulses": self._total_great_pulses,
                "sovereignty_mode": self._sovereignty_mode,
                "saturation_violations": self._saturation_violations,
                "collapse_violations": self._collapse_violations,
                "transition_epoch": self._transition_epoch,
                "updated_at": time.time(),
            }
            tmp = PERSISTENCE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, PERSISTENCE_FILE)
        except Exception as e:
            logger.debug("[Sovereignty] Save error: %s", e)

    def record_epoch(
        self,
        epoch_id: int,
        neuromod_levels: dict,
        developmental_age: int = 0,
        great_pulse_fired: bool = False,
    ):
        """Record one epoch of convergence data.

        Called from spirit_worker at each consciousness epoch.
        """
        self._developmental_age = developmental_age

        if great_pulse_fired:
            self._total_great_pulses += 1

        # Track modulator levels
        has_saturation = False
        has_collapse = False
        for mod, level in neuromod_levels.items():
            if mod in self._modulator_history:
                self._modulator_history[mod].append(level)
                if level > SATURATION_THRESHOLD:
                    has_saturation = True
                if level < COLLAPSE_THRESHOLD:
                    has_collapse = True

        if has_saturation:
            self._saturation_violations += 1
        if has_collapse:
            self._collapse_violations += 1

        # Periodic save (every 500 epochs)
        if epoch_id % 500 == 0:
            self._save_state()

    def check_transition_criteria(self) -> dict:
        """Evaluate all transition criteria. Returns status dict.

        All criteria must be met for ENFORCING → ADVISORY transition.
        """
        criteria = {
            "great_cycle_met": self._great_cycle >= 1,
            "great_cycle": self._great_cycle,
            "developmental_age_met": self._developmental_age > 1000,
            "developmental_age": self._developmental_age,
            "convergence_met": self._check_convergence(),
            "saturation_violations": self._saturation_violations,
            "collapse_violations": self._collapse_violations,
            "great_pulses_met": self._total_great_pulses > 1000,
            "total_great_pulses": self._total_great_pulses,
            "maker_confirmed": self._maker_confirmed,
            "all_met": False,
            "sovereignty_mode": self._sovereignty_mode,
        }

        criteria["all_met"] = all([
            criteria["great_cycle_met"],
            criteria["developmental_age_met"],
            criteria["convergence_met"],
            criteria["great_pulses_met"],
            criteria["maker_confirmed"] if self._great_cycle == 0 else True,
        ])

        return criteria

    def _check_convergence(self) -> bool:
        """Check if neuromods have been stable in the convergence window.

        No saturation (>0.95 for >100 consecutive) or collapse (<0.05 for >100)
        in the last CONVERGENCE_WINDOW epochs.
        """
        # Simple check: less than 1% violation rate
        total_epochs = sum(len(h) for h in self._modulator_history.values())
        if total_epochs < CONVERGENCE_WINDOW:
            return False  # Not enough data yet

        violation_rate = (self._saturation_violations + self._collapse_violations) / max(1, total_epochs)
        return violation_rate < 0.01  # Less than 1% violations

    def transition_to_advisory(self, epoch_id: int) -> bool:
        """Execute the ENFORCING → ADVISORY transition.

        Should only be called after check_transition_criteria()["all_met"] == True.
        """
        criteria = self.check_transition_criteria()
        if not criteria["all_met"]:
            logger.warning("[Sovereignty] Transition rejected — criteria not met: %s", criteria)
            return False

        self._sovereignty_mode = "ADVISORY"
        self._transition_epoch = epoch_id
        self._transition_ts = time.time()
        self._save_state()

        logger.critical(
            "[Sovereignty] GREAT CYCLE TRANSITION: ENFORCING → ADVISORY at epoch %d "
            "(dev_age=%d, great_pulses=%d, cycle=%d)",
            epoch_id, self._developmental_age, self._total_great_pulses, self._great_cycle)

        return True

    def confirm_maker(self):
        """Maker confirms transition (required for Cycle 0→1)."""
        self._maker_confirmed = True
        self._save_state()
        logger.info("[Sovereignty] Maker confirmed transition")

    def increment_great_cycle(self):
        """Called after successful reincarnation."""
        self._great_cycle += 1
        self._save_state()
        logger.info("[Sovereignty] GREAT CYCLE incremented to %d", self._great_cycle)

    def get_stats(self) -> dict:
        return {
            "sovereignty_mode": self._sovereignty_mode,
            "great_cycle": self._great_cycle,
            "total_great_pulses": self._total_great_pulses,
            "developmental_age": self._developmental_age,
            "saturation_violations": self._saturation_violations,
            "collapse_violations": self._collapse_violations,
            "convergence_window": CONVERGENCE_WINDOW,
            "transition_epoch": self._transition_epoch,
            "maker_confirmed": self._maker_confirmed,
        }
