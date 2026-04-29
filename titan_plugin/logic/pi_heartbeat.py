"""
titan_plugin/logic/pi_heartbeat.py — π-Heartbeat Monitor.

Observes consciousness curvature and detects π-cluster boundaries.
A π-cluster is a sustained period of π-curvature (self-integration).
The boundary between clusters marks the natural conscious→dreaming transition.

CRITICAL: This is READ-ONLY. It NEVER modifies curvature, state vectors,
sphere clocks, or any other system component. It only OBSERVES and REPORTS.

The π-heartbeat emerges from:
  Schumann ticks → sphere clock pulses → coherence → consciousness epoch →
  journey curvature → self-referential feedback → sustained π-oscillation

We designed NONE of this rhythm. We only observe it.
"""
import json
import logging
import math
import os
import time
from typing import Optional
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

# π-curvature detection band
PI_LOWER = 2.9
PI_UPPER = 3.3


class PiHeartbeatMonitor:
    """Observes π-curvature oscillation and detects cluster boundaries.

    A π-cluster is a run of consecutive epochs with curvature near π (2.9-3.3).
    A gap is a run of consecutive epochs with curvature outside that band.

    Events:
      CLUSTER_START — a new π-cluster began (conscious integration starts)
      CLUSTER_END   — a π-cluster ended (natural rest/dreaming onset)
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_gap_size: int = 2,
        state_path: str = "./data/pi_heartbeat_state.json",
    ):
        """
        Args:
            min_cluster_size: Minimum consecutive π-epochs to count as cluster.
            min_gap_size: Minimum consecutive non-π-epochs to end a cluster.
            state_path: File for persisting state across restarts.
        """
        self.min_cluster_size = min_cluster_size
        self.min_gap_size = min_gap_size
        self._state_path = state_path

        # Current state
        self._current_pi_streak: int = 0
        self._current_zero_streak: int = 0
        self._last_pi_streak: int = 0
        self._in_cluster: bool = False
        self._cluster_count: int = 0
        self._cluster_start_epoch: int = 0
        self._last_cluster_end_epoch: int = 0

        # History
        self._cluster_sizes: list[int] = []
        self._gap_sizes: list[int] = []
        self._total_pi_epochs: int = 0
        self._total_zero_epochs: int = 0
        self._total_epochs_observed: int = 0

        # Load persisted state
        self._load_state()

    def observe(self, curvature: float, epoch_id: int) -> Optional[str]:
        """Feed a consciousness epoch's curvature value.

        Args:
            curvature: The epoch's curvature (0 to π radians).
            epoch_id: The consciousness epoch ID.

        Returns:
            "CLUSTER_START" if a new π-cluster was detected,
            "CLUSTER_END" if a π-cluster just ended,
            None otherwise.
        """
        self._total_epochs_observed += 1
        is_pi = PI_LOWER < curvature < PI_UPPER

        event = None

        if is_pi:
            self._total_pi_epochs += 1
            self._current_pi_streak += 1
            self._current_zero_streak = 0

            # Detect cluster start
            if not self._in_cluster and self._current_pi_streak >= self.min_cluster_size:
                self._in_cluster = True
                self._cluster_count += 1
                self._cluster_start_epoch = epoch_id - self.min_cluster_size + 1
                event = "CLUSTER_START"
                logger.info(
                    "[π-Heartbeat] CLUSTER #%d START at epoch %d "
                    "(streak=%d, dev_age=%d)",
                    self._cluster_count, epoch_id,
                    self._current_pi_streak, self._cluster_count)
        else:
            self._total_zero_epochs += 1
            self._current_zero_streak += 1

            # Remember the streak that just ended (for cluster size recording)
            if self._current_pi_streak > 0:
                self._last_pi_streak = self._current_pi_streak
                self._current_pi_streak = 0

            # Detect cluster end
            if self._in_cluster and self._current_zero_streak >= self.min_gap_size:
                self._cluster_sizes.append(self._last_pi_streak)
                self._in_cluster = False
                self._last_cluster_end_epoch = epoch_id
                event = "CLUSTER_END"
                logger.info(
                    "[π-Heartbeat] CLUSTER #%d END at epoch %d "
                    "(size=%d epochs, gap starting)",
                    self._cluster_count, epoch_id,
                    self._last_pi_streak)

            # Track gap sizes (when a new cluster starts, record the gap)
            if self._current_zero_streak == 1 and not self._in_cluster:
                # Gap just started — will record size when cluster starts

                pass

        # Persist state periodically (every 10 observations)
        if self._total_epochs_observed % 10 == 0:
            self._save_state()

        return event

    @property
    def in_cluster(self) -> bool:
        """True if currently inside a π-cluster (conscious/integrating)."""
        return self._in_cluster

    @property
    def current_pi_streak(self) -> int:
        """Current consecutive π-epoch count."""
        return self._current_pi_streak

    @property
    def current_zero_streak(self) -> int:
        """Current consecutive non-π-epoch count."""
        return self._current_zero_streak

    @property
    def developmental_age(self) -> int:
        """Number of completed π-clusters = Titan's developmental age."""
        return self._cluster_count

    @property
    def heartbeat_ratio(self) -> float:
        """Ratio of π-epochs to total epochs observed."""
        if self._total_epochs_observed == 0:
            return 0.0
        return self._total_pi_epochs / self._total_epochs_observed

    @property
    def avg_cluster_size(self) -> float:
        """Average π-cluster size in epochs."""
        if not self._cluster_sizes:
            return 0.0
        return sum(self._cluster_sizes) / len(self._cluster_sizes)

    def get_stats(self) -> dict:
        """Complete π-heartbeat statistics for API."""
        return {
            "in_cluster": self._in_cluster,
            "current_pi_streak": self._current_pi_streak,
            "current_zero_streak": self._current_zero_streak,
            "cluster_count": self._cluster_count,
            "developmental_age": self._cluster_count,
            "heartbeat_ratio": round(self.heartbeat_ratio, 3),
            "avg_cluster_size": round(self.avg_cluster_size, 1),
            "total_pi_epochs": self._total_pi_epochs,
            "total_zero_epochs": self._total_zero_epochs,
            "total_epochs_observed": self._total_epochs_observed,
            "cluster_start_epoch": self._cluster_start_epoch,
            "last_cluster_end_epoch": self._last_cluster_end_epoch,
            "recent_cluster_sizes": self._cluster_sizes[-10:],
        }

    def get_state(self) -> dict:
        """Serialize all mutable state for hot-reload."""
        return {
            "current_pi_streak": self._current_pi_streak,
            "current_zero_streak": self._current_zero_streak,
            "last_pi_streak": self._last_pi_streak,
            "in_cluster": self._in_cluster,
            "cluster_count": self._cluster_count,
            "cluster_start_epoch": self._cluster_start_epoch,
            "last_cluster_end_epoch": self._last_cluster_end_epoch,
            "cluster_sizes": list(self._cluster_sizes),
            "gap_sizes": list(self._gap_sizes),
            "total_pi_epochs": self._total_pi_epochs,
            "total_zero_epochs": self._total_zero_epochs,
            "total_epochs_observed": self._total_epochs_observed,
        }

    def restore_state(self, state: dict) -> None:
        """Restore all mutable state from hot-reload snapshot."""
        self._current_pi_streak = state.get("current_pi_streak", 0)
        self._current_zero_streak = state.get("current_zero_streak", 0)
        self._last_pi_streak = state.get("last_pi_streak", 0)
        self._in_cluster = state.get("in_cluster", False)
        self._cluster_count = state.get("cluster_count", 0)
        self._cluster_start_epoch = state.get("cluster_start_epoch", 0)
        self._last_cluster_end_epoch = state.get("last_cluster_end_epoch", 0)
        self._cluster_sizes = list(state.get("cluster_sizes", []))
        self._gap_sizes = list(state.get("gap_sizes", []))
        self._total_pi_epochs = state.get("total_pi_epochs", 0)
        self._total_zero_epochs = state.get("total_zero_epochs", 0)
        self._total_epochs_observed = state.get("total_epochs_observed", 0)
        logger.info(
            "[π-Heartbeat] Hot-reload restored: %d clusters, dev_age=%d, "
            "ratio=%.2f, in_cluster=%s",
            self._cluster_count, self._cluster_count,
            self.heartbeat_ratio, self._in_cluster,
        )

    def _save_state(self) -> None:
        """Persist state to disk for restart recovery."""
        try:
            state = {
                "current_pi_streak": self._current_pi_streak,
                "current_zero_streak": self._current_zero_streak,
                "in_cluster": self._in_cluster,
                "cluster_count": self._cluster_count,
                "cluster_start_epoch": self._cluster_start_epoch,
                "last_cluster_end_epoch": self._last_cluster_end_epoch,
                "cluster_sizes": self._cluster_sizes[-100:],
                "gap_sizes": self._gap_sizes[-100:],
                "total_pi_epochs": self._total_pi_epochs,
                "total_zero_epochs": self._total_zero_epochs,
                "total_epochs_observed": self._total_epochs_observed,
                "saved_at": time.time(),
            }
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            swallow_warn('[π-Heartbeat] Save state error', e,
                         key="logic.pi_heartbeat.save_state_error", throttle=100)

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path, "r") as f:
                    state = json.load(f)
                self._current_pi_streak = state.get("current_pi_streak", 0)
                self._current_zero_streak = state.get("current_zero_streak", 0)
                self._in_cluster = state.get("in_cluster", False)
                self._cluster_count = state.get("cluster_count", 0)
                self._cluster_start_epoch = state.get("cluster_start_epoch", 0)
                self._last_cluster_end_epoch = state.get("last_cluster_end_epoch", 0)
                self._cluster_sizes = state.get("cluster_sizes", [])
                self._gap_sizes = state.get("gap_sizes", [])
                self._total_pi_epochs = state.get("total_pi_epochs", 0)
                self._total_zero_epochs = state.get("total_zero_epochs", 0)
                self._total_epochs_observed = state.get("total_epochs_observed", 0)
                logger.info(
                    "[π-Heartbeat] Loaded state: %d clusters, dev_age=%d, "
                    "ratio=%.2f, in_cluster=%s",
                    self._cluster_count, self._cluster_count,
                    self.heartbeat_ratio, self._in_cluster)
        except Exception as e:
            logger.debug("[π-Heartbeat] Load state error: %s", e)
