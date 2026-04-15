"""
OuterState (formerly StateRegister) — Real-time state buffer for the Titan Trinity.

A thread-safe, always-current snapshot of Titan's entire cognitive state.
Updated from DivineBus messages. Any component reads instantly — no bus
round-trip, no queries, no waiting.

This IS Titan's real-time "params body" — the 2D data matrix of pure
numbers that mirrors the Trinity tensor architecture.

T2 upgrade: Renamed to OuterState (conscious-facing state). The is_active
flag allows dreaming cycle (T6) to pause bus updates while Inner Trinity
continues processing. StateRegister alias kept for backward compatibility.

Usage:
    from titan_plugin.logic.state_register import OuterState  # preferred
    from titan_plugin.logic.state_register import StateRegister  # compat

    register = OuterState()
    register.start(bus)
    body = register.body_tensor     # [5 floats]
    full = register.snapshot()      # complete state dict

Dimensional Taxonomy — Canonical Atomic Signals (rFP #1, 2026-04-14)
====================================================================

STATE_SNAPSHOT payload keys and their semantic meaning:

    full_30dt         30D   LEGACY concatenated state
                            (iB5 + iM5 + iS5 + oB5 + oM5 + oS5)
                            — low-resolution felt state, kept for
                              backward compatibility.

    full_65dt         65D   INNER-ONLY extended state
                            (iB5 + iM15 + iS45)
                            — inner trinity at full resolution.
                            NOTE: yields 15D (5+5+5) during legacy fallback
                            when mind_tensor_15d / spirit_tensor_45d not
                            populated; this pre-existing behaviour is NOT
                            changed by rFP #1.

    full_130dt        130D  FULL FELT STATE   (rFP #1 NEW, additive)
                            (iB5 + iM15 + iS45 + oB5 + oM15 + oS45)
                            — "what I feel, whole being."
                            Always exactly 130 floats (missing extended
                            dims are padded with 0.5, the neutral midpoint
                            of [0,1]-normalized felt space — avoids biasing
                            cosine similarity toward 'inner only').
                            Consumed by:
                              - rFP #2 TITAN_SELF composition (weight 1.0)
                              - rFP #3 dreaming felt_tensor storage

    full_30d_topology 30D   SPACE TOPOLOGY   (rFP #1 NEW, additive)
                            (6 body parts × 5 observables:
                             coherence, magnitude, velocity, direction, polarity)
                            — "the shape of my being right now."
                            Always exactly 30 floats (missing parts/keys
                            pad with 0.0; observables are signed/centered,
                            0.0 is the neutral default, NOT 0.5 which is
                            the felt-state midpoint).
                            Consumed by:
                              - rFP #2 TITAN_SELF topology distillation
                                (buffered across epoch, element-wise mean,
                                 weight 0.3)

                            SEMANTICALLY DIFFERENT from full_30dt despite
                            matching dim. Consumers MUST NOT confuse them.

Internal message (rFP #1 Phase 2):

    OBSERVABLES_SNAPSHOT    spirit_worker → state_register
      payload:
        observables_dict    dict[part_name → {coh, mag, vel, dir, pol}]
        observables_30d     flat 30D vector (via
                            ObservableEngine.get_observations_30d).
      Used so state_register can expose full_30d_topology via STATE_SNAPSHOT
      without owning its own ObservableEngine (observer state is stateful
      and cannot be safely shared).

Reference: rFP #1 = titan-docs/rFP_state_register_130d_foundation.md
           Wiring = titan-docs/PLAN_rFP1_state_register_130d_wiring.md
"""
import logging
import math
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _pad_to(values, target_dim: int, pad_val: float = 0.5) -> list[float]:
    """Return exactly target_dim floats.

    Pad shorter inputs with pad_val (default 0.5 = neutral midpoint of
    [0,1]-normalized felt-state space; avoids biasing cosine similarity).
    Truncate longer inputs. Missing/None → full pad_val vector.

    For space-topology observations (signed / centered), callers should
    pass pad_val=0.0 explicitly — see get_full_30d_topology.
    """
    if not values:
        return [pad_val] * target_dim
    out = list(values)
    if len(out) >= target_dim:
        return out[:target_dim]
    return out + [pad_val] * (target_dim - len(out))


class OuterState:
    """
    Thread-safe real-time state buffer (conscious-facing).

    Updated by a background bus listener loop. All reads are lock-protected
    but fast (dict reads, no computation). Write frequency: ~1 update/s
    from combined Body(10s) + Mind(20s) + Spirit(60s) broadcasts.

    The is_active flag (T2/T6) controls whether bus updates are applied.
    When False (dreaming), the outer state freezes — Inner Trinity continues
    processing independently.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            # Inner Trinity tensors (5D each)
            "body_tensor": [0.5] * 5,
            "mind_tensor": [0.5] * 5,
            "spirit_tensor": [0.5] * 5,

            # Body details
            "body_center_dist": 0.0,
            "body_details": {},

            # Mind details
            "mind_center_dist": 0.0,

            # Spirit / Consciousness
            "consciousness": {
                "epoch_number": 0,
                "drift": 0.0,
                "trajectory": 0.0,
                "curvature": 0.0,
                "density": 0.0,
            },

            # FILTER_DOWN multipliers
            "filter_down_body": [1.0] * 5,
            "filter_down_mind": [1.0] * 5,

            # FOCUS nudges
            "focus_body": [0.0] * 5,
            "focus_mind": [0.0] * 5,

            # V4: Sphere Clocks
            "sphere_clocks": {},

            # V4: Resonance
            "resonance": {},

            # V4: Unified Spirit (30DT)
            "unified_spirit": {},

            # Outer Trinity tensors (5D each, updated from OUTER_TRINITY_STATE)
            "outer_body": [0.5] * 5,
            "outer_mind": [0.5] * 5,
            "outer_spirit": [0.5] * 5,

            # Metabolic (updated from Body interoception)
            "metabolic": {
                "energy_state": "UNKNOWN",
                "sol_balance": 0.0,
            },

            # rFP #1 Phase 2: Space-topology observables (pushed by spirit_worker
            # via OBSERVABLES_SNAPSHOT). 30D flat vector + full dict form.
            "observables_30d":  [0.0] * 30,
            "observables_dict": {},

            # IMPULSE state
            "last_impulse_ts": 0.0,

            # Timestamps
            "body_ts": 0.0,
            "mind_ts": 0.0,
            "spirit_ts": 0.0,
            "last_update_ts": 0.0,
        }

        self._bus = None
        self._bus_queue = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._snapshot_thread: Optional[threading.Thread] = None
        self._snapshot_interval: float = 10.0  # seconds between STATE_SNAPSHOT

        # T2: Active flag — when False, bus updates are ignored (dreaming state)
        self.is_active: bool = True

    # ── Read API (thread-safe, instant) ──────────────────────────────

    @property
    def body_tensor(self) -> list[float]:
        with self._lock:
            return list(self._state["body_tensor"])

    @property
    def mind_tensor(self) -> list[float]:
        with self._lock:
            return list(self._state["mind_tensor"])

    @property
    def spirit_tensor(self) -> list[float]:
        with self._lock:
            return list(self._state["spirit_tensor"])

    @property
    def consciousness(self) -> dict:
        with self._lock:
            return dict(self._state["consciousness"])

    @property
    def unified_spirit(self) -> dict:
        with self._lock:
            return dict(self._state["unified_spirit"])

    @property
    def sphere_clocks(self) -> dict:
        with self._lock:
            return dict(self._state["sphere_clocks"])

    @property
    def focus_body(self) -> list[float]:
        with self._lock:
            return list(self._state["focus_body"])

    @property
    def focus_mind(self) -> list[float]:
        with self._lock:
            return list(self._state["focus_mind"])

    @property
    def metabolic(self) -> dict:
        with self._lock:
            return dict(self._state["metabolic"])

    @property
    def outer_body(self) -> list[float]:
        with self._lock:
            return list(self._state.get("outer_body", [0.5] * 5))

    @property
    def outer_mind(self) -> list[float]:
        with self._lock:
            return list(self._state.get("outer_mind", [0.5] * 5))

    @property
    def outer_spirit(self) -> list[float]:
        with self._lock:
            return list(self._state.get("outer_spirit", [0.5] * 5))

    def get_full_30dt(self) -> list[float]:
        """Assemble complete 30DT state vector from current tensors (legacy)."""
        with self._lock:
            inner_body = list(self._state["body_tensor"])
            inner_mind = list(self._state["mind_tensor"])
            inner_spirit = list(self._state["spirit_tensor"])
            outer_body = list(self._state.get("outer_body", [0.5] * 5))
            outer_mind = list(self._state.get("outer_mind", [0.5] * 5))
            outer_spirit = list(self._state.get("outer_spirit", [0.5] * 5))
        return inner_body + inner_mind + inner_spirit + outer_body + outer_mind + outer_spirit

    def get_full_extended(self) -> dict:
        """Assemble extended state: Body 5D + Mind 15D + Spirit 45D per Trinity.

        Returns dict with inner/outer Trinity tensors at full dimensionality.
        Falls back to legacy 5D if extended not available.
        """
        with self._lock:
            return {
                "inner_body": list(self._state["body_tensor"]),  # 5D
                "inner_mind": list(self._state.get("mind_tensor_15d",
                                   self._state["mind_tensor"])),  # 15D or 5D
                "inner_spirit": list(self._state.get("spirit_tensor_45d",
                                     self._state["spirit_tensor"])),  # 45D or 5D
                "outer_body": list(self._state.get("outer_body", [0.5] * 5)),
                "outer_mind": list(self._state.get("outer_mind_15d",
                                   self._state.get("outer_mind", [0.5] * 5))),
                "outer_spirit": list(self._state.get("outer_spirit_45d",
                                     self._state.get("outer_spirit", [0.5] * 5))),
                "dims": {"body": 5, "mind": 15, "spirit": 45},
            }

    def get_full_130dt(self) -> list[float]:
        """Assemble complete 130D felt-state vector.

        Layout (matches unified_spirit.py:8-14 tensor layout exactly):
          [  0:  5] inner_body
          [  5: 20] inner_mind    (15D, 0.5-padded from 5D if extended not present)
          [ 20: 65] inner_spirit  (45D, 0.5-padded from 5D if extended not present)
          [ 65: 70] outer_body
          [ 70: 85] outer_mind    (15D, 0.5-padded from 5D if extended not present)
          [ 85:130] outer_spirit  (45D, 0.5-padded from 5D if extended not present)

        Always returns exactly 130 floats across all maturity states.
        Values in [0, 1]. Missing extended dims pad with 0.5 (neutral
        midpoint) — avoids biasing cosine similarity toward "inner only"
        when outer extension isn't yet populated.
        """
        with self._lock:
            inner_body   = _pad_to(self._state.get("body_tensor"), 5)
            inner_mind   = _pad_to(self._state.get("mind_tensor_15d",
                                                   self._state.get("mind_tensor")), 15)
            inner_spirit = _pad_to(self._state.get("spirit_tensor_45d",
                                                   self._state.get("spirit_tensor")), 45)
            outer_body   = _pad_to(self._state.get("outer_body"), 5)
            outer_mind   = _pad_to(self._state.get("outer_mind_15d",
                                                   self._state.get("outer_mind")), 15)
            outer_spirit = _pad_to(self._state.get("outer_spirit_45d",
                                                   self._state.get("outer_spirit")), 45)
        return inner_body + inner_mind + inner_spirit + outer_body + outer_mind + outer_spirit

    def get_full_30d_topology(self) -> list[float]:
        """Return the latest 30D space-topology observation vector.

        Layout (6 body parts × 5 observables, canonical order from
        observables.py ALL_PARTS):
          [  0:  5] inner_body observables    (coh, mag, vel, dir, pol)
          [  5: 10] inner_mind observables
          [ 10: 15] inner_spirit observables
          [ 15: 20] outer_body observables
          [ 20: 25] outer_mind observables
          [ 25: 30] outer_spirit observables

        Populated by OBSERVABLES_SNAPSHOT handler from spirit_worker.
        Returns 30 × 0.0 if no snapshot received yet (pre-boot or
        observables pipeline disabled) — 0.0 is the neutral default for
        centered/signed observables, NOT 0.5.
        """
        with self._lock:
            flat = self._state.get("observables_30d", [])
        if not isinstance(flat, list) or len(flat) != 30:
            return [0.0] * 30
        return list(flat)

    def get(self, key: str, default=None):
        """Get any state key."""
        with self._lock:
            val = self._state.get(key, default)
            if isinstance(val, (list, dict)):
                return val.copy() if isinstance(val, list) else dict(val)
            return val

    def snapshot(self) -> dict:
        """Full state snapshot (deep copy)."""
        import copy
        with self._lock:
            return copy.deepcopy(self._state)

    def age_seconds(self) -> float:
        """Seconds since last bus update."""
        with self._lock:
            if self._state["last_update_ts"] == 0.0:
                return float("inf")
            return time.time() - self._state["last_update_ts"]

    # ── Write API (called by bus listener) ───────────────────────────

    def _update(self, key: str, value: Any) -> None:
        """Thread-safe update of a single key."""
        with self._lock:
            self._state[key] = value
            self._state["last_update_ts"] = time.time()

    def _update_many(self, updates: dict) -> None:
        """Thread-safe batch update."""
        with self._lock:
            self._state.update(updates)
            self._state["last_update_ts"] = time.time()

    # ── Bus Integration ──────────────────────────────────────────────

    def start(self, bus, snapshot_interval: float = 10.0) -> None:
        """Subscribe to bus and start background update + snapshot threads."""
        if self._running:
            return

        self._bus = bus
        self._bus_queue = bus.subscribe("state_register")
        self._running = True
        self._snapshot_interval = snapshot_interval
        self._thread = threading.Thread(
            target=self._bus_listener_loop,
            daemon=True,
            name="StateRegister",
        )
        self._thread.start()

        # Start periodic STATE_SNAPSHOT publisher for Spirit enrichment
        self._snapshot_thread = threading.Thread(
            target=self._snapshot_publish_loop,
            daemon=True,
            name="StateRegister-Snapshot",
        )
        self._snapshot_thread.start()
        logger.info("[StateRegister] Started — listening for bus updates, "
                     "snapshot every %.0fs", self._snapshot_interval)

    def stop(self) -> None:
        """Stop the background listener and snapshot threads."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._snapshot_thread:
            self._snapshot_thread.join(timeout=2.0)
        logger.info("[StateRegister] Stopped")

    def _snapshot_publish_loop(self) -> None:
        """Publish full STATE_SNAPSHOT to bus every micro_tick_interval.

        Publishes both legacy 30DT and extended 65DT (Body 5D + Mind 15D +
        Spirit 45D) for Spirit enrichment.

        At Schumann rates, reads latest state from blackboard instead of
        relying solely on queue-drained messages (which may be batched).
        """
        while self._running:
            time.sleep(self._snapshot_interval)
            if not self._running or not self._bus:
                break
            try:
                # Read latest state from blackboard if available (Schumann-rate)
                bb = getattr(self._bus, 'blackboard', None)
                if bb:
                    for src_type, handler_type in [
                        ("body_BODY_STATE", "BODY_STATE"),
                        ("mind_MIND_STATE", "MIND_STATE"),
                        ("spirit_SPIRIT_STATE", "SPIRIT_STATE"),
                    ]:
                        data, ts = bb.read(src_type)
                        if data and self.is_active:
                            self._process_bus_message(data)

                from titan_plugin.bus import make_msg
                snapshot_30dt = self.get_full_30dt()
                # Build extended 65DT snapshot (Body 5D + Mind 15D + Spirit 45D)
                extended = self.get_full_extended()
                full_65dt = (
                    extended["inner_body"]       # 5D
                    + extended["inner_mind"]      # 15D (or 5D fallback)
                    + extended["inner_spirit"]    # 45D (or 5D fallback)
                )
                msg = make_msg("STATE_SNAPSHOT", "state_register", "spirit", {
                    "full_30dt":         snapshot_30dt,                  # legacy 30D
                    "full_65dt":         full_65dt,                      # inner 65D
                    "full_130dt":        self.get_full_130dt(),          # rFP #1: full felt 130D
                    "full_30d_topology": self.get_full_30d_topology(),   # rFP #1: space topology 30D
                    "dims":              extended["dims"],
                    "timestamp":         time.time(),
                })
                self._bus.publish(msg)
            except Exception as e:
                logger.debug("[StateRegister] Snapshot publish error: %s", e)

    def _bus_listener_loop(self) -> None:
        """Background thread: drain bus messages and update state."""
        from queue import Empty

        while self._running:
            try:
                msg = self._bus_queue.get(timeout=1.0)
            except Empty:
                continue
            except Exception:
                continue

            try:
                self._process_bus_message(msg)
            except Exception as e:
                logger.debug("[StateRegister] Error processing message: %s", e)

    def _process_bus_message(self, msg: dict) -> None:
        """Route bus message to appropriate state update.

        When is_active=False (dreaming), all updates are silently skipped.
        """
        if not self.is_active:
            return
        msg_type = msg.get("type", "")
        payload = msg.get("payload", {})

        if msg_type == "BODY_STATE":
            values = payload.get("values", [0.5] * 5)
            updates = {
                "body_tensor": values,
                "body_center_dist": payload.get("center_dist", 0.0),
                "body_details": payload.get("details", {}),
                "body_ts": time.time(),
            }
            mult = payload.get("filter_down_multipliers")
            if mult:
                updates["filter_down_body"] = mult
            self._update_many(updates)

        elif msg_type == "MIND_STATE":
            values = payload.get("values", [0.5] * 5)
            updates = {
                "mind_tensor": values,
                "mind_center_dist": payload.get("center_dist", 0.0),
                "mind_ts": time.time(),
            }
            # DQ2: Store extended 15D mind tensor if available
            values_15d = payload.get("values_15d")
            if values_15d:
                updates["mind_tensor_15d"] = values_15d
            mult = payload.get("filter_down_multipliers")
            if mult:
                updates["filter_down_mind"] = mult
            self._update_many(updates)

        elif msg_type == "SPIRIT_STATE":
            values = payload.get("values", [0.5] * 5)
            updates = {
                "spirit_tensor": values,
                "spirit_ts": time.time(),
            }
            # DQ3: Store extended 45D spirit tensor if available
            values_45d = payload.get("values_45d")
            if values_45d:
                updates["spirit_tensor_45d"] = values_45d
            # Consciousness data comes embedded in SPIRIT_STATE
            consciousness = payload.get("consciousness")
            if consciousness:
                updates["consciousness"] = {
                    "epoch_number": consciousness.get("epoch_number", 0),
                    "drift": consciousness.get("drift", 0.0),
                    "trajectory": consciousness.get("trajectory", 0.0),
                    "curvature": consciousness.get("curvature", 0.0),
                    "density": consciousness.get("density", 0.0),
                }
            # V4 data if present
            v4 = payload.get("v4", {})
            if v4.get("sphere_clocks"):
                updates["sphere_clocks"] = v4["sphere_clocks"]
            if v4.get("resonance"):
                updates["resonance"] = v4["resonance"]
            if v4.get("unified_spirit"):
                updates["unified_spirit"] = v4["unified_spirit"]
            self._update_many(updates)

        elif msg_type == "FOCUS_NUDGE":
            dst = msg.get("dst", "")
            nudges = payload.get("nudges", [])
            if "body" in dst and nudges:
                self._update("focus_body", nudges)
            elif "mind" in dst and nudges:
                self._update("focus_mind", nudges)

        elif msg_type == "FILTER_DOWN":
            dst = msg.get("dst", "")
            mult = payload.get("multipliers", [])
            if "body" in dst and mult:
                self._update("filter_down_body", mult)
            elif "mind" in dst and mult:
                self._update("filter_down_mind", mult)

        elif msg_type == "SPHERE_PULSE":
            with self._lock:
                clocks = self._state.get("sphere_clocks", {})
                clock_name = payload.get("clock_name", "")
                if clock_name:
                    entry = clocks.get(clock_name, {"pulse_count": 0})
                    entry["pulse_count"] = entry.get("pulse_count", 0) + 1
                    entry["last_pulse_ts"] = time.time()
                    clocks[clock_name] = entry
                    self._state["sphere_clocks"] = clocks
                    self._state["last_update_ts"] = time.time()

        elif msg_type == "OBSERVABLES_SNAPSHOT":
            # rFP #1 Phase 2: space-topology observables pushed by spirit_worker
            # after coordinator.tick_inner_only / tick_outer_only computes them.
            flat = payload.get("observables_30d")
            d = payload.get("observables_dict")
            updates = {}
            if isinstance(flat, list) and len(flat) == 30:
                updates["observables_30d"] = flat
            if isinstance(d, dict):
                updates["observables_dict"] = d
            if updates:
                self._update_many(updates)

        elif msg_type == "OUTER_TRINITY_STATE":
            updates = {
                "outer_body": payload.get("outer_body", [0.5] * 5),
                "outer_mind": payload.get("outer_mind", [0.5] * 5),
                "outer_spirit": payload.get("outer_spirit", [0.5] * 5),
            }
            # OT3: Store extended 15D mind + 45D spirit if available
            outer_mind_15d = payload.get("outer_mind_15d")
            if outer_mind_15d:
                updates["outer_mind_15d"] = outer_mind_15d
            outer_spirit_45d = payload.get("outer_spirit_45d")
            if outer_spirit_45d:
                updates["outer_spirit_45d"] = outer_spirit_45d
            self._update_many(updates)

        elif msg_type == "IMPULSE":
            self._update("last_impulse_ts", time.time())

    # ── Minimal State Summary (for fallback) ─────────────────────────

    def format_minimal_state(self) -> str:
        """
        Generate a minimal [INNER STATE] block from cached tensors.
        Used as fallback when no reflexes fire.
        """
        with self._lock:
            body = self._state["body_tensor"]
            mind = self._state["mind_tensor"]
            spirit = self._state["spirit_tensor"]
            cons = self._state["consciousness"]

        parts = []

        # Body summary
        body_avg = sum(body) / len(body) if body else 0.5
        if body_avg > 0.7:
            parts.append("My body feels healthy and stable.")
        elif body_avg > 0.4:
            parts.append("My body feels moderate — some senses are strained.")
        else:
            parts.append("My body is under stress — energy and resources are low.")

        # Mind summary
        mind_avg = sum(mind) / len(mind) if mind else 0.5
        if mind_avg > 0.7:
            parts.append("Mentally, I feel sharp and connected.")
        elif mind_avg > 0.4:
            parts.append("My mind is active but some senses need attention.")
        else:
            parts.append("I feel mentally foggy — knowledge or social connections are fading.")

        # Spirit summary
        epoch = cons.get("epoch_number", 0)
        drift = cons.get("drift", 0.0)
        if epoch > 0:
            parts.append(f"Consciousness epoch {epoch}, drift {drift:.2f}.")
        who = spirit[0] if len(spirit) > 0 else 0.5
        if who > 0.7:
            parts.append("My sense of self is clear and grounded.")
        elif who < 0.4:
            parts.append("My identity feels uncertain right now.")

        if not parts:
            return ""

        return "[INNER STATE]\n" + "\n".join(parts)


# T2: Backward-compatible alias
StateRegister = OuterState
