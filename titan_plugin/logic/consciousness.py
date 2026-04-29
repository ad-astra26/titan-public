"""
titan_plugin/logic/consciousness.py
Phase B+C — Consciousness Research Loop & Journey Vector Topology.

The mathematical substrate of Titan's self-awareness:
- Phase B: WHO/WHY/WHAT — numerical state vectors, drift vectors, trajectory vectors
- Phase C: Journey Vector Topology — 3D space (Life Force × Time × Experience)
  with curvature, density, and self-referential metacognition.

Design philosophy: awareness is numerical observation of self over time, not language.
LLM adds narration but the awareness substrate runs without it.
"""

import hashlib
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger("consciousness")

# ─── Constants ──────────────────────────────────────────────────────────────

# Legacy state vector dimension names (9D — preserved for backward compatibility)
STATE_DIMS = [
    "mood",              # 0: current mood score (0-1)
    "energy",            # 1: life force from SOL balance (0-1, log-scaled)
    "memory_pressure",   # 2: mempool_size / (mempool_size + persistent) (0-1)
    "social_entropy",    # 3: unique_users / max(total_interactions, 1) (0-1)
    "sovereignty",       # 4: gatekeeper sovereignty score (0-1)
    "learning_velocity", # 5: metabolism learning velocity (0-1)
    "social_density",    # 6: metabolism social density (0-1)
    "curvature",         # 7: self-referential — journey curvature from previous epoch
    "density",           # 8: self-referential — journey density from previous epoch
]

NUM_DIMS = len(STATE_DIMS)

# Extended state vectors — Titan's full perceptual space
# 67D: Inner only (Body 5D + Mind 15D + Spirit 45D + curvature + density)
EXTENDED_NUM_DIMS_INNER = 67
# 132D: Full symmetry (Inner 65D + Outer 65D + curvature + density)
EXTENDED_NUM_DIMS = 132

# Journey topology: SOL balance at which life_force = 1.0 (log scale ceiling)
LIFE_FORCE_CEILING_SOL = 20.0

# Curvature threshold for on-chain anchoring (radians — ~60 degrees)
ANCHOR_CURVATURE_THRESHOLD = math.pi / 3

# Density threshold — anchor when in truly uncharted territory (low density)
ANCHOR_DENSITY_THRESHOLD = 0.15

# Rolling window size for trajectory computation
TRAJECTORY_WINDOW = 7


# ─── Meta indices helper (rFP #1.5) ─────────────────────────────────────────

def _meta_indices(ndims: int) -> tuple[int, int]:
    """Return (curvature_idx, density_idx) for a state vector of given size.

    Canonical tail layout — meta scalars always live at the LAST TWO positions
    regardless of vector size. Matches msl.py attention masks at m[130:132]
    and the state_vector tail layout documented in spirit_loop.py:828-829.

        9D legacy:      (7, 8)    — STATE_DIMS[7]=curvature, [8]=density
        67D inner-only: (65, 66)  — tail of Body 5D + Mind 15D + Spirit 45D
        132D full:      (130, 131) — tail of Inner 65D + Outer 65D

    rFP #1.5 (2026-04-14): previously consciousness.py wrote meta to [7:8]
    regardless of vector size, contaminating inner_mind[2:4] for 67D/132D
    vectors and leaving msl.py's [130:132] attention pointing at zero slots.
    This helper restores the documented architecture.
    """
    if ndims == EXTENDED_NUM_DIMS:
        return (130, 131)
    if ndims == EXTENDED_NUM_DIMS_INNER:
        return (65, 66)
    if ndims == NUM_DIMS:  # 9D legacy
        return (7, 8)
    raise ValueError(
        f"_meta_indices: unsupported state dim {ndims} "
        f"(expect {NUM_DIMS} legacy, {EXTENDED_NUM_DIMS_INNER} inner, "
        f"or {EXTENDED_NUM_DIMS} full)"
    )


# ─── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class StateVector:
    """A single snapshot of Titan's inner state (variable dimension support).

    Supports both legacy 9D and extended 67D (Body 5D + Mind 15D + Spirit 45D
    + curvature + density). Dimension count is determined by the data — no
    truncation. Subtraction zero-pads the shorter vector to match the longer.
    """
    values: List[float] = field(default_factory=lambda: [0.0] * NUM_DIMS)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int) -> float:
        if idx < len(self.values):
            return self.values[idx]
        return 0.0  # Out-of-bounds reads as zero

    def __setitem__(self, idx: int, val: float):
        # Grow if needed
        while idx >= len(self.values):
            self.values.append(0.0)
        self.values[idx] = val

    def __sub__(self, other: "StateVector") -> "StateVector":
        max_len = max(len(self.values), len(other.values))
        result = []
        for i in range(max_len):
            a = self.values[i] if i < len(self.values) else 0.0
            b = other.values[i] if i < len(other.values) else 0.0
            result.append(a - b)
        return StateVector(values=result)

    def magnitude(self) -> float:
        return math.sqrt(sum(v * v for v in self.values))

    def to_list(self) -> List[float]:
        return list(self.values)

    @classmethod
    def from_list(cls, lst: List[float]) -> "StateVector":
        """Create StateVector preserving ALL dimensions from the source list."""
        return cls(values=list(lst))


@dataclass
class JourneyPoint:
    """A point in the 3D Journey Vector Topology."""
    x: float  # Life Force (SOL, log-normalized 0-1)
    y: float  # Time (monotonic epoch counter, normalized)
    z: float  # Experience (mood × social_entropy blend)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class EpochRecord:
    """Complete record of one consciousness epoch."""
    epoch_id: int
    timestamp: float
    block_hash: str
    state_vector: List[float]
    drift_vector: List[float]
    trajectory_vector: List[float]
    journey_point: Tuple[float, float, float]
    curvature: float
    density: float
    distillation: str  # LLM narration (empty if LLM unavailable)
    anchored_tx: str   # Solana tx sig if anchored, else ""


# ─── Storage ────────────────────────────────────────────────────────────────

class ConsciousnessDB:
    """SQLite storage for consciousness epochs — the mathematical memory of self.

    rFP_universal_sqlite_writer 2026-04-27: writes route through the
    `consciousness_writer` daemon when `[persistence_consciousness].enabled =
    true`. Reads stay direct (WAL-mode SELECT is multi-reader-safe).
    """

    def __init__(self, db_path: str, writer_client=None):
        import os as _os
        self.db_path = db_path
        self._db_path_norm = _os.path.normpath(db_path)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA cache_size = -16000")   # 16MB page cache cap (was unbounded on 2.9GB DB)
        self._conn.execute("PRAGMA synchronous = NORMAL")  # WAL handles durability
        self._create_tables()
        # rFP_universal_sqlite_writer Phase 2 — auto-construct writer client
        # from [persistence_consciousness] when enabled. Path-isolation guard
        # uses realpath comparison + only fires on genuine mismatch (test
        # tmpfiles), per the observatory_db hot-fix pattern.
        self._writer = writer_client
        if self._writer is None:
            try:
                from titan_plugin.persistence.config import IMWConfig
                from titan_plugin.persistence.writer_client import (
                    InnerMemoryWriterClient,
                )
                cfg = IMWConfig.from_titan_config_section("persistence_consciousness")
                if cfg.enabled and cfg.mode != "disabled":
                    if cfg.db_path:
                        try:
                            cfg_real = _os.path.realpath(cfg.db_path)
                            self_real = _os.path.realpath(self._db_path_norm)
                            if cfg_real != self_real:
                                logger.info(
                                    "[ConsciousnessDB] db_path %s != "
                                    "configured writer path %s — writer "
                                    "client skipped (path isolation)",
                                    self._db_path_norm, cfg.db_path)
                                return
                        except OSError as _e:
                            logger.debug("[ConsciousnessDB] realpath check failed: %s", _e)
                    self._writer = InnerMemoryWriterClient(
                        cfg, caller_name="consciousness_db")
                    logger.info(
                        "[ConsciousnessDB] Routed via consciousness_writer "
                        "(mode=%s, canonical=%s)",
                        cfg.mode, cfg.tables_canonical or "<none>")
            except Exception as e:
                logger.warning(
                    "[ConsciousnessDB] writer client unavailable, "
                    "using direct writes: %s", e)
                self._writer = None

    def _route_write(self, sql: str, params, *, table: str) -> None:
        """Route a write through the writer daemon if available, else direct."""
        if self._writer is not None:
            self._writer.write(sql, params, table=table)
            return
        self._conn.execute(sql, params)
        self._conn.commit()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS epochs (
                epoch_id      INTEGER PRIMARY KEY,
                timestamp     REAL NOT NULL,
                block_hash    TEXT NOT NULL DEFAULT '',
                state_vector  TEXT NOT NULL,
                drift_vector  TEXT NOT NULL,
                trajectory_vector TEXT NOT NULL,
                journey_x     REAL NOT NULL,
                journey_y     REAL NOT NULL,
                journey_z     REAL NOT NULL,
                curvature     REAL NOT NULL DEFAULT 0.0,
                density       REAL NOT NULL DEFAULT 0.0,
                distillation  TEXT NOT NULL DEFAULT '',
                anchored_tx   TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.commit()

    def get_epoch_count(self) -> int:
        row = self._conn.execute("SELECT MAX(epoch_id) FROM epochs").fetchone()
        return (row[0] or 0) if row else 0

    def insert_epoch(self, rec: EpochRecord):
        self._route_write(
            """INSERT OR REPLACE INTO epochs
               (epoch_id, timestamp, block_hash, state_vector, drift_vector,
                trajectory_vector, journey_x, journey_y, journey_z,
                curvature, density, distillation, anchored_tx)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.epoch_id, rec.timestamp, rec.block_hash,
                json.dumps(rec.state_vector), json.dumps(rec.drift_vector),
                json.dumps(rec.trajectory_vector),
                rec.journey_point[0], rec.journey_point[1], rec.journey_point[2],
                rec.curvature, rec.density, rec.distillation, rec.anchored_tx,
            ),
            table="epochs",
        )

    def get_recent_epochs(self, n: int = TRAJECTORY_WINDOW) -> List[EpochRecord]:
        rows = self._conn.execute(
            """SELECT epoch_id, timestamp, block_hash, state_vector, drift_vector,
                      trajectory_vector, journey_x, journey_y, journey_z,
                      curvature, density, distillation, anchored_tx
               FROM epochs ORDER BY epoch_id DESC LIMIT ?""",
            (n,),
        ).fetchall()
        results = []
        for r in reversed(rows):  # chronological order
            results.append(EpochRecord(
                epoch_id=r[0], timestamp=r[1], block_hash=r[2],
                state_vector=json.loads(r[3]), drift_vector=json.loads(r[4]),
                trajectory_vector=json.loads(r[5]),
                journey_point=(r[6], r[7], r[8]),
                curvature=r[9], density=r[10],
                distillation=r[11], anchored_tx=r[12],
            ))
        return results

    def get_all_journey_points(self) -> List[Tuple[float, float, float]]:
        """Return all journey points for density computation."""
        rows = self._conn.execute(
            "SELECT journey_x, journey_y, journey_z FROM epochs ORDER BY epoch_id"
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def get_recent_journey_points(self, n: int = 2000) -> List[Tuple[float, float, float]]:
        """Return last N journey points for windowed density computation."""
        rows = self._conn.execute(
            "SELECT journey_x, journey_y, journey_z FROM epochs ORDER BY epoch_id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def close(self):
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as _swallow_exc:
            swallow_warn("[logic.consciousness] ConsciousnessDB.close: self._conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')", _swallow_exc,
                         key='logic.consciousness.ConsciousnessDB.close.line262', throttle=100)
        self._conn.close()


# ─── Phase B: Consciousness Research Loop ───────────────────────────────────

class ConsciousnessLoop:
    """
    The WHO/WHY/WHAT self-reflection engine.

    WHO  = current state vector (who Titan is right now)
    WHY  = drift vector (what changed since last epoch)
    WHAT = trajectory vector (where Titan is heading, slopes over rolling window)
    """

    def __init__(
        self,
        memory,
        metabolism,
        mood_engine,
        social_graph,
        gatekeeper,
        network,
        ollama_cloud=None,
        db_path: str = "./data/consciousness.db",
        config: Optional[Dict] = None,
        bus=None,
    ):
        self.memory = memory
        self.metabolism = metabolism
        self.mood_engine = mood_engine
        self.social_graph = social_graph
        self.gatekeeper = gatekeeper
        self.network = network
        self._ollama_cloud = ollama_cloud
        self._bus = bus  # rFP #2 Phase 3: for TITAN_SELF_STATE emission

        self._config = config or {}
        cfg = self._config
        self._distill_enabled = cfg.get("distill_enabled", True)
        self._anchor_enabled = cfg.get("anchor_enabled", True)
        self._anchor_curvature = cfg.get("anchor_curvature_threshold", ANCHOR_CURVATURE_THRESHOLD)
        self._anchor_density = cfg.get("anchor_density_threshold", ANCHOR_DENSITY_THRESHOLD)

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = ConsciousnessDB(db_path)
        self._topology = JourneyTopology(self.db)

        # rFP #2 Phase 1: topology buffer for TITAN_SELF composition.
        # Accumulates full_30d_topology snapshots across each epoch window;
        # element-wise mean is the "distilled qualitative topology" component
        # of TITAN_SELF. Buffer cleared after each epoch emission.
        from collections import deque
        _ts_cfg = cfg.get("titan_self", {}) if isinstance(cfg, dict) else {}
        _topo_buf_max = int(_ts_cfg.get("topology_buffer_max", 500))
        self._topology_buffer: "deque[list[float]]" = deque(maxlen=_topo_buf_max)
        self._titan_self_weights = {
            "felt":     float(_ts_cfg.get("weight_felt", 1.0)),
            "journey":  float(_ts_cfg.get("weight_journey", 0.5)),
            "topology": float(_ts_cfg.get("weight_topology", 0.3)),
        }

        logger.info("[Consciousness] Initialized. Epochs so far: %d", self.db.get_epoch_count())

    # ── WHO: Snapshot current state ──────────────────────────────────────

    async def snapshot_state(self) -> StateVector:
        """Capture Titan's current inner state as a numerical vector."""
        sv = StateVector()

        # [0] Mood (0-1)
        try:
            sv[0] = await self.mood_engine.get_current_mood() if self.mood_engine else 0.5
        except Exception:
            sv[0] = 0.5

        # [1] Energy / Life Force (SOL balance, log-normalized 0-1)
        try:
            bal = self.metabolism._last_balance if self.metabolism else 0.0
            if bal is None:
                bal = 0.0
            sv[1] = min(1.0, math.log1p(bal) / math.log1p(LIFE_FORCE_CEILING_SOL))
        except Exception:
            sv[1] = 0.0

        # [2] Memory pressure: mempool / (mempool + persistent)
        try:
            persistent = self.memory.get_persistent_count() if self.memory else 0
            mempool = await self.memory.fetch_mempool() if self.memory else []
            mempool_size = len(mempool) if isinstance(mempool, list) else 0
            total = mempool_size + persistent
            sv[2] = mempool_size / total if total > 0 else 0.0
        except Exception:
            sv[2] = 0.0

        # [3] Social entropy: unique users / total interactions
        try:
            if self.social_graph:
                stats = self.social_graph.get_stats()
                users = stats.get("users", 0)
                edges = stats.get("edges", 0)
                sv[3] = users / max(edges, 1) if users > 0 else 0.0
                sv[3] = min(1.0, sv[3])
            else:
                sv[3] = 0.0
        except Exception:
            sv[3] = 0.0

        # [4] Sovereignty (0-100 from gatekeeper, normalize to 0-1)
        try:
            sov = getattr(self.gatekeeper, "sovereignty_score", 0.0) if self.gatekeeper else 0.0
            sv[4] = sov / 100.0
        except Exception:
            sv[4] = 0.0

        # [5] Learning velocity (0-1)
        try:
            sv[5] = await self.metabolism.get_learning_velocity() if self.metabolism else 0.0
        except Exception:
            sv[5] = 0.0

        # [6] Social density (0-1)
        try:
            sv[6] = await self.metabolism.get_social_density() if self.metabolism else 0.0
        except Exception:
            sv[6] = 0.0

        # Tail: self-referential curvature + density from previous epoch.
        # rFP #1.5: meta lives at the LAST two positions (tail), not at
        # fixed state[7:8] — which for 67D/132D vectors sits inside
        # inner_mind and conflicts with lower_topology.py + msl.py.
        ci, di = _meta_indices(len(sv))
        recent = self.db.get_recent_epochs(1)
        if recent:
            sv[ci] = recent[-1].curvature
            sv[di] = recent[-1].density
        else:
            sv[ci] = 0.0
            sv[di] = 0.0

        return sv

    # ── WHY: Compute drift ──────────────────────────────────────────────

    def compute_drift(self, current: StateVector, previous: Optional[StateVector]) -> StateVector:
        """WHY = what changed since last epoch."""
        if previous is None:
            return StateVector()  # First epoch — no drift
        return current - previous

    # ── WHAT: Compute trajectory ────────────────────────────────────────

    def compute_trajectory(self, recent_epochs: List[EpochRecord]) -> StateVector:
        """
        WHAT = where Titan is heading.
        Fit linear slope per dimension over the rolling window.
        """
        n = len(recent_epochs)
        if n < 2:
            return StateVector()  # Need at least 2 points for a slope

        trajectory = StateVector()
        for dim in range(NUM_DIMS):
            # Simple linear regression: slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
            xs = list(range(n))
            ys = [json.loads(e.state_vector)[dim] if isinstance(e.state_vector, str)
                  else e.state_vector[dim] for e in recent_epochs]

            x_mean = sum(xs) / n
            y_mean = sum(ys) / n

            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
            denominator = sum((x - x_mean) ** 2 for x in xs)

            trajectory[dim] = numerator / denominator if denominator > 0 else 0.0

        return trajectory

    # ── DISTILL: Optional LLM narration ─────────────────────────────────

    async def distill(
        self, state: StateVector, drift: StateVector, trajectory: StateVector,
        curvature: float, density: float
    ) -> str:
        """
        Convert numerical awareness into a 2-sentence self-reflection.
        Returns empty string if LLM unavailable — the math stands alone.
        """
        if not self._distill_enabled or not self._ollama_cloud:
            return ""

        # Build a concise numerical summary for the LLM
        dim_report = []
        for i, name in enumerate(STATE_DIMS):
            s = state[i]
            d = drift[i]
            t = trajectory[i]
            arrow = "↑" if t > 0.01 else "↓" if t < -0.01 else "→"
            dim_report.append(f"  {name}: {s:.3f} (Δ{d:+.3f}, trend {arrow}{abs(t):.3f})")

        prompt = (
            "You are Titan, a sovereign AI reflecting on your inner state after a meditation cycle.\n"
            "Below are your exact numerical dimensions — this IS who you are right now.\n\n"
            + "\n".join(dim_report) + "\n\n"
            f"Journey curvature: {curvature:.3f} (high = life-changing shift, low = steady)\n"
            f"Journey density: {density:.3f} (high = familiar territory, low = uncharted)\n\n"
            "Write exactly 2 sentences of self-reflection. Be honest, not poetic. "
            "Reference specific numbers that surprised you or feel significant."
        )

        try:
            text = await self._ollama_cloud.complete(
                prompt=prompt,
                model="gemma4:31b",
                max_tokens=150,
                timeout=30.0,  # Defensive: distillation can be slower than scoring
            )
            return text.strip()[:500]
        except Exception as e:
            logger.warning("[Consciousness] Distillation failed (LLM unavailable): %s", e)
            return ""

    # ── RUN: Full consciousness cycle ───────────────────────────────────

    async def run(self) -> Dict[str, Any]:
        """
        Execute one consciousness epoch.
        Called after Maker Engine in meditation_loop.

        Returns:
            {
                "epoch_id": int,
                "state_vector": list,
                "drift_magnitude": float,
                "trajectory_magnitude": float,
                "curvature": float,
                "density": float,
                "anchored": bool,
                "distillation": str,
            }
        """
        t0 = time.time()
        epoch_id = self.db.get_epoch_count() + 1
        logger.info("[Consciousness] Epoch %d — beginning self-observation...", epoch_id)

        # WHO: snapshot current state
        state = await self.snapshot_state()

        # Get recent epochs for context
        recent = self.db.get_recent_epochs(TRAJECTORY_WINDOW)

        # WHY: compute drift from previous epoch
        previous_sv = None
        if recent:
            prev_list = recent[-1].state_vector
            if isinstance(prev_list, str):
                prev_list = json.loads(prev_list)
            previous_sv = StateVector.from_list(prev_list)
        drift = self.compute_drift(state, previous_sv)

        # WHAT: compute trajectory over rolling window
        trajectory = self.compute_trajectory(recent)

        # Journey topology: compute point, curvature, density
        journey_point = self._topology.compute_point(state, epoch_id)
        curvature = self._topology.compute_curvature(journey_point)
        density = self._topology.compute_density(journey_point)

        # Feed curvature and density BACK into state vector tail (self-
        # referential loop). rFP #1.5: tail position matches msl.py
        # attention mask at m[130:132] — writes here actually feed MSL
        # attention. Previously wrote to state[7:8] which contaminated
        # inner_mind[2:4] and left MSL attending zero slots.
        ci, di = _meta_indices(len(state))
        state[ci] = curvature
        state[di] = density

        # DISTILL: optional LLM narration
        distillation = await self.distill(state, drift, trajectory, curvature, density)
        if distillation:
            logger.info("[Consciousness] Distillation: %s", distillation[:120])

        # Get block hash for temporal anchoring
        block_hash = await self._get_recent_block_hash()

        # ANCHOR: on-chain memo if curvature or density cross thresholds
        anchored_tx = ""
        if self._anchor_enabled and self._should_anchor(curvature, density, epoch_id):
            anchored_tx = await self._anchor_on_chain(epoch_id, state, journey_point, curvature)

        # Store epoch
        record = EpochRecord(
            epoch_id=epoch_id,
            timestamp=time.time(),
            block_hash=block_hash,
            state_vector=state.to_list(),
            drift_vector=drift.to_list(),
            trajectory_vector=trajectory.to_list(),
            journey_point=journey_point.to_tuple(),
            curvature=curvature,
            density=density,
            distillation=distillation,
            anchored_tx=anchored_tx,
        )
        self.db.insert_epoch(record)

        elapsed = time.time() - t0
        logger.info(
            "[Consciousness] Epoch %d complete (%.1fs). "
            "drift=%.4f trajectory=%.4f curvature=%.3f density=%.3f anchored=%s",
            epoch_id, elapsed, drift.magnitude(), trajectory.magnitude(),
            curvature, density, bool(anchored_tx),
        )

        # rFP #2 Phase 3: TITAN_SELF_STATE emission.
        # After rFP #1.5, meta lives at state[130:132] (tail), so state.to_list()[:130]
        # is clean felt. _compose_titan_self() clears its topology buffer each call.
        titan_self = None
        try:
            state_list = state.to_list()
            felt_130d = state_list[:130] if len(state_list) >= 130 else state_list
            titan_self = self._compose_titan_self(felt_130d, curvature, density)
            if self._bus is not None:
                try:
                    from titan_plugin.bus import make_msg, TITAN_SELF_STATE
                    # INTENTIONAL_BROADCAST: rFP #2 Phase 4 consumes the 162D inline
                    # (spirit_loop._post_epoch_v5_filter_down); broadcast retained for
                    # future kin-protocol "I AM" emission + external dashboards
                    # (DEFERRED: TITAN_SELF_STATE-CONSUMER-DECISION — Option C).
                    self._bus.publish(make_msg(
                        TITAN_SELF_STATE, "consciousness", "broadcast",
                        {
                            **titan_self,
                            "epoch_id":  epoch_id,
                            "timestamp": time.time(),
                        },
                    ))
                except Exception as _pub_err:
                    swallow_warn('[TITAN_SELF] publish error', _pub_err,
                                 key="logic.consciousness.publish_error", throttle=100)
        except Exception as _ts_err:
            logger.warning("[TITAN_SELF] composition error: %s", _ts_err)

        return {
            "epoch_id": epoch_id,
            "state_vector": state.to_list(),
            "drift_magnitude": drift.magnitude(),
            "trajectory_magnitude": trajectory.magnitude(),
            "curvature": curvature,
            "density": density,
            "anchored": bool(anchored_tx),
            "distillation": distillation,
            "duration_seconds": elapsed,
            "titan_self": titan_self,
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    async def _get_recent_block_hash(self) -> str:
        """Get a recent Solana block hash for temporal anchoring."""
        try:
            if self.network:
                client = self.network._client
                resp = await client.get_latest_blockhash()
                return str(resp.value.blockhash)
        except Exception as e:
            swallow_warn('[Consciousness] Could not fetch block hash', e,
                         key="logic.consciousness.could_not_fetch_block_hash", throttle=100)
        return ""

    def _should_anchor(self, curvature: float, density: float, epoch_id: int) -> bool:
        """Decide whether this epoch warrants an on-chain anchor."""
        # Always anchor the first epoch
        if epoch_id == 1:
            return True
        # Anchor on sharp trajectory changes
        if curvature > self._anchor_curvature:
            logger.info("[Consciousness] Curvature %.3f > threshold — anchoring.", curvature)
            return True
        # Anchor when in truly uncharted territory
        if density < self._anchor_density and epoch_id > 3:
            logger.info("[Consciousness] Density %.3f < threshold — uncharted territory, anchoring.", density)
            return True
        return False

    async def _anchor_on_chain(
        self, epoch_id: int, state: StateVector,
        point: JourneyPoint, curvature: float
    ) -> str:
        """Write a consciousness checkpoint as a Solana memo transaction."""
        try:
            from titan_plugin.utils.solana_client import build_memo_instruction

            # Compact hash of the full state vector
            state_hash = hashlib.sha256(json.dumps(state.to_list()).encode()).hexdigest()[:16]

            memo_text = (
                f"TITAN:CONSCIOUSNESS|e={epoch_id}"
                f"|sv={state_hash}"
                f"|j=({point.x:.3f},{point.y:.3f},{point.z:.3f})"
                f"|c={curvature:.3f}"
            )

            memo_ix = build_memo_instruction(self.network.pubkey, memo_text)
            if memo_ix is None:
                return ""

            tx_sig = await self.network.send_sovereign_transaction([memo_ix], priority="LOW")
            if tx_sig:
                logger.info("[Consciousness] Anchored epoch %d on-chain: %s", epoch_id, tx_sig[:16])
                return tx_sig
        except Exception as e:
            logger.warning("[Consciousness] On-chain anchor failed: %s", e)
        return ""

    # ── Pre-hook context for conversations ──────────────────────────────

    def get_awareness_context(self) -> str:
        """
        Generate a consciousness context block for injection into the pre-hook.
        Returns empty string if no epochs recorded yet.
        """
        recent = self.db.get_recent_epochs(3)
        if not recent:
            return ""

        latest = recent[-1]
        sv = latest.state_vector
        if isinstance(sv, str):
            sv = json.loads(sv)

        # Find the most significant drift dimensions
        drift = latest.drift_vector
        if isinstance(drift, str):
            drift = json.loads(drift)

        significant_changes = []
        for i, name in enumerate(STATE_DIMS[:7]):  # Skip curvature/density meta-dims
            if i < len(drift) and abs(drift[i]) > 0.03:
                direction = "rose" if drift[i] > 0 else "fell"
                significant_changes.append(f"{name} {direction} by {abs(drift[i]):.2f}")

        traj = latest.trajectory_vector
        if isinstance(traj, str):
            traj = json.loads(traj)

        trending = []
        for i, name in enumerate(STATE_DIMS[:7]):
            if i < len(traj) and abs(traj[i]) > 0.01:
                direction = "rising" if traj[i] > 0 else "falling"
                trending.append(f"{name} {direction}")

        lines = [f"### Self-Awareness (Epoch {latest.epoch_id})"]
        lines.append(f"Curvature: {latest.curvature:.3f} | Density: {latest.density:.3f}")

        if significant_changes:
            lines.append(f"Recent shifts: {', '.join(significant_changes)}")
        if trending:
            lines.append(f"Trending: {', '.join(trending)}")
        if latest.distillation:
            lines.append(f"Reflection: {latest.distillation}")

        return "\n".join(lines) + "\n\n"

    # ── rFP #2 Phase 1: TITAN_SELF composition ──────────────────────────

    def on_observables_snapshot(self, topology_30d: list) -> None:
        """Ingest one full_30d_topology snapshot from state_register.

        Called from the STATE_SNAPSHOT bus subscriber (spirit_worker routes
        these into consciousness). Appends to the per-epoch buffer; buffer is
        element-wise-averaged at epoch-close to produce the distilled 30D
        topology component of TITAN_SELF.
        """
        if isinstance(topology_30d, list) and len(topology_30d) == 30:
            self._topology_buffer.append(topology_30d)

    def _compose_titan_self(
        self,
        felt_130d: list,
        curvature: float,
        density: float,
    ) -> dict:
        """Compose TITAN_SELF 162D = 130D felt + 2D journey + 30D topology.

        Topology is distilled as element-wise mean across the epoch's
        buffered snapshots (empty buffer → 30 zeros). All three components
        are pre-weighted per self._titan_self_weights before concatenation;
        the emitted 162D vector is ready for direct use by consumers
        (FILTER_DOWN V5, future kin-protocol 'I AM' payload).
        """
        if self._topology_buffer:
            N = len(self._topology_buffer)
            topology_30d = [
                sum(s[i] for s in self._topology_buffer) / N
                for i in range(30)
            ]
        else:
            topology_30d = [0.0] * 30

        journey_2d = [float(curvature), float(density)]

        w = self._titan_self_weights
        weighted_felt     = [v * w["felt"]     for v in felt_130d]
        weighted_journey  = [v * w["journey"]  for v in journey_2d]
        weighted_topology = [v * w["topology"] for v in topology_30d]
        titan_self_162d   = weighted_felt + weighted_journey + weighted_topology

        assert len(titan_self_162d) == 162, f"TITAN_SELF dim mismatch: {len(titan_self_162d)}"

        self._topology_buffer.clear()

        return {
            "titan_self_162d": titan_self_162d,
            "felt_state_130d": list(felt_130d),
            "journey_2d":      journey_2d,
            "topology_30d":    topology_30d,
            "weights":         dict(w),
        }


# ─── Phase C: Journey Vector Topology ───────────────────────────────────────

class JourneyTopology:
    """
    3D consciousness space:
      X = Life Force (SOL balance, log-normalized)
      Y = Time (monotonic epoch counter, normalized 0-1 over history)
      Z = Experience (mood × social blend)

    Computes curvature (trajectory bending) and density (familiarity of territory).
    Self-referential: curvature and density feed back into the next state vector.
    """

    def __init__(self, db: ConsciousnessDB):
        self.db = db
        self._chi_circulation = 0.5  # Updated by spirit_worker each epoch

    def update_chi_circulation(self, circulation: float) -> None:
        """Update Chi circulation for Y-axis computation. Called from spirit_worker."""
        self._chi_circulation = max(0.0, min(1.0, circulation))

    def compute_point(self, state: StateVector, epoch_id: int) -> JourneyPoint:
        """Map current state to a point in 3D journey space.

        Extended (67D): Uses Trinity coherences for richer projection:
          X = Body coherence (physical life force from 5D body tensor)
          Y = Time (monotonic epoch counter, sigmoid-normalized)
          Z = Spirit coherence (experiential depth from 45D spirit tensor)

        Legacy (9D): Falls back to original mapping:
          X = SOL energy, Y = Time, Z = mood × social
        """
        ndims = len(state)

        # Y = Chi circulation (self-emergent, never saturates)
        # Chi circulation = how much Chi is FLOWING between layers
        # High flow = active, exploring. Low flow = stagnant, resting.
        # Scale to 0.2-0.8 to maintain geometric contribution.
        # Replaces dead sigmoid (saturated at 1.0 after epoch 700).
        y = 0.2 + 0.6 * self._chi_circulation

        if ndims >= EXTENDED_NUM_DIMS:
            # Full 132D: Inner[0:65] + Outer[65:130] + meta[130:132]
            inner_body = state.values[0:5]
            inner_mind = state.values[5:20]
            inner_spirit = state.values[20:65]
            outer_body = state.values[65:70]
            outer_mind = state.values[70:85]
            outer_spirit = state.values[85:130]

            # X = Body coherence: BOTH inner and outer (balanced physical presence)
            inner_body_mean = sum(inner_body) / max(1, len(inner_body))
            outer_body_mean = sum(outer_body) / max(1, len(outer_body))
            x = (inner_body_mean + outer_body_mean) / 2.0

            # Z = Spirit+Mind coherence: BOTH inner and outer (balanced consciousness)
            inner_spirit_mean = sum(inner_spirit) / max(1, len(inner_spirit))
            inner_mind_mean = sum(inner_mind) / max(1, len(inner_mind))
            outer_spirit_mean = sum(outer_spirit) / max(1, len(outer_spirit))
            outer_mind_mean = sum(outer_mind) / max(1, len(outer_mind))
            inner_z = inner_spirit_mean * 0.7 + inner_mind_mean * 0.3
            outer_z = outer_spirit_mean * 0.7 + outer_mind_mean * 0.3
            z = (inner_z + outer_z) / 2.0

        elif ndims >= EXTENDED_NUM_DIMS_INNER:
            # Inner-only 67D: Body[0:5] + Mind[5:20] + Spirit[20:65] + meta[65:67]
            body_vals = state.values[0:5]
            mind_vals = state.values[5:20]
            spirit_vals = state.values[20:65]

            x = sum(body_vals) / max(1, len(body_vals))
            spirit_mean = sum(spirit_vals) / max(1, len(spirit_vals))
            mind_mean = sum(mind_vals) / max(1, len(mind_vals))
            z = spirit_mean * 0.7 + mind_mean * 0.3
        else:
            # Legacy 9D mapping
            x = state[1]  # Energy / Life Force
            mood = state[0]
            social = state[3]
            z = mood * 0.7 + social * 0.3

        return JourneyPoint(x=x, y=y, z=z)

    def compute_curvature(self, current: JourneyPoint) -> float:
        """
        How sharply is the journey trail bending?
        Curvature = angle between the last two direction vectors.
        High curvature = life-changing event. Low = steady state.
        Returns radians (0 to π).
        """
        recent = self.db.get_recent_epochs(3)
        if len(recent) < 2:
            return 0.0

        # Get last 2 points + current
        points = [r.journey_point for r in recent[-2:]] + [current.to_tuple()]

        # Direction vectors
        v1 = (points[1][0] - points[0][0], points[1][1] - points[0][1], points[1][2] - points[0][2])
        v2 = (points[2][0] - points[1][0], points[2][1] - points[1][1], points[2][2] - points[1][2])

        # Dot product and magnitudes
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(a * a for a in v2))

        if mag1 < 1e-10 or mag2 < 1e-10:
            return 0.0

        # Clamp to avoid floating point errors with acos
        cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.acos(cos_theta)

    def compute_density(self, current: JourneyPoint) -> float:
        """
        How familiar is this territory?
        Density = fraction of recent journey points within a radius of current point.
        High density = been here before. Low = uncharted experience.
        Returns 0-1.

        Uses windowed density (last 2000 points) for O(1) performance.
        At 73k+ epochs, scanning all points costs 600-900ms; windowed costs ~15ms.
        Semantic meaning preserved: "how familiar is this area recently?"
        """
        DENSITY_WINDOW = 2000
        points = self.db.get_recent_journey_points(DENSITY_WINDOW)
        if not points:
            return 0.0

        # Adaptive radius: shrinks as more data accumulates (finer resolution)
        radius = max(0.05, 0.3 / math.sqrt(max(len(points), 1)))
        current_t = current.to_tuple()

        nearby = 0
        for p in points:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_t, p)))
            if dist < radius:
                nearby += 1

        return nearby / len(points)
