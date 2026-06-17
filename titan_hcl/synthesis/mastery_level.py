"""titan_hcl.synthesis.mastery_level — the self-emergent MasteryLevel primitive.

RFP_emergent_mastery_curriculum P3 / ARCHITECTURE_mastery_leveling.md §2.2 (D-SPEC-160).

`MasteryLevel = { EMA(symlog-value), fixed grade support, competence-GATED
running-max ratchet, milestone ticks }` — an ABILITY-driven level (not age,
INV-MC-1) derived from the IQL value function V(s):

  • the level is the bin of EMA(symlog-V̄) on a FIXED grade ladder (bins-as-grades;
    the two-hot idea applied at the LEVEL stage so IQL's scalar expectile critic
    is untouched — INV-ML-1);
  • the running-max ratchet only advances when a SCALE-FREE competence_rate
    confirms it (SOAR control/competence — so reward-scale inflation can't fake a
    level-up, INV-ML-2/3);
  • chunk-graduation events (new macros) fire milestone ticks (SOAR ②); the
    chunk COUNT is a SECONDARY structural signal, never the primary driver
    (INV-ML-5).

Reusable per-domain primitive (INV-ML-6): the routing instance is the first;
social/reasoning/research level instances reuse this class over their own
(value, competence) scalars. Pure-numpy, no torch, no SHM (the worker owns
persistence + the SHM publish).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Defaults mirror ARCHITECTURE_mastery_leveling.md §5 (the v1 design point); the
# worker overrides from config[synthesis][self_learning].
DEFAULT_N_GRADES = 10
DEFAULT_GRADE_LO = -5.0          # symlog-space; r∈[−1,1],γ=0.99 ⇒ symlog-V ∈ ~[−5,5]
DEFAULT_GRADE_HI = 5.0
DEFAULT_EMA_ALPHA = 0.05         # V̄ smoothing
DEFAULT_COMPETENCE_FLOOR_BASE = 0.55
DEFAULT_COMPETENCE_FLOOR_SLOPE = 0.02
DEFAULT_COMPETENCE_EMA_ALPHA = 0.05

MASTERY_LEVEL_SCHEMA_VERSION = 1


class MasteryLevel:
    """Emergent ability level from a value signal + a scale-free competence gate."""

    def __init__(
        self,
        *,
        n_grades: int = DEFAULT_N_GRADES,
        grade_lo: float = DEFAULT_GRADE_LO,
        grade_hi: float = DEFAULT_GRADE_HI,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        competence_floor_base: float = DEFAULT_COMPETENCE_FLOOR_BASE,
        competence_floor_slope: float = DEFAULT_COMPETENCE_FLOOR_SLOPE,
        competence_ema_alpha: float = DEFAULT_COMPETENCE_EMA_ALPHA,
    ):
        self.n_grades = int(n_grades)
        # n_grades bins ⇒ n_grades+1 edges; grade index ∈ [0, n_grades-1].
        self._support = np.linspace(
            float(grade_lo), float(grade_hi), self.n_grades + 1).astype(np.float64)
        self.ema_alpha = float(ema_alpha)
        self.competence_floor_base = float(competence_floor_base)
        self.competence_floor_slope = float(competence_floor_slope)
        self.competence_ema_alpha = float(competence_ema_alpha)
        # Mutable state
        self._ema_v: Optional[float] = None       # EMA of symlog-V̄
        self._competence_ema: float = 0.0
        self._running_max_grade: int = 0          # the RATCHET (INV-ML-2)
        self._last_n_chunks: int = 0
        self._n_value_milestones: int = 0
        self._n_chunk_milestones: int = 0
        self._updates: int = 0

    # -- core -----------------------------------------------------------
    def competence_floor(self, grade: int) -> float:
        """The competence the pupil must PROVE to ratchet PAST `grade` — rises
        with the grade so each level demands more genuine competence (INV-ML-2)."""
        return min(1.0, self.competence_floor_base
                   + self.competence_floor_slope * float(grade))

    def _grade_of(self, ema_v: float) -> int:
        # digitize → bin index in [1, n_grades]; shift to [0, n_grades-1].
        g = int(np.digitize(ema_v, self._support)) - 1
        return int(np.clip(g, 0, self.n_grades - 1))

    def update(self, v_symlog: float, competence_rate: float,
               n_chunks: int = 0) -> dict:
        """One level update from the current (symlog) value, an instantaneous
        scale-free competence_rate ∈ [0,1] (already the success/adv blend), and
        the current chunk count. Returns the level readout + any milestone ticks.

        The RATCHET is GATED: the running-max grade advances only when the
        smoothed competence clears `competence_floor(running_max_grade)` — so a
        V̄ rise from reward-scale inflation alone never advances the level."""
        v = float(v_symlog)
        c = float(np.clip(competence_rate, 0.0, 1.0))
        self._ema_v = v if self._ema_v is None else (
            (1.0 - self.ema_alpha) * self._ema_v + self.ema_alpha * v)
        self._competence_ema = (
            (1.0 - self.competence_ema_alpha) * self._competence_ema
            + self.competence_ema_alpha * c)
        milestones = []

        grade = self._grade_of(self._ema_v)
        # GATED running-max ratchet (INV-ML-2/3).
        if grade > self._running_max_grade and \
                self._competence_ema >= self.competence_floor(self._running_max_grade):
            self._running_max_grade = grade
            self._n_value_milestones += 1
            milestones.append("value")

        # Chunk graduation (SOAR ② — milestone only, NOT the primary driver; INV-ML-5).
        nc = int(n_chunks)
        if nc > self._last_n_chunks:
            self._n_chunk_milestones += (nc - self._last_n_chunks)
            self._last_n_chunks = nc
            milestones.append("chunk")

        self._updates += 1
        return self.readout(milestones=milestones)

    def readout(self, milestones: Optional[list] = None) -> dict:
        """Current level state. `level` is the ratcheted grade + a bounded,
        competence-scaled within-grade fraction (continuous, monotone in the
        ratchet — never drops below running_max_grade, INV-ML-2)."""
        rmg = self._running_max_grade
        ema_v = self._ema_v if self._ema_v is not None else float(self._support[0])
        lo = float(self._support[rmg])
        hi = float(self._support[min(rmg + 1, self.n_grades)])
        frac = 0.0 if hi <= lo else float(np.clip((ema_v - lo) / (hi - lo), 0.0, 1.0))
        level = float(rmg) + frac * self._competence_ema
        return {
            "level": level,
            "grade": rmg,
            "ema_v_symlog": float(ema_v),
            "competence": float(self._competence_ema),
            "n_chunks": int(self._last_n_chunks),
            "value_milestones": int(self._n_value_milestones),
            "chunk_milestones": int(self._n_chunk_milestones),
            "updates": int(self._updates),
            "milestones": list(milestones or []),
        }

    # -- persistence (worker-owned; JSON blob) --------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": MASTERY_LEVEL_SCHEMA_VERSION,
            "n_grades": self.n_grades,
            "ema_v": self._ema_v,
            "competence_ema": self._competence_ema,
            "running_max_grade": self._running_max_grade,
            "last_n_chunks": self._last_n_chunks,
            "n_value_milestones": self._n_value_milestones,
            "n_chunk_milestones": self._n_chunk_milestones,
            "updates": self._updates,
        }

    def load_dict(self, d: dict) -> bool:
        """Restore mutable state. Returns False (state untouched) on schema/grade
        mismatch → a fresh level relearns from the value function."""
        try:
            if not isinstance(d, dict):
                return False
            if int(d.get("schema_version", 0)) != MASTERY_LEVEL_SCHEMA_VERSION:
                return False
            if int(d.get("n_grades", self.n_grades)) != self.n_grades:
                return False
            self._ema_v = (None if d.get("ema_v") is None else float(d["ema_v"]))
            self._competence_ema = float(d.get("competence_ema", 0.0) or 0.0)
            self._running_max_grade = int(np.clip(
                int(d.get("running_max_grade", 0) or 0), 0, self.n_grades - 1))
            self._last_n_chunks = int(d.get("last_n_chunks", 0) or 0)
            self._n_value_milestones = int(d.get("n_value_milestones", 0) or 0)
            self._n_chunk_milestones = int(d.get("n_chunk_milestones", 0) or 0)
            self._updates = int(d.get("updates", 0) or 0)
            return True
        except Exception:
            return False


__all__ = ("MasteryLevel", "MASTERY_LEVEL_SCHEMA_VERSION")
