"""
titan_plugin/logic/dreaming.py — Dreaming Cycle Engine (T6).

Self-emergent dreaming via neurochemical competition:
  ONSET:  sleep_drive > wake_drive (no hardcoded threshold)
    sleep_drive = fatigue × (1 + GABA × 2) × (1 + experience_pressure)
    wake_drive  = NE × (0.5 + DA × 0.5) × max(0.3, 1 - fatigue × 0.5)
  WAKE:   consolidation-gated (distillation complete AND drain recovered)
  SAFETY: drain > 0.75 → force sleep, max 600s → force wake
  MAKER:  DREAM_WAKE_REQUEST → instant gentle wake

Fatigue score (10 components) still computed as observable — feeds CHIT[10],
ANANDA[10], REFLECTION hormone, NS training via inner_state.fatigue.

Architecture: sleep_drive/wake_drive for ONSET, consolidation for DURATION.
  - Active Titan: high fatigue + high exp pressure → dreams sooner, longer
  - Idle Titan: chi stagnation + curvature stagnation → dreams from boredom
  - Flow state: high DA → resists sleep (but fatigue dampens alertness)
  - Hypervigilant: high NE → delays sleep (but fatigue eventually wins)

Redesigned 2026-03-26: emergent fatigue + self-emergent onset v2.
Tuned 2026-03-26: sleep_drive uses 10-component fatigue (not raw drain alone),
  wake_drive dampened by fatigue, drain passive decay reduced for adenosine-like
  accumulation.  Fixes: multiplicative-drain collapse + equilibrium ceiling.
See PLAN_emergent_fatigue_dreaming.md and rFP_pi_decoupling_maturation.md.
"""
import json
import logging
import math
import os
import time

logger = logging.getLogger(__name__)

# DNA defaults — overridden by titan_params.toml [dreaming] section
_DNA_DEFAULTS = {
    # Outer fatigue weights (acting surface depletion)
    "outer_weight": 0.45,
    "w_coherence_depletion": 0.20,
    "w_magnitude_depletion": 0.15,
    "w_direction_erratic": 0.15,
    "w_experience_pressure": 0.30,
    "w_expression_repetitive": 0.20,
    # Inner fatigue weights (neurochemical consolidation need)
    "inner_weight": 0.55,
    "w_gaba_accumulation": 0.30,
    "w_neuromod_deviation": 0.25,
    "w_chi_stagnation": 0.20,
    "w_curvature_stagnation": 0.15,
    "w_volume_expansion": 0.10,
    # Self-emergent onset (no threshold — competition between drives)
    "pi_accelerator": 0.70,    # sleep_drive boost when CLUSTER_END fires
    "min_dream_s": 30.0,       # Minimum dream duration (sleep inertia)
    "max_dream_s": 600.0,      # Safety cap (stuck guard)
    # Consolidation-gated wake conditions (DNA ratios)
    "drain_recovery_frac": 0.40,   # drain must clear to 40% of onset level
    "distill_complete_frac": 0.20, # undistilled must drop to 20% of onset level
    "force_sleep_drain": 0.75,     # force sleep above this drain (biological fainting)
    "adenosine_amplifier": 1.0,    # metabolic_drain → sleep_drive multiplier
    "wake_inertia_s": 120.0,       # minimum awake time after dream (sleep refractory period)
    # Legacy (kept for fatigue observable, not used for onset)
    "fatigue_threshold": 0.65,
    "readiness_threshold": 0.75,
    "r_inner_weight": 0.55,
    "r_outer_weight": 0.45,
    # rFP #3 Phase 2: clustering within dream cycle
    # Cosine-sim threshold for merging temporally-adjacent significant
    # snapshots into a single cluster (emits one insight per cluster).
    # Higher → fewer/larger clusters; lower → more/smaller clusters.
    "cluster_merge_threshold": 0.85,
    # Expected dimensionality of felt_tensor — telemetry only, no enforcement.
    "felt_tensor_expected_dim": 130,
}


def _element_wise_mean(vectors: list[list[float]]) -> list[float]:
    """Compute element-wise mean of a list of same-dim vectors.

    rFP #3 Phase 2 helper. Returns empty list for empty input.
    """
    if not vectors:
        return []
    D = len(vectors[0])
    N = len(vectors)
    return [sum(v[i] for v in vectors) / N for i in range(D)]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 on dim mismatch.

    rFP #3 Phase 2 helper. Mirrors experiential_memory._cosine_sim but kept
    local to avoid import cycle; used by clustering in _distill_experiences.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot / (mag_a * mag_b)


class DreamingEngine:
    """Manages conscious/dreaming state transitions via neurochemical competition."""

    def __init__(
        self,
        fatigue_threshold: float | None = None,
        readiness_threshold: float | None = None,
        state_path: str | None = None,
        dna: dict | None = None,
    ):
        # Load DNA weights (titan_params.toml [dreaming] section)
        self._dna = dict(_DNA_DEFAULTS)
        if dna:
            self._dna.update(dna)

        self.fatigue_threshold = fatigue_threshold or self._dna["fatigue_threshold"]
        self.readiness_threshold = readiness_threshold or self._dna["readiness_threshold"]
        self._pi_accelerator = self._dna["pi_accelerator"]
        self._min_dream_s = self._dna["min_dream_s"]
        self._max_dream_s = self._dna["max_dream_s"]
        self._drain_recovery_frac = self._dna["drain_recovery_frac"]
        self._distill_complete_frac = self._dna["distill_complete_frac"]
        self._force_sleep_drain = self._dna["force_sleep_drain"]
        self._adenosine_amplifier = self._dna["adenosine_amplifier"]
        self._wake_inertia_s = self._dna["wake_inertia_s"]

        self._cycle_count: int = 0
        self._last_transition_ts: float = 0.0
        self._distilled_count: int = 0
        # I-017-UNIVERSAL fix (2026-04-13): rolling variance distribution for
        # _distill_experiences. The previous threshold (0.02) was calibrated
        # for unbounded tensors but body/mind/spirit are normalized to [0,1]
        # where typical homeostatic state has variance < 0.02 → distillation
        # produced 0 insights across 6,928 cycles on T1+T2+T3 combined.
        # Threshold lowered to 0.005 (Phase 1 of foundational healing rFP).
        # Variance samples enable empirical recalibration in v2.
        from collections import deque as _deque
        self._variance_samples: _deque = _deque(maxlen=500)
        self._distill_threshold: float = 0.005
        self._distill_attempts: int = 0
        self._distill_passed: int = 0

        # rFP #3 Phase 2: clustering within dream cycle (config-driven via DNA)
        self._cluster_merge_threshold: float = float(
            self._dna.get("cluster_merge_threshold", 0.85))
        self._felt_tensor_expected_dim: int = int(
            self._dna.get("felt_tensor_expected_dim", 130))

        # Fatigue/readiness (observable — feeds CHIT/ANANDA/REFLECTION)
        self.last_fatigue: float = 0.0
        self.last_readiness: float = 0.0
        self._last_fatigue_breakdown: dict = {}

        # Self-emergent drives (logged for observability)
        self.last_sleep_drive: float = 0.0
        self.last_wake_drive: float = 0.0

        # Persisted state (survives cold restarts via dreaming_state.json)
        self._epochs_since_dream: int = 0
        self._persisted_drain: float = 0.0  # metabolic_drain continuity
        self._state_path = state_path
        if state_path:
            self._load_state()

        # Onset state for consolidation-gated wake
        self._onset_drain: float = 0.0
        self._onset_undistilled: int = 0

        # π-accelerator flag (armed by spirit_worker on CLUSTER_END/START)
        self._pending_pi_accelerator: bool = False
        self._pending_pi_wake: bool = False

        # Wake inertia: timestamp of last wake-up (sleep refractory period)
        self._last_wake_ts: float = 0.0

    # ── Self-emergent drive computation ─────────────────────────

    def compute_sleep_drive(
        self, gaba_level: float, experience_pressure: float,
        metabolic_drain: float = 0.0,
    ) -> float:
        """Sleep drive = fatigue × GABA amplifier × learning pressure × adenosine.

        metabolic_drain acts as adenosine — a direct byproduct of neural program
        firing that monotonically accumulates during wakefulness and clears during
        sleep. It amplifies sleep_drive DIRECTLY (not just through fatigue),
        matching biology where adenosine directly inhibits wake-promoting neurons.
        """
        fatigue = self.last_fatigue  # 10-component composite from compute_fatigue()
        adenosine_amplifier = 1.0 + metabolic_drain * self._adenosine_amplifier
        return fatigue * (1.0 + gaba_level * 2.0) * (1.0 + experience_pressure) * adenosine_amplifier

    def compute_wake_drive(
        self, ne_level: float, da_level: float,
    ) -> float:
        """Wake drive = alertness (NE) × engagement (DA) × fatigue dampening.

        As fatigue builds, alertness weakens — biologically, prolonged waking
        reduces the effectiveness of catecholamine-driven wakefulness.
        Dampening floors at 0.3 to prevent instant sleep at high fatigue.
        """
        fatigue = self.last_fatigue
        base = ne_level * (0.5 + da_level * 0.5)
        dampening = max(0.3, 1.0 - fatigue * 0.5)
        return base * dampening

    # ── Fatigue computation (OUTER + INNER, 10 components) ────────
    # Still computed as OBSERVABLE for inner_state.fatigue consumers.
    # NOT used for onset — onset is sleep_drive vs wake_drive.

    def compute_fatigue(
        self,
        observables: dict[str, dict],
        topology: dict,
        neurochemical: dict | None = None,
        experience: dict | None = None,
    ) -> float:
        """
        Compute composite fatigue from observables + neurochemistry + experience.
        This is an OBSERVABLE — feeds inner_state.fatigue for CHIT/ANANDA/REFLECTION.
        NOT used for dreaming onset (that's sleep_drive vs wake_drive).
        """
        d = self._dna

        # ── OUTER fatigue: acting surface wearing out ──
        outer_coh = []
        outer_mag = []
        outer_dir = []
        for name, obs in observables.items():
            if name.startswith("outer_"):
                outer_coh.append(obs.get("coherence", 1.0))
                outer_mag.append(obs.get("magnitude", 0.5))
                outer_dir.append(obs.get("direction", 1.0))

        if outer_coh:
            o1_coh_depletion = 1.0 - (sum(outer_coh) / len(outer_coh))
            o2_mag_depletion = 1.0 - (sum(outer_mag) / len(outer_mag))
            o3_dir_erratic = 1.0 - (sum(outer_dir) / len(outer_dir))
        else:
            o1_coh_depletion = 0.0
            o2_mag_depletion = 0.0
            o3_dir_erratic = 0.0

        exp = experience or {}
        undistilled = exp.get("undistilled", 0)
        total_exp = max(1, exp.get("total", 1))
        o4_exp_pressure = min(1.0, undistilled / max(1, total_exp))
        o5_exp_repetitive = min(1.0, exp.get("repetitiveness", 0.0))

        outer_fatigue = (
            d["w_coherence_depletion"] * o1_coh_depletion
            + d["w_magnitude_depletion"] * o2_mag_depletion
            + d["w_direction_erratic"] * o3_dir_erratic
            + d["w_experience_pressure"] * o4_exp_pressure
            + d["w_expression_repetitive"] * o5_exp_repetitive
        )

        # ── INNER fatigue: neurochemical consolidation need ──
        neuro = neurochemical or {}

        metabolic_drain = neuro.get("metabolic_drain", 0.0)
        i1_gaba = min(1.0, metabolic_drain / 0.5)

        i2_neuromod_dev = min(1.0, neuro.get("neuromod_deviation", 0.0) * 3.0)

        chi_circ = neuro.get("chi_circulation", 0.5)
        i3_chi_stagnation = 1.0 - min(1.0, chi_circ * 10.0)

        curv_variance = neuro.get("curvature_variance", 0.5)
        i4_curv_stagnation = 1.0 - min(1.0, curv_variance * 2.0)

        topo_curv = topology.get("curvature", 0.0)
        i5_vol_expansion = min(1.0, max(0.0, -topo_curv))

        inner_fatigue = (
            d["w_gaba_accumulation"] * i1_gaba
            + d["w_neuromod_deviation"] * i2_neuromod_dev
            + d["w_chi_stagnation"] * i3_chi_stagnation
            + d["w_curvature_stagnation"] * i4_curv_stagnation
            + d["w_volume_expansion"] * i5_vol_expansion
        )

        fatigue = (
            d["outer_weight"] * outer_fatigue
            + d["inner_weight"] * inner_fatigue
        )
        fatigue = max(0.0, min(1.0, fatigue))

        # Periodic logging (every 50 epochs)
        if self._epochs_since_dream % 50 == 1:
            logger.info(
                "[Dreaming] Fatigue=%.4f (outer=%.3f inner=%.3f) "
                "o1_coh=%.2f o4_exp=%.2f o5_rep=%.2f i1_metab=%.2f i2_neuro=%.2f "
                "i3_chi=%.2f i4_curv=%.2f | sleep=%.4f wake=%.4f",
                fatigue, outer_fatigue, inner_fatigue,
                o1_coh_depletion, o4_exp_pressure, o5_exp_repetitive,
                i1_gaba, i2_neuromod_dev,
                i3_chi_stagnation, i4_curv_stagnation,
                self.last_sleep_drive, self.last_wake_drive)

        self._last_fatigue_breakdown = {
            "outer": round(outer_fatigue, 4),
            "inner": round(inner_fatigue, 4),
            "o1_coh": round(o1_coh_depletion, 4),
            "o4_exp": round(o4_exp_pressure, 4),
            "o5_rep": round(o5_exp_repetitive, 4),
            "i1_metab": round(i1_gaba, 4),
            "i2_neuro": round(i2_neuromod_dev, 4),
            "i3_chi": round(i3_chi_stagnation, 4),
            "i4_curv": round(i4_curv_stagnation, 4),
        }

        return fatigue

    # ── State transition check ────────────────────────────────────

    def check_transition(
        self,
        inner_state,
        observables: dict[str, dict],
        topology: dict,
        neurochemical: dict | None = None,
        experience: dict | None = None,
        pi_event: str | None = None,
    ) -> str | None:
        """
        Self-emergent dreaming: sleep_drive vs wake_drive competition.

        ONSET (awake → dreaming):
          sleep_drive = drain × (1 + GABA×2) × (1 + exp_pressure)
          wake_drive  = NE × (0.5 + DA×0.5)
          Triggers when sleep_drive > wake_drive (zero hardcoded threshold).

        WAKE (dreaming → awake):
          Consolidation-gated: drain recovered AND distillation complete.
          Minimum 30s sleep inertia, maximum 600s safety cap.

        Fatigue still written to inner_state.fatigue for downstream consumers.
        """
        if not inner_state:
            return None

        # Handle π-accelerator flags
        if pi_event == "CLUSTER_END":
            self._pending_pi_accelerator = True
        elif pi_event == "CLUSTER_START":
            self._pending_pi_wake = True

        neuro = neurochemical or {}
        exp = experience or {}

        if not inner_state.is_dreaming:
            # ── AWAKE: compute fatigue observable + check onset ──
            fatigue = self.compute_fatigue(
                observables, topology, neurochemical, experience)
            inner_state.fatigue = fatigue
            self.last_fatigue = fatigue

            self._epochs_since_dream += 1
            metabolic_drain = neuro.get("metabolic_drain", 0.0)
            if self._epochs_since_dream % 100 == 0:
                self._persisted_drain = metabolic_drain  # snapshot for restart continuity
                self.save_state()

            # Self-emergent onset: neurochemical competition
            gaba_level = neuro.get("gaba_level", 0.3)
            ne_level = neuro.get("ne_level", 0.5)
            da_level = neuro.get("da_level", 0.5)
            undistilled = exp.get("undistilled", 0)
            total_exp = max(1, exp.get("total", 1))
            exp_p = min(1.0, undistilled / max(1, total_exp))

            sd = self.compute_sleep_drive(gaba_level, exp_p, metabolic_drain)
            wd = self.compute_wake_drive(ne_level, da_level)

            # Sleep debt: biological adenosine accumulation.
            # Without this, a Titan stuck in an arousing monoculture
            # (e.g. RECALL → high NE → high wake_drive) can NEVER dream,
            # creating a vicious cycle where the learning loop that would
            # fix the monoculture is itself blocked by the monoculture.
            # Progressive boost: starts at 500 epochs, ramps over 1000.
            _sleep_debt_onset = 500
            _sleep_debt_ramp = 1000
            if self._epochs_since_dream > _sleep_debt_onset:
                _debt = (self._epochs_since_dream - _sleep_debt_onset) / _sleep_debt_ramp
                sd = sd * (1.0 + _debt)
                if self._epochs_since_dream % 100 == 1:
                    logger.info(
                        "[Dreaming] Sleep debt: epochs_since=%d debt=%.2f "
                        "sd=%.4f(boosted) wd=%.4f",
                        self._epochs_since_dream, _debt, sd, wd)

            # π-CLUSTER_END: boost sleep drive by 30%
            if self._pending_pi_accelerator:
                sd /= self._pi_accelerator  # divide by 0.70 = boost ~43%
                self._pending_pi_accelerator = False
                logger.info(
                    "[Dreaming] π-accelerator: sleep=%.4f(boosted) wake=%.4f",
                    sd, wd)

            self.last_sleep_drive = sd
            self.last_wake_drive = wd

            # ONSET: sleep_drive > wake_drive
            # Wake inertia: suppress onset for _wake_inertia_s after waking
            # (biological sleep refractory period — gives time for diverse
            # expression, metabolic drain accumulation, and o5_rep recovery)
            since_wake = time.time() - self._last_wake_ts if self._last_wake_ts > 0 else float('inf')
            in_wake_inertia = since_wake < self._wake_inertia_s

            if sd > wd and not in_wake_inertia:
                self._onset_drain = metabolic_drain
                # Use pre-dream tagged count (excludes any during-dream records)
                self._onset_undistilled = int(exp.get("pre_dream_undistilled", undistilled))
                return "BEGIN_DREAMING"
            elif sd > wd and in_wake_inertia and self._epochs_since_dream % 20 == 1:
                logger.info(
                    "[Dreaming] Wake inertia: sleep=%.4f>wake=%.4f SUPPRESSED (%.0fs/%.0fs remaining)",
                    sd, wd, since_wake, self._wake_inertia_s)

            # SAFETY: force sleep at extreme metabolic drain (overrides wake inertia)
            if metabolic_drain > self._force_sleep_drain:
                self._onset_drain = metabolic_drain
                self._onset_undistilled = int(undistilled)
                logger.warning(
                    "[Dreaming] FORCE SLEEP — drain=%.3f > %.3f",
                    metabolic_drain, self._force_sleep_drain)
                return "BEGIN_DREAMING"

        else:
            # ── DREAMING: check consolidation-gated wake ──
            dream_duration = (time.time() - self._last_transition_ts
                              if self._last_transition_ts else 0.0)

            # Minimum dream duration (sleep inertia — prevents micro-dreams)
            if dream_duration < self._min_dream_s:
                return None

            # SAFETY: force wake after max duration (stuck guard)
            if dream_duration >= self._max_dream_s:
                logger.warning(
                    "[Dreaming] FORCE WAKE — duration=%.0fs > max=%.0fs",
                    dream_duration, self._max_dream_s)
                return "END_DREAMING"

            # Consolidation-gated wake: both conditions must be met
            metabolic_drain = neuro.get("metabolic_drain", 0.0)
            # Use pre-dream tagged count — ignores experiences created during this dream
            undistilled = exp.get("pre_dream_undistilled", exp.get("undistilled", 0))

            drain_recovered = (
                metabolic_drain <= self._onset_drain * self._drain_recovery_frac
                or self._onset_drain < 0.01  # negligible onset drain
            )
            distill_done = (
                undistilled <= self._onset_undistilled * self._distill_complete_frac
                or self._onset_undistilled <= 5  # negligible onset backlog
            )

            # π-CLUSTER_START: skip distillation gate (natural awakening signal)
            if self._pending_pi_wake:
                distill_done = True
                self._pending_pi_wake = False
                logger.info(
                    "[Dreaming] π-wake: distillation gate bypassed (drain=%.4f)",
                    metabolic_drain)

            if drain_recovered and distill_done:
                return "END_DREAMING"

        return None

    # ── State transitions ─────────────────────────────────────────

    def begin_dreaming(self, inner_state) -> None:
        """Transition to dreaming state."""
        if not inner_state:
            return
        inner_state.is_dreaming = True
        inner_state.cycle_count += 1
        inner_state.last_cycle_ts = time.time()
        self._last_transition_ts = time.time()

        # Store onset state for consolidation-gated recovery
        self._dream_onset_fatigue = float(inner_state.fatigue)
        self._dream_fatigue = float(inner_state.fatigue)
        self._dream_epoch_count = 0
        self._wake_transition = False
        self._epochs_since_dream = 0
        self.save_state()

        logger.info(
            "[Dreaming] BEGIN cycle %d — sleep=%.4f>wake=%.4f "
            "drain=%.4f undist=%d fatigue=%.3f",
            inner_state.cycle_count,
            self.last_sleep_drive, self.last_wake_drive,
            self._onset_drain, self._onset_undistilled,
            inner_state.fatigue)

    def end_dreaming(self, inner_state, emot_cgn=None) -> dict:
        """Transition back to awake state. Returns dreaming summary.

        rFP_emot_cgn_v2 §7.3: if emot_cgn is provided, opportunistically
        seed emergent cluster slots from dream clusters (gated — active or
        shadow, both OK since seeding is observability-compatible) +
        trigger periodic recentering.
        """
        if not inner_state:
            return {}

        buffer = inner_state.drain_experience_buffer()
        distilled = self._distill_experiences(buffer)

        # EMOT-CGN dream-cluster integration (gated — see rFP §7.3).
        #
        # ⚠ DEAD PATH since Phase 1.6h cutover (2026-04-20): EMOT-CGN
        # moved from in-process (meta_reasoning._emot_cgn) to standalone
        # subprocess (modules/emot_cgn_worker.py). Neither
        # inner_coordinator.py call site (:201, :296) passes emot_cgn to
        # this method, and meta_reasoning.py:978 keeps _emot_cgn = None
        # as sentinel. This `if emot_cgn is not None:` branch is
        # therefore never entered in production.
        #
        # BUG #10 fix (2026-04-24): k-means recenter invocation relocated
        # to emot_cgn_worker main loop with matching K_RECENTER_CHECK_
        # INTERVAL_S tick. Dream-cluster seeding (seed_from_dream_clusters)
        # remains here for reference but is dead; reviving it requires a
        # new DREAM_CYCLE_END bus message + subscription in the worker —
        # not scoped now per Q3 audit TRANSITIONAL directive.
        if emot_cgn is not None:  # always False in production — kept for reference
            try:
                clusterer = getattr(emot_cgn, "_clusterer", None)
                if clusterer is not None and distilled:
                    clusterer.seed_from_dream_clusters(distilled)
                    clusterer.maybe_recenter(force=False)
            except Exception as _e:
                logger.debug("[Dreaming] EMOT-CGN hook error: %s", _e)

        inner_state.is_dreaming = False
        inner_state.fatigue = 0.0
        self._last_transition_ts = time.time()
        self._last_wake_ts = time.time()  # Start wake inertia refractory period
        self._cycle_count += 1
        self.save_state()

        duration = time.time() - inner_state.last_cycle_ts
        logger.info(
            "[Dreaming] END cycle %d — duration=%.1fs, distilled=%d, "
            "drain=%.4f→now, undist=%d→now",
            inner_state.cycle_count, duration, len(distilled),
            self._onset_drain, self._onset_undistilled)

        return {
            "cycle": inner_state.cycle_count,
            "experiences_processed": len(buffer),
            "distilled_insights": distilled,
            "duration_s": duration,
        }

    # ── Experience distillation ───────────────────────────────────

    def _distill_experiences(self, buffer: list[dict]) -> list[dict]:
        """Element-wise aggregation with temporal-coherence clustering.

        rFP #3 Phase 2 rewrite:
          - Reads `full_130dt` (preferred, rFP #1); falls back to `full_65dt`
            during the transition window; then `full_30dt`; finally legacy
            body/mind/spirit_tensor keys for test-producer compat.
          - Stores the ACTUAL felt tensor per insight (not just scalar mean),
            enabling `recall_by_state` cosine-similarity lookup.
          - Groups temporally-adjacent significant snapshots that share felt-
            similarity into clusters; emits ONE insight per cluster, preserving
            distinctness when a dream cycle contains multiple sub-experiences.

        THE ROOT FIX: prior version emitted `tensor_mean` (scalar) as the
        "felt_tensor" via experiential_memory's fallback — corrupting the
        column and crashing D6 recall with `float has no len()`. New insights
        carry the actual D-dim vector instead.
        """
        if not buffer:
            return []

        # Step 1: collect significant snapshots with their tensors
        significant = []
        for snapshot in buffer:
            tensors = (
                list(snapshot.get("full_130dt") or [])     # preferred (rFP #1)
                or list(snapshot.get("full_65dt") or [])   # fallback
                or list(snapshot.get("full_30dt") or [])   # legacy
            )
            if not tensors:
                # Backward compat: test-producer path uses body/mind/spirit keys
                for key in ("body_tensor", "mind_tensor", "spirit_tensor"):
                    t = snapshot.get(key, [])
                    if t:
                        tensors.extend(t)
            if not tensors:
                continue

            mean_val = sum(tensors) / len(tensors)
            variance = sum((v - mean_val) ** 2 for v in tensors) / len(tensors)

            self._distill_attempts += 1
            self._variance_samples.append(round(variance, 6))

            if variance > self._distill_threshold:
                self._distill_passed += 1
                significant.append({
                    "tensor":   tensors,
                    "variance": variance,
                    "ts":       snapshot.get("ts") or snapshot.get("timestamp", 0.0),
                })

        if not significant:
            return []

        # Step 2: greedy cosine-sim clustering to previous cluster's centroid.
        # Cross-dim pairs return similarity 0 (len-mismatch guard in
        # _cosine_similarity) → snapshots of different dims never merge,
        # giving graceful handling of mixed-dim transition-window buffers.
        clusters: list[list[dict]] = [[significant[0]]]
        centroid: list[float] = list(significant[0]["tensor"])

        for s in significant[1:]:
            sim = _cosine_similarity(s["tensor"], centroid)
            if sim > self._cluster_merge_threshold:
                clusters[-1].append(s)
                centroid = _element_wise_mean([m["tensor"] for m in clusters[-1]])
            else:
                clusters.append([s])
                centroid = list(s["tensor"])

        # Step 3: emit one insight per cluster.
        # Cast to native Python floats: state_register emits numpy-backed
        # tensors → arithmetic yields np.float64 → leaks into log format
        # (peak_sig=[np.float64(0.05)]) and into json.dumps risk paths.
        insights = []
        for idx, cluster in enumerate(clusters):
            D = len(cluster[0]["tensor"])
            felt_tensor = [float(x) for x in _element_wise_mean(
                [m["tensor"] for m in cluster])]
            peak_var = float(max(m["variance"] for m in cluster))
            insights.append({
                "significance": round(peak_var, 4),
                "felt_tensor":  felt_tensor,     # ← THE FIX: actual vector, not scalar
                "ts":           float(cluster[0]["ts"]) if cluster[0]["ts"] else 0.0,
                "dim":          D,               # telemetry: 130 (or 65 during transition)
                "num_samples":  len(cluster),
                "cluster_idx":  idx,
            })

        self._distilled_count += len(insights)
        return insights

    # ── State persistence ─────────────────────────────────────────

    def _load_state(self) -> None:
        """Load persisted dreaming state from disk."""
        if not self._state_path or not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)
            self._epochs_since_dream = int(data.get("epochs_since_dream", 0))
            self._cycle_count = int(data.get("cycle_count", 0))
            self._distilled_count = int(data.get("distilled_count", 0))
            self._persisted_drain = float(data.get("metabolic_drain", 0.0))
            # 2026-04-13 Phase 1 — persist attempts/passed too so all three
            # counters are lifetime, consistent with distilled_count. Without
            # this, distilled (persisted) and attempts (per-process) drift,
            # producing impossible-looking reads like "distilled > passed"
            # and false-positive "DISTILLATION DISCONNECTED" alerts when a
            # process restarts before its first dream cycle. Defaults to 0
            # for files saved by older code (backward compat).
            self._distill_attempts = int(data.get("distill_attempts", 0))
            self._distill_passed = int(data.get("distill_passed", 0))
            logger.info(
                "[Dreaming] Loaded state: epochs_since_dream=%d, cycles=%d, "
                "distilled=%d (attempts=%d passed=%d), drain=%.4f",
                self._epochs_since_dream, self._cycle_count,
                self._distilled_count, self._distill_attempts,
                self._distill_passed, self._persisted_drain)
        except Exception as e:
            logger.warning("[Dreaming] Failed to load state from %s: %s",
                           self._state_path, e)

    def save_state(self) -> None:
        """Persist dreaming state to disk (atomic write)."""
        if not self._state_path:
            return
        try:
            data = {
                "epochs_since_dream": self._epochs_since_dream,
                "cycle_count": self._cycle_count,
                "distilled_count": self._distilled_count,
                "metabolic_drain": self._persisted_drain,
                # 2026-04-13 Phase 1 — persist Phase 1 telemetry counters as
                # lifetime. See _load_state for full rationale.
                "distill_attempts": self._distill_attempts,
                "distill_passed": self._distill_passed,
            }
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.warning("[Dreaming] Failed to save state: %s", e)

    def get_stats(self) -> dict:
        # Variance distribution stats for I-017-UNIVERSAL recalibration (v2)
        var_stats = {}
        if self._variance_samples:
            samples = sorted(self._variance_samples)
            n = len(samples)
            var_stats = {
                "samples": n,
                "min": round(samples[0], 6),
                "p25": round(samples[n // 4], 6),
                "median": round(samples[n // 2], 6),
                "p75": round(samples[(3 * n) // 4], 6),
                "p95": round(samples[min(n - 1, (95 * n) // 100)], 6),
                "max": round(samples[-1], 6),
                "mean": round(sum(samples) / n, 6),
            }
        attempts = self._distill_attempts
        passed = self._distill_passed
        return {
            "cycle_count": self._cycle_count,
            "distilled_count": self._distilled_count,
            "last_fatigue": round(self.last_fatigue, 4),
            "last_sleep_drive": round(self.last_sleep_drive, 4),
            "last_wake_drive": round(self.last_wake_drive, 4),
            "fatigue_breakdown": self._last_fatigue_breakdown,
            "epochs_since_dream": self._epochs_since_dream,
            "onset_drain": round(self._onset_drain, 4),
            "onset_undistilled": self._onset_undistilled,
            # I-017-UNIVERSAL telemetry
            "distill_threshold": self._distill_threshold,
            "distill_attempts": attempts,
            "distill_passed": passed,
            "distill_pass_rate": (round(passed / attempts, 4)
                                  if attempts else None),
            "variance_distribution": var_stats,
        }
