"""
titan_plugin/logic/inner_coordinator.py — Inner Trinity Coordinator (T3).

The wise guardian — coordinates all Inner Trinity functions into a single
coherent tick. Replaces ad-hoc logic scattered through spirit_worker with
a clean orchestration layer.

Responsibilities:
  1. Compute observables for all 6 body parts (T1 ObservableEngine)
  2. Update InnerState with observables (T2)
  3. Assemble SpiritState from all sources (T2)
  4. Extract coherences for sphere clock consumption
  5. Run nervous system micro-programs (T4 hook — not yet active)
  6. Check fatigue/readiness (T6 hook — not yet active)
  7. Buffer outer snapshots for dreaming (T6 hook)

The coordinator is the SINGLE POINT where inner processing is orchestrated.
Spirit_worker calls coordinator.tick() and coordinator.on_outer_snapshot()
instead of managing individual components directly.
"""
import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


class InnerTrinityCoordinator:
    """The wise guardian — coordinates Inner Trinity functions."""

    def __init__(
        self,
        inner_state,
        spirit_state,
        observable_engine,
        vm=None,
        nervous_system=None,
        topology_engine=None,
        dreaming_engine=None,
        neural_nervous_system=None,
    ):
        """
        Args:
            inner_state: InnerState instance (T2)
            spirit_state: SpiritState instance (T2)
            observable_engine: ObservableEngine instance (T1)
            vm: TitanVM instance (T4 — optional)
            nervous_system: NervousSystem instance (T4 — optional, VM fallback)
            topology_engine: TopologyEngine instance (T5 — optional)
            dreaming_engine: DreamingEngine instance (T6 — optional)
            neural_nervous_system: V5 NeuralNervousSystem (T4 upgrade — optional)
        """
        self.inner = inner_state
        self.spirit = spirit_state
        self.observables = observable_engine
        self.vm = vm
        self.nervous_system = nervous_system
        self.neural_ns = neural_nervous_system
        self.topology = topology_engine
        self.dreaming = dreaming_engine
        self._tick_count: int = 0
        self._last_nervous_signals: list[dict] = []
        self._last_topology: dict = {}
        self._last_dreaming_event: str | None = None
        self._last_epoch_id: int = 0

        # Dream side-effect subsystems (set via set_dream_subsystems)
        self._exp_orchestrator = None
        self._life_force = None
        self._e_mem = None
        self._neuromod_system = None

    def set_dream_subsystems(self, exp_orchestrator=None, life_force=None,
                             e_mem=None, neuromod_system=None):
        """Wire subsystems needed for dream side-effects."""
        self._exp_orchestrator = exp_orchestrator
        self._life_force = life_force
        self._e_mem = e_mem
        self._neuromod_system = neuromod_system

    def tick(
        self,
        inner_tensors: dict[str, Sequence[float]],
        outer_tensors: Optional[dict[str, Sequence[float]]] = None,
        outer_snapshot: Optional[dict] = None,
    ) -> dict[str, dict]:
        """
        Main coordination tick — runs every spirit publish cycle (60s).

        Args:
            inner_tensors: {"inner_body": [...], "inner_mind": [...], "inner_spirit": [...]}
            outer_tensors: {"outer_body": [...], "outer_mind": [...], "outer_spirit": [...]}
                           If None, uses last known outer observables.
            outer_snapshot: Full outer state snapshot for SpiritState assembly.

        Returns:
            Dict of all observables {part_name: {5 observables}}.
        """
        self._tick_count += 1

        # 1. Compute observables for available tensors
        all_tensors = dict(inner_tensors)
        if outer_tensors:
            all_tensors.update(outer_tensors)

        all_obs = self.observables.observe_all(all_tensors)

        # 2. Update InnerState
        if self.inner:
            self.inner.update_observables(all_obs)

        # 3. Assemble SpiritState
        if self.spirit:
            self.spirit.assemble(
                outer_snapshot=outer_snapshot,
                observables=all_obs,
                inner_snapshot=self.inner.snapshot() if self.inner else None,
            )

        # 4. Run nervous system micro-programs (T4)
        if self.nervous_system:
            try:
                self._last_nervous_signals = self.nervous_system.evaluate(all_obs)
            except Exception as e:
                logger.debug("[Coordinator] NervousSystem error: %s", e)

        # 5. Compute space topology (T5)
        if self.topology:
            try:
                self._last_topology = self.topology.compute(all_obs)
                if self.inner:
                    self.inner.update_topology(self._last_topology)
            except Exception as e:
                logger.debug("[Coordinator] Topology error: %s", e)

        # 6. Emergent GREAT PULSE: topology convergence during dreaming (T7)
        #    Must check BEFORE dreaming transitions, so pulse fires at convergence
        #    peak while still dreaming (before waking up).
        self._last_dreaming_event = None
        if (self.topology and self.inner and self.inner.is_dreaming
                and self.topology.is_convergence_peak()):
            self._last_dreaming_event = "GREAT_PULSE"
            self._last_topology["great_pulse"] = {
                "cycle": self.inner.cycle_count,
                "volume": self._last_topology.get("volume", 0.0),
                "curvature": self._last_topology.get("curvature", 0.0),
                "tick": self._tick_count,
            }
            if self.spirit:
                self.spirit.enrichment_quality = min(
                    1.0, self.spirit.enrichment_quality + 0.05)
            logger.info("[Coordinator] GREAT PULSE at convergence! cycle=%d vol=%.4f",
                        self.inner.cycle_count,
                        self._last_topology.get("volume", 0.0))

        # 7. Check fatigue/readiness and dreaming transitions (T6)
        #    Only transition if no GREAT PULSE this tick (pulse takes priority)
        if self.dreaming and self.inner and self._last_dreaming_event != "GREAT_PULSE":
            try:
                transition = self.dreaming.check_transition(
                    self.inner, all_obs, self._last_topology)
                if transition == "BEGIN_DREAMING":
                    self.dreaming.begin_dreaming(self.inner)
                    self._last_dreaming_event = "BEGIN_DREAMING"
                elif transition == "END_DREAMING":
                    summary = self.dreaming.end_dreaming(self.inner)
                    self._last_dreaming_event = "END_DREAMING"
                    if summary:
                        self._last_topology["dreaming_summary"] = summary
            except Exception as e:
                logger.debug("[Coordinator] Dreaming check error: %s", e)

        return all_obs

    def coordinate(
        self,
        temporal: dict = None,
        neurochemical: dict | None = None,
        experience: dict | None = None,
        pi_event: str | None = None,
    ) -> dict:
        """
        Run T5-T7 coordination logic using observables already in InnerState.

        Call this after tick_inner_only() and/or tick_outer_only() have updated
        the observables. Runs: topology, GREAT PULSE check, dreaming transitions.
        NS evaluation handled by spirit_worker Tier 2 (has full observation data).

        Args:
            temporal: Optional dict with circadian features from π-heartbeat.
            neurochemical: GABA level/setpoint, neuromod_deviation, chi_circulation,
                          curvature_variance — for emergent fatigue computation.
            experience: undistilled, total, repetitiveness — for experience pressure.
            pi_event: "CLUSTER_END" | "CLUSTER_START" | None — π-accelerator.

        Returns:
            Dict with coordination results including any dreaming events.
        """
        self._tick_count += 1
        self._last_dreaming_event = None

        all_obs = self.inner.observables if self.inner else {}
        if not all_obs:
            return {"event": None}

        # T4/V5: Neural NS evaluation handled by spirit_worker Tier 2 block.
        # Maturity signals wired there (sphere clocks, resonance, consciousness).
        # _last_nervous_signals updated by spirit_worker after evaluation.
        # Removed 2026-03-24: double evaluation caused refractory trap blocking
        # 8 of 10 hormonal programs. See CRITICAL_BUG_20260324 for full analysis.

        # T5: Compute space topology
        if self.topology:
            try:
                self._last_topology = self.topology.compute(all_obs)
                if self.inner:
                    self.inner.update_topology(self._last_topology)
            except Exception as e:
                logger.debug("[Coordinator] Topology error: %s", e)

        # T7: Emergent GREAT PULSE — topology convergence during dreaming
        #     Must check BEFORE dreaming transitions so pulse fires at
        #     convergence peak while still dreaming (before waking up).
        if (self.topology and self.inner and self.inner.is_dreaming
                and self.topology.is_convergence_peak()):
            self._last_dreaming_event = "GREAT_PULSE"
            self._last_topology["great_pulse"] = {
                "cycle": self.inner.cycle_count,
                "volume": self._last_topology.get("volume", 0.0),
                "curvature": self._last_topology.get("curvature", 0.0),
                "tick": self._tick_count,
            }
            if self.spirit:
                self.spirit.enrichment_quality = min(
                    1.0, self.spirit.enrichment_quality + 0.05)
            logger.info("[Coordinator] GREAT PULSE at convergence! cycle=%d vol=%.4f",
                        self.inner.cycle_count,
                        self._last_topology.get("volume", 0.0))

        # T6: Check fatigue/readiness and dreaming transitions (EMERGENT)
        #     SINGLE AUTHORITY: all dreaming decisions flow through here.
        #     π-events passed as accelerator, not as direct trigger.
        #     Only transition if no GREAT PULSE this tick (pulse takes priority)
        if self.dreaming and self.inner and self._last_dreaming_event != "GREAT_PULSE":
            try:
                transition = self.dreaming.check_transition(
                    self.inner, all_obs, self._last_topology,
                    neurochemical=neurochemical,
                    experience=experience,
                    pi_event=pi_event)
                if transition == "BEGIN_DREAMING":
                    self.dreaming.begin_dreaming(self.inner)
                    self._last_dreaming_event = "BEGIN_DREAMING"
                    logger.info(
                        "[Coordinator] BEGIN DREAMING — cycle=%d fatigue=%.3f "
                        "breakdown=%s",
                        self.inner.cycle_count, self.dreaming.last_fatigue,
                        self.dreaming._last_fatigue_breakdown)
                    self._on_dream_begin()
                elif transition == "END_DREAMING":
                    summary = self.dreaming.end_dreaming(self.inner)
                    self._last_dreaming_event = "END_DREAMING"
                    if summary:
                        self._last_topology["dreaming_summary"] = summary
                    logger.info(
                        "[Coordinator] END DREAMING — cycle=%d readiness=%.3f "
                        "duration=%.1fs",
                        self.inner.cycle_count, self.dreaming.last_readiness,
                        summary.get("duration_s", 0) if summary else 0)
                    self._on_dream_end(summary)
            except Exception as e:
                logger.warning("[Coordinator] Dreaming check error: %s", e)

        return {
            "event": self._last_dreaming_event,
            "topology": self._last_topology,
            "nervous_signals": self._last_nervous_signals,
        }

    def tick_inner_only(
        self,
        body: Sequence[float],
        mind: Sequence[float],
        spirit: Sequence[float],
    ) -> tuple[dict[str, dict], dict[str, float]]:
        """
        Convenience: tick with just inner tensors.

        Returns:
            (inner_observables, inner_coherences) — coherences ready for sphere clocks.
        """
        inner_tensors = {
            "inner_body": body,
            "inner_mind": mind,
            "inner_spirit": spirit,
        }

        all_obs = self.observables.observe_inner(body, mind, spirit)

        # Update InnerState with inner observables (merge with existing outer)
        if self.inner:
            merged = dict(self.inner.observables)
            merged.update(all_obs)
            self.inner.update_observables(merged)

        coherences = {k: v["coherence"] for k, v in all_obs.items()}
        return all_obs, coherences

    def tick_outer_only(
        self,
        body: Sequence[float],
        mind: Sequence[float],
        spirit: Sequence[float],
    ) -> tuple[dict[str, dict], dict[str, float]]:
        """
        Convenience: tick with just outer tensors (from OUTER_TRINITY_STATE).

        Returns:
            (outer_observables, outer_coherences) — coherences ready for sphere clocks.
        """
        outer_obs = self.observables.observe_outer(body, mind, spirit)

        # Merge outer observables into InnerState
        if self.inner:
            merged = dict(self.inner.observables)
            merged.update(outer_obs)
            self.inner.update_observables(merged)

        coherences = {k: v["coherence"] for k, v in outer_obs.items()}
        return outer_obs, coherences

    def on_outer_snapshot(self, snapshot: dict) -> None:
        """
        Called when outer state snapshot arrives (STATE_SNAPSHOT from bus).

        Buffers the snapshot in InnerState for dreaming processing (T6).
        Also updates SpiritState with latest outer view.
        """
        if self.inner:
            self.inner.buffer_experience(snapshot)

        if self.spirit:
            self.spirit.assemble(outer_snapshot=snapshot)

    def assemble_spirit(self) -> None:
        """Force a SpiritState assembly from current InnerState."""
        if self.spirit and self.inner:
            self.spirit.assemble(
                observables=self.inner.observables,
                inner_snapshot=self.inner.snapshot(),
            )

    # ── Dream side-effects (moved from spirit_worker π-handler) ─────

    def _consume_dream_gaba(self, amount: float, source: str) -> None:
        """Consume GABA during dream consolidation — biological sleep pressure clearance.

        Each consolidation event (NS training, meta-reasoning, distillation)
        uses up a small amount of GABA. Dreams that do useful work end sooner.
        This couples dream duration to actual consolidation utility.
        """
        if not self._neuromod_system:
            return
        _gaba = self._neuromod_system.modulators.get("GABA")
        if _gaba:
            _before = _gaba.level
            _gaba.level = max(0.05, _gaba.level - amount)
            if _before - _gaba.level > 0.001:
                logger.debug("[DreamGABA] Consumed %.3f from %s (%.3f→%.3f)",
                             amount, source, _before, _gaba.level)

    def _on_dream_begin(self) -> None:
        """Side-effects when dreaming begins. All 5 actions from π-CLUSTER_END."""
        # 1. Life force dreaming state
        if self._life_force:
            try:
                self._life_force.set_dreaming(True)
            except Exception as e:
                logger.warning("[Coordinator] Life force dream start error: %s", e)

        # 2. NS dream consolidation (2× learning boost)
        if self.neural_ns:
            try:
                consolidation = self.neural_ns.consolidate_training(
                    boost_factor=2.0)
                logger.info("[Coordinator] Dream consolidation: %s",
                            {k: v.get("loss", "?")
                             for k, v in consolidation.items()}
                            if consolidation else "empty")
                if consolidation:
                    self._consume_dream_gaba(0.01, "ns_consolidation")
            except Exception as e:
                logger.warning("[Coordinator] Dream consolidation error: %s", e)

        # 2b. Reasoning engine dream consolidation
        if hasattr(self, '_reasoning_engine') and self._reasoning_engine:
            try:
                r_cons = self._reasoning_engine.consolidate_training(boost_factor=2.0)
                if r_cons.get("trained"):
                    logger.info("[Coordinator] Reasoning dream training: %d samples, loss=%.4f",
                                r_cons["samples"], r_cons["avg_loss"])
                    self._reasoning_engine.save_all()
                    self._consume_dream_gaba(0.01, "reasoning_consolidation")
            except Exception as e:
                logger.warning("[Coordinator] Reasoning consolidation error: %s", e)

        # 2c. Meta-wisdom decay during dreams (M2)
        if hasattr(self, '_meta_wisdom') and self._meta_wisdom:
            try:
                decay_stats = self._meta_wisdom.dream_decay()
                if decay_stats.get("pruned", 0) > 0 or decay_stats.get("remaining", 0) > 0:
                    logger.info("[Coordinator] Meta-wisdom decay: %d pruned, %d remaining (conf=%.3f, crystallized=%d)",
                                decay_stats["pruned"], decay_stats["remaining"],
                                decay_stats["avg_confidence"], decay_stats.get("crystallized", 0))
            except Exception as e:
                logger.warning("[Coordinator] Meta-wisdom decay error: %s", e)

        # 2d. Autoencoder dream training (M3)
        if hasattr(self, '_meta_autoencoder') and self._meta_autoencoder:
            try:
                if hasattr(self, '_chain_archive') and self._chain_archive:
                    ae_stats = self._meta_autoencoder.dream_train(
                        chain_archive=self._chain_archive, batch_size=32)
                    if ae_stats.get("trained"):
                        logger.info("[Coordinator] Autoencoder: recon=%.4f contrastive=%.4f "
                                    "%d samples, %d embeddings updated (step %d)",
                                    ae_stats["recon_loss"], ae_stats["contrastive_loss"],
                                    ae_stats["samples"], ae_stats["embeddings_updated"],
                                    ae_stats["total_steps"])
                        self._meta_autoencoder.save()
            except Exception as e:
                logger.warning("[Coordinator] Autoencoder training error: %s", e)

        # 2e. Meta-reasoning dream consolidation (System 1)
        if hasattr(self, '_meta_engine') and self._meta_engine:
            try:
                meta_cons = self._meta_engine.consolidate_training(boost_factor=2.0)
                if meta_cons.get("trained"):
                    logger.info("[Coordinator] Meta-reasoning dream: %d samples, loss=%.4f (updates=%d)",
                                meta_cons["samples"], meta_cons["avg_loss"],
                                meta_cons.get("total_updates", 0))
                    self._meta_engine.save_all()
                    self._consume_dream_gaba(0.015, "meta_reasoning_consolidation")
            except Exception as e:
                logger.warning("[Coordinator] Meta-reasoning consolidation error: %s", e)

        # 2f. Mini-reasoner dream consolidation
        if hasattr(self, '_mini_registry') and self._mini_registry:
            try:
                mini_stats = self._mini_registry.consolidate_all(boost_factor=2.0)
                logger.info("[Coordinator] Mini-reasoner dream training: %s",
                            {d: f"{s.get('samples', 0)}s/L{s.get('loss', 0):.4f}"
                             for d, s in mini_stats.items() if s.get("samples", 0) > 0})
                self._mini_registry.save_all()
            except Exception as e:
                logger.warning("[Coordinator] Mini-reasoner consolidation error: %s", e)

        # 3. Experience Orchestrator dream distillation
        if self._exp_orchestrator:
            try:
                _cycle = self.inner.cycle_count if self.inner else 1
                insights = self._exp_orchestrator.distill_cycle(
                    dream_cycle=_cycle,
                    current_epoch_id=self._last_epoch_id)
                logger.info("[Coordinator] Dream distillation: %d insights",
                            len(insights))
                if insights:
                    self._consume_dream_gaba(0.01 * min(len(insights), 5),
                                             "experience_distillation")
            except Exception as e:
                logger.warning("[Coordinator] Dream distillation error: %s", e)

        # 4. Neuromod clearance boost (GABA=1.0 normal, others=1+GABA×3)
        # Uses neuromodulator_system (neurotransmitter layer), not neural_ns._hormonal (program hormones)
        if self._neuromod_system:
            try:
                _gaba = self._neuromod_system.modulators.get("GABA")
                _gaba_val = _gaba.level if _gaba else 0.3
                for _mn, _mm in self._neuromod_system.modulators.items():
                    if _mn == "GABA":
                        # GABA: normal clearance during dreams. Production-side
                        # coupling to metabolic_drain handles natural decline.
                        _mm._dream_clearance_boost = 1.0
                    else:
                        _mm._dream_clearance_boost = 1.0 + _gaba_val * 3.0
                logger.info("[Coordinator] Neuromod dream boost: GABA=1.0x(normal), "
                            "others=%.2f", 1.0 + _gaba_val * 3.0)
            except Exception as e:
                logger.warning("[Coordinator] Neuromod boost error: %s", e)

    def _on_dream_end(self, summary: dict | None) -> None:
        """Side-effects when dreaming ends. All actions from π-CLUSTER_START."""
        # 5. Life force wake state
        if self._life_force:
            try:
                self._life_force.set_dreaming(False)
            except Exception as e:
                logger.warning("[Coordinator] Life force wake error: %s", e)

        # 6. Store distilled insights in e_mem
        if summary and self._e_mem:
            _distilled = summary.get("distilled_insights", [])
            for insight in _distilled:
                try:
                    self._e_mem.store_insight(
                        insight, self.inner.cycle_count if self.inner else 0)
                except Exception:
                    pass
            # rFP #3 Phase 3: clustering telemetry (tune via TUNING_DATABASE
            # ARCH-TUNE-005). Reading dims/cluster_sizes/peak_sig at a glance
            # shows whether clustering is fragmented (all size 1), over-merged
            # (one big cluster), or healthy.
            if _distilled:
                _dims = [int(ins.get("dim", 0)) for ins in _distilled]
                _sizes = [int(ins.get("num_samples", 1)) for ins in _distilled]
                # Defensive float() cast — dreaming.py now emits native Python
                # floats, but this belt-and-suspenders avoids np.float64(...)
                # leaks if anyone else feeds this log in the future.
                _sigs = [round(float(ins.get("significance", 0.0)), 4) for ins in _distilled]
                logger.info(
                    "[Coordinator] Dream cycle %d: %d insights → e_mem "
                    "dims=%s cluster_sizes=%s peak_sig=%s",
                    self.inner.cycle_count if self.inner else 0,
                    len(_distilled), _dims, _sizes, _sigs)
            # Prune stale insights every 10th cycle
            if self.inner and self.inner.cycle_count % 10 == 0:
                try:
                    self._e_mem.prune_stale()
                except Exception:
                    pass

        # 7. Neuromod restoration (reset clearance boost, resensitize)
        if self._neuromod_system:
            try:
                for _mm in self._neuromod_system.modulators.values():
                    _mm._dream_clearance_boost = 1.0
                    _mm.sensitivity = max(0.5, min(2.0,
                        (_mm.sensitivity + 1.0) / 2.0))
            except Exception as e:
                logger.warning("[Coordinator] Neuromod restore error: %s", e)

    def get_stats(self) -> dict:
        """Coordinator statistics."""
        # Serialize topology — distance_matrix has tuple keys, convert to strings
        topology = dict(self._last_topology)
        if "distance_matrix" in topology:
            topology["distance_matrix"] = {
                f"{a}:{b}": round(v, 6)
                for (a, b), v in topology["distance_matrix"].items()
            }

        dreaming_stats = {}
        if self.dreaming:
            dreaming_stats = {
                "fatigue": self.inner.fatigue if self.inner else 0.0,
                "readiness": self.inner.readiness if self.inner else 0.0,
                "fatigue_threshold": self.dreaming.fatigue_threshold,
                "readiness_threshold": self.dreaming.readiness_threshold,
                "experience_buffer_size": len(self.inner._experience_buffer) if self.inner else 0,
                "cycle_count": self.dreaming._cycle_count,
                # I-017 visibility: distilled insights accumulated across cycles
                "distilled_count": getattr(self.dreaming,
                                           "_distilled_count", 0),
            }

        neural_ns_stats = {}
        if self.neural_ns:
            neural_ns_stats = self.neural_ns.get_stats()

        return {
            "tick_count": self._tick_count,
            "has_vm": self.vm is not None,
            "has_nervous_system": self.nervous_system is not None,
            "has_neural_ns": self.neural_ns is not None,
            "has_topology": self.topology is not None,
            "has_dreaming": self.dreaming is not None,
            "is_dreaming": self.inner.is_dreaming if self.inner else False,
            "cycle_count": self.inner.cycle_count if self.inner else 0,
            "last_dreaming_event": self._last_dreaming_event,
            "dreaming": dreaming_stats,
            "nervous_signals": self._last_nervous_signals,
            "neural_nervous_system": neural_ns_stats,
            "topology": topology,
        }
