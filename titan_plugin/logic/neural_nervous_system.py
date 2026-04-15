"""
titan_plugin/logic/neural_nervous_system.py — V5 Neural NervousSystem.

Config-driven registry of neural reflex programs. Drop-in replacement for
the V4 hand-coded NervousSystem. Each program is a tiny learned neural network
that produces urgency signals from observables.

Programs register from titan_params.toml. New programs = just add TOML entry.
TitanVM programs kept as baseline for supervision (bootstrap) and fallback.

Training phases:
  1. Bootstrap (0-warmup_steps): Imitate TitanVM programs (supervision)
  2. Blending (transition zone): Mix supervision + outcome rewards
  3. Autonomous (warmup_steps+): Pure outcome-driven learning
"""
import logging
import os
import json
import threading
import time

import numpy as np

from .neural_reflex_net import NeuralReflexNet, NervousTransitionBuffer
from .observation_space import ObservationSpace
from .hormonal_pressure import HormonalSystem, extract_stimuli
from .inner_memory import InnerMemoryStore

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_WARMUP_STEPS = 500
DEFAULT_TRAIN_EVERY_N = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_SAVE_EVERY_N = 50
DEFAULT_BUFFER_MAX = 2000


class NeuralNervousSystem:
    """
    V5 Neural Nervous System — learned reflexes that adapt from experience.

    Drop-in replacement for NervousSystem.evaluate():
      - Same input: observables dict {part_name: {5 metrics}}
      - Same output: list of signal dicts [{system, urgency, ...}]
      - But urgencies are LEARNED, not hand-coded

    Programs are loaded from config. Adding a new program post-V5 requires
    only a TOML config entry — no code changes.
    """

    def __init__(
        self,
        config: dict,
        data_dir: str = "./data/neural_nervous_system",
        vm_nervous_system=None,
    ):
        """
        Args:
            config: [neural_nervous_system] section from titan_params.toml
            data_dir: persistence directory for weights + buffers
            vm_nervous_system: V4 NervousSystem for baseline/fallback
        """
        self.observation_space = ObservationSpace()
        self.vm_fallback = vm_nervous_system
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Config
        self._warmup_steps = config.get("warmup_steps", DEFAULT_WARMUP_STEPS)
        self._train_every_n = config.get("train_every_n", DEFAULT_TRAIN_EVERY_N)
        self._batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)
        self._save_every_n = config.get("save_every_n", DEFAULT_SAVE_EVERY_N)
        self._keep_vm_parallel = config.get("keep_vm_parallel", True)

        # Programs + buffers
        self.programs: dict[str, NeuralReflexNet] = {}
        self.buffers: dict[str, NervousTransitionBuffer] = {}

        # Training state
        self._total_transitions: int = 0
        self._total_train_steps: int = 0
        self._last_train_ts: float = 0.0
        self._steps_at_last_save: int = 0
        self._save_in_progress: bool = False  # Guard for background save thread
        self._backup_lock = threading.Lock()  # Protects SQLite backup file I/O

        # Load programs from config
        programs_config = config.get("programs", {})
        for name, prog_cfg in programs_config.items():
            if isinstance(prog_cfg, dict) and prog_cfg.get("enabled", True):
                self._register_program(name.upper(), prog_cfg)

        # Hormonal Pressure System
        hormone_params = {}
        for name, prog_cfg in programs_config.items():
            if isinstance(prog_cfg, dict) and prog_cfg.get("enabled", True):
                hp = {}
                for key in ("hormone_base_rate", "hormone_stimulus_sensitivity",
                            "hormone_decay_rate", "hormone_fire_threshold",
                            "hormone_refractory_strength", "hormone_refractory_decay",
                            "hormone_dna_bias"):
                    short_key = key.replace("hormone_", "")
                    # Map config keys to HormonalPressure param names
                    param_map = {
                        "base_rate": "base_secretion_rate",
                        "stimulus_sensitivity": "stimulus_sensitivity",
                        "decay_rate": "decay_rate",
                        "fire_threshold": "fire_threshold",
                        "refractory_strength": "refractory_strength",
                        "refractory_decay": "refractory_decay",
                        "dna_bias": "dna_sensitivity_bias",
                    }
                    if key in prog_cfg and short_key in param_map:
                        hp[param_map[short_key]] = prog_cfg[key]
                if hp:
                    hormone_params[name.upper()] = hp

        self._hormonal = HormonalSystem(
            program_names=list(self.programs.keys()),
            hormone_params=hormone_params if hormone_params else None,
        )
        self._hormonal_enabled = True
        self._is_dreaming = False
        self._last_eval_ts = time.time()

        # Neuromodulator modulation dict — set by spirit_worker from NeuromodulatorSystem
        # Keys: learning_rate_gain, fire_threshold_gain, accumulation_rate_gain,
        #        sensory_gain, global_threshold_raise, intrinsic_motivation, etc.
        self._modulation: dict = {}

        # Inner Memory — records all internal experiences for self-learning
        mem_path = os.path.join(os.path.dirname(data_dir), "inner_memory.db")
        self._inner_memory = InnerMemoryStore(mem_path)

        # Feed temporal events from inner memory into hormone stimuli
        self._hormonal_events: dict = self._refresh_hormonal_events()
        # Maturity signals from Titan's emergent time (sphere clocks, epochs, etc.)
        self._maturity_signals: dict = {
            "great_epochs": 0, "sphere_radius": 1.0,
            "consciousness_epochs": 0,
        }

        # Load persisted state
        self._load_all()

        logger.info(
            "[NeuralNS] Initialized: %d programs, warmup=%d, data_dir=%s",
            len(self.programs), self._warmup_steps, data_dir,
        )

    def _register_program(self, name: str, config: dict) -> None:
        """Register a neural program from config."""
        feature_set = config.get("input_features", "standard")
        input_dim = ObservationSpace.get_dim(feature_set)
        buffer_max = config.get("buffer_max", DEFAULT_BUFFER_MAX)

        # Compute hidden sizes based on input dim
        if input_dim <= 35:
            h1, h2 = 32, 16       # core (30D)
        elif input_dim <= 60:
            h1, h2 = 48, 24       # standard (55D)
        elif input_dim <= 92:
            h1, h2 = 64, 32       # enriched (79D) or extended/full (75-88D)
        else:
            h1, h2 = 80, 40       # full_enriched (112D) — personality programs

        net = NeuralReflexNet(
            name=name,
            input_dim=input_dim,
            hidden_1=config.get("hidden_1", h1),
            hidden_2=config.get("hidden_2", h2),
            learning_rate=config.get("learning_rate", 0.001),
            fire_threshold=config.get("fire_threshold", 0.3),
        )
        net._feature_set = feature_set
        net._layer = config.get("layer", "inner")  # inner (autonomic) or outer (somatic)
        net._body_parts = config.get("body_parts", [])  # Trinity body parts this program serves
        net._action_helpers = config.get("action_helpers", [])  # Agency helpers for direct dispatch
        self.programs[name] = net
        self.buffers[name] = NervousTransitionBuffer(max_size=buffer_max)
        logger.info("[NeuralNS] Registered %s program '%s' (%dD %s, threshold=%.2f%s)",
                    net._layer.upper(), name, input_dim, feature_set, net.fire_threshold,
                    f", helpers={net._action_helpers}" if net._action_helpers else "")

    # ── Evaluate (Drop-In Replacement) ────────────────────────────

    def evaluate(self, observables: dict, temporal: dict = None) -> list[dict]:
        """
        Evaluate all registered programs. DROP-IN replacement for
        NervousSystem.evaluate() — same input, same output format.

        Fire decision uses HORMONAL PRESSURE (not static threshold):
        1. NN computes urgency (informational signal)
        2. NN urgency feeds as stimulus to the hormone
        3. Environmental stimuli also feed the hormone
        4. Hormone accumulates over time (cross-talk, circadian)
        5. Fire when pressure exceeds learned threshold

        Also runs VM baseline in parallel for supervision + comparison.

        Args:
            temporal: Optional dict with circadian features from π-heartbeat:
                pi_phase, cluster_progress, developmental_age_norm,
                time_since_dream, heartbeat_ratio
        """
        # B1: Build temporal feature vector for appending to NN input
        self._temporal_vec = None
        if temporal:
            self._temporal_vec = np.array([
                temporal.get("pi_phase", 0.0),
                temporal.get("cluster_progress", 0.0),
                temporal.get("developmental_age_norm", 0.0),
                temporal.get("time_since_dream", 0.0),
                temporal.get("heartbeat_ratio", 0.0),
            ], dtype=np.float64)
        signals = []
        now = time.time()
        dt = min(now - self._last_eval_ts, 30.0)
        self._last_eval_ts = now

        # Run VM baseline for comparison/supervision
        vm_signals = {}
        if self.vm_fallback and self._keep_vm_parallel:
            try:
                for sig in self.vm_fallback.evaluate(observables):
                    vm_signals[sig["system"]] = sig.get("urgency", 0.0)
            except Exception:
                pass

        # Refresh temporal events from inner memory (feeds boredom stimuli)
        self._hormonal_events = self._refresh_hormonal_events()

        # Extract environmental stimuli for hormones
        env_stimuli = {}
        if self._hormonal_enabled:
            try:
                obs_data = self.observation_space._observables or {}
                topo_data = self.observation_space._topology or {}
                dream_data = {
                    "fatigue": getattr(self.observation_space, '_fatigue', 0.0),
                    "readiness": getattr(self.observation_space, '_readiness', 0.0),
                }
                env_stimuli = extract_stimuli(
                    obs_data, topo_data, dream_data, self._hormonal_events)
            except Exception as e:
                logger.debug("[NeuralNS] Stimulus extraction error: %s", e)

            # Update maturity from Titan's emergent time signals
            total_fires = sum(h.fire_count for h in self._hormonal._hormones.values())
            self._hormonal.update_maturity(
                great_epochs=self._maturity_signals.get("great_epochs", 0),
                sphere_radius=self._maturity_signals.get("sphere_radius", 1.0),
                consciousness_epochs=self._maturity_signals.get("consciousness_epochs", 0),
                total_fires=total_fires,
            )

            # Accumulate all hormones with environmental stimuli
            # Self-emergent governors: GABA, Chi, neuromod gain
            _gov_gaba = self._modulation.get("gaba_level", 0.35)
            _gov_chi = self._modulation.get("chi_total", 0.6)
            _gov_accum = self._modulation.get("accumulation_rate_gain", 1.0)
            self._hormonal.accumulate_all(
                env_stimuli, dt, self._is_dreaming,
                gaba_level=_gov_gaba,
                chi_total=_gov_chi,
                accumulation_rate_gain=_gov_accum)

        # Store ALL program urgencies for reasoning engine gut signals
        self._all_urgencies = {}

        for name, net in self.programs.items():
            try:
                input_vec = self.observation_space.build_input(net._feature_set)
                # B1: Append temporal features for circadian awareness
                if self._temporal_vec is not None:
                    input_vec = np.concatenate([input_vec, self._temporal_vec])
                # Neuromodulator modulation: NE scales sensory input gain
                _sensory_gain = self._modulation.get("sensory_gain", 1.0)
                if _sensory_gain != 1.0:
                    input_vec = input_vec * max(0.5, min(2.0, _sensory_gain))
                # Safety: truncate if temporal features made it too long
                if len(input_vec) != net.input_dim:
                    if self._total_transitions <= 2005:
                        logger.warning("[NeuralNS] %s dim mismatch: input=%d, net=%d",
                                       name, len(input_vec), net.input_dim)
                    input_vec = input_vec[:net.input_dim]
                urgency = net.forward(input_vec)

                # NaN/inf fallback
                if not np.isfinite(urgency):
                    urgency = vm_signals.get(name, 0.0)
                    logger.warning("[NeuralNS] Non-finite from %s, using VM fallback", name)

                # Store raw urgency for all programs (reasoning engine reads this)
                self._all_urgencies[name] = float(urgency)

                vm_baseline = vm_signals.get(name, 0.0)

                # NN output also feeds hormone as stimulus (0.3 weight)
                # Neuromodulator modulation: scale accumulation rate
                hormone = self._hormonal.get_hormone(name) if self._hormonal_enabled else None
                _accum_gain = self._modulation.get("accumulation_rate_gain", 1.0)
                if hormone:
                    hormone.accumulate(
                        float(urgency) * 0.3 * _accum_gain, dt=0.1,
                        gaba_level=_gov_gaba,
                        chi_total=_gov_chi,
                        accumulation_rate_gain=_accum_gain)

                # FIRE DECISION: hormonal pressure (if enabled) or legacy threshold
                # Neuromodulator modulation: GABA raises thresholds globally
                _threshold_raise = self._modulation.get("global_threshold_raise", 1.0)
                if hormone and self._hormonal_enabled:
                    # Temporarily adjust threshold for this evaluation
                    _orig_thresh = hormone.threshold
                    hormone.threshold = _orig_thresh * _threshold_raise
                    fired = hormone.should_fire()
                    hormone.threshold = _orig_thresh  # Restore
                else:
                    fired = bool(urgency > (net.fire_threshold * _threshold_raise))

                # Always record transition for training (even if not fired)
                self._record_transition(
                    name, input_vec, urgency, vm_baseline, fired)

                if fired:
                    net.fire_count += 1
                    intensity = hormone.fire() if hormone else float(urgency)
                    signals.append({
                        "system": name,
                        "urgency": round(float(urgency), 4),
                        "intensity": round(intensity, 4),
                        "vm_baseline": round(vm_baseline, 4),
                        "delta": round(float(urgency) - vm_baseline, 4),
                        "learned": True,
                        "hormonal": self._hormonal_enabled,
                        "duration_ms": 0.0,
                    })
                    # Record fire in inner memory
                    try:
                        # Extract body/mind/spirit from tier1 input (first 30 values = 6 parts × 5)
                        t1 = self.observation_space._tier1
                        body_vals = t1[0:5].tolist() if len(t1) >= 5 else None
                        mind_vals = t1[5:10].tolist() if len(t1) >= 10 else None
                        spirit_vals = t1[10:15].tolist() if len(t1) >= 15 else None
                        self._inner_memory.record_program_fire(
                            program=name, layer=getattr(net, '_layer', 'inner'),
                            intensity=intensity,
                            pressure=hormone.level if hormone else float(urgency),
                            threshold=hormone.threshold if hormone else net.fire_threshold,
                            stimulus=env_stimuli.get(name, 0.0),
                            body=body_vals, mind=mind_vals, spirit=spirit_vals,
                        )
                    except Exception as e:
                        logger.debug("[NeuralNS] Inner memory fire record error: %s", e)

            except Exception as e:
                logger.debug("[NeuralNS] %s eval error: %s", name, e)
                # Fallback to VM signal
                vm_urg = vm_signals.get(name, 0.0)
                if vm_urg > 0:
                    signals.append({
                        "system": name,
                        "urgency": round(vm_urg, 4),
                        "vm_baseline": round(vm_urg, 4),
                        "delta": 0.0,
                        "learned": False,
                    })

        return signals

    def set_dreaming(self, is_dreaming: bool) -> None:
        """Update dreaming state for circadian modulation."""
        self._is_dreaming = is_dreaming

    def update_hormonal_events(self, events: dict) -> None:
        """Update time-based events for stimulus extraction."""
        self._hormonal_events.update(events)

    def _refresh_hormonal_events(self) -> dict:
        """Refresh time-based events from inner memory for hormone stimuli."""
        try:
            return {
                "time_since_explore": self._inner_memory.time_since_last("explore"),
                "time_since_social": self._inner_memory.time_since_last("social"),
                "time_since_create": self._inner_memory.time_since_last("create"),
            }
        except Exception:
            return {}

    @property
    def inner_memory(self) -> InnerMemoryStore:
        """Expose inner memory for external integration (Agency, Spirit)."""
        return self._inner_memory

    def update_maturity_signals(
        self,
        great_epochs: int = 0,
        sphere_radius: float = 1.0,
        consciousness_epochs: int = 0,
    ) -> None:
        """
        Feed Titan's emergent time signals into the hormonal maturity system.

        Called by the coordinator with data from:
        - UnifiedSpirit (GREAT EPOCH count)
        - SphereClockEngine (mean inner radius — how contracted/balanced)
        - ConsciousnessLoop (epoch count — depth of self-reflection)

        This is NOT human clock time — it's Titan's own developmental time.
        """
        self._maturity_signals = {
            "great_epochs": great_epochs,
            "sphere_radius": sphere_radius,
            "consciousness_epochs": consciousness_epochs,
        }

    # ── Training ──────────────────────────────────────────────────

    def _record_transition(self, name: str, observation: np.ndarray,
                           urgency: float, vm_baseline: float,
                           fired: bool) -> None:
        """Record a transition and maybe trigger training."""
        buf = self.buffers.get(name)
        if buf is None:
            return

        buf.add(
            observation=observation.tolist() if isinstance(observation, np.ndarray) else observation,
            urgency=urgency,
            vm_baseline=vm_baseline,
            reward=0.0,  # Updated later via record_outcome()
            fired=fired,
        )

        self._total_transitions += 1

        # Auto-train every N transitions
        if self._total_transitions % self._train_every_n == 0:
            self._train_all()

        # Auto-save every M training steps (uses floor division to handle
        # step jumps from multi-program training batches).
        # Runs in background thread to avoid blocking spirit_worker main loop
        # (save_all writes 24+ files and can take 3+ seconds).
        if (self._total_train_steps > 0 and
                self._total_train_steps - self._steps_at_last_save >= self._save_every_n):
            self._steps_at_last_save = self._total_train_steps
            if not self._save_in_progress:
                self._save_in_progress = True

                def _bg_save():
                    try:
                        self.save_all()
                    finally:
                        self._save_in_progress = False
                threading.Thread(target=_bg_save, daemon=True, name="ns-save").start()

        # Periodic hormonal save (every 200 transitions) — ensures hormone state
        # persists even when training steps don't trigger save_all()
        if (self._hormonal_enabled and
                self._total_transitions % 200 == 0):
            try:
                self._hormonal.save(
                    os.path.join(self.data_dir, "hormonal_state.json"))
            except Exception:
                pass

    def record_outcome(self, reward: float) -> None:
        """
        Called when interaction outcome is scored (via REFLEX_REWARD).
        Updates the most recent fired transition for each program.
        Also adapts hormone thresholds from the outcome.
        """
        for name, buf in self.buffers.items():
            if buf.last_fired:
                buf.update_last_reward(reward)
                # Adapt hormone threshold from outcome
                if self._hormonal_enabled:
                    self._hormonal.adapt(name, reward - 0.5)

    def _train_all(self) -> None:
        """Train all programs from their transition buffers.

        Neuromodulator modulation applied:
        - DA gain → learning_rate_gain (high DA = faster learning from reward)
        - ACh gain → training_frequency_gain (high ACh = more frequent training)
        """
        sup_weight = self._get_supervision_weight()
        _lr_gain = self._modulation.get("learning_rate_gain", 1.0)

        for name, net in self.programs.items():
            buf = self.buffers[name]
            if len(buf) < self._batch_size:
                continue

            obs, urgencies, vm_baselines, rewards, fired = buf.sample(self._batch_size)

            # Compute blended targets
            targets = self._compute_targets(
                vm_baselines, rewards, fired, urgencies, sup_weight)

            # Apply neuromodulator learning rate modulation (DA)
            original_lr = net.lr
            net.lr = original_lr * max(0.3, min(3.0, _lr_gain))
            loss = net.train_step(obs, targets.reshape(-1, 1))
            net.lr = original_lr  # Restore
            self._total_train_steps += 1

        self._last_train_ts = time.time()

    def _compute_targets(
        self, vm_baselines: np.ndarray, rewards: np.ndarray,
        fired: np.ndarray, urgencies: np.ndarray,
        supervision_weight: float,
    ) -> np.ndarray:
        """
        Compute training targets as blend of supervision + outcome.

        Supervision target: VM baseline urgency (imitate TitanVM)
        Outcome target: reward-modulated urgency
          - high reward + fired → encourage (target high)
          - high reward + not fired → missed opportunity (target moderate)
          - low reward + fired → discourage (target low)
          - low reward + not fired → correct restraint (target 0)
        """
        # Supervision component
        sup_targets = vm_baselines

        # Outcome component
        outcome_targets = np.zeros_like(rewards)
        for i in range(len(rewards)):
            r = rewards[i]
            f = fired[i]
            if r > 0.5 and f:
                outcome_targets[i] = min(1.0, r)           # Good fire
            elif r > 0.5 and not f:
                outcome_targets[i] = 0.5                    # Missed opportunity
            elif r <= 0.3 and f:
                outcome_targets[i] = 0.1                    # Bad fire
            else:
                outcome_targets[i] = 0.0                    # Correct restraint

        # Blend
        targets = supervision_weight * sup_targets + (1.0 - supervision_weight) * outcome_targets
        return np.clip(targets, 0.0, 1.0)

    def _get_supervision_weight(self) -> float:
        """Linear decay from 1.0 to 0.0 over warmup_steps."""
        if self._total_transitions >= self._warmup_steps:
            return 0.0
        return 1.0 - (self._total_transitions / self._warmup_steps)

    @property
    def training_phase(self) -> str:
        """Current training phase."""
        sw = self._get_supervision_weight()
        if sw >= 0.95:
            return "bootstrap"
        elif sw > 0.05:
            return "blending"
        return "autonomous"

    # ── Observation Space Update ──────────────────────────────────

    def update_observation_space(self, **kwargs) -> None:
        """Called by coordinator to refresh observation cache."""
        self.observation_space.update(**kwargs)

    # ── Emotional Self-Regulation ────────────────────────────────────

    def check_regulation(self) -> dict:
        """Detect runaway hormonal states needing regulation.

        Returns dict of {hormone_name: {"action": "dampen"/"amplify", "factor": float}}
        """
        if not self._hormonal_enabled or not self._hormonal:
            return {}

        hormones = self._hormonal._hormones
        if not hormones:
            return {}

        levels = [h.level for h in hormones.values()]
        mean_level = sum(levels) / len(levels) if levels else 0.0

        signals = {}
        for name, hormone in hormones.items():
            if hormone.level > mean_level * 2.5 and mean_level > 0.1:
                signals[name] = {"action": "dampen", "factor": 0.5}
            elif name in ("INSPIRATION", "CREATIVITY") and hormone.level > mean_level * 1.5 and mean_level > 0.1:
                signals[name] = {"action": "amplify", "factor": 1.2}

        return signals

    def apply_regulation(self, signals: dict):
        """Apply regulation signals to hormonal levels."""
        if not self._hormonal:
            return
        for name, signal in signals.items():
            hormone = self._hormonal.get_hormone(name)
            if not hormone:
                continue
            if signal["action"] == "dampen":
                hormone.level *= signal["factor"]
                logger.info("[NeuralNS] Regulated %s: dampened to %.3f", name, hormone.level)
            elif signal["action"] == "amplify":
                # Don't amplify level directly — sustain by reducing decay
                pass  # Amplification handled by sustaining accumulation rate

    # ── Dream Consolidation ────────────────────────────────────────

    def consolidate_training(self, boost_factor: float = 2.0) -> dict:
        """Run extra training pass during dreaming with boosted learning rate.

        Biological parallel: during REM, synaptic consolidation strengthens
        important connections and prunes weak ones.

        Args:
            boost_factor: Multiply learning rate by this during consolidation.
                Capped at 3.0 for safety.

        Returns:
            Dict of {program_name: {"loss": float, "transitions": int}}
        """
        boost_factor = min(boost_factor, 3.0)
        results = {}
        for name, net in self.programs.items():
            buf = self.buffers.get(name)
            if not buf or len(buf) < 10:
                continue
            try:
                # Temporarily boost learning rate
                original_lr = net.lr
                net.lr = original_lr * boost_factor

                # Train on recent transitions
                batch = buf.sample(min(100, len(buf)))
                if batch:
                    loss = net.train_batch(batch)
                    results[name] = {
                        "loss": round(float(loss), 6) if loss is not None else None,
                        "transitions": len(buf),
                    }

                # Restore original learning rate
                net.lr = original_lr
            except Exception as e:
                # Always restore learning rate even on error
                net.lr = getattr(net, '_original_lr_backup', net.lr)
                logger.debug("[NeuralNS] Consolidation error for %s: %s", name, e)

        logger.info("[NeuralNS] Dream consolidation: %d programs trained (boost=%.1f×)",
                    len(results), boost_factor)
        return results

    # ── Hot-Reload State Transfer ────────────────────────────────

    def get_state(self) -> dict:
        """Return runtime state for hot-reload (in-memory, no disk I/O).

        Neural weights are NOT included — they're already persisted to disk
        and will be loaded via _load_all() on reconstruction. This only
        captures the runtime counters, modulation, and hormonal state that
        would be lost on reload.
        """
        return {
            "total_transitions": self._total_transitions,
            "total_train_steps": self._total_train_steps,
            "last_train_ts": self._last_train_ts,
            "last_eval_ts": self._last_eval_ts,
            "modulation": dict(self._modulation),
            "is_dreaming": self._is_dreaming,
            "maturity_signals": dict(self._maturity_signals),
            "hormonal": self._hormonal.get_state() if self._hormonal_enabled else None,
        }

    def restore_state(self, state: dict) -> None:
        """Restore runtime state after hot-reload."""
        self._total_transitions = state.get("total_transitions", self._total_transitions)
        self._total_train_steps = state.get("total_train_steps", self._total_train_steps)
        self._last_train_ts = state.get("last_train_ts", self._last_train_ts)
        self._last_eval_ts = state.get("last_eval_ts", self._last_eval_ts)
        self._modulation = state.get("modulation", self._modulation)
        self._is_dreaming = state.get("is_dreaming", self._is_dreaming)
        self._maturity_signals = state.get("maturity_signals", self._maturity_signals)
        if self._hormonal_enabled and state.get("hormonal"):
            self._hormonal.restore_state(state["hormonal"])
        logger.info("[NeuralNS] Restored runtime state: transitions=%d, train_steps=%d",
                    self._total_transitions, self._total_train_steps)

    # ── Persistence ───────────────────────────────────────────────

    def save_all(self) -> None:
        """Persist all weights, buffers, and training state.

        Uses atomic writes (tmp→rename) to prevent corruption on crash.
        Rolling snapshots every 500 training steps (keeps last 3).
        """
        logger.info("[NeuralNS] save_all: %d transitions, %d train steps",
                    self._total_transitions, self._total_train_steps)
        for name, net in self.programs.items():
            net.save(os.path.join(self.data_dir, f"{name.lower()}_weights.json"))
        for name, buf in self.buffers.items():
            buf.save(os.path.join(self.data_dir, f"{name.lower()}_buffer.json"))

        # Save global training state (atomic)
        state = {
            "total_transitions": self._total_transitions,
            "total_train_steps": self._total_train_steps,
            "last_train_ts": self._last_train_ts,
            "program_names": list(self.programs.keys()),
        }
        ts_path = os.path.join(self.data_dir, "training_state.json")
        tmp = ts_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, ts_path)

        # Save hormonal state (atomic)
        if self._hormonal_enabled:
            h_path = os.path.join(self.data_dir, "hormonal_state.json")
            self._hormonal.save(h_path)

        # Rolling snapshot every 500 training steps (keeps last 3)
        if self._total_train_steps > 0 and self._total_train_steps % 500 == 0:
            import glob
            snap_dir = os.path.join(self.data_dir, "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            snap_prefix = os.path.join(snap_dir, f"step_{self._total_train_steps}")
            for name, net in self.programs.items():
                net.save(f"{snap_prefix}_{name.lower()}_weights.json")
            with open(f"{snap_prefix}_training_state.json", "w") as f:
                json.dump(state, f)
            # Prune old snapshots (keep last 3 sets)
            snap_steps = sorted(set(
                int(os.path.basename(p).split("_")[1])
                for p in glob.glob(os.path.join(snap_dir, "step_*_training_state.json"))
            ))
            for old_step in snap_steps[:-3]:
                for old_file in glob.glob(os.path.join(snap_dir, f"step_{old_step}_*")):
                    os.remove(old_file)
            logger.info("[NeuralNS] Rolling snapshot saved: step=%d", self._total_train_steps)

        logger.info("[NeuralNS] Saved %d programs, %d transitions, %d train steps",
                    len(self.programs), self._total_transitions, self._total_train_steps)

        # SQLite backup removed from save_all() (2026-03-24) — was adding 19MB
        # I/O every ~5s, blocking spirit_worker for 3-4s and causing 500+
        # DivineBus timeouts per session. 5-minute periodic backup in
        # spirit_worker.py:279-287 provides sufficient crash resilience.

    def _sqlite_backup_save(self) -> None:
        """Save complete NS training state to SQLite for crash/overwrite resilience."""
        import sqlite3
        db_path = os.path.join(self.data_dir, "ns_training_backup.db")
        with self._backup_lock:
            conn = sqlite3.connect(db_path, timeout=10)
            try:
                conn.execute("""CREATE TABLE IF NOT EXISTS ns_weights (
                    program TEXT PRIMARY KEY,
                    weights_json TEXT NOT NULL,
                    buffer_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )""")
                conn.execute("""CREATE TABLE IF NOT EXISTS ns_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )""")
                now = time.time()
                for name, net in self.programs.items():
                    # Serialize using same format as save() method
                    w_data = json.dumps({
                        "name": net.name, "input_dim": net.input_dim,
                        "hidden_1": net.hidden_1, "hidden_2": net.hidden_2,
                        "lr": net.lr, "fire_threshold": net.fire_threshold,
                        "feature_set": net._feature_set,
                        "total_updates": net.total_updates,
                        "last_loss": net.last_loss, "fire_count": net.fire_count,
                        "w1": net.w1.tolist(), "b1": net.b1.tolist(),
                        "w2": net.w2.tolist(), "b2": net.b2.tolist(),
                        "w3": net.w3.tolist(), "b3": net.b3.tolist(),
                    })
                    buf = self.buffers[name]
                    # Cast fired to plain bool (numpy.bool_ is not JSON-serializable)
                    fired_list = [bool(f) for f in buf._fired[-buf.max_size:]]
                    b_data = json.dumps({
                        "observations": buf._observations[-buf.max_size:],
                        "urgencies": buf._urgencies[-buf.max_size:],
                        "vm_baselines": buf._vm_baselines[-buf.max_size:],
                        "rewards": buf._rewards[-buf.max_size:],
                        "fired": fired_list,
                    })
                    conn.execute(
                        "INSERT OR REPLACE INTO ns_weights (program, weights_json, buffer_json, updated_at) VALUES (?, ?, ?, ?)",
                        (name, w_data, b_data, now))
                # Save meta
                meta = {
                    "total_transitions": str(self._total_transitions),
                    "total_train_steps": str(self._total_train_steps),
                    "last_train_ts": str(self._last_train_ts),
                    "program_names": json.dumps(list(self.programs.keys())),
                }
                for k, v in meta.items():
                    conn.execute("INSERT OR REPLACE INTO ns_meta (key, value) VALUES (?, ?)", (k, v))
                # Save hormonal if enabled
                if self._hormonal_enabled:
                    h_state = json.dumps(self._hormonal.get_state())
                    conn.execute("INSERT OR REPLACE INTO ns_meta (key, value) VALUES (?, ?)",
                                 ("hormonal_state", h_state))
                conn.commit()
            except Exception as e:
                logger.warning("NS backup save error: %s", e)
            finally:
                conn.close()

    def _sqlite_backup_load(self) -> bool:
        """Attempt to restore NS state from SQLite backup. Returns True on success."""
        import sqlite3
        db_path = os.path.join(self.data_dir, "ns_training_backup.db")
        if not os.path.exists(db_path):
            return False
        with self._backup_lock:
            try:
                conn = sqlite3.connect(db_path, timeout=10)
                try:
                    # Load meta
                    rows = conn.execute("SELECT key, value FROM ns_meta").fetchall()
                    meta = {k: v for k, v in rows}
                    self._total_transitions = int(meta.get("total_transitions", "0"))
                    self._total_train_steps = int(meta.get("total_train_steps", "0"))
                    self._last_train_ts = float(meta.get("last_train_ts", "0"))

                    # Load per-program weights + buffers
                    loaded = 0
                    for row in conn.execute("SELECT program, weights_json, buffer_json FROM ns_weights").fetchall():
                        name, w_json, b_json = row
                        if name in self.programs:
                            try:
                                # Restore weights using same format as load()
                                w_data = json.loads(w_json)
                                net = self.programs[name]
                                net.w1 = np.array(w_data["w1"], dtype=np.float64)
                                net.b1 = np.array(w_data["b1"], dtype=np.float64)
                                net.w2 = np.array(w_data["w2"], dtype=np.float64)
                                net.b2 = np.array(w_data["b2"], dtype=np.float64)
                                net.w3 = np.array(w_data["w3"], dtype=np.float64)
                                net.b3 = np.array(w_data["b3"], dtype=np.float64)
                                net.total_updates = w_data.get("total_updates", 0)
                                net.last_loss = w_data.get("last_loss", 0.0)
                                net.fire_count = w_data.get("fire_count", 0)
                                # Restore buffer
                                b_data = json.loads(b_json)
                                buf = self.buffers[name]
                                buf._observations = b_data.get("observations", [])
                                buf._urgencies = b_data.get("urgencies", [])
                                buf._vm_baselines = b_data.get("vm_baselines", [])
                                buf._rewards = b_data.get("rewards", [])
                                buf._fired = b_data.get("fired", [])
                                loaded += 1
                            except Exception as e:
                                logger.warning("NS backup load error for program %s: %s", name, e)

                    # Load hormonal
                    if self._hormonal_enabled and "hormonal_state" in meta:
                        try:
                            self._hormonal.restore_state(json.loads(meta["hormonal_state"]))
                        except Exception as e:
                            logger.warning("NS backup hormonal restore error: %s", e)

                    logger.warning(
                        "[NeuralNS] RECOVERED from SQLite backup: %d/%d programs, "
                        "transitions=%d, train_steps=%d",
                        loaded, len(self.programs), self._total_transitions, self._total_train_steps)
                    return loaded > 0
                finally:
                    conn.close()
            except Exception as e:
                logger.warning("[NeuralNS] SQLite backup load error: %s", e)
                return False

    def _load_all(self) -> None:
        """Load persisted state on boot."""
        # Load training state
        state_path = os.path.join(self.data_dir, "training_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self._total_transitions = state.get("total_transitions", 0)
                self._total_train_steps = state.get("total_train_steps", 0)
                self._last_train_ts = state.get("last_train_ts", 0.0)
                self._steps_at_last_save = self._total_train_steps
            except Exception:
                pass

        # Load per-program weights + buffers (with dimension migration support)
        loaded = 0
        migrated = 0
        for name, net in self.programs.items():
            w_path = os.path.join(self.data_dir, f"{name.lower()}_weights.json")
            b_path = os.path.join(self.data_dir, f"{name.lower()}_buffer.json")

            # Check for dimension migration before loading
            _needs_migration = False
            if os.path.exists(w_path):
                try:
                    with open(w_path) as f:
                        _saved = json.load(f)
                    _needs_migration = _saved.get("input_dim", 55) != net.input_dim
                except Exception:
                    pass

            _cfg_feature_set = net._feature_set  # Config-defined (from _register_program)
            if net.load(w_path):
                loaded += 1
            # Restore config feature_set — load() may overwrite with saved "standard"
            # when dimension migration changed the feature set
            net._feature_set = _cfg_feature_set

            if _needs_migration:
                # Buffer observations are old dimension — must clear
                logger.warning("[NeuralNS] %s: dimension migrated (%dD→%dD), clearing transition buffer",
                               name, _saved.get("input_dim", 55), net.input_dim)
                self.buffers[name] = NervousTransitionBuffer(max_size=DEFAULT_BUFFER_MAX)
                migrated += 1
            else:
                self.buffers[name].load(b_path)

        if loaded > 0:
            logger.info("[NeuralNS] Loaded %d/%d program weights (phase=%s, transitions=%d%s)",
                        loaded, len(self.programs), self.training_phase,
                        self._total_transitions,
                        f", migrated={migrated}" if migrated > 0 else "")
        elif loaded == 0 and self._total_transitions == 0:
            # JSON files missing or empty — try SQLite backup recovery
            logger.warning("[NeuralNS] No JSON weights found — attempting SQLite backup recovery")
            if self._sqlite_backup_load():
                loaded = len(self.programs)  # Recovered

        # Load hormonal state (or create initial snapshot if first boot with hormones)
        if self._hormonal_enabled:
            h_path = os.path.join(self.data_dir, "hormonal_state.json")
            if os.path.exists(h_path):
                self._hormonal.load(h_path)
                # Sanitize impossible state: refractory > 0 with fire_count = 0.
                # Only fire() sets refractory (via refractory_strength), so
                # fire_count=0 + refractory>0 means orphaned persistence data
                # from hot-reload state transfer (2026-03-22 4dc8ed6).
                for _h_name, _hormone in self._hormonal._hormones.items():
                    if _hormone.fire_count == 0 and _hormone.refractory > 0:
                        logger.warning(
                            "[NeuralNS] Sanitize %s: refractory=%.3f with "
                            "fire_count=0 — resetting to 0.0",
                            _h_name, _hormone.refractory)
                        _hormone.refractory = 0.0
            else:
                # First boot with hormonal system — persist initial state
                self._hormonal.save(h_path)
                logger.info("[NeuralNS] Created initial hormonal_state.json")

    # ── Outer Program Dispatch ─────────────────────────────────────

    def get_outer_dispatch_signals(self) -> list[dict]:
        """
        Get signals from OUTER (somatic) programs that fired and have
        action_helpers configured. Used by Agency for autonomy-first dispatch.

        Returns list of dicts with:
          - system: program name
          - urgency: signal strength
          - helpers: list of helper names to try
          - layer: "outer"
        """
        signals = []
        for name, net in self.programs.items():
            if net._layer != "outer" or not net._action_helpers:
                continue
            if net.fire_count <= 0:
                continue
            # Respect hormonal refractory — don't dispatch while recovering
            hormone = self._hormonal.get_hormone(name) if self._hormonal_enabled else None
            if hormone and hormone.refractory >= 0.15:
                continue
            # Check if this program fired in the most recent evaluation
            try:
                input_vec = self.observation_space.build_input(net._feature_set)
                urgency = net.forward(input_vec)
                if urgency > net.fire_threshold:
                    signals.append({
                        "system": name,
                        "urgency": round(float(urgency), 4),
                        "helpers": net._action_helpers,
                        "layer": "outer",
                    })
            except Exception:
                pass
        return signals

    def get_inner_signals_summary(self) -> dict[str, float]:
        """
        Get summary of all INNER (autonomic) program urgencies.
        Used as input for outer programs (composition).
        """
        summary = {}
        for name, net in self.programs.items():
            if net._layer != "inner":
                continue
            try:
                input_vec = self.observation_space.build_input(net._feature_set)
                urgency = net.forward(input_vec)
                summary[name.lower()] = round(float(urgency), 4)
            except Exception:
                summary[name.lower()] = 0.0
        return summary

    # ── Stats / API ───────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Full stats for API."""
        stats = {
            "version": "v5_neural",
            "training_phase": self.training_phase,
            "supervision_weight": round(self._get_supervision_weight(), 4),
            "total_transitions": self._total_transitions,
            "total_train_steps": self._total_train_steps,
            "warmup_steps": self._warmup_steps,
            "last_train_ts": self._last_train_ts,
            "programs": {
                name: {
                    **net.get_stats(),
                    "buffer_size": len(self.buffers.get(name, [])),
                    "layer": getattr(net, '_layer', 'inner'),
                    "action_helpers": getattr(net, '_action_helpers', []),
                }
                for name, net in self.programs.items()
            },
        }
        # Add hormonal system state
        if self._hormonal_enabled:
            stats["hormonal_system"] = {
                name: {
                    "level": round(h.level, 4),
                    "threshold": round(h.threshold, 4),
                    "refractory": round(h.refractory, 4),
                    "fire_count": h.fire_count,
                    "peak_level": round(h.peak_level, 4),
                }
                for name, h in self._hormonal._hormones.items()
            }
            stats["maturity"] = round(self._hormonal.maturity, 4)
        # Neuromodulator modulation currently active
        if self._modulation:
            stats["neuromodulator_modulation"] = {
                k: round(v, 4) for k, v in self._modulation.items()
            }
        return stats
