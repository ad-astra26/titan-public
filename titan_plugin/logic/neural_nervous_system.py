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
import math
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


# PERSISTENCE_BY_DESIGN: NeuralNervousSystem._save_in_progress (async write
# guard flag) + _hormonal_enabled (runtime feature flag) + _steps_at_last_save
# (observability counter) are transient — reset on boot is correct. Actual
# nervous system state persists via neural_nervous_system/*.json pickle files.
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
        # Reward-modulated residual targets (2026-04-19 autonomous-collapse
        # fix). Replaces discrete-case classifier that reinforced "output 0"
        # when 98%+ of transitions had reward=0. See _compute_targets for
        # formula. Fire/no-fire gains control per-step target shift magnitude.
        self._nn_target_fire_gain = float(
            config.get("nn_target_fire_gain", 0.3))
        self._nn_target_nofire_gain = float(
            config.get("nn_target_nofire_gain", 0.1))

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

        # rFP β Stage 2: per-program reward EMA + audit log + eligibility params
        self._reward_ema_mean: dict[str, float] = {n: 0.0 for n in self.programs}
        self._reward_ema_var: dict[str, float] = {n: 1.0 for n in self.programs}
        self._reward_ema_alpha: float = 0.01  # ~1000-event horizon
        self._reward_log_path = os.path.join(self.data_dir, "reward_log.jsonl")
        self._reward_log_lines = 0
        self._reward_log_max_lines = 50000
        self._reward_log_enabled = True
        self._stratified_sampling_enabled: bool = True
        self._soft_fire_enabled: bool = True
        # Eligibility params from Stage 0.5 empirical analysis (or defaults)
        self._eligibility_params = self._load_eligibility_params()

        # Load persisted state
        self._load_all()

        # 2026-04-19: Rescue programs trapped at sigmoid(-10) saturation.
        # Sigmoid's vanishing gradient makes escape via gradient descent
        # astronomically slow once pre-sigmoid bias converges strongly
        # negative. This one-time load-time action resets the output layer
        # (preserving learned hidden features) on programs showing saturation.
        self._liberate_saturated_output_layers()

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
            except Exception as e:
                # Rate-limited WARNING — VM supervision breaking is important
                # but we don't want 10Hz log spam if it persists.
                if self._total_transitions % 100 == 0:
                    logger.warning("[NeuralNS] VM baseline eval failed: %s", e)

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

        # ── Hormone snapshot (rate-limited) ──
        # INNER-MEMORY-API-ORPHANS fix (2026-04-19): before today,
        # inner_memory.record_hormone_snapshot had 0 callers in the
        # codebase → hormone_snapshots table had 0 rows over 1+ month
        # of runtime. EMOT-CGN v2 rFP (locked 2026-04-19) requires this
        # history as its HORMONE_FIRE producer. Snapshot every 100
        # epoch-ticks (rate-limited to avoid SQLite write pressure —
        # per-fire records are already captured by record_program_fire).
        if self._hormonal_enabled and self._inner_memory is not None:
            try:
                self._hormone_snapshot_tick = getattr(
                    self, "_hormone_snapshot_tick", 0) + 1
                if self._hormone_snapshot_tick % 100 == 0:
                    _hs_levels = {
                        n: round(float(h.level), 4)
                        for n, h in self._hormonal._hormones.items()
                    }
                    _hs_thresh = {
                        n: round(float(h.threshold), 4)
                        for n, h in self._hormonal._hormones.items()
                    }
                    _hs_refr = {
                        n: round(float(getattr(h, "refractory_until", 0.0)), 4)
                        for n, h in self._hormonal._hormones.items()
                    }
                    _hs_fired = [
                        s.get("system") for s in signals
                        if s.get("system") and s.get("learned")
                    ]
                    _hs_epoch = int(
                        self._maturity_signals.get("consciousness_epochs", 0)
                        or self._total_transitions)
                    self._inner_memory.record_hormone_snapshot(
                        epoch_id=_hs_epoch,
                        levels=_hs_levels,
                        thresholds=_hs_thresh,
                        refractory=_hs_refr,
                        fired=_hs_fired,
                        stimuli=env_stimuli,
                    )
            except Exception as _hs_err:
                # Debug level — snapshot failures are non-critical; the
                # per-fire record_program_fire path already captures
                # the main signal for EMOT-CGN.
                logger.debug(
                    "[NeuralNS] Hormone snapshot failed: %s", _hs_err)

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
            except Exception as e:
                logger.warning("[NeuralNS] hormonal_state.json save failed: %s", e)

        # rFP β Stage 0: periodic health summary — every 1000 transitions
        # Surfaces "which programs are alive" in the brain log so operators
        # can catch regression without running arch_map. Rate-limited to once
        # per 1000 transitions (~ every 2-3 min at typical tick rates).
        if self._total_transitions > 0 and self._total_transitions % 1000 == 0:
            try:
                self._log_health_summary()
            except Exception as e:
                logger.warning("[NeuralNS] health summary log failed: %s", e)

    def get_health_snapshot(self) -> dict:
        """rFP β Stage 0: canonical health snapshot for /v4/ns-health endpoint,
        periodic log, and arch_map ns-health. Computes buffer-derived stats
        (urgency distribution, fire rate, %nonzero) that aren't in get_stats().

        Returns dict with:
          - training: {phase, supervision_weight, total_transitions, total_train_steps}
          - programs: per-program stats with urgency+vm_baseline+reward distributions
          - hormonal: per-hormone level/threshold/fire_count
          - verdicts: {DEAD: [...], LOW: [...], OK: [...]}
        """
        import numpy as _np  # keep local — heavy import
        snap = {
            "training": {
                "phase": self.training_phase,
                "supervision_weight": round(self._get_supervision_weight(), 4),
                "total_transitions": self._total_transitions,
                "total_train_steps": self._total_train_steps,
                "last_train_ts": self._last_train_ts,
            },
            "programs": {},
            "verdicts": {"DEAD": [], "LOW": [], "OK": []},
        }
        urgency_eps = 0.01  # match arch_map threshold (noise floor filter)
        for name, net in self.programs.items():
            buf = self.buffers.get(name)
            if buf is None or len(buf) == 0:
                snap["programs"][name] = {"status": "empty_buffer"}
                continue
            # Use internal buffer arrays (same as _sqlite_backup_save pattern)
            urgencies = list(buf._urgencies[-buf.max_size:]) if hasattr(buf, '_urgencies') else []
            vm_baselines = list(buf._vm_baselines[-buf.max_size:]) if hasattr(buf, '_vm_baselines') else []
            fired = list(buf._fired[-buf.max_size:]) if hasattr(buf, '_fired') else []
            rewards = list(buf._rewards[-buf.max_size:]) if hasattr(buf, '_rewards') else []
            n = len(urgencies)
            if n == 0:
                snap["programs"][name] = {"status": "empty_buffer"}
                continue
            avg_u = float(sum(urgencies) / n)
            max_u = float(max(urgencies))
            pct_nz_u = 100.0 * sum(1 for v in urgencies if abs(v) > urgency_eps) / n
            pct_nz_vm = 100.0 * sum(1 for v in vm_baselines if abs(v) > urgency_eps) / n if vm_baselines else 0.0
            fire_pct = 100.0 * sum(1 for f in fired if f) / n
            pct_nz_r = 100.0 * sum(1 for v in rewards if abs(v) > urgency_eps) / n if rewards else 0.0

            # Verdict — same logic as arch_map ns-signals
            is_dead = n > 500 and (pct_nz_u < 5 or (avg_u < 0.01 and max_u < 0.05))
            if is_dead:
                verdict = "DEAD"
            elif pct_nz_u < 30:
                verdict = "LOW"
            else:
                verdict = "OK"
            snap["verdicts"][verdict].append(name)

            snap["programs"][name] = {
                "status": "ok",
                "n": n,
                "avg_urgency": round(avg_u, 6),
                "max_urgency": round(max_u, 6),
                "pct_nonzero_urgency": round(pct_nz_u, 1),
                "pct_nonzero_vm_baseline": round(pct_nz_vm, 1),
                "fire_pct": round(fire_pct, 2),
                "pct_nonzero_reward": round(pct_nz_r, 1),
                "last_loss": round(float(getattr(net, 'last_loss', 0.0)), 8),
                "total_updates": int(getattr(net, 'total_updates', 0)),
                "fire_count": int(getattr(net, 'fire_count', 0)),
                "layer": getattr(net, '_layer', 'inner'),
                "feature_set": getattr(net, '_feature_set', '?'),
                "verdict": verdict,
            }

        # Hormonal snapshot
        if self._hormonal_enabled and self._hormonal:
            snap["hormonal"] = {
                "maturity": round(self._hormonal.maturity, 4),
                "hormones": {
                    name: {
                        "level": round(h.level, 4),
                        "threshold": round(h.threshold, 4),
                        "refractory": round(h.refractory, 4),
                        "fire_count": h.fire_count,
                    }
                    for name, h in self._hormonal._hormones.items()
                },
            }
        # NeuromodRewardObserver stats (set by spirit_worker on init)
        obs = getattr(self, "_neuromod_reward_observer", None)
        if obs is not None:
            try:
                snap["neuromod_reward_observer"] = obs.get_stats()
            except Exception:
                snap["neuromod_reward_observer"] = {"error": "stats_unavailable"}

        # Overall state for session-startup / CI gates
        n_dead = len(snap["verdicts"]["DEAD"])
        n_prog = sum(1 for p in snap["programs"].values() if p.get("status") == "ok")
        if n_dead == 0:
            snap["overall"] = "healthy"
        elif n_dead < n_prog / 2:
            snap["overall"] = "warning"
        else:
            snap["overall"] = "critical"
        snap["overall_counts"] = {
            "dead": n_dead, "low": len(snap["verdicts"]["LOW"]),
            "ok": len(snap["verdicts"]["OK"]), "total_programs": n_prog,
        }
        return snap

    def _log_health_summary(self) -> None:
        """Periodic one-line health summary to the brain log. Called every
        1000 transitions from _record_transition. rFP β § 4c observability.

        Phase 3 addition: also persists snapshot to
        data/neural_nervous_system/health_snapshot.json so the /v4/ns-health
        dashboard endpoint can read NS state from across the spirit_worker
        subprocess boundary. Per-titan unique by repo path (no collision
        between T1 ~/projects/titan/ and T3 ~/projects/titan3/).
        """
        snap = self.get_health_snapshot()
        counts = snap.get("overall_counts", {})
        overall = snap.get("overall", "unknown")
        # One-line summary
        dead = snap["verdicts"]["DEAD"]
        dead_str = ",".join(dead[:3]) + ("..." if len(dead) > 3 else "")
        logger.info(
            "[NeuralNS] health: %s  |  dead=%d/%d%s  |  low=%d  ok=%d  |  "
            "phase=%s sup_w=%.3f  transitions=%d train_steps=%d",
            overall, counts.get("dead", 0), counts.get("total_programs", 0),
            f" ({dead_str})" if dead else "",
            counts.get("low", 0), counts.get("ok", 0),
            snap["training"]["phase"], snap["training"]["supervision_weight"],
            snap["training"]["total_transitions"],
            snap["training"]["total_train_steps"],
        )
        # Persist snapshot for cross-process /v4/ns-health endpoint access.
        # Atomic write (tmp→rename) so dashboard readers never see partial JSON.
        try:
            snap["snapshot_ts"] = time.time()
            snap_path = os.path.join(self.data_dir, "health_snapshot.json")
            tmp = snap_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(snap, f, default=str)
            os.replace(tmp, snap_path)
        except Exception as e:
            logger.debug("[NeuralNS] health snapshot persist failed: %s", e)

    def _load_eligibility_params(self) -> dict:
        """rFP β Stage 2: load per-program eligibility params from Stage 0.5 output.

        Returns dict {program: {K, decay, freshness_window_s}} or empty dict
        if params file missing (will use built-in defaults).
        """
        params_path = os.path.join(self.data_dir, "eligibility_params.json")
        try:
            with open(params_path) as f:
                data = json.load(f)
            params = data.get("params", {})
            if params:
                logger.info(
                    "[NeuralNS] Loaded eligibility params for %d programs from %s "
                    "(generated %s)",
                    len(params), params_path,
                    data.get("generated_at", "unknown"))
            return params
        except FileNotFoundError:
            logger.info("[NeuralNS] No eligibility_params.json — using defaults "
                        "(K=1, decay=0.5)")
            return {}
        except Exception as e:
            logger.warning("[NeuralNS] Failed to load eligibility params: %s", e)
            return {}

    def _z_normalize_reward(self, program: str, reward: float) -> float:
        """rFP β Stage 2 § 4a-quat: per-program z-score normalization.

        Updates rolling EMA mean+var for the program, returns clipped z-score.
        Programs that fire at very different rates (METABOLISM rare vs
        REFLEX often) end up with comparable training-signal magnitudes.

        URGENCY OUTPUT stays raw [0, 1] — only the training reward is
        normalized. Downstream consumers (gut_signals, meta-reasoning,
        composites) keep semantic urgency meaning.
        """
        a = self._reward_ema_alpha
        prev_mean = self._reward_ema_mean.get(program, 0.0)
        prev_var = self._reward_ema_var.get(program, 1.0)
        new_mean = prev_mean + a * (reward - prev_mean)
        # Naive Welford-ish online variance update
        new_var = prev_var + a * ((reward - prev_mean) ** 2 - prev_var)
        self._reward_ema_mean[program] = new_mean
        self._reward_ema_var[program] = max(new_var, 1e-6)
        std = max(math.sqrt(self._reward_ema_var[program]), 0.01)
        z = (reward - new_mean) / std
        return max(-3.0, min(3.0, z))

    def _log_reward_event(self, program: str, reward_raw: float, reward_z: float,
                          source: str, k_applied: int, fired: bool) -> None:
        """rFP β Stage 2 § 4d: append to reward audit log.

        Rolling truncation when line count exceeds max (defaults to 50k).
        Daily archives + 30-day retention deferred to follow-up commit.
        """
        if not self._reward_log_enabled:
            return
        try:
            entry = {
                "ts": time.time(),
                "program": program,
                "reward_raw": round(float(reward_raw), 6),
                "reward_z": round(float(reward_z), 4),
                "source": source,
                "k": k_applied,
                "fired": bool(fired),
                "ema_mean": round(self._reward_ema_mean.get(program, 0.0), 4),
            }
            with open(self._reward_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self._reward_log_lines += 1
            # Rolling truncation every 1000 new events
            if self._reward_log_lines % 1000 == 0:
                self._truncate_reward_log()
        except Exception as e:
            logger.debug("[NeuralNS] reward log append failed: %s", e)

    def _truncate_reward_log(self) -> None:
        """Keep only the last `_reward_log_max_lines` lines of the audit log."""
        try:
            if not os.path.exists(self._reward_log_path):
                return
            with open(self._reward_log_path) as f:
                lines = f.readlines()
            if len(lines) > self._reward_log_max_lines:
                kept = lines[-self._reward_log_max_lines:]
                tmp = self._reward_log_path + ".tmp"
                with open(tmp, "w") as f:
                    f.writelines(kept)
                os.replace(tmp, self._reward_log_path)
                self._reward_log_lines = len(kept)
        except Exception as e:
            logger.debug("[NeuralNS] reward log truncate failed: %s", e)

    def record_outcome(self, reward: float, program: str | None = None,
                       k: int | None = None, decay: float | None = None,
                       source: str = "firehose") -> None:
        """rFP β Stage 2 — per-program reward routing with eligibility traces,
        z-score normalization, soft-fire propagation, and audit log.

        Args:
            reward: Raw reward value (any scale — z-normalized internally).
            program: If specified, only update that program's buffer.
                     If None, FIREHOSE mode — apply to all programs whose
                     last transition fired (preserves the old REFLEX_REWARD
                     behavior for backward compat).
            k: Override the eligibility K (number of recent fires to credit).
               If None, uses Stage-0.5 empirical params or default 1.
            decay: Override the decay factor (weight on each older fire).
                   If None, uses params or default 0.5.
            source: Free-text label for the audit log (e.g. "reasoning.commit",
                    "neuromod.NE_spike", "drain_delta", "firehose").

        Behavior:
            For each target program:
              1. Z-normalize reward against per-program rolling EMA
              2. If last transition FIRED → apply z-reward to last K fires
                 with eligibility decay (rFP β § 4a Option 1+ traces)
              3. If last transition NOT-FIRED but urgency was close to
                 threshold → apply soft-fire scaled reward (rFP β § 4a
                 Option 1 propagation, breaks class imbalance)
              4. Adapt hormone threshold from outcome
              5. Append audit log entry

        Backward compat: `record_outcome(0.5)` — single positional arg —
        retains the old firehose behavior. New code should use
        `record_outcome(reward=X, program=Y)` for per-program routing.
        """
        target_names = [program] if program else list(self.buffers.keys())
        for name in target_names:
            buf = self.buffers.get(name)
            if buf is None:
                if program:  # explicit program asked but missing — log
                    logger.warning(
                        "[NeuralNS] record_outcome: unknown program '%s' "
                        "(known: %s)", name, list(self.buffers.keys()))
                continue
            net = self.programs.get(name)
            if net is None:
                continue

            # 1. Z-normalize per-program (training signal equalization)
            z = self._z_normalize_reward(name, reward)

            # 2/3. Apply with eligibility traces or soft-fire
            eparams = self._eligibility_params.get(name, {})
            eK = k if k is not None else int(eparams.get("K", 1))
            eDecay = decay if decay is not None else float(eparams.get("decay", 0.5))

            applied_fired = False
            if buf.last_fired:
                # Eligibility traces: credit last K fires
                buf.update_recent_rewards(z, k=eK, decay=eDecay)
                applied_fired = True
            elif self._soft_fire_enabled:
                # Soft-fire propagation: scaled reward proportional to urgency
                threshold = getattr(net, 'fire_threshold', 0.3)
                buf.update_soft_reward(z, fire_threshold=threshold,
                                       soft_factor=0.5)

            # 4. Adapt hormone threshold from raw reward (preserves prior dynamics)
            if self._hormonal_enabled:
                self._hormonal.adapt(name, reward - 0.5)

            # 5. Audit log
            self._log_reward_event(name, reward, z, source,
                                   k_applied=eK, fired=applied_fired)

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

            # rFP β Stage 2 § 4a Option 3: stratified sampling combats
            # class imbalance (97-99% not-fired). Falls back to uniform
            # if one class is empty (early training).
            if self._stratified_sampling_enabled:
                obs, urgencies, vm_baselines, rewards, fired = buf.sample_stratified(self._batch_size)
            else:
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
        """Compute training targets as blend of supervision + outcome.

        Supervision target: VM baseline urgency (imitate TitanVM during
        warmup phase, supervision_weight 1.0 → 0.0 linear decay).

        Outcome target (2026-04-19 refactor — reward-modulated residual
        learning): Default (r ≈ 0) keeps target = current urgency so no
        collapse pressure builds from the 98%+ reward-zero transitions.
        Positive rewards bump target up (encourage); negative rewards bump
        target down (discourage). Fired transitions carry stronger gain
        because the NN's decision caused the outcome; not-fired transitions
        are passive observations with lower gain. tanh bounds per-step shift
        so extreme rewards can't spike targets beyond ±gain.

        Fixes three bugs in the prior discrete-case formula:
          (1) negative rewards silently mapped to target=0, no discouragement
          (2) r ∈ (0.3, 0.5] fell through to target=0 regardless of fire
          (3) r=0 + not_fired (98% of transitions) always target=0, which
              drove the NN to sigmoid(-10) = 4.5e-05 saturation.
        """
        # Supervision component (VM imitation during warmup)
        sup_targets = vm_baselines

        # Outcome component — reward-modulated residual from current urgency
        gain = np.where(
            fired, self._nn_target_fire_gain, self._nn_target_nofire_gain)
        delta = np.tanh(rewards) * gain
        outcome_targets = np.clip(urgencies + delta, 0.0, 1.0)

        # Extension #8 edge-case fix: dead-zone anti-fixed-point push.
        # When rewards ≈ 0 AND not fired (98%+ of transitions for rare-fire
        # programs like METABOLISM), residual delta = 0 → target = urgency.
        # Any urgency value is a fixed point, including near-zero sigmoid-
        # saturation. Observed 2026-04-19 evening on T2 METABOLISM after
        # deep liberation: fresh Xavier init landed in σ < 0.001 region,
        # NN learned to keep producing < 0.001, re-saturated within ~700
        # training steps. Push targets in dead zones (< 0.05 or > 0.95)
        # gently toward 0.5 only on passive transitions. Healthy programs
        # (target ∈ [0.05, 0.95]) are UNAFFECTED. Active transitions
        # (fired OR |reward| > 0) are UNAFFECTED. Pulls dead-zone programs
        # back into the trainable region within ~100-200 batches.
        in_dead_zone = (outcome_targets < 0.05) | (outcome_targets > 0.95)
        passive = (~fired) & (np.abs(rewards) < 1e-6)
        dormant_push = np.where(
            in_dead_zone & passive,
            0.01 * (0.5 - outcome_targets),
            0.0,
        )
        outcome_targets = np.clip(outcome_targets + dormant_push, 0.0, 1.0)

        # Blend supervision + outcome by phase weight
        targets = (supervision_weight * sup_targets +
                   (1.0 - supervision_weight) * outcome_targets)
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

    # ── Sigmoid-trap liberation (2026-04-19) ──────────────────────

    def _liberate_saturated_output_layers(self) -> None:
        """Rescue programs whose NN output is trapped at sigmoid saturation.

        Two tiers (both run per-program, deep takes precedence):

        SHALLOW (original, 2026-04-19): saved bias b3 < −5.0 → reinit w3+b3
            only. Preserves w1/b1/w2/b2. Handles output-bias saturation.

        DEEP / Extension #8 (2026-04-19 evening): buffer urgency trace
            shows pct_nonzero_urgency < 5% over ≥500 samples → the trap
            lives in hidden weights w2 producing z3 ≈ −10 regardless of
            b3. Reinit w2+b2+w3+b3 AND clear the replay buffer (stale
            urgency targets would re-saturate fresh NN via residual
            learning). Preserves w1+b1 so first-layer feature detectors
            survive. Runtime signal (pct_nz_u) is ground truth; saved b3
            is one proxy among many.

        Idempotent: healthy NNs (b3 near 0, pct_nz_u ≥ 5%) pass through
        unchanged. Works with _compute_targets residual learning.
        """
        import math
        SATURATION_B3_THRESHOLD = -5.0
        RUNTIME_DEAD_PCT_NZ_U = 5.0
        RUNTIME_DEAD_MIN_SAMPLES = 500
        URGENCY_EPS = 0.001

        shallow = []
        deep = []

        for name, net in self.programs.items():
            try:
                buf = self.buffers.get(name)
                pct_nz_u = None
                n_samples = 0
                if buf is not None and hasattr(buf, '_urgencies'):
                    urgencies = list(buf._urgencies[-buf.max_size:])
                    n_samples = len(urgencies)
                    if n_samples >= RUNTIME_DEAD_MIN_SAMPLES:
                        pct_nz_u = 100.0 * sum(
                            1 for v in urgencies if abs(v) > URGENCY_EPS
                        ) / n_samples

                b3_val = float(net.b3[0])
                runtime_dead = (
                    pct_nz_u is not None and pct_nz_u < RUNTIME_DEAD_PCT_NZ_U
                )
                b3_saturated = (b3_val < SATURATION_B3_THRESHOLD)

                if runtime_dead:
                    net.w2 = (np.random.randn(net.hidden_1, net.hidden_2).astype(np.float64)
                              * math.sqrt(2.0 / net.hidden_1))
                    net.b2 = np.zeros(net.hidden_2, dtype=np.float64)
                    net.w3 = (np.random.randn(net.hidden_2, 1).astype(np.float64)
                              * math.sqrt(2.0 / net.hidden_2))
                    net.b3 = np.zeros(1, dtype=np.float64)
                    if buf is not None:
                        buf._observations.clear()
                        buf._urgencies.clear()
                        buf._vm_baselines.clear()
                        buf._rewards.clear()
                        buf._fired.clear()
                        if hasattr(buf, '_last_fired_idx'):
                            buf._last_fired_idx = -1
                    deep.append(
                        f"{name}(pct_nz_u={pct_nz_u:.1f}%, n={n_samples}, "
                        f"b3={b3_val:.2f})"
                    )
                elif b3_saturated:
                    net.w3 = (np.random.randn(net.hidden_2, 1).astype(np.float64)
                              * math.sqrt(2.0 / net.hidden_2))
                    net.b3 = np.zeros(1, dtype=np.float64)
                    shallow.append(f"{name}(b3 was {b3_val:.2f})")
            except Exception as e:
                logger.debug(
                    "[NeuralNS] %s liberation check error: %s", name, e)

        if shallow:
            logger.warning(
                "[NeuralNS] SHALLOW liberation: %d programs — "
                "output layer w3+b3 reinit (b3 < %.1f). w1/b1/w2/b2 preserved: %s",
                len(shallow), SATURATION_B3_THRESHOLD, ", ".join(shallow))
        if deep:
            logger.warning(
                "[NeuralNS] DEEP liberation (Extension #8): %d programs — "
                "w2+b2+w3+b3 reinit + replay buffer cleared "
                "(runtime pct_nz_u < %.1f%% over ≥%d samples). w1/b1 preserved: %s",
                len(deep), RUNTIME_DEAD_PCT_NZ_U,
                RUNTIME_DEAD_MIN_SAMPLES, ", ".join(deep))

    # ── Outer Program Dispatch ─────────────────────────────────────

    def get_augmented_urgencies(self, hormone_blend: float = 0.3) -> dict:
        """rFP β Phase 3 — blend NN urgency with hormonal pressure for gut input.

        Pure NN urgency is one signal. Hormone level (which carries history
        from environmental + recent fires) is a complementary signal.
        Blending makes gut more robust when NN flickers or is collapsed.

        Args:
            hormone_blend: ratio of hormone pressure in the blend [0, 1].
                           Default 0.3 → 70% NN + 30% hormone-pressure-ratio.
                           Set to 0 to disable (pure NN urgencies returned).

        Returns:
            Dict of {program: blended_urgency} where blended_urgency is
              (1 - blend) × nn_urgency + blend × (hormone.level / threshold)
            Hormones above threshold contribute > 1.0 (clamped at 1.0).
        """
        nn_urgencies = dict(getattr(self, '_all_urgencies', {}))
        if hormone_blend <= 0 or not self._hormonal_enabled or not self._hormonal:
            return nn_urgencies

        out = {}
        for name, nn_urg in nn_urgencies.items():
            hormone = self._hormonal._hormones.get(name) if self._hormonal else None
            if hormone is None or hormone.threshold <= 0:
                out[name] = nn_urg
                continue
            hormone_ratio = min(1.0, hormone.level / max(hormone.threshold, 0.01))
            out[name] = (1.0 - hormone_blend) * nn_urg + hormone_blend * hormone_ratio
        return out

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
