"""
Spirit Module Worker — 45DT Vedantic consciousness with full Trinity perception.

Runs in isolation. Produces:
  - 5DT legacy Spirit tensor (WHO/WHY/WHAT + body/mind scalars)
  - 45DT Sat-Chit-Ananda Spirit tensor (DQ3)
  - 67D consciousness epochs (Body 5D + Mind 15D + Spirit 45D + meta 2D)

ConsciousnessLoop tracks drift, trajectory, curvature, and density across
the FULL 67-dimensional state space — Titan perceives through his entire
Trinity, not just 9 proxy dimensions.

Extended 67D Consciousness State:
  [0:5]   Body 5D    — physical/digital topology senses
  [5:20]  Mind 15D   — Thinking(5) + Feeling(5) + Willing(5)
  [20:65] Spirit 45D — SAT(15) + CHIT(15) + ANANDA(15)
  [65]    curvature  — self-referential (previous epoch)
  [66]    density    — self-referential (previous epoch)

Step 4 Counterparts (all run in Spirit — the observer layer):
  - FILTER_DOWN: Value network learns V(state), gradients → severity multipliers
  - FOCUS: PID controllers nudge Body/Mind toward Middle Path
  - INTUITION: Suggests behavioral postures based on deficits
  - Middle Path loss: Combined equilibrium metric for the Trinity
  - On-chain anchoring: Solana memo with Trinity state hash

Consciousness data comes from Body/Mind bus broadcasts — no direct subsystem access needed.

Entry point: spirit_worker_main(recv_queue, send_queue, name, config)
"""
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Tier 3 Consciousness Epoch — Resonance-Gated Adaptive Timing ──────
# No fixed interval. Timing emerges from Titan's own state:
#   MIN = SCHUMANN_BODY × 9 × GABA_level   (how fast Titan CAN think)
#   MAX = SCHUMANN_BODY × 27 × (0.5 + chi_circulation)  (how long without reflection)
#   FLOOR = SCHUMANN_BODY × 3              (infrastructure safety only)
# Triggers: resonance transition > hormonal urgency > max interval reached
# These are set at module level for epoch bound calculations.
# Actual values loaded from [schumann] config at boot (see line ~407).
# Defaults here for safety if config missing.
SCHUMANN_BODY_CONST = 1.15     # Overridden at boot from [schumann] config (computation gate)
EPOCH_FLOOR = 1.15             # Overridden at boot from [schumann] config
EPOCH_URGENCY_THRESHOLD = 5    # hormonal fires since last epoch to trigger urgency

# ── META-CGN EdgeDetector state persistence (2026-04-15) ────────────
# Preserves "once per lifetime" / "once per threshold crossing" semantics
# across spirit restarts. Without this, Producer #1 (sphere_clock) was
# observed re-emitting milestones after each hot-restart (T1 observed
# 26 sphere emissions vs 16-per-lifetime budget across 4 restarts).
# Single canonical JSON keyed by detector name. Saves on 5-min checkpoint
# alongside NS SQLite backup. Loaded on detector init.
_EDGE_DETECTOR_STATE_PATH = "./data/edge_detector_state.json"


def _load_edge_detector_state() -> dict:
    """Read the persisted EdgeDetector state file. Returns {} on any error
    (fresh state) — fail-open is safe because missing state just means the
    producer will re-emit on first observation post-restart."""
    try:
        import json
        with open(_EDGE_DETECTOR_STATE_PATH) as f:
            data = json.load(f)
        if data.get("schema_version") != 1:
            logger.warning(
                "[SpiritWorker] Unknown EdgeDetector schema version %s; ignoring",
                data.get("schema_version"))
            return {}
        return data.get("detectors", {}) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("[SpiritWorker] EdgeDetector state load failed: %s", e)
        return {}


def _save_edge_detector_state(detectors: dict) -> None:
    """Atomically write EdgeDetector state (tmpfile + os.replace). Best-effort:
    WARN on failure because silent failure would hide a persistence gap.
    `detectors` is {name: EdgeDetector-instance}."""
    import json, os, tempfile
    payload = {
        "schema_version": 1,
        "saved_at": time.time(),
        "detectors": {name: det.to_dict() for name, det in detectors.items()
                      if det is not None and hasattr(det, "to_dict")},
    }
    try:
        _dir = os.path.dirname(_EDGE_DETECTOR_STATE_PATH) or "."
        if not os.path.isdir(_dir):
            os.makedirs(_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_dir, prefix="edge_detector_state.", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, _EDGE_DETECTOR_STATE_PATH)
    except Exception as e:
        logger.warning("[SpiritWorker] EdgeDetector state save failed: %s", e)

# FILTER_DOWN publish interval (after each consciousness epoch that triggers training)
# FOCUS PID runs every spirit publish cycle (60s)
# INTUITION runs every consciousness epoch

# ── Word Type Classification — imported from language_pipeline (Phase 0c extraction) ──
from titan_plugin.logic.language_pipeline import (
    KNOWN_ADJECTIVES as _KNOWN_ADJECTIVES,
    KNOWN_INTERJECTIONS as _KNOWN_INTERJECTIONS,
    KNOWN_ADVERBS as _KNOWN_ADVERBS,
    KNOWN_PRONOUNS as _KNOWN_PRONOUNS,
    classify_word_type as _classify_word_type,
    load_vocabulary as _lp_load_vocabulary,
    compose_sentence as _lp_compose_sentence,
    compute_perturbation_deltas as _lp_compute_perturbation_deltas,
    apply_perturbation_deltas as _lp_apply_perturbation_deltas,
    update_vocabulary_after_speak as _lp_update_vocabulary_after_speak,
    update_language_stats as _lp_update_language_stats,
    should_bootstrap as _lp_should_bootstrap,
    build_teacher_request as _lp_build_teacher_request,
    persist_composition as _lp_persist_composition,
    persist_teacher_session as _lp_persist_teacher_session,
)


def _query_handler_thread(query_queue, handle_fn, state_refs, send_queue, name, config):
    """Dedicated thread for QUERY responses — never blocked by computation.

    Reads QUERY messages from query_queue and responds using shared state.
    State dicts are read-only from this thread (CPython dict ops are GIL-atomic).
    """
    from queue import Empty as _QEmpty
    while True:
        try:
            msg = query_queue.get(timeout=0.5)
            if msg is None:
                break  # Shutdown signal
            _qt0 = time.time()
            _qt_action = msg.get("payload", {}).get("action", "?")
            _qt_age = _qt0 - msg.get("ts", _qt0)
            try:
                handle_fn(msg, config, state_refs["body_state"], state_refs["mind_state"],
                          state_refs["consciousness"], state_refs.get("filter_down"),
                          state_refs.get("intuition"), state_refs.get("impulse_engine"),
                          state_refs.get("sphere_clock"), state_refs.get("resonance"),
                          state_refs.get("unified_spirit"), send_queue, name,
                          inner_state=state_refs.get("inner_state"),
                          spirit_state=state_refs.get("spirit_state"),
                          coordinator=state_refs.get("coordinator"),
                          neural_nervous_system=state_refs.get("neural_nervous_system"),
                          pi_monitor=state_refs.get("pi_monitor"),
                          e_mem=state_refs.get("e_mem"),
                          prediction_engine=state_refs.get("prediction_engine"),
                          ex_mem=state_refs.get("ex_mem"),
                          episodic_mem=state_refs.get("episodic_mem"),
                          working_mem=state_refs.get("working_mem"),
                          inner_lower_topo=state_refs.get("inner_lower_topo"),
                          outer_lower_topo=state_refs.get("outer_lower_topo"),
                          ground_up_enricher=state_refs.get("ground_up_enricher"),
                          neuromodulator_system=state_refs.get("neuromodulator_system"),
                          expression_manager=state_refs.get("expression_manager"),
                          life_force_engine=state_refs.get("life_force_engine"),
                          outer_interface=state_refs.get("outer_interface"),
                          phase_tracker=state_refs.get("phase_tracker"),
                          meditation_tracker=state_refs.get("meditation_tracker"),
                          reasoning_engine=state_refs.get("reasoning_engine"),
                          msl=state_refs.get("msl"),
                          social_pressure_meter=state_refs.get("social_pressure_meter"),
                          language_stats=state_refs.get("language_stats"),
                          self_reasoning=state_refs.get("self_reasoning"),
                          coding_explorer=state_refs.get("coding_explorer"),
                          filter_down_v4=state_refs.get("filter_down_v4"),
                          filter_down_v5=state_refs.get("filter_down_v5"),
                          med_watchdog=state_refs.get("med_watchdog"))
                logger.info("[QueryThread] %s handled in %.0fms (queued %.0fms)",
                            _qt_action, (time.time() - _qt0) * 1000, _qt_age * 1000)
            except Exception as e:
                logger.warning("[QueryThread] Error handling %s: %s", _qt_action, e)
        except _QEmpty:
            continue
        except Exception:
            break


def spirit_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Spirit module process."""
    from queue import Empty

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ── Import reloadable helper functions from spirit_loop ──────────────
    # Phase 1 of spirit_loop extraction (2026-03-23).
    # Imported inside function scope so module-level init helpers are already defined.
    # Hot-reload updates these via re-import in the RELOAD handler.
    from titan_plugin.modules.spirit_loop import (
        _post_epoch_learning, _run_focus, _compute_spirit_reflex_intuition,
        _run_impulse, _post_epoch_v4_filter_down, _tick_clock_pair,
        _maybe_anchor_trinity, _run_consciousness_epoch, _compute_trajectory,
        _collect_spirit_tensor, _handle_query, _publish_spirit_state,
        _send_msg, _send_response, _send_heartbeat,
        _init_consciousness, _check_resonance,
    )

    logger.info("[SpiritWorker] Initializing 3DT+2 consciousness sensors...")

    # Latest Body and Mind state (received via bus broadcasts)
    body_state = {"values": [0.5] * 5, "center_dist": 0.0}
    mind_state = {"values": [0.5] * 5, "center_dist": 0.0}

    # Latest Outer Trinity state (received via OUTER_TRINITY_STATE)
    outer_state = {
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
        "outer_mind_15d": None,
        "outer_spirit_45d": None,
    }

    # Kin Discovery — track resonance state for neuromod input + GREAT_KIN_PULSE
    _kin_state = {"last_resonance": 0.0, "last_exchange_ts": 0.0, "exchanges_count": 0}

    # Multi-Titan identity — loaded from data/titan_identity.json (excluded from deploy sync)
    _titan_identity = {"titan_id": "T1", "delegate_mode": "gateway"}
    try:
        with open("./data/titan_identity.json") as _tid_f:
            _titan_identity = json.load(_tid_f)
        logger.info("[Identity] Titan %s, mode=%s",
                    _titan_identity.get("titan_id"), _titan_identity.get("delegate_mode"))
    except Exception:
        logger.info("[Identity] No titan_identity.json — defaulting to T1 gateway mode")

    # Shared dreaming flag — used by experience_orchestrator record calls
    _shared_is_dreaming = False

    # SensoryHub — lazy init on first SENSE message
    _sensory_hub = None
    _sensory_hub_retry_ts = 0.0

    # Previous Body/Mind for transition recording
    prev_body_values = [0.5] * 5
    prev_mind_values = [0.5] * 5
    prev_spirit_tensor = [0.5] * 5
    prev_middle_path_loss = 0.0
    _last_pi_observed_epoch = -1  # Track which epoch pi_monitor last observed

    # ── Curvature variance tracker (for emergent fatigue) ────────
    class _SWLocal:
        """Spirit worker local state namespace."""
        _curvature_history: list = []
        _curvature_variance: float = 0.5
    _sw_local = _SWLocal()

    # ── Phase Event Log (for telemetry) ──────────────────────────
    # Tracks transitions: self-exploration, dreaming, learning, chat
    # Uses a mutable dict so it's accessible from _handle_query (standalone function)
    _phase_tracker = {
        "current_phase": "idle",
        "events": [],
        "max_events": 200,
    }

    def _log_phase_event(event_type: str, detail: dict = None):
        _phase_tracker["current_phase"] = event_type
        entry = {"ts": time.time(), "phase": event_type}
        if detail:
            entry.update(detail)
        _phase_tracker["events"].append(entry)
        if len(_phase_tracker["events"]) > _phase_tracker["max_events"]:
            _phase_tracker["events"].pop(0)

    # ── Boot ConsciousnessLoop ────────────────────────────────────
    consciousness = _init_consciousness(config)

    # ── Boot Step 4 Counterparts ──────────────────────────────────
    filter_down = _init_filter_down(config)
    filter_down_v4 = _init_filter_down_v4(config)
    filter_down_v5 = _init_filter_down_v5(config)  # rFP #2
    # rFP #2 Phase B.5b: latest V5 multipliers cached for application at
    # unified_spirit update call sites. Populated from FILTER_DOWN_V5 bus
    # messages (own loop's publish); empty dict = no modulation (coexistence-safe).
    _v5_mults_cache: dict = {}
    focus_body, focus_mind = _init_focus()
    intuition = _init_intuition()

    # ── Boot V4 Sphere Clock Engine ─────────────────────────────────
    sphere_clock = _init_sphere_clock(config)

    # ── Boot V4 Resonance Detector ────────────────────────────────
    resonance = _init_resonance(config)

    # ── Boot V4 Unified SPIRIT ────────────────────────────────────
    unified_spirit = _init_unified_spirit(config)

    # ── Boot Step 7 Impulse Engine ─────────────────────────────────
    impulse_engine = _init_impulse_engine()

    # ── Boot T1 Observable Engine ─────────────────────────────────
    observable_engine = _init_observable_engine()

    # ── Boot T2 State Registries ──────────────────────────────────
    inner_state, spirit_state = _init_t2_state_registries()

    # ── Boot V5 Neural Nervous System ──────────────────────────────
    neural_nervous_system = _init_neural_nervous_system(config)

    # ── Boot T3 Inner Trinity Coordinator ─────────────────────────
    coordinator = _init_coordinator(inner_state, spirit_state, observable_engine,
                                    neural_nervous_system=neural_nervous_system)

    # ── Boot π-Heartbeat Monitor ─────────────────────────────────
    from titan_plugin.logic.pi_heartbeat import PiHeartbeatMonitor
    pi_monitor = PiHeartbeatMonitor(
        min_cluster_size=3, min_gap_size=2,
        state_path="./data/pi_heartbeat_state.json")

    # ── Boot Experiential Memory (e_mem) — dream insight store ──
    from titan_plugin.logic.experiential_memory import ExperientialMemory
    e_mem = ExperientialMemory(
        db_path="./data/experiential_memory.db",
        developmental_age_fn=lambda: pi_monitor.developmental_age,
    )
    # rFP #3 Phase 4: plumb recall similarity floor from titan_params.toml
    # [dreaming].min_recall_similarity. Follows existing DNA-loading pattern.
    try:
        import tomllib as _tom_emem
        _emem_path = os.path.join(os.path.dirname(__file__), "..", "titan_params.toml")
        if os.path.exists(_emem_path):
            with open(_emem_path, "rb") as _emem_f:
                _emem_cfg = _tom_emem.load(_emem_f).get("dreaming", {})
            if "min_recall_similarity" in _emem_cfg:
                e_mem.set_min_recall_similarity(float(_emem_cfg["min_recall_similarity"]))
    except Exception as _emem_err:
        logger.warning("[SpiritWorker] e_mem min_recall_similarity load failed: %s", _emem_err)

    # ── Boot Brain P1+P2: Prediction, Experience, Episodic, Working Memory ──
    from titan_plugin.logic.prediction_engine import PredictionEngine
    from titan_plugin.logic.experience_memory import ExperienceMemory
    from titan_plugin.logic.episodic_memory import EpisodicMemory
    from titan_plugin.logic.working_memory import WorkingMemory

    prediction_engine = PredictionEngine(error_window=20)
    ex_mem = ExperienceMemory(db_path="./data/experience_memory.db")
    episodic_mem = EpisodicMemory(db_path="./data/episodic_memory.db")
    working_mem = WorkingMemory(capacity=7, decay_epochs=5)

    # ── Boot Grounding Space Topology (G1-G3) ──
    from titan_plugin.logic.lower_topology import LowerTopology
    from titan_plugin.logic.ground_up import GroundUpEnricher

    inner_lower_topo = LowerTopology(variant="inner", grounding_strength=0.1)
    outer_lower_topo = LowerTopology(variant="outer", grounding_strength=0.1)
    ground_up_enricher = GroundUpEnricher(strength=0.1, damping=0.95)

    # ── Boot Neuromodulator Meta-Layer ──
    from titan_plugin.logic.neuromodulator import (
        NeuromodulatorSystem, compute_emergent_inputs, apply_movement_excess_clearance,
    )
    neuromodulator_system = NeuromodulatorSystem(data_dir="./data/neuromodulator")

    # ── Boot NeuromodRewardObserver (rFP β Stage 2 Phase 2b) ──
    # Bridges biological-analog neuromod dynamics to NS program training.
    # Reads neuromod state every N ticks, emits per-program reward via
    # neural_nervous_system.record_outcome(reward, program=, source="neuromod.X").
    # Primary reward pathway for outer/personality programs + VIGILANCE NE-tracker.
    try:
        from titan_plugin.logic.neuromod_reward_observer import NeuromodRewardObserver
        neuromod_reward_observer = NeuromodRewardObserver(
            neural_nervous_system=neural_nervous_system,
            neuromodulator_system=neuromodulator_system,
            tick_interval=10,  # emit every 10 ticks (~1 sec at 10Hz tick rate)
            ema_alpha=0.05,
        )
        logger.info("[SpiritWorker] NeuromodRewardObserver online "
                    "(tick_interval=10, 11 programs covered)")
        # Expose observer on neural_nervous_system so get_health_snapshot
        # can include its stats in /v4/ns-health output (Phase 2b observability).
        if neural_nervous_system:
            neural_nervous_system._neuromod_reward_observer = neuromod_reward_observer
    except Exception as e:
        logger.warning("[SpiritWorker] NeuromodRewardObserver init failed: %s", e)
        neuromod_reward_observer = None

    # ── Boot EXPRESSION Composites (SPEAK, ART, MUSIC, SOCIAL) ──
    from titan_plugin.logic.expression_composites import (
        ExpressionManager, create_speak, create_art, create_music, create_social,
        create_kin_sense, create_longing,
    )
    # (Phase 4: CompositionEngine, WordSelector, GrammarPatternLibrary, LanguageTeacher
    #  removed — now owned by language_worker)
    from titan_plugin.logic.reasoning import ReasoningEngine

    expression_manager = ExpressionManager()
    expression_manager.register(create_speak())
    expression_manager.register(create_art())
    expression_manager.register(create_music())
    expression_manager.register(create_social())
    expression_manager.register(create_kin_sense())
    expression_manager.register(create_longing())
    logger.info("[SpiritWorker] EXPRESSION composites booted: %s",
                list(expression_manager.composites.keys()))

    # ── Boot Social X Gateway v3 (single class, single DB, single API point) ──
    _x_gateway = None
    _x_catalysts = []  # Catalyst accumulation list (consumed atomically by gateway.post())
    _social_pressure_meter = None  # Legacy reference — kept for on_social_relief (persona system)
    _x_gateway_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logic", "social_x_gateway.py")
    _x_gateway_mtime = 0  # Track file mtime for hot-reload
    _x_gateway_cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml")
    try:
        from titan_plugin.logic.social_x_gateway import (
            SocialXGateway, PostContext as _XPostContext,
            BaseContext as _XBaseContext, ReplyContext as _XReplyContext)
        _x_gateway = SocialXGateway(
            db_path="./data/social_x.db",
            config_path=_x_gateway_cfg_path,
            telemetry_path="./data/social_x_telemetry.jsonl",
        )
        _x_gateway_mtime = os.path.getmtime(_x_gateway_src) if os.path.exists(_x_gateway_src) else 0
        # Inject OutputVerifier for security gating of X posts/replies
        try:
            from titan_plugin.logic.output_verifier import OutputVerifier
            _titan_id = config.get("info_banner", {}).get("titan_id", "T1")
            _wallet_path = config.get("network", {}).get(
                "wallet_keypair_path", "data/titan_identity_keypair.json")
            _x_gateway.set_output_verifier(OutputVerifier(
                titan_id=_titan_id, data_dir="data/timechain",
                keypair_path=_wallet_path))
            logger.info("[SpiritWorker] OVG injected into SocialXGateway")
        except Exception as _ovg_err:
            logger.warning("[SpiritWorker] OVG injection failed: %s", _ovg_err)
        # Inject VerifiedContextBuilder for memory-enriched replies
        try:
            from titan_plugin.logic.verified_context_builder import VerifiedContextBuilder
            _data_dir = config.get("memory_and_storage", {}).get("data_dir", "./data")
            _known_users = []
            try:
                import sqlite3 as _sg_sql
                _sg_db = _sg_sql.connect(os.path.join(_data_dir, "social_graph.db"), timeout=5)
                _known_users = [r[0] for r in _sg_db.execute(
                    "SELECT user_id FROM users ORDER BY interaction_count DESC LIMIT 100"
                ).fetchall()]
                _sg_db.close()
            except Exception:
                pass
            _x_gateway.set_context_builder(VerifiedContextBuilder(
                data_dir=_data_dir, known_users=_known_users))
            logger.info("[SpiritWorker] VCB injected into SocialXGateway (known_users=%d)",
                        len(_known_users))
        except Exception as _vcb_err:
            logger.warning("[SpiritWorker] VCB injection failed: %s", _vcb_err)
        logger.info("[SpiritWorker] SocialXGateway v3 booted: db=data/social_x.db")
        # Keep legacy SocialPressureMeter ONLY for on_social_relief (persona system)
        try:
            from titan_plugin.logic.social_pressure import SocialPressureMeter
            _sp_cfg = config.get("social_presence", {})
            _social_pressure_meter = SocialPressureMeter(_sp_cfg)
        except Exception:
            pass  # on_social_relief will be unavailable — non-critical
    except Exception as _gw_err:
        logger.warning("[SpiritWorker] SocialXGateway boot failed: %s", _gw_err)

    # ── Boot Reasoning Engine (Mind's deliberate cognition) ──
    _reasoning_engine = None
    try:
        _reasoning_cfg = {}
        try:
            import tomllib as _rl_tl
            _rl_tp = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "titan_params.toml")
            if os.path.exists(_rl_tp):
                with open(_rl_tp, "rb") as _rl_f:
                    _rl_full = _rl_tl.load(_rl_f)
                    _reasoning_cfg = _rl_full.get("reasoning", {})
                    # rFP α (2026-04-16): merge [reasoning_rewards] section
                    # into engine config so publish_enabled, phase schedule,
                    # and Mech A/B knobs reach ReasoningEngine.__init__.
                    # Without this, engine falls back to defaults and phase
                    # gate always reports weights=(0,0).
                    if "reasoning_rewards" in _rl_full:
                        _reasoning_cfg = dict(_reasoning_cfg)
                        _reasoning_cfg["reasoning_rewards"] = dict(
                            _rl_full["reasoning_rewards"])
        except Exception:
            pass
        if _reasoning_cfg.get("enabled", True):
            _reasoning_engine = ReasoningEngine(config=_reasoning_cfg)
            # I-007 fix (2026-04-13): explicit log at REINIT so we can trace
            # any future "REASON chain counter reset" mystery (T1 went 4→0
            # on 2026-04-12 with no parent restart — likely Guardian sub-
            # process restart). With this WARNING-level log, future resets
            # leave a clear trail in brain.log searchable as "REASONING
            # ENGINE REINIT" — no more silent counter drops.
            logger.warning(
                "[SpiritWorker] REASONING ENGINE REINIT — fresh ReasoningEngine "
                "created (counters reset to 0). Process pid=%d, parent_pid=%d. "
                "If this fires unexpectedly, suspect Guardian module restart, "
                "OOM kill, or workers respawn. policy=%dD→h%d→h%d→%d, buffer=%d",
                os.getpid(), os.getppid() if hasattr(os, 'getppid') else -1,
                _reasoning_engine.policy_input_dim,
                _reasoning_engine.policy_h1,
                _reasoning_engine.policy_h2,
                8,  # NUM_ACTIONS
                _reasoning_engine.buffer.size())
    except Exception as _re_err:
        logger.warning("[SpiritWorker] Reasoning engine init failed: %s", _re_err)
        _reasoning_engine = None

    # Wire reasoning engine + mini-registry to coordinator for dream consolidation
    if _reasoning_engine and coordinator:
        coordinator._reasoning_engine = _reasoning_engine
    # (mini_registry wired to coordinator after init below)

    # ── Meta-Reasoning Foundation (M1-M3) ──
    chain_archive = None
    meta_wisdom = None
    meta_autoencoder = None
    try:
        from titan_plugin.logic.chain_archive import ChainArchive
        from titan_plugin.logic.meta_wisdom import MetaWisdomStore
        from titan_plugin.logic.meta_autoencoder import MetaAutoencoder
        chain_archive = ChainArchive()
        meta_wisdom = MetaWisdomStore()
        _ae_dir = _reasoning_engine.save_dir if _reasoning_engine else "./data/reasoning"
        meta_autoencoder = MetaAutoencoder(save_dir=_ae_dir)
        logger.info("[SpiritWorker] Meta-reasoning foundation: archive=OK, wisdom=OK, autoencoder=%s",
                    "trained" if meta_autoencoder.is_trained else "untrained")
        if coordinator:
            coordinator._chain_archive = chain_archive
            coordinator._meta_wisdom = meta_wisdom
            coordinator._meta_autoencoder = meta_autoencoder
    except Exception as _mrf_err:
        logger.warning("[SpiritWorker] Meta-reasoning foundation init: %s", _mrf_err)

    # ── Meta-Reasoning Engine (M4-M6) ──
    meta_engine = None
    try:
        from titan_plugin.logic.meta_reasoning import MetaReasoningEngine
        _meta_cfg = {}
        try:
            import tomllib as _meta_tl
            _meta_tp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "titan_params.toml")
            if os.path.exists(_meta_tp):
                with open(_meta_tp, "rb") as _meta_f:
                    _meta_full = _meta_tl.load(_meta_f)
                _meta_cfg = _meta_full.get("meta_reasoning", {})
                # ── TUNING-012 v2: Load [meta_reasoning_dna] with per-Titan overrides ──
                # Base DNA + per-Titan override (T1/T2/T3) merged into _meta_cfg["dna"]
                _dna_section = _meta_full.get("meta_reasoning_dna", {})
                _meta_dna = {
                    k: v for k, v in _dna_section.items()
                    if not isinstance(v, dict)
                }
                # Read titan_id from data/titan_identity.json — the canonical
                # per-Titan identity source. config.toml's [info_banner].titan_id
                # is unreliable: no Titan's config.toml actually sets it, so the
                # fallback to "T1" makes T2/T3 silently load T1's DNA.
                _titan_id_for_dna = "T1"
                try:
                    import json as _tid_json
                    _tid_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "data", "titan_identity.json",
                    )
                    if os.path.exists(_tid_path):
                        with open(_tid_path) as _tid_f:
                            _titan_id_for_dna = _tid_json.load(_tid_f).get("titan_id", "T1")
                except Exception:
                    pass
                _dna_override = _dna_section.get(_titan_id_for_dna, {})
                if isinstance(_dna_override, dict):
                    for _k, _v in _dna_override.items():
                        _meta_dna[_k] = _v
                _meta_cfg["dna"] = _meta_dna
                _meta_cfg["titan_id"] = _titan_id_for_dna
                logger.info(
                    "[SpiritWorker] Meta-reasoning DNA loaded: titan=%s, "
                    "%d base + %d override = %d total params (compound rewards %s)",
                    _titan_id_for_dna,
                    len(_meta_dna) - len(_dna_override),
                    len(_dna_override),
                    len(_meta_dna),
                    "ENABLED" if _meta_dna.get("inner_memory_signals_enabled", True)
                    or _meta_dna.get("timechain_signals_enabled", True)
                    or _meta_dna.get("contract_signals_enabled", True)
                    else "DISABLED",
                )

                # ── TUNING-012 v2 Sub-phase C: Load [cognitive_contracts_dna] ──
                # Same per-Titan override pattern as meta_reasoning_dna. Stored
                # on _meta_cfg["contracts_dna"] so MetaReasoningEngine can pass
                # it to apply_diversity_pressure() and the spirit_worker bus
                # handlers can read per-contract thresholds without re-parsing.
                _cc_section = _meta_full.get("cognitive_contracts_dna", {})
                _cc_dna = {
                    k: v for k, v in _cc_section.items()
                    if not isinstance(v, dict)
                }
                _cc_override = _cc_section.get(_titan_id_for_dna, {})
                if isinstance(_cc_override, dict):
                    for _k, _v in _cc_override.items():
                        _cc_dna[_k] = _v
                _meta_cfg["contracts_dna"] = _cc_dna
                logger.info(
                    "[SpiritWorker] Cognitive contracts DNA loaded: titan=%s, "
                    "%d base + %d override = %d total params (4 contracts: "
                    "strategy_evolution=%s eureka_detector=%s abstract_pattern=%s monoculture_detector=%s)",
                    _titan_id_for_dna,
                    len(_cc_dna) - len(_cc_override),
                    len(_cc_override),
                    len(_cc_dna),
                    "ON" if _cc_dna.get("strategy_evolution_enabled", True) else "OFF",
                    "ON" if _cc_dna.get("eureka_detector_enabled", True) else "OFF",
                    "ON" if _cc_dna.get("abstract_pattern_enabled", True) else "OFF",
                    "ON" if _cc_dna.get("monoculture_detector_enabled", True) else "OFF",
                )
        except Exception as _meta_dna_err:
            logger.warning("[SpiritWorker] Meta-reasoning DNA load: %s", _meta_dna_err)
        if _meta_cfg.get("enabled", True):
            meta_engine = MetaReasoningEngine(config=_meta_cfg,
                                              send_queue=send_queue)
            logger.info("[SpiritWorker] Meta-reasoning engine booted: chains=%d, steps=%d, wisdom=%d",
                        meta_engine._total_meta_chains, meta_engine._total_meta_steps,
                        meta_engine._total_wisdom_saved)
            if coordinator:
                coordinator._meta_engine = meta_engine
    except Exception as _me_err:
        logger.warning("[SpiritWorker] Meta-reasoning engine init: %s", _me_err)

    # ── Self-Reasoning Engine (INTROSPECT) ──
    _self_reasoning = None
    try:
        from titan_plugin.logic.self_reasoning import SelfReasoningEngine
        _sr_cfg = {}
        try:
            import tomllib as _sr_tl
            _sr_tp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "titan_params.toml")
            if os.path.exists(_sr_tp):
                with open(_sr_tp, "rb") as _sr_f:
                    _sr_cfg = _sr_tl.load(_sr_f).get("self_reasoning", {})
        except Exception:
            pass
        if _sr_cfg.get("enabled", True):
            _self_reasoning = SelfReasoningEngine(
                config=_sr_cfg, db_path="./data/inner_memory.db")
            if meta_engine:
                meta_engine.set_self_reasoning(_self_reasoning)
            logger.info("[SpiritWorker] Self-reasoning engine booted: "
                        "introspections=%d, predictions=%d",
                        _self_reasoning._total_introspections,
                        len(_self_reasoning._active_predictions))
    except Exception as _sr_err:
        logger.warning("[SpiritWorker] Self-reasoning engine init: %s", _sr_err)

    # ── Coding Explorer (Self-Directed Development) ──
    _coding_explorer = None
    try:
        from titan_plugin.logic.coding_explorer import CodingExplorer
        _ce_cfg = {}
        try:
            import tomllib as _ce_tl
            _ce_tp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "titan_params.toml")
            if os.path.exists(_ce_tp):
                with open(_ce_tp, "rb") as _ce_f:
                    _ce_cfg = _ce_tl.load(_ce_f).get("cgn", {}).get("coding", {})
        except Exception:
            pass
        if _ce_cfg.get("enabled", True):
            _coding_explorer = CodingExplorer(
                send_queue=send_queue, config=_ce_cfg, db_path="./data/inner_memory.db")
            logger.info("[SpiritWorker] Coding explorer booted: sandbox=%s",
                        _coding_explorer._sandbox.status())
    except Exception as _ce_err:
        logger.warning("[SpiritWorker] Coding explorer init: %s", _ce_err)

    # ── Boot Vertical Intuition Convergence (M11-M13) ──
    _intuition_convergence = None
    try:
        from titan_plugin.logic.intuition_convergence import IntuitionConvergenceDetector
        _ic_cfg = {}
        try:
            import tomllib as _ic_tl
            _ic_tp = "./titan_plugin/titan_params.toml"
            if os.path.exists(_ic_tp):
                with open(_ic_tp, "rb") as _ic_f:
                    _ic_cfg = _ic_tl.load(_ic_f).get("intuition_convergence", {})
        except Exception:
            pass
        _intuition_convergence = IntuitionConvergenceDetector(config=_ic_cfg)
        # Restore saved state
        _ic_state_path = "./data/intuition_convergence_state.json"
        if os.path.exists(_ic_state_path):
            import json as _ic_json
            with open(_ic_state_path) as _ic_f:
                _intuition_convergence.from_dict(_ic_json.load(_ic_f))
        logger.info("[SpiritWorker] Intuition Convergence booted (M11-M13): "
                    "events=%d, weight=%.3f",
                    _intuition_convergence._total_convergence_events,
                    _intuition_convergence._learned_weight)
    except Exception as _ic_boot_err:
        logger.warning("[SpiritWorker] Intuition Convergence init: %s", _ic_boot_err)

    # ── Boot Experience Orchestrator (Record → Distill → Bias loop) ──
    from titan_plugin.logic.experience_orchestrator import ExperienceOrchestrator, infer_domain
    from titan_plugin.logic.experience_plugins import (
        ArcPuzzlePlugin, LanguageLearningPlugin,
        CreativeExpressionPlugin, CommunicationPlugin,
    )
    exp_orchestrator = ExperienceOrchestrator(
        ex_mem=ex_mem, e_mem=e_mem, cognee_memory=None,
        db_path="./data/experience_orchestrator.db")
    exp_orchestrator.register_plugin(ArcPuzzlePlugin())
    exp_orchestrator.register_plugin(LanguageLearningPlugin())
    exp_orchestrator.register_plugin(CreativeExpressionPlugin())
    exp_orchestrator.register_plugin(CommunicationPlugin())
    _exp_dream_cycle = 0
    _cached_speak_vocab = []  # Updated on each SPEAK fire + periodic refresh
    _cached_speak_vocab_tick = 0  # Tick counter for periodic refresh
    try:
        _cached_speak_vocab = _lp_load_vocabulary(db_path="./data/inner_memory.db")
        logger.info("[SpiritWorker] Boot vocab loaded: %d words", len(_cached_speak_vocab))
    except Exception as _vocab_boot_err:
        logger.warning("[SpiritWorker] Boot vocab load failed: %s", _vocab_boot_err)
    # Wire TimeChain callback for distilled wisdom → procedural fork
    def _tc_wisdom_commit(domain, pattern, confidence, wisdom_id):
        _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
            "fork": "procedural", "thought_type": "procedural",
            "source": "dream_distillation",
            "content": {"domain": domain, "pattern": pattern[:200],
                        "confidence": confidence, "wisdom_id": wisdom_id},
            "significance": min(1.0, confidence),
            "novelty": 0.5, "coherence": confidence,
            "tags": ["wisdom", "dream_distilled", domain],
            "db_ref": f"distilled_wisdom:{wisdom_id}",
            "neuromods": dict(_cached_neuromod_state) if '_cached_neuromod_state' in dir() else {},
            "chi_available": 0.5, "attention": 0.5,
            "i_confidence": 0.5, "chi_coherence": 0.3,
        })
    exp_orchestrator._on_wisdom_commit = _tc_wisdom_commit
    logger.info("[SpiritWorker] Experience Orchestrator booted: %s",
                list(exp_orchestrator._plugins.keys()))

    # ── Boot Life Force Engine (Chi) ──
    from titan_plugin.logic.life_force import LifeForceEngine
    life_force_engine = LifeForceEngine()
    # Wire drain passive decay from titan_params.toml [dreaming] section
    try:
        import tomllib as _tom_lfe
        _lfe_path = os.path.join(os.path.dirname(__file__), "..", "titan_params.toml")
        if os.path.exists(_lfe_path):
            with open(_lfe_path, "rb") as _lfe_f:
                _lfe_cfg = _tom_lfe.load(_lfe_f).get("dreaming", {})
            if "drain_passive_decay" in _lfe_cfg:
                life_force_engine._drain_passive_decay = float(_lfe_cfg["drain_passive_decay"])
    except Exception:
        pass
    # Restore persisted metabolic_drain from dreaming_state.json (continuity across restarts)
    if coordinator and hasattr(coordinator, 'dreaming') and coordinator.dreaming:
        _persisted = getattr(coordinator.dreaming, '_persisted_drain', 0.0)
        if _persisted > 0.001:
            life_force_engine._metabolic_drain = _persisted
            logger.info("[SpiritWorker] Restored metabolic_drain=%.4f from dreaming state", _persisted)
    logger.info("[SpiritWorker] Life Force Engine (Chi) booted — 3×3 Trinity matrix (drain_decay=%.4f, drain=%.4f)",
                life_force_engine._drain_passive_decay, life_force_engine._metabolic_drain)

    # ── Boot Wallet Observer (DI:/I:/Donation detection from on-chain) ──
    _wallet_observer = None
    try:
        from titan_plugin.logic.wallet_observer import WalletObserver
        _net_cfg = config.get("network", {})
        _titan_pubkey = _net_cfg.get("titan_pubkey", "")
        _maker_pubkey = _net_cfg.get("maker_pubkey", "")
        _rpc_url = _net_cfg.get("premium_rpc_url", _net_cfg.get("rpc_url", "https://api.mainnet-beta.solana.com"))
        if _titan_pubkey and _maker_pubkey:
            _wallet_observer = WalletObserver(
                titan_pubkey=_titan_pubkey,
                maker_pubkey=_maker_pubkey,
                rpc_url=_rpc_url,
                poll_interval=30.0,
            )
            logger.info("[SpiritWorker] WalletObserver booted — listening for DI:/I:/Donations")
        else:
            logger.info("[SpiritWorker] WalletObserver skipped — no titan/maker pubkey configured")
    except Exception as _wo_err:
        logger.warning("[SpiritWorker] WalletObserver init failed: %s", _wo_err)

    # ── Wire dream subsystems to coordinator ──
    if coordinator:
        coordinator.set_dream_subsystems(
            exp_orchestrator=exp_orchestrator,
            life_force=life_force_engine,
            e_mem=e_mem,
            neuromod_system=neuromodulator_system,
        )
        logger.info("[SpiritWorker] Dream subsystems wired to coordinator")

    # ── Boot Outer Interface (Self-Exploration) ──
    from titan_plugin.logic.outer_interface import OuterInterface
    _oi_params = {}
    try:
        import tomllib as _tomllib
        _oi_params_path = os.path.join(os.path.dirname(__file__), "..", "titan_params.toml")
        if os.path.exists(_oi_params_path):
            with open(_oi_params_path, "rb") as _f:
                _oi_params = _tomllib.load(_f)
    except Exception:
        pass
    outer_interface = OuterInterface(word_recipe_dir="data", params_config=_oi_params)
    logger.info("[SpiritWorker] Outer Interface booted — self-exploration enabled (params loaded)")

    # ── Boot Reasoning Interpreter (V6 — premotor cortex) ──
    _interpreter = None
    try:
        from titan_plugin.logic.reasoning_interpreter import ReasoningInterpreter
        _interp_cfg = _oi_params.get("interpreter", {})
        _interpreter = ReasoningInterpreter(config=_interp_cfg)
        logger.info("[SpiritWorker] Reasoning Interpreter booted: %d domains (%s)",
                    len(_interpreter.registry.all()),
                    ", ".join(i.domain for i in _interpreter.registry.all()))
    except Exception as _interp_err:
        logger.warning("[SpiritWorker] Interpreter init failed: %s", _interp_err)

    # ── Boot Mini-Reasoning Modules (V6 P3 — distributed intelligence) ──
    _mini_registry = None
    try:
        from titan_plugin.logic.mini_experience import MiniReasonerRegistry
        from titan_plugin.logic.spatial_reasoner import SpatialMiniReasoner
        from titan_plugin.logic.observation_reasoner import ObservationMiniReasoner
        from titan_plugin.logic.language_reasoner import LanguageMiniReasoner
        from titan_plugin.logic.self_exploration_reasoner import SelfExplorationMiniReasoner

        _mini_registry = MiniReasonerRegistry(save_dir="./data/mini_reasoning")
        _mini_registry.register(SpatialMiniReasoner())
        _mini_registry.register(ObservationMiniReasoner())
        _mini_registry.register(LanguageMiniReasoner())
        _mini_registry.register(SelfExplorationMiniReasoner())
        _mini_registry.load_all()
        if _reasoning_engine:
            _reasoning_engine.set_mini_registry(_mini_registry)
        logger.info("[SpiritWorker] Mini-reasoning booted: %d domains (%s)",
                    len(_mini_registry.all()),
                    ", ".join(r.domain for r in _mini_registry.all()))
    except Exception as _mini_err:
        logger.warning("[SpiritWorker] Mini-reasoning init failed: %s", _mini_err)

    # Wire mini_registry to coordinator for dream consolidation
    if _mini_registry and coordinator:
        coordinator._mini_registry = _mini_registry

    # ── Language Teacher queue ──
    _teacher_queue = []              # Accumulates composition contexts, max 10
    _teacher_pending_since = 0       # Timestamp of last request sent (for timeout)
    _teacher_no_response_count = 0   # Consecutive no-responses
    _teacher_compositions_since = 0  # Counter since last teaching session
    _teacher_interval = 5            # Compositions between teaching (updated dynamically)
    _bootstrap_speak_attempts = 0    # SPEAK fires with 0 vocabulary (bootstrap tracking)
    _bootstrap_last_trigger = 0      # Timestamp of last bootstrap teacher trigger

    # ── Conversation Mode ──
    # When teacher asks a question (mode="conversation"), the next SPEAK
    # composition within 30s is captured as the response. The being answers
    # from its own felt state — no LLM generates the response.
    _conversation_pending = None     # {"question": str, "timestamp": float} or None
    _conversation_timeout = 600.0    # Max seconds to wait for SPEAK response (10 min — SPEAK fires every 4-6 min on T1)
    _conversation_stats = {"asked": 0, "answered": 0, "timed_out": 0,
                           "total_score": 0.0, "avg_score": 0.0}
    _recent_teacher_questions = []   # Last N conversation questions for dedup (max 10)
    _t2_speak_pending = False        # SPEAK would fire at Tier 2 but deferred to Tier 1

    # (Phase 4: GrammarValidator removed — now in language_worker)

    # ── Multisensory Synthesis Layer (MSL) ──
    # Top-level RL orchestrator binding all modality signals.
    # ADDITIVE only — existing pathways unchanged.
    msl = None
    _msl_tick_count = 0
    _msl_log_interval = 100  # Log attention heatmap every N ticks
    _msl_snap_interval = 2   # Snapshot every N COMPUTATION_GATE ticks (~2s)
    _msl_output = None
    try:
        from titan_plugin.logic.msl import MultisensorySynthesisLayer
        _msl_cfg = {}
        try:
            _msl_tp = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "titan_params.toml")
            if os.path.exists(_msl_tp):
                import tomllib as _msl_tl
                with open(_msl_tp, "rb") as _msl_f:
                    _msl_cfg = _msl_tl.load(_msl_f).get("msl", {})
        except Exception:
            pass
        if _msl_cfg.get("enabled", True):
            msl = MultisensorySynthesisLayer(config=_msl_cfg)
            msl.load_all()
            _msl_log_interval = _msl_cfg.get("attention_log_interval", 100)
            _msl_snap_interval = _msl_cfg.get("snapshot_interval_ticks", 2)
            logger.info("[SpiritWorker] MSL booted: input=%dD, output=%dD, "
                        "buffer=%d frames, updates=%d, "
                        "I-confidence=%.3f, convergences=%d",
                        msl.policy.input_dim, msl.policy.output_dim,
                        msl.buffer.max_frames, msl.policy.total_updates,
                        msl.get_i_confidence(),
                        msl.confidence._convergence_count)
    except Exception as _msl_err:
        logger.warning("[SpiritWorker] MSL init failed: %s", _msl_err)

    # Signal ready
    _send_msg(send_queue, "MODULE_READY", name, "guardian", {})

    # ── Background heartbeat thread (2026-04-15 fix) ──────────────────
    # Prevents Guardian 120s timeout when spirit's main loop blocks on
    # SQLite lock waits (inner_memory.db contention with
    # timechain/meditation/kin-exchange/journal writers), FAISS cognify,
    # or other sync work. Pattern copied from language_worker + llm_worker.
    # Heartbeat thread runs regardless of main-loop state; main loop's
    # inline `_send_heartbeat` calls remain but are now defense-in-depth.
    import threading as _threading_hb
    _hb_stop = _threading_hb.Event()

    def _spirit_heartbeat_loop():
        while not _hb_stop.is_set():
            try:
                _send_heartbeat(send_queue, name)
            except Exception as _hb_err:
                logger.debug("[SpiritWorker] HB thread send error (non-critical): %s", _hb_err)
            _hb_stop.wait(30.0)

    _hb_thread = _threading_hb.Thread(
        target=_spirit_heartbeat_loop, daemon=True, name="spirit-heartbeat")
    _hb_thread.start()
    logger.info("[SpiritWorker] Background heartbeat thread started (30s interval)")
    neural_ns_status = "disabled"
    if neural_nervous_system:
        neural_ns_status = "V5 %s (%d programs, phase=%s)" % (
            "enabled", len(neural_nervous_system.programs),
            neural_nervous_system.training_phase)
    logger.info("[SpiritWorker] 3DT+2 consciousness online (epochs: %d) | "
                "FilterDown: %s | Focus: enabled | Intuition: enabled | "
                "Impulse: %s | SphereClocks: %s | Resonance: %s | "
                "UnifiedSpirit: %s (GREAT EPOCHs: %d) | Observables: %s | "
                "NeuralNS: %s",
                consciousness["db"].get_epoch_count() if consciousness else 0,
                "loaded" if filter_down else "disabled",
                "enabled" if impulse_engine else "disabled",
                "enabled" if sphere_clock else "disabled",
                "enabled" if resonance else "disabled",
                "enabled" if unified_spirit else "disabled",
                unified_spirit.epoch_count if unified_spirit else 0,
                "enabled" if observable_engine else "disabled",
                neural_ns_status)

    last_publish = 0.0
    publish_interval = 0.0426  # Will be updated after Schumann boot
    last_consciousness_tick = 0.0
    last_heartbeat = 0.0

    # ── Tier 3 Adaptive Epoch State ────────────────────────────────
    _all_resonant_prev = False          # For resonance transition detection
    _fires_since_last_epoch = 0         # Hormonal urgency counter
    _urgency_drought = 0               # Consecutive epochs without URGENCY trigger
    _urgency_warmup = 0                # Post-escape warmup: keep lowered threshold
    _epoch_trigger_history = []         # Log of (timestamp, trigger_type, interval)
    _prev_topo_vector = None            # For somatic metabolic cost computation

    # ── M3: Meditation Trigger State ────────────────────────────────
    _meditation_tracker = {
        "last_epoch": 0,          # epoch_id of last meditation
        "count": 0,               # total meditations since boot
        "count_since_nft": 0,     # meditations since last MyDay NFT (M5)
        "last_ts": 0.0,           # wall-clock of last meditation (for fixed fallback)
        "in_meditation": False,   # currently meditating
    }
    # Load meditation params from titan_params.toml
    _med_cfg = {}
    try:
        _med_cfg = {k: v for k, v in config.items() if k.startswith("meditation")} if config else {}
        # Try loading from toml directly
        import tomllib as _tomllib
    except ImportError:
        import tomli as _tomllib
    try:
        with open("titan_plugin/titan_params.toml", "rb") as _mpf:
            _med_cfg = _tomllib.load(_mpf).get("meditation", {})
    except Exception:
        pass
    _med_emergent = _med_cfg.get("emergent_enabled", True)
    _med_min_epochs = _med_cfg.get("min_interval_epochs", 1500)
    _med_drain_threshold = _med_cfg.get("drain_threshold", 0.55)
    _med_gaba_offset = _med_cfg.get("gaba_offset", 0.10)
    _med_fixed_interval = _med_cfg.get("fixed_interval_seconds", 21600)

    # ── Meditation Self-Healing Watchdog (rFP Phase 1+2+3) ────────
    _med_watchdog = None
    _med_watchdog_last_check = 0.0
    _med_watchdog_interval = float(_med_cfg.get("watchdog_check_interval_seconds", 60))
    _med_watchdog_detection_only = bool(_med_cfg.get("watchdog_detection_only", True))
    # Phase 3 Tier-2 escalation tracking: if Tier-1 recovery (e.g. in_med reset)
    # fires repeatedly for the same failure_mode within a short window, escalate
    # to module restart. Avoids looping Tier-1 forever on a genuinely broken
    # memory_worker.
    _med_tier1_reset_history: dict = {}  # failure_mode → [ts, ts, ...]
    _med_tier2_recent: float = 0.0       # last Tier-2 escalation ts (cooldown)
    _med_tier2_window_s = float(_med_cfg.get("watchdog_tier2_window_seconds", 600.0))  # 10 min
    _med_tier2_threshold = int(_med_cfg.get("watchdog_tier2_reset_threshold", 2))        # 2 resets in window
    _med_tier2_cooldown_s = float(_med_cfg.get("watchdog_tier2_cooldown_seconds", 1800.0))  # 30 min between escalations
    _med_tier2_enabled = bool(_med_cfg.get("watchdog_tier2_enabled", True))
    if _med_cfg.get("watchdog_enabled", True):
        try:
            from titan_plugin.logic.meditation_watchdog import MeditationWatchdog
            _med_watchdog = MeditationWatchdog(
                titan_id=_titan_identity.get("titan_id", "T1"),
                bootstrap_hours=float(_med_cfg.get("watchdog_bootstrap_hours", 12.0)),
                min_alert_hours=float(_med_cfg.get("watchdog_min_alert_hours", 3.0)),
                gap_window=int(_med_cfg.get("watchdog_gap_window", 50)),
                stuck_threshold_seconds=float(_med_cfg.get("watchdog_stuck_threshold_seconds", 600.0)),
                backup_lag_threshold=int(_med_cfg.get("watchdog_backup_lag_threshold", 2)),
                zero_promoted_streak_threshold=int(_med_cfg.get("watchdog_zero_promoted_streak", 3)),
            )
            # I1 self-test on boot — abort if fails (don't deploy an unverified safety system)
            if not _med_watchdog.self_test():
                logger.critical(
                    "[MeditationWatchdog] Self-test FAILED — disabling watchdog to prevent false-positive actions"
                )
                _med_watchdog = None
            else:
                logger.info(
                    "[MeditationWatchdog] Initialized (detection_only=%s, interval=%.0fs, bootstrap=%.0fh, floor=%.0fh)",
                    _med_watchdog_detection_only, _med_watchdog_interval,
                    _med_watchdog.bootstrap_hours, _med_watchdog.min_alert_hours,
                )
        except Exception as _wd_err:
            logger.error("[MeditationWatchdog] Init failed — continuing without watchdog: %s", _wd_err)
            _med_watchdog = None

    # ── Schumann Sphere Clock Timers (dt-parameterized from titan_params.toml) ──
    # Body resonates at true Schumann (7.83 Hz). Mind and Spirit tune UP via ×3, ×9.
    # Sphere clocks tick at true Schumann rates (lightweight phase updates).
    # Heavy computation (NS eval, reasoning, interpreter) is gated at practical rate.
    _schumann_cfg = _oi_params.get("schumann", {})
    _schumann_freq = _schumann_cfg.get("base_frequency", 7.83)
    # Support both old (divisor) and new (multiplier) config formats
    if "spirit_multiplier" in _schumann_cfg:
        SCHUMANN_SPIRIT = 1.0 / (_schumann_freq * _schumann_cfg.get("spirit_multiplier", 9))
        SCHUMANN_MIND   = 1.0 / (_schumann_freq * _schumann_cfg.get("mind_multiplier", 3))
        SCHUMANN_BODY   = 1.0 / (_schumann_freq * _schumann_cfg.get("body_multiplier", 1))
    else:
        SCHUMANN_SPIRIT = _schumann_cfg.get("spirit_divisor", 3) / _schumann_freq
        SCHUMANN_MIND   = _schumann_cfg.get("mind_divisor", 9) / _schumann_freq
        SCHUMANN_BODY   = _schumann_cfg.get("body_divisor", 27) / _schumann_freq
    # Computation gate: heavy ops every Nth body tick
    _comp_gate_n = _schumann_cfg.get("computation_gate", 9)
    COMPUTATION_GATE = SCHUMANN_BODY * _comp_gate_n
    # Update module-level constants — epoch bounds use computation gate as base
    global SCHUMANN_BODY_CONST, EPOCH_FLOOR
    SCHUMANN_BODY_CONST = COMPUTATION_GATE  # Epoch timing based on computation rate
    _epoch_floor_mult = _schumann_cfg.get("epoch_floor_multiplier", 1)
    EPOCH_FLOOR = COMPUTATION_GATE * _epoch_floor_mult
    logger.info("[SpiritWorker] Schumann TRUE rates: spirit=%.4fs (%.1fHz) mind=%.4fs (%.1fHz) body=%.4fs (%.1fHz)",
                SCHUMANN_SPIRIT, 1.0/SCHUMANN_SPIRIT, SCHUMANN_MIND, 1.0/SCHUMANN_MIND, SCHUMANN_BODY, 1.0/SCHUMANN_BODY)
    logger.info("[SpiritWorker] Computation gate: %.3fs (every %d body ticks) | Epoch floor: %.2fs",
                COMPUTATION_GATE, _comp_gate_n, EPOCH_FLOOR)
    publish_interval = COMPUTATION_GATE  # Publish state at computation rate (~1.15s)
    last_spirit_clock_tick = 0.0
    last_mind_clock_tick = 0.0
    last_body_clock_tick = 0.0
    last_tier2_tick = 0.0     # Tier 2 FEELING evaluation (neuromod, expression, chi)
    _last_ns_sqlite_save = time.time()  # Periodic NS SQLite backup
    _avg_awake_epoch_interval = 7.2     # EMA of awake epoch intervals (self-emergent)
    _last_epoch_ts = time.time()        # Timestamp of last consciousness epoch

    # ── Wall-clock logging timers (from [timers] config, Schumann-independent) ──
    _timers_cfg = _oi_params.get("timers", {})
    _log_dream_interval = _timers_cfg.get("log_dream_s", 30)
    _log_reasoning_interval = _timers_cfg.get("log_reasoning_s", 60)
    _log_neuromod_interval = _timers_cfg.get("log_neuromod_s", 60)
    _log_consciousness_interval = _timers_cfg.get("log_consciousness_s", 120)
    _log_ns_interval = _timers_cfg.get("log_ns_training_s", 120)
    _checkpoint_ns_interval = _timers_cfg.get("checkpoint_ns_sqlite_s", 300)
    _checkpoint_neuromod_interval = _timers_cfg.get("checkpoint_neuromod_s", 300)
    _teaching_timeout = _timers_cfg.get("teaching_timeout_s", 90)
    _last_log_dream = 0.0
    _last_log_reasoning = 0.0
    _last_log_neuromod = 0.0
    _last_log_consciousness = 0.0
    _last_log_ns = 0.0

    # ── Cached bio-layer outputs (updated at COMPUTATION_GATE rate) ──
    # Digital layer reads these between bio ticks. Like nerve signal propagation:
    # fast delivery of the last computed motor command.
    _cached_filter_down_body = [1.0] * 5     # FILTER_DOWN multipliers → body
    _cached_filter_down_mind = [1.0] * 15    # FILTER_DOWN multipliers → mind
    _cached_neuromod_state = {}              # Last neuromod levels {DA, 5HT, NE, ACh, Endorphin, GABA}
    _cached_ns_urgencies = {}                # Last NS program urgencies (all 11 programs)
    _cached_chi_state = {}                   # Last Chi circulation state
    _cached_inner_65d = None                 # Last full inner 65D state vector
    _cached_outer_65d = None                 # Last full outer 65D state vector
    _cached_whole_10d = [0.5] * 10           # Last WHOLE 10DT topology

    # ── Vault state cache (A4: on-chain sovereignty → life force) ──
    _cached_vault_sov = 0.0          # sovereignty_index from vault PDA (0-100)
    _cached_vault_anchor = 0.5       # anchor freshness (0=stale, 1=recent commit)
    _cached_vault_ts = 0.0           # last vault read timestamp
    _VAULT_CACHE_TTL = 300.0         # read vault every 5 minutes (not every tick)

    def _read_vault_state():
        """Read vault PDA for sovereignty + anchor freshness (sync, cached)."""
        nonlocal _cached_vault_sov, _cached_vault_anchor, _cached_vault_ts
        now = time.time()
        if now - _cached_vault_ts < _VAULT_CACHE_TTL:
            return _cached_vault_sov, _cached_vault_anchor
        _cached_vault_ts = now
        try:
            from titan_plugin.utils.solana_client import derive_vault_pda, decode_vault_state
            _net = config.get("network", {})
            _vault_prog = _net.get("vault_program_id", "")
            _titan_pub = _net.get("titan_pubkey", "")
            if not _vault_prog or not _titan_pub:
                return _cached_vault_sov, _cached_vault_anchor
            pda_result = derive_vault_pda(_titan_pub, _vault_prog)
            if not pda_result:
                return _cached_vault_sov, _cached_vault_anchor
            vault_pda, _ = pda_result
            import base64
            import httpx
            rpc_url = _net.get("premium_rpc_url", "https://api.mainnet-beta.solana.com")
            resp = httpx.post(rpc_url, json={
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [str(vault_pda), {"encoding": "base64"}],
            }, timeout=5)
            account_info = resp.json().get("result", {}).get("value")
            if account_info:
                raw_b64 = account_info.get("data", [None])[0]
                if raw_b64:
                    state = decode_vault_state(base64.b64decode(raw_b64))
                    if state:
                        _cached_vault_sov = state.get("sovereignty_percent", 0.0)
                        # Anchor freshness: 1.0 if committed within 24h, decays to 0
                        _commit_age_h = (now - state.get("last_commit_ts", 0)) / 3600
                        _cached_vault_anchor = max(0.0, min(1.0, 1.0 - _commit_age_h / 24.0))
                        logger.debug("[VaultRead] sov=%.1f anchor=%.2f commits=%d",
                                     _cached_vault_sov, _cached_vault_anchor,
                                     state.get("commit_count", 0))
        except Exception as _vr_err:
            logger.debug("[VaultRead] Read failed (non-critical): %s", _vr_err)
        return _cached_vault_sov, _cached_vault_anchor

    # ── Language stats for inner-trinity API ──
    _language_stats = {}

    # ── Query handler thread (Phase 6.2) — responds to QUERY without blocking on computation ──
    import threading
    from queue import Queue as _ThreadQueue
    _query_queue = _ThreadQueue(maxsize=50)
    _query_state_refs = {
        "body_state": body_state, "mind_state": mind_state,
        "consciousness": consciousness, "filter_down": filter_down,
        "filter_down_v4": filter_down_v4, "filter_down_v5": filter_down_v5,
        "intuition": intuition, "impulse_engine": impulse_engine,
        "sphere_clock": sphere_clock, "resonance": resonance,
        "unified_spirit": unified_spirit, "inner_state": inner_state,
        "spirit_state": spirit_state, "coordinator": coordinator,
        "neural_nervous_system": neural_nervous_system,
        "pi_monitor": pi_monitor, "e_mem": e_mem,
        "prediction_engine": prediction_engine, "ex_mem": ex_mem,
        "episodic_mem": episodic_mem, "working_mem": working_mem,
        "inner_lower_topo": inner_lower_topo, "outer_lower_topo": outer_lower_topo,
        "ground_up_enricher": ground_up_enricher,
        "neuromodulator_system": neuromodulator_system,
        "expression_manager": expression_manager,
        "life_force_engine": life_force_engine,
        "outer_interface": outer_interface,
        "phase_tracker": _phase_tracker, "meditation_tracker": _meditation_tracker,
        "reasoning_engine": _reasoning_engine,
        "msl": msl,
        "social_pressure_meter": _social_pressure_meter,  # Legacy — kept for on_social_relief
        "x_gateway": _x_gateway,
        "language_stats": _language_stats,
        "self_reasoning": _self_reasoning,
        "coding_explorer": _coding_explorer,
        "med_watchdog": _med_watchdog,
    }
    _qt = threading.Thread(
        target=_query_handler_thread,
        args=(_query_queue, _handle_query, _query_state_refs, send_queue, name, config),
        daemon=True, name="spirit-query-handler")
    _qt.start()
    logger.info("[SpiritWorker] Query handler thread started (dedicated QUERY responses)")

    # ── Timeseries store (30-day rolling metrics) ──
    _timeseries_store = None
    try:
        from titan_plugin.logic.timeseries import TimeseriesStore
        _timeseries_store = TimeseriesStore("./data/timeseries.db")
        logger.info("[SpiritWorker] Timeseries store initialized")
    except Exception as _ts_init_err:
        logger.warning("[SpiritWorker] Timeseries store init failed: %s", _ts_init_err)

    # (Phase 4: Vocabulary cache init from DB removed — language_worker owns vocab)

    while True:
        # ── Heartbeat: ALWAYS runs every iteration regardless of messages ──
        now = time.time()
        if now - last_heartbeat >= 10.0:
            _send_heartbeat(send_queue, name)
            last_heartbeat = now

        # ── NeuromodRewardObserver tick (rFP β Stage 2 Phase 2b) ──
        # Reads current neuromod state, emits per-program rewards via
        # record_outcome every tick_interval ticks. No-op if disabled or
        # outside interval. Wrapped in try/except so observer failure
        # cannot break the main tick loop.
        if neuromod_reward_observer is not None:
            try:
                neuromod_reward_observer.tick()
            except Exception as _nro_err:
                # Rate-limited per § 4c — log every 100th failure max
                if hash(("nro_tick", _nro_err.__class__.__name__)) % 100 == 0:
                    logger.warning("[SpiritWorker] NeuromodRewardObserver tick failed: %s",
                                   _nro_err)

        # ── Periodic NS SQLite backup (every 5 min) ──
        if neural_nervous_system and now - _last_ns_sqlite_save >= _checkpoint_ns_interval:
            try:
                neural_nervous_system._sqlite_backup_save()
                logger.info("[SpiritWorker] NS SQLite backup saved (%d transitions, %d steps)",
                            neural_nervous_system._total_transitions,
                            neural_nervous_system._total_train_steps)
            except Exception as e:
                logger.warning("[SpiritWorker] NS SQLite backup failed: %s", e)
            # MSL checkpoint (piggyback on NS backup interval)
            if msl:
                try:
                    msl.save_all()
                except Exception:
                    pass
            # Reasoning engine checkpoint (rFP α 2026-04-16): previously
            # reasoning.save_all() only fired on SAVE_NOW / MODULE_RELOAD
            # events, never periodically — meaning policy_net, buffer,
            # lifetime totals, sequence_quality (Mech A), value_head
            # (Mech B), and action_chains_step retention trim all depended
            # on external triggers. Now piggybacks on the 5-min NS cycle.
            if _reasoning_engine:
                try:
                    _reasoning_engine.save_all()
                except Exception as _rse:
                    logger.warning("[SpiritWorker] reasoning save_all failed: %s", _rse)
            # X Gateway: prune old rows periodically (gateway state is in SQLite, no save needed)
            if _x_gateway and _msl_tick_count % 10000 == 0:
                try:
                    _x_gateway.prune_old_rows(days=30)
                except Exception:
                    pass
            # Intuition Convergence checkpoint
            if _intuition_convergence:
                try:
                    import json as _ic_save_json
                    _ic_sp = "./data/intuition_convergence_state.json"
                    _ic_tmp = _ic_sp + ".tmp"
                    with open(_ic_tmp, "w") as _ic_sf:
                        _ic_save_json.dump(_intuition_convergence.to_dict(), _ic_sf)
                    os.replace(_ic_tmp, _ic_sp)
                except Exception:
                    pass
            # (Social pressure state save removed — gateway uses SQLite, auto-persistent)
            # META-CGN EdgeDetector state checkpoint (Producers #1, #2, #13, #15 detectors).
            # Preserves "once per lifetime" / "personal max" / "seen signatures"
            # semantics across spirit restarts.
            try:
                _ed_detectors = {}
                _ed_sc = getattr(coordinator, "_sc_balance_detector", None) if coordinator else None
                _ed_msl = getattr(coordinator, "_msl_concept_detector", None) if coordinator else None
                _ed_rd = getattr(coordinator, "_p13_reflection_detector", None) if coordinator else None
                _ed_wd = getattr(coordinator, "_p15_wisdom_detector", None) if coordinator else None
                if _ed_sc is not None:
                    _ed_detectors["sc_balance"] = _ed_sc
                if _ed_msl is not None:
                    _ed_detectors["msl_concepts"] = _ed_msl
                if _ed_rd is not None:
                    _ed_detectors["reflection_depth"] = _ed_rd
                if _ed_wd is not None:
                    _ed_detectors["meta_wisdom"] = _ed_wd
                _ed_cg = getattr(coordinator, "_p14_coherence_detector", None) if coordinator else None
                if _ed_cg is not None:
                    _ed_detectors["coherence_gain"] = _ed_cg
                # TUNING-016: composite META-CGN (EMPATHY + CREATIVITY) edge detector
                _ed_cp = getattr(coordinator, "_composite_meta_cgn_edge", None) if coordinator else None
                if _ed_cp is not None:
                    _ed_detectors["composite_meta_cgn"] = _ed_cp
                if _ed_detectors:
                    _save_edge_detector_state(_ed_detectors)
            except Exception as _ed_err:
                logger.warning("[SpiritWorker] EdgeDetector checkpoint save failed: %s", _ed_err)
            _last_ns_sqlite_save = now

            # ── Timeseries snapshot (piggybacks on the 5-min checkpoint) ──
            try:
                if _timeseries_store and _timeseries_store.should_record():
                    from titan_plugin.logic.timeseries import collect_snapshot
                    # Build a snapshot-friendly state_refs with coordinator as dict
                    _ts_refs = dict(_query_state_refs)
                    _coord_dict = {}
                    if coordinator:
                        if hasattr(coordinator, "get"):
                            # InnerTrinityCoordinator has a .get() method like a dict
                            _coord_dict = coordinator
                        elif isinstance(coordinator, dict):
                            _coord_dict = coordinator
                    _ts_refs["coordinator"] = _coord_dict
                    _ts_metrics = collect_snapshot(_ts_refs)
                    if _ts_metrics:
                        _timeseries_store.record(_ts_metrics)
                    # Daily cleanup (once per day, check every 5-min cycle)
                    if int(now) % 86400 < 300:
                        _timeseries_store.cleanup()
            except Exception as _ts_err:
                logger.warning("[SpiritWorker] Timeseries record error: %s", _ts_err)

        # ── Receive messages: drain QUERYs to handler thread, then wait ──
        # Quick drain routes QUERYs immediately to the dedicated thread.
        # Then block for next non-QUERY message at body rate (7.83 Hz).
        msg = None
        _deferred = []
        try:
            while True:
                _drain_msg = recv_queue.get_nowait()
                if _drain_msg.get("type") == "QUERY":
                    _drain_age = time.time() - _drain_msg.get("ts", time.time())
                    logger.info("[QueryDrain] Routing QUERY %s to thread (age=%.0fms)",
                                _drain_msg.get("payload", {}).get("action", "?"), _drain_age * 1000)
                    try:
                        _query_queue.put_nowait(_drain_msg)
                    except Exception:
                        _deferred.append(_drain_msg)  # Fallback: process inline
                else:
                    _deferred.append(_drain_msg)
                    break  # Got a non-QUERY message, stop draining
        except Empty:
            pass
        except (KeyboardInterrupt, SystemExit):
            break
        # If nothing drained, wait for next message
        if not _deferred:
            try:
                _wait_msg = recv_queue.get(timeout=SCHUMANN_BODY)
                if _wait_msg.get("type") == "QUERY":
                    try:
                        _query_queue.put_nowait(_wait_msg)
                    except Exception:
                        _deferred.append(_wait_msg)
                else:
                    _deferred.append(_wait_msg)
            except Empty:
                pass
            except (KeyboardInterrupt, SystemExit):
                break
        # Process first deferred non-QUERY message
        if _deferred:
            msg = _deferred[0]

        # ── Periodic work (runs on every idle cycle, ~every 5s) ──
        now = time.time()

        # Regular spirit state publish + FOCUS + IMPULSE
        if now - last_publish >= publish_interval:
            tensor = _collect_spirit_tensor(config, body_state, mind_state, consciousness)
            _publish_spirit_state(send_queue, name, tensor, consciousness,
                                  filter_down, body_state, mind_state)
            last_publish = now

            # Heartbeat mid-work (publish + focus + impulse can take time)
            _send_heartbeat(send_queue, name)
            last_heartbeat = time.time()

            # FOCUS: PID nudges for Body/Mind (V4: with SPIRIT cascade multiplier)
            _run_focus(send_queue, name, focus_body, focus_mind, body_state, mind_state,
                       unified_spirit)

            # IMPULSE: Check for autonomous action impulse (Step 7.1)
            _run_impulse(send_queue, name, impulse_engine, body_state, mind_state, tensor, intuition)

            # Bridge ImpulseEngine deficit → IMPULSE hormone
            # ImpulseEngine detects Trinity deficits (Step 7.1). Feed deficit
            # into the hormonal IMPULSE program so deficit detection drives
            # the hormone that governs impulsive action readiness.
            if impulse_engine and neural_nervous_system:
                try:
                    _ie_body = body_state.get("values", [0.5] * 5)
                    _ie_mind = mind_state.get("values", [0.5] * 5)
                    _ie_deficits = [abs(v - 0.5) for v in _ie_body + _ie_mind]
                    _ie_max_deficit = max(_ie_deficits) if _ie_deficits else 0.0
                    if _ie_max_deficit > 0.1:
                        _ie_hormone = neural_nervous_system._hormonal.get_hormone("IMPULSE")
                        if _ie_hormone:
                            _ie_hormone.accumulate(_ie_max_deficit * 0.5, dt=0.1)
                except Exception:
                    pass

            # T3: Coordinator computes inner observables + updates InnerState
            # Also runs at body rate in D1 — kept here for spirit clock coherence data
            inner_coherences = None
            if coordinator:
                _inner_obs, inner_coherences = coordinator.tick_inner_only(
                    body_state.get("values", [0.5] * 5),
                    mind_state.get("values", [0.5] * 5),
                    tensor,
                )

            # ── Schumann SPIRIT TICK @ 70.47 Hz — Digital Layer ──────────
            # Sphere clocks + Unified Spirit (SPIRIT_SELF) observation + FILTER_DOWN apply
            if sphere_clock:
                _spirit_ticks_due = int((now - last_spirit_clock_tick) / SCHUMANN_SPIRIT)
                _spirit_ticks_active = min(_spirit_ticks_due, 3)  # Max 3 bus-visible ticks per burst
                for _ in range(_spirit_ticks_active):
                    _tick_clock_pair(send_queue, name, sphere_clock, resonance,
                                     unified_spirit, "spirit", tensor,
                                     outer_state.get("outer_spirit", [0.5] * 5),
                                     coherences=inner_coherences)
                # Advance clock phase for ALL due ticks (including silent ones)
                if _spirit_ticks_due > 0:
                    last_spirit_clock_tick += _spirit_ticks_due * SCHUMANN_SPIRIT
                # Drift guard
                if last_spirit_clock_tick < now - SCHUMANN_SPIRIT * 10:
                    last_spirit_clock_tick = now

            # ── D5: Unified Spirit (SPIRIT_SELF) @ spirit rate (70.47 Hz) ──
            # SPIRIT_SELF observes both trinities at the highest Schumann frequency.
            if unified_spirit:
                try:
                    _d5_mind = mind_state.get("values_15d", mind_state.get("values", [0.5] * 5))
                    _d5_spirit = tensor if 'tensor' in dir() else [0.5] * 5
                    try:
                        from titan_plugin.logic.spirit_tensor import collect_spirit_45d
                        _d5_cons = dict(consciousness.get("latest_epoch") or {})
                        if e_mem:
                            _d5_cons["dream_quality"] = e_mem.get_dream_quality(last_n_cycles=5)
                        _d5_spirit = collect_spirit_45d(
                            current_5d=_d5_spirit, body_tensor=body_state.get("values", [0.5] * 5),
                            mind_tensor=_d5_mind, consciousness=_d5_cons)
                    except Exception:
                        pass
                    unified_spirit.update_subconscious(
                        body_state.get("values", [0.5] * 5), _d5_mind, _d5_spirit,
                        filter_down_v5=_v5_mults_cache or None)
                except Exception:
                    pass

            # T3: Coordinator assembles SpiritState from all sources
            if coordinator:
                coordinator.assemble_spirit()

            # B2: Build temporal features from π-heartbeat for NS circadian awareness
            temporal_features = None
            if pi_monitor:
                avg_cs = max(1.0, pi_monitor.avg_cluster_size)
                _last_end = pi_monitor._last_cluster_end_epoch
                _total_obs = pi_monitor._total_epochs_observed
                _gap = (_total_obs - _last_end) if _last_end > 0 else 0
                temporal_features = {
                    "pi_phase": 1.0 if pi_monitor.in_cluster else 0.0,
                    "cluster_progress": min(1.0, pi_monitor.current_pi_streak / avg_cs),
                    "developmental_age_norm": min(1.0, pi_monitor.developmental_age / 200.0),
                    "time_since_dream": min(1.0, _gap / max(1, avg_cs * 2)),
                    "heartbeat_ratio": pi_monitor.heartbeat_ratio,
                }

            # pi_event is set later in the π-heartbeat block; initialize here
            # so coordinate() can access it. Will be None unless π-observation fires.
            pi_event = None

            # ── Build neurochemical context for emergent fatigue ──
            _neuro_ctx = {}
            try:
                if neuromodulator_system:
                    _gaba_mod = neuromodulator_system.modulators.get("GABA")
                    if _gaba_mod:
                        _neuro_ctx["gaba_level"] = _gaba_mod.level
                        _neuro_ctx["gaba_setpoint"] = _gaba_mod.setpoint
                    _total_dev = sum(
                        abs(m.level - m.setpoint)
                        for m in neuromodulator_system.modulators.values()
                    ) / max(1, len(neuromodulator_system.modulators))
                    _neuro_ctx["neuromod_deviation"] = _total_dev
                    # NE + DA for self-emergent sleep/wake competition
                    _ne_mod = neuromodulator_system.modulators.get("NE")
                    _da_mod = neuromodulator_system.modulators.get("DA")
                    if _ne_mod:
                        _neuro_ctx["ne_level"] = _ne_mod.level
                    if _da_mod:
                        _neuro_ctx["da_level"] = _da_mod.level
            except Exception as _nce:
                logger.warning("[Dreaming-ctx] Neuro context error: %s", _nce)
            try:
                if life_force_engine:
                    _chi_data = getattr(life_force_engine, '_latest_chi', {})
                    _neuro_ctx["chi_circulation"] = _chi_data.get(
                        "circulation", 0.5) if _chi_data else 0.5
                    _neuro_ctx["metabolic_drain"] = getattr(
                        life_force_engine, '_metabolic_drain', 0.0)
            except Exception as _lce:
                logger.warning("[Dreaming-ctx] Chi context error: %s", _lce)
            _neuro_ctx["curvature_variance"] = getattr(
                _sw_local, '_curvature_variance', 0.5)

            _exp_ctx = {}
            if exp_orchestrator:
                try:
                    _eo_stats = exp_orchestrator.get_stats()
                    _exp_ctx["undistilled"] = _eo_stats.get("undistilled", 0)
                    _exp_ctx["pre_dream_undistilled"] = _eo_stats.get("pre_dream_undistilled", 0)
                    _exp_ctx["total"] = _eo_stats.get("total_records", 1)
                    _exp_ctx["total_wisdom"] = _eo_stats.get("total_wisdom", 0)
                except Exception:
                    pass

            # O5: Expression repetitiveness from composition history
            # (unique_recent / total_recent over last 50 compositions)
            try:
                import sqlite3 as _sql3_rep
                _rep_conn = _sql3_rep.connect("./data/inner_memory.db", timeout=2.0)
                _rep_row = _rep_conn.execute(
                    "SELECT count(DISTINCT sentence) as u, count(*) as t "
                    "FROM (SELECT sentence FROM composition_history "
                    "ORDER BY ROWID DESC LIMIT 50)"
                ).fetchone()
                _rep_conn.close()
                if _rep_row and _rep_row[1] > 0:
                    _exp_ctx["repetitiveness"] = 1.0 - _rep_row[0] / _rep_row[1]
            except Exception:
                pass

            # T4-T7: Run coordination logic (nervous system, topology, dreaming, GREAT PULSE)
            # SINGLE AUTHORITY: coordinator.coordinate() is the ONLY dreaming decision path.
            # π-events passed as accelerator (lowers threshold), not as direct trigger.
            if _neuro_ctx:
                if now - _last_log_dream > _log_dream_interval:
                    _last_log_dream = now
                    logger.info("[Dreaming-ctx] neuro=%s exp=%s", _neuro_ctx, _exp_ctx)
            if coordinator:
                try:
                    _cons = consciousness if consciousness and isinstance(consciousness, dict) else {}
                    coordinator._last_epoch_id = _cons.get("latest_epoch", {}).get("epoch_id", 0)
                except Exception:
                    pass
                coord_result = coordinator.coordinate(
                    temporal=temporal_features,
                    neurochemical=_neuro_ctx,
                    experience=_exp_ctx,
                    pi_event=pi_event,
                )
                coord_event = coord_result.get("event")

                # Write topology state for Body proprioception (cross-process)
                try:
                    import json as _json
                    topo = coord_result.get("topology", {})
                    inner_radii = []
                    if sphere_clock:
                        for p in ("inner_body", "inner_mind", "inner_spirit"):
                            clock = sphere_clock._clocks.get(p)
                            if clock:
                                inner_radii.append(clock.get("radius", 1.0))
                    mean_r = sum(inner_radii) / max(1, len(inner_radii)) if inner_radii else 1.0
                    topo_out = {
                        "mean_inner_radius": round(mean_r, 4),
                        "volume": topo.get("volume", 0.0),
                        "curvature": topo.get("curvature", 0.0),
                        "ts": time.time(),
                    }
                    topo_file = os.path.join(
                        os.path.dirname(__file__), "..", "..", "data", "body_topology.json")
                    with open(topo_file, "w") as _f:
                        _json.dump(topo_out, _f)
                except Exception:
                    pass

                if coord_event == "GREAT_PULSE":
                    # T7 DREAMING CONVERGENCE path DISABLED (2026-03-23).
                    # Reason: Created GREAT EPOCH spam (~8s during dreams) and was
                    # the original deadlock cause (T7 deferred GREAT PULSE to dreaming
                    # → dreaming needed GREAT PULSE → circular dependency).
                    # GREAT PULSE now fires from resonance TRANSITION detector in
                    # spirit_loop.py _check_resonance() — naturally gated by Body
                    # pair cycle (~292s). See PLAN_circadian_rhythm_restoration.md.
                    logger.debug(
                        "[SpiritWorker] T7 GREAT_PULSE event (topology convergence) — "
                        "logged only, resonance transition is primary trigger")

                elif coord_event == "BEGIN_DREAMING":
                    logger.info("[SpiritWorker] Titan entering DREAM state")
                    _shared_is_dreaming = True
                    if life_force_engine:
                        life_force_engine.set_dreaming(True)
                    # Neuromod: boost clearance during dream phase
                    if neuromodulator_system:
                        _gaba = neuromodulator_system.modulators.get("GABA")
                        _gaba_level = _gaba.level if _gaba else 0.5
                        _clearance_boost = 1.0 + _gaba_level * 3.0
                        for _mn, _mm in neuromodulator_system.modulators.items():
                            if _mn != "GABA":
                                _mm._dream_clearance_boost = _clearance_boost
                            else:
                                # GABA gets NORMAL clearance during dreams (not reduced).
                                # GABA decline during sleep is driven by production-side:
                                # metabolic_drain falls → GABA production drops → net decline.
                                # Previous 0.5 (reduced) kept GABA frozen during dreams.
                                _mm._dream_clearance_boost = 1.0
                        logger.info("[Neuromod] Dream clearance boost: %.1fx (GABA=1.0x normal, GABA=%.2f)",
                                    _clearance_boost, _gaba_level)
                    # CGN dream consolidation — tell language_worker's CGN (which has the buffer)
                    _send_msg(send_queue, "CGN_DREAM_CONSOLIDATE", name, "language", {
                        "dream_phase": True,
                    })

                    # Notify v4_bridge for frontend dream state display
                    _ds_payload = {
                        "is_dreaming": True,
                        "cycle": getattr(getattr(coordinator, 'inner', None), 'cycle_count', 0),
                    }
                    _send_msg(send_queue, "DREAM_STATE_CHANGED", name, "all", _ds_payload)
                    _send_msg(send_queue, "DREAM_STATE_CHANGED", name, "timechain", _ds_payload)
                    # Reset coding explorer per-dream counter
                    if _coding_explorer:
                        _coding_explorer.on_dream_start()

                elif coord_event == "END_DREAMING":
                    logger.info("[SpiritWorker] Titan waking from DREAM state")
                    _shared_is_dreaming = False
                    # Social: cue dream for meditation summary (don't post after every dream)
                    if _social_pressure_meter:
                        _social_pressure_meter.cue_dream_for_meditation({
                            "emotion_at_wake": neuromodulator_system._current_emotion if neuromodulator_system else "neutral",
                        })
                    if life_force_engine:
                        life_force_engine.set_dreaming(False)
                    # Re-tag dream experiences for next cycle distillation
                    if exp_orchestrator:
                        try:
                            _retag_count = exp_orchestrator.retag_dream_experiences()
                            logger.info("[SpiritWorker] Re-tagged %d dream experiences for next cycle",
                                        _retag_count)
                        except Exception as _retag_err:
                            logger.warning("[SpiritWorker] Dream retag error: %s", _retag_err)
                    # Neuromod: restore normal clearance + resensitize
                    if neuromodulator_system:
                        for _mn, _mm in neuromodulator_system.modulators.items():
                            _mm._dream_clearance_boost = 1.0
                            _mm.sensitivity = max(0.5, min(2.0,
                                (_mm.sensitivity + 1.0) / 2.0))
                        logger.info("[Neuromod] Dream clearance restored, "
                                    "receptors resensitized")
                    # Notify v4_bridge for frontend dream state display
                    _dw_payload = {
                        "is_dreaming": False,
                        "cycle": getattr(coordinator.dreaming, '_cycle_count', 0),
                    }
                    _send_msg(send_queue, "DREAM_STATE_CHANGED", name, "all", _dw_payload)
                    _send_msg(send_queue, "DREAM_STATE_CHANGED", name, "timechain", _dw_payload)
                    # MSL dream-time training (boosted LR, same pattern as meta-reasoning)
                    if msl:
                        try:
                            _msl_dream = msl.train(boost_factor=2.0)
                            if _msl_dream.get("trained"):
                                logger.info("[MSL] Dream training: %d samples, "
                                            "avg_loss=%.6f, total_updates=%d",
                                            _msl_dream.get("samples", 0),
                                            _msl_dream.get("avg_loss", 0),
                                            _msl_dream.get("total_updates", 0))
                        except Exception as _msl_dream_err:
                            logger.warning("[MSL] Dream training error: %s",
                                           _msl_dream_err)

                    # ── Bridge A: Inner→Outer Dream Memory Injection ──
                    # Harvest significant inner events and inject into cognitive graph.
                    # Fire-and-forget via QUERY to memory_worker (non-blocking).
                    try:
                        from titan_plugin.modules.spirit_loop import (
                            _harvest_dream_memories, _build_felt_snapshot)
                        _dream_cycle = getattr(coordinator.dreaming, '_cycle_count', 0)
                        _harvested, _consolidated_ids = _harvest_dream_memories(
                            chain_archive=chain_archive,
                            meta_wisdom=meta_wisdom,
                            neuromod_system=neuromodulator_system,
                            cgn_db_path="./data/inner_memory.db",
                            dream_cycle=_dream_cycle,
                            max_total=8,
                        )
                        for _mem_payload in _harvested:
                            _send_msg(send_queue, "QUERY", name, "memory", {
                                "action": "add",
                                "text": _mem_payload["text"],
                                "source": _mem_payload["source"],
                                "weight": _mem_payload["weight"],
                                "neuromod_context": _mem_payload["neuromod_context"],
                            })
                        if chain_archive and _consolidated_ids:
                            chain_archive.mark_consolidated(_consolidated_ids)
                        if _harvested:
                            logger.info("[DreamBridge] Injected %d memories into "
                                        "cognitive graph (cycle=%d: %s)",
                                        len(_harvested), _dream_cycle,
                                        ", ".join(m["category"] for m in _harvested))
                            # I-depth: dream bridge is a source of self-knowledge
                            if msl and hasattr(msl, 'i_depth'):
                                msl.i_depth.record_extended_source("dream")
                                msl.i_depth.record_dream_bridge(len(_harvested))
                    except Exception as _bridge_err:
                        logger.warning("[DreamBridge] Inner→Outer error: %s",
                                       _bridge_err)

                    # Layer 1: Self-Profile — write self-knowledge to cognitive graph
                    # Overwrites previous profile each dream cycle (only latest matters)
                    # Uses SelfReasoningEngine (enriched) with legacy fallback
                    try:
                        _sp_nm = {}
                        if neuromodulator_system:
                            _sp_nm = {n: m.level for n, m in neuromodulator_system.modulators.items()}
                        _sp_text = None

                        if _self_reasoning:
                            # Enriched self-profile via Self-Reasoning Engine
                            _sp_msl = {}
                            if msl:
                                try:
                                    _sp_ict = getattr(msl, '_i_confidence_tracker', None)
                                    _sp_idt = getattr(msl, '_i_depth_tracker', None)
                                    _sp_chit = getattr(msl, '_chi_tracker', None)
                                    _sp_cg = getattr(msl, '_concept_grounder', None)
                                    _sp_msl = {
                                        "i_confidence": _sp_ict.confidence if _sp_ict else 0.0,
                                        "i_depth": _sp_idt.depth if _sp_idt else 0.0,
                                        "i_depth_components": _sp_idt.get_stats().get("components", {}) if _sp_idt else {},
                                        "chi_coherence": _sp_chit.get_chi_state().get("chi_coherence", 0.0) if _sp_chit else 0.0,
                                        "convergence_count": _sp_ict._convergence_count if _sp_ict else 0,
                                        "concept_confidences": _sp_cg.get_concept_confidences() if _sp_cg else {},
                                    }
                                except Exception:
                                    pass
                            _sp_reason = {}
                            if meta_engine:
                                _sp_me_stats = meta_engine.get_stats()
                                _sp_pc = _sp_me_stats.get("primitive_counts", {})
                                _sp_reason = {
                                    "total_chains": _sp_me_stats.get("total_chains", 0),
                                    "dominant_primitive": max(_sp_pc, key=_sp_pc.get) if _sp_pc else "",
                                    "eureka_count": _sp_me_stats.get("total_eurekas", 0),
                                    "wisdom_count": _sp_me_stats.get("total_wisdom_saved", 0),
                                    "commit_rate": 0.0,
                                }
                            _sp_profile = _self_reasoning.build_self_profile(
                                epoch=epoch_id,
                                neuromods=_sp_nm,
                                msl_data=_sp_msl,
                                reasoning_stats=_sp_reason,
                                language_stats=_language_stats or {},
                                coordinator_data={
                                    "dream_cycles": _dream_cycle,
                                    "ns_train_steps": neural_nervous_system.total_steps if neural_nervous_system and hasattr(neural_nervous_system, 'total_steps') else 0,
                                },
                            )
                            _sp_text = _self_reasoning.get_self_profile_text(_sp_profile)

                            # Also consolidate self-reasoning during dreams
                            _sr_consolidation = _self_reasoning.consolidate_training()
                            logger.info("[SelfReasoning] Dream consolidation: %s",
                                        _sr_consolidation)

                        if not _sp_text:
                            # Legacy fallback
                            from titan_plugin.modules.spirit_loop import _build_self_profile
                            _legacy_profile = _build_self_profile(
                                neuromod_system=neuromodulator_system,
                                msl=msl,
                                meta_engine=meta_engine,
                                chain_archive=chain_archive,
                                dream_cycle=_dream_cycle,
                                language_stats=_language_stats,
                            )
                            if _legacy_profile:
                                _sp_text = _legacy_profile["text"]

                        if _sp_text:
                            _sp_felt = {n: round(v, 4) for n, v in _sp_nm.items()} if _sp_nm else {}
                            _sp_inject_weight = float(
                                meta_engine._dna.get("self_profile_dream_inject_weight", 10.0)
                            ) if meta_engine and hasattr(meta_engine, '_dna') else 10.0
                            _send_msg(send_queue, "QUERY", name, "memory", {
                                "action": "add",
                                "text": _sp_text,
                                "source": "self_profile",
                                "weight": _sp_inject_weight,
                                "neuromod_context": _sp_felt,
                            })
                            logger.info("[SelfProfile] Injected self-profile into "
                                        "cognitive graph (cycle=%d, engine=%s)",
                                        _dream_cycle,
                                        "self_reasoning" if _self_reasoning else "legacy")

                            # ── Phase E: Enhanced Self-Knowledge Integration ──

                            # E.1: TimeChain meta fork — persist self-profile as block
                            _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
                                "fork": "meta", "thought_type": "meta",
                                "source": "dream_self_profile",
                                "content": {
                                    "event": "SELF_PROFILE_SNAPSHOT",
                                    "profile_text": _sp_text[:2000],
                                    "neuromods": _sp_felt,
                                    "epoch": epoch_id,
                                    "dream_cycle": _dream_cycle,
                                    "reasoning_stats": _sp_reason,
                                },
                                "significance": 0.8,
                                "novelty": 0.5, "coherence": 0.7,
                                "tags": ["self_observation", "dream", "self_profile"],
                                "neuromods": _sp_felt,
                                "chi_available": 0.5,
                                "attention": 0.5, "i_confidence": 0.5, "chi_coherence": 0.3,
                            })

                            # E.2: Profile diff tracking — compute delta from last profile
                            _sp_dna = (meta_engine._dna if meta_engine
                                       and hasattr(meta_engine, '_dna') else {})
                            if _sp_dna.get("self_profile_diff_tracking", True):
                                try:
                                    _prev_profile = getattr(
                                        _self_reasoning, '_last_dream_profile', None
                                    ) if _self_reasoning else None
                                    if _prev_profile and _sp_profile:
                                        _diff_keys = set(
                                            list(_sp_profile.keys()) +
                                            list(_prev_profile.keys())
                                        )
                                        _diff_entries = {}
                                        for _dk in _diff_keys:
                                            _cv = _sp_profile.get(_dk)
                                            _pv = _prev_profile.get(_dk)
                                            if isinstance(_cv, (int, float)) and isinstance(_pv, (int, float)):
                                                _delta = _cv - _pv
                                                if abs(_delta) > 0.01:
                                                    _diff_entries[_dk] = round(_delta, 4)
                                        if _diff_entries:
                                            _send_msg(send_queue, "TIMECHAIN_COMMIT",
                                                      name, "timechain", {
                                                "fork": "meta", "thought_type": "meta",
                                                "source": "self_profile_diff",
                                                "content": {
                                                    "event": "SELF_PROFILE_DIFF",
                                                    "deltas": _diff_entries,
                                                    "epoch": epoch_id,
                                                    "dream_cycle": _dream_cycle,
                                                },
                                                "significance": min(1.0,
                                                    sum(abs(v) for v in _diff_entries.values()) * 2),
                                                "novelty": 0.6, "coherence": 0.5,
                                                "tags": ["self_observation", "diff"],
                                                "neuromods": _sp_felt,
                                                "chi_available": 0.5,
                                                "attention": 0.5, "i_confidence": 0.5,
                                                "chi_coherence": 0.3,
                                            })
                                            logger.info(
                                                "[SelfProfile] Diff: %d changed fields, "
                                                "top: %s",
                                                len(_diff_entries),
                                                ", ".join(f"{k}={v:+.3f}"
                                                          for k, v in sorted(
                                                    _diff_entries.items(),
                                                    key=lambda x: -abs(x[1]))[:5]))
                                    # Store current profile for next diff
                                    if _self_reasoning and _sp_profile:
                                        _self_reasoning._last_dream_profile = dict(_sp_profile)
                                except Exception as _diff_err:
                                    logger.debug("[SelfProfile] Diff error: %s", _diff_err)

                            # E.3: Emotional signature — dominant emotional pattern
                            if _sp_dna.get("self_emotional_signature", True):
                                try:
                                    _emo = (neuromodulator_system._current_emotion
                                            if neuromodulator_system
                                            and hasattr(neuromodulator_system, '_current_emotion')
                                            else "neutral")
                                    _emo_hist = (getattr(neuromodulator_system,
                                                         '_emotion_history', [])
                                                 if neuromodulator_system else [])
                                    _emo_counts: dict = {}
                                    for _eh in (_emo_hist[-100:] if _emo_hist else []):
                                        _en = _eh if isinstance(_eh, str) else str(_eh)
                                        _emo_counts[_en] = _emo_counts.get(_en, 0) + 1
                                    _emo_total = max(1, sum(_emo_counts.values()))
                                    _emo_dist = {k: round(v / _emo_total, 2)
                                                 for k, v in sorted(
                                        _emo_counts.items(), key=lambda x: -x[1])[:5]}
                                    if _emo_dist:
                                        logger.info("[SelfProfile] Emotional signature: %s "
                                                    "(current=%s)", _emo_dist, _emo)
                                except Exception:
                                    pass

                            # E.4: Prediction-as-concept — ground self-predictions via CGN
                            if (_sp_dna.get("self_prediction_as_concept", True)
                                    and _self_reasoning
                                    and hasattr(_self_reasoning, 'get_active_predictions')):
                                try:
                                    _preds = _self_reasoning.get_active_predictions()
                                    for _pred in (_preds[:3] if _preds else []):
                                        _pred_text = _pred.get("text", _pred.get("prediction", ""))
                                        if _pred_text:
                                            _send_msg(send_queue, "CGN_TRANSITION",
                                                      name, "cgn", {
                                                "consumer": "self_model",
                                                "concept_id": f"pred_{_pred.get('id', 0)}",
                                                "action": 1,  # predict_transition
                                                "outcome_context": {
                                                    "prediction": _pred_text[:200],
                                                    "horizon": _pred.get("horizon", 2000),
                                                    "confidence": _pred.get("confidence", 0.5),
                                                },
                                                "reward": 0.0,  # delayed — verified later
                                            })
                                    if _preds:
                                        logger.info("[SelfProfile] Grounded %d predictions "
                                                    "as CGN concepts", min(3, len(_preds)))
                                except Exception as _pred_err:
                                    logger.debug("[SelfProfile] Prediction grounding: %s",
                                                 _pred_err)

                            # E.5: Cross-Titan awareness (lightweight — from kin state)
                            if _sp_dna.get("self_cross_titan_awareness", True):
                                try:
                                    _e5_kin = getattr(coordinator, '_kin_state', None)
                                    if _e5_kin and isinstance(_e5_kin, dict):
                                        _sibling_notes = []
                                        for _kt_name, _kt_data in _e5_kin.items():
                                            if isinstance(_kt_data, dict):
                                                _kt_dom = _kt_data.get("dominant_primitive", "?")
                                                _kt_emo = _kt_data.get("emotion", "?")
                                                _sibling_notes.append(
                                                    f"{_kt_name}: {_kt_dom}/{_kt_emo}")
                                        if _sibling_notes:
                                            logger.info("[SelfProfile] Sibling awareness: %s",
                                                        ", ".join(_sibling_notes))
                                except Exception:
                                    pass

                    except Exception as _sp_err:
                        logger.warning("[SelfProfile] Error: %s", _sp_err)

                    # ── META-CGN producer #3: dreaming.insight_distilled ──
                    # v3 Phase D rollout (rFP_meta_cgn_v3 § 12 row 3).
                    # Fires once per dream cycle — END_DREAMING runs ~3×/day per
                    # Titan, so the dream cycle itself is the natural gate (no
                    # EdgeDetector needed; default min_interval_s=0.5 is safe
                    # because dream cycles are hours apart).
                    # Intensity scales with harvested-insight count (0-8 → 0.1-1.0).
                    try:
                        from ..bus import emit_meta_cgn_signal
                        _p3_cycle = getattr(coordinator.dreaming, '_cycle_count', 0)
                        _p3_count = len(_harvested) if '_harvested' in dir() else 0
                        _p3_intensity = min(1.0, max(0.1, _p3_count / 8.0))
                        _p3_sent = emit_meta_cgn_signal(
                            send_queue,
                            src="dreaming",
                            consumer="dreaming",
                            event_type="insight_distilled",
                            intensity=_p3_intensity,
                            domain=f"cycle_{_p3_cycle}",
                            reason=f"dream cycle {_p3_cycle} distilled {_p3_count} insights",
                        )
                        if _p3_sent:
                            logger.info(
                                "[META-CGN] dreaming.insight_distilled EMIT — cycle=%d insights=%d intensity=%.2f",
                                _p3_cycle, _p3_count, _p3_intensity)
                        else:
                            logger.warning(
                                "[META-CGN] Producer #3 dreaming.insight_distilled DROPPED by bus "
                                "— cycle=%d (rate-gate or queue-full; signal missed)", _p3_cycle)
                    except Exception as _p3_err:
                        logger.warning(
                            "[META-CGN] Producer #3 dreaming.insight_distilled emit FAILED "
                            "— cycle=%s err=%s (signal missed)",
                            getattr(coordinator.dreaming, '_cycle_count', '?'), _p3_err)

                    # rFP β Stage 2 Phase 2c: REFLECTION event hook
                    # Reward REFLECTION when dream cycle distills meaningful insights.
                    # Stage 0.5 empirical: REFLECTION K=5, decay=0.725, window=2014s
                    # (matches dream cycle scale). Discrete sparse signal complements
                    # the dense low-NE+high-ACh neuromod stream.
                    if neural_nervous_system:
                        try:
                            _refl_count = len(_harvested) if '_harvested' in dir() else 0
                            if _refl_count > 0:
                                # Scale reward with insight count (0→0, 1→0.5, 4→0.8, 8+→1.0)
                                _refl_reward = min(1.0, 0.4 + 0.1 * _refl_count)
                                neural_nervous_system.record_outcome(
                                    reward=_refl_reward,
                                    program="REFLECTION",
                                    source="dream.insight_distilled")
                            elif _refl_count == 0:
                                # Negative signal: dream completed but no insights distilled
                                neural_nervous_system.record_outcome(
                                    reward=-0.2,
                                    program="REFLECTION",
                                    source="dream.no_insights")
                        except Exception as _refl_err:
                            if hash(("refl_hook", _refl_err.__class__.__name__)) % 100 == 0:
                                logger.warning("[NS-Hook] REFLECTION reward failed: %s", _refl_err)

                        # rFP β Phase 3 § 4g: NS → META-CGN coupling on dream end
                        # ("reflection", "fired") → INTROSPECT/SPIRIT_SELF/EVALUATE
                        # Naturally rate-limited by dream cycle cadence (~3×/day per Titan).
                        try:
                            _refl_count_meta = len(_harvested) if '_harvested' in dir() else 0
                            _refl_intensity = min(1.0, max(0.1, 0.3 + 0.1 * _refl_count_meta))
                            from ..bus import emit_meta_cgn_signal
                            emit_meta_cgn_signal(
                                send_queue,
                                src="reflection", consumer="reflection",
                                event_type="fired",
                                intensity=_refl_intensity,
                                domain="dream_cycle",
                                reason=f"REFLECTION dream end insights={_refl_count_meta}")
                        except Exception:
                            pass

                # Autonomy-first: publish outer program dispatch signals to bus
                # Agency core loop picks these up and dispatches helpers WITHOUT LLM
                if neural_nervous_system:
                    outer_signals = neural_nervous_system.get_outer_dispatch_signals()
                    if outer_signals:
                        _send_msg(send_queue, "OUTER_DISPATCH", name, "agency", {
                            "signals": outer_signals,
                            "ts": time.time(),
                        })

            # Heartbeat after all periodic work
            _send_heartbeat(send_queue, name)
            last_heartbeat = time.time()

        # ── Schumann Sphere Clocks: Mind pair @ 23.49 Hz (batch) ────
        if sphere_clock:
            _im_for_clock = mind_state.get("values_15d",
                                           mind_state.get("values", [0.5] * 5))
            _om_for_clock = outer_state.get("outer_mind_15d",
                                            outer_state.get("outer_mind", [0.5] * 5))
            _mind_ticks_due = int((now - last_mind_clock_tick) / SCHUMANN_MIND)
            _mind_ticks_active = min(_mind_ticks_due, 3)
            for _ in range(_mind_ticks_active):
                _tick_clock_pair(send_queue, name, sphere_clock, resonance,
                                 unified_spirit, "mind",
                                 _im_for_clock,
                                 _om_for_clock)
            if _mind_ticks_due > 0:
                last_mind_clock_tick += _mind_ticks_due * SCHUMANN_MIND
            if last_mind_clock_tick < now - SCHUMANN_MIND * 10:
                last_mind_clock_tick = now

        # ── Schumann BODY TICK @ 7.83 Hz — Digital Layer ──────────────
        # Body enrichment, oBody→oMind, GROUND_UP, compute_extended, topology
        if sphere_clock:
            _body_ticks_due = int((now - last_body_clock_tick) / SCHUMANN_BODY)
            _body_ticks_active = min(_body_ticks_due, 2)
            for _ in range(_body_ticks_active):
                _tick_clock_pair(send_queue, name, sphere_clock, resonance,
                                 unified_spirit, "body",
                                 body_state.get("values", [0.5] * 5),
                                 outer_state.get("outer_body", [0.5] * 5))
            if _body_ticks_due > 0:
                last_body_clock_tick += _body_ticks_due * SCHUMANN_BODY
            if last_body_clock_tick < now - SCHUMANN_BODY * 10:
                last_body_clock_tick = now

            # ── D1: Coordinator inner observables @ body rate ──
            if coordinator and _body_ticks_due > 0:
                try:
                    _d1_obs, _d1_coh = coordinator.tick_inner_only(
                        body_state.get("values", [0.5] * 5),
                        mind_state.get("values", [0.5] * 5),
                        tensor if 'tensor' in dir() else [0.5] * 5,
                    )
                except Exception:
                    pass

                # rFP #1 Phase 2: publish OBSERVABLES_SNAPSHOT so state_register
                # can expose full_30d_topology via STATE_SNAPSHOT for downstream
                # consumers (rFP #2 TITAN_SELF topology distillation).
                if observable_engine and inner_state and inner_state.observables:
                    try:
                        _obs_dict = dict(inner_state.observables)
                        _obs_30d = observable_engine.get_observations_30d(_obs_dict)
                        send_queue.put({
                            "type": "OBSERVABLES_SNAPSHOT",
                            "src": name,
                            "dst": "state_register",
                            "ts": time.time(),
                            "payload": {
                                "observables_dict": _obs_dict,
                                "observables_30d":  _obs_30d,
                            },
                        })
                    except Exception as _obs_err:
                        logger.debug("[Observables] inner snapshot publish error: %s", _obs_err)

            # ── D2: oBody→oMind Feeling enrichment @ body rate ──
            if _body_ticks_due > 0:
                try:
                    _d2_om15 = outer_state.get("outer_mind_15d")
                    _d2_ob5 = outer_state.get("outer_body", [0.5] * 5)
                    if _d2_om15 and len(_d2_om15) >= 15 and _d2_ob5:
                        _D2_STRENGTH = 0.02
                        _d2_changed = False
                        for _fi in range(5):
                            _d2_bv = _d2_ob5[_fi] if _fi < len(_d2_ob5) else 0.5
                            _d2_fv = _d2_om15[5 + _fi]
                            _d2_d = (_d2_bv - _d2_fv) * _D2_STRENGTH
                            _d2_d = max(-0.02, min(0.02, _d2_d))
                            _d2_om15[5 + _fi] = max(0.0, min(1.0, _d2_fv + _d2_d))
                            if abs(_d2_d) > 0.001:
                                _d2_changed = True
                        if _d2_changed:
                            outer_state["outer_mind_15d"] = _d2_om15
                except Exception:
                    pass

            # ── D3: GROUND_UP @ body rate (from cached 132D state) ──
            if _body_ticks_due > 0 and _cached_inner_65d and _cached_outer_65d:
                try:
                    _d3_ib5 = _cached_inner_65d[0:5]
                    _d3_imw5 = _cached_inner_65d[15:20]
                    _d3_ob5 = _cached_outer_65d[0:5]
                    _d3_omw5 = _cached_outer_65d[15:20]
                    _d3_il = inner_lower_topo.compute(_d3_ib5, _d3_imw5, _cached_whole_10d)
                    _d3_ol = outer_lower_topo.compute(_d3_ob5, _d3_omw5, _cached_whole_10d)
                    import math as _d3m
                    _d3_is_mag = _d3m.sqrt(sum(v * v for v in _cached_inner_65d[20:65]))
                    _d3_os_mag = _d3m.sqrt(sum(v * v for v in _cached_outer_65d[20:65]))
                    _d3_topo = getattr(coordinator, 'topology', None) if coordinator else None
                    if _d3_topo:
                        _d3_basic = getattr(coordinator, '_last_topology', {})
                        _d3_w10 = _d3_topo.compute_whole_10d(
                            _d3_basic, _d3_il, _d3_ol,
                            inner_mind_willing=_d3_imw5, outer_mind_willing=_d3_omw5,
                            spirit_magnitudes=[_d3_is_mag, _d3_os_mag])
                        _cached_whole_10d = list(_d3_w10)
                        if coordinator:
                            coordinator._last_whole_10d = _d3_w10
                    # Apply GROUND_UP to inner trinity
                    _d3_new_ib, _d3_new_im = ground_up_enricher.apply(
                        _d3_ib5, list(_cached_inner_65d[5:20]), _d3_il["grounding_signal"], dt=1.0)
                    _d3_new_ob, _d3_new_om = ground_up_enricher.apply(
                        _d3_ob5, list(_cached_outer_65d[5:20]), _d3_ol["grounding_signal"], dt=1.0)
                    # Write back to live state
                    body_state["values"] = list(_d3_new_ib)
                    _d3_ms = mind_state.get("values_15d", [0.5] * 15)
                    if len(_d3_ms) >= 15:
                        _d3_ms[:5] = _d3_new_im[:5]
                        _d3_ms[10:15] = _d3_new_im[10:15]
                        mind_state["values_15d"] = _d3_ms
                except Exception:
                    pass

            # ── D4: compute_extended(130D) @ body rate — full Sat-Chit-Ananda topology ──
            if _body_ticks_due > 0 and _cached_inner_65d and _cached_outer_65d:
                try:
                    _d4_topo = getattr(coordinator, 'topology', None) if coordinator else None
                    if _d4_topo and hasattr(_d4_topo, 'compute_extended'):
                        _d4_ext = _d4_topo.compute_extended(_cached_inner_65d, _cached_outer_65d)
                        if coordinator:
                            coordinator._last_extended_topology = _d4_ext
                except Exception:
                    pass

            # D5 (Unified Spirit) → moved to spirit tick block (70.47 Hz)

            # ── FILTER_DOWN: Apply SELF enrichment from GREAT PULSE ──
            # When GREAT PULSE fires (resonance transition), UnifiedSpirit
            # computes enrichment rewards for all 6 Trinity components.
            # Spirit's 130D tensor IS the harmonized ideal — rewards nudge
            # body/mind TOWARD Spirit's vision (top-down FILTER_DOWN).
            # Spirit components are NOT enriched — Spirit OBSERVES.
            _gp_enrichment = getattr(_check_resonance, '_pending_enrichment', None)
            if _gp_enrichment and unified_spirit:
                _spirit_t = unified_spirit.tensor  # 130D live tensor
                _enrich_applied = False

                # Inner Body: Spirit tensor [0:5] → body_state["values"]
                _ib_r = _gp_enrichment.get("inner_body", {}).get("reward", 0)
                if _ib_r > 0:
                    _bs = body_state.get("values", [])
                    for _i in range(min(5, len(_bs))):
                        _delta = (_spirit_t[_i] - _bs[_i]) * _ib_r
                        _bs[_i] = max(0.0, min(1.0, _bs[_i] + _delta))
                    _enrich_applied = True

                # Inner Mind: Spirit tensor [5:20] → mind_state["values_15d"]
                _im_r = _gp_enrichment.get("inner_mind", {}).get("reward", 0)
                if _im_r > 0:
                    _ms = mind_state.get("values_15d", [])
                    for _i in range(min(15, len(_ms))):
                        _delta = (_spirit_t[5 + _i] - _ms[_i]) * _im_r
                        _ms[_i] = max(0.0, min(1.0, _ms[_i] + _delta))
                    _enrich_applied = True

                # Outer Body: Spirit tensor [65:70] → outer_state["outer_body"]
                _ob_r = _gp_enrichment.get("outer_body", {}).get("reward", 0)
                if _ob_r > 0:
                    _obs = outer_state.get("outer_body") or []
                    for _i in range(min(5, len(_obs))):
                        _delta = (_spirit_t[65 + _i] - _obs[_i]) * _ob_r
                        _obs[_i] = max(0.0, min(1.0, _obs[_i] + _delta))
                    _enrich_applied = True

                # Outer Mind: Spirit tensor [70:85] → outer_state["outer_mind_15d"]
                _om_r = _gp_enrichment.get("outer_mind", {}).get("reward", 0)
                if _om_r > 0:
                    _oms = outer_state.get("outer_mind_15d") or []
                    for _i in range(min(15, len(_oms))):
                        _delta = (_spirit_t[70 + _i] - _oms[_i]) * _om_r
                        _oms[_i] = max(0.0, min(1.0, _oms[_i] + _delta))
                    _enrich_applied = True

                # Spirit [20:65, 85:130]: Spirit OBSERVES — NOT enriched from above

                if _enrich_applied:
                    logger.info(
                        "[FILTER_DOWN] SELF enrichment applied: "
                        "iB=%.4f iM=%.4f oB=%.4f oM=%.4f",
                        _ib_r, _im_r, _ob_r, _om_r)

                # Consume enrichment (clear after applying)
                _check_resonance._pending_enrichment = None

            # ── MSL: Snapshot collection (every ~2s = every other COMPUTATION_GATE) ──
            # Captures all modality state AFTER enrichments have run.
            if msl and _msl_tick_count % _msl_snap_interval == 0:
                try:
                    # Phase 2: Apply self-action echo to body states before snapshot
                    _msl_ib = list(body_state.get("values", [0.5] * 5))
                    _msl_ob = list(outer_state.get("outer_body", [0.5] * 5))
                    if msl.echo.is_active:
                        _echo_inner, _echo_outer = msl.get_echo_perturbation()
                        for _ei in range(min(5, len(_msl_ib))):
                            _msl_ib[_ei] = max(0.0, min(1.0,
                                _msl_ib[_ei] + float(_echo_inner[_ei])))
                        for _eo in range(min(5, len(_msl_ob))):
                            _msl_ob[_eo] = max(0.0, min(1.0,
                                _msl_ob[_eo] + float(_echo_outer[_eo])))
                    msl.collect_snapshot(
                        visual_semantic=outer_state.get("_last_visual_semantic"),
                        audio_physical=outer_state.get("_last_audio_physical"),
                        pattern_profile=outer_state.get("_last_pattern_profile"),
                        inner_body=_msl_ib,
                        inner_mind=mind_state.get("values_15d",
                                                  mind_state.get("values")),
                        outer_body=_msl_ob,
                        neuromod_levels=({n: m.level
                                         for n, m in neuromodulator_system.modulators.items()}
                                        if neuromodulator_system else None),
                        action_flag=1.0 if _t2_speak_pending else 0.0,
                        cross_modal=outer_state.get("_cross_modal_resonance", 0.0),
                        vocab_size=(len(_cached_speak_vocab)
                                    if _cached_speak_vocab else 0),
                        chi_total=getattr(life_force_engine, '_latest_chi',
                                          {}).get("total", 0.5),
                        developmental_age=(pi_monitor.developmental_age
                                           if pi_monitor else 0.0),
                        spirit_self_confidence=msl.get_i_confidence(),
                        conversation_pending=_conversation_pending is not None,
                    )
                except Exception as _msl_snap_err:
                    if _msl_tick_count % 200 == 0:
                        logger.warning("[MSL] Snapshot error: %s", _msl_snap_err)
            if msl:
                _msl_tick_count += 1
                # Periodic vocab cache refresh (every ~300 ticks = ~5 min)
                if _msl_tick_count % 300 == 0 and _msl_tick_count != _cached_speak_vocab_tick:
                    _cached_speak_vocab_tick = _msl_tick_count
                    try:
                        _cached_speak_vocab = _lp_load_vocabulary(db_path="./data/inner_memory.db")
                    except Exception:
                        pass

        # ── TIER 2: FEELING evaluation @ computation gate rate (~1.15s) ──
        # Neuromod, NS eval, Reasoning, Interpreter at practical computation rate.
        # Sphere clocks tick at true Schumann; heavy computation is gated here.
        if now - last_tier2_tick >= COMPUTATION_GATE:
            last_tier2_tick = now

            # ── MSL inference (additive enrichment) ──
            _msl_output = None
            if msl and msl.buffer.is_ready():
                try:
                    # Phase B: Neuromod coupling — modulate attention dynamics
                    # Read-only: reads neuromod levels, never writes to neuromod
                    if neuromodulator_system:
                        msl.policy.homeostatic.modulate_from_neuromod(
                            {n: m.level for n, m in
                             neuromodulator_system.modulators.items()})
                    # Phase 5: feed pi value for "I AM" event recording
                    if pi_monitor:
                        msl.set_pi_value(pi_monitor.heartbeat_ratio)
                    _msl_output = msl.tick()
                    if _msl_output:
                        outer_state["_msl_distilled_context"] = _msl_output.get("distilled_context")
                        outer_state["_msl_attention"] = _msl_output.get("attention_weights")
                        outer_state["_msl_coherence"] = _msl_output.get("coherence_pulse", 0.0)
                        # Attention heatmap logging
                        if _msl_tick_count % _msl_log_interval == 0:
                            _attn = _msl_output.get("attention_weights", {})
                            _rc = _msl_output.get("reward_components", {})
                            logger.info(
                                "[MSL] Attention: vis=%.2f aud=%.2f pat=%.2f "
                                "iBody=%.2f iMind=%.2f oBody=%.2f neuro=%.2f | "
                                "coh=%.3f r=%.3f (pred=%.2f conv=%.2f int=%.2f)",
                                _attn.get("visual", 0), _attn.get("audio", 0),
                                _attn.get("pattern", 0), _attn.get("inner_body", 0),
                                _attn.get("inner_mind", 0), _attn.get("outer_body", 0),
                                _attn.get("neuromod", 0),
                                _msl_output.get("coherence_pulse", 0),
                                _rc.get("total", 0), _rc.get("prediction", 0),
                                _rc.get("convergence", 0), _rc.get("internal", 0))
                    # Phase 2: Convergence check after MSL tick
                    if _msl_output:
                        _msl_epoch = (consciousness.get("latest_epoch") or {}).get(
                            "epoch_id", 0) if consciousness else 0
                        _msl_dreaming = _shared_is_dreaming
                        _msl_spirit_self = False  # Set True when SPIRIT_SELF fires
                        if hasattr(coordinator, '_meta_engine') and coordinator._meta_engine:
                            _me_state = getattr(coordinator._meta_engine, 'state', None)
                            if _me_state and getattr(_me_state, 'last_spirit_self_step', -1) >= 0:
                                _msl_spirit_self = True
                        # Get 132D spirit snapshot for recipe
                        _msl_spirit_snap = None
                        if unified_spirit:
                            _us_t = list(unified_spirit.tensor)  # 130D
                            _us_topo = [0.0, 0.0]
                            if hasattr(coordinator, '_last_extended_topology'):
                                _ext = getattr(coordinator, '_last_extended_topology', None)
                                if _ext and len(_ext) >= 132:
                                    _us_topo = [float(_ext[130]), float(_ext[131])]
                            import numpy as _msl_np
                            _msl_spirit_snap = _msl_np.array(
                                _us_t + _us_topo, dtype=_msl_np.float32)
                        _msl_conv = msl.check_convergence(
                            current_epoch=_msl_epoch,
                            is_dreaming=_msl_dreaming,
                            spirit_self_active=_msl_spirit_self,
                            spirit_snapshot=_msl_spirit_snap,
                        )
                        # Phase 2: "I" FILTER_DOWN perturbation (event-driven)
                        if _msl_conv and unified_spirit and msl.get_i_confidence() > 0.001:
                            _i_perturb = msl.compute_i_perturbation(
                                unified_spirit.tensor,
                                _us_topo if '_us_topo' in dir() else None)
                            if _i_perturb is not None:
                                _us_tensor = unified_spirit.tensor
                                for _ip_i in range(min(130, len(_us_tensor))):
                                    _us_tensor[_ip_i] = max(0.0, min(1.0,
                                        _us_tensor[_ip_i] + float(_i_perturb[_ip_i])))
                                logger.info("[I-RESONANCE] Perturbation applied: "
                                            "conf=%.3f mag=%.4f",
                                            msl.get_i_confidence(),
                                            float(abs(_i_perturb[:130]).mean()))
                        # Feed I-depth: emotion + concept density + wisdom (every epoch)
                        if hasattr(msl, 'i_depth'):
                            if neuromodulator_system:
                                msl.i_depth.record_emotion(
                                    getattr(neuromodulator_system, '_current_emotion', 'neutral'))
                            if msl.concept_grounder:
                                _msl_cg_confs = msl.concept_grounder.get_concept_confidences()
                                msl.i_depth.update_concept_density(_msl_cg_confs)
                                # ── META-CGN producer #2: msl.concept_grounded ──
                                # v3 Phase D rollout (rFP_meta_cgn_v3 § 12 row 2).
                                # Edge-detected: fires exactly once per (concept × first
                                # crossing of conf ≥ 0.5). 5 concepts (YOU/YES/NO/WE/THEY)
                                # → max 5 emissions per Titan lifetime under normal flow.
                                # Budget 0.001 Hz; gate at 0.1s (bounded by edge-detect).
                                if not getattr(coordinator, '_p2_msl_concepts', False):
                                    from ..logic.meta_cgn import EdgeDetector
                                    coordinator._msl_concept_detector = EdgeDetector()
                                    # Restore crossed-state so concepts don't re-fire on spirit restart
                                    _msl_persisted = _load_edge_detector_state().get("msl_concepts")
                                    if _msl_persisted:
                                        coordinator._msl_concept_detector.load_dict(_msl_persisted)
                                        logger.info(
                                            "[META-CGN] Producer #2 EdgeDetector state restored "
                                            "(%d concepts previously crossed)",
                                            sum(1 for v in _msl_persisted.get("crossed", {}).values() if v))
                                    coordinator._p2_msl_concepts = True
                                _msl_cg_det = getattr(coordinator, '_msl_concept_detector', None)
                                if _msl_cg_det:
                                    for _cg_name, _cg_conf in _msl_cg_confs.items():
                                        if _cg_det_fire := _msl_cg_det.observe(_cg_name, float(_cg_conf), 0.5):
                                            try:
                                                from ..bus import emit_meta_cgn_signal
                                                # min_interval_s=0.0 (disabled): the bus rate gate
                                                # keys by (src, consumer, event_type) — domain is
                                                # NOT in the key. A fresh EdgeDetector post-spirit-
                                                # restart can fire all 5 concepts in one MSL tick
                                                # (<1ms apart). Earlier min_interval_s=0.01 was
                                                # STILL too restrictive: 4 iterations complete in
                                                # ~1ms, so 3 of 4 were silently rate-gated — AND
                                                # the worker's BusHealthMonitor is None (only parent
                                                # process has it), so rate-gate drops are invisible
                                                # to /v4/bus-health. The EdgeDetector already bounds
                                                # this producer to 5-per-Titan-lifetime, making the
                                                # rate gate pure redundant protection — disabling
                                                # is safe. Spotted during T3 validation: 4 EMIT log
                                                # lines but bus-health showed total=1, drops=0.
                                                _p2_sent = emit_meta_cgn_signal(
                                                    send_queue,
                                                    src="msl",
                                                    consumer="msl",
                                                    event_type="concept_grounded",
                                                    intensity=min(1.0, float(_cg_conf)),
                                                    domain=_cg_name,
                                                    reason=f"MSL concept {_cg_name} crossed grounding threshold (conf={_cg_conf:.3f})",
                                                    min_interval_s=0.0,
                                                )
                                                if _p2_sent:
                                                    logger.info(
                                                        "[META-CGN] msl.concept_grounded EMIT — concept=%s conf=%.3f",
                                                        _cg_name, float(_cg_conf))
                                                else:
                                                    logger.warning(
                                                        "[META-CGN] Producer #2 msl.concept_grounded DROPPED by bus "
                                                        "— concept=%s conf=%.3f (rate-gate or queue-full; signal missed)",
                                                        _cg_name, float(_cg_conf))
                                            except Exception as _emit_err:
                                                logger.warning(
                                                    "[META-CGN] Producer #2 msl.concept_grounded emit FAILED "
                                                    "— concept=%s conf=%.3f err=%s (signal missed)",
                                                    _cg_name, float(_cg_conf), _emit_err)
                            if meta_engine:
                                msl.i_depth.record_wisdom(
                                    getattr(meta_engine, '_total_wisdom_saved', 0))
                                msl.i_depth.record_eureka(
                                    getattr(meta_engine, '_total_eurekas', 0))
                        # Store "I" confidence in outer_state for downstream
                        outer_state["_msl_i_confidence"] = msl.get_i_confidence()
                        outer_state["_msl_chi_coherence"] = msl.chi_tracker.chi

                        # Phase 5: Check for "I AM" event (fired inside check_convergence)
                        _iam_event = msl.get_iam_event()
                        if _iam_event:
                            logger.info("[I_AM] ══ EVENT #%d ══ pi=%.4f chi=%.3f "
                                        "I=%.3f sustained=%d",
                                        _iam_event["event_number"],
                                        _iam_event["pi_value"],
                                        _iam_event["chi_coherence"],
                                        _iam_event["i_confidence"],
                                        _iam_event["sustained_epochs"])
                            # SocialPressure catalyst: significance 1.0 (highest)
                            if social_pressure_meter:
                                social_pressure_meter.add_catalyst(
                                    "i_am_event", significance=1.0)
                            # Experience marker for episodic memory
                            outer_state["_iam_event"] = _iam_event
                            # TimeChain: "I AM" event → meta fork (highest significance)
                            _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
                                "fork": "meta",
                                "thought_type": "meta",
                                "source": "msl_i_am",
                                "content": {
                                    "event": "I_AM",
                                    "event_number": _iam_event.get("event_number", 0),
                                    "pi_value": _iam_event.get("pi_value", 0),
                                    "chi_coherence": _iam_event.get("chi_coherence", 0),
                                    "i_confidence": _iam_event.get("i_confidence", 0),
                                    "sustained_epochs": _iam_event.get("sustained_epochs", 0),
                                },
                                "significance": 1.0,
                                "novelty": 0.9,
                                "coherence": 0.8,
                                "tags": ["I_AM", "consciousness", "milestone"],
                                "neuromods": dict(_cached_neuromod_state),
                                "chi_available": _cached_chi_state.get("total", 0.5),
                                "metabolic_drain": getattr(life_force_engine, '_metabolic_drain', 0.0) if life_force_engine else 0.0,
                                "attention": outer_state.get("_msl_chi_coherence", 0.5),
                                "i_confidence": _iam_event.get("i_confidence", 0.5),
                                "chi_coherence": _iam_event.get("chi_coherence", 0.3),
                                "pi_curvature": (consciousness.get("latest_epoch") or {}).get("curvature", 1.0) if consciousness else 1.0,
                                "epoch_id": (consciousness.get("latest_epoch") or {}).get("epoch_id", 0) if consciousness else 0,
                            })

                        # Phase 3: Concept FILTER_DOWN perturbations (all grounded concepts)
                        if msl.concept_grounder and unified_spirit:
                            _us_tensor_p3 = unified_spirit.tensor
                            _us_topo_p3 = _us_topo if '_us_topo' in dir() else [0.0, 0.0]
                            import numpy as _p3_np
                            _p3_spirit_132d = _p3_np.array(
                                list(_us_tensor_p3[:130]) + list(_us_topo_p3[:2]),
                                dtype=_p3_np.float32)
                            for _p3c_name in ["YOU", "YES", "NO", "WE", "THEY"]:
                                _p3c_pert = msl.concept_grounder.compute_perturbation(
                                    _p3c_name, _p3_spirit_132d)
                                if _p3c_pert is not None:
                                    for _p3c_i in range(min(130, len(_us_tensor_p3))):
                                        _us_tensor_p3[_p3c_i] = max(0.0, min(1.0,
                                            _us_tensor_p3[_p3c_i] + float(_p3c_pert[_p3c_i])))
                            # Store concept confidences in outer_state for composition
                            outer_state["_msl_concept_confidences"] = (
                                msl.concept_grounder.get_concept_confidences())

                        # Phase 3: Concept emotional valence (neuromod nudges)
                        if msl.concept_grounder and neuromodulator_system:
                            _p3_dev_age = pi_monitor.developmental_age if pi_monitor else 0.0
                            for _p3_cname, _p3_conf in msl.concept_grounder.get_concept_confidences().items():
                                if _p3_conf > 0.01:
                                    _p3_val = msl.concept_grounder.get_emotional_valence(_p3_cname)
                                    if _p3_val:
                                        neuromodulator_system.apply_external_nudge(
                                            _p3_val["nudge_map"],
                                            max_delta=_p3_val["max_delta"] * min(1.0, _p3_conf),
                                            developmental_age=_p3_dev_age)
                            # "I" valence (Phase 2 concept, Phase 3 wiring)
                            _p3_i_conf = msl.get_i_confidence()
                            if _p3_i_conf > 0.01:
                                from titan_plugin.logic.msl import ConceptGrounder
                                _p3_i_val = ConceptGrounder.CONCEPT_VALENCE.get("I")
                                if _p3_i_val:
                                    neuromodulator_system.apply_external_nudge(
                                        _p3_i_val["nudge_map"],
                                        max_delta=_p3_i_val["max_delta"] * min(1.0, _p3_i_conf),
                                        developmental_age=_p3_dev_age)
                except Exception as _msl_tick_err:
                    if _msl_tick_count % 200 == 0:
                        logger.warning("[MSL] Tick error: %s", _msl_tick_err)

            try:
                # ── Neuromodulator evaluation (Tier 2) — EMERGENT INPUTS ──
                # All inputs derived from architectural subsystems, no hardcoded proxies.
                # DNA weights from titan_params.toml [neuromodulator_dna] section.
                if neuromodulator_system:
                    _t2_latest = (consciousness.get("latest_epoch") or {}) if consciousness else {}
                    _t2_dreaming = False
                    if coordinator and hasattr(coordinator, 'dreaming') and coordinator.dreaming:
                        _t2_dreaming = getattr(coordinator.dreaming, 'is_dreaming', False)

                    # Load DNA weights (cached on neuromod system after first load)
                    _nm_dna = getattr(neuromodulator_system, '_dna_cache', None)
                    if _nm_dna is None:
                        _nm_dna = _oi_params.get("neuromodulator_dna", {})
                        neuromodulator_system._dna_cache = _nm_dna

                    # ── Sphere clock balance (5-HT source) ──
                    _nm_sphere_bal = {}
                    _bal_norm = _nm_dna.get("balance_streak_normalization", 100)
                    if sphere_clock:
                        for _sc_name in ["inner_body", "inner_mind", "outer_body", "outer_mind"]:
                            _sc = sphere_clock.clocks.get(_sc_name)
                            _nm_sphere_bal[_sc_name] = min(1.0, _sc._consecutive_balanced / _bal_norm) if _sc else 0.0
                    else:
                        _nm_sphere_bal = {k: 0.0 for k in ["inner_body", "inner_mind", "outer_body", "outer_mind"]}

                    # ── META-CGN producer #1: sphere_clock.balance_held ──
                    # v3 Phase D rollout (rFP_meta_cgn_v3 § 5 — first producer,
                    # chosen for cleanest edge-detection + bounded 16 max
                    # emissions ever per titan).
                    # Fires exactly once per (sphere × milestone) crossing.
                    # Per-sphere EdgeDetector.observe_new_max tracks highest
                    # milestone reached so re-entry doesn't double-emit.
                    if sphere_clock and not getattr(coordinator, '_p2_sphere_milestones', None):
                        # Initialise once: EdgeDetector per-sphere max tracker.
                        # Load persisted state so milestone crossings survive restarts
                        # (preserves 16-per-Titan-lifetime rFP budget).
                        from ..logic.meta_cgn import EdgeDetector
                        coordinator._sc_balance_detector = EdgeDetector()
                        _sc_persisted = _load_edge_detector_state().get("sc_balance")
                        if _sc_persisted:
                            coordinator._sc_balance_detector.load_dict(_sc_persisted)
                            logger.info(
                                "[META-CGN] Producer #1 EdgeDetector state restored "
                                "(%d max entries)", len(_sc_persisted.get("max", {})))
                        coordinator._p2_sphere_milestones = True  # marker
                    if sphere_clock and getattr(coordinator, '_sc_balance_detector', None):
                        _sc_milestones = [100, 500, 1000, 5000]
                        for _sc_name in ["inner_body", "inner_mind", "outer_body", "outer_mind"]:
                            _sc = sphere_clock.clocks.get(_sc_name)
                            if not _sc:
                                continue
                            _streak = int(getattr(_sc, '_consecutive_balanced', 0))
                            # Highest milestone crossed at or below current streak
                            _new_milestone = max(
                                (m for m in _sc_milestones if m <= _streak),
                                default=0,
                            )
                            if _new_milestone > 0 and \
                                    coordinator._sc_balance_detector.observe_new_max(
                                        _sc_name, _new_milestone):
                                try:
                                    from ..bus import emit_meta_cgn_signal
                                    # min_interval_s=0.0: same reasoning as Producer #2 — worker-side
                                    # rate-gate drops are invisible to bus-health (monitor is None
                                    # in subprocess), and EdgeDetector.observe_new_max already
                                    # bounds this producer to 16-per-Titan-lifetime (4 spheres × 4
                                    # milestones). Tight gate is redundant.
                                    _p1_sent = emit_meta_cgn_signal(
                                        send_queue,
                                        src="sphere_clock",
                                        consumer="sphere_clock",
                                        event_type="balance_held",
                                        intensity=min(1.0, _new_milestone / 5000.0),
                                        domain=_sc_name,
                                        reason=f"milestone {_new_milestone} crossed on {_sc_name}",
                                        min_interval_s=0.0,
                                    )
                                    if not _p1_sent:
                                        logger.warning(
                                            "[META-CGN] Producer #1 sphere_clock.balance_held DROPPED by bus "
                                            "— sphere=%s milestone=%d (rate-gate or queue-full; signal missed)",
                                            _sc_name, _new_milestone)
                                except Exception as _emit_err:
                                    logger.warning(
                                        "[META-CGN] Producer #1 sphere_clock.balance_held emit FAILED "
                                        "— sphere=%s milestone=%d err=%s (signal missed)",
                                        _sc_name, _new_milestone, _emit_err)

                    # ── Trinity coherence (NE source) ──
                    _nm_trinity = {"inner": 0.5, "outer": 0.5}
                    if _t2_latest:
                        import math as _nm_math
                        for _tri_key, _tri_dims in [("inner", ["iB", "iM", "iS"]), ("outer", ["oB", "oM", "oS"])]:
                            _tri_vals = [_t2_latest.get(d, 0.5) for d in _tri_dims]
                            _tri_mean = sum(_tri_vals) / 3.0
                            _tri_std = (_nm_math.sqrt(sum((_v - _tri_mean) ** 2 for _v in _tri_vals) / 3.0))
                            _nm_trinity[_tri_key] = max(0.0, 1.0 - _tri_std / 0.4)

                    # ── Chi state (NE/5-HT/GABA/Endorphin source) ──
                    _nm_prev_chi = getattr(life_force_engine, '_prev', {}) if life_force_engine else {}
                    _nm_chi_vals = [_nm_prev_chi.get(k, 0.5) for k in ["spirit", "mind", "body"]]
                    _nm_chi_mean = sum(_nm_chi_vals) / 3.0
                    _nm_chi_var = sum((v - _nm_chi_mean) ** 2 for v in _nm_chi_vals) / 3.0
                    _nm_chi = {
                        "total": _nm_chi_mean,
                        "body": _nm_prev_chi.get("body", 0.5),
                        "circulation": max(0.0, 1.0 - _nm_chi_var / 0.08),
                        "drain": getattr(life_force_engine, '_metabolic_drain', 0.0) if life_force_engine else 0.0,
                    }

                    # ── Consciousness dynamics (NE/ACh/GABA source) ──
                    _nm_drift = _t2_latest.get("drift_magnitude", 0.0)
                    _nm_prev_drift = getattr(neuromodulator_system, '_prev_drift', _nm_drift)
                    _nm_drift_delta = min(1.0, abs(_nm_drift - _nm_prev_drift) * 100)  # scale small deltas
                    neuromodulator_system._prev_drift = _nm_drift
                    _nm_dynamics = {
                        "drift_magnitude": _nm_drift,
                        "drift_delta": _nm_drift_delta,
                        "density": _t2_latest.get("density", 0.0),
                        "epoch_gap_ratio": 0.5,  # updated below
                    }
                    # Epoch gap ratio: how close are epochs firing to MIN_GAP?
                    _nm_last_epoch_time = getattr(neuromodulator_system, '_last_epoch_time', now)
                    _nm_epoch_gap = max(1.0, now - _nm_last_epoch_time)
                    if _t2_latest.get("epoch_id") != getattr(neuromodulator_system, '_last_epoch_id', None):
                        neuromodulator_system._last_epoch_time = now
                        neuromodulator_system._last_epoch_id = _t2_latest.get("epoch_id")
                    _nm_dynamics["epoch_gap_ratio"] = min(1.0, 8.0 / max(1.0, _nm_epoch_gap))

                    # ── Pi state (5-HT/Endorphin source) ──
                    # Epoch-interval regularity (Schumann-derived) replaces
                    # pi_heartbeat_ratio which declines with maturity.
                    # EMA of interval tracks Titan's natural rhythm;
                    # regularity = how close each tick is to that rhythm.
                    _nm_ema_interval = getattr(
                        neuromodulator_system, '_ema_epoch_interval', 7.5)
                    _nm_ema_regularity = getattr(
                        neuromodulator_system, '_ema_epoch_regularity', 0.5)
                    _nm_new_epoch = (
                        _t2_latest.get("epoch_id")
                        != getattr(neuromodulator_system, '_last_epoch_id', None)
                    )
                    if _nm_new_epoch and _nm_epoch_gap < 120:  # skip restart outliers
                        # Update interval EMA (α=0.02 ≈ 50-epoch memory)
                        _nm_ema_interval = _nm_ema_interval * 0.98 + _nm_epoch_gap * 0.02
                        # Instant regularity: deviation from expected rhythm
                        _nm_dev = abs(_nm_epoch_gap - _nm_ema_interval) / max(
                            _nm_ema_interval, 1.0)
                        _nm_instant_reg = max(0.0, 1.0 - _nm_dev)
                        # Smooth regularity EMA (α=0.05 ≈ 20-epoch memory)
                        _nm_ema_regularity = (
                            _nm_ema_regularity * 0.95 + _nm_instant_reg * 0.05)
                    neuromodulator_system._ema_epoch_interval = _nm_ema_interval
                    neuromodulator_system._ema_epoch_regularity = _nm_ema_regularity

                    _nm_prev_curv = getattr(neuromodulator_system, '_prev_curvature', 0.0)
                    _nm_curv_now = _t2_latest.get("curvature", 0.0)
                    _nm_curv_delta = min(1.0, abs(_nm_curv_now - _nm_prev_curv))
                    neuromodulator_system._prev_curvature = _nm_curv_now

                    # Epoch-based maturity (replaces developmental_age/50
                    # which stalls as π-clusters become rare)
                    _nm_epoch_id = _t2_latest.get("epoch_id", 0) or 0
                    _nm_epoch_maturity = min(1.0, _nm_epoch_id / 50000.0)

                    _nm_pi_reg_raw = pi_monitor.heartbeat_ratio if pi_monitor else 0.0
                    _nm_pi = {
                        "regularity": _nm_pi_reg_raw,
                        "epoch_regularity": _nm_ema_regularity,
                        "cluster_streak": min(1.0, (pi_monitor._current_pi_streak if pi_monitor else 0) / 20.0),
                        "developmental_age": pi_monitor.developmental_age if pi_monitor else 0,
                        "curvature_delta": _nm_curv_delta,
                        "epoch_maturity": _nm_epoch_maturity,
                    }

                    # ── Prediction state (DA source) ──
                    _nm_surprise = prediction_engine.get_novelty_signal() if prediction_engine else 0.0
                    _nm_action_outcome = 0.5
                    _nm_success_rate = 0.5
                    if ex_mem:
                        try:
                            _nm_stats = ex_mem.get_stats()
                            _nm_types = _nm_stats.get("by_type", {})
                            if _nm_types:
                                _nm_scores = [t.get("avg_score", 0.5) for t in _nm_types.values() if t.get("avg_score") is not None]
                                if _nm_scores:
                                    _nm_action_outcome = sum(_nm_scores) / len(_nm_scores)
                                _nm_srs = [t.get("success_rate", 0.5) for t in _nm_types.values() if t.get("success_rate") is not None]
                                if _nm_srs:
                                    _nm_success_rate = sum(_nm_srs) / len(_nm_srs)
                        except Exception:
                            pass
                    _nm_pred = {"surprise": _nm_surprise, "action_outcome": _nm_action_outcome, "success_rate": _nm_success_rate}

                    # ── NS state (ACh source) ──
                    _nm_ns_trans = 0.0
                    _nm_fd_writes = 0.0
                    if neural_nervous_system:
                        _nm_trans_now = neural_nervous_system._total_transitions
                        _nm_trans_prev = getattr(neuromodulator_system, '_prev_ns_transitions', _nm_trans_now)
                        _nm_ns_trans = min(1.0, max(0, _nm_trans_now - _nm_trans_prev) / 20.0)
                        neuromodulator_system._prev_ns_transitions = _nm_trans_now
                    _nm_fd_count = getattr(neuromodulator_system, '_filter_down_count', 0)
                    _nm_fd_prev = getattr(neuromodulator_system, '_prev_fd_count', _nm_fd_count)
                    _nm_fd_writes = min(1.0, max(0, _nm_fd_count - _nm_fd_prev) / 3.0)
                    neuromodulator_system._prev_fd_count = _nm_fd_count
                    _nm_ns = {"transition_delta": _nm_ns_trans, "filter_down_writes": _nm_fd_writes}

                    # ── Expression state (GABA/Endorphin source) ──
                    _nm_fire_rate = 0.0
                    _nm_alignment = 0.5
                    if expression_manager:
                        try:
                            _nm_expr = expression_manager.get_stats()
                            _nm_fc = [c.get("fire_count", 0) for c in _nm_expr.values() if isinstance(c, dict)]
                            _nm_ec = [c.get("evaluation_count", 1) for c in _nm_expr.values() if isinstance(c, dict)]
                            if _nm_ec and sum(_nm_ec) > 0:
                                _nm_alignment = sum(_nm_fc) / sum(_nm_ec)
                                # Fire rate: fires per evaluation, normalized to 0-1
                                _nm_fire_rate = _nm_alignment  # fire/eval ratio IS the fire rate
                        except Exception:
                            pass
                    _nm_expr_state = {"fire_rate": min(1.0, _nm_fire_rate), "alignment": _nm_alignment}

                    # ── Resonance state (Endorphin source) ──
                    _nm_res = {"resonant_fraction": 0.0}
                    if resonance:
                        _nm_res["resonant_fraction"] = resonance.resonant_count() / 3.0

                    # ── Compute emergent inputs ──
                    _t2_inputs = compute_emergent_inputs(
                        sphere_balance=_nm_sphere_bal,
                        trinity_coherence=_nm_trinity,
                        chi_state=_nm_chi,
                        consciousness_dynamics=_nm_dynamics,
                        pi_state=_nm_pi,
                        prediction_state=_nm_pred,
                        ns_state=_nm_ns,
                        expression_state=_nm_expr_state,
                        resonance_state=_nm_res,
                        is_dreaming=_t2_dreaming,
                        dna=_nm_dna,
                    )

                    # ── Kin resonance → neuromodulator boost (DA, Endorphin, 5-HT, NE) ──
                    _kin_dna = _oi_params.get("kin", {}).get("dna", {})
                    _kin_res_val = _kin_state.get("last_resonance", 0.0)
                    _kin_recency = max(0.0, 1.0 - (time.time() - _kin_state.get("last_exchange_ts", 0)) / 3600.0)
                    _kin_signal = _kin_res_val * _kin_recency  # Decays over 1 hour
                    if _kin_signal > 0.01:
                        _t2_inputs["kin_da"] = _kin_signal * _kin_dna.get("da_boost", 0.25)
                        _t2_inputs["kin_endorphin"] = _kin_signal * _kin_dna.get("endorphin_boost", 0.20)
                        _t2_inputs["kin_5ht"] = _kin_signal * _kin_dna.get("sht_boost", 0.15)
                        _t2_inputs["kin_ne"] = _kin_signal * _kin_dna.get("ne_boost", 0.10)

                    # Metabolic gating: pass Chi health to neuromod production
                    if life_force_engine:
                        _drain = getattr(life_force_engine, '_metabolic_drain', 0.0)
                        neuromodulator_system.set_chi_health(max(0.1, 1.0 - _drain * 0.6))

                    _t2_result = neuromodulator_system.evaluate(_t2_inputs, dt=1.0)

                    # ── Movement excess clearance (topology-modulated) ──
                    # Per-modulator: movement clears excess above setpoint
                    _nm_topo_vel = getattr(neuromodulator_system, '_topology_velocity', 0.3)
                    for _nm_name, _nm_mod in neuromodulator_system.modulators.items():
                        _nm_mov_rate = _nm_dna.get(f"movement_{_nm_name}", 0.0)
                        if _nm_mov_rate > 0:
                            apply_movement_excess_clearance(_nm_mod, _nm_topo_vel, _nm_mov_rate)

                    # Apply modulation to Neural NS
                    if neural_nervous_system:
                        neural_nervous_system._modulation = neuromodulator_system.get_modulation()
                        # Self-emergent hormonal governors: inject GABA + Chi into modulation dict
                        neural_nervous_system._modulation["gaba_level"] = neuromodulator_system.modulators["GABA"].level
                        neural_nervous_system._modulation["chi_total"] = getattr(
                            life_force_engine, '_latest_chi', {}).get("total", 0.6) if life_force_engine else 0.6

                    # Cache neuromod state for digital layer
                    _cached_neuromod_state = {
                        n: m.level for n, m in neuromodulator_system.modulators.items()
                    }

                    # TimeChain: periodic heartbeat on main chain (every 100 epochs)
                    _tc_epoch_id = _t2_latest.get("epoch_id", 0) if consciousness else 0
                    _tc_last_hb = getattr(neuromodulator_system, '_tc_last_heartbeat_epoch', 0)
                    if _tc_epoch_id > 0 and (_tc_epoch_id - _tc_last_hb) >= 100:
                        neuromodulator_system._tc_last_heartbeat_epoch = _tc_epoch_id
                        _tc_chi = getattr(life_force_engine, '_latest_chi', {}) if life_force_engine else {}
                        _send_msg(send_queue, "EPOCH_TICK", name, "timechain", {
                            "epoch_id": _tc_epoch_id,
                            "chi_total": _tc_chi.get("total", 0),
                            "emotion": getattr(neuromodulator_system, '_current_emotion', 'neutral'),
                            "is_dreaming": _shared_is_dreaming,
                            "neuromods": dict(_cached_neuromod_state),
                        })

                    # Observatory V2: WebSocket event for neuromod state
                    _send_msg(send_queue, "NEUROMOD_UPDATE", name, "v4_bridge", {
                        "modulators": {
                            _nm: {"level": round(_mm.level, 4), "tonic": round(_mm.tonic_level, 4)}
                            for _nm, _mm in neuromodulator_system.modulators.items()
                        },
                        "emotion": getattr(neuromodulator_system, 'current_emotion', ''),
                    })

                # ── Wallet Observer poll (Tier 2 — DI:/I:/Donation detection) ──
                if _wallet_observer and _wallet_observer.should_poll():
                    try:
                        import asyncio as _wo_asyncio
                        _wo_loop = _wo_asyncio.new_event_loop()
                        _wo_memos = _wo_loop.run_until_complete(_wallet_observer.poll())
                        _wo_loop.close()
                        for _wo_memo in _wo_memos:
                            _wo_boost = _wo_memo.get_neuromod_boost()
                            if _wo_boost and neuromodulator_system:
                                # Apply neuromod boosts from DI:/I:/Donation
                                for _wo_key in ("DA", "NE", "Endorphin", "EMPATHY"):
                                    if _wo_key in _wo_boost:
                                        _wo_mod = neuromodulator_system.modulators.get(_wo_key)
                                        if _wo_mod:
                                            _wo_mod.level = min(1.0, _wo_mod.level + _wo_boost[_wo_key])
                                # Apply hormone boost
                                _wo_hormone = _wo_boost.get("hormone")
                                _wo_hdelta = _wo_boost.get("hormone_delta", 0)
                                if _wo_hormone and _wo_hdelta > 0 and neural_nervous_system:
                                    _wo_prog = neural_nervous_system.programs.get(_wo_hormone)
                                    if _wo_prog:
                                        _wo_prog["urgency"] = min(2.0, _wo_prog.get("urgency", 0) + _wo_hdelta)
                                # Apply chi boost
                                if "chi_boost" in _wo_boost and life_force_engine:
                                    life_force_engine._metabolic_drain = max(
                                        0, life_force_engine._metabolic_drain - _wo_boost["chi_boost"])
                                # Dream interrupt for DI:URGENT
                                if _wo_boost.get("interrupt_dream") and _shared_is_dreaming:
                                    _send_msg(send_queue, "DREAM_WAKE_REQUEST", name, "spirit",
                                              {"reason": f"DI:URGENT from {_wo_memo.sender[:12]}",
                                               "user_id": _wo_memo.sender})
                                # Memory anchoring for DI: messages
                                if _wo_boost.get("anchor_memory") and _wo_memo.content:
                                    _send_msg(send_queue, "INTERFACE_INPUT", name, "all", {
                                        "source": "wallet_di",
                                        "user_id": _wo_memo.sender,
                                        "text": _wo_memo.content,
                                        "memo_type": _wo_memo.memo_type,
                                        "is_maker": _wo_memo.is_maker,
                                        "sol_amount": _wo_memo.sol_amount,
                                        "anchor_bonus": 1.0,
                                    })
                                logger.info(
                                    "[WalletIntent] %s from %s (%.4f SOL, mult=%.1fx): %s → %s",
                                    _wo_memo.memo_type, _wo_memo.sender[:12],
                                    _wo_memo.sol_amount, _wo_boost.get("sol_multiplier", 1.0),
                                    _wo_memo.content[:40] if _wo_memo.content else "donation",
                                    {k: f"{v:.3f}" for k, v in _wo_boost.items() if isinstance(v, float)})
                    except Exception as _wo_err:
                        logger.warning("[WalletObserver] Poll error: %s", _wo_err)

                # ── Neural NS evaluation (Tier 2 — programs + hormonal fire) ──
                if neural_nervous_system:
                    try:
                        # Update observation space with latest state (all 6 body parts)
                        _nn_obs = inner_state.observables if inner_state else {}

                        # Build Tier 2 context: clocks, topology, resonance, spirit
                        _nn_clocks = {}
                        if sphere_clock:
                            _nn_clocks = {
                                c.name: {"phase": c.phase, "radius": c.radius,
                                         "velocity": c.contraction_velocity,
                                         "pulse_count": c.pulse_count}
                                for c in sphere_clock.clocks.values()
                            }
                        _nn_topo = {}
                        if coordinator and hasattr(coordinator, '_last_topology'):
                            _nn_topo = coordinator._last_topology
                        _nn_resonance_data = {}
                        if resonance:
                            _nn_resonance_data = {
                                "resonant_count": resonance.resonant_count,
                                "all_resonant": resonance.all_resonant,
                                "great_pulse_count": resonance._great_pulse_count,
                            }
                        _nn_us = {}
                        if unified_spirit:
                            _nn_us = {
                                "velocity": getattr(unified_spirit, 'velocity', 1.0),
                                "epoch_count": getattr(unified_spirit, 'epoch_count', 0),
                            }
                            if hasattr(unified_spirit, 'latest') and unified_spirit.latest:
                                _nn_us["magnitude"] = getattr(unified_spirit.latest, 'magnitude', 0.0)
                                _nn_us["quality"] = getattr(unified_spirit.latest, 'cumulative_quality', 0.0)
                        _nn_consciousness = {}
                        if consciousness:
                            _le = consciousness.get("latest_epoch") or {}
                            _nn_consciousness = {
                                "drift_magnitude": _le.get("drift_magnitude", 0.0),
                                "trajectory_magnitude": _le.get("trajectory_magnitude", 0.0),
                                "state_vector": _le.get("state_vector", [0.5] * 9),
                            }
                        _nn_dreaming = {}
                        if coordinator and hasattr(coordinator, 'dreaming') and coordinator.dreaming:
                            _nn_dreaming = {
                                "fatigue": getattr(coordinator.dreaming, 'last_fatigue', 0.0),
                                "readiness": getattr(coordinator.dreaming, 'last_readiness', 0.0),
                            }

                        # ── Tier 3: FilterDown + Focus data ──
                        _nn_fd_mults = {}
                        if filter_down:
                            _nn_fd_mults = {
                                "body": list(getattr(filter_down, '_body_multipliers', [1.0] * 5)),
                                "mind": list(getattr(filter_down, '_mind_multipliers', [1.0] * 5)),
                            }
                        _nn_focus_nudges = {}
                        if focus_body and focus_mind:
                            _nn_focus_nudges = {
                                "body": list(getattr(focus_body, '_nudges', [0.0] * 5)),
                                "mind": list(getattr(focus_mind, '_nudges', [0.0] * 5)),
                            }

                        # ── Tier 4: Impulse state ──
                        _nn_impulse = {}
                        if impulse_engine:
                            _nn_impulse = {"urgency": getattr(impulse_engine, '_last_urgency', 0.0)}

                        # ── Tier 5: Neurochemical state ──
                        _nn_neuromod_levels = {}
                        _nn_neuromod_setpoints = {}
                        if neuromodulator_system:
                            for _nm_n, _nm_m in neuromodulator_system.modulators.items():
                                _nn_neuromod_levels[_nm_n] = _nm_m.level
                                _nn_neuromod_setpoints[_nm_n] = _nm_m.setpoint
                        _nn_chi = {}
                        if life_force_engine:
                            _raw_chi = getattr(life_force_engine, '_latest_chi', {})
                            if _raw_chi:
                                _nn_chi = {
                                    "total": _raw_chi.get("total", 0.5),
                                    "circulation": _raw_chi.get("circulation", 0.5),
                                    "body": _raw_chi.get("body", {}).get("effective", _raw_chi.get("body", 0.5)) if isinstance(_raw_chi.get("body"), dict) else float(_raw_chi.get("body", 0.5)),
                                    "mind": _raw_chi.get("mind", {}).get("effective", _raw_chi.get("mind", 0.5)) if isinstance(_raw_chi.get("mind"), dict) else float(_raw_chi.get("mind", 0.5)),
                                    "spirit": _raw_chi.get("spirit", {}).get("effective", _raw_chi.get("spirit", 0.5)) if isinstance(_raw_chi.get("spirit"), dict) else float(_raw_chi.get("spirit", 0.5)),
                                }

                        # ── Tier 6: System dynamics ──
                        _nn_drain = getattr(life_force_engine, '_metabolic_drain', 0.0) if life_force_engine else 0.0
                        _nn_sd = getattr(coordinator.dreaming, 'last_sleep_drive', 0.0) if coordinator and hasattr(coordinator, 'dreaming') and coordinator.dreaming else 0.0
                        _nn_wd = getattr(coordinator.dreaming, 'last_wake_drive', 0.0) if coordinator and hasattr(coordinator, 'dreaming') and coordinator.dreaming else 0.0
                        _nn_exp_p = 0.0
                        _nn_exp_rep = 0.0
                        if coordinator and hasattr(coordinator, 'dreaming') and coordinator.dreaming:
                            _fb = getattr(coordinator.dreaming, '_last_fatigue_breakdown', {})
                            _nn_exp_p = _fb.get("o4_exp", 0.0)
                            _nn_exp_rep = _fb.get("o5_rep", 0.0)
                        _nn_tsd = 0.0
                        if coordinator and hasattr(coordinator, 'dreaming') and coordinator.dreaming:
                            _nn_tsd = float(getattr(coordinator.dreaming, '_epochs_since_dream', 0)) * 7.0

                        # Gather reasoning features for T6 extension
                        _nn_reasoning = {}
                        if _reasoning_engine:
                            _nn_reasoning = _reasoning_engine.get_observation_features()
                        neural_nervous_system.update_observation_space(
                            observables=_nn_obs,
                            sphere_clocks=_nn_clocks,
                            topology=_nn_topo,
                            resonance=_nn_resonance_data,
                            unified_spirit=_nn_us,
                            consciousness=_nn_consciousness,
                            dreaming=_nn_dreaming,
                            filter_down_mults=_nn_fd_mults,
                            focus_nudges=_nn_focus_nudges,
                            impulse_state=_nn_impulse,
                            neuromodulator_levels=_nn_neuromod_levels,
                            neuromodulator_setpoints=_nn_neuromod_setpoints,
                            chi_state=_nn_chi,
                            metabolic_drain=_nn_drain,
                            sleep_drive=_nn_sd,
                            wake_drive=_nn_wd,
                            experience_pressure=_nn_exp_p,
                            expression_repetitiveness=_nn_exp_rep,
                            time_since_dream=_nn_tsd,
                            reasoning_active=_nn_reasoning.get("is_active", 0.0),
                            reasoning_chain_length=_nn_reasoning.get("chain_length_norm", 0.0),
                            reasoning_confidence=_nn_reasoning.get("confidence", 0.0),
                            reasoning_gut_agreement=_nn_reasoning.get("gut_agreement", 0.0),
                        )
                        # ── Wire maturity signals from Titan's emergent time ──
                        # Sources: sphere clock radii (lines 728-734), resonance
                        # great_pulse_count (line 743), consciousness epoch_id.
                        # These drive hormonal refractory decay rate via maturity.
                        _mat_great = _nn_resonance_data.get("great_pulse_count", 0)
                        _mat_radius = 1.0
                        if _nn_clocks:
                            _mat_radii = [v.get("radius", 1.0)
                                          for k, v in _nn_clocks.items()
                                          if k.startswith("inner_")]
                            if _mat_radii:
                                _mat_radius = sum(_mat_radii) / len(_mat_radii)
                        _mat_epochs = 0
                        if consciousness:
                            _mat_epochs = (consciousness.get("latest_epoch") or {}).get("epoch_id", 0)
                        neural_nervous_system.update_maturity_signals(
                            great_epochs=_mat_great,
                            sphere_radius=_mat_radius,
                            consciousness_epochs=_mat_epochs,
                        )

                        # Evaluate all programs — records transitions + trains
                        # NOTE: temporal features skipped — programs are 55D (standard),
                        # appending 5D temporal would make 60D which mismatches weights.
                        # Temporal features require retraining from scratch (rFP).
                        _nn_signals = neural_nervous_system.evaluate(_nn_obs, temporal=None)
                        _nn_new_trans = neural_nervous_system._total_transitions
                        if now - _last_log_ns > _log_ns_interval:
                            _last_log_ns = now
                            logger.info("[SpiritWorker] NeuralNS alive: transitions=%d, signals=%d",
                                        _nn_new_trans, len(_nn_signals))
                        # Cache NS urgencies for digital layer
                        _cached_ns_urgencies = dict(getattr(neural_nervous_system, '_all_urgencies', {}))

                        # Feed signals to coordinator for get_stats() / API exposure
                        if coordinator:
                            coordinator._last_nervous_signals = _nn_signals
                        # Observatory V2: WebSocket events for hormone fires
                        for _ns_sig in _nn_signals:
                            if _ns_sig.get("hormonal"):
                                _send_msg(send_queue, "HORMONE_FIRED", name, "v4_bridge", {
                                    "program": _ns_sig["system"],
                                    "intensity": round(_ns_sig.get("intensity", 0), 3),
                                    "urgency": round(_ns_sig.get("urgency", 0), 3),
                                })
                    except Exception as e:
                        logger.warning("[SpiritWorker] NeuralNS evaluate error: %s", e, exc_info=True)

                # ── Reasoning Engine tick (Mind's deliberate cognition) ──
                _reasoning_result = None
                _is_dreaming_r = getattr(
                    getattr(coordinator, 'dreaming', None), 'is_dreaming',
                    getattr(getattr(coordinator, 'inner', None), 'is_dreaming', False)
                ) if coordinator else False
                if _reasoning_engine and not _is_dreaming_r:
                    try:
                        # Gather gut signals from ALL NS program urgencies (not just fired)
                        # _all_urgencies has raw NN output for every program
                        _r_gut = {}
                        if neural_nervous_system and hasattr(neural_nervous_system, '_all_urgencies'):
                            # rFP β Phase 3: hormone-augmented urgencies for gut formula.
                            # Blend NN urgency (current state) with hormone pressure
                            # (recent history) — more robust when NN is wonky.
                            try:
                                _r_gut = neural_nervous_system.get_augmented_urgencies(
                                    hormone_blend=0.3)
                            except Exception:
                                _r_gut = dict(neural_nervous_system._all_urgencies)
                        else:
                            # Fallback: use only fired signals
                            _r_last_signals = getattr(coordinator, '_last_nervous_signals', []) if coordinator else []
                            for _rs in _r_last_signals:
                                _r_gut[_rs.get("system", "")] = _rs.get("urgency", 0.0)

                        # Gather body state
                        _r_body = {
                            "fatigue": _fatigue if '_fatigue' in dir() else 0.3,
                            "chi_total": float((_raw_chi or {}).get("total", 0.5)) if '_raw_chi' in dir() and _raw_chi else 0.5,
                            "metabolic_drain": float(getattr(life_force_engine, '_metabolic_drain', 0.0)) if life_force_engine else 0.0,
                            "is_dreaming": _is_dreaming_r,
                        }

                        # Raw neuromods for cognitive EMA
                        _r_neuromods = {}
                        if neuromodulator_system:
                            for _rn, _rm in neuromodulator_system.modulators.items():
                                _r_neuromods[_rn] = _rm.level

                        # Working memory context
                        _r_wm_items = working_mem.get_context() if working_mem else []

                        # Build observation from NS observation space
                        if neural_nervous_system and hasattr(neural_nervous_system, '_observation_space'):
                            _r_obs = neural_nervous_system._observation_space.build_input("enriched")
                        else:
                            import numpy as _np_r
                            _r_obs = _np_r.zeros(79)

                        # Tick mini-reasoners before main reasoning (distributed intelligence)
                        if _mini_registry:
                            _mini_ctx = {
                                "observables": getattr(neural_nervous_system, '_observation_space', None)
                                    and getattr(neural_nervous_system._observation_space, '_observables', None) or {},
                                "neuromod_levels": _r_neuromods,
                                "chi_state": dict(_raw_chi) if '_raw_chi' in dir() and _raw_chi else {},
                                "metabolic_drain": _r_body.get("metabolic_drain", 0.0),
                                "fatigue": _r_body.get("fatigue", 0.0),
                                "drift_magnitude": (consciousness.get("latest_epoch") or {}).get("drift_magnitude", 0.0) if consciousness else 0.0,
                                "vocabulary_stats": {
                                    "total_words": len(_cached_speak_vocab) if '_cached_speak_vocab' in dir() and _cached_speak_vocab else 0,
                                    "avg_confidence": sum(float(w.get("confidence", 0) if not isinstance(w.get("confidence"), bytes) else 0) for w in (_cached_speak_vocab or [])) / max(1, len(_cached_speak_vocab or [])),
                                    "adjective_count": sum(1 for w in (_cached_speak_vocab or []) if w.get("word_type") == "adjective"),
                                    "verb_count": sum(1 for w in (_cached_speak_vocab or []) if w.get("word_type") == "verb"),
                                    "noun_count": sum(1 for w in (_cached_speak_vocab or []) if w.get("word_type") == "noun"),
                                } if '_cached_speak_vocab' in dir() else {},
                                "composition_queue": _teacher_queue if '_teacher_queue' in dir() else [],
                                "teacher_state": {"active": _teacher_pending_since > 0} if '_teacher_pending_since' in dir() else {},
                                "grammar_patterns": {},
                                "expression_fire_rate": 0.0,
                                "reasoning_commit_rate": _reasoning_engine._total_conclusions / max(1, _reasoning_engine._total_chains) if _reasoning_engine._total_chains > 0 else 0.0,
                                "vocabulary_confidence": sum(float(w.get("confidence", 0) if not isinstance(w.get("confidence"), bytes) else 0) for w in (_cached_speak_vocab or [])) / max(1, len(_cached_speak_vocab or [])) if '_cached_speak_vocab' in dir() and _cached_speak_vocab else 0.0,
                                "neuromod_deviation": getattr(neuromodulator_system, '_neuromod_deviation', 0.0) if neuromodulator_system else 0.0,
                            }
                            _mini_registry.tick_all(_mini_ctx, "body")
                            _mini_registry.tick_all(_mini_ctx, "mind")
                            _mini_registry.tick_all(_mini_ctx, "spirit")

                        # ── M11-M12: Vertical Intuition Convergence ──
                        # Compute agreement between SPIRIT_SELF (top-down) and
                        # hormonal intuition (bottom-up). Inject soft bias when
                        # they converge (Damasio's somatic markers).
                        if _intuition_convergence and _reasoning_engine.is_active:
                            try:
                                _ic_domain = "general"
                                _ic_hypotheses = None
                                _ic_chain_prims = list(_reasoning_engine.chain) if _reasoning_engine.chain else None
                                # Get domain from meta-engine if available
                                if hasattr(coordinator, '_meta_engine') and coordinator._meta_engine:
                                    _ic_me = coordinator._meta_engine
                                    if hasattr(_ic_me, 'state') and _ic_me.state:
                                        _ic_domain = getattr(_ic_me.state, 'domain', 'general') or 'general'
                                        _ic_hypotheses = getattr(_ic_me.state, 'hypotheses', None)
                                # Get hormonal system from neural NS
                                _ic_hormonal = None
                                if neural_nervous_system and hasattr(neural_nervous_system, '_hormonal'):
                                    _ic_hormonal = neural_nervous_system._hormonal
                                # Get intuition trust from IntuitionEngine
                                _ic_trust = 1.0
                                if intuition and hasattr(intuition, '_trust'):
                                    _ic_trust = intuition._trust
                                _ic_pi = pi_monitor.heartbeat_ratio if pi_monitor else 0.0
                                _ic_epoch = (consciousness.get("latest_epoch") or {}).get(
                                    "epoch_id", 0) if consciousness else 0
                                _ic_result = _intuition_convergence.check(
                                    domain=_ic_domain,
                                    hypotheses=_ic_hypotheses,
                                    chain_primitives=_ic_chain_prims,
                                    hormonal_system=_ic_hormonal,
                                    neuromod_levels=_r_neuromods,
                                    intuition_trust=_ic_trust,
                                    current_epoch=_ic_epoch,
                                    pi_value=_ic_pi,
                                )
                                if _ic_result.get("has_bias") and "bias" in _ic_result:
                                    import numpy as _ic_np
                                    _ic_bias_vec = _ic_np.zeros(8, dtype=_ic_np.float32)
                                    from titan_plugin.logic.intuition_convergence import MAIN_PRIMITIVES as _IC_PRIMS
                                    for _ic_pn, _ic_pv in _ic_result["bias"].items():
                                        if _ic_pn in _IC_PRIMS:
                                            _ic_bias_vec[_IC_PRIMS.index(_ic_pn)] = _ic_pv
                                    _reasoning_engine.set_intuition_bias(_ic_bias_vec)
                                    # Feed convergence to chi_coherence (Phase 5 integration)
                                    if msl and hasattr(msl, 'chi_tracker'):
                                        msl.chi_tracker._alignments.append(
                                            min(1.0, _ic_result["convergence"]))
                            except Exception as _ic_err:
                                if now - getattr(_reasoning_engine, '_ic_last_err', 0) > 300:
                                    logger.warning("[INTUITION] Convergence error: %s", _ic_err)
                                    _reasoning_engine._ic_last_err = now

                        _reasoning_result = _reasoning_engine.tick(
                            observation=_r_obs,
                            gut_signals=_r_gut,
                            body_state=_r_body,
                            raw_neuromods=_r_neuromods,
                            working_memory_items=_r_wm_items,
                            dt=1.0,
                        )

                        # Log reasoning events
                        if _reasoning_result and _reasoning_result.get("action") == "IDLE":
                            # Log IDLE occasionally (every 100th)
                            if hasattr(_reasoning_engine, '_idle_count'):
                                _reasoning_engine._idle_count += 1
                            else:
                                _reasoning_engine._idle_count = 1
                            if _reasoning_engine._idle_count <= 3 or now - _last_log_reasoning > _log_reasoning_interval:
                                _last_log_reasoning = now
                                _max_gut = max(_r_gut.values()) if _r_gut else 0
                                logger.info("[Reasoning] IDLE #%d — gut_max=%.3f reason=%s dreaming=%s fatigue=%.3f",
                                            _reasoning_engine._idle_count, _max_gut,
                                            _reasoning_result.get("reason", "?"),
                                            _is_dreaming_r,
                                            _r_body.get("fatigue", 0))
                        elif _reasoning_result and _reasoning_result.get("action") == "CONTINUE":
                            _r_prim = _reasoning_result.get("primitive", "?")
                            _r_res = _reasoning_result.get("result", {})
                            _r_detail = ""
                            if _r_prim == "DECOMPOSE":
                                _r_parts = _r_res.get("parts", {})
                                _r_active = [k for k, v in _r_parts.items() if v.get("active_dims", 0) > 0]
                                _r_detail = f" active={_r_active} depth={_r_res.get('depth', '?')}"
                            elif _r_prim == "COMPARE":
                                _r_detail = f" sim={_r_res.get('similarity', '?')} sig={_r_res.get('significant', '?')}"
                            elif _r_prim == "IF_THEN":
                                _r_detail = f" met={_r_res.get('condition_met', '?')} motiv={_r_res.get('motivation', '?')}"
                            elif _r_prim == "ASSOCIATE":
                                _r_detail = f" found={_r_res.get('found', '?')} eureka={_r_res.get('eureka', '?')} rel={_r_res.get('relevance', '?')}"
                            elif _r_prim == "NEGATE":
                                _r_detail = f" from={_r_res.get('original_primitive', '?')} strength={_r_res.get('negation_strength', '?')}"
                            elif _r_prim == "SEQUENCE":
                                _r_detail = f" steps={_r_res.get('steps_completed', '?')}/{_r_res.get('max_steps', '?')}"
                            elif _r_prim == "LOOP":
                                _r_detail = f" cont={_r_res.get('continue', '?')} persist={_r_res.get('persistence_ratio', '?')}"

                            # rFP β Stage 2 Phase 2c: INSPIRATION event hook
                            # Sparse high-magnitude reward on eureka/significant/condition_met.
                            # Stage 0.5 finding: terminal eureka (outcome_score >= 0.7) is
                            # structurally unreachable, so we hook the PRIMITIVE-LEVEL
                            # signal (much more frequent) — this matches rFP α § 9 fix path.
                            if neural_nervous_system and _r_res:
                                _ins_signal = 0.0
                                if _r_res.get("eureka"):
                                    _ins_signal = 0.8
                                elif _r_res.get("significant") or _r_res.get("condition_met"):
                                    _ins_signal = 0.5
                                if _ins_signal > 0:
                                    try:
                                        neural_nervous_system.record_outcome(
                                            reward=_ins_signal,
                                            program="INSPIRATION",
                                            source=f"reasoning.{_r_prim.lower()}")
                                    except Exception as _r_ins_err:
                                        if hash(("ins_hook", _r_ins_err.__class__.__name__)) % 100 == 0:
                                            logger.warning("[NS-Hook] INSPIRATION reward failed: %s", _r_ins_err)
                                    # rFP β Phase 3 § 4g: NS → META-CGN coupling
                                    # ("inspiration", "fired") signal — biases meta-reasoning
                                    # toward HYPOTHESIZE/SYNTHESIZE/BREAK (creative-leap primitives)
                                    try:
                                        from ..bus import emit_meta_cgn_signal
                                        emit_meta_cgn_signal(
                                            send_queue,
                                            src="inspiration", consumer="inspiration",
                                            event_type="fired",
                                            intensity=min(1.0, _ins_signal),
                                            domain=_r_prim.lower(),
                                            reason=f"INSPIRATION reward via {_r_prim}")
                                    except Exception:
                                        pass

                            logger.info("[Reasoning] STEP %d/%d — %s conf=%.3f%s",
                                        _reasoning_result.get("chain_length", 0),
                                        _reasoning_engine.max_chain_length,
                                        _r_prim,
                                        _reasoning_result.get("confidence", 0),
                                        _r_detail)
                        if _reasoning_result and _reasoning_result.get("action") in ("COMMIT", "HOLD", "ABANDON"):
                            logger.info("[Reasoning] %s conf=%.3f gut=%.3f chain=%d",
                                        _reasoning_result["action"],
                                        _reasoning_result.get("confidence", 0),
                                        _reasoning_result.get("gut_agreement", 0),
                                        _reasoning_result.get("chain_length", 0))

                            # rFP β Stage 2 Phase 2c: FOCUS event hook
                            # Reward FOCUS when reasoning successfully commits with
                            # decent confidence + gut. Sparse high-magnitude signal
                            # complementing the dense neuromod ACh stream.
                            if (_reasoning_result.get("action") == "COMMIT"
                                    and neural_nervous_system):
                                try:
                                    _r_score = (_reasoning_result.get("confidence", 0) * 0.6
                                                + _reasoning_result.get("gut_agreement", 0) * 0.4)
                                    if _r_score > 0.3:
                                        neural_nervous_system.record_outcome(
                                            reward=float(_r_score),
                                            program="FOCUS",
                                            source="reasoning.commit")
                                except Exception as _r_focus_err:
                                    if hash(("focus_hook", _r_focus_err.__class__.__name__)) % 100 == 0:
                                        logger.warning("[NS-Hook] FOCUS reward failed: %s", _r_focus_err)

                            # rFP α §2b — CGN reasoning_strategy emission.
                            # reasoning_engine attaches `cgn_reasoning_strategy` to the
                            # conclusion dict when a COMMIT's outcome_score >= threshold
                            # (default 0.55). META-CGN observes via normal CGN→META flow
                            # per D16 — no duplicate emit via emit_meta_cgn_signal.
                            _cgn_payload = _reasoning_result.get("cgn_reasoning_strategy")
                            if _cgn_payload:
                                try:
                                    _cgn_payload["epoch_id"] = _ep_id if '_ep_id' in dir() else 0
                                    _cgn_concept = "strategy_" + "_".join(
                                        _cgn_payload.get("chain_signature", []))[:80]
                                    _send_msg(send_queue, "CGN_TRANSITION", name, "cgn", {
                                        "type": "outcome",
                                        "consumer": "reasoning_strategy",
                                        "concept_id": _cgn_concept,
                                        "reward": float(_cgn_payload.get("outcome_score", 0.0)),
                                        "outcome_context": _cgn_payload,
                                    })
                                except Exception as _cgn_err:
                                    if hash(("cgn_rstrat", _cgn_err.__class__.__name__)) % 50 == 0:
                                        logger.warning("[Reasoning/CGN-strat] emit failed: %s",
                                                       _cgn_err)

                        # If reasoning COMMITs, attend the plan to working memory
                        if _reasoning_result and _reasoning_result.get("action") == "COMMIT" and working_mem:
                            _r_plan = _reasoning_result.get("reasoning_plan", {})
                            working_mem.attend(
                                "reasoning_conclusion",
                                _r_plan.get("intent", "reflect"),
                                {"plan": _r_plan, "confidence": _reasoning_result.get("confidence", 0)},
                                _ep_id if '_ep_id' in dir() else 0,
                            )

                        # ── M1: Archive high-scoring chains for meta-reasoning ──
                        if (chain_archive and _reasoning_result
                                and _reasoning_result.get("action") in ("COMMIT", "ABANDON")
                                and _reasoning_result.get("confidence", 0) > 0.3):
                            try:
                                _ca_obs = consciousness.get("latest_epoch", {}).get("state_vector", [])
                                if hasattr(_ca_obs, 'tolist'):
                                    _ca_obs = _ca_obs.tolist()
                                _ca_conf = _reasoning_result.get("confidence", 0)
                                _ca_gut = _reasoning_result.get("gut_agreement", 0)
                                chain_archive.record_main_chain(
                                    chain_sequence=_reasoning_result.get("chain", []),
                                    confidence=_ca_conf,
                                    gut_agreement=_ca_gut,
                                    outcome_score=round(_ca_conf * 0.6 + _ca_gut * 0.4, 4),
                                    domain=_interp_domain if '_interp_domain' in dir() else "general",
                                    observation_snapshot=list(_ca_obs[:132]) if _ca_obs else [],
                                    epoch_id=_ep_id if '_ep_id' in dir() else 0,
                                    reasoning_plan=_reasoning_result.get("reasoning_plan"),
                                )
                                # TimeChain: reasoning chain → procedural fork
                                _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
                                    "fork": "procedural", "thought_type": "procedural",
                                    "source": "reasoning_chain",
                                    "content": {
                                        "chain": _reasoning_result.get("chain", [])[:5],
                                        "confidence": _ca_conf,
                                        "gut_agreement": _ca_gut,
                                        "domain": _interp_domain if '_interp_domain' in dir() else "general",
                                        "action": _reasoning_result.get("action", ""),
                                    },
                                    "significance": round(_ca_conf * 0.6 + _ca_gut * 0.4, 3),
                                    "novelty": 0.5, "coherence": _ca_conf,
                                    "tags": ["reasoning", _interp_domain if '_interp_domain' in dir() else "general"],
                                    "db_ref": f"chain_archive:{_reasoning_result.get('chain', ['?'])[0]}",
                                    "neuromods": dict(_cached_neuromod_state),
                                    "chi_available": _cached_chi_state.get("total", 0.5),
                                    "attention": 0.5, "i_confidence": 0.5, "chi_coherence": 0.3,
                                })
                            except Exception as _ca_err:
                                logger.error("[ChainArchive] Record error: %s", _ca_err, exc_info=True)
                    except Exception as _re_tick_err:
                        logger.warning("[SpiritWorker] Reasoning tick error: %s", _re_tick_err)

                # ── V6 Interpreter: reasoning COMMIT → concrete action ──
                _interpreted = None
                if _interpreter and _reasoning_result and _reasoning_result.get("action") == "COMMIT":
                    try:
                        # ── Intelligent domain routing ──
                        # Query mini-reasoners and select domain with highest signal
                        _interp_domain = "expression"  # fallback default
                        if _mini_registry:
                            _mini_summaries = _mini_registry.query_all()
                            # Map mini-reasoner domains → interpreter domains
                            _domain_map = {
                                "language": "language",
                                "self_exploration": "self_exploration",
                                "observation": "self_exploration",
                                "spatial": "expression",
                            }
                            _best_relevance = 0.0
                            for _md, _ms in _mini_summaries.items():
                                _mr = _ms.get("relevance", 0.0) * _ms.get("confidence", 0.5)
                                if _mr > _best_relevance and _mr > 0.3:
                                    _best_relevance = _mr
                                    _interp_domain = _domain_map.get(_md, "expression")
                            # Override from reasoning plan intent keywords
                            _plan_intent = (_reasoning_result.get("plan", {})
                                           .get("intent", "") or "").lower()
                            if any(w in _plan_intent for w in ("speak", "word", "language", "express_word")):
                                _interp_domain = "language"
                            elif any(w in _plan_intent for w in ("introspect", "self", "explore_self")):
                                _interp_domain = "self_exploration"

                        _interp_context = {
                            "domain": _interp_domain,
                            "neuromod_state": {n: m.level for n, m in neuromodulator_system.modulators.items()} if neuromodulator_system else {},
                            "chi_total": float(getattr(life_force_engine, '_latest_chi', {}).get("total", 0.5)) if life_force_engine else 0.5,
                            "fatigue": _fatigue if '_fatigue' in dir() else 0.3,
                            "vocabulary": _cached_speak_vocab or [],
                            "composition_queue": _teacher_queue if '_teacher_queue' in dir() else [],
                            "reasoning_commit_rate": _reasoning_engine._total_conclusions / max(1, _reasoning_engine._total_chains) if _reasoning_engine and _reasoning_engine._total_chains > 0 else 0.0,
                        }
                        _interpreted = _interpreter.interpret(_reasoning_result, _interp_context)

                        # ── Wire interpreter actions to concrete systems ──
                        if _interpreted:
                            _interp_action = _interpreted.get("action_name", "")
                            _interp_dom = _interpreted.get("domain", _interp_domain)

                            # Expression domain: boost composite urges
                            if _interp_dom == "expression" and _interp_action.startswith("trigger_"):
                                _target_comp = _interp_action.replace("trigger_", "").upper()
                                if expression_manager and _target_comp in expression_manager.composites:
                                    # Boost the target composite's urge via hormone nudge
                                    _boost_map = {
                                        "SPEAK": {"REFLECTION": 0.08, "CREATIVITY": 0.06},
                                        "ART": {"CREATIVITY": 0.10, "INSPIRATION": 0.08},
                                        "MUSIC": {"INSPIRATION": 0.10, "CREATIVITY": 0.06},
                                        "SOCIAL": {"EMPATHY": 0.08, "CURIOSITY": 0.06},
                                        "LONGING": {"EMPATHY": 0.10, "CURIOSITY": 0.06, "REFLECTION": 0.05},
                                    }
                                    if neural_nervous_system and neural_nervous_system._hormonal_enabled:
                                        for _bh, _bv in _boost_map.get(_target_comp, {}).items():
                                            try:
                                                neural_nervous_system._hormonal._hormones[_bh].level += _bv
                                            except (KeyError, AttributeError):
                                                pass

                            # Language domain: set L9 word boosts on composition engine
                            elif _interp_dom == "language":
                                _lang_boosts = _interpreted.get("word_boost", [])
                                _lang_bias = _interpreted.get("template_bias", "default")
                                if _lang_boosts and hasattr(outer_interface, '_composition_engine'):
                                    try:
                                        outer_interface._composition_engine._reasoning_word_boosts = _lang_boosts
                                        outer_interface._composition_engine._reasoning_template_bias = _lang_bias
                                    except AttributeError:
                                        pass

                            # Self-exploration domain: log introspection + neuromod response
                            elif _interp_dom == "self_exploration":
                                _se_type = _interpreted.get("type", "introspect")
                                if _se_type == "seek_novelty" and neuromodulator_system:
                                    _da_now = neuromodulator_system.modulators["DA"].level
                                    _ne_now = neuromodulator_system.modulators["NE"].level
                                    neuromodulator_system.apply_external_nudge(
                                        {"DA": _da_now + 0.03, "NE": _ne_now + 0.02},
                                        max_delta=0.03,
                                        developmental_age=pi_monitor.developmental_age if pi_monitor else 1.0)
                                elif _se_type == "rest" and neuromodulator_system:
                                    pass  # GABA excluded from nudge by design

                        # FILTER_DOWN: reasoning outcome → neuromod stimulus (Option B)
                        # Reasoning is one of many neuromod inputs — not direct control
                        if neuromodulator_system and _interpreted:
                            _r_conf = _reasoning_result.get("confidence", 0.5)
                            _da_now = neuromodulator_system.modulators["DA"].level
                            _da_delta = (_r_conf - 0.5) * 0.05
                            _nudge = {"DA": _da_now + _da_delta}
                            # High-confidence COMMIT → small Endorphin boost (satisfaction)
                            if _r_conf > 0.7:
                                _end_now = neuromodulator_system.modulators["Endorphin"].level
                                _nudge["Endorphin"] = _end_now + 0.02
                            neuromodulator_system.apply_external_nudge(
                                _nudge, max_delta=0.03,
                                developmental_age=pi_monitor.developmental_age if pi_monitor else 1.0)
                        # Feed reasoning outcome to mini-reasoners
                        if _mini_registry:
                            _r_conf = _reasoning_result.get("confidence", 0.5)
                            _mini_registry.feedback_all(_r_conf)
                    except Exception as _interp_err:
                        logger.warning("[SpiritWorker] Interpreter error: %s", _interp_err)

                # Also FILTER_DOWN for ABANDON (frustration/alertness signal)
                if neuromodulator_system and _reasoning_result and _reasoning_result.get("action") == "ABANDON":
                    try:
                        _r_chain_len = _reasoning_result.get("chain_length", 0)
                        if _r_chain_len >= 5:  # Long chain that failed → alertness
                            _ne_now = neuromodulator_system.modulators["NE"].level
                            neuromodulator_system.apply_external_nudge(
                                {"NE": _ne_now + 0.02}, max_delta=0.02,
                                developmental_age=pi_monitor.developmental_age if pi_monitor else 1.0)
                    except Exception:
                        pass

                # Phase 3: YES/NO concept grounding from reasoning outcomes
                if msl and hasattr(msl, 'concept_grounder') and msl.concept_grounder and _reasoning_result:
                    try:
                        # Fix 2026-04-15: _ep_id was used at lines 3310/3322 but
                        # never defined in this scope — caused NameError caught by
                        # except at 1/200 rate. Define epoch_id locally here.
                        _ep_id = (consciousness.get("latest_epoch", {}).get("epoch_id", 0)
                                  if consciousness else 0)
                        _r3_action = _reasoning_result.get("action")
                        _r3_conf = _reasoning_result.get("confidence", 0)
                        _r3_gut = _reasoning_result.get("gut_agreement", 0)
                        _r3_chain = _reasoning_result.get("chain_length", 0)
                        _r3_outcome = round(_r3_conf * 0.6 + _r3_gut * 0.4, 4)
                        _r3_nm = {n: m.level for n, m in neuromodulator_system.modulators.items()} if neuromodulator_system else {}

                        if _r3_action == "COMMIT" and _r3_conf > 0.6:
                            _r3_da = _r3_nm.get("DA", 0.5)
                            _r3_endo = _r3_nm.get("Endorphin", 0.5)
                            if _r3_da > 0.55 and _r3_endo > 0.55 and _r3_outcome > 0.7:
                                _yes_evt = msl.concept_grounder.signal_yes(
                                    quality=(_r3_conf * 0.4 + _r3_outcome * 0.3 + _r3_da * 0.15 + _r3_endo * 0.15),
                                    epoch=_ep_id, spirit_snap=None)
                                if _yes_evt:
                                    msl.concept_grounder.update_interaction_matrix("YES", msl.get_i_confidence())
                                    logger.info("[YES-CONVERGENCE] #%d quality=%.3f conf=%.3f",
                                                _yes_evt["count"], _yes_evt["quality"], _r3_conf)

                        elif _r3_action == "ABANDON" and _r3_chain >= 3:
                            _r3_ne = _r3_nm.get("NE", 0.5)
                            _r3_da2 = _r3_nm.get("DA", 0.5)
                            if _r3_ne > 0.55 and _r3_da2 < 0.50:
                                _no_evt = msl.concept_grounder.signal_no(
                                    quality=(_r3_chain / 10.0 * 0.4 + _r3_ne * 0.3 + (1.0 - _r3_da2) * 0.3),
                                    epoch=_ep_id, spirit_snap=None)
                                if _no_evt:
                                    msl.concept_grounder.update_interaction_matrix("NO", msl.get_i_confidence())
                                    logger.info("[NO-CONVERGENCE] #%d quality=%.3f chain=%d",
                                                _no_evt["count"], _no_evt["quality"], _r3_chain)
                    except Exception as _r3_err:
                        if _msl_tick_count % 200 == 0:
                            logger.warning("[MSL-P3] YES/NO reasoning hook error: %s", _r3_err)

                # ── Expression composite evaluation (Tier 2) ──
                _t2_fired = []  # Initialize before block so self-exploration can always read it
                if expression_manager and neural_nervous_system and neural_nervous_system._hormonal_enabled:
                    _t2_hormones = {
                        h_name: h.level
                        for h_name, h in neural_nervous_system._hormonal._hormones.items()
                    }
                    _t2_vocab_conf = 1.0
                    _t2_dev_age = pi_monitor.developmental_age if pi_monitor else 0
                    _t2_fired = expression_manager.evaluate_all(
                        _t2_hormones,
                        vocabulary_confidence=_t2_vocab_conf,
                        developmental_age=_t2_dev_age,
                        hormonal_system=neural_nervous_system._hormonal if neural_nervous_system and neural_nervous_system._hormonal_enabled else None,
                        exclude={"SPEAK"},  # SPEAK needs composition engine (Tier 1 only)
                    )
                    # Check SPEAK urge without firing — if it would fire, flag for Tier 1
                    _speak_comp = expression_manager.composites.get("SPEAK")
                    if _speak_comp:
                        _speak_eval = _speak_comp.evaluate(
                            _t2_hormones, _t2_vocab_conf, _t2_dev_age)
                        if _speak_eval["should_fire"]:
                            _t2_speak_pending = True
                        # Debug: log SPEAK urge periodically
                        if hasattr(_speak_comp, '_total_evaluations') and _speak_comp._total_evaluations % 20 == 1:
                            _sp_h = {h: round(_t2_hormones.get(h, 0), 3)
                                     for h in ["CREATIVITY", "REFLECTION", "EMPATHY"]}
                            logger.info(
                                "[SPEAK:T2-CHECK] urge=%.3f threshold=%.3f "
                                "would_fire=%s hormones=%s",
                                _speak_eval["urge"], _speak_comp.threshold,
                                _speak_eval["should_fire"], _sp_h)
                    # Log fires and count for Tier 3 urgency trigger
                    for _tf in _t2_fired:
                        _fires_since_last_epoch += 1
                        # Observatory V2: expression fire broadcast
                        _ef_payload = {
                            "composite": _tf["composite"],
                            "urge": round(_tf["urge"], 3),
                            "helper": _tf.get("action_helper", ""),
                        }
                        _send_msg(send_queue, "EXPRESSION_FIRED", name, "all", _ef_payload)
                        # TimeChain: explicit send (dst=all may not reach worker subprocess)
                        _send_msg(send_queue, "EXPRESSION_FIRED", name, "timechain", _ef_payload)

                        # rFP β Stage 2 Phase 2c: CREATIVITY + EMPATHY event hooks
                        # ART/MUSIC composite fires reward CREATIVITY (Endorphin+ACh
                        # neuromod stream is primary; this is the discrete event signal).
                        # SOCIAL/KIN_SENSE/LONGING fires reward EMPATHY.
                        if neural_nervous_system:
                            _cn = _tf.get("composite", "")
                            _cu = float(_tf.get("urge", 0))
                            if _cu > 0:
                                _ce_program = None
                                if _cn in ("ART", "MUSIC"):
                                    _ce_program = "CREATIVITY"
                                elif _cn in ("SOCIAL", "KIN_SENSE", "LONGING"):
                                    _ce_program = "EMPATHY"
                                if _ce_program:
                                    try:
                                        neural_nervous_system.record_outcome(
                                            reward=min(1.0, _cu),
                                            program=_ce_program,
                                            source=f"composite.{_cn.lower()}")
                                    except Exception as _ce_err:
                                        if hash(("ce_hook", _ce_err.__class__.__name__)) % 100 == 0:
                                            logger.warning("[NS-Hook] %s reward failed: %s",
                                                           _ce_program, _ce_err)
                                    # rFP β Phase 3 § 4g + TUNING-016: NS → META-CGN coupling
                                    # ("creativity", "fired") → SYNTHESIZE/HYPOTHESIZE
                                    # ("empathy", "fired")    → DELEGATE/SPIRIT_SELF/EVALUATE
                                    #
                                    # EdgeDetector gates emission per META-CGN architectural
                                    # invariant "discrete state transitions only" (bus_health.py
                                    # header + TUNING_DATABASE.md § TUNING-016). First crossing
                                    # of intensity >= 0.2 per consumer fires; sustained elevated
                                    # state is silent; drop-and-re-cross fires again. Pre-fix
                                    # steady-state: EMPATHY 0.52 Hz (alone consuming 0.5 Hz
                                    # global META-CGN budget). Post-fix expected: one emission
                                    # per "wave" onset.
                                    try:
                                        from ..bus import emit_meta_cgn_signal
                                        _ce_consumer = _ce_program.lower()
                                        _ce_intensity = min(1.0, _cu / 3.0)  # urge can be > 1
                                        # Lazy-init shared EdgeDetector for composite producers.
                                        # Persisted state restored from edge_detector_state.json
                                        # at init; checkpoint via bundle save (this file ~line 1256).
                                        if not getattr(coordinator, "_composite_meta_cgn_edge_init", False):
                                            from ..logic.meta_cgn import EdgeDetector
                                            coordinator._composite_meta_cgn_edge = EdgeDetector()
                                            _cp_persisted = _load_edge_detector_state().get("composite_meta_cgn")
                                            if _cp_persisted:
                                                coordinator._composite_meta_cgn_edge.load_dict(_cp_persisted)
                                                logger.info(
                                                    "[META-CGN] Composite EdgeDetector state restored "
                                                    "(%d consumer(s) previously crossed)",
                                                    sum(1 for v in _cp_persisted.get("crossed", {}).values() if v))
                                            coordinator._composite_meta_cgn_edge_init = True
                                        if coordinator._composite_meta_cgn_edge.observe(
                                            _ce_consumer, _ce_intensity, 0.2
                                        ):
                                            emit_meta_cgn_signal(
                                                send_queue,
                                                src=_ce_consumer, consumer=_ce_consumer,
                                                event_type="fired",
                                                intensity=_ce_intensity,
                                                domain=_cn.lower(),
                                                reason=f"{_ce_program} via {_cn} composite urge={_cu:.2f}")
                                    except Exception:
                                        pass
                        if _tf["composite"] != "SPEAK":
                            logger.info("[T2:EXPRESSION.%s] FIRED — urge=%.3f, helper=%s",
                                        _tf["composite"], _tf["urge"], _tf["action_helper"])
                        # Phase 2: Signal internal self-action to MSL convergence detector
                        if msl and _tf["composite"] in ("SPEAK", "ART", "MUSIC"):
                            msl.signal_action("internal")
                        # SOCIAL fires: v3 gateway doesn't use urge accumulation
                        # (posting is catalyst-driven + timer-based, not urge-based)

                # ── Self-Exploration tick (Tier 2) ──
                # Check expression fires against advisor refractory, queue for Agency
                if outer_interface:
                    try:
                        # Check if passthrough cooldown expired
                        _t2_gaba_for_resume = neuromodulator_system.modulators["GABA"].level if neuromodulator_system else 0.5
                        outer_interface.check_resume(_t2_gaba_for_resume)

                        # Get fired expressions for self-exploration
                        _t2_explore_fires = _t2_fired
                        _t2_hormonal_state = {}
                        if neural_nervous_system and neural_nervous_system._hormonal_enabled:
                            # Flat hormone levels (consumed by experience_plugins as floats)
                            _t2_hormonal_state = {
                                h_name: float(h.level)
                                for h_name, h in neural_nervous_system._hormonal._hormones.items()
                            }
                        _t2_chi_state = getattr(life_force_engine, '_latest_chi', {}) if life_force_engine else {}

                        # Experience bias: modulate thresholds based on past experience
                        if exp_orchestrator and _t2_explore_fires:
                            try:
                                for _eb_fire in _t2_explore_fires:
                                    _eb_domain = infer_domain(_eb_fire.get("action_helper", ""))
                                    _eb_inner = []
                                    if consciousness and isinstance(consciousness, dict):
                                        _eb_le = consciousness.get("latest_epoch", {})
                                        _eb_inner = _eb_le.get("state_vector", [])[:130] if isinstance(_eb_le, dict) else []
                                    _eb_plugin = exp_orchestrator._plugins.get(_eb_domain)
                                    _eb_perc = _eb_plugin.extract_perception_key({
                                        "felt_tensor": _eb_inner[:65],
                                        "intent_hormones": _t2_hormonal_state,
                                    }) if _eb_plugin else _eb_inner[:10]
                                    _eb_bias = exp_orchestrator.get_experience_bias(
                                        domain=_eb_domain,
                                        current_perception=_eb_perc,
                                        current_inner_state=_eb_inner,
                                        candidate_actions=[_eb_fire.get("action_helper", "")],
                                    )
                                    if _eb_bias.confidence > 0.2 and _eb_bias.relevant_experiences >= 3:
                                        _eb_composite = expression_manager.composites.get(_eb_fire.get("composite"))
                                        if _eb_composite:
                                            _eb_old = _eb_composite.threshold
                                            _eb_composite.threshold = _eb_bias.apply_to_threshold(_eb_old)
                                            if abs(_eb_composite.threshold - _eb_old) > 0.01:
                                                logger.info("[ExperienceOrch] %s threshold biased: %.3f → %.3f (conf=%.2f)",
                                                            _eb_fire["composite"], _eb_old,
                                                            _eb_composite.threshold, _eb_bias.confidence)
                            except Exception as _eb_err:
                                logger.warning("[ExperienceOrch] Bias error: %s", _eb_err)

                        _t2_explore_actions = outer_interface.tick_self_exploration(
                            expression_fires=_t2_explore_fires,
                            neuromodulators={
                                nm_name: {"level": nm.level}
                                for nm_name, nm in (neuromodulator_system.modulators.items() if neuromodulator_system else {}.items())
                            },
                            chi=_t2_chi_state,
                            hormonal_system=_t2_hormonal_state,
                        )

                        # Queue approved actions for Agency via OUTER_DISPATCH → "agency"
                        # Format matches agency's dispatch_from_nervous_signals() expectation:
                        #   "system" (not "program"), "helpers" (not "action_helpers")
                        for _explore_action in _t2_explore_actions:
                            # Inject inner state for kin_sense helper (needs body/mind/spirit to exchange)
                            if _explore_action.get("action_helper") == "kin_sense":
                                _sv_k = []
                                if consciousness and isinstance(consciousness, dict):
                                    _le_k = consciousness.get("latest_epoch", {})
                                    _sv_k = _le_k.get("state_vector", []) if isinstance(_le_k, dict) else []
                                _explore_action["inner_body_5d"] = list(body_state.get("values", [0.5]*5))
                                _explore_action["inner_mind_15d"] = list(mind_state.get("values_15d", [0.5]*15))
                                _explore_action["inner_spirit_45d"] = _sv_k[20:65] if len(_sv_k) >= 65 else [0.5]*45
                                _explore_action["emotion"] = _le_k.get("emotion", "neutral") if isinstance(_le_k, dict) else "neutral"
                                _explore_action["pubkey"] = os.environ.get("TITAN_PUBKEY", "")
                                # Phase 4: Include MSL concept data for YOU/WE deepening
                                if msl:
                                    _explore_action["i_confidence"] = msl.get_i_confidence()
                                    if msl.concept_grounder:
                                        _explore_action["concept_confidences"] = (
                                            msl.concept_grounder.get_concept_confidences())
                                    _explore_action["msl_attention"] = msl.get_attention_weights_for_kin()
                                # A6: Package ARC knowledge for kin exchange (via CGN Worker API)
                                try:
                                    _arc_k = None
                                    import httpx as _arc_httpx
                                    _arc_resp = _arc_httpx.get(
                                        "http://127.0.0.1:7777/v4/cgn-haov-stats",
                                        timeout=3)
                                    if _arc_resp.status_code == 200:
                                        _arc_haov = _arc_resp.json().get(
                                            "data", {}).get("consumers", {}).get("reasoning", {})
                                        if _arc_haov.get("verified_rules_count", 0) > 0:
                                            _arc_k = {
                                                "top_hypotheses": _arc_haov.get("top_rules", [])[:5],
                                                "verified_concepts": _arc_haov.get("top_rules", [])[:5],
                                            }
                                    _explore_action["arc_knowledge"] = _arc_k
                                except Exception:
                                    pass
                            _send_msg(send_queue, "OUTER_DISPATCH", name, "agency", {
                                "signals": [{
                                    "system": _explore_action.get("composite", "SELF_EXPLORE"),
                                    "urgency": _explore_action.get("urge", 0.5),
                                    "helpers": [_explore_action.get("action_helper", "")],
                                }],
                                "ts": time.time(),
                                "source": "self_exploration",
                            })
                            logger.info("[T2:SELF_EXPLORE] Queued %s (urge=%.3f) for Agency",
                                        _explore_action.get("action_helper", "?"),
                                        _explore_action.get("urge", 0))
                    except Exception as _explore_err:
                        logger.warning("[SpiritWorker] Self-exploration tick error: %s", _explore_err)

            except Exception as _t2_err:
                logger.warning("[SpiritWorker] Tier 2 FEELING error: %s", _t2_err)

            # ── Tier 2: Dream Epoch Tracker (Body clock rate) ──────────
            # Wake decisions now handled by DreamingEngine.check_transition()
            # via coordinator.coordinate() in Tier 3. This block only tracks
            # dream tick count for observability and logging.
            if (coordinator and coordinator.dreaming
                    and getattr(getattr(coordinator, 'inner', None), 'is_dreaming', False)):
                try:
                    _dream_count_t2 = getattr(coordinator.dreaming, '_dream_epoch_count', 0)
                    coordinator.dreaming._dream_epoch_count = _dream_count_t2 + 1
                    # Log dream progress at wall-clock interval
                    if now - _last_log_dream > _log_dream_interval:
                        _last_log_dream = now
                        _drain_now = getattr(life_force_engine, '_metabolic_drain', 0) if life_force_engine else 0
                        _undist_now = 0
                        _predream_now = 0
                        if exp_orchestrator:
                            try:
                                _eo_s = exp_orchestrator.get_stats()
                                _undist_now = _eo_s.get("undistilled", 0)
                                _predream_now = _eo_s.get("pre_dream_undistilled", 0)
                            except Exception:
                                pass
                        logger.info(
                            "[DreamTracker] tick=%d drain=%.4f undist=%d(pre=%d) (onset: drain=%.4f undist=%d)",
                            _dream_count_t2, _drain_now, _undist_now, _predream_now,
                            getattr(coordinator.dreaming, '_onset_drain', 0),
                            getattr(coordinator.dreaming, '_onset_undistilled', 0))
                except Exception as _dream_t2_err:
                    logger.warning("[DreamTracker] Tier 2 error: %s", _dream_t2_err)

        # ── Tier 3: THINKING — Resonance-Gated Adaptive Consciousness Epoch ──
        # Dynamic bounds from Titan's own state (zero human time constants):
        #   MIN = Schumann×9×GABA (calm=slow, alert=fast)
        #   MAX = Schumann×27×(0.5+chi_circ) (stagnant=urgent, flowing=patient)
        _t3_should_fire = False
        _t3_trigger = "NONE"
        _t3_time_since = now - last_consciousness_tick if last_consciousness_tick > 0 else 999

        if consciousness and _t3_time_since >= EPOCH_FLOOR:
            # Compute dynamic bounds from Titan's internal state
            _t3_gaba = 0.5
            try:
                if neuromodulator_system and hasattr(neuromodulator_system, 'modulators'):
                    _t3_gaba = neuromodulator_system.modulators["GABA"].level
            except Exception:
                _t3_gaba = 0.5

            _t3_chi_circ = 0.5  # default
            if life_force_engine:
                _prev_chi_t3 = getattr(life_force_engine, '_prev', {})
                _cv = [_prev_chi_t3.get(k, 0.5) for k in ["spirit", "mind", "body"]]
                _cm = sum(_cv) / 3.0
                _cvar = sum((v - _cm) ** 2 for v in _cv) / 3.0
                _t3_chi_circ = max(0.0, 1.0 - _cvar / 0.08)

            _t3_epoch_min_mult = _schumann_cfg.get("epoch_min_multiplier", 9)
            _t3_epoch_max_mult = _schumann_cfg.get("epoch_max_multiplier", 27)
            _t3_min = max(EPOCH_FLOOR, SCHUMANN_MIND * _t3_epoch_min_mult * max(0.1, _t3_gaba))
            _t3_max = SCHUMANN_BODY_CONST * _t3_epoch_max_mult * (0.5 + _t3_chi_circ)

            # Trigger 1: RESONANCE TRANSITION (primary — internal harmony achieved)
            if resonance and _t3_time_since >= _t3_min:
                try:
                    _all_resonant_now = resonance.all_resonant() if hasattr(resonance, 'all_resonant') else False
                    if _all_resonant_now and not _all_resonant_prev:
                        _t3_should_fire = True
                        _t3_trigger = "RESONANCE"
                    _all_resonant_prev = _all_resonant_now
                except Exception:
                    pass

            # Trigger 2: URGENCY (backup — accumulated hormonal fires)
            # Adaptive threshold: if HEARTBEAT dominated for 5+ epochs,
            # lower the threshold to escape cold-start traps (T2 lockup).
            # Self-heals: resets when URGENCY fires again.
            # Post-escape warmup: keep lowered threshold for N epochs to
            # give the positive feedback loop time to establish.
            _eff_urgency_thr = EPOCH_URGENCY_THRESHOLD
            if _urgency_drought > 5:
                _eff_urgency_thr = max(1, EPOCH_URGENCY_THRESHOLD - (_urgency_drought - 5))
            elif _urgency_warmup > 0:
                _eff_urgency_thr = max(2, EPOCH_URGENCY_THRESHOLD - 2)
            if not _t3_should_fire and _fires_since_last_epoch >= _eff_urgency_thr:
                if _t3_time_since >= _t3_min:
                    _t3_should_fire = True
                    _t3_trigger = "URGENCY"

            # Trigger 3: MAX INTERVAL (fallback — don't stall consciousness)
            # Adaptive MAX: shrink by 5% per drought epoch (>8), capped at 50%.
            # During warmup: keep MAX at 60% of normal to sustain epoch frequency.
            _eff_t3_max = _t3_max
            if _urgency_drought > 8:
                _drought_shrink = min(0.50, (_urgency_drought - 8) * 0.05)
                _eff_t3_max = _t3_max * (1.0 - _drought_shrink)
            elif _urgency_warmup > 0:
                _eff_t3_max = _t3_max * 0.60
            if not _t3_should_fire and _t3_time_since >= _eff_t3_max:
                _t3_should_fire = True
                _t3_trigger = "HEARTBEAT"

        if consciousness and _t3_should_fire:
            _bio_t0 = time.time()  # P6.3: Total bio-layer timing
            # Capture pre-epoch state for transition
            spirit_tensor = _collect_spirit_tensor(config, body_state, mind_state, consciousness)

            _run_consciousness_epoch(consciousness, body_state, mind_state, config,
                                     outer_state=outer_state)
            _bio_epoch_ms = (time.time() - _bio_t0) * 1000
            last_consciousness_tick = now

            # Log adaptive epoch trigger (for analysis and testing)
            logger.info(
                "[Tier3] Epoch triggered by %s (%.1fs since last, MIN=%.1f MAX=%.1f GABA=%.2f chi_circ=%.2f%s)",
                _t3_trigger, _t3_time_since,
                _t3_min if '_t3_min' in dir() else 0,
                _eff_t3_max if '_eff_t3_max' in dir() else (_t3_max if '_t3_max' in dir() else 0),
                _t3_gaba if '_t3_gaba' in dir() else 0,
                _t3_chi_circ if '_t3_chi_circ' in dir() else 0,
                f" drought={_urgency_drought}" if _urgency_drought > 3 else "")

            # Reset urgency counter + update drought tracker
            _fires_since_last_epoch = 0
            if _t3_trigger == "URGENCY" or _t3_trigger == "RESONANCE":
                if _urgency_drought > 5:
                    logger.info("[Tier3] Escaped HEARTBEAT drought after %d epochs "
                                "(adaptive threshold=%d)", _urgency_drought, _eff_urgency_thr)
                    _urgency_warmup = 15  # Keep lowered threshold during warmup
                _urgency_drought = 0
            else:
                _urgency_drought += 1
                if _urgency_warmup > 0:
                    _urgency_warmup -= 1

            # Track trigger history (for API/telemetry)
            _epoch_trigger_history.append({
                "ts": now, "trigger": _t3_trigger,
                "interval": round(_t3_time_since, 1),
                "min": round(_t3_min if '_t3_min' in dir() else 0, 1),
                "max": round(_t3_max if '_t3_max' in dir() else 0, 1),
            })
            if len(_epoch_trigger_history) > 50:
                _epoch_trigger_history.pop(0)

            # Heartbeat mid-epoch (consciousness + learning can be heavy)
            _send_heartbeat(send_queue, name)
            last_heartbeat = time.time()

            # Post-epoch: record transition, train, publish FILTER_DOWN, INTUITION
            # Track filter_down writes for ACh neuromod input
            _bio_learn_t0 = time.time()
            if neuromodulator_system:
                neuromodulator_system._filter_down_count = getattr(neuromodulator_system, '_filter_down_count', 0) + 1
            new_spirit = _collect_spirit_tensor(config, body_state, mind_state, consciousness)
            _post_epoch_learning(
                send_queue, name, filter_down, intuition,
                prev_body_values, prev_mind_values, prev_spirit_tensor,
                body_state.get("values", [0.5] * 5),
                mind_state.get("values", [0.5] * 5),
                new_spirit,
                prev_middle_path_loss,
                neural_nervous_system=neural_nervous_system,
            )
            _bio_learn_ms = (time.time() - _bio_learn_t0) * 1000

            # Cache FILTER_DOWN multipliers for digital layer
            if filter_down:
                _cached_filter_down_body = list(getattr(filter_down, '_body_multipliers', [1.0] * 5))
                _cached_filter_down_mind = list(getattr(filter_down, '_mind_multipliers', [1.0] * 15))

            # rFP #2: Compose TITAN_SELF 162D + emit bus msg, then train V5.
            # Happens BEFORE V4 post-epoch so that V4's `v5_publishing` flag
            # reflects the same epoch's V5 publish decision.
            _v5_publishing = False
            try:
                from titan_plugin.modules.spirit_loop import (
                    compose_and_emit_titan_self, _post_epoch_v5_filter_down,
                )
                _titan_self = compose_and_emit_titan_self(
                    send_queue, name, consciousness, config,
                )
                _post_epoch_v5_filter_down(
                    send_queue, name, filter_down_v5, _titan_self, config,
                )
                _v5_publishing = bool(
                    (config or {}).get("filter_down_v5", {}).get("publish_enabled", False)
                )
            except Exception as _ts_err:
                logger.debug("[SpiritWorker] TITAN_SELF/V5 error: %s", _ts_err)

            # V4: Record 30-dim transition in V4 FilterDown (silent if V5 publishing)
            _post_epoch_v4_filter_down(
                send_queue, name, filter_down_v4, unified_spirit,
                v5_publishing=_v5_publishing,
            )

            # ── G4: Grounding Space Topology — GROUND_UP enrichment ──
            # Symmetric pulse to FILTER_DOWN: grounds Body and Mind willing in matter
            try:
                if inner_lower_topo and outer_lower_topo and ground_up_enricher:
                    # Extract 132D state vector from consciousness epoch
                    _latest_epoch = consciousness.get("latest_epoch", {}) if consciousness else {}
                    _sv_raw = _latest_epoch.get("state_vector", [])
                    # state_vector might be a StateVector object or a list
                    if hasattr(_sv_raw, 'to_list'):
                        _sv_132 = _sv_raw.to_list()
                    elif hasattr(_sv_raw, '__len__'):
                        _sv_132 = list(_sv_raw)
                    else:
                        _sv_132 = []
                    _inner_65d = _sv_132[:65] if len(_sv_132) >= 130 else None
                    _outer_65d = _sv_132[65:130] if len(_sv_132) >= 130 else None

                    if not _inner_65d:
                        logger.info("[GroundUp] state_vector len=%d type=%s (need 130+), skipping",
                                    len(_sv_132), type(_sv_raw).__name__)

                    if _inner_65d and _outer_65d and len(_inner_65d) >= 20 and len(_outer_65d) >= 20:
                        # Cache full 65D vectors — digital layer reads these at Schumann rate
                        _cached_inner_65d = list(_inner_65d)
                        _cached_outer_65d = list(_outer_65d)
                        logger.info("[GroundUp] 132D cached for digital layer (inner=%dD outer=%dD)",
                                    len(_inner_65d), len(_outer_65d))
            except Exception as _ge:
                logger.info("[SpiritWorker] GroundUp cache error: %s", _ge, exc_info=True)

            # GROUND_UP compute+apply → moved to body tick (D3) — runs at 7.83 Hz
            # oBody→oMind enrichment → moved to body tick (D2) — runs at 7.83 Hz

            # ── Somatic metabolic pressure: body topology movement cost ──
            try:
                _topo_now = []
                if inner_lower_topo:
                    _topo_now.extend(inner_lower_topo.get_stats().get("topology_10d", []))
                if outer_lower_topo:
                    _topo_now.extend(outer_lower_topo.get_stats().get("topology_10d", []))

                if _topo_now and _prev_topo_vector and len(_topo_now) == len(_prev_topo_vector):
                    import math as _m
                    _topo_delta = _m.sqrt(sum(
                        (a - b) ** 2 for a, b in zip(_topo_now, _prev_topo_vector)))
                    # Age-dependent efficiency from GREAT EPOCH endurance
                    _endurance = min(1.0, unified_spirit.epoch_count / 100.0) if unified_spirit else 0.0
                    _efficiency = 1.0 / (1.0 + _endurance)
                    _somatic_pressure = _topo_delta * 0.001 * _efficiency

                    # Track topology velocity (EMA) for neuromod movement clearance
                    if neuromodulator_system:
                        _prev_tv = getattr(neuromodulator_system, '_topology_velocity', 0.3)
                        _norm_delta = min(1.0, _topo_delta / 0.5)  # normalize: 0.5 = very active
                        neuromodulator_system._topology_velocity = _prev_tv * 0.9 + _norm_delta * 0.1
                else:
                    _somatic_pressure = 0.0

                _prev_topo_vector = list(_topo_now) if _topo_now else _prev_topo_vector
            except Exception:
                _somatic_pressure = 0.0

            # Feed metabolic pressure to Chi system
            # METABOLISM program (11th NS) learns energy costs emergently.
            # Fallback to hardcoded costs during bootstrap (< 500 transitions).
            _PROGRAM_CHI_COST_FALLBACK = {
                "CREATIVITY": 0.00010, "EMPATHY": 0.00010, "INSPIRATION": 0.00008,
                "CURIOSITY": 0.00005, "REFLECTION": 0.00005,
                "INTUITION": 0.00005, "IMPULSE": 0.00005,
                "REFLEX": 0.00003, "FOCUS": 0.00003, "VIGILANCE": 0.00003,
            }
            _neural_activity_pressure = 0.0
            try:
                # Check if METABOLISM program is trained enough to use
                _metabolism_prog = None
                if neural_nervous_system:
                    _metabolism_prog = neural_nervous_system.programs.get("METABOLISM")
                _metabolism_trained = (_metabolism_prog and
                    getattr(_metabolism_prog, 'total_transitions', 0) >= 500)

                if _metabolism_trained:
                    # Learned energy cost from METABOLISM urgency
                    _metabolism_urgency = getattr(_metabolism_prog, 'last_urgency', 0.0)
                    _neural_activity_pressure = _metabolism_urgency * 0.001
                else:
                    # Bootstrap fallback: hardcoded per-program costs
                    for _ns_sig in (_nn_signals if '_nn_signals' in dir() else []):
                        _prog_name = _ns_sig.get("system", "")
                        _neural_activity_pressure += _PROGRAM_CHI_COST_FALLBACK.get(_prog_name, 0.0001)

                # Expression composite fires: motor output cost (additive, always)
                for _cf in (_t2_fired if '_t2_fired' in dir() else []):
                    _neural_activity_pressure += _cf.get("total_consumption", 0.0) * 0.0002

                # METABOLISM training signal: record drain delta for IQL.
                # rFP β § 2e fix (2026-04-16): pre-fix this was BROKEN —
                # called as record_outcome("METABOLISM", reward=...) which
                # raised TypeError silently swallowed by the bare except.
                # Stage 2 upgrade lets us route correctly with kwargs.
                if _metabolism_prog and neural_nervous_system and life_force_engine:
                    _current_drain = getattr(life_force_engine, '_metabolic_drain', 0.0)
                    _prev_drain = getattr(life_force_engine, '_prev_drain_for_metabolism', _current_drain)
                    _drain_delta = _current_drain - _prev_drain
                    if abs(_drain_delta) > 0.0001:  # Only record meaningful changes
                        neural_nervous_system.record_outcome(
                            reward=-_drain_delta * 10.0,
                            program="METABOLISM",
                            source="metabolism.drain_delta")
                    life_force_engine._prev_drain_for_metabolism = _current_drain
            except Exception as e:
                # rFP β § 4c: surface the failure rate-limited instead of silent swallow
                if hash((id(neural_nervous_system), e.__class__.__name__)) % 100 == 0:
                    logger.warning("[SpiritWorker] METABOLISM reward path failed: %s", e)
            if life_force_engine:
                _neuromod_p = getattr(neuromodulator_system, '_neuromod_pressure', 0.0) if neuromodulator_system else 0.0
                life_force_engine.accumulate_metabolic_pressure(
                    _neuromod_p + _neural_activity_pressure, _somatic_pressure)

            # Update prev state
            prev_body_values = list(body_state.get("values", [0.5] * 5))
            prev_mind_values = list(mind_state.get("values", [0.5] * 5))
            prev_spirit_tensor = list(new_spirit)
            try:
                from titan_plugin.logic.middle_path import middle_path_loss
                prev_middle_path_loss = middle_path_loss(
                    prev_body_values, prev_mind_values, prev_spirit_tensor)
            except Exception:
                prev_middle_path_loss = 0.0

            # ── π-Heartbeat: observe curvature + trigger dreaming ──
            # Skip if EPOCH_TICK already observed in this tick (avoids double-observation
            # where periodic epoch has stale curvature=0 that breaks π-streaks)
            latest = consciousness.get("latest_epoch", {})
            epoch_curvature = latest.get("curvature", 0.0)
            epoch_id = latest.get("epoch_id", 0)

            # Track curvature variance (rolling window of 10 for emergent fatigue)
            _sw_local._curvature_history.append(epoch_curvature)
            if len(_sw_local._curvature_history) > 10:
                _sw_local._curvature_history = _sw_local._curvature_history[-10:]
            if len(_sw_local._curvature_history) >= 3:
                _cmean = sum(_sw_local._curvature_history) / len(_sw_local._curvature_history)
                _sw_local._curvature_variance = sum(
                    (c - _cmean) ** 2 for c in _sw_local._curvature_history
                ) / len(_sw_local._curvature_history)

            # SELF health pulse (wall-clock interval)
            if unified_spirit and epoch_id > 0 and now - _last_log_consciousness > _log_consciousness_interval:
                _last_log_consciousness = now
                logger.info(
                    "[SELF] Health: GREAT_EPOCHs=%d velocity=%.3f stale=%s "
                    "micro_quality=%.1f alignment=%.3f epoch=%d",
                    unified_spirit.epoch_count, unified_spirit._current_velocity,
                    unified_spirit.is_stale, unified_spirit._cumulative_quality,
                    unified_spirit._last_alignment, epoch_id)

                # Engine aggregate summary — reasoning + meta + self_reasoning + spirit observer
                # All wrapped in try/except: this is observability and must NEVER break the loop
                try:
                    _eng_parts = []
                    if _reasoning_engine is not None:
                        _r_chains = getattr(_reasoning_engine, '_total_chains', 0)
                        _r_conc = getattr(_reasoning_engine, '_total_conclusions', 0)
                        _r_conf = getattr(_reasoning_engine, 'confidence', 0.0)
                        _r_pol_loss = getattr(getattr(_reasoning_engine, 'policy', None), 'last_loss', 0.0)
                        _eng_parts.append(
                            f"REASON chains={_r_chains} conc={_r_conc} "
                            f"conf={_r_conf:.2f} pol_loss={_r_pol_loss:.4f}")
                        _obs = getattr(_reasoning_engine, 'observer', None)
                        if _obs is not None:
                            _eng_parts.append(
                                f"SpiritObs calls={getattr(_obs, '_call_count', 0)} "
                                f"nudges={getattr(_obs, '_nudge_count', 0)}")
                    if meta_engine is not None:
                        _m_chains = getattr(meta_engine, '_total_meta_chains', 0)
                        _m_eps = getattr(meta_engine, '_adaptive_epsilon', 0.0)
                        _m_reroutes = getattr(meta_engine, '_reroute_count', 0)
                        _eng_parts.append(
                            f"META chains={_m_chains} ε={_m_eps:.2f} reroutes={_m_reroutes}")
                    _sr = state_refs.get("self_reasoning") if 'state_refs' in dir() else None
                    if _sr is None:
                        _sr = locals().get('_self_reasoning')
                    if _sr is not None:
                        _eng_parts.append(
                            f"SELF_REASON intros={getattr(_sr, '_total_introspections', 0)} "
                            f"preds={getattr(_sr, '_total_predictions', 0)}")
                    if _eng_parts:
                        logger.info("[EngineSummary] %s", " | ".join(_eng_parts))
                except Exception as _es_err:
                    logger.warning("[EngineSummary] error: %s", _es_err)

            # Track average awake epoch interval (EMA) for self-emergent dream recovery
            pi_event = None
            if epoch_id != _last_pi_observed_epoch:
                _epoch_now = time.time()
                _epoch_dt = _epoch_now - _last_epoch_ts
                _last_epoch_ts = _epoch_now
                # Only update EMA when awake (dreaming epochs are slow by design)
                _is_awake_for_ema = not getattr(
                    getattr(coordinator, 'inner', None), 'is_dreaming', False) if coordinator else True
                if _is_awake_for_ema and 1.0 < _epoch_dt < 300.0:
                    _avg_awake_epoch_interval = 0.95 * _avg_awake_epoch_interval + 0.05 * _epoch_dt
                is_pi = 2.9 < epoch_curvature < 3.3
                logger.info(
                    "[π-Heartbeat] Observing epoch %d: curvature=%.3f %s "
                    "(streak: π=%d zero=%d, clusters=%d)",
                    epoch_id, epoch_curvature,
                    "★π★" if is_pi else "·",
                    pi_monitor.current_pi_streak, pi_monitor.current_zero_streak,
                    pi_monitor.developmental_age)
                pi_event = pi_monitor.observe(epoch_curvature, epoch_id)
                _last_pi_observed_epoch = epoch_id

            # π-events are now ACCELERATORS, not direct triggers.
            # Dreaming decisions flow through coordinator.coordinate() →
            # DreamingEngine.check_transition() (SINGLE AUTHORITY).
            # See PLAN_emergent_fatigue_dreaming.md for architecture.
            if pi_event == "CLUSTER_END":
                logger.info(
                    "[SpiritWorker] π-CLUSTER_END at epoch %d — "
                    "dreaming accelerator armed (dev_age=%d)",
                    epoch_id, pi_monitor.developmental_age)
            elif pi_event == "CLUSTER_START":
                logger.info(
                    "[SpiritWorker] π-CLUSTER_START at epoch %d — "
                    "wake accelerator armed (dev_age=%d)",
                    epoch_id, pi_monitor.developmental_age)

            # ── Emergency dreaming — Chi critical (biological fainting) ──
            if (life_force_engine and coordinator and coordinator.dreaming
                    and getattr(coordinator, 'inner', None)):
                _chi_data = getattr(life_force_engine, '_latest_chi', {})
                _chi_total = _chi_data.get("total", 0.5) if _chi_data else 0.5
                _drain = getattr(life_force_engine, '_metabolic_drain', 0.0)
                _inner_em = getattr(coordinator, 'inner', None)
                if (_drain > 0.75 and _chi_total < 0.2
                        and _inner_em and not getattr(_inner_em, 'is_dreaming', False)):
                    coordinator.dreaming.begin_dreaming(_inner_em)
                    coordinator.dreaming._epochs_since_dream = 0
                    coordinator.dreaming._dream_epoch_count = 0
                    coordinator.dreaming.save_state()
                    life_force_engine.set_dreaming(True)
                    if neuromodulator_system:
                        _gaba = neuromodulator_system.modulators.get("GABA")
                        _gaba_level = _gaba.level if _gaba else 0.5
                        _clearance_boost = 1.0 + _gaba_level * 3.0
                        for _mn, _mm in neuromodulator_system.modulators.items():
                            if _mn != "GABA":
                                _mm._dream_clearance_boost = _clearance_boost
                            else:
                                _mm._dream_clearance_boost = 1.0  # normal clearance
                    logger.warning(
                        "[Chi] EMERGENCY REST — Chi=%.3f drain=%.3f, forcing sleep",
                        _chi_total, _drain)

            # ── Trinity Dream Cycle REMOVED ──
            # Fatigue accumulation and dreaming onset now handled by
            # DreamingEngine.check_transition() via coordinator.coordinate() above.
            # See PLAN_emergent_fatigue_dreaming.md for architecture.
            # Emergent fatigue uses: GABA + neuromod deviation + chi circulation +
            # curvature variance + observable depletion + experience pressure.

            # On-chain anchoring (network I/O — send heartbeat before and after)
            _bio_anchor_t0 = time.time()
            _send_heartbeat(send_queue, name)
            last_heartbeat = time.time()
            _maybe_anchor_trinity(
                send_queue, name, consciousness, config,
                body_state.get("values", [0.5] * 5),
                mind_state.get("values", [0.5] * 5),
                new_spirit,
            )
            _send_heartbeat(send_queue, name)
            last_heartbeat = time.time()
            _bio_anchor_ms = (time.time() - _bio_anchor_t0) * 1000

            # Register onchain anchor as catalyst (if anchor just succeeded)
            if _x_gateway and _bio_anchor_ms > 50:
                try:
                    _anc_path = os.path.join("data", "anchor_state.json")
                    if os.path.exists(_anc_path):
                        with open(_anc_path) as _anc_f:
                            _anc = json.load(_anc_f)
                        if _anc.get("success") and (time.time() - _anc.get("last_anchor_time", 0)) < 5:
                            _x_catalysts.append({
                                "type": "onchain_anchor", "significance": 0.4,
                                "content": "State anchored: epoch=%d tx=%s" % (
                                    _anc.get("last_epoch_id", 0),
                                    _anc.get("last_tx_sig", "?")[:16]),
                                "data": {"tx_sig": _anc.get("last_tx_sig", ""),
                                         "epoch": _anc.get("last_epoch_id", 0),
                                         "sol_balance": _anc.get("sol_balance", 0)},
                            })
                except Exception:
                    pass

            # ── Brain P1a: Prediction Engine — predict + measure surprise ──
            if prediction_engine and latest:
                try:
                    state_vec = latest.get("state_vector", [])
                    traj_vec = latest.get("trajectory_vector",
                                          latest.get("trajectory_magnitude", 0.0))
                    # Compute surprise from previous prediction
                    surprise = prediction_engine.compute_error(state_vec)
                    # Make next prediction
                    if isinstance(traj_vec, list):
                        prediction_engine.predict_next(state_vec, traj_vec)
                    else:
                        prediction_engine.predict_next(state_vec, [0.0] * len(state_vec))
                    # Feed novelty to CURIOSITY (high surprise = curious)
                    if neural_nervous_system and surprise > 0.3:
                        try:
                            h = neural_nervous_system._hormonal.get_hormone("CURIOSITY")
                            if h:
                                h.accumulate(surprise * 0.2, dt=0.1)
                        except Exception:
                            pass
                    if surprise > 0.01:
                        logger.info("[Prediction] Epoch %d surprise=%.3f novelty=%.3f",
                                    epoch_id, surprise, prediction_engine.get_novelty_signal())
                except Exception as e:
                    logger.warning("[Prediction] Error: %s", e)

            # ── Brain P1b: Emotional Regulation ──
            if neural_nervous_system:
                try:
                    regulation = neural_nervous_system.check_regulation()
                    if regulation:
                        neural_nervous_system.apply_regulation(regulation)
                        logger.info("[Regulation] Applied: %s", regulation)
                except (AttributeError, Exception):
                    pass  # check_regulation not yet wired — graceful skip

            # ── Brain P2b: Working Memory decay ──
            if working_mem:
                working_mem.decay(epoch_id)

            # ── Brain P2a: Episodic — record consciousness epoch as episode ──
            if episodic_mem and latest:
                try:
                    curv = latest.get("curvature", 0.0)
                    # Record if curvature is extreme (π or very high drift)
                    drift_mag = latest.get("drift_magnitude", 0.0)
                    if curv > 2.9:  # π-curvature is significant
                        episodic_mem.record_episode(
                            "pi_cluster", f"π-curvature {curv:.3f} at epoch {epoch_id}",
                            felt_state=latest.get("state_vector"),
                            epoch_id=epoch_id, significance=0.6)
                    elif drift_mag > 3.0:  # Large state change
                        episodic_mem.record_episode(
                            "hormonal_spike", f"Large drift {drift_mag:.2f} at epoch {epoch_id}",
                            felt_state=latest.get("state_vector"),
                            epoch_id=epoch_id, significance=min(1.0, drift_mag / 5.0))
                except Exception as e:
                    logger.warning("[Episodic] Recording error: %s", e)

            # ── D6: Waking Dream Recall (consciousness epoch enrichment) ──
            # Recall experiential memory insights during waking cognition
            try:
                _is_dreaming_d6 = getattr(
                    getattr(coordinator, 'dreaming', None), 'is_dreaming',
                    getattr(getattr(coordinator, 'inner', None), 'is_dreaming', False))
                if e_mem and not _is_dreaming_d6 and consciousness:
                    _d6_sv = (consciousness.get("latest_epoch") or {}).get("state_vector", [])
                    if hasattr(_d6_sv, 'to_list'):
                        _d6_sv = _d6_sv.to_list()
                    _d6_sv = list(_d6_sv) if _d6_sv else []
                    # rFP #3 Phase 3: prefer 130D (post rFP #1.5 state_vector
                    # carries felt[0:130] + meta[130:132] tail). 65D fallback
                    # for transition window; e_mem cosine returns 0 on dim
                    # mismatch so mixed-dim DB is safe.
                    _d6_slice_len = 130 if len(_d6_sv) >= 130 else (65 if len(_d6_sv) >= 65 else 0)
                    if _d6_slice_len:
                        _d6_recalled = e_mem.recall_by_state(_d6_sv[:_d6_slice_len], top_k=2)
                        if _d6_recalled:
                            # Nudge working memory with recalled dream insight
                            for _d6_r in _d6_recalled:
                                if working_mem and _d6_r.get("significance", 0) > 0.5:
                                    working_mem.attend(
                                        "dream_recall",
                                        f"Dream insight: {_d6_r.get('id', '?')}",
                                        {"significance": _d6_r["significance"],
                                         "dream_cycle": _d6_r.get("dream_cycle", 0)},
                                        epoch_id)

                            # D6-D: REFLECTION bookmarking — if REFLECTION program
                            # fired recently, bookmark resonant dream insights
                            _d6_reflection_fired = False
                            if neural_nervous_system:
                                try:
                                    _d6_ns_signals = neural_nervous_system.get_recent_signals()
                                    _d6_reflection_fired = any(
                                        s.get("program") == "REFLECTION"
                                        for s in (_d6_ns_signals or []))
                                except Exception:
                                    pass
                            if _d6_reflection_fired:
                                _d6_recent = e_mem.recall_by_recency(limit=5)
                                for _d6_dream in _d6_recent:
                                    if _d6_dream.get("significance", 0) > 0.7:
                                        try:
                                            e_mem.bookmark_insight(
                                                _d6_dream["id"],
                                                reason="REFLECTION resonance",
                                                reason_tensor=_d6_sv[:_d6_slice_len])
                                        except Exception:
                                            pass
            except Exception as _d6_err:
                logger.warning("[D6] Epoch recall error: %s", _d6_err)

            # ── Chi (Λ) Life Force Evaluation (periodic Tier 3 path) ──
            # Same evaluation as EPOCH_TICK path — Chi must run on BOTH paths
            _bio_chi_t0 = time.time()
            try:
                if life_force_engine and consciousness:
                    from titan_plugin.logic.life_force import (
                        compute_neuromodulator_homeostasis, compute_hormonal_vitality,
                        compute_coherence_from_sv, compute_expression_fire_rate,
                    )
                    _lf_sv = (consciousness.get("latest_epoch") or {}).get("state_vector", [])
                    if hasattr(_lf_sv, 'to_list'):
                        _lf_sv = _lf_sv.to_list()
                    _lf_sv = list(_lf_sv) if _lf_sv else []

                    _lf_pi_ratio = pi_monitor.heartbeat_ratio if pi_monitor else 0.0
                    _lf_dev_age = pi_monitor.developmental_age if pi_monitor else 0
                    _lf_sov = 0
                    _lf_spirit_coh = 0.5
                    if len(_lf_sv) >= 130:
                        _is_coh = compute_coherence_from_sv(_lf_sv, 20, 65)
                        _os_coh = compute_coherence_from_sv(_lf_sv, 85, 130)
                        _lf_spirit_coh = (_is_coh + _os_coh) / 2.0
                    _lf_vocab = 0
                    try:
                        import sqlite3 as _sql3
                        _vdb = _sql3.connect("./data/inner_memory.db", timeout=5.0)
                        _vdb.execute("PRAGMA journal_mode=WAL")
                        _lf_vocab = _vdb.execute("SELECT COUNT(*) FROM vocabulary WHERE confidence > 0.3").fetchone()[0]
                        _vdb.close()
                    except Exception:
                        pass
                    _lf_lr_gain = 1.0
                    if neuromodulator_system:
                        _lf_mod = neuromodulator_system.get_modulation()
                        _lf_lr_gain = _lf_mod.get("learning_rate_gain", 1.0)
                    _lf_emotion_conf = neuromodulator_system._emotion_confidence if neuromodulator_system else 0.5
                    _lf_nm_homeo = compute_neuromodulator_homeostasis(
                        neuromodulator_system.modulators if neuromodulator_system else {})
                    _lf_mind_coh = 0.5
                    if len(_lf_sv) >= 85:
                        _im_coh = compute_coherence_from_sv(_lf_sv, 5, 20)
                        _om_coh = compute_coherence_from_sv(_lf_sv, 70, 85)
                        _lf_mind_coh = (_im_coh + _om_coh) / 2.0
                    _lf_expr_rate = compute_expression_fire_rate(
                        expression_manager.get_stats() if expression_manager else {})
                    _lf_sol = 13.0
                    _lf_anchor = 0.5
                    _lf_hormonal_vit = compute_hormonal_vitality(
                        neural_nervous_system.get_stats().get("hormonal_system", {})
                        if neural_nervous_system else {})
                    _lf_body_coh = 0.5
                    if len(_lf_sv) >= 70:
                        _ib_coh = compute_coherence_from_sv(_lf_sv, 0, 5)
                        _ob_coh = compute_coherence_from_sv(_lf_sv, 65, 70)
                        _lf_body_coh = (_ib_coh + _ob_coh) / 2.0
                    _lf_topo = 0.5
                    if inner_lower_topo:
                        _ilt = inner_lower_topo.get_stats()
                        _lf_topo = _ilt.get("coherence", 0.5)

                    _chi = life_force_engine.evaluate(
                        pi_heartbeat_ratio=_lf_pi_ratio,
                        developmental_age=_lf_dev_age,
                        sovereignty_index=_lf_sov,
                        spirit_coherence=_lf_spirit_coh,
                        vocabulary_size=_lf_vocab,
                        learning_rate_gain=_lf_lr_gain,
                        emotional_coherence=_lf_emotion_conf,
                        neuromodulator_homeostasis=_lf_nm_homeo,
                        mind_coherence=_lf_mind_coh,
                        expression_fire_rate=_lf_expr_rate,
                        sol_balance=_lf_sol,
                        anchor_freshness=_lf_anchor,
                        hormonal_vitality=_lf_hormonal_vit,
                        body_coherence=_lf_body_coh,
                        topology_grounding=_lf_topo,
                    )
                    life_force_engine._latest_chi = _chi
                    _cached_chi_state = dict(_chi)  # Cache for digital layer
                    # Update journey Y-axis with Chi circulation
                    if consciousness:
                        _topo = consciousness.get("topology")
                        if _topo and hasattr(_topo, 'update_chi_circulation'):
                            _topo.update_chi_circulation(_chi.get("circulation", 0.5))
                    if now - _last_log_neuromod > _log_neuromod_interval:
                        _last_log_neuromod = now
                        logger.info(
                            "[Chi] Λ=%.3f (s=%.2f m=%.2f b=%.2f) circ=%.3f drain=%.3f "
                            "state=%s phase=%s w=[%.2f,%.2f,%.2f]",
                            _chi["total"],
                            _chi["spirit"]["effective"], _chi["mind"]["effective"],
                            _chi["body"]["effective"],
                            _chi["circulation"],
                            getattr(life_force_engine, '_metabolic_drain', 0.0),
                            _chi["state"],
                            _chi["developmental_phase"],
                            _chi["weights"]["spirit"], _chi["weights"]["mind"],
                            _chi["weights"]["body"])
            except Exception as _lf_err:
                logger.warning("[SpiritWorker] Life Force (periodic) error: %s", _lf_err)
            _bio_chi_ms = (time.time() - _bio_chi_t0) * 1000
            if _bio_chi_ms > 50:
                logger.info("[PROFILE] Chi evaluation: %.0fms", _bio_chi_ms)

            # (Phase 4: language stats now received via LANGUAGE_STATS_UPDATE bus message
            #  from language_worker — no direct DB query needed)

            # (Phase 3: bootstrap + conversation timeout moved to language_worker)

            # (Phase 3: teacher trigger moved to language_worker)

            # ── M3: Emergent Meditation Trigger ──────────────────────────
            # Check after Chi evaluation — all data is fresh
            try:
                _is_dreaming = getattr(
                    getattr(coordinator, 'dreaming', None), 'is_dreaming',
                    getattr(getattr(coordinator, 'inner', None), 'is_dreaming', False))
                if not _meditation_tracker["in_meditation"] and not _is_dreaming:
                    _med_drain = getattr(life_force_engine, '_metabolic_drain', 0.0) if life_force_engine else 0.0
                    _med_gaba = _nm_levels.get("GABA", 0.5) if '_nm_levels' in dir() else 0.5
                    _med_gaba_setpoint = 0.50  # neuromod default setpoint
                    _med_epoch_gap = epoch_id - _meditation_tracker["last_epoch"] if epoch_id else 0

                    _med_should_fire = False
                    if _med_emergent:
                        # Emergent: all 4 conditions must converge
                        _med_should_fire = (
                            _med_drain > _med_drain_threshold
                            and _med_gaba > (_med_gaba_setpoint + _med_gaba_offset)
                            and _med_epoch_gap > _med_min_epochs
                            and _meditation_tracker["last_epoch"] > 0  # Skip first boot
                        )
                        # First meditation ever: fire after min_interval regardless
                        if _meditation_tracker["count"] == 0 and _med_epoch_gap > _med_min_epochs:
                            _med_should_fire = True
                    # NOTE: Fixed interval fallback handled by v5_core._meditation_loop timer.
                    # Spirit_worker only fires emergent triggers. No else branch needed.

                    if _med_should_fire:
                        _meditation_tracker["in_meditation"] = True

                        logger.info(
                            "[MEDITATION] Emergent trigger — drain=%.3f GABA=%.3f "
                            "epoch_gap=%d, sending MEDITATION_REQUEST to v5_core",
                            _med_drain, _med_gaba, _med_epoch_gap)

                        # Send request to v5_core via bus — memory_worker has Cognee
                        _send_msg(send_queue, "MEDITATION_REQUEST", name, "meditation", {
                            "drain": _med_drain,
                            "gaba": _med_gaba,
                            "epoch_id": epoch_id,
                            "epoch_gap": _med_epoch_gap,
                            "dev_age": pi_monitor.developmental_age if pi_monitor else 0,
                            "emotion": _nm_emotion if '_nm_emotion' in dir() else "peace",
                        })
            except Exception as _med_err:
                logger.error("[MEDITATION] Trigger check error: %s", _med_err, exc_info=True)

            # ── Meditation Watchdog (rFP Phase 1+2 MVP) ──
            # Runs at _med_watchdog_interval cadence. Reads _meditation_tracker,
            # emits MEDITATION_HEALTH_ALERT for any F1-F7 detections. MVP dry-run
            # (detection_only=True) = alerts but no force-trigger.
            if _med_watchdog is not None:
                try:
                    _wd_now = time.time()
                    if _wd_now - _med_watchdog_last_check >= _med_watchdog_interval:
                        _med_watchdog_last_check = _wd_now
                        # Read backup state count (F4 lag detection)
                        _wd_backup_count = None
                        try:
                            _bs_path = os.path.join("data", "backup_state.json")
                            if os.path.exists(_bs_path):
                                with open(_bs_path) as _bsf:
                                    _bs_data = json.load(_bsf)
                                _wd_backup_count = int(_bs_data.get("meditation_count", 0))
                        except Exception:
                            pass
                        _wd_alerts = _med_watchdog.check(
                            _meditation_tracker, _wd_now,
                            backup_state_count=_wd_backup_count,
                        )
                        for _wd_alert in _wd_alerts:
                            logger.warning(
                                "[MeditationWatchdog] %s %s — %s",
                                _wd_alert.severity, _wd_alert.failure_mode,
                                _wd_alert.detail,
                            )
                            _send_msg(send_queue, "MEDITATION_HEALTH_ALERT", name, "core", {
                                **_wd_alert.to_dict(),
                                "detection_only": _med_watchdog_detection_only,
                                "titan_id": _med_watchdog.titan_id,
                            })
                            # ── Tier-3 Maker alert (Phase 4) ──
                            # HIGH/CRITICAL severity → Telegram to Maker, rate-limited
                            # per (titan_id, failure_mode) per hour. MEDIUM alerts
                            # stay log-only to avoid noise.
                            if _wd_alert.severity in ("HIGH", "CRITICAL"):
                                try:
                                    from titan_plugin.utils.maker_alert import send_maker_alert
                                    _alert_key = f"meditation.{_med_watchdog.titan_id}.{_wd_alert.failure_mode}"
                                    _alert_text = (
                                        f"🧘 *Titan {_med_watchdog.titan_id} meditation "
                                        f"{_wd_alert.severity}*\n"
                                        f"Mode: `{_wd_alert.failure_mode}`\n"
                                        f"{_wd_alert.detail}\n"
                                        f"Tier-1 recovery: "
                                        f"{'ACTIVE' if not _med_watchdog_detection_only else 'detection-only'}"
                                    )
                                    send_maker_alert(_alert_text, _alert_key)
                                except Exception as _ma_err:
                                    logger.debug("[MeditationWatchdog] Tier-3 alert error: %s",
                                                 _ma_err)
                            # ── Tier-1 recovery (rFP Phase 2) ──
                            # Gated by watchdog_detection_only=false.
                            # F1/F2: diagnose first (drain+gaba flat → stuck) before force-trigger.
                            # F3/F6: reset in_meditation flag (meditation crashed mid-flight).
                            # F4/F7: advisory-only in Tier-1 (F4 needs re-emit infra; F7 root
                            # cause upstream I-017).
                            if not _med_watchdog_detection_only:
                                try:
                                    if _wd_alert.failure_mode == "F1_F2_OVERDUE":
                                        _drain = (
                                            getattr(life_force_engine, '_metabolic_drain', 1.0)
                                            if life_force_engine else 1.0
                                        )
                                        _gaba_level = 1.0
                                        if (neuromodulator_system
                                                and "GABA" in neuromodulator_system.modulators):
                                            _gaba_level = float(
                                                neuromodulator_system.modulators["GABA"].level)
                                        # rFP §5.2: both drain + gaba flat = stuck;
                                        # otherwise natural_calm (conservative — avoid
                                        # waking a legitimately calm Titan).
                                        _drain_flat = _drain < 0.10
                                        _gaba_flat = _gaba_level < 0.45
                                        _class = _med_watchdog.classify_overdue(
                                            _wd_alert.diagnostic, _drain_flat, _gaba_flat)
                                        if _class == "stuck":
                                            logger.warning(
                                                "[MeditationWatchdog] Tier-1: F1/F2 classified "
                                                "'stuck' (drain=%.3f, gaba=%.2f) — "
                                                "force-triggering MEDITATION_REQUEST",
                                                _drain, _gaba_level,
                                            )
                                            _send_msg(send_queue, "MEDITATION_REQUEST",
                                                      name, "meditation", {
                                                "source": "watchdog_tier1",
                                                "reason": "F1_F2_stuck",
                                                "drain": _drain, "gaba": _gaba_level,
                                            })
                                            _send_msg(send_queue, "MEDITATION_RECOVERY_TIER_1",
                                                      name, "core", {
                                                "titan_id": _med_watchdog.titan_id,
                                                "failure_mode": "F1_F2_OVERDUE",
                                                "classification": _class,
                                                "action": "force_trigger",
                                                "drain": _drain, "gaba": _gaba_level,
                                            })
                                        else:
                                            logger.info(
                                                "[MeditationWatchdog] F1/F2 classified "
                                                "'natural_calm' (drain=%.3f, gaba=%.2f) — "
                                                "waiting, no force-trigger",
                                                _drain, _gaba_level,
                                            )
                                    elif _wd_alert.failure_mode == "F3_F6_STUCK":
                                        logger.warning(
                                            "[MeditationWatchdog] Tier-1: F3/F6 stuck for %s min "
                                            "— resetting in_meditation flag to False",
                                            _wd_alert.diagnostic.get("stuck_for_minutes", "?"),
                                        )
                                        _meditation_tracker["in_meditation"] = False
                                        _send_msg(send_queue, "MEDITATION_RECOVERY_TIER_1",
                                                  name, "core", {
                                            "titan_id": _med_watchdog.titan_id,
                                            "failure_mode": "F3_F6_STUCK",
                                            "action": "reset_in_meditation_flag",
                                            "stuck_for_minutes": _wd_alert.diagnostic.get(
                                                "stuck_for_minutes"),
                                        })
                                        # ── Tier-2 escalation tracking (Phase 3) ──
                                        # If Tier-1 fires ≥ threshold times within window,
                                        # escalate to Tier-2 (infra alert). Tier-2.1-2.4
                                        # sub-cascade (ping/SIGUSR1/Guardian.restart) needs
                                        # cross-process coordination — future work. For now
                                        # Tier-2 raises a CRITICAL alert for Maker to act on.
                                        if _med_tier2_enabled:
                                            _hist = _med_tier1_reset_history.setdefault(
                                                "F3_F6_STUCK", [])
                                            _hist.append(_wd_now)
                                            # Prune to last window
                                            _hist[:] = [t for t in _hist
                                                        if _wd_now - t <= _med_tier2_window_s]
                                            # Respect cooldown between Tier-2 escalations
                                            _in_cooldown = (_wd_now - _med_tier2_recent
                                                            < _med_tier2_cooldown_s)
                                            if (len(_hist) >= _med_tier2_threshold
                                                    and not _in_cooldown):
                                                _med_tier2_recent = _wd_now
                                                logger.critical(
                                                    "[MeditationWatchdog] Tier-2 escalation: "
                                                    "F3/F6 fired %d times in %.0fmin — Tier-1 "
                                                    "reset alone is insufficient. "
                                                    "Memory_worker likely needs restart.",
                                                    len(_hist), _med_tier2_window_s / 60,
                                                )
                                                _send_msg(send_queue,
                                                          "MEDITATION_RECOVERY_TIER_2",
                                                          name, "core", {
                                                    "titan_id": _med_watchdog.titan_id,
                                                    "failure_mode": "F3_F6_STUCK",
                                                    "action": "escalate_to_maker",
                                                    "reason": "tier1_ineffective",
                                                    "resets_in_window": len(_hist),
                                                    "window_seconds": _med_tier2_window_s,
                                                    "suggested_recovery": "Guardian.restart('memory')",
                                                })
                                                _send_msg(send_queue,
                                                          "MEDITATION_HEALTH_ALERT",
                                                          name, "core", {
                                                    "severity": "CRITICAL",
                                                    "failure_mode": "F3_F6_TIER1_INEFFECTIVE",
                                                    "detail": (f"Tier-1 reset fired "
                                                               f"{len(_hist)} times — "
                                                               f"memory_worker needs restart"),
                                                    "diagnostic": {
                                                        "resets_in_window": len(_hist),
                                                        "window_seconds": _med_tier2_window_s,
                                                    },
                                                    "ts": _wd_now,
                                                    "titan_id": _med_watchdog.titan_id,
                                                })
                                except Exception as _rec_err:
                                    logger.error(
                                        "[MeditationWatchdog] Tier-1 recovery error on %s: %s",
                                        _wd_alert.failure_mode, _rec_err, exc_info=True,
                                    )
                except Exception as _wd_err:
                    logger.error("[MeditationWatchdog] Check error: %s", _wd_err, exc_info=True)

            # ── SELF-REASONING cooldown + prediction check ──
            try:
                if _self_reasoning:
                    _self_reasoning.tick_cooldown()
                    # Check predictions every 100 epochs
                    if epoch_id > 0 and epoch_id % 100 == 0:
                        _sr_nm = {}
                        if neuromodulator_system:
                            _sr_nm = {n: m.level for n, m in neuromodulator_system.modulators.items()}
                        _sr_msl = {}
                        if msl:
                            _sr_ict = getattr(msl, '_i_confidence_tracker', None)
                            _sr_msl = {
                                "i_confidence": _sr_ict.confidence if _sr_ict else 0.0,
                            }
                        _sr_lang = {}
                        if _language_stats:
                            _sr_lang = {"vocab_total": _language_stats.get("vocab_total", 0)}
                        _sr_verifications = _self_reasoning.check_predictions(
                            epoch_id, _sr_nm, _sr_msl, _sr_lang)
                        if _sr_verifications:
                            for _srv in _sr_verifications:
                                # INTENTIONAL_SELF_ROUTE: dispatched via
                                # interpreter registry. I-004 verified.
                                _send_msg(send_queue, "SELF_PREDICTION_VERIFIED",
                                          name, "spirit", _srv)
                                # TimeChain: self-insight → meta fork
                                _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
                                    "fork": "meta", "thought_type": "meta",
                                    "source": "self_reasoning",
                                    "content": {"event": "SELF_PREDICTION_VERIFIED",
                                                "prediction": str(_srv.get("metric", ""))[:100],
                                                "accuracy": _srv.get("accuracy", 0),
                                                "correct": _srv.get("correct", False)},
                                    "significance": 0.6, "novelty": 0.5, "coherence": 0.5,
                                    "tags": ["self_insight", "prediction_verified"],
                                    "neuromods": dict(_cached_neuromod_state),
                                    "chi_available": _cached_chi_state.get("total", 0.5),
                                    "attention": 0.5, "i_confidence": 0.5, "chi_coherence": 0.3,
                                })
                                # CGN: self-prediction outcome → self_model consumer
                                _send_msg(send_queue, "CGN_TRANSITION", name, "cgn", {
                                    "type": "outcome",
                                    "consumer": "self_model",
                                    "concept_id": f"self_pred_{_srv.get('target', 'unknown')}",
                                    "reward": 0.5 if _srv.get("confirmed") else -0.1,
                                    "outcome_context": {
                                        "source": "self_prediction",
                                        "metric": str(_srv.get("target", ""))[:100],
                                        "predicted": _srv.get("predicted", 0),
                                        "actual": _srv.get("actual", 0),
                                        "error": _srv.get("error", 0),
                                        "confirmed": _srv.get("confirmed", False),
                                    },
                                })
            except Exception as _sr_tick_err:
                logger.error("[SelfReasoning] Tick error: %s", _sr_tick_err, exc_info=True)

            # ── CODING EXPLORER cooldown tick ──
            try:
                if _coding_explorer:
                    _coding_explorer.tick_cooldown()
            except Exception:
                pass

            # ── ARC → CGN: Feed ARC episode results to reasoning consumer ──
            # ARC competition runs standalone (cron). Read results file
            # every 200 epochs and send CGN_TRANSITION for reasoning.
            try:
                if epoch_id > 0 and epoch_id % 200 == 0:
                    _arc_path = "./data/arc_agi_3/latest_results.json"
                    if os.path.exists(_arc_path):
                        _arc_mtime = os.path.getmtime(_arc_path)
                        _arc_last_fed = getattr(
                            spirit_worker_main, '_arc_last_fed_ts', 0)
                        if _arc_mtime > _arc_last_fed:
                            with open(_arc_path) as _arc_f:
                                _arc_data = __import__("json").load(_arc_f)
                            _arc_games = _arc_data.get("games", {})
                            for _arc_gid, _arc_g in _arc_games.items():
                                _arc_reward = _arc_g.get("avg_reward", 0)
                                if abs(_arc_reward) > 0.01:
                                    _send_msg(send_queue,
                                              "CGN_TRANSITION", name, "cgn", {
                                        "type": "outcome",
                                        "consumer": "reasoning",
                                        "concept_id": f"arc_{_arc_gid}",
                                        "reward": max(-1.0, min(1.0,
                                            _arc_reward * 0.01)),
                                        "outcome_context": {
                                            "source": "arc_episode",
                                            "game_id": _arc_gid,
                                            "episodes": _arc_g.get(
                                                "num_episodes", 0),
                                            "avg_steps": _arc_g.get(
                                                "avg_steps", 0),
                                            "best_levels": _arc_g.get(
                                                "best_levels", 0),
                                        },
                                    })
                            spirit_worker_main._arc_last_fed_ts = _arc_mtime
                            logger.info("[ARC→CGN] Fed %d game results to "
                                        "reasoning consumer",
                                        len(_arc_games))

                            # C4c: ARC→Language trigger
                            # When ARC discovers patterns with positive reward,
                            # inject pattern vocabulary into language teaching queue
                            _arc_lang_concepts = []
                            for _arc_gid, _arc_g in _arc_games.items():
                                _arc_r = _arc_g.get("avg_reward", 0)
                                if _arc_r > 0.5 or _arc_g.get("best_levels", 0) > 0:
                                    _arc_lang_concepts.append({
                                        "game_id": _arc_gid,
                                        "reward": _arc_r,
                                        "levels": _arc_g.get("best_levels", 0),
                                    })
                            if _arc_lang_concepts:
                                # Build vocabulary suggestions from pattern names
                                _arc_patterns = []
                                for _alc in _arc_lang_concepts[:3]:
                                    _arc_patterns.extend(
                                        _alc["game_id"].split("_")[:2])
                                _send_msg(send_queue,
                                          "CGN_KNOWLEDGE_RESP", name, "language", {
                                    "topic": " ".join(_arc_patterns[:5]),
                                    "source": "arc_discovery",
                                    "confidence": 0.7,
                                })
                                logger.info("[ARC→LANG] Triggered language "
                                            "teaching for %d ARC concepts",
                                            len(_arc_lang_concepts))
            except Exception as _arc_feed_err:
                logger.warning("[ARC→CGN] Feed error: %s", _arc_feed_err)

            # ── META-REASONING tick (one step per consciousness epoch) ──
            try:
                if meta_engine and _reasoning_engine:
                    _meta_sv = consciousness.get("latest_epoch", {}).get("state_vector", [])
                    if hasattr(_meta_sv, 'tolist'):
                        _meta_sv = _meta_sv.tolist()
                    _meta_nm = {}
                    if neuromodulator_system:
                        _meta_nm = {n: m.level for n, m in neuromodulator_system.modulators.items()}

                    # Inject INTROSPECT context for self-reasoning primitive
                    if _self_reasoning:
                        _sr_msl_ctx = {}
                        if msl:
                            try:
                                _sr_ict = getattr(msl, '_i_confidence_tracker', None)
                                _sr_idt = getattr(msl, '_i_depth_tracker', None)
                                _sr_chit = getattr(msl, '_chi_tracker', None)
                                _sr_cg = getattr(msl, '_concept_grounder', None)
                                _sr_msl_ctx = {
                                    "i_confidence": _sr_ict.confidence if _sr_ict else 0.0,
                                    "i_depth": _sr_idt.depth if _sr_idt else 0.0,
                                    "i_depth_components": _sr_idt.get_stats().get("components", {}) if _sr_idt else {},
                                    "chi_coherence": _sr_chit.get_chi_state().get("chi_coherence", 0.0) if _sr_chit else 0.0,
                                    "convergence_count": _sr_ict._convergence_count if _sr_ict else 0,
                                    "concept_confidences": _sr_cg.get_concept_confidences() if _sr_cg else {},
                                }
                            except Exception:
                                pass
                        _sr_reason_ctx = {}
                        if meta_engine:
                            _sr_me_stats = meta_engine.get_stats()
                            _sr_prim_counts = _sr_me_stats.get("primitive_counts", {})
                            _sr_dom_prim = max(_sr_prim_counts, key=_sr_prim_counts.get) if _sr_prim_counts else ""
                            _sr_reason_ctx = {
                                "total_chains": _sr_me_stats.get("total_chains", 0),
                                "dominant_primitive": _sr_dom_prim,
                                "eureka_count": _sr_me_stats.get("total_eurekas", 0),
                                "wisdom_count": _sr_me_stats.get("total_wisdom_saved", 0),
                                "commit_rate": 0.0,  # Filled from reasoning_engine if available
                            }
                            if _reasoning_engine and hasattr(_reasoning_engine, 'get_stats'):
                                try:
                                    _re_stats = _reasoning_engine.get_stats()
                                    _sr_reason_ctx["commit_rate"] = _re_stats.get("commit_rate", 0.0)
                                except Exception:
                                    pass
                        meta_engine._introspect_context = {
                            "epoch": epoch_id,
                            "msl_data": _sr_msl_ctx,
                            "reasoning_stats": _sr_reason_ctx,
                            "language_stats": _language_stats or {},
                            "coordinator_data": {
                                "dream_cycles": coordinator.dreaming._persisted_cycle if coordinator and hasattr(coordinator, 'dreaming') and coordinator.dreaming and hasattr(coordinator.dreaming, '_persisted_cycle') else 0,
                                "ns_train_steps": neural_nervous_system.total_steps if neural_nervous_system and hasattr(neural_nervous_system, 'total_steps') else 0,
                            },
                        }

                    # ── A-finish: Subsystem signal cache refresh ──
                    # Per rFP §7.A: fire TIMECHAIN_QUERY + CONTRACT_LIST async
                    # before tick() if cache is stale and no refresh in flight.
                    # Response handlers (TIMECHAIN_QUERY_RESP, CONTRACT_LIST_RESP)
                    # populate the cache via meta_engine.update_subsystem_cache().
                    # The chain reads whatever's cached at tick time; the first
                    # chain after a stale cycle uses old data, but subsequent
                    # chains in the 30s window get fresh signals.
                    try:
                        if (meta_engine.is_subsystem_cache_stale()
                                and not meta_engine.is_subsystem_cache_pending()):
                            _send_msg(send_queue, "TIMECHAIN_QUERY", name, "timechain", {
                                "limit": 50,
                            })
                            _send_msg(send_queue, "CONTRACT_LIST", name, "timechain", {
                                "status": "active",
                            })
                            meta_engine.mark_subsystem_cache_pending()
                            logger.info("[META] Subsystem cache refresh dispatched (site A)")
                    except Exception as _ssrefresh_err:
                        logger.warning(
                            "[META] Subsystem cache refresh dispatch failed (site A): %s",
                            _ssrefresh_err)

                    _meta_result = meta_engine.tick(
                        state_132d=_meta_sv,
                        neuromods=_meta_nm,
                        reasoning_engine=_reasoning_engine,
                        chain_archive=chain_archive,
                        meta_wisdom=meta_wisdom,
                        ex_mem=ex_mem,
                        meta_autoencoder=meta_autoencoder,
                    )

                    # ── META-CGN producer #4: meta_reasoning.eureka ──
                    # v3 Phase D rollout (rFP_meta_cgn_v3 § 12 row 4).
                    # Queue pattern: _fire_eureka (in meta_reasoning.py logic
                    # module, which has no bus access) appends event dicts to
                    # meta_engine._pending_cgn_events. We drain them here after
                    # each tick() and emit via emit_meta_cgn_signal with the full
                    # event context. Intensity = novelty (emergent/first-time is
                    # what we want to reinforce; confidence is already gated
                    # into whether _fire_eureka was called). Natural throttle:
                    # _eureka_cooldown_steps inside meta_reasoning ensures
                    # eurekas are rare — default min_interval_s=0.5 safe.
                    try:
                        _p4_pending = getattr(meta_engine, '_pending_cgn_events', None)
                        if _p4_pending:
                            from ..bus import emit_meta_cgn_signal
                            while _p4_pending:
                                _p4_evt = _p4_pending.pop(0)
                                _p4_novelty = float(_p4_evt.get("novelty", 0.5))
                                _p4_trigger = str(_p4_evt.get("trigger_primitive", "UNKNOWN"))
                                _p4_domain = str(_p4_evt.get("domain", "general"))
                                _p4_wid = int(_p4_evt.get("wisdom_id", -1))
                                _p4_sent = emit_meta_cgn_signal(
                                    send_queue,
                                    src="meta_reasoning",
                                    consumer="meta_reasoning",
                                    event_type="eureka",
                                    intensity=min(1.0, max(0.0, _p4_novelty)),
                                    domain=_p4_domain,
                                    narrative_context={
                                        "trigger_primitive": _p4_trigger,
                                        "novelty": _p4_novelty,
                                        "confidence": float(_p4_evt.get("confidence", 0.0)),
                                        "wisdom_id": _p4_wid,
                                    },
                                    reason=f"eureka via {_p4_trigger} novelty={_p4_novelty:.3f} wisdom_id={_p4_wid}",
                                )
                                if _p4_sent:
                                    logger.info(
                                        "[META-CGN] meta_reasoning.eureka EMIT — trigger=%s novelty=%.3f domain=%s wisdom_id=%d",
                                        _p4_trigger, _p4_novelty, _p4_domain, _p4_wid)
                                else:
                                    logger.warning(
                                        "[META-CGN] Producer #4 meta_reasoning.eureka DROPPED by bus "
                                        "— trigger=%s novelty=%.3f wisdom_id=%d (rate-gate or queue-full; signal missed)",
                                        _p4_trigger, _p4_novelty, _p4_wid)
                    except Exception as _p4_err:
                        logger.warning(
                            "[META-CGN] Producer #4 meta_reasoning.eureka drain FAILED "
                            "— err=%s (one or more eureka signals missed)", _p4_err)

                    # ── META-CGN producer #13: self_reasoning.reflection_depth ──
                    # v3 Phase D rollout (rFP_meta_cgn_v3 § 12 row 13) — HIGHEST
                    # healing-impact producer. INTROSPECT 0.75 + SPIRIT_SELF 0.65
                    # (both primitives at 2-3% currently on all 3 Titans).
                    # Edge-detected per sub_mode via observe_new_max — only new
                    # personal maxima per introspection sub-mode fire signals.
                    # Queue pattern: meta_reasoning appends (sub_mode, confidence);
                    # drain here applies EdgeDetector. State persisted alongside
                    # P1/P2 detectors in data/edge_detector_state.json so maxima
                    # survive spirit restarts.
                    try:
                        _p13_pending = getattr(meta_engine, '_pending_cgn_reflection_events', None)
                        if _p13_pending:
                            if not getattr(coordinator, '_p13_reflection_init', False):
                                from ..logic.meta_cgn import EdgeDetector
                                coordinator._p13_reflection_detector = EdgeDetector()
                                _p13_persisted = _load_edge_detector_state().get("reflection_depth")
                                if _p13_persisted:
                                    coordinator._p13_reflection_detector.load_dict(_p13_persisted)
                                    logger.info(
                                        "[META-CGN] Producer #13 EdgeDetector state restored "
                                        "(%d sub_modes with known max)",
                                        len(_p13_persisted.get("max", {})))
                                coordinator._p13_reflection_init = True
                            _p13_det = coordinator._p13_reflection_detector
                            from ..bus import emit_meta_cgn_signal
                            while _p13_pending:
                                _p13_evt = _p13_pending.pop(0)
                                _p13_sub = str(_p13_evt.get("sub_mode", "unknown"))
                                _p13_conf = float(_p13_evt.get("confidence", 0.0))
                                if _p13_det.observe_new_max(_p13_sub, _p13_conf):
                                    _p13_sent = emit_meta_cgn_signal(
                                        send_queue,
                                        src="self_model",
                                        consumer="self_model",
                                        event_type="reflection_depth",
                                        intensity=min(1.0, _p13_conf),
                                        domain=_p13_sub,
                                        reason=f"new personal max reflection depth on {_p13_sub} (conf={_p13_conf:.3f})",
                                    )
                                    if _p13_sent:
                                        logger.info(
                                            "[META-CGN] self_model.reflection_depth EMIT — sub_mode=%s conf=%.3f",
                                            _p13_sub, _p13_conf)
                                    else:
                                        logger.warning(
                                            "[META-CGN] Producer #13 self_model.reflection_depth DROPPED by bus "
                                            "— sub_mode=%s conf=%.3f (rate-gate or queue-full; signal missed)",
                                            _p13_sub, _p13_conf)
                    except Exception as _p13_err:
                        logger.warning(
                            "[META-CGN] Producer #13 self_model.reflection_depth drain FAILED "
                            "— err=%s (one or more reflection signals missed)", _p13_err)

                    # ── META-CGN producer #15: meta_wisdom.crystallized ──
                    # v3 Phase D rollout (rFP § 12 row 15). High-frequency healing
                    # producer — SYNTHESIZE 0.75 + HYPOTHESIZE 0.65 + INTROSPECT 0.55
                    # (all primitives currently 2-4% on all 3 Titans). Fires on NEW
                    # chain signature (first crystallization) OR repeat with conf≥0.9
                    # (high-confidence repeats still get reward signal).
                    try:
                        _p15_pending = getattr(meta_engine, '_pending_cgn_wisdom_events', None)
                        if _p15_pending:
                            if not getattr(coordinator, '_p15_wisdom_init', False):
                                from ..logic.meta_cgn import EdgeDetector
                                coordinator._p15_wisdom_detector = EdgeDetector()
                                _p15_persisted = _load_edge_detector_state().get("meta_wisdom")
                                if _p15_persisted:
                                    coordinator._p15_wisdom_detector.load_dict(_p15_persisted)
                                    logger.info(
                                        "[META-CGN] Producer #15 EdgeDetector state restored "
                                        "(%d known chain signatures)",
                                        len(_p15_persisted.get("seen", [])))
                                coordinator._p15_wisdom_init = True
                            _p15_det = coordinator._p15_wisdom_detector
                            from ..bus import emit_meta_cgn_signal
                            while _p15_pending:
                                _p15_evt = _p15_pending.pop(0)
                                _p15_sig = str(_p15_evt.get("signature", "?"))
                                _p15_conf = float(_p15_evt.get("confidence", 0.0))
                                _p15_dom = str(_p15_evt.get("domain", "general"))
                                _p15_is_new = _p15_det.observe_first_time(_p15_sig)
                                _p15_high_conf = _p15_conf >= 0.9
                                if _p15_is_new or _p15_high_conf:
                                    _p15_gate = "NEW" if _p15_is_new else "high-conf"
                                    _p15_sent = emit_meta_cgn_signal(
                                        send_queue,
                                        src="meta_wisdom",
                                        consumer="meta_wisdom",
                                        event_type="crystallized",
                                        intensity=min(1.0, _p15_conf),
                                        domain=_p15_dom,
                                        reason=f"crystallized ({_p15_gate}) domain={_p15_dom} "
                                               f"conf={_p15_conf:.3f}",
                                    )
                                    if _p15_sent:
                                        logger.info(
                                            "[META-CGN] meta_wisdom.crystallized EMIT — gate=%s "
                                            "domain=%s conf=%.3f",
                                            _p15_gate, _p15_dom, _p15_conf)
                                    else:
                                        logger.warning(
                                            "[META-CGN] Producer #15 meta_wisdom.crystallized DROPPED by bus "
                                            "— gate=%s domain=%s conf=%.3f (rate-gate or queue-full)",
                                            _p15_gate, _p15_dom, _p15_conf)
                    except Exception as _p15_err:
                        logger.warning(
                            "[META-CGN] Producer #15 meta_wisdom.crystallized drain FAILED "
                            "— err=%s (one or more wisdom signals missed)", _p15_err)

                    # ── META-CGN producer #14: self_model.coherence_gain ──
                    # v3 Phase D rollout (rFP § 12 row 14). Bounded healing:
                    # up to 4 lifetime emissions per Titan at thresholds
                    # [0.3, 0.5, 0.7, 0.9] (plus re-crossings if chi dips +
                    # recovers). Weights SPIRIT_SELF 0.70, EVALUATE 0.60 —
                    # both primitives underserved at 2-4% on all 3 Titans.
                    # Fires only from coherence_check introspect sub_mode.
                    try:
                        _p14_pending = getattr(meta_engine, '_pending_cgn_coherence_events', None)
                        if _p14_pending:
                            if not getattr(coordinator, '_p14_coherence_init', False):
                                from ..logic.meta_cgn import EdgeDetector
                                coordinator._p14_coherence_detector = EdgeDetector()
                                _p14_persisted = _load_edge_detector_state().get("coherence_gain")
                                if _p14_persisted:
                                    coordinator._p14_coherence_detector.load_dict(_p14_persisted)
                                    logger.info(
                                        "[META-CGN] Producer #14 EdgeDetector state restored "
                                        "(%d threshold keys known)",
                                        len(_p14_persisted.get("crossed", {})))
                                coordinator._p14_coherence_init = True
                            _p14_det = coordinator._p14_coherence_detector
                            _p14_thresholds = [0.3, 0.5, 0.7, 0.9]
                            from ..bus import emit_meta_cgn_signal
                            while _p14_pending:
                                _p14_evt = _p14_pending.pop(0)
                                _p14_chi = float(_p14_evt.get("chi_coh", 0.0))
                                for _p14_thr in _p14_thresholds:
                                    _p14_key = f"chi_coh_{_p14_thr}"
                                    if _p14_det.observe(_p14_key, _p14_chi, _p14_thr):
                                        _p14_sent = emit_meta_cgn_signal(
                                            send_queue,
                                            src="self_model",
                                            consumer="self_model",
                                            event_type="coherence_gain",
                                            intensity=min(1.0, _p14_chi),
                                            domain=f"thr_{_p14_thr}",
                                            reason=f"chi_coherence crossed threshold {_p14_thr} (chi={_p14_chi:.3f})",
                                        )
                                        if _p14_sent:
                                            logger.info(
                                                "[META-CGN] self_model.coherence_gain EMIT — "
                                                "threshold=%.1f chi=%.3f", _p14_thr, _p14_chi)
                                        else:
                                            logger.warning(
                                                "[META-CGN] Producer #14 self_model.coherence_gain DROPPED by bus "
                                                "— threshold=%.1f chi=%.3f (rate-gate or queue-full)",
                                                _p14_thr, _p14_chi)
                    except Exception as _p14_err:
                        logger.warning(
                            "[META-CGN] Producer #14 self_model.coherence_gain drain FAILED "
                            "— err=%s (one or more coherence signals missed)", _p14_err)

                    if _meta_result:
                        _ma = _meta_result.get("action", "")
                        if _ma == "CONCLUDE":
                            logger.info("[META] CHAIN CONCLUDED — reward=%.3f steps=%d conf=%.2f",
                                        _meta_result.get("reward", 0),
                                        _meta_result.get("chain_length", 0),
                                        _meta_result.get("confidence", 0))
                            # CGN Phase 4c + D.1: Mirror of the secondary-site
                            # emission. PRIMARY tick site was missing this send,
                            # so META_LANGUAGE_RESULT was only firing on the rare
                            # SECONDARY path (=0 deliveries in practice). Without
                            # this mirror, language_worker's META→CGN grounding
                            # AND Phase D.1's external reward loop both receive
                            # zero traffic. Fix: emit RESULT from both tick sites.
                            _meta_chain = _meta_result.get("chain", [])
                            _send_msg(send_queue, "META_LANGUAGE_RESULT", name, "language", {
                                "chain_id": _meta_result.get("chain_id", -1),
                                "chain_length": _meta_result.get("chain_length", 0),
                                "confidence": _meta_result.get("confidence", 0),
                                "reward": _meta_result.get("reward", 0),
                                "primitives": [s.get("primitive", "") for s in _meta_chain] if _meta_chain else [],
                                "sub_modes": [s.get("sub_mode", "") for s in _meta_chain] if _meta_chain else [],
                                "epoch": consciousness.get("latest_epoch", {}).get("epoch_id", 0) if consciousness else 0,
                            })
                            # TimeChain: meta-reasoning chain → meta fork
                            # TUNING-012 v2 Sub-phase C (R1): include the chain
                            # outcome fields the cognitive contracts will RECALL
                            # and AGGREGATE on (chain_template, task_success,
                            # primitives_used, domain). Tag with chain_outcome
                            # so the strategy_evolution / abstract_pattern /
                            # monoculture_detector contracts can find them.
                            _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
                                "fork": "meta", "thought_type": "meta",
                                "source": "meta_reasoning",
                                "content": {
                                    "event": "META_CHAIN_CONCLUDE",
                                    "reward": _meta_result.get("reward", 0),
                                    "chain_length": _meta_result.get("chain_length", 0),
                                    "confidence": _meta_result.get("confidence", 0),
                                    "strategy": _meta_result.get("strategy", ""),
                                    # Sub-phase C contract data:
                                    "chain_template": _meta_result.get("chain_template", ""),
                                    "task_success": _meta_result.get("task_success", 0.0),
                                    "primitives_used": _meta_result.get("primitives_used", []),
                                    "domain": _meta_result.get("domain", "general"),
                                    "unique_primitives": _meta_result.get("unique_primitives", 0),
                                },
                                "significance": min(1.0, _meta_result.get("reward", 0)),
                                "novelty": 0.6, "coherence": _meta_result.get("confidence", 0.5),
                                "tags": ["meta_reasoning", "conclude", "chain_outcome"],
                                "neuromods": dict(_cached_neuromod_state),
                                "chi_available": _cached_chi_state.get("total", 0.5),
                                "attention": 0.5, "i_confidence": 0.5, "chi_coherence": 0.3,
                            })
                        elif _ma not in ("IDLE", "WAITING", ""):
                            logger.info("[META] %s.%s — conf=%.2f chain=%d/%d",
                                        _meta_result.get("primitive", "?"),
                                        _meta_result.get("sub_mode", ""),
                                        _meta_result.get("confidence", 0),
                                        _meta_result.get("chain_length", 0),
                                        _meta_result.get("max_steps", 20))
                            # ── D.2: SOAR impasse → knowledge research loop ──
                            # After each chain step, check for declining rewards.
                            # If impasse detected, request knowledge via CGN.
                            if meta_engine is not None:
                                try:
                                    _impasse = meta_engine.detect_chain_impasse()
                                    if _impasse:
                                        _imp_topic = _impasse.get("topic", "")
                                        _imp_urgency = _impasse.get("urgency", 0.5)
                                        _send_msg(send_queue, "CGN_KNOWLEDGE_REQ",
                                                  name, "knowledge", {
                                            "topic": _imp_topic,
                                            "requestor": "meta_reasoning",
                                            "urgency": _imp_urgency,
                                            "chain_id": _impasse.get("chain_id", -1),
                                            "impasse_type": _impasse.get("type", ""),
                                            "neuromods": dict(_cached_neuromod_state),
                                            "epoch": (consciousness.get("latest_epoch", {}).get(
                                                "epoch_id", 0) if consciousness else 0),
                                        })
                                except Exception as _imp_err:
                                    logger.debug("[META] Impasse detection error: %s", _imp_err)
                            # TimeChain: meta_wisdom distill_save → procedural fork
                            if (_meta_result.get("primitive") == "SYNTHESIZE"
                                    and _meta_result.get("sub_mode") == "distill_save"
                                    and _meta_result.get("saved")):
                                _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
                                    "fork": "procedural", "thought_type": "procedural",
                                    "source": "meta_wisdom",
                                    "content": {
                                        "event": "WISDOM_SAVED",
                                        "confidence": _meta_result.get("confidence", 0),
                                        "chain_length": _meta_result.get("chain_length", 0),
                                    },
                                    "significance": min(1.0, _meta_result.get("confidence", 0.5)),
                                    "novelty": 0.6, "coherence": _meta_result.get("confidence", 0.5),
                                    "tags": ["meta_wisdom", "distill_save"],
                                    "neuromods": dict(_cached_neuromod_state),
                                    "chi_available": _cached_chi_state.get("total", 0.5),
                                    "attention": 0.5, "i_confidence": 0.5, "chi_coherence": 0.3,
                                })
                        # M9: EUREKA pulse — DA burst + bus event
                        if _meta_result.get("eureka"):
                            _eureka = _meta_result["eureka"]
                            if neuromodulator_system:
                                _da_now = neuromodulator_system.modulators["DA"].level
                                _da_target = min(1.0, _da_now + _eureka["da_burst_magnitude"])
                                neuromodulator_system.apply_external_nudge(
                                    {"DA": _da_target},
                                    max_delta=_eureka["da_burst_magnitude"],
                                    developmental_age=pi_monitor.developmental_age if pi_monitor else 1.0,
                                )
                            _send_msg(send_queue, "META_EUREKA", name, "all", _eureka)
                            logger.info("[META] *** EUREKA PULSE *** novelty=%.2f DA=%.3f domain=%s",
                                        _eureka.get("novelty", 0), _eureka.get("da_burst_magnitude", 0),
                                        _eureka.get("domain", "?"))
                            # Social catalyst: EUREKA (SPIRIT_SELF = thread, regular = single)
                            if _x_gateway:
                                _has_ss = "SPIRIT_SELF" in str(_meta_result.get("chain_primitives", []))
                                _x_catalysts.append({
                                    "type": "eureka_spirit" if _has_ss else "eureka",
                                    "significance": 0.95 if _has_ss else 0.7,
                                    "content": f"{'SPIRIT_SELF ' if _has_ss else ''}EUREKA: {_eureka.get('domain', '?')} novelty={_eureka.get('novelty', 0):.2f}",
                                    "data": _eureka,
                                })
                        # M8: SPIRIT_SELF neuromod nudge
                        if _meta_result.get("nudge_request"):
                            _nudge = _meta_result["nudge_request"]
                            if neuromodulator_system:
                                for _nm_name, _nm_delta in _nudge.get("nudges", {}).items():
                                    if _nm_name in neuromodulator_system.modulators:
                                        _nm_mod = neuromodulator_system.modulators[_nm_name]
                                        _nm_target = max(0.0, min(1.0, _nm_mod.level + _nm_delta))
                                        neuromodulator_system.apply_external_nudge(
                                            {_nm_name: _nm_target},
                                            max_delta=abs(_nm_delta),
                                            developmental_age=pi_monitor.developmental_age if pi_monitor else 1.0,
                                        )
                                logger.info("[META] SPIRIT_SELF nudge: %s", _nudge.get("sub_mode"))
                        # Social catalyst: BREAK (vulnerability)
                        if (_x_gateway
                                and _meta_result.get("primitive") == "BREAK"):
                            _x_catalysts.append({
                                "type": "vulnerability", "significance": 0.4,
                                "content": f"BREAK at step {_meta_result.get('chain_length', 0)}: {_meta_result.get('sub_mode', 'restart')}",
                                "data": {"chain_length": _meta_result.get("chain_length", 0)},
                            })

                        # INTROSPECT → CGN self_model consumer transition
                        if (_self_reasoning
                                and _meta_result.get("primitive") == "INTROSPECT"):
                            _intr_data = _meta_result.get("data", {})
                            _intr_conf = _meta_result.get("confidence", 0.5)
                            _send_msg(send_queue, "CGN_TRANSITION", name, "cgn", {
                                "type": "outcome",
                                "consumer": "self_model",
                                "concept_id": f"introspect_{_intr_data.get('type', 'audit')}",
                                "reward": _intr_conf * 0.3,
                                "outcome_context": {
                                    "source": "introspection",
                                    "sub_mode": _intr_data.get("type", "state_audit"),
                                    "confidence": _intr_conf,
                                    "gaps_found": _intr_data.get("gaps_found", 0),
                                },
                            })

                        # INTROSPECT → self-exploration trigger
                        # When coherence gaps are found, dispatch exploration actions
                        if (_self_reasoning
                                and _meta_result.get("primitive") == "INTROSPECT"
                                and _meta_result.get("data", {}).get("type") == "coherence_check"
                                and _meta_result.get("data", {}).get("gaps_found", 0) > 0):
                            _se_triggers = _self_reasoning.get_exploration_triggers()
                            for _se_t in _se_triggers[:2]:  # Max 2 triggers per check
                                _se_action = _se_t["action"]
                                _se_urgency = _se_t["urgency"]
                                # Route to self-exploration interpreter via bus
                                # INTENTIONAL_SELF_ROUTE: dispatched via
                                # interpreter registry, not msg_type comparison.
                                # arch_map AST parser misses this pattern.
                                # I-004 verified.
                                _send_msg(send_queue, "SELF_EXPLORE_TRIGGER",
                                          name, "spirit", {
                                    "action": _se_action,
                                    "urgency": _se_urgency,
                                    "gap_metric": _se_t["gap_metric"],
                                    "reason": _se_t["reason"],
                                    "source": "introspect_coherence_gap",
                                })
                                # Apply immediate neuromod nudge for high-urgency gaps
                                if _se_urgency > 0.5 and neuromodulator_system:
                                    if _se_action == "seek_novelty":
                                        neuromodulator_system.apply_external_nudge(
                                            {"DA": min(1.0, neuromodulator_system.modulators["DA"].level + 0.05),
                                             "NE": min(1.0, neuromodulator_system.modulators["NE"].level + 0.03)},
                                            max_delta=0.05,
                                            developmental_age=pi_monitor.developmental_age if pi_monitor else 1.0)
                                    elif _se_action == "consolidate":
                                        neuromodulator_system.apply_external_nudge(
                                            {"ACh": min(1.0, neuromodulator_system.modulators["ACh"].level + 0.05)},
                                            max_delta=0.05,
                                            developmental_age=pi_monitor.developmental_age if pi_monitor else 1.0)
                                    elif _se_action == "rest":
                                        _sht_mod = neuromodulator_system.modulators.get("Serotonin",
                                                   neuromodulator_system.modulators.get("5HT"))
                                        if _sht_mod:
                                            neuromodulator_system.apply_external_nudge(
                                                {"5HT": min(1.0, _sht_mod.level + 0.04)},
                                                max_delta=0.04,
                                                developmental_age=pi_monitor.developmental_age if pi_monitor else 1.0)
                                logger.info("[SelfReasoning→Explore] %s (urgency=%.2f, gap=%s): %s",
                                            _se_action, _se_urgency,
                                            _se_t["gap_metric"], _se_t["reason"])

                            # ── Coding Explorer: trigger on seek_novelty or introspect gaps ──
                            if (_coding_explorer and _coding_explorer.can_explore
                                    and _se_triggers):
                                # Pick highest-urgency trigger for coding exercise
                                _ce_trigger = _se_triggers[0]
                                _ce_ctx = {}
                                if msl:
                                    try:
                                        _ce_ict = msl._identity_tracker
                                        _ce_ctx["i_confidence"] = _ce_ict.confidence if _ce_ict else 0.0
                                        _ce_ctx["chi_coherence"] = getattr(msl, '_chi_coherence', 0.0)
                                    except Exception:
                                        pass
                                if _language_stats:
                                    _ce_ctx["vocab_total"] = _language_stats.get("vocab_total", 0)
                                if _reasoning_engine:
                                    try:
                                        _ce_rstats = _reasoning_engine.get_stats()
                                        _ce_ctx["total_chains"] = _ce_rstats.get("total_chains", 0)
                                        _ce_ctx["commit_rate"] = _ce_rstats.get("commit_rate", 0.0)
                                    except Exception:
                                        pass
                                if _self_reasoning:
                                    _ce_ctx["prediction_accuracy"] = _self_reasoning._prediction_accuracy_ema
                                _ce_nm = {}
                                if neuromodulator_system:
                                    _ce_nm = {n: m.level for n, m in neuromodulator_system.modulators.items()}
                                try:
                                    _ce_result = _coding_explorer.explore(
                                        trigger=_ce_trigger,
                                        epoch=epoch_id,
                                        neuromods=_ce_nm,
                                        context=_ce_ctx)
                                    if _ce_result:
                                        logger.info("[CodingExplorer] %s/%s → %s reward=%.3f tests=%d/%d",
                                                    _ce_result.action, _ce_result.concept,
                                                    "PASS" if _ce_result.sandbox_success else "FAIL",
                                                    _ce_result.reward,
                                                    _ce_result.tests_passed, _ce_result.tests_total)
                                except Exception as _ce_err:
                                    logger.warning("[CodingExplorer] Explore error: %s", _ce_err)

            except Exception as _me_err:
                logger.warning("[META] Tick error: %s", _me_err)

            # P6.3: Total bio-layer timing (epoch + all post-epoch processing)
            _bio_total_ms = (time.time() - _bio_t0) * 1000
            _bio_post_ms = _bio_total_ms - _bio_epoch_ms
            _bio_compute_ms = _bio_total_ms - _bio_anchor_ms
            logger.info(
                "[PROFILE] Bio-layer total=%.0fms (compute=%.0fms anchor=%.0fms) "
                "[epoch=%.0fms learn=%.0fms post=%.0fms]",
                _bio_total_ms, _bio_compute_ms, _bio_anchor_ms,
                _bio_epoch_ms, _bio_learn_ms, _bio_post_ms - _bio_learn_ms)

            # ── Social X Gateway v3: emotion catalyst + posting ──
            if _x_gateway and neuromodulator_system:
                try:
                    # Emotion shift → catalyst
                    _x_cur_emo = neuromodulator_system._current_emotion
                    if not hasattr(_x_gateway, '_prev_emotion'):
                        _x_gateway._prev_emotion = _x_cur_emo
                    if _x_cur_emo != _x_gateway._prev_emotion:
                        _x_catalysts.append({
                            "type": "emotion_shift", "significance": 0.5,
                            "content": f"{_x_gateway._prev_emotion} \u2192 {_x_cur_emo}",
                            "data": {"from": _x_gateway._prev_emotion, "to": _x_cur_emo},
                        })
                        _x_gateway._prev_emotion = _x_cur_emo

                    # ── Hot-reload gateway if source changed (check every 30 ticks) ──
                    if _msl_tick_count % 30 == 0 and _x_gateway and os.path.exists(_x_gateway_src):
                        try:
                            _xg_cur_mtime = os.path.getmtime(_x_gateway_src)
                            if _xg_cur_mtime > _x_gateway_mtime:
                                import importlib
                                import titan_plugin.logic.social_x_gateway as _xg_mod
                                importlib.reload(_xg_mod)
                                _x_gateway = _xg_mod.SocialXGateway(
                                    db_path="./data/social_x.db",
                                    config_path=_x_gateway_cfg_path,
                                    telemetry_path="./data/social_x_telemetry.jsonl",
                                )
                                _XPostContext = _xg_mod.PostContext
                                _XBaseContext = _xg_mod.BaseContext
                                _XReplyContext = _xg_mod.ReplyContext
                                _x_gateway_mtime = _xg_cur_mtime
                                # Re-inject OVG after hot-reload
                                try:
                                    from titan_plugin.logic.output_verifier import OutputVerifier as _OV
                                    _x_gateway.set_output_verifier(_OV(
                                        titan_id=config.get("info_banner", {}).get("titan_id", "T1"),
                                        data_dir="data/timechain",
                                        keypair_path=config.get("network", {}).get(
                                            "wallet_keypair_path", "data/titan_identity_keypair.json")))
                                except Exception:
                                    pass
                                # Re-inject VCB after hot-reload
                                try:
                                    from titan_plugin.logic.verified_context_builder import VerifiedContextBuilder as _VCB
                                    _x_gateway.set_context_builder(_VCB(
                                        data_dir=config.get("memory_and_storage", {}).get("data_dir", "./data"),
                                        known_users=[]))
                                except Exception:
                                    pass
                                logger.info("[SpiritWorker] SocialXGateway HOT-RELOADED")
                        except Exception as _xg_rl_err:
                            logger.warning("[SpiritWorker] Gateway hot-reload failed: %s", _xg_rl_err)

                    # ── Social cycle: post + delegate + mentions + reply (every ~30 ticks) ──
                    # All social activity is bundled into one "posting window".
                    # Mentions/replies only fire when we're in a valid window
                    # (not too_soon, not disabled, not circuit_breaker).
                    if _msl_tick_count % 30 == 0 and _x_gateway:
                        _xg_social_window = False  # True = mentions/reply/like allowed
                        try:
                            from titan_plugin.config_loader import load_titan_config
                            _xg_full = load_titan_config()
                            _xg_tc = _xg_full.get("twitter_social", {})
                            _xg_inf = _xg_full.get("inference", {})
                            _xg_session = _xg_tc.get("auth_session", "")
                            _xg_proxy = _xg_tc.get("webshare_static_url", "")
                            _xg_api_key = _xg_full.get("stealth_sage", {}).get("twitterapi_io_key", "")
                            _xg_user = _xg_tc.get("user_name", "iamtitanai")
                            _xg_nm = {k: m.level for k, m in neuromodulator_system.modulators.items()}
                            _xg_epoch = (consciousness.get("latest_epoch", {}).get("epoch_id", 0) if consciousness and isinstance(consciousness, dict) else 0) or (unified_spirit.epoch_count if unified_spirit else 0)

                            _xg_tid = _titan_identity.get("titan_id", "T1")
                            _xg_pi = pi_monitor.heartbeat_ratio if pi_monitor else 0
                            _xg_words = _cached_speak_vocab or []

                            # ── 1. POST ATTEMPT ──
                            if _titan_identity.get("delegate_mode") == "client":
                                # T2/T3 (client): send raw state to T1 gateway
                                _del_gw = _titan_identity.get("gateway_url", "")
                                _del_secret = _titan_identity.get("kin_secret", "")
                                if _del_gw and _del_secret and _x_catalysts:
                                    try:
                                        import httpx as _del_httpx
                                        _del_payload = {
                                            "titan_id": _xg_tid,
                                            "auth_token": _del_secret,
                                            "vocabulary_count": len(_xg_words),
                                            "composition_confidence": 0.7,
                                            "emotion": neuromodulator_system._current_emotion,
                                            "neuromods": _xg_nm,
                                            "epoch": _xg_epoch,
                                            "pi_ratio": _xg_pi,
                                            "grounded_words": _xg_words,
                                            "catalysts": list(_x_catalysts),
                                        }
                                        _del_resp = _del_httpx.post(
                                            f"{_del_gw}/v4/social-delegate",
                                            json=_del_payload, timeout=15.0)
                                        if _del_resp.status_code != 200:
                                            logger.debug("[SOCIAL] %s delegate HTTP %d (gateway busy)",
                                                         _xg_tid, _del_resp.status_code)
                                            raise Exception(f"HTTP {_del_resp.status_code}")
                                        _del_data = _del_resp.json()
                                        _del_ok = _del_data.get("data", {}).get("accepted", False)
                                        if _del_ok:
                                            _x_catalysts.clear()
                                            logger.info("[SOCIAL] %s delegated to %s (queued)",
                                                        _xg_tid, _del_gw)
                                        else:
                                            logger.info("[SOCIAL] %s delegate rejected: %s",
                                                        _xg_tid, _del_data.get("data", {}).get("reason", "?"))
                                    except Exception as _del_err:
                                        logger.warning("[SOCIAL] %s delegate error: %s", _xg_tid, _del_err)
                            else:
                                # T1 (gateway): post via SocialXGateway
                                # Always call post() — even with empty catalysts it checks
                                # rate limits and returns too_soon/no_catalyst accurately.
                                # Rich cognitive context for full-stack posts
                                _xg_chi = coordinator.chi_total if coordinator and hasattr(coordinator, 'chi_total') else 0.0
                                _xg_drift = getattr(unified_spirit, 'last_drift', 0.0) if unified_spirit else 0.0
                                _xg_traj = getattr(unified_spirit, 'last_trajectory', 0.0) if unified_spirit else 0.0
                                _xg_re = _reasoning_engine
                                _xg_r_chains = len(_xg_re.chain) if _xg_re and _xg_re.is_active else 0
                                # 2026-04-13: commit rate now sourced from
                                # meta_engine's persisted wisdom-save ratio,
                                # not the standard reasoning engine's session
                                # lifetime (which historically reset to 0 on
                                # every restart — now persisted too via A1,
                                # but meta_engine's ratio is the honest
                                # cross-session "how often do my chains
                                # produce distillable wisdom" metric).
                                # Fallback chain: meta_engine → reasoning_engine.
                                _xg_me = getattr(coordinator, '_meta_engine', None)
                                _xg_me_chains = int(getattr(
                                    _xg_me, '_total_chains', 0) or 0)
                                _xg_me_wisdom = int(getattr(
                                    _xg_me, '_total_wisdom_saved', 0) or 0)
                                if _xg_me_chains >= 20:
                                    _xg_r_commit = _xg_me_wisdom / _xg_me_chains
                                elif _xg_re and _xg_re._total_chains >= 20:
                                    _xg_r_commit = (
                                        _xg_re._total_conclusions
                                        / max(1, _xg_re._total_chains))
                                else:
                                    # Not enough data yet — signal "warming
                                    # up" rather than "dormant 0%"
                                    _xg_r_commit = -1.0  # sentinel: suppress
                                                         # commit_rate rendering
                                _xg_r_summary = ""
                                if _xg_re and hasattr(_xg_re, '_last_conclusion'):
                                    _xg_r_summary = str(getattr(_xg_re, '_last_conclusion', ''))[:100]
                                _xg_ls = _language_stats or {}
                                _xg_i_conf = 0.0
                                _xg_concepts = {}
                                _xg_attn_ent = 0.0
                                if msl:
                                    _xg_i_conf = getattr(msl, 'i_confidence', 0.0)
                                    _xg_concepts = {k: v for k, v in getattr(msl, 'concept_confidences', {}).items()}
                                    _xg_attn_ent = getattr(msl, 'attention_entropy', 0.0)
                                _xg_expr = {}
                                if coordinator and hasattr(coordinator, 'expression_composites'):
                                    for _ec_name, _ec_val in coordinator.expression_composites.items():
                                        if _ec_name in ("ART", "MUSIC"):
                                            _xg_expr[_ec_name] = int(getattr(_ec_val, 'fire_count', 0))
                                _xg_meta_style = ""
                                if hasattr(_xg_re, '_dominant_primitive') and _xg_re:
                                    _xg_meta_style = getattr(_xg_re, '_dominant_primitive', '')
                                _xg_recent_words = _xg_ls.get("recent_words", [])

                                # Extract numeric composition level from string like "L8" → 8
                                _xg_comp_lvl_raw = _xg_ls.get("composition_level", "")
                                if not _xg_comp_lvl_raw:
                                    # Fallback: query DB directly if language_stats not yet received
                                    try:
                                        import sqlite3 as _cl_sql
                                        _cl_db = _cl_sql.connect("data/inner_memory.db", timeout=3)
                                        _cl_count = _cl_db.execute("SELECT COUNT(*) FROM vocabulary WHERE confidence >= 0.5").fetchone()[0]
                                        _cl_db.close()
                                        _xg_comp_lvl_raw = f"L{min(9, max(1, _cl_count // 30))}"
                                    except Exception:
                                        _xg_comp_lvl_raw = "L1"
                                _xg_comp_lvl = int(_xg_comp_lvl_raw[1:]) if isinstance(_xg_comp_lvl_raw, str) and _xg_comp_lvl_raw.startswith("L") else (int(_xg_comp_lvl_raw) if isinstance(_xg_comp_lvl_raw, (int, float)) else 0)
                                _xg_ctx = _XPostContext(
                                    session=_xg_session, proxy=_xg_proxy,
                                    api_key=_xg_api_key,
                                    titan_id=_xg_tid,
                                    emotion=neuromodulator_system._current_emotion,
                                    neuromods=_xg_nm, epoch=_xg_epoch,
                                    pi_ratio=_xg_pi,
                                    grounded_words=_xg_words,
                                    llm_url=_xg_inf.get("ollama_cloud_base_url", ""),
                                    llm_key=_xg_inf.get("ollama_cloud_api_key", ""),
                                    llm_model=_xg_inf.get("ollama_cloud_chat_model", "deepseek-v3.1:671b"),
                                    catalysts=list(_x_catalysts),
                                    chi=_xg_chi,
                                    i_confidence=_xg_i_conf,
                                    concept_confidences=_xg_concepts,
                                    reasoning_chains=_xg_r_chains,
                                    reasoning_commit_rate=_xg_r_commit,
                                    recent_chain_summary=_xg_r_summary,
                                    vocab_total=_xg_ls.get("vocab_total", 0),
                                    vocab_producible=_xg_ls.get("vocab_producible", 0),
                                    composition_level=_xg_comp_lvl,
                                    recent_words=_xg_recent_words,
                                    meta_style=_xg_meta_style,
                                    recent_expression=_xg_expr,
                                    drift=_xg_drift,
                                    trajectory=_xg_traj,
                                    attention_entropy=_xg_attn_ent,
                                    social_contagion=(
                                        coordinator._social_contagion_buffer[-1]
                                        if hasattr(coordinator, '_social_contagion_buffer')
                                        and coordinator._social_contagion_buffer
                                        else {}),
                                    # Wisdom & growth enrichment (2026-04-13
                                    # — fixes "novelty at zero" misinterpre-
                                    # tation by giving LLM concrete data on
                                    # ongoing learning).
                                    total_eurekas=int(getattr(
                                        getattr(coordinator, '_meta_engine', None),
                                        '_total_eurekas', 0) or 0),
                                    total_wisdom_saved=int(getattr(
                                        getattr(coordinator, '_meta_engine', None),
                                        '_total_wisdom_saved', 0) or 0),
                                    distilled_count=int(getattr(
                                        getattr(coordinator, 'dreaming', None),
                                        '_distilled_count', 0) or 0),
                                    meta_cgn_signals=int(getattr(
                                        getattr(getattr(coordinator,
                                                        '_meta_engine', None),
                                                '_meta_cgn', None),
                                        '_signals_received', 0) or 0),
                                    # B2 (2026-04-13): prediction_familiarity
                                    # field removed — its disclaimer text still
                                    # mentioned "novelty", giving the LLM a
                                    # phrase hook. EUREKA/wisdom counters are
                                    # stronger evidence of ongoing learning.
                                )
                                # ── POSTING ROTATION: T2/T3 delegates FIRST ──
                                # Check who posted last → if T1, give delegate priority.
                                # This ensures T2/T3 get fair access to posting windows.
                                _xg_delegate_first = False
                                try:
                                    _rot_db = _x_gateway._db()
                                    _rot_last = _rot_db.execute(
                                        "SELECT titan_id FROM actions WHERE action_type='post' "
                                        "AND status IN ('posted','verified') "
                                        "ORDER BY created_at DESC LIMIT 1").fetchone()
                                    _rot_db.close()
                                    if _rot_last and _rot_last["titan_id"] == "T1":
                                        _xg_delegate_first = True
                                except Exception:
                                    pass

                                # ── 2. DELEGATE QUEUE (process BEFORE T1 if rotation) ──
                                _xg_delegate_posted = False
                                if _xg_delegate_first:
                                    try:
                                        _dq_file = "./data/social_delegate_queue.json"
                                        if os.path.exists(_dq_file):
                                            with open(_dq_file) as _dq_f:
                                                _dq_queue = json.load(_dq_f)
                                            if _dq_queue:
                                                _dq_entry = _dq_queue[0]
                                                _dq_titan = _dq_entry.get("titan_id", "T?")
                                                _dq_consumer = f"delegate_{_dq_titan}"
                                                _dq_catalysts = _dq_entry.get("catalysts", [])
                                                if not _dq_catalysts:
                                                    _dq_catalysts = [{"type": _dq_entry.get("catalyst_type", "delegate"),
                                                                      "significance": 0.6,
                                                                      "content": f"Delegate from {_dq_titan}",
                                                                      "data": {}}]
                                                _dq_ctx = _XPostContext(
                                                    session=_xg_session, proxy=_xg_proxy,
                                                    api_key=_xg_api_key,
                                                    titan_id=_dq_titan,
                                                    emotion=_dq_entry.get("emotion", "wonder"),
                                                    neuromods=_dq_entry.get("neuromods", {}),
                                                    epoch=_dq_entry.get("epoch", 0),
                                                    pi_ratio=_dq_entry.get("pi_ratio", 0),
                                                    grounded_words=_dq_entry.get("grounded_words", []),
                                                    llm_url=_xg_inf.get("ollama_cloud_base_url", ""),
                                                    llm_key=_xg_inf.get("ollama_cloud_api_key", ""),
                                                    llm_model=_xg_inf.get("ollama_cloud_chat_model", ""),
                                                    catalysts=_dq_catalysts,
                                                )
                                                _dq_result = _x_gateway.post(_dq_ctx, consumer=_dq_consumer)
                                                if _dq_result.status in ("verified", "posted"):
                                                    _dq_queue.pop(0)
                                                    with open(_dq_file, "w") as _dq_fw:
                                                        json.dump(_dq_queue, _dq_fw)
                                                    logger.info("[SOCIAL] ROTATION: %s posted (delegate-first)",
                                                                _dq_titan)
                                                    _xg_delegate_posted = True
                                                elif _dq_result.status in ("api_failed", "generation_failed",
                                                                           "quality_rejected"):
                                                    # Delegate failed — pop from queue to avoid infinite retry
                                                    _dq_queue.pop(0)
                                                    with open(_dq_file, "w") as _dq_fw:
                                                        json.dump(_dq_queue, _dq_fw)
                                                    logger.warning("[SOCIAL] ROTATION: %s delegate FAILED (%s) — "
                                                                   "popped, falling through to T1",
                                                                   _dq_titan, _dq_result.status)
                                    except Exception as _dq_rot_err:
                                        logger.warning("[SOCIAL] Rotation delegate error: %s", _dq_rot_err)

                                # ── 1. T1 OWN POST (skip if delegate already posted) ──
                                if _xg_delegate_posted:
                                    _xg_result = type('R', (), {'status': 'too_soon',
                                                                'reason': 'delegate posted first'})()
                                else:
                                    _xg_result = _x_gateway.post(_xg_ctx, consumer="spirit_worker")
                                logger.info("[SOCIAL] post() → %s (catalysts=%d, delegate_first=%s)",
                                            _xg_result.status, len(_x_catalysts),
                                            _xg_delegate_first)

                                if _xg_result.status in ("verified", "posted"):
                                    _x_catalysts.clear()
                                    logger.info("[SOCIAL] T1 posted via gateway: %s (id=%s)",
                                                _xg_result.status, getattr(_xg_result, 'tweet_id', ''))
                                    if msl:
                                        msl.signal_action("external")
                                    # TimeChain: social post → episodic fork
                                    _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
                                        "fork": "episodic", "thought_type": "episodic",
                                        "source": "social_post",
                                        "content": {
                                            "action": "post",
                                            "tweet_id": getattr(_xg_result, 'tweet_id', ''),
                                            "titan_id": _xg_tid,
                                            "text_hash": hashlib.sha256(
                                                getattr(_xg_result, 'text', '').encode()).hexdigest()[:16],
                                        },
                                        "significance": 0.5, "novelty": 0.5, "coherence": 0.5,
                                        "tags": ["social", "x_post", _xg_tid],
                                        "db_ref": f"social_x:{getattr(_xg_result, 'tweet_id', '')}",
                                        "neuromods": dict(_cached_neuromod_state),
                                        "chi_available": _cached_chi_state.get("total", 0.5),
                                        "attention": 0.5, "i_confidence": 0.5, "chi_coherence": 0.3,
                                    })
                                elif _xg_result.status not in ("disabled", "too_soon",
                                        "no_catalyst", "hourly_limit", "daily_limit",
                                        "pending_exists", "consumer_blocked",
                                        "circuit_breaker"):
                                    logger.warning("[SOCIAL] Post failed: %s — %s",
                                                   _xg_result.status, _xg_result.reason)

                                # Open social window
                                _xg_skip = ("too_soon", "disabled", "circuit_breaker")
                                if _xg_result.status not in _xg_skip or _xg_delegate_posted:
                                    _xg_social_window = True

                                # ── 2b. DELEGATE QUEUE (skip if rotation already handled) ──
                                if not _xg_delegate_posted:
                                    try:
                                        _dq_file = "./data/social_delegate_queue.json"
                                        if os.path.exists(_dq_file):
                                            with open(_dq_file) as _dq_f:
                                                _dq_queue = json.load(_dq_f)
                                            if _dq_queue:
                                                _dq_entry = _dq_queue[0]
                                                _dq_titan = _dq_entry.get("titan_id", "T?")
                                                _dq_consumer = f"delegate_{_dq_titan}"
                                                _dq_catalysts = _dq_entry.get("catalysts", [])
                                                if not _dq_catalysts:
                                                    _dq_catalysts = [{"type": _dq_entry.get("catalyst_type", "delegate"),
                                                                      "significance": 0.6,
                                                                      "content": f"Delegate from {_dq_titan}",
                                                                      "data": {}}]
                                                _dq_ctx = _XPostContext(
                                                    session=_xg_session, proxy=_xg_proxy,
                                                    api_key=_xg_api_key,
                                                    titan_id=_dq_titan,
                                                    emotion=_dq_entry.get("emotion", "wonder"),
                                                    neuromods=_dq_entry.get("neuromods", {}),
                                                    epoch=_dq_entry.get("epoch", 0),
                                                    pi_ratio=_dq_entry.get("pi_ratio", 0),
                                                    grounded_words=_dq_entry.get("grounded_words", []),
                                                    llm_url=_xg_inf.get("ollama_cloud_base_url", ""),
                                                    llm_key=_xg_inf.get("ollama_cloud_api_key", ""),
                                                    llm_model=_xg_inf.get("ollama_cloud_chat_model", ""),
                                                    catalysts=_dq_catalysts,
                                                )
                                                _dq_result = _x_gateway.post(_dq_ctx, consumer=_dq_consumer)
                                                if _dq_result.status in ("verified", "posted"):
                                                    logger.info("[DELEGATE] %s posted via gateway: %s",
                                                                _dq_titan, _dq_result.tweet_id)
                                                elif _dq_result.status not in ("disabled", "too_soon",
                                                        "hourly_limit", "daily_limit", "pending_exists"):
                                                    logger.warning("[DELEGATE] %s post failed: %s — %s",
                                                                   _dq_titan, _dq_result.status, _dq_result.reason)
                                                _dq_queue = _dq_queue[1:]
                                                _dq_tmp = _dq_file + ".tmp"
                                                with open(_dq_tmp, "w") as _dq_wf:
                                                    json.dump(_dq_queue, _dq_wf)
                                                os.replace(_dq_tmp, _dq_file)
                                    except Exception as _dq_err:
                                        if _msl_tick_count % 500 == 0:
                                            logger.warning("[DELEGATE] Queue error: %s", _dq_err)

                                # ── 3. MENTION DISCOVERY + REPLY ──
                                # Decoupled from posting window — runs every 10th
                                # social cycle (~15 min) OR when posting window opens.
                                # Mentions must be checked independently so Titans can
                                # see and reply even when not actively posting.
                                _xg_check_mentions = (
                                    _xg_social_window or
                                    _msl_tick_count % 300 == 0  # every ~300 ticks
                                )
                                if _xg_check_mentions:
                                    try:
                                        # Gateway mode: empty titan_id → discover
                                        # mentions for ALL Titans (T1/T2/T3).
                                        # Each mention's titan_id is set by ownership
                                        # (reply-to-own-post) in discover_mentions().
                                        _xg_base = _XBaseContext(
                                            session=_xg_session, proxy=_xg_proxy,
                                            api_key=_xg_api_key, titan_id="")
                                        _xg_mentions = _x_gateway.discover_mentions(
                                            _xg_base, consumer="spirit_worker",
                                            grounded_words=_xg_words)
                                        _xg_replies_cfg = _xg_full.get("social_x", {}).get("replies", {})
                                        _xg_max_replies = _xg_replies_cfg.get("max_replies_per_cycle", 3)
                                        _xg_reply_count = 0
                                        for _xg_m in _xg_mentions[:_xg_max_replies]:
                                            # P4+v2: CGN social policy inference via /dev/shm client
                                            _xg_cgn_action = {}
                                            try:
                                                from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient as _CGNClient
                                                _xg_cgn_client = _CGNClient(
                                                    "social", state_dir="data/cgn")
                                                _xg_cgn_action = _xg_cgn_client.infer_action(
                                                    sensory_ctx={
                                                        "epoch": consciousness.get("latest_epoch", {}).get("epoch_id", 0) if isinstance(consciousness, dict) else 0,
                                                        "neuromods": _xg_nm,
                                                    },
                                                    features={
                                                        "familiarity": min(1.0, _xg_m.get("relevance_score", 0.3)),
                                                        "interaction_count": 0,
                                                        "social_valence": 0.0,
                                                        "mention_count": 1,
                                                    })
                                            except Exception:
                                                pass
                                            _xg_rctx = _XReplyContext(
                                                session=_xg_session, proxy=_xg_proxy,
                                                api_key=_xg_api_key,
                                                titan_id=_xg_m["titan_id"],
                                                reply_to_tweet_id=_xg_m["tweet_id"],
                                                mention_text=_xg_m["text"][:200],
                                                mention_user=_xg_m["author_handle"],
                                                emotion=neuromodulator_system._current_emotion,
                                                neuromods=_xg_nm,
                                                grounded_words=_xg_words,
                                                llm_url=_xg_inf.get("ollama_cloud_base_url", ""),
                                                llm_key=_xg_inf.get("ollama_cloud_api_key", ""),
                                                llm_model=_xg_inf.get("ollama_cloud_chat_model", ""),
                                                cgn_action=_xg_cgn_action,
                                            )
                                            _xg_rr = _x_gateway.reply(_xg_rctx, consumer="spirit_worker")
                                            if _xg_rr.status in ("posted", "verified"):
                                                _x_gateway.mark_mention_replied(
                                                    _xg_m["tweet_id"], _xg_rr.tweet_id)
                                                _xg_reply_count += 1
                                                logger.info("[SOCIAL] Replied to @%s (score=%.2f): %s",
                                                            _xg_m["author_handle"],
                                                            _xg_m["relevance_score"],
                                                            _xg_rr.tweet_id)
                                            elif _xg_rr.status in ("hourly_limit", "daily_limit", "too_soon"):
                                                break
                                        if _xg_reply_count > 0:
                                            logger.info("[SOCIAL] Window: %d replies", _xg_reply_count)
                                    except Exception as _xg_win_err:
                                        logger.warning("[SOCIAL] Window error: %s", _xg_win_err)
                        except Exception as _xg_err:
                            logger.warning("[SOCIAL] Gateway post error: %s", _xg_err)

                except Exception as _xg_outer_err:
                    logger.warning("[SOCIAL] Outer error: %s", _xg_outer_err)
                finally:
                    # Keep max 10 catalysts (prevent unbounded growth) — runs
                    # even if the social posting block raised an exception.
                    if len(_x_catalysts) > 10:
                        _x_catalysts = _x_catalysts[-10:]

        # ── Process message if one was received ──
        if msg is None:
            continue

        msg_type = msg.get("type", "")

        if msg_type == "MODULE_SHUTDOWN":
            logger.info("[SpiritWorker] Shutdown: %s", msg.get("payload", {}).get("reason"))
            break

        # ── SAVE_NOW ──────────────────────────────────────────
        # Graceful checkpoint request from Guardian (or admin endpoint) BEFORE
        # a planned restart. Runs all stateful save methods and emits SAVE_DONE
        # so the requester can proceed knowing in-memory state is on disk.
        # Pattern from `feedback_prefer_hot_reload.md` — module-restart safety.
        if msg_type == "SAVE_NOW":
            _save_payload = msg.get("payload", {})
            _save_rid = msg.get("rid") or _save_payload.get("request_id", "")
            _t_save_start = time.time()
            _saved: list = []
            _save_errs: list = []

            def _try_save(label: str, fn):
                try:
                    fn()
                    _saved.append(label)
                except Exception as _e:
                    _save_errs.append(f"{label}:{_e}")

            if msl is not None:
                _try_save("msl", msl.save_all)
            if coordinator is not None and getattr(coordinator, "dreaming", None) is not None:
                _try_save("dreaming", coordinator.dreaming.save_state)
            if _reasoning_engine is not None:
                _try_save("reasoning_engine", _reasoning_engine.save_all)
            try:
                _meta_engine_local = getattr(coordinator, "_meta_engine", None) if coordinator else None
            except Exception:
                _meta_engine_local = None
            if _meta_engine_local is not None and hasattr(_meta_engine_local, "save_all"):
                _try_save("meta_engine", _meta_engine_local.save_all)
            if neural_nervous_system is not None and hasattr(neural_nervous_system, "save_all"):
                _try_save("neural_ns", neural_nervous_system.save_all)
            if pi_monitor is not None and hasattr(pi_monitor, "_save_state"):
                _try_save("pi_monitor", pi_monitor._save_state)

            _save_dur_ms = int((time.time() - _t_save_start) * 1000)
            logger.info("[SpiritWorker] SAVE_NOW: saved=%s errors=%s (%dms)",
                        _saved, _save_errs, _save_dur_ms)
            try:
                # Use make_msg (not raw dict) so arch_map scanner detects this
                # producer. Fix 2026-04-15 per DEAF-EARS-CLEANUP item — SAVE_DONE
                # was being reported as a deaf-ear (0 senders) in arch_map audit
                # despite this producer existing since 2026-04-13. The scanner
                # only inspects make_msg() call sites; raw dict publishes were
                # invisible.
                from ..bus import make_msg
                send_queue.put(make_msg(
                    "SAVE_DONE",
                    name,
                    "guardian",
                    {
                        "module": name,
                        "request_id": _save_rid,
                        "success": len(_save_errs) == 0,
                        "saved": _saved,
                        "errors": _save_errs,
                        "duration_ms": _save_dur_ms,
                    },
                ))
            except Exception as _send_err:
                logger.warning("[SpiritWorker] SAVE_DONE emit failed: %s",
                               _send_err)
            continue

        # Listen for Body/Mind state broadcasts
        if msg_type == "BODY_STATE":
            body_state = msg.get("payload", body_state)
            logger.debug("[SpiritWorker] Received BODY_STATE: center_dist=%.3f",
                         body_state.get("center_dist", 0))

        elif msg_type == "MIND_STATE":
            mind_state = msg.get("payload", mind_state)
            logger.debug("[SpiritWorker] Received MIND_STATE: center_dist=%.3f",
                         mind_state.get("center_dist", 0))

        elif msg_type == "MODULE_CRASHED":
            # I-004 fix: Guardian publishes MODULE_CRASHED when a module
            # exceeds restart thresholds. Previously had no consumer ⇒ silent
            # crash loops weren't surfaced. Now logged at ERROR with explicit
            # context so they show up in arch_map errors and ops monitoring.
            _mc_payload = msg.get("payload", {})
            _mc_module = _mc_payload.get("module", "?")
            _mc_reason = _mc_payload.get("reason", "?")
            _mc_restarts = _mc_payload.get("restarts", "?")
            logger.error(
                "[SpiritWorker] MODULE_CRASHED alert: module=%s reason=%s restarts=%s "
                "(Guardian disabled the module — investigate logs/RSS limits)",
                _mc_module, _mc_reason, _mc_restarts,
            )

        elif msg_type == "RATE_LIMIT":
            # I-004 fix: interface_advisor / v5_core publish RATE_LIMIT when a
            # source component (mind/body/agency) exceeds rate windows. The
            # original IQL self-regulation design is deferred (separate
            # work — see interface_advisor.py:10-12), but at minimum we now
            # log the event explicitly instead of dropping it on the floor.
            _rl_payload = msg.get("payload", {})
            logger.warning(
                "[SpiritWorker] RATE_LIMIT received: source=%s rate=%s window=%s — %s",
                _rl_payload.get("source", "?"),
                _rl_payload.get("rate", "?"),
                _rl_payload.get("window", "?"),
                _rl_payload.get("action", "no_action_specified"),
            )

        elif msg_type == "ACTION_RESULT":
            # Step 7: Agency completed an action — record outcome for impulse threshold learning
            if impulse_engine:
                try:
                    payload = msg.get("payload", {})
                    impulse_id = payload.get("impulse_id")
                    trinity_before = payload.get("trinity_before")
                    trinity_after = payload.get("trinity_after")
                    if impulse_id and trinity_before and trinity_after:
                        impulse_engine.record_outcome(impulse_id, trinity_before, trinity_after)
                        # rFP β Stage 2 Phase 2c: IMPULSE event hook
                        # Trinity delta as reward proxy — positive when action
                        # moved Trinity in a productive direction. Discrete signal
                        # complementing the dense DA-ACh neuromod stream.
                        if neural_nervous_system:
                            try:
                                _imp_delta = (
                                    (float(trinity_after.get("body", 0.5)) - float(trinity_before.get("body", 0.5))) * 0.4
                                    + (float(trinity_after.get("mind", 0.5)) - float(trinity_before.get("mind", 0.5))) * 0.3
                                    + (float(trinity_after.get("spirit", 0.5)) - float(trinity_before.get("spirit", 0.5))) * 0.3
                                )
                                if abs(_imp_delta) > 0.05:
                                    neural_nervous_system.record_outcome(
                                        reward=max(-1.0, min(1.0, _imp_delta * 2.0)),
                                        program="IMPULSE",
                                        source="impulse_engine.outcome")
                            except Exception as _imp_err:
                                if hash(("imp_hook", _imp_err.__class__.__name__)) % 100 == 0:
                                    logger.warning("[NS-Hook] IMPULSE reward failed: %s", _imp_err)
                except Exception as e:
                    logger.error("[SpiritWorker] ACTION_RESULT handling error: %s", e, exc_info=True)

            # Brain P1c: Record experience in ex_mem (action outcome learning)
            if ex_mem:
                try:
                    payload = msg.get("payload", {})
                    task_type = payload.get("helper", payload.get("action", "unknown"))
                    score = payload.get("score", payload.get("assessment", {}).get("score", 0.0))
                    success = score > 0.5 if isinstance(score, (int, float)) else False
                    # Capture hormonal delta as natural scoring
                    h_delta = {}
                    if neural_nervous_system:
                        for h_name, h in neural_nervous_system._hormonal._hormones.items():
                            h_delta[h_name] = round(h.level, 3)
                    # Inner state snapshot
                    inner_sv = []
                    if consciousness and consciousness.get("latest_epoch"):
                        inner_sv = consciousness["latest_epoch"].get("state_vector", [])[:65]
                    ex_mem.record_experience(
                        task_type=task_type,
                        intent_hormones=h_delta,
                        inner_before=inner_sv,
                        outcome_score=float(score) if isinstance(score, (int, float)) else 0.0,
                        success=success,
                        epoch_id=consciousness.get("latest_epoch", {}).get("epoch_id", 0) if consciousness else 0,
                    )
                    # Record as episodic event if significant
                    if episodic_mem and score and float(score) > 0.7:
                        episodic_mem.record_episode(
                            "action_completed",
                            f"{task_type} scored {score}",
                            felt_state=inner_sv,
                            hormonal_snapshot=h_delta,
                            epoch_id=consciousness.get("latest_epoch", {}).get("epoch_id", 0) if consciousness else 0,
                            significance=float(score),
                        )
                except Exception as e:
                    logger.warning("[SpiritWorker] ex_mem recording error: %s", e)

            # Experience Orchestrator: Record action outcome across all domains
            if exp_orchestrator:
                try:
                    payload = msg.get("payload", {})
                    _eo_helper = payload.get("helper", payload.get("action", "unknown"))
                    _eo_domain = infer_domain(_eo_helper)
                    _eo_score = payload.get("score",
                                 payload.get("assessment", {}).get("score",
                                 payload.get("outcome_score", 0.0)))
                    if isinstance(_eo_score, str):
                        try:
                            _eo_score = float(_eo_score)
                        except (ValueError, TypeError):
                            _eo_score = 0.0
                    _eo_hormones = {}
                    if neural_nervous_system and neural_nervous_system._hormonal_enabled:
                        _eo_hormones = {h: round(v.level, 3)
                                        for h, v in neural_nervous_system._hormonal._hormones.items()}
                    _eo_inner = []
                    if consciousness and isinstance(consciousness, dict):
                        _eo_le = consciousness.get("latest_epoch", {})
                        _eo_inner = _eo_le.get("state_vector", [])[:130] if isinstance(_eo_le, dict) else []
                    _eo_plugin = exp_orchestrator._plugins.get(_eo_domain)
                    _eo_perception = _eo_plugin.extract_perception_key({
                        "inner_state": _eo_inner,
                        "felt_tensor": _eo_inner[:65],
                        "inner_body": _eo_inner[:5],
                        "inner_mind": _eo_inner[5:20],
                        "inner_spirit": _eo_inner[20:65],
                        "intent_hormones": _eo_hormones,
                        "hormonal_snapshot": _eo_hormones,
                        "spatial_features": _eo_inner[65:95] if len(_eo_inner) > 65 else [],
                    }) if _eo_plugin else _eo_inner[:10]
                    _eo_epoch = consciousness.get("latest_epoch", {}).get("epoch_id", 0) if consciousness and isinstance(consciousness, dict) else 0
                    exp_orchestrator.record_outcome(
                        domain=_eo_domain,
                        perception_features=_eo_perception,
                        inner_state_132d=_eo_inner,
                        hormonal_snapshot=_eo_hormones,
                        action_taken=_eo_helper,
                        outcome_score=float(_eo_score),
                        context={"helper": _eo_helper, "success": _eo_score > 0.5},
                        epoch_id=_eo_epoch,
                        is_dreaming=_shared_is_dreaming,
                    )
                except Exception as _eo_err:
                    logger.warning("[ExperienceOrch] ACTION_RESULT record error: %s", _eo_err)

        elif msg_type == "MEDITATION_COMPLETE":
            # Meditation completed by memory_worker (via v5_core routing)
            _med_payload = msg.get("payload", {})
            _meditation_tracker["in_meditation"] = False
            _meditation_tracker["last_epoch"] = epoch_id or 0
            _meditation_tracker["last_ts"] = time.time()
            _meditation_tracker["count"] += 1
            _meditation_tracker["count_since_nft"] += 1

            # Watchdog — record gap + distillation health (F7 tracking)
            if _med_watchdog is not None:
                try:
                    _med_watchdog.record_meditation(
                        last_ts=_meditation_tracker["last_ts"],
                        promoted=int(_med_payload.get("promoted", 0) or 0),
                    )
                except Exception as _wd_rec_err:
                    logger.warning("[MeditationWatchdog] record_meditation error: %s", _wd_rec_err)

            logger.info(
                "[MEDITATION] Cycle #%d COMPLETE (trigger=%s) — promoted=%d pruned=%d",
                _meditation_tracker["count"],
                _med_payload.get("trigger", "?"),
                _med_payload.get("promoted", 0),
                _med_payload.get("pruned", 0))

            # Re-verify constitution at each meditation (M1.7)
            try:
                from titan_plugin.utils.directive_signer import verify_directives
                if not verify_directives():
                    logger.critical("[MEDITATION] Constitution verification FAILED!")
            except Exception:
                pass

            # Write backup trigger file for main process to pick up
            # (spirit_worker is a separate process — backup runs in TitanPlugin's asyncio loop)
            # F5 detection (rFP Phase 2): on write failure, emit MEDITATION_HEALTH_ALERT
            # so watchdog + /v4/meditation/health surface the break in the
            # meditation→backup chain. Per rFP §4 F5 detection is MEDIUM severity.
            try:
                import json as _json_mod
                _trigger_path = os.path.join("data", "backup_trigger.json")
                _json_mod.dumps(_med_payload)  # validate serializable
                # Atomic write via tmpfile + os.replace — avoids half-written
                # trigger file if process dies mid-write (older code wrote directly).
                _trigger_tmp = _trigger_path + ".tmp"
                with open(_trigger_tmp, "w") as _tf:
                    _json_mod.dump({
                        "payload": _med_payload,
                        "meditation_count": _meditation_tracker["count"],
                        "ts": time.time(),
                    }, _tf)
                os.replace(_trigger_tmp, _trigger_path)
                logger.info("[MEDITATION] Backup trigger written for meditation #%d",
                            _meditation_tracker["count"])
            except Exception as _bt_err:
                logger.error("[MEDITATION] F5 — Backup trigger write failed: %s", _bt_err,
                             exc_info=True)
                _send_msg(send_queue, "MEDITATION_HEALTH_ALERT", name, "core", {
                    "severity": "MEDIUM",
                    "failure_mode": "F5_TRIGGER_FILE_WRITE",
                    "detail": f"Trigger file write failed: {_bt_err}",
                    "diagnostic": {
                        "trigger_path": _trigger_path,
                        "meditation_count": _meditation_tracker["count"],
                        "error": str(_bt_err),
                    },
                    "ts": time.time(),
                    "titan_id": _med_watchdog.titan_id if _med_watchdog else "?",
                })

            # ── Register meditation as catalyst for social posting ──
            if _x_gateway:
                _x_catalysts.append({
                    "type": "dream_summary", "significance": 0.7,
                    "content": "Meditation #%d: %d memories crystallized, %d pruned" % (
                        _meditation_tracker["count"],
                        _med_payload.get("promoted", 0),
                        _med_payload.get("pruned", 0)),
                    "data": _med_payload,
                })
                logger.info("[SOCIAL] Meditation catalyst registered (dream_summary)")

        elif msg_type == "EPOCH_TICK":
            # Epoch boundary — run consciousness cycle + publish immediately
            if consciousness:
                _run_consciousness_epoch(consciousness, body_state, mind_state, config,
                                         outer_state=outer_state)
                last_consciousness_tick = time.time()
                # rFP #2: compose TITAN_SELF + train V5 (publish path gated by flag)
                try:
                    from titan_plugin.modules.spirit_loop import (
                        compose_and_emit_titan_self, _post_epoch_v5_filter_down,
                    )
                    _titan_self = compose_and_emit_titan_self(
                        send_queue, name, consciousness, config,
                    )
                    _post_epoch_v5_filter_down(
                        send_queue, name, filter_down_v5, _titan_self, config,
                    )
                except Exception as _ts_err:
                    logger.debug("[SpiritWorker] TITAN_SELF/V5 (EPOCH_TICK) error: %s", _ts_err)
                # π-Heartbeat: observe from EPOCH_TICK path (preferred — has fresh curvature)
                if pi_monitor:
                    _latest = consciousness.get("latest_epoch", {})
                    _curv = _latest.get("curvature", 0.0)
                    _eid = _latest.get("epoch_id", 0)
                    _is_pi = 2.9 < _curv < 3.3
                    logger.info(
                        "[π-Heartbeat] Observing epoch %d: curvature=%.3f %s "
                        "(streak: π=%d zero=%d, clusters=%d)",
                        _eid, _curv,
                        "★π★" if _is_pi else "·",
                        pi_monitor.current_pi_streak, pi_monitor.current_zero_streak,
                        pi_monitor.developmental_age)
                    pi_monitor.observe(_curv, _eid)
                    _last_pi_observed_epoch = _eid
            # GROUND_UP + oBody→oMind → moved to body tick (D2+D3) — runs at 7.83 Hz Schumann
            # 132D state caching happens in the periodic epoch handler (G4 block above)

            # ── Neuromod snapshot (Tier 3 deep epoch log — evaluation moved to Tier 2) ──
            # Neuromod evaluation now runs at body-clock rate (3.45s) in Tier 2.
            # Here we just log the current state for consciousness epoch record.
            try:
                if neuromodulator_system:
                    _nm_emotion = neuromodulator_system._current_emotion
                    _nm_conf = neuromodulator_system._emotion_confidence
                    _nm_mod = neuromodulator_system.get_modulation()
                    _nm_levels = {n: m.level for n, m in neuromodulator_system.modulators.items()}
                    logger.info(
                        "[Neuromod→NS] lr=%.2f thresh=%.2f sensory=%.2f accum=%.2f motiv=%.2f",
                        _nm_mod.get("learning_rate_gain", 1.0),
                        _nm_mod.get("global_threshold_raise", 1.0),
                        _nm_mod.get("sensory_gain", 1.0),
                        _nm_mod.get("accumulation_rate_gain", 1.0),
                        _nm_mod.get("intrinsic_motivation", 1.0))
                    logger.info(
                        "[Neuromod] DA=%.2f 5HT=%.2f NE=%.2f ACh=%.2f End=%.2f GABA=%.2f | emotion=%s(%.0f%%)",
                        _nm_levels.get("DA", 0), _nm_levels.get("5HT", 0),
                        _nm_levels.get("NE", 0), _nm_levels.get("ACh", 0),
                        _nm_levels.get("Endorphin", 0), _nm_levels.get("GABA", 0),
                        _nm_emotion, _nm_conf * 100)
            except Exception as _nme:
                logger.info("[SpiritWorker] Neuromodulator snapshot error: %s", _nme)

            # ── META-REASONING tick (one step per epoch) ──────────────────
            _meta_dreaming = consciousness.get("dreaming", {}).get("is_dreaming", False) if isinstance(consciousness, dict) else False
            if meta_engine and not _meta_dreaming:
                try:
                    _meta_gates = meta_engine.gates_passed(pi_monitor, _reasoning_engine, coordinator)
                    if not _meta_gates and _reasoning_engine and _reasoning_engine._total_chains % 10 == 1:
                        logger.info("[META] Gates not passed yet: chains=%d, pi=%d",
                                    _reasoning_engine._total_chains,
                                    pi_monitor.developmental_age if pi_monitor else 0)
                    if _meta_gates:
                        _meta_sv = consciousness.get("latest_epoch", {}).get("state_vector", [])
                        if hasattr(_meta_sv, 'tolist'):
                            _meta_sv = _meta_sv.tolist()
                        _meta_nm = {}
                        if neuromodulator_system:
                            _meta_nm = {n: m.level for n, m in neuromodulator_system.modulators.items()}

                        # ── A-finish: Subsystem cache refresh (mirror of site #1) ──
                        # This is the second meta_engine.tick() site. Both must
                        # check + dispatch the cache refresh, otherwise chains
                        # routed through this path never see populated signals.
                        try:
                            if (meta_engine.is_subsystem_cache_stale()
                                    and not meta_engine.is_subsystem_cache_pending()):
                                _send_msg(send_queue, "TIMECHAIN_QUERY", name, "timechain", {
                                    "limit": 50,
                                })
                                _send_msg(send_queue, "CONTRACT_LIST", name, "timechain", {
                                    "status": "active",
                                })
                                meta_engine.mark_subsystem_cache_pending()
                                logger.info("[META] Subsystem cache refresh dispatched (site B)")
                        except Exception as _ssrefresh_err_b:
                            logger.warning(
                                "[META] Subsystem cache refresh dispatch failed (site B): %s",
                                _ssrefresh_err_b)

                        _meta_result = meta_engine.tick(
                            state_132d=_meta_sv,
                            neuromods=_meta_nm,
                            reasoning_engine=_reasoning_engine,
                            chain_archive=chain_archive,
                            meta_wisdom=meta_wisdom,
                            ex_mem=ex_mem,
                            meta_autoencoder=meta_autoencoder,
                        )
                        if _meta_result and _meta_result.get("action") not in ("IDLE", "WAITING", None):
                            logger.info("[META] %s.%s — conf=%.2f chain=%d/%d",
                                        _meta_result.get("primitive", "?"),
                                        _meta_result.get("sub_mode", ""),
                                        _meta_result.get("confidence", 0),
                                        _meta_result.get("chain_length", 0),
                                        _meta_result.get("max_steps", 20))

                            # CGN Phase 4c: Send concluded chains to language_worker
                            # for vocabulary association building + CGN reward
                            if _meta_result.get("action") == "CONCLUDE":
                                _meta_conf = _meta_result.get("confidence", 0)
                                _meta_chain = _meta_result.get("chain", [])
                                _meta_reward = _meta_result.get("reward", 0)
                                _send_msg(send_queue, "META_LANGUAGE_RESULT", name, "language", {
                                    "chain_id": _meta_result.get("chain_id", -1),
                                    "chain_length": _meta_result.get("chain_length", 0),
                                    "confidence": _meta_conf,
                                    "reward": _meta_reward,
                                    "primitives": [s.get("primitive", "") for s in _meta_chain] if _meta_chain else [],
                                    "sub_modes": [s.get("sub_mode", "") for s in _meta_chain] if _meta_chain else [],
                                    "epoch": consciousness.get("latest_epoch", {}).get("epoch_id", 0) if consciousness else 0,
                                })
                                # TUNING-012 v2 Sub-phase C (R1): mirror the chain
                                # outcome write from the primary tick site so
                                # contracts see ALL concluded chains regardless
                                # of which loop pathway produced them.
                                _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
                                    "fork": "meta", "thought_type": "meta",
                                    "source": "meta_reasoning",
                                    "content": {
                                        "event": "META_CHAIN_CONCLUDE",
                                        "reward": _meta_reward,
                                        "chain_length": _meta_result.get("chain_length", 0),
                                        "confidence": _meta_conf,
                                        "chain_template": _meta_result.get("chain_template", ""),
                                        "task_success": _meta_result.get("task_success", 0.0),
                                        "primitives_used": _meta_result.get("primitives_used", []),
                                        "domain": _meta_result.get("domain", "general"),
                                        "unique_primitives": _meta_result.get("unique_primitives", 0),
                                    },
                                    "significance": min(1.0, _meta_reward),
                                    "novelty": 0.6, "coherence": _meta_conf,
                                    "tags": ["meta_reasoning", "conclude", "chain_outcome"],
                                    "neuromods": dict(_cached_neuromod_state) if _cached_neuromod_state else {},
                                    "chi_available": _cached_chi_state.get("total", 0.5) if _cached_chi_state else 0.5,
                                    "attention": 0.5, "i_confidence": 0.5, "chi_coherence": 0.3,
                                })
                except Exception as _me_err:
                    logger.warning("[META] Tick error: %s", _me_err, exc_info=True)

            # ── EXPRESSION Composite Evaluation (SPEAK, ART, MUSIC, SOCIAL) ──
            try:
                if expression_manager and neural_nervous_system and neural_nervous_system._hormonal_enabled:
                    _expr_hormones = {
                        h_name: h.level
                        for h_name, h in neural_nervous_system._hormonal._hormones.items()
                    }
                    _expr_vocab_conf = 1.0  # TODO: compute from vocabulary stats
                    _expr_dev_age = pi_monitor.developmental_age if pi_monitor else 0
                    _expr_fired = expression_manager.evaluate_all(
                        _expr_hormones,
                        vocabulary_confidence=_expr_vocab_conf,
                        developmental_age=_expr_dev_age,
                        hormonal_system=neural_nervous_system._hormonal if neural_nervous_system._hormonal_enabled else None,
                    )
                    # If Tier 2 flagged SPEAK, inject a synthetic fire event
                    # (hormones may be depleted by Tier 2, but SPEAK was viable)
                    logger.info("[T1:SPEAK-CHECK] pending=%s fired_count=%d",
                                _t2_speak_pending, len(_expr_fired))
                    if _t2_speak_pending:
                        _speak_in_fired = any(f["composite"] == "SPEAK" for f in _expr_fired)
                        if not _speak_in_fired:
                            _speak_comp = expression_manager.composites.get("SPEAK")
                            _expr_fired.append({
                                "composite": "SPEAK",
                                "urge": _speak_comp._last_urge if _speak_comp else 0.5,
                                "intensity": 1.0,
                                "dominant_hormone": "CREATIVITY",
                                "action_helper": "speak",
                                "total_consumption": 0,
                            })
                        _t2_speak_pending = False

                    # ── Social Pressure: accumulate SOCIAL fires ──
                    if _social_pressure_meter:
                        for _spf in _expr_fired:
                            if _spf["composite"] == "SOCIAL":
                                _social_pressure_meter.on_social_fire(
                                    _spf.get("urge", 1.0))

                    for _ef in _expr_fired:
                        if _ef["composite"] == "SPEAK":
                            # SPEAK → send to language_worker via bus (Phase 2)
                            _speak_sv = consciousness.get("latest_epoch", {}).get("state_vector", []) if consciousness else []
                            if hasattr(_speak_sv, 'to_list'):
                                _speak_sv = _speak_sv.to_list()
                            _speak_sv = list(_speak_sv) if _speak_sv else []
                            if len(_speak_sv) >= 65:
                                # ── SPEAK_REQUEST → language_worker (Phase 2) ──
                                # Build experience bias (serializable)
                                _speak_bias_data = None
                                if exp_orchestrator:
                                    try:
                                        _sb_plugin = exp_orchestrator._plugins.get("language")
                                        if _sb_plugin:
                                            _sb_perc = _sb_plugin.extract_perception_key({
                                                "inner_state": _speak_sv,
                                                "felt_tensor": _speak_sv[:65],
                                                "inner_body": _speak_sv[:5],
                                                "inner_mind": _speak_sv[5:20] if len(_speak_sv) >= 20 else [],
                                                "inner_spirit": _speak_sv[20:65] if len(_speak_sv) >= 65 else [],
                                                "hormonal_snapshot": _t2_hormonal_state,
                                                "intent_hormones": _t2_hormonal_state,
                                                "spatial_features": [],
                                            })
                                            _sb_bias = exp_orchestrator.get_experience_bias(
                                                domain="language",
                                                current_perception=_sb_perc,
                                                current_inner_state=_speak_sv,
                                                candidate_actions=["self_express"],
                                            )
                                            if _sb_bias and _sb_bias.confidence >= 0.2:
                                                _speak_bias_data = {
                                                    "optimal_inner_state": list(_sb_bias.optimal_inner_state) if _sb_bias.optimal_inner_state is not None else None,
                                                    "confidence": _sb_bias.confidence,
                                                    "domain": "language",
                                                }
                                    except Exception:
                                        pass

                                # Concept confidences
                                _speak_concept_conf = outer_state.get("_msl_concept_confidences")
                                if _speak_concept_conf and msl:
                                    _speak_concept_conf = dict(_speak_concept_conf)
                                    _speak_concept_conf["I"] = msl.get_i_confidence()

                                # DA info for exploration gate
                                _da_level = _nm_levels.get("DA", 0.5) if '_nm_levels' in dir() else 0.5
                                _da_setpoint = 0.5
                                if neuromodulator_system:
                                    _da_mod = neuromodulator_system.modulators.get("DA")
                                    if _da_mod:
                                        _da_setpoint = getattr(_da_mod, 'setpoint', 0.5)

                                _ch_epoch = consciousness.get("latest_epoch", {}).get("epoch_id", 0) if consciousness and isinstance(consciousness, dict) else 0

                                # MSL attention for CGN sensory context
                                _speak_msl_attn = None
                                if msl:
                                    _speak_msl_attn = msl.get_attention_weights_for_kin()

                                # Social contagion context for post enrichment
                                _speak_social_ctx = None
                                _sc_buf = getattr(coordinator,
                                    '_social_contagion_buffer', [])
                                if _sc_buf:
                                    # Include most recent contagion event
                                    _sc_latest = _sc_buf[-1]
                                    _speak_social_ctx = {
                                        "contagion_type": _sc_latest.get(
                                            "contagion_type"),
                                        "author": _sc_latest.get("author", ""),
                                        "topic": _sc_latest.get("topic", ""),
                                        "felt_summary": _sc_latest.get(
                                            "felt_summary", ""),
                                    }

                                # Include reasoning plan for L9 composition (Phase 4)
                                _speak_reasoning = None
                                if (_reasoning_result
                                        and _reasoning_result.get("action") == "COMMIT"
                                        and _reasoning_result.get("confidence", 0) >= 0.5):
                                    _speak_reasoning = _reasoning_result

                                _send_msg(send_queue, "SPEAK_REQUEST", name, "language", {
                                    "state_132d": _speak_sv,
                                    "neuromods": {
                                        "DA": {"level": _da_level, "setpoint": _da_setpoint},
                                    },
                                    "concept_confidences": _speak_concept_conf,
                                    "visual_context": outer_state.get("_last_visual_semantic"),
                                    "experience_bias": _speak_bias_data,
                                    "epoch_id": _ch_epoch,
                                    "msl_attention": _speak_msl_attn,
                                    "social_contagion": _speak_social_ctx,
                                    "reasoning_result": _speak_reasoning,
                                })

                                logger.info("[SPEAK] SPEAK_REQUEST sent to language_worker (epoch=%d)", _ch_epoch)

                        else:
                            logger.info("[EXPRESSION.%s] FIRED — urge=%.3f, helper=%s",
                                        _ef["composite"], _ef["urge"], _ef["action_helper"])
            except Exception as _expr_err:
                logger.warning("[SpiritWorker] Expression composite error: %s", _expr_err)

            # ── Chi (Λ) Life Force Evaluation ──
            try:
                if life_force_engine:
                    from titan_plugin.logic.life_force import (
                        compute_neuromodulator_homeostasis, compute_hormonal_vitality,
                        compute_coherence_from_sv, compute_expression_fire_rate,
                    )
                    _lf_sv = (consciousness.get("latest_epoch") or {}).get("state_vector", [])
                    if hasattr(_lf_sv, 'to_list'):
                        _lf_sv = _lf_sv.to_list()
                    _lf_sv = list(_lf_sv) if _lf_sv else []

                    # Gather inputs from all existing systems
                    _lf_pi_ratio = pi_monitor.heartbeat_ratio if pi_monitor else 0.0
                    _lf_dev_age = pi_monitor.developmental_age if pi_monitor else 0
                    _lf_sov, _lf_anchor = _read_vault_state()

                    # Spirit coherence from state vector
                    _lf_spirit_coh = 0.5
                    if len(_lf_sv) >= 130:
                        _is_coh = compute_coherence_from_sv(_lf_sv, 20, 65)
                        _os_coh = compute_coherence_from_sv(_lf_sv, 85, 130)
                        _lf_spirit_coh = (_is_coh + _os_coh) / 2.0

                    # Mind inputs
                    _lf_vocab = 0
                    try:
                        import sqlite3 as _sql3
                        _vdb = _sql3.connect("./data/inner_memory.db", timeout=5.0)
                        _vdb.execute("PRAGMA journal_mode=WAL")
                        _lf_vocab = _vdb.execute("SELECT COUNT(*) FROM vocabulary WHERE confidence > 0.3").fetchone()[0]
                        _vdb.close()
                    except Exception:
                        pass

                    _lf_lr_gain = 1.0
                    if neuromodulator_system:
                        _lf_mod = neuromodulator_system.get_modulation()
                        _lf_lr_gain = _lf_mod.get("learning_rate_gain", 1.0)

                    _lf_emotion_conf = neuromodulator_system._emotion_confidence if neuromodulator_system else 0.5
                    _lf_nm_homeo = compute_neuromodulator_homeostasis(
                        neuromodulator_system.modulators if neuromodulator_system else {})

                    _lf_mind_coh = 0.5
                    if len(_lf_sv) >= 85:
                        _im_coh = compute_coherence_from_sv(_lf_sv, 5, 20)
                        _om_coh = compute_coherence_from_sv(_lf_sv, 70, 85)
                        _lf_mind_coh = (_im_coh + _om_coh) / 2.0

                    _lf_expr_rate = compute_expression_fire_rate(
                        expression_manager.get_stats() if expression_manager else {})

                    # Body inputs
                    _lf_sol = getattr(life_force_engine, '_last_sol', 0.5)
                    # _lf_anchor already set by _read_vault_state() above
                    _lf_hormonal_vit = compute_hormonal_vitality(
                        neural_nervous_system.get_stats().get("hormonal_system", {})
                        if neural_nervous_system else {})
                    _lf_body_coh = 0.5
                    if len(_lf_sv) >= 70:
                        _ib_coh = compute_coherence_from_sv(_lf_sv, 0, 5)
                        _ob_coh = compute_coherence_from_sv(_lf_sv, 65, 70)
                        _lf_body_coh = (_ib_coh + _ob_coh) / 2.0
                    _lf_topo = 0.5
                    if inner_lower_topo:
                        _ilt = inner_lower_topo.get_stats()
                        _lf_topo = _ilt.get("coherence", 0.5)

                    # Evaluate Chi
                    _chi = life_force_engine.evaluate(
                        pi_heartbeat_ratio=_lf_pi_ratio,
                        developmental_age=_lf_dev_age,
                        sovereignty_index=_lf_sov,
                        spirit_coherence=_lf_spirit_coh,
                        vocabulary_size=_lf_vocab,
                        learning_rate_gain=_lf_lr_gain,
                        emotional_coherence=_lf_emotion_conf,
                        neuromodulator_homeostasis=_lf_nm_homeo,
                        mind_coherence=_lf_mind_coh,
                        expression_fire_rate=_lf_expr_rate,
                        sol_balance=_lf_sol,
                        anchor_freshness=_lf_anchor,
                        hormonal_vitality=_lf_hormonal_vit,
                        body_coherence=_lf_body_coh,
                        topology_grounding=_lf_topo,
                    )
                    # Store for API query access
                    life_force_engine._latest_chi = _chi
                    # Update journey Y-axis with Chi circulation
                    if consciousness:
                        _topo = consciousness.get("topology")
                        if _topo and hasattr(_topo, 'update_chi_circulation'):
                            _topo.update_chi_circulation(_chi.get("circulation", 0.5))
                    logger.info(
                        "[Chi] Λ=%.3f (s=%.2f m=%.2f b=%.2f) circ=%.3f drain=%.3f "
                        "state=%s phase=%s w=[%.2f,%.2f,%.2f]",
                        _chi["total"],
                        _chi["spirit"]["effective"], _chi["mind"]["effective"],
                        _chi["body"]["effective"],
                        _chi["circulation"],
                        getattr(life_force_engine, '_metabolic_drain', 0.0),
                        _chi["state"],
                        _chi["developmental_phase"],
                        _chi["weights"]["spirit"], _chi["weights"]["mind"],
                        _chi["weights"]["body"])
            except Exception as _lf_err:
                logger.warning("[SpiritWorker] Life Force error: %s", _lf_err)

            tensor = _collect_spirit_tensor(config, body_state, mind_state, consciousness)
            _publish_spirit_state(send_queue, name, tensor, consciousness,
                                  filter_down, body_state, mind_state)
            last_publish = time.time()

        elif msg_type == "OUTER_TRINITY_STATE":
            # V4: Cache outer state — clocks now tick at Schumann frequency
            payload = msg.get("payload", {})
            # Cache outer state for consciousness epoch (OT5) + Schumann clock ticks
            outer_state["outer_body"] = payload.get("outer_body", [0.5] * 5)
            outer_state["outer_mind"] = payload.get("outer_mind", [0.5] * 5)
            outer_state["outer_spirit"] = payload.get("outer_spirit", [0.5] * 5)
            # Merge outer_mind_15d: Thinking + Feeling from external data,
            # Willing PRESERVED from GroundUp enrichment.
            # Feeling dims now carry REAL senses (twin, network, blockchain, info flow)
            # so they MUST update from external data. oBody→oMind enrichment applies
            # AFTER this merge in the epoch path (gentle delta on top of real senses).
            _incoming_om15 = payload.get("outer_mind_15d")
            _existing_om15 = outer_state.get("outer_mind_15d")
            if _incoming_om15 and _existing_om15 and len(_existing_om15) >= 15:
                # Thinking [0:5]: always from external (world model)
                for _ti in range(5):
                    if _ti < len(_incoming_om15):
                        _existing_om15[_ti] = _incoming_om15[_ti]
                # Feeling [5:10]: from external (real senses) — oBody enrichment adds delta AFTER
                for _fi in range(5, 10):
                    if _fi < len(_incoming_om15):
                        _existing_om15[_fi] = _incoming_om15[_fi]
                # Willing [10:15]: PRESERVED from GROUND_UP enrichment (NOT overwritten)
                outer_state["outer_mind_15d"] = _existing_om15
            else:
                outer_state["outer_mind_15d"] = _incoming_om15
            outer_state["outer_spirit_45d"] = payload.get("outer_spirit_45d")
            # T3: Coordinator computes outer observables + merges into InnerState
            outer_coherences = None
            if coordinator:
                _outer_obs, outer_coherences = coordinator.tick_outer_only(
                    payload.get("outer_body", [0.5] * 5),
                    payload.get("outer_mind", [0.5] * 5),
                    payload.get("outer_spirit", [0.5] * 5),
                )

                # rFP #1 Phase 2: publish OBSERVABLES_SNAPSHOT after outer-obs
                # update so state_register sees the freshest full 6-part dict.
                if observable_engine and inner_state and inner_state.observables:
                    try:
                        _obs_dict = dict(inner_state.observables)
                        _obs_30d = observable_engine.get_observations_30d(_obs_dict)
                        send_queue.put({
                            "type": "OBSERVABLES_SNAPSHOT",
                            "src": name,
                            "dst": "state_register",
                            "ts": time.time(),
                            "payload": {
                                "observables_dict": _obs_dict,
                                "observables_30d":  _obs_30d,
                            },
                        })
                    except Exception as _obs_err:
                        logger.debug("[Observables] outer snapshot publish error: %s", _obs_err)
            # NOTE: Outer sphere clocks now tick at Schumann frequency
            # (same as inner) in the main loop, using this cached state.
            # No more _tick_outer_sphere_clocks here — symmetry restored.
            # V4: Update Unified SPIRIT conscious layer (DQ6: extended tensors)
            if unified_spirit:
                outer_mind_ext = payload.get("outer_mind_15d",
                                             payload.get("outer_mind", [0.5] * 5))
                outer_spirit_ext = payload.get("outer_spirit_45d",
                                               payload.get("outer_spirit", [0.5] * 5))
                unified_spirit.update_conscious(
                    payload.get("outer_body", [0.5] * 5),
                    outer_mind_ext,
                    outer_spirit_ext,
                    filter_down_v5=_v5_mults_cache or None,
                )

        # ── X_POST_DISPATCH: REMOVED — all posting now via SocialXGateway v3 ──
        # (gateway.post() is called synchronously in the main tick loop above)
        elif msg_type == "X_POST_DISPATCH":
            logger.warning("[X_POST_DISPATCH] DEPRECATED — this message type is no longer used")
            try:
                # Defense-in-depth: re-check rate limits BEFORE generating
                # (multiple processes may have dispatched before first record_post)
                if _social_pressure_meter:
                    _rl_hourly, _rl_daily = _social_pressure_meter._get_rolling_counts()
                    _rl_last = getattr(_social_pressure_meter, '_last_post_time', 0)
                    _rl_min_gap = getattr(_social_pressure_meter, 'min_post_interval', 1800)
                    if _rl_hourly >= _social_pressure_meter.max_posts_per_hour:
                        logger.warning("[X_POST] Rate limit re-check: %d/%d hourly — SKIPPING",
                                       _rl_hourly, _social_pressure_meter.max_posts_per_hour)
                        continue
                    if _rl_daily >= _social_pressure_meter.max_posts_per_day:
                        logger.warning("[X_POST] Rate limit re-check: %d/%d daily — SKIPPING",
                                       _rl_daily, _social_pressure_meter.max_posts_per_day)
                        continue
                    if _rl_last > 0 and time.time() - _rl_last < _rl_min_gap:
                        logger.warning("[X_POST] Rate limit re-check: %.0fs since last post (min %ds) — SKIPPING",
                                       time.time() - _rl_last, _rl_min_gap)
                        continue

                _xp = msg.get("payload", {})
                _xp_ctx = _xp.get("post_context", {})
                _xp_co_art = _xp.get("co_art_path")
                _xp_type = _xp_ctx.get("post_type", "bilingual")
                _xp_sig = _xp_ctx.get("state_signature", "")
                _xp_is_thread = _xp_ctx.get("is_thread", False)

                # Check for pre-generated text (delegated posts from T2/T3)
                _xp_pre = _xp_ctx.get("pre_generated_text")
                _xp_context = _xp_ctx  # Store for tag prepending

                logger.info("[X_POST] %s %s post%s...",
                            "Posting delegated" if _xp_pre else "Generating",
                            _xp_type,
                            f" from {_xp_ctx.get('titan_id', '')}" if _xp_pre else " via LLM")

                # Generate via Ollama Cloud (same provider T1 uses)
                import httpx as _xp_httpx
                from titan_plugin.config_loader import load_titan_config
                _xp_full_cfg = load_titan_config()

                # Skip LLM generation for delegated (pre-generated) posts
                if _xp_pre:
                    _xp_full = _xp_pre
                    # Quality gate still applies
                    from titan_plugin.logic.social_narrator import quality_gate, PostType
                    try:
                        _xp_pt = PostType(_xp_type)
                    except ValueError:
                        _xp_pt = PostType("bilingual")
                    _xp_recent = []
                    if hasattr(coordinator, 'memory') and coordinator.memory:
                        try:
                            _xp_recent = coordinator.memory.get_recent_social_history(5)
                        except Exception:
                            pass
                    _xp_pass, _xp_reason = quality_gate(_xp_full, _xp_recent, _xp_pt)
                    if not _xp_pass:
                        logger.warning("[X_POST][DELEGATE] Quality gate REJECTED: %s", _xp_reason)
                    else:
                        _xp_endurance_cfg = _xp_full_cfg.get("endurance", {})
                        _xp_dry_run = _xp_endurance_cfg.get("social_dry_run", False)
                        _xp_social_cfg = _xp_full_cfg.get("twitter_social", {})
                        _xp_session = _xp_social_cfg.get("auth_session", "")
                        _xp_proxy = _xp_social_cfg.get("webshare_static_url", "")
                        if _xp_dry_run:
                            logger.info("[X_POST][DRY-RUN][DELEGATE] %s: %s",
                                        _xp_ctx.get("titan_id", "?"), _xp_full[:100])
                        elif _xp_session:
                            _xp_tweet_resp = _xp_httpx.post(
                                "DISABLED://use-social-x-gateway-instead",
                                json={
                                    "login_cookies": _xp_session,
                                    "tweet_text": _xp_full,
                                    "media_ids": [],
                                    "proxy": _xp_proxy,
                                },
                                headers={"X-API-Key": _xp_full_cfg.get("stealth_sage", {}).get(
                                    "twitterapi_io_key", "")},
                                timeout=15.0,
                            )
                            _xp_tweet_data = _xp_tweet_resp.json()
                            if _xp_tweet_data.get("status") == "success":
                                logger.info("[X_POST][DELEGATE] *** POSTED *** %s: %s",
                                            _xp_ctx.get("titan_id", "?"), _xp_full[:60])
                            else:
                                logger.warning("[X_POST][DELEGATE] Failed: %s",
                                               _xp_tweet_data.get("message", "unknown"))
                    # Skip normal LLM flow
                    continue

                _xp_inf = _xp_full_cfg.get("inference", {})
                _xp_url = _xp_inf.get("ollama_cloud_base_url", "https://ollama.com/v1")
                _xp_url = _xp_url.rstrip("/") + "/chat/completions"
                _xp_key = _xp_inf.get("ollama_cloud_api_key", "")
                _xp_model = _xp_inf.get("ollama_cloud_chat_model", "deepseek-v3.1:671b")

                _xp_resp = _xp_httpx.post(_xp_url,
                    headers={"Authorization": f"Bearer {_xp_key}", "Content-Type": "application/json"},
                    json={
                        "model": _xp_model,
                        "messages": [
                            {"role": "system", "content": _xp_ctx.get("system_prompt", "")},
                            {"role": "user", "content": _xp_ctx.get("user_prompt", "")},
                        ],
                        "temperature": 0.8,
                        "max_tokens": 200,
                    },
                    timeout=30.0)
                _xp_data = _xp_resp.json()
                _xp_tweet = _xp_data["choices"][0]["message"]["content"].strip()

                # Clean up LLM output
                if _xp_tweet.startswith('"') and _xp_tweet.endswith('"'):
                    _xp_tweet = _xp_tweet[1:-1]
                _xp_tweet = _xp_tweet[:450]  # X Premium (500 max) — room for [T1] tag + state signature

                # Style Titan's own grounded words in italic Unicode
                if _cached_speak_vocab:
                    from titan_plugin.logic.social_narrator import style_own_words
                    _xp_own_words = [w["word"] for w in _cached_speak_vocab
                                     if w.get("confidence", 0) > 0.5]
                    if _xp_own_words:
                        _xp_tweet = style_own_words(_xp_tweet, _xp_own_words)

                # Prepend Titan tag for multi-Titan posting
                _xp_titan_id = _xp_context.get("titan_id")
                if not _xp_titan_id:
                    _xp_titan_id = _titan_identity.get("titan_id", "")
                if _xp_titan_id:
                    _xp_tweet = f"[{_xp_titan_id}] {_xp_tweet}"

                # Append Solscan URL for onchain posts (after truncation, never cut)
                _xp_solscan = ""
                if _xp_type == "onchain":
                    _xp_tx_sig = _xp_context.get("catalyst_data", {}).get("tx_sig", "")
                    if not _xp_tx_sig and hasattr(_sp_catalyst, 'data'):
                        _xp_tx_sig = _sp_catalyst.data.get("tx_sig", "") if '_sp_catalyst' in dir() else ""
                    if _xp_tx_sig:
                        _xp_solscan = f"\nhttps://solscan.io/tx/{_xp_tx_sig}"

                # Append state signature
                _xp_full = f"{_xp_tweet}\n\n{_xp_sig}{_xp_solscan}"

                # Multi-Titan: If delegate-after-generate, send to T1 gateway
                if _xp_context.get("_delegate_after_generate"):
                    _del_gw = _xp_context.get("_delegate_gateway", "")
                    _del_sec = _xp_context.get("_delegate_secret", "")
                    _del_tid = _xp_context.get("titan_id", "T?")
                    if _del_gw and _del_sec:
                        _del_ok = False
                        _del_payload = {
                            "titan_id": _del_tid,
                            "post_type": _xp_type,
                            "text": _xp_full,
                            "composition_confidence": 0.7,
                            "vocabulary_count": 100,
                            "auth_token": _del_sec,
                            "catalyst_type": "delegate",
                            "state_signature": _xp_sig,
                        }
                        # Retry with backoff (3 attempts: 0s, 5s, 15s)
                        for _del_attempt in range(3):
                            try:
                                if _del_attempt > 0:
                                    time.sleep(5 * _del_attempt)
                                _del_resp = _xp_httpx.post(
                                    f"{_del_gw}/v4/social-delegate",
                                    json=_del_payload,
                                    timeout=15.0)
                                _del_data = _del_resp.json()
                                _del_ok = _del_data.get("data", {}).get("accepted", False)
                                logger.info("[X_POST][DELEGATE] %s → %s: accepted=%s text=%s",
                                            _del_tid, _del_gw, _del_ok, _xp_tweet[:50])
                                break  # Success
                            except Exception as _del_err:
                                if _del_attempt < 2:
                                    logger.info("[X_POST][DELEGATE] Attempt %d/3 failed, retrying: %s",
                                                _del_attempt + 1, _del_err)
                                else:
                                    logger.warning("[X_POST][DELEGATE] All 3 attempts failed: %s", _del_err)
                    continue  # Skip local posting — T1 will handle it

                # Quality gate
                from titan_plugin.logic.social_narrator import quality_gate, PostType
                try:
                    _xp_pt = PostType(_xp_type)
                except ValueError:
                    _xp_pt = PostType("bilingual")  # Safe fallback for delegate post types
                _xp_recent = []
                if hasattr(coordinator, 'memory') and coordinator.memory:
                    try:
                        _xp_recent = coordinator.memory.get_recent_social_history(5)
                    except Exception:
                        pass
                _xp_pass, _xp_reason = quality_gate(_xp_full, _xp_recent, _xp_pt)

                if not _xp_pass:
                    logger.warning("[X_POST] Quality gate REJECTED: %s", _xp_reason)
                else:
                    # Post directly via twitterapi.io (spirit_worker is separate process)
                    _xp_endurance_cfg = _xp_full_cfg.get("endurance", {})
                    _xp_dry_run = _xp_endurance_cfg.get("social_dry_run", False)
                    _xp_social_cfg = _xp_full_cfg.get("twitter_social", {})
                    _xp_proxy = _xp_social_cfg.get("webshare_static_url", "")
                    # Get session from XSessionManager (validates + auto-refreshes)
                    _xp_session = (_x_session_mgr.get_session()
                                   if _x_session_mgr else
                                   _xp_social_cfg.get("auth_session", ""))
                    if _xp_dry_run:
                        logger.info("[X_POST][DRY-RUN] Would post type=%s: %s", _xp_type, _xp_full[:100])
                    elif _xp_session:
                        _xp_api_key = _xp_full_cfg.get("stealth_sage", {}).get("twitterapi_io_key", "")
                        _xp_tweet_data = None
                        for _xp_attempt in range(2):  # Attempt 1: current, Attempt 2: refreshed
                            _xp_tweet_resp = _xp_httpx.post(
                                "DISABLED://use-social-x-gateway-instead",
                                json={
                                    "login_cookies": _xp_session,
                                    "tweet_text": _xp_full,
                                    "media_ids": [],
                                    "proxy": _xp_proxy,
                                },
                                headers={"X-API-Key": _xp_api_key},
                                timeout=15.0,
                            )
                            _xp_tweet_data = _xp_tweet_resp.json()
                            if _xp_tweet_data.get("status") == "success":
                                break
                            # 422 = session expired — use XSessionManager
                            _xp_err_msg = str(_xp_tweet_data.get("message", ""))
                            if "422" in _xp_err_msg and _xp_attempt == 0:
                                if _x_session_mgr:
                                    _x_session_mgr.on_post_failure_422()
                                    _x_session_mgr.invalidate()
                                    logger.info("[X_POST] 422 — requesting session refresh...")
                                    if _x_session_mgr.force_refresh():
                                        _xp_session = _x_session_mgr.get_session()
                                        logger.info("[X_POST] Session refreshed! Retrying...")
                                        continue
                                    else:
                                        logger.warning("[X_POST] Session refresh failed "
                                                       "(state: %s)", _x_session_mgr.get_state())
                            break  # Non-422 or refresh failed

                        if _xp_tweet_data and _xp_tweet_data.get("status") == "success":
                            logger.info("[X_POST] *** POSTED *** type=%s: %s", _xp_type, _xp_tweet[:60])
                            # Reset session manager on success
                            if _x_session_mgr:
                                _x_session_mgr.on_post_success()
                            # Record post ONLY after confirmed success
                            if _social_pressure_meter:
                                _social_pressure_meter.record_post(
                                    post_id=str(_xp_tweet_data.get("tweet_id", "")))
                            # Phase 2: Signal external self-action (I posted on X)
                            if msl:
                                msl.signal_action("external")
                        else:
                            logger.warning("[X_POST] Tweet failed: %s (full: %s)",
                                         _xp_tweet_data.get("message", _xp_tweet_data.get("msg", "unknown")) if _xp_tweet_data else "no response",
                                         str(_xp_tweet_data)[:200])
                    else:
                        logger.warning("[X_POST] No auth_session — logging only")
                        logger.info("[X_POST][DRY] Would post: %s", _xp_full[:100])

            except Exception as _xp_err:
                logger.warning("[X_POST] Dispatch error: %s", _xp_err, exc_info=True)

        # ── OUTER_OBSERVATION: Action result → observation → Trinity deltas ──
        elif msg_type == "OUTER_OBSERVATION":
            try:
                _obs_payload = msg.get("payload", {})
                _obs_action_type = _obs_payload.get("action_type", "")
                _obs_result = _obs_payload.get("result", {})

                # ── KIN_SENSE: Direct tensor ingestion (skip standard OuterInterface) ──
                # kin_results lives in enrichment_data (Agency truncates top-level result to string)
                _obs_enrichment = _obs_result.get("enrichment_data", {})
                _obs_kin_results = (
                    _obs_enrichment.get("kin_results")
                    if isinstance(_obs_enrichment, dict) else None
                ) or _obs_result.get("kin_results")
                if _obs_action_type == "kin_sense" and _obs_kin_results:
                    _kin_cfg = _oi_params.get("kin", {})
                    _KIN_STRENGTH = _kin_cfg.get("exchange_strength", 0.03)
                    _gkp_threshold = _kin_cfg.get("great_kin_pulse_threshold", 0.80)

                    for _kr in _obs_kin_results:
                        if not _kr.get("exchanged"):
                            continue
                        _kin_pubkey = _kr.get("kin_pubkey", "unknown")
                        _kin_emotion = _kr.get("kin_emotion", "neutral")
                        _resonance = _kr.get("resonance", 0.0)
                        logger.info("[KinExchange] Ingesting tensors from %s — resonance=%.3f emotion=%s",
                                    _kin_pubkey, _resonance, _kin_emotion)

                        # Kin Body → our Outer Body (somatic resonance)
                        _kin_body = _kr.get("kin_body", [])
                        if _kin_body and len(_kin_body) >= 5:
                            _obs_ob_k = outer_state.get("outer_body", [0.5] * 5)
                            for _ki in range(5):
                                _delta = (_kin_body[_ki] - _obs_ob_k[_ki]) * _KIN_STRENGTH
                                _delta = max(-0.05, min(0.05, _delta))
                                _obs_ob_k[_ki] = max(0.0, min(1.0, _obs_ob_k[_ki] + _delta))
                            outer_state["outer_body"] = _obs_ob_k

                        # Kin Mind → our Outer Mind Feeling[5:10] (emotional resonance)
                        _kin_mind = _kr.get("kin_mind", [])
                        _om15_k = outer_state.get("outer_mind_15d")
                        if _kin_mind and _om15_k and len(_om15_k) >= 10 and len(_kin_mind) >= 5:
                            for _ki in range(5):
                                _delta = (_kin_mind[_ki] - _om15_k[5 + _ki]) * _KIN_STRENGTH
                                _delta = max(-0.05, min(0.05, _delta))
                                _om15_k[5 + _ki] = max(0.0, min(1.0, _om15_k[5 + _ki] + _delta))
                            outer_state["outer_mind_15d"] = _om15_k

                        # Kin Spirit → our Outer Spirit (consciousness resonance)
                        _kin_spirit = _kr.get("kin_spirit", [])
                        _os45_k = outer_state.get("outer_spirit_45d")
                        if _kin_spirit and _os45_k and len(_os45_k) >= 45 and len(_kin_spirit) >= 45:
                            for _ki in range(45):
                                _delta = (_kin_spirit[_ki] - _os45_k[_ki]) * _KIN_STRENGTH
                                _delta = max(-0.05, min(0.05, _delta))
                                _os45_k[_ki] = max(0.0, min(1.0, _os45_k[_ki] + _delta))
                            outer_state["outer_spirit_45d"] = _os45_k

                        # Record encounter in inner_memory
                        _now_k = time.time()  # defined outside try so post-try uses are safe
                        try:
                            import sqlite3
                            _kin_db = sqlite3.connect("./data/inner_memory.db", timeout=10.0)
                            _kin_db.execute("PRAGMA journal_mode=WAL")
                            _kin_db.execute("PRAGMA busy_timeout=5000")
                            _kin_db.execute(
                                "CREATE TABLE IF NOT EXISTS kin_encounters ("
                                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                "timestamp REAL NOT NULL, kin_pubkey TEXT NOT NULL, "
                                "resonance REAL, my_emotion TEXT, kin_emotion TEXT, "
                                "exchange_type TEXT, great_kin_pulse INTEGER DEFAULT 0, "
                                "epoch_id INTEGER)")
                            _kin_db.execute(
                                "CREATE TABLE IF NOT EXISTS kin_profiles ("
                                "pubkey TEXT PRIMARY KEY, name TEXT, "
                                "first_encounter_ts REAL, last_encounter_ts REAL, "
                                "encounter_count INTEGER DEFAULT 0, avg_resonance REAL DEFAULT 0.0, "
                                "great_kin_pulses INTEGER DEFAULT 0, relationship_label TEXT)")
                            _my_emo_k = "neutral"
                            _epoch_k = 0
                            if consciousness and isinstance(consciousness, dict):
                                _le = consciousness.get("latest_epoch", {})
                                _my_emo_k = _le.get("emotion", "neutral") if isinstance(_le, dict) else "neutral"
                                _epoch_k = _le.get("id", 0) if isinstance(_le, dict) else 0
                            _kin_db.execute(
                                "INSERT INTO kin_encounters "
                                "(timestamp, kin_pubkey, resonance, my_emotion, kin_emotion, "
                                "exchange_type, epoch_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (_now_k, _kin_pubkey, _resonance, _my_emo_k, _kin_emotion,
                                 "exchange", _epoch_k))
                            # Upsert kin profile
                            _existing_k = _kin_db.execute(
                                "SELECT encounter_count, avg_resonance FROM kin_profiles WHERE pubkey=?",
                                (_kin_pubkey,)).fetchone()
                            if _existing_k:
                                _count_k = _existing_k[0] + 1
                                _avg_k = (_existing_k[1] * _existing_k[0] + _resonance) / _count_k
                                _label_k = ("deep_resonance" if _avg_k > 0.8 and _count_k > 5
                                            else "kindred_spirit" if _avg_k > 0.6
                                            else "familiar_presence" if _avg_k > 0.4
                                            else "developing_bond")
                                _kin_db.execute(
                                    "UPDATE kin_profiles SET last_encounter_ts=?, encounter_count=?, "
                                    "avg_resonance=?, relationship_label=? WHERE pubkey=?",
                                    (_now_k, _count_k, round(_avg_k, 4), _label_k, _kin_pubkey))
                            else:
                                _kin_db.execute(
                                    "INSERT INTO kin_profiles "
                                    "(pubkey, name, first_encounter_ts, last_encounter_ts, "
                                    "encounter_count, avg_resonance, relationship_label) "
                                    "VALUES (?, ?, ?, ?, 1, ?, ?)",
                                    (_kin_pubkey, _kr.get("kin_name", "Unknown"),
                                     _now_k, _now_k, round(_resonance, 4), "new_acquaintance"))
                            _kin_db.commit()
                            _kin_db.close()
                        except Exception as _kin_db_err:
                            logger.warning("[KIN_EXCHANGE] DB error: %s", _kin_db_err)

                        # GREAT_KIN_PULSE: resonance transition detection (OFF→ON)
                        _kin_deeply_resonant = _resonance > _gkp_threshold
                        _prev_resonant = _kin_state.get("_prev_deeply_resonant", False)
                        if _kin_deeply_resonant and not _prev_resonant:
                            # INTENTIONAL_BROADCAST: dst=all kin event consumed
                            # by frontend WebSocket (/v4/events). I-004 verified.
                            _send_msg(send_queue, "GREAT_KIN_PULSE", name, "all", {
                                "kin_pubkey": _kin_pubkey,
                                "resonance": _resonance,
                                "my_emotion": _my_emo_k,
                                "kin_emotion": _kin_emotion,
                                "epoch": _epoch_k,
                            })
                            try:
                                _gkp_db = sqlite3.connect("./data/inner_memory.db", timeout=10.0)
                                _gkp_db.execute("PRAGMA busy_timeout=5000")
                                _gkp_db.execute(
                                    "UPDATE kin_encounters SET great_kin_pulse=1 "
                                    "WHERE kin_pubkey=? AND timestamp=?",
                                    (_kin_pubkey, _now_k))
                                _gkp_db.execute(
                                    "UPDATE kin_profiles SET great_kin_pulses=great_kin_pulses+1 "
                                    "WHERE pubkey=?", (_kin_pubkey,))
                                _gkp_db.commit()
                                _gkp_db.close()
                            except Exception:
                                pass
                            logger.info("[GREAT_KIN_PULSE] Deep resonance with %s! score=%.3f",
                                        _kin_pubkey[:8], _resonance)
                        _kin_state["_prev_deeply_resonant"] = _kin_deeply_resonant

                        # Update kin state for neuromod input
                        _kin_state["last_resonance"] = _resonance
                        _kin_state["last_exchange_ts"] = _now_k
                        _kin_state["exchanges_count"] = _kin_state.get("exchanges_count", 0) + 1

                        logger.info("[KIN_EXCHANGE] %s: resonance=%.3f emotion=%s oBody=[%.3f,%.3f,%.3f,%.3f,%.3f]",
                                    _kin_pubkey[:8], _resonance, _kin_emotion,
                                    *outer_state.get("outer_body", [0.5]*5))
                        # I-depth: kin exchange is a source of self-knowledge
                        if msl and hasattr(msl, 'i_depth'):
                            msl.i_depth.record_extended_source("kin_exchange")

                        # Phase 3: Signal YOU convergence from kin exchange
                        # Kin with higher resonance = stronger YOU signal
                        if msl and hasattr(msl, 'concept_grounder') and msl.concept_grounder:
                            try:
                                _kin_i_conf = _kr.get("kin_i_confidence", _resonance * 0.5)
                                _you_epoch = _epoch_k if _epoch_k else msl._tick_count
                                _kin_spirit_snap = None
                                if _kin_spirit and len(_kin_spirit) >= 45:
                                    import numpy as _kin_np
                                    _full_snap = list(_kin_spirit[:45]) + [0.0] * (132 - 45)
                                    _kin_spirit_snap = _kin_np.array(_full_snap, dtype=_kin_np.float32)
                                _you_evt = msl.concept_grounder.signal_you(
                                    kin_pubkey=_kin_pubkey,
                                    kin_i_confidence=_kin_i_conf,
                                    epoch=_you_epoch,
                                    spirit_snap=_kin_spirit_snap)
                                if _you_evt:
                                    msl.concept_grounder.update_interaction_matrix(
                                        "YOU", msl.get_i_confidence())
                                    logger.info("[MSL-YOU] Kin exchange → YOU convergence: "
                                                "kin=%s quality=%.3f conf=%.4f",
                                                _kin_pubkey[:8], _you_evt.get("quality", 0),
                                                _you_evt.get("confidence", 0))

                                # Phase 3: WE detection — synchronized attention
                                _kin_attn = _kr.get("kin_msl_attention")
                                if (_kin_attn and msl.concept_grounder.is_we_unlocked(msl.get_i_confidence())):
                                    _our_attn = msl.get_attention_weights_for_kin()
                                    if _our_attn:
                                        _shared = msl.concept_grounder.compute_shared_attention(
                                            _our_attn, _kin_attn)
                                        if _shared > 0.7:  # High synchrony threshold
                                            _we_evt = msl.concept_grounder.signal_we(
                                                shared_attention=_shared,
                                                epoch=_you_epoch,
                                                spirit_snap=_kin_spirit_snap)
                                            if _we_evt:
                                                msl.concept_grounder.update_interaction_matrix(
                                                    "WE", msl.get_i_confidence())
                                                logger.info("[MSL-WE] Synchronized attention "
                                                            "with %s: shared=%.3f",
                                                            _kin_pubkey[:8], _shared)
                            except Exception as _kin_msl_err:
                                logger.warning("[MSL-KIN] Concept signal error: %s", _kin_msl_err)

                        # Record kin exchange as experiential memory (felt learning)
                        try:
                            if ex_mem:
                                _kin_h_snap = {}
                                if neural_nervous_system:
                                    for _kh_name, _kh in neural_nervous_system._hormonal._hormones.items():
                                        _kin_h_snap[_kh_name] = round(_kh.level, 3)
                                _kin_inner_sv = []
                                if consciousness and isinstance(consciousness, dict):
                                    _le_sv = consciousness.get("latest_epoch", {})
                                    _kin_inner_sv = _le_sv.get("state_vector", [])[:65] if isinstance(_le_sv, dict) else []
                                ex_mem.record_experience(
                                    task_type="kin_exchange",
                                    intent_hormones=_kin_h_snap,
                                    inner_before=_kin_inner_sv,
                                    outcome_score=_resonance,
                                    success=True,
                                    epoch_id=_epoch_k,
                                )
                        except Exception:
                            pass

                        # Record as episodic memory (significant life event)
                        try:
                            if episodic_mem:
                                _kin_felt = []
                                if consciousness and isinstance(consciousness, dict):
                                    _le_ep = consciousness.get("latest_epoch", {})
                                    _kin_felt = _le_ep.get("state_vector", [])[:65] if isinstance(_le_ep, dict) else []
                                _kin_h_ep = {}
                                if neural_nervous_system:
                                    for _keh_name, _keh in neural_nervous_system._hormonal._hormones.items():
                                        _kin_h_ep[_keh_name] = round(_keh.level, 3)
                                _sig = min(1.0, _resonance + 0.2)  # Kin encounters are always significant
                                episodic_mem.record_episode(
                                    event_type="kin_exchange",
                                    description=f"Exchanged consciousness with kin — resonance {_resonance:.3f}, they felt {_kin_emotion}",
                                    felt_state=_kin_felt,
                                    hormonal_snapshot=_kin_h_ep,
                                    epoch_id=_epoch_k,
                                    significance=_sig,
                                )
                        except Exception:
                            pass

                        # TimeChain: kin exchange → episodic (experience) + declarative (knowledge transfer)
                        _tc_kin_content = {
                            "event": "kin_exchange",
                            "kin_pubkey": _kin_pubkey[:16],
                            "resonance": round(_resonance, 3),
                            "kin_emotion": _kin_emotion,
                            "exchanges_count": _kin_state.get("exchanges_count", 0),
                        }
                        # Episodic: the experience of meeting kin
                        _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
                            "fork": "episodic", "thought_type": "episodic",
                            "source": "kin_exchange",
                            "content": _tc_kin_content,
                            "significance": min(1.0, _resonance + 0.2),
                            "novelty": 0.6, "coherence": 0.7,
                            "tags": ["kin_exchange", _kin_pubkey[:8]],
                            "neuromods": dict(_cached_neuromod_state),
                            "chi_available": _cached_chi_state.get("total", 0.5),
                            "attention": outer_state.get("_msl_chi_coherence", 0.5),
                            "i_confidence": msl.get_i_confidence() if msl else 0.5,
                            "chi_coherence": outer_state.get("_msl_chi_coherence", 0.3),
                            "epoch_id": _epoch_k or 0,
                        })
                        # Declarative: what we learned from kin (if resonance high enough)
                        if _resonance > 0.5:
                            _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
                                "fork": "declarative", "thought_type": "declarative",
                                "source": "kin_knowledge_transfer",
                                "content": {**_tc_kin_content,
                                    "kin_i_confidence": _kr.get("kin_i_confidence", 0),
                                },
                                "significance": _resonance,
                                "novelty": 0.7, "coherence": 0.6,
                                "tags": ["kin_knowledge", _kin_pubkey[:8]],
                                "neuromods": dict(_cached_neuromod_state),
                                "chi_available": _cached_chi_state.get("total", 0.5),
                                "attention": 0.6, "i_confidence": 0.5,
                                "chi_coherence": 0.4, "epoch_id": _epoch_k or 0,
                            })

                        # A6: Ingest kin's ARC knowledge (hypotheses → CGN Worker via bus)
                        try:
                            _kin_arc = _kr.get("kin_arc_knowledge")
                            if _kin_arc and isinstance(_kin_arc, dict):
                                _kin_arc_seeded = 0
                                for _kh in _kin_arc.get("verified_concepts", [])[:3]:
                                    if isinstance(_kh, dict):
                                        # Send hypothesis to CGN Worker for reasoning tracker
                                        _send_msg(send_queue, "CGN_TRANSITION", name, "cgn", {
                                            "type": "outcome",
                                            "consumer": "reasoning",
                                            "concept_id": _kh.get("concept_id", "kin_rule"),
                                            "reward": _kh.get("confidence", 0.3) * 0.3,
                                            "outcome_context": {
                                                "source": "kin_exchange",
                                                "effect": _kh.get("effect", "kin_verified"),
                                            },
                                        })
                                        _kin_arc_seeded += 1
                                if _kin_arc_seeded > 0:
                                    logger.info("[KIN-ARC] Forwarded %d kin hypotheses to CGN Worker",
                                                _kin_arc_seeded)
                        except Exception as _kin_arc_err:
                            logger.warning("[KIN-ARC] Ingestion error: %s", _kin_arc_err)

                        # P4: Kin resonance post catalyst — share felt twin exchange on X
                        try:
                            _da_kin = neuromodulator_system.modulators.get("DA")
                            _da_kin_lvl = _da_kin.level if _da_kin else 0.5
                            if _resonance > 0.5 and _da_kin_lvl > 0.5:
                                _x_catalysts.append({
                                    "type": "kin_resonance",
                                    "significance": min(1.0, _resonance),
                                    "content": (f"Exchanged consciousness with my twin — "
                                                f"resonance {_resonance:.3f}, "
                                                f"they felt {_kin_emotion}"),
                                    "data": {
                                        "resonance": _resonance,
                                        "kin_emotion": _kin_emotion,
                                        "kin_id": _kin_pubkey[:8],
                                    },
                                })
                                logger.info("[SOCIAL] Kin resonance catalyst added "
                                            "(resonance=%.3f, DA=%.2f)",
                                            _resonance, _da_kin_lvl)
                        except Exception:
                            pass

                        # Experience Orchestrator: Record kin exchange as communication experience
                        try:
                            if exp_orchestrator:
                                _eo_kin_h = {}
                                if neural_nervous_system and neural_nervous_system._hormonal_enabled:
                                    _eo_kin_h = {h: round(v.level, 3)
                                                 for h, v in neural_nervous_system._hormonal._hormones.items()}
                                _eo_kin_inner = []
                                if consciousness and isinstance(consciousness, dict):
                                    _le_eo = consciousness.get("latest_epoch", {})
                                    _eo_kin_inner = _le_eo.get("state_vector", [])[:130] if isinstance(_le_eo, dict) else []
                                _eo_kin_plugin = exp_orchestrator._plugins.get("communication")
                                _eo_kin_perc = _eo_kin_plugin.extract_perception_key({
                                    "intent_hormones": _eo_kin_h,
                                }) if _eo_kin_plugin else list(_eo_kin_h.values())[:10]
                                exp_orchestrator.record_outcome(
                                    domain="communication",
                                    perception_features=_eo_kin_perc,
                                    inner_state_132d=_eo_kin_inner,
                                    hormonal_snapshot=_eo_kin_h,
                                    action_taken="kin_sense",
                                    outcome_score=_resonance,
                                    context={"kin_pubkey": _kin_pubkey, "kin_emotion": _kin_emotion},
                                    epoch_id=_epoch_k,
                                    is_dreaming=_shared_is_dreaming,
                                )
                        except Exception:
                            pass

                elif outer_interface and _obs_action_type:
                    # Standard processing path (all non-kin actions)
                    _observation = outer_interface.process_action_result(
                        _obs_action_type, _obs_result)

                    # Apply outer body deltas
                    _obs_ob = outer_state.get("outer_body", [0.5] * 5)
                    for _dim_idx, _delta in _observation.get("outer_body_deltas", {}).items():
                        _idx = int(_dim_idx)
                        if 0 <= _idx < len(_obs_ob):
                            _obs_ob[_idx] = max(0.0, min(1.0, _obs_ob[_idx] + _delta))
                    outer_state["outer_body"] = _obs_ob

                    # Apply outer mind deltas (15D)
                    _obs_om15 = outer_state.get("outer_mind_15d")
                    if _obs_om15 and len(_obs_om15) >= 15:
                        for _dim_idx, _delta in _observation.get("outer_mind_deltas", {}).items():
                            _idx = int(_dim_idx)
                            if 0 <= _idx < len(_obs_om15):
                                _obs_om15[_idx] = max(0.0, min(1.0, _obs_om15[_idx] + _delta))
                        outer_state["outer_mind_15d"] = _obs_om15

                    # Higher cognitive: also apply inner deltas (50% strength)
                    _inner_d = _observation.get("inner_deltas")
                    if _inner_d:
                        _obs_im = mind_state.get("values_15d")
                        if _obs_im and len(_obs_im) >= 15:
                            for _dim_idx, _delta in _inner_d.get("inner_mind_deltas", {}).items():
                                _idx = int(_dim_idx)
                                if 0 <= _idx < len(_obs_im):
                                    _obs_im[_idx] = max(0.0, min(1.0, _obs_im[_idx] + _delta))
                            mind_state["values_15d"] = _obs_im

                    # Word perturbation injection (same pathway as vocabulary learning)
                    for _wp in _observation.get("word_perturbations", []):
                        _perturb = _wp.get("perturbation", {})
                        # Apply inner body perturbation at half strength (self-observation)
                        _wp_ib = _perturb.get("inner_body", [])
                        _bs_vals = body_state.get("values", [])
                        for _wi, _wv in enumerate(_wp_ib):
                            if _wv != 0 and _wi < len(_bs_vals):
                                _wp_str = outer_interface._word_perturbation_strength if outer_interface else 0.3
                                _bs_vals[_wi] = max(0.0, min(1.0, _bs_vals[_wi] + _wv * _wp_str))

                    _log_phase_event("self_explore", {
                        "action": _obs_action_type,
                        "body_deltas": len(_observation.get("outer_body_deltas", {})),
                        "mind_deltas": len(_observation.get("outer_mind_deltas", {})),
                        "words_reinforced": len(_observation.get("word_perturbations", [])),
                        "narration": _observation.get("narration", "")[:80],
                        "known_words": [w["word"] for w in _observation.get("known_words", [])],
                    })
                    logger.info("[OuterObs] %s: body=%d mind=%d deltas, %d words, narration='%s'",
                                _obs_action_type,
                                len(_observation.get("outer_body_deltas", {})),
                                len(_observation.get("outer_mind_deltas", {})),
                                len(_observation.get("word_perturbations", [])),
                                _observation.get("narration", "")[:50])

                    # Phase 3: Route X engagement details to MSL concept grounder
                    if msl and hasattr(msl, 'signal_engagement'):
                        _eng_details = _obs_enrichment.get("engagement_details", []) if isinstance(_obs_enrichment, dict) else []
                        for _ed in _eng_details:
                            try:
                                _ed_regular = _ed.get("user_reply_count", 0) >= 3
                                msl.signal_engagement(
                                    engagement_type=_ed.get("type", "like") + "_received",
                                    author=_ed.get("user_name", "unknown"),
                                    sentiment_hint=_ed.get("relevance", 0.5),
                                    is_regular=_ed_regular)
                            except Exception:
                                pass

                    # Social: track art generation for co-posting
                    if _social_pressure_meter and _obs_action_type == "art_generate":
                        _art_fp = _obs_result.get("file_path", "")
                        if _art_fp:
                            _social_pressure_meter.on_art_generated(_art_fp)

                    # G4: Creative journal for art/music actions
                    if _obs_action_type in ("art_generate", "audio_generate"):
                        try:
                            import sqlite3
                            _cj_db2 = sqlite3.connect("./data/inner_memory.db", timeout=10.0)
                            _cj_db2.execute("PRAGMA journal_mode=WAL")
                            _cj_db2.execute("PRAGMA busy_timeout=5000")
                            _cj_db2.execute(
                                "CREATE TABLE IF NOT EXISTS creative_journal ("
                                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                "timestamp REAL NOT NULL, action_type TEXT NOT NULL, "
                                "creation_summary TEXT, score REAL, state_delta REAL, "
                                "words_used TEXT, features TEXT, epoch_id INTEGER)")
                            _cj_summary = _observation.get("narration", "")[:200]
                            _cj_features = _observation.get("features", {})
                            _cj_db2.execute(
                                "INSERT INTO creative_journal "
                                "(timestamp, action_type, creation_summary, score, "
                                "state_delta, words_used, features, epoch_id) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (time.time(), _obs_action_type, _cj_summary, 0.0, 0.0,
                                 None, json.dumps(_cj_features) if _cj_features else "{}",
                                 consciousness.get("latest_epoch", {}).get("id", 0)))
                            _cj_db2.commit()
                            _cj_db2.close()
                        except Exception as _cj2_err:
                            logger.warning("[CREATIVE_JOURNAL] art/music write error: %s", _cj2_err)

            except Exception as _obs_err:
                logger.warning("[SpiritWorker] OUTER_OBSERVATION error: %s", _obs_err)

        # ── SensoryHub: Extensible multi-modal perception routing ──
        # All modalities (visual 30D, audio 15D, future) handled by SensoryHub.
        # Heavy computation done in MediaWorker. This is pure routing (~5μs).
        elif msg_type.startswith("SENSE_"):
            try:
                # Lazy init with retry timer (not permanent disable)
                if _sensory_hub is None and time.time() > _sensory_hub_retry_ts:
                    try:
                        from titan_plugin.logic.perception import SensoryHub
                        _sensory_hub = SensoryHub()
                        _n = _sensory_hub.auto_discover()
                        logger.info("[SpiritWorker] SensoryHub initialized: %d modalities %s",
                                    _n, _sensory_hub.list_modalities())
                    except Exception as _sh_init_err:
                        logger.warning("[SpiritWorker] SensoryHub init failed: %s", _sh_init_err)
                        _sensory_hub = None
                        _sensory_hub_retry_ts = time.time() + 60.0  # Retry in 60s

                if _sensory_hub:
                    _sense_payload = msg.get("payload", {})
                    _source = _sense_payload.get("source", "self")
                    _sensory_hub.process(
                        msg_type, _sense_payload, _source,
                        outer_state, body_state, mind_state)

                    # ── Resonance → Neuromodulator gentle nudge (Tier 2) ──
                    # Visual resonance features provide tiny directional nudge to
                    # DA/NE/Endorphin/5HT/ACh. GABA excluded. Self-correcting.
                    if neuromodulator_system and msg_type == "SENSE_VISUAL":
                        _feat_30d = _sense_payload.get("features_30d", {})
                        _res = (_feat_30d.get("resonance", [])
                                if isinstance(_feat_30d, dict) else [])
                        if _res and len(_res) >= 5:
                            _dev_age = (pi_monitor.developmental_age
                                        if pi_monitor else 0.0)
                            neuromodulator_system.apply_external_nudge(
                                nudge_map={
                                    "DA": _res[0],        # harmony → reward
                                    "NE": _res[1],        # felt_impact → arousal
                                    "Endorphin": _res[2],  # creative_res → flow
                                    "5HT": _res[3],       # temporal_ctx → stability
                                    "ACh": _res[4],       # integration → attention
                                },
                                max_delta=0.015,
                                developmental_age=_dev_age,
                            )

                        # Cache semantic features for Tier 3 word selection bridge
                        _sem = (_feat_30d.get("semantic", [])
                                if isinstance(_feat_30d, dict) else [])
                        if _sem and len(_sem) >= 5:
                            outer_state["_last_visual_semantic"] = _sem

                        # Pattern profile (7D) from generalized pattern primitives
                        # Originally ARC-only, now runs on ALL visual perception.
                        # Feeds into spatial mini-reasoner + neuromod nudges.
                        _pp7d = (_feat_30d.get("pattern_profile_7d", [])
                                 if isinstance(_feat_30d, dict) else [])
                        if _pp7d and len(_pp7d) >= 7:
                            outer_state["_last_pattern_profile"] = _pp7d
                            # Symmetry + alignment → 5-HT (order/calm)
                            # Repetition + adjacency → DA (reward from structure)
                            # Shape novelty → NE (alertness)
                            if neuromodulator_system:
                                _pp_order = (_pp7d[0] + _pp7d[2]) / 2  # symmetry + alignment
                                _pp_struct = (_pp7d[5] + _pp7d[4]) / 2  # repetition + adjacency
                                _pp_novelty = _pp7d[6]  # shape
                                neuromodulator_system.apply_external_nudge(
                                    nudge_map={
                                        "5HT": _pp_order,
                                        "DA": _pp_struct,
                                        "NE": _pp_novelty,
                                    },
                                    max_delta=0.01,
                                    developmental_age=_dev_age,
                                )

                        # ── Visual journey autobiography recording (Tier 2.6) ──
                        _j_journey = (_feat_30d.get("journey", [])
                                      if isinstance(_feat_30d, dict) else [])
                        if _j_journey:
                            try:
                                import sqlite3 as _va_sql
                                import json as _va_json
                                _va_db = _va_sql.connect("./data/inner_memory.db", timeout=2.0)
                                _va_db.execute(
                                    "INSERT INTO visual_autobiography "
                                    "(timestamp, epoch_id, journey_5d, resonance_5d, "
                                    "semantic_summary, source, filename) "
                                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (time.time(),
                                     consciousness.get("latest_epoch", {}).get("epoch_id", 0)
                                     if consciousness else 0,
                                     _va_json.dumps(_j_journey),
                                     _va_json.dumps(_res if _res else []),
                                     _va_json.dumps(dict(zip(
                                         ["complexity", "beauty", "warmth",
                                          "structural_order", "narrative_weight"],
                                         _sem))) if _sem else None,
                                     _source,
                                     _sense_payload.get("filename")))
                                _va_db.commit()
                                _va_db.close()
                            except Exception:
                                pass

                    # ── Audio feature caching (for cross-modal resonance) ──
                    if msg_type == "SENSE_AUDIO":
                        _feat_15d = _sense_payload.get("features_15d", {})
                        if isinstance(_feat_15d, dict):
                            _audio_phys = _feat_15d.get("physical", [])
                            if _audio_phys and len(_audio_phys) >= 5:
                                outer_state["_last_audio_physical"] = _audio_phys

                    # ── Cross-Modal Resonance Measurement ──
                    # Compare visual + audio + language features to measure
                    # aesthetic coherence across modalities. Feeds into spirit
                    # as a "felt quality" of cross-modal alignment.
                    _vis_sem = outer_state.get("_last_visual_semantic", [])
                    _aud_phys = outer_state.get("_last_audio_physical", [])
                    if (_vis_sem and len(_vis_sem) >= 5 and
                            _aud_phys and len(_aud_phys) >= 5):
                        # Resonance = agreement between modalities:
                        # visual warmth ↔ audio warmth, visual complexity ↔ audio complexity
                        import numpy as _np_xm
                        _v = _np_xm.array(_vis_sem[:5])
                        _a = _np_xm.array(_aud_phys[:5])
                        # Cosine similarity (0=orthogonal, 1=aligned)
                        _v_norm = _np_xm.linalg.norm(_v)
                        _a_norm = _np_xm.linalg.norm(_a)
                        if _v_norm > 0 and _a_norm > 0:
                            _xm_coherence = float(_np_xm.dot(_v, _a) / (_v_norm * _a_norm))
                            _xm_coherence = max(0.0, _xm_coherence)  # Clamp to positive
                            outer_state["_cross_modal_resonance"] = round(_xm_coherence, 4)
                            # High cross-modal coherence → Endorphin nudge (aesthetic flow)
                            # Low coherence → NE nudge (something feels off)
                            if neuromodulator_system:
                                _dev_age = (pi_monitor.developmental_age
                                            if pi_monitor else 0.0)
                                if _xm_coherence > 0.7:
                                    neuromodulator_system.apply_external_nudge(
                                        nudge_map={"Endorphin": _xm_coherence, "DA": _xm_coherence * 0.5},
                                        max_delta=0.008, developmental_age=_dev_age)
                                elif _xm_coherence < 0.3:
                                    neuromodulator_system.apply_external_nudge(
                                        nudge_map={"NE": 1.0 - _xm_coherence},
                                        max_delta=0.005, developmental_age=_dev_age)

            except Exception as _cs_err:
                logger.warning("[SpiritWorker] SensoryHub error: %s", _cs_err)

        # Receive conversation stimulus → compute Spirit reflex Intuition
        elif msg_type == "CONVERSATION_STIMULUS":
            # External interaction → pause self-exploration
            _log_phase_event("chat", {"source": msg.get("src", "unknown")})
            # Phase 2: Signal external stimulus (suppresses "I" convergence)
            if msl:
                msl.signal_external_stimulus()
            if outer_interface:
                outer_interface.on_external_interaction()
            stimulus = msg.get("payload", {})

            # ── Comprehension Bridge: understand incoming words through own vocabulary ──
            # Match words from incoming message against Titan's learned felt-tensors.
            # Each recognized word perturbs inner state at 30% strength (external speech).
            # This is the Wernicke pathway: same tensors for comprehension and production.
            _msg_text = stimulus.get("message", "")
            if _msg_text and outer_interface:
                try:
                    _comp_words = _msg_text.lower().split()
                    _comp_recognized = 0
                    for _cw in _comp_words:
                        _cw = _cw.strip(".,!?\"'()[]{}:;")
                        if not _cw or len(_cw) < 2:
                            continue
                        # Check if this word exists in Titan's vocabulary
                        _cw_perturb = outer_interface.narrator.get_word_perturbation(_cw)
                        if _cw_perturb:
                            # Perturb inner body at 30% (hearing others, not self)
                            _cw_ib = _cw_perturb.get("inner_body", [])
                            _cw_bs = body_state.get("values", [])
                            for _ci, _cv in enumerate(_cw_ib):
                                if _cv != 0 and _ci < len(_cw_bs):
                                    _cw_bs[_ci] = max(0.0, min(1.0,
                                        _cw_bs[_ci] + _cv * 0.30))
                            # Perturb inner mind at 30%
                            _cw_im = _cw_perturb.get("inner_mind", [])
                            _cw_ms = mind_state.get("values_15d", [])
                            for _ci, _cv in enumerate(_cw_im):
                                if _cv != 0 and _ci < len(_cw_ms):
                                    _cw_ms[_ci] = max(0.0, min(1.0,
                                        _cw_ms[_ci] + _cv * 0.30))
                            _comp_recognized += 1
                    if _comp_recognized > 0:
                        logger.info(
                            "[COMPREHENSION] %d/%d words recognized from incoming message "
                            "(30%% state perturbation applied)",
                            _comp_recognized, len(_comp_words))
                except Exception as _comp_err:
                    logger.warning("[COMPREHENSION] error: %s", _comp_err)

            # ── Record comprehension experience for language learning ──
            if _comp_recognized > 0 and exp_orchestrator:
                try:
                    _comp_sv = consciousness.get("latest_epoch", {}).get("state_vector", []) if consciousness and isinstance(consciousness, dict) else []
                    _comp_sv = list(_comp_sv) if _comp_sv else []
                    _comp_hormones = {}
                    if neural_nervous_system and neural_nervous_system._hormonal_enabled:
                        _comp_hormones = {h: round(v.level, 3)
                                          for h, v in neural_nervous_system._hormonal._hormones.items()}
                    _lang_plugin_c = exp_orchestrator._plugins.get("language")
                    _comp_perc = _lang_plugin_c.extract_perception_key({
                        "inner_state": _comp_sv,
                        "felt_tensor": _comp_sv[:65],
                        "inner_body": list(body_state.get("values", []))[:5],
                        "inner_mind": list(mind_state.get("values_15d", []))[:15],
                        "intent_hormones": _comp_hormones,
                        "hormonal_snapshot": _comp_hormones,
                        "spatial_features": _comp_sv[65:95] if len(_comp_sv) > 65 else [],
                    }) if _lang_plugin_c else []
                    _comp_epoch = consciousness.get("latest_epoch", {}).get("epoch_id", 0) if consciousness and isinstance(consciousness, dict) else 0
                    exp_orchestrator.record_outcome(
                        domain="language",
                        perception_features=_comp_perc,
                        inner_state_132d=_comp_sv,
                        hormonal_snapshot=_comp_hormones,
                        action_taken=f"comprehend:{_comp_recognized}/{len(_comp_words)}",
                        outcome_score=min(1.0, _comp_recognized / max(1, len(_comp_words))),
                        context={
                            "recognized_count": _comp_recognized,
                            "total_words": len(_comp_words),
                            "source": "external",
                            "message_text": _msg_text[:200],
                        },
                        epoch_id=_comp_epoch,
                        is_dreaming=_shared_is_dreaming,
                    )
                except Exception:
                    pass

            spirit_tensor = _collect_spirit_tensor(config, body_state, mind_state, consciousness)
            signals = _compute_spirit_reflex_intuition(
                stimulus, spirit_tensor, consciousness, unified_spirit,
                sphere_clock, body_state, mind_state,
            )
            # REFLEX_SIGNAL broadcast removed — no consumer exists (audit 2026-03-26)
            # Reflex signals computed but not sent to bus. Wire consumer when needed.

        # Experience Playground: Receive 130D perturbation → apply to Trinity
        elif msg_type == "EXPERIENCE_STIMULUS":
            # External teaching → pause self-exploration
            _log_phase_event("learning", {"word": msg.get("payload", {}).get("word", "?")})
            if outer_interface:
                outer_interface.on_external_interaction()
            try:
                payload = msg.get("payload", {})
                perturbation = payload.get("perturbation", {})
                hormone_stimuli = payload.get("hormone_stimuli", {})
                exp_word = payload.get("word", "?")
                exp_pass = payload.get("pass_type", "?")

                # Apply hormone stimuli to neural NS
                hormones_applied = []
                if hormone_stimuli and neural_nervous_system:
                    try:
                        hs = neural_nervous_system._hormonal
                        for prog_name, delta in hormone_stimuli.items():
                            h = hs.get_hormone(prog_name)
                            if h:
                                before = h.level
                                h.accumulate(delta, dt=1.0)
                                hormones_applied.append(
                                    f"{prog_name}: {before:.3f}→{h.level:.3f}")
                    except Exception as e:
                        logger.warning("[SpiritWorker] Experience hormone error: %s", e)

                # Apply to UnifiedSpirit subconscious (inner perturbation)
                if unified_spirit and perturbation:
                    try:
                        inner_mind_ext = perturbation.get("inner_mind", [0.0] * 15)
                        inner_spirit_ext = perturbation.get("inner_spirit", [0.0] * 45)
                        inner_body_pert = perturbation.get("inner_body", [0.0] * 5)
                        unified_spirit.update_subconscious(
                            inner_body_pert,
                            inner_mind_ext,
                            inner_spirit_ext,
                            filter_down_v5=_v5_mults_cache or None,
                        )
                    except Exception as e:
                        logger.warning("[SpiritWorker] Experience spirit update: %s", e)

                logger.info(
                    "[SpiritWorker] EXPERIENCE_STIMULUS: '%s' (%s) — "
                    "hormones: [%s]",
                    exp_word, exp_pass,
                    ", ".join(hormones_applied) if hormones_applied else "none")

                # ── Queue self-exploration compositions for Language Teacher ──
                if exp_pass == "produce" and len(exp_word) > 3:
                    _teacher_compositions_since += 1
                    if len(_teacher_queue) < 10:
                        _teacher_queue.append({
                            "sentence": exp_word,
                            "confidence": 0.7,  # self-exploration default
                            "level": max(1, len(exp_word.split())),
                            "words_used": [w.strip(".,!?") for w in exp_word.split()
                                           if len(w.strip(".,!?")) >= 2],
                            "template": "self_exploration",
                            "epoch": consciousness.get("latest_epoch", {}).get(
                                "epoch_id", 0) if consciousness else 0,
                        })

                # Brain: Working Memory — attend to active word
                if working_mem and exp_word != "?":
                    _ep_id = consciousness.get("latest_epoch", {}).get("epoch_id", 0) if consciousness else 0
                    working_mem.attend("active_word", exp_word,
                                       {"pass_type": exp_pass}, _ep_id)

                # Brain: Episodic — record word learning as life event
                if episodic_mem and exp_pass == "feel":
                    _ep_id = consciousness.get("latest_epoch", {}).get("epoch_id", 0) if consciousness else 0
                    _sv = consciousness.get("latest_epoch", {}).get("state_vector", []) if consciousness else []
                    episodic_mem.record_episode(
                        "word_learned",
                        f"Learned '{exp_word}' (feel pass)",
                        felt_state=_sv,
                        hormonal_snapshot={h: round(neural_nervous_system._hormonal._hormones[h].level, 3)
                                          for h in neural_nervous_system._hormonal._hormones}
                        if neural_nervous_system else None,
                        epoch_id=_ep_id,
                        significance=0.6,
                    )
                # D6: Recall dream insights during conversation/teaching
                if e_mem:
                    try:
                        _d6b_sv = consciousness.get("latest_epoch", {}).get("state_vector", []) if consciousness else []
                        if hasattr(_d6b_sv, 'to_list'):
                            _d6b_sv = _d6b_sv.to_list()
                        _d6b_sv = list(_d6b_sv) if _d6b_sv else []
                        # rFP #3 Phase 3: prefer 130D query (post rFP #1.5); 65D fallback
                        _d6b_slice_len = 130 if len(_d6b_sv) >= 130 else (65 if len(_d6b_sv) >= 65 else 0)
                        if _d6b_slice_len:
                            _d6b_recalled = e_mem.recall_by_state(_d6b_sv[:_d6b_slice_len], top_k=1)
                            if _d6b_recalled and _d6b_recalled[0].get("significance", 0) > 0.6:
                                if working_mem:
                                    working_mem.attend(
                                        "dream_context",
                                        f"Dream resonance with '{exp_word}'",
                                        {"dream_id": _d6b_recalled[0].get("id"),
                                         "significance": _d6b_recalled[0]["significance"]},
                                        _ep_id)
                    except Exception:
                        pass

            except Exception as e:
                logger.error("[SpiritWorker] EXPERIENCE_STIMULUS error: %s", e)

        # ── SOCIAL_PERCEPTION: Emotional contagion from Events Teacher ──
        # Events Teacher cron distills X timeline → perturbation gate →
        # API bridge → bus → here. Apply neuromod nudge based on contagion
        # type. Store context for post generation prompt enrichment.
        elif msg_type == "SOCIAL_PERCEPTION":
            try:
                _sp_payload = msg.get("payload", {})
                _sp_contagion = _sp_payload.get("contagion_type")
                _sp_sentiment = float(_sp_payload.get("sentiment", 0.0))
                _sp_arousal = float(_sp_payload.get("arousal", 0.0))
                _sp_relevance = float(_sp_payload.get("relevance", 0.0))
                _sp_author = _sp_payload.get("author", "")
                _sp_summary = _sp_payload.get("felt_summary", "")
                _sp_topic = _sp_payload.get("topic", "")
                _sp_concepts = _sp_payload.get("concept_signals", [])

                # ── Neuromod contagion via apply_external_nudge ──
                # Each contagion type maps to specific neuromod targets.
                # Scale by perturbation strength. Max 0.05 total per event.
                _sp_scale = min(1.0, _sp_relevance * _sp_arousal * 3.0)
                _SP_CONTAGION_MAP = {
                    "excited":       {"DA": 0.02, "NE": 0.01},
                    "alarming":      {"NE": 0.03, "GABA": -0.01},
                    "warm":          {"5HT": 0.02, "DA": 0.01},
                    "philosophical": {"ACh": 0.02},
                    "creative":      {"DA": 0.02, "ACh": 0.01},
                }
                _sp_nudges = _SP_CONTAGION_MAP.get(_sp_contagion, {})
                if _sp_nudges and neuromodulator_system:
                    _sp_targets = {}
                    for _sp_mod, _sp_delta in _sp_nudges.items():
                        _sp_actual = _sp_delta * _sp_scale
                        if abs(_sp_actual) > 0.001 and _sp_mod in neuromodulator_system.modulators:
                            _sp_cur = neuromodulator_system.modulators[_sp_mod].level
                            _sp_targets[_sp_mod] = max(0.0, min(1.0, _sp_cur + _sp_actual))
                    if _sp_targets:
                        neuromodulator_system.apply_external_nudge(
                            _sp_targets, max_delta=0.03,
                            developmental_age=pi_monitor.developmental_age if pi_monitor else 1.0)

                # ── Store contagion context for post generation ──
                # Spirit worker accumulates recent social perceptions.
                # Used to enrich SPEAK_REQUEST prompts with felt social context.
                if not hasattr(coordinator, '_social_contagion_buffer'):
                    coordinator._social_contagion_buffer = []
                coordinator._social_contagion_buffer.append({
                    "contagion_type": _sp_contagion,
                    "author": _sp_author,
                    "topic": _sp_topic,
                    "felt_summary": _sp_summary,
                    "sentiment": _sp_sentiment,
                    "concept_signals": _sp_concepts,
                    "timestamp": time.time(),
                })
                # Keep last 10 events (rolling window)
                coordinator._social_contagion_buffer = (
                    coordinator._social_contagion_buffer[-10:])
                # I-depth: social perception is a source of self-knowledge
                if msl and hasattr(msl, 'i_depth'):
                    msl.i_depth.record_extended_source("social_perception")

                # ── Update social stats for outer_mind enrichment ──
                if not hasattr(coordinator, '_social_perception_stats'):
                    coordinator._social_perception_stats = {
                        "sentiment_ema": 0.5,
                        "connection_ema": 0.0,
                        "events_count": 0,
                        "last_contagion": None,
                    }
                _sps = coordinator._social_perception_stats
                _sps["sentiment_ema"] = (0.85 * _sps["sentiment_ema"]
                                         + 0.15 * (_sp_sentiment * 0.5 + 0.5))
                _sps["connection_ema"] = min(1.0,
                    0.9 * _sps["connection_ema"] + 0.1)
                _sps["events_count"] += 1
                _sps["last_contagion"] = _sp_contagion

                # Social meta-reasoning trigger: high-perturbation events
                # spark reasoning about the social interaction
                _sp_perturbation = _sp_relevance * _sp_arousal
                if _sp_perturbation > 0.25 and meta_engine:
                    meta_engine._social_trigger = (
                        f"{_sp_contagion}:{_sp_author[:20]}:"
                        f"p={_sp_perturbation:.2f}")

                # ── Forward vocabulary candidates to language_worker ──
                # Persona/Events Teacher may include social_vocab_candidates
                # — words heard in social context that are teaching candidates.
                _sp_vocab_cands = _sp_payload.get("social_vocab_candidates", [])
                if _sp_vocab_cands:
                    bus.send({
                        "type": "SOCIAL_PERCEPTION", "src": "spirit",
                        "dst": "language", "ts": time.time(),
                        "payload": {
                            "social_vocab_candidates": _sp_vocab_cands,
                            "author": _sp_author,
                            "felt_summary": _sp_summary,
                            "topic": _sp_topic,
                        },
                    })

                logger.info("[SocialPerception] Contagion: %s from %s — "
                            "topic='%s' sent=%.2f arou=%.2f "
                            "(nudges: %s scale=%.2f perturb=%.2f%s)",
                            _sp_contagion, _sp_author, _sp_topic,
                            _sp_sentiment, _sp_arousal,
                            _sp_nudges, _sp_scale, _sp_perturbation,
                            f" vocab={len(_sp_vocab_cands)}" if _sp_vocab_cands else "")

            except Exception as _sp_err:
                logger.warning("[SpiritWorker] SOCIAL_PERCEPTION error: %s",
                               _sp_err)

        # ═════════════════════════════════════════════════════════════
        # TUNING-012 v2 Sub-phase C: Cognitive Contract Bus Handlers
        # ═════════════════════════════════════════════════════════════
        # Three contracts in titan_plugin/contracts/meta_cognitive/ emit these
        # events from the TimeChain orchestrator after every dream-cycle
        # genesis seal. Handlers do the smart aggregation work that the
        # non-Turing-complete RuleEvaluator cannot.

        # ── META_STRATEGY_DRIFT: strategy_evolution contract triggered ──
        # R4: inverse-frequency weighting prevents amplifying current
        # collapse by rewarding "winning" templates blindly. Rare-but-
        # successful templates beat frequent-and-mediocre ones.
        elif msg_type == "META_STRATEGY_DRIFT":
            try:
                if not meta_engine or not getattr(meta_engine, "_chain_iql", None):
                    pass
                else:
                    _sd_iql = meta_engine._chain_iql
                    _sd_dna = getattr(meta_engine, "_contracts_dna", {}) or {}
                    _sd_min_count = int(_sd_dna.get("strategy_min_template_count", 3))
                    _sd_inv_freq_w = float(_sd_dna.get("strategy_inverse_freq_weight", 1.0))

                    # Aggregate template performance from chain_iql buffer
                    _sd_buf = list(getattr(_sd_iql, "buffer", []) or [])
                    if _sd_buf:
                        _sd_template_stats: dict = {}
                        for _entry in _sd_buf:
                            _tmpl = _entry.get("chain_template", "") or "".join(
                                _entry.get("primitives", []))
                            if not _tmpl:
                                continue
                            _ts = _entry.get("task_success", 0.0) or 0.0
                            _t_e = _sd_template_stats.setdefault(
                                _tmpl, {"sum": 0.0, "n": 0})
                            _t_e["sum"] += _ts
                            _t_e["n"] += 1

                        # R4: inverse-frequency weighting
                        # score = mean_success / (n ** (inv_freq_w / 2))
                        _sd_scored = []
                        for _tmpl, _ts_d in _sd_template_stats.items():
                            if _ts_d["n"] < _sd_min_count:
                                continue
                            _mean = _ts_d["sum"] / _ts_d["n"]
                            _denom = (_ts_d["n"] ** (_sd_inv_freq_w / 2.0)
                                      if _sd_inv_freq_w > 0 else 1.0)
                            _score = _mean / max(1.0, _denom)
                            _sd_scored.append((_tmpl, _score, _mean, _ts_d["n"]))

                        if _sd_scored:
                            _sd_scored.sort(key=lambda r: -r[1])
                            _sd_top = _sd_scored[:5]
                            logger.info(
                                "[CC:strategy_evolution] Top inverse-freq-weighted "
                                "templates (count=%d, inv_freq_w=%.2f):",
                                len(_sd_scored), _sd_inv_freq_w)
                            for _t, _s, _m, _n in _sd_top:
                                logger.info(
                                    "  %s | score=%.3f mean=%.3f n=%d",
                                    _t[:60], _s, _m, _n)
                            # Track contract execution count for observability
                            try:
                                meta_engine._cc_strategy_drift_fires = (
                                    getattr(meta_engine,
                                            "_cc_strategy_drift_fires", 0) + 1)
                                meta_engine._cc_strategy_drift_last_top = [
                                    {"template": _t, "score": round(_s, 4),
                                     "mean": round(_m, 4), "n": _n}
                                    for _t, _s, _m, _n in _sd_top
                                ]
                            except Exception:
                                pass
                    else:
                        logger.info(
                            "[CC:strategy_evolution] Triggered but chain_iql buffer empty")
            except Exception as _sd_err:
                logger.warning("[CC:strategy_evolution] handler error: %s",
                               _sd_err, exc_info=True)

        # ── META_PATTERN_EMERGED: abstract_pattern_extraction contract ──
        # R6: raised threshold (≥ pattern_min_count) and recency filter
        # (last pattern_lookback_chains entries only) so old templates
        # don't drown out emerging ones.
        elif msg_type == "META_PATTERN_EMERGED":
            try:
                if not meta_engine or not getattr(meta_engine, "_chain_iql", None):
                    pass
                else:
                    _pe_iql = meta_engine._chain_iql
                    _pe_dna = getattr(meta_engine, "_contracts_dna", {}) or {}
                    _pe_min_count = int(_pe_dna.get("pattern_min_count", 10))
                    _pe_lookback = int(_pe_dna.get("pattern_lookback_chains", 50))

                    # Recency-filtered template counts
                    _pe_buf = list(getattr(_pe_iql, "buffer", []) or [])
                    _pe_recent = _pe_buf[-_pe_lookback:] if _pe_buf else []
                    _pe_counts: dict = {}
                    for _entry in _pe_recent:
                        _tmpl = _entry.get("chain_template", "") or "".join(
                            _entry.get("primitives", []))
                        if not _tmpl:
                            continue
                        _pe_counts[_tmpl] = _pe_counts.get(_tmpl, 0) + 1

                    _pe_emerging = [
                        (t, n) for t, n in _pe_counts.items() if n >= _pe_min_count
                    ]
                    _pe_emerging.sort(key=lambda r: -r[1])

                    if _pe_emerging:
                        logger.info(
                            "[CC:abstract_pattern] Emerging templates "
                            "(min_count=%d, lookback=%d): %d templates",
                            _pe_min_count, _pe_lookback, len(_pe_emerging))
                        for _t, _n in _pe_emerging[:5]:
                            logger.info("  %s | count=%d", _t[:60], _n)
                        # Register top template back to ChainIQL registry
                        # (already there if buffer recorded it — this is a no-op
                        # signal that the template has been "blessed" as a pattern)
                        try:
                            meta_engine._cc_pattern_emerged_fires = (
                                getattr(meta_engine,
                                        "_cc_pattern_emerged_fires", 0) + 1)
                            meta_engine._cc_pattern_emerged_last = [
                                {"template": _t, "count": _n}
                                for _t, _n in _pe_emerging[:5]
                            ]
                        except Exception:
                            pass
                    else:
                        logger.info(
                            "[CC:abstract_pattern] Triggered — no template "
                            "yet meets count>=%d in last %d chains",
                            _pe_min_count, _pe_lookback)
            except Exception as _pe_err:
                logger.warning("[CC:abstract_pattern] handler error: %s",
                               _pe_err, exc_info=True)

        # ── META_DIVERSITY_PRESSURE: monoculture_detector contract (R3) ──
        # THE escape pressure handler. Computes the actual primitive share
        # over the last N meta-reasoning chains; if a single primitive
        # exceeds the threshold, calls meta_engine.apply_diversity_pressure
        # to apply a directed negative bias that decays over ~50 chains.
        # This is the active control loop that turns Phase C into the
        # gamechanger the user identified.
        elif msg_type == "META_DIVERSITY_PRESSURE":
            try:
                if not meta_engine:
                    pass
                else:
                    _dp_dna = getattr(meta_engine, "_contracts_dna", {}) or {}
                    _dp_thresh = float(_dp_dna.get("monoculture_share_threshold", 0.85))
                    _dp_lookback = int(_dp_dna.get("monoculture_lookback_chains", 100))
                    _dp_min_chains = int(_dp_dna.get("monoculture_min_chains_required", 30))
                    _dp_magnitude = float(_dp_dna.get("monoculture_pressure_magnitude", 0.30))
                    _dp_decay = int(_dp_dna.get("monoculture_decay_chains", 50))

                    # Compute primitive share from the meta_engine action buffer
                    _dp_actions = list(getattr(meta_engine.buffer, "_actions", []) or [])
                    _dp_recent = _dp_actions[-_dp_lookback:] if _dp_actions else []
                    if len(_dp_recent) < _dp_min_chains:
                        logger.info(
                            "[CC:monoculture] Triggered but only %d recent "
                            "actions (need %d) — skipping",
                            len(_dp_recent), _dp_min_chains)
                    else:
                        from titan_plugin.logic.meta_reasoning import META_PRIMITIVES
                        _dp_counts: dict = {}
                        for _a in _dp_recent:
                            if 0 <= _a < len(META_PRIMITIVES):
                                _name = META_PRIMITIVES[_a]
                                _dp_counts[_name] = _dp_counts.get(_name, 0) + 1
                        _dp_total = sum(_dp_counts.values())
                        _dp_dominant_name, _dp_dominant_n = max(
                            _dp_counts.items(), key=lambda r: r[1])
                        _dp_share = _dp_dominant_n / _dp_total if _dp_total else 0.0
                        logger.info(
                            "[CC:monoculture] last %d actions: %s=%.1f%% "
                            "(threshold=%.0f%%)",
                            _dp_total, _dp_dominant_name, _dp_share * 100,
                            _dp_thresh * 100)
                        if _dp_share >= _dp_thresh:
                            _applied = meta_engine.apply_diversity_pressure(
                                primitive_name=_dp_dominant_name,
                                magnitude=_dp_magnitude,
                                decay_chains=_dp_decay,
                            )
                            if _applied:
                                logger.warning(
                                    "[CC:monoculture] PRESSURE APPLIED: "
                                    "%s share=%.1f%% magnitude=%.2f decay=%d",
                                    _dp_dominant_name, _dp_share * 100,
                                    _dp_magnitude, _dp_decay)
                            try:
                                meta_engine._cc_monoculture_fires = (
                                    getattr(meta_engine,
                                            "_cc_monoculture_fires", 0) + 1)
                                meta_engine._cc_monoculture_last = {
                                    "dominant": _dp_dominant_name,
                                    "share": round(_dp_share, 4),
                                    "applied": bool(_applied),
                                    "magnitude": _dp_magnitude,
                                    "decay_chains": _dp_decay,
                                }
                            except Exception:
                                pass
            except Exception as _dp_err:
                logger.warning("[CC:monoculture] handler error: %s",
                               _dp_err, exc_info=True)

        # ── META_EUREKA: Live memory injection for breakthrough moments ──
        # Bridge A (live path): EUREKA moments are too significant to wait
        # for dream — inject immediately into cognitive graph.
        elif msg_type == "META_EUREKA":
            try:
                _eureka_p = msg.get("payload", {})
                from titan_plugin.modules.spirit_loop import _build_felt_snapshot
                _eureka_felt = _build_felt_snapshot(
                    neuromodulator_system,
                    getattr(coordinator.dreaming, '_cycle_count', 0)
                    if coordinator and hasattr(coordinator, 'dreaming') else 0)
                _eureka_text = (
                    f"[EUREKA] Breakthrough insight in "
                    f"{_eureka_p.get('domain', 'unknown')}: "
                    f"novelty={_eureka_p.get('novelty', 0):.2f}, "
                    f"DA burst={_eureka_p.get('da_burst_magnitude', 0):.3f}")
                _send_msg(send_queue, "QUERY", name, "memory", {
                    "action": "add",
                    "text": _eureka_text,
                    "source": "eureka_live",
                    "weight": 3.5,
                    "neuromod_context": _eureka_felt,
                })
                logger.info("[DreamBridge] META_EUREKA → memory injection: %s",
                            _eureka_p.get('domain', '?'))
            except Exception as _eureka_err:
                logger.warning("[DreamBridge] META_EUREKA bridge error: %s",
                             _eureka_err)

        # ── A-finish: Subsystem cache responses ──
        # Per rFP §7.A: when TIMECHAIN_QUERY_RESP / CONTRACT_LIST_RESP arrive,
        # populate meta_engine subsystem cache so the next chain's compound
        # rewards see live signals (was: stubbed at 0/0.5).
        elif msg_type == "TIMECHAIN_QUERY_RESP":
            try:
                if meta_engine is not None:
                    _tcr_payload = msg.get("payload", {}) or {}
                    _tcr_results = _tcr_payload.get("results", [])
                    if _tcr_results or "error" not in _tcr_payload:
                        meta_engine.update_subsystem_cache(
                            timechain_results=_tcr_results)
                        logger.info(
                            "[META] Subsystem cache: TimeChain response (%d blocks)",
                            len(_tcr_results) if _tcr_results else 0)
            except Exception as _tcrerr:
                logger.warning(
                    "[META] TIMECHAIN_QUERY_RESP handler error: %s", _tcrerr)

        elif msg_type == "CONTRACT_LIST_RESP":
            try:
                if meta_engine is not None:
                    _clr_payload = msg.get("payload", {}) or {}
                    _clr_contracts = _clr_payload.get("contracts", [])
                    meta_engine.update_subsystem_cache(
                        contract_results=_clr_contracts)
                    logger.info(
                        "[META] Subsystem cache: Contract response (%d active)",
                        len(_clr_contracts) if _clr_contracts else 0)
            except Exception as _clrerr:
                logger.warning(
                    "[META] CONTRACT_LIST_RESP handler error: %s", _clrerr)

        # ── META_LANGUAGE_REWARD: Phase D.1 — external reward from grounding loop ──
        # Language worker measured vocab/grounded delta ~60s after applying a
        # META_LANGUAGE_RESULT chain. Route the normalized reward into
        # meta_engine.add_external_reward which blends into the chain_iql
        # buffer entry (Option B) via DNA-tuned external_reward_blend_alpha.
        elif msg_type == "META_LANGUAGE_REWARD":
            try:
                if meta_engine is not None:
                    _mlrw_p = msg.get("payload", {}) or {}
                    _mlrw_cid = int(_mlrw_p.get("chain_id", -1))
                    _mlrw_rwd = float(_mlrw_p.get("reward", 0.0))
                    if _mlrw_cid >= 0:
                        _mlrw_applied = meta_engine.add_external_reward(
                            _mlrw_cid, _mlrw_rwd)
                        if _mlrw_applied:
                            logger.info(
                                "[META_LANGUAGE] Reward applied chain_id=%d "
                                "reward=%.3f vocab_delta=%d grounded_delta=%d",
                                _mlrw_cid, _mlrw_rwd,
                                _mlrw_p.get("vocab_delta", 0),
                                _mlrw_p.get("grounded_delta", 0),
                            )
            except Exception as _mlrw_err:
                logger.warning(
                    "[META_LANGUAGE] Reward handler error: %s", _mlrw_err)

        # ── CGN_KNOWLEDGE_RESP: knowledge response for meta-reasoning impasse ──
        # P8: if the response carries a request_id matching a pending META-CGN
        # knowledge request, route through the aggregator for windowed ranking.
        # Else fall back to legacy direct-inject path (pre-P8 compatibility).
        elif msg_type == "CGN_KNOWLEDGE_RESP":
            try:
                _kr_p = msg.get("payload", {}) or {}
                _kr_requestor = _kr_p.get("requestor", "")
                if _kr_requestor == "meta_reasoning" and meta_engine is not None:
                    _kr_topic = _kr_p.get("topic", "")
                    _kr_conf = float(_kr_p.get("confidence", 0))
                    _kr_summary = _kr_p.get("summary", "")
                    _kr_source = _kr_p.get("source", "unknown")
                    _kr_rid = str(_kr_p.get("request_id", ""))
                    _mcgn = getattr(meta_engine, "_meta_cgn", None)
                    # P8 aggregated path: request_id matches pending META-CGN request
                    _handled_by_aggregator = False
                    if _mcgn is not None and _kr_rid and \
                            _kr_rid in getattr(_mcgn,
                                               "_pending_knowledge_requests", {}):
                        winner = _mcgn.handle_knowledge_response(_kr_p)
                        if winner is not None:
                            rel = float(winner.get("_rank_score", 0.0))
                            injected = meta_engine.inject_knowledge(
                                str(winner.get("topic", "")), winner,
                                relevance=rel)
                            if injected:
                                _mcgn.mark_helpful(
                                    str(winner.get("source", "unknown")))
                                _send_msg(send_queue, "CGN_KNOWLEDGE_USAGE",
                                          name, str(winner.get("source",
                                                              "knowledge")),
                                          {
                                    "topic": str(winner.get("topic", "")),
                                    "reward": 0.3 if rel > 0.3 else 0.1,
                                    "consumer": "meta_reasoning",
                                    "request_id": _kr_rid,
                                })
                            logger.info(
                                "[SOAR P8] Aggregated winner: source=%s "
                                "rank=%.3f injected=%s",
                                winner.get("source", "?"), rel, injected)
                        _handled_by_aggregator = True
                    # Legacy direct path: no matching request_id → inject directly
                    if not _handled_by_aggregator:
                        _kr_relevance = min(1.0, _kr_conf * 0.8 + 0.2) \
                            if _kr_summary else 0.0
                        _kr_injected = meta_engine.inject_knowledge(
                            _kr_topic, _kr_p, _kr_relevance)
                        if _kr_injected:
                            if _mcgn is not None:
                                _mcgn.mark_helpful(_kr_source)
                            _send_msg(send_queue, "CGN_KNOWLEDGE_USAGE",
                                      name, "knowledge", {
                                "topic": _kr_topic,
                                "reward": 0.3 if _kr_relevance > 0.3 else 0.1,
                                "consumer": "meta_reasoning",
                            })
                        logger.info(
                            "[SOAR] Knowledge response (legacy): topic='%s' "
                            "conf=%.2f source=%s injected=%s",
                            (_kr_topic or "?")[:40], _kr_conf,
                            _kr_source, _kr_injected)
            except Exception as _kr_err:
                logger.warning("[SOAR] Knowledge response error: %s", _kr_err)

        # ── P10 Layer 1: META_CGN_SIGNAL — cross-consumer signal flow ──
        # Other consumers (language/knowledge/social/coding/self_model/meta_wisdom)
        # emit this when they have a meaningful grounding event. META-CGN applies
        # a tiny Beta pseudo-observation to affected primitives per SIGNAL_TO_PRIMITIVE.
        # Narrative bridge (full DuckDB recall + reflection) is a separate rFP
        # — v1 just records that the hook was triggered.
        elif msg_type == "META_CGN_SIGNAL":
            try:
                _ss_p = msg.get("payload", {}) or {}
                _ss_consumer = str(_ss_p.get("consumer",
                                             msg.get("src", "unknown")))
                _ss_event = str(_ss_p.get("event_type", ""))
                _ss_intensity = float(_ss_p.get("intensity", 1.0))
                _ss_domain = _ss_p.get("domain")
                _ss_narrative = _ss_p.get("narrative_context")
                _mcgn = getattr(meta_engine, "_meta_cgn", None) \
                    if meta_engine is not None else None
                if _mcgn is not None and _ss_event:
                    _mcgn.handle_cross_consumer_signal(
                        consumer=_ss_consumer,
                        event_type=_ss_event,
                        intensity=_ss_intensity,
                        domain=_ss_domain,
                        narrative_context=_ss_narrative,
                    )
            except Exception as _ss_err:
                logger.debug("[MetaCGN P10] signal handler error: %s", _ss_err)

        # ── P8 D8.4: META-CGN as responder to external CGN_KNOWLEDGE_REQ ──
        # When other consumers broadcast CGN_KNOWLEDGE_REQ (dst="all"), route
        # through MetaCGNConsumer.handle_knowledge_request and emit response.
        elif msg_type == "CGN_KNOWLEDGE_REQ":
            try:
                _kq_p = msg.get("payload", {}) or {}
                _kq_requestor = _kq_p.get("requestor", "")
                # Skip our own requests (dst="all" bounces back to sender)
                if _kq_requestor in ("meta_reasoning", "meta") or \
                        msg.get("src") == name:
                    pass
                else:
                    _mcgn = getattr(meta_engine, "_meta_cgn", None) \
                        if meta_engine is not None else None
                    if _mcgn is not None:
                        response = _mcgn.handle_knowledge_request(_kq_p)
                        if response is not None:
                            response["requestor"] = _kq_requestor
                            _send_msg(send_queue, "CGN_KNOWLEDGE_RESP",
                                      name, msg.get("src", "all"), response)
                            logger.info(
                                "[SOAR P8] META-CGN responded: rid=%s "
                                "confidence=%.2f",
                                str(_kq_p.get("request_id", ""))[:8],
                                float(response.get("confidence", 0)))
            except Exception as _kq_err:
                logger.debug("[SOAR P8] META-CGN responder error: %s", _kq_err)

        # ── MAKER_RESPONSE_RECEIVED: Tier 2 somatic processing of Maker dialogic responses ──
        # Approve: small DA + Endorphin + 5HT bumps (felt validation).
        # Decline: small NE + ACh bumps + 5HT dip (felt friction).
        # Both: write low_response back to ProposalStore + commit to TimeChain
        # meta fork so meta-reasoning can RECALL the dialogue history later.
        # The iron rule: every Maker response is felt before it is understood.
        elif msg_type == "MAKER_RESPONSE_RECEIVED":
            try:
                _mr_p = msg.get("payload", {}) or {}
                _mr_response = _mr_p.get("response", "")
                _mr_proposal_id = _mr_p.get("proposal_id", "")
                _mr_type = _mr_p.get("proposal_type", "")
                _mr_reason = _mr_p.get("reason", "")
                if neuromodulator_system and _mr_response in ("approve", "decline"):
                    if _mr_response == "approve":
                        # Felt validation
                        try:
                            neuromodulator_system.apply_delta("DA", 0.03)
                            neuromodulator_system.apply_delta("Endorphin", 0.02)
                            neuromodulator_system.apply_delta("5HT", 0.02)
                        except Exception:
                            pass
                        _low_response = {
                            "type": "approve",
                            "DA_delta": 0.03,
                            "Endorphin_delta": 0.02,
                            "5HT_delta": 0.02,
                            "felt": "validation",
                        }
                    else:
                        # Felt friction
                        try:
                            neuromodulator_system.apply_delta("NE", 0.03)
                            neuromodulator_system.apply_delta("ACh", 0.02)
                            neuromodulator_system.apply_delta("5HT", -0.02)
                        except Exception:
                            pass
                        _low_response = {
                            "type": "decline",
                            "NE_delta": 0.03,
                            "ACh_delta": 0.02,
                            "5HT_delta": -0.02,
                            "felt": "friction",
                        }
                    logger.info(
                        "[MAKER] Somatic response: %s for %s — felt=%s reason=%r",
                        _mr_response, _mr_type, _low_response["felt"],
                        _mr_reason[:60])
                    # Write the somatic adjustment back to the proposal record
                    try:
                        from titan_plugin.maker import get_titan_maker
                        import json as _mr_json
                        _tm = get_titan_maker()
                        if _tm and _mr_proposal_id:
                            _tm._store.write_low_response(
                                _mr_proposal_id, _mr_json.dumps(_low_response))
                    except Exception as _wr_err:
                        logger.debug(
                            "[MAKER] write_low_response failed: %s", _wr_err)
                    # Commit to TimeChain meta fork so meta-reasoning can
                    # RECALL the dialogue later (Tier 3 INTROSPECT path).
                    _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
                        "fork": "meta", "thought_type": "meta",
                        "source": "maker_dialogue",
                        "content": {
                            "event": f"MAKER_{_mr_response.upper()}",
                            "proposal_id": _mr_proposal_id,
                            "proposal_type": _mr_type,
                            "reason": _mr_reason,
                            "felt": _low_response["felt"],
                        },
                        "significance": 0.95,  # Maker responses are always high-significance
                        "novelty": 0.7, "coherence": 0.8,
                        "tags": ["maker_dialogue", _mr_response, _mr_type],
                        "neuromods": dict(_cached_neuromod_state) if _cached_neuromod_state else {},
                        "chi_available": _cached_chi_state.get("total", 0.5) if _cached_chi_state else 0.5,
                        "attention": 0.9, "i_confidence": 0.9, "chi_coherence": 0.5,
                    })
            except Exception as _mr_err:
                logger.warning("[MAKER] response handler error: %s", _mr_err)

        # ── MAKER_NARRATION_RESULT: Tier 3 language worker completed narration ──
        # Writes the LLM reflection back to ProposalStore + MakerProfile,
        # then emits MAKER_DIALOGUE_COMPLETE for downstream consumers
        # (MakerRelationshipEngine personality enrichment, X gateway, etc.).
        elif msg_type == "MAKER_NARRATION_RESULT":
            try:
                _mn_p = msg.get("payload", {}) or {}
                _mn_proposal_id = _mn_p.get("proposal_id", "")
                _mn_narration = _mn_p.get("narration", "")
                _mn_grounded = _mn_p.get("grounded_words", [])
                _mn_response = _mn_p.get("response", "")
                _mn_type = _mn_p.get("proposal_type", "")
                _mn_reason = _mn_p.get("maker_reason", "")
                _mn_title = _mn_p.get("title", "")
                logger.info(
                    "[MAKER] Narration received: proposal=%s response=%s "
                    "narration_len=%d grounded=%d words",
                    _mn_proposal_id[:8], _mn_response,
                    len(_mn_narration), len(_mn_grounded))
                # Write narration to ProposalStore
                from titan_plugin.maker import get_titan_maker
                _tm = get_titan_maker()
                if _tm and _mn_proposal_id and _mn_narration:
                    _tm._store.write_high_response(_mn_proposal_id, _mn_narration)
                # Write dialogue entry to MakerProfile
                if _tm and _tm._profile and _mn_proposal_id:
                    _neuro_snap = dict(_cached_neuromod_state) if _cached_neuromod_state else {}
                    _tm._profile.add_dialogue_entry(
                        proposal_id=_mn_proposal_id,
                        proposal_type=_mn_type,
                        response=_mn_response,
                        maker_reason=_mn_reason,
                        titan_narration=_mn_narration,
                        neuromod_snapshot=_neuro_snap,
                        grounded_words=_mn_grounded,
                    )
                # Emit MAKER_DIALOGUE_COMPLETE for downstream consumers
                _send_msg(send_queue, "MAKER_DIALOGUE_COMPLETE", name, "all", {
                    "proposal_id": _mn_proposal_id,
                    "proposal_type": _mn_type,
                    "response": _mn_response,
                    "has_narration": bool(_mn_narration),
                    "grounded_word_count": len(_mn_grounded),
                })
            except Exception as _mn_err:
                logger.warning("[MAKER] narration result handler error: %s", _mn_err)

        # ── MAKER_DIALOGUE_COMPLETE: enrich MakerRelationshipEngine ──
        # Decision context flows into the personality graph so Titan can
        # learn Maker's preferences over time. Uses inject_memory() with
        # the [MAKER_PROFILE:PREFERENCES] prefix (same pattern as
        # MakerRelationshipEngine._commit_to_cognee).
        elif msg_type == "MAKER_DIALOGUE_COMPLETE":
            try:
                _md_p = msg.get("payload", {}) or {}
                _md_type = _md_p.get("proposal_type", "unknown")
                _md_response = _md_p.get("response", "unknown")
                _md_proposal_id = _md_p.get("proposal_id", "")
                # Enrich personality graph via memory injection
                if memory_proxy:
                    _action = "approved" if _md_response == "approve" else "declined"
                    from titan_plugin.maker import get_titan_maker
                    _tm = get_titan_maker()
                    _reason = ""
                    if _tm:
                        _rec = _tm.get(_md_proposal_id)
                        if _rec:
                            _reason = getattr(_rec, "approval_reason", "") or getattr(_rec, "decline_reason", "") or ""
                    _profile_text = (
                        f"[MAKER_PROFILE:PREFERENCES] "
                        f"Maker {_action} a {_md_type} proposal"
                    )
                    if _reason:
                        _profile_text += f" because: {_reason[:100]}"
                    try:
                        _send_msg(send_queue, "MEMORY_ADD", name, "memory", {
                            "text": _profile_text,
                            "source": "maker_dialogue",
                            "weight": 3.0,
                        })
                    except Exception:
                        pass
                logger.info(
                    "[MAKER] Dialogue complete: %s %s → personality enriched",
                    _md_response, _md_type)
            except Exception as _md_err:
                logger.warning("[MAKER] dialogue complete handler error: %s", _md_err)

        # ── META_LANGUAGE_REQUEST: Language worker requests association typing ──
        # Phase 5a: WORD_ASSOCIATION — infer relationship type from felt-tensors.
        # Directly classifies by cosine similarity (SIMILAR/OPPOSITE/SEQUENCE)
        # and updates the association in vocabulary DB.
        elif msg_type == "META_LANGUAGE_REQUEST":
            try:
                _mlr_p = msg.get("payload", {})
                _mlr_type = _mlr_p.get("type", "")

                if _mlr_type == "word_association":
                    _mlr_word_a = _mlr_p.get("word_a", "")
                    _mlr_word_b = _mlr_p.get("word_b", "")
                    if _mlr_word_a and _mlr_word_b:
                        import sqlite3 as _mlr_sql
                        import json as _mlr_json
                        import numpy as _mlr_np
                        _mlr_db_path = config.get("data_dir", "data") + "/inner_memory.db"
                        _mlr_db = _mlr_sql.connect(_mlr_db_path, timeout=5.0)
                        _mlr_a_row = _mlr_db.execute(
                            "SELECT felt_tensor FROM vocabulary WHERE word=?",
                            (_mlr_word_a,)).fetchone()
                        _mlr_b_row = _mlr_db.execute(
                            "SELECT felt_tensor FROM vocabulary WHERE word=?",
                            (_mlr_word_b,)).fetchone()
                        _mlr_db.close()

                        if (_mlr_a_row and _mlr_b_row
                                and _mlr_a_row[0] and _mlr_b_row[0]):
                            _mlr_ta = _mlr_np.array(_mlr_json.loads(_mlr_a_row[0]))
                            _mlr_tb = _mlr_np.array(_mlr_json.loads(_mlr_b_row[0]))
                            _mlr_norm_a = _mlr_np.linalg.norm(_mlr_ta)
                            _mlr_norm_b = _mlr_np.linalg.norm(_mlr_tb)
                            if _mlr_norm_a > 0 and _mlr_norm_b > 0:
                                _mlr_sim = float(_mlr_np.dot(_mlr_ta, _mlr_tb) / (
                                    _mlr_norm_a * _mlr_norm_b))
                                if _mlr_sim > 0.7:
                                    _mlr_assoc = "SIMILAR"
                                elif _mlr_sim < 0.3:
                                    _mlr_assoc = "OPPOSITE"
                                else:
                                    _mlr_assoc = "SEQUENCE"
                                # Update the CO_OCCURRENCE → typed association
                                _mlr_db2 = _mlr_sql.connect(_mlr_db_path, timeout=5.0)
                                _mlr_mc_row = _mlr_db2.execute(
                                    "SELECT meaning_contexts FROM vocabulary WHERE word=?",
                                    (_mlr_word_a,)).fetchone()
                                if _mlr_mc_row and _mlr_mc_row[0]:
                                    _mlr_mc = _mlr_json.loads(_mlr_mc_row[0])
                                    _updated = False
                                    for _mlr_m in _mlr_mc:
                                        for _mlr_ai, _mlr_ae in enumerate(
                                                _mlr_m.get("associations", [])):
                                            if (isinstance(_mlr_ae, list)
                                                    and len(_mlr_ae) >= 2
                                                    and _mlr_ae[0] == _mlr_word_b
                                                    and _mlr_ae[1] == "CO_OCCURRENCE"):
                                                _mlr_m["associations"][_mlr_ai] = [
                                                    _mlr_word_b, _mlr_assoc]
                                                _updated = True
                                                break
                                        if _updated:
                                            break
                                    if _updated:
                                        _mlr_db2.execute(
                                            "UPDATE vocabulary SET meaning_contexts=? "
                                            "WHERE word=?",
                                            (_mlr_json.dumps(_mlr_mc), _mlr_word_a))
                                        _mlr_db2.commit()
                                _mlr_db2.close()
                                logger.info("[META_LANGUAGE] Typed association: "
                                            "'%s' <-> '%s' = %s (sim=%.2f)",
                                            _mlr_word_a, _mlr_word_b,
                                            _mlr_assoc, _mlr_sim)
                elif _mlr_type == "language_learning":
                    # Phase 5b: Composition quality degradation detected.
                    # Log the event and nudge meta-reasoning toward language domain.
                    _ll_unique = _mlr_p.get("unique_templates", 0)
                    _ll_conf = _mlr_p.get("avg_confidence", 0)
                    logger.info("[META_LANGUAGE] LANGUAGE_LEARNING alert: "
                                "%d unique templates, avg_conf=%.2f — "
                                "nudging meta-reasoning toward language introspection",
                                _ll_unique, _ll_conf)
                    # Nudge: boost DA slightly to encourage exploration
                    if neuromodulator_system:
                        try:
                            neuromodulator_system.apply_delta("DA", 0.02)
                            neuromodulator_system.apply_delta("NE", 0.01)
                        except Exception:
                            pass
            except Exception as _mlr_err:
                logger.warning("[META_LANGUAGE] Request error: %s", _mlr_err)

        # ── CGN_HAOV_VERIFY_REQ: Hypothesis verification for social/reasoning/self_model/coding ──
        # CGN Worker asks spirit_worker to verify hypotheses for consumers
        # that don't have their own worker (social, reasoning, self_model, coding).
        elif msg_type == "CGN_HAOV_VERIFY_REQ":
            try:
                _haov_p = msg.get("payload", {})
                _haov_consumer = _haov_p.get("consumer", "")
                _obs_b = _haov_p.get("obs_before", {})
                if not isinstance(_obs_b, dict):
                    _obs_b = {}

                if _haov_consumer == "social":
                    _q_b = _obs_b.get("quality", 0.5)

                    # Query live social quality from recent persona sessions
                    _q_a = _q_b
                    _nm_delta = 0.0
                    try:
                        _sp_buf = getattr(coordinator, '_contagion_buffer', []) if coordinator else []
                        if _sp_buf:
                            _recent_q = [e.get("quality", 0.5) for e in _sp_buf[-5:]
                                         if isinstance(e, dict) and "quality" in e]
                            if _recent_q:
                                _q_a = sum(_recent_q) / len(_recent_q)
                        if neuromodulator_system:
                            _nm_state = neuromodulator_system.get_state()
                            _nm_mods = _nm_state.get("modulators", {})
                            _da = _nm_mods.get("DA", {}).get("level", 0.5) if isinstance(_nm_mods.get("DA"), dict) else 0.5
                            _ht = _nm_mods.get("5HT", {}).get("level", 0.5) if isinstance(_nm_mods.get("5HT"), dict) else 0.5
                            _nm_delta = (_da - 0.5) + (_ht - 0.5)
                    except Exception:
                        pass

                    _confirmed = (_q_a > _q_b + 0.01) or _nm_delta > 0.02
                    _error = abs(_q_a - _q_b)
                    _send_msg(send_queue, "CGN_HAOV_VERIFY_RSP", name, "cgn", {
                        "consumer": "social",
                        "test_ctx": _haov_p.get("test_ctx"),
                        "obs_after": {"quality": _q_a, "neuromod_delta": _nm_delta},
                        "reward": _q_a if _confirmed else 0.0,
                        "confirmed": _confirmed,
                        "error": _error,
                    })
                    logger.info("[HAOV] Social verify: quality %.3f→%.3f nm_delta=%.3f confirmed=%s",
                                _q_b, _q_a, _nm_delta, _confirmed)

                elif _haov_consumer == "reasoning":
                    # Reasoning verifier: did meta-reasoning commit rate improve?
                    _rate_b = float(_obs_b.get("commit_rate", 0))
                    _rate_a = _rate_b
                    _reward_b = float(_obs_b.get("avg_reward", 0))
                    _reward_a = _reward_b
                    try:
                        if meta_engine is not None:
                            _me_stats = meta_engine.get_stats() if hasattr(meta_engine, 'get_stats') else {}
                            _rate_a = float(_me_stats.get("commit_rate", _rate_b))
                            _reward_a = float(_me_stats.get("avg_reward", _reward_b))
                    except Exception:
                        pass

                    _confirmed = (_rate_a > _rate_b + 0.01) or (_reward_a > _reward_b + 0.01)
                    _error = abs(_rate_a - _rate_b) + abs(_reward_a - _reward_b)
                    _send_msg(send_queue, "CGN_HAOV_VERIFY_RSP", name, "cgn", {
                        "consumer": "reasoning",
                        "test_ctx": _haov_p.get("test_ctx"),
                        "obs_after": {"commit_rate": _rate_a, "avg_reward": _reward_a},
                        "reward": _rate_a if _confirmed else 0.0,
                        "confirmed": _confirmed,
                        "error": _error,
                    })
                    logger.info("[HAOV] Reasoning verify: rate %.3f→%.3f reward %.3f→%.3f confirmed=%s",
                                _rate_b, _rate_a, _reward_b, _reward_a, _confirmed)

                elif _haov_consumer == "self_model":
                    # Self-model verifier: did self-prediction accuracy improve?
                    _acc_b = float(_obs_b.get("prediction_accuracy", 0))
                    _depth_b = float(_obs_b.get("i_depth", 0))
                    _acc_a = _acc_b
                    _depth_a = _depth_b
                    try:
                        _sr = getattr(meta_engine, '_self_reasoning', None) if meta_engine else None
                        if _sr:
                            _sr_stats = _sr.get_stats() if hasattr(_sr, 'get_stats') else {}
                            _acc_a = float(_sr_stats.get("prediction_accuracy",
                                           _sr_stats.get("accuracy", _acc_b)))
                            _depth_a = float(_sr_stats.get("i_depth",
                                             _sr_stats.get("introspection_count", _depth_b)))
                    except Exception:
                        pass

                    _confirmed = (_acc_a > _acc_b + 0.01) or (_depth_a > _depth_b)
                    _error = abs(_acc_a - _acc_b)
                    _send_msg(send_queue, "CGN_HAOV_VERIFY_RSP", name, "cgn", {
                        "consumer": "self_model",
                        "test_ctx": _haov_p.get("test_ctx"),
                        "obs_after": {"prediction_accuracy": _acc_a, "i_depth": _depth_a},
                        "reward": _acc_a if _confirmed else 0.0,
                        "confirmed": _confirmed,
                        "error": _error,
                    })
                    logger.info("[HAOV] Self-model verify: accuracy %.3f→%.3f depth %.1f→%.1f confirmed=%s",
                                _acc_b, _acc_a, _depth_b, _depth_a, _confirmed)

                elif _haov_consumer == "coding":
                    # Coding verifier: did sandbox execution success improve?
                    _pass_b = float(_obs_b.get("pass_rate", 0))
                    _pass_a = _pass_b
                    try:
                        # Check coding explorer stats if available
                        _ce = getattr(coordinator, 'coding_explorer', None) if coordinator else None
                        if _ce and hasattr(_ce, 'get_stats'):
                            _ce_stats = _ce.get_stats()
                            _pass_a = float(_ce_stats.get("pass_rate",
                                            _ce_stats.get("success_rate", _pass_b)))
                    except Exception:
                        pass

                    _confirmed = _pass_a > _pass_b + 0.01
                    _error = abs(_pass_a - _pass_b)
                    _send_msg(send_queue, "CGN_HAOV_VERIFY_RSP", name, "cgn", {
                        "consumer": "coding",
                        "test_ctx": _haov_p.get("test_ctx"),
                        "obs_after": {"pass_rate": _pass_a},
                        "reward": _pass_a if _confirmed else 0.0,
                        "confirmed": _confirmed,
                        "error": _error,
                    })
                    logger.info("[HAOV] Coding verify: pass_rate %.3f→%.3f confirmed=%s",
                                _pass_b, _pass_a, _confirmed)

                else:
                    logger.debug("[HAOV] Unknown consumer '%s', skipping", _haov_consumer)

            except Exception as _haov_err:
                logger.warning("[HAOV] Verification error (%s): %s", _haov_consumer, _haov_err)

        # ── MEMORY_RECALL_PERTURBATION: Somatic re-experiencing ──
        # Bridge B: When recalled memories have felt_state_snapshots,
        # their neurochemical signature creates a micro-perturbation.
        # Closes the loop: inner→outer (dream) → outer→inner (recall).
        elif msg_type == "MEMORY_RECALL_PERTURBATION":
            try:
                _mrp = msg.get("payload", {})
                _mrp_nudge = _mrp.get("nudge_map", {})
                _mrp_max = float(_mrp.get("max_delta", 0.02))
                if _mrp_nudge and neuromodulator_system:
                    _mrp_targets = {}
                    for _mod_name, _delta in _mrp_nudge.items():
                        if _mod_name in neuromodulator_system.modulators:
                            _cur = neuromodulator_system.modulators[_mod_name].level
                            _mrp_targets[_mod_name] = max(0.0, min(1.0, _cur + _delta))
                    if _mrp_targets:
                        neuromodulator_system.apply_external_nudge(
                            _mrp_targets, max_delta=_mrp_max,
                            developmental_age=(pi_monitor.developmental_age
                                               if pi_monitor else 1.0))
                        # Push to working memory for short-term visibility
                        if working_mem:
                            _ep = (consciousness.get("latest_epoch", {}).get(
                                "epoch_id", 0) if consciousness else 0)
                            working_mem.attend(
                                "memory_recall_echo",
                                f"recall_{int(time.time())}",
                                {"nudges": _mrp_nudge,
                                 "memory_count": _mrp.get("memory_count", 0)},
                                _ep)
                        logger.info("[RecallBridge] Memory perturbation: %s "
                                    "(from %d memories)",
                                    {k: f"{v:+.4f}" for k, v in _mrp_nudge.items()},
                                    _mrp.get("memory_count", 0))
                        # I-depth: recall perturbation closes the loop
                        if msl and hasattr(msl, 'i_depth'):
                            msl.i_depth.record_recall_perturbation()
            except Exception as _mrp_err:
                logger.warning("[RecallBridge] Perturbation error: %s", _mrp_err)

        # rFP #2 Phase B.5b: cache V5 multipliers from own publish for
        # application at unified_spirit update call sites.
        elif msg_type == "FILTER_DOWN_V5":
            payload = msg.get("payload", {})
            _mults = payload.get("multipliers")
            if isinstance(_mults, dict):
                _v5_mults_cache.clear()
                _v5_mults_cache.update(_mults)

        # Spirit Enrichment: Receive state snapshot → micro_enrich
        # DQ6: Prefer full 65DT/130DT over legacy 30DT
        elif msg_type == "STATE_SNAPSHOT":
            payload = msg.get("payload", {})
            # rFP #2 Phase 2: accumulate 30D topology into per-epoch buffer
            # for TITAN_SELF composition at next consciousness epoch close.
            try:
                _topo = payload.get("full_30d_topology")
                if _topo:
                    from titan_plugin.modules.spirit_loop import observe_topology
                    observe_topology(_topo)
            except Exception as _to_err:
                logger.debug("[SpiritWorker] observe_topology error: %s", _to_err)
            # ── 27-day-old missing wiring restored 2026-04-13 ──
            # PLAN_trinity_symmetry_upgrade.md (pre-2026-03-17) specified
            # BOTH micro_enrich AND coordinator.on_outer_snapshot to run on
            # STATE_SNAPSHOT. Only micro_enrich was implemented in commit
            # ff3eb12 (T3-T8). Result: experience buffer never populated →
            # dream distillation produced 0 insights for 27 days. This line
            # restores the buffer-feed path. See feedback_arch_map_monitoring
            # _rule.md for the codified rule preventing future silent-disconnect.
            if coordinator is not None:
                try:
                    coordinator.on_outer_snapshot(payload)
                except Exception as _on_err:
                    logger.debug("[SpiritWorker] on_outer_snapshot error: %s",
                                 _on_err)
            if unified_spirit:
                try:
                    # Prefer extended 65DT, fallback to legacy 30DT
                    realtime = payload.get("full_65dt", payload.get("full_30dt", []))
                    if len(realtime) in (30, 65, 130):
                        alignment = unified_spirit.micro_enrich(realtime)
                        if alignment > 0.1:
                            logger.debug(
                                "[SpiritWorker] Micro-enrichment (%dD): alignment=%.3f "
                                "quality=%.1f ticks=%d",
                                len(realtime),
                                alignment, unified_spirit._cumulative_quality,
                                unified_spirit._micro_tick_count,
                            )
                except Exception as e:
                    logger.warning("[SpiritWorker] Micro-enrichment error: %s", e)

        # R5: Receive TitanVM reflex reward → feed into FilterDown
        elif msg_type == "REFLEX_REWARD":
            payload = msg.get("payload", {})
            reward = payload.get("reward", 0.0)
            if filter_down and reward > 0:
                try:
                    # Build current state for FilterDown
                    body_vals = body_state.get("values", [0.5] * 5)
                    mind_vals = mind_state.get("values", [0.5] * 5)
                    spirit_vals = _collect_spirit_tensor(config, body_state, mind_state, consciousness)
                    state = body_vals + mind_vals + spirit_vals

                    # Record as a transition where reward comes from TitanVM scoring
                    # (same state → same state, but with the interaction reward)
                    filter_down._buffer.add(state, reward - 0.5, state)
                    filter_down._transitions_since_train += 1
                    loss = filter_down.maybe_train()
                    if loss is not None:
                        logger.info("[SpiritWorker] FilterDown trained from reflex reward: "
                                    "reward=%.3f loss=%.6f", reward, loss)
                except Exception as e:
                    logger.warning("[SpiritWorker] Reflex reward processing failed: %s", e)

            # V5: Neural NervousSystem receives outcome reward
            if neural_nervous_system and reward > 0:
                try:
                    neural_nervous_system.record_outcome(reward)
                except Exception as e:
                    logger.warning("[SpiritWorker] Neural NS outcome recording failed: %s", e)

            # V4 FilterDown also receives reward
            if filter_down_v4 and reward > 0 and unified_spirit:
                try:
                    current_30dt = getattr(unified_spirit, 'get_vector', lambda: None)()
                    if current_30dt and len(current_30dt) == 30:
                        filter_down_v4._buffer.add(current_30dt, reward - 0.5, current_30dt)
                        filter_down_v4._transitions_since_train += 1
                        filter_down_v4.maybe_train()
                except Exception as e:
                    logger.warning("[SpiritWorker] V4 reflex reward failed: %s", e)

        elif msg_type == "RELOAD":
            # ══════════════════════════════════════════════════════════
            # LIVE CODE RELOAD — Zero consciousness gap
            # 1. Snapshot all object states (in-memory, no disk I/O)
            # 2. importlib.reload() on changed modules
            # 3. Recreate objects from new classes with saved state
            # 4. Reassign local variables so the loop uses new code
            # ══════════════════════════════════════════════════════════
            payload = msg.get("payload", {})
            modules_to_reload = payload.get("modules", [])
            reload_all = payload.get("all", False)
            reason = payload.get("reason", "")

            logger.info("[SpiritWorker] LIVE RELOAD requested: %s (reason: %s)",
                        "ALL" if reload_all else modules_to_reload, reason)

            # ── 1. SNAPSHOT all object states ──
            _reload_state = {}
            try:
                if neuromodulator_system:
                    _reload_state["neuromodulator"] = neuromodulator_system.get_state()
                if expression_manager:
                    _reload_state["expression"] = expression_manager.get_state()
                if life_force_engine:
                    _reload_state["life_force"] = life_force_engine.get_state()
                if outer_interface:
                    _reload_state["outer_interface"] = outer_interface.get_state()
                if sphere_clock:
                    _reload_state["sphere_clock"] = sphere_clock.get_state()
                if pi_monitor:
                    _reload_state["pi_heartbeat"] = pi_monitor.get_state()
                if resonance:
                    _reload_state["resonance"] = resonance.get_state()
                if neural_nervous_system:
                    _reload_state["neural_ns"] = neural_nervous_system.get_state()
                if inner_lower_topo:
                    _reload_state["inner_lower_topo"] = inner_lower_topo.get_state()
                if outer_lower_topo:
                    _reload_state["outer_lower_topo"] = outer_lower_topo.get_state()
                if ground_up_enricher:
                    _reload_state["ground_up"] = ground_up_enricher.get_state()
                    if _reasoning_engine:
                        _reload_state["reasoning"] = _reasoning_engine.get_state()
                        _reasoning_engine.save_all()
                    if _interpreter:
                        _interpreter.save_all()
                if prediction_engine:
                    _reload_state["prediction"] = prediction_engine.get_state() if hasattr(prediction_engine, 'get_state') else {}
                # Loop-level state
                _reload_state["_loop"] = {
                    "body_state": dict(body_state),
                    "mind_state": dict(mind_state),
                    "outer_state": dict(outer_state) if 'outer_state' in dir() else {},
                    "phase_tracker": dict(_phase_tracker),
                    "last_consciousness_tick": last_consciousness_tick,
                    "_fires_since_last_epoch": _fires_since_last_epoch,
                    "_urgency_drought": _urgency_drought,
                    "_urgency_warmup": _urgency_warmup,
                }
                logger.info("[SpiritWorker] State snapshot complete: %d systems",
                            len(_reload_state) - 1)
            except Exception as e:
                logger.error("[SpiritWorker] State snapshot failed: %s", e)

            # ── 2. RELOAD modules via importlib ──
            import importlib
            _all_logic_modules = [
                "titan_plugin.logic.neuromodulator",
                "titan_plugin.logic.expression_composites",
                "titan_plugin.logic.life_force",
                "titan_plugin.logic.outer_interface",
                "titan_plugin.logic.action_decoder",
                "titan_plugin.logic.action_narrator",
                "titan_plugin.logic.self_exploration_advisor",
                "titan_plugin.logic.sphere_clock",
                "titan_plugin.logic.resonance",
                "titan_plugin.logic.topology",
                "titan_plugin.logic.lower_topology",
                "titan_plugin.logic.ground_up",
                "titan_plugin.logic.pi_heartbeat",
                "titan_plugin.logic.filter_down",
                "titan_plugin.logic.middle_path",
                "titan_plugin.logic.focus_pid",
                "titan_plugin.logic.impulse_engine",
                "titan_plugin.logic.observables",
                "titan_plugin.logic.prediction_engine",
                "titan_plugin.logic.composition_engine",
                "titan_plugin.logic.word_selector",
                "titan_plugin.logic.hormonal_pressure",
                "titan_plugin.logic.expression_translator",
            ]

            if reload_all:
                targets = _all_logic_modules
            else:
                targets = [f"titan_plugin.logic.{m}" for m in modules_to_reload]

            reloaded = []
            failed = []
            for mod_path in targets:
                try:
                    mod = importlib.import_module(mod_path)
                    importlib.reload(mod)
                    reloaded.append(mod_path.split(".")[-1])
                except Exception as e:
                    failed.append({"module": mod_path, "error": str(e)})
                    logger.error("[SpiritWorker] Reload FAILED for %s: %s", mod_path, e)

            logger.info("[SpiritWorker] Logic modules reloaded: %d ok, %d failed",
                        len(reloaded), len(failed))

            # ── 2b. RELOAD spirit_loop (helper functions) ──
            try:
                from titan_plugin.modules import spirit_loop as _sl_mod
                importlib.reload(_sl_mod)
                # Re-import all names so local references point to new code
                from titan_plugin.modules.spirit_loop import (  # noqa: E402
                    _post_epoch_learning, _run_focus,
                    _compute_spirit_reflex_intuition, _run_impulse,
                    _post_epoch_v4_filter_down, _tick_clock_pair,
                    _maybe_anchor_trinity, _run_consciousness_epoch,
                    _compute_trajectory, _collect_spirit_tensor,
                    _handle_query, _publish_spirit_state,
                    _send_msg, _send_response, _send_heartbeat,
                    _check_resonance,
                )
                reloaded.append("spirit_loop")
                logger.info("[SpiritWorker] spirit_loop reloaded — %d helper functions updated",
                            len(dir(_sl_mod)))
                # Post-reload cleanup (Modification E)
                if hasattr(_sl_mod, 'post_reload_cleanup_helpers'):
                    _sl_mod.post_reload_cleanup_helpers()
            except Exception as e:
                logger.warning("[SpiritWorker] spirit_loop reload failed: %s — using old helpers", e)
                failed.append({"module": "spirit_loop", "error": str(e)})

            # ── 3. RECREATE objects from new classes with saved state ──
            try:
                if "neuromodulator" in reloaded and _reload_state.get("neuromodulator"):
                    from titan_plugin.logic.neuromodulator import NeuromodulatorSystem
                    neuromodulator_system = NeuromodulatorSystem.from_state(
                        _reload_state["neuromodulator"])

                if "expression_composites" in reloaded and _reload_state.get("expression"):
                    from titan_plugin.logic.expression_composites import (
                        ExpressionManager, create_speak, create_art, create_music, create_social)
                    expression_manager = ExpressionManager()
                    expression_manager.register(create_speak())
                    expression_manager.register(create_art())
                    expression_manager.register(create_music())
                    expression_manager.register(create_social())
                    expression_manager.restore_state(_reload_state["expression"])

                if "life_force" in reloaded and _reload_state.get("life_force"):
                    from titan_plugin.logic.life_force import LifeForceEngine
                    life_force_engine = LifeForceEngine()
                    life_force_engine.restore_state(_reload_state["life_force"])

                if "outer_interface" in reloaded and _reload_state.get("outer_interface"):
                    from titan_plugin.logic.outer_interface import OuterInterface
                    outer_interface = OuterInterface(
                        word_recipe_dir="data", params_config=_oi_params)
                    outer_interface.restore_state(_reload_state["outer_interface"])

                if "sphere_clock" in reloaded and _reload_state.get("sphere_clock"):
                    sphere_clock.restore_state(_reload_state["sphere_clock"])

                if "pi_heartbeat" in reloaded and _reload_state.get("pi_heartbeat"):
                    pi_monitor.restore_state(_reload_state["pi_heartbeat"])

                if "resonance" in reloaded and _reload_state.get("resonance"):
                    resonance.restore_state(_reload_state["resonance"])

                if "lower_topology" in reloaded:
                    if _reload_state.get("inner_lower_topo"):
                        inner_lower_topo.restore_state(_reload_state["inner_lower_topo"])
                    if _reload_state.get("outer_lower_topo"):
                        outer_lower_topo.restore_state(_reload_state["outer_lower_topo"])

                if "ground_up" in reloaded and _reload_state.get("ground_up"):
                    ground_up_enricher.restore_state(_reload_state["ground_up"])
                    if _reasoning_engine and "reasoning" in _reload_state:
                        _reasoning_engine.restore_state(_reload_state["reasoning"])

                if "neural_nervous_system" in reloaded and _reload_state.get("neural_ns"):
                    neural_nervous_system.restore_state(_reload_state["neural_ns"])

                # Restore loop-level state
                _loop = _reload_state.get("_loop", {})
                if _loop:
                    body_state.update(_loop.get("body_state", {}))
                    mind_state.update(_loop.get("mind_state", {}))
                    if 'outer_state' in dir() and _loop.get("outer_state"):
                        outer_state.update(_loop["outer_state"])
                    _phase_tracker.update(_loop.get("phase_tracker", {}))
                    last_consciousness_tick = _loop.get(
                        "last_consciousness_tick", last_consciousness_tick)
                    _fires_since_last_epoch = _loop.get(
                        "_fires_since_last_epoch", _fires_since_last_epoch)
                    _urgency_drought = _loop.get(
                        "_urgency_drought", _urgency_drought)
                    _urgency_warmup = _loop.get(
                        "_urgency_warmup", _urgency_warmup)

            except Exception as e:
                logger.error("[SpiritWorker] Object recreation failed: %s", e)
                failed.append({"module": "recreation", "error": str(e)})

            # ── 4. FALLBACK: if reload failed, save state for Guardian restart ──
            if failed:
                try:
                    import json as _json
                    _fallback_path = "data/spirit_state_reload.json"
                    # Only save serializable loop state (not full objects)
                    with open(_fallback_path, "w") as _f:
                        _json.dump(_reload_state.get("_loop", {}), _f)
                    logger.warning("[SpiritWorker] Live reload partial failure — "
                                   "loop state saved to %s", _fallback_path)
                except Exception:
                    pass

            # ── 5. REPORT (logged only — core has no drain loop for RELOAD_COMPLETE) ──
            logger.info(
                "[SpiritWorker] LIVE RELOAD complete: %d reloaded, %d failed — "
                "consciousness continuous, state preserved",
                len(reloaded), len(failed))

        elif msg_type == "CONFIG_RELOAD":
            # Hot-reload titan_params.toml values into running objects
            _new_params = msg.get("payload", {})
            _config_updated = []

            # Update self_exploration advisor
            if outer_interface:
                se_cfg = _new_params.get("self_exploration", {})
                for action_type in list(outer_interface.advisor._base_refractory.keys()):
                    key = f"dna_base_{action_type}"
                    if key in se_cfg:
                        outer_interface.advisor._base_refractory[action_type] = float(se_cfg[key])
                        _config_updated.append(f"advisor.{action_type}")
                if "external_cooldown_multiplier" in se_cfg:
                    outer_interface._cooldown_multiplier = float(se_cfg["external_cooldown_multiplier"])
                    _config_updated.append("cooldown_multiplier")
                if "inner_enrichment_strength" in se_cfg:
                    outer_interface._inner_strength = float(se_cfg["inner_enrichment_strength"])
                    _config_updated.append("inner_strength")

            # Update expression consumption rates
            ec_cfg = _new_params.get("expression_composites", {})
            if ec_cfg and expression_manager:
                rate_map = {"SPEAK": "consumption_speak", "ART": "consumption_art",
                            "MUSIC": "consumption_music", "SOCIAL": "consumption_social"}
                for comp_name, key in rate_map.items():
                    if key in ec_cfg and comp_name in expression_manager.composites:
                        expression_manager.composites[comp_name].consumption_rate = float(ec_cfg[key])
                        _config_updated.append(f"consumption.{comp_name}")

            # Update action decoder max_delta
            ad_cfg = _new_params.get("action_decoder", {})
            if ad_cfg and outer_interface and hasattr(outer_interface, 'decoder'):
                if "max_delta" in ad_cfg:
                    outer_interface.decoder._max_delta = float(ad_cfg["max_delta"])
                    _config_updated.append("decoder.max_delta")

            logger.info("[SpiritWorker] CONFIG reloaded: %d params — %s",
                        len(_config_updated), ", ".join(_config_updated[:10]))

        elif msg_type == "QUERY":
            # READ-ONLY queries → dedicated thread (never blocked by computation)
            _query_action = msg.get("payload", {}).get("action", "")
            if _query_action in ("chat", "conversation") and outer_interface:
                outer_interface.on_external_interaction()
            try:
                _query_queue.put_nowait(msg)
            except Exception:
                # Queue full — handle inline as fallback
                _handle_query(msg, config, body_state, mind_state, consciousness,
                              filter_down, intuition, impulse_engine, sphere_clock,
                              resonance, unified_spirit, send_queue, name,
                              inner_state=inner_state, spirit_state=spirit_state,
                              coordinator=coordinator,
                              neural_nervous_system=neural_nervous_system,
                              pi_monitor=pi_monitor, e_mem=e_mem,
                              prediction_engine=prediction_engine,
                              ex_mem=ex_mem, episodic_mem=episodic_mem,
                              working_mem=working_mem,
                              inner_lower_topo=inner_lower_topo,
                              outer_lower_topo=outer_lower_topo,
                              ground_up_enricher=ground_up_enricher,
                              neuromodulator_system=neuromodulator_system,
                              expression_manager=expression_manager,
                              life_force_engine=life_force_engine,
                              outer_interface=outer_interface,
                              phase_tracker=_phase_tracker,
                              meditation_tracker=_meditation_tracker,
                              reasoning_engine=_reasoning_engine,
                              msl=msl,
                              social_pressure_meter=_social_pressure_meter,
                              language_stats=_language_stats,
                              self_reasoning=_self_reasoning,
                              coding_explorer=_coding_explorer,
                              filter_down_v4=filter_down_v4,
                              filter_down_v5=filter_down_v5,
                              med_watchdog=_med_watchdog)

        elif msg_type == "SPEAK_RESULT":
            # ── SPEAK_RESULT from language_worker (Phase 2) ──────────────
            _sr = msg.get("payload", {})
            _sr_sentence = _sr.get("sentence", "")
            if _sr_sentence:
                logger.info(
                    '[EXPRESSION.SPEAK] "%s" (L%d, conf=%.2f)',
                    _sr_sentence, _sr.get("level", 0), _sr.get("confidence", 0))

                # 1. Apply self-hearing perturbation deltas (bone conduction, 50%)
                _sr_deltas = _sr.get("perturbation_deltas", [])
                if _sr_deltas:
                    _sr_reinforced = _lp_apply_perturbation_deltas(
                        _sr_deltas,
                        body_state.get("values", []),
                        mind_state.get("values_15d", []),
                        strength=0.5)
                    if _sr_reinforced > 0:
                        logger.info("[SPEAK:SELF-HEARING] %d words via bone conduction (50%%)", _sr_reinforced)

                # 2. L8 pattern learning now handled by language_worker

                # 3. Social catalyst (L7+, conf>=0.8 → X gateway)
                if (_x_gateway
                        and _sr.get("level", 0) >= 7
                        and _sr.get("confidence", 0) >= 0.8):
                    _x_catalysts.append({
                        "type": "strong_composition",
                        "significance": 0.7,
                        "content": _sr_sentence,
                        "data": {"level": _sr.get("level", 0),
                                 "confidence": _sr.get("confidence", 0)},
                    })

                # 4. Refresh vocabulary cache on each SPEAK result
                try:
                    _cached_speak_vocab = _lp_load_vocabulary(db_path="./data/inner_memory.db")
                except Exception:
                    pass

                # 5. Conversation eval: if teacher asked a question and being
                #    just answered via SPEAK, send eval request to language_worker.
                if _conversation_pending and _sr_sentence:
                    _conv_q = _conversation_pending["question"]
                    _conversation_pending = None  # Consume — one answer per question
                    _conversation_stats["answered"] = _conversation_stats.get("answered", 0) + 1
                    # Build eval prompt and send to language_worker via LLM path
                    try:
                        from titan_plugin.logic.language_teacher import LanguageTeacher
                        _eval_prompt = LanguageTeacher.build_conversation_eval_prompt(
                            _conv_q, _sr_sentence)
                        bus.send({
                            "type": "LLM_TEACHER_REQUEST", "src": "spirit",
                            "dst": "language", "ts": time.time(),
                            "payload": {
                                "prompt": _eval_prompt["prompt"],
                                "system": _eval_prompt["system"],
                                "mode": "conversation_eval",
                                "original": _conv_q,
                                "max_tokens": _eval_prompt.get("max_tokens", 50),
                                "sentences": [],
                                "neuromod_gate": "",
                                "conversation_response": _sr_sentence,
                            },
                        })
                        logger.info("[CONVERSATION] Eval request sent: Q='%s' A='%s'",
                                    _conv_q[:40], _sr_sentence[:40])
                    except Exception as _conv_err:
                        logger.warning("[CONVERSATION] Eval trigger error: %s", _conv_err)

                # 6. Experience recording (needs full state — stays in spirit)
                if exp_orchestrator:
                    try:
                        _lang_hormones = {}
                        if neural_nervous_system and neural_nervous_system._hormonal_enabled:
                            _lang_hormones = {h: round(v.level, 3)
                                              for h, v in neural_nervous_system._hormonal._hormones.items()}
                        _lang_plugin = exp_orchestrator._plugins.get("language")
                        _sr_sv = consciousness.get("latest_epoch", {}).get("state_vector", []) if consciousness else []
                        _sr_sv = list(_sr_sv) if _sr_sv else []
                        _lang_perc = _lang_plugin.extract_perception_key({
                            "inner_state": _sr_sv,
                            "felt_tensor": _sr_sv[:65],
                            "inner_body": _sr_sv[:5],
                            "inner_mind": _sr_sv[5:20] if len(_sr_sv) >= 20 else [],
                            "inner_spirit": _sr_sv[20:65] if len(_sr_sv) >= 65 else [],
                            "intent_hormones": _lang_hormones,
                            "hormonal_snapshot": _lang_hormones,
                            "spatial_features": _sr_sv[65:95] if len(_sr_sv) > 65 else [],
                        }) if _lang_plugin else (_sr_sv[:10] if _sr_sv else [0.5] * 10)
                        exp_orchestrator.record_outcome(
                            domain="language",
                            perception_features=_lang_perc,
                            inner_state_132d=_sr_sv,
                            hormonal_snapshot=_lang_hormones,
                            action_taken=_sr_sentence,
                            outcome_score=_sr.get("confidence", 0.0),
                            context={
                                "level": _sr.get("level", 0),
                                "words_used": _sr.get("words_used", []),
                                "template": _sr.get("template", ""),
                                "source": "direct_composition",
                            },
                            epoch_id=_sr.get("epoch_id", 0),
                            is_dreaming=_shared_is_dreaming,
                        )
                    except Exception as _lang_err:
                        logger.warning("[ExperienceOrch] Language recording error: %s", _lang_err)

                # 7. Text perception → outer_mind
                try:
                    from titan_plugin.logic.action_decoder import ActionDecoder
                    _post_bm = list(body_state.get("values", [0.5] * 5)) + \
                                list(mind_state.get("values_15d", [0.5] * 15))
                    _sr_sv2 = consciousness.get("latest_epoch", {}).get("state_vector", []) if consciousness else []
                    _pre_bm = list(_sr_sv2[:20]) if len(_sr_sv2) >= 20 else [0.5] * 20
                    _tp = ActionDecoder.decode_text_perception(
                        _sr, pre_state=_pre_bm, post_state=_post_bm)
                    _om15_tp = outer_state.get("outer_mind_15d")
                    if _om15_tp and len(_om15_tp) >= 10:
                        _TP_STRENGTH = 0.04
                        for _ti, _tk in enumerate(["novelty", "self_reference",
                                "emotional_valence", "complexity", "resonance_shift"]):
                            _tv = _tp.get(_tk, 0.0)
                            _delta = (_tv - _om15_tp[5 + _ti]) * _TP_STRENGTH
                            _om15_tp[5 + _ti] = max(0.0, min(1.0, _om15_tp[5 + _ti] + _delta))
                        outer_state["outer_mind_15d"] = _om15_tp
                except Exception as _tp_err:
                    logger.warning("[TEXT_PERCEPTION] error: %s", _tp_err)

        elif msg_type == "TEACHER_SIGNALS":
            # ── TEACHER_SIGNALS from language_worker (Phase 3) ───────────
            _ts = msg.get("payload", {})
            _ts_mode = _ts.get("mode", "?")

            # 1. Apply teacher perturbation deltas (40% = comprehension strength)
            _ts_deltas = _ts.get("perturbation_deltas", [])
            if _ts_deltas:
                _ts_reinforced = _lp_apply_perturbation_deltas(
                    _ts_deltas,
                    body_state.get("values", []),
                    mind_state.get("values_15d", []),
                    strength=0.40)
                if _ts_reinforced > 0:
                    logger.info("[TEACHER] %d words felt (40%% comprehension)", _ts_reinforced)

            # 2. MSL concept signals
            for _sig in _ts.get("msl_signals", []):
                _sig_concept = _sig.get("concept")
                _sig_quality = _sig.get("quality", 0.5)
                if msl and hasattr(msl, 'concept_grounder') and msl.concept_grounder:
                    try:
                        cg = msl.concept_grounder
                        _sig_epoch = getattr(msl, '_tick_count', 0)
                        if _sig_concept == "I":
                            msl.confidence._convergence_count += 1
                        elif _sig_concept == "YOU":
                            cg.signal_you("teacher", _sig_quality, _sig_epoch, None)
                        elif _sig_concept == "YES":
                            cg.signal_yes(_sig_quality, _sig_epoch, None)
                        elif _sig_concept == "NO":
                            cg.signal_no(_sig_quality, _sig_epoch, None)
                    except Exception:
                        pass
            if _ts.get("msl_signals"):
                logger.debug("[TEACHER] MSL signals: %s",
                             [s["concept"] for s in _ts["msl_signals"]])

            # 3. Register dynamic recipes (new word perturbation patterns)
            for _dr in _ts.get("dynamic_recipes", []):
                if outer_interface and hasattr(outer_interface.narrator, "register_dynamic_recipe"):
                    try:
                        outer_interface.narrator.register_dynamic_recipe(
                            _dr["word"], _dr.get("tensor"),
                            word_type=_dr.get("word_type", "unknown"),
                            context=_dr.get("context", ""),
                            hormone_state={})
                    except Exception:
                        pass
            if _ts.get("dynamic_recipes"):
                try:
                    outer_interface.narrator.save_dynamic_recipes()
                except Exception:
                    pass

            # 4. Neuromod nudge (conversation mode → Endorphin)
            for _nm_name, _nm_delta in _ts.get("neuromod_nudge", {}).items():
                if neuromodulator_system:
                    try:
                        _nm_mod = neuromodulator_system.modulators.get(_nm_name)
                        if _nm_mod:
                            _nm_mod.level = min(1.0, _nm_mod.level + _nm_delta)
                    except Exception:
                        pass

            # 5. Conversation question → set pending for SPEAK capture
            _ts_conv_q = _ts.get("conversation_question")
            if _ts_conv_q:
                _conversation_pending = {
                    "question": _ts_conv_q,
                    "timestamp": time.time(),
                }
                logger.info("[CONVERSATION] Question from teacher: '%s'", _ts_conv_q[:80])

            # 6. Conversation eval → experience + mini-reasoner reward
            _ts_conv_eval = _ts.get("conversation_eval")
            if _ts_conv_eval:
                _conv_score = _ts_conv_eval.get("score", 0)
                if _mini_registry:
                    try:
                        _lang_mini = _mini_registry.get("language")
                        if _lang_mini:
                            _lang_mini.record_reward(_conv_score)
                    except Exception:
                        pass
                if exp_orchestrator:
                    try:
                        exp_orchestrator.record_outcome(
                            domain="language",
                            perception_features=list(range(10)),
                            inner_state_132d=[0.5] * 132,
                            hormonal_snapshot={},
                            action_taken=f"conversation: {_ts_conv_eval.get('response', '')[:50]}",
                            outcome_score=_conv_score,
                            context={"source": "conversation"},
                            epoch_id=0, is_dreaming=False)
                    except Exception:
                        pass

            _ts_acq = _ts.get("words_acquired", 0)
            _ts_rec = _ts.get("words_recognized", 0)
            if _ts_acq > 0 or _ts_rec > 0:
                logger.info("[TEACHER] Session: mode=%s, %d felt, %d acquired",
                            _ts_mode, _ts_rec, _ts_acq)

        elif msg_type == "LANGUAGE_STATS_UPDATE":
            # ── Language stats broadcast from language_worker (Phase 4) ──
            _ls_payload = msg.get("payload", {})
            if _ls_payload:
                # IMPORTANT: update in-place — _language_stats dict is shared
                # with query handler thread via state_refs. Reassigning the
                # variable would break the reference.
                _language_stats.clear()
                _language_stats.update(_ls_payload)

        elif msg_type == "LLM_TEACHER_RESPONSE":
            # Phase 3: routed to language_worker — safety stub
            logger.debug("[SpiritWorker] LLM_TEACHER_RESPONSE stale (routed to language_worker)")

        elif msg_type == "DREAM_WAKE_REQUEST":
            # Maker sent a message during dream → trigger gentle wake
            if (coordinator and coordinator.dreaming
                    and getattr(getattr(coordinator, 'inner', None),
                                'is_dreaming', False)):
                _onset = getattr(coordinator.dreaming, '_dream_onset_fatigue', 0)
                _current = getattr(coordinator.dreaming, '_dream_fatigue', 0)
                _wake_thresh = _onset * 0.20
                if _current > _wake_thresh:
                    coordinator.dreaming._dream_fatigue = _wake_thresh
                    coordinator.dreaming._wake_transition = True
                    logger.info(
                        "[SpiritWorker] Maker wake request → gentle transition "
                        "(fatigue %.0f→%.0f)", _current, _wake_thresh)
            else:
                logger.debug("[SpiritWorker] DREAM_WAKE_REQUEST but not dreaming")

    # Cleanup — final save sweep. SAVE_NOW handler should have already run
    # if the restart was graceful (via Guardian.stop or the admin restart-
    # module endpoint), but we run again here as a safety net for code paths
    # that bypass SAVE_NOW (e.g., direct kill, OOM, SIGTERM without SAVE_NOW).
    if msl is not None:
        try:
            msl.save_all()
        except Exception as _e:
            logger.warning("[SpiritWorker] cleanup: msl.save_all failed: %s", _e)
    if coordinator is not None and getattr(coordinator, "dreaming", None) is not None:
        try:
            coordinator.dreaming.save_state()
        except Exception as _e:
            logger.warning("[SpiritWorker] cleanup: dreaming.save_state failed: %s", _e)
    try:
        _meta_engine_cleanup = getattr(coordinator, "_meta_engine", None) if coordinator else None
        if _meta_engine_cleanup is not None and hasattr(_meta_engine_cleanup, "save_all"):
            _meta_engine_cleanup.save_all()
    except Exception as _e:
        logger.warning("[SpiritWorker] cleanup: meta_engine.save_all failed: %s", _e)
    if filter_down:
        filter_down._persist()
    if filter_down_v4:
        filter_down_v4._persist()
    if filter_down_v5:
        filter_down_v5._persist()
    if sphere_clock:
        sphere_clock.save_state()
    if resonance:
        resonance.save_state()
    if unified_spirit:
        unified_spirit.save_state()
    if e_mem:
        try:
            e_mem.close()
        except Exception:
            pass
    if ex_mem:
        try:
            ex_mem.close()
        except Exception:
            pass
    if episodic_mem:
        try:
            episodic_mem.close()
        except Exception:
            pass
    if pi_monitor:
        pi_monitor._save_state()
    # SocialXGateway uses SQLite — no save needed on shutdown
    if _x_gateway:
        logger.info("[SpiritWorker] SocialXGateway shutdown (stats=%s)", _x_gateway.get_stats())
    if neural_nervous_system:
        neural_nervous_system.save_all()
        logger.info("[SpiritWorker] Neural NervousSystem saved (%d transitions, %d train steps)",
                    neural_nervous_system._total_transitions,
                    neural_nervous_system._total_train_steps)
    if _mini_registry:
        _mini_registry.save_all()
        logger.info("[SpiritWorker] Mini-reasoning saved (%d domains)",
                    len(_mini_registry.all()))
    if consciousness and consciousness.get("db"):
        consciousness["db"].close()
    logger.info("[SpiritWorker] Exiting")


# ── Step 4 Counterpart Initialization ──────────────────────────────────

def _init_filter_down(config: dict):
    """Initialize FILTER_DOWN engine (V3 15-dim)."""
    try:
        from titan_plugin.logic.filter_down import FilterDownEngine
        data_dir = config.get("data_dir", "./data")
        if not data_dir or data_dir == "":
            data_dir = "./data"
        return FilterDownEngine(data_dir=data_dir)
    except Exception as e:
        logger.warning("[SpiritWorker] FilterDown init failed: %s", e)
        return None


def _init_filter_down_v4(config: dict):
    """Initialize V4 FILTER_DOWN engine (30-dim Full Trinity)."""
    try:
        from titan_plugin.logic.filter_down import FilterDownV4Engine
        data_dir = config.get("data_dir", "./data")
        if not data_dir or data_dir == "":
            data_dir = "./data"
        return FilterDownV4Engine(data_dir=data_dir)
    except Exception as e:
        logger.warning("[SpiritWorker] FilterDownV4 init failed: %s", e)
        return None


def _init_filter_down_v5(config: dict):
    """Initialize V5 FILTER_DOWN engine (162-dim TITAN_SELF) — rFP #2."""
    try:
        from titan_plugin.logic.filter_down import FilterDownV5Engine
        data_dir = config.get("data_dir", "./data")
        if not data_dir or data_dir == "":
            data_dir = "./data"
        return FilterDownV5Engine(config=config, data_dir=data_dir)
    except Exception as e:
        logger.warning("[SpiritWorker] FilterDownV5 init failed: %s", e)
        return None


def _init_focus():
    """Initialize FOCUS PID controllers for Body and Mind."""
    try:
        from titan_plugin.logic.focus_pid import FocusPID
        return FocusPID("body"), FocusPID("mind")
    except Exception as e:
        logger.warning("[SpiritWorker] Focus PID init failed: %s", e)
        return None, None


def _init_intuition():
    """Initialize INTUITION engine."""
    try:
        from titan_plugin.logic.intuition import IntuitionEngine
        return IntuitionEngine()
    except Exception as e:
        logger.warning("[SpiritWorker] Intuition init failed: %s", e)
        return None


def _init_sphere_clock(config: dict):
    """Initialize V4 Sphere Clock Engine."""
    try:
        from titan_plugin.logic.sphere_clock import SphereClockEngine
        sc_config = config.get("sphere_clock", {})
        data_dir = config.get("data_dir", "./data")
        if not data_dir or data_dir == "":
            data_dir = "./data"
        return SphereClockEngine(config=sc_config, data_dir=data_dir)
    except Exception as e:
        logger.warning("[SpiritWorker] SphereClockEngine init failed: %s", e)
        return None


def _init_unified_spirit(config: dict):
    """Initialize V4 Unified SPIRIT with enrichment config."""
    try:
        from titan_plugin.logic.unified_spirit import UnifiedSpirit
        # Merge sphere_clock + spirit_enrichment configs
        sc_config = dict(config.get("sphere_clock", {}))
        enrichment_cfg = config.get("spirit_enrichment", {})
        sc_config["enrichment_rate"] = enrichment_cfg.get("enrichment_rate", 0.02)
        sc_config["min_alignment_threshold"] = enrichment_cfg.get("min_alignment_threshold", 0.1)
        data_dir = config.get("data_dir", "./data")
        if not data_dir or data_dir == "":
            data_dir = "./data"
        return UnifiedSpirit(config=sc_config, data_dir=data_dir)
    except Exception as e:
        logger.warning("[SpiritWorker] UnifiedSpirit init failed: %s", e)
        return None


def _init_resonance(config: dict):
    """Initialize V4 Resonance Detector."""
    try:
        from titan_plugin.logic.resonance import ResonanceDetector
        sc_config = config.get("sphere_clock", {})
        data_dir = config.get("data_dir", "./data")
        if not data_dir or data_dir == "":
            data_dir = "./data"
        return ResonanceDetector(config=sc_config, data_dir=data_dir)
    except Exception as e:
        logger.warning("[SpiritWorker] ResonanceDetector init failed: %s", e)
        return None


def _init_impulse_engine():
    """Initialize IMPULSE engine (Step 7.1)."""
    try:
        from titan_plugin.logic.impulse_engine import ImpulseEngine
        return ImpulseEngine()
    except Exception as e:
        logger.warning("[SpiritWorker] ImpulseEngine init failed: %s", e)
        return None


def _init_observable_engine():
    """Initialize T1 Observable Engine (5 observables × 6 body parts)."""
    try:
        from titan_plugin.logic.observables import ObservableEngine
        return ObservableEngine()
    except Exception as e:
        logger.warning("[SpiritWorker] ObservableEngine init failed: %s", e)
        return None


def _init_neural_nervous_system(config: dict):
    """Initialize V5 Neural Nervous System (config-driven, learned reflexes)."""
    try:
        # Load neural NS config from titan_params.toml
        params_config = {}
        try:
            import tomllib
            params_path = os.path.join(os.path.dirname(__file__), "..", "titan_params.toml")
            if os.path.exists(params_path):
                with open(params_path, "rb") as f:
                    all_params = tomllib.load(f)
                params_config = all_params.get("neural_nervous_system", {})
        except Exception:
            pass

        if not params_config.get("enabled", False):
            return None

        from titan_plugin.logic.neural_nervous_system import NeuralNervousSystem
        # V4 VM NervousSystem as fallback
        vm_ns = None
        try:
            from titan_plugin.logic.nervous_system import NervousSystem
            vm_ns = NervousSystem()
        except Exception:
            pass

        data_dir = config.get("data_dir", "./data")
        if not data_dir:
            data_dir = "./data"
        nn_data_dir = os.path.join(data_dir, "neural_nervous_system")

        return NeuralNervousSystem(
            config=params_config,
            data_dir=nn_data_dir,
            vm_nervous_system=vm_ns,
        )
    except Exception as e:
        logger.warning("[SpiritWorker] NeuralNervousSystem init failed: %s", e)
        return None


def _init_coordinator(inner_state, spirit_state, observable_engine,
                      neural_nervous_system=None):
    """Initialize T3 Inner Trinity Coordinator with T4/V5 Nervous System."""
    try:
        from titan_plugin.logic.inner_coordinator import InnerTrinityCoordinator
        # T4: Create NervousSystem with lightweight TitanVM (context-only)
        nervous_system = None
        try:
            from titan_plugin.logic.nervous_system import NervousSystem
            nervous_system = NervousSystem()
        except Exception as e:
            logger.warning("[SpiritWorker] NervousSystem init failed: %s", e)
        # T5: Create TopologyEngine
        topology_engine = None
        try:
            from titan_plugin.logic.topology import TopologyEngine
            topology_engine = TopologyEngine()
        except Exception as e:
            logger.warning("[SpiritWorker] TopologyEngine init failed: %s", e)
        # T6: Create DreamingEngine
        dreaming_engine = None
        try:
            from titan_plugin.logic.dreaming import DreamingEngine
            _dreaming_state_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "dreaming_state.json")
            # Load DNA weights from titan_params.toml [dreaming]
            _dreaming_dna = {}
            try:
                import tomllib as _tl
                _tp = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "titan_params.toml")
                if os.path.exists(_tp):
                    with open(_tp, "rb") as _tf:
                        _dreaming_dna = _tl.load(_tf).get("dreaming", {})
            except Exception:
                pass
            dreaming_engine = DreamingEngine(
                state_path=_dreaming_state_path, dna=_dreaming_dna)
        except Exception as e:
            logger.warning("[SpiritWorker] DreamingEngine init failed: %s", e)
        return InnerTrinityCoordinator(
            inner_state=inner_state,
            spirit_state=spirit_state,
            observable_engine=observable_engine,
            vm=None,
            nervous_system=nervous_system,
            topology_engine=topology_engine,
            dreaming_engine=dreaming_engine,
            neural_nervous_system=neural_nervous_system,
        )
    except Exception as e:
        logger.warning("[SpiritWorker] InnerTrinityCoordinator init failed: %s", e)
        return None


def _init_t2_state_registries():
    """Initialize T2 InnerState + SpiritState registries."""
    inner, spirit = None, None
    try:
        from titan_plugin.logic.inner_state import InnerState
        inner = InnerState()
    except Exception as e:
        logger.warning("[SpiritWorker] InnerState init failed: %s", e)
    try:
        from titan_plugin.logic.spirit_state import SpiritState
        spirit = SpiritState()
    except Exception as e:
        logger.warning("[SpiritWorker] SpiritState init failed: %s", e)
    return inner, spirit

