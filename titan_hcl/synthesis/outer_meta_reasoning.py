"""titan_hcl/synthesis/outer_meta_reasoning.py — OML Phase C, piece 5.

The OUTER **two-level reasoner's** top engine (RFP §7.C, both-engines reuse):
``OuterMetaReasoningEngine`` runs the SAME 9-meta-primitive / 28-sub-mode flow
(FORMULATE→RECALL→HYPOTHESIZE→DELEGATE→EVALUATE→BREAK→SYNTHESIZE→SPIRIT_SELF→
INTROSPECT) as the inner ``MetaReasoningEngine`` — by **inheriting the shared
``PrimitiveHandlersMixin``** (piece 4) — but with its OWN outer orchestration:

  • **NO inner coupling** — no meta-CGN, no grounded-V Option-β, no teacher-bias,
    no repeat/entry/diversity logit bias, no neuromod normalization, no
    matrix-seed, no felt-cluster emit, no chain_iql template bias, no EUREKA.
    ``_send_queue=None`` disables every inner-coupled bus emit in the handlers.
  • **Selector = a FRESH ``MetaPolicy``** (9 meta-actions) **+ fresh
    ``SubModePolicy`` per primitive** — the direct parallel to piece-3's fresh
    ``ReasoningPolicyNet``. (The Phase-1 ``OuterMetaPolicy`` is NOT this — it is
    the 5-action hot-path agno consumer of the composites this engine produces.)
  • **DELEGATE → the piece-3 ``OuterReasoningEngine``, SYNCHRONOUSLY** — sets the
    8-D strategy bias, drives ``run_chain()`` inline, resolves via the inherited
    ``_check_delegate`` (vs the inner's async poll across epochs).
  • **EVALUATE ← oracle** — an injected ``oracle_verify`` callable scores the
    delegated result on the verifiable lane (the §1.3 ``40320 ✓``).
  • **conclude → composite** — a winning chain produces the Reasoning-composite
    payload (the bus emit + ``SynthesisWriter`` write is piece 6).

It is a NEW, SEPARATE class/instance/state — it does NOT touch or run the live
inner ``cognitive_worker`` engines. Zero edits to ``meta_reasoning.py``: it only
INHERITS the mixin + IMPORTS the reusable building blocks. The inner reasoning
loop (CGN / meta-CGN / felt) is therefore structurally untouched.
"""
import logging
import os
import time
from typing import Optional

import numpy as np

# Zero-inner-touch reuse: inherit the shared handler mixin + import the reusable
# building blocks. We never construct the inner MetaReasoningEngine.
from titan_hcl.logic.meta_reasoning import (
    META_POLICY_INPUT_DIM,
    META_PRIMITIVES,
    NUM_META_ACTIONS,
    STEP_REWARDS,
    SUB_MODES,
    MetaChainState,
    MetaPolicy,
    MetaTransitionBuffer,
    PrimitiveHandlersMixin,
    SubModePolicy,
)

logger = logging.getLogger(__name__)

# Outer plane has NO neuromod coupling — the handlers + input builders read a
# neuromod dict to set thresholds; neutral 0.5 is the honest "no signal".
_NEUTRAL_NEUROMODS = {
    "DA": 0.5, "5-HT": 0.5, "5HT": 0.5, "NE": 0.5, "ACh": 0.5,
    "Endorphin": 0.5, "GABA": 0.5,
}

# A neutral 132-D problem-state seed (outer has no felt state vector). The real
# problem-derived sv is supplied by piece 6; neutral keeps the meta-input
# state-summary block well-defined for the offline build/test.
_NEUTRAL_SV_DIM = 132


class OuterMetaReasoningEngine(PrimitiveHandlersMixin):
    """Outer meta-reasoning chain-builder (piece 5)."""

    def __init__(self, config: Optional[dict] = None, save_dir: Optional[str] = None):
        cfg = config or {}
        self._config = dict(cfg)
        self.min_chain_length = int(cfg.get("min_chain_length", 3))
        self.max_steps = int(cfg.get("max_steps", 20))

        # ── Selector policies — FRESH instances (own weights) ──────────────
        self.meta_policy = MetaPolicy(
            input_dim=META_POLICY_INPUT_DIM,
            lr=float(cfg.get("meta_lr", 0.001)))
        self.sub_mode_policies = {
            prim: SubModePolicy(len(modes), lr=float(cfg.get("sub_lr", 0.001)))
            for prim, modes in SUB_MODES.items()
        }
        self.buffer = MetaTransitionBuffer(max_size=int(cfg.get("buffer_size", 1000)))

        # ── Instance attrs the inherited handlers + input builders read ────
        # (inner-__init__ defaults — kept identical for faithful reuse). These
        # are read via `self.` from the relocated mixin handlers / the meta
        # input builders; NONE are inner-CGN coupled.
        self._delegate_bias_strength = float(cfg.get("delegate_bias_strength", 0.5))
        self._delegate_max_bias = float(cfg.get("delegate_max_bias", 3.0))
        self._delegate_max_wait_ticks = int(cfg.get("delegate_max_wait_ticks", 12))
        self._spirit_self_cooldown_max = int(cfg.get("spirit_self_cooldown", 5))
        self._spirit_self_gate = int(cfg.get("gate_spirit_self_chains", 50))
        self._baseline_confidence = 0.5
        self._total_wisdom_saved = 0
        self._ema_state = np.zeros(_NEUTRAL_SV_DIM, dtype=np.float32)
        self._ema_alpha = 0.02
        self._strategy_history = np.zeros(12, dtype=np.float32)
        self._total_eurekas = 0
        self._total_meta_chains = 0
        self._total_meta_steps = 0
        self._next_chain_id = 0
        self._eureka_cooldown_steps = 0

        # ── Isolation switches ─────────────────────────────────────────────
        # send_queue=None → every inner-coupled bus emit in the handlers is
        # skipped (felt-cluster, META_INTROSPECT_REQUEST, meta_outer_reward …).
        self._send_queue = None
        # Recall reader (set via set_outer_reader); inert until wired.
        self._outer_reader = None
        self._outer_future = None
        # Verifiable-lane oracle (injected via set_oracle); inert until wired.
        self._oracle_verify = None

        # Chain state.
        self.state = MetaChainState()
        self._problem: dict = {}
        # Per-chain transitions for terminal credit assignment.
        self._chain_transitions: list[tuple] = []

        # Lifetime telemetry.
        self._action_counts: dict[str, int] = {}
        self._last_outcome: str = ""

        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.meta_policy.load(os.path.join(save_dir, "outer_meta_policy_net.json"))

    # ── Injected dependencies ──────────────────────────────────────────────
    def set_outer_reader(self, reader) -> None:
        """Wire the OuterContextReader (recall seam). May be None → inert."""
        self._outer_reader = reader

    def set_oracle(self, oracle_verify) -> None:
        """Wire the verifiable-lane oracle: ``oracle_verify(problem, delegate_results)
        -> reward ∈ [-1, 1]``. May be None → EVALUATE/conclude skip oracle scoring."""
        self._oracle_verify = oracle_verify

    # ── Recall seam (outer copy — verbatim-equivalent to the inner methods;
    #    reads only self._outer_reader / self._outer_future / self.state) ────
    def _outer_enabled(self) -> bool:
        if self._outer_reader is None:
            return False
        try:
            return bool(self._outer_reader.is_active())
        except Exception:
            return False

    def _dispatch_outer_fetch(self) -> None:
        if self._outer_future is not None:
            return
        if not self._outer_enabled():
            return
        if not self.state.needs_outer:
            return
        try:
            self._outer_future = self._outer_reader.compose_recall_query(
                dict(self.state.entity_refs))
        except Exception as e:  # noqa: BLE001
            logger.debug("[OuterMeta] outer dispatch failed: %s", e)
            self._outer_future = None

    def _await_outer_context(self, timeout_s: float = 0.2) -> Optional[dict]:
        if not self._outer_enabled():
            return None
        if self.state.outer_context:
            return self.state.outer_context
        if self._outer_future is None:
            if self.state.entity_refs:
                self.state.needs_outer = dict(self.state.entity_refs)
                self._dispatch_outer_fetch()
            if self._outer_future is None:
                return None
        try:
            ctx = self._outer_future.result(timeout=float(timeout_s))
        except Exception:
            return None
        if isinstance(ctx, dict):
            self.state.outer_context = ctx
            return ctx
        return None

    def _reset_outer_state(self) -> None:
        self._outer_future = None

    def _post_formulate_detect_entities(self) -> None:
        if not self._outer_enabled():
            return
        if self.state.entity_refs or self.state.needs_outer:
            return
        refs: dict = {}
        fo = self.state.formulate_output or {}
        pid = fo.get("participant_person_id")
        if isinstance(pid, str) and pid:
            refs["primary_person"] = pid
        topic = self.state.impasse_topic or fo.get("impasse_topic")
        if isinstance(topic, str) and topic:
            refs["current_topic"] = topic
        if not refs:
            return
        self.state.entity_refs = refs
        self.state.needs_outer = dict(refs)
        self._dispatch_outer_fetch()

    # ── Policy-input builders (outer copies of the inner 80-D / 30-D builders;
    #    deterministic over sv/nm/self.state/self._ema_state/_strategy_history) ─
    def _build_meta_input(self, sv, nm, chain_archive, meta_autoencoder=None) -> list:
        inp: list = []
        sv_arr = np.array(sv[:132], dtype=np.float32)
        inp.extend([
            float(np.mean(sv_arr[0:5])), float(np.mean(sv_arr[5:20])),
            float(np.mean(sv_arr[20:65])), float(np.mean(sv_arr[65:70])),
            float(np.mean(sv_arr[70:85])), float(np.mean(sv_arr[85:130])),
        ])
        inp.extend([
            float(np.std(sv_arr[0:5])), float(np.std(sv_arr[5:20])),
            float(np.std(sv_arr[20:65])), float(np.std(sv_arr[65:70])),
            float(np.std(sv_arr[70:85])), float(np.std(sv_arr[85:130])),
        ])
        for i_start, i_end, o_start, o_end in [(0, 5, 65, 70), (5, 20, 70, 85), (20, 65, 85, 130)]:
            inner = sv_arr[i_start:i_end]
            outer = sv_arr[o_start:o_end]
            min_len = min(len(inner), len(outer))
            if min_len > 0:
                corr = float(np.corrcoef(inner[:min_len], outer[:min_len])[0, 1])
                inp.append(0.0 if np.isnan(corr) else corr)
            else:
                inp.append(0.0)
        deviation = np.abs(sv_arr - self._ema_state)
        top5 = np.argsort(deviation)[-5:][::-1]
        inp.extend([float(d / 132.0) for d in top5])
        # Problem embedding (16D) — hash of the formulated template (the outer
        # engine has no trained meta_autoencoder; the hash fallback is faithful).
        if self.state.formulate_output:
            template = self.state.formulate_output.get("problem_template", "")
            inp.extend([(hash((template, i)) % 1000) / 1000.0 for i in range(16)])
        else:
            inp.extend([0.5] * 16)
        inp.extend(self._strategy_history.tolist())  # 12D
        del_vec = [0.0] * 8
        for i, dr in enumerate(self.state.delegate_results[-4:]):
            del_vec[i * 2] = dr.get("confidence", 0.5)
            del_vec[i * 2 + 1] = dr.get("gut_agreement", 0.5)
        inp.extend(del_vec)
        inp.extend([0.0] * 8)  # archive stats — outer chain_archive optional
        n = len(self.state.chain)
        inp.extend([
            n / max(self.state.max_steps, 1), self.state.confidence,
            float(self._strategy_history[:6].sum()),
            1.0 if self.state.awaiting_delegate else 0.0,
        ])
        inp.extend([
            nm.get("DA", 0.5), nm.get("5HT", 0.5), nm.get("NE", 0.5),
            nm.get("ACh", 0.5), nm.get("Endorphin", 0.5), nm.get("GABA", 0.5),
        ])
        inp.extend([
            float(self.state.break_count) / max(self.state.max_breaks, 1),
            len(self.state.checkpoints) / 5.0,
            float(self.state.spirit_self_cooldown) / max(self._spirit_self_cooldown_max, 1),
            1.0 if self.state.last_spirit_self_step >= 0 else 0.0,
            float(self._total_eurekas) / max(self._total_meta_chains, 1),
            1.0 if self._total_meta_chains >= self._spirit_self_gate else 0.0,
        ])
        inp = inp[:META_POLICY_INPUT_DIM]
        while len(inp) < META_POLICY_INPUT_DIM:
            inp.append(0.0)
        return inp

    def _build_sub_mode_input(self, sv, nm, primitive) -> list:
        from titan_hcl.logic.meta_reasoning import SUB_MODE_INPUT_DIM
        inp: list = []
        sv_arr = np.array(sv[:132], dtype=np.float32)
        inp.extend([
            float(np.mean(sv_arr[0:5])), float(np.mean(sv_arr[5:20])),
            float(np.mean(sv_arr[20:65])), float(np.mean(sv_arr[65:70])),
            float(np.mean(sv_arr[70:85])), float(np.mean(sv_arr[85:130])),
        ])
        deviation = np.abs(sv_arr - self._ema_state)
        inp.extend([
            float(np.mean(deviation[0:5])), float(np.mean(deviation[5:20])),
            float(np.mean(deviation[20:65])), float(np.mean(deviation[65:70])),
            float(np.mean(deviation[70:85])), float(np.mean(deviation[85:130])),
        ])
        n = len(self.state.chain)
        inp.extend([
            1.0 if self.state.formulate_output else 0.0,
            1.0 if self.state.recalled_data else 0.0,
            min(len(self.state.hypotheses) / 3.0, 1.0),
            min(len(self.state.delegate_results) / 3.0, 1.0),
        ])
        inp.extend([
            n / max(self.state.max_steps, 1), self.state.confidence,
            float(self.state.break_count) / 3.0,
            1.0 if self.state.awaiting_delegate else 0.0,
        ])
        inp.extend([
            nm.get("DA", 0.5), nm.get("5HT", 0.5), nm.get("NE", 0.5),
            nm.get("ACh", 0.5), nm.get("Endorphin", 0.5), nm.get("GABA", 0.5),
        ])
        inp.extend([
            float(self.state.break_count) / max(self.state.max_breaks, 1),
            len(self.state.checkpoints) / 5.0,
            float(self.state.spirit_self_cooldown) / max(self._spirit_self_cooldown_max, 1),
            1.0 if self.state.last_spirit_self_step >= 0 else 0.0,
        ])
        inp = inp[:SUB_MODE_INPUT_DIM]
        while len(inp) < SUB_MODE_INPUT_DIM:
            inp.append(0.0)
        return inp

    def _get_temperature(self, nm) -> float:
        da = nm.get("DA", 0.5)
        ne = nm.get("NE", 0.5)
        return float(max(0.3, min(1.5, 0.7 + da * 0.6 - ne * 0.3)))

    # ── Dispatcher (own copy — calls the inherited mixin handlers) ──────────
    def _execute(self, prim, sub, sv, nm, reasoning_engine,
                 chain_archive, meta_wisdom, exp_orchestrator, autoencoder):
        if prim == "FORMULATE":
            result = self._prim_formulate(sub, sv, nm, meta_wisdom, autoencoder)
            self._post_formulate_detect_entities()
            return result
        if prim == "RECALL":
            return self._prim_recall(sub, chain_archive, meta_wisdom, exp_orchestrator)
        if prim == "HYPOTHESIZE":
            return self._prim_hypothesize(sub, nm)
        if prim == "DELEGATE":
            return self._prim_delegate(sub, reasoning_engine)
        if prim == "SYNTHESIZE":
            return self._prim_synthesize(sub, meta_wisdom, autoencoder, sv)
        if prim == "EVALUATE":
            return self._prim_evaluate(sub, nm)
        if prim == "BREAK":
            return self._prim_break(sub, sv, reasoning_engine)
        if prim == "SPIRIT_SELF":
            return self._prim_spirit_self(sub, nm)
        if prim == "INTROSPECT":
            return self._prim_introspect(sub, sv, nm)
        return {"primitive": prim, "error": "unknown"}

    # ── Chain orchestration ─────────────────────────────────────────────────
    def _start_chain(self, reason: str, sv, problem: Optional[dict]) -> None:
        """Minimal outer seed — sets current_topic + entry_primitive from the
        drawn problem; NO felt-cluster emit / emot-shm / chain_iql (inner-only)."""
        self.state = MetaChainState()
        self.state.is_active = True
        self.state.trigger_reason = reason
        self.state.start_time = time.time()
        self.state.max_steps = self.max_steps
        self.state.pre_state_132d = list(sv[:132])
        p = problem or {}
        topic = str(p.get("topic") or p.get("goal_class") or "")
        entry = str(p.get("entry_primitive") or "")
        if topic:
            self.state.entity_refs["current_topic"] = topic
            self.state.grounding_concept = topic
        if entry:
            self.state.entry_primitive = entry
        self.state.chain_id = self._next_chain_id
        self._next_chain_id += 1
        self._total_meta_chains += 1
        self._chain_transitions = []
        self._reset_outer_state()

    def _should_conclude(self) -> bool:
        n = len(self.state.chain)
        if n >= self.state.max_steps:
            return True
        if n >= self.min_chain_length and self.state.confidence > 0.8:
            return True
        if (self.state.chain_results
                and self.state.chain_results[-1].get("recommendation") == "conclude"
                and n >= self.min_chain_length):
            return True
        return False

    def run_chain(self, problem: Optional[dict] = None, sv=None, nm=None,
                  reasoning_engine=None, chain_archive=None, meta_wisdom=None,
                  exp_orchestrator=None, meta_autoencoder=None,
                  max_steps: Optional[int] = None) -> dict:
        """Run one full outer meta-chain to conclusion. Returns the conclusion
        dict (the composite payload). DELEGATE drives ``reasoning_engine``
        (the piece-3 OuterReasoningEngine) synchronously."""
        self._problem = dict(problem or {})
        if sv is None:
            sv = [0.5] * _NEUTRAL_SV_DIM
        nm = nm or _NEUTRAL_NEUROMODS
        self._start_chain(self._problem.get("reason", "outer_explore"), sv, self._problem)

        cap = max_steps if max_steps is not None else (self.max_steps + 4)
        for _ in range(cap):
            meta_input = self._build_meta_input(sv, nm, chain_archive, meta_autoencoder)
            temperature = self._get_temperature(nm)
            prim_idx = int(self.meta_policy.select_action(meta_input, temperature))
            prim_name = META_PRIMITIVES[prim_idx]

            # Minimal state-based gates (faithful subset — prevent pathology;
            # NO inner maturity/CGN gates). Reroute to EVALUATE.
            if prim_name == "BREAK" and self.state.break_count >= self.state.max_breaks:
                prim_name, prim_idx = "EVALUATE", META_PRIMITIVES.index("EVALUATE")
            elif prim_name == "INTROSPECT" and self.state.introspect_used:
                prim_name, prim_idx = "EVALUATE", META_PRIMITIVES.index("EVALUATE")
            self.state.spirit_self_cooldown = max(0, self.state.spirit_self_cooldown - 1)

            sub_input = self._build_sub_mode_input(sv, nm, prim_name)
            sub_idx = int(self.sub_mode_policies[prim_name].select_action(sub_input, temperature))
            sub_name = SUB_MODES[prim_name][sub_idx]

            if prim_name == "EVALUATE":
                self.state.pre_eval_confidence = float(self.state.confidence)

            result = self._execute(prim_name, sub_name, sv, nm, reasoning_engine,
                                   chain_archive, meta_wisdom, exp_orchestrator,
                                   meta_autoencoder)

            # DELEGATE → drive the outer reasoning engine SYNCHRONOUSLY, then
            # resolve via the inherited _check_delegate (reads conf/gut, appends
            # to delegate_results, clears bias, blends meta confidence).
            if (prim_name == "DELEGATE" and self.state.awaiting_delegate
                    and reasoning_engine is not None):
                try:
                    reasoning_engine.run_chain()  # advances _total_chains by 1
                except Exception as e:  # noqa: BLE001
                    logger.debug("[OuterMeta] delegate run_chain failed: %s", e)
                self._check_delegate(reasoning_engine)

            # EVALUATE ← oracle (verifiable lane): score the delegated result.
            if (prim_name == "EVALUATE" and self._oracle_verify is not None
                    and self.state.delegate_results):
                try:
                    ov = float(self._oracle_verify(self._problem, self.state.delegate_results))
                    ov = max(-1.0, min(1.0, ov))
                    self.state.outer_context["oracle_verdict"] = ov
                    # nudge meta confidence toward the verdict (±, bounded)
                    self.state.confidence = float(np.clip(
                        self.state.confidence + (ov * 0.2), 0.0, 1.0))
                    result = {**result, "oracle_verdict": ov,
                              "confidence": self.state.confidence}
                except Exception as e:  # noqa: BLE001
                    logger.debug("[OuterMeta] oracle_verify failed: %s", e)

            if prim_name == "RECALL":
                self.state.recall_history.append(
                    {"source": sub_name, "count": result.get("count", 0)})

            # Record transition + step reward (STEP_REWARDS catalog + bonuses).
            step_key = f"{prim_name}.{sub_name}"
            step_reward = STEP_REWARDS.get(step_key, 0.0)
            if result.get("count", 0) > 0:
                step_reward += min(result["count"] * 0.02, 0.10)
            if result.get("wisdom_found"):
                step_reward += 0.05
            self.buffer.record(meta_input, prim_idx, sub_idx, step_reward)
            # Store the EXACT inputs used at selection time (sub_input depends on
            # the per-step chain state — rebuilding it post-chain would mis-credit).
            self._chain_transitions.append(
                (meta_input, prim_idx, sub_input, sub_idx, prim_name))

            # Update chain state.
            self.state.chain.append(step_key)
            self.state.chain_results.append(result)
            self.state.step_rewards.append(step_reward)
            self.state.confidence = result.get("confidence", self.state.confidence)
            self._total_meta_steps += 1

            if prim_name in ("FORMULATE", "SYNTHESIZE"):
                self._save_checkpoint()  # inherited (mixin)

            # Strategy-history EMA (faithful to inner).
            sh_len = min(NUM_META_ACTIONS, 6)
            one_hot = np.zeros(sh_len, dtype=np.float32)
            if prim_idx < sh_len:
                one_hot[prim_idx] = 1.0
            self._strategy_history[:sh_len] = (
                0.9 * self._strategy_history[:sh_len] + 0.1 * one_hot)
            self._strategy_history[sh_len:] = 0.95 * self._strategy_history[sh_len:]
            if prim_idx < len(self._strategy_history) - sh_len:
                self._strategy_history[sh_len + prim_idx] = 1.0

            if self._should_conclude():
                return self._conclude_chain(sv)

        return self._conclude_chain(sv)

    def _conclude_chain(self, sv) -> dict:
        """Conclude → the Reasoning-composite payload (the bus emit +
        SynthesisWriter write is piece 6). Computes the chain reward: the oracle
        verdict if the verifiable lane ran, else a confidence-derived reward."""
        oracle_verdict = self.state.outer_context.get("oracle_verdict")
        if oracle_verdict is None and self._oracle_verify is not None and self.state.delegate_results:
            try:
                oracle_verdict = max(-1.0, min(1.0, float(
                    self._oracle_verify(self._problem, self.state.delegate_results))))
            except Exception:  # noqa: BLE001
                oracle_verdict = None
        reward = (float(oracle_verdict) if oracle_verdict is not None
                  else (self.state.confidence - self._baseline_confidence) * 2.0)
        reward = max(-1.0, min(1.0, reward))

        verified = oracle_verdict is not None and oracle_verdict > 0
        self.state.chain_succeeded = 1.0 if reward > 0 else 0.0
        self.state.is_active = False
        self._action_counts["conclude"] = self._action_counts.get("conclude", 0) + 1
        self._last_outcome = f"reward={reward:+.3f} verified={verified}"

        composite = {
            "action": "conclude",
            "topic": self.state.entity_refs.get("current_topic", ""),
            "chain": list(self.state.chain),
            "chain_length": len(self.state.chain),
            "confidence": round(self.state.confidence, 4),
            "reward": round(reward, 4),
            "oracle_verdict": oracle_verdict,
            "verified": verified,
            "delegate_results": list(self.state.delegate_results),
            "duration_s": round(time.time() - self.state.start_time, 2),
            "chain_id": self.state.chain_id,
            # idea_type=procedural (synthesis SPEC §6.2.8) — a Reasoning composite.
            "idea_type": "procedural",
        }
        logger.debug("[OuterMeta] conclude — chain=%d conf=%.3f reward=%.3f verified=%s",
                     composite["chain_length"], self.state.confidence, reward, verified)
        return composite

    def train_terminal(self, reward: float, baseline: float = 0.0) -> dict:
        """Train the fresh MetaPolicy + the per-primitive SubModePolicies toward
        the terminal reward, crediting every (primitive, sub-mode) taken in the
        chain just concluded (REINFORCE w/ baseline). Updates the EMA baseline."""
        transitions = self._chain_transitions
        if not transitions:
            return {"trained": 0, "reason": "no_transitions"}
        # EMA reward baseline (REINFORCE).
        self._baseline_confidence = (
            0.95 * self._baseline_confidence + 0.05 * float(reward))
        adv = float(reward) - float(baseline)
        trained = 0
        for meta_input, prim_idx, sub_input, sub_idx, prim_name in transitions:
            self.meta_policy.train_step(meta_input, prim_idx, adv)
            self.sub_mode_policies[prim_name].train_step(sub_input, sub_idx, adv)
            trained += 1
        self._chain_transitions = []
        return {"trained": trained, "reward": float(reward),
                "advantage": round(adv, 4),
                "meta_updates": int(self.meta_policy.total_updates)}

    # ── Persistence / stats ─────────────────────────────────────────────────
    def save_all(self) -> None:
        if not self.save_dir:
            return
        self.meta_policy.save(os.path.join(self.save_dir, "outer_meta_policy_net.json"))

    def get_stats(self) -> dict:
        return {
            "total_meta_chains": self._total_meta_chains,
            "total_meta_steps": self._total_meta_steps,
            "action_counts": dict(self._action_counts),
            "last_outcome": self._last_outcome,
            "meta_policy_updates": int(self.meta_policy.total_updates),
            "buffer_size": self.buffer.size() if hasattr(self.buffer, "size") else None,
        }


__all__ = ["OuterMetaReasoningEngine"]
