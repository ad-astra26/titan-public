"""Self-Learning Worker — L2 owner of the OUTER meta-reasoning policy.

RFP_synthesis_self_learning_meta_reasoning Phase 1 (§7.A, L1-L3). The OFF-HOT-PATH
half of the verifiable-lane closed loop:

  agno DECIDE (exploit) ──SELF_LEARN_DECISION──▶ stash by parent_tool_call_tx
  oracle verdict @ dream-flush ──SELF_LEARN_REWARD──▶ JOIN on parent_tool_call_tx
      → OuterMetaPolicy.learn(features, action, reward)  (REINFORCE-with-baseline)
      → publish updated weights to SHM (read O(1) by the next DECIDE turn)
      → record the reward tuple; when a goal_class accrues enough verified wins,
        distill a macro-strategy ──SELF_LEARN_MACRO_READY──▶ synthesis_worker
        (writes the `Reasoning` node under `Self` via the single SynthesisWriter)

State ownership (G21 / INV-OML-8): this worker owns ONLY `data/self_learning.duckdb`
(policy weights + pending decisions + reward tuples + explore log). It NEVER opens a
spine native handle — all spine writes route through synthesis_worker's SynthesisWriter
(INV-Syn-19/28). The reward is ASYNC (the verdict arrives at the dream-flush, not at
turn time), so the decision and reward are two events joined on `parent_tool_call_tx`
— the load-bearing detail (§1.2 E4).

Exploration (L3) is idle-only + metabolically gated (INV-OML-9): never on a live user
turn. Phase 1's concrete explore action is experience-replay on accumulated reward
tuples (self-contained, sharpens the policy between sparse live rewards); the active
idle problem-generation request (SELF_LEARN_EXPLORE_REQUEST) is emitted only when its
idle-loop consumer is wired (config `explore_request_enabled`, default off — Phase 2).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from queue import Empty

import duckdb
import numpy as np

from titan_hcl import bus
from titan_hcl.bus import (
    SELF_LEARN_DECISION,
    SELF_LEARN_EXPLORE_REQUEST,
    SELF_LEARN_MACRO_READY,
    SELF_LEARN_REWARD,
)
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _sev
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)
from titan_hcl.synthesis.outer_meta_policy import (
    MSL_CONTEXT_DIM,
    NUM_OUTER_ACTIONS,
    OUTER_ACTIONS,
    OUTER_META_POLICY_STATE_SPEC,
    OUTER_POLICY_INPUT_DIM,
    OuterMetaPolicy,
    _BASE_FEATURE_NAMES,
    action_index_to_name,
    structural_target_action,
)
from titan_hcl.synthesis.mastery_level import (
    MASTERY_LEVEL_STATE_SPEC,
    MasteryLevel,
    mastery_readout_to_flat,
)
from titan_hcl.synthesis.outer_meta_reasoning import OuterMetaReasoningEngine
from titan_hcl.synthesis.outer_reasoning import OuterReasoningEngine
from titan_hcl.params import get_params

logger = logging.getLogger("self_learning")

_HEARTBEAT_INTERVAL_S = 30.0
_WORKER_READY: bool = False
_BOOT_DEADLINE = None

# Defaults (overridable via config[synthesis][self_learning]).
_DEFAULTS = {
    "enabled": False,                  # spawn-gate (module_catalog) + agno consume-gate
    "policy_lr": 0.01,
    "baseline_alpha": 0.05,
    "pending_ttl_s": 1800.0,           # drop un-joined decisions older than this
    "explore_interval_s": 120.0,       # idle EXPLORE tick cadence
    "explore_chi_floor": 0.30,         # metabolic floor for exploration (INV-OML-9)
    "explore_replay_batch": 16,        # experience-replay batch size
    "explore_balanced": True,          # replay BALANCED across actions (deadlock fix, step 1)
    # ── Active idle action-space exploration (deadlock fix step 2, §7.C/§24.6).
    #    On recalled contexts, teach the policy the VERIFIABLE structural target
    #    (structural_target_action) directly via cross-entropy — so the under-used
    #    actions (esp. IDK, which has NO live reward stream) get learned on the
    #    contexts where they are structurally correct, off the live path
    #    (INV-OML-9). The target is VERIFIABLE (the memory-search "does he know?"
    #    signal + tool-intent + skill-match + metabolic affordability), so we
    #    teach it directly rather than Boltzmann-sample-and-reward (which decays
    #    to a degenerate single-action policy — REINFORCE's advantage → 0 at
    #    convergence). Stochastic discovery belongs to LIVE turns (unknown
    #    outcomes); idle consolidates toward the verifiable oracle, live verdicts
    #    refine BEYOND it. This is what actually closes G5/GB8.
    "explore_structural": True,        # run the structural-verifier bootstrap pass
    # Lever 1 (§24.9 — THE routing-collapse fix). The structural cross-entropy
    # pass is the ONLY feature-CONDITIONAL signal (the live judge rewards a single
    # action regardless of features → collapse). Offline lever experiments
    # (`/tmp/lever_lab.py`, 5 seeds) reproduced the collapse at routing-acc 0.33
    # and showed cranking THIS pass (freq 24→64 contexts/tick + advantage 1→3)
    # restores feature-conditional routing → 1.00; tightening max_weight_norm
    # HURT and entropy-reg was neutral, so only this knob moved. (2026-06-12.)
    "explore_structural_batch": 64,    # recalled contexts taught per idle tick (Lever 1: was 24)
    "explore_structural_advantage": 3.0,  # cross-entropy push magnitude toward the verified target (Lever 1: was 1.0)
    "explore_know_threshold": 0.65,    # recall_top_cosine ≥ this ⇒ "he knows" (→ direct).
    #   Calibrated 0.5→0.65 (§24.9) to MATCH the grounded router's recall_known_floor
    #   (grounded_router.py default 0.65). The live T3 soak showed the 0.5 boundary made
    #   moderate-recall informational prompts (recall ~0.5–0.65 — a false-positive "knows"
    #   signal) route `direct` where grounded sends them `research`. Same boundary ⇒ the
    #   seed + idle pass agree with the symbolic router on "does he actually know?".
    "explore_skill_floor": 0.3,        # skill_utility ≥ this + matched ⇒ reuse (→ skill_delegate)
    # Break D (RFP_synthesis_reuse_and_routing_revival) — anti-collapse. Each idle
    # tick re-teaches the balanced SYNTHETIC lane set (the cold-start lanes) this
    # many passes, UNCONDITIONALLY, so feature-discrimination is maintained against
    # the dense quality-judge `direct` stream on a restored (never-re-seeded) policy.
    # 0 disables (pure real-context teaching — the old, collapse-preserving behavior).
    "structural_synthetic_passes": 3,
    "explore_research_chi_floor": 0.4, # chi ≥ this ⇒ research affordable; else honest IDK
    "explore_request_enabled": False,  # active idle problem-gen (LLM-judge layer; Phase 2 consumer)
    "macro_min_wins": 5,               # verified wins of one (goal_class,action) → distil
    "macro_refine_min_growth": 5,      # §7.D D.4c — +N verified wins since last emit → a successor composite
    # §7.D D.4b — macro-of-macros: when the deliberation for goal_class G builds on
    # ≥ macro_compose_min_children ALREADY-emitted macros whose 30-D mean-feature
    # signature is cosine ≥ macro_compose_floor to G's (the D7-honest reuse signal —
    # numpy over the worker's OWN verified macros, no 384-D embedder), the new macro
    # is emitted `composed_from` those child macro labels → REASONING_COMPOSED_FROM
    # edges (parent-of-children provenance, not the D.1 leaf-join). Conservative
    # floor so only genuinely feature-near verified macros compose.
    "macro_compose_floor": 0.85,       # §7.D D.4b — cosine over 30-D signatures
    "macro_compose_min_children": 2,   # §7.D D.4b — ≥2 children → a macro-of-macros
    # Fix 2 (§24.9 — cold-start feature-discriminating seed). A FRESH policy's
    # argmax is ~uniform; the first dense live reward stream (the turn-judge,
    # which rewards a single action regardless of features) collapses it to that
    # action BEFORE it learns to discriminate on features. We warm-start a
    # cold-started policy from the SAME verifiable structural oracle the idle pass
    # teaches (`structural_target_action`) on a synthetic balanced feature set, so
    # it routes computable→tool / known→direct / unknowable→research /
    # skill→skill_delegate from boot. Cold-start ONLY (a restored policy already
    # carries learned routing); live verdicts + Lever-1's boosted idle pass refine
    # beyond it. Deterministic (fixed rng) → identical warm-start every cold boot.
    "cold_start_seed_enabled": True,   # warm-start a cold policy toward feature-conditional routing
    "cold_start_seed_epochs": 300,     # passes over the synthetic balanced lane set
    "cold_start_seed_advantage": 3.0,  # cross-entropy push magnitude for the seed
    # ── Phase C (§7.C piece 6) — the OUTER two-level reasoner as the continuous
    #    idle-time deliberative learner. When enabled, the explore tick runs the
    #    OuterMetaReasoningEngine over OuterReasoningEngine on a verified-win
    #    goal_class → trains the outer policies + emits the (verified) macro;
    #    the reactive Phase-1 `_maybe_distill_macro` is SUPERSEDED (fallback when
    #    disabled). Numpy-only, INV-OML-8 isolation preserved.
    "outer_meta_enabled": True,        # explore-tick deliberative macro path (supersede)
    "outer_meta_max_steps": 16,        # max meta-chain length per deliberation
    # ── Anti-runaway regularization (2026-06-11). The unregularized REINFORCE
    #    let the off-policy `tool` attribution explode the policy weights
    #    (scores ~1100 → always-tool collapse). weight_decay = decoupled L2
    #    pull/step; max_weight_norm = hard per-matrix Frobenius cap.
    "weight_decay": 0.001,
    "max_weight_norm": 6.0,
    # ── Phase B (§7.B) — non-verifiable reward source weights (the "bigger delta
    #    for Maker" mechanic). reward × weight → bigger advantage → bigger nudge.
    "judge_reward_weight": 1.0,        # LLM turn-judge (metered tier)
    "user_reward_weight": 1.0,         # ordinary user rating (reward-only nudge)
    "maker_reward_weight": 2.0,        # Maker rating — more authority over his own Titan
    # ── Canonical-IQL routing learner (RFP_emergent_mastery_curriculum P2) ──
    # FULL IQL (learned Q + expectile-V + AWR + Bellman-with-V(s') + Polyak),
    # mirroring CGN-IQL by hand in numpy on OuterMetaPolicy (INV-MC-6). The
    # foundation that gives V(s) for the emergent MasteryLevel (P3). DEFAULT ON
    # (all-flags-default-on); flag-off ⇒ byte-identical legacy REINFORCE path.
    # When on: _handle_reward only BUFFERS (record_reward_tuple); learning is the
    # offline train_iql pass in the idle _explore_tick (pure-offline, like CGN
    # consolidate). next_state from per-goal_class trajectory links.
    "oml_iql_enabled": True,
    "iql_tau": 0.7,                    # expectile asymmetry (optimistic V)
    "iql_beta": 3.0,                   # AWR temperature
    "iql_gamma": 0.99,                 # Bellman discount (full IQL — uses V(s'))
    "iql_polyak": 0.005,              # Q-target EMA rate
    "iql_adv_clip": 100.0,            # AWR weight cap
    "iql_lr": 0.003,                  # SGD step for the V/Q/π updates
    "iql_steps": 20,                  # gradient steps per idle consolidation pass
    "iql_batch_size": 32,             # transitions per step
    "iql_replay_window": 2000,        # most-recent reward_tuples drawn for trajectory linking
    # ── Emergent MasteryLevel (P3, ARCHITECTURE_mastery_leveling.md §2.2/§5) ──
    # Self-emergent ability level from the IQL value V(s): EMA(symlog-V̄) binned
    # on a fixed grade ladder, ratcheted only when a SCALE-FREE competence_rate
    # confirms it (SOAR ① — anti reward-scale-inflation). Computed each idle
    # _explore_tick after train_iql; published to SHM (G21) for dashboard/P4-P5.
    "level_n_grades": 10,
    "level_grade_lo": -5.0,           # symlog-space grade support (r∈[−1,1],γ=0.99)
    "level_grade_hi": 5.0,
    "level_ema_alpha": 0.05,          # V̄ smoothing
    "level_competence_floor_base": 0.55,   # competence gate at grade 0
    "level_competence_floor_slope": 0.02,  # per-grade floor increment
    "competence_w_succ": 0.6,         # weight: verified-success-rate
    "competence_w_adv": 0.4,          # weight: advantage-positive-rate
    "competence_window": 200,         # recent decisions for success_rate
    "competence_ema_alpha": 0.05,     # competence_rate smoothing
    # ── P4 — level-driven reward shaping (ARCHITECTURE_mastery_leveling.md §4) ──
    # As the level rises, the easy `direct` path's POSITIVE reward decays (coasting
    # pays less → the policy explores harder paths). Applied to the IQL TRAINING
    # transitions only — the buffered reward_tuples stay RAW so the scale-free
    # competence signal stays correctness-grounded (INV-ML-3). Floor>0 keeps
    # `direct` a valid learned action (INV-MC-4). Other actions are never damped.
    "level_shaping_enabled": True,
    "direct_damping_floor": 0.3,      # min multiplier on a positive direct reward (INV-MC-4)
    "direct_damping_slope": 0.07,     # per-level decay (level 10 → 1−0.7 = floor)
    "direct_ungrounded_recall_threshold": 0.5,  # recall_top_cosine below this = ungrounded
    "direct_ungrounded_extra_damping": 0.7,     # ungrounded direct damped MORE
    "direct_graduated_floor": 0.6,    # mastered (chunked) goal_class → ease damping (higher floor)
    # ── P5 — level-adaptive co-adaptive teacher (ARCHITECTURE_mastery_leveling.md §4) ──
    # The teacher is driven by the EMERGENT ratcheted level (NOT a hand-drawn curve —
    # Q2 resolved 2026-06-18): `level_norm = level / level_n_grades` ∈ [0,1] linearly
    # maps the reward-source mix. (a) relative-to-self + (b) level-rubric live in the
    # synthesis TurnJudge (teacher_relself_*); (c) the graduated source-mix is here:
    # as level↑ the llm_judge weight DECAYS toward a floor>0 (the non-verifiable lane
    # never fully starves — INV-MC-4 spirit) while oracle/maker RISE. "self" is NOT a
    # source (Q4 → RFP_introspective_inner_turn's grounded predict-vs-measure signal).
    # _REWARD_SOURCE_RANK authority order is UNCHANGED — only the magnitudes graduate.
    # flag-off ⇒ static Phase-B weights, byte-identical (INV-MC-7).
    "teacher_coadaptive_enabled": True,
    "teacher_relself_gain": 1.0,        # (a) relative-to-self strength (read by synthesis TurnJudge)
    "teacher_relself_ema_alpha": 0.1,   # (a) recent-self EMA smoothing (read by synthesis TurnJudge)
    "teacher_judge_weight_floor": 0.3,  # (c) llm_judge weight floor>0 at max level (INV-MC-4 spirit)
    "teacher_authority_rise_gain": 1.0, # (c) oracle/maker rise: weight = base × (1 + gain·level_norm)
    # ── P7 — clean-baseline reset (uncollapse) ──────────────────────────────
    # One-shot reset trigger: if this sentinel file exists at worker boot, the
    # collapsed routing policy + IQL nets + level + replay buffer are CLEARED →
    # a true cold-start relearn on the IQL loop (the acceptance-proof reset). The
    # worker DELETES the sentinel after consuming it (so it never loops). To reset
    # a Titan: `touch <reset_sentinel_path>` on its box, then full-restart it.
    "reset_sentinel_path": "data/.reset_routing_policy",
}
# Reward-source authority rank (Phase B corrective-delta): a higher-rank source
# may correct a lower-rank applied reward; same-or-lower is ignored (no double-train).
# `oracle` (§24.10, Fix 1) = the OVG numeric-VERIFICATION outcome — objective
# correctness for a verifiable claim, so it out-ranks ALL subjective sources
# (the quality turn-judge, a user/Maker rating): a numerically-wrong answer is
# wrong regardless of how it reads. (Emitted at verdict-time on the direct path
# today; the rank governs any future join-path correction of the turn-judge.)
_REWARD_SOURCE_RANK = {"llm_judge": 0, "user": 1, "maker": 2, "oracle": 3}
_SURVIVAL_STATES = frozenset({"SURVIVAL", "STARVATION"})


def _graduated_source_weight(src: str, base_weight: float, level_norm: float,
                             cfg) -> float:
    """P5 (c) — the level-graduated reward-source weight (ARCHITECTURE §4).

    EMERGENT (Q2): the multiplier is a MINIMAL LINEAR map on the agent's own proven
    `level_norm` ∈ [0,1] (no hand-drawn curve, only config gains/floors):
      • `llm_judge`        → DECAYS  `base × max(judge_floor, 1 − level_norm)` — wean
        off the noisy LLM crutch as proven ability rises, but a floor>0 keeps the
        non-verifiable lane a live signal (INV-MC-4 spirit; the relative-to-self
        rubric is what keeps that floored stream non-stale, INV-MC-3).
      • `oracle` / `maker` → RISE    `base × (1 + rise_gain · level_norm)` — lean on
        the correctness-/authority-grounded signals more as he graduates.
      • everything else (`user`)     → unchanged base.
    flag-off (`teacher_coadaptive_enabled=false`) or an unpublished level
    (`level_norm<=0`) ⇒ returns `base_weight` unchanged (byte-identical, INV-MC-7)."""
    if not bool(cfg.get("teacher_coadaptive_enabled", True)) or level_norm <= 0.0:
        return base_weight
    ln = max(0.0, min(1.0, float(level_norm)))
    if src == "llm_judge":
        floor = float(cfg.get("teacher_judge_weight_floor", 0.3))
        return base_weight * max(floor, 1.0 - ln)
    if src in ("oracle", "maker"):
        rise = float(cfg.get("teacher_authority_rise_gain", 1.0))
        return base_weight * (1.0 + rise * ln)
    return base_weight


# ── The worker's OWN durable store (G21 — NOT synthesis.duckdb) ─────────────
class _SelfLearningStore:
    """Single-process owned (this worker only) → plain duckdb is safe. Soft-fail."""

    def __init__(self, path: str = os.path.join("data", "self_learning.duckdb")):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = duckdb.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS policy_state ("
            " id INTEGER PRIMARY KEY, weights_json VARCHAR,"
            " total_updates INTEGER, reward_baseline DOUBLE, ts DOUBLE)")
        # IQL nets (V/Q/Q-target) durable artifact — SEPARATE from policy_state
        # (which carries the π SHM flat). Keeps the agno-read SHM slot
        # byte-identical (P2 §7.P2 step 5). Worker-only; G21 single-writer.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS policy_iql_state ("
            " id INTEGER PRIMARY KEY, iql_json VARCHAR,"
            " total_iql_updates BIGINT, ts DOUBLE)")
        # MasteryLevel state (P3) — emergent level ratchet/EMA, worker-only (G21).
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS mastery_level_state ("
            " id INTEGER PRIMARY KEY, state_json VARCHAR, ts DOUBLE)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_decisions ("
            " parent_tool_call_tx VARCHAR PRIMARY KEY, features_json VARCHAR,"
            " action INTEGER, goal_class VARCHAR, turn_id VARCHAR, ts DOUBLE,"
            " applied_reward DOUBLE, applied_source VARCHAR)")
        # Back-compat: existing fleet dbs (created pre-§7.B) lack the corrective-
        # delta columns — add them idempotently so a deployed worker self-migrates.
        for _col, _ty in (("applied_reward", "DOUBLE"), ("applied_source", "VARCHAR")):
            try:
                self._conn.execute(
                    f"ALTER TABLE pending_decisions ADD COLUMN IF NOT EXISTS "
                    f"{_col} {_ty}")
            except Exception as e:  # noqa: BLE001
                logger.debug("[self_learning] pending_decisions ALTER %s: %s", _col, e)
        self._conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS seq_reward_tuples START 1")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS reward_tuples ("
            " id INTEGER PRIMARY KEY DEFAULT nextval('seq_reward_tuples'),"
            " features_json VARCHAR, action INTEGER, reward DOUBLE,"
            " goal_class VARCHAR, ts DOUBLE)")
        self._conn.execute(
            "CREATE SEQUENCE IF NOT EXISTS seq_explore_log START 1")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS explore_log ("
            " id INTEGER PRIMARY KEY DEFAULT nextval('seq_explore_log'),"
            " kind VARCHAR, goal_class VARCHAR, detail VARCHAR, ts DOUBLE)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS macro_emitted ("
            " goal_class VARCHAR, action INTEGER, ts DOUBLE,"
            " version INTEGER DEFAULT 1, wins_at_emit INTEGER DEFAULT 0,"
            " PRIMARY KEY (goal_class, action))")
        # §7.D D.4c — additive columns on a pre-D.4c macro_emitted table (refinement
        # versioning / mutate-not-update). Idempotent (DuckDB IF NOT EXISTS).
        for _col, _ddl in (("version", "INTEGER DEFAULT 1"),
                           ("wins_at_emit", "INTEGER DEFAULT 0")):
            try:
                self._conn.execute(
                    f"ALTER TABLE macro_emitted ADD COLUMN IF NOT EXISTS {_col} {_ddl}")
            except Exception as _e:  # noqa: BLE001
                logger.debug("[self_learning] macro_emitted ALTER %s soft-fail: %s",
                             _col, _e)

    # -- policy weights -------------------------------------------------
    def load_policy_flat(self):
        try:
            row = self._conn.execute(
                "SELECT weights_json, total_updates, reward_baseline "
                "FROM policy_state WHERE id=0").fetchone()
            if not row or not row[0]:
                return None
            return (json.loads(row[0]), int(row[1] or 0), float(row[2] or 0.0))
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] load_policy_flat soft-fail: %s", e)
            return None

    def save_policy_flat(self, flat_list, total_updates, reward_baseline) -> None:
        try:
            self._conn.execute(
                "INSERT INTO policy_state (id, weights_json, total_updates, "
                "reward_baseline, ts) VALUES (0,?,?,?,?) ON CONFLICT (id) DO UPDATE "
                "SET weights_json=excluded.weights_json, "
                "total_updates=excluded.total_updates, "
                "reward_baseline=excluded.reward_baseline, ts=excluded.ts",
                [json.dumps(flat_list), int(total_updates), float(reward_baseline),
                 time.time()])
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] save_policy_flat soft-fail: %s", e)

    # -- IQL nets persistence (P2 — worker-only, separate from the π SHM flat) --
    def save_iql_flat(self, iql_flat_list, total_iql_updates) -> None:
        try:
            self._conn.execute(
                "INSERT INTO policy_iql_state (id, iql_json, total_iql_updates, ts) "
                "VALUES (0,?,?,?) ON CONFLICT (id) DO UPDATE "
                "SET iql_json=excluded.iql_json, "
                "total_iql_updates=excluded.total_iql_updates, ts=excluded.ts",
                [json.dumps(iql_flat_list), int(total_iql_updates), time.time()])
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] save_iql_flat soft-fail: %s", e)

    def load_iql_flat(self):
        try:
            row = self._conn.execute(
                "SELECT iql_json FROM policy_iql_state WHERE id=0").fetchone()
            if not row or not row[0]:
                return None
            return json.loads(row[0])
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] load_iql_flat soft-fail: %s", e)
            return None

    def iql_transitions(self, n: int):
        """Most-recent reward tuples WITH goal_class + ts, oldest→newest, for
        per-goal_class trajectory linking (P2 full-IQL next_state). Returns
        [(features, action, reward, goal_class, ts), …]."""
        try:
            rows = self._conn.execute(
                "SELECT features_json, action, reward, goal_class, ts FROM ("
                "  SELECT features_json, action, reward, goal_class, ts, id"
                "  FROM reward_tuples ORDER BY id DESC LIMIT ?"
                ") ORDER BY ts ASC, id ASC", [int(n)]).fetchall()
            return [(json.loads(r[0]), int(r[1]), float(r[2]),
                     str(r[3] or ""), float(r[4] or 0.0)) for r in rows]
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] iql_transitions soft-fail: %s", e)
            return []

    # -- MasteryLevel signals + persistence (P3) ------------------------
    def success_rate(self, window: int) -> float:
        """Scale-free verified-success rate (SOAR ① / INV-ML-3): fraction of the
        last `window` decisions with reward>0. Dimensionless ∈ [0,1] — cannot be
        fooled by reward-scale inflation. 0.0 on an empty buffer."""
        try:
            row = self._conn.execute(
                "SELECT AVG(CASE WHEN reward > 0 THEN 1.0 ELSE 0.0 END) FROM ("
                "  SELECT reward FROM reward_tuples ORDER BY id DESC LIMIT ?)",
                [int(window)]).fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] success_rate soft-fail: %s", e)
            return 0.0

    def chunk_count(self) -> int:
        """Number of distilled macros = chunked (mastered) routines (SOAR ②).
        Secondary structural-competence signal (INV-ML-5)."""
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM macro_emitted").fetchone()
            return int(row[0]) if row and row[0] else 0
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] chunk_count soft-fail: %s", e)
            return 0

    def is_graduated(self, goal_class: str) -> bool:
        """Has ANY action for this goal_class been chunked? → it's a mastered
        (graduated) class (the SOAR ② graduation map; P4/P5 read this)."""
        try:
            row = self._conn.execute(
                "SELECT 1 FROM macro_emitted WHERE goal_class=? LIMIT 1",
                [str(goal_class or "")]).fetchone()
            return bool(row)
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] is_graduated soft-fail: %s", e)
            return False

    def frontier_goal_classes(self, limit: int = 16):
        """Goal_classes SEEN in reward_tuples but NOT yet chunked — the un-mastered
        frontier P4/P5 steer exploration/teacher toward (SOAR ② graduation)."""
        try:
            rows = self._conn.execute(
                "SELECT DISTINCT r.goal_class FROM reward_tuples r "
                "WHERE r.goal_class <> '' AND r.goal_class NOT IN "
                "(SELECT goal_class FROM macro_emitted) LIMIT ?",
                [int(limit)]).fetchall()
            return [str(x[0]) for x in rows]
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] frontier_goal_classes soft-fail: %s", e)
            return []

    def reset_policy_artifacts(self) -> dict:
        """P7 — therapeutic clean-baseline reset (§24.7 precedent). Clears ALL
        durable routing-policy state so the next boot is a true cold-start on the
        IQL loop: the π weights (policy_state), the IQL nets (policy_iql_state),
        the MasteryLevel (mastery_level_state), AND the experience-replay buffer
        (reward_tuples — the collapsed-era samples would re-collapse a fresh
        policy). Macros (macro_emitted) are KEPT — they're verified mastered
        routines, not collapsed scaffolding. Returns the cleared counts. Only the
        relearnable RL scaffolding is cleared (INV-OML-1); memory/timechain are
        untouched (directive_memory_preservation)."""
        counts = {}
        for tbl in ("policy_state", "policy_iql_state", "mastery_level_state",
                    "reward_tuples"):
            try:
                n = self._conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
                self._conn.execute(f"DELETE FROM {tbl}")
                counts[tbl] = int(n[0]) if n and n[0] else 0
            except Exception as e:  # noqa: BLE001
                logger.debug("[self_learning] reset %s soft-fail: %s", tbl, e)
                counts[tbl] = -1
        return counts

    def save_mastery_state(self, state_dict) -> None:
        try:
            self._conn.execute(
                "INSERT INTO mastery_level_state (id, state_json, ts) "
                "VALUES (0,?,?) ON CONFLICT (id) DO UPDATE "
                "SET state_json=excluded.state_json, ts=excluded.ts",
                [json.dumps(state_dict), time.time()])
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] save_mastery_state soft-fail: %s", e)

    def load_mastery_state(self):
        try:
            row = self._conn.execute(
                "SELECT state_json FROM mastery_level_state WHERE id=0").fetchone()
            return json.loads(row[0]) if row and row[0] else None
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] load_mastery_state soft-fail: %s", e)
            return None

    # -- pending decisions (the async-join stash) -----------------------
    def stash_decision(self, *, tx, features, action, goal_class, turn_id) -> None:
        try:
            self._conn.execute(
                "INSERT INTO pending_decisions (parent_tool_call_tx, features_json, "
                "action, goal_class, turn_id, ts) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT (parent_tool_call_tx) DO UPDATE SET "
                "features_json=excluded.features_json, action=excluded.action, "
                "goal_class=excluded.goal_class, turn_id=excluded.turn_id, ts=excluded.ts, "
                "applied_reward=NULL, applied_source=NULL",
                [str(tx), json.dumps(list(features)), int(action),
                 str(goal_class or ""), str(turn_id or ""), time.time()])
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] stash_decision soft-fail: %s", e)

    # -- Phase B (§7.B): peek (no delete) + mark, so a 2nd higher-authority reward
    #    (user/Maker after the judge) can apply a corrective delta vs the prior.
    def peek_decision(self, tx):
        try:
            row = self._conn.execute(
                "SELECT features_json, action, goal_class, applied_reward, applied_source "
                "FROM pending_decisions WHERE parent_tool_call_tx=?", [str(tx)]).fetchone()
            if not row:
                return None
            return (json.loads(row[0]), int(row[1]), str(row[2] or ""),
                    (None if row[3] is None else float(row[3])),
                    (None if row[4] is None else str(row[4])))
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] peek_decision soft-fail: %s", e)
            return None

    def mark_rewarded(self, tx, applied_reward, applied_source) -> None:
        try:
            self._conn.execute(
                "UPDATE pending_decisions SET applied_reward=?, applied_source=?, ts=? "
                "WHERE parent_tool_call_tx=?",
                [float(applied_reward), str(applied_source), time.time(), str(tx)])
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] mark_rewarded soft-fail: %s", e)

    def pop_decision(self, tx):
        try:
            row = self._conn.execute(
                "SELECT features_json, action, goal_class FROM pending_decisions "
                "WHERE parent_tool_call_tx=?", [str(tx)]).fetchone()
            if not row:
                return None
            self._conn.execute(
                "DELETE FROM pending_decisions WHERE parent_tool_call_tx=?", [str(tx)])
            return (json.loads(row[0]), int(row[1]), str(row[2] or ""))
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] pop_decision soft-fail: %s", e)
            return None

    def prune_pending(self, ttl_s: float) -> int:
        try:
            cutoff = time.time() - float(ttl_s)
            n = self._conn.execute(
                "SELECT COUNT(*) FROM pending_decisions WHERE ts < ?", [cutoff]).fetchone()
            self._conn.execute("DELETE FROM pending_decisions WHERE ts < ?", [cutoff])
            return int(n[0]) if n else 0
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] prune_pending soft-fail: %s", e)
            return 0

    # -- reward tuples + macro accounting -------------------------------
    def record_reward_tuple(self, *, features, action, reward, goal_class) -> None:
        try:
            self._conn.execute(
                "INSERT INTO reward_tuples (features_json, action, reward, goal_class, ts) "
                "VALUES (?,?,?,?,?)",
                [json.dumps(list(features)), int(action), float(reward),
                 str(goal_class or ""), time.time()])
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] record_reward_tuple soft-fail: %s", e)

    def clear_reward_tuples(self) -> int:
        """Wipe the experience-replay history (§24.7). Called on a THERAPEUTIC
        cold-start (pathological self-heal or a schema-version discard): the old
        tuples encode the collapsed/old-schema behavior, and replaying them would
        re-collapse the fresh policy. Returns the count cleared. Soft-fail."""
        try:
            n = self._conn.execute("SELECT COUNT(*) FROM reward_tuples").fetchone()
            self._conn.execute("DELETE FROM reward_tuples")
            return int(n[0]) if n and n[0] else 0
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] clear_reward_tuples soft-fail: %s", e)
            return 0

    def recent_reward_tuples(self, n: int):
        try:
            rows = self._conn.execute(
                "SELECT features_json, action, reward FROM reward_tuples "
                "ORDER BY id DESC LIMIT ?", [int(n)]).fetchall()
            return [(json.loads(r[0]), int(r[1]), float(r[2])) for r in rows]
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] recent_reward_tuples soft-fail: %s", e)
            return []

    def balanced_reward_tuples(self, n: int):
        """Replay batch BALANCED across actions (deadlock fix, tuning step 1).

        `recent_reward_tuples` is strict time-order → dominated by the dense
        `tool` stream, so replay just re-reinforces the always-tool collapse. This
        draws the most-recent tuples PER ACTION (round-robin), so the minority
        direct/research/IDK samples train at parity with `tool`. Combined with the
        per-action baseline, that lets a positive non-tool reward actually move
        its action. Falls back to the most-recent within each action's slice."""
        try:
            per_action = max(1, int(n) // max(1, NUM_OUTER_ACTIONS) + 1)
            rows = self._conn.execute(
                "SELECT features_json, action, reward FROM ("
                "  SELECT features_json, action, reward, id,"
                "    ROW_NUMBER() OVER (PARTITION BY action ORDER BY id DESC) AS rn"
                "  FROM reward_tuples"
                ") WHERE rn <= ? ORDER BY rn ASC, id DESC LIMIT ?",
                [int(per_action), int(n)]).fetchall()
            return [(json.loads(r[0]), int(r[1]), float(r[2])) for r in rows]
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] balanced_reward_tuples soft-fail: %s", e)
            return []

    def distinct_recent_contexts(self, n: int):
        """Diverse recalled CONTEXTS (feature vec + goal_class) for active idle
        exploration (step 2) — the most-recent few per goal_class, round-robin,
        so exploration probes the whole context space (conversational/unknowable
        included) rather than just the dense `tool` region. Returns the FEATURE
        vector only (the action taken / its reward are irrelevant — exploration
        chooses a fresh action and scores it structurally)."""
        try:
            per_class = max(1, int(n) // 4 + 1)
            rows = self._conn.execute(
                "SELECT features_json, goal_class FROM ("
                "  SELECT features_json, goal_class, id,"
                "    ROW_NUMBER() OVER (PARTITION BY goal_class ORDER BY id DESC) AS rn"
                "  FROM reward_tuples"
                ") WHERE rn <= ? ORDER BY rn ASC, id DESC LIMIT ?",
                [int(per_class), int(n)]).fetchall()
            return [(json.loads(r[0]), str(r[1] or "")) for r in rows]
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] distinct_recent_contexts soft-fail: %s", e)
            return []

    def win_count(self, goal_class: str, action: int) -> int:
        """Verified wins (reward>0) of a (goal_class, action) — the macro trigger."""
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM reward_tuples WHERE goal_class=? AND action=? "
                "AND reward > 0", [str(goal_class or ""), int(action)]).fetchone()
            return int(row[0]) if row and row[0] else 0
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] win_count soft-fail: %s", e)
            return 0

    def macro_already_emitted(self, goal_class: str, action: int) -> bool:
        try:
            row = self._conn.execute(
                "SELECT 1 FROM macro_emitted WHERE goal_class=? AND action=?",
                [str(goal_class or ""), int(action)]).fetchone()
            return bool(row)
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] macro_already_emitted soft-fail: %s", e)
            return False

    def mark_macro_emitted(self, goal_class: str, action: int,
                           version: int = 1, wins_at_emit: int = 0) -> None:
        """§7.D D.4c — record the emit + its version + the win_count at emit-time
        (the refinement baseline). On a successor (version>1) UPDATE in place; the
        prior macro RECORD is never overwritten (it lives under its own ::v{n} id —
        mutate-not-update, INV-OML-5)."""
        try:
            self._conn.execute(
                "INSERT INTO macro_emitted (goal_class, action, ts, version, "
                "wins_at_emit) VALUES (?,?,?,?,?) "
                "ON CONFLICT (goal_class, action) DO UPDATE SET "
                "ts=excluded.ts, version=excluded.version, "
                "wins_at_emit=excluded.wins_at_emit",
                [str(goal_class or ""), int(action), time.time(),
                 int(version), int(wins_at_emit)])
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] mark_macro_emitted soft-fail: %s", e)

    def macro_version(self, goal_class: str, action: int) -> tuple:
        """Return (version, wins_at_emit) for an emitted (goal_class, action), or
        (0, 0) if never emitted."""
        try:
            row = self._conn.execute(
                "SELECT version, wins_at_emit FROM macro_emitted "
                "WHERE goal_class=? AND action=?",
                [str(goal_class or ""), int(action)]).fetchone()
            if not row:
                return (0, 0)
            return (int(row[0] or 1), int(row[1] or 0))
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] macro_version soft-fail: %s", e)
            return (0, 0)

    def refinement_candidates(self, min_growth: int, limit: int = 4):
        """§7.D D.4c — already-emitted (goal_class, action) whose verified wins have
        GROWN by ≥ `min_growth` since the last emit → a successor composite is worth
        distilling (the strategy got more evidence). Most-grown first."""
        try:
            rows = self._conn.execute(
                "SELECT goal_class, action, version, wins_at_emit, wins FROM ("
                "  SELECT me.goal_class AS goal_class, me.action AS action, "
                "    me.version AS version, me.wins_at_emit AS wins_at_emit, "
                "    (SELECT COUNT(*) FROM reward_tuples rt "
                "     WHERE rt.goal_class=me.goal_class AND rt.action=me.action "
                "     AND rt.reward>0) AS wins "
                "  FROM macro_emitted me"
                ") WHERE wins - wins_at_emit >= ? "
                "ORDER BY wins - wins_at_emit DESC LIMIT ?",
                [int(min_growth), int(limit)]).fetchall()
            return [(str(r[0]), int(r[1]), int(r[2] or 1), int(r[3] or 0)) for r in rows]
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] refinement_candidates soft-fail: %s", e)
            return []

    def candidate_macro_classes(self, min_wins: int, limit: int = 8):
        """Phase-C piece 6 — (goal_class, action) classes with ≥ `min_wins`
        verified wins (reward>0) that have NOT yet been macro-emitted, most-won
        first. The explore-tick deliberative path draws from here."""
        try:
            rows = self._conn.execute(
                "SELECT rt.goal_class, rt.action, COUNT(*) AS wins "
                "FROM reward_tuples rt "
                "WHERE rt.reward > 0 AND rt.goal_class <> '' "
                "GROUP BY rt.goal_class, rt.action "
                "HAVING COUNT(*) >= ? "
                "AND NOT EXISTS (SELECT 1 FROM macro_emitted me "
                "                WHERE me.goal_class = rt.goal_class "
                "                AND me.action = rt.action) "
                "ORDER BY wins DESC LIMIT ?",
                [int(min_wins), int(limit)]).fetchall()
            return [(str(r[0]), int(r[1])) for r in rows]
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] candidate_macro_classes soft-fail: %s", e)
            return []

    def related_emitted_macros(self, signature, *, exclude_goal_class,
                               exclude_action, floor, limit=4):
        """§7.D D.4b — ALREADY-emitted macros whose 30-D mean-feature signature is
        cosine ≥ `floor` to `signature` (excluding the target class). This is the
        D7-honest macro-of-macros reuse signal: the worker is numpy-only with no
        384-D embedder (so it cannot faiss-search the composite library), but it
        DOES own every emitted macro's 30-D signature (`mean_features`), so a
        cosine over THAT space is a real "operates in the same problem-region"
        signal — the parent composite genuinely builds on these verified macros.
        numpy dot/norm RELEASE the GIL (no heartbeat starvation; the macro count is
        small). Returns [(goal_class, action, cosine), ...] most-similar first.
        Soft → []."""
        try:
            sig = np.asarray(signature, dtype=np.float32)
            n_sig = float(np.linalg.norm(sig))
            if n_sig <= 0.0:
                return []
            rows = self._conn.execute(
                "SELECT goal_class, action FROM macro_emitted").fetchall()
            scored = []
            for r in rows:
                gc, act = str(r[0]), int(r[1])
                if gc == str(exclude_goal_class) and act == int(exclude_action):
                    continue
                v = self.mean_features(gc, act)
                if v is None:
                    continue
                vv = np.asarray(v, dtype=np.float32)
                nv = float(np.linalg.norm(vv))
                if nv <= 0.0:
                    continue
                cos = float(np.dot(sig, vv) / (n_sig * nv))
                if cos >= float(floor):
                    scored.append((gc, act, cos))
            scored.sort(key=lambda t: -t[2])
            return scored[:int(limit)]
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] related_emitted_macros soft-fail: %s", e)
            return []

    def mean_features(self, goal_class: str, action: int):
        """Mean winning feature vector for a (goal_class, action) — the macro signature."""
        try:
            rows = self._conn.execute(
                "SELECT features_json FROM reward_tuples WHERE goal_class=? AND "
                "action=? AND reward > 0", [str(goal_class or ""), int(action)]).fetchall()
            vecs = [json.loads(r[0]) for r in rows if r and r[0]]
            vecs = [v for v in vecs if len(v) == OUTER_POLICY_INPUT_DIM]
            if not vecs:
                return None
            n = len(vecs)
            return [sum(v[i] for v in vecs) / n for i in range(OUTER_POLICY_INPUT_DIM)]
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] mean_features soft-fail: %s", e)
            return None

    def log_explore(self, kind: str, goal_class: str, detail: str) -> None:
        try:
            self._conn.execute(
                "INSERT INTO explore_log (kind, goal_class, detail, ts) VALUES (?,?,?,?)",
                [str(kind), str(goal_class or ""), str(detail or ""), time.time()])
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] log_explore soft-fail: %s", e)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


def _cfg(config: dict) -> dict:
    sub = (get_params("synthesis") or {}).get("self_learning", {}) or {}
    out = dict(_DEFAULTS)
    out.update({k: v for k, v in sub.items() if k in _DEFAULTS})
    # Caller-supplied overrides take precedence. Called ONCE at boot
    # (self_learning_worker_main:837) with `full_config`, whose
    # [synthesis][self_learning] == the get_params boot snapshot → a no-op in
    # production (no SHM-liveness regression). In tests it applies injected flags
    # (e.g. oml_iql_enabled=False to exercise the legacy REINFORCE path). Without
    # this the `config` arg was silently ignored — a latent regression that left
    # the legacy-path tests asserting against the IQL-default-on behavior.
    ov = ((config or {}).get("synthesis", {}) or {}).get("self_learning", {}) or {}
    out.update({k: v for k, v in ov.items() if k in _DEFAULTS})
    return out


def _publish_weights(writer, policy) -> None:
    if writer is None:
        return
    try:
        writer.write(policy.to_flat())
    except Exception as e:  # noqa: BLE001
        logger.debug("[self_learning] SHM publish soft-fail: %s", e)


@with_error_envelope(module_name="self_learning", subsystem="entry",
                     severity=_sev.FATAL)
def self_learning_worker_main(recv_queue, send_queue, name: str,
                              config: dict) -> None:
    """Main loop for the self-learning policy-owner subprocess."""
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    full_config = config or {}
    cfg = _cfg(full_config)

    _state_writer = None
    try:
        from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
        _state_writer = ModuleStateWriter(
            module_name=name, layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT)
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning("[self_learning] ModuleStateWriter init failed: %s", _sw_err)

    from titan_hcl.core.state_registry import (
        StateRegistryWriter, ensure_shm_root, resolve_titan_id)
    titan_id = ((get_params("info_banner") or {}).get("titan_id")
                or resolve_titan_id())
    logger.info("[self_learning] Booting — titan_id=%s enabled=%s", titan_id,
                cfg["enabled"])

    try:
        store = _SelfLearningStore()
    except Exception as e:
        logger.error("[self_learning] store init failed: %s — exiting", e)
        return

    # P7 — one-shot clean-baseline reset (uncollapse). If the sentinel exists,
    # CLEAR the collapsed policy + IQL nets + level + replay buffer BEFORE loading
    # → a guaranteed cold-start relearn on the IQL loop. Consume (delete) the
    # sentinel so it never loops. (§24.7 therapeutic-clear precedent.)
    try:
        _reset_sentinel = str(cfg.get("reset_sentinel_path",
                                      "data/.reset_routing_policy"))
        if _reset_sentinel and os.path.exists(_reset_sentinel):
            _cleared = store.reset_policy_artifacts()
            try:
                os.remove(_reset_sentinel)
            except Exception:  # noqa: BLE001
                pass
            logger.warning("[self_learning] 🔄 P7 CLEAN-BASELINE RESET (uncollapse) "
                           "— cleared %s; cold-start relearn on the IQL loop",
                           _cleared)
    except Exception as e:  # noqa: BLE001
        logger.warning("[self_learning] reset-sentinel check failed: %s", e)

    # Policy — restore from our own store, else fresh (cold-start).
    _wd = float(cfg["weight_decay"])
    _mwn = float(cfg["max_weight_norm"])
    policy = OuterMetaPolicy(lr=float(cfg["policy_lr"]), weight_decay=_wd, max_weight_norm=_mwn)
    _cold_start = True   # True unless a healthy persisted policy is restored below (Fix 2 §24.9)
    loaded = store.load_policy_flat()
    if loaded is not None:
        try:
            import numpy as _np
            restored = OuterMetaPolicy.from_flat(_np.asarray(loaded[0], dtype=_np.float32))
            restored.lr = float(cfg["policy_lr"])
            restored.weight_decay = _wd
            restored.max_weight_norm = _mwn
            # Self-heal a persisted RUNAWAY policy (the pre-2026-06-11 unregularized
            # collapse: `tool` weights exploded → always-tool). A pathological
            # restore is discarded for a fresh cold-start — the regularized
            # train_step then re-learns without re-exploding.
            if restored.is_pathological():
                logger.warning("[self_learning] RESTORED POLICY IS PATHOLOGICAL "
                               "(runaway weights, updates=%d) — re-initializing "
                               "(regularized cold-start)", restored.total_updates)
                policy = OuterMetaPolicy(lr=float(cfg["policy_lr"]),
                                         weight_decay=_wd, max_weight_norm=_mwn)
                # Therapeutic cold-start ⇒ the reward-tuple history encodes the
                # COLLAPSED behavior (the always-tool era) — replaying it would
                # re-collapse the fresh policy (live-verified 2026-06-12). Clear it
                # so the cold-start is a truly clean baseline (§24.7).
                _n = store.clear_reward_tuples()
                logger.warning("[self_learning] cleared %d collapsed-era reward tuples "
                               "(clean cold-start baseline)", _n)
            else:
                policy = restored
                _cold_start = False   # healthy restore — keep learned routing, do NOT re-seed
                logger.info("[self_learning] policy restored (updates=%d, baseline=%.3f)",
                            policy.total_updates, policy.reward_baseline)
        except Exception as e:  # noqa: BLE001
            # Restore failed (e.g. a schema-version flat-dim change discarded the
            # persisted policy) ⇒ cold-start. The reward-tuple history belongs to the
            # OLD policy/schema (and likely the collapsed era) — clear it for a clean
            # baseline (§24.7); the regularized + structurally-explored policy re-learns.
            logger.warning("[self_learning] policy restore failed (cold-start): %s", e)
            try:
                _n = store.clear_reward_tuples()
                logger.warning("[self_learning] cleared %d stale reward tuples "
                               "(clean cold-start baseline)", _n)
            except Exception:  # noqa: BLE001
                pass

    # Fix 2 (§24.9) — cold-start feature-discriminating seed. ONLY on a genuine
    # cold-start (fresh / pathological re-init / restore-fail), BEFORE the SHM
    # publish so the seeded weights are what the agno DECIDE path reads from boot.
    # A restored policy already carries learned routing → never re-seed it.
    if _cold_start and bool(cfg.get("cold_start_seed_enabled", True)):
        try:
            _seeded = _seed_cold_start(policy, cfg)
            if _seeded:
                logger.info("[self_learning] cold-start seed: %d structural steps "
                            "→ feature-conditional warm-start (§24.9)", _seeded)
                store.save_policy_flat(policy.to_flat().tolist(),
                                       policy.total_updates, policy.reward_baseline)
        except Exception as e:  # noqa: BLE001
            logger.warning("[self_learning] cold-start seed failed (continuing "
                           "unseeded): %s", e)

    # P2 — restore the IQL nets (V/Q/Q-target), separate from the π flat. A
    # cold-start (fresh / pathological / restore-fail) gets a FRESH init: the old
    # V/Q encode collapsed-era value, same reasoning as clearing reward tuples.
    if bool(cfg.get("oml_iql_enabled", True)):
        try:
            import numpy as _np
            _iql_loaded = store.load_iql_flat() if not _cold_start else None
            if _iql_loaded is not None and policy.iql_from_flat(
                    _np.asarray(_iql_loaded, dtype=_np.float32)):
                logger.info("[self_learning] IQL nets restored (iql_updates=%d)",
                            policy.total_iql_updates)
            else:
                policy.init_iql()
                logger.info("[self_learning] IQL nets fresh-init (cold-start=%s)",
                            _cold_start)
        except Exception as e:  # noqa: BLE001
            logger.warning("[self_learning] IQL init/restore failed (fresh): %s", e)
            try:
                policy.init_iql()
            except Exception:  # noqa: BLE001
                pass

    # SHM weight publisher (single writer — INV-OML-8 / G21).
    _shm_writer = None
    try:
        _shm_writer = StateRegistryWriter(
            OUTER_META_POLICY_STATE_SPEC, ensure_shm_root(titan_id))
        _publish_weights(_shm_writer, policy)
    except Exception as e:  # noqa: BLE001
        logger.warning("[self_learning] SHM writer init failed: %s", e)

    # P3 — MasteryLevel (emergent ability level from V(s)) + its SHM slot.
    # Only when IQL is on (the level reads the IQL value); else None (no level).
    _mastery = None
    _level_writer = None
    if bool(cfg.get("oml_iql_enabled", True)):
        try:
            _mastery = MasteryLevel(
                n_grades=int(cfg["level_n_grades"]),
                grade_lo=float(cfg["level_grade_lo"]),
                grade_hi=float(cfg["level_grade_hi"]),
                ema_alpha=float(cfg["level_ema_alpha"]),
                competence_floor_base=float(cfg["level_competence_floor_base"]),
                competence_floor_slope=float(cfg["level_competence_floor_slope"]),
                competence_ema_alpha=float(cfg["competence_ema_alpha"]))
            # A cold-start policy relearns the level too; only restore on a healthy
            # policy restore (same reasoning as the IQL nets).
            _ms = store.load_mastery_state() if not _cold_start else None
            if _ms is not None and _mastery.load_dict(_ms):
                logger.info("[self_learning] MasteryLevel restored (grade=%d)",
                            _mastery.readout()["grade"])
            _level_writer = StateRegistryWriter(
                MASTERY_LEVEL_STATE_SPEC, ensure_shm_root(titan_id))
        except Exception as e:  # noqa: BLE001
            logger.warning("[self_learning] MasteryLevel init failed: %s", e)
            _mastery = None

    # Metabolic gate reader (soft — cold default permits, but survival blocks).
    try:
        from titan_hcl.proxies.life_force_proxy import LifeForceShmReader
        _life = LifeForceShmReader()
    except Exception:  # noqa: BLE001
        _life = None

    # ── Phase C piece 6 — the OUTER two-level reasoner (numpy-only; INV-OML-8) ──
    # OuterMetaReasoningEngine (reuses the meta handlers via PrimitiveHandlersMixin)
    # over OuterReasoningEngine (reuses reasoning.py's primitive funcs). The
    # explore tick runs them on verified-win classes. EVALUATE ← an in-worker
    # win-verification oracle (reads only this worker's own duckdb — no
    # coding_sandbox import, isolation preserved).
    _outer_reason = None
    _outer_meta = None
    if cfg["enabled"] and cfg["outer_meta_enabled"]:
        try:
            _outer_dir = os.path.join("data", "self_learning_outer")
            _outer_reason = OuterReasoningEngine(save_dir=_outer_dir)
            _outer_meta = OuterMetaReasoningEngine(
                config={"max_steps": int(cfg["outer_meta_max_steps"])},
                save_dir=_outer_dir)

            def _win_oracle(problem, _delegate_results, _store=store):
                gc = str(problem.get("goal_class", "") or "")
                act = int(problem.get("action", 0))
                return 1.0 if _store.win_count(gc, act) > 0 else -1.0

            _outer_meta.set_oracle(_win_oracle)
            logger.info("[self_learning] OuterMetaReasoningEngine wired "
                        "(deliberative macro path; in-worker win-oracle)")
        except Exception as e:  # noqa: BLE001
            logger.warning("[self_learning] outer reasoner init failed "
                           "(deliberative path disabled): %s", e)
            _outer_reason = None
            _outer_meta = None

    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception:  # noqa: BLE001
            pass

    last_heartbeat = 0.0
    last_explore = time.time()
    last_prune = time.time()
    processed = 0
    trained = 0
    errors = 0

    while True:
        now = time.time()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            try:
                send_queue.put({
                    "type": bus.MODULE_HEARTBEAT, "src": name, "dst": "guardian",
                    "payload": {"alive": True, "ts": now, "processed": processed,
                                "trained": trained, "errors": errors}, "ts": now})
            except Exception:  # noqa: BLE001
                pass
            if _state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
                try:
                    _state_writer.heartbeat()
                except Exception:  # noqa: BLE001
                    pass
            last_heartbeat = now

        # Periodic prune of un-joined decisions (top of loop, NOT in except Empty).
        if now - last_prune >= 300.0:
            store.prune_pending(cfg["pending_ttl_s"])
            last_prune = now

        # Idle EXPLORE tick — metabolically gated (INV-OML-9), top of loop.
        if now - last_explore >= float(cfg["explore_interval_s"]):
            try:
                _explore_tick(cfg, store, policy, _shm_writer, _life, send_queue, name,
                              _outer_reason, _outer_meta,
                              mastery=_mastery, level_writer=_level_writer)
            except Exception as e:  # noqa: BLE001
                logger.debug("[self_learning] explore tick soft-fail: %s", e)
            last_explore = now

        try:
            msg = recv_queue.get(timeout=0.5)
        except Empty:
            continue
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(msg, dict):
            continue

        msg_type = msg.get("type")

        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:  # noqa: BLE001
            pass

        if msg_type == "MODULE_PROBE_REQUEST":
            try:
                from titan_hcl.core.probe_dispatcher import handle_module_probe_request
                handle_module_probe_request(
                    msg, send_queue=send_queue, module_name=name,
                    state_writer=_state_writer, probe_fn=None)
            except Exception as _pe:  # noqa: BLE001
                logger.warning("[self_learning] PROBE handler failed: %s", _pe)
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[self_learning] Shutdown — saving policy (updates=%d)",
                        policy.total_updates)
            try:
                store.save_policy_flat(policy.to_flat().tolist(),
                                       policy.total_updates, policy.reward_baseline)
            except Exception:  # noqa: BLE001
                pass
            store.close()
            return

        payload = msg.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        if msg_type == SELF_LEARN_DECISION:
            try:
                store.stash_decision(
                    tx=payload.get("parent_tool_call_tx"),
                    features=payload.get("features") or [],
                    action=int(payload.get("action", 0)),
                    goal_class=payload.get("goal_class"),
                    turn_id=payload.get("turn_id"))
                processed += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                logger.debug("[self_learning] decision stash failed: %s", e)
            continue

        if msg_type == SELF_LEARN_REWARD:
            try:
                # P5 (c) — the in-process emergent level (self_learning is the sole
                # MasteryLevel writer). None/0 when not yet published ⇒ static mix.
                _lvl_norm = 0.0
                if _mastery is not None:
                    try:
                        _lvl_norm = max(0.0, min(1.0, float(
                            _mastery.readout().get("level", 0.0))
                            / float(max(1, int(cfg["level_n_grades"])))))
                    except Exception:  # noqa: BLE001
                        _lvl_norm = 0.0
                if _handle_reward(payload, store, policy, _shm_writer, cfg,
                                  send_queue, name, level_norm=_lvl_norm):
                    trained += 1
                processed += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                logger.warning("[self_learning] reward join failed: %s", e)
            continue


def _handle_reward(payload, store, policy, shm_writer, cfg, send_queue, name,
                   level_norm: float = 0.0) -> bool:
    """One policy update from a verdict-time OR genuinely-async reward.

    v1.1 (Phase 1, INV-OML-12): the synthesis-side C1 capture emits
    `(features, action, reward, goal_class)` DIRECTLY at verdict-time (verifiable
    tool lane) — decision and outcome travel together, no join.

    Phase B (§7.B): non-verifiable turns reward LATER (LLM turn-judge / user /
    Maker), keyed on `tx`(=reasoning_id) → join the stashed decision. The reward
    is scaled by its SOURCE WEIGHT (Maker > user → bigger delta), and a higher-
    authority second reward (user/Maker after the judge) applies a CORRECTIVE
    DELTA over the prior — a refinement of the same turn, never a double-train."""
    features = payload.get("features")
    action = payload.get("action")
    goal_class = str(payload.get("goal_class", "") or "")
    source = str(payload.get("source", "") or "")
    record_tuple = True
    macro_reward = 0.0
    if features is not None and action is not None:
        # v1.1 direct path — verifiable lane, decision + outcome together.
        if len(features) != OUTER_POLICY_INPUT_DIM:
            return False
        action = int(action)
        train_reward = float(payload.get("reward", 0.0))
        macro_reward = train_reward
    else:
        # Phase B async path — join a stashed decision by reasoning_id (tx).
        tx = payload.get("parent_tool_call_tx")
        if not tx:
            return False
        decision = store.peek_decision(tx)
        if decision is None:
            return False  # no matching decision (pruned / not ours) — not an error
        features, action, goal_class, applied_reward, applied_source = decision
        if len(features) != OUTER_POLICY_INPUT_DIM:
            return False
        src = source or "llm_judge"
        # P5 (c) — the level-graduated source weight (EMERGENT, level_norm-driven;
        # flag-off / unpublished level ⇒ the static Phase-B base, byte-identical).
        _base_w = float(cfg.get(f"{src}_reward_weight", 1.0))
        effective = float(payload.get("reward", 0.0)) * _graduated_source_weight(
            src, _base_w, level_norm, cfg)
        if applied_source is None:
            train_reward = effective                 # first reward for this turn
            macro_reward = effective
        elif (_REWARD_SOURCE_RANK.get(src, 0)
              > _REWARD_SOURCE_RANK.get(applied_source, 0)):
            train_reward = effective - float(applied_reward or 0.0)  # higher-auth correction
            record_tuple = False                     # refine the policy, not a new sample
        else:
            return False                             # same-or-lower authority → ignore
        store.mark_rewarded(tx, effective, src)
    # P2 (RFP_emergent_mastery_curriculum) — FULL-IQL pure-offline path. When on,
    # the per-turn REINFORCE update is SKIPPED: the turn only BUFFERS the sample;
    # the V/Q/π learning is the offline train_iql pass in the idle _explore_tick
    # (mirrors CGN consolidate). The π SHM flat is unchanged here, so no republish.
    # Flag-off ⇒ byte-identical legacy REINFORCE+publish path (INV-MC-7).
    _iql_on = bool(cfg.get("oml_iql_enabled", True))
    _sample_reward = train_reward
    if _iql_on:
        # The async-correction branch set record_tuple=False to avoid a REINFORCE
        # double-train; under IQL we DO want the corrected sample, labelled with
        # the ABSOLUTE corrected reward (`effective`), not the REINFORCE delta.
        if not record_tuple:
            _sample_reward = effective
            record_tuple = True
    else:
        policy.learn(features, action, train_reward,
                     baseline_alpha=float(cfg["baseline_alpha"]))
        store.save_policy_flat(policy.to_flat().tolist(),
                               policy.total_updates, policy.reward_baseline)
        _publish_weights(shm_writer, policy)
    if record_tuple:
        store.record_reward_tuple(
            features=features, action=action,
            reward=(_sample_reward if _iql_on else train_reward),
            goal_class=goal_class)
    if _iql_on:
        logger.info("[self_learning] buffered (IQL): action=%s reward=%+.2f src=%s "
                    "goal=%s (offline train in explore tick)",
                    action_index_to_name(action), _sample_reward,
                    source or "direct", goal_class or "-")
    else:
        logger.info("[self_learning] trained: action=%s reward=%+.2f src=%s goal=%s "
                    "updates=%d baseline=%.3f", action_index_to_name(action),
                    train_reward, source or "direct", goal_class or "-",
                    policy.total_updates, policy.reward_baseline)
    # Macro distillation (S1) — a (goal_class, action) with enough verified wins.
    # Phase-C piece 6: when the deliberative explore-tick path is enabled it is
    # the SOLE macro source (it would otherwise be pre-empted by this reactive
    # path firing the instant the threshold is crossed); the reactive path is
    # the FALLBACK when `outer_meta_enabled=False`.
    if macro_reward > 0 and goal_class and not cfg["outer_meta_enabled"]:
        _maybe_distill_macro(goal_class, action, store, cfg, send_queue, name)
    return True


def _emit_macro(goal_class, action, store, send_queue, name,
                composed_from=None, version=1) -> None:
    """Emit the `SELF_LEARN_MACRO_READY` S2 contract + mark emitted. Shared by the
    Phase-1 reactive path (`_maybe_distill_macro`), the Phase-C deliberative
    explore-tick path (`_outer_deliberate`), and the §7.D D.4c refinement path.

    `version` (D.4c, mutate-not-update / INV-OML-5): v1 keeps the canonical label
    `macro::{gc}::{action}` (back-compat); a successor (v>1) is a NEW record
    `macro::{gc}::{action}::v{n}` whose `composed_from` carries the parent label as
    lineage — the prior macro is never overwritten. `composed_from` extra entries
    (D.4b child composites) are merged in. The synthesis-side handler joins the
    verified tool_use leaves when composed_from is empty (D.1)."""
    win_count = store.win_count(goal_class, action)
    signature = store.mean_features(goal_class, action)
    if signature is None:
        return
    action_name = action_index_to_name(action)
    base_label = f"macro::{goal_class}::{action_name}"
    label = base_label if int(version) <= 1 else f"{base_label}::v{int(version)}"
    lineage = list(composed_from or [])
    if int(version) > 1 and base_label not in lineage:
        lineage.append(base_label)        # successor cites its predecessor (lineage)
    payload = {
        "goal_class": goal_class,
        "action": int(action),
        "action_name": action_name,
        "signature": signature,
        "b_i": 1.0, "c": 1.0,
        "time_cost": 1.0,                 # oracle-verified → proficient (B1)
        "use_count": win_count,
        "verified": True,                  # only verified classes reach here
        "label": label,
        "version": int(version),
    }
    if lineage:
        payload["composed_from"] = lineage
    try:
        send_queue.put({
            "type": SELF_LEARN_MACRO_READY, "src": name, "dst": "synthesis",
            "ts": time.time(), "payload": payload})
        store.mark_macro_emitted(goal_class, action, version=int(version),
                                 wins_at_emit=win_count)
        store.log_explore("macro_emitted", goal_class,
                          f"{action_name} v{int(version)} wins={win_count}")
        logger.info("[self_learning] macro-strategy distilled: %s → %s v%d (wins=%d)",
                    goal_class, action_name, int(version), win_count)
    except Exception as e:  # noqa: BLE001
        logger.debug("[self_learning] macro emit soft-fail: %s", e)


def _research_affordable(life, cfg) -> bool:
    """Can Titan afford a research request right now? (research costs an oracle/web
    request — FC-6). Below the chi floor the honest action is IDK, not research.
    No metabolic reader → assume affordable (cold default permits)."""
    if life is None:
        return True
    try:
        return float(life.get_chi_total()) >= float(cfg["explore_research_chi_floor"])
    except Exception:  # noqa: BLE001
        return True


# ── Fix 2 (§24.9) — cold-start feature-discriminating seed ──────────────────
# The lane archetypes: only the DISCRIMINATING base features (the ones
# `structural_target_action` reads — recall@1, skill_util@3, skill_matched@4,
# requires_tool@6, has_code@7) are pinned; everything else is jittered per
# sample so the net keys on the base features, not a memorized vector. We seed
# the 4 feature-DISCRIMINABLE lanes; IDK is NOT seeded — it differs from research
# only by metabolic affordability (NOT a feature), so seeding both on the same
# low-recall vec would teach two actions for one feature-region. IDK stays a
# downstream metabolic conversion (MODE_SHADOW) + the idle pass under genuine
# starvation (structural_target → IDK when not affordable).
#   (lane label, {base_feature_index: pinned_value}, affordable)
_COLD_START_LANES: tuple = (
    ("tool",           {6: 1.0, 7: 1.0}, True),   # computational shape → verify deterministically
    ("direct",         {1: 0.85},        True),   # dereferenceable knowledge present → he KNOWS
    ("research",       {1: 0.05},        True),   # does not know, affordable → find out
    ("skill_delegate", {4: 1.0, 3: 0.8}, True),   # proficient learned skill matches → reuse
)


def _synth_feature_vec(rng, overrides: dict) -> np.ndarray:
    """A jittered synthetic 30-D OuterFeatures vector for cold-start seeding.

    The non-discriminating features are randomized so the net learns to KEY ON
    the pinned base features rather than memorizing one vector: the base block is
    low noise [0,0.2] (kept below every structural threshold so an un-pinned base
    feature never spuriously trips the ladder), the MSL block is full-range
    [-1,1] noise (decorrelating it — w1's HE-random MSL rows otherwise inject
    live noise into the score), the retrieval prior is [0,0.3] noise. `bias`=1.0;
    the `overrides` then pin this vector squarely into one structural lane."""
    n_base = len(_BASE_FEATURE_NAMES)
    vec = np.empty(OUTER_POLICY_INPUT_DIM, dtype=np.float32)
    vec[:n_base] = rng.uniform(0.0, 0.2, size=n_base).astype(np.float32)
    vec[0] = 1.0  # bias
    vec[n_base:n_base + MSL_CONTEXT_DIM] = rng.uniform(
        -1.0, 1.0, size=MSL_CONTEXT_DIM).astype(np.float32)
    vec[n_base + MSL_CONTEXT_DIM:] = rng.uniform(
        0.0, 0.3, size=OUTER_POLICY_INPUT_DIM - n_base - MSL_CONTEXT_DIM).astype(np.float32)
    for idx, val in overrides.items():
        vec[idx] = float(val)
    return vec


def _seed_cold_start(policy, cfg) -> int:
    """Fix 2 (§24.9) — warm-start a cold policy toward feature-conditional routing
    BEFORE the first live turn, from the verifiable structural oracle.

    A fresh policy's argmax is ~uniform; the first dense live reward stream (the
    turn-judge, which rewards response QUALITY of a single action regardless of
    features) collapses it to that action before it learns to DISCRIMINATE on
    features (the routing-collapse failure mode, quantified live across T1/T2/T3
    2026-06-12). We teach the SAME verifiable structural target the idle pass
    teaches (`structural_target_action`) on a synthetic balanced lane set via the
    cross-entropy `train_step` (the proven robust rule, 8/8 seeds — NOT
    Boltzmann, which decays to a degenerate single action). Deterministic (fixed
    rng) → identical warm-start every cold boot; bounded by the §24.2 weight
    regularization. Live verdicts + Lever-1's boosted idle pass refine beyond it
    (a prior, not a freeze). Returns the number of cross-entropy steps taken."""
    epochs = int(cfg.get("cold_start_seed_epochs", 300))
    adv = float(cfg.get("cold_start_seed_advantage", 3.0))
    if epochs <= 0 or adv <= 0:
        return 0
    know_thr = float(cfg["explore_know_threshold"])
    skill_floor = float(cfg["explore_skill_floor"])
    rng = np.random.default_rng(0)   # deterministic warm-start
    n = 0
    for _ in range(epochs):
        for _label, overrides, affordable in _COLD_START_LANES:
            vec = _synth_feature_vec(rng, overrides)
            target = structural_target_action(
                vec, affordable=affordable,
                know_threshold=know_thr, skill_floor=skill_floor)
            policy.train_step(vec, target, advantage=adv)
            n += 1
    return n


def _structural_explore(cfg, store, policy, shm_writer, life, name) -> None:
    """Active idle action-space exploration (deadlock fix step 2, §7.C/§24.6).

    For each recalled context, teach the policy the VERIFIABLE structural target
    (`structural_target_action`) directly via cross-entropy (`train_step` toward
    the known-correct action) — so the under-used actions get learned on the
    contexts where they are structurally correct, off the live path (INV-OML-9).
    Esp. `IDK`: it has NO live reward stream, so this idle bootstrap is the only
    place it can be learned. The oracle's know/don't-know axis is the
    memory-search signal (recall_top_cosine captured per turn — "does a
    dereferenceable thought exist?"); research-vs-IDK is metabolic affordability.

    We teach the verifiable target DIRECTLY rather than Boltzmann-sample-and-
    reward: the target is known, and sampling + REINFORCE's decaying advantage
    provably collapses to a degenerate single-action policy in this net (verified
    across seeds). `train_step(target, advantage>0)` is a cross-entropy step
    (gradient = probs − onehot_target) → it robustly learns the feature-
    conditional routing (8/8 seeds with the live multi-feature recall signal),
    bounded by the existing weight regularization. Stochastic discovery stays on
    LIVE turns (unknown outcomes, per-action-baseline `learn`); idle consolidates
    toward the verifiable oracle, live verdicts refine BEYOND it. This is what
    closes G5/GB8."""
    affordable = _research_affordable(life, cfg)
    adv = float(cfg["explore_structural_advantage"])
    know_thr = float(cfg["explore_know_threshold"])
    skill_floor = float(cfg["explore_skill_floor"])
    taught = 0

    # Break D (RFP_synthesis_reuse_and_routing_revival) — anti-collapse maintenance.
    # The real-context teaching below keys on `distinct_recent_contexts`, which on a
    # collapsed Titan are ALL direct-shaped (high recall; skill/tool/MSL features 0)
    # → structural_target returns `direct` for every one → the idle pass REINFORCES
    # the collapse instead of breaking it (verified live: T1 routed 344/344 direct).
    # The balanced SYNTHETIC lanes (the SAME set the cold-start seed uses) re-teach
    # feature-discrimination EVERY tick regardless of live-traffic bias, so the dense
    # quality-judge `direct` stream cannot monopolize a restored policy (which never
    # re-runs the cold-start seed). Runs UNCONDITIONALLY (most important when there
    # are no/biased recent contexts). Off the live path (INV-OML-9); cross-entropy
    # toward the verifiable structural target (the proven-robust rule, NOT Boltzmann).
    _syn_passes = int(cfg.get("structural_synthetic_passes", 3))
    if _syn_passes > 0:
        _rng = np.random.default_rng(0)   # deterministic balanced anchors
        for _ in range(_syn_passes):
            for _label, _overrides, _affordable in _COLD_START_LANES:
                _vec = _synth_feature_vec(_rng, _overrides)
                _target = structural_target_action(
                    _vec, affordable=_affordable,
                    know_threshold=know_thr, skill_floor=skill_floor)
                policy.train_step(_vec, _target, advantage=adv)
                taught += 1

    contexts = store.distinct_recent_contexts(int(cfg["explore_structural_batch"]))
    for features, _goal_class in (contexts or []):
        if len(features) != OUTER_POLICY_INPUT_DIM:
            continue
        vec = np.asarray(features, dtype=np.float32)
        target = structural_target_action(
            vec, affordable=affordable, know_threshold=know_thr, skill_floor=skill_floor)
        policy.train_step(vec, target, advantage=adv)   # cross-entropy → verifiable target
        taught += 1
    if taught:
        store.save_policy_flat(policy.to_flat().tolist(),
                               policy.total_updates, policy.reward_baseline)
        _publish_weights(shm_writer, policy)
        store.log_explore(
            "structural", "",
            f"n={taught} synthetic_passes={_syn_passes} affordable={affordable}")


def _maybe_distill_macro(goal_class, action, store, cfg, send_queue, name) -> None:
    if store.macro_already_emitted(goal_class, action):
        return
    if store.win_count(goal_class, action) < int(cfg["macro_min_wins"]):
        return
    _emit_macro(goal_class, action, store, send_queue, name)


def _outer_deliberate(cfg, store, send_queue, name, outer_reason, outer_meta) -> None:
    """Phase-C piece 6 — ONE outer meta-reasoning deliberation per explore tick.

    Draws a verified-win `goal_class` not yet macro-emitted → seeds the lower
    `OuterReasoningEngine` with its 30-D mean-feature signature → runs the
    `OuterMetaReasoningEngine` meta-chain (which DELEGATEs to the lower engine +
    EVALUATEs via the in-worker win-oracle) → trains the outer policies. On a
    VERIFIED composite, emits the macro via the shared S2 contract. This is the
    continuous idle-time deliberative learner ("otherwise dead code → useful")."""
    # §7.D D.4c — prefer a FRESH class (v1); else refine an already-emitted class
    # whose verified wins GREW ≥ macro_refine_min_growth since last emit → a
    # successor composite v{n+1} (mutate-not-update; the prior macro is untouched).
    candidates = store.candidate_macro_classes(int(cfg["macro_min_wins"]), limit=4)
    version = 1
    if candidates:
        goal_class, action = candidates[0]
    else:
        refine = store.refinement_candidates(
            int(cfg["macro_refine_min_growth"]), limit=4)
        if not refine:
            return
        goal_class, action, _prev_version, _wins_at_emit = refine[0]
        version = int(_prev_version) + 1
    signature = store.mean_features(goal_class, action)
    if signature is None or len(signature) != OUTER_POLICY_INPUT_DIM:
        return
    outer_reason.set_problem(signature)
    problem = {"topic": goal_class, "goal_class": goal_class, "action": int(action),
               "entry_primitive": "RECALL", "reason": "outer_deliberate"}
    composite = outer_meta.run_chain(problem=problem, reasoning_engine=outer_reason)
    outer_meta.train_terminal(float(composite.get("reward", 0.0)))
    try:
        outer_reason.save_all()
        outer_meta.save_all()
    except Exception:  # noqa: BLE001
        pass
    store.log_explore(
        "outer_deliberate", goal_class,
        f"v{version} verified={composite.get('verified')} "
        f"reward={composite.get('reward')} chain={composite.get('chain_length')}")
    if composite.get("verified"):
        # §7.D D.4b — macro-of-macros: if this verified deliberation builds on
        # ≥ macro_compose_min_children already-emitted macros feature-near G's
        # signature, emit `composed_from` those child macro labels (REASONING_
        # COMPOSED_FROM edges) instead of the D.1 leaf-join. The children's
        # canonical v1 labels (`macro::{gc}::{action_name}`) are the spine ids.
        composed_children = None
        try:
            related = store.related_emitted_macros(
                signature, exclude_goal_class=goal_class,
                exclude_action=int(action),
                floor=float(cfg["macro_compose_floor"]), limit=4)
            if len(related) >= int(cfg["macro_compose_min_children"]):
                composed_children = [
                    f"macro::{gc}::{action_index_to_name(a)}"
                    for gc, a, _cos in related]
                store.log_explore(
                    "macro_of_macros", goal_class,
                    f"children={len(composed_children)} "
                    f"floor={cfg['macro_compose_floor']}")
        except Exception as _d4b_err:  # noqa: BLE001
            logger.debug("[self_learning] D.4b compose detect soft-fail: %s",
                         _d4b_err)
        _emit_macro(goal_class, action, store, send_queue, name,
                    composed_from=composed_children, version=version)


def _build_routing_transitions(rows):
    """Per-goal_class trajectory linking for the full-IQL routing learner (P2),
    mirroring cgn._build_trajectory_links (concept_id ≙ goal_class). `rows` =
    [(features, action, reward, goal_class, ts), …] already global-sorted by ts
    ASC. Returns transition dicts {state, action, reward, next_state, terminal};
    next_state = the SAME goal_class's next state in time, tail terminal
    (V(s')=0). This is what makes it FULL IQL (V(s') participates), not bandit."""
    from collections import defaultdict
    valid = []
    groups = defaultdict(list)
    for feats, action, reward, gc, _ts in rows:
        if len(feats) != OUTER_POLICY_INPUT_DIM:
            continue
        valid.append((feats, int(action), float(reward), str(gc or "")))
        groups[gc].append(len(valid) - 1)
    out = []
    for _gc, idxs in groups.items():
        for j, ix in enumerate(idxs):
            feats, action, reward, gcl = valid[ix]
            nxt = valid[idxs[j + 1]][0] if j + 1 < len(idxs) else None
            out.append({"state": feats, "action": action, "reward": reward,
                        "next_state": nxt, "terminal": nxt is None,
                        "goal_class": gcl})
    return out


# Action index for the easy path (`direct`) — the only action P4 damps.
_DIRECT_ACTION_IDX = OUTER_ACTIONS.index("direct")


def _direct_damping(level: float, *, floor: float, slope: float,
                    graduated: bool, graduated_floor: float) -> float:
    """P4 damping multiplier for a POSITIVE `direct` reward — decays as the level
    rises (coasting pays less), bounded below by `floor` (>0, INV-MC-4). A
    mastered (chunked/graduated) goal_class eases the damping to `graduated_floor`
    (coasting on a mastered skill is fine — SOAR ② graduation map)."""
    d = max(float(floor), 1.0 - float(slope) * float(level))
    if graduated:
        d = max(d, float(graduated_floor))
    return d


def _shape_transitions_for_level(transitions, level, cfg, store):
    """P4 — shape the IQL TRAINING transitions by the current MasteryLevel
    (ARCHITECTURE_mastery_leveling.md §4). Mutates `reward` in place:
    a POSITIVE `direct` reward is damped by `_direct_damping(level)`, extra-damped
    when ungrounded (low recall_top_cosine = state[1]); other actions and negative
    rewards are untouched (we never ease a penalty). Returns the count shaped."""
    floor = float(cfg["direct_damping_floor"])
    slope = float(cfg["direct_damping_slope"])
    grad_floor = float(cfg["direct_graduated_floor"])
    recall_thr = float(cfg["direct_ungrounded_recall_threshold"])
    ungrounded_mul = float(cfg["direct_ungrounded_extra_damping"])
    shaped = 0
    for t in transitions:
        if t["action"] != _DIRECT_ACTION_IDX or t["reward"] <= 0.0:
            continue  # only POSITIVE direct rewards decay
        graduated = store.is_graduated(t.get("goal_class", ""))
        d = _direct_damping(level, floor=floor, slope=slope,
                            graduated=graduated, graduated_floor=grad_floor)
        try:
            recall = float(t["state"][1])  # recall_top_cosine
        except Exception:  # noqa: BLE001
            recall = 1.0
        if recall < recall_thr:
            d *= ungrounded_mul          # ungrounded direct damped MORE
        t["reward"] = float(t["reward"]) * d
        shaped += 1
    return shaped


def _update_mastery_level(cfg, store, policy, transitions, mastery, level_writer):
    """P3 — recompute the emergent MasteryLevel after an IQL pass and publish it.
    V̄ = mean symlog-V over the batch states; competence_rate = the scale-free
    blend (verified-success-rate + advantage-positive-rate) that GATES the ratchet
    (SOAR ①); n_chunks = distilled-macro count (SOAR ②). Persist + SHM-publish."""
    if mastery is None or not transitions:
        return
    try:
        import numpy as _np
        states = _np.array([t["state"] for t in transitions], dtype=_np.float32)
        v_sym = float(_np.mean([policy.value_symlog(s) for s in states]))
        succ = store.success_rate(int(cfg["competence_window"]))
        adv_pos = policy.advantage_positive_rate()
        competence = (float(cfg["competence_w_succ"]) * succ
                      + float(cfg["competence_w_adv"]) * adv_pos)
        n_chunks = store.chunk_count()
        readout = mastery.update(v_sym, competence, n_chunks)
        store.save_mastery_state(mastery.to_dict())
        if level_writer is not None:
            level_writer.write(mastery_readout_to_flat(readout))
        if readout.get("milestones"):
            store.log_explore(
                "mastery", "",
                f"level={readout['level']:.3f} grade={readout['grade']} "
                f"competence={readout['competence']:.3f} chunks={readout['n_chunks']} "
                f"milestones={','.join(readout['milestones'])}")
    except Exception as e:  # noqa: BLE001
        logger.debug("[self_learning] mastery level update soft-fail: %s", e)


def _explore_tick(cfg, store, policy, shm_writer, life, send_queue, name,
                  outer_reason=None, outer_meta=None, *,
                  mastery=None, level_writer=None) -> None:
    """Idle EXPLORE (L3) — metabolically gated (INV-OML-9). Passes:
    (1) balanced experience-replay (deadlock fix step 1 — minority actions at
    parity); (2) active structural exploration (step 2 — the verifiable structural
    oracle, the G5/GB8 closer); (3) Phase-C piece 6 — ONE outer meta-reasoning
    deliberation (the continuous deliberative learner). The LLM-judge counterfactual
    layer rides the `explore_request` hook (Phase 2 consumer)."""
    # Metabolic floor: never explore in survival/starvation or while dreaming.
    if life is not None:
        try:
            if life.is_dreaming():
                return
            if life.get_state() in _SURVIVAL_STATES:
                return
            if life.get_chi_total() < float(cfg["explore_chi_floor"]):
                return
        except Exception:  # noqa: BLE001
            return  # can't confirm metabolic headroom → conservatively skip
    # (1) experience-replay learning.
    if bool(cfg.get("oml_iql_enabled", True)):
        # FULL-IQL consolidation (P2) — REPLACES the REINFORCE replay. Build
        # per-goal_class trajectories (next_state/terminal) from the recent
        # reward tuples and run ONE offline train_iql pass; persist the π flat
        # (SHM) + the separate IQL nets. This is the sole policy-learning pass
        # under IQL (the per-turn _handle_reward only buffers).
        rows = store.iql_transitions(int(cfg.get("iql_replay_window", 2000)))
        transitions = _build_routing_transitions(rows)
        # P4 — level-driven reward shaping on the TRAINING transitions (the level
        # as of the previous tick; EMA-slow so this is the current curriculum).
        if (mastery is not None and len(transitions) >= 2
                and bool(cfg.get("level_shaping_enabled", True))):
            _level_now = float(mastery.readout().get("level", 0.0))
            _ns = _shape_transitions_for_level(transitions, _level_now, cfg, store)
            if _ns:
                store.log_explore("shape", "", f"level={_level_now:.2f} direct_damped={_ns}")
        if len(transitions) >= 2:
            stats = policy.train_iql(
                transitions,
                tau=float(cfg["iql_tau"]), beta=float(cfg["iql_beta"]),
                gamma=float(cfg["iql_gamma"]), polyak=float(cfg["iql_polyak"]),
                adv_clip=float(cfg["iql_adv_clip"]), lr=float(cfg["iql_lr"]),
                steps=int(cfg["iql_steps"]), batch_size=int(cfg["iql_batch_size"]))
            store.save_policy_flat(policy.to_flat().tolist(),
                                   policy.total_updates, policy.reward_baseline)
            store.save_iql_flat(policy.iql_to_flat().tolist(),
                                policy.total_iql_updates)
            _publish_weights(shm_writer, policy)
            store.log_explore(
                "iql", "",
                f"trans={stats.get('transitions')} v={stats.get('v_loss', 0.0):.4f} "
                f"q={stats.get('q_loss', 0.0):.4f} p={stats.get('policy_loss', 0.0):.4f} "
                f"iql_updates={stats.get('iql_updates')}")
            # P3 — recompute + publish the emergent MasteryLevel off this pass.
            _update_mastery_level(cfg, store, policy, transitions,
                                  mastery, level_writer)
    else:
        # Legacy REINFORCE balanced experience-replay (flag-off; INV-MC-7).
        _batch_n = int(cfg["explore_replay_batch"])
        if cfg.get("explore_balanced", True):
            batch = store.balanced_reward_tuples(_batch_n)
        else:
            batch = store.recent_reward_tuples(_batch_n)
        replayed = 0
        for features, action, reward in batch:
            if len(features) != OUTER_POLICY_INPUT_DIM:
                continue
            policy.learn(features, action, reward, baseline_alpha=float(cfg["baseline_alpha"]))
            replayed += 1
        if replayed:
            store.save_policy_flat(policy.to_flat().tolist(),
                                   policy.total_updates, policy.reward_baseline)
            _publish_weights(shm_writer, policy)
            store.log_explore("replay", "", f"batch={replayed}balanced={cfg.get('explore_balanced', True)}")
    # (2) active structural exploration — the G5/GB8 closer (deadlock fix step 2).
    if cfg.get("explore_structural", True):
        _structural_explore(cfg, store, policy, shm_writer, life, name)
    # (3) Phase C piece 6 — ONE outer meta-reasoning deliberation per tick.
    if (outer_meta is not None and outer_reason is not None
            and cfg["outer_meta_enabled"]):
        try:
            _outer_deliberate(cfg, store, send_queue, name, outer_reason, outer_meta)
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] outer deliberation soft-fail: %s", e)
    # (4) LLM-judge counterfactual layer (Phase 2 consumer; off by default).
    if cfg.get("explore_request_enabled"):
        try:
            send_queue.put({
                "type": SELF_LEARN_EXPLORE_REQUEST, "src": name, "dst": "agno",
                "ts": time.time(), "payload": {"goal_class": "", "prompt_hint": ""}})
        except Exception:  # noqa: BLE001
            pass


__all__ = ["self_learning_worker_main"]
