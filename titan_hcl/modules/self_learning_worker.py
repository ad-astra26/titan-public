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
    OUTER_META_POLICY_STATE_SPEC,
    OUTER_POLICY_INPUT_DIM,
    OuterMetaPolicy,
    action_index_to_name,
)

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
    "explore_request_enabled": False,  # active idle problem-gen (Phase 2 consumer)
    "macro_min_wins": 5,               # verified wins of one (goal_class,action) → distil
}
_SURVIVAL_STATES = frozenset({"SURVIVAL", "STARVATION"})


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
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_decisions ("
            " parent_tool_call_tx VARCHAR PRIMARY KEY, features_json VARCHAR,"
            " action INTEGER, goal_class VARCHAR, turn_id VARCHAR, ts DOUBLE)")
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
            " PRIMARY KEY (goal_class, action))")

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

    # -- pending decisions (the async-join stash) -----------------------
    def stash_decision(self, *, tx, features, action, goal_class, turn_id) -> None:
        try:
            self._conn.execute(
                "INSERT INTO pending_decisions (parent_tool_call_tx, features_json, "
                "action, goal_class, turn_id, ts) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT (parent_tool_call_tx) DO UPDATE SET "
                "features_json=excluded.features_json, action=excluded.action, "
                "goal_class=excluded.goal_class, turn_id=excluded.turn_id, ts=excluded.ts",
                [str(tx), json.dumps(list(features)), int(action),
                 str(goal_class or ""), str(turn_id or ""), time.time()])
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] stash_decision soft-fail: %s", e)

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

    def recent_reward_tuples(self, n: int):
        try:
            rows = self._conn.execute(
                "SELECT features_json, action, reward FROM reward_tuples "
                "ORDER BY id DESC LIMIT ?", [int(n)]).fetchall()
            return [(json.loads(r[0]), int(r[1]), float(r[2])) for r in rows]
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] recent_reward_tuples soft-fail: %s", e)
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

    def mark_macro_emitted(self, goal_class: str, action: int) -> None:
        try:
            self._conn.execute(
                "INSERT INTO macro_emitted (goal_class, action, ts) VALUES (?,?,?) "
                "ON CONFLICT (goal_class, action) DO NOTHING",
                [str(goal_class or ""), int(action), time.time()])
        except Exception as e:  # noqa: BLE001
            logger.debug("[self_learning] mark_macro_emitted soft-fail: %s", e)

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
    sub = ((config or {}).get("synthesis", {}) or {}).get("self_learning", {}) or {}
    out = dict(_DEFAULTS)
    out.update({k: v for k, v in sub.items() if k in _DEFAULTS})
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
    titan_id = ((full_config.get("info_banner", {}) or {}).get("titan_id")
                or resolve_titan_id())
    logger.info("[self_learning] Booting — titan_id=%s enabled=%s", titan_id,
                cfg["enabled"])

    try:
        store = _SelfLearningStore()
    except Exception as e:
        logger.error("[self_learning] store init failed: %s — exiting", e)
        return

    # Policy — restore from our own store, else fresh (cold-start).
    policy = OuterMetaPolicy(lr=float(cfg["policy_lr"]))
    loaded = store.load_policy_flat()
    if loaded is not None:
        try:
            import numpy as _np
            restored = OuterMetaPolicy.from_flat(_np.asarray(loaded[0], dtype=_np.float32))
            policy = restored
            policy.lr = float(cfg["policy_lr"])
            logger.info("[self_learning] policy restored (updates=%d, baseline=%.3f)",
                        policy.total_updates, policy.reward_baseline)
        except Exception as e:  # noqa: BLE001
            logger.warning("[self_learning] policy restore failed (cold-start): %s", e)

    # SHM weight publisher (single writer — INV-OML-8 / G21).
    _shm_writer = None
    try:
        _shm_writer = StateRegistryWriter(
            OUTER_META_POLICY_STATE_SPEC, ensure_shm_root(titan_id))
        _publish_weights(_shm_writer, policy)
    except Exception as e:  # noqa: BLE001
        logger.warning("[self_learning] SHM writer init failed: %s", e)

    # Metabolic gate reader (soft — cold default permits, but survival blocks).
    try:
        from titan_hcl.proxies.life_force_proxy import LifeForceShmReader
        _life = LifeForceShmReader()
    except Exception:  # noqa: BLE001
        _life = None

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
                _explore_tick(cfg, store, policy, _shm_writer, _life, send_queue, name)
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
                if _handle_reward(payload, store, policy, _shm_writer, cfg,
                                  send_queue, name):
                    trained += 1
                processed += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                logger.warning("[self_learning] reward join failed: %s", e)
            continue


def _handle_reward(payload, store, policy, shm_writer, cfg, send_queue, name) -> bool:
    """One policy update from a verdict-time reward.

    v1.1 (Phase 1, INV-OML-12): the synthesis-side C1 capture emits
    `(features, action, reward, goal_class)` DIRECTLY at verdict-time — decision
    and outcome travel together (the per-use Reasoning record carries both), so
    there is NO cross-process async-join. The legacy stash/join path is kept,
    dormant, for Phase-2 genuinely-async rewards (LLM-judge / user feedback)."""
    features = payload.get("features")
    action = payload.get("action")
    goal_class = str(payload.get("goal_class", "") or "")
    if features is not None and action is not None:
        # v1.1 direct path — decision + outcome together.
        if len(features) != OUTER_POLICY_INPUT_DIM:
            return False
        action = int(action)
        reward = float(payload.get("reward", 0.0))
    else:
        # Phase-2 (parked) — async reward joins a stashed decision by tx.
        tx = payload.get("parent_tool_call_tx")
        if not tx:
            return False
        decision = store.pop_decision(tx)
        if decision is None:
            return False  # no matching decision (pruned / not ours) — not an error
        features, action, goal_class = decision
        if len(features) != OUTER_POLICY_INPUT_DIM:
            return False
        reward = float(payload.get("reward", 0.0))
    policy.learn(features, action, reward, baseline_alpha=float(cfg["baseline_alpha"]))
    store.save_policy_flat(policy.to_flat().tolist(),
                           policy.total_updates, policy.reward_baseline)
    _publish_weights(shm_writer, policy)
    store.record_reward_tuple(features=features, action=action, reward=reward,
                              goal_class=goal_class)
    logger.info("[self_learning] trained: action=%s reward=%+.0f goal=%s "
                "updates=%d baseline=%.3f", action_index_to_name(action),
                reward, goal_class or "-", policy.total_updates, policy.reward_baseline)
    # Macro distillation (S1) — a (goal_class, action) with enough verified wins.
    if reward > 0 and goal_class:
        _maybe_distill_macro(goal_class, action, store, cfg, send_queue, name)
    return True


def _maybe_distill_macro(goal_class, action, store, cfg, send_queue, name) -> None:
    if store.macro_already_emitted(goal_class, action):
        return
    if store.win_count(goal_class, action) < int(cfg["macro_min_wins"]):
        return
    signature = store.mean_features(goal_class, action)
    if signature is None:
        return
    try:
        send_queue.put({
            "type": SELF_LEARN_MACRO_READY, "src": name, "dst": "synthesis",
            "ts": time.time(),
            "payload": {
                "goal_class": goal_class,
                "action": int(action),
                "action_name": action_index_to_name(action),
                "signature": signature,
                "b_i": 1.0, "c": 1.0,
                "time_cost": 1.0,                 # oracle-verified → proficient (B1)
                "use_count": store.win_count(goal_class, action),
                "verified": True,                  # only verified wins reach here
                "label": f"macro::{goal_class}::{action_index_to_name(action)}",
            }})
        store.mark_macro_emitted(goal_class, action)
        store.log_explore("macro_emitted", goal_class, action_index_to_name(action))
        logger.info("[self_learning] macro-strategy distilled: %s → %s (wins=%d)",
                    goal_class, action_index_to_name(action),
                    store.win_count(goal_class, action))
    except Exception as e:  # noqa: BLE001
        logger.debug("[self_learning] macro emit soft-fail: %s", e)


def _explore_tick(cfg, store, policy, shm_writer, life, send_queue, name) -> None:
    """Idle EXPLORE (L3) — metabolically gated (INV-OML-9). Phase-1 action =
    experience-replay on accumulated reward tuples (self-contained; sharpens the
    policy between sparse live rewards). The active idle problem-generation
    request is emitted only when its consumer is wired (Phase 2)."""
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
    batch = store.recent_reward_tuples(int(cfg["explore_replay_batch"]))
    if not batch:
        return
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
        store.log_explore("replay", "", f"batch={replayed}")
    # Active idle problem-generation (Phase 2 consumer; off by default).
    if cfg.get("explore_request_enabled"):
        try:
            send_queue.put({
                "type": SELF_LEARN_EXPLORE_REQUEST, "src": name, "dst": "agno",
                "ts": time.time(), "payload": {"goal_class": "", "prompt_hint": ""}})
        except Exception:  # noqa: BLE001
            pass


__all__ = ["self_learning_worker_main"]
