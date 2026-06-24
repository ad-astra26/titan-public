"""pattern_logic_worker — the cross-substrate pattern/model faculty (Phase 1).

RFP_pattern_logic.md §7.1 (LOCKED 2026-06-24). A second-order L2 worker that
observes verified-transitions from BOTH substrates — the OUTER synthesis verdict
stream (`VERIFIED_TRANSITION` broadcast) and the INNER CGN `reasoning_strategy`
HAOV (read-only snapshot) — clusters the recurring ones into PATTERNs, promotes
the oracle-grounded ones into MODELs, and OFFERs them back (OUTER: a reasoning-
composite → OML `composite_match` φ(s); INNER: the `vec[17]` model-sig cache the
CGN grounding path reads — exported here, consumed by cgn._build_state_vector).

G21 sole-writer of `data/pattern_logic.duckdb`. The heavy RECOGNISE→CONSTRUCT→OFFER
pass runs at a BOUNDED, EMERGENT cadence (CGN_IMPASSE / dream tick / a floor
interval — never per-tick, INV-PL-3) on a daemon thread, OFF the recv loop, so the
heartbeat is never starved (the GIL-starvation scar). The recv loop only drains the
queue, records outer transitions, and arms the pass.

CONSTRUCT note (Phase 1): every observed transition is ALREADY an oracle/CGN-grounded
verdict, so promotion accrues on accumulated oracle-terminated evidence (INV-PL-2
satisfied) — no fresh cross-process oracle call. Active cheapest-oracle RE-test of a
borderline case is a Phase-2 enhancement (needs the cross-process oracle RPC).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from queue import Empty
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from titan_hcl import bus
from titan_hcl.synthesis import pattern_op_taxonomy as taxo
from titan_hcl.synthesis.pattern_particle_store import PatternParticleStore
from titan_hcl.synthesis.consolidation import _default_cosine
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed)

logger = logging.getLogger(__name__)

# §10.B — Python L2/L3 modules MUST publish MODULE_HEARTBEAT every 10s
# (MODULE_HEARTBEAT_INTERVAL_S=10, MODULE_HEARTBEAT_TIMEOUT_S=90).
_HEARTBEAT_INTERVAL_S = 10.0
_POLL_INTERVAL_S = 0.2
# Phase 11 §11.I.5 boot-grace heartbeat state (set per-boot in the entry fn).
_WORKER_READY: bool = False
_BOOT_DEADLINE = None

# Defaults — mirror config.toml [pattern_logic]; emergence over determinism.
_DEFAULTS = {
    "enabled": True,
    "min_interval_s": 300.0,        # bounded-cadence floor (never per-tick)
    "cos_thresh": 0.85,             # signature-cluster gate (ConsolidationPass parity)
    "min_cluster_size": 3,          # transitions to form a PATTERN
    "min_sources": 2,               # distinct sources/domains to PROPOSE (Q2 / INV-PL-4)
    "max_cluster_window": 500,      # cap the pass size (GIL-safe)
    "c0": 1.0,                      # confidence saturation constant
    "promote_floor": 0.85,          # PROMOTE c-gate (Q2)
    "min_transitions": 5,           # PROMOTE evidence-mass gate (Q2)
    "f_floor": 0.7,                 # PROMOTE truth-fraction gate (a MODEL must work)
}

_HAOV_SNAPSHOT_NAME = "haov_reasoning_strategy_snapshot.json"


# ── embedding (shared 384-d bge space for BOTH substrates) ───────────────────
def _embed(embedder: Any, text: str, dim: int = 384) -> Optional[np.ndarray]:
    """L2-normalized embedding of a context string, or None on failure/empty."""
    if not text:
        return None
    try:
        vec = np.asarray(embedder.encode(text), dtype=np.float32).reshape(-1)
        return vec if vec.size else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("[pattern_logic] embed failed: %s", exc)
        return None


# ── inner op derivation (best-effort from the HAOV rule/effect text) ─────────
_INNER_OP_KEYWORDS = (
    ("research", "RESEARCH"), ("fetch", "RESEARCH"), ("lookup", "RESEARCH"),
    ("recall", "RECALL"), ("direct", "RECALL"), ("memory", "RECALL"),
    ("compare", "COMPARE"), ("delta", "COMPARE"), ("diff", "COMPARE"),
    ("compute", "COMPUTE"), ("compose", "COMPOSE"), ("transform", "TRANSFORM"),
    ("verify", "VERIFY"), ("tool", "TOOL"), ("skill", "SKILL"),
)


def _inner_op(rule: str, predicted_effect: str) -> str:
    """Heuristic op for an inner HAOV hypothesis. Falls back to RECALL (the inner
    reasoning faculty's default 'derive from what I have'). Live data refines this."""
    hay = f"{rule} {predicted_effect}".lower()
    for kw, op in _INNER_OP_KEYWORDS:
        if kw in hay:
            return op
    return "RECALL"


# ── OBSERVE: normalize one outer VERIFIED_TRANSITION into the store ──────────
def record_outer_transition(store: PatternParticleStore, embedder: Any,
                            payload: Dict[str, Any]) -> Optional[int]:
    """Record one outer verified-transition. Returns the tx id, or None if skipped."""
    ctx = str(payload.get("context", "") or "")
    sig = _embed(embedder, ctx)
    if sig is None:
        return None
    op = taxo.op_for_oracle(str(payload.get("oracle_id", "") or ""))
    frame = str(payload.get("frame", "") or "general")
    return store.record_transition(
        signature=sig.tolist(), operation=op, frame=frame,
        verdict=bool(payload.get("verdict")), substrate="outer",
        source=str(payload.get("source", "tool_verdict")), context_label=ctx)


# ── OBSERVE: ingest the inner reasoning_strategy HAOV snapshot (read-only) ────
def ingest_inner_snapshot(store: PatternParticleStore, embedder: Any,
                          snapshot_path: str, seen: Dict[str, tuple]) -> int:
    """Read the HAOV snapshot; record the NET-NEW confirmations (TRUE) and
    falsifications (FALSE) per hypothesis as inner verified-transitions. `seen`
    tracks last (conf, fals) per rule to ingest only deltas. Returns #recorded."""
    if not os.path.exists(snapshot_path):
        return 0
    try:
        import json
        with open(snapshot_path) as f:
            snap = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[pattern_logic] inner snapshot read failed: %s", exc)
        return 0
    recorded = 0
    for h in snap.get("hypotheses", []) or []:
        rule = str(h.get("rule", "") or "")
        if not rule:
            continue
        conf = int(h.get("confirmations", 0) or 0)
        fals = int(h.get("falsifications", 0) or 0)
        prev_conf, prev_fals = seen.get(rule, (0, 0))
        d_conf = max(0, conf - prev_conf)
        d_fals = max(0, fals - prev_fals)
        if d_conf == 0 and d_fals == 0:
            continue
        ctx = f"{rule} {h.get('action_context', '')}"
        sig = _embed(embedder, ctx)
        if sig is None:
            continue
        op = _inner_op(rule, str(h.get("predicted_effect", "") or ""))
        # PROVENANCE: frame = the CGN consumer; source = the HAOV `rule` name (so a
        # model built from this transition knows which inner rule to corroborate).
        for _ in range(d_conf):
            store.record_transition(signature=sig.tolist(), operation=op,
                                    frame="reasoning_strategy", verdict=True,
                                    substrate="inner", source=rule, context_label=ctx)
            recorded += 1
        for _ in range(d_fals):
            store.record_transition(signature=sig.tolist(), operation=op,
                                    frame="reasoning_strategy", verdict=False,
                                    substrate="inner", source=rule, context_label=ctx)
            recorded += 1
        seen[rule] = (conf, fals)
    return recorded


# ── RECOGNISE + CONSTRUCT (the bounded-cadence emergence pass) ───────────────
def _match_existing(store: PatternParticleStore, tx: Dict[str, Any],
                    cos_thresh: float) -> Optional[Dict[str, Any]]:
    """Find an ACTIVE particle of the SAME op whose signature is within cos_thresh
    of this transition (the accumulation path). Returns the particle dict or None."""
    best, best_sim = None, cos_thresh
    txsig = tuple(float(x) for x in tx["signature"])
    if not txsig:
        return None
    for p in store.active_particles():
        if p["operation"] != tx["operation"]:
            continue
        psig = tuple(float(x) for x in p["signature"])
        if len(psig) != len(txsig):
            continue
        sim = _default_cosine(txsig, psig)
        if sim >= best_sim:
            best, best_sim = p, sim
    return best


def recognise_and_construct(store: PatternParticleStore, cfg: Dict[str, Any],
                            offer_sink: Callable[[Dict[str, Any]], None]) -> Dict[str, int]:
    """One bounded-cadence pass: accumulate unclustered transitions into matching
    particles (reinforce/cite), cluster the rest into new PATTERNs (INV-PL-4
    cross-substrate/cross-domain), promote eligible PATTERNs → MODELs (mutate-not-
    update), and hand each newly-promoted MODEL to `offer_sink`."""
    cos_thresh = float(cfg["cos_thresh"])
    txs = store.recent_transitions(only_unclustered=True,
                                   limit=int(cfg["max_cluster_window"]))
    stats = {"matched": 0, "patterns": 0, "models": 0, "cited": 0}

    # 1. Accumulate: match each unclustered tx to an existing particle (same op).
    leftover: List[Dict[str, Any]] = []
    for tx in txs:
        p = _match_existing(store, tx, cos_thresh)
        if p is None:
            leftover.append(tx)
            continue
        store.merge_evidence(p["id"], verdict=tx["verdict"], source=tx["source"],
                             ts=tx["ts"])
        store._attach_transitions([tx["id"]], p["id"])
        stats["matched"] += 1
        # MODEL reused-and-verified-true = the self-reinforcing G-REUSE signal.
        if p["kind"] == "MODEL" and tx["verdict"]:
            store.cite_model(p["id"])
            stats["cited"] += 1

    # 2. Cluster the leftovers (greedy, within-op, by signature cosine).
    clusters: List[List[Dict[str, Any]]] = []
    for tx in leftover:
        txsig = tuple(float(x) for x in tx["signature"])
        placed = False
        for cl in clusters:
            head = cl[0]
            if head["operation"] != tx["operation"]:
                continue
            if _default_cosine(txsig, tuple(float(x) for x in head["signature"])) >= cos_thresh:
                cl.append(tx)
                placed = True
                break
        if not placed:
            clusters.append([tx])

    # 3. PROPOSE patterns from viable clusters (INV-PL-4: cross-substrate OR
    #    cross-domain; Q2: >= min_sources distinct sources).
    for cl in clusters:
        if len(cl) < int(cfg["min_cluster_size"]):
            continue
        substrates = {t["substrate"] for t in cl}
        frames = {t["frame"] for t in cl}
        sources = {(t["substrate"], t["source"], t["frame"]) for t in cl}
        cross = (len(substrates) >= 2) or (len(frames) >= 2)
        if not cross or len(sources) < int(cfg["min_sources"]):
            continue
        sigs = np.asarray([t["signature"] for t in cl], dtype=np.float32)
        centroid = sigs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        # Frame = the actionable DOMAIN. Prefer an OUTER transition's frame (the
        # goal_class the OML composite_match keys on) over the inner consumer label
        # ("reasoning_strategy"), which is not a routable domain.
        outer_frames = [t["frame"] for t in cl if t["substrate"] == "outer"]
        model_frame = outer_frames[0] if outer_frames else cl[0]["frame"]
        pid = store.propose_pattern(
            signature=centroid.tolist(), operation=cl[0]["operation"],
            frame=model_frame,
            evidence=[{"verdict": t["verdict"], "source": t["source"], "ts": t["ts"]}
                      for t in cl],
            n_sources=len(sources), tx_ids=[t["id"] for t in cl])
        stats["patterns"] += 1
        logger.info("[pattern_logic] PROPOSED pattern %s op=%s frame=%s "
                    "(substrates=%d frames=%d n=%d)", pid, cl[0]["operation"],
                    model_frame, len(substrates), len(frames), len(cl))

    # 4. PROMOTE eligible PATTERNs → MODELs.
    for p in store.active_particles(kind="PATTERN"):
        if store.eligible_for_promotion(p["id"]):
            mid = store.promote_to_model(p["id"])
            stats["models"] += 1
            model = store.get_particle(mid)
            logger.info("[pattern_logic] PROMOTED %s → MODEL %s op=%s frame=%s "
                        "f=%.3f c=%.3f", p["id"], mid, model["operation"],
                        model["frame"], model["f"], model["c"])
            offer_sink(model)
    return stats


# ── OFFER ────────────────────────────────────────────────────────────────────
def build_offer_event(model: Dict[str, Any], name: str) -> Dict[str, Any]:
    """A PATTERN_MODEL_READY bus event persisting the MODEL as an OML composite."""
    op = model["operation"]
    rid = f"plmodel::{model['frame']}::{op}::{model['id']}"
    return {
        "type": bus.PATTERN_MODEL_READY, "src": name, "dst": "synthesis",
        "ts": time.time(),
        "payload": {
            "reasoning_id": rid,
            "goal_class": model["frame"],
            "action": taxo.oml_action_for_op(op),
            "signature": [float(x) for x in model["signature"]],
            "c": float(model["c"]),
            "use_count": int(model.get("use_count", 1)) or 1,
        },
    }


def build_corroboration_events(store: PatternParticleStore, model: Dict[str, Any],
                               name: str) -> List[Dict[str, Any]]:
    """OFFER-inner (rule-keyed corroboration, RFP §7.1/§VC-2): one CGN_MODEL_
    CORROBORATION per CGN consumer whose HAOV rules fed this MODEL. The shared key is
    the symbolic HAOV `rule` name (no embedding-space match). strength = model.c (the
    promoted confidence; the cluster already passed the cos_thresh join gate). Empty
    list if the model had no inner provenance (a purely-outer cluster)."""
    parent = model.get("parent_id")
    if not parent:
        return []
    rules_by_consumer: Dict[str, List[str]] = {}
    for consumer, rule in store.inner_rules_for_particle(parent):
        rules_by_consumer.setdefault(consumer, []).append(rule)
    strength = max(0.0, min(1.0, float(model.get("c", 0.0))))
    events: List[Dict[str, Any]] = []
    for consumer, rules in rules_by_consumer.items():
        events.append({
            "type": bus.CGN_MODEL_CORROBORATION, "src": name, "dst": "cgn",
            "ts": time.time(),
            "payload": {"consumer": consumer, "rules": rules, "strength": strength},
        })
    return events


# ── the process entry point ──────────────────────────────────────────────────
def _resolve_cfg(config: dict) -> Dict[str, Any]:
    raw = {}
    try:
        raw = dict((config or {}).get("pattern_logic") or {})
    except Exception:  # noqa: BLE001
        raw = {}
    cfg = dict(_DEFAULTS)
    cfg.update({k: raw[k] for k in _DEFAULTS if k in raw})
    return cfg


def pattern_logic_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """L2 worker entry. recv loop = drain + record outer + arm the pass; a daemon
    thread runs the bounded-cadence RECOGNISE→CONSTRUCT→OFFER pass."""
    # Phase 11 §11.I.5 — boot-grace heartbeat state (reset before the loop).
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()
    cfg = _resolve_cfg(config)
    if not cfg.get("enabled", True):
        logger.info("[pattern_logic] disabled by config — idle heartbeat only")

    # State dir / paths.
    data_dir = ((config or {}).get("memory_and_storage") or {}).get("data_dir") or "data"
    db_path = os.path.join(data_dir, "pattern_logic.duckdb")
    haov_snapshot_path = os.path.join(data_dir, _HAOV_SNAPSHOT_NAME)

    # Lifecycle SHM slot (Phase 11).
    _state_writer = None
    try:
        from titan_hcl.core.module_state import ModuleStateWriter, BootPriority
        _state_writer = ModuleStateWriter(module_name=name, layer="L2",
                                          boot_priority=BootPriority.OPTIONAL_POST_BOOT)
        _state_writer.write_state("starting")
    except Exception as exc:  # noqa: BLE001
        logger.debug("[pattern_logic] state writer unavailable: %s", exc)

    store = PatternParticleStore(
        db_path, c0=float(cfg["c0"]), promote_floor=float(cfg["promote_floor"]),
        min_transitions=int(cfg["min_transitions"]), f_floor=float(cfg["f_floor"]))

    # Embedder — ONE shared 384-d space for both substrates, LAZY-LOADED on first
    # actual use (a verdict to record, or the first contemplate pass). Loading the
    # bge embedder (~130 MB) eagerly at boot put TWO embedders on the shared T2+T3
    # box at once → a cold-boot memory/CPU spike that flapped both workers
    # (2026-06-24). Lazy load eliminates the boot spike; on a low-traffic Titan the
    # embedder may never load. Cached after first load (kept — llama.cpp models
    # don't cheaply unload; the steady-state ~130 MB is acceptable, the boot spike
    # was not).
    _embedder_box = {"e": None, "tried": False}

    def _embedder():
        if _embedder_box["e"] is None and not _embedder_box["tried"]:
            _embedder_box["tried"] = True
            try:
                from titan_hcl.utils.text_embedder import get_text_embedder
                _embedder_box["e"] = get_text_embedder()
                logger.info("[pattern_logic] embedder lazy-loaded on first use")
            except Exception as exc:  # noqa: BLE001
                logger.warning("[pattern_logic] embedder unavailable (%s) — "
                               "OBSERVE inert", exc)
        return _embedder_box["e"]

    trigger = threading.Event()
    stop = threading.Event()
    # Restart-safety (kill-respawn allowlist): reconstruct the HAOV-snapshot dedup
    # state from the durable store so a respawn never re-ingests the snapshot.
    try:
        inner_seen: Dict[str, tuple] = store.inner_ingest_counts()
    except Exception:  # noqa: BLE001
        inner_seen = {}

    def _offer_sink(model: Dict[str, Any]) -> None:
        # OFFER-outer: persist the MODEL as a reasoning-composite (→ OML composite_match).
        try:
            send_queue.put_nowait(build_offer_event(model, name))
        except Exception as exc:  # noqa: BLE001
            logger.debug("[pattern_logic] outer offer emit failed: %s", exc)
        # OFFER-inner: corroborate the contributing inner HAOV rules (→ confidence boost).
        try:
            for ev in build_corroboration_events(store, model, name):
                send_queue.put_nowait(ev)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[pattern_logic] inner corroboration emit failed: %s", exc)

    def _contemplate_loop() -> None:
        while not stop.is_set():
            fired = trigger.wait(timeout=float(cfg["min_interval_s"]))
            trigger.clear()
            if stop.is_set():
                break
            if not cfg.get("enabled", True):
                continue
            _emb = _embedder()  # lazy — loads on the first pass that has work
            if _emb is None:
                continue
            try:
                n_inner = ingest_inner_snapshot(store, _emb, haov_snapshot_path,
                                                inner_seen)
                stats = recognise_and_construct(store, cfg, _offer_sink)
                if n_inner or any(stats.values()):
                    st = store.get_stats()
                    logger.info("[pattern_logic] pass: inner+=%d matched=%d "
                                "patterns+=%d models+=%d cited=%d (models_active=%d) "
                                "trigger=%s", n_inner, stats["matched"],
                                stats["patterns"], stats["models"], stats["cited"],
                                st["models_active"], "event" if fired else "interval")
            except Exception:  # noqa: BLE001
                logger.exception("[pattern_logic] contemplate pass failed")

    contemplate = threading.Thread(target=_contemplate_loop, name="pl-contemplate",
                                   daemon=True)
    contemplate.start()

    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception:  # noqa: BLE001
            pass

    # §7.D — register a bus-INDEPENDENT shutdown save (the load-bearing graceful-exit
    # save path; the floor SIGTERM handler runs it on EVERY exit). All transitions /
    # particles already write-through to duckdb (autocommit-durable), so this only
    # closes the connection cleanly — but registering it is the contract.
    try:
        from titan_hcl.core.worker_shutdown import register_shutdown_save
        register_shutdown_save(name, lambda: store.close())
    except Exception as _rs:  # noqa: BLE001
        logger.debug("[pattern_logic] register_shutdown_save skipped: %s", _rs)

    logger.info("[pattern_logic] booted — db=%s embedder=lazy cadence=%.0fs",
                db_path, float(cfg["min_interval_s"]))

    last_heartbeat = 0.0
    recorded = 0
    while True:
        now = time.time()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            try:
                send_queue.put_nowait({
                    "type": bus.MODULE_HEARTBEAT, "src": name, "dst": "guardian",
                    "payload": {"alive": True, "ts": now, "recorded": recorded},
                    "ts": now})
            except Exception:  # noqa: BLE001
                pass
            if _state_writer is not None and shm_heartbeat_allowed(
                    _WORKER_READY, _BOOT_DEADLINE):
                try:
                    _state_writer.heartbeat()
                except Exception:  # noqa: BLE001
                    pass
            last_heartbeat = now

        try:
            msg = recv_queue.get(timeout=_POLL_INTERVAL_S)
        except Empty:
            continue
        except Exception:  # noqa: BLE001
            continue

        msg_type = msg.get("type") if isinstance(msg, dict) else None
        if msg_type is None:
            continue

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch (REQUIRED
        # for every spawn-mode worker; BUS_HANDOFF/ADOPT_ACK/HANDOFF_CANCELED). Its
        # absence is what made this worker fail the supervision handshake → the
        # in-process watcher SIGTERM'd it ~28s after boot → shm_pid_dead restart loop.
        from titan_hcl.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        # ── SAVE_NOW → SAVE_DONE (§11.H.9 / D-SPEC-146) — REQUIRED. The guardian's
        # stop() publishes SAVE_NOW and waits the full save_timeout=30s for a matching
        # SAVE_DONE (core.py:897-982). A worker that never replies stalls EVERY stop
        # for 30s — and under restart-module that window perpetuates a shm_pid_dead
        # restart loop (the bug this worker hit live, 2026-06-24). State is already
        # duckdb-durable (write-through autocommit), so there's nothing to flush — ack
        # immediately, echoing the request_id so the orchestrator proceeds at once.
        if msg_type == bus.SAVE_NOW:
            _save_rid = (msg.get("payload") or {}).get("request_id")
            _t0 = time.time()
            try:
                send_queue.put_nowait({
                    "type": bus.SAVE_DONE, "src": name, "dst": "guardian",
                    "ts": time.time(),
                    "payload": {"module": name, "request_id": _save_rid,
                                "saved": True, "errors": 0,
                                "duration_ms": int((time.time() - _t0) * 1000)}})
            except Exception:  # noqa: BLE001
                pass
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[pattern_logic] MODULE_SHUTDOWN — exiting")
            break

        if msg_type == bus.MODULE_PROBE_REQUEST and _state_writer is not None:
            try:
                from titan_hcl.core.probe_dispatcher import handle_module_probe_request
                handle_module_probe_request(msg, send_queue=send_queue,
                                            module_name=name, state_writer=_state_writer,
                                            probe_fn=None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[pattern_logic] probe handler failed: %s", exc)
            continue

        if msg_type == bus.VERIFIED_TRANSITION:
            _emb = _embedder() if cfg.get("enabled", True) else None  # lazy load on first verdict
            if _emb is not None:
                try:
                    _pl = msg.get("payload") or {}
                    _tid = record_outer_transition(store, _emb, _pl)
                    if _tid is not None:
                        recorded += 1
                    logger.info("[pattern_logic] VERIFIED_TRANSITION recv ctx=%r op<-%s "
                                "verdict=%s recorded=%s total=%d",
                                str(_pl.get("context", ""))[:40], _pl.get("oracle_id"),
                                _pl.get("verdict"), _tid is not None, recorded)
                except Exception:  # noqa: BLE001
                    logger.warning("[pattern_logic] outer record failed", exc_info=True)
            continue

        if msg_type == "CGN_IMPASSE":
            # Bounded-cadence emergent trigger (INV-PL-3) — cgn_worker broadcasts
            # "CGN_IMPASSE" (dst="all", a raw type string, not a bus const) when a
            # consumer is stuck; a good moment to consolidate what we've seen.
            trigger.set()
            continue

    stop.set()
    trigger.set()
    try:
        store.close()
    except Exception:  # noqa: BLE001
        pass
