"""Felt-Teaching Worker — L2 consumer (RFP_inner_outer_felt_teaching_bridge §7.4).

The consumer half of the inner↔outer loop. Outer (Mind/synthesis) lived experience
teaches inner (Spirit/CGN) felt-grounding:

  consumes ENGRAM_FELT_CANDIDATE (synthesis Phase-3 producer's bus handoff — an Engram
  Object with no CGN felt-grounding, seeded with the felt-state it was lived under)
    → LanguageTeacher.build_felt_perturbation (prompt-build; LLM via this worker's own
      provider — best-effort)
    → CGNConsumerClient.send_transition({"type":"experience", …, "outcome_context":…})
      (cgn_worker → cgn.record_experience → buffers a complete (state, action, reward)
      transition then self-matches via record_outcome → value_net Sigma micro-update +
      concept-journey growth + felt-centroid materialization fed from
      outcome_context.felt_state)
    + emit_cross_insight (peer consumers learn from it).

`felt_teaching` is a FULL CGN consumer + IQL/value-net contributor — yet still
PROPOSE-ONLY: it feeds a complete experience transition; CGN's value_net + cross-consumer
maturity (≥2 distinct consumers) own the actual grounding. (record_outcome alone was a
no-op — this consumer never grounds, so its match loop never found a pending transition;
the experience path is what actually reaches IQL — fixed 2026-06-20, INV-Syn-ENG-4.) It NEVER writes a grounding
or emits CGN_CONCEPT_GROUNDED. The reward is conservatively bounded so one humble voice
cannot skew the shared V(s) (CGN's freq_scale de-weights any dominant consumer).

The loop closes elsewhere: when a 2nd consumer also grounds the Object, CGN matures it
→ CGN_CONCEPT_GROUNDED → synthesis's grounded-set → next dream the Object is no longer a
gap → the source Engram's axis_felt rises (= G4 / EEL-G7).

State ownership (G21): synthesis owns synthesis.duckdb exclusively, so this worker keeps
its OWN durable store (data/felt_teaching.duckdb) for the candidate status lifecycle +
dedup/retry. The bus event is the live handoff; the worker reconciles against ITS OWN
store, never a cross-process read of synthesis.duckdb.

frame_dependent (BRAIN-INV-18): the worker maintains its own grounded-view from
CGN_CONCEPT_GROUNDED. If a candidate's Object is ALREADY in that view (it got grounded
between the producer's gap-detection and now — a stale-gap/race), the lived felt is
taught as a domain_hint-scoped frame (status="frame_dependent", outcome_context.frame)
rather than a fresh base grounding — never an overwrite. NOTE: this is label-level
("base already grounded → frame-scope"); a true felt-vector conflict comparison is not
possible without CGN exposing per-concept felt-state (a future enhancement).
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from queue import Empty

import duckdb

from titan_hcl import bus
from titan_hcl.bus import CGN_CONCEPT_GROUNDED, ENGRAM_FELT_CANDIDATE
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _sev
from titan_hcl.logic.cgn_types import normalize_neuromods
from titan_hcl.synthesis.felt_bridge import normalize_label

logger = logging.getLogger("felt_teaching")

_HEARTBEAT_INTERVAL_S = 30.0
# felt-dict keys that are metadata, NOT neuromod levels (matches consolidation.py).
_FELT_META_KEYS = frozenset({"emotion", "emotion_confidence", "dream_cycle", "ts"})
_SETPOINT_CENTRE = 0.5
# Recurrence (count of distinct source Engrams an Object appears in) at which the
# evidence reaches full confidence weight. Keeps a one-off Object's reward modest.
_RECURRENCE_NORM = 3.0
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)
from titan_hcl.params import get_params

_WORKER_READY: bool = False
_BOOT_DEADLINE = None  # boot-grace deadline (monotonic); None=no grace


# ── Reward (Q3 — felt_evidence_strength, v1 bootstrap) ──────────────────────
def _felt_magnitude(felt: dict) -> float:
    """Normalized felt intensity ∈ [0,1] = ‖levels − 0.5‖ / max_dev over the numeric
    neuromod levels (metadata excluded). Empty / level-less felt → 0.0."""
    levels = [float(v) for k, v in (felt or {}).items()
              if k not in _FELT_META_KEYS and isinstance(v, (int, float))]
    if not levels:
        return 0.0
    dev = math.sqrt(sum((x - _SETPOINT_CENTRE) ** 2 for x in levels))
    max_dev = math.sqrt(len(levels)) * _SETPOINT_CENTRE
    return min(1.0, dev / max_dev) if max_dev > 0 else 0.0


def felt_evidence_strength(felt: dict, recurrence: int) -> float:
    """Q3 reward ∈ [0,1] = mean lived-neuromod magnitude × recurrence confidence.
    Strong, consistently-felt, recurring Objects are stronger teaching evidence;
    conservatively bounded so one humble voice can't skew the shared V(s)."""
    mag = _felt_magnitude(felt)
    conf = min(1.0, max(1, int(recurrence)) / _RECURRENCE_NORM)
    return round(max(0.0, min(1.0, mag * conf)), 4)


# ── The worker's OWN durable store (G21 — NOT synthesis.duckdb) ─────────────
class _FeltTeachingStore:
    """felt_teaching_worker's own state — status lifecycle + dedup/retry + recurrence.
    Single-process owned (this worker only) → plain duckdb is safe. Soft-fail."""

    def __init__(self, path: str = os.path.join("data", "felt_teaching.duckdb")):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = duckdb.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS felt_candidates ("
            " object_label VARCHAR, source_engram VARCHAR, source_version INTEGER,"
            " felt_state_json VARCHAR, domain_hint VARCHAR, reward DOUBLE,"
            " status VARCHAR, ts DOUBLE,"
            " PRIMARY KEY (object_label, source_engram, source_version))")

    def status_of(self, label: str, engram: str, version: int):
        try:
            row = self._conn.execute(
                "SELECT status FROM felt_candidates WHERE object_label=? AND "
                "source_engram=? AND source_version=?",
                [label, str(engram), int(version)]).fetchone()
            return row[0] if row else None
        except Exception as e:  # noqa: BLE001
            logger.debug("[felt_teaching] status_of soft-fail: %s", e)
            return None

    def recurrence(self, label: str) -> int:
        """Distinct source Engrams this Object has appeared in (≥1 incl. this one)."""
        try:
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT source_engram) FROM felt_candidates "
                "WHERE object_label=?", [label]).fetchone()
            return int(row[0]) if row and row[0] else 0
        except Exception as e:  # noqa: BLE001
            logger.debug("[felt_teaching] recurrence soft-fail: %s", e)
            return 0

    def upsert(self, *, label, engram, version, felt_json, domain_hint, reward,
               status) -> None:
        try:
            self._conn.execute(
                "INSERT INTO felt_candidates (object_label, source_engram, "
                "source_version, felt_state_json, domain_hint, reward, status, ts) "
                "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT (object_label, source_engram, "
                "source_version) DO UPDATE SET reward=excluded.reward, "
                "status=excluded.status, ts=excluded.ts",
                [label, str(engram), int(version), felt_json, domain_hint or "",
                 float(reward), status, time.time()])
        except Exception as e:  # noqa: BLE001
            logger.debug("[felt_teaching] upsert soft-fail: %s", e)

    def mark_matured(self, label: str) -> None:
        """A grounded Object (CGN_CONCEPT_GROUNDED) → mark its candidate rows matured."""
        try:
            self._conn.execute(
                "UPDATE felt_candidates SET status='matured' WHERE object_label=? "
                "AND status<>'matured'", [label])
        except Exception as e:  # noqa: BLE001
            logger.debug("[felt_teaching] mark_matured soft-fail: %s", e)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


@with_error_envelope(module_name="felt_teaching", subsystem="entry",
                     severity=_sev.FATAL)
def felt_teaching_worker_main(recv_queue, send_queue, name: str,
                              config: dict) -> None:
    """Main loop for the felt-teaching consumer subprocess."""
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    full_config = config or {}

    # ── SHM state-slot writer (Phase 11 §11.I.5; G21 single-writer) ──
    _state_writer = None
    try:
        from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
        _state_writer = ModuleStateWriter(
            module_name=name, layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT)
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning("[felt_teaching] ModuleStateWriter init failed "
                       "(SHM slot absent): %s", _sw_err)

    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = ((get_params("info_banner") or {}).get("titan_id")
                or resolve_titan_id())
    logger.info("[felt_teaching] Booting — titan_id=%s", titan_id)

    # Own durable store (NOT synthesis.duckdb — G21).
    try:
        store = _FeltTeachingStore()
    except Exception as e:
        logger.error("[felt_teaching] store init failed: %s — exiting", e)
        return

    # CGN consumer client (record_outcome / emit_cross_insight only — never ground(),
    # so the SHM weights are never loaded; lean footprint).
    from titan_hcl.logic.cgn_consumer_client import CGNConsumerClient
    from titan_hcl.logic.language_teacher import LanguageTeacher
    client = CGNConsumerClient(
        consumer_name="felt_teaching", send_queue=send_queue, module_name=name,
        titan_id=titan_id, config=get_params("cgn") or full_config)
    teacher = LanguageTeacher()

    # Self-register (cgn_worker also pre-registers as a safety net).
    try:
        send_queue.put({
            "type": "CGN_REGISTER", "src": name, "dst": "cgn",
            "ts": time.time(),
            "payload": {
                "name": "felt_teaching", "feature_dims": 30, "action_dims": 8,
                "action_names": ["reinforce", "explore", "differentiate",
                                 "consolidate", "associate", "dissociate",
                                 "deepen", "stabilize"],
                "reward_source": "felt_grounding_evidence",
                "max_buffer_size": 500, "consolidation_priority": 2,
            }})
    except Exception:  # noqa: BLE001
        pass

    # Best-effort LLM provider for the felt-perturbation (mirrors synthesis_worker —
    # missing key/import → no perturbation text, the record_outcome evidence still
    # flows). The perturbation is the teaching gesture; CGN's signal is record_outcome.
    provider = None
    try:
        inference_cfg = (get_params("inference") or {})
        if inference_cfg.get("ollama_cloud_api_key"):
            from titan_hcl.inference import (
                get_provider as _get_provider, resolve_internal_provider_name)
            provider = _get_provider(
                resolve_internal_provider_name(inference_cfg), inference_cfg)
    except Exception as e:  # noqa: BLE001
        logger.info("[felt_teaching] perturbation provider unavailable "
                    "(record_outcome still flows): %s", e)

    # Own grounded-view (event-sourced from CGN_CONCEPT_GROUNDED; label-level) for
    # frame_dependent detection.
    grounded_view: set[str] = set()

    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception:  # noqa: BLE001
            pass

    last_heartbeat = 0.0
    processed = 0
    errors = 0

    while True:
        now = time.time()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            try:
                send_queue.put({
                    "type": bus.MODULE_HEARTBEAT, "src": name, "dst": "guardian",
                    "payload": {"alive": True, "ts": now, "processed": processed,
                                "errors": errors}, "ts": now})
            except Exception:  # noqa: BLE001
                pass
            if _state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
                try:
                    _state_writer.heartbeat()
                except Exception:  # noqa: BLE001
                    pass
            last_heartbeat = now

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
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request)
                handle_module_probe_request(
                    msg, send_queue=send_queue, module_name=name,
                    state_writer=_state_writer, probe_fn=None)
            except Exception as _pe:  # noqa: BLE001
                logger.warning("[felt_teaching] PROBE handler failed: %s", _pe)
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[felt_teaching] Shutdown — exiting (processed=%d)",
                        processed)
            store.close()
            return

        payload = msg.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        if msg_type == CGN_CONCEPT_GROUNDED:
            lbl = normalize_label(payload.get("concept_id"))
            if lbl:
                grounded_view.add(lbl)
                store.mark_matured(lbl)
            continue

        if msg_type == ENGRAM_FELT_CANDIDATE:
            try:
                _process_candidate(payload, store, client, teacher, provider,
                                   grounded_view)
                processed += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                logger.warning("[felt_teaching] candidate processing failed: %s", e)
            continue


def _process_candidate(payload, store, client, teacher, provider,
                       grounded_view) -> None:
    """Process one ENGRAM_FELT_CANDIDATE → drive CGN via a complete "experience"
    transition (propose-only value-net contributor; INV-Syn-ENG-4)."""
    label = normalize_label(payload.get("object_label"))
    if not label:
        return
    engram = payload.get("source_engram") or ""
    version = int(payload.get("source_version") or 0)
    felt_state = payload.get("felt_state") or {}
    if not isinstance(felt_state, dict):
        felt_state = {}
    domain_hint = str(payload.get("domain_hint") or "")

    # Reconcile against OUR OWN store (not synthesis.duckdb) — skip if already driven.
    if store.status_of(label, engram, version) in ("grounding", "frame_dependent",
                                                    "matured"):
        return

    # frame_dependent (BRAIN-INV-18): the base is already grounded (in our event-sourced
    # view) → teach the lived felt as a domain-scoped frame, never a base overwrite.
    is_frame = label in grounded_view
    status = "frame_dependent" if is_frame else "grounding"

    import json as _json
    recurrence = store.recurrence(label) + 1  # incl. this occurrence
    reward = felt_evidence_strength(felt_state, recurrence)

    # Felt-perturbation teaching gesture (prompt-build in LanguageTeacher; LLM here).
    perturbation = None
    if provider is not None:
        try:
            import asyncio as _asyncio
            spec = teacher.build_felt_perturbation(label, felt_state, domain_hint)
            perturbation = _asyncio.run(provider.complete(
                prompt=spec["prompt"], system=spec["system"], temperature=0.4,
                max_tokens=spec.get("max_tokens", 80), timeout=30.0))
        except Exception as e:  # noqa: BLE001
            logger.debug("[felt_teaching] perturbation LLM soft-fail: %s", e)

    outcome_context = {
        "felt_state": felt_state,
        "source_engram": engram,
        "source_version": version,
        "domain_hint": domain_hint,
        "recurrence": recurrence,
    }
    if is_frame:  # F⊗C — scope the felt to the domain frame (never a base overwrite)
        outcome_context["frame"] = domain_hint or "default"
    if perturbation:
        outcome_context["perturbation"] = perturbation[:240]

    # PROPOSE-ONLY value-net contributor (INV-Syn-ENG-4). record_outcome alone is
    # a NO-OP here — this consumer never ground()s, so cgn.record_outcome's match
    # loop (cgn.py:594) finds no pending transition → the reward AND the per-concept
    # felt centroid (cgn.py:620) were both silently dropped. record_felt_experience
    # sends a COMPLETE "experience" transition (felt_state → 30D state) carrying
    # outcome_context → cgn.record_experience buffers (s,a,r) + self-matches via
    # record_outcome → Sigma V(s) update + concept-journey + felt-centroid. Still
    # propose-only: action 0 is the single "teach" gesture; no CGN_CONCEPT_GROUNDED.
    client.record_felt_experience(
        concept_id=label, neuromods=normalize_neuromods(felt_state),
        reward=reward, outcome_context=outcome_context,
        metadata={"action_name": "teach_felt", "encounter_type": "teaching",
                  "domain_hint": domain_hint, "recurrence": int(recurrence)})
    try:
        client.emit_cross_insight(reward, {"concept_id": label,
                                           "domain_hint": domain_hint})
    except Exception:  # noqa: BLE001
        pass

    store.upsert(label=label, engram=engram, version=version,
                 felt_json=_json.dumps(felt_state, sort_keys=True),
                 domain_hint=domain_hint, reward=reward, status=status)


__all__ = ["felt_teaching_worker_main", "felt_evidence_strength"]
