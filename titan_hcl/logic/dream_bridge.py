"""titan_hcl/logic/dream_bridge.py — Bridge A: inner→outer dream-memory harvest.

rFP §3G Phase 10G — RESTORED dropped orchestration loop. The harvest body was
``spirit_loop._harvest_dream_memories``; its invoker (the spirit_worker
END_DREAMING handler) was deleted in the D8-3 spirit→cognitive migration and
never re-homed, so inner→outer dream-memory injection silently stopped (zero
invoker; ``MEMORY_ADD`` had no emitter; the ``[DREAM_WISDOM]/[EUREKA]/…`` tags
that ``agno_hooks`` recall-perturbation (Bridge B) consumes were never produced).
Confirmed 100% dropped (audit_phase10_relocation_liveness_findings) → restored
per ``feedback_never_delete_live_logic_fix_dont_delete`` and Maker greenlight
(2026-05-28).

This module owns ONLY the pure harvest (sqlite reads + dict assembly + dedup).
The orchestration — firing on the dreaming→waking falling edge, emitting
``MEMORY_ADD`` (felt-tagged for Bridge B), marking consolidated chains, and
recording the msl dream-bridge metric — lives in ``cognitive_worker`` (which
owns the DreamingEngine + chain_archive + meta_wisdom engines), per the Maker
decision to home it there on dream-end.

Pure-ish: sqlite reads of ``data/inner_memory.db`` (vocabulary +
composition_history + meta_wisdom) + ``events_teacher.db``; no torch / no cgn
package import. The felt snapshot is passed in pre-built by the orchestrator.
"""
from __future__ import annotations

import os
import sqlite3

from titan_hcl.utils.silent_swallow import swallow_warn
from titan_hcl.logic.spirit_helpers import _load_bridge_dedup, _save_bridge_dedup


def harvest_dream_memories(
    chain_archive, meta_wisdom, felt: dict,
    cgn_db_path: str, dream_cycle: int, max_total: int = 8,
) -> tuple:
    """Harvest significant inner events for cognitive-graph injection.

    Called on the dreaming→waking falling edge (END_DREAMING). Returns
    ``(memories_list, chain_ids_to_mark)``. Each memory is a dict with:
    text, source, weight, neuromod_context (felt), category.
    Deduplicates against previous injections via dream_bridge_dedup.json.

    ``felt`` is the pre-built felt-state snapshot (neuromod levels + emotion)
    stored as ``neuromod_context`` so Bridge B recall perturbation can
    re-experience the somatic state at injection time.

    Sources (with caps):
      1. Crystallized meta-wisdom (max 3, weight 3.0)
      2. High-scoring reasoning chains (max 2, weight 3.0)
      3. CGN grounding milestones (max 2, weight 2.5)
      4. High-quality compositions (max 1, weight 2.0)
      5. Recent significant social interactions (max 2, weight 2.0)
    """
    memories: list = []
    chain_ids_to_mark: list = []
    dedup = _load_bridge_dedup()
    # vocabulary + composition_history + meta_wisdom all live in inner_memory.db
    inner_mem_db = os.path.join(os.path.dirname(cgn_db_path), "inner_memory.db") \
        if os.path.basename(cgn_db_path) != "inner_memory.db" else cgn_db_path

    # 1. Crystallized meta-wisdom (max 3)
    if meta_wisdom:
        try:
            conn = sqlite3.connect(inner_mem_db, timeout=10)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, problem_pattern, strategy_sequence, confidence, "
                "times_reused FROM meta_wisdom "
                "WHERE crystallized = 1 ORDER BY confidence DESC LIMIT 3"
            ).fetchall()
            conn.close()
            for r in rows:
                if r["id"] in dedup.get("wisdom_ids", []):
                    continue  # Already injected
                strategy = r["strategy_sequence"] or "unknown"
                if isinstance(strategy, str) and len(strategy) > 200:
                    strategy = strategy[:200]
                dedup.setdefault("wisdom_ids", []).append(r["id"])
                memories.append({
                    "text": (f"[DREAM_WISDOM] After dream cycle #{dream_cycle}, "
                             f"a reasoning pattern crystallized: '{r['problem_pattern'][:100]}' "
                             f"— strategy: {strategy}, confidence: {r['confidence']:.2f}, "
                             f"reused {r['times_reused']}x"),
                    "source": "dream_consolidation",
                    "weight": 3.0,
                    "neuromod_context": felt,
                    "category": "wisdom",
                })
        except Exception as e:
            swallow_warn('[DreamBridge] Meta-wisdom harvest failed', e,
                         key="logic.dream_bridge.meta_wisdom_harvest_failed", throttle=100)

    # 2. High-scoring unconsolidated reasoning chains (max 2)
    if chain_archive:
        try:
            chains = chain_archive.get_unconsolidated(limit=20)
            top_chains = sorted(
                [c for c in chains if c.get("outcome_score", 0) > 0.7],
                key=lambda c: -c.get("outcome_score", 0))[:2]
            for c in top_chains:
                chain_ids_to_mark.append(c["id"])
                memories.append({
                    "text": (f"[EUREKA] High-confidence reasoning chain "
                             f"(score={c['outcome_score']:.2f}): "
                             f"domain={c.get('domain', 'general')}, "
                             f"strategy={c.get('strategy_label', 'emergent')}, "
                             f"{c.get('chain_length', 0)} steps deep"),
                    "source": "dream_consolidation",
                    "weight": 3.0,
                    "neuromod_context": felt,
                    "category": "eureka",
                })
        except Exception as e:
            swallow_warn('[DreamBridge] Chain harvest failed', e,
                         key="logic.dream_bridge.chain_harvest_failed", throttle=100)

    # 3. CGN grounding milestones (max 2)
    try:
        conn = sqlite3.connect(cgn_db_path, timeout=5)
        rows = conn.execute(
            "SELECT word, cross_modal_conf, confidence, "
            "times_encountered, times_produced "
            "FROM vocabulary WHERE cross_modal_conf > 0.15 "
            "ORDER BY cross_modal_conf DESC LIMIT 2"
        ).fetchall()
        conn.close()
        for r in rows:
            if r[0] in dedup.get("cgn_words", []):
                continue  # Already injected
            xm = float(r[1]) if not isinstance(r[1], bytes) else 0.0
            dedup.setdefault("cgn_words", []).append(r[0])
            memories.append({
                "text": (f"[CGN_MILESTONE] The word '{r[0]}' reached deep grounding: "
                         f"cross-modal confidence {xm:.3f}, "
                         f"encountered {r[3]}x, produced {r[4]}x — "
                         f"I truly feel what this word means"),
                "source": "dream_consolidation",
                "weight": 2.5,
                "neuromod_context": felt,
                "category": "cgn_milestone",
            })
    except Exception as e:
        swallow_warn('[DreamBridge] CGN milestone harvest failed', e,
                     key="logic.dream_bridge.cgn_milestone_harvest_failed", throttle=100)

    # 4. High-quality compositions (max 1)
    try:
        conn = sqlite3.connect(cgn_db_path, timeout=5)
        row = conn.execute(
            "SELECT id, sentence, level, confidence FROM composition_history "
            "WHERE level >= 7 AND confidence > 0.8 "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row and row[0] not in dedup.get("composition_ids", []):
            dedup.setdefault("composition_ids", []).append(row[0])
            memories.append({
                "text": (f"[COMPOSITION] I composed at level {row[2]} "
                         f"(confidence {row[3]:.2f}): '{row[1]}'"),
                "source": "dream_consolidation",
                "weight": 2.0,
                "neuromod_context": felt,
                "category": "composition",
            })
    except Exception as e:
        swallow_warn('[DreamBridge] Composition harvest failed', e,
                     key="logic.dream_bridge.composition_harvest_failed", throttle=100)

    # 5. P4: Recent significant social interactions (max 2)
    try:
        _et_path = os.path.join(os.path.dirname(cgn_db_path), "events_teacher.db")
        if os.path.exists(_et_path):
            _et_conn = sqlite3.connect(_et_path, timeout=5)
            _et_conn.row_factory = sqlite3.Row
            _et_rows = _et_conn.execute(
                "SELECT author, topic, felt_summary, contagion_type, "
                "sentiment, arousal FROM felt_experiences "
                "WHERE arousal > 0.5 "
                "ORDER BY created_at DESC LIMIT 4"
            ).fetchall()
            _et_conn.close()
            _soc_count = 0
            for _sr in _et_rows:
                if _soc_count >= 2:
                    break
                _soc_key = f"{_sr['author']}:{_sr['topic']}"
                if _soc_key in dedup.get("social_interactions", []):
                    continue
                dedup.setdefault("social_interactions", []).append(_soc_key)
                # Trim dedup list
                dedup["social_interactions"] = dedup["social_interactions"][-50:]
                memories.append({
                    "text": (f"[SOCIAL_INTERACTION] Felt {_sr['contagion_type'] or 'connection'} "
                             f"from @{_sr['author']} about '{_sr['topic']}': "
                             f"{_sr['felt_summary']}"),
                    "source": "dream_consolidation",
                    "weight": 2.0,
                    "neuromod_context": felt,
                    "category": "social_interaction",
                })
                _soc_count += 1
    except Exception as e:
        swallow_warn('[DreamBridge] Social interaction harvest failed', e,
                     key="logic.dream_bridge.social_interaction_harvest_failed", throttle=100)

    # Cap total + save dedup state
    memories = memories[:max_total]
    _save_bridge_dedup(dedup)
    return memories, chain_ids_to_mark
