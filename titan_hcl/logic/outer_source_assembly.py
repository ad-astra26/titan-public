"""
outer_source_assembly — SPEC §9.F stateless library for assembling the
outer-trinity source dict, SHM-direct.

Phase C dissolution of ``TitanHCL._gather_outer_sources`` + the
``OUTER_SOURCES_SNAPSHOT`` bus broadcast (RFP_phase_c_titan_hcl_cleanup §2
Phase C; AUDIT_phase_c_gather_outer_sources_dissolution_20260522.md). The
broadcast carried architectural STATE over the bus — a Preamble G18 violation.
This module replaces it: each in-parent sensor-refresh sidecar calls
``assemble_outer_sources(keys, ctx)`` with its ``SOURCE_KEYS`` subset and the
function reads every source SHM-direct / file / utility / in-process, with NO
bus state transport and NO process-resident gather loop.

§9.F contract: this is a **stateless library** ("pure transform over SHM-read
state should NOT be carved into a worker"). The only stateful adjunct is
``OuterHeavyStatsRefresher`` (a G20 background refresher for the heavy DB
counts — DB queries must never run inline on the sidecar hot path); the
sidecars' host (titan_HCL) owns one instance and passes its cache via ``ctx``.

Source taxonomy (per AUDIT §2 + the live-T1 SHM probe §7):
  - SHM-direct (the G18 fix vs the old bus-cache/proxy):
      agency_stats, helper_statuses, assessment_stats, social_perception_stats,
      memory_status, memory_stats/memory_growth_metrics, knowledge_graph_stats,
      hormone_levels, cgn_stats, output_verifier_stats, language_stats,
      meta_reasoning_stats/meta_cgn_stats, expression_translator_stats,
      soul_health, timechain_genesis_stats.
  - FILE (any process): anchor_state/sol_balance/recovery_stats,
      genesis_record_exists/solana_local_stats, jailbreak_alerts_stats,
      world_footprint_extra_counts.
  - UTIL (importable, per-process): system_sensor_stats, network_monitor_stats,
      tx_latency_stats, block_delta_stats.
  - IN-PROCESS (via ctx, parent-resident): bus_stats, art/audio/text counts
      (observatory_db), uptime_seconds.
  - HEAVY (G20 in-process refresher cache via ctx): inner_memory_stats,
      social_x_gateway_stats, events_teacher_stats, community_engagement_stats.
  - DERIVED: substrate_success_rate, llm_avg_latency.

Stateful breath trackers (expr_window / willing_window / outer_body_change /
pi_heartbeat_hrv / outer_spirit_self_change / outer_spirit_history_stats) are
NOT assembled here — they relocate into their single owning sidecar (they
accumulate; §9.F is for pure transforms). The sidecar computes them from the
assembled dict + SHM and merges the breath keys.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Context ──────────────────────────────────────────────────────────


@dataclass
class OuterSourceContext:
    """References the assembler needs. Constructed once by the sidecar host.

    Mandatory: ``shm_bank`` (ShmReaderBank), ``titan_id``, ``data_dir``,
    ``start_time``. Optional in-process providers are read only when the
    requested key needs them (absent provider → key omitted, never crashes)."""

    shm_bank: Any
    titan_id: str
    data_dir: str
    start_time: float
    # Optional in-process providers (parent-resident):
    bus_stats_provider: Optional[Callable[[], dict]] = None
    observatory_db: Any = None
    heavy_stats: dict = field(default_factory=dict)
    # OuterSpiritHistory accumulator (the 6th stateful tracker). Owned by the
    # spirit sidecar; its light per-tick ingests run in the spirit provider,
    # its heavy refresh_dream_recall() runs in OuterHeavyStatsRefresher (G20).
    outer_spirit_history: Any = None
    # mtime/ttl caches for file reads (per-context, NOT global — keeps the
    # function stateless across contexts while avoiding re-reading files every
    # tick within one sidecar).
    _file_cache: dict = field(default_factory=dict)


# ── File readers (moved verbatim from plugin; pure given data_dir) ───


def _compute_jailbreak_stats(ctx: OuterSourceContext) -> dict:
    """data/jailbreak_alerts.json → 24h aggregates. mtime-cached.
    SPEC §23.9 SAT[3] boundary_enforcement + willing[13] protective_response."""
    import json as _json
    path = os.path.join(ctx.data_dir, "jailbreak_alerts.json")
    if not os.path.exists(path):
        return {"threats_detected_24h": 0, "blocked_24h": 0,
                "confirmed_threats_24h": 0, "total_alerts": 0}
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    cache = ctx._file_cache.get("jailbreak")
    if cache and cache.get("_mtime") == mtime:
        return cache["stats"]
    with open(path) as f:
        alerts = _json.load(f)
    if not isinstance(alerts, list):
        alerts = []
    now = time.time()
    cutoff = now - 86400
    recent = [a for a in alerts if isinstance(a, dict)
              and a.get("timestamp", 0) > cutoff]
    blocked = sum(1 for a in recent if a.get("score", 0) >= 0.9)
    confirmed = sum(1 for a in recent
                    if a.get("score", 1.0) < 1.0 and a.get("adversary_type"))
    severity_avg = (sum(max(0, 1.0 - float(a.get("score", 1.0)))
                        for a in recent) / len(recent)) if recent else 0.0
    defended_all_time = sum(1 for a in alerts if isinstance(a, dict)
                            and a.get("score", 0) >= 0.9)
    stats = {
        "threats_detected_24h": len(recent),
        "blocked_24h": blocked,
        "confirmed_threats_24h": confirmed,
        "severity_avg_24h": round(severity_avg, 4),
        "total_alerts": len(alerts),
        "defended_all_time": defended_all_time,
        "defended_per_hour": round(blocked / 24.0, 4),
        "defended_per_day": float(blocked),
    }
    ctx._file_cache["jailbreak"] = {"_mtime": mtime, "stats": stats}
    return stats


def _count_artifact_dirs(ctx: OuterSourceContext) -> dict:
    """arweave inscriptions + meditation memos counts (world_footprint). 60s cache."""
    cache = ctx._file_cache.get("artifact_dirs", {})
    now = time.time()
    if cache.get("_ts", 0) > now - 60:
        return cache["counts"]
    counts = {}
    try:
        arw_dir = os.path.join(ctx.data_dir, "arweave_devnet")
        counts["arweave_inscriptions"] = (
            sum(1 for f in os.listdir(arw_dir) if f.endswith(".tags.json"))
            if os.path.isdir(arw_dir) else 0)
    except Exception:
        counts["arweave_inscriptions"] = 0
    try:
        med_dir = os.path.join(ctx.data_dir, "meditation_memos")
        counts["meditation_memos"] = (
            sum(1 for _ in os.listdir(med_dir))
            if os.path.isdir(med_dir) else 0)
    except Exception:
        counts["meditation_memos"] = 0
    ctx._file_cache["artifact_dirs"] = {"_ts": now, "counts": counts}
    return counts


def _read_anchor_file(ctx: OuterSourceContext) -> Optional[dict]:
    """data/anchor_state.json — mtime-cached."""
    import json as _json
    path = os.path.join(ctx.data_dir, "anchor_state.json")
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    cache = ctx._file_cache.get("anchor")
    if cache and cache.get("_mtime") == mtime:
        return cache["data"]
    try:
        with open(path) as f:
            data = _json.load(f)
        ctx._file_cache["anchor"] = {"_mtime": mtime, "data": data}
        return data
    except Exception:
        return None


# ── Heavy DB stats refresher (G20 — never inline on the sidecar hot path) ─


class OuterHeavyStatsRefresher:
    """Background refresher for the heavy DB counts the outer dims consume.

    G20: "Heavy producers (DB queries, external RPC) get their own background
    refresher thread following the _heavy_stats_cache pattern." Moved verbatim
    from the deleted ``TitanHCL._ensure_heavy_stats_refresher``. The host
    (titan_HCL) starts ONE instance; its ``cache`` is shared into each sidecar's
    ``OuterSourceContext.heavy_stats``. Single owner → no G21 concern (it's an
    in-process cache, not an SHM slot)."""

    def __init__(self, titan_id: str, data_dir: str, is_x_gateway: bool,
                 outer_spirit_history: Any = None):
        self.titan_id = str(titan_id).upper()
        self.data_dir = data_dir
        self.is_x_gateway = is_x_gateway
        # OSH whose heavy refresh_dream_recall() (SQL COUNT against
        # experiential_memory) must run off the spirit sidecar hot path (G20).
        self.outer_spirit_history = outer_spirit_history
        self.cache: dict = {}
        self._started = False
        self._social_x_reader = None
        self._events_teacher_reader = None

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        t = threading.Thread(target=self._loop, name="outer_heavy_stats_refresher",
                             daemon=True)
        t.start()
        logger.info("[OuterHeavyStats] refresher thread started (60s cadence)")

    def _loop(self) -> None:
        time.sleep(20)  # let producers boot
        while True:
            try:
                self.cache["inner_memory_stats"] = self._read_inner_memory()
            except Exception as _e:
                logger.debug("[OuterHeavyStats] inner_memory: %s", _e)
            try:
                self.cache["social_x_gateway_stats"] = self._read_social_x()
            except Exception as _e:
                logger.debug("[OuterHeavyStats] social_x: %s", _e)
            try:
                self.cache["events_teacher_stats"] = self._read_events_teacher()
            except Exception as _e:
                logger.debug("[OuterHeavyStats] events_teacher: %s", _e)
            try:
                ce = self._read_community_engagement()
                if ce is not None:
                    self.cache["community_engagement_stats"] = ce
            except Exception as _e:
                logger.debug("[OuterHeavyStats] community_engagement: %s", _e)
            # OSH dream_recall — heavy SQL COUNT against experiential_memory;
            # G19/G20 says NEVER inline in the gather hot path (was refreshed
            # here in the deleted parent _heavy_stats_refresher).
            try:
                osh = self.outer_spirit_history
                if osh is not None and hasattr(osh, "refresh_dream_recall"):
                    osh.refresh_dream_recall()
            except Exception as _e:
                logger.debug("[OuterHeavyStats] osh dream_recall: %s", _e)
            time.sleep(60)

    def _read_inner_memory(self) -> dict:
        import sqlite3 as _sql
        db_path = os.path.join(self.data_dir, "inner_memory.db")
        if not os.path.exists(db_path):
            return {}
        stats: dict = {}
        conn = _sql.connect(f"file:{db_path}?mode=ro&immutable=0",
                            uri=True, timeout=2.0)
        try:
            conn.execute("PRAGMA busy_timeout=2000")
            for table in ("hormone_snapshots", "program_fires", "action_chains",
                          "creative_works", "event_markers", "vocabulary"):
                try:
                    c = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = int(c.fetchone()[0])
                except Exception:
                    stats[table] = 0
        finally:
            conn.close()
        return stats

    def _read_social_x(self) -> dict:
        if self._social_x_reader is None:
            from titan_hcl.logic.social_x_gateway import SocialXGateway
            self._social_x_reader = SocialXGateway()
        return self._social_x_reader.get_stats()

    def _read_events_teacher(self) -> dict:
        if self._events_teacher_reader is None:
            from titan_hcl.logic.events_teacher import EventsTeacherDB
            self._events_teacher_reader = EventsTeacherDB()
        return self._events_teacher_reader.get_stats(self.titan_id)

    def _read_community_engagement(self) -> Optional[dict]:
        # T1 owns social_x.db locally; T2/T3 reach T1 over HTTP (Phase 2.5.E).
        if self.is_x_gateway and self.titan_id == "T1":
            sxg = self._social_x_reader
            if sxg is None:
                from titan_hcl.logic.social_x_gateway import SocialXGateway
                self._social_x_reader = SocialXGateway()
                sxg = self._social_x_reader
            if hasattr(sxg, "get_community_engagement_stats"):
                return sxg.get_community_engagement_stats(
                    is_x_gateway=True, titan_id="T1")
            return None
        try:
            import urllib.request as _ur
            import json as _json
            url = (f"http://10.135.0.3:7777/v4/community-engagement-stats"
                   f"?titan_id={self.titan_id}")
            resp = _ur.urlopen(url, timeout=8.0)
            body = _json.loads(resp.read())
            if body.get("status") == "ok":
                stats = body.get("data") or {}
                stats["gateway_role"] = "kin-rpc"
                stats["titan_id"] = self.titan_id
                return stats
        except Exception:
            pass
        return None


# ── The assembler ────────────────────────────────────────────────────


def assemble_outer_sources(keys: Iterable[str], ctx: OuterSourceContext) -> dict:
    """Assemble exactly the requested source keys, SHM-direct + file/util/
    in-process. Stateless. Shared SHM reads are computed once. A key whose
    source is unavailable is omitted (the Rust daemon falls back to its
    documented default — same contract as ``_normalize_sources`` None)."""
    want = set(keys)
    out: dict = {}
    bank = ctx.shm_bank

    def _need(*ks: str) -> bool:
        return any(k in want for k in ks)

    # ── SHM: agency (agency_stats + helper_statuses) ─────────────────
    if _need("agency_stats", "helper_statuses", "substrate_success_rate"):
        ag = bank.read_agency_state() or {}
        if "agency_stats" in want and ag:
            out["agency_stats"] = ag
        if "helper_statuses" in want:
            hs = ag.get("helper_statuses")
            if hs is not None:
                out["helper_statuses"] = hs

    if "assessment_stats" in want:
        a = bank.read_assessment_state()
        if a:
            out["assessment_stats"] = a

    if "social_perception_stats" in want:
        sp = bank.read_social_perception_state()
        if sp:
            out["social_perception_stats"] = sp

    # ── SHM: memory (memory_status + memory_stats + knowledge_graph) ──
    if _need("memory_status", "memory_stats", "memory_growth_metrics",
             "knowledge_graph_stats"):
        mem = bank.read_memory_state() or {}
        if mem:
            if "memory_status" in want:
                out["memory_status"] = mem
            if _need("memory_stats", "memory_growth_metrics"):
                # outer_mind/outer_spirit read growth fields under these keys.
                growth = {
                    "learning_velocity": mem.get("learning_velocity", 0.0),
                    "directive_alignment": mem.get("directive_alignment", 0.0),
                    "effective_nodes_24h": mem.get("effective_nodes_24h", 0),
                    "high_quality_count": mem.get("high_quality_count", 0),
                    "persistent_count": mem.get("persistent_count", 0),
                }
                if "memory_stats" in want:
                    out["memory_stats"] = growth
                if "memory_growth_metrics" in want:
                    out["memory_growth_metrics"] = growth
            if "knowledge_graph_stats" in want:
                # transform kg_node_count→node_count (tensor's expected names).
                out["knowledge_graph_stats"] = {
                    "node_count": mem.get("kg_node_count", 0),
                    "edge_count": mem.get("kg_edge_count", 0),
                    "total_entities": mem.get("kg_node_count", 0),
                    "total_edges": mem.get("kg_edge_count", 0),
                }

    # ── SHM: hormone_levels (titanvm NS-program urgencies, D-SPEC-81) ─
    if "hormone_levels" in want:
        tvm = bank.read_titanvm_registers() or {}
        progs = tvm.get("programs", {}) if isinstance(tvm, dict) else {}
        if isinstance(progs, dict) and progs:
            out["hormone_levels"] = {
                name: float(p.get("urgency", 0.0))
                for name, p in progs.items() if isinstance(p, dict)
            }

    # ── SHM: cgn_stats (cgn_engine_state — C.2b added avg_reward/grounded) ─
    if "cgn_stats" in want:
        cgn = bank.read_cgn_engine_state()
        if cgn:
            out["cgn_stats"] = cgn

    if "output_verifier_stats" in want:
        ov = bank.read_output_verifier_state()
        if ov:
            out["output_verifier_stats"] = ov

    if _need("language_stats", "vocab_stats"):
        lang = bank.read_language_state()
        if lang:
            if "language_stats" in want:
                out["language_stats"] = lang
            if "vocab_stats" in want:
                out["vocab_stats"] = lang

    # ── SHM: meta_reasoning + meta_cgn (C.2a enriched the meta_cgn block) ─
    if _need("meta_reasoning_stats", "meta_cgn_stats"):
        mr = bank.read_meta_reasoning_state() or {}
        if mr:
            if "meta_reasoning_stats" in want:
                out["meta_reasoning_stats"] = mr
            if "meta_cgn_stats" in want:
                mcgn = mr.get("meta_cgn") or {}
                # Project to ONLY the 8 fields the tensors consume (keeps the
                # sensor_cache_outer_*.bin payload small — the full block is large).
                out["meta_cgn_stats"] = {
                    k: mcgn[k] for k in (
                        "knowledge_helpful_ratio", "knowledge_helpful_by_source",
                        "knowledge_responses_received", "usage_gini",
                        "eureka_accelerated_updates", "eureka_accelerated_per_hour",
                        "primitives_total", "primitives_grounded",
                    ) if k in mcgn
                }

    # ── SHM: expression_translator_stats (expression_state slot) ─────
    if "expression_translator_stats" in want:
        es = bank.read_expression_state()
        if es:
            out["expression_translator_stats"] = es

    # ── SHM: soul_health (soul_state.soul_initialized) ───────────────
    if "soul_health" in want:
        soul = bank.read_soul_state() or {}
        out["soul_health"] = 0.9 if soul.get("soul_initialized") else 0.2

    # ── SHM: inner_perception_stats (parent-published slot, C.7) ─────
    if "inner_perception_stats" in want:
        ip = bank.read_inner_perception_state()
        if ip:
            out["inner_perception_stats"] = ip

    # ── SHM: timechain_genesis_stats ─────────────────────────────────
    if "timechain_genesis_stats" in want:
        tc = bank.read_timechain_state()
        if isinstance(tc, dict):
            out["timechain_genesis_stats"] = {
                "total_blocks": float(tc.get("total_blocks", 0) or 0),
                "recent_anchor_age_s": float(
                    tc.get("recent_anchor_age_s", 86400.0) or 86400.0),
            }

    # ── FILE: anchor_state / sol_balance / recovery_stats ────────────
    if _need("anchor_state", "sol_balance", "recovery_stats"):
        anchor = _read_anchor_file(ctx)
        if isinstance(anchor, dict):
            if "anchor_state" in want:
                out["anchor_state"] = anchor
            if "sol_balance" in want:
                sb = anchor.get("sol_balance")
                if isinstance(sb, (int, float)):
                    out["sol_balance"] = float(sb)
            if "recovery_stats" in want:
                out["recovery_stats"] = {
                    "consecutive_failures": int(
                        anchor.get("consecutive_failures", 0) or 0),
                    "last_anchor_time": anchor.get("last_anchor_time", 0.0),
                    "anchor_count": int(anchor.get("anchor_count", 0) or 0),
                }

    # ── FILE: genesis_record_exists / solana_local_stats ─────────────
    if _need("genesis_record_exists", "solana_local_stats"):
        genesis_path = os.path.join(ctx.data_dir, "genesis_record.json")
        genesis_exists = os.path.exists(genesis_path)
        soul = bank.read_soul_state() or {}
        identity_ok = 1.0 if soul.get("soul_initialized") else 0.0
        if "genesis_record_exists" in want:
            out["genesis_record_exists"] = bool(genesis_exists)
        if "solana_local_stats" in want:
            out["solana_local_stats"] = {
                "identity_verified": identity_ok,
                "genesis_nft_exists": 1.0 if genesis_exists else 0.0,
            }

    if "jailbreak_alerts_stats" in want:
        out["jailbreak_alerts_stats"] = _compute_jailbreak_stats(ctx)

    if "world_footprint_extra_counts" in want:
        out["world_footprint_extra_counts"] = _count_artifact_dirs(ctx)

    # ── UTIL: system_sensor / network_monitor / timechain_v2 ─────────
    if "system_sensor_stats" in want:
        try:
            from titan_hcl.utils import system_sensor as _ss
            out["system_sensor_stats"] = _ss.get_all_stats()
        except Exception as _e:
            logger.debug("[OuterSourceAssembly] system_sensor: %s", _e)

    if "network_monitor_stats" in want:
        try:
            from titan_hcl.utils import network_monitor as _nm
            out["network_monitor_stats"] = _nm.get_all_stats(
                rpc_url=None, bus_stats=out.get("bus_stats"))
        except Exception as _e:
            logger.debug("[OuterSourceAssembly] network_monitor: %s", _e)

    if _need("tx_latency_stats", "block_delta_stats"):
        try:
            from titan_hcl.logic.timechain_v2 import (
                get_tx_latency_stats, get_block_delta_stats)
            if "tx_latency_stats" in want:
                out["tx_latency_stats"] = get_tx_latency_stats()
            if "block_delta_stats" in want:
                out["block_delta_stats"] = get_block_delta_stats()
        except Exception as _e:
            logger.debug("[OuterSourceAssembly] timechain_v2: %s", _e)

    # ── IN-PROCESS (via ctx): bus_stats / observatory counts / uptime ─
    if "bus_stats" in want and ctx.bus_stats_provider is not None:
        try:
            out["bus_stats"] = ctx.bus_stats_provider()
        except Exception as _e:
            logger.debug("[OuterSourceAssembly] bus_stats: %s", _e)

    if _need("art_count_100", "audio_count_100", "art_count_500",
             "audio_count_500", "text_count_500") and ctx.observatory_db is not None:
        odb = ctx.observatory_db
        try:
            if _need("art_count_100"):
                out["art_count_100"] = len(
                    odb.get_expressive_archive(type_="art", limit=100))
            if _need("audio_count_100"):
                out["audio_count_100"] = len(
                    odb.get_expressive_archive(type_="audio", limit=100))
            if _need("art_count_500"):
                out["art_count_500"] = len(
                    odb.get_expressive_archive(type_="art", limit=500))
            if _need("audio_count_500"):
                out["audio_count_500"] = len(
                    odb.get_expressive_archive(type_="audio", limit=500))
            if _need("text_count_500"):
                try:
                    out["text_count_500"] = len(
                        odb.get_expressive_archive(type_="text", limit=500))
                except Exception:
                    out["text_count_500"] = 0
        except Exception as _e:
            logger.debug("[OuterSourceAssembly] observatory counts: %s", _e)

    if "uptime_seconds" in want:
        out["uptime_seconds"] = time.time() - ctx.start_time

    # ── HEAVY (G20 in-process refresher cache via ctx) ───────────────
    for hk in ("inner_memory_stats", "social_x_gateway_stats",
               "events_teacher_stats", "community_engagement_stats"):
        if hk in want:
            v = ctx.heavy_stats.get(hk)
            if isinstance(v, dict):
                out[hk] = v

    # ── DERIVED ──────────────────────────────────────────────────────
    if "substrate_success_rate" in want:
        ag = out.get("agency_stats") or {}
        tot = float(ag.get("total_actions", 0) or 0)
        fail = float(ag.get("failed_actions", 0) or 0)
        if tot > 0:
            succ = (tot - fail) / tot
        else:
            et = out.get("expression_translator_stats") or {}
            et_tot = float(et.get("total_actions", 0) or 0)
            et_learn = float(et.get("learned_actions", 0) or 0)
            succ = (et_learn / et_tot) if et_tot > 0 else 0.7
        out["substrate_success_rate"] = max(0.0, min(1.0, succ))

    if "llm_avg_latency" in want:
        out["llm_avg_latency"] = 0.0

    return out
