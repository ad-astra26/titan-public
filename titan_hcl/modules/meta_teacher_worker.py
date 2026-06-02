"""
Meta-Teacher Worker — Guardian-managed module for philosophical critique
of meta-reasoning chains.

Subscribes to META_CHAIN_COMPLETE from spirit (meta_reasoning._conclude_chain),
samples which chains to critique (uncertainty + random + rate cap + domain
balance), calls Ollama Cloud light-tier model for critique, emits
META_TEACHER_FEEDBACK (→ chain_iql reward shaping) and META_TEACHER_GROUNDING
(→ meta_cgn β-posterior nudge).

Entry: meta_teacher_worker_main(recv_queue, send_queue, name, config)

rFP: titan-docs/rFP_titan_meta_reasoning_teacher.md §3, §4, §7
"""
import asyncio
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from queue import Empty
from typing import Optional
from titan_hcl.utils.silent_swallow import swallow_warn
from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)

CRITIQUES_JSONL_FILENAME = "critiques.jsonl"
ADOPTION_JSON_FILENAME = "adoption_metrics.json"

# Phase B (rFP_meta_teacher_v2) — Maker INFO channel.
# Non-actionable notifications surface here; ProposalStore reserved for
# action-required items (contract bundles, config changes, etc).
MAKER_INFO_FILENAME = "maker_info_log.jsonl"
MAKER_INFO_RETENTION_DAYS = 30
# Default 24h cadence per rFP §2 Phase B — override via config.
DEFAULT_STILL_NEEDS_PUSH_INFO_CADENCE_S = 24 * 3600.0
# Archival run cadence — once daily at most.
ARCHIVAL_CHECK_CADENCE_S = 24 * 3600.0
# Phase C (rFP_meta_teacher_v2) — voice self-assessment LLM call timeout.
DEFAULT_VOICE_SELF_ASSESS_TIMEOUT_S = 30.0


def _send_msg(send_queue, msg_type, src, dst, payload, rid=None):
    """Thin wrapper matching other workers' _send_msg pattern."""
    try:
        send_queue.put_nowait({
            "type": msg_type, "src": src, "dst": dst,
            "ts": time.time(), "rid": rid, "payload": payload,
        })
    except Exception:
        from titan_hcl.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


_last_hb_ts: float = 0.0


# Phase 11 §11.I.5 (Chunk 11N) — module-level readiness sentinel; gates
# SHM-slot heartbeat() (legacy bus heartbeat fires unconditionally for
# the boot window so guardian_HCL's stale-heartbeat detector doesn't
# kill a slow boot).
_WORKER_READY: bool = False


def _send_heartbeat(send_queue, name: str,
                    state_writer: Optional[object] = None) -> None:
    """Send MODULE_HEARTBEAT to Guardian (throttled to ≤1 per 3s).

    Phase 11 §11.I.5: also publishes state_writer.heartbeat() on the SHM
    slot once _WORKER_READY is True. SHM writes are best-effort.
    """
    global _last_hb_ts
    now = time.time()
    if now - _last_hb_ts < 3.0:
        return
    _last_hb_ts = now
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        rss_mb = 0
    try:
        send_queue.put_nowait({
            "type": "MODULE_HEARTBEAT", "src": name, "dst": "guardian",
            "ts": now, "rid": None,
            "payload": {"rss_mb": round(rss_mb, 1)},
        })
    except Exception:
        pass
    if state_writer is not None and _WORKER_READY:
        try:
            state_writer.heartbeat()
        except Exception:  # noqa: BLE001 — never crash the heartbeat
            pass


def _load_adoption_state(data_dir: str, current_version: int) -> dict:
    """Restore adoption_metrics from disk. Returns {} if absent/invalid.

    v2 (2026-04-24): version-aware. The persisted file may be either:
      - Legacy format: {"outer_spirit": 0.36, "inner_spirit": 0.19, ...}
        → treat as v1, reset on any non-1 current_version
      - Versioned format: {"_prompt_version": 2, "by_domain": {...}}
        → keep if version matches, reset otherwise

    Version-mismatch resets: the adoption metric's semantics change when
    SYSTEM_PROMPT_VERSION bumps (suggestions change from "full toolkit" to
    "missing-only"), so cross-version EMAs are incomparable. Better to
    reset to empty and let fresh data accumulate under the new semantics.
    """
    path = os.path.join(data_dir, "meta_teacher", ADOPTION_JSON_FILENAME)
    try:
        with open(path, "r") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return {}
        # Versioned format
        if "_prompt_version" in d:
            if int(d.get("_prompt_version", 0)) == current_version:
                raw = d.get("by_domain", {})
                if isinstance(raw, dict):
                    return {str(k): float(v) for k, v in raw.items()}
            else:
                logger.info(
                    "[MetaTeacher] adoption_metrics version mismatch "
                    "(saved=%s, current=%s) — resetting EMAs to empty",
                    d.get("_prompt_version"), current_version)
                return {}
        # Legacy flat format — only keep if current_version is 1
        if current_version == 1:
            return {str(k): float(v) for k, v in d.items()
                    if isinstance(v, (int, float))}
        logger.info(
            "[MetaTeacher] adoption_metrics legacy format found but "
            "current_version=%s — resetting EMAs to empty (v2 semantics "
            "are incomparable with v1 data)", current_version)
        return {}
    except Exception:
        return {}


def _save_adoption_state(
    data_dir: str, state: dict, current_version: int,
) -> None:
    """Persist adoption_metrics to disk (best-effort).

    v2: writes versioned format {"_prompt_version": N, "by_domain": {...}}
    so future loads can detect semantics mismatches and reset.
    """
    dir_path = os.path.join(data_dir, "meta_teacher")
    try:
        os.makedirs(dir_path, exist_ok=True)
        payload = {
            "_prompt_version": int(current_version),
            "by_domain": {str(k): float(v) for k, v in state.items()},
        }
        with open(os.path.join(dir_path, ADOPTION_JSON_FILENAME), "w") as f:
            json.dump(payload, f)
    except Exception as e:
        swallow_warn('[MetaTeacher] adoption save failed', e,
                     key="modules.meta_teacher_worker.adoption_save_failed", throttle=100)


def _append_maker_info_jsonl(data_dir: str, entry: dict) -> None:
    """Append one INFO row to data/meta_teacher/maker_info_log.jsonl.

    Phase B Maker INFO channel — non-actionable notifications (still_needs_push
    list, voice changes in Phase C, etc.) surface here for Observatory UI
    rendering. Rolling daily files with 30d retention; Maker sees on login.
    """
    dir_path = os.path.join(data_dir, "meta_teacher")
    try:
        os.makedirs(dir_path, exist_ok=True)
        date_tag = time.strftime("%Y%m%d", time.gmtime())
        path = os.path.join(
            dir_path, f"{MAKER_INFO_FILENAME.split('.')[0]}.{date_tag}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        cutoff_ts = time.time() - (MAKER_INFO_RETENTION_DAYS * 86400)
        for fname in os.listdir(dir_path):
            prefix = MAKER_INFO_FILENAME.split(".")[0] + "."
            if not fname.startswith(prefix) or not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(dir_path, fname)
            try:
                if os.path.getmtime(fpath) < cutoff_ts:
                    os.remove(fpath)
            except Exception:
                pass
    except Exception as e:
        logger.debug("[MetaTeacher] maker info jsonl append failed: %s", e)


def _append_critique_jsonl(data_dir: str, entry: dict, retention_days: int) -> None:
    """Append one critique entry. Rotation handled via daily files.

    Naming: critiques.YYYYMMDD.jsonl — old files beyond retention pruned.
    """
    dir_path = os.path.join(data_dir, "meta_teacher")
    try:
        os.makedirs(dir_path, exist_ok=True)
        date_tag = time.strftime("%Y%m%d", time.gmtime())
        path = os.path.join(dir_path, f"critiques.{date_tag}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        # Prune old files
        cutoff_ts = time.time() - (retention_days * 86400)
        for fname in os.listdir(dir_path):
            if not fname.startswith("critiques.") or not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(dir_path, fname)
            try:
                if os.path.getmtime(fpath) < cutoff_ts:
                    os.remove(fpath)
            except Exception:
                pass
    except Exception as e:
        logger.debug("[MetaTeacher] critique jsonl append failed: %s", e)


def _load_meta_teacher_llm_ctx(full_config: dict):
    """Phase 3 Chunk ψ (D-SPEC-88, 2026-05-18) — resolve LLM call context
    for /v4/llm-distill. Returns (api_base, internal_key, model) — model
    via `get_model_for_task("meta_teacher")` retained for parity with the
    old _load_ollama_client return shape. (None, None, None) when no
    internal_key configured.
    """
    try:
        from titan_hcl.inference import get_model_for_task
        api_cfg = full_config.get("api", {}) or {}
        internal_key = api_cfg.get("internal_key", "") or ""
        if not internal_key:
            # The worker's spawned `config` may not carry the [api] section
            # (module configs are partial). Fall back to the canonical merged
            # config (config.toml + ~/.titan/secrets.toml) — the key IS there.
            # Without this the meta-teacher's LLM critique silently noops:
            # every critique returns default score=0.50, reward_w=0, 0 primitives
            # (llm_ok=False) — i.e. the teacher "runs" but teaches nothing.
            try:
                from titan_hcl.config_loader import load_titan_config
                _canon_api = load_titan_config().get("api", {}) or {}
                internal_key = _canon_api.get("internal_key", "") or ""
                if internal_key:
                    api_cfg = _canon_api
            except Exception as _ck_err:
                logger.warning(
                    "[MetaTeacher] canonical config load failed: %s", _ck_err)
        if not internal_key:
            logger.warning(
                "[MetaTeacher] No api.internal_key — LLM calls will noop")
            return None, None, None
        api_port = int(api_cfg.get("port", 7777))
        api_base = f"http://127.0.0.1:{api_port}"
        model = get_model_for_task("meta_teacher")
        return api_base, internal_key, model
    except Exception as e:
        logger.error("[MetaTeacher] LLM context init failed: %s", e)
        return None, None, None


async def _call_llm(api_base, internal_key, model, system_prompt, user_prompt, timeout_s):
    """Invoke /v4/llm-distill. Returns text response or empty string on failure.

    Phase 3 Chunk ψ — replaces direct OllamaCloudClient.complete(...). All
    LLM traffic appears in llm_state.bin. Same temperature (0.3) +
    max_tokens (300) as before.
    """
    if not internal_key:
        return ""
    try:
        from titan_hcl.logic.llm_distill_client import distill_via_http_async
        return await distill_via_http_async(
            text=user_prompt,
            instruction=system_prompt,
            api_base=api_base,
            internal_key=internal_key,
            model=model,
            max_tokens=300,
            temperature=0.3,
            consumer="meta_teacher",
            timeout_s=timeout_s,
        )
    except Exception as e:
        logger.debug("[MetaTeacher] LLM call failed: %s", e)
        return ""


@with_error_envelope(module_name="meta_teacher", subsystem="entry", severity=_phase11_sev.FATAL)
def meta_teacher_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Meta-Teacher module process.

    Args:
        recv_queue: bus → worker
        send_queue: worker → bus
        name: module name ("meta_teacher")
        config: dict with keys from [meta_teacher] TOML section + inherited
                [inference] credentials + data_dir
    """
    global _WORKER_READY
    _WORKER_READY = False

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ── Phase 11 §11.I.5 (Chunk 11N) — SHM state-slot writer ──
    # Constructed BEFORE the slow MetaTeacher + Memory + Voice + Peer init
    # so the slot publishes state="starting" immediately. Heartbeats during
    # boot keep the slot's last_heartbeat fresh.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import (
            BootPriority,
            ModuleStateWriter,
        )
        _state_writer = ModuleStateWriter(
            module_name=name,
            layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[MetaTeacher] Phase 11 ModuleStateWriter init failed: %s",
            _sw_err)

    from titan_hcl.logic.meta_teacher import MetaTeacher, build_system_prompt
    from titan_hcl.logic.meta_teacher_prompts import (
        build_user_prompt, SYSTEM_PROMPT_VERSION,
    )
    # Phase B: two-tier memory (hot deque + cold per-topic_key journal).
    from titan_hcl.logic.meta_teacher_memory import (
        TeacherMemory, canonical_topic_key,
    )
    # Phase C: autonomous voice-tuning state.
    from titan_hcl.logic.meta_teacher_voice import TeacherVoice
    # Phase D.1: cross-Titan peer exchange client.
    from titan_hcl.logic.meta_teacher_peer import PeerExchangeClient

    data_dir = config.get("data_dir", "./data")
    logger.info(
        "[MetaTeacher] Initializing (data_dir=%s, enabled=%s, sample=%s, "
        "prompt_version=%s, memory=%s)",
        data_dir, config.get("enabled"), config.get("sample_mode"),
        SYSTEM_PROMPT_VERSION,
        bool(config.get("teaching_memory_enabled", False)))

    teacher = MetaTeacher(config)
    # Restore adoption EMAs from disk — version-aware: v1→v2→v3 bump resets.
    teacher.adoption_ema_by_domain.update(
        _load_adoption_state(data_dir, SYSTEM_PROMPT_VERSION))
    # rFP_teachers_update F5 (2026-05-26): bootstrap the 24h critique window
    # from disk so SHM-published dashboard stats are accurate from the first
    # heartbeat tick (steady-state appends in record_critique_entry).
    try:
        _bootstrap_n = teacher.bootstrap_24h_window(data_dir)
        if _bootstrap_n > 0:
            logger.info(
                "[MetaTeacher] 24h window bootstrapped: %d critiques loaded "
                "from disk (rFP_teachers_update F5)", _bootstrap_n)
    except Exception as _bs_err:
        logger.warning(
            "[MetaTeacher] bootstrap_24h_window failed (continuing with "
            "empty window): %s", _bs_err)

    # Phase B: TeacherMemory — hot/cold tiers. load() lazily reads journal.
    teacher_memory = TeacherMemory(config, data_dir=data_dir)
    if teacher_memory.enabled:
        try:
            teacher_memory.load()
        except Exception as _tm_err:
            logger.warning(
                "[MetaTeacher] TeacherMemory load failed (continuing disabled): %s",
                _tm_err)

    # Phase C: TeacherVoice — voice_state.json + signed journal. Always
    # constructed (snapshot endpoint reads even when disabled), but the
    # self-assess loop only runs when voice_tuning_enabled.
    teacher_voice = TeacherVoice(config, data_dir=data_dir)
    try:
        teacher_voice.load()
    except Exception as _tv_err:
        logger.warning(
            "[MetaTeacher] TeacherVoice load failed (continuing default): %s",
            _tv_err)

    # Phase D.1: PeerExchangeClient — cross-Titan teaching exchange. HTTP
    # transport (DivineBus is in-process only). Always constructed; outbound
    # queries gated behind peer_exchange_enabled.
    titan_id = str(config.get("titan_id") or os.environ.get(
        "TITAN_ID") or "t1").lower()
    peer_endpoints_cfg = config.get("peers", {}) or {}
    peer_client = PeerExchangeClient(
        config, data_dir=data_dir,
        my_titan_id=titan_id, peer_endpoints=peer_endpoints_cfg)
    try:
        peer_client.load()
    except Exception as _pe_err:
        logger.warning(
            "[MetaTeacher] PeerExchangeClient load failed (continuing default): %s",
            _pe_err)

    # ── Phase G (RFP_cgn_enhancements §9.3 / proto-SPEC §9.5c) — binding store ──
    # The teacher is the SOLE writer of reasoning_bindings.db (§G21): on each
    # critiqued chain it mints/refines a ReasoningBinding (context_signature →
    # recommended primitive) that the meta-reasoning policy reads READ-ONLY and
    # turns into a logit bias. It also applies the G.iv recognized/produced
    # counters reported by the policy in META_CHAIN_COMPLETE.binding_outcome.
    binding_store = None
    try:
        from titan_hcl.logic.reasoning_binding import ReasoningBindingStore
        binding_store = ReasoningBindingStore(
            db_path=os.path.join(data_dir, "meta_teacher", "reasoning_bindings.db"))
        logger.info(
            "[MetaTeacher] ReasoningBindingStore ready (Phase G — %d bindings)",
            binding_store.count())
    except Exception as _rb_err:
        logger.warning(
            "[MetaTeacher] ReasoningBindingStore init failed (Phase G disabled): %s",
            _rb_err)
        binding_store = None

    inference_cfg = config.get("inference", {}) or {}
    # Phase 3 Chunk ψ (D-SPEC-88, 2026-05-18) — LLM calls now route through
    # /v4/llm-distill. Old (llm_client, llm_model) replaced by
    # (llm_api_base, llm_internal_key, llm_model).
    llm_api_base, llm_internal_key, llm_model = _load_meta_teacher_llm_ctx(config)
    system_prompt = build_system_prompt()

    # Track pending suggestions per domain for adoption-signal: maps
    # domain -> deque of {chain_id, topic_key, suggested_primitives, ts}.
    # Phase B adds topic_key so adoption observed on follow-up chain
    # retroactively updates the correct cold entry.
    pending_suggestions: dict[str, deque] = {}
    retention_days = int(config.get("critique_log_retention_days", 7))
    _save_interval_s = 60.0
    _last_save_ts = time.time()

    # Phase B: still_needs_push INFO cadence state.
    info_cadence_s = float(config.get(
        "still_needs_push_info_cadence_seconds",
        DEFAULT_STILL_NEEDS_PUSH_INFO_CADENCE_S))
    last_info_ts: float = 0.0
    last_info_hash: str = ""
    last_archival_ts: float = 0.0

    # ── Phase A.4 (D-SPEC-70 v1.10.0) — meta_teacher_state.bin publisher ──
    # G21 single-writer; consumed by api_subprocess StateAccessor.meta_teacher
    # (replaces meta_teacher.stats bus-cache per Preamble G18).
    meta_teacher_state_publisher = None
    try:
        from titan_hcl.logic.meta_teacher_state_publisher import (
            MetaTeacherStatePublisher,
        )
        meta_teacher_state_publisher = MetaTeacherStatePublisher(
            titan_id=titan_id)
        meta_teacher_state_publisher.publish(teacher, memory=teacher_memory)
        logger.info(
            "[MetaTeacher] meta_teacher_state publisher attached "
            "(G21 single-writer; Phase A.4 / D-SPEC-70; "
            "rFP_teachers_update F5 dashboard payload included)")
    except Exception as _err:
        logger.warning(
            "[MetaTeacher] meta_teacher_state publisher init failed: %s — "
            "api_subprocess will read cold-boot stubs from meta_teacher_state",
            _err)

    # ── Background heartbeat thread ──────────────────────────────────
    _hb_stop = threading.Event()

    def _heartbeat_loop():
        while not _hb_stop.is_set():
            _send_heartbeat(send_queue, name, state_writer=_state_writer)
            # Phase A.4 — refresh meta_teacher_state.bin every heartbeat (30s)
            # so readers see fresh ts even if no critiques arrived recently.
            # rFP_teachers_update F5: also threads teacher_memory through so
            # the SHM payload carries cold-tier counts + still_needs_push.
            if meta_teacher_state_publisher is not None:
                try:
                    meta_teacher_state_publisher.publish(
                        teacher, memory=teacher_memory)
                except Exception:
                    pass
            _hb_stop.wait(30.0)

    hb_thread = threading.Thread(
        target=_heartbeat_loop, daemon=True, name="meta-teacher-heartbeat")
    hb_thread.start()

    # Single-threaded asyncio runner for LLM calls
    async def _critique_chain(payload: dict) -> tuple:
        """Returns (critique_dict|None, raw_response)."""
        if not llm_internal_key or llm_model is None:
            return None, ""
        try:
            user_prompt = build_user_prompt(payload)
        except Exception as e:
            logger.warning("[MetaTeacher] build_user_prompt failed: %s", e)
            return None, ""
        raw = await _call_llm(
            llm_api_base, llm_internal_key, llm_model,
            system_prompt, user_prompt,
            float(config.get("llm_timeout_s", 30.0)))
        # v2: pass used_primitives so parse_critique can defensively strip
        # any LLM suggestion that violates the "NOT USED only" rule.
        return teacher.parse_critique(
            raw, used_primitives=payload.get("primitives_used", []),
        ), raw

    def _maybe_emit_info(now: float) -> None:
        """Phase B: Emit still_needs_push INFO to Maker when:
        1. teaching memory is enabled, AND
        2. ≥ info_cadence_s since last emission, AND
        3. still_needs_push list hash has changed since last emission.

        Writes to data/meta_teacher/maker_info_log.YYYYMMDD.jsonl so the
        Observatory /v4/meta-teacher/maker-info endpoint can surface it.
        No bus message — the INFO channel is a log-tail surface, not a
        subscribe-worthy event (matches "non-actionable" semantics).
        """
        nonlocal last_info_ts, last_info_hash
        if not teacher_memory.enabled:
            return
        if (now - last_info_ts) < info_cadence_s:
            return
        try:
            cur_hash = teacher_memory.still_needs_push_hash()
        except Exception as e:
            logger.debug("[MetaTeacher] still_needs_push_hash failed: %s", e)
            return
        if cur_hash == last_info_hash:
            # Cadence elapsed but list unchanged — bump timestamp to spread
            # next check without emitting noise.
            last_info_ts = now
            return
        try:
            topics = teacher_memory.still_needs_push_list(limit=10)
        except Exception as e:
            logger.debug("[MetaTeacher] still_needs_push_list failed: %s", e)
            return
        info_entry = {
            "ts": now,
            "kind": "still_needs_push",
            "hash": cur_hash,
            "topics": topics,
            "prompt_version": SYSTEM_PROMPT_VERSION,
        }
        _append_maker_info_jsonl(data_dir, info_entry)
        last_info_ts = now
        last_info_hash = cur_hash
        logger.info(
            "[MetaTeacher] Emitted still_needs_push INFO (%d stuck topics, "
            "hash=%s)", len(topics), cur_hash)

    def _maybe_archive(now: float) -> None:
        """Run cold-tier archive sweep at most once per day."""
        nonlocal last_archival_ts
        if not teacher_memory.enabled:
            return
        if (now - last_archival_ts) < ARCHIVAL_CHECK_CADENCE_S:
            return
        last_archival_ts = now
        try:
            archived = teacher_memory.archive_inactive(now=now)
            if archived:
                logger.info(
                    "[MetaTeacher] Archived %d inactive topics (daily sweep)",
                    archived)
        except Exception as e:
            logger.debug("[MetaTeacher] archive_inactive failed: %s", e)

    # ── Phase C: voice self-assess helpers ───────────────────────────
    def _gather_voice_stats() -> dict:
        """Assemble adoption / quality / suggestion-frequency stats for the
        voice self-assessment LLM call.

        Pure read-side aggregation over teacher.adoption_ema_by_domain,
        teacher_memory cold tier, and teacher.recent_critiques. No bus
        round-trips, no DB calls.
        """
        adop = dict(teacher.adoption_ema_by_domain)
        # Quality delta by domain — average of recent cold-tier rows' deltas
        qd_by_domain: dict[str, list[float]] = {}
        try:
            if teacher_memory.enabled:
                for tk, cold in teacher_memory._cold.items():   # noqa: SLF001
                    crits = cold.get("quality_trajectory") or []
                    if not crits:
                        continue
                    # Use last-row's domain hint if present (cold doesn't
                    # store domain directly; infer from latest hot entry).
                    dom = "general"
                    qd_by_domain.setdefault(dom, []).append(
                        float(cold.get("quality_delta", 0.0)))
        except Exception as _e:
            logger.debug("[MetaTeacher] voice stats cold scan failed: %s", _e)
        qd_avg = {
            k: round(sum(v) / max(1, len(v)), 4)
            for k, v in qd_by_domain.items()}
        snp_topics: list[dict] = []
        snp_count = 0
        try:
            if teacher_memory.enabled:
                snp_topics = teacher_memory.still_needs_push_list(limit=5)
                snp_count = teacher_memory.snapshot().get(
                    "still_needs_push_count", 0)
        except Exception as _e:
            logger.debug("[MetaTeacher] voice stats SNP fetch failed: %s", _e)
        # Primitive suggestion frequency over recent critiques
        sug_freq: dict[str, int] = {}
        try:
            for c in list(teacher.recent_critiques)[-200:]:
                for p in (c.get("suggested_primitives") or []):
                    sug_freq[str(p)] = sug_freq.get(str(p), 0) + 1
        except Exception as _e:
            logger.debug("[MetaTeacher] voice stats freq scan failed: %s", _e)
        return {
            "adoption_by_domain": {k: round(float(v), 3)
                                    for k, v in adop.items()},
            "quality_delta_by_domain": qd_avg,
            "still_needs_push_count": snp_count,
            "still_needs_push_topics": [
                {"topic_key": t.get("topic_key"),
                 "n": int(t.get("critique_count", 0))} for t in snp_topics],
            "primitive_suggestion_freq": sug_freq,
            "current_biases": dict(
                teacher_voice.snapshot().get("domain_biases", {})),
        }

    async def _self_assess_voice() -> tuple:
        """Async LLM call for voice self-assessment. Returns (update|None, raw)."""
        if not llm_internal_key or llm_model is None:
            return None, ""
        stats = _gather_voice_stats()
        try:
            user_prompt = teacher_voice.build_self_assess_prompt(stats)
        except Exception as e:
            logger.warning("[MetaTeacher] build_self_assess_prompt failed: %s", e)
            return None, ""
        raw = await _call_llm(
            llm_api_base, llm_internal_key, llm_model,
            system_prompt, user_prompt,
            float(config.get(
                "voice_self_assess_timeout_s",
                DEFAULT_VOICE_SELF_ASSESS_TIMEOUT_S)))
        return teacher_voice.parse_self_assess_response(raw), raw

    def _maybe_run_voice_self_assess() -> None:
        """Run voice self-assess + apply + Maker INFO if rate-limit elapsed.

        Called from main-loop housekeeping. No-op when voice disabled or
        rate-limit not satisfied. On apply, emits an INFO row to
        maker_info_log.YYYYMMDD.jsonl with kind=voice_change.
        """
        if not teacher_voice.enabled:
            return
        if not teacher_voice.should_self_assess(teacher.total_observed):
            return
        try:
            update, raw = asyncio.run(_self_assess_voice())
        except Exception as e:
            logger.warning("[MetaTeacher] self_assess_voice failed: %s", e)
            return
        if update is None:
            logger.debug(
                "[MetaTeacher] voice self-assess returned no update (raw=%d chars)",
                len(raw or ""))
            return
        applied, reason = teacher_voice.apply_voice_update(update)
        if not applied:
            logger.info(
                "[MetaTeacher] voice update REJECTED: %s (reasoning=%r)",
                reason, str(update.get("reasoning") or "")[:80])
            return
        # Apply succeeded — emit Maker INFO + log.
        now = time.time()
        if teacher_voice.maker_info_due(now):
            info_entry = {
                "ts": now,
                "kind": "voice_change",
                "applied_count": teacher_voice.snapshot().get(
                    "applied_count", 0),
                "reasoning": str(update.get("reasoning") or "")[:200],
                "diff": {
                    "domain_bias": update.get("domain_bias"),
                    "style_hint": update.get("style_hint"),
                    "topic_suppression": update.get("topic_suppression"),
                },
                "after_hash": teacher_voice.snapshot().get(
                    "current_state_hash"),
                "prompt_version": SYSTEM_PROMPT_VERSION,
            }
            _append_maker_info_jsonl(data_dir, info_entry)
            teacher_voice.mark_maker_info_emitted(now)
        logger.info(
            "[MetaTeacher] voice update APPLIED (applied_count=%d, reasoning=%r)",
            teacher_voice.snapshot().get("applied_count", 0),
            str(update.get("reasoning") or "")[:80])

    # ── Phase 11 §11.I.2 — slot transition: starting → booted ──
    # (legacy boot-signal bus emit deleted per locked D2 / no-shim policy)
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[MetaTeacher] Phase 11 write_state(booted) failed: %s",
                _swb_err)
    logger.info(
        "[MetaTeacher] Ready — subscribed on META_CHAIN_COMPLETE "
        "(Phase 11 SHM slot=booted; awaiting MODULE_PROBE_REQUEST)")

    # ── Main loop ────────────────────────────────────────────────────
    # ── Microkernel v2 Phase B.1 §6 — readiness/hibernate reporter ──
    from titan_hcl.core.readiness_reporter import trivial_reporter
    def _b1_save_state():
        return []
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L2", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    while True:
        _send_heartbeat(send_queue, name, state_writer=_state_writer)

        # Periodic persistence + Phase B/C/D housekeeping (info + archive +
        # voice self-assess + peer query log rotation).
        now = time.time()
        if now - _last_save_ts >= _save_interval_s:
            _save_adoption_state(
                data_dir, dict(teacher.adoption_ema_by_domain),
                SYSTEM_PROMPT_VERSION)
            _maybe_emit_info(now)
            _maybe_archive(now)
            # Phase C: run voice self-assessment if eval_interval elapsed and
            # the rate-limit budget is full. Cheap when disabled (no-op).
            try:
                _maybe_run_voice_self_assess()
            except Exception as _ve:
                logger.debug(
                    "[MetaTeacher] voice self-assess tick failed: %s", _ve)
            # Phase D.1: rotate peer_query_log retention.
            try:
                if peer_client.enabled:
                    peer_client.prune_old_logs(now=now)
            except Exception as _pe:
                logger.debug(
                    "[MetaTeacher] peer_client.prune_old_logs failed: %s", _pe)
            _last_save_ts = now

        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            continue

        msg_type = msg.get("type", "")
        payload = msg.get("payload", {}) or {}

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ──
        if msg_type == bus.MODULE_PROBE_REQUEST and _state_writer is not None:
            try:
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request,
                )
                handle_module_probe_request(
                    msg,
                    probe_fn=None,
                    send_queue=send_queue,
                    module_name=name,
                    state_writer=_state_writer,
                )
            except Exception as _probe_err:  # noqa: BLE001
                logger.warning(
                    "[MetaTeacher] MODULE_PROBE_REQUEST handler failed: %s",
                    _probe_err)
            continue

        # ── Microkernel v2 Phase B.1 §6 — shadow swap dispatch ────
        if _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
        from titan_hcl.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[MetaTeacher] MODULE_SHUTDOWN — persisting + exiting")
            _save_adoption_state(
                data_dir, dict(teacher.adoption_ema_by_domain),
                SYSTEM_PROMPT_VERSION)
            _hb_stop.set()
            return

        if msg_type == bus.QUERY:
            qtype = payload.get("query_type", "")
            rid = msg.get("rid")
            if qtype == "get_meta_teacher_status":
                _send_msg(
                    send_queue, bus.QUERY_RESPONSE, name,
                    msg.get("src", ""), teacher.telemetry(), rid=rid)
            elif qtype == "get_meta_teacher_critiques":
                limit = int(payload.get("limit", 50))
                _send_msg(
                    send_queue, bus.QUERY_RESPONSE, name,
                    msg.get("src", ""),
                    {"critiques": list(teacher.recent_critiques)[-limit:]},
                    rid=rid)
            continue

        if msg_type != bus.META_CHAIN_COMPLETE:
            continue

        teacher.total_observed += 1

        # ── Phase G (RFP_cgn_enhancements §9.3 G.iv) — binding outcome counters ──
        # The policy reports, per chain, whether it CHOSE a matched binding's
        # recommended primitive (recognized) and whether the UNBIASED policy
        # already ranked it top (produced). Apply BEFORE the sampling gate so the
        # curriculum advances for every chain, not just critiqued ones. Teacher is
        # the sole writer (§G21) — the policy never writes the store.
        if binding_store is not None:
            _bo = payload.get("binding_outcome") or {}
            _bid = int(_bo.get("binding_id", -1))
            if _bid >= 0:
                try:
                    if _bo.get("produced"):
                        binding_store.record_produced(_bid)
                    elif _bo.get("recognized"):
                        binding_store.record_recognized(_bid)
                except Exception as _bo_err:
                    logger.debug(
                        "[MetaTeacher] binding outcome apply failed: %s", _bo_err)

        # Phase A: outer_summary / step_arguments may ride in the payload.
        outer_summary = payload.get("outer_summary")

        # Adoption signal: if we have a pending suggestion for this domain,
        # compare this chain's primitives to suggested ones.
        #
        # v2 (2026-04-24) semantics — suggested_primitives are now a set of
        # MISSING-from-prior-chain primitives (rFP §v2 bump). "Adopted" = True
        # iff Titan used at least ONE of those suggested-missing primitives
        # in a subsequent chain (non-empty intersection).
        #
        # v3 Phase B: when teaching memory is enabled, also retroactively
        # feed adoption into the cold-tier entry for that topic_key —
        # lets importance_weight track real learning signal over time.
        try:
            domain = str(payload.get("domain", "general"))
            dq = pending_suggestions.get(domain)
            if dq:
                sug = dq.popleft()  # match oldest pending first
                suggested = set(sug.get("suggested_primitives") or [])
                actual = set(payload.get("primitives_used", []))
                if suggested:
                    adopted = bool(suggested & actual)
                    teacher.update_adoption(domain, adopted)
                    # F1 (rFP_teachers_update): make adoption observable in
                    # real-time. Without this, the only signal of the loop's
                    # effect was the jsonl on disk (the worker logged
                    # Initializing/Ready/still_needs_push and went silent).
                    logger.info(
                        "[MetaTeacher] adoption: chain %s domain=%s applied=%s "
                        "(suggested=%d ∩ actual=%d) → adoption_ema=%.2f",
                        payload.get("chain_id", "?"), domain,
                        "YES" if adopted else "NO",
                        len(suggested), len(actual),
                        teacher.adoption_ema_by_domain.get(domain, 0.0))
                    # Phase B: feed adoption back to memory cold tier.
                    prior_topic_key = sug.get("topic_key")
                    if teacher_memory.enabled and prior_topic_key:
                        try:
                            teacher_memory.record_adoption(
                                prior_topic_key, adopted,
                                suggested_primitives=list(
                                    sug.get("suggested_primitives") or []),
                            )
                        except Exception as _tm_err:
                            logger.debug(
                                "[MetaTeacher] memory.record_adoption failed: %s",
                                _tm_err)
                    # Phase D.2: when this critique was informed by a peer
                    # query, credit/debit the peer-query policy. Quality
                    # delta isn't observable here (we only see one chain
                    # forward), so we approximate "negative outcome" as
                    # not-adopted. PeerExchangeClient owns the policy.
                    try:
                        if sug.get("peer_query_used"):
                            quality_delta = 0.0 if adopted else -0.05
                            peer_client.record_outcome(
                                chain_id=int(sug.get("chain_id", -1)),
                                domain=domain,
                                adopted=adopted,
                                quality_delta=quality_delta,
                            )
                    except Exception as _pq_err:
                        logger.debug(
                            "[MetaTeacher] peer.record_outcome failed: %s",
                            _pq_err)
                # else: empty suggestion → skip EMA update entirely
        except Exception as _ad_err:
            logger.debug("[MetaTeacher] adoption update error: %s", _ad_err)

        # Sampling gate
        sample_ok, reason = teacher.should_sample(payload)
        # F1 (rFP_teachers_update): emit the sampling decision so the loop is
        # observable in journalctl in real time. Previously the only signal
        # was the on-disk critiques jsonl — no live telemetry for whether
        # should_sample() was accepting or rate-limiting.
        logger.info(
            "[MetaTeacher] sample chain=%s domain=%s mode=%s reason=%s",
            payload.get("chain_id", "?"),
            payload.get("domain", "general"),
            "ACCEPT" if sample_ok else "SKIP", reason)
        if not sample_ok:
            continue

        # Phase B: retrieve similar prior critiques BEFORE calling LLM so the
        # teacher can see patterns. topic_key derived once; reused when the
        # critique is absorbed post-LLM + when pending suggestion is recorded.
        topic_key: Optional[str] = None
        memory_hits: list = []
        retrieved_ids: list = []
        if teacher_memory.enabled:
            try:
                topic_key = canonical_topic_key(
                    outer_summary,
                    primitives_used=payload.get("primitives_used"),
                    domain=payload.get("domain", ""),
                )
                memory_hits = teacher_memory.retrieve_similar(
                    topic_key, outer_summary)
                retrieved_ids = [h["topic_key"] for h in memory_hits]
                if memory_hits:
                    # prompt-builder reads from payload["_memory_hits"]
                    payload = dict(payload)
                    payload["_memory_hits"] = memory_hits
            except Exception as _rt_err:
                logger.debug(
                    "[MetaTeacher] memory.retrieve_similar failed: %s",
                    _rt_err)
        # Even when teaching memory is disabled, derive topic_key for Phase
        # C voice composition + Phase D.1 peer trigger gating below.
        if topic_key is None:
            try:
                topic_key = canonical_topic_key(
                    outer_summary,
                    primitives_used=payload.get("primitives_used"),
                    domain=payload.get("domain", ""),
                )
            except Exception:
                topic_key = None

        # Phase C: compose voice section for the LLM prompt (per-domain
        # biases + style hints + topic suppressions). Empty string when
        # voice is disabled or no relevant entries exist.
        try:
            voice_section = teacher_voice.compose_user_prompt_section(
                str(payload.get("domain", "general")), topic_key=topic_key)
            if voice_section:
                payload = dict(payload)
                payload["_voice_section"] = voice_section
        except Exception as _vs_err:
            logger.debug(
                "[MetaTeacher] voice.compose_user_prompt_section failed: %s",
                _vs_err)

        # Phase D.1: surface most-recent peer-teacher observation for this
        # topic. PeerExchangeClient caches inbound responses keyed by
        # topic_key; non-empty string is rendered into the user prompt as
        # "Peer teacher observation".
        try:
            if peer_client.enabled and topic_key:
                peer_obs = peer_client.format_recent_observation_for_topic(
                    topic_key)
                if peer_obs:
                    payload = dict(payload)
                    payload["_peer_observation"] = peer_obs
        except Exception as _po_err:
            logger.debug(
                "[MetaTeacher] peer.format_recent_observation_for_topic failed: %s",
                _po_err)

        # LLM call (async)
        try:
            critique, raw = asyncio.run(_critique_chain(payload))
        except Exception as e:
            logger.warning("[MetaTeacher] async critique failed: %s", e)
            critique, raw = None, ""

        teacher._record_sample(payload)

        fb_payload = teacher.build_feedback_payload(
            payload, critique, retrieved_context_ids=retrieved_ids or None)
        gr_payloads = teacher.build_grounding_payloads(payload, critique)

        # F1 (rFP_teachers_update): emit the critique result so the loop's
        # output is visible in real-time. quality_score + suggested_primitives
        # + llm_ok tell the operator at a glance whether the critique fired
        # and what it returned.
        _sug_count = len(fb_payload.get("suggested_primitives") or [])
        logger.info(
            "[MetaTeacher] critique chain=%s domain=%s score=%.2f reward_w=%.2f "
            "primitives_suggested=%d llm_ok=%s",
            payload.get("chain_id", "?"),
            payload.get("domain", "general"),
            float(fb_payload.get("quality_score", 0.0) or 0.0),
            float(fb_payload.get("reward_bonus", 0.0) or 0.0),
            _sug_count, bool(fb_payload.get("llm_ok", False)))

        # ── Phase G (RFP_cgn_enhancements §9.3 G.ii) — mint/refine binding ──
        # When the critique recommends a primitive AND the chain carried its
        # numeric context_signature, mint (or corroborate) a ReasoningBinding in
        # that cosine space: "in this inner context, the proper next primitive is
        # X." The policy reads these RO and biases toward them — the strong
        # channel the scalar reward_bonus (≤0.05) never provided. The legacy
        # META_TEACHER_FEEDBACK reward/suggestion path is KEPT below (complementary).
        if binding_store is not None:
            try:
                from titan_hcl.logic.reasoning_binding import BINDING_PRIMITIVES
                _ctx_sig = payload.get("context_signature")
                _suggested = fb_payload.get("suggested_primitives") or []
                if _ctx_sig and _suggested:
                    _rec_prim = str(_suggested[0]).split(".", 1)[0].upper()
                    if _rec_prim in BINDING_PRIMITIVES:
                        _principles = fb_payload.get("principles_invoked") or []
                        _cats = fb_payload.get("critique_categories") or []
                        _label = str(
                            (_principles[0] if _principles else
                             (_cats[0] if _cats else "teacher_recommendation")))[:64]
                        _sub = str(_suggested[0]).split(".", 1)[1] if "." in str(
                            _suggested[0]) else ""
                        _mid = binding_store.mint_or_refine(
                            _ctx_sig, _rec_prim, recommended_sub_action=_sub,
                            principle_label=_label)
                        logger.info(
                            "[MetaTeacher] binding mint/refine id=%s prim=%s "
                            "label=%s domain=%s (n_bindings=%d)",
                            _mid, _rec_prim, _label,
                            payload.get("domain", "general"), binding_store.count())
            except Exception as _mint_err:
                logger.debug(
                    "[MetaTeacher] binding mint failed: %s", _mint_err)

        # Persist-first, publish-second: write the critique to disk before
        # emitting bus messages. Guarantees that a crash between emit and
        # write can't lose the record, and fixes a test-race where downstream
        # observers (test harness, Observatory) saw feedback before the jsonl
        # existed.
        entry = {
            "ts": time.time(),
            "chain_id": int(payload.get("chain_id", 0)),
            "domain": str(payload.get("domain", "general")),
            "quality_score": fb_payload["quality_score"],
            "critique_categories": fb_payload["critique_categories"],
            "critique_text": fb_payload["critique_text"],
            "suggested_primitives": fb_payload["suggested_primitives"] or [],
            "confidence": fb_payload["confidence"],
            "reward_bonus": fb_payload["reward_bonus"],
            "principles_invoked": fb_payload["principles_invoked"],
            "llm_ok": fb_payload["llm_ok"],
            "sample_reason": reason,
            "context_summary": payload.get("context_summary") or {},
            "prompt_version": SYSTEM_PROMPT_VERSION,
        }
        if retrieved_ids:
            entry["retrieved_context_ids"] = list(retrieved_ids)
        if topic_key:
            entry["topic_key"] = topic_key
        teacher.record_critique_entry(entry)
        _append_critique_jsonl(data_dir, entry, retention_days)

        # Phase B: absorb the critique into teaching memory. Happens BEFORE
        # feedback emit so memory cold-tier update is on-disk before any
        # observer can read the critique entry.
        if teacher_memory.enabled and topic_key:
            try:
                teacher_memory.add_critique(entry, outer_summary)
            except Exception as _tm_err:
                logger.debug(
                    "[MetaTeacher] memory.add_critique failed: %s", _tm_err)

        # Phase C: bump rate-limit counter for voice self-assess. Always-on
        # increment (cheap when voice is disabled). Also remember whether
        # voice contributed to this critique — used for Maker INFO context
        # when voice fires; not exposed in feedback payload.
        try:
            teacher_voice.notify_critique()
        except Exception as _vc_err:
            logger.debug(
                "[MetaTeacher] voice.notify_critique failed: %s", _vc_err)

        # Phase D.1: maybe issue a peer query for this topic. Trigger fires
        # when the topic is on the still_needs_push list, the per-topic
        # cooldown has elapsed, the per-Titan rate limit has slack, AND the
        # peer policy gate (D.2) approves. Async HTTP call wrapped in
        # asyncio.run; failures are logged at DEBUG and never block the
        # critique loop.
        if (peer_client.enabled and topic_key
                and teacher_memory.enabled):
            try:
                cold_entry = teacher_memory._cold.get(topic_key, {})  # noqa: SLF001
                if cold_entry.get("still_needs_push"):
                    asyncio.run(
                        peer_client.maybe_issue_query(
                            topic_key=topic_key,
                            domain=str(payload.get("domain", "general")),
                            cold_entry=cold_entry,
                            chain_id=int(payload.get("chain_id", 0)),
                        )
                    )
            except Exception as _peq_err:
                logger.debug(
                    "[MetaTeacher] peer.maybe_issue_query failed: %s", _peq_err)

        # Publish META_TEACHER_FEEDBACK (dst="spirit" — routed to chain_iql)
        _send_msg(
            send_queue, bus.META_TEACHER_FEEDBACK, name, "spirit", fb_payload)

        # Publish per-primitive META_TEACHER_GROUNDING (dst="spirit" —
        # spirit_worker routes to meta_cgn.handle_teacher_grounding)
        for gp in gr_payloads:
            _send_msg(
                send_queue, bus.META_TEACHER_GROUNDING, name, "spirit", gp)

        # If there's a suggestion, queue it for adoption matching on a
        # subsequent same-domain chain. Phase B: include topic_key so memory
        # cold-tier adoption_trajectory lands on the correct entry. Phase
        # D.2: include peer_query_used flag so adoption-time reward shaping
        # can credit peer queries that landed.
        if fb_payload.get("suggested_primitives"):
            try:
                domain = str(payload.get("domain", "general"))
                peer_used = False
                try:
                    peer_used = bool(
                        peer_client.consume_peer_query_marker_for_chain(
                            int(payload.get("chain_id", 0))))
                except Exception:
                    peer_used = False
                pending_suggestions.setdefault(domain, deque(maxlen=20)).append({
                    "chain_id": int(payload.get("chain_id", 0)),
                    "topic_key": topic_key,
                    "suggested_primitives": list(fb_payload["suggested_primitives"]),
                    "peer_query_used": bool(peer_used),
                    "ts": time.time(),
                })
            except Exception:
                pass
