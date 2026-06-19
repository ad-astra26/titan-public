"""
social_worker_post_dispatch — Phase C-S9 chunk 9Q.

Owns the post-dispatch orchestration tick migrated out of
``spirit_worker.py:7770-8400``. Under
``microkernel.social_worker_enabled = true`` this module is the SOLE
driver of ``SocialXGateway.post()`` calls (`feedback_social_x_gateway_post_is_sole_sanctioned_x_path`).

Architecture per SPEC G18-G22:

  - All cross-process cognitive state is read from canonical Rust L0+L1
    SHM slots via ``ShmReaderBank`` (no sync bus.request, no in-process
    engine peeks across worker boundaries). One SHM read per tick per
    slot. Phase B.5 closure 2026-05-18 migrated this module off the
    retired Python-wrapper slots (consciousness_state /
    spirit_supplemental_state); canonical pattern mirrors
    SpiritAccessor.get_coordinator in state_accessor.py.
  - Bus is used for notifications only (``TIMECHAIN_COMMIT`` on success).
  - Gateway publish callback (``_x_post_published_callback`` wired in
    chunk 9F) emits ``X_POST_PUBLISHED`` after every successful post.
  - File-based delegate queue (``data/social_delegate_queue.json``)
    remains the canonical cross-Titan rotation channel — the same file
    spirit_worker used. social_worker reads/writes it identically.
  - F-phase pre-post ``meta_service_client.send_meta_request`` consultation
    runs each tick when a post is about to happen, with the same dry-run
    ``outcome_reward=0.0`` pattern as the legacy spirit_worker path (rFP
    §16.1 Session 1). Outcome emission closes the request_id after
    gateway.post() returns.
  - Mention discovery + reply cycle gated by 30-min cooldown (config:
    ``[social_x.replies].mention_check_cooldown_seconds``).
  - Catalyst clearing on verified/posted status — meter holds the
    consumable catalyst list internally per chunk 9I dual-mode pattern.

Tick cadence: 30s default (matches legacy ``_msl_tick_count % 30``
under 1Hz tick → ~30s; configurable via
``[social_x].post_dispatch_tick_interval_seconds``).

T1 path (l0_rust_enabled=false): post_context section + coordinator
subdict in ``spirit_supplemental_state.bin`` are populated by
spirit_worker. social_worker reads them here.

T3 path (l0_rust_enabled=true): same SHM slot, but the writer is
cognitive_worker via its call to ``start_snapshot_builder_threads``.
state_refs there carries pi_monitor / expression_manager / coordinator
identically (cognitive_worker.py:988-995). G21 holds — single writer
per Titan, mutually exclusive by flag.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from titan_hcl import bus
from titan_hcl.api.shm_reader_bank import ShmReaderBank
from titan_hcl.core.state_registry import (
    ensure_shm_root,
    resolve_titan_id,
)
from titan_hcl.logic.social_x_gateway import (
    ActionResult,
    BaseContext,
    PostContext,
    ReplyContext,
)
from titan_hcl.params import get_params
from titan_hcl.params import load_titan_params

logger = logging.getLogger(__name__)

# Default mention-discovery cooldown — matches legacy
# [social_x.replies].mention_check_cooldown_seconds in spirit_worker:8315-8316.
_DEFAULT_MENTION_COOLDOWN_S = 1800.0
# Default post-dispatch tick cadence (legacy ran every 30 1Hz ticks of MSL).
_DEFAULT_TICK_INTERVAL_S = 30.0
# Default delegate-queue path (legacy: ./data/social_delegate_queue.json).
_DEFAULT_DELEGATE_QUEUE = "data/social_delegate_queue.json"
# Default twitter_social user (canary value; config overrides).
_DEFAULT_X_USER = "your_x_handle"


class PostDispatchOrchestrator:
    """Single-tick post-dispatch orchestration for social_worker.

    Reads cognitive state from SHM, builds PostContext, calls
    SocialXGateway.post() (the SOLE sanctioned X path), handles delegate
    rotation, F-phase meta consultation, mention discovery + reply cycle,
    and TIMECHAIN_COMMIT emission on success.
    """

    def __init__(self, gateway, meter, titan_id: str, send_queue,
                 worker_name: str,
                 is_canonical_poller: bool = True,
                 api_base: str = "http://127.0.0.1:7777",
                 internal_key: str = "") -> None:
        self._gateway = gateway
        self._meter = meter
        self._titan_id = titan_id
        self._send_queue = send_queue
        self._worker_name = worker_name
        # Phase 3 Chunk ω-bis (D-SPEC-88, 2026-05-18) — composition routes
        # through POST {api_base}/v4/llm-distill (same path as out-of-kernel
        # cron callers). social_worker is L2 + subprocess; the FastAPI app
        # runs in api_subprocess. Using HTTP avoids reimplementing the bus
        # reply-poll pattern for what's a rare (1-2/hr) posting event.
        self._api_base = api_base.rstrip("/")
        self._internal_key = internal_key
        # Phase C-S9 chunk 9M: under fleet mode only the canonical poller
        # publishes MENTION_RECEIVED to other social_workers. Non-canonical
        # Titans receive them via _handle_polling_broadcast (chunk 9O).
        self._is_canonical_poller = bool(is_canonical_poller)
        # Tracks the last row timestamps we already broadcast, so we
        # only emit *_RECEIVED / *_CAPTURED / *_TAKEN for *newly*
        # inserted rows.
        self._last_mention_broadcast_ts: float = 0.0
        self._last_felt_experience_broadcast_ts: float = 0.0
        self._last_engagement_snapshot_broadcast_ts: float = 0.0

        # Canonical Rust L0+L1 SHM reads via ShmReaderBank (rFP
        # phase_c_state_read_unification Phase B.5 closure 2026-05-18).
        # Replaces the retired Python-wrapper slots
        # (consciousness_state / spirit_supplemental_state) — readers
        # below mirror the same canonical-slot pattern SpiritAccessor
        # uses in api_subprocess (state_accessor.py:201). Readers attach
        # lazily on first read; G18 (state via SHM) + G21 (single writer)
        # honored across all reads in this module.
        self._shm_root: Path = ensure_shm_root(titan_id)
        self._shm_bank: ShmReaderBank = ShmReaderBank(titan_id=titan_id)

        # Mention-cycle cooldown state — preserves across ticks (legacy
        # used _x_gateway._last_mention_check_ts; we keep it on the
        # orchestrator instance to avoid mutating the gateway).
        self._last_mention_check_ts: float = 0.0
        # rFP X-post PART C2 (INV-XEFF-4) — adaptive mention-poll backoff:
        # consecutive empty polls widen the effective cooldown (×2 each, capped),
        # any discovered mention resets it. ≈5% mention hit-rate ⇒ cuts most of
        # the mentions-poll spend during quiet periods. Maker-ratified 2026-06-03.
        self._mention_empty_streak: int = 0
        # rFP X-post PART B4 (INV-XENG-4) — organic auto-follow: periodic, gated,
        # DISABLED by default ([social_x.auto_follow].enabled). Grows the curated
        # following toward recurring high-relevance voices via gateway.follow().
        self._last_auto_follow_ts: float = 0.0
        from titan_hcl.logic.social_x.auto_follow import AutoFollowPolicy
        self._auto_follow = AutoFollowPolicy(
            gateway=gateway,
            social_x_db=getattr(gateway, "_db_path", "./data/social_x.db"))

        # F-phase pre-post pending request_ids → (sent_ts, kind).
        # Outcome emission pops these after gateway.post returns.
        self._meta_pending: dict[str, tuple[float, str]] = {}

        logger.info(
            "[PostDispatch] orchestrator initialized (titan_id=%s, "
            "shm_root=%s)", titan_id, self._shm_root)

    # ── Compose post text via /v4/llm-distill (D-SPEC-88 Chunk ω-bis) ──
    def _compose_post_text(self, descriptor) -> str:
        """Call POST {api_base}/v4/llm-distill to compose post text.

        Same endpoint cron callers use (events_teacher_run.py). Sync httpx
        is acceptable here — the orchestrator runs inside social_worker's
        sync tick loop, which fires once per few seconds. The LLM round-
        trip itself blocks for 5-45s, but X posts are 1-2 per hour so the
        absolute time spent in this method per Titan-day is bounded.

        Returns the composed text, or "" on failure (caller treats as
        generation_failed).
        """
        if not self._internal_key:
            logger.warning("[PostDispatch] no internal_key — cannot compose; "
                           "treating as generation_failed")
            return ""
        try:
            import httpx
            payload = {
                "text": descriptor.user_prompt,
                "instruction": descriptor.system_prompt,
                "max_tokens": descriptor.max_tokens,
                "temperature": descriptor.temperature,
                "model": "deepseek-v3.1:671b",
                "consumer": f"social_x_post.{self._worker_name}",
                "timeout_s": 45.0,
            }
            resp = httpx.post(
                f"{self._api_base}/v4/llm-distill",
                headers={"X-Titan-Internal-Key": self._internal_key,
                         "Content-Type": "application/json"},
                json=payload,
                timeout=50.0)
            body = resp.json()
            if body.get("status") != "ok":
                logger.warning(
                    "[PostDispatch] /v4/llm-distill status=%s error=%s",
                    body.get("status"), body.get("error"))
                return ""
            text = (body.get("text") or "").strip()
            # Strip wrapping quotes (matches the old _generate_text behavior).
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            return text
        except Exception as exc:
            logger.warning("[PostDispatch] compose raised: %s", exc)
            return ""

    # ── Canonical SHM read helpers (Phase B.5 D-SPEC-78) ────────────────

    def _read_neuromods_scalar(self) -> dict[str, float]:
        """Return ``{name: level}`` from canonical ``neuromod_state.bin``
        (Rust L0). ShmReaderBank schema:
        ``{modulators: {DA: {level, gain, phasic, tonic}, ...}}``.
        Returns empty dict if SHM unavailable (cold-boot tolerant)."""
        payload = self._shm_bank.read_neuromod() or {}
        modulators = payload.get("modulators") or {}
        return {
            name: float(entry.get("level", 0.5) or 0.5)
            for name, entry in modulators.items()
            if isinstance(entry, dict)
        }

    # ── Context build from SHM ──────────────────────────────────────────

    def _build_post_context(self, *, full_config: dict,
                            catalysts: list) -> PostContext | None:
        """Assemble a PostContext from canonical Rust L0+L1 SHM slots.

        Migrated 2026-05-18 (rFP_phase_c_state_read_unification Phase B.5
        closure — social_worker_post_dispatch was the missed consumer
        from Phase B.4's 8-site migration). Reads now follow the same
        canonical-slot pattern as SpiritAccessor.get_coordinator /
        SpiritAccessor.get_v4_state. The retired Python-wrapper slots
        (consciousness_state.bin + spirit_supplemental_state.bin) are
        no longer read; unified_spirit_metadata.bin reads come from the
        Rust-owned canonical writer (titan-unified-spirit-rs
        MetadataPublisher per Phase B.0).

        Returns None only on config-load failure; otherwise returns a
        context populated with whatever SHM slots could be read this
        tick (cold-boot fields zero/empty).
        """
        # Config blocks (loaded fresh per tick to pick up live config edits).
        tc = get_params("twitter_social") or {}
        inf = get_params("inference") or {}
        sage = get_params("stealth_sage") or {}
        session = tc.get("auth_session", "")
        proxy = tc.get("webshare_static_url", "")
        api_key = sage.get("twitterapi_io_key", "")

        # Canonical Rust L0+L1 + L2 reads via ShmReaderBank.
        bank = self._shm_bank
        chi_payload = bank.read_chi() or {}
        epoch_payload = bank.read_epoch() or {}
        unified = bank.read_unified_spirit_metadata() or {}
        reasoning_block = bank.read_reasoning_state() or {}
        meta_block = bank.read_meta_reasoning_state() or {}
        dreaming_block = bank.read_dream_state() or {}
        expression_payload = bank.read_expression_state() or {}
        mind = bank.read_mind_state() or {}
        # MSL self-model slot (msl_state.bin, published by cognitive_worker
        # via MSLStatePublisher). The CANONICAL source for I-confidence +
        # concept confidences — mind_state does NOT carry them (i_confidence
        # was read from `mind` → always 0.0 → posts publicly reported
        # "I-confidence: 0.000" while the real grounded value is ~0.95).
        # 2026-05-29 fix; same class as the sovereignty=0.0 live-signal gap.
        msl = bank.read_msl_state() or {}
        lang = bank.read_language_state() or {}
        social_perception = bank.read_social_perception_state() or {}
        # D-SPEC-85 v1.25.0 — consciousness_age slot carries Titan's
        # "main age" (lifetime self-observation tick counter). Distinct
        # from unified_spirit.epoch_count (slower GreatEpoch counter).
        consc_age = bank.read_consciousness_age() or {}

        # Neuromods — scalar {name: level} dict for downstream consumers
        # (SocialXGateway._build_state_signature reads .get(code, 0.5)).
        neuromods = self._read_neuromods_scalar()

        # Emotion — the canonical felt label is the EMOTIONAL CGN's current
        # state (emot-cgn), read via the emotion bundle (v3 affective region →
        # legacy human-readable label). This is the DEEP affective source the
        # rest of the gateway already consults for emotion context — not a cosine
        # match over raw neuromod levels, and not the old hardcoded "wonder" that
        # made EVERY post's footer report "◇ wonder" regardless of real state
        # (2026-06-18, Maker: "we've got emot-cgn for this"). Falls back to
        # "wonder" only when the emot reader is inactive (cold boot).
        emotion = "wonder"
        try:
            from titan_hcl.logic.emot_bundle_protocol import (
                read_full_emotion_context)
            _emo_ctx = read_full_emotion_context()
            if _emo_ctx and _emo_ctx.get("legacy_label"):
                emotion = str(_emo_ctx["legacy_label"]).lower()
        except Exception as _emo_err:
            logger.debug("[PostDispatch] emot-cgn emotion read failed: %s",
                         _emo_err)

        # Epoch — prefer unified_spirit.latest_epoch.epoch_id (Rust L1),
        # then unified_spirit.epoch_count, then Rust L0 epoch_counter.bin.
        # This is the GreatEpoch counter (nuance), NOT consciousness age.
        latest_epoch_block = unified.get("latest_epoch")
        if isinstance(latest_epoch_block, dict):
            epoch = int(latest_epoch_block.get("epoch_id", 0) or 0)
        else:
            epoch = 0
        if not epoch:
            epoch = int(unified.get("epoch_count", 0) or 0)
        if not epoch:
            epoch = int(epoch_payload.get("epoch", 0) or 0)

        # Consciousness age — lifetime self-observation tick counter
        # (Titan's "main age", canonical SHM source per D-SPEC-85).
        consciousness_age = int(consc_age.get("age_epochs", 0) or 0)

        # pi_ratio — was post_context.pi_ratio (retired). The π-pulse
        # cadence still lives in pi_heartbeat_state.bin (Rust L0); no
        # direct ratio is published. Leave 0.0 until a successor field
        # is defined; SocialXGateway._build_state_signature suppresses
        # it at 0.0.
        pi_ratio = 0.0

        grounded_words = list(lang.get("recent_words") or [])

        # Rich cognitive enrichment ─ canonical slots.
        chi = float(chi_payload.get("total", 0.0) or 0.0)
        drift = float(unified.get("last_drift", 0.0) or 0.0)
        trajectory = float(unified.get("last_trajectory", 0.0) or 0.0)

        i_confidence = float(msl.get("i_confidence", 0.0) or 0.0)
        concept_confidences = dict(
            msl.get("concept_confidences") or {})
        attention_entropy = float(
            mind.get("attention_entropy", 0.0) or 0.0)

        reasoning_chains = int(
            reasoning_block.get("active_chain_count", 0) or 0)
        # Commit rate: prefer meta_engine's persisted wisdom-save ratio
        # (mirrors legacy fallback chain).
        me_chains = int(meta_block.get("total_chains", 0) or 0)
        me_wisdom = int(meta_block.get("total_wisdom_saved", 0) or 0)
        re_chains = int(reasoning_block.get("total_chains", 0) or 0)
        re_concl = int(reasoning_block.get("total_conclusions", 0) or 0)
        if me_chains >= 20:
            reasoning_commit_rate = me_wisdom / me_chains
        elif re_chains >= 20:
            reasoning_commit_rate = re_concl / max(1, re_chains)
        else:
            reasoning_commit_rate = -1.0  # sentinel: suppress render

        recent_chain_summary = str(
            reasoning_block.get("last_conclusion") or "")[:100]
        meta_style = str(reasoning_block.get("dominant_primitive") or "")

        vocab_total = int(lang.get("vocab_total", 0) or 0)
        vocab_producible = int(lang.get("vocab_producible", 0) or 0)
        composition_level_raw = lang.get("composition_level", "")
        if isinstance(composition_level_raw, str) and \
                composition_level_raw.startswith("L"):
            try:
                composition_level = int(composition_level_raw[1:])
            except ValueError:
                composition_level = 0
        elif isinstance(composition_level_raw, (int, float)):
            composition_level = int(composition_level_raw)
        else:
            composition_level = 0

        # Expression fire counts — canonical expression_state.bin
        # (producer: expression_worker per SPEC §1 glossary / D-SPEC-53).
        recent_expression = dict(
            expression_payload.get("fire_counts") or {})

        # Social contagion — canonical social_perception_state.bin
        # (Session 4 publisher kept per Phase B.5 Maker greenlight).
        social_contagion = dict(
            social_perception.get("contagion_latest") or {})

        # Wisdom/growth counters from meta_reasoning + dreaming.
        total_eurekas = int(meta_block.get("total_eurekas", 0) or 0)
        total_wisdom_saved = int(
            meta_block.get("total_wisdom_saved", 0) or 0)
        distilled_count = int(
            dreaming_block.get("distilled_count", 0) or 0)
        meta_cgn_signals = int(
            meta_block.get("meta_cgn_signals", 0) or 0)
        crystallized_samples = list(
            meta_block.get("crystallized_samples") or [])[:3]

        creative_works_samples = self._fetch_creative_works_samples()

        # SOCIAL-MEMORY-ENRICHMENT (2026-05-26) — read persistent + mempool
        # counts from the canonical `memory_state.bin` SHM slot (G18,
        # MEMORY_STATE_SPEC, owned by memory_worker per G21). Free read —
        # the publisher refreshes ~1Hz, so it's always within freshness for
        # post cadence.
        memory_state = bank.read_memory_state() or {}
        memory_persistent_count = int(
            memory_state.get("persistent_count", 0) or 0)
        memory_mempool_size = int(memory_state.get("mempool_size", 0) or 0)

        return PostContext(
            session=session, proxy=proxy, api_key=api_key,
            titan_id=self._titan_id, emotion=emotion, neuromods=neuromods,
            epoch=epoch, consciousness_age=consciousness_age,
            pi_ratio=pi_ratio, grounded_words=grounded_words,
            llm_url=inf.get("ollama_cloud_base_url", ""),
            llm_key=inf.get("ollama_cloud_api_key", ""),
            llm_model=inf.get(
                "ollama_cloud_chat_model", "deepseek-v3.1:671b"),
            catalysts=list(catalysts),
            chi=chi, i_confidence=i_confidence,
            concept_confidences=concept_confidences,
            reasoning_chains=reasoning_chains,
            reasoning_commit_rate=reasoning_commit_rate,
            recent_chain_summary=recent_chain_summary,
            vocab_total=vocab_total, vocab_producible=vocab_producible,
            composition_level=composition_level,
            recent_words=grounded_words,
            meta_style=meta_style,
            recent_expression=recent_expression,
            drift=drift, trajectory=trajectory,
            attention_entropy=attention_entropy,
            social_contagion=social_contagion,
            total_eurekas=total_eurekas,
            total_wisdom_saved=total_wisdom_saved,
            distilled_count=distilled_count,
            meta_cgn_signals=meta_cgn_signals,
            crystallized_samples=crystallized_samples,
            creative_works_samples=creative_works_samples,
            memory_persistent_count=memory_persistent_count,
            memory_mempool_size=memory_mempool_size,
        )

    def _fetch_creative_works_samples(self) -> list:
        """Cheap read-only query against inner_memory.db for up to 3 recent
        creative_works rows. Mirrors spirit_worker:7976-7995 verbatim.
        Returns empty list on any DB error (table missing, DB locked, etc.)."""
        try:
            db_path = "data/inner_memory.db"
            if not os.path.exists(db_path):
                return []
            con = sqlite3.connect(db_path, timeout=2.0)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT work_type, timestamp, triggering_program, "
                "assessment_score, hormone_level_at_creation "
                "FROM creative_works ORDER BY timestamp DESC LIMIT 3"
            ).fetchall()
            con.close()
            return [dict(r) for r in rows]
        except Exception as _err:
            logger.debug(
                "[PostDispatch] creative_works fetch failed: %s", _err)
            return []

    # ── Bus publish helper (gateway → bus for grounding-gate
    #     CGN_KNOWLEDGE_REQ etc., mirrors spirit_worker's
    #     _x_gateway_bus_publish closure) ──

    def _bus_publish(self, msg: dict) -> None:
        """Forward gateway-emitted bus messages onto the worker's send_queue."""
        try:
            self._send_queue.put(msg)
        except Exception as _err:
            logger.debug(
                "[PostDispatch] bus forward failed: %s", _err)

    # ── F-phase pre-post + outcome emission ──

    def _meta_pre_post(self, full_config: dict,
                      catalyst_type: str) -> str:
        """Pre-post F-phase meta consultation (rFP §16.1 Session 1 dry-run).
        Same shape as spirit_worker:8140-8182 — fire-and-forget request
        with a 300ms time budget; outcome emission closes it after
        gateway.post() returns.

        Returns the request_id (empty string on failure)."""
        try:
            from titan_hcl.logic.meta_service_client import (
                send_meta_request as _sm_send,
            )
            from titan_hcl.logic.social_narrator import (
                build_social_meta_context_30d as _sm_build_ctx,
            )
            # Build neuromods/hormones/chi snapshot inputs from canonical
            # SHM slots (Phase B.5: chi_state.bin Rust L1 + neuromod_state.bin
            # Rust L0; retired spirit_supplemental_state.bin no longer read).
            # We pass None for hormones (legacy code only used it if
            # _cached_hormone_state was defined; harmless when missing).
            neuromods = self._read_neuromods_scalar()
            chi = self._shm_bank.read_chi() or {}
            ctx_vec = _sm_build_ctx(
                neuromods=neuromods, hormones=None, chi=chi)
            req_id = _sm_send(
                consumer_id="social",
                question_type="formulate_strategy",
                context_vector=ctx_vec,
                time_budget_ms=300,
                constraints={
                    "confidence_threshold": 0.4,
                    "allow_timechain_query": False,
                },
                payload_snippet=f"post catalyst={catalyst_type}",
                send_queue=self._send_queue,
                src=self._worker_name,
            )
            if req_id:
                self._meta_pending[req_id] = (time.time(), "post")
            return req_id or ""
        except Exception as _err:
            logger.debug(
                "[PostDispatch] meta pre-post skipped: %s", _err)
            return ""

    def _meta_outcome(self, req_id: str, status: str) -> None:
        """Outcome emission for F-phase request — dry-run reward=0.0."""
        if not req_id:
            return
        try:
            from titan_hcl.logic.meta_service_client import (
                send_meta_outcome as _sm_sink,
            )
            _sm_sink(
                request_id=req_id,
                consumer_id="social",
                outcome_reward=0.0,
                actual_primitive_used=None,
                context=f"session_1_dry_run status={status}",
                send_queue=self._send_queue,
                src=self._worker_name,
            )
            self._meta_pending.pop(req_id, None)
        except Exception as _err:
            logger.debug(
                "[PostDispatch] meta outcome skipped: %s", _err)

    # ── Delegate queue rotation (mirrors spirit_worker:8059-8127 +
    #     8253-8301) ──

    def _delegate_first_check(self) -> bool:
        """Read the latest verified-post titan_id from social_x.db; if
        the last post was T1, give the delegate queue priority this tick."""
        try:
            rot_db = self._gateway._db()
            row = rot_db.execute(
                "SELECT titan_id FROM actions WHERE action_type='post' "
                "AND status IN ('posted','verified') "
                "ORDER BY created_at DESC LIMIT 1").fetchone()
            rot_db.close()
            return bool(row and row["titan_id"] == "T1")
        except Exception:
            return False

    def _process_delegate_queue(self, *, full_config: dict,
                                pop_on_failure: bool) -> str | None:
        """Try to post the head of the delegate queue.

        Returns one of:
          - "verified"/"posted" on success (caller advances rotation)
          - other ActionResult.status on rate-limit/etc (entry kept)
          - None if queue is empty / can't be read
        """
        inf = get_params("inference") or {}
        tc = get_params("twitter_social") or {}
        sage = get_params("stealth_sage") or {}
        dq_file = _DEFAULT_DELEGATE_QUEUE
        try:
            if not os.path.exists(dq_file):
                return None
            with open(dq_file) as f:
                queue = json.load(f) or []
            if not queue:
                return None
            entry = queue[0]
            dq_titan = entry.get("titan_id", "T?")
            dq_consumer = f"delegate_{dq_titan}"
            dq_catalysts = entry.get("catalysts") or [{
                "type": entry.get("catalyst_type", "delegate"),
                "significance": 0.6,
                "content": f"Delegate from {dq_titan}",
                "data": {},
            }]
            ctx = PostContext(
                session=tc.get("auth_session", ""),
                proxy=tc.get("webshare_static_url", ""),
                api_key=sage.get("twitterapi_io_key", ""),
                titan_id=dq_titan,
                emotion=entry.get("emotion", "wonder"),
                neuromods=entry.get("neuromods", {}),
                epoch=int(entry.get("epoch", 0) or 0),
                pi_ratio=float(entry.get("pi_ratio", 0.0) or 0.0),
                grounded_words=list(entry.get("grounded_words") or []),
                llm_url=inf.get("ollama_cloud_base_url", ""),
                llm_key=inf.get("ollama_cloud_api_key", ""),
                llm_model=inf.get("ollama_cloud_chat_model", ""),
                catalysts=dq_catalysts,
            )
            # Phase 3 Chunk ω-bis (D-SPEC-88, 2026-05-18) — two-call shape.
            # 1) prepare_post() runs gates + archetype + grounding + prompt build.
            # 2) /v4/llm-distill composes the post text (via llm_worker bus).
            # 3) gateway.post() validates + transports.
            err, desc = self._gateway.prepare_post(
                ctx, consumer=dq_consumer, bus=self._bus_publish)
            if err is not None:
                result = err
            elif desc is not None and desc.post_type == "amplify":
                # AMPLIFY = native retweet, no LLM compose (Maker 2026-05-30).
                _arc = getattr(ctx, "archetype_candidate", None)
                _meta = getattr(_arc, "metadata", {}) or {}
                _target = str(_meta.get("retweet_target_id", "") or "")
                if not _target:
                    result = ActionResult(status="failed",
                                          reason="amplify_no_target")
                else:
                    result = self._gateway.retweet(
                        _target, ctx, consumer=dq_consumer,
                        author=str(_meta.get("author", "") or ""),
                        source_id=str(_meta.get("amplify_source_id", "") or ""))
            else:
                ctx.composed_text = self._compose_post_text(desc)
                if not ctx.composed_text:
                    result = ActionResult(
                        status="generation_failed",
                        reason="composer returned empty text (LLM timeout or error)")
                    self._gateway._log_telemetry({
                        "event": "post_generation_failed",
                        "titan_id": ctx.titan_id, "post_type": desc.post_type,
                    })
                else:
                    result = self._gateway.post(
                        ctx, consumer=dq_consumer, descriptor=desc,
                        bus=self._bus_publish)
            advance = False
            if result.status in ("verified", "posted", "unverified"):
                # 'unverified' = soft-failed but the tweet likely landed; advance
                # the delegate queue so it is NOT re-delegated (2026-06-13).
                advance = True
                logger.info(
                    "[PostDispatch] DELEGATE %s posted (%s): %s",
                    dq_titan, result.status, getattr(result, "tweet_id", ""))
            elif pop_on_failure and result.status in (
                    "api_failed", "generation_failed", "quality_rejected"):
                advance = True
                logger.warning(
                    "[PostDispatch] DELEGATE %s failed (%s) — popping",
                    dq_titan, result.status)
            elif result.status not in (
                    "disabled", "too_soon", "hourly_limit", "daily_limit",
                    "pending_exists"):
                logger.warning(
                    "[PostDispatch] DELEGATE %s post failed: %s — %s",
                    dq_titan, result.status, result.reason)
            if advance:
                queue = queue[1:]
                tmp = dq_file + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(queue, f)
                os.replace(tmp, dq_file)
            return result.status
        except Exception as _err:
            logger.warning(
                "[PostDispatch] delegate queue error: %s", _err)
            return None

    # ── Polling broadcast (chunk 9N) ──

    def _broadcast_new_mentions(self) -> int:
        """Phase C-S9 chunk 9N. After a discover_mentions cycle, emit
        MENTION_RECEIVED for each row inserted since the last broadcast
        so non-canonical Titans can write the row into their own local
        mention_tracking tables (chunk 9O consumer).

        Returns the count of MENTION_RECEIVED events emitted.

        Only the canonical poller broadcasts; non-canonical Titans
        receive these events. Idempotency on the consumer side is
        enforced by ``INSERT OR IGNORE INTO mention_tracking`` on the
        ``tweet_id`` UNIQUE constraint."""
        if not self._is_canonical_poller:
            return 0
        try:
            db = self._gateway._db()
            cutoff = self._last_mention_broadcast_ts
            rows = db.execute(
                "SELECT tweet_id, author, author_handle, text, our_post_id, "
                "titan_id, status, relevance_score, discovered_at "
                "FROM mention_tracking WHERE discovered_at > ? "
                "ORDER BY discovered_at ASC", (cutoff,)).fetchall()
            db.close()
        except Exception as _err:
            logger.debug(
                "[PostDispatch] mention broadcast read failed: %s", _err)
            return 0
        count = 0
        max_ts = self._last_mention_broadcast_ts
        for row in rows:
            try:
                row_dict = dict(row) if hasattr(row, "keys") else {
                    "tweet_id": row[0], "author": row[1],
                    "author_handle": row[2], "text": row[3],
                    "our_post_id": row[4], "titan_id": row[5],
                    "status": row[6], "relevance_score": row[7],
                    "discovered_at": row[8],
                }
                self._send_queue.put({
                    "type": bus.MENTION_RECEIVED,
                    "src": self._worker_name,
                    "dst": "all",
                    "payload": row_dict,
                    "ts": time.time(),
                })
                ts = float(row_dict.get("discovered_at", 0) or 0)
                if ts > max_ts:
                    max_ts = ts
                count += 1
            except Exception as _err:
                logger.debug(
                    "[PostDispatch] MENTION_RECEIVED publish failed: %s",
                    _err)
        self._last_mention_broadcast_ts = max_ts
        if count > 0:
            logger.info(
                "[PostDispatch] broadcast %d MENTION_RECEIVED events "
                "(canonical poller=T_TEST? %s)", count,
                self._titan_id)
        return count

    def _broadcast_new_felt_experiences(self) -> int:
        """Phase C-S9 chunk 9N. Poll events_teacher.db for felt_experiences
        rows inserted since the last broadcast and emit
        FELT_EXPERIENCE_CAPTURED for each so non-canonical Titans can
        write them into their own local DBs (chunk 9O consumer).

        Reads-only against events_teacher.db (canonical writer is
        EventsTeacher itself which is cron-based + uses its own
        ``events_teacher_writer`` daemon for writes — we never write
        here, so G21 holds). Only the canonical poller broadcasts."""
        if not self._is_canonical_poller:
            return 0
        try:
            con = sqlite3.connect("data/events_teacher.db", timeout=2.0)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT id, titan_id, source, author, topic, sentiment, "
                "arousal, relevance, concept_signals, semantic_concepts, "
                "felt_summary, contagion_type, mode, window_id, "
                "created_at FROM felt_experiences WHERE created_at > ? "
                "ORDER BY created_at ASC LIMIT 50",
                (self._last_felt_experience_broadcast_ts,)).fetchall()
            con.close()
        except Exception as _err:
            logger.debug(
                "[PostDispatch] felt_experiences broadcast read failed: %s",
                _err)
            return 0
        count = 0
        max_ts = self._last_felt_experience_broadcast_ts
        for row in rows:
            try:
                payload = dict(row)
                self._send_queue.put({
                    "type": bus.FELT_EXPERIENCE_CAPTURED,
                    "src": self._worker_name,
                    "dst": "all",
                    "payload": payload,
                    "ts": time.time(),
                })
                ts = float(payload.get("created_at", 0) or 0)
                if ts > max_ts:
                    max_ts = ts
                count += 1
            except Exception as _err:
                logger.debug(
                    "[PostDispatch] FELT_EXPERIENCE_CAPTURED publish "
                    "failed: %s", _err)
        self._last_felt_experience_broadcast_ts = max_ts
        if count > 0:
            logger.info(
                "[PostDispatch] broadcast %d FELT_EXPERIENCE_CAPTURED "
                "events", count)
        return count

    def _broadcast_new_engagement_snapshots(self) -> int:
        """Phase C-S9 chunk 9N. Poll events_teacher.db for engagement
        snapshots inserted since the last broadcast and emit
        ENGAGEMENT_SNAPSHOT_TAKEN for each so non-canonical Titans can
        write them into their own local DBs."""
        if not self._is_canonical_poller:
            return 0
        try:
            con = sqlite3.connect("data/events_teacher.db", timeout=2.0)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT id, titan_id, tweet_id, likes, replies, quotes, "
                "delta_likes, delta_replies, delta_quotes, checked_at "
                "FROM engagement_snapshots WHERE checked_at > ? "
                "ORDER BY checked_at ASC LIMIT 50",
                (self._last_engagement_snapshot_broadcast_ts,)
            ).fetchall()
            con.close()
        except Exception as _err:
            logger.debug(
                "[PostDispatch] engagement_snapshots broadcast read "
                "failed: %s", _err)
            return 0
        count = 0
        max_ts = self._last_engagement_snapshot_broadcast_ts
        for row in rows:
            try:
                payload = dict(row)
                self._send_queue.put({
                    "type": bus.ENGAGEMENT_SNAPSHOT_TAKEN,
                    "src": self._worker_name,
                    "dst": "all",
                    "payload": payload,
                    "ts": time.time(),
                })
                ts = float(payload.get("checked_at", 0) or 0)
                if ts > max_ts:
                    max_ts = ts
                count += 1
            except Exception as _err:
                logger.debug(
                    "[PostDispatch] ENGAGEMENT_SNAPSHOT_TAKEN publish "
                    "failed: %s", _err)
        self._last_engagement_snapshot_broadcast_ts = max_ts
        if count > 0:
            logger.info(
                "[PostDispatch] broadcast %d ENGAGEMENT_SNAPSHOT_TAKEN "
                "events", count)
        return count

    # ── Mention discovery + reply cycle ──

    def _run_mention_cycle(self, *, full_config: dict,
                           grounded_words: list, neuromods: dict,
                           emotion: str, now: float) -> int:
        """30-min-cooldown mention discovery + reply loop. Mirrors
        spirit_worker:8303-8390. Returns number of replies posted."""
        sx_cfg = get_params("social_x") or {}
        replies_cfg = sx_cfg.get("replies") or {}
        # INV-XEFF (2026-06-18): the 3 Titans share ONE @your_x_handle account, so
        # 3 independent mention polls = 3× the paid /twitter/user/mentions reads
        # of the SAME inbox. Collapse to a SINGLE poller — the deterministic
        # owner of the account handle under the same fleet partition used for
        # tagging (drift-proof: matching rosters → exactly one box polls). That
        # owner then replies to ALL mentions (the per-mention reply partition is
        # moot once a single box owns the shared inbox — see the reply loop).
        self._mention_single_owner = bool(
            sx_cfg.get("mention_poll_single_owner", True))
        if self._mention_single_owner:
            from titan_hcl.logic.social_x.archetypes.base import (
                engagement_owner_for)
            _roster = sx_cfg.get("engagement_fleet") or ["T1", "T2", "T3"]
            _acct = (get_params("twitter_social") or {}).get(
                "user_name") or "your_x_handle"
            _acct_owner = engagement_owner_for(_acct, _roster)
            if _acct_owner and _acct_owner != self._titan_id:
                return 0  # another box owns the shared-account mention poll
        base_cooldown = float(replies_cfg.get(
            "mention_check_cooldown_seconds",
            _DEFAULT_MENTION_COOLDOWN_S))
        # C2 (INV-XEFF-4): adaptive backoff. Effective cooldown widens with the
        # consecutive-empty streak (×2 each) up to a cap, then resets to base on
        # the next discovered mention. Bounds responsiveness (cap) and the streak
        # exponent (no runaway bigint). enabled=true by default; emergent, not a
        # static cron cut (INV-XEFF-5).
        backoff_on = bool(replies_cfg.get("mention_backoff_enabled", True))
        cap = float(replies_cfg.get("mention_backoff_cap_seconds", 7200.0))
        if backoff_on:
            effective_cooldown = min(
                base_cooldown * (2 ** min(self._mention_empty_streak, 16)),
                cap)
        else:
            effective_cooldown = base_cooldown
        if now - self._last_mention_check_ts < effective_cooldown:
            return 0
        self._last_mention_check_ts = now

        tc = get_params("twitter_social") or {}
        inf = get_params("inference") or {}
        sage = get_params("stealth_sage") or {}
        max_replies = int(replies_cfg.get("max_replies_per_cycle", 3))
        reply_count = 0
        try:
            base = BaseContext(
                session=tc.get("auth_session", ""),
                proxy=tc.get("webshare_static_url", ""),
                api_key=sage.get("twitterapi_io_key", ""),
                titan_id="")  # empty → discover for ALL Titans
            mentions = self._gateway.discover_mentions(
                base, consumer=self._worker_name,
                grounded_words=grounded_words) or []
            # C2: any discovered mention resets the backoff to base (stay
            # responsive while conversation is live); an empty poll widens it.
            if mentions:
                self._mention_empty_streak = 0
            else:
                self._mention_empty_streak = min(
                    self._mention_empty_streak + 1, 16)
                logger.debug(
                    "[PostDispatch] Mention poll empty (streak=%d) — next "
                    "poll backed off toward %.0fs cap",
                    self._mention_empty_streak,
                    float((get_params("social_x") or {}).get(
                        "replies", {}).get("mention_backoff_cap_seconds", 7200.0)))
            # Fleet reply partition (RFP_fleet_x_engagement_coordination
            # INV-FX-7): only the Titan that OWNS a mention's author replies to
            # it, so the shared @your_x_handle account never double-replies to one
            # person across T1/T2/T3. Same deterministic author-hash as proactive
            # engagement; zero coordination.
            _sx_cfg = get_params("social_x") or {}
            # When single-owner polling is on, THIS box is the sole poller of the
            # shared inbox, so it must reply to ALL mentions — the per-mention
            # author partition (which existed to stop 3 boxes double-replying) is
            # redundant and would silently drop every mention not hashing to this
            # box (they'd never be answered, since no other box polls).
            _reply_partition_on = (
                not getattr(self, "_mention_single_owner", False)
                and bool(_sx_cfg.get("engagement_partition_enabled", True)))
            _reply_roster = _sx_cfg.get("engagement_fleet") or ["T1", "T2", "T3"]
            from titan_hcl.logic.social_x.archetypes.base import (
                engagement_owner_for,
            )
            for m in mentions[:max_replies]:
                if _reply_partition_on:
                    _owner = engagement_owner_for(
                        m.get("author_handle") or m.get("author") or "",
                        _reply_roster)
                    if _owner and _owner != self._titan_id:
                        continue
                cgn_action: dict = {}
                cgn_client = None
                # §7.D-A2: keep the inference inputs so we can record the matching
                # IQL transition after the reply (state must equal what was inferred).
                _social_sctx = {"epoch": 0, "neuromods": neuromods}
                _social_feats = {
                    "familiarity": min(1.0, m.get("relevance_score", 0.3)),
                    "interaction_count": 0,
                    "social_valence": 0.0,
                    "mention_count": 1,
                }
                try:
                    from titan_hcl.logic.cgn_consumer_client import (
                        CGNConsumerClient,
                    )
                    cgn_client = CGNConsumerClient(
                        "social", state_dir="data/cgn")
                    cgn_action = cgn_client.infer_action(
                        sensory_ctx=_social_sctx, features=_social_feats)
                except Exception:
                    pass
                rctx = ReplyContext(
                    session=tc.get("auth_session", ""),
                    proxy=tc.get("webshare_static_url", ""),
                    api_key=sage.get("twitterapi_io_key", ""),
                    titan_id=m["titan_id"],
                    reply_to_tweet_id=m["tweet_id"],
                    mention_text=m["text"][:200],
                    mention_user=m["author_handle"],
                    emotion=emotion,
                    neuromods=neuromods,
                    grounded_words=grounded_words,
                    llm_url=inf.get("ollama_cloud_base_url", ""),
                    llm_key=inf.get("ollama_cloud_api_key", ""),
                    llm_model=inf.get("ollama_cloud_chat_model", ""),
                    cgn_action=cgn_action,
                )
                rr = self._gateway.reply(
                    rctx, consumer=self._worker_name)
                if rr.status in ("posted", "verified"):
                    self._gateway.mark_mention_replied(
                        m["tweet_id"], rr.tweet_id)
                    reply_count += 1
                    # §7.D-A2: close the social CGN-IQL loop — social INFERRED an
                    # engage action above but never recorded the outcome, so its
                    # consumer learned nothing (formed=0, fed=0). Record the
                    # (perception state, inferred action, reward) as a single-shot
                    # experience (state built identically to inference via the
                    # shared helper). ⚠ Reward here is PROXIMAL — relevance-weighted
                    # successful engagement; the TRUE reward is the delayed
                    # `engagement_reciprocity` (replies/likes), which needs a
                    # tweet_id-keyed deferred-reward path (documented refinement,
                    # RFP §7.D-A2 social). This first wire gives the social IQL real
                    # state→action coverage; the delayed reward sharpens the policy.
                    if cgn_client is not None:
                        try:
                            _rel = float(m.get("relevance_score", 0.3) or 0.3)
                            cgn_client.record_experience(
                                _social_sctx, _social_feats,
                                action=int(cgn_action.get("action_index", 1)),
                                reward=0.2 + 0.5 * _rel,
                                concept_id="social_"
                                + str(m.get("author_handle", ""))[:30])
                        except Exception:
                            pass
                    logger.info(
                        "[PostDispatch] Replied to @%s (score=%.2f): %s",
                        m["author_handle"], m["relevance_score"],
                        rr.tweet_id)
                elif rr.status in (
                        "hourly_limit", "daily_limit", "too_soon"):
                    break
            if reply_count > 0:
                logger.info(
                    "[PostDispatch] Mention cycle: %d replies",
                    reply_count)
        except Exception as _err:
            logger.warning(
                "[PostDispatch] mention cycle error: %s", _err)
        return reply_count

    # ── TIMECHAIN_COMMIT emission on successful post ──

    def _emit_timechain_commit(self, result, full_config: dict) -> None:
        """Fork an episodic-thought TimeChain entry for a successful post.
        Mirrors spirit_worker:8224-8240 verbatim. Best-effort; never
        raises."""
        try:
            tweet_id = getattr(result, "tweet_id", "") or ""
            text = getattr(result, "text", "") or ""
            text_hash = hashlib.sha256(
                text.encode("utf-8")).hexdigest()[:16]
            # neuromod / chi snapshots — read fresh from canonical SHM
            # for the TimeChain entry (Phase B.5: chi_state.bin Rust L1
            # + neuromod_state.bin Rust L0; retired
            # spirit_supplemental_state.bin no longer read).
            neuromods = self._read_neuromods_scalar()
            chi_block = self._shm_bank.read_chi() or {}
            self._send_queue.put({
                "type": bus.TIMECHAIN_COMMIT,
                "src": self._worker_name,
                "dst": "timechain",
                "payload": {
                    "fork": "episodic",
                    "thought_type": "episodic",
                    "source": "social_post",
                    "content": {
                        "action": "post",
                        "tweet_id": tweet_id,
                        "titan_id": self._titan_id,
                        "text_hash": text_hash,
                    },
                    "significance": 0.5,
                    "novelty": 0.5,
                    "coherence": 0.5,
                    "tags": ["social", "x_post", self._titan_id],
                    "db_ref": f"social_x:{tweet_id}",
                    "neuromods": dict(neuromods),
                    "chi_available": float(
                        chi_block.get("total", 0.5) or 0.5),
                    "attention": 0.5,
                    "i_confidence": 0.5,
                    "chi_coherence": 0.3,
                },
                "ts": time.time(),
            })
        except Exception as _err:
            logger.debug(
                "[PostDispatch] TIMECHAIN_COMMIT emit failed: %s", _err)

    # ── Main tick entrypoint ──

    def _maybe_auto_follow(self, *, ctx) -> None:
        """rFP PART B4: periodic organic auto-follow, gated + OFF by default.

        Cheap no-op unless `[social_x.auto_follow].enabled` is true; then runs at
        most once per `check_interval_s` (default 1h). Never raises into the tick.
        """
        try:
            sx_cfg = self._gateway._load_config()
            af = sx_cfg.get("auto_follow", {}) or {}
            if not af.get("enabled", False):
                return
            now = time.time()
            interval = float(af.get("check_interval_s", 3600))
            if now - self._last_auto_follow_ts < interval:
                return
            self._last_auto_follow_ts = now
            n = self._auto_follow.run(
                titan_id=ctx.titan_id, context=ctx, config=sx_cfg)
            if n:
                logger.info("[PostDispatch] auto-follow: %d new follow(s)", n)
        except Exception as _err:
            logger.warning("[PostDispatch] auto-follow tick failed: %s", _err)

    def run_tick(self) -> None:
        """One post-dispatch orchestration tick.

        Sequence (mirrors spirit_worker:7772-8400):
          1. Load fresh config + secrets.
          2. Drain the meter's consumable catalyst list.
          3. Decide delegate-first vs T1-own.
          4. (delegate-first) Process delegate queue head.
          5. (else) F-phase pre-post → gateway.post() → outcome emit.
          6. On verified/posted: clear meter catalysts + emit
             TIMECHAIN_COMMIT + msl.signal_action proxy.
          7. Open social-window if post wasn't no-op.
          8. (delegate-second) Process delegate queue head if rotation
             didn't already.
          9. Mention discovery + reply cycle (30-min cooldown).
        """
        try:
            full_config = load_titan_params()
        except Exception as _err:
            logger.warning(
                "[PostDispatch] config load failed — skipping tick: %s",
                _err)
            return

        # Drain catalysts the meter has accumulated since last tick.
        # The meter's drain method clears its internal list and returns
        # the dicts for inclusion in PostContext (mirrors legacy
        # _x_catalysts.append → list(_x_catalysts) handoff).
        catalysts = self._drain_meter_catalysts()

        ctx = self._build_post_context(
            full_config=full_config, catalysts=catalysts)
        if ctx is None:
            return  # config load failed earlier; abort tick

        # ── Organic auto-follow (rFP PART B4) — periodic, gated, OFF by default ──
        self._maybe_auto_follow(ctx=ctx)

        # ── Delegate-first rotation decision ──
        delegate_first = self._delegate_first_check()

        delegate_posted = False
        if delegate_first:
            status = self._process_delegate_queue(
                full_config=full_config, pop_on_failure=False)
            delegate_posted = status in ("verified", "posted")

        # ── T1 own post path ──
        result = None
        force_ungrounded = any(
            bool(c.get("force_ungrounded")) for c in catalysts)
        if not delegate_posted:
            catalyst_type = catalysts[0].get("type", "") if catalysts else ""
            req_id = self._meta_pre_post(full_config, catalyst_type)
            try:
                # Phase 3 Chunk ω-bis (D-SPEC-88, 2026-05-18) — two-call shape.
                err, desc = self._gateway.prepare_post(
                    ctx, consumer=self._worker_name,
                    bus=self._bus_publish,
                    force_ungrounded=force_ungrounded)
                if err is not None:
                    result = err
                elif desc is not None and desc.post_type == "amplify":
                    # AMPLIFY = native retweet, no LLM compose (Maker
                    # 2026-05-30). The dispatcher picked the amplify archetype;
                    # short-circuit to a retweet of the target post. The action
                    # row written by gateway.retweet() feeds the per-author
                    # cooldown + amplify dedup.
                    _arc = getattr(ctx, "archetype_candidate", None)
                    _meta = getattr(_arc, "metadata", {}) or {}
                    _target = str(_meta.get("retweet_target_id", "") or "")
                    if not _target:
                        result = ActionResult(status="failed",
                                              reason="amplify_no_target")
                    else:
                        result = self._gateway.retweet(
                            _target, ctx, consumer=self._worker_name,
                            author=str(_meta.get("author", "") or ""),
                            source_id=str(_meta.get("amplify_source_id", "") or ""))
                else:
                    ctx.composed_text = self._compose_post_text(desc)
                    if not ctx.composed_text:
                        result = ActionResult(
                            status="generation_failed",
                            reason="composer returned empty text (LLM timeout or error)")
                        self._gateway._log_telemetry({
                            "event": "post_generation_failed",
                            "titan_id": ctx.titan_id, "post_type": desc.post_type,
                        })
                    else:
                        result = self._gateway.post(
                            ctx, consumer=self._worker_name, descriptor=desc,
                            bus=self._bus_publish,
                            force_ungrounded=force_ungrounded)
            except Exception as _err:
                logger.warning(
                    "[PostDispatch] gateway.post raised: %s", _err)
                result = None
            status = getattr(result, "status", "error") if result else "error"
            self._meta_outcome(req_id, status)

            if result is not None and result.status in (
                    "verified", "posted"):
                logger.info(
                    "[PostDispatch] posted via gateway: %s (id=%s)",
                    result.status,
                    getattr(result, "tweet_id", ""))
                # Clear catalysts on success (meter holds them).
                self._clear_meter_catalysts()
                # NOTE: legacy spirit_worker:8221-8222 also called
                # ``msl.signal_action("external")`` here as an exploration
                # nudge. Under L2 split MSL lives in mind_worker; rather
                # than add a new bus event for this single in-process
                # signal, the nudge is left as a follow-up — mention/
                # post telemetry already feeds mind_worker via
                # X_POST_PUBLISHED (chunk 9F).
                self._emit_timechain_commit(result, full_config)
            elif result is not None and result.status == "unverified":
                # Soft-failed but the tweet likely landed (twitterapi.io couldn't
                # parse its own response). NOT a failure — the row is S_UNVERIFIED
                # and counts toward the daily latch + budget so it cannot re-fire
                # duplicates. No tweet_id, so we skip the timechain commit. The
                # min-interval (now unverified-aware) blocks any catalyst re-fire.
                logger.info(
                    "[PostDispatch] posted UNVERIFIED via gateway (likely landed,"
                    " id unparseable): %s", getattr(result, "reason", ""))
            elif result is not None and result.status not in (
                    "disabled", "too_soon", "no_catalyst",
                    "hourly_limit", "daily_limit", "pending_exists",
                    "must_post_hard_cap",
                    "consumer_blocked", "circuit_breaker"):
                logger.warning(
                    "[PostDispatch] post failed: %s — %s",
                    result.status, getattr(result, "reason", ""))

        # ── Decide social-window open ──
        skip_set = ("too_soon", "disabled", "circuit_breaker")
        social_window = delegate_posted or (
            result is not None and result.status not in skip_set)

        # ── Delegate-second rotation (only if rotation didn't already) ──
        if not delegate_posted and not delegate_first:
            self._process_delegate_queue(
                full_config=full_config, pop_on_failure=False)

        # ── Mention discovery + reply cycle (30-min cooldown) ──
        if social_window:
            self._run_mention_cycle(
                full_config=full_config,
                grounded_words=ctx.grounded_words,
                neuromods=ctx.neuromods,
                emotion=ctx.emotion,
                now=time.time(),
            )
            # Chunk 9N: broadcast newly-discovered mentions + felt
            # experiences + engagement snapshots to other Titans
            # (canonical poller only — gated inside each method).
            self._broadcast_new_mentions()
            self._broadcast_new_felt_experiences()
            self._broadcast_new_engagement_snapshots()

    # ── Meter catalyst handoff ─────────────────────────────────────────
    # The meter (SocialPressureMeter) accumulates CatalystEvent instances
    # via on_catalyst_event. Legacy spirit_worker used a parallel list
    # `_x_catalysts`. Under chunk 9I dual-mode, the meter's internal list
    # is the source of truth. These helpers translate between meter shape
    # (CatalystEvent) and the dict form gateway.post + force_ungrounded
    # expect.

    def _drain_meter_catalysts(self) -> list[dict]:
        """Return current catalyst snapshot as legacy-shaped dicts.

        Does NOT clear the meter's list; clearing happens via
        _clear_meter_catalysts only on verified/posted (mirrors
        spirit_worker:8217-8218 `_x_catalysts.clear()`)."""
        if self._meter is None:
            return []
        out: list[dict] = []
        for ev in list(getattr(self._meter, "catalyst_events", []) or []):
            try:
                out.append({
                    "type": getattr(ev, "type", "unknown"),
                    "significance": float(
                        getattr(ev, "significance", 0.5) or 0.5),
                    "content": str(getattr(ev, "content", "") or ""),
                    "data": dict(getattr(ev, "data", {}) or {}),
                })
            except Exception:
                continue
        return out

    def _clear_meter_catalysts(self) -> None:
        """Empty the meter's catalyst buffer on successful post."""
        if self._meter is None:
            return
        try:
            buf = getattr(self._meter, "catalyst_events", None)
            if buf is not None:
                buf.clear()
        except Exception as _err:
            logger.debug(
                "[PostDispatch] catalyst clear failed: %s", _err)


__all__ = ("PostDispatchOrchestrator",)
