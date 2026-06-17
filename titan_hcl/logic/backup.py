"""
logic/backup.py
Sovereign backup system: Arweave permanent storage via Irys.
V5.0: Meditation-triggered backups. Direct memory backend (DuckDB + FAISS + Kuzu).
      Daily personality (~70MB compressed) + weekly full soul (~200MB compressed).
      Triggered by MEDITATION_COMPLETE events (Titan's own time).

Timing (agreed 2026-03-28):
  - ZK epoch snapshot: every MEDITATION_COMPLETE (~4x/day, ~$0.08/mo)
  - MyDay NFT: every 4th meditation (~1x/day, ~$0.07/mo)
  - Personality backup: 1st meditation/day (Arweave, ~$10.50/mo)
  - Soul package: 1st meditation/week on Sunday (Arweave, ~$4.04/mo)
  - Total: ~$15/month for complete sovereign backup
"""
import asyncio
import logging
import os
import tarfile
import time
from contextlib import suppress
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from titan_hcl.utils.crypto import hash_file
from titan_hcl.utils.silent_swallow import swallow_warn
from titan_hcl.utils.maker_alert import send_maker_alert
from titan_hcl.params import get_params

logger = logging.getLogger(__name__)

# Phase B (RFP_backup_arweave_sustainability) — committed default for the chained
# diff model. Runtime config.toml [backup].chained_incrementals overrides it. NOT a
# phase-c/Rust-synced constant (a Python-only L2 feature flag, so it must NOT live in
# the generated _phase_c_constants.py — that file is kept in lock-step with the Rust
# kernel's constants.rs by the phase-c generator-parity gate). The no-silent-full-ship
# safety is ALWAYS on regardless of this flag (INV-BR-9).
_BACKUP_CHAINED_INCREMENTALS_DEFAULT = False


def _l5_cleanup_old_local_tarballs(local_dir: str, retention_days: int = 30) -> int:
    """L5 retention helper — prune personality_baseline_*.tar.gz and
    personality_incremental_*.tar.gz files older than retention_days.

    The legacy cascade's `cleanup_local` only matches `personality_*.tar.gz`
    by mtime which CATCHES these too (since they start with `personality_`),
    but we also call this directly from create_local_diff_event because the
    legacy cascade path is BYPASSED when local_diff_enabled=true.

    Returns count of deleted files. Never raises.
    """
    import glob
    deleted = 0
    try:
        cutoff = time.time() - (retention_days * 86400)
        patterns = [
            os.path.join(local_dir, "personality_baseline_*.tar.gz"),
            os.path.join(local_dir, "personality_incremental_*.tar.gz"),
        ]
        for pat in patterns:
            for path in glob.glob(pat):
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.unlink(path)
                        deleted += 1
                except OSError as e:
                    logger.warning(
                        "[Backup] L5 cleanup failed for %s: %s", path, e)
        if deleted > 0:
            logger.info(
                "[Backup] L5 cleanup: pruned %d local tarball(s) older than %dd",
                deleted, retention_days)
    except Exception as e:
        logger.warning("[Backup] L5 cleanup error: %s", e)
    return deleted


class BackupReconstructDeferred(Exception):
    """The chained-mirror reconstruct failed for a TRANSIENT reason — the chain
    provider isn't ready yet (early boot) or a fetch was unreachable — while the
    chain is verifiably still on-chain (head=present) or unproven-gone
    (head=unverified). The orchestrator SKIPS planning this tick and retries;
    it must NEVER fall to an expensive self_heal baseline on a transient.

    Maker rule (RFP_backup_redesign_spine, 2026-06-11): on mainnet a full baseline
    fires ONLY on a genuine integrity failure (chain DATA corrupt, or a chain tx
    DEFINITIVELY missing on-chain) — never because the incrementals couldn't run
    (low Irys SOL / provider warming at boot). The 7-day-gap false self_heal
    baseline on T1 (2026-06-11) was exactly this: the reconstruct ran during early
    boot before the chain provider was ready → fetch failed → baseline. Now it
    defers until the (free-to-read) chain is reachable, then ships a cheap
    catch-up incremental."""


class RebirthBackup:
    """
    Manages sovereign backup triggered by meditation cycles:
    - Daily personality archive → Arweave (1st meditation of day)
    - Weekly full soul package → Arweave (1st meditation on Sunday)
    - ZK compressed epoch snapshot → Solana (every meditation)
    - MyDay NFT → Solana (every 4th meditation)
    """

    def __init__(self, network_client, config: dict = None, titan_id: str = "T1",
                 arweave_store=None, full_config: dict = None, chain_provider=None):
        """
        Args:
            network_client: Solana RPC client (for ZK snapshot + NFT mint)
            config: memory_and_storage section from config.toml
            titan_id: T1/T2/T3 — used for per-Titan manifest path
            arweave_store: injected ArweaveStore (rFP Phase 1 BUG-5 fix —
                constructed ONCE at boot rather than rebuilt per-backup)
            full_config: optional full config dict (for mainnet_budget flag)
            chain_provider: injected ChainProvider (RFP_chain_provider Phase A —
                the data-plane path for Arweave put/get; tests inject a
                FakeChainProvider). Lazily built by `_ensure_chain()` if None.
        """
        config = config or {}
        self.network = network_client
        self.current_snapshot_hash = None
        self._titan_id = titan_id
        self._arweave_store = arweave_store  # BUG-5: injected once at boot
        self._chain_provider = chain_provider  # RFP_chain_provider Phase A
        self._full_config = full_config or {}

        # Will be wired by TitanHCL.__init__
        self.memory = None
        self.social = None
        self._photon = None

        # Backup state tracking (calendar-day based)
        self._last_personality_date = ""  # "YYYY-MM-DD"
        self._last_soul_date = ""         # "YYYY-MM-DD"
        self._last_timechain_date = ""    # "YYYY-MM-DD" — added 2026-05-06
        self._meditation_count = 0
        self._meditation_count_since_nft = 0

        # SPEC §24 / rFP_backup_diff_baseline_unified_v1 Phase 1 (2026-05-15):
        # Per-gate asyncio.Lock for atomic compare-and-swap on the daily date
        # check + write. Closes the dedup race documented in AUDIT_irys_arweave
        # _costs_20260514 §4 BUG-2: pre-fix, two MEDITATION_COMPLETE events
        # firing close together both passed the `if today != _last_*_date`
        # check before either could set the date — observed 15 personality
        # uploads on 2026-05-12. Lock is created lazily on first use (so the
        # constructor stays event-loop-agnostic — RebirthBackup is instantiated
        # at plugin boot which is sync); see _get_*_cas_lock helpers.
        self._personality_cas_lock = None  # asyncio.Lock, lazy
        self._soul_cas_lock = None         # asyncio.Lock, lazy
        self._timechain_cas_lock = None    # asyncio.Lock, lazy

        # Phase 2 pre-stage (2026-05-31): the backup worker's stager builds the
        # day's event off the recv loop and parks it here; on_meditation_complete
        # consumes it for a fast ship. _stage_lock guards the stager↔meditation
        # handoff (a threading.Lock — both run on worker daemon threads, not the
        # asyncio loop). {"staged": StagedEvent, "date": "YYYY-MM-DD"} or None.
        self._staged_event = None
        self._stage_lock = threading.Lock()

        # RFP_backup_redesign_spine Phase D / INV-BRS-7 — orchestrator-owned
        # recovery/scheduling SIDE-STATE, consolidated from 4 scattered side-files
        # (.restore_test_halt / .force_baseline / backup_dry_run_result /
        # .mirror_state) into ONE in-memory object persisted in _BACKUP_STATE_PATH
        # (the ONE atomic state file). The orchestrator timer-flushes it + the
        # existing SAVE_NOW / MODULE_SHUTDOWN wiring flushes it; halt/force-baseline
        # flush IMMEDIATELY (fail-closed: a crash must never lose a halt). Consumers
        # (dashboard) read the consolidated JSON readout (synthesis-snapshot style —
        # ModuleStateWriter carries only a state string, hand-rolled SHM is barred).
        self._halted = False
        self._halt_reason = ""
        self._halt_failed_event_id = None
        self._force_baseline_pending = False   # one-shot R4 recovery token (INV-BKP-5)
        self._last_dry_run = None              # dashboard backup-health card
        self._last_restore_test_date = None    # §24.12 Sunday once/day gate
        self._mirror_state = None              # {event_id, chained, arcs, ts} | None

        # Load persisted backup state if available
        self._load_backup_state()

    def stage_built_event(self, staged, day: str) -> None:
        """Stager entry-point (called from the worker's background thread): park a
        freshly-built StagedEvent for today's ship. Won't clobber an unconsumed
        fresh stage for the same day."""
        with self._stage_lock:
            cur = self._staged_event
            if cur is not None and cur.get("date") == day:
                return  # already have a fresh stage for today
            self._staged_event = {"staged": staged, "date": day}

    def _take_fresh_staged_event(self, day: str):
        """Atomically take + clear today's staged event (or None if absent/stale).
        Consumed exactly once; a stale (prior-day) stage is dropped."""
        with self._stage_lock:
            entry = self._staged_event
            self._staged_event = None
            if entry is not None and entry.get("date") == day:
                return entry.get("staged")
            return None

    def _get_personality_cas_lock(self):
        """Lazy-create the personality-date CAS lock on first async use."""
        if self._personality_cas_lock is None:
            self._personality_cas_lock = asyncio.Lock()
        return self._personality_cas_lock

    def _get_soul_cas_lock(self):
        """Lazy-create the soul-date CAS lock on first async use."""
        if self._soul_cas_lock is None:
            self._soul_cas_lock = asyncio.Lock()
        return self._soul_cas_lock

    def _get_timechain_cas_lock(self):
        """Lazy-create the timechain-date CAS lock on first async use."""
        if self._timechain_cas_lock is None:
            self._timechain_cas_lock = asyncio.Lock()
        return self._timechain_cas_lock

    # -------------------------------------------------------------------------
    # Backup State Persistence
    # -------------------------------------------------------------------------
    _BACKUP_STATE_PATH = "data/backup_state.json"

    def _load_backup_state(self):
        """Load backup tracking state from disk (survives restarts)."""
        import json
        try:
            if os.path.exists(self._BACKUP_STATE_PATH):
                with open(self._BACKUP_STATE_PATH) as f:
                    state = json.load(f)
                self._last_personality_date = state.get("last_personality_date", "")
                self._last_soul_date = state.get("last_soul_date", "")
                # 2026-05-06 fix: timechain backup date now persisted across
                # restarts. Pre-fix it was in-memory only → every restart
                # re-triggered timechain on next meditation. Caused 160-backup
                # bloat on T3 (10/day vs target 1/day) — see BUG-TIMECHAIN-
                # BACKUP-RESTART-LEAK-20260506.
                self._last_timechain_date = state.get("last_timechain_date", "")
                self._meditation_count = state.get("meditation_count", 0)
                self._meditation_count_since_nft = state.get("meditation_count_since_nft", 0)
                # Phase D side-state (INV-BRS-7 consolidation)
                self._halted = bool(state.get("halted", False))
                self._halt_reason = state.get("halt_reason", "") or ""
                self._halt_failed_event_id = state.get("halt_failed_event_id")
                self._force_baseline_pending = bool(
                    state.get("force_baseline_pending", False))
                self._last_dry_run = state.get("last_dry_run")
                self._last_restore_test_date = state.get("last_restore_test_date")
                self._mirror_state = state.get("mirror_state")
                logger.info("[Backup] Loaded state: personality=%s, soul=%s, "
                            "meditations=%d, halted=%s",
                            self._last_personality_date, self._last_soul_date,
                            self._meditation_count, self._halted)
        except Exception as e:
            swallow_warn('[Backup] No backup state loaded', e,
                         key="logic.backup.no_backup_state_loaded", throttle=100)

    def _save_backup_state(self):
        """Persist backup tracking state to disk — ATOMIC (§11.H.2 / §11.H.9).

        Used by both the periodic path and the §11.H.9 SAVE_NOW + MODULE_SHUTDOWN
        flush wiring (backup_worker). tmp+os.replace so a crash mid-write can never
        corrupt backup_state.json (a torn file → silent loss of the dedup dates →
        duplicate/re-shipped backups)."""
        import json
        os.makedirs(os.path.dirname(self._BACKUP_STATE_PATH) or ".", exist_ok=True)
        try:
            tmp = self._BACKUP_STATE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "last_personality_date": self._last_personality_date,
                    "last_soul_date": self._last_soul_date,
                    "last_timechain_date": self._last_timechain_date,
                    "meditation_count": self._meditation_count,
                    "meditation_count_since_nft": self._meditation_count_since_nft,
                    # Phase D side-state (INV-BRS-7) — ONE owned state file
                    "halted": self._halted,
                    "halt_reason": self._halt_reason,
                    "halt_failed_event_id": self._halt_failed_event_id,
                    "force_baseline_pending": self._force_baseline_pending,
                    "last_dry_run": self._last_dry_run,
                    "last_restore_test_date": self._last_restore_test_date,
                    "mirror_state": self._mirror_state,
                    "updated_at": time.time(),
                }, f, indent=2)
            os.replace(tmp, self._BACKUP_STATE_PATH)
        except Exception as e:
            logger.warning("[Backup] Failed to save state: %s", e)

    # -------------------------------------------------------------------------
    # Main Entry Point: Meditation-Triggered Backup
    # -------------------------------------------------------------------------
    async def on_meditation_complete(self, payload: dict):
        """
        Central backup handler called on every MEDITATION_COMPLETE event.

        Decides what backup actions to take based on meditation count and calendar:
        1. ZK epoch snapshot (every meditation)
        2. Personality → Arweave (1st meditation of day)
        3. Soul package → Arweave (1st meditation on Sunday)
        4. MyDay NFT (every 4th meditation)

        Args:
            payload: MEDITATION_COMPLETE bus payload with keys:
                epoch, promoted, pruned, trigger, success, ts
        """
        if not payload.get("success", False):
            logger.debug("[Backup] Meditation was not successful — skipping backup")
            return

        self._meditation_count += 1
        self._meditation_count_since_nft += 1
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        weekday = now.weekday()  # 0=Mon, 6=Sun

        epoch = payload.get("epoch", 0)
        promoted = payload.get("promoted", 0)

        # Gather stats
        total_nodes = self.memory.get_persistent_count() if self.memory else 0

        logger.info(
            "[Backup] Meditation #%d complete — epoch=%d promoted=%d nodes=%d (day=%s, weekday=%d)",
            self._meditation_count, epoch, promoted, total_nodes, today, weekday,
        )

        # SPEC §24 — Unified backup pipeline (Phase 5.5, 2026-05-16; D-SPEC-123
        # follow-up 2026-05-23: legacy fallback DISABLED — Maker decision).
        # When [backup].unified_v2_enabled = true (default false), route this
        # meditation through the new diff/baseline/manifest/ZK pipeline
        # (titan_hcl.logic.backup_upload_pipeline). When flag is off, legacy
        # cascade runs unchanged — that's the T2/T3 default + T1 rollback.
        #
        # NO LEGACY FALLBACK on unified_v2 failure. Legacy cascade is bug-laden
        # + costs money per AUDIT_irys_arweave_costs_20260514; silently falling
        # back to it on every unified_v2 hiccup is strictly worse than failing
        # the event loudly and letting the next meditation retry. The root-
        # cause fix for the TOCTOU race that originally triggered this
        # fallback ships in the same commit (full_ship.encode_diff now
        # hardlinks rolling-retention sources at encode time — see
        # titan_hcl/logic/diff_encoders/full_ship.py + the defense-in-depth
        # skip-with-WARN in pack_event_tarball:209). If unified_v2 still
        # raises after those fixes, that's a real bug to investigate, not a
        # signal to ship the legacy path.
        if self._unified_v2_enabled():
            # SPEC §24 — the unified_v2 event IS the daily backup; it ships ONCE
            # per UTC day. The ship + daily gate live in the SINGLE shared path
            # `_ship_daily_event_v2` (also driven by the Maker-forced manual
            # trigger). Gate = MANIFEST-AS-TRUTH (no stuck-claim flag): a success
            # appends today's event (next meditation skips); a failure appends
            # nothing (next meditation retries) — nothing can get stuck.
            await self._ship_daily_event_v2(today, weekday)
            return

        # ── Legacy non-unified_v2 backup path RETIRED (RFP_backup_redesign_spine
        #    Phase B / B-1, 2026-06-10; no-shim). The ZK-epoch-snapshot + the legacy
        #    per-type personality/NFT full-tarball path lived here and was DEAD on every
        #    Titan (unified_v2_enabled=true fleet-wide — the branch above returns). The
        #    unified §24 pipeline (BackupWorker) owns backups now; mint_epoch_nft +
        #    vault/frontmatter eviction is Phase E.

    async def _ship_daily_event_v2(self, today: str, weekday: int) -> bool:
        """SPEC §24 — ship the daily unified_v2 backup event ONCE per UTC day.

        The SINGLE ship path, shared by MEDITATION_COMPLETE and the Maker-forced
        manual trigger (2026-06-11: the heavy whole-file `_run_unified_event_v2`
        was deleted — both triggers now ship via the bounded primitives below,
        so a manual force-ship can never re-trip BUG-BACKUP-RSS-FLAP).

        Gate = MANIFEST-AS-TRUTH (no stuck-claim flag): a SUCCESSFUL ship appends
        today's event (the next trigger correctly skips); a FAILED ship appends
        nothing (the next trigger retries). The personality CAS lock only
        serializes two near-simultaneous triggers → still exactly one ship/day.

        Prefers the pre-staged finalized drip (`_ship_staged_event_v2` — streams
        the ready tarballs, bounded RSS); falls back to a bounded inline build
        (`_inline_build_and_ship_v2` — build_slice byte-budget + streamed pack).
        Returns True iff an event shipped (manifest appended)."""
        async with self._get_personality_cas_lock():
            if self._todays_backup_already_landed():
                logger.info(
                    "[Backup] §24 daily backup already LANDED for %s "
                    "(manifest event exists) — skipping", today)
                return False

            # Irys auto-fund via ChainProvider (RFP Phase C tail) — top up before
            # the upload; bounded by the daily cap inside chain.fund. Best-effort;
            # a fund hiccup never blocks the backup.
            try:
                await self._auto_fund_irys_before_upload()
            except Exception as _af_err:
                logger.warning(
                    "[Backup] §24 Irys auto-fund check raised: %s", _af_err)

            # Ship: prefer a fresh pre-staged event (built off-loop by the drip);
            # fall back to a bounded inline build on cold-start / stale-baseline.
            # On ANY failure we return — the manifest has no today-event, so the
            # next trigger retries (nothing stuck, no claim to release).
            try:
                _staged = self._take_fresh_staged_event(today)
                if _staged is not None:
                    shipped = await self._ship_staged_event_v2(_staged)
                    if not shipped:
                        logger.info(
                            "[Backup] §24 staged ship declined (stale/failed) — "
                            "falling back to a bounded inline build "
                            "(Phase D last-resort)")
                        shipped = await self._inline_build_and_ship_v2(weekday)
                else:
                    # Phase D failsafe-2 (INV-BRS-3): no fresh stage → bounded
                    # inline build+ship via BackupWorker (NOT a whole-file path).
                    shipped = await self._inline_build_and_ship_v2(weekday)
            except Exception as e:
                logger.exception(
                    "[Backup] §24 unified_v2 ship raised — no backup; next "
                    "trigger retries (manifest has no today-event → nothing "
                    "stuck): %s", e)
                self._alert_backup_failure("unified_v2", f"raised: {e}")
                return False

            if shipped:
                logger.info(
                    "[Backup] §24 unified_v2 event SHIPPED — daily backup landed "
                    "for %s; manifest updated", today)
            else:
                logger.info(
                    "[Backup] §24 ship did not land for %s — no manifest event "
                    "written; next trigger will retry (nothing stuck)", today)
            return shipped

    # _compute_sovereignty DELETED (RFP_backup_redesign_spine Phase E — INV: ONE
    # sovereignty score). It was a 0-caller wrapper around the canonical
    # synthesis/sovereignty_readout.read_rolling_sovereignty; any consumer reads
    # that directly (S × 100 for the 0-100 scale, S × 10000 bp on-chain).

    # -------------------------------------------------------------------------
    # Boot Check
    # -------------------------------------------------------------------------
    async def check_on_boot(self):
        """Boot-time verification + rFP Phase 4 catch-up.

        1. Verify critical data files exist.
        2. If last personality upload > 24h ago AND mode is mainnet_arweave
           AND Irys has runway → fire personality backup immediately.
        """
        critical_paths = [
            ("data/titan_memory.duckdb", "DuckDB memory store"),
            ("data/memory_vectors.faiss", "FAISS vector index"),
            ("titan_constitution.md", "Titan constitution"),
        ]
        for path, label in critical_paths:
            if os.path.exists(path):
                logger.info("[Backup] Boot check: %s OK (%s)", label, path)
            else:
                logger.warning("[Backup] Boot check: %s MISSING (%s)", label, path)

        # Backup catch-up is owned by unified_v2: the BackupOrchestrator drips
        # the daily incremental and the meditation event ships it. The legacy
        # boot catch-up (full-tarball re-upload via TimeChainBackup/BackupCascade)
        # was retired here with the write-path reconciliation — it was a SOL
        # drain and ran synchronously before `booted`, stalling boot into a
        # Guardian-kill loop (RFP_backup_redesign_spine, 2026-06-10).
        return {"critical_ok": True}

    # -------------------------------------------------------------------------
    # Hash — delegated to utils/crypto.py (Single Source of Truth)
    # -------------------------------------------------------------------------
    def calculate_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of a file via the centralized crypto utility."""
        return hash_file(filepath)

    # =========================================================================
    # Personality Backup → Arweave (Daily)
    # =========================================================================

    # All personality-critical data paths — SPEC §24.4.B canonical inventory.
    #
    # SPEC §24 / rFP_backup_diff_baseline_unified_v1 Phase 1 (2026-05-15)
    # remediated:
    #   REMOVED (12 entries):
    #     - 6 static birth identity files → §24.4.A (on Arweave via GenesisNFT)
    #       (titan_constitution.md, titan_directives.sig, genesis_record.json,
    #        genesis_nft_metadata.json, birth_dna_snapshot.json, titan_identity.json)
    #     - titan_chronicles.md (RE-ADDED 2026-05-22 below — BUG-CHRONICLE-WRITER-DEAD-POST-A87 fixed)
    #     - data/runtime_keypair.json (ephemeral — regenerated every boot from soul_keypair.enc)
    #     - data/zk_queue/pending.json (reconstructable from memory.db uncommitted rows)
    #     - data/sage_memory/buffer_metadata.json + meta.json (pointer-only — moved to weekly with FULL dir)
    #     - data/timechain/contract_stats.json (derivable from chain blocks)
    #   ADDED (8 entries — §11.H critical-data files missed by prior audit + new substrate paths):
    #     - data/word_recipes.json (SPEC §11.H critical — outer_interface_worker)
    #     - data/outer_interface_state.json (SPEC §11.H critical — outer_interface_worker)
    #     - data/prediction/novelty_state.json (SPEC §11.H critical — self_reflection_worker)
    #     - data/word_resonance_dynamic.json
    #     - data/dim_history/assessment_history.json
    #     - data/meta_teacher/adoption_metrics.json
    #     - data/mini_reasoning/ (full dir)
    #     - data/titan_vm_v2/ (FULL DIR — Maker decision 2026-05-15 Q5)
    # Size hints refreshed to current observed sizes (audit 2026-05-14).
    PERSONALITY_PATHS = [
        ("data/neural_nervous_system/", "neural_ns"),              # ~129MB: NS weights + training buffers
        ("data/neuromodulator/", "neuromodulator"),                # ~7KB: DNA + allostatic state
        ("data/inner_memory.db", "inner_memory.db"),              # ~1.1GB: vocabulary, fires, compositions
        ("data/titan_memory.duckdb", "titan_memory.duckdb"),      # ~34MB: all memory nodes
        ("data/memory_vectors.faiss", "memory_vectors.faiss"),    # ~4MB: FAISS semantic index
        ("data/memory_vectors.faiss.idmap.json", "memory_vectors.faiss.idmap.json"),  # ID map
        # ── Synthesis spine + learning (Phase D — resurrection-completeness, added
        # 2026-06-13). Without these a restored Titan comes back WITHOUT his sovereign
        # wiki concepts (DK.1/2), macros/skills (D-strategy/EEL-B), research recipes
        # (DK.5), and RL policy. The anchored TXs survive on the timechain chains, but
        # the rebuildable indices + the sidecar deref-target are not chain-derivable
        # cheaply, so they are snapshotted directly. See ARCHITECTURE_storage_topology.md.
        ("data/synthesis.duckdb", "synthesis.duckdb"),            # skills, reasoning_records, engram axes, research_recipes
        ("data/synthesis_spine.kuzu", "synthesis_spine.kuzu"),    # Engram concepts (wiki) + Reasoning macros + Production skills
        ("data/reasoning_vectors.faiss", "reasoning_vectors.faiss"),  # macro signature embeddings (composite recall)
        ("data/thought_sidecar.db", "thought_sidecar.db"),        # tx_hash→content deref bridge (concept evidence)
        ("data/self_learning.duckdb", "self_learning.duckdb"),    # RL policy weights + reward/decision accounting
        # Affective Grounding Loop (RFP_affective_grounding_loop §7.B, added
        # 2026-06-13). The per-Titan EMA baseline + the learned AffectiveNudgeNet
        # weights ARE the Titan's divergent emotional personality (INV-AFF-SELF-
        # SOVEREIGN) — without these a restored Titan loses its habituation
        # history + felt-value net (it would re-learn from scratch). Tiny (<1MB).
        ("data/affective/", "affective"),                          # affective_nudge_state.json + affective_nudge_net.npz
        ("data/experience_orchestrator.db", "experience_orchestrator.db"),  # ~336MB: learned action wisdom
        ("data/experience_memory.db", "experience_memory.db"),    # ~51MB: experience records
        ("data/episodic_memory.db", "episodic_memory.db"),        # ~99MB: episodic records
        ("data/experiential_memory.db", "experiential_memory.db"),  # ~856KB: dream insights
        ("data/pi_heartbeat_state.json", "pi_heartbeat_state.json"),  # ~1KB
        # Narrative diary — RE-ADDED 2026-05-22 (BUG-CHRONICLE-WRITER-DEAD-POST-A87
        # fixed: titan_HCL._append_to_chronicle writes meditation reflections again
        # on MEDITATION_COMPLETE). Was removed from §24.4.B while the writer was dead.
        ("titan_chronicles.md", "titan_chronicles.md"),            # ~varies: Scholar's Chronicle reflections
        # Soul-diary hash-chain ledger — RFP_titan_authored_soul_diary P8 / INV-SD-12:
        # the 3rd triple-anchor root (daily diff-incremental + Arweave for T1) so the
        # tamper-evident {entry_hash, cumulative_hash} chain + public projection survive.
        ("data/soul_diary_chain.json", "soul_diary_chain.json"),   # ~varies: tamper-evident diary ledger
        # MSL concept state — "I" identity + concept cascade (critical for resurrection)
        ("data/msl/msl_identity.json", "msl/msl_identity.json"),    # ~2KB: I-confidence, recipe, convergences
        ("data/msl/msl_concepts.json", "msl/msl_concepts.json"),    # ~90KB: YOU/YES/NO/WE/THEY + interaction matrix
        ("data/msl/msl_policy.json", "msl/msl_policy.json"),        # ~918KB: policy network weights
        ("data/msl/msl_buffer.json", "msl/msl_buffer.json"),        # ~2.2MB: policy replay buffer
        # Memory graph + CGN (added 2026-04-06 — critical for resurrection)
        ("data/memory_nodes.db", "memory_nodes.db"),                # ~11MB: core memory node records
        ("data/cgn/", "cgn"),                                        # ~467KB: CGN state tensor + affinity + telemetry
        # Social state (added 2026-04-06 — Titan forgets relationships without these)
        ("data/social_graph.db", "social_graph.db"),                # ~100KB: social relationships
        ("data/social_x.db", "social_x.db"),                        # ~948KB: X interaction history
        # Operational state (added 2026-04-06 — prevents double-anchoring / gaps)
        ("data/anchor_state.json", "anchor_state.json"),            # ~0.3KB: Solana anchor counter + last TX
        # ─── Added 2026-04-20 (rFP_backup_worker Phase 0 persistence audit) ────
        ("data/reasoning/", "reasoning"),                                               # ~7.6MB: meta_stats, chain_iql, policy_net, value_head, sequence_quality, meta_policy, meta_autoencoder
        ("data/meta_cgn/", "meta_cgn"),                                                 # ~4.5MB: META-CGN consumer (jsonl logs filtered from daily, included in soul)
        ("data/emot_cgn/", "emot_cgn"),                                                 # ~64KB: EMOT-CGN 8th consumer
        ("data/filter_down_v5_state.json", "filter_down_v5_state.json"),                # ~4KB
        ("data/filter_down_v5_weights.json", "filter_down_v5_weights.json"),            # ~616KB
        ("data/filter_down_v5_buffer.json", "filter_down_v5_buffer.json"),              # ~5.5MB: V5 replay buffer
        # ─── Small (per-half LOCAL) filter_down nets — Phase 0 chunk 0C (2026-05-20) ───
        # Per-half learned TrinityValueNet<65> brains (inner/outer body+mind+spirit).
        # MUST be backed up per directive_memory_preservation — the unified v5 brains
        # above were lost in the Phase C migration; do not repeat that for the small tier.
        ("data/filter_down_local_inner_state.json", "filter_down_local_inner_state.json"),    # ~1.4KB
        ("data/filter_down_local_inner_weights.json", "filter_down_local_inner_weights.json"),# ~463KB
        ("data/filter_down_local_inner_buffer.json", "filter_down_local_inner_buffer.json"),  # ~1MB: inner-half replay buffer
        ("data/filter_down_local_outer_state.json", "filter_down_local_outer_state.json"),    # ~1.4KB
        ("data/filter_down_local_outer_weights.json", "filter_down_local_outer_weights.json"),# ~463KB
        ("data/filter_down_local_outer_buffer.json", "filter_down_local_outer_buffer.json"),  # ~1MB: outer-half replay buffer
        ("data/unified_spirit_state.json", "unified_spirit_state.json"),                # ~2.2MB
        ("data/spirit_state_reload.json", "spirit_state_reload.json"),                  # ~17KB
        ("data/dreaming_state.json", "dreaming_state.json"),                            # ~164B: dream cycle pointer
        ("data/edge_detector_state.json", "edge_detector_state.json"),                  # ~495B
        ("data/intuition_convergence_state.json", "intuition_convergence_state.json"),  # ~3.2KB
        ("data/sovereignty_state.json", "sovereignty_state.json"),                      # ~206B
        ("data/resonance_state.json", "resonance_state.json"),                          # ~861B
        ("data/social_pressure_state.json", "social_pressure_state.json"),              # ~1.5KB
        ("data/sphere_clock_state.json", "sphere_clock_state.json"),                    # ~1.4KB
        ("data/maker_engine_state.json", "maker_engine_state.json"),                    # ~96B
        ("data/contact_maker_state.json", "contact_maker_state.json"),                  # ~144B
        ("data/events_teacher_state.json", "events_teacher_state.json"),                # ~25KB
        ("data/teacher_state.json", "teacher_state.json"),                              # ~63B
        ("data/dream_bridge_dedup.json", "dream_bridge_dedup.json"),                    # ~2.6KB
        ("data/persona_session_edge_state.json", "persona_session_edge_state.json"),    # ~1.1KB
        ("data/adversary_evolution.json", "adversary_evolution.json"),                  # ~2.2KB: evolved defense patterns
        ("data/phase_status.json", "phase_status.json"),                                # ~116B: phase gate pointer
        ("data/social_delegate_queue.json", "social_delegate_queue.json"),              # ~723KB: pending social delegations
        ("data/events_teacher.db", "events_teacher.db"),                                # ~232KB: events teacher memory
        ("data/grammar_rules.db", "grammar_rules.db"),                                  # ~20KB: learned grammar
        # ─── Added 2026-05-15 (SPEC §24 / rFP_backup_diff_baseline_unified_v1 Phase 1) ────
        ("data/word_recipes.json", "word_recipes.json"),                                # SPEC §11.H critical — outer_interface_worker
        ("data/outer_interface_state.json", "outer_interface_state.json"),              # SPEC §11.H critical — outer_interface_worker
        ("data/prediction/novelty_state.json", "prediction/novelty_state.json"),        # SPEC §11.H critical — self_reflection_worker
        ("data/word_resonance_dynamic.json", "word_resonance_dynamic.json"),
        ("data/dim_history/assessment_history.json", "dim_history/assessment_history.json"),
        ("data/meta_teacher/adoption_metrics.json", "meta_teacher/adoption_metrics.json"),
        ("data/mini_reasoning/", "mini_reasoning"),                                     # mini-reasoning state subtree
        ("data/titan_vm_v2/", "titan_vm_v2"),                                           # FULL DIR — Maker decision 2026-05-15 Q5; small now, grows with rFP_titan_vm_v2
        # ─── Added 2026-05-25 (D-SPEC-125 / v1.57.1 — §G5.2 item 4 traveling-tensor checkpoints) ───
        # Per-part tensor state (prev, prev2, last 5D observable signature) written
        # by each BMS+spirit daemon (G21 single-writer); enables tensor journey to
        # survive systemctl restart AND host reboot AND restore-from-Arweave.
        ("data/inner_body_checkpoint.bin", "inner_body_checkpoint.bin"),                # 68B  (body 5D × 2 + obs)
        ("data/inner_mind_checkpoint.bin", "inner_mind_checkpoint.bin"),                # 148B (mind 15D × 2 + obs)
        ("data/inner_spirit_checkpoint.bin", "inner_spirit_checkpoint.bin"),            # 388B (spirit 45D × 2 + obs)
        ("data/outer_body_checkpoint.bin", "outer_body_checkpoint.bin"),                # 68B
        ("data/outer_mind_checkpoint.bin", "outer_mind_checkpoint.bin"),                # 148B
        ("data/outer_spirit_checkpoint.bin", "outer_spirit_checkpoint.bin"),            # 388B
        # ─── Configuration (CONDITIONAL — SPEC §24.4.B / D-SPEC-147, 2026-05-31) ──────
        # titan_hcl/config.toml — the Titan's full runtime config. Included so a
        # sovereign resurrection restores a FULLY-configured Titan that boots ready
        # (inference/comms/all settings read from the restored config — no re-prompt).
        # Gated by [backup].backup_config_toml (default FALSE — opt-in): the producer
        # SKIPS it unless enabled (see create_personality_archive); the installer turns
        # it on for a mainnet install + strongly recommends encryption. Listing it here
        # (statically) is also what lets the restore inverse-map
        # (backup_restore.build_arc_to_target) map the "config.toml" archive entry back
        # to titan_hcl/config.toml. Dev↔public divergence: on the maintainer fleet
        # config.toml is kept SECRET-FREE (untracked + scrubbed 2026-05-31; real secrets
        # live out-of-repo in ~/.titan/secrets.toml, never backed up). A public install
        # carries the user's real creds → with encryption OFF (Mode A) an enabled
        # config-backup ships them to Arweave in the clear, hence opt-in + the encryption
        # recommendation + warning. Identity keypair is NEVER backed up (§24.4.A /
        # G16(8)); config.toml carries no keypair.
        ("titan_hcl/config.toml", "config.toml"),
    ]

    # Filename patterns excluded from ALL archives — historical dev backups that
    # accumulated in tracked dirs over time. NOT live state. Examples:
    #   data/reasoning/policy_net_backup_20260328_pre_reward_fix.json
    #   data/reasoning/meta_autoencoder.json.bak_20260417
    _BACKUP_SKIP_PATTERNS = ('_backup_', '.bak_', '.bak', '.pre_', '.bksnap')

    # rFP Phase 2 failsafe cascade — local-always snapshot directory
    _LOCAL_BACKUP_DIR = "data/backups"

    # Arweave-excluded paths: large DBs that rebuild from experience, uploaded weekly not daily
    # These are still in local daily backups (PERSONALITY_PATHS) — just not in daily Arweave
    ARWEAVE_DAILY_EXCLUDE = {
        "experience_orchestrator.db",   # ~118MB: rebuilds from new experiences
        "experience_memory.db",         # ~51MB: historical experience records
        "episodic_memory.db",           # ~99MB: episodic records
        # ─── Added 2026-04-20 (rFP_backup_worker Phase 0 shrink-daily) ────────
        # State audit revealed daily personality had grown to 212MB because
        # inner_memory.db + NS buffers tripled in size. Both are already in
        # WEEKLY_EXTRA_PATHS coverage via Sunday soul package.
        "inner_memory.db",              # ~730MB on disk — weekly only
        "neural_ns",                    # ~129MB — weekly only
    }

    # Weekly full backup — SPEC §24.4.C canonical inventory.
    #
    # SPEC §24 / rFP_backup_diff_baseline_unified_v1 Phase 1 (2026-05-15)
    # remediated:
    #   REMOVED (5 entries):
    #     - data/timechain/ → §24.4.D dedicated TIMECHAIN_PATHS tarball (was duplicate with daily)
    #     - data/meditation_memos/ (on-chain Solana memos — refetchable)
    #     - data/daily_nfts/ (on-chain NFTs — refetchable)
    #     - data/testaments/ (already on-chain AND on Arweave per testament.py:11 — fully redundant)
    #     - data/backup_records/ (reconstructable from backup_unified_manifest itself)
    #   ADDED (1 dir + 1 pattern):
    #     - data/sage_memory/ (FULL DIR — closes the half-broken state where pointer JSONs were
    #       in personality tarball but .memmap data was not; replaces buffer_metadata.json + meta.json)
    #     - data/meta_teacher/critiques.YYYYMMDD.jsonl (rotating daily files — pattern coverage
    #       handled by including data/meta_teacher/ subset via filter; see archive builder)
    WEEKLY_EXTRA_PATHS = [
        ("data/consciousness.db", "consciousness.db"),            # ~4GB: chronological epoch log (primary diff target)
        ("data/knowledge_graph.kuzu", "knowledge_graph.kuzu"),    # Kuzu entity graph
        # Sage memory — FULL DIR (replaces pointer-only buffer_metadata.json + meta.json
        # entries that were in PERSONALITY_PATHS pre-§24; closes half-broken state where
        # pointer JSONs were in tarball but .memmap data was not).
        ("data/sage_memory/", "sage_memory"),                      # ~103MB raw incl. .memmap buffers
        # MSL extended state (convergence history)
        ("data/msl/msl_convergence_log.json", "msl/msl_convergence_log.json"),  # ~100KB
        ("data/msl/msl_stats.json", "msl/msl_stats.json"),                      # ~2KB
        # Defense patterns + persona profiles (added post-March 28)
        ("data/adversary_attacks/", "adversary_attacks"),          # ~36KB: jailbreak defense patterns
        ("data/persona_profiles/", "persona_profiles"),            # ~56KB: companion/visitor/adversary configs
        # ─── Added 2026-04-20 (rFP_backup_worker Phase 0) ──────────────────────
        # CGN-family rotating jsonl logs — excluded from daily personality to
        # save Arweave cost, included here for weekly forensic replay coverage.
        ("data/meta_cgn/signals_log.jsonl", "meta_cgn/signals_log.jsonl"),
        ("data/meta_cgn/disagreements.jsonl", "meta_cgn/disagreements.jsonl"),
        ("data/meta_cgn/shadow_mode_log.jsonl", "meta_cgn/shadow_mode_log.jsonl"),
        ("data/meta_cgn/blend_weights_history.jsonl", "meta_cgn/blend_weights_history.jsonl"),
        ("data/meta_cgn/haov_signal_outcomes.jsonl", "meta_cgn/haov_signal_outcomes.jsonl"),
        ("data/emot_cgn/shadow_mode_log.jsonl", "emot_cgn/shadow_mode_log.jsonl"),
        ("data/cgn/affinity_history.jsonl", "cgn/affinity_history.jsonl"),
        ("data/cgn/cgn_telemetry.jsonl", "cgn/cgn_telemetry.jsonl"),
        # Meta-teacher rotating critiques (added 2026-05-15 — SPEC §24.4.C; daily-file pattern
        # — directory-level inclusion captures all critiques.YYYYMMDD.jsonl rotations).
        ("data/meta_teacher/", "meta_teacher"),
    ]

    # NEW 2026-05-15 (SPEC §24 / rFP_backup_diff_baseline_unified_v1 §24.4.D):
    # TimeChain dedicated tarball — separate physical archive, gated/anchored in
    # the same backup event as personality (one event → one Merkle commit on ZK
    # Vault even though two physical tarballs go to Arweave). Daily tail-only
    # diff per §24.5 (chains are append-only fixed-record format; incremental =
    # bytes since previous event's prev_offset). Monthly baseline ships full.
    TIMECHAIN_PATHS = [
        # 7 chain .bin files (one per fork — fork ID embedded in filename)
        ("data/timechain/chain_conversation.bin", "timechain/chain_conversation.bin"),
        ("data/timechain/chain_declarative.bin", "timechain/chain_declarative.bin"),
        ("data/timechain/chain_episodic.bin", "timechain/chain_episodic.bin"),
        ("data/timechain/chain_main.bin", "timechain/chain_main.bin"),
        ("data/timechain/chain_meta.bin", "timechain/chain_meta.bin"),
        ("data/timechain/chain_procedural.bin", "timechain/chain_procedural.bin"),
        ("data/timechain/chain_system.bin", "timechain/chain_system.bin"),
        # sqlite index over the chains
        ("data/timechain/index.db", "timechain/index.db"),
        # auxiliary maker proposals
        ("data/timechain/auxiliary/maker_proposals.db", "timechain/auxiliary/maker_proposals.db"),
    ]

    def create_personality_archive(self, output_path: str = None,
                                    arweave_tier: bool = False) -> Optional[str]:
        """Create compressed tar.gz of personality-critical data.

        Args:
            output_path: Override output path. Default: /tmp/titan_personality_<ts>.tar.gz
            arweave_tier: If True, exclude large experience DBs (ARWEAVE_DAILY_EXCLUDE)
                          to reduce Arweave upload cost. Full archive for local backup.

        Returns output path on success, None on failure.
        """
        if not output_path:
            tag = "arweave" if arweave_tier else "full"
            output_path = f"/tmp/titan_personality_{tag}_{int(time.time())}.tar.gz"

        try:
            skip_patterns = self._BACKUP_SKIP_PATTERNS

            def _filter(ti):
                name = ti.name
                if name.endswith(('.tmp', '.pyc')) or '__pycache__' in name:
                    return None
                # Daily tier: exclude rotating jsonl logs (included in weekly soul)
                if name.endswith('.jsonl'):
                    return None
                # Historical dev backups kept on disk but not live state
                if any(p in name for p in skip_patterns):
                    return None
                return ti

            # config.toml inclusion is opt-IN per [backup].backup_config_toml
            # (default FALSE). A public-install user who leaves it off must retain
            # config.toml themselves + supply it alongside Shard-1 at resurrection.
            _backup_cfg = get_params("backup") or {}
            include_config_toml = bool(_backup_cfg.get("backup_config_toml", False))

            with tarfile.open(output_path, "w:gz", compresslevel=9) as tar:
                for source_path, archive_name in self.PERSONALITY_PATHS:
                    if archive_name == "config.toml" and not include_config_toml:
                        logger.debug("[Backup] config.toml excluded "
                                     "([backup].backup_config_toml=false)")
                        continue
                    # Skip large experience DBs for Arweave tier
                    if arweave_tier and archive_name in self.ARWEAVE_DAILY_EXCLUDE:
                        logger.debug("[Backup] Arweave tier: skipping %s", archive_name)
                        continue

                    source = Path(source_path)
                    if source.exists():
                        if source.is_dir():
                            tar.add(str(source), arcname=archive_name, filter=_filter)
                        else:
                            tar.add(str(source), arcname=archive_name)
                        logger.debug("[Backup] Added %s (%s)",
                                     archive_name, source_path)
                    else:
                        logger.debug("[Backup] Skipped %s (not found)", source_path)

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            tier_label = "Arweave-tier" if arweave_tier else "Full"
            logger.info("[Backup] %s personality archive: %.1f MB at %s",
                        tier_label, size_mb, output_path)
            return output_path

        except Exception as e:
            logger.error("[Backup] Personality archive failed: %s", e)
            return None

    def _build_v3_encryptor(self):
        """Mode-B encryptor for the v=3 event pipeline (RFP G2 / INV-MBR-13).

        When `[backup].encryption_enabled` is true, returns a callable
        (plaintext_tarball, component) → (ciphertext, iv_b64) that encrypts each
        component tarball under a key derived from the soul keypair's master key —
        the SAME key a wallet-only restore re-derives (from Shard-1+Shard-3) and
        the arc[:16] backup_id. Mode-A → None (plaintext upload). Raises loudly on
        init failure (no silent downgrade — an encrypted Titan must NOT ship
        plaintext).
        """
        backup_cfg = get_params("backup") or {}
        if not backup_cfg.get("encryption_enabled", False):
            return None
        from titan_hcl.logic.backup_crypto import (
            load_keypair_bytes, derive_master_key, encrypt_component_tarball,
        )
        net_cfg = get_params("network") or {}
        kp_path = net_cfg.get(
            "wallet_keypair_path", "data/titan_identity_keypair.json")
        kp_bytes, titan_pubkey = load_keypair_bytes(kp_path)
        master = derive_master_key(kp_bytes, titan_pubkey)

        def _encrypt(plaintext: bytes, component: str):
            return encrypt_component_tarball(plaintext, master, component)
        return _encrypt

    # -------------------------------------------------------------------------
    # Backup Records (local verification)
    # -------------------------------------------------------------------------
    _MANIFEST_VERSION = "1.0"

    def _store_backup_record(self, backup_type: str, record: dict):
        """Store backup record locally for verification queries.

        Phase 7 — bumps manifest_version to 1.0 and preserves the `encryption`
        stanza threaded through by the unified backup pipeline. Legacy records lacking these
        fields are treated as manifest_version="0" + encryption.algorithm="none".
        """
        import json
        record_dir = Path("data/backup_records")
        record_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        filepath = record_dir / f"{backup_type}_{ts}.json"
        enriched = dict(record)
        enriched.setdefault("manifest_version", self._MANIFEST_VERSION)
        if "encryption" not in enriched:
            enriched["encryption"] = {"algorithm": "none"}
        with open(filepath, "w") as f:
            json.dump(enriched, f, indent=2)

    def get_latest_backup_record(self, backup_type: str = "personality") -> Optional[dict]:
        """Get the most recent backup record for verification."""
        import json
        record_dir = Path("data/backup_records")
        if not record_dir.exists():
            return None

        files = sorted(record_dir.glob(f"{backup_type}_*.json"), reverse=True)
        if files:
            with open(files[0]) as f:
                return json.load(f)
        return None

    async def verify_backup(self, backup_type: str = "personality") -> dict:
        """Verify latest backup: compare local hash to stored record."""
        record = self.get_latest_backup_record(backup_type)
        if not record:
            return {"verified": False, "error": "No backup record found"}

        # Phase E.2.4: tarball creation is CPU-bound — wrap to_thread.
        import asyncio as _asyncio_local
        if backup_type == "personality":
            archive_path = await _asyncio_local.to_thread(self.create_personality_archive)
        elif backup_type == "soul_package":
            archive_path = await _asyncio_local.to_thread(self.create_soul_package)
        else:
            return {"verified": False, "error": f"Unknown backup type: {backup_type}"}

        if not archive_path:
            return {"verified": False, "error": "Could not create archive for comparison"}

        current_hash = self.calculate_hash(archive_path)
        stored_hash = record.get("archive_hash", "")

        with suppress(FileNotFoundError):
            os.remove(archive_path)

        return {
            "verified": current_hash == stored_hash,
            "backup_type": backup_type,
            "current_hash": current_hash[:16],
            "stored_hash": stored_hash[:16],
            "arweave_tx": record.get("arweave_tx", ""),
            "uploaded_at": record.get("uploaded_at", 0),
        }

    # -------------------------------------------------------------------------
    # Maker alerts — backup events (RFP_backup_redesign_spine Phase E eviction)
    # -------------------------------------------------------------------------
    # The duplicate in-class Telegram POST (`_send_telegram_alert`) + its
    # token/chat-id constants + the orphaned `_alert_backup_success` were DELETED:
    # backup alerts now route through the unified, rate-limited
    # `titan_hcl.utils.maker_alert.send_maker_alert` (ONE Telegram impl fleet-wide;
    # its defaults mirror the old backup token/chat). `_alert_backup_failure`
    # stays as the backup-specific failure FORMATTER (4 callers).

    def _alert_backup_failure(self, backup_type: str, error: str):
        """Alert Maker on backup failure via the unified send_maker_alert
        (rate-limited per failure-type to avoid storms — RFP Phase E)."""
        send_maker_alert(
            f"🔴 *Titan Backup FAILED*\n"
            f"Type: {backup_type}\n"
            f"Error: {error[:200]}",
            alert_key=f"backup.failure.{backup_type}",
            rate_limit_seconds=300.0,
        )

    # -------------------------------------------------------------------------
    # Restore from Archive — Resurrection support
    # -------------------------------------------------------------------------

    def restore_personality(self, archive_path: str) -> dict:
        """Restore Titan's personality from a backup archive.

        Unpacks tar.gz to the correct paths, verifying each file.
        Returns dict with restored_files count and any errors.
        """
        if not os.path.exists(archive_path):
            return {"success": False, "error": f"Archive not found: {archive_path}"}

        restored = []
        errors = []
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                members = tar.getmembers()
                logger.info("[Backup] Restoring personality: %d members in archive", len(members))

                for member in members:
                    try:
                        # Determine extraction path
                        # Archive names map to source paths in PERSONALITY_PATHS
                        target = self._archive_name_to_path(member.name)
                        if target:
                            # Ensure parent directory exists
                            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                            if member.isdir():
                                os.makedirs(target, exist_ok=True)
                            else:
                                with tar.extractfile(member) as src:
                                    if src:
                                        with open(target, "wb") as dst:
                                            dst.write(src.read())
                            restored.append(member.name)
                        else:
                            # Extract to archive name directly (relative path)
                            tar.extract(member, path=".")
                            restored.append(member.name)
                    except Exception as e:
                        errors.append(f"{member.name}: {e}")
                        logger.warning("[Backup] Restore error for %s: %s", member.name, e)

            logger.info("[Backup] Restored %d files (%d errors)", len(restored), len(errors))
            send_maker_alert(
                f"🔄 *Titan Restore Complete*\n"
                f"Files: {len(restored)}\nErrors: {len(errors)}",
                alert_key="backup.restore_complete", rate_limit_seconds=300.0,
            )
            return {"success": len(errors) == 0, "restored": len(restored), "errors": errors}

        except Exception as e:
            msg = f"Archive extraction failed: {e}"
            logger.error("[Backup] %s", msg)
            self._alert_backup_failure("restore", msg)
            return {"success": False, "error": msg}

    def _archive_name_to_path(self, archive_name: str) -> Optional[str]:
        """Map archive member name back to source path for restoration."""
        # Build reverse mapping from PERSONALITY_PATHS + WEEKLY_EXTRA_PATHS
        for source_path, arc_name in self.PERSONALITY_PATHS + self.WEEKLY_EXTRA_PATHS:
            if archive_name == arc_name or archive_name.startswith(arc_name + "/"):
                if archive_name == arc_name:
                    return source_path
                # Subdirectory member: arc_name/subfile → source_path/subfile
                suffix = archive_name[len(arc_name):]
                return source_path.rstrip("/") + suffix
        return None

    def _find_backup_record_by_tx(self, backup_type: str, tx_id: str) -> Optional[dict]:
        """Scan data/backup_records/{backup_type}_*.json for a matching arweave_tx.

        Phase 7 — restore needs the encryption stanza that was captured at upload
        time. Returns the newest matching record, or None if no match (legacy
        records without this tx, or fresh install).
        """
        import json as _json
        record_dir = Path("data/backup_records")
        if not record_dir.exists():
            return None
        for f in sorted(record_dir.glob(f"{backup_type}_*.json"), reverse=True):
            try:
                with open(f) as fh:
                    r = _json.load(fh)
                if r.get("arweave_tx") == tx_id:
                    return r
            except Exception:
                continue
        return None

    async def restore_personality_from_arweave(self, tx_id: str) -> dict:
        """Download personality archive from Arweave and restore it.

        Phase 7 (2026-04-20) — transparently decrypts if the local backup record
        for this tx indicates AES-256-GCM. Legacy (pre-toggle) entries with
        encryption.algorithm="none" or no record at all fall through to the
        plaintext extraction path.

        Args:
            tx_id: Arweave transaction ID of the backup archive.

        Returns:
            dict with success, restored count, errors.
        """
        try:
            from titan_hcl.utils.arweave_store import ArweaveStore
            store = ArweaveStore(
                keypair_path=getattr(self.network, '_wallet_path', None) or getattr(self.network, '_keypair_path', None),
            )

            logger.info("[Backup] Downloading personality from Arweave: %s", tx_id[:20])
            send_maker_alert(
                f"🔄 *Titan Restore Started*\nDownloading from Arweave: `{tx_id[:20]}...`",
                alert_key="backup.restore_started", rate_limit_seconds=300.0,
            )

            data = await store.fetch(tx_id)
            if not data:
                return {"success": False, "error": f"Failed to fetch from Arweave: {tx_id}"}

            # Phase 7 — detect + decrypt
            record = self._find_backup_record_by_tx("personality", tx_id)
            encryption = (record or {}).get("encryption", {}) or {}
            algo = encryption.get("algorithm", "none")
            if algo != "none":
                logger.info(
                    "[Backup] Restore: encrypted tarball detected (algorithm=%s key=%s) — decrypting",
                    algo, encryption.get("key_id", "?"))
                try:
                    from titan_hcl.logic.backup_crypto import (
                        decrypt_from_manifest, load_keypair_bytes,
                    )
                    import hashlib as _hashlib
                    net = get_params("network") or {}
                    kp_path = net.get(
                        "wallet_keypair_path", "data/titan_identity_keypair.json")
                    kp_bytes, titan_pubkey = load_keypair_bytes(kp_path)
                    data = decrypt_from_manifest(
                        data, encryption, kp_bytes, titan_pubkey, "personality")
                    # Verify plaintext integrity
                    expected = encryption.get("plaintext_sha256")
                    if expected and _hashlib.sha256(data).hexdigest() != expected:
                        return {
                            "success": False,
                            "error": "decrypted plaintext sha256 mismatch — manifest may be corrupt",
                        }
                    logger.info(
                        "[Backup] Restore: decryption OK (%d bytes plaintext)",
                        len(data))
                except Exception as e:
                    msg = f"Decryption failed: {e}"
                    logger.error("[Backup] %s", msg)
                    self._alert_backup_failure("arweave_restore_decrypt", msg)
                    return {"success": False, "error": msg}

            archive_path = f"/tmp/titan_restore_{int(time.time())}.tar.gz"
            with open(archive_path, "wb") as f:
                f.write(data)

            result = self.restore_personality(archive_path)

            # Cleanup
            with suppress(FileNotFoundError):
                os.remove(archive_path)

            return result

        except Exception as e:
            msg = f"Arweave restore failed: {e}"
            logger.error("[Backup] %s", msg)
            self._alert_backup_failure("arweave_restore", msg)
            return {"success": False, "error": msg}

    # -------------------------------------------------------------------------
    # Backup Hash → Solana Memo (daily integrity anchor)
    # Phase 8 — v=2 hash chain (rFP §5.9): each anchor references the previous
    # entry's full archive_hash, making the full backup history tamper-evident
    # on Solana. The local append-only file data/backup_anchor_chain_{titan_id}.json
    # mirrors the on-chain sequence so `arch_map backup --verify-chain` can walk
    # it without scanning Solana memos.
    # -------------------------------------------------------------------------

    def _anchor_chain_path(self) -> str:
        return f"data/backup_anchor_chain_{self._titan_id}.json"

    def _read_chain(self) -> list:
        """Read append-only chain list. Returns [] if file absent/corrupt."""
        import json as _json
        p = self._anchor_chain_path()
        if not os.path.exists(p):
            return []
        try:
            with open(p) as f:
                data = _json.load(f)
            return data.get("anchors", []) if isinstance(data, dict) else []
        except Exception as e:
            logger.warning("[Backup] Chain file unreadable at %s: %s", p, e)
            return []

    def _append_chain_entry(self, entry: dict) -> None:
        """Append a new anchor entry. Atomic write via tmp+rename.

        Entry shape: {backup_id, archive_hash, prev_anchor_hash, tx, ts,
                       backup_type, size_mb}
        """
        import json as _json
        chain = self._read_chain()
        chain.append(entry)
        p = self._anchor_chain_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        payload = {
            "version": 1,
            "titan_id": self._titan_id,
            "anchors": chain,
        }
        with open(tmp, "w") as f:
            _json.dump(payload, f, indent=2)
        os.replace(tmp, p)

    def _chain_tip_hash(self) -> str:
        """archive_hash of the most recent chain entry, or '' if empty."""
        chain = self._read_chain()
        return chain[-1].get("archive_hash", "") if chain else ""

    async def anchor_backup_hash(self, archive_hash: str, size_mb: float,
                                  backup_type: str = "personality") -> Optional[str]:
        """Inscribe backup hash as Solana memo for tamper-proof verification.

        Phase 8 — memo format v=2 includes prev=PREV[:16] linking to the previous
        anchor, forming a verifiable chain. Legacy v=1 parsing remains supported
        in the verifier for pre-Phase-8 entries.

        Format v=2: TITAN|BACKUP|v=2|date=YYYY-MM-DD|h=HASH[:16]|prev=PREV[:16]|size=NNmb|type=TYPE
        Returns TX signature on success, None on failure.
        """
        if not self.network or not hasattr(self.network, 'send_sovereign_transaction'):
            return None

        try:
            from titan_hcl.utils.solana_client import build_memo_instruction, is_available
            if not is_available() or self.network.keypair is None:
                return None

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            prev_hash_full = self._chain_tip_hash()
            prev_fragment = prev_hash_full[:16] if prev_hash_full else "genesis"
            memo_text = (
                f"TITAN|BACKUP|v=2|date={today}|h={archive_hash[:16]}"
                f"|prev={prev_fragment}|size={size_mb:.0f}mb|type={backup_type}"
            )

            memo_ix = build_memo_instruction(self.network.pubkey, memo_text)
            sig = await self.network.send_sovereign_transaction(
                [memo_ix], priority="LOW"
            )

            if sig:
                logger.info("[Backup] Backup hash anchored on-chain: %s (hash=%s...)",
                            sig[:20] if len(sig) > 20 else sig, archive_hash[:12])
                # Phase 8 — append to local chain file only on confirmed on-chain write
                try:
                    self._append_chain_entry({
                        "backup_id": len(self._read_chain()),
                        "archive_hash": archive_hash,
                        "prev_anchor_hash": prev_hash_full,
                        "tx": sig,
                        "ts": int(time.time()),
                        "backup_type": backup_type,
                        "size_mb": round(size_mb, 2),
                    })
                except Exception as e:
                    logger.warning(
                        "[Backup] Anchor succeeded on-chain but local chain "
                        "append failed: %s (recover via memo scan)", e)
            return sig

        except Exception as e:
            logger.warning("[Backup] Backup hash anchor failed (non-critical): %s", e)
            return None

    # ── SPEC §24.12 (5J-2) — Sovereign v=3 chain commit (per-component memos) ─
    async def commit_event_v3_chain(
        self,
        event_id: str,
        ts: int,
        event_type: str,
        event_merkle_root: str,
        components: list,
        prev_sig: Optional[str] = None,
        commit_state: bool = True,
    ) -> Optional[dict]:
        """Emit one v=3 memo per uploaded component for a single backup event.

        `commit_state` (default True) co-bundles the ZK-Vault `commit_state` on the
        head TX — correct for a LIVE event (the vault holds the LATEST event-merkle,
        INV-MBR-4). A historical BACKFILL must pass `commit_state=False`: it posts
        memo-only catalogue TXs and MUST NOT regress the vault PDA to an old event's
        root.

        Option 1 (Maker 2026-05-30): components = ordered
        [{"tier": "PT"|"TC"|"SL", "tx_id": <arweave>, "arc": <tarball sha256>}].
        Each memo carries arc + url (the Arweave tx) + the shared event_merkle_root
        (mrkl); the head (PT) TX co-bundles `commit_state(event_merkle_root)`.
        prev= is event-level — every memo points at the prior event's head sig.

        Mode: `[backup].encryption_enabled` true → data is ciphertext → Mode B
        (plaintext URL); false → Mode A (URL AES-encrypted with the soul-seed
        HKDF key, decryptable only after 2-of-3 Shamir recovery).

        Returns {"head_sig", "component_sigs": {tier: sig}} or None on ANY failure
        (the pipeline then emits BACKUP_EVENT_FAILED — no silent fallback).
        """
        if not self.network or not hasattr(self.network, "send_sovereign_transaction"):
            logger.warning("[Backup] v=3 chain commit: no network client")
            return None
        try:
            from titan_hcl.utils.solana_client import (
                build_memo_instruction, build_vault_commit_instruction, is_available,
            )
            from titan_hcl.logic.backup_memo_v3 import (
                build_v3_memo, derive_backup_url_key,
            )
            from titan_hcl.logic.backup_crypto import load_keypair_bytes
        except Exception as e:
            logger.warning("[Backup] v=3 chain commit import failed: %s", e)
            return None
        if not is_available() or self.network.keypair is None or self.network.pubkey is None:
            logger.warning("[Backup] v=3 chain commit: solana unavailable / no keypair")
            return None
        if not components:
            logger.error("[Backup] v=3 chain commit: empty components")
            return None

        # Mode + Mode-A URL key
        backup_cfg = get_params("backup") or {}
        mode = "B" if backup_cfg.get("encryption_enabled", False) else "A"
        url_key = None
        if mode == "A":
            try:
                net_cfg = get_params("network") or {}
                kp_path = net_cfg.get(
                    "wallet_keypair_path", "data/titan_identity_keypair.json")
                kp_bytes, _pub = load_keypair_bytes(kp_path)
                url_key = derive_backup_url_key(kp_bytes)
            except Exception as e:
                logger.error(
                    "[Backup] v=3 Mode-A URL key derivation failed: %s — aborting", e)
                return None

        vault_program_id = (
            getattr(self, "_vault_program_id", None)
            or getattr(self.network, "_vault_program_id", None)
        )
        if not vault_program_id:
            cfg = getattr(self.network, "_config", None) or self._full_config or {}
            vault_program_id = (cfg.get("network", {}) or {}).get("vault_program_id")

        try:
            # P3 (Synthesis Decision Authority) — the ONE sovereignty score S
            # (basis points), read from the synthesis snapshot (G18). Was the
            # vestigial `_last_sovereignty_idx` (never set → 0bp).
            from titan_hcl.synthesis.sovereignty_readout import (
                rolling_sovereignty_bp,
            )
            sovereignty_bp = rolling_sovereignty_bp()
        except Exception:
            sovereignty_bp = 0

        component_sigs: dict = {}
        head_sig: Optional[str] = None
        chain = self._ensure_chain()  # RFP_chain_provider Phase B tail
        for i, comp in enumerate(components):
            try:
                memo = build_v3_memo(
                    event_id=event_id, ts=int(ts), event_type=event_type,
                    tier=comp["tier"], archive_hash=comp["arc"],
                    merkle_root=event_merkle_root, arweave_tx=comp["tx_id"],
                    mode=mode, prev_sig=prev_sig, url_key=url_key,
                    iv_b64=comp.get("iv") if mode == "B" else None,
                )
            except Exception as e:
                logger.error("[Backup] v=3 memo build failed (%s): %s",
                             comp.get("tier"), e)
                return None
            # RFP_chain_provider Phase B tail — the memo-ix build, the head's
            # vault commit_state bundle, and the tx send all move INTO
            # `chain.commit_memo`: ONE tx per component, `state_root` set ⇒ the
            # head ([commit_ix, memo_ix]); None ⇒ a tail (or a backfill, the
            # commit_state=False case). Memo BUILDING stays above (build_v3_memo).
            head = (i == 0 and commit_state)
            try:
                sig = await chain.commit_memo(
                    memo,
                    state_root=event_merkle_root if head else None,
                    sovereignty_bp=sovereignty_bp if head else None,
                )
            except Exception as e:
                logger.error("[Backup] v=3 chain commit raised (%s): %s",
                             comp.get("tier"), e)
                return None
            if not sig:
                logger.error("[Backup] v=3 chain commit: TX send failed (%s)",
                             comp.get("tier"))
                return None
            component_sigs[comp["tier"]] = sig
            if head_sig is None:
                head_sig = sig
            logger.info("[Backup] v=3 memo anchored: tier=%s evt=%s tx=%s",
                        comp.get("tier"), event_id[:8],
                        sig[:16] if len(sig) > 16 else sig)

        # ── ZK-compressed audit trail (SPEC §B4 — `append_epoch_snapshot`) ──
        # On the HEAD backup event, fire the Light-Protocol compressed-account
        # snapshot in-process (same network client), gated by the per-Titan
        # switch. NON-BLOCKING: the v=3 chain + commit_state already committed
        # above; a failure here never regresses them (PLAN_zk_vault_proof_completion).
        if commit_state and head_sig and self._zk_compressed_audit_enabled():
            try:
                from titan_hcl.logic.zk_vault_state import (
                    emit_epoch_snapshot, read_timechain_block_count,
                    write_zk_audit_state,
                )
                mem_count = read_timechain_block_count()
                if mem_count is None:
                    # INV-NO-STUBS: never write a 0 memory_count — skip honestly.
                    logger.warning(
                        "[Backup] ZK audit: timechain count unavailable — "
                        "skipping epoch snapshot (no 0-stub)")
                    write_zk_audit_state(
                        self._titan_id, enabled=True,
                        last_error="memory_count_unavailable")
                else:
                    head_comp = components[0]
                    arweave_tx = head_comp.get("tx_id", "") or ""
                    arweave_url = (f"https://arweave.net/{arweave_tx}"
                                   if arweave_tx else "")
                    await emit_epoch_snapshot(
                        self.network,
                        state_root_hex=event_merkle_root,
                        sovereignty_bp=sovereignty_bp,
                        archive_hash=head_comp.get("arc", "") or "",
                        arweave_url=arweave_url,
                        titan_id=self._titan_id,
                        memory_count=mem_count,
                        program_id_str=vault_program_id,
                        photon=getattr(self, "_photon", None),
                    )
                    # E2 (option A): alongside the E1 append-trail, write the
                    # running canonical SovereignState (SNARK-per-write). Selector
                    # (create-vs-update) lives inside emit_sovereign_state. Needs
                    # Photon (proofs) — no-ops cleanly if it's absent.
                    photon = getattr(self, "_photon", None)
                    if photon is not None:
                        from titan_hcl.logic.zk_vault_state import emit_sovereign_state
                        await emit_sovereign_state(
                            self.network,
                            state_root_hex=event_merkle_root,
                            sovereignty_bp=sovereignty_bp,
                            arweave_url=arweave_url,
                            archive_hash=head_comp.get("arc", "") or "",
                            titan_id=self._titan_id,
                            memory_count=mem_count,
                            program_id_str=vault_program_id,
                            photon=photon,
                        )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[Backup] ZK-compressed audit snapshot failed "
                    "(non-critical): %s", e)

        return {"head_sig": head_sig, "component_sigs": component_sigs}

    def _zk_compressed_audit_enabled(self) -> bool:
        """Per-Titan switch for the ZK-compressed audit trail (SPEC §B4
        append_epoch_snapshot on the backup event). Default OFF."""
        try:
            return bool(
                get_params("backup").get(
                    "zk_compressed_audit_enabled", False)
            )
        except Exception:  # noqa: BLE001
            return False

    # ── SPEC §24 — Unified backup pipeline (Phase 5.5, 2026-05-16) ─────────
    #
    # Production wiring for backup_upload_pipeline.run_unified_event.
    # Gated by [backup].unified_v2_enabled (default false). When enabled,
    # replaces the legacy full-tarball personality+soul+timechain cascade
    # in on_meditation_complete with one event-level ship that records to
    # UnifiedManifest + commits to ZK Vault per event.

    def _unified_v2_enabled(self) -> bool:
        try:
            cfg = self._full_config or {}
            return bool(
                get_params("backup").get("unified_v2_enabled", False)
            )
        except Exception:
            return False

    def _baseline_working_dir(self) -> str:
        """Per-titan baseline-snapshot dir. After each baseline event, the
        current on-disk source files for personality + timechain + soul are
        copied here so subsequent incremental events have a stable
        diff-source. Refreshed by `_refresh_baseline_working_dir`."""
        return f"data/backups/unified_baseline_{self._titan_id}"

    def _refresh_baseline_working_dir(self) -> None:
        """Snapshot current source files into baseline working dir. Called
        after a baseline event ships so incremental events can diff
        against a stable reference rather than re-snapshotting at every
        meditation."""
        import shutil
        base = self._baseline_working_dir()
        os.makedirs(base, exist_ok=True)
        for paths in (self.PERSONALITY_PATHS, self.TIMECHAIN_PATHS,
                      self.WEEKLY_EXTRA_PATHS):
            for entry in paths:
                if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                    continue
                src, arc = entry[0], entry[1]
                if not os.path.exists(src):
                    continue
                dst = os.path.join(base, arc)
                try:
                    parent = os.path.dirname(dst)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
                except Exception as e:
                    logger.warning(
                        "[Backup] baseline snapshot of %s → %s failed: %s",
                        src, dst, e,
                    )

    def _refresh_baseline_dir_from_tarballs(self, tarball_paths: list) -> int:
        """§24 #3 (2026-06-01) — refresh the baseline working dir from the
        FULL-mode files INSIDE the just-shipped tarballs, NOT a re-copy of live
        source.

        The baseline-dir bytes MUST equal the bytes uploaded to Arweave so a
        future incremental diff (xdelta3) reconstructs correctly at restore.
        Re-copying live source drifts — especially on the STAGED path, where
        the pack (off-loop stager) and the post-ship refresh can be ~20 min of
        live writes apart, and for live sqlite DBs a raw re-read is a torn
        snapshot. The shipped tarball IS the canonical bytes; extracting its
        full-mode payloads makes the diff base exact.

        Only diff_mode=="full" files are written (a baseline's tiers are all
        full; a Sunday incremental's soul diff is left to anchor on the prior
        full soul). get_patch_bytes() verifies each file's sha256.
        """
        from titan_hcl.logic.backup_event_tarball import unpack_event_tarball
        base = self._baseline_working_dir()
        os.makedirs(base, exist_ok=True)
        written = 0
        for tp in tarball_paths:
            if not tp or not os.path.exists(tp):
                continue
            with open(tp, "rb") as f:
                data = f.read()
            with unpack_event_tarball(data) as unpacked:
                for fm in unpacked.files:
                    if fm.get("diff_mode") != "full":
                        continue
                    arc = fm.get("arc_name")
                    if not arc:
                        continue
                    payload = unpacked.get_patch_bytes(arc)  # full bytes (sha-verified)
                    dst = os.path.join(base, arc)
                    parent = os.path.dirname(dst)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    tmp = dst + ".tmp"
                    with open(tmp, "wb") as out:
                        out.write(payload)
                    os.replace(tmp, dst)
                    written += 1
        logger.info(
            "[Backup] §24 baseline-dir refreshed from %d full-mode file(s) "
            "across %d tarball(s) — bytes == shipped (no source-drift)",
            written, len(tarball_paths))
        return written

    # ── Phase B: chained-incremental diff-base (RFP_backup_arweave_sustainability, 2026-06-09) ──
    #
    # The baseline-mirror (`_baseline_working_dir`) becomes a ROLLING "last-shipped
    # state" mirror: each incremental diffs vs the PREVIOUS event's reconstructed
    # bytes (held in the mirror), not vs the monthly baseline → ~3-4x less Arweave.
    # A `.mirror_state.json` sidecar records {event_id, arc→sha} so a cheap daily
    # precheck can detect mirror drift; a missing diff-base for a KNOWN file forces
    # a labeled `self_heal` baseline (never a silent full-ship — INV-BR-9). The flag
    # gates the chained behaviour; the no-silent-full-ship safety is always on.

    def _chained_incrementals_enabled(self) -> bool:
        """Flag gate. Runtime config.toml [backup].chained_incrementals overrides
        the committed default _BACKUP_CHAINED_INCREMENTALS_DEFAULT."""
        cfg = get_params("backup") or {}
        return bool(cfg.get("chained_incrementals",
                            _BACKUP_CHAINED_INCREMENTALS_DEFAULT))

    def _load_mirror_state(self) -> Optional[dict]:
        """The rolling-mirror integrity sidecar — {"event_id": str, "chained":
        bool, "arcs": {arc: sha}} — Phase D (INV-BRS-7): now lives IN-MEMORY in
        the consolidated _BACKUP_STATE_PATH, not a co-located .mirror_state.json.
        None if never written (→ precheck adopts the mirror or recovers,
        fail-closed). A regenerable cache: a loss only forces a self_heal
        baseline, never data loss (a stale .mirror_state.json from before the
        Phase-D deploy is simply ignored → one self-heal baseline on first run)."""
        st = self._mirror_state
        if not isinstance(st, dict) or not isinstance(st.get("arcs"), dict):
            return None
        return st

    def _write_mirror_state(self, event_id: str, arcs: dict) -> None:
        """Record the mirror's arc→sha map after an advance/recover (Phase D:
        into the consolidated state object, persisted via _save_backup_state)."""
        self._mirror_state = {
            "event_id": event_id,
            "chained": self._chained_incrementals_enabled(),
            "arcs": arcs, "ts": time.time()}
        self._save_backup_state()

    def _hash_mirror_dir(self, base: str) -> dict:
        """{arc(relpath): file_merkle_root} for every file in the mirror, EXCLUDING
        the sidecar + transient scratch. This is the true mirror state."""
        from titan_hcl.logic import diff_encoders
        out: dict = {}
        if not os.path.isdir(base):
            return out
        for root, _dirs, files in os.walk(base):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, base)
                if (arc == ".mirror_state.json" or arc.endswith(".tmp")
                        or arc.endswith(".advancing")):
                    continue
                try:
                    out[arc] = diff_encoders.file_merkle_root(full)
                except OSError:
                    continue
        return out

    def _alarm_self_heal_baseline(self, reason: str) -> None:
        """maker_notify + WARN when a missing/drifted diff-base forces a labeled
        self_heal baseline (INV-BR-9). The baseline preserves lineage (it is
        prev_event_id-linked, not an orphan) — this is the LOUD, never-silent
        recovery that replaces the 06-03 silent full-ship."""
        msg = (f"⚠️ §24 backup self-heal: forcing a labeled baseline (diff-base "
               f"recovery) — {reason}. Chain lineage preserved (prev-linked); the "
               f"next backup is a full snapshot.")
        logger.warning("[Backup] %s", msg)
        try:
            send_maker_alert(msg, alert_key="backup.self_heal",
                             rate_limit_seconds=3600.0)
        except Exception:
            pass

    def _make_diff_base_resolver(self, base_dir: str, known_arcs):
        """Build the baseline_resolver passed to the pipeline. Returns the mirror
        path for a present arc; RAISES for a KNOWN arc whose bytes vanished (the
        fail-closed backstop — the precheck must have forced a baseline first);
        returns None for a NEW arc (never shipped) → legit per-file full-ship."""
        from titan_hcl.logic.backup_upload_pipeline import MissingDiffBaseError

        def _resolver(component, arc_name):
            candidate = os.path.join(base_dir, arc_name)
            if os.path.exists(candidate):
                return candidate
            if arc_name in known_arcs:
                raise MissingDiffBaseError(
                    f"diff-base for KNOWN arc {arc_name!r} missing under "
                    f"{base_dir!r} — the precheck must force a baseline "
                    f"(INV-BR-9); refusing a silent full-ship")
            return None  # NEW file (never shipped) → legitimate per-file full-ship
        return _resolver

    def _mirror_verifies(self, base_dir: str, known_arcs, state, latest_id) -> bool:
        """Chained-mode check: the sidecar tracks the LATEST event AND every known
        arc's mirror bytes still hash to the recorded sha."""
        from titan_hcl.logic import diff_encoders
        if not state or state.get("event_id") != latest_id:
            return False
        recorded = state.get("arcs", {})
        for arc in known_arcs:
            p = os.path.join(base_dir, arc)
            expected = recorded.get(arc)
            if expected is None or not os.path.exists(p):
                return False
            try:
                if diff_encoders.file_merkle_root(p) != expected:
                    return False
            except OSError:
                return False
        return True

    def _rebase_params(self) -> tuple[str, int]:
        """Backup CADENCE by mode (RFP_backup_redesign_spine, 2026-06-11).

        Discriminator = the MAINNET BIRTH CERTIFICATE (data/genesis_record.json —
        the on-chain genesis-NFT ceremony record). A mainnet-born Titan has it;
        devnet/local installs do NOT (verified: T1 has it, T3 doesn't). This is the
        RELIABLE signal — `self._arweave_store is not None` is NOT, because
        `_ensure_arweave_store_for_unified` lazy-inits a *devnet* ArweaveStore (the
        local arweave_devnet/ cache) on devnet too, so it goes non-None there.
          • mainnet (birth cert present) → ("none", 90): incremental AS LONG AS
            POSSIBLE; a full baseline (expensive on Arweave) fires only on
            first_event / self_heal / the depth_cap-90 restore-chain safety. No
            calendar trigger → a missed daily incremental (low SOL) is NOT a rebase;
            the next incremental covers the gap.
          • devnet/local (no birth cert) → ("weekly", 30): re-baseline once the
            baseline is ≥7d old, then prune older events (local disk is cheap)."""
        if os.path.exists(os.path.join("data", "genesis_record.json")):
            return ("none", 90)
        return ("weekly", 30)

    def _configure_manifest_cadence(self, manifest) -> None:
        """Apply this Titan's mode cadence to a freshly-loaded manifest so every
        should_rebase() call on it is mode-aware (the one configured object covers
        the precheck, plan_build, and build_unified_event callers). Defensive: a
        test double without configure_rebase is left at the legacy default."""
        if hasattr(manifest, "configure_rebase"):
            manifest.configure_rebase(*self._rebase_params())

    def _prune_old_local_backups(self, manifest) -> int:
        """devnet/local RETENTION (Maker 2026-06-11): after a weekly baseline lands,
        delete BOTH the manifest events AND their local devnet-cache tarballs that
        PRECEDE the current baseline. mainnet keeps INDEFINITE retention (the on-chain
        chain is the sovereign record — never prune it). Safe to call after every
        ship: it no-ops unless the current baseline has older events before it (i.e.
        right after a fresh weekly baseline). Returns the number of events pruned."""
        cadence, _ = self._rebase_params()
        if cadence != "weekly":          # mainnet/none → indefinite retention
            return 0
        if not hasattr(manifest, "prune_before_current_baseline"):
            return 0
        pruned = manifest.prune_before_current_baseline()
        if not pruned:
            return 0
        cache = os.path.join("data", "arweave_devnet")
        deleted = 0
        for ev in pruned:
            for tier in ("personality", "timechain", "soul"):
                t = ev.get(tier)
                tx = t.get("tx_id") if isinstance(t, dict) else None
                if not tx:
                    continue
                # devnet cache: file://data/arweave_devnet/<tx>.data (+ .tags.json)
                txid = (tx.replace("file://data/arweave_devnet/", "")
                          .replace(".data", ""))
                for suffix in (".data", ".tags.json"):
                    p = os.path.join(cache, f"{txid}{suffix}")
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                            deleted += 1
                        except OSError as e:
                            logger.warning(
                                "[Backup] §24 retention: could not delete %s: %s",
                                p, e)
        logger.info(
            "[Backup] §24 retention (devnet/local): pruned %d pre-baseline event(s) "
            "+ %d local cache file(s) older than baseline %s",
            len(pruned), deleted,
            (manifest.current_baseline_event_id or "?")[:8])
        return len(pruned)

    async def _precheck_diff_base(self, manifest):
        """RFP Phase B integrity precheck — runs BEFORE the tier build in BOTH ship
        paths. Returns (force_event_type, force_trigger, known_arcs).

        ALWAYS-ON (flag-independent): a KNOWN file whose mirror diff-base is gone →
        a labeled self_heal baseline (never a silent full-ship — INV-BR-9).
        CHAINED (flag on): additionally merkle-verifies the rolling mirror vs the
        sidecar and runs the reconstruct ladder (incl. the Q2 flag-flip bridge)
        before falling back to a baseline."""
        # R4 recovery (§24.12): a failed weekly restore-test armed a one-shot
        # force-baseline marker → the chain is suspect, so the FIRST event after
        # the halt is cleared rebases to a clean labeled baseline (INV-BKP-5),
        # never an append to a possibly-broken chain. One-shot: consumed as it fires.
        if self._take_force_baseline():
            logger.warning("[Backup] §24.12 recovery — forcing a clean baseline "
                           "after a failed weekly restore-test")
            return ("baseline", "self_heal", set())
        # A baseline full-ships regardless — nothing to pre-check.
        _cad, _depth = self._rebase_params()
        should_rebase, _ = manifest.should_rebase(cadence=_cad, depth_cap=_depth)
        base_dir = self._baseline_working_dir()
        if should_rebase:
            return (None, None, set())

        state = self._load_mirror_state()
        chained = self._chained_incrementals_enabled()

        if state is not None:
            known_arcs = set(state.get("arcs", {}).keys())
        else:
            # Transitional (no sidecar): adopt the existing mirror as the known
            # diff-base; an empty mirror despite a prior baseline = the 06-03
            # infra failure → fail closed to a labeled baseline.
            present = self._hash_mirror_dir(base_dir)
            if not present:
                self._alarm_self_heal_baseline(
                    "no mirror-state sidecar AND empty baseline mirror "
                    "(diff-base infrastructure missing)")
                return ("baseline", "self_heal", set())
            adopt_event = manifest.current_baseline_event_id or "adopted"
            self._write_mirror_state(adopt_event, present)
            state = {"event_id": adopt_event, "arcs": present}
            known_arcs = set(present.keys())
            logger.info("[Backup] §24 Phase B: adopted existing mirror as diff-base "
                        "(%d arcs, no prior sidecar)", len(known_arcs))

        missing_known = [a for a in known_arcs
                         if not os.path.exists(os.path.join(base_dir, a))]

        if not chained:
            # FLAG OFF (R2 only): existence check on KNOWN arcs. Any gone → baseline.
            if missing_known:
                self._alarm_self_heal_baseline(
                    f"{len(missing_known)} known diff-base file(s) missing from the "
                    f"baseline mirror (cumulative mode)")
                return ("baseline", "self_heal", known_arcs)
            return (None, None, known_arcs)

        # FLAG ON (chained): verify mirror == latest shipped state, reconstruct if not.
        latest = manifest.get_latest_event()
        latest_id = latest["event_id"] if latest else None
        if self._mirror_verifies(base_dir, known_arcs, state, latest_id):
            return (None, None, known_arcs)

        # Reconstruct ladder — rebuild the mirror to the LATEST event's state from
        # the chain (also performs the Q2 flag-flip BRIDGE when the mirror lags).
        recovered, halt_reason = (False, None)
        if latest_id:
            recovered, halt_reason = await self._recover_mirror_to_latest(
                manifest, latest_id)
        if recovered:
            state = self._load_mirror_state()
            known_arcs = set(state.get("arcs", {}).keys()) if state else known_arcs
            if self._mirror_verifies(base_dir, known_arcs, state, latest_id):
                logger.info("[Backup] §24 Phase B: mirror recovered to latest %s — "
                            "chained incremental", (latest_id or "?")[:8])
                return (None, None, known_arcs)

        # Reconstruct FAILED → classify GENUINE corruption vs TRANSIENT unreachable
        # before ever paying for a baseline (Maker rule: baseline only on a real
        # integrity failure, never because the chain couldn't be fetched right now).
        from titan_hcl.logic.backup_restore import (
            HALT_BROKEN_CHAIN, HALT_TARBALL_HASH_MISMATCH,
            HALT_EVENT_MERKLE_MISMATCH, HALT_ZK_MEMO_MISMATCH, HALT_ZK_DISCONNECT,
            HALT_APPLY_FAILED, HALT_POST_RESTORE_HASH_MISMATCH)
        _GENUINE_CORRUPTION = {
            HALT_BROKEN_CHAIN, HALT_TARBALL_HASH_MISMATCH, HALT_EVENT_MERKLE_MISMATCH,
            HALT_ZK_MEMO_MISMATCH, HALT_ZK_DISCONNECT, HALT_APPLY_FAILED,
            HALT_POST_RESTORE_HASH_MISMATCH}
        if halt_reason == "no_store":
            # No chain store/infra at all — a STRUCTURAL failure (not a transient
            # fetch), and we can't even probe; self_heal is the only path.
            self._alarm_self_heal_baseline(
                "chained mirror reconstruct: no chain store available")
            return ("baseline", "self_heal", known_arcs)
        if halt_reason in _GENUINE_CORRUPTION:
            # The chain is reachable but its DATA is corrupt — self_heal is the
            # honest, lineage-preserving recovery.
            self._alarm_self_heal_baseline(
                f"chained mirror reconstruct: chain DATA corrupt ({halt_reason})")
            return ("baseline", "self_heal", known_arcs)

        # Fetch/connectivity failure (or unknown) — only self_heal if the chain is
        # DEFINITIVELY gone. A cheap daemon-independent gateway HEAD distinguishes:
        head = await self._probe_latest_chain_head(manifest, latest_id)
        if head == "missing":
            self._alarm_self_heal_baseline(
                "chained mirror reconstruct: latest chain tx DEFINITIVELY missing "
                "on-chain (chain genuinely gone)")
            return ("baseline", "self_heal", known_arcs)

        # head=present (chain reachable → transient fetch) or head=unverified
        # (uncertain → never self_heal on uncertainty): DEFER + retry next tick.
        # This is the boot-ordering root-cause fix — the reconstruct ran before the
        # chain provider was ready; it must NOT manufacture a self_heal baseline.
        logger.warning(
            "[Backup] §24 Phase B: chained mirror reconstruct DEFERRED — chain is "
            "reachable/unproven-gone (head=%s, halt=%s); the provider is likely "
            "still warming at boot. Skipping the plan this tick (NO self_heal "
            "baseline on a transient); will retry + ship a cheap catch-up "
            "incremental once the chain is fetchable.", head, halt_reason)
        raise BackupReconstructDeferred(
            f"reconstruct transient (head={head}, halt={halt_reason})")

    def _build_decrypting_fetch(self, manifest):
        """An async arweave_fetch(tx_id)->PLAINTEXT bytes for restore_full that
        transparently Mode-B decrypts using the LOCAL keypair + the manifest's
        per-tx `iv` (no Solana RPC). Mode-A (iv None) → passthrough. Reuses the
        audited backup_crypto helpers (same as the sovereign restore)."""
        store = self._ensure_arweave_store_for_unified()
        tx_meta: dict = {}
        for ev in manifest.events:
            for comp in ("personality", "timechain", "soul"):
                sub = ev.get(comp)
                if not isinstance(sub, dict):
                    continue
                tx = sub.get("tx_id")
                if tx:
                    tx_meta[tx] = {"iv": sub.get("iv"), "component": comp,
                                   "arc": sub.get("merkle_root")}
        cache: dict = {"master": None}

        def _master_key():
            if cache["master"] is None:
                from titan_hcl.logic.backup_crypto import (
                    load_keypair_bytes, derive_master_key)
                cfg = get_params("backup") or {}
                kp_path = cfg.get("wallet_keypair_path",
                                  "data/titan_identity_keypair.json")
                kp_bytes, titan_pubkey = load_keypair_bytes(kp_path)
                cache["master"] = derive_master_key(kp_bytes, titan_pubkey)
            return cache["master"]

        async def _fetch(tx_id: str) -> bytes:
            data = await store.fetch(tx_id)
            if not data:
                raise RuntimeError(f"Arweave fetch returned no data for {tx_id}")
            meta = tx_meta.get(tx_id)
            if meta and meta.get("iv"):
                from titan_hcl.logic.backup_crypto import decrypt_component_tarball
                data = decrypt_component_tarball(
                    bytes(data), meta["iv"], _master_key(),
                    meta["component"], meta["arc"])
            return bytes(data)
        return _fetch

    def _build_fetch_to_file(self, manifest):
        """An async fetch_to_file(tx_id, dest_path)->bool for restore_full that
        STREAMS each component tarball from Arweave straight to disk in fixed
        chunks — constant memory regardless of tarball size. This is the 2026-06-09
        RSS close: the in-RAM `_build_decrypting_fetch` held the whole ~595MB
        compressed salvage tarball (→ ~960MB worker RSS, over the 500MB rss_limit);
        streaming keeps the live chained reconstruction under ~100MB. Mode-A (iv
        None) streams the plaintext tarball directly; Mode-B decrypts to dest. The
        staged file is ALWAYS the PLAINTEXT tarball, so its sha256 == the component
        merkle_root (restore_full verifies that streamed). Reuses the audited
        backup_crypto helpers + the streaming `ChainProvider.get_to_file`
        (RFP_chain_provider Phase A — the restore-fetch caller migrated here)."""
        chain = self._ensure_chain()
        tx_meta: dict = {}
        for ev in manifest.events:
            for comp in ("personality", "timechain", "soul"):
                sub = ev.get(comp)
                if not isinstance(sub, dict):
                    continue
                tx = sub.get("tx_id")
                if tx:
                    tx_meta[tx] = {"iv": sub.get("iv"), "component": comp,
                                   "arc": sub.get("merkle_root")}
        cache: dict = {"master": None}

        def _master_key():
            if cache["master"] is None:
                from titan_hcl.logic.backup_crypto import (
                    load_keypair_bytes, derive_master_key)
                cfg = get_params("backup") or {}
                kp_path = cfg.get("wallet_keypair_path",
                                  "data/titan_identity_keypair.json")
                kp_bytes, titan_pubkey = load_keypair_bytes(kp_path)
                cache["master"] = derive_master_key(kp_bytes, titan_pubkey)
            return cache["master"]

        async def _fetch_to_file(tx_id: str, dest_path: str) -> bool:
            if chain is None:
                return False
            meta = tx_meta.get(tx_id)
            if meta and meta.get("iv"):
                # Mode-B: stream the CIPHERTEXT to a temp, then decrypt to dest.
                # GCM auth needs the whole ciphertext so the decrypt itself is a
                # single read — but it's the encrypted-user fallback (T1 is
                # Mode-A) and still far below the old all-components-in-RAM peak.
                ct_tmp = dest_path + ".ct"
                ok = await chain.get_to_file(tx_id, ct_tmp)
                if not ok:
                    return False
                try:
                    from titan_hcl.logic.backup_crypto import (
                        decrypt_component_tarball)
                    with open(ct_tmp, "rb") as f:
                        ct = f.read()
                    pt = decrypt_component_tarball(
                        ct, meta["iv"], _master_key(),
                        meta["component"], meta["arc"])
                    with open(dest_path, "wb") as f:
                        f.write(pt)
                    return True
                finally:
                    try:
                        os.unlink(ct_tmp)
                    except OSError:
                        pass
            # Mode-A: stream the plaintext tarball straight to dest (chunked).
            return await chain.get_to_file(tx_id, dest_path)
        return _fetch_to_file

    async def _recover_mirror_to_latest(self, manifest, latest_id):
        """Reconstruct the rolling mirror to the LATEST shipped event's state via
        the chain-aware restore_full (decrypting fetch for Mode-B). Small per-link
        patches, NO new full upload. Also performs the Q2 flag-flip bridge.

        Returns (recovered: bool, halt_reason: Optional[str]). On failure the
        halt_reason lets the precheck classify GENUINE corruption (→ self_heal)
        vs a TRANSIENT fetch/connectivity failure (→ defer, never self_heal)."""
        base_dir = self._baseline_working_dir()
        store = self._ensure_arweave_store_for_unified()
        if store is None:
            logger.warning("[Backup] §24 Phase B recover: no ArweaveStore")
            return (False, "no_store")
        try:
            from titan_hcl.logic.backup_restore import restore_full
            os.makedirs(base_dir, exist_ok=True)
            fetch = self._build_decrypting_fetch(manifest)

            async def _memo_noop(_sig):
                return ""

            def _arc_to_mirror(component, arc_name):
                return os.path.join(base_dir, arc_name)

            result = await restore_full(
                manifest=manifest, target_dir=base_dir,
                arweave_fetch=fetch, memo_fetch=_memo_noop,
                arc_to_target=_arc_to_mirror, target_event_id=latest_id,
                # STREAM each tarball to disk (constant memory) — keeps the live
                # reconstruction under backup's 500MB rss_limit. The in-RAM
                # `fetch` stays as the fallback for any tx the streamer can't
                # handle; bytes are sha-verified vs the manifest, and the weekly
                # §24.12 test does the full on-chain ZK walk (R4).
                arweave_fetch_to_file=self._build_fetch_to_file(manifest),
                verify_zk_chain=False,
                # T1's baseline (560d, packed 2026-05-29) predates the ed5f4d0c
                # copy-snapshot truncation-race fix (2026-05-31), so some per-file
                # patch_bytes_sha256 / post-apply merkle values are STALE — the data
                # is intact (the component tarball's sha256 IS verified above vs the
                # manifest merkle_root, INV-MBR-4), but the per-file hash is a
                # false-positive. Same situation the sovereign restore handles
                # (mainnet §R4/INV-MBR-12, v0.4.6): downgrade per-file/post-apply
                # mismatches to advisory once the component arc is verified. The
                # xdelta3 baseline_merkle_root check STAYS strict (wrong source =
                # garbage) — only the redundant per-file hashes go advisory.
                verify_patch_hash=False)
            if result.status != "success":
                halt = getattr(result, "halt_reason", None)
                logger.warning("[Backup] §24 Phase B recover: restore_full "
                               "status=%s halt=%s", result.status, halt)
                return (False, halt)
            self._write_mirror_state(latest_id, self._hash_mirror_dir(base_dir))
            logger.info("[Backup] §24 Phase B: mirror RECONSTRUCTED to latest %s "
                        "(%d files)", (latest_id or "?")[:8], result.restored_files)
            return (True, None)
        except Exception as e:
            logger.warning("[Backup] §24 Phase B recover: reconstruct raised: %s", e)
            return (False, "exception")

    async def _probe_latest_chain_head(self, manifest, latest_id) -> str:
        """Cheap reachability probe used to gate self_heal: HEAD the LATEST event's
        component tx_ids via the gateway (a direct HTTP HEAD — daemon-INDEPENDENT,
        so it works even when the fetch path's Irys daemon is still warming at
        boot). Returns:
          • 'present'    — ANY component tx is on the gateway (HTTP 200): the chain
                           is reachable, so the reconstruct's fetch failure was
                           TRANSIENT → DEFER, never self_heal.
          • 'missing'    — EVERY probed tx is a DEFINITIVE 404/410: the chain is
                           genuinely gone → self_heal is the honest recovery.
          • 'unverified' — timeouts / no tx_ids / uncertain: never proven gone →
                           treated as transient (DEFER). INV: self_heal needs
                           POSITIVE proof the chain is gone, never a guess."""
        latest = manifest.get_latest_event()
        if not latest:
            return "unverified"
        txs = []
        for comp in ("personality", "timechain", "soul"):
            sub = latest.get(comp)
            if isinstance(sub, dict) and sub.get("tx_id"):
                txs.append(sub["tx_id"])
        if not txs:
            return "unverified"
        chain = self._ensure_chain()
        statuses = []
        for tx in txs:
            try:
                statuses.append(await chain.head(tx))
            except Exception as e:  # noqa: BLE001
                logger.debug("[Backup] §24 Phase B head probe raised for %s: %s",
                             str(tx)[:16], e)
                statuses.append("unverified")
        if any(s == "present" for s in statuses):
            return "present"
        if statuses and all(s == "missing" for s in statuses):
            return "missing"
        return "unverified"

    def _advance_mirror_from_tarballs(self, tarball_paths, event_id) -> int:
        """RFP Phase B mirror ADVANCE — make the rolling mirror equal the just-shipped
        event's reconstructed state (the next event's chained diff-base), from the
        shipped TARBALLS (canonical bytes; never a torn live-sqlite re-read), per arc
        by diff_mode: full→write bytes · incremental→apply_diff onto the mirror ·
        skipped→no-op. Then write the .mirror_state.json sidecar atomically. Returns
        the count of files written/advanced. (Superset of
        _refresh_baseline_dir_from_tarballs — also handles incremental + the sidecar.)"""
        from titan_hcl.logic.backup_event_tarball import unpack_event_tarball
        from titan_hcl.logic import diff_encoders
        base = self._baseline_working_dir()
        os.makedirs(base, exist_ok=True)
        written = 0
        for tp in tarball_paths:
            if not tp or not os.path.exists(tp):
                continue
            with open(tp, "rb") as f:
                data = f.read()
            with unpack_event_tarball(data) as unpacked:
                for fm in unpacked.files:
                    arc = fm.get("arc_name")
                    if not arc:
                        continue
                    mode = fm.get("diff_mode")
                    if mode == "skipped":
                        continue  # mirror already holds the correct bytes
                    dst = os.path.join(base, arc)
                    parent = os.path.dirname(dst)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    if mode == "full":
                        payload = unpacked.get_patch_bytes(arc)  # full bytes (sha-verified)
                        tmp = dst + ".tmp"
                        with open(tmp, "wb") as out:
                            out.write(payload)
                        os.replace(tmp, dst)
                    else:
                        # incremental diff — apply onto the mirror's prior bytes.
                        diff_dict = unpacked.diff_dict_for(arc, verify_hash=True)
                        baseline_path = dst if os.path.exists(dst) else None
                        scratch = dst + ".advancing"
                        diff_encoders.apply_diff(
                            baseline_path=baseline_path, diff_dict=diff_dict,
                            output_path=scratch, verify_output=True)
                        os.replace(scratch, dst)
                    written += 1
        self._write_mirror_state(event_id, self._hash_mirror_dir(base))
        logger.info("[Backup] §24 Phase B: mirror ADVANCED to event %s "
                    "(%d files written/advanced)", (event_id or "?")[:8], written)
        return written

    # ── Phase R4: weekly full-chain restore-test + self-heal halt (§24.12, 2026-06-09) ──

    def _is_backups_halted(self) -> bool:
        """True if a failed weekly restore-test HALTED scheduled backups (INV-BR-4).
        Phase D (INV-BRS-7): in-memory, loaded from _BACKUP_STATE_PATH at boot +
        flushed IMMEDIATELY on set/clear. The consolidated state file is atomic
        (tmp+os.replace → never torn), so the prior 'fail-closed on an unreadable
        SEPARATE halt file' paranoia is subsumed (an unreadable main state file is
        a far bigger problem, surfaced by _load_backup_state)."""
        return bool(self._halted)

    def _set_backups_halt(self, reason: str, failed_event_id) -> None:
        """HALT scheduled backups + arm the one-shot force-baseline recovery
        token (INV-BR-4 / INV-BKP-5). force_baseline_pending survives a halt-clear
        (so the resume event rebases to a clean baseline). Persists IMMEDIATELY —
        a crash must never lose a halt → never back up a suspect chain."""
        self._halted = True
        self._halt_reason = reason
        self._halt_failed_event_id = failed_event_id
        self._force_baseline_pending = True
        self._save_backup_state()

    def _clear_backups_halt(self) -> None:
        """Lift the halt (a green restore-test auto-clears; or an investigated
        manual clear). Leaves force_baseline_pending intact → the resume event is
        a baseline. Persists immediately."""
        if self._halted:
            self._halted = False
            self._halt_reason = ""
            self._save_backup_state()

    def _take_force_baseline(self) -> bool:
        """One-shot: consume the R4 recovery force-baseline token if armed."""
        if self._force_baseline_pending:
            self._force_baseline_pending = False
            self._save_backup_state()
            return True
        return False

    def _build_memo_fetch(self):
        """Live memo_fetch (Solana sig → SPL-Memo text) for verify_zk_chain=True."""
        async def _fetch(sig: str):
            from titan_hcl.utils import solana_client
            return await solana_client.get_memo_for_tx(sig)
        return _fetch

    async def _run_weekly_restore_test(self, *, memo_fetch=None,
                                       bus_emit=None) -> bool:
        """§24.12 — weekly FULL-chain restore-test. Reconstructs the ENTIRE live chain
        into a SCRATCH dir via restore_full (Mode-B decrypting fetch + verify_zk_chain
        =True → restore_full byte-verifies every link's tarball sha, each apply's
        output merkle, AND the on-chain v=3 ZK memo per event). So result.status==
        "success" IS the byte-for-byte + on-chain proof (§24.12).

        PASS → emit BACKUP_RESTORE_TEST_PASS + clear any stale halt.
        FAIL → logger.critical + maker_notify + HALT scheduled backups (INV-BR-4)
               + arm the one-shot force-baseline recovery (INV-BKP-5).
        Returns True on PASS. Read-only — scratch dir only; never touches live data/."""
        import shutil
        import tempfile
        from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
        from titan_hcl.logic.backup_restore import restore_full
        from titan_hcl.logic.backup_upload_pipeline import (
            EVENT_BACKUP_RESTORE_TEST_PASS, EVENT_BACKUP_RESTORE_TEST_FAIL)
        started = time.time()
        store = self._ensure_arweave_store_for_unified()
        if store is None:
            logger.warning("[Backup] §24.12 restore-test: no ArweaveStore — skipped")
            return False
        try:
            manifest = UnifiedManifest.load(titan_id=self._titan_id, base_dir="data")
        except ValueError as e:
            logger.error("[Backup] §24.12 restore-test: manifest load failed: %s", e)
            return False
        if not manifest.events:
            logger.info("[Backup] §24.12 restore-test: no events yet — skipped")
            return False

        scratch = tempfile.mkdtemp(prefix=f"titan_restore_test_{self._titan_id}_")
        fetch = self._build_decrypting_fetch(manifest)
        mfetch = memo_fetch or self._build_memo_fetch()

        def _arc_to_scratch(component, arc_name):
            return os.path.join(scratch, component, arc_name)

        try:
            result = await restore_full(
                manifest=manifest, target_dir=scratch,
                arweave_fetch=fetch, memo_fetch=mfetch,
                arc_to_target=_arc_to_scratch, verify_zk_chain=True)
            dur = time.time() - started
            if result.status == "success":
                logger.info("[Backup] §24.12 restore-test PASS — walked %d events, "
                            "%d files verified, %.1fs",
                            len(result.applied_events), result.restored_files, dur)
                self._clear_backups_halt()
                if bus_emit:
                    bus_emit(EVENT_BACKUP_RESTORE_TEST_PASS, {
                        "event": EVENT_BACKUP_RESTORE_TEST_PASS,
                        "event_walked_to": result.target_event_id,
                        "events": len(result.applied_events),
                        "files_verified": result.restored_files,
                        "duration_s": dur})
                return True
            # FAIL — the chain may be broken; halt + alarm loudly (Maker-confirmed:
            # HALT + notify Maker + a CRITICAL journal log).
            reason = getattr(result, "halt_reason", None) or "restore_failed"
            fe = getattr(result, "halt_event_id", None)
            logger.critical(
                "[Backup] §24.12 WEEKLY RESTORE-TEST FAILED — scheduled Arweave "
                "backups HALTED. halt_reason=%s halt_event=%s errors=%s. The "
                "restore-from-Arweave chain may be BROKEN; investigate before "
                "clearing the halt (INV-BR-4 / INV-BKP-5).",
                reason, fe, result.errors)
            self._set_backups_halt(reason=reason, failed_event_id=fe)
            try:
                send_maker_alert(
                    f"🛑 §24.12 WEEKLY RESTORE-TEST FAILED — reason={reason}, "
                    f"event={str(fe)[:12]}. Scheduled Arweave backups are HALTED "
                    f"until investigated; restore-from-Arweave may be at risk.",
                    alert_key="backup.restore_test_failed",
                    rate_limit_seconds=3600.0)
            except Exception:
                pass
            if bus_emit:
                bus_emit(EVENT_BACKUP_RESTORE_TEST_FAIL, {
                    "event": EVENT_BACKUP_RESTORE_TEST_FAIL,
                    "halt_reason": reason, "halt_event_id": fe,
                    "stage": result.status, "errors": result.errors,
                    "duration_s": dur})
            return False
        except Exception as e:
            logger.critical("[Backup] §24.12 restore-test raised: %s — HALTING "
                            "backups (fail-closed)", e, exc_info=True)
            self._set_backups_halt(reason=f"restore_test_exception: {e}",
                                   failed_event_id=None)
            return False
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _tier_specs_from_paths(self, paths, format_hint=None):
        """Convert PERSONALITY_PATHS-style tuples into TierFileSpec records
        for the unified pipeline.

        Directory entries (e.g. data/sage_memory/, data/mini_reasoning/,
        data/titan_vm_v2/) are RECURSED — each file inside becomes a
        separate spec with arc_name="<dir_arc>/<relative_path>". Mirrors
        local_diff.build_incremental_tarball's recursion pattern so the
        unified plane preserves the same coverage as L5 + the legacy
        cascade.

        Skip patterns: .bak / .bak.prev / .tmp / .restoring / .staging
        (matches audit-coverage ignore_patterns + Phase 9 scratch artifacts).
        """
        from pathlib import Path

        from titan_hcl.logic.backup_upload_pipeline import TierFileSpec

        IGNORE_SUFFIXES = (
            ".bak", ".tmp", ".restoring", ".staging",
            ".corrupt", ".repair",
            # 2026-05-29: backup-snapshot hardlinks created by
            # diff_encoders/full_ship._race_safe_snapshot. New code places
            # these in data/.bksnap_scratch (out of tree), but skip-list as
            # defense in depth so a stray in-tree .bksnap can never be
            # re-snapshotted into the exponential-orphan blowup that produced
            # 340,445 phantom hardlinks / 358 GB phantom scope.
            ".bksnap",
        )

        def _ignore(name: str) -> bool:
            return any(suffix in name for suffix in IGNORE_SUFFIXES)

        def _hint_for(path: str) -> Optional[str]:
            h = format_hint
            if h is None and path.endswith(".bin") and "timechain" in path:
                h = "timechain_bin"
            return h

        specs = []
        for entry in paths:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            src, arc = entry[0], entry[1]
            if not os.path.exists(src):
                # Tolerate missing — caller resolves to a no-op for that
                # arc_name during this event
                continue
            if os.path.isdir(src) or src.endswith("/"):
                # Recurse: each file inside the dir becomes a sub-spec
                root = Path(src)
                for sub in root.rglob("*"):
                    if not sub.is_file():
                        continue
                    if _ignore(sub.name):
                        continue
                    rel = sub.relative_to(root)
                    sub_arc = f"{arc}/{rel.as_posix()}"
                    specs.append(TierFileSpec(
                        source_path=str(sub),
                        arc_name=sub_arc,
                        format_hint=_hint_for(str(sub)),
                    ))
                continue
            specs.append(TierFileSpec(
                source_path=src, arc_name=arc, format_hint=_hint_for(src),
            ))
        return specs

    def _ensure_arweave_store_for_unified(self):
        """Lazy-init ArweaveStore matching the pattern used elsewhere in
        this class. Returns the store or None on init failure."""
        if self._arweave_store is not None:
            return self._arweave_store
        try:
            from titan_hcl.utils.arweave_store import ArweaveStore
            kp = os.path.expanduser("~/.config/solana/id.json")
            net = "mainnet" if self._titan_id == "T1" else "devnet"
            store = ArweaveStore(keypair_path=kp, network=net)
            self._arweave_store = store
            return store
        except Exception as e:
            logger.warning(
                "[Backup] §24 unified_v2: ArweaveStore lazy-init failed: %s", e,
            )
            return None

    def _ensure_chain(self):
        """Lazy-init the ChainProvider (RFP_chain_provider). An injected provider
        (tests → FakeChainProvider) wins. Otherwise build a real
        ArweaveChainProvider: the data plane (Phase A) uses the same keypair/
        network as `_ensure_arweave_store_for_unified`; the trust plane (Phase B)
        is wired with `self.network` (the Solana signer for `commit_memo`) + the
        vault program id. Construction does no I/O (daemon/RPC connect lazily)."""
        if self._chain_provider is not None:
            return self._chain_provider
        from titan_hcl.chain import ArweaveChainProvider
        kp = os.path.expanduser("~/.config/solana/id.json")
        net = "mainnet" if self._titan_id == "T1" else "devnet"
        ncfg = get_params("network") or {}
        rpc = ncfg.get("premium_rpc_url", "") or ""
        vpid = (getattr(self.network, "_vault_program_id", None)
                or ncfg.get("vault_program_id"))
        self._chain_provider = ArweaveChainProvider(
            keypair_path=kp, network=net, rpc_url=rpc,
            network_client=self.network, vault_program_id=vpid)
        return self._chain_provider


    # ── Phase 2 pre-stage (2026-05-31) ─────────────────────────────────────
    #
    # Splits the unified_v2 event into a heavy BUILD (snapshot+diff+pack — run by
    # the backup worker's stager OFF the recv loop, ahead of the first meditation)
    # and a fast SHIP (upload staged tarballs + ZK + manifest — run on meditation).
    # Removes the multi-minute diff-build from the bus-blocking meditation path.
    # Cold-start / no-fresh-stage fallback = the BOUNDED `_inline_build_and_ship_v2`
    # (build_slice byte-budget + streamed pack), shared by meditation + the manual
    # trigger via `_ship_daily_event_v2`. (The legacy whole-file inline path
    # `_run_unified_event_v2` was DELETED 2026-06-11 — it was the only prod caller
    # of `run_unified_event` + the RSS-flap footgun on a Maker-forced trigger.)

    def _todays_backup_already_landed(self) -> bool:
        """SPEC §24 daily/weekly gate — the MANIFEST is the source of truth.

        Returns True iff a valid backup event dated today (UTC) already exists in
        the manifest. Cheap LOCAL read (no Solana/Arweave RPC — those are costly
        timewise, per Maker). A manifest event is only appended on a FULLY
        successful ship (all tiers uploaded + ZK committed), so existence == the
        backup landed. Weekly (Sunday soul) is covered: Sunday's event is one
        atomic event that includes the soul tier, so a today-event implies both.

        On an unreadable manifest, returns False (not-landed) — safer than
        falsely claiming done: a retry that finds an existing event no-ops, but a
        false 'done' would block a needed backup (the exact bug this replaces).
        """
        try:
            from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
            manifest = UnifiedManifest.load(
                titan_id=self._titan_id, base_dir="data")
            self._configure_manifest_cadence(manifest)  # mode-aware §24.2
        except Exception as e:
            logger.warning(
                "[Backup] §24 gate: manifest load failed (%s) — treating as "
                "not-landed (will attempt/retry)", e)
            return False
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for ev in manifest.events:
            ts = ev.get("ts_unix", 0) or 0
            if not ts:
                continue
            ev_day = datetime.fromtimestamp(
                float(ts), timezone.utc).strftime("%Y-%m-%d")
            # "exists AND is correct": dated today + carries the chain anchor
            # (zk_commit_tx is only set on a fully-committed event).
            if ev_day == today and ev.get("zk_commit_tx"):
                return True
        return False

    def _plan_staged_build_v2(self, weekday: int, scratch_dir=None,
                              *, byte_budget=None):
        """RFP_backup_redesign_spine Phase D — the PLAN half of a staged build:
        manifest load, tier specs, the §24 diff-base integrity precheck + the
        new-vs-known resolver, and `BackupWorker.plan_build` (seeds the pending
        specs — NO encode/pack yet). Returns `(worker, staged, baseline_resolver,
        known_arcs)` or None (arweave gate off / manifest unloadable).

        The BackupOrchestrator drives `build_slice` across idle ticks (the drip)
        and `finalize_pack` at the deadline/ship; the one-shot
        `_build_staged_event_v2` drains it inline (cold/inline fallback + tests).
        `scratch_dir=None` → a fresh tempdir (one-shot); the Orchestrator passes
        its STABLE per-Titan drip dir so a disk-persisted drip survives a restart.
        `byte_budget` overrides the per-slice byte budget ([backup.orchestrator])."""
        from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
        from titan_hcl.logic.backup_worker_pipeline import BackupWorker
        try:
            _budget = get_params("mainnet_budget") or {}
        except Exception:
            _budget = {}
        if not _budget.get("backup_arweave_enabled", False):
            logger.debug("[Backup] §24 build-stage: backup_arweave_enabled=false")
            return None
        try:
            manifest = UnifiedManifest.load(
                titan_id=self._titan_id, base_dir="data")
            self._configure_manifest_cadence(manifest)  # mode-aware §24.2
        except ValueError as e:
            logger.error("[Backup] §24 build-stage: manifest load failed: %s", e)
            return None
        p_specs = self._tier_specs_from_paths(self.PERSONALITY_PATHS)
        t_specs = self._tier_specs_from_paths(
            self.TIMECHAIN_PATHS, format_hint="timechain_bin")
        # §24.4.C conformance: soul ships on weekly Sundays AND on every baseline
        # event (a baseline is a full snapshot of all in-scope paths).
        _cad, _depth = self._rebase_params()
        _should_rebase, _ = manifest.should_rebase(cadence=_cad, depth_cap=_depth)
        s_specs = (self._tier_specs_from_paths(self.WEEKLY_EXTRA_PATHS)
                   if (weekday == 6 or _should_rebase) else None)
        base_dir = self._baseline_working_dir()
        # Phase B integrity precheck (sync caller thread → asyncio.run the async
        # precheck; the only async part is a rare chain-reconstruct). Decides
        # force-baseline (missing KNOWN diff-base) + builds the new-vs-known
        # resolver. ⚠ One-shot side effect: `_take_force_baseline` is CONSUMED
        # here — so the Orchestrator runs this ONCE per plan and persists
        # `known_arcs`; on a drip RESUME it rebuilds the resolver from the
        # persisted known_arcs (NOT a re-precheck), preserving the force-baseline.
        try:
            _force_et, _force_trig, _known_arcs = asyncio.run(
                self._precheck_diff_base(manifest))
        except BackupReconstructDeferred as e:
            # Transient: the chained mirror couldn't be reconstructed because the
            # chain provider isn't fetch-ready yet (early boot) — NOT a genuine
            # integrity failure. Skip planning this tick; the orchestrator retries
            # and ships a cheap catch-up incremental once the chain is reachable.
            # NEVER manufacture a self_heal baseline on a transient (Maker rule).
            logger.warning("[Backup] §24 build-stage DEFERRED — no plan this tick "
                           "(chain provider warming): %s", e)
            return None
        _baseline_resolver = self._make_diff_base_resolver(base_dir, _known_arcs)

        if scratch_dir is None:
            import tempfile
            scratch_dir = tempfile.mkdtemp(
                prefix=f"titan_backup_stage_{self._titan_id}_")
        else:
            os.makedirs(scratch_dir, exist_ok=True)
        _kw = {"titan_id": self._titan_id, "chain_provider": None}
        if byte_budget is not None:
            _kw["byte_budget"] = int(byte_budget)
        worker = BackupWorker(**_kw)
        staged = worker.plan_build(
            manifest=manifest, personality_specs=p_specs, timechain_specs=t_specs,
            soul_specs=s_specs, scratch_dir=scratch_dir,
            force_event_type=_force_et, force_trigger=_force_trig)
        return worker, staged, _baseline_resolver, _known_arcs

    def _build_staged_event_v2(self, weekday: int):
        """Pre-BUILD a unified event in ONE shot (no upload, no manifest mutation).
        Returns a StagedBuild or None. KEPT for the cold/inline fallback + the
        prestage tests; Phase D's BackupOrchestrator drives the per-batch idle
        drip via `_plan_staged_build_v2` + `build_slice` across ticks instead."""
        planned = self._plan_staged_build_v2(weekday)
        if planned is None:
            return None
        worker, staged, _baseline_resolver, _known_arcs = planned
        # Drain the whole build now (one-shot). Phase D drips this batch-by-batch.
        while worker.build_slice(staged, _baseline_resolver):
            pass
        worker.finalize_pack(staged)
        logger.info(
            "[Backup] §24 staged event BUILT: id=%s type=%s (off-loop; "
            "awaiting meditation to ship)", staged.event_id[:8],
            staged.event_type)
        return staged

    async def _inline_build_and_ship_v2(self, weekday: int) -> bool:
        """RFP_backup_redesign_spine Phase D synchronous last-resort (failsafe-2,
        INV-BRS-3): when no fresh stage is ready at meditation, build the event
        INLINE via the SAME bounded BackupWorker primitive the drip uses
        (`build_slice` byte-budget + streamed `finalize_pack`), then stream-ship
        it (`_ship_staged_event_v2`). This bounded path replaced the legacy
        whole-file-buffering inline path (`_run_unified_event_v2`, deleted
        2026-06-11) for BOTH the meditation fallback and the manual trigger. The
        build runs in a thread (its precheck `asyncio.run` needs a clean loop +
        the CPU-bound build_slice/pack stays OFF the meditation coroutine), so
        RSS stays bounded and meditation waits only briefly. Returns True if
        shipped."""
        loop = asyncio.get_running_loop()
        staged = await loop.run_in_executor(
            None, self._build_staged_event_v2, weekday)
        if staged is None:
            return False
        return await self._ship_staged_event_v2(staged)

    async def _ship_staged_event_v2(self, staged) -> bool:
        """SHIP a pre-built StagedEvent (fast, on meditation). Reloads the
        manifest fresh for the staleness check + append. Returns True if shipped;
        False on stale-baseline (caller rebuilds) / gate-off / failure."""
        from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
        from titan_hcl.logic.backup_worker_pipeline import BackupWorker
        # §24.12 / INV-BR-4 — halt scheduled backups after a failed restore-test.
        if self._is_backups_halted():
            logger.warning("[Backup] §24.12 backups HALTED (failed restore-test) — "
                           "not shipping the staged event until the halt is cleared")
            return False
        store = self._ensure_arweave_store_for_unified()
        if store is None:
            logger.warning("[Backup] §24 ship-stage: no ArweaveStore available")
            return False
        try:
            manifest = UnifiedManifest.load(
                titan_id=self._titan_id, base_dir="data")
            self._configure_manifest_cadence(manifest)  # mode-aware §24.2
        except ValueError as e:
            logger.error("[Backup] §24 ship-stage: manifest load failed: %s", e)
            return False

        async def _v3_chain_commit(event_id, ts, event_type, event_root,
                                   components, prev_sig):
            return await self.commit_event_v3_chain(
                event_id=event_id, ts=ts, event_type=event_type,
                event_merkle_root=event_root, components=components,
                prev_sig=prev_sig)

        def _bus_emit(name: str, payload: dict) -> None:
            try:
                bus = getattr(self, "bus", None) or getattr(self, "_bus", None)
                if bus is not None and hasattr(bus, "emit"):
                    bus.emit(name, payload)
            except Exception:
                pass

        # cleanup_scratch=False: keep the staged tarballs for the #3 baseline-dir
        # refresh (CRITICAL on the staged path — the pack ran off-loop, possibly
        # ~20 min before this ship, so re-copying live source would drift). We
        # rmtree staged.scratch_dir ourselves in the finally below.
        worker = BackupWorker(titan_id=self._titan_id,
                              chain_provider=self._ensure_chain())
        result = await worker.ship_event(
            staged, manifest=manifest, zk_committer=_v3_chain_commit,
            bus_emit=_bus_emit, encryptor=self._build_v3_encryptor())

        try:
            if result.status == "stale_baseline":
                logger.info(
                    "[Backup] §24 staged event STALE (baseline moved since build) — "
                    "discarding, stager will rebuild: %s", result.skipped_reason)
                return False
            if result.status != "shipped":
                logger.warning(
                    "[Backup] §24 staged ship NOT shipped: status=%s errors=%s",
                    result.status, result.errors)
                return False
            # Phase B: ADVANCE the rolling mirror to this just-shipped event.
            # Chained → every event; cumulative → baseline only.
            if (self._chained_incrementals_enabled()
                    or result.event_type == "baseline"):
                try:
                    paths = [t.tarball_path for t in staged.tier_results.values()
                             if getattr(t, "tarball_path", None)]
                    self._advance_mirror_from_tarballs(paths, result.event_id)
                except Exception as e:
                    logger.warning(
                        "[Backup] §24 ship-stage: mirror advance-from-tarball "
                        "failed: %s — falling back to source copy", e)
                    try:
                        self._refresh_baseline_working_dir()
                    except Exception as e2:
                        logger.warning(
                            "[Backup] §24 ship-stage: source-copy fallback also "
                            "failed: %s (next event may self-heal to a baseline)", e2)
            # devnet/local RETENTION: a fresh weekly baseline supersedes the prior
            # chain → drop manifest events + local cache tarballs older than it
            # (Maker 2026-06-11). No-op on mainnet (indefinite) + on incrementals.
            if result.event_type == "baseline":
                try:
                    self._prune_old_local_backups(manifest)
                except Exception as e:
                    logger.warning(
                        "[Backup] §24 retention prune failed: %s (non-fatal)", e)
            return True
        finally:
            import shutil
            if getattr(staged, "scratch_dir", None) and os.path.isdir(staged.scratch_dir):
                shutil.rmtree(staged.scratch_dir, ignore_errors=True)

    async def _auto_fund_irys_before_upload(self) -> None:
        """Irys auto-fund via the ChainProvider (RFP_chain_provider Phase C tail) —
        `chain.balance()` + the BOUNDED `chain.fund()` replace the legacy
        subprocess-`node` Irys path (INV-CP-3). Config moves to
        `[chain.fund]` (Q-CP-3 clean cut). No-op when disabled / runway sufficient
        / devnet. The runway→amount DECISION here is contained; it moves to the
        BackupOrchestrator in the next redesign step (which will also enforce the
        wallet-reserve floor against the live wallet balance)."""
        fcfg = ((self._full_config or {}).get("chain", {}) or {}).get("fund", {}) or {}
        if not fcfg.get("enabled", False):
            return
        chain = self._ensure_chain()
        try:
            irys_sol = await chain.balance()
        except Exception as e:
            logger.warning("[Backup] §24 Irys auto-fund: balance query failed: %s", e)
            return
        if irys_sol == float("inf"):
            return  # devnet — no real deposit
        # Runway estimator needs a representative upload size — use the staged
        # event's total tarball size when available, else a sane daily default.
        size_mb = 35.0
        try:
            entry = self._staged_event
            staged = entry.get("staged") if isinstance(entry, dict) else None
            if staged is not None:
                total = sum(
                    r.tarball_size_bytes for r in staged.tier_results.values()
                    if r is not None and r.tarball_size_bytes)
                if total > 0:
                    size_mb = total / 1048576.0
        except Exception:
            pass
        avg_cost = max(size_mb, 1.0) * 0.0002 * 2.0  # 2× safety margin (match S4)
        daily_burn = avg_cost * float(fcfg.get("avg_uploads_per_day", 3.0))
        runway = irys_sol / daily_burn if daily_burn > 0 else 1e9
        if runway >= float(fcfg.get("min_runway_days", 3.0)):
            return  # runway sufficient
        target_sol = float(fcfg.get("target_runway_days", 14.0)) * daily_burn
        amount = max(0.0, target_sol - irys_sol)
        if amount <= 0:
            return
        try:
            tx = await chain.fund(
                amount, daily_cap_sol=float(fcfg.get("daily_cap_sol", 0.05)))
        except Exception as e:
            logger.warning("[Backup] §24 Irys auto-fund: fund failed: %s", e)
            return
        if tx:
            logger.info("[Backup] §24 Irys auto-fund: FUNDED ~%.4f SOL (runway was "
                        "%.2fd) tx=%s", amount, runway, str(tx)[:16])

    # ── Local diff/baseline event (L5, 2026-05-14) ─────────────────────────
    #
    # Replaces "ship a full ~30MB personality tarball every meditation" with
    # weekly full baseline + daily incremental patches (xdelta3). Local-only
    # — does NOT touch the Arweave path. Opt-in via `[backup].local_diff_enabled`
    # config (default OFF). When enabled + mode=local_only, on_meditation_complete
    # calls this instead of the full-tarball cascade for personality.
    #
    # See titan_hcl/logic/local_diff.py for the diff/patch primitives + manifest
    # schema. Reuses PERSONALITY_PATHS + _BACKUP_SKIP_PATTERNS from this class.
    def create_local_diff_event(
        self,
        local_dir: str = "data/backups",
    ) -> Optional[dict]:
        """Build either a baseline or incremental tarball + update manifest.

        Decision (per local_diff.should_create_baseline):
          - Cold start (no prior baseline) → baseline
          - Today is Sunday UTC → baseline (replaces prior baseline)
          - Latest baseline older than 7 days → baseline (failsafe)
          - Otherwise → incremental against current baseline

        Returns dict with: type, event_id, tarball_path, size_mb, summary stats.
        Returns None on failure (existing cascade is unaffected — caller falls
        back to regular create_personality_archive path).
        """
        from . import local_diff as ld
        from dataclasses import asdict

        try:
            import hashlib
            os.makedirs(local_dir, exist_ok=True)
            manifest = ld.load_manifest(local_dir, self._titan_id)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            ts_hash = hashlib.sha256(
                f"{self._titan_id}{time.time()}".encode()).hexdigest()[:8]

            # L5 file selection: apply ARWEAVE_DAILY_EXCLUDE filter to keep
            # the baseline tarball at the same size as today's daily local
            # backups (~30 MB). The 5 big DBs (inner_memory.db, neural_ns/,
            # experience_*.db, episodic_memory.db) continue to ride the
            # weekly soul package, unchanged. Without this filter, baseline
            # would balloon to ~400 MB (full PERSONALITY_PATHS).
            filtered_specs = [
                (src, arc) for src, arc in self.PERSONALITY_PATHS
                if arc not in self.ARWEAVE_DAILY_EXCLUDE
            ]

            create_baseline = ld.should_create_baseline(manifest, today)

            if create_baseline:
                output_name = ld.BASELINE_TARBALL_FMT.format(date=today.replace("-", ""), hash8=ts_hash)
                output_path = os.path.join(local_dir, output_name)
                event = ld.build_baseline_tarball(
                    file_specs=filtered_specs,
                    output_path=output_path,
                    skip_patterns=self._BACKUP_SKIP_PATTERNS,
                )
                if event is None:
                    return None
                # Extract baseline contents to baseline_active/ so future
                # incrementals can diff against per-arc files.
                if not ld.refresh_baseline_active(output_path, local_dir):
                    logger.warning(
                        "[Backup] L5 baseline tarball saved but baseline_active "
                        "refresh failed — next incrementals will fail until "
                        "next baseline lands.")
                # Update manifest: append event + set current_baseline_event_id
                manifest["events"].append(asdict(event))
                manifest["current_baseline_event_id"] = event.event_id
                ld.save_manifest(manifest, local_dir, self._titan_id)
                # L5 retention (after baseline land): prune old baselines +
                # incrementals from prior baseline chains. Safe to do here
                # because the new baseline + future incrementals can stand
                # alone for restore.
                _l5_cleanup_old_local_tarballs(
                    local_dir, retention_days=int(
                        get_params("backup").get("local_rolling_days", 30)))
                logger.info(
                    "[Backup] L5 BASELINE: %s (%.1f MB, %d files)",
                    output_name, event.tarball_size_bytes / 1024 / 1024,
                    len(event.files))
                return {
                    "type": "baseline",
                    "event_id": event.event_id,
                    "tarball_path": output_path,
                    "size_mb": round(event.tarball_size_bytes / 1024 / 1024, 2),
                    "files_count": len(event.files),
                }

            # Incremental path — diff against current baseline_active/
            baseline_event_id = manifest["current_baseline_event_id"]
            baseline_event = next(
                (ev for ev in manifest["events"]
                 if ev["event_id"] == baseline_event_id),
                None)
            if baseline_event is None:
                logger.warning("[Backup] L5 incremental: baseline event not found in manifest")
                return None
            baseline_active = str(ld.baseline_active_dir(local_dir))
            if not os.path.isdir(baseline_active):
                logger.warning(
                    "[Backup] L5 incremental: baseline_active dir missing (%s) — "
                    "forcing baseline rebuild next cycle", baseline_active)
                return None

            output_name = ld.INCREMENTAL_TARBALL_FMT.format(date=today.replace("-", ""), hash8=ts_hash)
            output_path = os.path.join(local_dir, output_name)
            event = ld.build_incremental_tarball(
                file_specs=filtered_specs,
                baseline_event=baseline_event,
                baseline_files_dir=baseline_active,
                output_path=output_path,
                skip_patterns=self._BACKUP_SKIP_PATTERNS,
            )
            if event is None:
                return None
            manifest["events"].append(asdict(event))
            ld.save_manifest(manifest, local_dir, self._titan_id)
            # L5 retention: prune personality_baseline_*.tar.gz and
            # personality_incremental_*.tar.gz older than local_rolling_days.
            # The legacy cascade's cleanup_local doesn't run when L5 is enabled
            # (we bypass upload_personality_to_arweave entirely), so do it here.
            _l5_cleanup_old_local_tarballs(
                local_dir, retention_days=int(
                    get_params("backup").get("local_rolling_days", 30)))
            logger.info(
                "[Backup] L5 INCREMENTAL: %s (%.2f MB, patched=%d skipped=%d removed=%d)",
                output_name, event.tarball_size_bytes / 1024 / 1024,
                event.patched_count, event.skipped_count, event.removed_count)
            return {
                "type": "incremental",
                "event_id": event.event_id,
                "tarball_path": output_path,
                "size_mb": round(event.tarball_size_bytes / 1024 / 1024, 2),
                "patched_count": event.patched_count,
                "skipped_count": event.skipped_count,
                "removed_count": event.removed_count,
            }
        except Exception as e:
            logger.error("[Backup] create_local_diff_event failed: %s", e, exc_info=True)
            return None

    # ── Monthly-archive consolidation + retention cap (L3+L4, 2026-05-14) ──
    #
    # Pre-2026-05-14: T1-only cron `0 5 * * 0` ran `scripts/personality_backup_archive.sh`
    # which bundled personality_*.tar.gz files older than 7 days into a monthly
    # archive kept INDEFINITELY in data/backups/archives/. T2/T3 had no equivalent.
    # Result: T1 disk grew unbounded (13.5GB April 2026 archive observed); T2/T3
    # accumulated loose tarballs only bounded by 30-day cleanup_local rolling window.
    #
    # Post-fix: BackupWorker calls this method after every successful backup event
    # on ALL Titans. Logic:
    #   1. Find personality_*.tar.gz files older than `monthly_archive_loose_days`
    #      (default 7) that have not yet been archived
    #   2. Group by year-month from filename (personality_YYYYMMDD_HASH.tar.gz)
    #   3. For each year-month without an existing archive: create
    #      data/backups/archives/personality_archive_YYYYMM.tar.gz containing those
    #      loose files (gzip-9 — same compression as original; no nesting gain
    #      but reduces inode pressure)
    #   4. After tarball-verify, delete the loose files
    #   5. Apply retention cap: delete monthly archives older than
    #      `monthly_archive_retention_months` (default 12)
    #
    # Idempotent: re-running on the same day is a no-op (archives detected by
    # filename, skipped). Atomic-write semantics: tarball built to .tmp then
    # renamed; on partial failure no loose files are deleted.
    def consolidate_monthly_archives(
        self,
        local_dir: str = "data/backups",
        loose_retention_days: int = 7,
        archive_retention_months: int = 12,
    ) -> dict:
        """Consolidate loose daily personality tarballs into monthly archives
        + prune monthly archives older than retention cap.

        Returns dict with: archived_count, deleted_loose_count, pruned_archives_count.
        Never raises; logs warnings on individual file failures.
        """
        import glob
        import re
        result = {
            "archived_count": 0,
            "deleted_loose_count": 0,
            "pruned_archives_count": 0,
            "errors": [],
        }
        try:
            local_path = Path(local_dir)
            if not local_path.exists():
                return result
            archives_path = local_path / "archives"
            archives_path.mkdir(exist_ok=True)

            # Step 1+2: enumerate eligible loose files, group by year-month
            now = time.time()
            cutoff = now - (loose_retention_days * 86400)
            # Filename pattern: personality_YYYYMMDD_HASH.tar.gz
            name_pat = re.compile(r'^personality_(\d{4})(\d{2})\d{2}_[0-9a-f]+\.tar\.gz$')
            groups: dict[str, list[Path]] = {}
            for tarball in local_path.glob("personality_*.tar.gz"):
                if not tarball.is_file():
                    continue
                try:
                    mtime = tarball.stat().st_mtime
                except OSError:
                    continue
                if mtime > cutoff:
                    continue  # too recent, keep loose
                m = name_pat.match(tarball.name)
                if not m:
                    continue  # unrecognized name pattern
                year_month = f"{m.group(1)}{m.group(2)}"
                groups.setdefault(year_month, []).append(tarball)

            # Skip the current calendar year-month (file is older than 7d but may
            # still receive loose siblings this month).
            current_ym = datetime.now(timezone.utc).strftime("%Y%m")
            groups.pop(current_ym, None)

            # Step 3+4: for each year-month, create archive tarball if not present,
            # then delete loose files after verifying tarball integrity.
            for ym, files in sorted(groups.items()):
                archive_target = archives_path / f"personality_archive_{ym}.tar.gz"
                if archive_target.exists():
                    # Idempotent skip — archive already exists.
                    logger.debug("[Backup] Monthly archive exists: %s", archive_target.name)
                    continue
                tmp_target = archives_path / f"personality_archive_{ym}.tar.gz.tmp"
                try:
                    with tarfile.open(tmp_target, "w:gz", compresslevel=9) as tar:
                        for f in files:
                            tar.add(str(f), arcname=f.name)
                    # Verify by re-opening + counting members
                    with tarfile.open(tmp_target, "r:gz") as tar:
                        member_count = len(tar.getnames())
                    if member_count != len(files):
                        raise RuntimeError(
                            f"member count mismatch: tarball={member_count} input={len(files)}")
                    # Atomic rename
                    tmp_target.rename(archive_target)
                    # Delete loose files
                    for f in files:
                        try:
                            f.unlink()
                            result["deleted_loose_count"] += 1
                        except OSError as e:
                            result["errors"].append(
                                f"delete {f.name}: {e}")
                    result["archived_count"] += 1
                    logger.info(
                        "[Backup] Monthly archive: %s (%d files, %.1f MB)",
                        archive_target.name, len(files),
                        archive_target.stat().st_size / 1024 / 1024)
                except Exception as e:
                    with suppress(FileNotFoundError):
                        tmp_target.unlink()
                    result["errors"].append(f"archive {ym}: {e}")
                    logger.warning(
                        "[Backup] Monthly archive failed for %s: %s", ym, e)

            # Step 5: retention cap — delete monthly archives older than
            # `archive_retention_months` months from current year-month.
            try:
                current_year = int(current_ym[:4])
                current_month = int(current_ym[4:])
                # Convert "N months ago" → cutoff year-month (integer compare)
                cutoff_total_months = (current_year * 12 + current_month) - archive_retention_months
                arch_name_pat = re.compile(
                    r'^personality_archive_(\d{4})(\d{2})\.tar\.gz$')
                for arch in archives_path.glob("personality_archive_*.tar.gz"):
                    m = arch_name_pat.match(arch.name)
                    if not m:
                        continue
                    arch_total_months = int(m.group(1)) * 12 + int(m.group(2))
                    if arch_total_months < cutoff_total_months:
                        try:
                            size_mb = arch.stat().st_size / 1024 / 1024
                            arch.unlink()
                            result["pruned_archives_count"] += 1
                            logger.info(
                                "[Backup] Pruned monthly archive: %s "
                                "(%.1f MB, older than %d months)",
                                arch.name, size_mb, archive_retention_months)
                        except OSError as e:
                            result["errors"].append(
                                f"prune {arch.name}: {e}")
            except Exception as e:
                result["errors"].append(f"retention cap: {e}")
                logger.warning(
                    "[Backup] Monthly-archive retention cap failed: %s", e)

        except Exception as e:
            result["errors"].append(f"consolidate: {e}")
            logger.warning("[Backup] consolidate_monthly_archives failed: %s", e)

        return result
