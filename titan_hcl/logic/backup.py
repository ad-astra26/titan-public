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

logger = logging.getLogger(__name__)


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


class RebirthBackup:
    """
    Manages sovereign backup triggered by meditation cycles:
    - Daily personality archive → Arweave (1st meditation of day)
    - Weekly full soul package → Arweave (1st meditation on Sunday)
    - ZK compressed epoch snapshot → Solana (every meditation)
    - MyDay NFT → Solana (every 4th meditation)
    """

    def __init__(self, network_client, config: dict = None, titan_id: str = "T1",
                 arweave_store=None, full_config: dict = None):
        """
        Args:
            network_client: Solana RPC client (for ZK snapshot + NFT mint)
            config: memory_and_storage section from config.toml
            titan_id: T1/T2/T3 — used for per-Titan manifest path
            arweave_store: injected ArweaveStore (rFP Phase 1 BUG-5 fix —
                constructed ONCE at boot rather than rebuilt per-backup)
            full_config: optional full config dict (for mainnet_budget flag)
        """
        config = config or {}
        self.network = network_client
        self.current_snapshot_hash = None
        self._titan_id = titan_id
        self._arweave_store = arweave_store  # BUG-5: injected once at boot
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
                logger.info("[Backup] Loaded state: personality=%s, soul=%s, meditations=%d",
                            self._last_personality_date, self._last_soul_date, self._meditation_count)
        except Exception as e:
            swallow_warn('[Backup] No backup state loaded', e,
                         key="logic.backup.no_backup_state_loaded", throttle=100)

    def _save_backup_state(self):
        """Persist backup tracking state to disk."""
        import json
        os.makedirs(os.path.dirname(self._BACKUP_STATE_PATH) or ".", exist_ok=True)
        try:
            with open(self._BACKUP_STATE_PATH, "w") as f:
                json.dump({
                    "last_personality_date": self._last_personality_date,
                    "last_soul_date": self._last_soul_date,
                    "last_timechain_date": self._last_timechain_date,
                    "meditation_count": self._meditation_count,
                    "meditation_count_since_nft": self._meditation_count_since_nft,
                    "updated_at": time.time(),
                }, f, indent=2)
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
            # CRITICAL (2026-05-30): the §24 unified_v2 event IS the daily
            # backup — it must fire ONCE per calendar day, on the 1st
            # meditation (the documented contract: "Personality → Arweave
            # (1st meditation of day)", on_meditation_complete docstring).
            #
            # Pre-fix, this branch called _run_unified_event_v2 on EVERY
            # meditation and returned early — the per-tier daily CAS gates
            # below (line ~305 `today != _last_personality_date`) were
            # DEAD CODE for the unified_v2 path. run_unified_event ships
            # whenever the diff is non-empty, so a 2nd meditation-with-
            # changes the same day shipped a 2nd Arweave backup (observed
            # 2× / day on T1: 05-29 16:22+17:02, 05-30 02:25+06:25). That
            # also caused proof_day to find a fresh anchor more than once a
            # day. Restore the daily gate by claiming the calendar day here
            # (same CAS pattern + failure-reset as the legacy personality
            # gate) so the unified event ships exactly once per day.
            # SPEC §24 daily/weekly gate — MANIFEST IS THE SINGLE SOURCE OF TRUTH
            # (2026-05-31 redesign, Maker request). The prior pattern CLAIMED the
            # calendar day (persisting _last_personality_date) BEFORE the ship; a
            # ship failure OR a mid-ship process death (boot-storm / restart) then
            # left the day stuck-claimed → every later meditation skipped → no
            # backup landed for days (the proof_day-silence + 05-31 stuck-claim
            # root cause). Fix: there is NO separate claim flag. The gate asks the
            # manifest "did today's event already land?" — a cheap LOCAL read (no
            # RPC). A SUCCESSFUL ship appends today's event (next meditation then
            # correctly skips); a FAILED ship appends nothing (next meditation
            # correctly retries). Nothing can get stuck. The CAS lock is kept
            # purely to serialize two near-simultaneous meditations → still ships
            # exactly once. Weekly (Sunday soul) is covered automatically: Sunday's
            # event is ONE atomic event that includes the soul tier.
            async with self._get_personality_cas_lock():
                if self._todays_backup_already_landed():
                    logger.info(
                        "[Backup] §24 daily backup already LANDED for %s "
                        "(manifest event exists) — skipping (meditation #%d)",
                        today, self._meditation_count)
                    return

                # Irys auto-fund (re-homed 2026-05-31) — top up before the upload,
                # same caps (auto_fund_enabled / daily cap / runway / reserve).
                # Best-effort; a fund hiccup never blocks the backup.
                try:
                    self._auto_fund_irys_before_upload()
                except Exception as _af_err:
                    logger.warning(
                        "[Backup] §24 Irys auto-fund check raised: %s", _af_err)

                # Ship. Phase 2: prefer a fresh pre-staged event (built off-loop
                # by the stager); fall back to an inline build on cold-start /
                # stale-baseline. On ANY failure we simply return — the manifest
                # has no today-event, so the next meditation retries (nothing
                # stuck, no claim to release).
                try:
                    _staged = self._take_fresh_staged_event(today)
                    if _staged is not None:
                        shipped = await self._ship_staged_event_v2(_staged)
                        if not shipped:
                            logger.info(
                                "[Backup] §24 staged ship declined (stale/failed)"
                                " — falling back to inline build")
                            shipped = await self._run_unified_event_v2(
                                weekday=weekday)
                    else:
                        shipped = await self._run_unified_event_v2(
                            weekday=weekday)
                except Exception as e:
                    logger.exception(
                        "[Backup] §24 unified_v2 ship raised — no backup this "
                        "meditation; next meditation retries (manifest has no "
                        "today-event → nothing stuck): %s", e)
                    self._alert_backup_failure("unified_v2", f"raised: {e}")
                    return

                if shipped:
                    logger.info(
                        "[Backup] §24 unified_v2 event SHIPPED — daily backup "
                        "landed for %s (meditation #%d); manifest updated",
                        today, self._meditation_count)
                else:
                    logger.info(
                        "[Backup] §24 ship did not land for %s — no manifest "
                        "event written; next meditation will retry (nothing "
                        "stuck)", today)
                return

        # 1. ZK Epoch Snapshot (every meditation) — DISABLED for mainnet MVM
        #    TimeChain merkle root is now committed via vault in meditation.py.
        #    ZK compressed snapshots are redundant (raw data → Arweave instead).
        #    Re-enable via [mainnet_budget] zk_compression_enabled = true
        sovereignty_idx = 0
        try:
            sovereignty_idx = await self._compute_sovereignty()
            _budget = self.network._config.get("mainnet_budget", {}) if hasattr(self.network, '_config') else {}
            if _budget.get("zk_compression_enabled", False):
                archive_hash = f"meditation_{epoch}_{int(time.time())}"
                await self._zk_epoch_snapshot(
                    archive_hash, None, total_nodes, sovereignty_idx,
                )
            else:
                logger.debug("[Backup] ZK epoch snapshot skipped (zk_compression_enabled=false)")
        except Exception as e:
            logger.warning("[Backup] ZK epoch snapshot failed: %s", e)

        # 2. Personality backup (1st meditation of day)
        # SPEC §24 / rFP_backup_diff_baseline_unified_v1 Phase 1 (2026-05-15):
        # CAS-gated date check closes the dedup race documented in
        # AUDIT_irys_arweave_costs_20260514 §4 BUG-2 (15 personality uploads
        # on 2026-05-12 from concurrent MEDITATION_COMPLETE events all
        # passing the bare `if today != _last_personality_date` check before
        # any of them could set the date). The lock is held only across
        # the CAS itself — the upload runs outside the lock so cascading
        # work doesn't serialize.
        do_personality = False
        async with self._get_personality_cas_lock():
            if today != self._last_personality_date:
                self._last_personality_date = today
                self._save_backup_state()
                do_personality = True
        if do_personality:
            logger.info("[Backup] Daily personality backup triggered (1st meditation of %s)", today)
            # L5 (2026-05-14): if [backup].local_diff_enabled=true, route
            # through the local diff/baseline engine instead of the full
            # tarball cascade. Saves ~4× local disk by shipping daily
            # incremental patches against a weekly baseline (xdelta3).
            # Local-only — doesn't touch Arweave path. Default false; flip
            # on after observing manifest behavior.
            _backup_cfg = (self._full_config.get("backup", {}) or {})
            _local_diff_enabled = bool(_backup_cfg.get("local_diff_enabled", False))
            _local_dir = _backup_cfg.get("local_dir", "data/backups")
            if _local_diff_enabled:
                # 2026-05-23 D-SPEC-123 follow-up — Maker policy: NO legacy
                # fallback on L5 failure either. Legacy cascade is bug-laden
                # + costs money. When L5 is the chosen path, an L5 failure
                # logs ERROR + Maker-notify; the next meditation retries.
                # Silent fallback to the legacy upload pipeline was strictly
                # worse than failing visibly.
                try:
                    diff_result = await asyncio.to_thread(
                        self.create_local_diff_event, _local_dir)
                    if diff_result:
                        logger.info(
                            "[Backup] L5 personality (%s): %s (%.2fMB)",
                            diff_result["type"], diff_result["event_id"][:8],
                            diff_result["size_mb"])
                    else:
                        logger.error(
                            "[Backup] L5 personality returned None — NO "
                            "legacy fallback (Maker policy); next meditation "
                            "will retry")
                        self._alert_backup_failure(
                            "L5_personality", "create_local_diff_event "
                            "returned None")
                except Exception as e:
                    logger.exception(
                        "[Backup] L5 personality crashed — NO legacy "
                        "fallback (Maker policy); next meditation will "
                        "retry: %s", e)
                    self._alert_backup_failure("L5_personality", str(e))
            else:
                try:
                    result = await self.upload_personality_to_arweave()
                    if result:
                        _tx = result.get("arweave_tx", "?")
                        _sz = result.get("size_mb", 0)
                        _hash = result.get("archive_hash", "")
                        logger.info("[Backup] Personality → Arweave: tx=%s (%.1fMB)", _tx[:20], _sz)
                        # Update titan.md frontmatter
                        self._update_titan_frontmatter(
                            sovereignty_milestone=sovereignty_idx,
                            epochs_completed=epoch,
                        )
                        # Anchor backup hash on-chain (daily integrity proof)
                        await self.anchor_backup_hash(_hash, _sz, "personality")
                        # Update vault shadow_url_hash with backup archive hash
                        await self._update_vault_shadow_hash(_hash)
                        # Alert Maker via Telegram
                        self._alert_backup_success("personality", _sz, _hash, _tx)
                    else:
                        self._alert_backup_failure("personality", "Upload returned None")
                except Exception as e:
                    logger.error("[Backup] Daily personality backup failed: %s", e)
                    self._alert_backup_failure("personality", str(e))
                    # Reset date so it retries next meditation
                    self._last_personality_date = ""
                    self._save_backup_state()

        # 3. Soul package (1st meditation on Sunday)
        # SPEC §24 Phase 1 (2026-05-15) — CAS lock per personality rationale.
        do_soul = False
        async with self._get_soul_cas_lock():
            if weekday == 6 and today != self._last_soul_date:
                self._last_soul_date = today
                self._save_backup_state()
                do_soul = True
        if do_soul:
            logger.info("[Backup] Weekly soul package triggered (Sunday %s)", today)
            try:
                result = await self.upload_soul_package_to_arweave()
                if result:
                    _tx = result.get("arweave_tx", "?")
                    _sz = result.get("size_mb", 0)
                    _hash = result.get("archive_hash", "")
                    logger.info("[Backup] Soul package → Arweave: tx=%s (%.1fMB)", _tx[:20], _sz)
                    self._alert_backup_success("soul_package", _sz, _hash, _tx)
                else:
                    self._alert_backup_failure("soul_package", "Upload returned None")
            except Exception as e:
                logger.error("[Backup] Weekly soul package failed: %s", e)
                self._alert_backup_failure("soul_package", str(e))
                self._last_soul_date = ""
                self._save_backup_state()

        # 5. TimeChain backup (daily, alongside personality)
        # SPEC §24 Phase 1 (2026-05-15) — CAS lock per personality rationale.
        # 2026-05-06 fix preserved: persist timechain date BEFORE upload so
        # restarts don't re-trigger (was the BUG-TIMECHAIN-BACKUP-RESTART-LEAK
        # source — pre-fix that wrote 160 backups in 16d on T3).
        do_timechain = False
        async with self._get_timechain_cas_lock():
            if today != self._last_timechain_date:
                self._last_timechain_date = today
                self._save_backup_state()
                do_timechain = True
        if do_timechain:
            try:
                from titan_hcl.logic.timechain_backup import TimeChainBackup
                # rFP Phase 1 BUG-5: use injected ArweaveStore (constructed once at
                # boot) instead of re-reading config + reconstructing per-backup.
                # Falls back to config-read only if injection didn't happen (legacy boot path).
                _tc_arweave = self._arweave_store
                if _tc_arweave is None:
                    try:
                        _tc_budget = self._full_config.get("mainnet_budget", {})
                        if _tc_budget.get("backup_arweave_enabled", False):
                            _tc_net_cfg = self._full_config.get("network", {})
                            _tc_net = _tc_net_cfg.get("solana_network", "devnet")
                            if _tc_net == "mainnet-beta":
                                _tc_net = "mainnet"
                            _tc_kp = _tc_net_cfg.get("wallet_keypair_path", "")
                            if _tc_kp:
                                from titan_hcl.utils.arweave_store import ArweaveStore
                                _tc_arweave = ArweaveStore(keypair_path=_tc_kp, network=_tc_net)
                    except Exception as _ae:
                        swallow_warn('[Backup] TimeChain ArweaveStore fallback init', _ae,
                                     key="logic.backup.timechain_arweavestore_fallback_init", throttle=100)

                tc_backup = TimeChainBackup(
                    data_dir="data/timechain",
                    titan_id=self._titan_id,
                    arweave_store=_tc_arweave,
                )
                # rFP_backup_worker Phase 0: tarball path (proven working via cron).
                # rFP Phase 2 extension 2026-04-20: full 10-step cascade applied via
                # full_config passthrough → S2 validate + S3 local + S4 balance +
                # S6 verify + S10 cleanup now cover TimeChain uploads too.
                tc_retention = int(self._full_config.get("backup", {}).get(
                    "local_rolling_days", 30))
                tc_tx = await tc_backup.snapshot_to_arweave(
                    full_config=self._full_config,
                    retention_days=tc_retention,
                )
                if tc_tx:
                    logger.info("[Backup] TimeChain → Arweave: tx=%s", tc_tx[:20])
                elif _tc_arweave is None:
                    logger.info("[Backup] TimeChain Arweave upload skipped (no ArweaveStore)")
                else:
                    logger.warning("[Backup] TimeChain Arweave upload returned None")
            except Exception as e:
                logger.warning("[Backup] TimeChain backup failed: %s", e)

        # 4. MyDay NFT (every Nth meditation — L12 housekeeping closure
        # 2026-05-26). Two improvements over the previous code:
        #   (a) honor the `daily_nft_enabled` config flag at
        #       config.toml [mainnet_budget] line 465 — it was being
        #       ignored, so the mint was attempted on every fleet
        #       member regardless of the gate (Maker comment said
        #       "enable after discussion"; gate now actually gates).
        #   (b) read the threshold from `titan_params.toml`
        #       `meditations_per_daily_nft` (currently 4) instead of
        #       hardcoded 4 — same value today, but param-driven so
        #       tuning doesn't require a code change.
        mainnet_budget_cfg = (self._full_config.get("mainnet_budget")
                               if isinstance(self._full_config, dict)
                               else {}) or {}
        daily_nft_enabled = bool(
            mainnet_budget_cfg.get("daily_nft_enabled", False))
        meditations_per_nft = int(
            self._full_config.get("meditations_per_daily_nft", 4)
            if isinstance(self._full_config, dict) else 4)
        if (daily_nft_enabled
                and self._meditation_count_since_nft >= meditations_per_nft):
            self._meditation_count_since_nft = 0
            self._save_backup_state()
            try:
                from titan_hcl.logic.reflection import ReflectionLogic
                reflection = ReflectionLogic(None)
                diary_entry = await reflection.generate_myday_diary_entry(
                    nodes_count=total_nodes,
                    learning_score=min(100.0, total_nodes * 2.5),
                    unique_souls=0,
                    social_score=0.0,
                    mood_score=0.5,
                    sovereignty_index=sovereignty_idx,
                )
                nft_addr = await self.mint_epoch_nft(
                    epoch=int(time.time()),
                    sovereignty_idx=sovereignty_idx,
                    diary_entry=diary_entry,
                    total_nodes=total_nodes,
                )
                if nft_addr:
                    logger.info("[Backup] MyDay NFT minted: %s", nft_addr)
            except Exception as e:
                swallow_warn('[Backup] MyDay NFT skipped', e,
                             key="logic.backup.myday_nft_skipped", throttle=100)
        elif not daily_nft_enabled and self._meditation_count_since_nft >= meditations_per_nft:
            # Gate is OFF — log once per overflow to make the disable
            # visible without spamming. Counter does NOT reset so when
            # the gate flips on, the next meditation mints immediately.
            logger.debug(
                "[Backup] MyDay NFT skipped (daily_nft_enabled=false, "
                "count=%d threshold=%d)",
                self._meditation_count_since_nft, meditations_per_nft)

        # Save state after all operations
        self._save_backup_state()

    async def _compute_sovereignty(self) -> float:
        """Compute sovereignty index from available data."""
        try:
            from titan_hcl.logic.reflection import ReflectionLogic
            reflection = ReflectionLogic(None)
            return await reflection.get_sovereignty_stats(None)
        except Exception:
            return 50.0  # Default if computation unavailable

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

        # rFP Phase 4 — age-based catch-up
        summary = {"critical_ok": True, "catchup_fired": False,
                    "last_age_h": None}
        try:
            from titan_hcl.logic.backup_cascade import BackupCascade
            is_local_only = BackupCascade(
                full_config=self._full_config,
                arweave_store=self._arweave_store,
            ).is_local_only()
            latest = self.get_latest_backup_record("personality")
            if latest and "uploaded_at" in latest:
                last_age_s = time.time() - float(latest["uploaded_at"])
                last_age_h = last_age_s / 3600.0
                summary["last_age_h"] = round(last_age_h, 1)
                if last_age_h > 24 and not is_local_only:
                    if self._unified_v2_enabled():
                        # Maker policy 2026-05-23 (D-SPEC-123 follow-up): NO legacy
                        # full-tarball upload path when unified_v2 owns backups.
                        # This boot catch-up (upload_personality_to_arweave, ~50MB)
                        # was an OVERLOOKED legacy callsite — commit abbf6f84 retired
                        # the on_meditation_complete fallback but missed this one. It
                        # (a) re-uploaded a full tarball on EVERY worker restart — the
                        # SOL drain Maker flagged ("I'm not paying for full daily
                        # reuploads") — and (b) ran SYNCHRONOUSLY here, before the
                        # worker's `booted` transition, so on a ~176k-file personality
                        # tree the ~200s build kept the worker in `starting` past
                        # Guardian's stale-heartbeat threshold → kill → restart →
                        # catch-up fires again → boot-loop, which is WHY unified_v2
                        # never got to ship its first baseline. get_latest_backup_record
                        # reads legacy data/backup_records/ which unified_v2 never
                        # writes, so last_age_h is ALWAYS stale under unified_v2 — the
                        # >24h guard can never gate this off on its own. Unified events
                        # fire on MEDITATION_COMPLETE; no boot catch-up is needed.
                        # See RFP_phase_c_enhancements §3B.0 bugs #1/#2/#3.
                        logger.info(
                            "[Backup] Boot catch-up SKIPPED — unified_v2 owns "
                            "backups; legacy full-upload path retired here per Maker "
                            "2026-05-23 policy (last legacy record %.1fh ago is "
                            "expected: unified_v2 writes the manifest, not "
                            "backup_records/)", last_age_h)
                        summary["catchup_skipped_unified_v2"] = True
                    else:
                        logger.info(
                            "[Backup] Boot catch-up: last personality upload %.1fh ago — firing now",
                            last_age_h)
                        summary["catchup_fired"] = True
                        try:
                            result = await self.upload_personality_to_arweave()
                            if result:
                                tx = result.get("arweave_tx", "local_only")
                                logger.info("[Backup] Boot catch-up complete: %s", tx)
                                summary["catchup_result"] = tx
                        except Exception as e:
                            logger.warning("[Backup] Boot catch-up failed: %s", e)
                            summary["catchup_error"] = str(e)
            elif not latest:
                logger.info("[Backup] Boot: no prior personality record (first run?)")
        except Exception as e:
            swallow_warn('[Backup] Boot catch-up check error', e,
                         key="logic.backup.boot_catch_up_check_error", throttle=100)
        return summary

    # -------------------------------------------------------------------------
    # Hash — delegated to utils/crypto.py (Single Source of Truth)
    # -------------------------------------------------------------------------
    def calculate_hash(self, filepath: str) -> str:
        """Calculate SHA-256 hash of a file via the centralized crypto utility."""
        return hash_file(filepath)

    # -------------------------------------------------------------------------
    # Soul Frontmatter
    # -------------------------------------------------------------------------
    def _update_titan_frontmatter(self, sovereignty_milestone: float, epochs_completed: int) -> None:
        """Update the YAML frontmatter in titan.md with current epoch stats."""
        import re

        titan_path = os.path.join(os.path.dirname(__file__), "..", "..", "titan.md")
        try:
            with open(titan_path, "r") as f:
                content = f.read()

            fm_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if not fm_match:
                logger.debug("[Backup] No YAML frontmatter in titan.md, skipping update.")
                return

            frontmatter = fm_match.group(1)
            rest = content[fm_match.end():]

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            updates = {
                "sovereignty_milestone": f"{sovereignty_milestone:.1f}",
                "epochs_completed": str(epochs_completed),
                "last_rebirth": now_iso,
            }

            for key, value in updates.items():
                pattern = rf'^{key}:.*$'
                replacement = f'{key}: {value}'
                if re.search(pattern, frontmatter, re.MULTILINE):
                    frontmatter = re.sub(pattern, replacement, frontmatter, flags=re.MULTILINE)
                else:
                    frontmatter += f"\n{key}: {value}"

            with open(titan_path, "w") as f:
                f.write(f"---\n{frontmatter}\n---\n{rest}")

            logger.info("[Backup] Updated titan.md frontmatter: sovereignty=%.1f, epochs=%d",
                       sovereignty_milestone, epochs_completed)
        except Exception as e:
            logger.warning("[Backup] Failed to update titan.md frontmatter: %s", e)

    # -------------------------------------------------------------------------
    # Vault Shadow Hash Update (daily — backup archive hash → vault PDA)
    # -------------------------------------------------------------------------
    async def _update_vault_shadow_hash(self, archive_hash: str):
        """Update the vault's shadow_url_hash with the backup archive hash.

        This stores the backup verification hash on-chain in the vault PDA,
        making it queryable via Photon alongside the TimeChain merkle root.
        """
        if not self.network or self.network.pubkey is None:
            return
        try:
            import hashlib
            from titan_hcl.utils.solana_client import (
                build_vault_update_shadow_instruction, is_available,
            )
            if not is_available():
                return

            vault_program_id = getattr(self.network, '_vault_program_id', None)
            if not vault_program_id:
                # Try config
                cfg = getattr(self.network, '_config', {})
                vault_program_id = cfg.get("network", {}).get("vault_program_id", "")
            if not vault_program_id:
                return

            # Convert hex hash string to 32-byte hash
            hash_bytes = hashlib.sha256(archive_hash.encode("utf-8")).digest()

            ix = build_vault_update_shadow_instruction(
                self.network.pubkey, hash_bytes, vault_program_id,
            )
            if ix:
                sig = await self.network.send_sovereign_transaction([ix], priority="LOW")
                if sig:
                    logger.info("[Backup] Vault shadow hash updated: %s (tx=%s)",
                                archive_hash[:12], sig[:16] if len(sig) > 16 else sig)
        except Exception as e:
            swallow_warn('[Backup] Vault shadow hash update failed (non-critical)', e,
                         key="logic.backup.vault_shadow_hash_update_failed_non_crit", throttle=100)

    # -------------------------------------------------------------------------
    # ZK Epoch Snapshot (Solana — every meditation)
    # -------------------------------------------------------------------------
    async def _zk_epoch_snapshot(
        self, archive_hash: str, arweave_url: str | None,
        total_nodes: int, sovereignty_idx: float,
    ):
        """Create a ZK-compressed epoch snapshot on-chain via Light Protocol."""
        import hashlib as _hashlib
        from titan_hcl.utils.solana_client import (
            build_append_epoch_snapshot_instruction, is_available,
        )

        if not is_available() or self.network.pubkey is None:
            return

        vault_program_id = getattr(self, "_vault_program_id", None)
        if not vault_program_id:
            return

        try:
            state_root = _hashlib.sha256(archive_hash.encode("utf-8")).digest()
            url_str = arweave_url or f"local://{archive_hash}"
            url_hash = _hashlib.sha256(url_str.encode("utf-8")).digest()
            sovereignty_bp = int(sovereignty_idx * 100)

            ix = build_append_epoch_snapshot_instruction(
                authority_pubkey=self.network.pubkey,
                state_root=state_root,
                memory_count=total_nodes,
                sovereignty_score=sovereignty_bp,
                shadow_url_hash=url_hash,
                program_id_str=vault_program_id,
            )
            if ix:
                tx_sig = await self.network.send_sovereign_transaction(
                    [ix], priority="HIGH",
                )
                if tx_sig:
                    logger.info(
                        "[Backup] ZK epoch snapshot: memories=%d, sovereignty=%dbp",
                        total_nodes, sovereignty_bp,
                    )
                    # rFP_x_voice_enrichment §4.3.1 — PROOF_DAY needs the
                    # ZK Vault snapshot signature to render the "Seal" URL
                    # (iamtitan.tech/tx/{sig}). Persist alongside the
                    # backup_anchor_chain so the archetype can read the
                    # most-recent vault commit on demand.
                    self._persist_vault_snapshot(
                        tx_sig=tx_sig,
                        archive_hash=archive_hash,
                        memory_count=total_nodes,
                        sovereignty_bp=sovereignty_bp,
                        arweave_url=arweave_url or "",
                    )
        except Exception as e:
            swallow_warn('[Backup] ZK epoch snapshot skipped', e,
                         key="logic.backup.zk_epoch_snapshot_skipped", throttle=100)

    def _vault_snapshots_path(self) -> str:
        """Per-Titan ZK Vault snapshot history file (rFP_x_voice_enrichment §4.3.1)."""
        return f"data/zk_vault_snapshots_{self._titan_id}.json"

    def _persist_vault_snapshot(self, *, tx_sig: str, archive_hash: str,
                                 memory_count: int, sovereignty_bp: int,
                                 arweave_url: str = "") -> None:
        """Append a successful ZK Vault snapshot to the per-Titan history file.

        Atomic write via tmp+rename. Bounded to last 200 entries (~6 months
        at one meditation per ~daily). Failures are non-critical — the
        on-chain TX succeeded; this file is purely a queryable mirror.
        """
        import json as _json
        try:
            p = self._vault_snapshots_path()
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            existing: list = []
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        loaded = _json.load(f)
                    if isinstance(loaded, dict):
                        existing = loaded.get("snapshots", []) or []
                    elif isinstance(loaded, list):
                        existing = loaded
                except Exception:
                    existing = []
            existing.append({
                "tx_sig": tx_sig,
                "archive_hash": archive_hash,
                "memory_count": int(memory_count),
                "sovereignty_bp": int(sovereignty_bp),
                "arweave_url": arweave_url,
                "ts": int(time.time()),
            })
            existing = existing[-200:]
            payload = {
                "version": 1,
                "titan_id": self._titan_id,
                "snapshots": existing,
            }
            tmp = p + ".tmp"
            with open(tmp, "w") as f:
                _json.dump(payload, f, indent=2)
            os.replace(tmp, p)
        except Exception as e:
            swallow_warn('[Backup] vault snapshot persist failed', e,
                         key="logic.backup.vault_snapshot_persist_failed", throttle=100)

    # -------------------------------------------------------------------------
    # MyDay NFT (Solana — every 4th meditation)
    # -------------------------------------------------------------------------
    async def mint_epoch_nft(
        self, epoch: int, sovereignty_idx: float, diary_entry: str,
        total_nodes: int, art_path: str = None,
    ) -> Optional[str]:
        """Mint a MyDay Epoch NFT via Metaplex Core."""
        if self.network.keypair is None:
            logger.debug("[Backup] Cannot mint epoch NFT — no wallet keypair.")
            return None

        try:
            from solders.keypair import Keypair as SoldersKeypair
            from titan_hcl.utils.solana_client import (
                build_mpl_core_create_v1, is_available,
            )

            if not is_available():
                return None

            asset_kp = SoldersKeypair()
            asset_pubkey = asset_kp.pubkey()

            date_str = datetime.fromtimestamp(epoch).strftime("%Y-%m-%d")
            name = f"Titan Epoch {date_str}"

            # Use Arweave gateway for metadata (will be populated by backup record)
            latest = self.get_latest_backup_record("personality")
            if latest and latest.get("permanent_url"):
                uri = latest["permanent_url"]
            else:
                uri = f"ar://titan/epoch_{epoch}.json"

            attributes = {
                "Type": "Epoch",
                "Date": date_str,
                "Sovereignty": f"{sovereignty_idx:.1f}%",
                "Memory_Nodes": str(total_nodes),
                "Diary": diary_entry[:64],
            }

            ix = build_mpl_core_create_v1(
                asset_pubkey=asset_pubkey,
                payer_pubkey=self.network.pubkey,
                name=name[:32],
                uri=uri,
                attributes=attributes,
            )
            if ix is None:
                return None

            sig = await self.network.send_sovereign_transaction(
                [ix], priority="MEDIUM", extra_signers=[asset_kp],
            )

            if sig:
                addr = str(asset_pubkey)
                logger.info("[Backup] Epoch NFT minted: %s (TX: %s)", addr, sig)
                return addr

        except Exception as e:
            swallow_warn('[Backup] Epoch NFT mint failed', e,
                         key="logic.backup.epoch_nft_mint_failed", throttle=100)

        return None

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
        ("data/experience_orchestrator.db", "experience_orchestrator.db"),  # ~336MB: learned action wisdom
        ("data/experience_memory.db", "experience_memory.db"),    # ~51MB: experience records
        ("data/episodic_memory.db", "episodic_memory.db"),        # ~99MB: episodic records
        ("data/experiential_memory.db", "experiential_memory.db"),  # ~856KB: dream insights
        ("data/pi_heartbeat_state.json", "pi_heartbeat_state.json"),  # ~1KB
        # Narrative diary — RE-ADDED 2026-05-22 (BUG-CHRONICLE-WRITER-DEAD-POST-A87
        # fixed: titan_HCL._append_to_chronicle writes meditation reflections again
        # on MEDITATION_COMPLETE). Was removed from §24.4.B while the writer was dead.
        ("titan_chronicles.md", "titan_chronicles.md"),            # ~varies: Scholar's Chronicle reflections
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
            _backup_cfg = (self._full_config or {}).get("backup", {}) or {}
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

    async def upload_personality_to_arweave(
        self, archive_path: str = None, network: str = None,
    ) -> Optional[dict]:
        """Upload personality archive to Arweave (permanent).

        Returns dict with arweave_tx, archive_hash, size_mb.
        Network defaults to configured solana_network (mainnet-beta or devnet).
        Gated by [mainnet_budget] backup_arweave_enabled in config.toml.
        """
        # Resolve network from merged config if not explicitly passed
        if network is None:
            try:
                from titan_hcl.config_loader import load_titan_config
                cfg = load_titan_config()
                network = cfg.get("network", {}).get("solana_network", "devnet")
                # Map "mainnet-beta" → "mainnet" for ArweaveStore
                if network == "mainnet-beta":
                    network = "mainnet"
                # Check arweave gate
                budget = cfg.get("mainnet_budget", {})
                if not budget.get("backup_arweave_enabled", False):
                    logger.debug("[Backup] Arweave backup disabled (backup_arweave_enabled=false)")
                    return None
            except Exception:
                network = "devnet"

        # rFP Phase 2 cascade: S1 build → S2 validate → S3 local-always →
        # S4 balance → S5 upload → S6 verify → S7 manifest → S10 cleanup
        if not archive_path:
            import asyncio as _asyncio_local
            archive_path = await _asyncio_local.to_thread(
                self.create_personality_archive, arweave_tier=True)
        if not archive_path or not os.path.exists(archive_path):
            return None

        # Resolve ArweaveStore (prefer injected per rFP BUG-5)
        store = self._arweave_store
        if store is None:
            try:
                from titan_hcl.utils.arweave_store import ArweaveStore
                store = ArweaveStore(
                    keypair_path=(getattr(self.network, '_wallet_path', None)
                                  or getattr(self.network, '_keypair_path', None)),
                    network=network,
                )
            except Exception as e:
                logger.error("[Backup] ArweaveStore build failed: %s", e)
                with suppress(FileNotFoundError):
                    os.remove(archive_path)
                return None

        async def _upload_personality(path):
            """S5 closure — ArweaveStore upload (record-writing moved to main flow
            so the cascade-enriched dict, including Phase 7 encryption stanza,
            lands in the manifest)."""
            archive_hash = self.calculate_hash(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            tx_id = await store.upload_file(
                path,
                tags={
                    "Type": "Titan-Personality-Backup",
                    "Archive-Hash": archive_hash[:16],
                    "Size-MB": f"{size_mb:.1f}",
                    "Timestamp": str(int(time.time())),
                },
                content_type="application/gzip",
            )
            if not tx_id:
                return None
            logger.info("[Backup] Personality uploaded to Arweave: %s (%.1fMB, hash=%s...)",
                        tx_id[:16] if not tx_id.startswith("devnet") else tx_id,
                        size_mb, archive_hash[:12])
            return {
                "arweave_tx": tx_id,
                "archive_hash": archive_hash,
                "size_mb": round(size_mb, 2),
                "permanent_url": store.get_permanent_url(tx_id),
                "uploaded_at": time.time(),
            }

        from titan_hcl.logic.backup_cascade import BackupCascade
        retention = int(self._full_config.get("backup", {}).get("local_rolling_days", 30))
        local_dir = self._full_config.get("backup", {}).get(
            "local_dir", self._LOCAL_BACKUP_DIR)
        cascade = BackupCascade(full_config=self._full_config,
                                 arweave_store=store, local_dir=local_dir)
        # 2026-04-30 — wire Telegram notifier for auto-fund alerts (closes
        # rFP §5.5 silent-depletion gap). BackupCascade.auto_fund_irys_if_needed
        # invokes this callback on every successful auto-fund event.
        cascade._telegram_notifier = self._send_telegram_alert
        cascade_result = await cascade.run(
            archive_path, "personality", _upload_personality,
            get_latest_record_fn=self.get_latest_backup_record,
            retention_days=retention,
            encryption=self._build_encryption_context(),
        )

        # Persist the cascade-enriched record (includes Phase 7 encryption stanza)
        if cascade_result and cascade_result.get("arweave_tx"):
            self._store_backup_record("personality", cascade_result)

        # Cleanup temp build artifact (local copy is in data/backups/)
        if archive_path and archive_path.startswith("/tmp/"):
            with suppress(FileNotFoundError):
                os.remove(archive_path)

        return cascade_result

    # =========================================================================
    # Soul Package → Arweave (Weekly)
    # =========================================================================

    def create_soul_package(self, output_path: str = None) -> Optional[str]:
        """Create full soul package: personality + consciousness + knowledge graph.

        This is the weekly backup — everything needed to fully resurrect Titan.

        Returns output path on success, None on failure.
        """
        if not output_path:
            output_path = f"/tmp/titan_soul_{int(time.time())}.tar.gz"

        try:
            all_paths = self.PERSONALITY_PATHS + self.WEEKLY_EXTRA_PATHS
            skip_patterns = self._BACKUP_SKIP_PATTERNS

            def _filter(ti):
                name = ti.name
                if name.endswith(('.tmp', '.pyc')) or '__pycache__' in name:
                    return None
                # Weekly soul INCLUDES .jsonl (forensic replay tier)
                # Only skip historical dev backups — one-time snapshots not live state
                if any(p in name for p in skip_patterns):
                    return None
                return ti

            with tarfile.open(output_path, "w:gz", compresslevel=9) as tar:
                for source_path, archive_name in all_paths:
                    source = Path(source_path)
                    if source.exists():
                        if source.is_dir():
                            tar.add(str(source), arcname=archive_name, filter=_filter)
                        else:
                            tar.add(str(source), arcname=archive_name)
                        logger.debug("[Backup] Soul package: added %s", archive_name)

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info("[Backup] Soul package: %.1f MB at %s", size_mb, output_path)
            return output_path

        except Exception as e:
            logger.error("[Backup] Soul package failed: %s", e)
            return None

    async def upload_soul_package_to_arweave(
        self, network: str = "devnet"
    ) -> Optional[dict]:
        """Upload full soul package to Arweave (weekly).

        Includes personality + consciousness.db + Kuzu graph.
        ~200MB compressed, ~$1 on Arweave via Irys.
        """
        # rFP Phase 2 cascade for soul (weekly): same 10-step failsafe flow.
        # Phase E.2.4: ~200MB tarball + gzip-9 compression — wrap to_thread.
        import asyncio as _asyncio_local
        archive_path = await _asyncio_local.to_thread(self.create_soul_package)
        if not archive_path or not os.path.exists(archive_path):
            return None

        store = self._arweave_store
        if store is None:
            try:
                from titan_hcl.utils.arweave_store import ArweaveStore
                store = ArweaveStore(
                    keypair_path=(getattr(self.network, '_wallet_path', None)
                                  or getattr(self.network, '_keypair_path', None)),
                    network=network,
                )
            except Exception as e:
                logger.error("[Backup] ArweaveStore build failed: %s", e)
                with suppress(FileNotFoundError):
                    os.remove(archive_path)
                return None

        async def _upload_soul(path):
            """S5 closure — record-writing moved to main flow (see personality)."""
            archive_hash = self.calculate_hash(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            tx_id = await store.upload_file(
                path,
                tags={
                    "Type": "Titan-Soul-Package",
                    "Archive-Hash": archive_hash[:16],
                    "Size-MB": f"{size_mb:.1f}",
                    "Timestamp": str(int(time.time())),
                },
                content_type="application/gzip",
            )
            if not tx_id:
                return None
            logger.info("[Backup] Soul package uploaded to Arweave: %s (%.1fMB)",
                        tx_id[:16] if not tx_id.startswith("devnet") else tx_id,
                        size_mb)
            return {
                "arweave_tx": tx_id,
                "archive_hash": archive_hash,
                "size_mb": round(size_mb, 2),
                "permanent_url": store.get_permanent_url(tx_id),
                "uploaded_at": time.time(),
            }

        from titan_hcl.logic.backup_cascade import BackupCascade
        retention = int(self._full_config.get("backup", {}).get(
            "soul_local_rolling_days", 90))
        local_dir = self._full_config.get("backup", {}).get(
            "local_dir", self._LOCAL_BACKUP_DIR)
        cascade = BackupCascade(full_config=self._full_config,
                                 arweave_store=store, local_dir=local_dir)
        # 2026-04-30 — wire Telegram notifier for auto-fund alerts (soul path).
        cascade._telegram_notifier = self._send_telegram_alert
        cascade_result = await cascade.run(
            archive_path, "soul", _upload_soul,
            get_latest_record_fn=self.get_latest_backup_record,
            retention_days=retention,
            encryption=self._build_encryption_context(),
        )

        # Persist cascade-enriched record (Phase 7 — captures encryption stanza)
        if cascade_result and cascade_result.get("arweave_tx"):
            self._store_backup_record("soul_package", cascade_result)

        if archive_path and archive_path.startswith("/tmp/"):
            with suppress(FileNotFoundError):
                os.remove(archive_path)

        return cascade_result

    # -------------------------------------------------------------------------
    # Phase 7 — Encryption context (opt-in via [backup].encryption_enabled)
    # -------------------------------------------------------------------------
    def _build_encryption_context(self) -> Optional[dict]:
        """Delegate to shared helper. Returns None when encryption disabled."""
        from titan_hcl.logic.backup_crypto import build_encryption_context_from_config
        return build_encryption_context_from_config(self._full_config)

    # -------------------------------------------------------------------------
    # Backup Records (local verification)
    # -------------------------------------------------------------------------
    _MANIFEST_VERSION = "1.0"

    def _store_backup_record(self, backup_type: str, record: dict):
        """Store backup record locally for verification queries.

        Phase 7 — bumps manifest_version to 1.0 and preserves the `encryption`
        stanza threaded through by BackupCascade. Legacy records lacking these
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
    # Telegram Alerts — Notify Maker on backup events
    # -------------------------------------------------------------------------
    _TELEGRAM_BOT_TOKEN = "8531091229:AAElGsqbsLDvDfxaCwMwG1qGdBzyVlksI0c"
    _TELEGRAM_CHAT_ID = "6345894322"

    def _send_telegram_alert(self, message: str):
        """Send backup alert to Maker via Telegram. Non-blocking, fire-and-forget.

        Phase E.2.4 fix: previous version used sync httpx.post inside a sync
        method called from async contexts. The "non-blocking" comment was
        aspirational — the call actually blocked the event loop for up to
        10s on slow Telegram API responses. Now uses a daemon thread so
        it's truly fire-and-forget regardless of caller context.
        """
        def _post():
            try:
                import httpx
                url = f"https://api.telegram.org/bot{self._TELEGRAM_BOT_TOKEN}/sendMessage"
                httpx.post(url, json={
                    "chat_id": self._TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "Markdown",
                }, timeout=10)
            except Exception as e:
                swallow_warn('[Backup] Telegram alert failed (non-critical)', e,
                             key="logic.backup.telegram_alert_failed_non_critical", throttle=100)
        import threading
        threading.Thread(target=_post, daemon=True,
                         name="telegram-alert").start()

    def _alert_backup_success(self, backup_type: str, size_mb: float,
                               archive_hash: str, arweave_tx: str = ""):
        """Alert Maker on successful backup."""
        tx_info = f"\nArweave: `{arweave_tx[:20]}...`" if arweave_tx else "\nStorage: local"
        self._send_telegram_alert(
            f"✅ *Titan Backup OK*\n"
            f"Type: {backup_type}\n"
            f"Size: {size_mb:.1f} MB\n"
            f"Hash: `{archive_hash[:16]}`{tx_info}"
        )

    def _alert_backup_failure(self, backup_type: str, error: str):
        """Alert Maker on backup failure — immediate."""
        self._send_telegram_alert(
            f"🔴 *Titan Backup FAILED*\n"
            f"Type: {backup_type}\n"
            f"Error: {error[:200]}"
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
            self._send_telegram_alert(
                f"🔄 *Titan Restore Complete*\n"
                f"Files: {len(restored)}\nErrors: {len(errors)}"
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
            self._send_telegram_alert(
                f"🔄 *Titan Restore Started*\nDownloading from Arweave: `{tx_id[:20]}...`"
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
                    net = (self._full_config or {}).get("network", {}) or {}
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

    # ── SPEC §24.7 — ZK Vault Merkle commit per unified-manifest event ─────
    #
    # Distinct from `anchor_backup_hash` above (which is the legacy Phase 8
    # per-asset v=2 chain). This method anchors the EVENT-level Merkle root
    # — one event = personality + timechain + soul together = one Solana
    # commit — as required by SPEC §24.7 for the Arweave plane's
    # backup_unified_manifest.
    #
    # The two anchor channels coexist:
    #   anchor_backup_hash       — per-asset chain (TITAN|BACKUP|v=2|...)
    #   commit_event_merkle_to_zk_vault  — per-event chain (v=2;event_id=...)
    # Restore walks both chains independently per §24.7 + §11.H.4.
    async def commit_event_merkle_to_zk_vault(
        self,
        event_id: str,
        event_merkle_root: str,
        prev_event_merkle_root: Optional[str] = None,
    ) -> Optional[str]:
        """SPEC §24.7 — submit event_merkle_root as v=2 Solana memo.

        Builds the canonical `v=2;event_id={id};root={root[:32]};prev={prev[:16]}`
        memo (or `prev=genesis` for first event) and sends via existing
        `network.send_sovereign_transaction([memo_ix], priority="LOW")`.

        Returns Solana tx_id on success — caller stores in the manifest
        event's `zk_commit_tx` field. Returns None on failure (caller
        emits BACKUP_EVENT_FAILED + retries on next meditation cycle).
        """
        from titan_hcl.logic.backup_zk_commit import build_zk_memo

        if not self.network or not hasattr(self.network, "send_sovereign_transaction"):
            return None

        try:
            from titan_hcl.utils.solana_client import (
                build_memo_instruction, is_available)
            if not is_available() or self.network.keypair is None:
                return None

            memo_text = build_zk_memo(
                event_id=event_id,
                event_merkle_root=event_merkle_root,
                prev_event_merkle_root=prev_event_merkle_root,
            )
            memo_ix = build_memo_instruction(self.network.pubkey, memo_text)
            sig = await self.network.send_sovereign_transaction(
                [memo_ix], priority="LOW"
            )
            if sig:
                logger.info(
                    "[Backup] §24.7 event Merkle anchored on-chain: tx=%s "
                    "event_id=%s root=%s...",
                    sig[:20] if len(sig) > 20 else sig,
                    event_id[:8], event_merkle_root[:16])
            return sig
        except Exception as e:
            logger.warning("[Backup] §24.7 ZK Vault event commit failed: %s", e)
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
    ) -> Optional[dict]:
        """Emit one v=3 memo per uploaded component for a single backup event.

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
        backup_cfg = (self._full_config or {}).get("backup", {}) or {}
        mode = "B" if backup_cfg.get("encryption_enabled", False) else "A"
        url_key = None
        if mode == "A":
            try:
                net_cfg = (self._full_config or {}).get("network", {}) or {}
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
            sovereignty_bp = int(float(getattr(self, "_last_sovereignty_idx", 0.0)) * 100)
        except Exception:
            sovereignty_bp = 0

        component_sigs: dict = {}
        head_sig: Optional[str] = None
        for i, comp in enumerate(components):
            try:
                memo = build_v3_memo(
                    event_id=event_id, ts=int(ts), event_type=event_type,
                    tier=comp["tier"], archive_hash=comp["arc"],
                    merkle_root=event_merkle_root, arweave_tx=comp["tx_id"],
                    mode=mode, prev_sig=prev_sig, url_key=url_key,
                )
            except Exception as e:
                logger.error("[Backup] v=3 memo build failed (%s): %s",
                             comp.get("tier"), e)
                return None
            memo_ix = build_memo_instruction(self.network.pubkey, memo)
            if memo_ix is None:
                logger.error("[Backup] v=3 memo ix build failed (%s)", comp.get("tier"))
                return None
            ixs = [memo_ix]
            if i == 0:  # head memo co-bundles commit_state(event_merkle_root)
                try:
                    state_root = bytes.fromhex(event_merkle_root)
                except ValueError:
                    logger.error("[Backup] v=3: event_merkle_root not valid hex")
                    return None
                commit_ix = build_vault_commit_instruction(
                    self.network.pubkey, state_root, sovereignty_bp, vault_program_id)
                if commit_ix is not None:
                    ixs = [commit_ix, memo_ix]
                else:
                    logger.warning(
                        "[Backup] v=3: commit_state ix unavailable — head memo-only")
            sig = await self.network.send_sovereign_transaction(ixs, priority="LOW")
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
        return {"head_sig": head_sig, "component_sigs": component_sigs}

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
                cfg.get("backup", {}).get("unified_v2_enabled", False)
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

    async def _run_unified_event_v2(self, weekday: int) -> bool:
        """Production wiring for backup_upload_pipeline.run_unified_event.

        Returns True if the event was shipped (manifest updated + ZK
        commit landed), False otherwise. Caller in on_meditation_complete
        uses the bool to decide whether to fall through to legacy upload.
        """
        from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
        from titan_hcl.logic.backup_upload_pipeline import (
            run_unified_event,
        )

        # Gate on Arweave-enabled config.
        #
        # 2026-05-17 — definitive cause of "Phase 11 flip shipped but no
        # Arweave TX": pre-fix this gate read from `self.network._config`
        # via `hasattr(self.network, "_config")`. BackupWorker subprocess
        # builds RebirthBackup with `network_client=None` (subprocess has
        # no direct Solana client; line ~153 of modules/backup_worker.py),
        # so `self.network is None` → `hasattr(None, "_config")` is False
        # → `_budget = {}` → `_budget.get("backup_arweave_enabled", False)`
        # is False → gate silently returns False at DEBUG level (invisible
        # in default-INFO brain log). For 36h after the Phase 11 flip:
        # config flag True + ArweaveStore wired + meditations firing, but
        # ZERO §24 v2 event-ship attempts.
        #
        # Correct source: self._full_config — same path that the sibling
        # gate `_unified_v2_enabled()` (backup.py:1746) reads. Already set
        # by the RebirthBackup constructor + matches whatever
        # `config_loader.load_titan_config()` produces.
        try:
            _budget = (self._full_config or {}).get("mainnet_budget", {}) or {}
        except Exception:
            _budget = {}
        if not _budget.get("backup_arweave_enabled", False):
            logger.debug(
                "[Backup] §24 unified_v2: backup_arweave_enabled=false — skipping",
            )
            return False

        store = self._ensure_arweave_store_for_unified()
        if store is None:
            logger.warning(
                "[Backup] §24 unified_v2: no ArweaveStore available",
            )
            return False

        # Load (or initialize) the manifest
        try:
            manifest = UnifiedManifest.load(
                titan_id=self._titan_id, base_dir="data",
            )
        except ValueError as e:
            logger.error(
                "[Backup] §24 unified_v2: manifest load failed: %s "
                "(manual recovery required)", e,
            )
            return False

        # Build per-tier specs from the existing class-level path tuples
        p_specs = self._tier_specs_from_paths(self.PERSONALITY_PATHS)
        t_specs = self._tier_specs_from_paths(
            self.TIMECHAIN_PATHS, format_hint="timechain_bin",
        )
        # §24.4.C conformance: the soul tier ships on weekly Sundays
        # (incremental diff) AND on every baseline event (a baseline is a full
        # snapshot of ALL in-scope paths per §24.2 — soul included regardless
        # of weekday). Gating on weekday==6 ALONE was the bug that left the
        # soul tier without an Arweave baseline (so every Sunday full-shipped
        # consciousness.db). _refresh_baseline_working_dir then keeps the
        # baseline-dir soul == the shipped soul, so weekly Sundays diff.
        _should_rebase, _ = manifest.should_rebase()
        s_specs = (
            self._tier_specs_from_paths(self.WEEKLY_EXTRA_PATHS)
            if (weekday == 6 or _should_rebase) else None
        )

        # Baseline resolver — for incremental events, point at the
        # baseline working dir (refreshed after each baseline ship). For
        # the first event after enabling unified_v2 the working dir may
        # be empty; the pipeline falls back to full-ship in that case.
        base_dir = self._baseline_working_dir()

        def _baseline_resolver(component, arc_name):
            candidate = os.path.join(base_dir, arc_name)
            return candidate if os.path.exists(candidate) else None

        # Arweave uploader — wraps store.upload_file with a temp file
        async def _arweave_upload(data: bytes, tags: dict) -> str:
            import tempfile
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".tar.gz",
                prefix=f"titan_unified_{self._titan_id}_",
            ) as f:
                f.write(data)
                tmp_path = f.name
            try:
                tx = await store.upload_file(tmp_path, tags=tags)
                if not tx:
                    raise RuntimeError(
                        "ArweaveStore.upload_file returned empty tx_id"
                    )
                return tx
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # v=3 sovereign chain committer (5J-2) — emits per-component memos with
        # the Arweave URL on-chain + commit_state on the head, replacing the
        # v=2 merkle-only anchor. prev_sig threads event-heads back to genesis.
        async def _v3_chain_commit(event_id: str, ts: int, event_type: str,
                                   event_root: str, components: list,
                                   prev_sig) -> Optional[dict]:
            return await self.commit_event_v3_chain(
                event_id=event_id, ts=ts, event_type=event_type,
                event_merkle_root=event_root, components=components,
                prev_sig=prev_sig,
            )

        # Bus emit — wire to plugin/bus if available; non-fatal on absence
        def _bus_emit(name: str, payload: dict) -> None:
            try:
                bus = getattr(self, "bus", None)
                if bus is None:
                    bus = getattr(self, "_bus", None)
                if bus is not None and hasattr(bus, "emit"):
                    bus.emit(name, payload)
            except Exception:
                pass

        result = await run_unified_event(
            titan_id=self._titan_id, manifest=manifest,
            personality_specs=p_specs, timechain_specs=t_specs,
            soul_specs=s_specs, baseline_resolver=_baseline_resolver,
            arweave_uploader=_arweave_upload, zk_committer=_v3_chain_commit,
            bus_emit=_bus_emit,
        )

        if result.status != "shipped":
            logger.warning(
                "[Backup] §24 unified_v2 event NOT shipped: status=%s "
                "errors=%s", result.status, result.errors,
            )
            return False

        # On baseline events, refresh the working dir so subsequent
        # incrementals diff against the just-shipped baseline state
        if result.event_type == "baseline":
            try:
                self._refresh_baseline_working_dir()
            except Exception as e:
                logger.warning(
                    "[Backup] §24 unified_v2: baseline working dir refresh "
                    "failed: %s (next incremental may full-ship)", e,
                )
        return True

    # ── Phase 2 pre-stage (2026-05-31) ─────────────────────────────────────
    #
    # Splits the unified_v2 event into a heavy BUILD (snapshot+diff+pack — run by
    # the backup worker's stager OFF the recv loop, ahead of the first meditation)
    # and a fast SHIP (upload staged tarballs + ZK + manifest — run on meditation).
    # Removes the multi-minute diff-build from the bus-blocking meditation path.
    # _run_unified_event_v2 above is the unchanged inline path (manual trigger +
    # cold-start fallback when no fresh stage exists).

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

    def _build_staged_event_v2(self, weekday: int):
        """Pre-BUILD a unified event (no upload, no manifest mutation). Returns a
        StagedEvent or None (arweave gate off / manifest unloadable). Mirrors
        _run_unified_event_v2's spec/resolver wiring."""
        from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
        from titan_hcl.logic.backup_upload_pipeline import build_unified_event
        try:
            _budget = (self._full_config or {}).get("mainnet_budget", {}) or {}
        except Exception:
            _budget = {}
        if not _budget.get("backup_arweave_enabled", False):
            logger.debug("[Backup] §24 build-stage: backup_arweave_enabled=false")
            return None
        try:
            manifest = UnifiedManifest.load(
                titan_id=self._titan_id, base_dir="data")
        except ValueError as e:
            logger.error("[Backup] §24 build-stage: manifest load failed: %s", e)
            return None
        p_specs = self._tier_specs_from_paths(self.PERSONALITY_PATHS)
        t_specs = self._tier_specs_from_paths(
            self.TIMECHAIN_PATHS, format_hint="timechain_bin")
        # §24.4.C conformance (see _run_unified_event_v2): soul ships on weekly
        # Sundays AND on every baseline event (full snapshot of all paths).
        _should_rebase, _ = manifest.should_rebase()
        s_specs = (self._tier_specs_from_paths(self.WEEKLY_EXTRA_PATHS)
                   if (weekday == 6 or _should_rebase) else None)
        base_dir = self._baseline_working_dir()

        def _baseline_resolver(component, arc_name):
            candidate = os.path.join(base_dir, arc_name)
            return candidate if os.path.exists(candidate) else None

        staged = build_unified_event(
            titan_id=self._titan_id, manifest=manifest,
            personality_specs=p_specs, timechain_specs=t_specs,
            soul_specs=s_specs, baseline_resolver=_baseline_resolver,
        )
        logger.info(
            "[Backup] §24 staged event BUILT: id=%s type=%s (off-loop; "
            "awaiting meditation to ship)", staged.event_id[:8],
            staged.event_type)
        return staged

    async def _ship_staged_event_v2(self, staged) -> bool:
        """SHIP a pre-built StagedEvent (fast, on meditation). Reloads the
        manifest fresh for the staleness check + append. Returns True if shipped;
        False on stale-baseline (caller rebuilds) / gate-off / failure."""
        from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
        from titan_hcl.logic.backup_upload_pipeline import ship_staged_event
        store = self._ensure_arweave_store_for_unified()
        if store is None:
            logger.warning("[Backup] §24 ship-stage: no ArweaveStore available")
            return False
        try:
            manifest = UnifiedManifest.load(
                titan_id=self._titan_id, base_dir="data")
        except ValueError as e:
            logger.error("[Backup] §24 ship-stage: manifest load failed: %s", e)
            return False

        async def _arweave_upload(data: bytes, tags: dict) -> str:
            import tempfile
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".tar.gz",
                prefix=f"titan_unified_{self._titan_id}_",
            ) as f:
                f.write(data)
                tmp_path = f.name
            try:
                tx = await store.upload_file(tmp_path, tags=tags)
                if not tx:
                    raise RuntimeError(
                        "ArweaveStore.upload_file returned empty tx_id")
                return tx
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

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

        result = await ship_staged_event(
            staged, manifest=manifest, arweave_uploader=_arweave_upload,
            zk_committer=_v3_chain_commit, bus_emit=_bus_emit)

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
        if result.event_type == "baseline":
            try:
                self._refresh_baseline_working_dir()
            except Exception as e:
                logger.warning(
                    "[Backup] §24 ship-stage: baseline working dir refresh "
                    "failed: %s (next incremental may full-ship)", e)
        return True

    def _auto_fund_irys_before_upload(self) -> None:
        """Re-homed Irys auto-fund (2026-05-31). The auto-fund hook lived only in
        the legacy BackupCascade.run() path, which the unified_v2 migration
        retired — so it stopped firing after 2026-05-19 and the deposit went red.
        Re-invoke it from the unified_v2 daily path, before the upload, with the
        SAME caps (config-gated by [backup].auto_fund_enabled + daily cap +
        runway floor + wallet reserve; the audit log + Telegram alert are emitted
        inside auto_fund_irys_if_needed). No-op when disabled / runway sufficient.
        """
        bcfg = (self._full_config or {}).get("backup", {}) or {}
        if not bcfg.get("auto_fund_enabled", False):
            return
        from titan_hcl.logic.backup_cascade import BackupCascade
        cascade = BackupCascade(full_config=self._full_config)
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
        result = cascade.auto_fund_irys_if_needed(size_mb)
        action = (result or {}).get("action")
        if action == "funded":
            logger.info(
                "[Backup] §24 Irys auto-fund: FUNDED %.4f SOL (runway was %.2fd) "
                "tx=%s", result.get("amount_sol", 0.0),
                result.get("runway_before_days", 0.0),
                str(result.get("tx_id", ""))[:16])
        elif action and action != "no_action":
            logger.info(
                "[Backup] §24 Irys auto-fund: %s (%s)",
                action, result.get("reason", ""))

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
                        self._full_config.get("backup", {}).get("local_rolling_days", 30)))
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
                    self._full_config.get("backup", {}).get("local_rolling_days", 30)))
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
