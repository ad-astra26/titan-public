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
import logging
import os
import tarfile
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from titan_plugin.utils.crypto import hash_file
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)


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

        # Will be wired by TitanPlugin.__init__
        self.memory = None
        self.social = None
        self._photon = None

        # Backup state tracking (calendar-day based)
        self._last_personality_date = ""  # "YYYY-MM-DD"
        self._last_soul_date = ""         # "YYYY-MM-DD"
        self._meditation_count = 0
        self._meditation_count_since_nft = 0

        # Load persisted backup state if available
        self._load_backup_state()

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
        if today != self._last_personality_date:
            self._last_personality_date = today
            self._save_backup_state()
            logger.info("[Backup] Daily personality backup triggered (1st meditation of %s)", today)
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
        if weekday == 6 and today != self._last_soul_date:
            self._last_soul_date = today
            self._save_backup_state()
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
        if today != getattr(self, '_last_timechain_date', ''):
            self._last_timechain_date = today
            try:
                from titan_plugin.logic.timechain_backup import TimeChainBackup
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
                                from titan_plugin.utils.arweave_store import ArweaveStore
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

        # 4. MyDay NFT (every 4th meditation)
        meditations_per_nft = 4
        if self._meditation_count_since_nft >= meditations_per_nft:
            self._meditation_count_since_nft = 0
            self._save_backup_state()
            try:
                from titan_plugin.logic.reflection import ReflectionLogic
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

        # Save state after all operations
        self._save_backup_state()

    async def _compute_sovereignty(self) -> float:
        """Compute sovereignty index from available data."""
        try:
            from titan_plugin.logic.reflection import ReflectionLogic
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
            from titan_plugin.logic.backup_cascade import BackupCascade
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
            from titan_plugin.utils.solana_client import (
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
        from titan_plugin.utils.solana_client import (
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
        except Exception as e:
            swallow_warn('[Backup] ZK epoch snapshot skipped', e,
                         key="logic.backup.zk_epoch_snapshot_skipped", throttle=100)

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
            from titan_plugin.utils.solana_client import (
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

    # All personality-critical data paths
    PERSONALITY_PATHS = [
        ("data/neural_nervous_system/", "neural_ns"),              # ~84MB: NS weights + training buffers
        ("data/neuromodulator/", "neuromodulator"),                # ~7KB: DNA + allostatic state
        ("data/inner_memory.db", "inner_memory.db"),              # ~127MB: vocabulary, fires, compositions
        ("data/titan_memory.duckdb", "titan_memory.duckdb"),      # ~34MB: all memory nodes
        ("data/memory_vectors.faiss", "memory_vectors.faiss"),    # ~4MB: FAISS semantic index
        ("data/memory_vectors.faiss.idmap.json", "memory_vectors.faiss.idmap.json"),  # ID map
        ("data/experience_orchestrator.db", "experience_orchestrator.db"),  # ~118MB: learned action wisdom
        ("data/experience_memory.db", "experience_memory.db"),    # ~51MB: experience records
        ("data/episodic_memory.db", "episodic_memory.db"),        # ~99MB: episodic records
        ("data/experiential_memory.db", "experiential_memory.db"),  # ~856KB: dream insights
        ("data/pi_heartbeat_state.json", "pi_heartbeat_state.json"),  # ~1KB
        ("titan_constitution.md", "titan_constitution.md"),       # ~7KB: immutable identity
        ("titan_chronicles.md", "titan_chronicles.md"),           # ~3KB: diary
        ("data/titan_directives.sig", "titan_directives.sig"),    # ~200B: hash + Ed25519 sig
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
        ("data/titan_identity.json", "titan_identity.json"),        # ~0.1KB: T1/T2/T3 identity config
        # ─── Added 2026-04-20 (rFP_backup_worker Phase 0 persistence audit) ────
        # Systems shipped after rFP written (2026-04-12). Silently omitted from
        # daily backup until this audit. Includes reasoning subsystem which had
        # a 2-day silent data-loss bug closed 2026-04-19 (BUG-META-STATS-
        # PERSISTENCE) — that state now has offsite Arweave coverage.
        ("data/reasoning/", "reasoning"),                                               # ~7.6MB: meta_stats, chain_iql, policy_net, value_head, sequence_quality, meta_policy, meta_autoencoder
        ("data/meta_cgn/", "meta_cgn"),                                                 # ~4.5MB: META-CGN consumer (jsonl logs filtered from daily, included in soul)
        ("data/emot_cgn/", "emot_cgn"),                                                 # ~64KB: EMOT-CGN 8th consumer (shipped 2026-04-20)
        ("data/filter_down_v5_state.json", "filter_down_v5_state.json"),                # ~4KB
        ("data/filter_down_v5_weights.json", "filter_down_v5_weights.json"),            # ~616KB
        ("data/filter_down_v5_buffer.json", "filter_down_v5_buffer.json"),              # ~5.5MB: V5 replay buffer
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
        ("data/runtime_keypair.json", "runtime_keypair.json"),                          # ~295B: runtime-rotated keypair
        ("data/genesis_record.json", "genesis_record.json"),                            # ~1.6KB
        ("data/genesis_nft_metadata.json", "genesis_nft_metadata.json"),                # ~1.9KB
        ("data/birth_dna_snapshot.json", "birth_dna_snapshot.json"),                    # ~7.1KB
        ("data/zk_queue/pending.json", "zk_queue/pending.json"),                        # ~19KB
        ("data/timechain/contract_stats.json", "timechain/contract_stats.json"),        # ~2.5KB
        ("data/sage_memory/buffer_metadata.json", "sage_memory/buffer_metadata.json"),  # ~20B
        ("data/sage_memory/meta.json", "sage_memory/meta.json"),                        # ~489B
        ("data/events_teacher.db", "events_teacher.db"),                                # ~232KB: events teacher memory
        ("data/grammar_rules.db", "grammar_rules.db"),                                  # ~20KB: learned grammar
    ]

    # Filename patterns excluded from ALL archives — historical dev backups that
    # accumulated in tracked dirs over time. NOT live state. Examples:
    #   data/reasoning/policy_net_backup_20260328_pre_reward_fix.json
    #   data/reasoning/meta_autoencoder.json.bak_20260417
    _BACKUP_SKIP_PATTERNS = ('_backup_', '.bak_', '.bak', '.pre_')

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

    # Weekly full backup — adds consciousness history + TimeChain (large, append-only)
    WEEKLY_EXTRA_PATHS = [
        ("data/consciousness.db", "consciousness.db"),            # ~1.7GB: 410K+ epochs of life history
        ("data/knowledge_graph.kuzu", "knowledge_graph.kuzu"),    # Kuzu entity graph
        # MSL extended state (convergence history)
        ("data/msl/msl_convergence_log.json", "msl/msl_convergence_log.json"),  # ~100KB
        ("data/msl/msl_stats.json", "msl/msl_stats.json"),                      # ~2KB
        # TimeChain — Titan's own blockchain (added 2026-04-06)
        ("data/timechain/", "timechain"),                          # ~47MB: all fork chains + index
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
        # Backup lifecycle history + Titan cycle artifacts
        ("data/backup_records/", "backup_records"),                # daily backup records
        ("data/daily_nfts/", "daily_nfts"),                        # MyDay NFT records
        ("data/testaments/", "testaments"),                        # great-cycle testament history
        ("data/meditation_memos/", "meditation_memos"),            # meditation memo history
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

            with tarfile.open(output_path, "w:gz", compresslevel=9) as tar:
                for source_path, archive_name in self.PERSONALITY_PATHS:
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
                from titan_plugin.config_loader import load_titan_config
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
                from titan_plugin.utils.arweave_store import ArweaveStore
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

        from titan_plugin.logic.backup_cascade import BackupCascade
        retention = int(self._full_config.get("backup", {}).get("local_rolling_days", 30))
        local_dir = self._full_config.get("backup", {}).get(
            "local_dir", self._LOCAL_BACKUP_DIR)
        cascade = BackupCascade(full_config=self._full_config,
                                 arweave_store=store, local_dir=local_dir)
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
                from titan_plugin.utils.arweave_store import ArweaveStore
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

        from titan_plugin.logic.backup_cascade import BackupCascade
        retention = int(self._full_config.get("backup", {}).get(
            "soul_local_rolling_days", 90))
        local_dir = self._full_config.get("backup", {}).get(
            "local_dir", self._LOCAL_BACKUP_DIR)
        cascade = BackupCascade(full_config=self._full_config,
                                 arweave_store=store, local_dir=local_dir)
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
        from titan_plugin.logic.backup_crypto import build_encryption_context_from_config
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
            from titan_plugin.utils.arweave_store import ArweaveStore
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
                    from titan_plugin.logic.backup_crypto import (
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
            from titan_plugin.utils.solana_client import build_memo_instruction, is_available
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
