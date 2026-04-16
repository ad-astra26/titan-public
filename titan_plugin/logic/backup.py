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
            logger.debug("[Backup] No backup state loaded: %s", e)

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
                        logger.debug("[Backup] TimeChain ArweaveStore fallback init: %s", _ae)

                tc_backup = TimeChainBackup(
                    data_dir="data/timechain",
                    titan_id=self._titan_id,
                    arweave_store=_tc_arweave,
                )
                # rFP_backup_worker Phase 0: use tarball path (proven working via cron).
                # JSON+base64 path previously silently returned None on mainnet — retired 2026-04-13.
                tc_tx = await tc_backup.snapshot_to_arweave()
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
                logger.debug("[Backup] MyDay NFT skipped: %s", e)

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
        """Boot-time verification: check that critical data files exist."""
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
            logger.debug("[Backup] Vault shadow hash update failed (non-critical): %s", e)

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
            logger.debug("[Backup] ZK epoch snapshot skipped: %s", e)

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
            logger.debug("[Backup] Epoch NFT mint failed: %s", e)

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
    ]

    # Arweave-excluded paths: large DBs that rebuild from experience, uploaded weekly not daily
    # These are still in local daily backups (PERSONALITY_PATHS) — just not in daily Arweave
    ARWEAVE_DAILY_EXCLUDE = {
        "experience_orchestrator.db",   # ~118MB: rebuilds from new experiences
        "experience_memory.db",         # ~51MB: historical experience records
        "episodic_memory.db",           # ~99MB: episodic records
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
            with tarfile.open(output_path, "w:gz", compresslevel=9) as tar:
                for source_path, archive_name in self.PERSONALITY_PATHS:
                    # Skip large experience DBs for Arweave tier
                    if arweave_tier and archive_name in self.ARWEAVE_DAILY_EXCLUDE:
                        logger.debug("[Backup] Arweave tier: skipping %s", archive_name)
                        continue

                    source = Path(source_path)
                    if source.exists():
                        if source.is_dir():
                            def _filter(ti):
                                if ti.name.endswith(('.tmp', '.pyc')) or '__pycache__' in ti.name:
                                    return None
                                return ti
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

        if not archive_path:
            # Use arweave_tier to exclude large experience DBs (saves ~$30/month).
            # Phase E.2.4: tarball creation is CPU-bound (Zstd/gzip compression
            # over potentially hundreds of MB) — wrap in to_thread so event
            # loop stays responsive during backup.
            import asyncio as _asyncio_local
            archive_path = await _asyncio_local.to_thread(
                self.create_personality_archive, arweave_tier=True)
        if not archive_path or not os.path.exists(archive_path):
            return None

        archive_hash = self.calculate_hash(archive_path)
        size_mb = os.path.getsize(archive_path) / (1024 * 1024)

        try:
            from titan_plugin.utils.arweave_store import ArweaveStore
            store = ArweaveStore(
                keypair_path=getattr(self.network, '_wallet_path', None) or getattr(self.network, '_keypair_path', None),
                network=network,
            )

            tx_id = await store.upload_file(
                archive_path,
                tags={
                    "Type": "Titan-Personality-Backup",
                    "Archive-Hash": archive_hash[:16],
                    "Size-MB": f"{size_mb:.1f}",
                    "Timestamp": str(int(time.time())),
                },
                content_type="application/gzip",
            )

            if tx_id:
                result = {
                    "arweave_tx": tx_id,
                    "archive_hash": archive_hash,
                    "size_mb": round(size_mb, 2),
                    "permanent_url": store.get_permanent_url(tx_id),
                    "uploaded_at": time.time(),
                }
                logger.info("[Backup] Personality uploaded to Arweave: %s (%.1fMB, hash=%s...)",
                            tx_id[:16] if not tx_id.startswith("devnet") else tx_id,
                            size_mb, archive_hash[:12])
                self._store_backup_record("personality", result)

                # Cleanup temp archive
                with suppress(FileNotFoundError):
                    os.remove(archive_path)

                return result

        except Exception as e:
            logger.error("[Backup] Arweave personality upload failed: %s", e)

        # Cleanup on failure too
        if archive_path:
            with suppress(FileNotFoundError):
                os.remove(archive_path)

        return None

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
            with tarfile.open(output_path, "w:gz", compresslevel=9) as tar:
                for source_path, archive_name in all_paths:
                    source = Path(source_path)
                    if source.exists():
                        if source.is_dir():
                            def _filter(ti):
                                if ti.name.endswith(('.tmp', '.pyc')) or '__pycache__' in ti.name:
                                    return None
                                return ti
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
        # Phase E.2.4: ~200MB tarball + gzip-9 compression — wrap to_thread.
        import asyncio as _asyncio_local
        archive_path = await _asyncio_local.to_thread(self.create_soul_package)
        if not archive_path or not os.path.exists(archive_path):
            return None

        try:
            archive_hash = self.calculate_hash(archive_path)
            size_mb = os.path.getsize(archive_path) / (1024 * 1024)

            from titan_plugin.utils.arweave_store import ArweaveStore
            store = ArweaveStore(
                keypair_path=getattr(self.network, '_wallet_path', None) or getattr(self.network, '_keypair_path', None),
                network=network,
            )

            tx_id = await store.upload_file(
                archive_path,
                tags={
                    "Type": "Titan-Soul-Package",
                    "Archive-Hash": archive_hash[:16],
                    "Size-MB": f"{size_mb:.1f}",
                    "Timestamp": str(int(time.time())),
                },
                content_type="application/gzip",
            )

            if tx_id:
                result = {
                    "arweave_tx": tx_id,
                    "archive_hash": archive_hash,
                    "size_mb": round(size_mb, 2),
                    "permanent_url": store.get_permanent_url(tx_id),
                    "uploaded_at": time.time(),
                }
                self._store_backup_record("soul_package", result)
                logger.info("[Backup] Soul package uploaded to Arweave: %s (%.1fMB)",
                            tx_id[:16] if not tx_id.startswith("devnet") else tx_id,
                            size_mb)
                return result

        except Exception as e:
            logger.error("[Backup] Soul package Arweave upload failed: %s", e)
        finally:
            with suppress(FileNotFoundError):
                os.remove(archive_path)

        return None

    # -------------------------------------------------------------------------
    # Backup Records (local verification)
    # -------------------------------------------------------------------------
    def _store_backup_record(self, backup_type: str, record: dict):
        """Store backup record locally for verification queries."""
        import json
        record_dir = Path("data/backup_records")
        record_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        filepath = record_dir / f"{backup_type}_{ts}.json"
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)

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
                logger.debug("[Backup] Telegram alert failed (non-critical): %s", e)
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

    async def restore_personality_from_arweave(self, tx_id: str) -> dict:
        """Download personality archive from Arweave and restore it.

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

            archive_path = f"/tmp/titan_restore_{int(time.time())}.tar.gz"
            data = await store.fetch(tx_id)
            if not data:
                return {"success": False, "error": f"Failed to fetch from Arweave: {tx_id}"}

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
    # -------------------------------------------------------------------------

    async def anchor_backup_hash(self, archive_hash: str, size_mb: float,
                                  backup_type: str = "personality") -> Optional[str]:
        """Inscribe backup hash as Solana memo for tamper-proof verification.

        Format: TITAN|BACKUP|date=YYYY-MM-DD|h=HASH[:16]|size=NNmb|type=TYPE
        Returns TX signature on success, None on failure.
        """
        if not self.network or not hasattr(self.network, 'send_sovereign_transaction'):
            return None

        try:
            from titan_plugin.utils.solana_client import build_memo_instruction, is_available
            if not is_available() or self.network.keypair is None:
                return None

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            memo_text = (
                f"TITAN|BACKUP|date={today}|h={archive_hash[:16]}"
                f"|size={size_mb:.0f}mb|type={backup_type}"
            )

            memo_ix = build_memo_instruction(self.network.pubkey, memo_text)
            sig = await self.network.send_sovereign_transaction(
                [memo_ix], priority="LOW"
            )

            if sig:
                logger.info("[Backup] Backup hash anchored on-chain: %s (hash=%s...)",
                            sig[:20] if len(sig) > 20 else sig, archive_hash[:12])
            return sig

        except Exception as e:
            logger.warning("[Backup] Backup hash anchor failed (non-critical): %s", e)
            return None
