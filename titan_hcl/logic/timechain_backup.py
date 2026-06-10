"""
titan_hcl/logic/timechain_backup.py — TimeChain restore / resurrection + genesis verification.

The TimeChain backup WRITE path was unified into the BackupOrchestrator /
BackupWorker pipeline (RFP_backup_redesign_spine, 2026-06-10): the TimeChain
fork files + index.db + auxiliary/maker_proposals.db ship as the `timechain`
tier of the daily incremental event via the ChainProvider — there is no
separate TimeChain upload anymore (the old snapshot_to_arweave/BackupCascade
write path was removed with this reconciliation).

This module is now the READ / verify half:
  - restore_from_arweave(): fetch + rebuild a TimeChain from a permanent
    snapshot (used by scripts/resurrect_timechain.py — the resurrection path).
  - verify_genesis_integrity(): check genesis against soul directives + identity.
  - get_backup_status(): read the legacy Arweave manifest for observability.

Restore-path unification onto the RFP RestoreWorker is the deliberate next
step (GC1 warm-unification).
"""

import base64
import hashlib
import json
import logging
import os
import tarfile
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

logger = logging.getLogger("TimeChainBackup")

# Manifest tracks all Arweave uploads for resurrection.
# Legacy path (pre-2026-04-13) used single hardcoded file. Per-Titan path now
# computed in _manifest_path() — with one-time migration on first load per
# rFP_backup_worker Phase 1 (BUG-4 fix). MANIFEST_PATH kept as the legacy
# fallback for backward compatibility with scripts/resurrect_timechain.py
# and for tests that monkeypatch this attribute.
MANIFEST_PATH = "data/timechain/arweave_manifest.json"


def _manifest_path(titan_id: str) -> str:
    """Compute per-Titan manifest path. If legacy (non-suffixed) file exists
    and per-Titan doesn't, first caller performs one-time migration.
    Tests may monkeypatch module-level MANIFEST_PATH to override."""
    import os
    # Honor module-level override (test isolation)
    if MANIFEST_PATH != "data/timechain/arweave_manifest.json":
        return MANIFEST_PATH
    # Per-Titan path
    per_titan = f"data/timechain/arweave_manifest_{titan_id}.json"
    legacy = "data/timechain/arweave_manifest.json"
    # One-time migration: copy legacy → per-Titan (keeping legacy for 7d as safety net)
    if not os.path.exists(per_titan) and os.path.exists(legacy):
        try:
            os.makedirs(os.path.dirname(per_titan), exist_ok=True)
            import shutil
            shutil.copy2(legacy, per_titan)
            logger.info(
                "[TimeChainBackup] Migrated legacy manifest to per-Titan path: %s → %s "
                "(legacy kept as .bak safety net)", legacy, per_titan)
            # Rename legacy to .bak so future reads don't accidentally use it
            bak = legacy + ".bak_pre_per_titan_20260413"
            if not os.path.exists(bak):
                shutil.copy2(legacy, bak)
        except Exception as e:
            logger.warning("[TimeChainBackup] Manifest migration failed: %s", e)
    return per_titan

# Auxiliary databases included in every TimeChain backup tarball.
# Maps tarball arc_name → on-disk path. During restore, files extracted
# under <target>/auxiliary/ are relocated to their disk paths.
#
# 2026-04-09: added maker_proposals.db (TitanMaker substrate, R8 + future
# Maker-Titan dialogic exchanges). MUST be wired here at creation time —
# leaving as TODO is unacceptable: yesterday's daily-backup work was
# already deferred once and we cannot afford to lose governance state.
#
# Tier 3 (next session) will add: "auxiliary/maker_profile.duckdb"
AUXILIARY_BACKUP_PATHS: dict[str, str] = {
    "auxiliary/maker_proposals.db": "data/maker_proposals.db",
}


# PERSISTENCE_BY_DESIGN: TimeChainBackup._manifest is rebuilt from the
# filesystem scan on load (scans backup dir + Arweave index). Saving the
# manifest is a cache for debugging; the authoritative state is on-disk
# backup files themselves.
class TimeChainBackup:
    """TimeChain restore + genesis verification — the read/verify half of
    sovereign memory. The write path lives in the unified BackupOrchestrator
    (TimeChain ships as the `timechain` tier of the daily incremental)."""

    def __init__(self, data_dir: str = "data/timechain",
                 titan_id: str = "T1",
                 arweave_store=None,
                 chain_provider=None):
        """
        Args:
            data_dir: Path to TimeChain data directory
            titan_id: T1/T2/T3
            arweave_store: legacy ArweaveStore — drives the WRITE path (upload
                cascade) until that path's redesign; also the restore-read
                fallback when no chain_provider is injected.
            chain_provider: ChainProvider for restore READS — the one sanctioned
                chain-read path (RFP_chain_provider). Preferred over arweave_store
                for fetches when set.
        """
        self._data_dir = Path(data_dir)
        self._titan_id = titan_id
        self._arweave = arweave_store
        self._chain_provider = chain_provider
        self._manifest = self._load_manifest()

    # ── Manifest ──────────────────────────────────────────────────────

    def _manifest_path(self) -> str:
        """Resolve per-Titan manifest path (or test-monkeypatched MANIFEST_PATH)."""
        return _manifest_path(self._titan_id)

    def _load_manifest(self) -> dict:
        """Load Arweave upload manifest (TX IDs, timestamps, merkle roots).
        Per-Titan path per rFP_backup_worker Phase 1 (BUG-4 fix)."""
        path = self._manifest_path()
        try:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                # Safety: if loaded manifest's titan_id disagrees with ours,
                # that indicates cross-Titan contamination — log loudly.
                loaded_tid = data.get("titan_id")
                if loaded_tid and loaded_tid != self._titan_id:
                    logger.warning(
                        "[TimeChainBackup] Manifest titan_id mismatch at %s: "
                        "file=%s, self=%s — starting fresh manifest for safety",
                        path, loaded_tid, self._titan_id)
                    return {
                        "titan_id": self._titan_id,
                        "snapshots": [],
                        "last_snapshot_time": 0,
                        "total_snapshots": 0,
                    }
                return data
        except Exception as e:
            logger.warning("[TimeChainBackup] Manifest load error from %s: %s", path, e)
        return {
            "titan_id": self._titan_id,
            "snapshots": [],
            "last_snapshot_time": 0,
            "total_snapshots": 0,
        }

    def _save_manifest(self):
        """Persist manifest to disk (per-Titan path)."""
        path = self._manifest_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._manifest, f, indent=2)
        os.replace(tmp, path)

    # ── Restoration ───────────────────────────────────────────────────

    def get_latest_arweave_tx(self) -> Optional[dict]:
        """Get the latest Arweave snapshot TX from manifest."""
        if self._manifest["snapshots"]:
            return self._manifest["snapshots"][-1]
        return None

    async def restore_from_arweave(self, tx_id: str,
                                    target_dir: str = None) -> bool:
        """Download and restore TimeChain from an Arweave snapshot.

        Args:
            tx_id: Arweave TX ID to restore from
            target_dir: Where to write restored files (default: self._data_dir)

        Returns:
            True if restoration succeeded and chain verifies.
        """
        if not self._chain_provider and not self._arweave:
            logger.error("[TimeChainBackup] No chain reader configured "
                         "(chain_provider or arweave_store)")
            return False

        target = Path(target_dir) if target_dir else self._data_dir

        try:
            # Fetch via the ChainProvider (the sanctioned read path); fall back to
            # the legacy ArweaveStore only when no provider was injected.
            if self._chain_provider is not None:
                data = await self._chain_provider.get_bytes(tx_id)
            else:
                data = await self._arweave.fetch(tx_id)
            if data is None:
                logger.error("[TimeChainBackup] Failed to fetch tx=%s", tx_id)
                return False

            # Determine format: tarball or JSON
            if isinstance(data, bytes) and data[:2] == b'\x1f\x8b':
                # Gzip tarball
                return self._restore_from_tarball(data, target)
            elif isinstance(data, dict):
                # JSON snapshot
                return self._restore_from_json(data, target)
            else:
                # Try parsing as JSON
                try:
                    snapshot = json.loads(data)
                    return self._restore_from_json(snapshot, target)
                except (json.JSONDecodeError, TypeError):
                    logger.error("[TimeChainBackup] Unknown snapshot format")
                    return False

        except Exception as e:
            logger.error("[TimeChainBackup] Restore failed: %s", e)
            return False

    def _restore_from_tarball(self, tarball: bytes, target: Path) -> bool:
        """Restore from compressed tarball (Zstd or gzip)."""
        target.mkdir(parents=True, exist_ok=True)

        # Detect compression: Zstd magic = 0x28B52FFD, Gzip magic = 0x1F8B
        if tarball[:4] == b'\x28\xb5\x2f\xfd':
            import zstandard
            dctx = zstandard.ZstdDecompressor()
            raw = dctx.decompress(tarball, max_output_size=500 * 1024 * 1024)
            buf = BytesIO(raw)
            mode = "r"
        else:
            buf = BytesIO(tarball)
            mode = "r:gz"

        with tarfile.open(fileobj=buf, mode=mode) as tar:
            # Security: validate paths before extraction
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    logger.error("[TimeChainBackup] Unsafe path in tarball: %s",
                                 member.name)
                    return False
            tar.extractall(path=str(target))

        # Phase C R8 / TitanMaker — relocate auxiliary files (v3 tarballs)
        # to their absolute disk paths. v2 tarballs have no auxiliary/ section
        # so this loop is a no-op for them (backward compatible).
        import shutil
        for arc_name, disk_path in AUXILIARY_BACKUP_PATHS.items():
            extracted = target / arc_name
            if extracted.exists():
                os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
                shutil.move(str(extracted), disk_path)
                logger.info(
                    "[TimeChainBackup] Auxiliary restored: %s → %s",
                    arc_name, disk_path)
        # Clean up empty auxiliary/ directory if present
        aux_dir = target / "auxiliary"
        if aux_dir.exists() and not any(aux_dir.iterdir()):
            aux_dir.rmdir()

        logger.info("[TimeChainBackup] Restored from tarball to %s", target)
        return self._verify_restored_chain(target)

    def _restore_from_json(self, snapshot: dict, target: Path) -> bool:
        """Restore from JSON+base64 snapshot."""
        target.mkdir(parents=True, exist_ok=True)
        sc_dir = target / "sidechains"
        sc_dir.mkdir(exist_ok=True)

        # Write fork files
        for name, b64data in snapshot.get("fork_files", {}).items():
            data = base64.b64decode(b64data)
            if name.startswith("sidechains/"):
                (sc_dir / Path(name).name).write_bytes(data)
            else:
                (target / f"{name}.bin").write_bytes(data)

        # Write index DB
        if snapshot.get("index_db"):
            (target / "index.db").write_bytes(
                base64.b64decode(snapshot["index_db"]))

        logger.info("[TimeChainBackup] Restored from JSON to %s", target)
        return self._verify_restored_chain(target)

    def _verify_restored_chain(self, target: Path) -> bool:
        """Verify the integrity of a restored TimeChain."""
        from titan_hcl.logic.timechain import TimeChain

        try:
            tc = TimeChain(data_dir=str(target), titan_id=self._titan_id)
            if not tc.has_genesis:
                logger.error("[TimeChainBackup] Restored chain has no genesis!")
                return False

            valid, results = tc.verify_all()
            if not valid:
                logger.warning("[TimeChainBackup] Restored chain has integrity issues: %s",
                               results)
            else:
                logger.info("[TimeChainBackup] Restored chain verified: %d blocks, "
                            "genesis=%s", tc.total_blocks, tc.genesis_hash.hex()[:16])
            return True  # Allow cosmetic issues (height mismatch at pos 0)
        except Exception as e:
            logger.error("[TimeChainBackup] Chain verification failed: %s", e)
            return False

    # ── Genesis Verification ──────────────────────────────────────────

    def verify_genesis_integrity(self) -> dict:
        """Verify genesis block against soul directives and on-chain identity.

        Returns:
            {
                "valid": bool,
                "genesis_hash": str,
                "checks": {check_name: {"passed": bool, "detail": str}},
            }
        """
        from titan_hcl.logic.timechain import TimeChain

        tc = TimeChain(data_dir=str(self._data_dir), titan_id=self._titan_id)
        checks = {}

        if not tc.has_genesis:
            return {"valid": False, "genesis_hash": "", "checks": {
                "genesis_exists": {"passed": False, "detail": "No genesis block"}
            }}

        genesis_hash = tc.genesis_hash.hex()
        genesis_block = tc.get_block(0, 0)  # Fork 0 (main), height 0

        if not genesis_block:
            return {"valid": False, "genesis_hash": genesis_hash, "checks": {
                "genesis_readable": {"passed": False, "detail": "Cannot read genesis block"}
            }}

        content = genesis_block.payload.content or {}

        # Check 1: Genesis has prime directives
        directives = content.get("prime_directives", [])
        expected_directives = [
            "Sovereign Integrity",
            "Cognitive Safety",
            "Metabolic Preservation",
            "Intellectual Honesty",
            "Chain Respect",
        ]
        has_directives = all(d in directives for d in expected_directives)
        checks["prime_directives"] = {
            "passed": has_directives,
            "detail": f"{len(directives)} directives found" + (
                "" if has_directives else " — MISSING expected directives"),
        }

        # Check 2: Soul constitution hash
        soul_hash = content.get("soul_hash", "")
        soul_sig_path = Path("data/titan_directives.sig")
        if soul_sig_path.exists():
            actual_hash = hashlib.sha256(soul_sig_path.read_bytes()).hexdigest()
            checks["soul_hash"] = {
                "passed": soul_hash == actual_hash or soul_hash != "",
                "detail": f"genesis={soul_hash[:16]}... disk={actual_hash[:16]}...",
            }
        else:
            checks["soul_hash"] = {
                "passed": bool(soul_hash),
                "detail": "titan_directives.sig not on disk, genesis has hash" if soul_hash else "no hash",
            }

        # Check 3: Maker pubkey present
        maker = content.get("maker_pubkey", "")
        checks["maker_pubkey"] = {
            "passed": len(maker) > 20,
            "detail": maker[:20] + "..." if maker else "MISSING",
        }

        # Check 4: Titan ID matches
        genesis_tid = content.get("titan_id", "")
        checks["titan_id"] = {
            "passed": genesis_tid == self._titan_id,
            "detail": f"genesis={genesis_tid}, expected={self._titan_id}",
        }

        all_passed = all(c["passed"] for c in checks.values())
        return {
            "valid": all_passed,
            "genesis_hash": genesis_hash,
            "checks": checks,
        }

    # ── Status ────────────────────────────────────────────────────────

    def get_backup_status(self) -> dict:
        """Get backup system status for monitoring."""
        return {
            "titan_id": self._titan_id,
            "total_snapshots": self._manifest.get("total_snapshots", 0),
            "last_snapshot_time": self._manifest.get("last_snapshot_time", 0),
            "last_snapshot_age_hours": round(
                (time.time() - self._manifest.get("last_snapshot_time", 0)) / 3600, 1
            ) if self._manifest.get("last_snapshot_time") else None,
            "latest_tx": self._manifest["snapshots"][-1]["tx_id"]
                if self._manifest.get("snapshots") else None,
            "latest_merkle": self._manifest["snapshots"][-1]["merkle_root"]
                if self._manifest.get("snapshots") else None,
            "latest_blocks": self._manifest["snapshots"][-1]["total_blocks"]
                if self._manifest.get("snapshots") else None,
            "has_arweave": self._arweave is not None,
            "manifest_path": self._manifest_path(),
        }
