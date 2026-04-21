"""
Reusable 10-step failsafe cascade (rFP_backup_worker §5.3).

Used by:
  - RebirthBackup (personality + soul uploads)
  - TimeChainBackup (TimeChain tarball uploads)

Cascade steps:
  S1  Build tarball (caller-provided path)
  S2  Validate via extract-test
  S3  Save local copy — IRREDUCIBLE safety net (always runs)
  I5  Diff audit (observability; doesn't block)
  S4  Check Irys balance (mainnet_arweave mode only)
  S5  Upload via caller-provided async fn
  S6  Verify upload via gateway HEAD (propagation-tolerant retry)
  S7  Manifest update — caller's upload_fn owns this
  S8  Anchor on-chain — caller owns this (RebirthBackup.anchor_backup_hash)
  S9  Emit BACKUP_SUCCEEDED — backup_worker emits
  S10 Cleanup old local snapshots older than retention_days
"""

import base64
import glob
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from contextlib import suppress
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class BackupCascade:
    """Failsafe cascade orchestrator — single source of truth for rFP §5.3."""

    def __init__(self, full_config: Optional[dict] = None,
                 arweave_store=None,
                 local_dir: str = "data/backups"):
        self.full_config = full_config or {}
        self.arweave_store = arweave_store
        self.local_dir = local_dir

    # ── Mode detection ──────────────────────────────────────────────────

    def is_local_only(self) -> bool:
        """rFP Phase 3 — [backup].mode == 'local_only' (explicit) or fallback."""
        try:
            mode = (self.full_config.get("backup", {})
                    .get("mode", "") or "").strip().lower()
            if mode == "local_only":
                return True
            if mode == "mainnet_arweave":
                return False
            # Fallback: no arweave store → local only
            return self.arweave_store is None
        except Exception:
            return self.arweave_store is None

    # ── Encryption (Phase 7 — opt-in, between S2 and S3) ────────────────

    @staticmethod
    def _build_encryption_manifest(plaintext: bytes, ciphertext: bytes,
                                     iv: bytes, tag: bytes,
                                     backup_id: str, backup_type: str,
                                     tier: str) -> dict:
        from titan_plugin.logic.backup_crypto import ALGORITHM_ID, key_id
        return {
            "algorithm": ALGORITHM_ID,
            "key_id": key_id(backup_id, backup_type),
            "tier": tier,
            "iv_b64": base64.b64encode(iv).decode("ascii"),
            "auth_tag_b64": base64.b64encode(tag).decode("ascii"),
            "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
            "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
            "backup_id": backup_id,
        }

    def _encrypt_file(self, plaintext_path: str, backup_type: str,
                       encryption: dict) -> Optional[tuple]:
        """Read plaintext tarball, encrypt, write {path}.enc. Returns (enc_path, manifest)."""
        try:
            from titan_plugin.logic.backup_crypto import derive_backup_key, encrypt_tarball
            with open(plaintext_path, "rb") as f:
                plaintext = f.read()
            plaintext_sha = hashlib.sha256(plaintext).hexdigest()
            backup_id = plaintext_sha[:16]
            bkey = derive_backup_key(
                encryption["master_key"], backup_id, backup_type)
            ct, iv, tag = encrypt_tarball(plaintext, bkey)
            enc_path = plaintext_path + ".enc"
            tmp = enc_path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(ct)
            os.replace(tmp, enc_path)
            manifest = self._build_encryption_manifest(
                plaintext, ct, iv, tag, backup_id, backup_type,
                encryption.get("tier", "private"))
            logger.info(
                "[Cascade] Encrypted %s: %d→%d bytes (tier=%s key=%s)",
                backup_type, len(plaintext), len(ct), manifest["tier"],
                manifest["key_id"])
            return enc_path, manifest
        except Exception as e:
            logger.error("[Cascade] Encryption FAIL (%s): %s",
                         backup_type, e, exc_info=True)
            return None

    def _encrypt_bytes(self, plaintext: bytes, backup_type: str,
                        encryption: dict) -> Optional[tuple]:
        """Encrypt in-memory plaintext. Returns (ciphertext_bytes, manifest)."""
        try:
            from titan_plugin.logic.backup_crypto import derive_backup_key, encrypt_tarball
            plaintext_sha = hashlib.sha256(plaintext).hexdigest()
            backup_id = plaintext_sha[:16]
            bkey = derive_backup_key(
                encryption["master_key"], backup_id, backup_type)
            ct, iv, tag = encrypt_tarball(plaintext, bkey)
            manifest = self._build_encryption_manifest(
                plaintext, ct, iv, tag, backup_id, backup_type,
                encryption.get("tier", "private"))
            logger.info(
                "[Cascade] Encrypted %s bytes: %d→%d (tier=%s key=%s)",
                backup_type, len(plaintext), len(ct), manifest["tier"],
                manifest["key_id"])
            return ct, manifest
        except Exception as e:
            logger.error("[Cascade] Encryption FAIL (%s): %s",
                         backup_type, e, exc_info=True)
            return None

    @staticmethod
    def _cleanup_tmp_encrypted(plaintext_path: str, encrypted_path: str) -> None:
        """Remove the transient .enc sibling created by _encrypt_file if the
        plaintext source was in /tmp (caller will remove the plaintext itself).
        """
        if (encrypted_path != plaintext_path
                and encrypted_path.endswith(".enc")
                and encrypted_path.startswith("/tmp/")):
            with suppress(FileNotFoundError):
                os.remove(encrypted_path)

    # ── Cascade step helpers ────────────────────────────────────────────

    @staticmethod
    def validate_tarball(archive_path: str) -> bool:
        """S2 — extract-test tarball to catch silent corruption.

        Supports tar.gz (native tarfile) and tar.zst (via zstandard stream reader).
        Python's tarfile module doesn't recognize zstd as a compression format,
        so we decompress manually and feed raw tar bytes through tarfile.
        """
        try:
            count = 0
            if archive_path.endswith(".tar.zst") or archive_path.endswith(".zst"):
                # TimeChain uses zstd-19 — decompress stream → native tar reader
                import zstandard
                from io import BytesIO
                dctx = zstandard.ZstdDecompressor()
                with open(archive_path, "rb") as f_in:
                    raw_tar = dctx.stream_reader(f_in).read()
                with tarfile.open(fileobj=BytesIO(raw_tar), mode="r|") as tf:
                    for _m in tf:
                        count += 1
            else:
                # tar.gz (personality + soul) — native auto-detect
                with tarfile.open(archive_path, "r:*") as tf:
                    for _m in tf:
                        count += 1
            logger.debug("[Cascade] S2 validate OK: %s (%d members)",
                         os.path.basename(archive_path), count)
            return count > 0
        except Exception as e:
            logger.error("[Cascade] S2 validate FAIL: %s (%s)",
                         os.path.basename(archive_path), e)
            return False

    def save_local(self, archive_path: str, backup_type: str,
                    archive_hash: str) -> Optional[str]:
        """S3 — always save local copy (irreducible safety net)."""
        try:
            os.makedirs(self.local_dir, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y%m%d")
            # Determine extension from source
            if archive_path.endswith(".tar.zst"):
                ext = "tar.zst"
            elif archive_path.endswith(".tar.gz"):
                ext = "tar.gz"
            else:
                ext = "tar.gz"  # default
            local_name = f"{backup_type}_{today}_{archive_hash[:12]}.{ext}"
            local_path = os.path.join(self.local_dir, local_name)
            if os.path.exists(local_path):
                logger.debug("[Cascade] S3 local exists: %s", local_path)
                return local_path
            # If archive_path is in-memory bytes, caller must write it first.
            # Here we assume it's on disk.
            if os.path.isfile(archive_path):
                shutil.copy2(archive_path, local_path)
            else:
                logger.warning("[Cascade] S3 source missing on disk: %s",
                               archive_path)
                return None
            sz_mb = os.path.getsize(local_path) / 1024 / 1024
            logger.info("[Cascade] S3 local saved: %s (%.1f MB)",
                        os.path.basename(local_path), sz_mb)
            return local_path
        except Exception as e:
            logger.error("[Cascade] S3 local save FAIL: %s", e)
            return None

    def save_local_bytes(self, data: bytes, backup_type: str,
                          archive_hash: str, ext: str = "tar.zst"
                          ) -> Optional[str]:
        """S3 variant for in-memory byte buffers (TimeChain uses this)."""
        try:
            os.makedirs(self.local_dir, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y%m%d")
            local_name = f"{backup_type}_{today}_{archive_hash[:12]}.{ext}"
            local_path = os.path.join(self.local_dir, local_name)
            if os.path.exists(local_path):
                logger.debug("[Cascade] S3 local exists: %s", local_path)
                return local_path
            tmp = local_path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, local_path)
            sz_mb = len(data) / 1024 / 1024
            logger.info("[Cascade] S3 local saved: %s (%.1f MB)",
                        os.path.basename(local_path), sz_mb)
            return local_path
        except Exception as e:
            logger.error("[Cascade] S3 local save FAIL: %s", e)
            return None

    def check_irys_balance(self, size_mb: float) -> tuple[bool, float, str]:
        """S4 — query Irys deposit, return (sufficient?, sol, reason)."""
        try:
            keypair = self.full_config.get("network", {}).get(
                "wallet_keypair_path", "")
            if not keypair or not os.path.exists(keypair):
                return False, 0.0, "no_keypair"
            out = subprocess.check_output(
                ["node", "scripts/irys_upload.js", "balance", keypair,
                 "https://api.mainnet-beta.solana.com"],
                env={**os.environ, "NODE_PATH": _node_path()},
                timeout=30,
            )
            data = json.loads(out.decode())
            if data.get("status") != "ok":
                return False, 0.0, str(data.get("message", "unknown"))
            sol = float(data.get("balance_readable", 0))
            estimated_cost = size_mb * 0.0002
            required = estimated_cost * 2.0  # 2× safety margin
            ok = sol >= required
            reason = f"need={required:.6f} have={sol:.6f} size={size_mb:.1f}MB"
            return ok, sol, reason
        except Exception as e:
            return False, 0.0, f"balance_query_failed: {e}"

    @staticmethod
    def verify_upload(tx_id: str, expected_sha256: str, size_mb: float) -> bool:
        """S6 — propagation-tolerant HEAD verify (retry for Arweave mining)."""
        if not tx_id:
            return False
        if isinstance(tx_id, str) and tx_id.startswith("devnet"):
            return False  # devnet pseudo-tx never retrievable
        url = f"https://arweave.net/{tx_id}"
        delays = [15, 30, 45]  # 15s + 45s + 90s = ~2.5 min retry window
        for attempt, delay in enumerate(delays):
            time.sleep(delay)
            try:
                if size_mb < 50:
                    # Small: full fetch + hash compare
                    with urllib.request.urlopen(url, timeout=60) as resp:
                        data = resp.read()
                    actual = hashlib.sha256(data).hexdigest()
                    if actual == expected_sha256:
                        logger.info(
                            "[Cascade] S6 verify OK (hash match) tx=%s attempt=%d",
                            tx_id[:16], attempt + 1)
                        return True
                    logger.warning(
                        "[Cascade] S6 HASH MISMATCH tx=%s expected=%s actual=%s",
                        tx_id[:16], expected_sha256[:16], actual[:16])
                    return False
                # Large: HEAD-only reachability check
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=15) as resp:
                    if 200 <= resp.status < 300:
                        logger.info(
                            "[Cascade] S6 verify OK (HEAD 200) tx=%s attempt=%d",
                            tx_id[:16], attempt + 1)
                        return True
            except Exception as e:
                logger.debug("[Cascade] S6 attempt %d: %s", attempt + 1, e)
        logger.warning(
            "[Cascade] S6 verify FAIL — tx=%s not reachable after %.0fs "
            "(may still propagate)",
            tx_id[:16], sum(delays))
        return False

    def cleanup_local(self, backup_type: str, retention_days: int = 30) -> int:
        """S10 — remove local snapshots older than retention_days."""
        try:
            patterns = [
                os.path.join(self.local_dir, f"{backup_type}_*.tar.gz"),
                os.path.join(self.local_dir, f"{backup_type}_*.tar.zst"),
            ]
            cutoff = time.time() - (retention_days * 86400)
            deleted = 0
            for pattern in patterns:
                for path in glob.glob(pattern):
                    with suppress(OSError):
                        if os.path.getmtime(path) < cutoff:
                            os.remove(path)
                            deleted += 1
            if deleted:
                logger.info(
                    "[Cascade] S10 cleanup: removed %d old %s snapshot(s)",
                    deleted, backup_type)
            return deleted
        except Exception as e:
            logger.warning("[Cascade] S10 cleanup error: %s", e)
            return 0

    def diff_audit(self, backup_type: str, current_size_mb: float,
                    get_latest_record_fn: Optional[Callable] = None
                    ) -> Optional[dict]:
        """I5 — sequential-backup diff audit (optional if no record-getter)."""
        if get_latest_record_fn is None:
            return None
        try:
            prev = get_latest_record_fn(backup_type)
            if not prev:
                return None
            prev_size = float(prev.get("size_mb", 0))
            if prev_size > 0:
                delta_pct = abs(current_size_mb - prev_size) / prev_size
                if delta_pct > 0.30:
                    severity = ("ERROR" if current_size_mb < prev_size * 0.5
                                else "WARNING")
                    return {
                        "severity": severity,
                        "type": "size_delta",
                        "backup_type": backup_type,
                        "current_mb": round(current_size_mb, 2),
                        "previous_mb": round(prev_size, 2),
                        "delta_pct": round(delta_pct * 100, 1),
                    }
            return None
        except Exception as e:
            logger.debug("[Cascade] Diff audit error: %s", e)
            return None

    # ── Orchestrator ────────────────────────────────────────────────────

    async def run(
        self,
        archive_path: str,
        backup_type: str,
        upload_fn: Callable,
        get_latest_record_fn: Optional[Callable] = None,
        retention_days: int = 30,
        encryption: Optional[dict] = None,
    ) -> Optional[dict]:
        """Execute cascade around a file-based upload function.

        Args:
            archive_path: tarball path on disk (from S1 build)
            backup_type: "personality" | "soul" | "timechain"
            upload_fn: async callable (archive_path) -> dict | None with 'arweave_tx'
            get_latest_record_fn: optional (backup_type) -> dict for diff audit
            retention_days: S10 cleanup threshold
            encryption: optional {master_key: bytes, tier: str}. When provided,
                the plaintext tarball is AES-256-GCM encrypted between S2 validate
                and S3 local save; S3/S5/S6/S8 operate on the encrypted tarball.
        """
        if not archive_path or not os.path.exists(archive_path):
            logger.error("[Cascade] S1 FAIL — no archive for %s", backup_type)
            return None

        archive_hash = _hash_file(archive_path)
        size_mb = os.path.getsize(archive_path) / (1024 * 1024)

        # S2 validate (on plaintext — catches corruption BEFORE encryption locks in bad data)
        if not self.validate_tarball(archive_path):
            return {
                "cascade_fail": "S2_validate",
                "archive_hash": archive_hash,
                "size_mb": round(size_mb, 2),
                "backup_type": backup_type,
            }

        # Phase 7 — encrypt (opt-in). If encrypted, swap to .enc for all downstream.
        encryption_manifest: Optional[dict] = None
        _plaintext_path = archive_path  # keep reference for cleanup
        if encryption and encryption.get("master_key"):
            enc = self._encrypt_file(archive_path, backup_type, encryption)
            if enc is None:
                return {
                    "cascade_fail": "encryption",
                    "archive_hash": archive_hash,
                    "size_mb": round(size_mb, 2),
                    "backup_type": backup_type,
                }
            archive_path, encryption_manifest = enc
            archive_hash = _hash_file(archive_path)
            size_mb = os.path.getsize(archive_path) / (1024 * 1024)

        # S3 local-always
        local_path = self.save_local(archive_path, backup_type, archive_hash)

        # I5 diff audit (non-blocking)
        diff_alert = self.diff_audit(backup_type, size_mb, get_latest_record_fn)

        # Mode gate: local_only skips S4-S9
        if self.is_local_only():
            self.cleanup_local(backup_type, retention_days)
            self._cleanup_tmp_encrypted(_plaintext_path, archive_path)
            return {
                "mode": "local_only",
                "local_path": local_path,
                "archive_hash": archive_hash,
                "size_mb": round(size_mb, 2),
                "uploaded_at": time.time(),
                "backup_type": backup_type,
                **({"encryption": encryption_manifest} if encryption_manifest else {}),
                **({"diff_alert": diff_alert} if diff_alert else {}),
            }

        # S4 balance
        sufficient, irys_sol, reason = self.check_irys_balance(size_mb)
        if not sufficient:
            logger.warning(
                "[Cascade] S4 FAIL — Irys balance insufficient (%s)", reason)
            self.cleanup_local(backup_type, retention_days)
            self._cleanup_tmp_encrypted(_plaintext_path, archive_path)
            return {
                "mode": "local_fallback_low_balance",
                "local_path": local_path,
                "archive_hash": archive_hash,
                "size_mb": round(size_mb, 2),
                "irys_sol": round(irys_sol, 6),
                "reason": reason,
                "uploaded_at": time.time(),
                "backup_type": backup_type,
                **({"encryption": encryption_manifest} if encryption_manifest else {}),
                **({"diff_alert": diff_alert} if diff_alert else {}),
            }

        # S5 upload
        try:
            result = await upload_fn(archive_path)
        except Exception as e:
            logger.error("[Cascade] S5 upload error: %s", e, exc_info=True)
            self._cleanup_tmp_encrypted(_plaintext_path, archive_path)
            return {
                "cascade_fail": "S5_upload",
                "error": str(e),
                "local_path": local_path,
                "archive_hash": archive_hash,
                "size_mb": round(size_mb, 2),
                "backup_type": backup_type,
                **({"encryption": encryption_manifest} if encryption_manifest else {}),
            }

        if not result or not result.get("arweave_tx"):
            self._cleanup_tmp_encrypted(_plaintext_path, archive_path)
            return {
                "cascade_fail": "S5_upload_none",
                "local_path": local_path,
                "archive_hash": archive_hash,
                "size_mb": round(size_mb, 2),
                "backup_type": backup_type,
                **({"encryption": encryption_manifest} if encryption_manifest else {}),
            }

        tx_id = result["arweave_tx"]

        # S6 verify (devnet skipped automatically) — on the BYTES that were uploaded
        if isinstance(tx_id, str) and not tx_id.startswith("devnet"):
            result["verified"] = self.verify_upload(tx_id, archive_hash, size_mb)
        else:
            result["verified"] = "skipped_devnet"

        # S10 cleanup
        self.cleanup_local(backup_type, retention_days)
        self._cleanup_tmp_encrypted(_plaintext_path, archive_path)

        result["local_path"] = local_path
        result["backup_type"] = backup_type
        result["archive_hash"] = archive_hash  # ensure caller sees encrypted hash
        result["size_mb"] = round(size_mb, 2)
        if encryption_manifest:
            result["encryption"] = encryption_manifest
        if diff_alert:
            result["diff_alert"] = diff_alert
        return result

    async def run_bytes(
        self,
        tarball_bytes: bytes,
        archive_hash: str,
        backup_type: str,
        upload_fn: Callable,
        get_latest_record_fn: Optional[Callable] = None,
        retention_days: int = 30,
        ext: str = "tar.zst",
        encryption: Optional[dict] = None,
    ) -> Optional[dict]:
        """Cascade variant for in-memory byte buffers (TimeChain tarball).

        TimeChainBackup.create_snapshot_tarball returns bytes directly (the
        tarball is built in-memory via BytesIO). Rather than writing to /tmp
        just to let `run()` read it back, this variant operates directly on
        bytes + hash — we persist to S3 local path, then upload from there.

        Phase 7 — if encryption dict is provided, plaintext is AES-256-GCM
        encrypted between S2 validate and S3; S3/S5/S6/S8 operate on ciphertext.
        """
        size_mb = len(tarball_bytes) / (1024 * 1024)

        # S2 validate on plaintext — write to tmp to extract-test
        tmp_path = os.path.join("/tmp", f"cascade_{backup_type}_{int(time.time())}.{ext}")
        try:
            with open(tmp_path, "wb") as f:
                f.write(tarball_bytes)
            if not self.validate_tarball(tmp_path):
                return {
                    "cascade_fail": "S2_validate",
                    "archive_hash": archive_hash,
                    "size_mb": round(size_mb, 2),
                    "backup_type": backup_type,
                }

            # Phase 7 — encrypt (opt-in). If encrypted, swap payload + hash + ext.
            encryption_manifest: Optional[dict] = None
            if encryption and encryption.get("master_key"):
                enc = self._encrypt_bytes(tarball_bytes, backup_type, encryption)
                if enc is None:
                    return {
                        "cascade_fail": "encryption",
                        "archive_hash": archive_hash,
                        "size_mb": round(size_mb, 2),
                        "backup_type": backup_type,
                    }
                tarball_bytes, encryption_manifest = enc
                archive_hash = encryption_manifest["ciphertext_sha256"]
                size_mb = len(tarball_bytes) / (1024 * 1024)
                ext = ext + ".enc"

            # S3 local save (persists encrypted bytes if encryption enabled)
            local_path = self.save_local_bytes(
                tarball_bytes, backup_type, archive_hash, ext=ext)

            diff_alert = self.diff_audit(backup_type, size_mb, get_latest_record_fn)

            if self.is_local_only():
                self.cleanup_local(backup_type, retention_days)
                return {
                    "mode": "local_only",
                    "local_path": local_path,
                    "archive_hash": archive_hash,
                    "size_mb": round(size_mb, 2),
                    "uploaded_at": time.time(),
                    "backup_type": backup_type,
                    **({"encryption": encryption_manifest} if encryption_manifest else {}),
                    **({"diff_alert": diff_alert} if diff_alert else {}),
                }

            sufficient, irys_sol, reason = self.check_irys_balance(size_mb)
            if not sufficient:
                logger.warning(
                    "[Cascade] S4 FAIL — Irys balance insufficient (%s)", reason)
                self.cleanup_local(backup_type, retention_days)
                return {
                    "mode": "local_fallback_low_balance",
                    "local_path": local_path,
                    "archive_hash": archive_hash,
                    "size_mb": round(size_mb, 2),
                    "irys_sol": round(irys_sol, 6),
                    "reason": reason,
                    "uploaded_at": time.time(),
                    "backup_type": backup_type,
                    **({"encryption": encryption_manifest} if encryption_manifest else {}),
                    **({"diff_alert": diff_alert} if diff_alert else {}),
                }

            try:
                # upload_fn receives bytes (not path) for this variant
                result = await upload_fn(tarball_bytes)
            except Exception as e:
                logger.error("[Cascade] S5 upload error: %s", e, exc_info=True)
                return {
                    "cascade_fail": "S5_upload",
                    "error": str(e),
                    "local_path": local_path,
                    "archive_hash": archive_hash,
                    "size_mb": round(size_mb, 2),
                    "backup_type": backup_type,
                    **({"encryption": encryption_manifest} if encryption_manifest else {}),
                }

            if not result or not result.get("arweave_tx"):
                return {
                    "cascade_fail": "S5_upload_none",
                    "local_path": local_path,
                    "archive_hash": archive_hash,
                    "size_mb": round(size_mb, 2),
                    "backup_type": backup_type,
                    **({"encryption": encryption_manifest} if encryption_manifest else {}),
                }

            tx_id = result["arweave_tx"]
            if isinstance(tx_id, str) and not tx_id.startswith("devnet"):
                result["verified"] = self.verify_upload(tx_id, archive_hash, size_mb)
            else:
                result["verified"] = "skipped_devnet"

            self.cleanup_local(backup_type, retention_days)
            result["local_path"] = local_path
            result["backup_type"] = backup_type
            result["archive_hash"] = archive_hash
            result["size_mb"] = round(size_mb, 2)
            if encryption_manifest:
                result["encryption"] = encryption_manifest
            if diff_alert:
                result["diff_alert"] = diff_alert
            return result
        finally:
            with suppress(FileNotFoundError):
                os.remove(tmp_path)


# ── Module helpers ──────────────────────────────────────────────────────

def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _node_path() -> str:
    try:
        return subprocess.check_output(
            ["npm", "root", "-g"], timeout=10
        ).decode().strip()
    except Exception:
        return "/usr/lib/node_modules"
