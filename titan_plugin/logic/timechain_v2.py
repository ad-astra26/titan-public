"""
TimeChain v2 — Cognitive Memory Engine

Phase 1: Mempool + Genesis Chain + Block Builder (LIVE)
Phase 2: Consumer API — recall/check/compare/aggregate (LIVE)
Phase 3a: Smart Contract Engine — Ed25519 signing + meta fork storage

The existing commit_block() remains the sole write path to chain files.
The orchestrator batches, filters, and aggregates before calling it.

Feature-flagged: [timechain.v2] enabled = true in titan_params.toml

See: titan-docs/rFP_timechain_v2_cognitive_engine.md
"""

import hashlib
import json
import logging
import math
import os
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import msgpack
except ImportError:
    msgpack = None  # type: ignore

logger = logging.getLogger(__name__)

# Fork name → fork_id mapping (matches timechain.py constants)
FORK_IDS = {
    "main": 0, "declarative": 1, "procedural": 2,
    "episodic": 3, "meta": 4, "conversation": 5,
}


# ═══════════════════════════════════════════════════════════════════════
# Transaction — Atomic unit of cognitive memory
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Transaction:
    """A single cognitive event submitted to the TimeChain mempool."""

    tx_type: str            # "episodic", "declarative", "procedural", "meta", "main"
    source: str             # "expression_art", "meditation", "language_teacher", ...
    epoch_id: int
    significance: float
    content: dict
    neuromod_snapshot: dict
    tags: list
    timestamp: float
    fork_name: str          # "episodic", "declarative", ...

    # Original payload preserved for commit_block() compatibility
    original_payload: dict = field(default_factory=dict, repr=False)

    # Computed fields
    tx_hash: bytes = field(default=b"", repr=False)
    mempool_action: str = "buffer"  # "include", "aggregate", "buffer", "drop"

    def compute_hash(self) -> bytes:
        """SHA-256 of canonical JSON serialization."""
        canonical = json.dumps({
            "t": self.tx_type, "s": self.source,
            "e": self.epoch_id, "g": self.significance,
            "c": self.content, "a": self.tags,
            "ts": self.timestamp,
        }, sort_keys=True, separators=(",", ":")).encode()
        self.tx_hash = hashlib.sha256(canonical).digest()
        return self.tx_hash

    @classmethod
    def from_commit_payload(cls, payload: dict) -> "Transaction":
        """Construct from a TIMECHAIN_COMMIT bus message payload."""
        tx = cls(
            tx_type=payload.get("thought_type", "episodic"),
            source=payload.get("source", "unknown"),
            epoch_id=payload.get("epoch_id", 0),
            significance=payload.get("significance", 0.0),
            content=payload.get("content", {}),
            neuromod_snapshot={
                k: payload.get("neuromods", {}).get(k, 0.0)
                for k in ("DA", "5HT", "NE", "GABA", "ACh")
            },
            tags=payload.get("tags", []),
            timestamp=payload.get("timestamp", time.time()),
            fork_name=payload.get("fork", "episodic"),
            original_payload=payload,
        )
        tx.compute_hash()
        return tx

    def to_storage_dict(self) -> dict:
        """Minimal dict for WAL persistence (msgpack-friendly)."""
        return {
            "t": self.tx_type, "s": self.source, "e": self.epoch_id,
            "g": self.significance, "c": self.content,
            "nm": self.neuromod_snapshot, "a": self.tags,
            "ts": self.timestamp, "f": self.fork_name,
            "op": self.original_payload, "act": self.mempool_action,
        }

    @classmethod
    def from_storage_dict(cls, d: dict) -> "Transaction":
        """Reconstruct from WAL storage dict."""
        tx = cls(
            tx_type=d["t"], source=d["s"], epoch_id=d["e"],
            significance=d["g"], content=d["c"],
            neuromod_snapshot=d.get("nm", {}), tags=d.get("a", []),
            timestamp=d["ts"], fork_name=d["f"],
            original_payload=d.get("op", {}),
            mempool_action=d.get("act", "buffer"),
        )
        tx.compute_hash()
        return tx


# ═══════════════════════════════════════════════════════════════════════
# Consumer API — Query Dataclasses (Phase 2)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RecallQuery:
    """Query blocks from TimeChain. Delegates to v1 SQLite index."""
    fork: str = ""                    # Fork name filter ("" = all)
    thought_type: str = ""
    source: str = ""
    tag_contains: str = ""
    since_hours: float = 0            # Recency window (0 = no limit)
    since_epoch: int = 0              # Epoch-based window
    significance_min: float = 0.0
    significance_max: float = 1.0
    limit: int = 10
    order: str = "desc"               # "asc" or "desc" by epoch
    include_content: bool = False     # Load full block content (expensive)


@dataclass
class CheckQuery:
    """Quick boolean check — does X exist in memory?"""
    fork: str = ""
    source: str = ""
    tag_contains: str = ""
    since_hours: float = 0
    since_epoch: int = 0
    significance_min: float = 0.0


@dataclass
class CompareQuery:
    """Compare a state field across two time windows via genesis blocks."""
    fork: str = "main"
    field: str = ""                   # State field (e.g., "vocab_size")
    window_a_hours: float = 6         # Recent window
    window_b_hours: float = 12        # Older window


@dataclass
class AggregateQuery:
    """Aggregate over blocks (count, sum, avg, max, min)."""
    fork: str = ""
    op: str = "count"                 # "count", "sum", "avg", "max", "min"
    field: str = "significance"
    source: str = ""
    thought_type: str = ""
    since_hours: float = 24


@dataclass
class SimilarQuery:
    """F-phase (rFP §9.2): semantic embedding similarity over TimeChain blocks.

    Returns ranked list of {block_hash, similarity, payload_summary,
    fork, thought_type, epoch} for blocks whose stored context_embedding
    exceeds `threshold` cosine similarity to `query_vector`.

    Session 1 ships numpy cosine search over blocks whose `payload.context_embedding`
    field is set. Session 2 will add a FAISS index built during dream-
    consolidation cadence for constant-time kNN.

    Payload convention: writers (meta-reasoning in Session 2) include
    `context_embedding: list[float]` in the block payload at seal time.
    Blocks without this field are skipped (SIMILAR returns empty when no
    blocks have embeddings yet — honest "no data" signal).
    """
    query_vector: list = field(default_factory=list)  # float[N] — typ. 132D
    threshold: float = 0.75
    limit: int = 10
    fork: str = ""                    # Fork filter ("" = all)
    thought_type: str = ""
    since_hours: float = 72           # Recency filter (default 3 days)
    since_epoch: int = 0
    embedding_version: int = 0        # If > 0, only return matching version
                                       # (guards against autoencoder drift
                                       # across retrains — rFP §14.9)


# ═══════════════════════════════════════════════════════════════════════
# Smart Contract — Signed cognitive filter/trigger definitions (Phase 3a)
# ═══════════════════════════════════════════════════════════════════════

# Valid contract types
CONTRACT_TYPES = ("filter", "trigger", "genesis", "executor")
# Status lifecycle: draft → pending_approval → active / rejected
CONTRACT_STATUSES = ("draft", "pending_approval", "active", "rejected", "disabled")


@dataclass
class Contract:
    """A signed cognitive contract stored on the meta fork.

    Contracts define filter rules, post-seal triggers, or genesis-scoped
    reasoning logic. All contracts must be Ed25519-signed by Titan or Maker.
    Titan-authored contracts require Maker co-signature to activate.
    """
    contract_id: str              # Unique ID (e.g., "episodic_significance_gate")
    version: int = 1              # Incremented on update
    contract_type: str = "filter" # "filter", "trigger", "genesis", "executor"
    author: str = ""              # "titan" or "maker"
    description: str = ""
    rules: list = field(default_factory=list)     # JSON rule definitions
    triggers: list = field(default_factory=list)  # Events that activate this contract
    fork_scope: str = ""          # Fork this contract applies to ("" = all)

    # Signing
    signature: str = ""           # Ed25519 signature hex
    signer_pubkey: str = ""       # Signer's public key hex
    approver_signature: str = ""  # Maker co-signature for Titan-authored contracts
    approver_pubkey: str = ""     # Maker's public key hex

    # Lifecycle
    status: str = "draft"         # "draft", "pending_approval", "active", "rejected", "disabled"
    rejection_reason: str = ""
    created_at: float = 0.0
    activated_at: float = 0.0

    # Execution stats (updated at runtime, not part of signed content)
    execution_count: int = 0
    last_executed: float = 0.0

    def canonical_json(self) -> str:
        """Canonical JSON for signing — deterministic, excludes runtime fields."""
        return json.dumps({
            "id": self.contract_id,
            "v": self.version,
            "type": self.contract_type,
            "author": self.author,
            "desc": self.description,
            "rules": self.rules,
            "triggers": self.triggers,
            "fork": self.fork_scope,
        }, sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> bytes:
        """SHA-256 of canonical content — this is what gets signed."""
        return hashlib.sha256(self.canonical_json().encode()).digest()

    def to_dict(self) -> dict:
        """Full dict for storage/API (includes runtime fields)."""
        return {
            "contract_id": self.contract_id,
            "version": self.version,
            "contract_type": self.contract_type,
            "author": self.author,
            "description": self.description,
            "rules": self.rules,
            "triggers": self.triggers,
            "fork_scope": self.fork_scope,
            "signature": self.signature,
            "signer_pubkey": self.signer_pubkey,
            "approver_signature": self.approver_signature,
            "approver_pubkey": self.approver_pubkey,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "created_at": self.created_at,
            "activated_at": self.activated_at,
            "execution_count": self.execution_count,
            "last_executed": self.last_executed,
            "content_hash": self.content_hash().hex(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Contract":
        """Reconstruct from stored dict."""
        return cls(
            contract_id=d["contract_id"],
            version=d.get("version", 1),
            contract_type=d.get("contract_type", "filter"),
            author=d.get("author", ""),
            description=d.get("description", ""),
            rules=d.get("rules", []),
            triggers=d.get("triggers", []),
            fork_scope=d.get("fork_scope", ""),
            signature=d.get("signature", ""),
            signer_pubkey=d.get("signer_pubkey", ""),
            approver_signature=d.get("approver_signature", ""),
            approver_pubkey=d.get("approver_pubkey", ""),
            status=d.get("status", "draft"),
            rejection_reason=d.get("rejection_reason", ""),
            created_at=d.get("created_at", 0.0),
            activated_at=d.get("activated_at", 0.0),
            execution_count=d.get("execution_count", 0),
            last_executed=d.get("last_executed", 0.0),
        )


def sign_contract(contract: Contract, keypair) -> Contract:
    """Sign a contract with an Ed25519 keypair (Titan or Maker).

    Args:
        contract: Contract to sign.
        keypair: solders.keypair.Keypair instance.

    Returns:
        Contract with signature and signer_pubkey populated.
    """
    content_hash = contract.content_hash()
    sig = keypair.sign_message(content_hash)
    contract.signature = str(sig)
    contract.signer_pubkey = str(keypair.pubkey())
    contract.created_at = contract.created_at or time.time()
    return contract


def approve_contract(contract: Contract, maker_keypair) -> Contract:
    """Maker co-signs a Titan-authored contract to activate it.

    Args:
        contract: Contract with status="pending_approval".
        maker_keypair: Maker's solders.keypair.Keypair.

    Returns:
        Contract with approver_signature, status="active".
    """
    content_hash = contract.content_hash()
    sig = maker_keypair.sign_message(content_hash)
    contract.approver_signature = str(sig)
    contract.approver_pubkey = str(maker_keypair.pubkey())
    contract.status = "active"
    contract.activated_at = time.time()
    return contract


def verify_contract_signature(contract: Contract,
                              titan_pubkey: str,
                              maker_pubkey: str) -> tuple[bool, str]:
    """Verify a contract's Ed25519 signature(s).

    Returns:
        (valid, reason) — True if all required signatures check out.
    """
    try:
        from solders.signature import Signature
        from solders.pubkey import Pubkey
    except ImportError:
        return False, "solders not available"

    if not contract.signature:
        return False, "no_signature"
    if not contract.signer_pubkey:
        return False, "no_signer_pubkey"

    # Check signer is Titan or Maker
    if contract.signer_pubkey not in (titan_pubkey, maker_pubkey):
        return False, f"unknown_signer:{contract.signer_pubkey[:16]}"

    # Verify primary signature
    content_hash = contract.content_hash()
    try:
        sig = Signature.from_string(contract.signature)
        pk = Pubkey.from_string(contract.signer_pubkey)
        if not sig.verify(pk, content_hash):
            return False, "invalid_signature"
    except Exception as e:
        return False, f"signature_error:{e}"

    # Titan-authored contracts need Maker approval to be active
    if contract.author == "titan" and contract.status == "active":
        if not contract.approver_signature:
            return False, "titan_contract_missing_maker_approval"
        if contract.approver_pubkey != maker_pubkey:
            return False, f"approver_not_maker:{contract.approver_pubkey[:16]}"
        try:
            asig = Signature.from_string(contract.approver_signature)
            apk = Pubkey.from_string(contract.approver_pubkey)
            if not asig.verify(apk, content_hash):
                return False, "invalid_approver_signature"
        except Exception as e:
            return False, f"approver_sig_error:{e}"

    return True, "valid"


# PERSISTENCE_BY_DESIGN: ContractStore._bundle_verified + _last_reload are
# in-memory throttle/verification state recomputed on each contract bundle
# load. Contract data persists via the chain itself, not via ContractStore
# state files.
class ContractStore:
    """Manages contract lifecycle: deploy, load, hot-reload from meta fork.

    Contracts are stored as blocks on the meta fork (hash-chained, auditable).
    Active contracts are cached in-memory for fast evaluation in P3b.
    """

    def __init__(self, timechain, titan_pubkey: str, maker_pubkey: str):
        self._tc = timechain
        self._titan_pubkey = titan_pubkey
        self._maker_pubkey = maker_pubkey
        self._contracts: dict[str, Contract] = {}  # id → Contract
        self._last_reload = 0.0

        # Load existing contracts from meta fork
        self._load_from_chain()

    def _load_from_chain(self):
        """Scan meta fork for contract_deploy blocks and rebuild registry."""
        try:
            blocks = self._tc.query_blocks(
                thought_type="contract_deploy",
                fork_id=FORK_IDS.get("meta", 4),
                limit=500)
            loaded = 0
            for b in blocks:
                try:
                    bh = b.get("block_hash", "")
                    if isinstance(bh, str):
                        bh = bytes.fromhex(bh)
                    block = self._tc.get_block_by_hash(bh)
                    if not block or not hasattr(block, "payload"):
                        continue
                    content = block.payload.content or {}
                    contract_data = content.get("contract")
                    if not contract_data:
                        continue
                    c = Contract.from_dict(contract_data)
                    # Only keep latest version per contract_id
                    existing = self._contracts.get(c.contract_id)
                    if not existing or c.version > existing.version:
                        self._contracts[c.contract_id] = c
                        loaded += 1
                except Exception:
                    continue
            if loaded:
                logger.info("[ContractStore] Loaded %d contracts from meta fork", loaded)
        except Exception as e:
            logger.error("[ContractStore] Chain scan failed: %s", e, exc_info=True)
        self._last_reload = time.time()

    def deploy(self, contract: Contract, send_queue=None,
               worker_name: str = "timechain") -> tuple[bool, str]:
        """Deploy a signed contract to the meta fork.

        Returns:
            (success, reason)
        """
        # Validate contract type
        if contract.contract_type not in CONTRACT_TYPES:
            return False, f"invalid_type:{contract.contract_type}"

        # Validate signature
        valid, reason = verify_contract_signature(
            contract, self._titan_pubkey, self._maker_pubkey)
        if not valid:
            logger.warning("[ContractStore] Rejected contract '%s': %s",
                           contract.contract_id, reason)
            return False, reason

        # Set status based on author
        if contract.author == "maker" or contract.approver_signature:
            contract.status = "active"
            contract.activated_at = contract.activated_at or time.time()
        elif contract.author == "titan":
            contract.status = "pending_approval"

        # Store in registry
        self._contracts[contract.contract_id] = contract

        # Commit to meta fork
        from titan_plugin.logic.timechain import BlockPayload
        payload = BlockPayload(
            thought_type="contract_deploy",
            source="contract_engine",
            content={"contract": contract.to_dict()},
            significance=1.0,
            confidence=1.0,
            tags=["contract", contract.contract_id, contract.contract_type],
        )

        block = self._tc.commit_block(
            fork_id=FORK_IDS.get("meta", 4),
            epoch_id=0,  # Will be set by commit_block
            payload=payload,
            pot_nonce=1,  # Contracts bypass PoT (sovereignty is via Ed25519)
            chi_spent=0.005,
            neuromod_state={},
        )

        if block:
            logger.info(
                "[ContractStore] Deployed '%s' v%d (type=%s, author=%s, status=%s)",
                contract.contract_id, contract.version,
                contract.contract_type, contract.author, contract.status)

            # Emit deployment event
            if send_queue:
                try:
                    send_queue.put_nowait({
                        "type": "CONTRACT_DEPLOYED",
                        "src": worker_name,
                        "dst": "all",
                        "ts": time.time(),
                        "payload": {
                            "contract_id": contract.contract_id,
                            "version": contract.version,
                            "type": contract.contract_type,
                            "author": contract.author,
                            "status": contract.status,
                            "block_hash": block.block_hash.hex(),
                        },
                    })
                except Exception:
                    pass
            return True, "deployed"

        return False, "commit_failed"

    def approve(self, contract_id: str, maker_keypair,
                send_queue=None, worker_name: str = "timechain") -> tuple[bool, str]:
        """Maker approves a pending Titan-authored contract."""
        contract = self._contracts.get(contract_id)
        if not contract:
            return False, "not_found"
        if contract.status != "pending_approval":
            return False, f"wrong_status:{contract.status}"

        approve_contract(contract, maker_keypair)
        # Re-deploy with updated status
        return self.deploy(contract, send_queue, worker_name)

    def reject(self, contract_id: str, reason: str = "") -> tuple[bool, str]:
        """Maker rejects a pending contract."""
        contract = self._contracts.get(contract_id)
        if not contract:
            return False, "not_found"
        contract.status = "rejected"
        contract.rejection_reason = reason
        logger.info("[ContractStore] Rejected '%s': %s", contract_id, reason)
        return True, "rejected"

    def get_active(self, contract_type: str = None,
                   fork_scope: str = None) -> list[Contract]:
        """Get all active contracts, optionally filtered."""
        result = []
        for c in self._contracts.values():
            if c.status != "active":
                continue
            if contract_type and c.contract_type != contract_type:
                continue
            if fork_scope and c.fork_scope and c.fork_scope != fork_scope:
                continue
            result.append(c)
        return result

    def get_all(self) -> list[Contract]:
        """Get all contracts (any status)."""
        return list(self._contracts.values())

    def get(self, contract_id: str) -> Optional[Contract]:
        return self._contracts.get(contract_id)

    def get_pending(self) -> list[Contract]:
        """Get contracts pending Maker approval."""
        return [c for c in self._contracts.values()
                if c.status == "pending_approval"]

    def propose(self, name: str, contract_type: str, rules: list,
                description: str, titan_keypair,
                triggers: list = None, fork_scope: str = "",
                send_queue=None, worker_name: str = "timechain") -> tuple[bool, str]:
        """Titan proposes a new contract (P3d). Signed by Titan, pending Maker approval.

        Args:
            titan_keypair: solders.keypair.Keypair instance or bytes (will be converted).
        """
        if contract_type not in CONTRACT_TYPES:
            return False, f"invalid_type:{contract_type}"
        if name in self._contracts:
            return False, "already_exists"

        contract = Contract(
            contract_id=name,
            contract_type=contract_type,
            author="titan",
            description=description,
            rules=rules,
            triggers=triggers or [],
            fork_scope=fork_scope,
            created_at=time.time(),
        )
        try:
            # Convert bytes to Keypair if needed
            kp = titan_keypair
            if isinstance(titan_keypair, bytes):
                from solders.keypair import Keypair as _Kp
                kp = _Kp.from_bytes(titan_keypair)
            sign_contract(contract, kp)
        except Exception as e:
            return False, f"sign_failed:{e}"

        # Deploy (will set status=pending_approval for titan-authored)
        return self.deploy(contract, send_queue, worker_name)

    def hot_reload(self):
        """Reload contracts from chain if stale (>60s since last reload)."""
        if time.time() - self._last_reload > 60:
            self._load_from_chain()

    def get_stats(self) -> dict:
        by_status = {}
        by_type = {}
        for c in self._contracts.values():
            by_status[c.status] = by_status.get(c.status, 0) + 1
            by_type[c.contract_type] = by_type.get(c.contract_type, 0) + 1
        return {
            "total": len(self._contracts),
            "by_status": by_status,
            "by_type": by_type,
            "last_reload": self._last_reload,
        }

    def bootstrap_builtin_contracts(self, keypair_bytes: bytes = None):
        """Deploy built-in filter/trigger/genesis contracts on first boot.

        These replace hardcoded mempool logic with signed, auditable contracts.
        Only deploys if the contract doesn't already exist.
        """
        BUILTIN = [
            # P3b: Filter contracts (replace hardcoded mempool rules)
            Contract(
                contract_id="noise_floor_gate",
                contract_type="filter",
                author="system",
                description="Drop noise below sig 0.05 except protected sources",
                rules=[{
                    "op": "AND",
                    "clauses": [
                        {"op": "IF", "field": "significance", "cmp": "LT", "value": 0.05},
                        {"op": "IF", "field": "source", "cmp": "NOT_IN", "value": [
                            "meditation", "dream_insight", "conversation",
                            "kin_exchange", "genesis_chain"]},
                    ],
                    "then": {"action": "drop"},
                }],
                status="active",
                created_at=time.time(),
                activated_at=time.time(),
            ),
            Contract(
                contract_id="episodic_significance_gate",
                contract_type="filter",
                author="system",
                description="Aggregate low-sig expression events instead of individual blocks",
                rules=[{
                    "op": "AND",
                    "clauses": [
                        {"op": "IF", "field": "significance", "cmp": "LT", "value": 0.3},
                        {"op": "IF", "field": "source", "cmp": "IN", "value": [
                            "expression_art", "expression_music",
                            "expression_kin_sense", "expression_longing"]},
                    ],
                    "then": {"action": "aggregate"},
                }],
                status="active",
                created_at=time.time(),
                activated_at=time.time(),
            ),
            Contract(
                contract_id="high_significance_fast_track",
                contract_type="filter",
                author="system",
                description="Immediately include high-significance TXs (bypass batching)",
                rules=[{
                    "op": "IF",
                    "field": "significance",
                    "cmp": "GTE",
                    "value": 0.9,
                    "then": {"action": "include"},
                }],
                status="active",
                created_at=time.time(),
                activated_at=time.time(),
            ),
            # P3c: Genesis trigger contracts
            Contract(
                contract_id="cognitive_stall_detector",
                contract_type="genesis",
                author="system",
                description="Detect vocab stall when reasoning is active",
                rules=[{
                    "op": "AND",
                    "clauses": [
                        {"op": "TREND", "field": "vocab_size", "window": 3,
                         "cmp": "EQ", "value": "flat"},
                        {"op": "IF", "field": "meta_chains", "cmp": "GT", "value": 10},
                    ],
                    "then": {"action": "emit", "event": "COGNITIVE_STALL"},
                }],
                status="active",
                created_at=time.time(),
                activated_at=time.time(),
            ),
            Contract(
                contract_id="milestone_tracker",
                contract_type="genesis",
                author="system",
                description="Detect developmental milestones from genesis state",
                rules=[
                    {
                        "op": "AND",
                        "clauses": [
                            {"op": "IF", "field": "i_confidence", "cmp": "GTE", "value": 0.95},
                            {"op": "IF", "field": "vocab_size", "cmp": "GTE", "value": 300},
                        ],
                        "then": {"action": "emit", "event": "MILESTONE_REACHED",
                                 "data": {"milestone": "sovereign_speaker"}},
                    },
                    {
                        "op": "IF", "field": "vocab_size", "cmp": "GTE", "value": 500,
                        "then": {"action": "emit", "event": "MILESTONE_REACHED",
                                 "data": {"milestone": "fluent_speaker"}},
                    },
                ],
                status="active",
                created_at=time.time(),
                activated_at=time.time(),
            ),
            Contract(
                contract_id="homeostatic_alert",
                contract_type="genesis",
                author="system",
                description="Alert on sustained neuromodulator imbalance",
                rules=[{
                    "op": "DELTA", "field": "neuromods.GABA", "n_back": 3,
                    "cmp": "LT", "value": -0.1,
                    "then": {"action": "emit", "event": "HOMEOSTATIC_ALERT",
                             "data": {"neuromod": "GABA", "trend": "declining"}},
                }],
                status="active",
                created_at=time.time(),
                activated_at=time.time(),
            ),
        ]

        deployed = 0
        for contract in BUILTIN:
            if contract.contract_id not in self._contracts:
                # Sign with Titan key if available, otherwise mark as system
                if keypair_bytes:
                    try:
                        sign_contract(contract, keypair_bytes)
                    except Exception:
                        pass  # System contracts work without signature
                self._contracts[contract.contract_id] = contract
                # Commit to meta fork
                try:
                    from titan_plugin.logic.timechain import BlockPayload
                    payload = BlockPayload(
                        thought_type="contract_deploy",
                        source="contract_engine",
                        content={"contract": contract.to_dict()},
                        significance=1.0,
                        confidence=1.0,
                        tags=["contract", contract.contract_id,
                              contract.contract_type, "builtin"],
                    )
                    self._tc.commit_block(
                        fork_id=FORK_IDS.get("meta", 4),
                        epoch_id=0,
                        payload=payload,
                        pot_nonce=1,
                        chi_spent=0.001,
                        neuromod_state={},
                    )
                    deployed += 1
                except Exception as e:
                    logger.warning("[ContractStore] Failed to persist builtin %s: %s",
                                 contract.contract_id, e)
                    deployed += 1  # Still in-memory

        if deployed:
            logger.info("[ContractStore] Bootstrapped %d built-in contracts", deployed)

    def compute_bundle_hash_and_names(
        self, contracts_dir: str = None
    ) -> tuple[str, list[str]]:
        """Compute the deterministic bundle hash + contract names list.

        Mirrors the bundle hashing inside load_meta_cognitive_contracts but
        factored out so TitanMaker (R8 autoseed) can compute the hash without
        re-loading the contracts. Used at boot to determine if a Maker
        signature is needed and to seed a TitanMaker proposal if so.
        """
        import os
        if contracts_dir is None:
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            contracts_dir = os.path.join(here, "contracts", "meta_cognitive")
        if not os.path.isdir(contracts_dir):
            return "", []
        json_files = sorted(
            f for f in os.listdir(contracts_dir)
            if f.endswith(".json") and not f.startswith(".")
        )
        if not json_files:
            return "", []
        bundle_hasher = hashlib.sha256()
        names = []
        for fname in json_files:
            fpath = os.path.join(contracts_dir, fname)
            try:
                with open(fpath, "rb") as f:
                    raw = f.read()
                bundle_hasher.update(fname.encode())
                bundle_hasher.update(b"\x00")
                bundle_hasher.update(raw)
                bundle_hasher.update(b"\x00")
                d = json.loads(raw.decode())
                cname = d.get("contract_id", fname.replace(".json", ""))
                names.append(cname)
            except Exception:
                names.append(fname.replace(".json", ""))
        return bundle_hasher.hexdigest(), names

    @property
    def bundle_verified(self) -> bool:
        """Whether the loaded Phase C cognitive contract bundle has Maker
        signature verification (R8 ceremony complete). Set by
        load_meta_cognitive_contracts when .bundle_signature.json matches.
        """
        return getattr(self, "_bundle_verified", False)

    def load_meta_cognitive_contracts(
        self,
        contracts_dir: str = None,
        keypair_bytes: bytes = None,
        bundle_signature_path: str = None,
    ) -> int:
        """TUNING-012 v2 Sub-phase C: load Phase C cognitive contracts from JSON.

        Reads contract JSONs from titan_plugin/contracts/meta_cognitive/ and
        registers them via the same path as built-in contracts. The contracts
        live INSIDE the Python package (not data/) because data/ is per-Titan
        runtime state and must not sync between Titans via git pull. Cognitive
        contracts are SOURCE artifacts that ship with the codebase.

        Implements R8 — bundled Maker ceremony — by verifying a single SHA-256
        hash over all JSON contents matches the embedded approver signature,
        instead of requiring per-contract signatures.

        Args:
            contracts_dir: Override the default titan_plugin/contracts/meta_cognitive path.
            keypair_bytes: Optional Titan signing keypair (auto-signs at boot).
            bundle_signature_path: Path to bundle signature file. When present,
                verifies the bundle hash + Maker signature before activating
                contracts. Defaults to <contracts_dir>/.bundle_signature.json
                (per-Titan, NOT shipped via git — each Titan can hold its own
                Maker-signed bundle independently).

        Returns:
            Number of contracts loaded into the store.
        """
        import os
        if contracts_dir is None:
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            contracts_dir = os.path.join(here, "contracts", "meta_cognitive")
        if not os.path.isdir(contracts_dir):
            return 0

        # Discover JSON contract files (sorted for deterministic bundle hash)
        json_files = sorted(
            f for f in os.listdir(contracts_dir)
            if f.endswith(".json") and not f.startswith(".")
        )
        if not json_files:
            return 0

        # Compute deterministic bundle hash for R8 (bundled Maker ceremony)
        bundle_hasher = hashlib.sha256()
        contract_dicts = []
        for fname in json_files:
            fpath = os.path.join(contracts_dir, fname)
            try:
                with open(fpath, "rb") as f:
                    raw = f.read()
                bundle_hasher.update(fname.encode())
                bundle_hasher.update(b"\x00")
                bundle_hasher.update(raw)
                bundle_hasher.update(b"\x00")
                d = json.loads(raw.decode())
                contract_dicts.append((fname, d))
            except Exception as e:
                logger.warning(
                    "[ContractStore] Skipping malformed cognitive contract %s: %s",
                    fname, e)
        bundle_hash = bundle_hasher.hexdigest()

        # R8: verify Maker bundle signature if present (allow boot without
        # signature for first deploy — contracts activate as system-signed)
        if bundle_signature_path is None:
            bundle_signature_path = os.path.join(
                contracts_dir, ".bundle_signature.json")
        bundle_verified = False
        bundle_meta = {}
        if os.path.exists(bundle_signature_path):
            try:
                with open(bundle_signature_path) as bs_f:
                    bundle_meta = json.load(bs_f)
                if bundle_meta.get("bundle_hash") == bundle_hash:
                    bundle_verified = True
                    logger.info(
                        "[ContractStore] Cognitive contract bundle verified: "
                        "hash=%s... signer=%s",
                        bundle_hash[:16],
                        bundle_meta.get("approver_pubkey", "?")[:16],
                    )
                else:
                    logger.warning(
                        "[ContractStore] Bundle signature stale (file hash %s "
                        "≠ live %s) — contracts will load as system-signed",
                        bundle_meta.get("bundle_hash", "?")[:16],
                        bundle_hash[:16],
                    )
            except Exception as bs_err:
                logger.warning(
                    "[ContractStore] Bundle signature read failed: %s", bs_err)
        else:
            logger.info(
                "[ContractStore] No bundle signature yet (hash=%s...) — "
                "Phase C contracts loading as system-signed (Maker ceremony pending)",
                bundle_hash[:16],
            )

        deployed = 0
        for fname, d in contract_dicts:
            contract_id = d.get("contract_id", "")
            if not contract_id:
                continue
            if contract_id in self._contracts:
                # Already loaded — skip without warning (idempotent boot)
                continue
            now = time.time()
            contract = Contract(
                contract_id=contract_id,
                version=d.get("version", 1),
                contract_type=d.get("contract_type", "genesis"),
                author=d.get("author", "titan"),
                description=d.get("description", ""),
                rules=d.get("rules", []),
                triggers=d.get("triggers", []),
                fork_scope=d.get("fork_scope", "meta"),
                status="active" if d.get("status", "active") == "active" else "draft",
                created_at=now,
                activated_at=now,
            )
            # If bundle verified, embed approver signature on each contract
            if bundle_verified:
                contract.approver_signature = bundle_meta.get(
                    "approver_signature", "")
                contract.approver_pubkey = bundle_meta.get(
                    "approver_pubkey", "")
            # Auto-sign with Titan key if available
            if keypair_bytes:
                try:
                    sign_contract(contract, keypair_bytes)
                except Exception:
                    pass

            self._contracts[contract_id] = contract
            # Persist to meta fork (same path as built-in contracts)
            try:
                from titan_plugin.logic.timechain import BlockPayload
                payload = BlockPayload(
                    thought_type="contract_deploy",
                    source="contract_engine",
                    content={
                        "contract": contract.to_dict(),
                        "phase": d.get("phase", ""),
                        "bundle_hash": bundle_hash,
                        "bundle_verified": bundle_verified,
                    },
                    significance=1.0,
                    confidence=1.0,
                    tags=["contract", contract_id, contract.contract_type,
                          "meta_cognitive", "tuning_012_v2"],
                )
                self._tc.commit_block(
                    fork_id=FORK_IDS.get("meta", 4),
                    epoch_id=0,
                    payload=payload,
                    pot_nonce=1,
                    chi_spent=0.001,
                    neuromod_state={},
                )
            except Exception as persist_err:
                logger.warning(
                    "[ContractStore] Failed to persist meta-cognitive %s: %s",
                    contract_id, persist_err)
            deployed += 1
            logger.info(
                "[ContractStore] Loaded meta-cognitive contract '%s' (%s, %d rules, "
                "approver=%s)",
                contract_id, contract.contract_type, len(contract.rules),
                "MAKER" if bundle_verified else "system",
            )

        # Persist verification state for the bundle_verified property
        # (consumed by TitanMaker R8 autoseed at boot to decide if a Maker
        # signing proposal needs to be created).
        self._bundle_verified = bundle_verified
        if deployed:
            logger.info(
                "[ContractStore] Loaded %d meta-cognitive contracts (bundle_hash=%s..., "
                "verified=%s)",
                deployed, bundle_hash[:16], bundle_verified,
            )
        return deployed


# ═══════════════════════════════════════════════════════════════════════
# RuleEvaluator — Non-Turing-complete contract rule interpreter
# ═══════════════════════════════════════════════════════════════════════

class RuleEvaluator:
    """Evaluates contract rules against a context dict.

    Non-Turing-complete: no loops, no recursion, max 50 rules, max 3 queries.
    Rules evaluated top-to-bottom. First action-producing rule wins.

    Rule format: {"op": "IF", "field": "significance", "cmp": "LT", "value": 0.05,
                  "then": {"action": "drop"}}
    Compound: {"op": "AND", "clauses": [...], "then": {"action": "aggregate"}}
    """

    MAX_RULES = 50
    MAX_QUERIES = 3

    def __init__(self, orchestrator=None):
        self._orchestrator = orchestrator  # For RECALL queries (Consumer API)

    def evaluate(self, rules: list[dict], context: dict,
                 genesis_states: list[dict] = None) -> Optional[dict]:
        """Evaluate rules against context. Returns action dict or None."""
        if len(rules) > self.MAX_RULES:
            logger.warning("[RuleEval] Contract exceeds %d rule limit", self.MAX_RULES)
            return None

        variables = {}
        query_count = 0

        for rule in rules:
            op = rule.get("op", "").upper()

            # Variable binding (RECALL query)
            if op == "RECALL" and query_count < self.MAX_QUERIES:
                query_count += 1
                result = self._exec_recall(rule, context)
                store_as = rule.get("store", "")
                if store_as.startswith("$"):
                    variables[store_as] = result
                continue

            # Evaluate condition
            matched = self._eval_condition(rule, context, variables, genesis_states)
            if matched:
                action = rule.get("then", {})
                if action and action.get("action"):
                    return action
        return None

    def _eval_condition(self, rule: dict, ctx: dict, variables: dict,
                        genesis_states: list[dict] = None) -> bool:
        """Evaluate a single condition or compound condition."""
        op = rule.get("op", "").upper()

        if op == "IF":
            return self._eval_comparison(rule, ctx, variables)
        elif op == "AND":
            return all(self._eval_condition(c, ctx, variables, genesis_states)
                       for c in rule.get("clauses", []))
        elif op == "OR":
            return any(self._eval_condition(c, ctx, variables, genesis_states)
                       for c in rule.get("clauses", []))
        elif op == "NOT":
            inner = rule.get("clause", {})
            return not self._eval_condition(inner, ctx, variables, genesis_states)
        elif op in ("TREND", "DELTA", "SINCE", "STATE_AT"):
            return self._eval_genesis_primitive(op, rule, ctx, genesis_states)
        return False

    def _eval_comparison(self, rule: dict, ctx: dict, variables: dict) -> bool:
        """Evaluate IF comparison: field CMP value."""
        field = rule.get("field", "")
        cmp_op = rule.get("cmp", "").upper()
        target = rule.get("value")

        # Resolve field value — support dotted paths (e.g., "neuromods.DA")
        actual = self._resolve_field(field, ctx, variables)
        if actual is None:
            return False

        # Resolve target if it's a variable reference
        if isinstance(target, str) and target.startswith("$"):
            target = variables.get(target)

        try:
            if cmp_op == "GT":
                return float(actual) > float(target)
            elif cmp_op == "LT":
                return float(actual) < float(target)
            elif cmp_op == "GTE":
                return float(actual) >= float(target)
            elif cmp_op == "LTE":
                return float(actual) <= float(target)
            elif cmp_op == "EQ":
                return actual == target
            elif cmp_op == "NEQ":
                return actual != target
            elif cmp_op == "IN":
                return actual in (target if isinstance(target, (list, tuple, set)) else [target])
            elif cmp_op == "NOT_IN":
                return actual not in (target if isinstance(target, (list, tuple, set)) else [target])
            elif cmp_op == "BETWEEN":
                if isinstance(target, list) and len(target) == 2:
                    return target[0] <= float(actual) <= target[1]
            elif cmp_op == "STARTSWITH":
                return str(actual).startswith(str(target))
        except (TypeError, ValueError):
            return False
        return False

    def _resolve_field(self, field: str, ctx: dict, variables: dict):
        """Resolve a dotted field path or variable from context."""
        if field.startswith("$"):
            return variables.get(field)
        parts = field.split(".")
        val = ctx
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p)
            else:
                return None
        return val

    def _exec_recall(self, rule: dict, ctx: dict) -> Any:
        """Execute a RECALL query via Consumer API."""
        if not self._orchestrator:
            return None
        try:
            result = self._orchestrator.recall(
                fork=rule.get("fork", ""),
                source=rule.get("source", ""),
                since_hours=rule.get("since_hours", 24),
                limit=rule.get("limit", 10),
                significance_min=rule.get("significance_min", 0.0),
            )
            # Apply aggregate if requested
            agg = rule.get("agg", "").upper()
            if agg == "COUNT":
                return len(result)
            elif agg == "AVG" and result:
                field = rule.get("agg_field", "significance")
                vals = [r.get(field, 0) for r in result]
                return sum(vals) / len(vals) if vals else 0
            elif agg == "MAX" and result:
                field = rule.get("agg_field", "significance")
                return max(r.get(field, 0) for r in result)
            elif agg == "MIN" and result:
                field = rule.get("agg_field", "significance")
                return min(r.get(field, 0) for r in result)
            elif agg == "SUM" and result:
                field = rule.get("agg_field", "significance")
                return sum(r.get(field, 0) for r in result)
            return result
        except Exception as e:
            logger.error("[RuleEval] RECALL query failed: %s", e, exc_info=True)
            return None

    def _eval_genesis_primitive(self, op: str, rule: dict, ctx: dict,
                                genesis_states: list[dict] = None) -> bool:
        """Evaluate genesis-scoped primitives: TREND, DELTA, SINCE, STATE_AT."""
        if not genesis_states:
            return False

        field = rule.get("field", "")
        cmp_op = rule.get("cmp", "EQ").upper()
        target = rule.get("value")
        window = rule.get("window", 3)

        if op == "TREND":
            # Get field values from last N genesis blocks
            vals = []
            for gs in genesis_states[:window]:
                v = self._resolve_field(field, gs, {})
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            if len(vals) < 2:
                return False
            # Determine trend direction
            diffs = [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]
            avg_diff = sum(diffs) / len(diffs)
            if abs(avg_diff) < 0.01:
                trend = "flat"
            elif avg_diff > 0:
                trend = "rising"
            else:
                trend = "falling"
            # Compare trend to target
            return self._simple_compare(trend, cmp_op, target)

        elif op == "DELTA":
            n_back = rule.get("n_back", 1)
            if n_back >= len(genesis_states):
                return False
            current_val = self._resolve_field(field, genesis_states[0], {})
            past_val = self._resolve_field(field, genesis_states[n_back], {})
            if current_val is None or past_val is None:
                return False
            try:
                delta = float(current_val) - float(past_val)
            except (TypeError, ValueError):
                return False
            return self._simple_compare(delta, cmp_op, target)

        elif op == "SINCE":
            event_type = rule.get("event_type", "")
            for i, gs in enumerate(genesis_states):
                if gs.get("trigger") == event_type or gs.get("emotion") == event_type:
                    return self._simple_compare(i, cmp_op, target)
            return False

        elif op == "STATE_AT":
            n_back = rule.get("n_back", 0)
            if n_back < len(genesis_states):
                store_as = rule.get("store", "")
                if store_as.startswith("$"):
                    # This is a variable binding — handled via context
                    pass
                return True
            return False

        return False

    @staticmethod
    def _simple_compare(actual, cmp_op: str, target) -> bool:
        """Simple comparison helper."""
        try:
            if cmp_op == "EQ":
                return actual == target
            elif cmp_op == "NEQ":
                return actual != target
            elif cmp_op == "GT":
                return float(actual) > float(target)
            elif cmp_op == "LT":
                return float(actual) < float(target)
            elif cmp_op == "GTE":
                return float(actual) >= float(target)
            elif cmp_op == "LTE":
                return float(actual) <= float(target)
        except (TypeError, ValueError):
            pass
        return False


# ═══════════════════════════════════════════════════════════════════════
# BloomFilter — O(1) probabilistic duplicate detection
# ═══════════════════════════════════════════════════════════════════════

class BloomFilter:
    """Pure Python bloom filter for mempool TX dedup.

    Default: 100K capacity, 0.1% false positive rate → ~175 KB memory.
    Rotated daily to prevent saturation.
    """

    def __init__(self, capacity: int = 100_000, fp_rate: float = 0.001):
        self._capacity = capacity
        self._fp_rate = fp_rate
        # Optimal bit count and hash count
        self._num_bits = max(64, int(-capacity * math.log(fp_rate) / (math.log(2) ** 2)))
        self._num_hashes = max(1, int((self._num_bits / capacity) * math.log(2)))
        self._bits = bytearray(self._num_bits // 8 + 1)
        self._count = 0
        self._created_at = time.time()

    def _hash_positions(self, item: bytes) -> list[int]:
        """Compute hash positions using double hashing (SHA-256 based)."""
        h = hashlib.sha256(item).digest()
        h1 = int.from_bytes(h[:8], "big")
        h2 = int.from_bytes(h[8:16], "big")
        return [(h1 + i * h2) % self._num_bits for i in range(self._num_hashes)]

    def add(self, item: bytes) -> None:
        """Add an item to the filter."""
        for pos in self._hash_positions(item):
            self._bits[pos // 8] |= (1 << (pos % 8))
        self._count += 1

    def might_contain(self, item: bytes) -> bool:
        """Check if an item might be in the filter (false positives possible)."""
        return all(
            self._bits[pos // 8] & (1 << (pos % 8))
            for pos in self._hash_positions(item)
        )

    def clear(self) -> None:
        """Reset the filter (daily rotation)."""
        self._bits = bytearray(self._num_bits // 8 + 1)
        self._count = 0
        self._created_at = time.time()

    @property
    def size_kb(self) -> float:
        return len(self._bits) / 1024

    def should_rotate(self) -> bool:
        """True if filter is >24h old or >80% capacity."""
        age_hours = (time.time() - self._created_at) / 3600
        return age_hours > 24 or self._count > self._capacity * 0.8


# ═══════════════════════════════════════════════════════════════════════
# MempoolWAL — Crash-safe mempool persistence
# ═══════════════════════════════════════════════════════════════════════

class MempoolWAL:
    """SQLite WAL-mode database for mempool crash recovery.

    Stores pending TXs and aggregate counters. On crash restart,
    pending TXs are recovered and re-submitted to the mempool.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, timeout=10.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS pending_txs (
                tx_hash     BLOB PRIMARY KEY,
                fork_name   TEXT NOT NULL,
                tx_type     TEXT NOT NULL,
                source      TEXT NOT NULL,
                significance REAL NOT NULL,
                epoch_id    INTEGER NOT NULL,
                timestamp   REAL NOT NULL,
                payload     BLOB NOT NULL,
                action      TEXT DEFAULT 'buffer',
                created_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS aggregates (
                agg_key     TEXT PRIMARY KEY,
                fork_name   TEXT NOT NULL,
                count       INTEGER DEFAULT 0,
                sig_sum     REAL DEFAULT 0,
                sig_max     REAL DEFAULT 0,
                epoch_min   INTEGER DEFAULT 0,
                epoch_max   INTEGER DEFAULT 0,
                type_counts TEXT DEFAULT '{}',
                updated_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pending_fork
                ON pending_txs(fork_name);
            CREATE INDEX IF NOT EXISTS idx_pending_action
                ON pending_txs(action);
        """)
        self._conn.commit()

    def insert(self, tx: Transaction) -> None:
        """Insert a pending TX (idempotent via PRIMARY KEY)."""
        payload_bytes = json.dumps(tx.to_storage_dict()).encode()
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO pending_txs "
                "(tx_hash, fork_name, tx_type, source, significance, "
                "epoch_id, timestamp, payload, action, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (tx.tx_hash, tx.fork_name, tx.tx_type, tx.source,
                 tx.significance, tx.epoch_id, tx.timestamp,
                 payload_bytes, tx.mempool_action, time.time()))
            self._conn.commit()
        except sqlite3.Error as e:
            logger.warning("[MempoolWAL] Insert failed: %s", e)

    def get_pending(self, fork_name: str = None) -> list[Transaction]:
        """Get all pending TXs, optionally filtered by fork."""
        if fork_name:
            rows = self._conn.execute(
                "SELECT payload FROM pending_txs WHERE fork_name = ? "
                "ORDER BY timestamp", (fork_name,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload FROM pending_txs ORDER BY timestamp").fetchall()
        txs = []
        for row in rows:
            try:
                d = json.loads(row["payload"])
                txs.append(Transaction.from_storage_dict(d))
            except Exception as e:
                logger.warning("[MempoolWAL] Corrupt TX skipped: %s", e)
        return txs

    def get_pending_forks(self) -> list[str]:
        """Get list of fork names that have pending TXs."""
        rows = self._conn.execute(
            "SELECT DISTINCT fork_name FROM pending_txs").fetchall()
        return [r["fork_name"] for r in rows]

    def pending_count(self, fork_name: str = None) -> int:
        if fork_name:
            row = self._conn.execute(
                "SELECT COUNT(*) as c FROM pending_txs WHERE fork_name = ?",
                (fork_name,)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) as c FROM pending_txs").fetchone()
        return row["c"] if row else 0

    def mark_sealed(self, tx_hashes: list[bytes]) -> None:
        """Remove sealed TXs from pending."""
        if not tx_hashes:
            return
        self._conn.executemany(
            "DELETE FROM pending_txs WHERE tx_hash = ?",
            [(h,) for h in tx_hashes])
        self._conn.commit()

    def update_aggregate(self, agg_key: str, fork_name: str,
                         count: int, sig_sum: float, sig_max: float,
                         epoch_min: int, epoch_max: int,
                         type_counts: dict) -> None:
        """Upsert an aggregate counter."""
        self._conn.execute(
            "INSERT INTO aggregates "
            "(agg_key, fork_name, count, sig_sum, sig_max, "
            "epoch_min, epoch_max, type_counts, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(agg_key) DO UPDATE SET "
            "count=count+excluded.count, "
            "sig_sum=sig_sum+excluded.sig_sum, "
            "sig_max=MAX(sig_max, excluded.sig_max), "
            "epoch_min=MIN(epoch_min, excluded.epoch_min), "
            "epoch_max=MAX(epoch_max, excluded.epoch_max), "
            "type_counts=excluded.type_counts, "
            "updated_at=excluded.updated_at",
            (agg_key, fork_name, count, sig_sum, sig_max,
             epoch_min, epoch_max, json.dumps(type_counts), time.time()))
        self._conn.commit()

    def get_aggregates(self) -> list[dict]:
        """Get all aggregate counters."""
        rows = self._conn.execute("SELECT * FROM aggregates").fetchall()
        return [dict(r) for r in rows]

    def clear_aggregate(self, agg_key: str) -> None:
        self._conn.execute("DELETE FROM aggregates WHERE agg_key = ?", (agg_key,))
        self._conn.commit()

    def recover(self) -> tuple[list[Transaction], list[dict]]:
        """Recover pending TXs + aggregates after crash. Returns (txs, aggs)."""
        txs = self.get_pending()
        aggs = self.get_aggregates()
        logger.info("[MempoolWAL] Recovered %d pending TXs, %d aggregates",
                    len(txs), len(aggs))
        return txs, aggs

    def close(self):
        self._conn.close()


# ═══════════════════════════════════════════════════════════════════════
# Mempool — Dedup + Aggregation + WAL
# ═══════════════════════════════════════════════════════════════════════

class Mempool:
    """Transaction pool with bloom filter dedup and expression aggregation.

    Low-significance expression fires are aggregated in-memory (100:1).
    All other TXs are WAL-persisted for crash recovery.
    """

    def __init__(self, data_dir: str, config: dict):
        self._config = config
        self._bloom = BloomFilter(
            capacity=config.get("bloom_capacity", 100_000),
            fp_rate=config.get("bloom_fp_rate", 0.001),
        )
        wal_path = os.path.join(data_dir, "mempool.wal.db")
        self._wal = MempoolWAL(wal_path)

        # Aggregation config
        self._agg_threshold = config.get("aggregate_threshold_significance", 0.3)
        self._agg_batch_size = config.get("aggregate_batch_size", 100)
        self._agg_sources = set(config.get("aggregate_sources", [
            "expression_art", "expression_music",
            "expression_kin_sense", "expression_longing",
        ]))

        # In-memory aggregate buckets: {fork_name: {agg_key: {...}}}
        self._agg_buckets: dict[str, dict] = {}

        # Contract-based filtering (P3b) — set by Orchestrator after init
        self._contract_store: Optional[ContractStore] = None
        self._rule_evaluator: Optional[RuleEvaluator] = None
        self._contract_filter_evals = 0
        self._contract_filter_hits = 0

        # Stats
        self._total_submitted = 0
        self._total_aggregated = 0
        self._total_duplicates = 0
        self._total_dropped = 0
        self._total_queued = 0

        # Recover from crash
        recovered_txs, recovered_aggs = self._wal.recover()
        if recovered_txs:
            for tx in recovered_txs:
                self._bloom.add(tx.tx_hash)
            logger.info("[Mempool] Recovered %d TXs from WAL", len(recovered_txs))
        for agg in recovered_aggs:
            fork = agg.get("fork_name", "episodic")
            if fork not in self._agg_buckets:
                self._agg_buckets[fork] = {}
            self._agg_buckets[fork][agg["agg_key"]] = {
                "count": agg["count"],
                "sig_sum": agg["sig_sum"],
                "sig_max": agg["sig_max"],
                "epoch_min": agg["epoch_min"],
                "epoch_max": agg["epoch_max"],
                "type_counts": json.loads(agg.get("type_counts", "{}")),
            }

    def set_contract_filter(self, contract_store: "ContractStore",
                            rule_evaluator: "RuleEvaluator"):
        """Attach contract-based filtering (called by Orchestrator after init)."""
        self._contract_store = contract_store
        self._rule_evaluator = rule_evaluator

    def submit(self, tx: Transaction) -> str:
        """Submit TX to mempool. Returns: 'queued'|'aggregated'|'duplicate'|'dropped'."""
        self._total_submitted += 1

        # 1. Bloom filter dedup
        if self._bloom.might_contain(tx.tx_hash):
            self._total_duplicates += 1
            return "duplicate"
        self._bloom.add(tx.tx_hash)

        # 2. Contract-based filter evaluation (P3b) — replaces hardcoded rules
        contract_result = self._evaluate_filter_contracts(tx)
        if contract_result:
            action = contract_result.get("action", "").lower()
            if action == "drop":
                self._total_dropped += 1
                return "dropped"
            elif action == "aggregate":
                self._aggregate(tx)
                self._total_aggregated += 1
                return "aggregated"
            elif action == "include":
                tx.mempool_action = "include"
                # Fall through to WAL persist
            # Any other action → fall through to hardcoded defaults

        # 3. Hardcoded fallbacks (backward compat — used when no contracts match)
        if not contract_result:
            # Aggregate-only for low-significance expression sources
            source_key = f"expression_{tx.source.split('_', 1)[-1]}" if "_" in tx.source else tx.source
            if (tx.source in self._agg_sources or source_key in self._agg_sources) \
                    and tx.significance < self._agg_threshold:
                self._aggregate(tx)
                self._total_aggregated += 1
                return "aggregated"

            # Drop noise below hard floor
            if tx.significance < 0.05 and tx.source not in (
                    "meditation", "dream_insight", "conversation",
                    "kin_exchange", "genesis_chain"):
                self._total_dropped += 1
                return "dropped"

            # Mark high-significance for immediate include
            if tx.significance >= self._config.get("seal_significance_immediate", 0.9):
                tx.mempool_action = "include"

        # 4. WAL persist
        self._wal.insert(tx)
        self._total_queued += 1
        return "queued"

    def _evaluate_filter_contracts(self, tx: Transaction) -> Optional[dict]:
        """Evaluate active filter contracts against a TX. Returns action or None."""
        if not self._contract_store or not self._rule_evaluator:
            return None
        try:
            filters = [c for c in self._contract_store.get_all()
                       if c.status == "active" and c.contract_type == "filter"]
            if not filters:
                return None

            # Build TX context for rule evaluation
            ctx = {
                "significance": tx.significance,
                "source": tx.source,
                "fork": tx.fork_name,
                "thought_type": tx.tx_type,
                "epoch_id": tx.epoch_id,
            }

            self._contract_filter_evals += 1
            for contract in filters:
                result = self._rule_evaluator.evaluate(contract.rules, ctx)
                if result:
                    contract.execution_count += 1
                    contract.last_executed = time.time()
                    self._contract_filter_hits += 1
                    return result
        except Exception as e:
            logger.warning("[Mempool] Contract filter error: %s", e, exc_info=True)
        return None

    def _aggregate(self, tx: Transaction) -> None:
        """Accumulate into aggregate bucket. Flush when batch_size reached."""
        fork = tx.fork_name
        if fork not in self._agg_buckets:
            self._agg_buckets[fork] = {}

        agg_key = f"agg_{fork}_{tx.source}"
        bucket = self._agg_buckets[fork].get(agg_key)
        if not bucket:
            bucket = {
                "count": 0, "sig_sum": 0.0, "sig_max": 0.0,
                "epoch_min": tx.epoch_id, "epoch_max": tx.epoch_id,
                "type_counts": {},
            }
            self._agg_buckets[fork][agg_key] = bucket

        bucket["count"] += 1
        bucket["sig_sum"] += tx.significance
        bucket["sig_max"] = max(bucket["sig_max"], tx.significance)
        bucket["epoch_min"] = min(bucket["epoch_min"], tx.epoch_id)
        bucket["epoch_max"] = max(bucket["epoch_max"], tx.epoch_id)

        # Track expression sub-types from content
        etype = tx.content.get("composite", tx.source)
        bucket["type_counts"][etype] = bucket["type_counts"].get(etype, 0) + 1

        # Persist aggregate to WAL for crash recovery
        if bucket["count"] % 10 == 0:  # Persist every 10 events
            self._wal.update_aggregate(
                agg_key, fork, bucket["count"],
                bucket["sig_sum"], bucket["sig_max"],
                bucket["epoch_min"], bucket["epoch_max"],
                bucket["type_counts"])

    def flush_aggregates(self, fork_name: str = None) -> list[Transaction]:
        """Convert mature aggregate buckets into summary TXs."""
        summary_txs = []
        forks = [fork_name] if fork_name else list(self._agg_buckets.keys())

        for fork in forks:
            buckets = self._agg_buckets.get(fork, {})
            flush_keys = []
            for agg_key, bucket in buckets.items():
                if bucket["count"] >= self._agg_batch_size or fork_name:
                    # Create summary TX
                    sig_avg = bucket["sig_sum"] / max(bucket["count"], 1)
                    summary_tx = Transaction(
                        tx_type="episodic",
                        source="expression_aggregate",
                        epoch_id=bucket["epoch_max"],
                        significance=min(1.0, sig_avg * 1.2),  # Slight boost
                        content={
                            "aggregate": True,
                            "count": bucket["count"],
                            "sig_avg": round(sig_avg, 4),
                            "sig_max": round(bucket["sig_max"], 4),
                            "types": bucket["type_counts"],
                            "epoch_range": [bucket["epoch_min"], bucket["epoch_max"]],
                        },
                        neuromod_snapshot={},
                        tags=["aggregate", agg_key],
                        timestamp=time.time(),
                        fork_name=fork,
                        mempool_action="include",
                    )
                    summary_tx.compute_hash()
                    summary_txs.append(summary_tx)
                    flush_keys.append(agg_key)

            for key in flush_keys:
                del buckets[key]
                self._wal.clear_aggregate(key)

        return summary_txs

    def get_sealable(self, fork_name: str) -> list[Transaction]:
        """Get all pending TXs for a fork (from WAL)."""
        return self._wal.get_pending(fork_name)

    def get_pending_forks(self) -> list[str]:
        """Get forks with pending TXs (from WAL + aggregate buckets)."""
        wal_forks = set(self._wal.get_pending_forks())
        agg_forks = set(self._agg_buckets.keys())
        return list(wal_forks | agg_forks)

    def pending_count(self) -> int:
        return self._wal.pending_count()

    def mark_sealed(self, tx_hashes: list[bytes]) -> None:
        self._wal.mark_sealed(tx_hashes)

    def rotate_bloom(self) -> None:
        """Rotate bloom filter (daily)."""
        old_count = self._bloom._count
        self._bloom.clear()
        logger.info("[Mempool] Bloom filter rotated (was %d items)", old_count)

    def get_stats(self) -> dict:
        agg_total = sum(
            b["count"] for buckets in self._agg_buckets.values()
            for b in buckets.values())
        stats = {
            "total_submitted": self._total_submitted,
            "total_queued": self._total_queued,
            "total_aggregated": self._total_aggregated,
            "total_duplicates": self._total_duplicates,
            "total_dropped": self._total_dropped,
            "pending_wal": self._wal.pending_count(),
            "pending_aggregates": agg_total,
            "bloom_count": self._bloom._count,
            "bloom_size_kb": self._bloom.size_kb,
            "contract_filter_evals": self._contract_filter_evals,
            "contract_filter_hits": self._contract_filter_hits,
        }
        return stats

    def close(self):
        self._wal.close()


# ═══════════════════════════════════════════════════════════════════════
# BlockBuilder — Seals mempool TXs into blocks
# ═══════════════════════════════════════════════════════════════════════

class BlockBuilder:
    """Seals pending TXs into batched blocks via existing TimeChain.commit_block()."""

    def __init__(self, timechain, config: dict):
        self._tc = timechain
        self._config = config
        self._total_sealed = 0

    def seal_fork(self, mempool: Mempool, fork_name: str,
                  trigger: str, current_epoch: int,
                  send_queue, worker_name: str) -> Optional:
        """Seal all pending TXs for a fork into one batched block.

        Returns the sealed Block or None if nothing to seal.
        """
        # Collect sealable TXs
        txs = mempool.get_sealable(fork_name)
        # Flush any mature aggregates for this fork
        agg_txs = mempool.flush_aggregates(fork_name)
        all_txs = txs + agg_txs

        if not all_txs:
            return None

        fork_id = FORK_IDS.get(fork_name)
        if fork_id is None:
            # Try sidechain lookup
            fork_id = self._tc.get_sidechain_for_topic(fork_name)
            if fork_id is None:
                logger.warning("[BlockBuilder] Unknown fork: %s", fork_name)
                return None

        # Compute TX merkle root (lazy — only at seal time)
        tx_hashes = [tx.tx_hash for tx in all_txs]
        tx_merkle = self._merkle_tree(tx_hashes) if tx_hashes else b"\x00" * 32

        # Compute significance histogram (10 buckets: 0-0.1, 0.1-0.2, ..., 0.9-1.0)
        histogram = self._compute_histogram([tx.significance for tx in all_txs])

        # Compute summary stats
        sigs = [tx.significance for tx in all_txs]
        sig_avg = sum(sigs) / len(sigs) if sigs else 0.0
        sig_max = max(sigs) if sigs else 0.0
        epoch_ids = [tx.epoch_id for tx in all_txs if tx.epoch_id > 0]
        epoch_range = [min(epoch_ids), max(epoch_ids)] if epoch_ids else [current_epoch, current_epoch]

        # Build batched block content
        # Individual TX data packed into content for full verifiability
        tx_summaries = []
        for tx in all_txs:
            tx_summaries.append({
                "hash": tx.tx_hash.hex()[:16],
                "type": tx.tx_type,
                "source": tx.source,
                "sig": round(tx.significance, 3),
                "epoch": tx.epoch_id,
                "tags": tx.tags[:3],  # Truncate to keep block size manageable
            })

        content = {
            "v2": True,
            "sealed_by": trigger,
            "tx_count": len(all_txs),
            "tx_merkle_root": tx_merkle.hex(),
            "sig_histogram": histogram,
            "sig_avg": round(sig_avg, 4),
            "sig_max": round(sig_max, 4),
            "epoch_range": epoch_range,
            "tx_summaries": tx_summaries,
        }

        # Use the most common thought_type from TXs
        thought_types = {}
        for tx in all_txs:
            thought_types[tx.tx_type] = thought_types.get(tx.tx_type, 0) + 1
        primary_type = max(thought_types, key=thought_types.get) if thought_types else "episodic"

        # Aggregate neuromod snapshot (average across TXs that have it)
        nm_sums = {}
        nm_count = 0
        for tx in all_txs:
            if tx.neuromod_snapshot:
                for k, v in tx.neuromod_snapshot.items():
                    if v:
                        nm_sums[k] = nm_sums.get(k, 0.0) + float(v)
                nm_count += 1
        avg_neuromods = {k: v / nm_count for k, v in nm_sums.items()} if nm_count else {}

        # Import BlockPayload from timechain
        from titan_plugin.logic.timechain import BlockPayload

        block_payload = BlockPayload(
            thought_type=primary_type,
            source=f"v2_batch_{trigger}",
            content=content,
            significance=sig_avg,
            confidence=sig_max,
            tags=["v2_batch", trigger, fork_name],
        )

        # Commit via existing TimeChain
        block = self._tc.commit_block(
            fork_id=fork_id,
            epoch_id=current_epoch,
            payload=block_payload,
            pot_nonce=1,  # Batched blocks bypass individual PoT
            chi_spent=sum(tx.significance * 0.001 for tx in all_txs),
            neuromod_state=avg_neuromods,
        )

        if block:
            # Mark TXs as sealed in WAL
            mempool.mark_sealed(tx_hashes)
            self._total_sealed += 1

            logger.info(
                "[BlockBuilder] Sealed %s block #%d: %d TXs, sig_avg=%.3f, "
                "trigger=%s, merkle=%s",
                fork_name, block.header.block_height, len(all_txs),
                sig_avg, trigger, tx_merkle.hex()[:12])

            # Emit TIMECHAIN_SEALED event
            if send_queue:
                try:
                    send_queue.put_nowait({
                        "type": "TIMECHAIN_SEALED",
                        "src": worker_name,
                        "dst": "all",
                        "ts": time.time(),
                        "payload": {
                            "fork": fork_name,
                            "fork_id": fork_id,
                            "block_height": block.header.block_height,
                            "block_hash": block.block_hash.hex(),
                            "tx_count": len(all_txs),
                            "tx_merkle_root": tx_merkle.hex(),
                            "sig_avg": sig_avg,
                            "trigger": trigger,
                        },
                    })
                except Exception:
                    pass

        return block

    def seal_all_forks(self, mempool: Mempool, trigger: str,
                       current_epoch: int,
                       send_queue, worker_name: str) -> int:
        """Seal all forks that have pending TXs. Returns sealed block count."""
        sealed = 0
        for fork_name in mempool.get_pending_forks():
            block = self.seal_fork(
                mempool, fork_name, trigger, current_epoch,
                send_queue, worker_name)
            if block:
                sealed += 1
        return sealed

    def _merkle_tree(self, hashes: list[bytes]) -> bytes:
        """Compute merkle root from TX hashes."""
        if not hashes:
            return b"\x00" * 32
        if len(hashes) == 1:
            return hashes[0]
        # Pad to even
        if len(hashes) % 2:
            hashes = hashes + [hashes[-1]]
        next_level = []
        for i in range(0, len(hashes), 2):
            combined = hashlib.sha256(hashes[i] + hashes[i + 1]).digest()
            next_level.append(combined)
        return self._merkle_tree(next_level)

    def _compute_histogram(self, significances: list[float]) -> list[int]:
        """10-bucket significance histogram [0-0.1, 0.1-0.2, ..., 0.9-1.0]."""
        hist = [0] * 10
        for s in significances:
            bucket = min(9, int(s * 10))
            hist[bucket] += 1
        return hist


# ═══════════════════════════════════════════════════════════════════════
# GenesisChain — Autobiographical spine
# ═══════════════════════════════════════════════════════════════════════

class GenesisChain:
    """Developmental state snapshots + fork heartbeats on FORK_MAIN.

    Genesis blocks capture WHO Titan is at a point in time:
    neuromods, vocabulary, I-confidence, reasoning, pi-rate, etc.
    Fork heartbeats prove each fork's integrity from genesis alone.
    """

    FORK_MAIN = 0

    def __init__(self, timechain, config: dict):
        self._tc = timechain
        self._config = config
        self._last_seal_time = time.time()
        self._last_seal_epoch = 0
        self._seal_count = 0
        self._fallback_hours = config.get("genesis_seal_fallback_hours", 6)

    def should_seal(self, trigger: str) -> bool:
        """Check if a genesis block should be sealed.

        Genesis seals ONLY on meditation (~4-5x/day) + timer fallback.
        Dream/emotion events trigger fork batch sealing, not genesis.
        """
        if trigger == "meditation":
            return self._config.get("genesis_seal_on_meditation", True)
        if trigger == "timer":
            elapsed_h = (time.time() - self._last_seal_time) / 3600
            return elapsed_h >= self._fallback_hours
        if trigger in ("boot", "shutdown"):
            return True
        return False

    def collect_fork_heartbeats(self) -> list[dict]:
        """Collect current tip hash + height from every active fork."""
        from titan_plugin.logic.timechain import FORK_NAMES
        heartbeats = []
        try:
            for fork_id, (tip_height, tip_hash) in self._tc._fork_tips.items():
                if fork_id == self.FORK_MAIN:
                    continue  # Skip self
                fork_name = FORK_NAMES.get(fork_id, f"fork_{fork_id}")
                heartbeats.append({
                    "fork": fork_name,
                    "fork_id": fork_id,
                    "tip_hash": tip_hash.hex() if isinstance(tip_hash, bytes) else str(tip_hash),
                    "tip_height": tip_height,
                })
        except Exception as e:
            logger.warning("[GenesisChain] Error collecting heartbeats: %s", e)
        return heartbeats

    def seal(self, current_epoch: int, state_snapshot: dict,
             trigger: str, mempool_stats: dict,
             send_queue, worker_name: str,
             cognitive_work: dict = None,
             pot_validator=None) -> Optional:
        """Seal a genesis block on FORK_MAIN with developmental state.

        Extended PoT: genesis blocks carry proof of cognitive work done
        since the last genesis seal, validated through the PoT system.
        """

        fork_heartbeats = self.collect_fork_heartbeats()
        cw = cognitive_work or {}

        # Compute proof hash: SHA256(cognitive_work + state + heartbeats)
        proof_data = json.dumps({
            "cognitive_work": cw,
            "state": state_snapshot,
            "heartbeat_count": len(fork_heartbeats),
            "epoch": current_epoch,
            "trigger": trigger,
        }, sort_keys=True, separators=(",", ":")).encode()
        proof_hash = hashlib.sha256(proof_data).hexdigest()

        content = {
            "v2": True,
            "genesis_version": 2,
            "state": state_snapshot,
            "fork_heartbeats": fork_heartbeats,
            "sealed_by": trigger,
            "seal_number": self._seal_count,
            "tx_summary": mempool_stats,
            "cognitive_work": cw,
            "proof_hash": proof_hash,
        }

        # Extended PoT: run actual validation if validator available
        pot_result = {}
        pot_nonce = 1
        chi_spent = 0.002
        if pot_validator and state_snapshot.get("neuromods"):
            try:
                nm = state_snapshot.get("neuromods", {})
                pot = pot_validator.create_pot(
                    chi_available=state_snapshot.get("chi_total", 1.0),
                    metabolic_drain=0.1,
                    attention=0.8,
                    i_confidence=state_snapshot.get("i_confidence", 0.5),
                    chi_coherence=0.7,
                    neuromods=nm,
                    novelty=0.5,
                    significance=1.0,
                    coherence=0.9,
                    source="genesis_chain",
                    thought_type="main",
                    fork_name="main",
                    pi_curvature=1.0,
                )
                pot_result = {
                    "pot_score": round(pot.pot_score, 4),
                    "threshold": round(pot.threshold, 4),
                    "chi_cost": round(pot.chi_cost, 6),
                    "valid": pot.valid,
                }
                pot_nonce = pot.nonce
                chi_spent = pot.chi_cost
            except Exception as e:
                logger.error("[GenesisChain] PoT validation failed: %s", e, exc_info=True)

        content["proof_of_thought"] = pot_result

        from titan_plugin.logic.timechain import BlockPayload

        payload = BlockPayload(
            thought_type="genesis",
            source="genesis_chain",
            content=content,
            significance=1.0,
            confidence=1.0,
            tags=["genesis", "state_snapshot", trigger],
        )

        block = self._tc.commit_block(
            fork_id=self.FORK_MAIN,
            epoch_id=current_epoch,
            payload=payload,
            pot_nonce=pot_nonce,
            chi_spent=chi_spent,
            neuromod_state=state_snapshot.get("neuromods", {}),
        )

        if block:
            self._last_seal_time = time.time()
            self._last_seal_epoch = current_epoch
            self._seal_count += 1

            logger.info(
                "[GenesisChain] Sealed genesis #%d (epoch=%d, trigger=%s, "
                "%d fork heartbeats, vocab=%s, I=%.3f)",
                self._seal_count, current_epoch, trigger,
                len(fork_heartbeats),
                state_snapshot.get("vocab_size", "?"),
                state_snapshot.get("i_confidence", 0))

            # Emit genesis event
            if send_queue:
                try:
                    send_queue.put_nowait({
                        "type": "TIMECHAIN_GENESIS",
                        "src": worker_name,
                        "dst": "all",
                        "ts": time.time(),
                        "payload": {
                            "block_height": block.header.block_height,
                            "block_hash": block.block_hash.hex(),
                            "trigger": trigger,
                            "seal_number": self._seal_count,
                            "fork_heartbeat_count": len(fork_heartbeats),
                            "state_epoch": current_epoch,
                        },
                    })
                except Exception:
                    pass

        return block


# ═══════════════════════════════════════════════════════════════════════
# TimeChainOrchestrator — Top-level coordinator
# ═══════════════════════════════════════════════════════════════════════

# PERSISTENCE_BY_DESIGN: TimeChainOrchestrator._mempool and _contract_store
# are object references to sub-components (Mempool, ContractStore) that are
# instantiated fresh in __init__ and own their own persistence. The
# references themselves are not state to persist.
class TimeChainOrchestrator:
    """TimeChain v2 orchestrator — mempool + batched sealing + genesis chain.

    Wraps the existing TimeChain instance. All TIMECHAIN_COMMIT messages
    route through submit() → mempool → seal → commit_block().

    Phase 2: Consumer API — recall(), check(), compare(), aggregate()
    allow modules to QUERY memories, not just write them.
    """

    def __init__(self, timechain, data_dir: str, config: dict,
                 send_queue=None, worker_name: str = "timechain",
                 pot_validator=None, api_port: int = 7777,
                 titan_pubkey: str = "", maker_pubkey: str = ""):
        self._tc = timechain
        self._config = config
        self._data_dir = data_dir
        self._send_queue = send_queue
        self._worker_name = worker_name
        self._pot_validator = pot_validator
        self._api_port = api_port
        self._titan_pubkey = titan_pubkey
        self._maker_pubkey = maker_pubkey

        # Components
        self._mempool = Mempool(data_dir, config)
        self._builder = BlockBuilder(timechain, config)
        self._genesis = GenesisChain(timechain, config)

        # Contract store (Phase 3a) — Ed25519-signed contracts on meta fork
        self._contract_store: Optional[ContractStore] = None
        self._rule_evaluator = RuleEvaluator(orchestrator=self)
        if titan_pubkey or maker_pubkey:
            try:
                self._contract_store = ContractStore(
                    timechain, titan_pubkey, maker_pubkey)
                # Bootstrap built-in contracts on first boot
                self._contract_store.bootstrap_builtin_contracts()
                # TUNING-012 v2 Sub-phase C: load Phase C cognitive contracts
                # from titan_plugin/contracts/meta_cognitive/. Verifies the bundled
                # Maker signature (R8) if .bundle_signature.json is present.
                try:
                    self._contract_store.load_meta_cognitive_contracts()
                except Exception as cc_err:
                    logger.warning(
                        "[Orchestrator] load_meta_cognitive_contracts failed: %s",
                        cc_err,
                    )
                logger.info("[Orchestrator] ContractStore loaded — %d contracts, contracts=%d",
                            len(self._contract_store.get_all()),
                            len([c for c in self._contract_store.get_all()
                                 if c.status == "active"]))
                # Pass contract store + evaluator to mempool for filter contracts
                self._mempool.set_contract_filter(
                    self._contract_store, self._rule_evaluator)
                # NOTE: TitanMaker substrate (R8 autoseed) is initialized in
                # create_app() in the MAIN process, not here. The orchestrator
                # runs in the timechain_worker subprocess and a Python singleton
                # cannot span processes. ContractStore + TitanMaker compute the
                # bundle hash from the same on-disk JSON files via the shared
                # helper at titan_plugin.maker.contract_bundle.
            except Exception as e:
                logger.warning("[Orchestrator] ContractStore init failed: %s", e)

        # Per-fork seal timers (fixes global timer starvation bug)
        self._fork_seal_times: dict[str, float] = {}
        self._global_seal_time = time.time()
        self._seal_max_txs = config.get("seal_max_txs", 500)
        self._seal_max_time_s = config.get("seal_max_time_s", 300)
        self._seal_immediate_sig = config.get("seal_significance_immediate", 0.9)

        # State tracking for genesis snapshots
        self._last_neuromods: dict = {}
        self._last_emotion: str = ""
        self._current_epoch: int = 0
        self._tick_count: int = 0

        # Cognitive work tracking — evidence for Extended PoT genesis blocks
        self._cognitive_work = self._empty_cognitive_work()

        # NOTE: Arweave backup is handled by RebirthBackup (TitanCore._backup_loop).
        # The orchestrator handles sealing + genesis + birth block, NOT backup.
        # TimeChain chain files are included in the weekly soul package.

        logger.info(
            "[Orchestrator] Initialized — mempool(bloom=%dKB), "
            "seal(max_txs=%d, max_time=%ds), genesis(fallback=%dh), "
            "api_port=%d, contracts=%d",
            self._mempool._bloom.size_kb,
            self._seal_max_txs, self._seal_max_time_s,
            self._genesis._fallback_hours, self._api_port,
            len(self._contract_store.get_all()) if self._contract_store else 0)

        # Phase 4: Birth block (one-time, after all other init)
        self._ensure_birth_block()

    # ── Phase 4: Birth Block ─────────────────────────────────────────

    def _ensure_birth_block(self):
        """Create genesis birth identity block if missing (one-time).

        Records Titan's birth DNA, constitution, on-chain addresses,
        and prime directives as the foundational block on FORK_MAIN.
        This block is the anchor for restoration and sovereignty proof.
        """
        from titan_plugin.logic.timechain import FORK_MAIN, BlockPayload

        # Check flag file first (fastest, survives index rebuilds)
        _birth_flag = os.path.join(self._data_dir, ".birth_block_created")
        if os.path.exists(_birth_flag):
            logger.info("[Orchestrator] Birth block exists (flag file)")
            return
        # Check index DB for thought_type='birth' (handles pre-flag chains)
        try:
            import sqlite3 as _sq
            _db_path = os.path.join(self._data_dir, "index.db")
            _conn = _sq.connect(_db_path)
            _row = _conn.execute(
                "SELECT block_height FROM block_index "
                "WHERE fork_id = ? AND thought_type = 'birth' LIMIT 1",
                (FORK_MAIN,)).fetchone()
            _conn.close()
            if _row:
                logger.info("[Orchestrator] Birth block exists at height %d", _row[0])
                # Write flag file for next time
                try:
                    with open(_birth_flag, "w") as _bf:
                        _bf.write("exists\n")
                except Exception:
                    pass
                return
        except Exception:
            pass

        # Build birth block content
        import hashlib
        content = {
            "event": "BIRTH_IDENTITY",
            "schema_version": "1.0",
            "titan_id": self._tc._titan_id,
            "architecture_version": "v6-132D",
            "maker_pubkey": self._maker_pubkey,
            "titan_pubkey": self._titan_pubkey,
        }

        # Birth DNA hash
        try:
            from titan_plugin.logic.birth_dna import compute_dna_hash
            content["birth_dna_hash"] = compute_dna_hash()
        except Exception as e:
            logger.warning("[Orchestrator] Birth DNA hash failed: %s", e)
            content["birth_dna_hash"] = ""

        # Constitution hash
        try:
            from titan_plugin.utils.directive_signer import compute_constitution_hash
            content["constitution_hash"] = compute_constitution_hash()
        except Exception:
            content["constitution_hash"] = ""

        # Soul directives hash
        try:
            sig_path = os.path.join("data", "titan_directives.sig")
            if os.path.exists(sig_path):
                with open(sig_path, "rb") as f:
                    content["soul_hash"] = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass

        # Prime directives (must match verify_genesis_integrity expectations)
        content["prime_directives"] = [
            "Sovereign Integrity",
            "Cognitive Safety",
            "Metabolic Preservation",
            "Intellectual Honesty",
            "Chain Respect",
        ]

        # On-chain addresses from config
        content["zk_vault_program_id"] = self._config.get("vault_program_id", "")
        content["genesis_nft_address"] = self._config.get("genesis_nft_address", "")

        # Birth timestamp from existing genesis block or now
        try:
            genesis_block = self._tc.get_block(FORK_MAIN, 0)
            if genesis_block:
                content["born"] = genesis_block.header.timestamp
        except Exception:
            content["born"] = time.time()

        # Commit birth block
        try:
            payload = BlockPayload(
                thought_type="birth",
                source="genesis_chain",
                content=content,
                significance=1.0,
                confidence=1.0,
                tags=["birth", "identity", "sovereign"],
            )
            block = self._tc.commit_block(
                fork_id=FORK_MAIN,
                epoch_id=0,
                payload=payload,
                pot_nonce=0,
                chi_spent=0.0,
                neuromod_state={"DA": 0.5, "5HT": 0.5, "NE": 0.5,
                                "ACh": 0.5, "Endorphin": 0.5, "GABA": 0.3},
            )
            if block:
                logger.warning(
                    "[Orchestrator] *** BIRTH BLOCK CREATED *** "
                    "titan=%s hash=%s height=%d dna=%s",
                    self._tc._titan_id,
                    block.block_hash.hex()[:16],
                    block.header.block_height,
                    content.get("birth_dna_hash", "?")[:16])
                # Write flag file to prevent duplicate creation on next restart
                try:
                    with open(_birth_flag, "w") as _bf:
                        _bf.write(f"{block.block_hash.hex()}\n")
                except Exception:
                    pass
            else:
                logger.error("[Orchestrator] Birth block commit returned None")
        except Exception as e:
            logger.error("[Orchestrator] Birth block creation failed: %s", e)

    @staticmethod
    def _empty_cognitive_work() -> dict:
        return {
            "reasoning_chains": 0,
            "words_learned": 0,
            "experiences_distilled": 0,
            "expression_fires": 0,
            "pi_clusters": 0,
            "fork_blocks_sealed": 0,
            "txs_submitted": 0,
            "txs_aggregated": 0,
        }

    # ── Write API ──

    def submit(self, commit_payload: dict, src: str) -> str:
        """Process a TIMECHAIN_COMMIT payload. Returns action taken."""
        tx = Transaction.from_commit_payload(commit_payload)
        action = self._mempool.submit(tx)

        # Track cognitive work from source
        self._cognitive_work["txs_submitted"] += 1
        if action == "aggregated":
            self._cognitive_work["txs_aggregated"] += 1
        if tx.source.startswith("expression_"):
            self._cognitive_work["expression_fires"] += 1
        elif tx.source in ("meta_reasoning", "reasoning"):
            self._cognitive_work["reasoning_chains"] += 1
        elif tx.source in ("language_teacher", "word_learned"):
            self._cognitive_work["words_learned"] += 1
        elif tx.source in ("experience_orchestrator", "experience"):
            self._cognitive_work["experiences_distilled"] += 1
        elif tx.source == "pi_cluster":
            self._cognitive_work["pi_clusters"] += 1

        # Check for immediate seal trigger (per-fork timer)
        if tx.significance >= self._seal_immediate_sig and action == "queued":
            block = self._builder.seal_fork(
                self._mempool, tx.fork_name, "significance",
                tx.epoch_id, self._send_queue, self._worker_name)
            if block:
                self._fork_seal_times[tx.fork_name] = time.time()
                self._cognitive_work["fork_blocks_sealed"] += 1

        return action

    def tick(self, current_epoch: int, neuromods: dict, is_dreaming: bool):
        """Called every ~5s. Per-fork adaptive sealing + genesis timer."""
        self._current_epoch = current_epoch
        self._last_neuromods = neuromods or {}

        now = time.time()
        total_sealed = 0

        # Per-fork adaptive sealing
        pending_forks = self._mempool.get_pending_forks()
        for fork_name in pending_forks:
            fork_last = self._fork_seal_times.get(fork_name, self._global_seal_time)
            elapsed = now - fork_last

            sealable = self._mempool.get_sealable(fork_name)
            pending = len(sealable)
            if pending == 0:
                continue

            should_seal = False
            trigger = ""

            if pending >= self._seal_max_txs:
                should_seal, trigger = True, "count"
            elif elapsed >= self._seal_max_time_s:
                should_seal, trigger = True, "time"

            if should_seal:
                block = self._builder.seal_fork(
                    self._mempool, fork_name, trigger, current_epoch,
                    self._send_queue, self._worker_name)
                if block:
                    self._fork_seal_times[fork_name] = now
                    self._cognitive_work["fork_blocks_sealed"] += 1
                    total_sealed += 1

        # Periodic status log
        self._tick_count += 1
        if self._tick_count % 100 == 1:
            agg_total = sum(
                b["count"] for buckets in self._mempool._agg_buckets.values()
                for b in buckets.values())
            pending_total = self._mempool.pending_count()
            logger.info(
                "[Orchestrator] tick #%d: epoch=%d pending=%d agg=%d "
                "forks=%d sealed=%d contracts(eval=%d hit=%d)",
                self._tick_count, current_epoch, pending_total,
                agg_total, len(pending_forks), total_sealed,
                self._mempool._contract_filter_evals,
                self._mempool._contract_filter_hits)

        # Genesis timer check
        if self._genesis.should_seal("timer"):
            self._seal_genesis(current_epoch, neuromods, "timer")

        # Bloom filter rotation
        if self._mempool._bloom.should_rotate():
            self._mempool.rotate_bloom()

        # Persist contract stats for API (every 20 ticks = ~100s)
        if self._tick_count % 20 == 0 and self._contract_store:
            self._persist_contract_stats()

    # ── Event handlers ──

    def on_meditation_complete(self, epoch_id: int, state_snapshot: dict = None):
        """Seal all forks + genesis on meditation (primary genesis trigger)."""
        sealed = self._builder.seal_all_forks(
            self._mempool, "meditation", epoch_id,
            self._send_queue, self._worker_name)
        self._cognitive_work["fork_blocks_sealed"] += sealed

        # Reset all fork timers
        now = time.time()
        for fork in self._fork_seal_times:
            self._fork_seal_times[fork] = now
        self._global_seal_time = now

        # Seal genesis with full state + cognitive work + PoT
        self._seal_genesis(epoch_id, self._last_neuromods, "meditation")

        logger.info("[Orchestrator] Meditation seal: %d fork blocks + genesis", sealed)

        # NOTE: Arweave backup is handled by RebirthBackup (in TitanCore._backup_loop),
        # which processes MEDITATION_COMPLETE trigger files. TimeChain chain files are
        # included in the weekly soul package — no separate upload needed here.
        # The orchestrator's role is sealing + genesis, not backup.

    # ── Phase 4: Integrated Backup ──────────────────────────────────

    def _trigger_backup(self, epoch_id: int):
        """Trigger Arweave backup in background thread (non-blocking).

        Backup takes 5-30s (disk read + compress + upload) and MUST NOT
        block the main worker loop. Runs in a daemon thread.
        """
        import threading

        def _do_backup():
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                tx_id = loop.run_until_complete(self._backup.snapshot_to_arweave())
                loop.close()
                if tx_id:
                    self._last_backup_ts = time.time()
                    logger.warning(
                        "[Orchestrator] *** ARWEAVE BACKUP COMPLETE *** "
                        "tx=%s epoch=%d blocks=%d",
                        tx_id[:24] if tx_id else "none", epoch_id,
                        self._tc.total_blocks)
                    if self._send_queue:
                        # INTENTIONAL_BROADCAST: observability-only Arweave-
                        # backup confirmation. Frontend dashboard + audit
                        # consumer the stream; no in-process handler needed.
                        self._send_queue.put({
                            "type": "TIMECHAIN_BACKUP_COMPLETE",
                            "src": self._worker_name, "dst": "all",
                            "ts": time.time(),
                            "payload": {
                                "tx_id": tx_id, "epoch_id": epoch_id,
                                "blocks": self._tc.total_blocks,
                            },
                        })
                else:
                    logger.error("[Orchestrator] Arweave backup FAILED at epoch=%d", epoch_id)
            except Exception as e:
                logger.error("[Orchestrator] Arweave backup error: %s", e)

        t = threading.Thread(target=_do_backup, daemon=True, name="arweave-backup")
        t.start()
        logger.info("[Orchestrator] Arweave backup started (background, epoch=%d)", epoch_id)

    # UNUSED_PUBLIC_API: reserved for TIMECHAIN-ANCHOR-WIRING — shipped
    # half-built in commit a45f18d (2026-04-11 Phase 4 Step 3) but never
    # activated: zero callers + zero handlers. Current on-chain write path
    # is the agency-initiated MemoInscribeHelper (autonomous memo decisions).
    # Full activation awaits rFP_physical_time_anchoring.md ratification
    # (currently DRAFT) — write side AND read side (block-ts → oBody
    # physical-time ingestion) need matched design before wiring. See
    # DEFERRED: TIMECHAIN-ANCHOR-WIRING for the full deferral record.
    def _anchor_to_solana(self, epoch_id: int):
        """Emit TIMECHAIN_ANCHOR event for spirit_loop to inscribe on-chain.

        Reuses the existing Solana memo infrastructure — spirit_loop picks up
        the anchor data and includes it in the next epoch memo inscription.
        """
        try:
            merkle = self._tc.compute_merkle_root()
            if not merkle:
                return
            if self._send_queue:
                self._send_queue.put({
                    "type": "TIMECHAIN_ANCHOR",
                    "src": self._worker_name, "dst": "spirit",
                    "ts": time.time(),
                    "payload": {
                        "epoch_id": epoch_id,
                        "merkle_root": merkle.hex()[:32],
                        "total_blocks": self._tc.total_blocks,
                        "trigger": "meditation",
                    },
                })
        except Exception as e:
            logger.debug("[Orchestrator] Solana anchor emit failed: %s", e)

    def on_dream_boundary(self, epoch_id: int, is_start: bool):
        """Seal fork batches on dream boundary (NOT genesis)."""
        sealed = self._builder.seal_all_forks(
            self._mempool,
            "dream_start" if is_start else "dream_end",
            epoch_id, self._send_queue, self._worker_name)
        if sealed:
            self._cognitive_work["fork_blocks_sealed"] += sealed
            now = time.time()
            for fork in self._mempool.get_pending_forks():
                self._fork_seal_times[fork] = now

    def on_emotion_shift(self, epoch_id: int, new_emotion: str):
        """Track emotion for next genesis snapshot. No seal triggered."""
        self._last_emotion = new_emotion

    def shutdown(self):
        """Flush mempool on graceful shutdown."""
        if self._mempool.pending_count() > 0:
            sealed = self._builder.seal_all_forks(
                self._mempool, "shutdown", self._current_epoch,
                self._send_queue, self._worker_name)
            logger.info("[Orchestrator] Shutdown flush: %d blocks sealed", sealed)

        # Seal final genesis
        self._seal_genesis(self._current_epoch, self._last_neuromods, "shutdown")
        self._mempool.close()

    # ── Consumer API (Phase 2) ──

    def recall(self, query: "RecallQuery") -> list[dict]:
        """Query blocks from TimeChain index. Returns list of block metadata."""
        fork_id = FORK_IDS.get(query.fork) if query.fork else None
        epoch_range = None
        if query.since_epoch:
            epoch_range = (query.since_epoch, 999_999_999)
        elif query.since_hours and query.since_hours > 0:
            # ~1600 epochs/hour is approximate T1 rate
            since_epoch = max(0, self._current_epoch - int(query.since_hours * 1600))
            epoch_range = (since_epoch, 999_999_999)

        results = self._tc.query_blocks(
            thought_type=query.thought_type or None,
            source=query.source or None,
            fork_id=fork_id,
            tag=query.tag_contains or None,
            epoch_range=epoch_range,
            limit=query.limit * 2,  # Over-fetch for post-filter
        )

        # Post-filter significance range
        if query.significance_min > 0 or query.significance_max < 1.0:
            results = [
                r for r in results
                if query.significance_min <= r.get("significance", 0) <= query.significance_max
            ]

        # Sort
        results.sort(
            key=lambda r: r.get("epoch_id", 0),
            reverse=(query.order == "desc"))

        results = results[:query.limit]

        # Optionally load full block content (expensive)
        if query.include_content:
            for r in results:
                try:
                    bh = r.get("block_hash", "")
                    if isinstance(bh, str):
                        bh = bytes.fromhex(bh)
                    block = self._tc.get_block_by_hash(bh)
                    if block and hasattr(block, "payload"):
                        r["content"] = block.payload.content
                except Exception:
                    pass

        return results

    def check(self, query: "CheckQuery") -> bool:
        """Quick boolean existence check."""
        rq = RecallQuery(
            fork=query.fork, source=query.source,
            tag_contains=query.tag_contains,
            since_hours=query.since_hours,
            since_epoch=query.since_epoch,
            significance_min=query.significance_min,
            limit=1)
        return len(self.recall(rq)) > 0

    def compare(self, query: "CompareQuery") -> dict:
        """Compare a state field across two time windows via genesis blocks."""
        epochs_per_hour = 1600
        now_epoch = self._current_epoch

        # Window A: recent
        a_start = now_epoch - int(query.window_a_hours * epochs_per_hour)
        a_results = self.recall(RecallQuery(
            fork=query.fork or "main", thought_type="genesis",
            since_epoch=max(0, a_start),
            include_content=True, limit=20, order="desc"))

        # Window B: older
        b_start = now_epoch - int(query.window_b_hours * epochs_per_hour)
        b_end = a_start
        b_results = []
        try:
            b_raw = self._tc.query_blocks(
                thought_type="genesis", fork_id=0,
                epoch_range=(max(0, b_start), b_end), limit=20)
            # Load content for window B
            for r in b_raw:
                try:
                    bh = r.get("block_hash", "")
                    if isinstance(bh, str):
                        bh = bytes.fromhex(bh)
                    block = self._tc.get_block_by_hash(bh)
                    if block and hasattr(block, "payload"):
                        r["content"] = block.payload.content
                except Exception:
                    pass
            b_results = b_raw
        except Exception:
            pass

        def _extract(results, fld):
            for r in results:
                content = r.get("content", {})
                state = content.get("state", {})
                if fld in state:
                    return float(state[fld])
            return None

        a_val = _extract(a_results, query.field)
        b_val = _extract(b_results, query.field)

        if a_val is None or b_val is None:
            return {"direction": "unknown", "delta": 0,
                    "a_value": a_val, "b_value": b_val}

        delta = a_val - b_val
        if delta > 0.01:
            direction = "rising"
        elif delta < -0.01:
            direction = "falling"
        else:
            direction = "flat"
        return {"direction": direction, "delta": round(delta, 4),
                "a_value": a_val, "b_value": b_val}

    def similar(self, query: "SimilarQuery") -> list[dict]:
        """F-phase (rFP §9.2): semantic embedding similarity over blocks.

        Session 1: linear-scan + numpy cosine over blocks within the recency
        window, skipping any without `payload.context_embedding`. Returns
        empty list when no blocks have embeddings yet.

        Session 2 will accelerate via FAISS index (config path
        faiss_index_path; rebuilt during dream-consolidation per rFP §9.3).

        Latency guarantee: ≤50ms for ~10k recent blocks (rFP §9.5). The
        recency filter (default 72h) keeps the scanned set bounded.
        """
        try:
            import numpy as _np
        except ImportError:
            return []

        q_vec = list(query.query_vector or [])
        if not q_vec:
            return []
        try:
            q = _np.asarray(q_vec, dtype=_np.float32)
        except Exception:
            return []
        q_norm = float(_np.linalg.norm(q))
        if q_norm <= 0:
            return []

        # Pull candidate blocks via the existing recall path (bounded by
        # recency window + fork + thought_type filters) then load their
        # full content to access context_embedding.
        recall_q = RecallQuery(
            fork=query.fork,
            thought_type=query.thought_type,
            since_hours=query.since_hours or 0,
            since_epoch=query.since_epoch or 0,
            limit=min(max(10, query.limit * 20), 2000),  # scan a superset
            order="desc",
            include_content=True,
        )
        try:
            candidates = self.recall(recall_q)
        except Exception as e:
            logger.debug("[Orchestrator] similar() recall failed: %s", e)
            return []

        scored: list = []
        threshold = float(query.threshold)
        for blk in candidates:
            payload = blk.get("payload") or blk.get("content") or {}
            if not isinstance(payload, dict):
                continue
            emb = payload.get("context_embedding")
            if not emb or not isinstance(emb, list):
                continue
            if query.embedding_version > 0:
                v = payload.get("embedding_version", 0)
                if v != query.embedding_version:
                    continue
            if len(emb) != len(q_vec):
                continue  # dimension mismatch → skip
            try:
                b = _np.asarray(emb, dtype=_np.float32)
                b_norm = float(_np.linalg.norm(b))
                if b_norm <= 0:
                    continue
                sim = float(_np.dot(q, b) / (q_norm * b_norm))
            except Exception:
                continue
            if sim >= threshold:
                scored.append({
                    "block_hash": blk.get("block_hash", ""),
                    "similarity": round(sim, 4),
                    "fork": blk.get("fork", ""),
                    "thought_type": blk.get("thought_type", ""),
                    "epoch": blk.get("epoch_id", 0),
                    "payload_summary": str(payload.get("summary",
                                                         ""))[:160],
                })

        scored.sort(key=lambda d: -d["similarity"])
        return scored[:max(1, int(query.limit))]

    def aggregate(self, query: "AggregateQuery") -> float:
        """Aggregate over blocks via direct SQL on index. Efficient."""
        fork_id = FORK_IDS.get(query.fork) if query.fork else None
        epoch_start = 0
        if query.since_hours and query.since_hours > 0:
            epoch_start = max(0, self._current_epoch - int(query.since_hours * 1600))

        # Whitelist fields to prevent SQL injection
        safe_fields = {"significance", "chi_spent", "epoch_id", "block_height"}
        agg_field = query.field if query.field in safe_fields else "significance"

        sql_ops = {
            "count": "COUNT(*)",
            "sum": f"SUM({agg_field})",
            "avg": f"AVG({agg_field})",
            "max": f"MAX({agg_field})",
            "min": f"MIN({agg_field})",
        }
        sql_op = sql_ops.get(query.op, "COUNT(*)")

        conditions = ["epoch_id >= ?"]
        params: list = [epoch_start]
        if fork_id is not None:
            conditions.append("fork_id = ?")
            params.append(fork_id)
        if query.source:
            conditions.append("source = ?")
            params.append(query.source)
        if query.thought_type:
            conditions.append("thought_type = ?")
            params.append(query.thought_type)

        where = " AND ".join(conditions)
        sql = f"SELECT {sql_op} FROM block_index WHERE {where}"

        try:
            row = self._tc._index_db.execute(sql, params).fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception as e:
            logger.warning("[Orchestrator] aggregate() failed: %s", e)
            return 0.0

    # ── Contract API (Phase 3a) ──

    def deploy_contract(self, contract_dict: dict,
                        keypair=None) -> tuple[bool, str]:
        """Deploy a signed contract. Returns (success, reason)."""
        if not self._contract_store:
            return False, "contract_store_not_initialized"

        contract = Contract.from_dict(contract_dict)

        # If keypair provided, sign now (API-based deployment)
        if keypair:
            sign_contract(contract, keypair)

        ok, reason = self._contract_store.deploy(
            contract, self._send_queue, self._worker_name)

        # Hot-reload after deploy
        if ok and self._contract_store:
            self._contract_store.hot_reload()

        return ok, reason

    def approve_contract(self, contract_id: str,
                         maker_keypair) -> tuple[bool, str]:
        """Maker approves a pending Titan-authored contract."""
        if not self._contract_store:
            return False, "contract_store_not_initialized"
        return self._contract_store.approve(
            contract_id, maker_keypair,
            self._send_queue, self._worker_name)

    def reject_contract(self, contract_id: str,
                        reason: str = "") -> tuple[bool, str]:
        """Maker rejects a pending contract."""
        if not self._contract_store:
            return False, "contract_store_not_initialized"
        return self._contract_store.reject(contract_id, reason)

    def list_contracts(self, contract_type: str = None,
                       status: str = None) -> list[dict]:
        """List contracts, optionally filtered."""
        if not self._contract_store:
            return []
        contracts = self._contract_store.get_all()
        if contract_type:
            contracts = [c for c in contracts if c.contract_type == contract_type]
        if status:
            contracts = [c for c in contracts if c.status == status]
        return [c.to_dict() for c in contracts]

    def get_contract(self, contract_id: str) -> Optional[dict]:
        """Get a single contract by ID."""
        if not self._contract_store:
            return None
        c = self._contract_store.get(contract_id)
        return c.to_dict() if c else None

    # ── Stats ──

    def get_stats(self) -> dict:
        """Full orchestrator stats for monitoring."""
        return {
            "v2_enabled": True,
            "mempool": self._mempool.get_stats(),
            "builder": {
                "total_sealed_blocks": self._builder._total_sealed,
            },
            "genesis": {
                "seal_count": self._genesis._seal_count,
                "last_seal_epoch": self._genesis._last_seal_epoch,
                "last_seal_age_s": time.time() - self._genesis._last_seal_time,
            },
            "sealing": {
                "fork_timers": {
                    k: round(time.time() - v, 1)
                    for k, v in self._fork_seal_times.items()
                },
                "max_txs": self._seal_max_txs,
                "max_time_s": self._seal_max_time_s,
            },
            "cognitive_work": dict(self._cognitive_work),
            "contracts": self._contract_store.get_stats() if self._contract_store else {},
            "contract_executions": {
                c.contract_id: {
                    "type": c.contract_type,
                    "executions": c.execution_count,
                    "last_executed": round(time.time() - c.last_executed, 1)
                        if c.last_executed else None,
                }
                for c in (self._contract_store.get_all()
                          if self._contract_store else [])
                if c.status == "active"
            },
        }

    # ── Internal ──

    def _persist_contract_stats(self):
        """Write live contract stats to JSON file for API access."""
        try:
            stats = {
                "contracts": {},
                "mempool_filter_evals": self._mempool._contract_filter_evals,
                "mempool_filter_hits": self._mempool._contract_filter_hits,
                "updated_at": time.time(),
            }
            for c in self._contract_store.get_all():
                if c.status == "active":
                    stats["contracts"][c.contract_id] = {
                        "type": c.contract_type,
                        "execution_count": c.execution_count,
                        "last_executed": c.last_executed,
                        "description": c.description,
                    }
            path = os.path.join(self._data_dir, "contract_stats.json")
            with open(path, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.warning("[Orchestrator] contract_stats write failed: %s", e)

    def _seal_genesis(self, epoch_id: int, neuromods: dict, trigger: str):
        """Seal genesis block with enriched state + cognitive work + PoT."""
        state = self._build_state_snapshot(epoch_id, neuromods)
        self._genesis.seal(
            epoch_id, state, trigger,
            self._mempool.get_stats(),
            self._send_queue, self._worker_name,
            cognitive_work=dict(self._cognitive_work),
            pot_validator=self._pot_validator)
        # Reset cognitive work counters after genesis seal
        self._cognitive_work = self._empty_cognitive_work()
        # P3c: Evaluate genesis trigger contracts after seal
        self._evaluate_trigger_contracts("genesis", state)

    def _evaluate_trigger_contracts(self, contract_type: str, state: dict):
        """Evaluate trigger/genesis contracts after seal events (P3c).

        Fires bus events when contract conditions match developmental state.
        """
        if not self._contract_store or not self._rule_evaluator:
            return
        try:
            contracts = [c for c in self._contract_store.get_all()
                         if c.status == "active" and c.contract_type == contract_type]
            if not contracts:
                return

            # For genesis contracts, load recent genesis states for TREND/DELTA
            genesis_states = []
            if contract_type == "genesis":
                try:
                    blocks = self._tc.query_blocks(
                        thought_type="genesis", fork_id=0, limit=10)
                    for b in blocks:
                        bh = b.get("block_hash", "")
                        if isinstance(bh, str):
                            bh = bytes.fromhex(bh)
                        block = self._tc.get_block_by_hash(bh)
                        if block and hasattr(block, "payload"):
                            gs = block.payload.content.get("state", {})
                            if gs:
                                genesis_states.append(gs)
                except Exception:
                    pass

            for contract in contracts:
                result = self._rule_evaluator.evaluate(
                    contract.rules, state, genesis_states=genesis_states)
                if result and result.get("action") == "emit":
                    event_name = result.get("event", "CONTRACT_TRIGGER")
                    contract.execution_count += 1
                    contract.last_executed = time.time()
                    logger.info("[Orchestrator] Contract '%s' triggered: %s",
                                contract.contract_id, event_name)
                    if self._send_queue:
                        try:
                            self._send_queue.put_nowait({
                                "type": event_name,
                                "src": self._worker_name,
                                "dst": "all",
                                "ts": time.time(),
                                "payload": {
                                    "contract": contract.contract_id,
                                    "data": result.get("data", {}),
                                    "state_epoch": state.get("epoch_id", 0),
                                },
                            })
                        except Exception:
                            pass
        except Exception as e:
            logger.error("[Orchestrator] Trigger eval error: %s", e, exc_info=True)

    def _build_state_snapshot(self, epoch_id: int, neuromods: dict) -> dict:
        """Build developmental state from live API (enriched, not hardcoded)."""
        snapshot = {
            "epoch_id": epoch_id,
            "neuromods": {
                k: round(float(neuromods.get(k, 0)), 4)
                for k in ("DA", "5HT", "NE", "GABA", "ACh")
            },
            "emotion": self._last_emotion,
            "vocab_size": 0, "productive_vocab": 0, "i_confidence": 0.0,
            "pi_rate": 0.0, "cluster_count": 0, "dream_cycles": 0,
            "meta_chains": 0, "reasoning_commit_rate": 0.0,
            "chi_total": 0.0, "metabolic_tier": "", "sol_balance": 0.0,
        }

        # Enrich from localhost API (same process, fast).
        # CRITICAL: each call is in its own try/except so a slow /health
        # (mainnet Solana RPC) does NOT prevent inner-trinity enrichment.
        # Without this split, contract evaluators see all-zero context
        # whenever /health times out → contracts never fire on mainnet.
        import httpx
        base = f"http://127.0.0.1:{self._api_port}"

        # 1) SOL balance from /health (can be slow on mainnet — short timeout)
        try:
            r = httpx.get(f"{base}/health", timeout=3.0)
            if r.status_code == 200:
                h = r.json().get("data", {})
                snapshot["sol_balance"] = h.get("sol_balance", 0)
        except Exception as e:
            logger.debug("[Orchestrator] /health enrichment skipped: %s", e)

        # 2) Everything critical from inner-trinity (local, fast)
        try:
            r = httpx.get(f"{base}/v4/inner-trinity", timeout=5.0)
            if r.status_code == 200:
                d = r.json().get("data", {})
                lang = d.get("language", {})
                snapshot["vocab_size"] = lang.get("vocab_total", 0)
                snapshot["productive_vocab"] = lang.get("vocab_producible", 0)
                snapshot["i_confidence"] = d.get("msl", {}).get("i_confidence", 0.0)
                snapshot["dream_cycles"] = d.get("dreaming", {}).get("cycle_count", 0)
                meta = d.get("meta_reasoning", {})
                snapshot["meta_chains"] = meta.get("total_chains", 0)
                buf = meta.get("buffer_size", 0)
                chains = meta.get("total_chains", 0)
                snapshot["reasoning_commit_rate"] = round(
                    chains / max(1, buf), 3) if buf else 0.0
                snapshot["chi_total"] = d.get("chi", {}).get("total", 0.0)
                snapshot["metabolic_tier"] = d.get("metabolic_tier", "")
                pi = d.get("pi_heartbeat", {})
                snapshot["pi_rate"] = pi.get("heartbeat_ratio", 0.0)
                snapshot["cluster_count"] = pi.get("cluster_count", 0)
            logger.info(
                "[Orchestrator] State enriched: vocab=%d I=%.3f pi=%.3f chains=%d dream_cycles=%d",
                snapshot["vocab_size"], snapshot["i_confidence"],
                snapshot["pi_rate"], snapshot["meta_chains"],
                snapshot["dream_cycles"])
        except Exception as e:
            logger.warning("[Orchestrator] inner-trinity enrichment FAILED: %s", e)

        return snapshot
