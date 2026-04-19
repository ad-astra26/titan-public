"""
TimeChain — Proof of Thought Tripartite Memory Architecture.

Cryptographically provable temporal ordering for all of Titan's memory formation.
Every thought entering long-term memory passes a Proof of Thought (PoT) validation
gate, then is committed as an immutable block chained by SHA-256 hashes.

Synthesis of: Satoshi's blockchain (2008) x ACT-R tripartite memory (Anderson, 1998)
              x Titan's metabolic/neuromod architecture.

Architecture:
  - Fork 0: Main Chain (heartbeats, Merkle roots, lifecycle events)
  - Fork 1: Declarative (facts, concepts, vocabulary, knowledge)
  - Fork 2: Procedural (skills, strategies, HAOV rules, reasoning patterns)
  - Fork 3: Episodic (experiences, conversations, dreams, expression)
  - Fork 4: Meta (thoughts about thinking, "I AM" events, self-model updates)
  - Fork 5+: Topic Sidechains (auto-created when tag count >= 3 on a primary fork)

Storage: append-only binary files (one per fork) + SQLite index for queries.
"""

import hashlib
import logging
import os
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import msgpack

logger = logging.getLogger("TimeChain")

# ── Constants ──────────────────────────────────────────────────────────
CHAIN_VERSION = 1
HEADER_SIZE = 128  # fixed header bytes
MAX_CROSS_REFS = 4
GENESIS_PREV_HASH = b"\x00" * 32
GENESIS_NEUROMOD_HASH = b"\x00" * 16

# Fork IDs for primary forks
FORK_MAIN = 0
FORK_DECLARATIVE = 1
FORK_PROCEDURAL = 2
FORK_EPISODIC = 3
FORK_META = 4
FORK_CONVERSATION = 5   # Verified external outputs (chat, X, Telegram, agent)
FORK_SIDECHAIN_START = 6

# Primary fork names
FORK_NAMES = {
    FORK_MAIN: "main",
    FORK_DECLARATIVE: "declarative",
    FORK_PROCEDURAL: "procedural",
    FORK_EPISODIC: "episodic",
    FORK_META: "meta",
    FORK_CONVERSATION: "conversation",
}

# Sidechain auto-fork threshold
SIDECHAIN_TAG_THRESHOLD = 3

# Header struct format (128 bytes total):
# B   = version (1)
# Q   = block_height (8)
# d   = timestamp (8)
# Q   = epoch_id (8)
# 32s = prev_hash (32)
# 32s = payload_hash (32)
# H   = fork_id (2)
# Q   = fork_parent (8)
# I   = pot_nonce (4)
# f   = chi_spent (4)
# 16s = neuromod_hash (16)
# B   = cross_refs count (1)
# 4s  = reserved (4)
# Total: 1+8+8+8+32+32+2+8+4+4+16+1+4 = 128
HEADER_FORMAT = ">BQdQ32s32sHQIf16sB4s"
assert struct.calcsize(HEADER_FORMAT) == HEADER_SIZE

# Cross-reference entry: fork_id(H=2) + block_height(Q=8) = 10 bytes
CROSS_REF_FORMAT = ">HQ"
CROSS_REF_SIZE = 10


def sha256(data: bytes) -> bytes:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    """Compute SHA-256 hash as hex string."""
    return hashlib.sha256(data).hexdigest()


# ── Data Structures ────────────────────────────────────────────────────

@dataclass
class CrossRef:
    """Reference to a block on another fork."""
    fork_id: int
    block_height: int

    def to_bytes(self) -> bytes:
        return struct.pack(CROSS_REF_FORMAT, self.fork_id, self.block_height)

    @classmethod
    def from_bytes(cls, data: bytes) -> "CrossRef":
        fork_id, height = struct.unpack(CROSS_REF_FORMAT, data)
        return cls(fork_id=fork_id, block_height=height)


@dataclass
class BlockHeader:
    """Fixed-size 128-byte block header."""
    version: int
    block_height: int
    timestamp: float
    epoch_id: int
    prev_hash: bytes           # SHA-256 of previous block header
    payload_hash: bytes        # SHA-256 of payload bytes
    fork_id: int
    fork_parent: int           # block height on fork 0 where this fork branched
    pot_nonce: int
    chi_spent: float
    neuromod_hash: bytes       # hash of 6-neuromod state
    cross_ref_count: int
    reserved: bytes = b"\x00" * 4

    def to_bytes(self) -> bytes:
        return struct.pack(
            HEADER_FORMAT,
            self.version,
            self.block_height,
            self.timestamp,
            self.epoch_id,
            self.prev_hash,
            self.payload_hash,
            self.fork_id,
            self.fork_parent,
            self.pot_nonce,
            self.chi_spent,
            self.neuromod_hash,
            self.cross_ref_count,
            self.reserved,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "BlockHeader":
        vals = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
        return cls(
            version=vals[0],
            block_height=vals[1],
            timestamp=vals[2],
            epoch_id=vals[3],
            prev_hash=vals[4],
            payload_hash=vals[5],
            fork_id=vals[6],
            fork_parent=vals[7],
            pot_nonce=vals[8],
            chi_spent=vals[9],
            neuromod_hash=vals[10],
            cross_ref_count=vals[11],
            reserved=vals[12],
        )

    def compute_hash(self) -> bytes:
        """Hash of the header bytes — this IS the block hash."""
        return sha256(self.to_bytes())


@dataclass
class BlockPayload:
    """Variable-size msgpack-encoded block payload."""
    thought_type: str        # declarative|procedural|episodic|meta|genesis
    source: str              # perception|reasoning|social|dream|knowledge|teacher|kin|maker
    content: dict            # the actual thought content
    felt_tensor: bytes = b"" # compressed 130D felt tensor
    significance: float = 0.0
    confidence: float = 0.0
    tags: list[str] = field(default_factory=list)
    db_ref: str = ""         # pointer to existing SQLite/DuckDB row

    def to_bytes(self) -> bytes:
        return msgpack.packb({
            "t": self.thought_type,
            "s": self.source,
            "c": self.content,
            "f": self.felt_tensor,
            "g": self.significance,
            "n": self.confidence,
            "a": self.tags,
            "d": self.db_ref,
        }, use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BlockPayload":
        d = msgpack.unpackb(data, raw=False)
        return cls(
            thought_type=d.get("t", ""),
            source=d.get("s", ""),
            content=d.get("c", {}),
            felt_tensor=d.get("f", b""),
            significance=d.get("g", 0.0),
            confidence=d.get("n", 0.0),
            tags=d.get("a", []),
            db_ref=d.get("d", ""),
        )


@dataclass
class Block:
    """A complete TimeChain block: header + cross-refs + payload."""
    header: BlockHeader
    cross_refs: list[CrossRef] = field(default_factory=list)
    payload: BlockPayload = field(default_factory=lambda: BlockPayload(
        thought_type="", source="", content={}))

    @property
    def block_hash(self) -> bytes:
        return self.header.compute_hash()

    @property
    def block_hash_hex(self) -> str:
        return self.block_hash.hex()

    def to_bytes(self) -> bytes:
        """Serialize entire block to bytes: header + cross_refs + payload_len + payload."""
        parts = [self.header.to_bytes()]
        for ref in self.cross_refs[:MAX_CROSS_REFS]:
            parts.append(ref.to_bytes())
        payload_bytes = self.payload.to_bytes()
        # 4-byte payload length prefix
        parts.append(struct.pack(">I", len(payload_bytes)))
        parts.append(payload_bytes)
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Block":
        """Deserialize block from bytes."""
        offset = 0
        header = BlockHeader.from_bytes(data[offset:offset + HEADER_SIZE])
        offset += HEADER_SIZE

        cross_refs = []
        for _ in range(header.cross_ref_count):
            ref = CrossRef.from_bytes(data[offset:offset + CROSS_REF_SIZE])
            cross_refs.append(ref)
            offset += CROSS_REF_SIZE

        payload_len = struct.unpack(">I", data[offset:offset + 4])[0]
        offset += 4
        payload = BlockPayload.from_bytes(data[offset:offset + payload_len])
        offset += payload_len

        return cls(header=header, cross_refs=cross_refs, payload=payload)

    def total_size(self) -> int:
        """Total serialized size of this block."""
        return (HEADER_SIZE
                + self.header.cross_ref_count * CROSS_REF_SIZE
                + 4  # payload length prefix
                + len(self.payload.to_bytes()))


# ── TimeChain Core ─────────────────────────────────────────────────────

# PERSISTENCE_BY_DESIGN: TimeChain._fork_tips is an observability index
# saved for fast queries but rebuilt from the authoritative chain files on
# load (chain files themselves are the source of truth for fork tips).
class TimeChain:
    """Append-only hash-chained memory architecture with tripartite forks.

    Each fork is stored as a separate binary file. An SQLite index provides
    fast querying by tag, type, epoch, source, etc. The chain files are the
    source of truth; the index is always rebuildable.
    """

    def __init__(self, data_dir: str, titan_id: str = "T1",
                 auto_sidechain: bool = True):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._sidechain_dir = self._data_dir / "sidechains"
        self._sidechain_dir.mkdir(exist_ok=True)
        self._titan_id = titan_id
        self._auto_sidechain = auto_sidechain

        # Fork tip cache: fork_id -> (height, hash)
        self._fork_tips: dict[int, tuple[int, bytes]] = {}

        # Tag frequency tracking for auto-sidechain
        self._tag_counts: dict[int, dict[str, int]] = {}

        # Total block count across all forks
        self._total_blocks = 0

        # Initialize index database
        self._index_db_path = self._data_dir / "index.db"
        self._init_index_db()

        # Load fork registry and tip state from index
        self._load_fork_state()

    # ── Database Init ──────────────────────────────────────────────────

    def _init_index_db(self):
        """Create index database tables if they don't exist."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS block_index (
                    block_hash   BLOB PRIMARY KEY,
                    fork_id      INTEGER NOT NULL,
                    block_height INTEGER NOT NULL,
                    timestamp    REAL NOT NULL,
                    epoch_id     INTEGER NOT NULL,
                    thought_type TEXT,
                    source       TEXT,
                    significance REAL,
                    chi_spent    REAL,
                    neuromod_da  REAL,
                    neuromod_ach REAL,
                    neuromod_ne  REAL,
                    tags         TEXT,
                    cross_refs   TEXT,
                    db_ref       TEXT,
                    compacted    INTEGER DEFAULT 0,
                    file_offset  INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_bi_fork_height
                    ON block_index(fork_id, block_height);
                CREATE INDEX IF NOT EXISTS idx_bi_epoch
                    ON block_index(epoch_id);
                CREATE INDEX IF NOT EXISTS idx_bi_type
                    ON block_index(thought_type);
                CREATE INDEX IF NOT EXISTS idx_bi_tags
                    ON block_index(tags);
                CREATE INDEX IF NOT EXISTS idx_bi_source
                    ON block_index(source);

                CREATE TABLE IF NOT EXISTS fork_registry (
                    fork_id      INTEGER PRIMARY KEY,
                    fork_name    TEXT NOT NULL,
                    fork_type    TEXT NOT NULL,
                    parent_fork  INTEGER,
                    parent_block INTEGER,
                    created_at   REAL NOT NULL,
                    tip_height   INTEGER DEFAULT 0,
                    tip_hash     BLOB,
                    topic        TEXT,
                    compacted    INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp      REAL NOT NULL,
                    epoch_id       INTEGER NOT NULL,
                    merkle_root    BLOB NOT NULL,
                    total_blocks   INTEGER NOT NULL,
                    fork_tips      TEXT NOT NULL
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def _load_fork_state(self):
        """Load fork tips and tag counts from index DB, reconcile with chain files."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            rows = conn.execute(
                "SELECT fork_id, tip_height, tip_hash FROM fork_registry"
            ).fetchall()
            for fork_id, height, tip_hash in rows:
                # Fix genesis-era bug: tip_height=0 with NULL hash means "no blocks"
                if height == 0 and tip_hash is None:
                    self._fork_tips[fork_id] = (-1, GENESIS_PREV_HASH)
                    conn.execute(
                        "UPDATE fork_registry SET tip_height=-1 WHERE fork_id=? AND tip_hash IS NULL",
                        (fork_id,))
                else:
                    self._fork_tips[fork_id] = (height, tip_hash or GENESIS_PREV_HASH)

            # Reconcile: verify chain file tip matches index tip
            for fork_id, (idx_height, idx_hash) in list(self._fork_tips.items()):
                path = self._get_chain_file_path(fork_id)
                if not path.exists():
                    continue
                file_tip = self._get_file_tip_height(path)
                if file_tip is not None and file_tip != idx_height:
                    gap = abs(file_tip - idx_height)
                    # Small gaps (1-3 blocks) are expected from atomic commit timing
                    if gap > 3:
                        logger.warning(
                            "[TimeChain] Fork %d tip mismatch: index=%d, file=%d (gap=%d) — "
                            "reconciling from chain file",
                            fork_id, idx_height, file_tip, gap)
                    else:
                        logger.debug(
                            "[TimeChain] Fork %d tip reconcile: index=%d → file=%d",
                            fork_id, idx_height, file_tip)
                    # Read actual last block hash from file
                    last_block = self._read_last_block_from_file(fork_id, path)
                    if last_block:
                        new_hash = last_block.block_hash
                        self._fork_tips[fork_id] = (file_tip, new_hash)
                        conn.execute(
                            "UPDATE fork_registry SET tip_height=?, tip_hash=? "
                            "WHERE fork_id=?",
                            (file_tip, new_hash, fork_id))
                        conn.commit()

            # Count total blocks
            row = conn.execute("SELECT COUNT(*) FROM block_index").fetchone()
            self._total_blocks = row[0] if row else 0

            # Load tag counts for sidechain auto-creation
            for fork_id in range(FORK_DECLARATIVE, FORK_META + 1):
                tag_rows = conn.execute(
                    "SELECT tags FROM block_index WHERE fork_id = ? AND tags != ''",
                    (fork_id,)
                ).fetchall()
                counts: dict[str, int] = {}
                for (tags_json,) in tag_rows:
                    try:
                        tags = eval(tags_json) if tags_json else []
                        for tag in tags:
                            counts[tag] = counts.get(tag, 0) + 1
                    except Exception:
                        pass
                self._tag_counts[fork_id] = counts
        finally:
            conn.close()

    # ── Genesis ────────────────────────────────────────────────────────

    def create_genesis(self, genesis_content: dict,
                       birth_timestamp: float = 0.0) -> Block:
        """Create the genesis block (block 0 on main chain).

        The genesis block is the root of the entire TimeChain.
        It contains Titan's birth certificate and prime directives.
        All subsequent blocks chain back to genesis via hashes.

        Args:
            genesis_content: Birth certificate data (maker pubkey, soul hash, etc.)
            birth_timestamp: Titan's birth timestamp (defaults to now)

        Returns:
            The genesis Block.
        """
        if FORK_MAIN in self._fork_tips and self._fork_tips[FORK_MAIN][0] >= 0:
            existing = self._fork_tips.get(FORK_MAIN, (-1, b""))
            if existing[0] >= 0:
                logger.info("[TimeChain] Genesis already exists (height %d)", existing[0])
                return self.get_block(FORK_MAIN, 0)

        ts = birth_timestamp or time.time()

        payload = BlockPayload(
            thought_type="genesis",
            source="maker",
            content=genesis_content,
            significance=1.0,
            confidence=1.0,
            tags=["genesis", self._titan_id],
            db_ref="",
        )
        payload_bytes = payload.to_bytes()

        header = BlockHeader(
            version=CHAIN_VERSION,
            block_height=0,
            timestamp=ts,
            epoch_id=0,
            prev_hash=GENESIS_PREV_HASH,
            payload_hash=sha256(payload_bytes),
            fork_id=FORK_MAIN,
            fork_parent=0,
            pot_nonce=0,  # No PoT for genesis
            chi_spent=0.0,
            neuromod_hash=GENESIS_NEUROMOD_HASH,
            cross_ref_count=0,
        )

        block = Block(header=header, cross_refs=[], payload=payload)

        # Register primary forks
        self._register_primary_forks(ts)

        # Write genesis to main chain file
        self._append_block_to_file(block)
        self._index_block(block, file_offset=0)
        self._update_fork_tip(FORK_MAIN, 0, block.block_hash)
        self._total_blocks += 1

        logger.info("[TimeChain] Genesis created — hash=%s titan=%s",
                    block.block_hash_hex[:16], self._titan_id)
        return block

    def _register_primary_forks(self, created_at: float):
        """Register the 5 primary forks in the fork registry."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            for fork_id, fork_name in FORK_NAMES.items():
                conn.execute("""
                    INSERT OR IGNORE INTO fork_registry
                    (fork_id, fork_name, fork_type, parent_fork, parent_block,
                     created_at, tip_height, tip_hash, topic, compacted)
                    VALUES (?, ?, 'primary', 0, 0, ?, -1, NULL, NULL, 0)
                """, (fork_id, fork_name, created_at))
                if fork_id not in self._fork_tips:
                    self._fork_tips[fork_id] = (-1, GENESIS_PREV_HASH)
            conn.commit()
        finally:
            conn.close()

    @property
    def has_genesis(self) -> bool:
        tip = self._fork_tips.get(FORK_MAIN, (-1, b""))
        return tip[0] >= 0

    @property
    def genesis_hash(self) -> bytes:
        if not self.has_genesis:
            return GENESIS_PREV_HASH
        genesis = self.get_block(FORK_MAIN, 0)
        return genesis.block_hash if genesis else GENESIS_PREV_HASH

    # ── Block Commit ───────────────────────────────────────────────────

    def commit_block(self, fork_id: int, epoch_id: int,
                     payload: BlockPayload, pot_nonce: int,
                     chi_spent: float, neuromod_state: dict,
                     cross_refs: Optional[list[CrossRef]] = None,
                     timestamp: Optional[float] = None) -> Optional[Block]:
        """Commit a new block to the specified fork.

        Args:
            fork_id: Target fork (FORK_DECLARATIVE, FORK_PROCEDURAL, etc.)
            epoch_id: Current epoch at time of commitment
            payload: Block content
            pot_nonce: Proof of Thought validation nonce
            chi_spent: Metabolic energy consumed
            neuromod_state: Dict with DA, ACh, NE, 5HT, GABA, endorphin levels
            cross_refs: Optional cross-fork references (max 4)
            timestamp: Override timestamp (defaults to now)

        Returns:
            The committed Block, or None if fork doesn't exist.
        """
        if fork_id not in self._fork_tips:
            logger.warning("[TimeChain] Fork %d not registered", fork_id)
            return None

        refs = (cross_refs or [])[:MAX_CROSS_REFS]
        ts = timestamp or time.time()

        # Get previous block hash for this fork
        tip_height, tip_hash = self._fork_tips[fork_id]
        new_height = tip_height + 1 if tip_height >= 0 else 0

        # For non-main forks with no blocks yet, prev_hash = genesis hash
        if tip_height < 0:
            prev_hash = self.genesis_hash
        else:
            prev_hash = tip_hash

        # Compute neuromod hash
        neuromod_hash = self._compute_neuromod_hash(neuromod_state)

        # Build block
        payload_bytes = payload.to_bytes()
        header = BlockHeader(
            version=CHAIN_VERSION,
            block_height=new_height,
            timestamp=ts,
            epoch_id=epoch_id,
            prev_hash=prev_hash,
            payload_hash=sha256(payload_bytes),
            fork_id=fork_id,
            fork_parent=self._get_fork_parent(fork_id),
            pot_nonce=pot_nonce,
            chi_spent=chi_spent,
            neuromod_hash=neuromod_hash,
            cross_ref_count=len(refs),
        )

        block = Block(header=header, cross_refs=refs, payload=payload)

        # Get file offset before writing
        chain_file = self._get_chain_file_path(fork_id)
        file_offset = chain_file.stat().st_size if chain_file.exists() else 0

        # Write to chain file
        self._append_block_to_file(block)

        # Atomic index + fork tip update (single transaction prevents mismatch)
        block_hash = block.block_hash
        self._index_and_update_tip(block, file_offset, neuromod_state,
                                   fork_id, new_height, block_hash)
        self._total_blocks += 1

        # Track tags for auto-sidechain
        if self._auto_sidechain and payload.tags:
            self._track_tags_for_sidechain(fork_id, payload.tags)

        return block

    def _compute_neuromod_hash(self, neuromod_state: dict) -> bytes:
        """Hash the 6-neuromodulator state into 16 bytes."""
        if not neuromod_state:
            return GENESIS_NEUROMOD_HASH
        # Pack 6 float32 values, hash to 16 bytes
        vals = [
            float(neuromod_state.get("DA", 0)),
            float(neuromod_state.get("ACh", 0)),
            float(neuromod_state.get("NE", 0)),
            float(neuromod_state.get("5HT", 0)),
            float(neuromod_state.get("GABA", 0)),
            float(neuromod_state.get("endorphin", 0)),
        ]
        raw = struct.pack(">6f", *vals)
        return hashlib.md5(raw).digest()  # 16 bytes

    def _get_fork_parent(self, fork_id: int) -> int:
        """Get the main chain block height where this fork branched."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            row = conn.execute(
                "SELECT parent_block FROM fork_registry WHERE fork_id = ?",
                (fork_id,)
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    # ── File I/O ───────────────────────────────────────────────────────

    def _get_chain_file_path(self, fork_id: int) -> Path:
        """Get the binary file path for a fork."""
        name = FORK_NAMES.get(fork_id)
        if name:
            return self._data_dir / f"chain_{name}.bin"
        return self._sidechain_dir / f"sc_{fork_id:04d}.bin"

    def _get_file_tip_height(self, path: Path) -> Optional[int]:
        """Scan a chain file to find the last block's height."""
        try:
            last_height = None
            with open(path, "rb") as f:
                while True:
                    header_data = f.read(HEADER_SIZE)
                    if len(header_data) < HEADER_SIZE:
                        break
                    header = BlockHeader.from_bytes(header_data)
                    last_height = header.block_height
                    # Skip cross-refs + payload
                    f.read(header.cross_ref_count * CROSS_REF_SIZE)
                    len_data = f.read(4)
                    if len(len_data) < 4:
                        break
                    payload_len = struct.unpack(">I", len_data)[0]
                    f.read(payload_len)
            return last_height
        except Exception:
            return None

    def _read_last_block_from_file(self, fork_id: int,
                                    path: Path) -> Optional[Block]:
        """Read the last block from a chain file."""
        try:
            last_offset = 0
            with open(path, "rb") as f:
                while True:
                    pos = f.tell()
                    header_data = f.read(HEADER_SIZE)
                    if len(header_data) < HEADER_SIZE:
                        break
                    header = BlockHeader.from_bytes(header_data)
                    last_offset = pos
                    # Skip cross-refs + payload
                    f.read(header.cross_ref_count * CROSS_REF_SIZE)
                    len_data = f.read(4)
                    if len(len_data) < 4:
                        break
                    payload_len = struct.unpack(">I", len_data)[0]
                    f.read(payload_len)
            return self._read_block_at_offset(fork_id, last_offset)
        except Exception:
            return None

    def _append_block_to_file(self, block: Block):
        """Append a block to its fork's chain file."""
        path = self._get_chain_file_path(block.header.fork_id)
        with open(path, "ab") as f:
            f.write(block.to_bytes())

    def _read_block_at_offset(self, fork_id: int, offset: int) -> Optional[Block]:
        """Read a single block from a chain file at the given byte offset."""
        path = self._get_chain_file_path(fork_id)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                # Read header
                header_data = f.read(HEADER_SIZE)
                if len(header_data) < HEADER_SIZE:
                    return None
                header = BlockHeader.from_bytes(header_data)

                # Read cross-refs
                cross_ref_data = f.read(header.cross_ref_count * CROSS_REF_SIZE)
                cross_refs = []
                for i in range(header.cross_ref_count):
                    start = i * CROSS_REF_SIZE
                    ref = CrossRef.from_bytes(
                        cross_ref_data[start:start + CROSS_REF_SIZE])
                    cross_refs.append(ref)

                # Read payload length + payload
                len_data = f.read(4)
                if len(len_data) < 4:
                    return None
                payload_len = struct.unpack(">I", len_data)[0]
                payload_data = f.read(payload_len)
                if len(payload_data) < payload_len:
                    return None
                payload = BlockPayload.from_bytes(payload_data)

                return Block(header=header, cross_refs=cross_refs, payload=payload)
        except Exception as e:
            logger.error("[TimeChain] Read error at fork=%d offset=%d: %s",
                        fork_id, offset, e)
            return None

    # ── Index ──────────────────────────────────────────────────────────

    def _index_and_update_tip(self, block: Block, file_offset: int,
                              neuromod_state: Optional[dict],
                              fork_id: int, height: int,
                              block_hash: bytes):
        """Atomically index a block AND update fork tip in one transaction.

        This prevents the race condition where block_index has the block but
        fork_registry tip lags behind (causing mismatch warnings on next init).
        """
        nm = neuromod_state or {}
        refs_str = str([f"{r.fork_id}:{r.block_height}" for r in block.cross_refs])
        tags_str = str(block.payload.tags) if block.payload.tags else ""

        conn = sqlite3.connect(str(self._index_db_path))
        try:
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                INSERT OR REPLACE INTO block_index
                (block_hash, fork_id, block_height, timestamp, epoch_id,
                 thought_type, source, significance, chi_spent,
                 neuromod_da, neuromod_ach, neuromod_ne,
                 tags, cross_refs, db_ref, compacted, file_offset)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """, (
                block.block_hash,
                block.header.fork_id,
                block.header.block_height,
                block.header.timestamp,
                block.header.epoch_id,
                block.payload.thought_type,
                block.payload.source,
                block.payload.significance,
                block.header.chi_spent,
                nm.get("DA", 0.0),
                nm.get("ACh", 0.0),
                nm.get("NE", 0.0),
                tags_str,
                refs_str,
                block.payload.db_ref,
                file_offset,
            ))
            conn.execute("""
                UPDATE fork_registry SET tip_height = ?, tip_hash = ?
                WHERE fork_id = ?
            """, (height, block_hash, fork_id))
            conn.commit()
        except Exception as e:
            logger.error("[TimeChain] Atomic index+tip failed fork=%d height=%d: %s",
                         fork_id, height, e)
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            conn.close()
        # Always update in-memory tip (file write already succeeded)
        self._fork_tips[fork_id] = (height, block_hash)

    def _index_block(self, block: Block, file_offset: int,
                     neuromod_state: Optional[dict] = None):
        """Add a block to the SQLite index (standalone, used by rebuild)."""
        nm = neuromod_state or {}
        refs_str = str([f"{r.fork_id}:{r.block_height}" for r in block.cross_refs])
        tags_str = str(block.payload.tags) if block.payload.tags else ""

        conn = sqlite3.connect(str(self._index_db_path))
        try:
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("""
                INSERT OR REPLACE INTO block_index
                (block_hash, fork_id, block_height, timestamp, epoch_id,
                 thought_type, source, significance, chi_spent,
                 neuromod_da, neuromod_ach, neuromod_ne,
                 tags, cross_refs, db_ref, compacted, file_offset)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """, (
                block.block_hash,
                block.header.fork_id,
                block.header.block_height,
                block.header.timestamp,
                block.header.epoch_id,
                block.payload.thought_type,
                block.payload.source,
                block.payload.significance,
                block.header.chi_spent,
                nm.get("DA", 0.0),
                nm.get("ACh", 0.0),
                nm.get("NE", 0.0),
                tags_str,
                refs_str,
                block.payload.db_ref,
                file_offset,
            ))
            conn.commit()
        finally:
            conn.close()

    def _update_fork_tip(self, fork_id: int, height: int, block_hash: bytes):
        """Update the fork tip in memory and DB (standalone, used by reconciliation)."""
        self._fork_tips[fork_id] = (height, block_hash)
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("""
                UPDATE fork_registry SET tip_height = ?, tip_hash = ?
                WHERE fork_id = ?
            """, (height, block_hash, fork_id))
            conn.commit()
        finally:
            conn.close()

    # ── Block Retrieval ────────────────────────────────────────────────

    def get_block(self, fork_id: int, height: int) -> Optional[Block]:
        """Retrieve a block by fork and height."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            row = conn.execute(
                "SELECT file_offset FROM block_index WHERE fork_id = ? AND block_height = ?",
                (fork_id, height)
            ).fetchone()
            if not row:
                return None
            return self._read_block_at_offset(fork_id, row[0])
        finally:
            conn.close()

    def get_block_by_hash(self, block_hash: bytes) -> Optional[Block]:
        """Retrieve a block by its hash."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            row = conn.execute(
                "SELECT fork_id, file_offset FROM block_index WHERE block_hash = ?",
                (block_hash,)
            ).fetchone()
            if not row:
                return None
            return self._read_block_at_offset(row[0], row[1])
        finally:
            conn.close()

    def get_fork_tip(self, fork_id: int) -> tuple[int, bytes]:
        """Get the tip (height, hash) for a fork."""
        return self._fork_tips.get(fork_id, (-1, GENESIS_PREV_HASH))

    def get_recent_blocks(self, fork_id: int, n: int = 10) -> list[dict]:
        """Get the N most recent blocks on a fork (metadata only, from index)."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            rows = conn.execute("""
                SELECT block_hash, block_height, timestamp, epoch_id,
                       thought_type, source, significance, chi_spent, tags
                FROM block_index
                WHERE fork_id = ?
                ORDER BY block_height DESC
                LIMIT ?
            """, (fork_id, n)).fetchall()
            return [
                {
                    "block_hash": r[0].hex() if isinstance(r[0], bytes) else r[0],
                    "height": r[1],
                    "timestamp": r[2],
                    "epoch_id": r[3],
                    "thought_type": r[4],
                    "source": r[5],
                    "significance": r[6],
                    "chi_spent": r[7],
                    "tags": r[8],
                }
                for r in rows
            ]
        finally:
            conn.close()

    # ── Chain Verification ─────────────────────────────────────────────

    def verify_fork(self, fork_id: int) -> tuple[bool, str]:
        """Verify hash chain integrity for an entire fork.

        Reads every block from the chain file and verifies:
        1. prev_hash chains correctly
        2. payload_hash matches actual payload

        Returns (valid, error_message).
        """
        path = self._get_chain_file_path(fork_id)
        if not path.exists():
            return True, "Fork file does not exist (no blocks)"

        prev_hash = GENESIS_PREV_HASH
        # For non-main forks, first block's prev_hash = genesis hash
        if fork_id != FORK_MAIN and self.has_genesis:
            prev_hash = self.genesis_hash

        height = 0
        height_offset = 0  # Non-zero for forks with genesis-era off-by-one
        try:
            with open(path, "rb") as f:
                while True:
                    pos = f.tell()
                    header_data = f.read(HEADER_SIZE)
                    if len(header_data) == 0:
                        break  # EOF
                    if len(header_data) < HEADER_SIZE:
                        return False, f"Truncated header at height {height}"

                    header = BlockHeader.from_bytes(header_data)

                    # On first block, detect genesis-era height offset (blocks
                    # created before the tip_height=-1 fix start at 1, not 0)
                    if height == 0 and header.block_height == 1 and fork_id != FORK_MAIN:
                        height_offset = 1

                    # Verify height sequence (accounting for offset)
                    if header.block_height != height + height_offset:
                        return False, (f"Height mismatch at pos {pos}: "
                                      f"expected {height + height_offset}, "
                                      f"got {header.block_height}")

                    # Verify prev_hash chain
                    if height == 0 and fork_id == FORK_MAIN:
                        expected_prev = GENESIS_PREV_HASH
                    elif height == 0 and height_offset == 0:
                        expected_prev = self.genesis_hash
                    elif height == 0 and height_offset > 0:
                        # Genesis-era forks used GENESIS_PREV_HASH as first prev
                        expected_prev = GENESIS_PREV_HASH
                    else:
                        expected_prev = prev_hash

                    if header.prev_hash != expected_prev:
                        return False, (f"Chain break at height {height}: "
                                      f"prev_hash mismatch (expected "
                                      f"{expected_prev.hex()[:12]}..., got "
                                      f"{header.prev_hash.hex()[:12]}...)")

                    # Read cross-refs (skip)
                    f.read(header.cross_ref_count * CROSS_REF_SIZE)

                    # Read payload and verify hash
                    len_data = f.read(4)
                    if len(len_data) < 4:
                        return False, f"Truncated payload length at height {height}"
                    payload_len = struct.unpack(">I", len_data)[0]
                    payload_data = f.read(payload_len)
                    if len(payload_data) < payload_len:
                        return False, f"Truncated payload at height {height}"

                    if sha256(payload_data) != header.payload_hash:
                        return False, f"Payload tampered at height {height}"

                    prev_hash = sha256(header_data)
                    height += 1

        except Exception as e:
            return False, f"Verification error: {e}"

        return True, f"Fork {fork_id} valid ({height} blocks)"

    def verify_all(self) -> tuple[bool, list[str]]:
        """Verify all forks. Returns (all_valid, list_of_results)."""
        results = []
        all_valid = True
        for fork_id in sorted(self._fork_tips.keys()):
            valid, msg = self.verify_fork(fork_id)
            results.append(f"Fork {fork_id} ({FORK_NAMES.get(fork_id, f'sc_{fork_id}')}): {msg}")
            if not valid:
                all_valid = False
        return all_valid, results

    # ── Merkle Checkpointing ───────────────────────────────────────────

    def compute_merkle_root(self) -> bytes:
        """Compute Merkle root of all fork tip hashes.

        This is what gets anchored on Solana — a single hash representing
        the entire TimeChain state at this moment.
        """
        tip_hashes = []
        for fork_id in sorted(self._fork_tips.keys()):
            _, tip_hash = self._fork_tips[fork_id]
            tip_hashes.append(tip_hash)

        if not tip_hashes:
            return GENESIS_PREV_HASH

        # Build Merkle tree
        return self._merkle_tree(tip_hashes)

    def _merkle_tree(self, hashes: list[bytes]) -> bytes:
        """Compute Merkle root from a list of hashes."""
        if not hashes:
            return sha256(b"empty")
        if len(hashes) == 1:
            return hashes[0]

        # Pad to even
        if len(hashes) % 2 != 0:
            hashes.append(hashes[-1])

        next_level = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i + 1]
            next_level.append(sha256(combined))

        return self._merkle_tree(next_level)

    def create_checkpoint(self, epoch_id: int) -> dict:
        """Create a Merkle checkpoint of current state."""
        merkle_root = self.compute_merkle_root()
        tips = {str(k): v[0] for k, v in self._fork_tips.items()}

        conn = sqlite3.connect(str(self._index_db_path))
        try:
            conn.execute("""
                INSERT INTO checkpoints
                (timestamp, epoch_id, merkle_root, total_blocks, fork_tips)
                VALUES (?, ?, ?, ?, ?)
            """, (time.time(), epoch_id, merkle_root, self._total_blocks,
                  str(tips)))
            conn.commit()
        finally:
            conn.close()

        return {
            "merkle_root": merkle_root.hex(),
            "total_blocks": self._total_blocks,
            "epoch_id": epoch_id,
            "fork_tips": tips,
        }

    # ── Topic Sidechain Management ─────────────────────────────────────

    def _track_tags_for_sidechain(self, fork_id: int, tags: list[str]):
        """Track tag frequency and auto-create sidechains."""
        if fork_id not in self._tag_counts:
            self._tag_counts[fork_id] = {}

        for tag in tags:
            self._tag_counts[fork_id][tag] = \
                self._tag_counts[fork_id].get(tag, 0) + 1

            if (self._tag_counts[fork_id][tag] >= SIDECHAIN_TAG_THRESHOLD
                    and not self._sidechain_exists(tag)):
                self._create_sidechain(tag, fork_id)

    def _sidechain_exists(self, topic: str) -> bool:
        """Check if a topic sidechain already exists."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            row = conn.execute(
                "SELECT fork_id FROM fork_registry WHERE topic = ?",
                (topic,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def _create_sidechain(self, topic: str, parent_fork: int) -> int:
        """Create a new topic sidechain."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            # Find next available fork_id
            row = conn.execute(
                "SELECT MAX(fork_id) FROM fork_registry"
            ).fetchone()
            new_id = max(FORK_SIDECHAIN_START, (row[0] or 0) + 1)

            parent_tip = self._fork_tips.get(parent_fork, (-1, b""))[0]

            conn.execute("""
                INSERT INTO fork_registry
                (fork_id, fork_name, fork_type, parent_fork, parent_block,
                 created_at, tip_height, tip_hash, topic, compacted)
                VALUES (?, ?, 'topic_sidechain', ?, ?, ?, -1, NULL, ?, 0)
            """, (new_id, f"topic:{topic}", parent_fork, parent_tip,
                  time.time(), topic))
            conn.commit()

            self._fork_tips[new_id] = (-1, self.genesis_hash)
            logger.info("[TimeChain] Auto-sidechain created: fork=%d topic=%s "
                       "parent=%d", new_id, topic, parent_fork)
            return new_id
        finally:
            conn.close()

    def get_sidechain_for_topic(self, topic: str) -> Optional[int]:
        """Get the fork_id for a topic sidechain, if it exists."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            row = conn.execute(
                "SELECT fork_id FROM fork_registry WHERE topic = ?",
                (topic,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    # ── Query Interface ────────────────────────────────────────────────

    def query_blocks(self, thought_type: Optional[str] = None,
                     source: Optional[str] = None,
                     fork_id: Optional[int] = None,
                     tag: Optional[str] = None,
                     epoch_range: Optional[tuple[int, int]] = None,
                     neuromod_filter: Optional[dict] = None,
                     limit: int = 50) -> list[dict]:
        """Query the block index with filters."""
        conditions = []
        params = []

        if thought_type:
            conditions.append("thought_type = ?")
            params.append(thought_type)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if fork_id is not None:
            conditions.append("fork_id = ?")
            params.append(fork_id)
        if tag:
            conditions.append("tags LIKE ?")
            params.append(f"%'{tag}'%")
        if epoch_range:
            conditions.append("epoch_id BETWEEN ? AND ?")
            params.extend(epoch_range)
        if neuromod_filter:
            for key, (op, val) in neuromod_filter.items():
                col_map = {"DA": "neuromod_da", "ACh": "neuromod_ach",
                           "NE": "neuromod_ne"}
                col = col_map.get(key)
                if col:
                    conditions.append(f"{col} {op} ?")
                    params.append(val)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""SELECT block_hash, fork_id, block_height, timestamp, epoch_id,
                         thought_type, source, significance, chi_spent, tags
                  FROM block_index WHERE {where}
                  ORDER BY timestamp DESC LIMIT ?"""
        params.append(limit)

        conn = sqlite3.connect(str(self._index_db_path))
        try:
            rows = conn.execute(sql, params).fetchall()
            return [
                {
                    "block_hash": r[0].hex() if isinstance(r[0], bytes) else r[0],
                    "fork_id": r[1],
                    "height": r[2],
                    "timestamp": r[3],
                    "epoch_id": r[4],
                    "thought_type": r[5],
                    "source": r[6],
                    "significance": r[7],
                    "chi_spent": r[8],
                    "tags": r[9],
                }
                for r in rows
            ]
        finally:
            conn.close()

    # ── Stats ──────────────────────────────────────────────────────────

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    def get_fork_stats(self) -> dict:
        """Get statistics for all forks."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            forks = conn.execute("""
                SELECT fork_id, fork_name, fork_type, tip_height, topic, compacted
                FROM fork_registry ORDER BY fork_id
            """).fetchall()

            stats = {}
            for fork_id, name, ftype, tip_h, topic, compacted in forks:
                # Count blocks by type
                row = conn.execute(
                    "SELECT COUNT(*), SUM(chi_spent), AVG(significance) "
                    "FROM block_index WHERE fork_id = ?",
                    (fork_id,)
                ).fetchone()

                stats[fork_id] = {
                    "name": name,
                    "type": ftype,
                    "tip_height": tip_h,
                    "topic": topic,
                    "compacted": bool(compacted),
                    "block_count": row[0] if row else 0,
                    "total_chi_spent": round(row[1] or 0, 6),
                    "avg_significance": round(row[2] or 0, 4),
                }
            return stats
        finally:
            conn.close()

    def get_chain_status(self) -> dict:
        """Get overall TimeChain status for dashboard/monitoring."""
        fork_stats = self.get_fork_stats()
        merkle = self.compute_merkle_root()

        # Total chi spent
        total_chi = sum(f["total_chi_spent"] for f in fork_stats.values())

        # Latest Solana anchor transaction (for Solscan link)
        anchor_info = {}
        try:
            import json as _json
            _anchor_path = os.path.join(os.path.dirname(self._data_dir), "anchor_state.json")
            if os.path.exists(_anchor_path):
                with open(_anchor_path) as _af:
                    _as = _json.load(_af)
                    anchor_info = {
                        "last_tx_sig": _as.get("last_tx_sig", ""),
                        "anchor_count": _as.get("anchor_count", 0),
                        "last_epoch_id": _as.get("last_epoch_id", 0),
                    }
        except Exception:
            pass

        return {
            "titan_id": self._titan_id,
            "genesis_exists": self.has_genesis,
            "genesis_hash": self.genesis_hash.hex() if self.has_genesis else None,
            "total_blocks": self._total_blocks,
            "total_forks": len(fork_stats),
            "total_chi_spent": round(total_chi, 6),
            "merkle_root": merkle.hex(),
            "anchor": anchor_info,
            "forks": fork_stats,
        }

    # ── Dream Compaction ───────────────────────────────────────────────

    def get_compactable_sidechains(self, min_blocks: int = 5) -> list[dict]:
        """Find sidechains eligible for dream compaction."""
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            rows = conn.execute("""
                SELECT fr.fork_id, fr.fork_name, fr.topic, fr.tip_height,
                       COUNT(bi.block_hash) as block_count
                FROM fork_registry fr
                LEFT JOIN block_index bi ON fr.fork_id = bi.fork_id
                    AND bi.compacted = 0
                WHERE fr.fork_type = 'topic_sidechain'
                    AND fr.compacted = 0
                GROUP BY fr.fork_id
                HAVING block_count >= ?
            """, (min_blocks,)).fetchall()

            return [
                {
                    "fork_id": r[0],
                    "name": r[1],
                    "topic": r[2],
                    "tip_height": r[3],
                    "uncompacted_blocks": r[4],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def compact_sidechain(self, fork_id: int, epoch_id: int,
                          summary: dict, neuromod_state: dict) -> Optional[Block]:
        """Compact a sidechain into a summary block on the Meta fork.

        Does NOT delete original blocks — marks them as compacted.
        Creates a compaction block on Meta fork with Merkle root of compacted blocks.
        """
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            # Get all uncompacted blocks on this sidechain
            rows = conn.execute("""
                SELECT block_hash, block_height, significance
                FROM block_index
                WHERE fork_id = ? AND compacted = 0
                ORDER BY block_height
            """, (fork_id,)).fetchall()

            if not rows:
                return None

            # Compute Merkle root of compacted block hashes
            block_hashes = [r[0] for r in rows]
            compaction_merkle = self._merkle_tree(block_hashes)
            avg_sig = sum(r[2] or 0 for r in rows) / len(rows)

            # Get sidechain info
            sc_info = conn.execute(
                "SELECT fork_name, topic FROM fork_registry WHERE fork_id = ?",
                (fork_id,)
            ).fetchone()
            topic = sc_info[1] if sc_info else "unknown"

            # Mark blocks as compacted
            conn.execute(
                "UPDATE block_index SET compacted = 1 WHERE fork_id = ? AND compacted = 0",
                (fork_id,)
            )
            conn.execute(
                "UPDATE fork_registry SET compacted = 1 WHERE fork_id = ?",
                (fork_id,)
            )
            conn.commit()
        finally:
            conn.close()

        # Create compaction block on Meta fork
        payload = BlockPayload(
            thought_type="meta",
            source="dream",
            content={
                "type": "compaction",
                "source_fork": fork_id,
                "topic": topic,
                "blocks_compacted": len(block_hashes),
                "compaction_merkle": compaction_merkle.hex(),
                "avg_significance": round(avg_sig, 4),
                "summary": summary,
            },
            significance=avg_sig,
            tags=[topic, "compaction"],
        )

        return self.commit_block(
            fork_id=FORK_META,
            epoch_id=epoch_id,
            payload=payload,
            pot_nonce=1,  # Compaction always passes PoT
            chi_spent=0.001,  # Minimal chi for dream compaction
            neuromod_state=neuromod_state,
            cross_refs=[CrossRef(fork_id=fork_id, block_height=block_hashes[-1][0]
                                 if isinstance(block_hashes[-1], tuple)
                                 else len(block_hashes) - 1)],
        )

    # ── Rebuild Index ──────────────────────────────────────────────────

    def rebuild_index(self):
        """Rebuild the entire index from chain files.

        Use when index.db is corrupted. Chain files are source of truth.
        """
        logger.info("[TimeChain] Rebuilding index from chain files...")
        # Drop and recreate tables
        conn = sqlite3.connect(str(self._index_db_path))
        try:
            conn.executescript("""
                DELETE FROM block_index;
                DELETE FROM checkpoints;
            """)
            conn.commit()
        finally:
            conn.close()

        self._total_blocks = 0
        self._fork_tips = {}

        # Scan all chain files
        for fork_id, name in FORK_NAMES.items():
            path = self._get_chain_file_path(fork_id)
            if path.exists():
                self._scan_chain_file(fork_id, path)

        # Scan sidechains
        for sc_file in sorted(self._sidechain_dir.glob("sc_*.bin")):
            fork_id_str = sc_file.stem.split("_")[1]
            fork_id = int(fork_id_str)
            self._scan_chain_file(fork_id, sc_file)

        logger.info("[TimeChain] Index rebuilt: %d blocks across %d forks",
                    self._total_blocks, len(self._fork_tips))

    def _scan_chain_file(self, fork_id: int, path: Path):
        """Scan a chain file and index all blocks."""
        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                header_data = f.read(HEADER_SIZE)
                if len(header_data) < HEADER_SIZE:
                    break

                header = BlockHeader.from_bytes(header_data)

                # Skip cross-refs
                f.read(header.cross_ref_count * CROSS_REF_SIZE)

                # Read payload
                len_data = f.read(4)
                if len(len_data) < 4:
                    break
                payload_len = struct.unpack(">I", len_data)[0]
                payload_data = f.read(payload_len)
                if len(payload_data) < payload_len:
                    break

                payload = BlockPayload.from_bytes(payload_data)
                block = Block(header=header, cross_refs=[], payload=payload)

                self._index_block(block, offset)
                self._update_fork_tip(fork_id, header.block_height,
                                      block.block_hash)
                self._total_blocks += 1
