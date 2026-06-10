"""Synthesis Engine plug protocols — interface-complete, implementation-incremental.

Five `Protocol`s + their data classes that fix the engine's shape forever, even
though concrete implementations land incrementally per phase of
`rFP_outer_memory_enhancement.md §18` (see `ARCHITECTURE_synthesis_engine.md §3.5`).

Phase 1 (D-SPEC-123, 2026-05-23): protocols + dataclasses defined here. No
concrete implementations yet — concrete plug registrations land per phase:
  - Phase 2: Timechain SubstratePlug + SC ops (SEARCH/FORK_READ/DIFF/CROSS_REF)
  - Phase 3: Episodic conversational-fork tagging surface
  - Phase 4: Kuzu/FAISS SubstratePlugs + Concept-spine writes
  - Phase 6: TruthOraclePlug concrete (coding_sandbox / solana_rpc / web_api / x_oracle)
  - Phase 6: MeaningOraclePlug concrete (CGN — sole grounding authority per INV-Syn-1)
  - Phase 6: ProofStrategyPlug concrete (Merkle default; ZK targeted, cost-gated)
  - Phase 6: ToolPlug concrete (coding_sandbox + events_teacher + knowledge + x_research)

Binding invariants (SPEC §25.1):
  INV-Syn-1: CGN is sole grounding authority — no plug creates groundings.
  INV-Syn-2: Timechain canonical; SC ops via in-process RuleEvaluator (G19).
  INV-Syn-3: synthesis_worker is sole writer (G21) for activation_state /
             procedural_skills / actr_buffers / synth_status.bin.
  INV-Syn-4: cross-process reads watermark-gated via BridgeRecall (G18).
  INV-Syn-5: reinforcement use-gated — only items the LLM cited/acted upon.

Substrate read calls are READ-ONLY across the process boundary and gated by
the synth_status.bin watermark. Substrate writes happen ONLY inside the
synthesis_worker process (INV-Syn-3 / G21).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, runtime_checkable


# ─────────────────────────────────────────────────────────────────────────
# Shared scalar types
# ─────────────────────────────────────────────────────────────────────────

# Which canonical substrate. Values mirror the fork names used by
# timechain_v2.FORK_IDS where applicable; "kuzu"/"faiss"/"duckdb" are
# non-Timechain substrates.
SubstrateName = Literal["kuzu", "faiss", "timechain", "duckdb"]


# Which SC op the substrate is being asked to execute. Phase 1 defines the
# enum; Phase 2 ships the actual RuleEvaluator implementations. Plugs that
# do not understand a given op return WriteResult.unsupported / a Record
# list shaped as a single Record with kind="unsupported_op".
SCOp = Literal[
    "search",      # arch §12.1: SEARCH — semantic similarity over a fork
    "fork_read",   # arch §12.1: FORK_READ — cross-fork query with filters
    "diff",        # arch §12.1: DIFF — temporal/state delta
    "cross_ref",   # arch §12.1: CROSS_REF — references to a TX across forks
    "get",         # generic point-lookup (kuzu/duckdb)
    "scan",        # generic range/full-scan (duckdb)
    "knn",         # FAISS top-k cosine similarity
]


# ─────────────────────────────────────────────────────────────────────────
# Substrate query / record / write result / health
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SubstrateQuery:
    """A read request issued to a SubstratePlug.

    `op` is the SC op (or generic primitive) the substrate is being asked
    to execute. `args` carries op-specific parameters (e.g. for SEARCH:
    `{"fork": "episodic", "query_embedding": [...], "limit": 50}`).

    `watermark_ts` is the consumer's tolerance: if the substrate's freshness
    is older than `watermark_ts`, the plug MAY soft-fail with
    `Record(kind="stale", ...)` per INV-Syn-4.
    """

    op: SCOp
    args: dict[str, Any] = field(default_factory=dict)
    watermark_ts: Optional[float] = None     # consumer's freshness tolerance
    limit: int = 50


@dataclass
class Record:
    """A single result row from SubstratePlug.read().

    `kind` discriminates: "row" (normal result), "stale" (watermark miss —
    consumer should degrade to last-known-consistent state), "unsupported_op"
    (plug does not implement this op yet — e.g. Timechain SC ops pre-Phase-2),
    "error" (transient — caller may retry).

    `data` is op-specific. For SEARCH: `{"tx_hash": ..., "score": ...}`.
    For FORK_READ: a row dict from `data/timechain/index.db`. For KNN:
    `{"id": ..., "distance": ...}`.
    """

    kind: Literal["row", "stale", "unsupported_op", "error"]
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class WriteResult:
    """Result of SubstratePlug.write() — synthesis_worker process only (G21).

    `accepted` is the count of records the substrate actually committed
    (e.g. CREATE IF NOT EXISTS may accept 0 on duplicate). `rejected` is
    the count refused (e.g. CGN-grounding write attempted — INV-Syn-1).
    `evidence_ref` is an opaque cross-process pointer (TX hash, row id,
    blob hash) the caller may stash for later audit.
    """

    accepted: int
    rejected: int = 0
    evidence_ref: Optional[str] = None


@dataclass
class SubstrateHealth:
    """Health snapshot a SubstratePlug emits — feeds the synth_status.bin
    watermark composer (INV-Syn-3 / INV-Syn-4).

    `last_consistent_event_ts` is the most recent monotonic event timestamp
    the substrate guarantees fully consistent across its indices. Consumers
    in other processes compare their `watermark_ts` tolerance against this.

    `lag_s` is the substrate's perceived staleness (now − last_consistent);
    the synthesis_worker may degrade ranking weights when lag grows.
    """

    name: SubstrateName
    last_consistent_event_ts: float
    lag_s: float
    healthy: bool = True
    error: Optional[str] = None


@runtime_checkable
class SubstratePlug(Protocol):
    """Where memory lives — read-only across process boundaries, watermark-gated.

    Concrete implementations land per arch §3.3: Phase 2 = Timechain (wraps
    RuleEvaluator in-process — G19; SC ops `search`/`fork_read`/`diff`/
    `cross_ref` ship here); Phase 4 = Kuzu + FAISS; Phase 1 (this) = DuckDB
    (the `activation_state` substrate the synthesis_worker writes through
    its own in-process `TitanDuckDB` handle — the only "write" surface in
    Phase 1).
    """

    name: SubstrateName

    def read(self, q: SubstrateQuery) -> list[Record]:
        """Read-only, watermark-gated (INV-Syn-4). Returns `Record(kind=
        "unsupported_op", ...)` for ops the plug has not implemented yet
        (e.g. Timechain SC ops pre-Phase-2)."""
        ...

    def write(self, recs: list[Record]) -> WriteResult:
        """Write — synthesis_worker process ONLY (G21 / INV-Syn-3). Non-
        synthesis-worker callers MUST NOT invoke this; it is here so the
        worker itself can adopt every substrate through a single contract."""
        ...

    def health(self) -> SubstrateHealth:
        """Health snapshot feeding the synth_status.bin watermark
        composition. Called every 60s by the recompute loop; emit cheaply."""
        ...


# ─────────────────────────────────────────────────────────────────────────
# Truth oracle
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class OracleClaim:
    """A claim submitted to a TruthOraclePlug for verification.

    `domain` is the claim namespace ("solana_tx", "code_correctness",
    "web_fact", "x_event_real", etc.) — the plug's `can_handle(domain)`
    gates whether it accepts.

    `payload` is the claim body — opaque to the engine, interpreted by the
    plug (e.g. for `code_correctness`: `{"language": "python", "code": ...,
    "expected_stdout": ...}`; for `solana_tx`: `{"signature": ...,
    "expected_status": "confirmed"}`).
    """

    domain: str
    payload: dict[str, Any]
    importance: float = 0.5         # 0..1 — multiplies metabolic gate (INV-7)


@dataclass
class OracleVerdict:
    """Verdict the truth oracle returns. The verdict itself is anchored on
    the Timechain by the synthesis_worker (arch §11.1) — verification is
    provenanced, not just the attempt.
    """

    oracle_id: str                  # e.g. "coding_sandbox" / "solana_rpc" / "x_oracle"
    verdict: Literal["true", "false", "unknown"]
    evidence_ref: str               # opaque pointer (CAS hash / RPC sig / X tweet_id)
    cost: float                     # SOL units (free oracles report 0.0)
    latency_ms: int
    ts: float                       # monotonic time of verdict


@runtime_checkable
class TruthOraclePlug(Protocol):
    """'Is it true?' — verdict anchored on-chain (arch §11.1).

    Day-one set (arch §11.1 + SPEC §25.3, expanded D-SPEC-123 2026-05-23):
      - coding_sandbox (incl. deterministic math+code; doubles as ToolPlug)
      - solana_rpc + explorers (Helius today; sovereign light node = TARGET)
      - web_api
      - x_oracle (X as second-source companion to web for real-time events /
        commentary / billion-person discussion — more efficient than web
        search for current information; doubles as ToolPlug via x_research)

    Cost class gates the metabolic budget (INV-7): free oracles always-on,
    metered ones gated by `importance × balance`.
    """

    oracle_id: str
    cost_class: Literal["free", "metered"]

    def can_handle(self, domain: str) -> bool: ...

    def verify(self, claim: OracleClaim) -> OracleVerdict: ...


# ─────────────────────────────────────────────────────────────────────────
# Meaning oracle (CGN — sole grounding authority per INV-Syn-1)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class ConceptRef:
    """Reference to a concept in the meaning oracle's namespace. `concept_id`
    is the CGN node id; `version` follows arch §10 versioned concepts (a
    refined concept produces v(n+1) while v(n) stays immutable on-chain)."""

    concept_id: str
    version: int = 0


@dataclass
class MeaningStrand:
    """What a concept means, from CGN's perspective. The four strands of the
    concept spine (arch §6): declarative (what it is), procedural (skills
    using it), episodic (encounters), felt (emotional ground via emot-CGN).
    Each strand is a Timechain-anchored hash list."""

    concept: ConceptRef
    declarative_anchors: list[str] = field(default_factory=list)
    procedural_anchors: list[str] = field(default_factory=list)
    episodic_anchors: list[str] = field(default_factory=list)
    felt_anchors: list[str] = field(default_factory=list)


@dataclass
class FeltContext:
    """Current felt state passed to CGN.ground() to ground a concept in
    Titan's lived emotional moment. Sourced from emot-CGN by the caller."""

    valence: float                  # -1..+1
    arousal: float                  # 0..1
    neuromods: dict[str, float] = field(default_factory=dict)   # DA/5HT/NE/GABA/ACh


@dataclass
class Grounding:
    """Result of CGN.ground() — the felt-state link CGN attached to a concept.
    Returned read-only to the synthesis_worker; CGN is the SOLE authority
    that creates these (INV-Syn-1)."""

    concept: ConceptRef
    grounding_id: str
    strength: float                 # 0..1
    ts: float


@runtime_checkable
class MeaningOraclePlug(Protocol):
    """'What does it mean / how does it feel?' — CGN, the sole grounding
    authority (INV-Syn-1 / INV-1). The synthesis engine REQUESTS groundings;
    it never creates them.
    """

    def meaning_of(self, concept: ConceptRef) -> MeaningStrand:
        """Read-only query: what does CGN currently associate with this
        concept across all four strands of the spine?"""
        ...

    def ground(self, concept: ConceptRef, felt: FeltContext) -> Grounding:
        """Ask CGN to ground this concept in the given felt-state context.
        CGN is the sole authority — synthesis_worker MUST NOT bypass."""
        ...


# ─────────────────────────────────────────────────────────────────────────
# Proof strategy (Merkle default; ZK targeted, cost-gated — INV-7)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Proof:
    """A proof attached to a graduation / tombstone / inclusion claim.

    For Merkle: `commitment` is the Merkle root; `payload_ref` is the CAS
    hash of the proof body (leaf list + inclusion paths). For ZK:
    `commitment` is the verifier-key commit; `payload_ref` is the
    serialized proof + public inputs (ZK Vault stores actual bytes).
    """

    strategy: Literal["merkle", "zk"]
    commitment: bytes
    payload_ref: Optional[str] = None  # CAS hash / ZK Vault id
    cost: float = 0.0


@runtime_checkable
class ProofStrategyPlug(Protocol):
    """Merkle (default workhorse) / ZK (targeted, opt-in per fork, cost-gated)
    per arch §11.2 + INV-7. Energy-consistent: ZK proving on CPU would blow
    the no-GPU-farm budget, so Merkle-default is principled, not merely
    simpler."""

    strategy: Literal["merkle", "zk"]

    def commit(self, payload: bytes) -> Proof:
        """Compute the commitment (root / verifier-key commit) for the
        payload. For Merkle: SHA-256 root. For ZK: prover output."""
        ...

    def verify(self, proof: Proof, payload: Optional[bytes] = None) -> bool:
        """Verify a previously-issued proof. Merkle verification needs the
        payload (or its CAS-stored leaves); ZK verification does not."""
        ...


# ─────────────────────────────────────────────────────────────────────────
# Tool (the engine's single action surface — INV-12 per arch §11.3)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """A tool invocation issued by the synthesis engine. Every tool call is
    logged as a procedural-fork TX (arch §8.2) and oracle-verified where
    possible — every tool use is anchored + skill-compilable (arch §8).
    """

    tool_id: str                    # "coding_sandbox" | "events_teacher" | "x_research" | ...
    args: dict[str, Any]
    parent_chat_tx: Optional[str] = None   # links the call to the chat turn that issued it
    parent_goal: Optional[str] = None      # the goal-buffer entry driving the call
    parent_skill_id: Optional[str] = None  # set when invoked through procedural_skill_match
    # RFP_synthesis_self_learning_meta_reasoning v1.1 — the OuterMetaPolicy decision
    # that chose to fire this tool, carried so the verdict-time C1 capture can write
    # the per-use Reasoning record + train the policy (decision+outcome together,
    # INV-OML-12). IN-MEMORY ONLY — NOT written to the on-chain tool_call TX content
    # (the chain stays a lean pointer; features live in the DuckDB Reasoning record).
    decision_features: Optional[list] = None   # the 11-D OuterFeatures vector
    decision_action: Optional[int] = None      # the chosen action index


@dataclass
class ToolResult:
    """Result of a ToolPlug.invoke(). The result is logged as a procedural
    TX with the full payload content-addressed (Phase-0 CAS) so the chain
    carries only the hash reference."""

    tool_id: str
    success: bool
    result_summary: str             # human-readable summary
    result_full_hash: Optional[str] = None   # CAS hash of the full payload
    latency_ms: int = 0
    exception: Optional[str] = None


@runtime_checkable
class ToolPlug(Protocol):
    """The engine's single action surface (INV-12). Day-one set per arch
    §11.3 + SPEC §25.3:
      - coding_sandbox (doubles as TruthOraclePlug)
      - events_teacher (X event distillation)
      - knowledge / StealthSage research worker (web + doc search)
      - x_research (active X mining — post + fetch latest events / topic
        commentary / discussion threads; doubles as TruthOraclePlug via
        x_oracle, expanded D-SPEC-123 2026-05-23)

    Tools that double as oracles return a `TruthOraclePlug` from
    `.oracle()`; pure-tool plugs return None.
    """

    tool_id: str

    def capabilities(self) -> list[str]:
        """List of capability tags the tool exposes (e.g. for x_research:
        `["post", "fetch_thread", "fetch_topic", "fetch_account"]`).
        Used by the engine's tool-selection in `RECALL.procedural_skill_match`."""
        ...

    def invoke(self, call: ToolCall) -> ToolResult: ...

    def oracle(self) -> Optional[TruthOraclePlug]:
        """Return the truth-oracle surface for this tool if it has one
        (e.g. coding_sandbox / x_oracle), else None."""
        ...


# ─────────────────────────────────────────────────────────────────────────
# Public re-export surface (kept minimal — concrete plug imports stay in
# the synthesis_worker process; cross-process consumers use BridgeRecall
# patterns + the synth_status.bin watermark, never these classes directly).
# ─────────────────────────────────────────────────────────────────────────

__all__ = [
    # Substrate
    "SubstrateName", "SCOp", "SubstrateQuery", "Record",
    "WriteResult", "SubstrateHealth", "SubstratePlug",
    # Truth oracle
    "OracleClaim", "OracleVerdict", "TruthOraclePlug",
    # Meaning oracle
    "ConceptRef", "MeaningStrand", "FeltContext", "Grounding",
    "MeaningOraclePlug",
    # Proof
    "Proof", "ProofStrategyPlug",
    # Tool
    "ToolCall", "ToolResult", "ToolPlug",
]
