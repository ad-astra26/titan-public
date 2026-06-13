"""MakerStore — Titan's persistent, sovereign model of his Maker (the human he is
bonded to).

RFP_missions_and_the_maker_model §7.1. Mirrors `ReasoningStore` / the
`record_maker_assessment` write-path: every fact about the Maker is persisted as ONE
deref-able record across two substrates, written through the single SynthesisWriter
(INV-Syn-19/28 — G21 single-writer):

  • DuckDB `maker_facts` — the recallable scalars (value, provenance, confidence,
    significance, research_urgency, version/superseded). PK-only (the actr_buffers
    crash-class rule — no secondary index).
  • Kuzu  `MakerFact` node + `MAKER_HAS_FACT` edge under the `Maker` hub under `Self`
    (Self -[SELF_HAS_MAKER]-> Maker -[MAKER_HAS_FACT]-> MakerFact) — the graph
    structure ("what do I know about my Maker?").

Invariants realized:
  • INV-MIS-EPISTEMIC-HONESTY — every fact carries provenance ∈ {maker-told, observed,
    inferred, researched} + a confidence that is NEVER 1.0 (capped at 0.98), even for
    maker-told. Nothing about the Maker is taken as absolute truth.
  • INV-MIS-SOVEREIGN-KNOWLEDGE — this is Titan's own knowledge; there is no Maker-side
    inspect/override surface here (none is ever built — a backdoor is forbidden).

Dynamic placement (v0): a new (category, value) either REINFORCES the current fact of
that category (same value → bump confidence/updated_at, same node) or VERSIONS it (a
contradicting value → supersede the old, create version N+1). `significance` (1..100)
and `research_urgency` are SET here on create and MAINTAINED by the Phase-1b synthesis
routine (decay/bump/flag). Every write is soft-fail — a model-write must NEVER break
the chat/verdict path.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Optional

from titan_hcl.synthesis.writer import resolve_writer

logger = logging.getLogger(__name__)

# Confidence is never absolute — INV-MIS-EPISTEMIC-HONESTY (nothing about the Maker is
# 100% truth, not even maker-told).
_CONF_CEIL: float = 0.98
# Provenance of a Maker fact (INV-MIS-EPISTEMIC-HONESTY). `birth-cert` = seeded from
# Titan's identity/config at boot (wallet, X handle, pubkey); the rest are learned.
_PROVENANCE = ("maker-told", "observed", "inferred", "researched", "birth-cert")


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return s or "misc"


class MakerStore:
    """Sole writer of `maker_facts` + the Kuzu Maker/MakerFact nodes. Synthesis-
    worker-side; all mutations on the writer thread (INV-Syn-19/28)."""

    def __init__(
        self,
        duckdb_conn: Any,
        *,
        graph: Any = None,
        writer: Any = None,
        clock: Callable[[], float] = time.time,
    ):
        self._db = duckdb_conn
        self._graph = graph                 # core.direct_memory KnowledgeGraph (Kuzu)
        self._writer = resolve_writer(writer)
        self._clock = clock
        self.facts_written = 0
        self._init_schema()

    # ── schema (PK-only — crash-class rule) ──────────────────────────────
    def _init_schema(self) -> None:
        def _create() -> None:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS maker_facts ("
                "  fact_id          TEXT    PRIMARY KEY,"
                "  category         TEXT    NOT NULL,"
                "  value            TEXT    NOT NULL,"
                "  provenance       TEXT,"               # maker-told|observed|inferred|researched
                "  confidence       DOUBLE,"             # [0, 0.98] — never absolute
                "  significance     DOUBLE,"             # [1, 100] — set here, maintained by Phase-1b
                "  research_urgency TEXT,"               # none|low|high|stale (set by Phase-1b/mission)
                "  source_turn      TEXT,"
                "  version          INTEGER DEFAULT 1,"
                "  superseded       INTEGER DEFAULT 0,"
                "  created_at       DOUBLE  NOT NULL,"
                "  updated_at       DOUBLE  NOT NULL"
                ")")
        try:
            self._writer.submit_sync(_create)
        except Exception as e:  # noqa: BLE001
            logger.warning("[MakerStore] schema init failed: %s", e)

    @staticmethod
    def _initial_significance(confidence: float) -> float:
        """Set-on-create significance (1..100) from confidence; the Phase-1b routine
        does decay-on-silence + bump-on-use afterward."""
        return round(min(100.0, max(1.0, float(confidence) * 70.0)), 1)

    # ── WRITE: one provenance-tagged Maker fact (dynamic placement) ───────
    def record_fact(
        self, *, category: str, value: str, provenance: str = "maker-told",
        confidence: float = 0.7, source_turn: str = "",
    ) -> str:
        """Persist one fact about the Maker (DuckDB scalars + the Kuzu MakerFact node
        under the Maker hub). Dynamic placement: same-category same-value → reinforce
        (bump confidence, same node); same-category different-value → supersede + new
        version. Returns the fact_id (or '' on no-op/failure). Soft-fail."""
        cat = str(category or "").strip().lower()
        val = str(value or "").strip()
        if not cat or not val:
            return ""
        prov = provenance if provenance in _PROVENANCE else "maker-told"
        conf = float(min(_CONF_CEIL, max(0.0, confidence)))

        def _do() -> str:
            now = float(self._clock())
            # The current (un-superseded) fact of this category, if any.
            cur = self._db.execute(
                "SELECT fact_id, value, confidence, version FROM maker_facts "
                "WHERE category=? AND superseded=0 ORDER BY version DESC LIMIT 1",
                [cat]).fetchone()
            if cur is not None:
                cur_fid, cur_val, cur_conf, cur_ver = (
                    str(cur[0]), str(cur[1]), float(cur[2]), int(cur[3]))
                if cur_val.strip().lower() == val.lower():
                    # REINFORCE — same fact restated. Bump confidence (toward, not to,
                    # certainty) + refresh significance/updated_at. No new node.
                    new_conf = min(_CONF_CEIL, max(cur_conf, conf))
                    self._db.execute(
                        "UPDATE maker_facts SET confidence=?, significance=?, "
                        "updated_at=? WHERE fact_id=?",
                        [new_conf, self._initial_significance(new_conf), now, cur_fid])
                    return cur_fid
                # CONTRADICTION — a newer value supersedes the old; version up.
                self._db.execute(
                    "UPDATE maker_facts SET superseded=1, updated_at=? WHERE fact_id=?",
                    [now, cur_fid])
                if self._graph is not None:
                    try:
                        self._graph.spine_supersede_maker_fact(cur_fid)
                    except Exception as e:  # noqa: BLE001
                        logger.debug("[MakerStore] supersede kuzu soft-fail: %s", e)
                new_ver = cur_ver + 1
            else:
                new_ver = 1

            fid = f"maker:{_slug(cat)}:v{new_ver}"
            sig = self._initial_significance(conf)
            self._db.execute(
                "INSERT INTO maker_facts (fact_id, category, value, provenance, "
                "confidence, significance, research_urgency, source_turn, version, "
                "superseded, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT (fact_id) DO NOTHING",
                [fid, cat, val, prov, conf, sig, "none",
                 str(source_turn or "")[:280], new_ver, 0, now, now])
            if self._graph is not None:
                try:
                    self._graph.spine_create_maker_fact_node(
                        fact_id=fid, category=cat, value=val, provenance=prov,
                        confidence=conf, significance=sig, research_urgency="none",
                        version=new_ver, created_at=now, updated_at=now)
                    self._graph.spine_link_maker_fact(fid)
                except Exception as e:  # noqa: BLE001
                    logger.debug("[MakerStore] record_fact kuzu soft-fail: %s", e)
            self.facts_written += 1
            return fid

        try:
            return str(self._writer.submit_sync(_do) or "")
        except Exception as e:  # noqa: BLE001
            logger.warning("[MakerStore] record_fact write failed: %s", e)
            return ""

    # ── READ: "what do I know about my Maker?" ───────────────────────────
    def recall(self, category: Optional[str] = None) -> list[dict]:
        """Return the current (un-superseded) facts about the Maker, ranked by
        significance. Filters to one `category` if given. Read on the writer thread
        (the guarded conn rejects off-thread native ops)."""
        cols = ("fact_id", "category", "value", "provenance", "confidence",
                "significance", "research_urgency", "version", "created_at",
                "updated_at")

        def _do() -> list[dict]:
            sel = (
                "SELECT fact_id, category, value, provenance, confidence, "
                "significance, research_urgency, version, created_at, updated_at "
                "FROM maker_facts WHERE superseded=0")
            if category:
                rows = self._db.execute(
                    sel + " AND category=? ORDER BY significance DESC",
                    [str(category).strip().lower()]).fetchall()
            else:
                rows = self._db.execute(
                    sel + " ORDER BY significance DESC").fetchall()
            return [dict(zip(cols, r)) for r in rows]

        try:
            return list(self._writer.submit_sync(_do) or [])
        except Exception as e:  # noqa: BLE001
            logger.warning("[MakerStore] recall failed: %s", e)
            return []
