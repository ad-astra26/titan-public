"""CGN lexicon snapshot exporter (Phase 8.Y fold-in / D-SPEC-PHASE8).

Closes the P7 follow-up gap: `agno_worker._ground_for_goal_hook` reads
`plugin.cgn_lexicon` but no loader populates it → `concept_ids` stuck
at `[]` in production. P8.Y:

- CGN-side exporter writes `data/cgn_lexicon_snapshot.json` on a
  5-min cadence + on every grounding-mutation bus event. Atomic
  tmp+rename, like every other Phase 4-7 snapshot.
- Emits `CGN_LEXICON_UPDATED` bus event with payload
  `{ts, lexicon_size, snapshot_path}`.
- Agno-side loader (in modules/agno_worker.py) reads at boot + on
  the bus event into `plugin.cgn_lexicon`.

Soft-fail throughout: if the source DB is missing or empty, the
snapshot is still written (with an empty mapping). If `_ground_for_goal_hook`
sees a missing/empty `plugin.cgn_lexicon`, it falls back to
`concept_ids: []` (current production behavior, no regression).

Source of truth: `data/inner_memory.db` table `knowledge_concepts`.
Each `topic` column value becomes a concept_id; the lowercase form of
the topic + its tokens become lexicon keys. Confidence is the tie-breaker
when multiple topics share a token (highest confidence wins).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from typing import Optional

logger = logging.getLogger(__name__)


DEFAULT_INNER_MEMORY_DB = "data/inner_memory.db"
DEFAULT_SNAPSHOT_NAME = "cgn_lexicon_snapshot.json"

# Cap the lexicon to keep snapshot file size + agno-side dict footprint bounded.
# Per arch §6 + INV-Syn-18: spreading-activation uses at most 20 entities
# per chat. The lexicon needs enough coverage that common tokens hit, but
# doesn't need every word in the corpus.
DEFAULT_MAX_ENTRIES = 5000

# Minimum word length to enter the lexicon. Mirrors agno_worker's
# `_ground_for_goal_hook` filter (`len(t) > 2`).
MIN_TOKEN_LEN = 3

# Word boundary tokenizer — case-folded, strips punctuation. Matches the
# canonical CGN vocabulary tokenization style.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens of length ≥ MIN_TOKEN_LEN. Stable across exports."""
    if not text:
        return []
    return [
        m.group(0).lower()
        for m in _TOKEN_RE.finditer(text)
        if len(m.group(0)) >= MIN_TOKEN_LEN
    ]


def build_lexicon_from_db(
    *,
    inner_memory_db: str = DEFAULT_INNER_MEMORY_DB,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> dict[str, str]:
    """Build {token: concept_id} mapping from knowledge_concepts.

    Strategy:
      1. SELECT topic, confidence FROM knowledge_concepts ORDER BY confidence DESC.
      2. For each row: tokenize the topic + add (token → topic) entries.
      3. Higher-confidence rows process first so they win the dict slot
         when two topics share a token (dict overwrite semantics).
      4. Cap at max_entries; subsequent rows beyond cap stop early.

    Returns {} on any I/O / parse error (caller treats as 'no lexicon').
    """
    if not os.path.exists(inner_memory_db):
        return {}
    try:
        conn = sqlite3.connect(
            f"file:{inner_memory_db}?mode=ro&immutable=0", uri=True, timeout=5.0,
        )
    except Exception as e:
        logger.warning("[cgn.lexicon_exporter] open failed: %s", e)
        return {}
    try:
        conn.row_factory = sqlite3.Row
        try:
            rows = list(conn.execute(
                "SELECT topic, confidence FROM knowledge_concepts "
                "WHERE topic IS NOT NULL AND topic != '' "
                "ORDER BY confidence DESC, topic ASC LIMIT ?",
                [int(max_entries)],
            ).fetchall())
        except sqlite3.Error as e:
            logger.debug("[cgn.lexicon_exporter] query failed: %s", e)
            return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Higher-confidence topics seed the dict first so they win the
    # token-collision tie-break (we want the most-grounded concept to be
    # the canonical answer for "what does this token mean?").
    out: dict[str, str] = {}
    for row in rows:
        topic = str(row["topic"] or "").strip()
        if not topic:
            continue
        # The concept_id IS the topic string (canonical per CGN).
        # The lowercase topic itself is also a lexicon key (cheap exact match).
        lowered = topic.lower()
        if MIN_TOKEN_LEN <= len(lowered):
            out.setdefault(lowered, topic)
        for tok in tokenize(topic):
            out.setdefault(tok, topic)
        if len(out) >= max_entries:
            break
    return out


def write_snapshot(
    lexicon: dict[str, str],
    snapshot_path: str,
    *,
    clock=time.time,
) -> tuple[bool, dict]:
    """Atomic tmp+rename write of cgn_lexicon_snapshot.json.

    Payload schema:
        {version: 1, ts: <wall-clock>, lexicon_size: int, lexicon: dict}

    Returns (success, payload). Caller treats success=False as "skip the
    CGN_LEXICON_UPDATED emit"; the agno-side loader keeps the prior
    lexicon (no regression)."""
    payload = {
        "version": 1,
        "ts": float(clock()),
        "lexicon_size": len(lexicon),
        "lexicon": dict(lexicon),
    }
    try:
        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        tmp_path = snapshot_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, snapshot_path)
        return True, payload
    except Exception as e:
        logger.warning(
            "[cgn.lexicon_exporter] snapshot write failed (%s): %s",
            snapshot_path, e,
        )
        return False, payload


def export_lexicon(
    *,
    inner_memory_db: str = DEFAULT_INNER_MEMORY_DB,
    snapshot_path: Optional[str] = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    clock=time.time,
) -> Optional[dict]:
    """One-call export: build the lexicon + write the snapshot. Returns
    the payload dict on success, None on write failure."""
    if snapshot_path is None:
        data_dir = os.environ.get("TITAN_DATA_DIR", "data")
        snapshot_path = os.path.join(data_dir, DEFAULT_SNAPSHOT_NAME)
    lexicon = build_lexicon_from_db(
        inner_memory_db=inner_memory_db, max_entries=max_entries,
    )
    ok, payload = write_snapshot(lexicon, snapshot_path, clock=clock)
    if not ok:
        return None
    return payload


def load_lexicon_snapshot(snapshot_path: Optional[str] = None) -> dict[str, str]:
    """Read the snapshot back into a {token: concept_id} dict.

    Used by agno_worker's `_load_cgn_lexicon` at boot + on
    CGN_LEXICON_UPDATED bus event. Returns {} on missing / corrupt
    (P7 hook falls back to `concept_ids: []` — no regression)."""
    if snapshot_path is None:
        data_dir = os.environ.get("TITAN_DATA_DIR", "data")
        snapshot_path = os.path.join(data_dir, DEFAULT_SNAPSHOT_NAME)
    if not os.path.exists(snapshot_path):
        return {}
    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(
            "[cgn.lexicon_exporter] snapshot read failed (%s): %s",
            snapshot_path, e,
        )
        return {}
    if not isinstance(data, dict):
        return {}
    lex = data.get("lexicon")
    if not isinstance(lex, dict):
        return {}
    # Defensive cast — every value must be a string concept_id.
    return {
        str(k).lower(): str(v)
        for k, v in lex.items()
        if isinstance(k, str) and isinstance(v, str)
    }
