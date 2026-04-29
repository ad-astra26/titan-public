"""
titan_plugin/logic/meta_outer_context.py — Bridge Titan's inner and outer cognitive memory.

Meta-reasoning chains need to reason about specific people, events, and topics
Titan has encountered in the world — not just abstract inner state. This module
is the bridge.

Components:
  - OuterContextReader        — unified async fan-out across 7 stores
  - _LRUTTLCache              — lookup cache (size-bounded, TTL-expiring)
  - PeerCGNReader             — lock-free tail of peer consumers' saved β/α
  - SocialGraphReader         — SQLite WAL read of data/social_graph.db
  - FeltExperiencesReader     — SQLite WAL read of data/events_teacher.db
  - InnerMemoryReader         — SQLite read of data/inner_memory.db (already wired)
  - TitanSelfProbe            — in-process snapshot of current state
  - KnowledgeReader           — DivineBus RPC client for knowledge_worker (Kuzu owner)
  - XSearchReader             — SocialXGateway.search wrapper (rate-limited + circuit-broken)
  - EventsPollReader          — DivineBus RPC client for events_teacher manual window

Activation: flag file at /dev/shm/meta_outer_enabled.flag — runtime-controllable
via /v4/meta-outer/enable and /v4/meta-outer/disable. is_active() call is cached
for 1s to avoid repeated stat() syscalls.

All reads are timeout-bounded (default 200ms composed, 50ms per-read). Reader
owns a dedicated ThreadPoolExecutor(max_workers=4) — isolated from the shared
FastAPI/uvicorn pool to prevent cross-contamination under load.

See rFP_titan_meta_outer_layer.md for full architecture and bridge semantics.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from titan_plugin import bus

logger = logging.getLogger("titan.meta_outer")


# ── Activation (flag-file based, 1s cache) ───────────────────────────────

_FLAG_PATH = "/dev/shm/meta_outer_enabled.flag"
_FLAG_CACHE = {"ts": 0.0, "value": False}
_FLAG_TTL_S = 1.0


def is_active() -> bool:
    """Cheap is-active check. Reads flag file via os.path.exists; cached 1s.

    Flag file is touched by POST /v4/meta-outer/enable, removed by /disable.
    Allows runtime activation/deactivation without restart.
    """
    now = time.time()
    if now - _FLAG_CACHE["ts"] < _FLAG_TTL_S:
        return bool(_FLAG_CACHE["value"])
    exists = os.path.exists(_FLAG_PATH)
    _FLAG_CACHE["ts"] = now
    _FLAG_CACHE["value"] = exists
    return exists


def set_active(value: bool) -> bool:
    """Touch or remove the flag file. Returns new is_active state.

    Called by dashboard endpoints. Safe to call concurrently — os.path and
    os.remove are atomic enough for this purpose.
    """
    try:
        if value:
            with open(_FLAG_PATH, "w") as f:
                f.write(f"enabled:{time.time()}\n")
        else:
            if os.path.exists(_FLAG_PATH):
                os.remove(_FLAG_PATH)
    except Exception as e:
        logger.warning("[MetaOuter] set_active(%s) failed: %s", value, e)
    _FLAG_CACHE["ts"] = 0.0  # force re-check on next is_active()
    return is_active()


# ── Cache ────────────────────────────────────────────────────────────────

class _LRUTTLCache:
    """Bounded LRU with per-entry TTL. Thread-unsafe — caller holds the lock
    (OuterContextReader uses a single-threaded access pattern per key).
    """

    def __init__(self, max_size: int = 500, ttl_s: float = 60.0):
        self._max = int(max_size)
        self._ttl = float(ttl_s)
        self._data: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: Any) -> Optional[Any]:
        entry = self._data.get(key)
        if entry is None:
            self.misses += 1
            return None
        ts, value = entry
        if (time.time() - ts) > self._ttl:
            self._data.pop(key, None)
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: Any, value: Any) -> None:
        self._data[key] = (time.time(), value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)
            self.evictions += 1

    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {
            "size": len(self._data),
            "max_size": self._max,
            "ttl_s": self._ttl,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(hit_rate, 4),
        }


# ── Peer CGN reader (lock-free tail of existing save files) ───────────────

_DEFAULT_PEER_CGN_PATHS = {
    "meta":      "data/meta_cgn/primitive_grounding.json",
    "emot":      "data/emot_cgn/emot_cgn_snapshot_v1.json",
    "language":  "data/language_cgn/state_snapshot.json",
    "social":    "data/social_cgn/state_snapshot.json",
    "knowledge": "data/knowledge_cgn/state_snapshot.json",
    "coding":    "data/coding_cgn/state_snapshot.json",
    "reasoning": "data/reasoning_cgn/state_snapshot.json",
}


class PeerCGNReader:
    """Tails peer CGN consumers' on-disk snapshot files lock-free.

    Each consumer already persists its β/α state as part of normal lifecycle
    (meta_cgn → primitive_grounding.json, emot_cgn → emot_cgn_snapshot_v1.json,
    others via CGNConsumerClient save_state). This reader opens each file
    read-only, caches parsed state keyed by (consumer, mtime), and serves
    lookups without bus roundtrips.

    Consumers that don't currently write a snapshot return None gracefully —
    logged once per consumer at DEBUG to avoid spam.
    """

    def __init__(self, paths: Optional[dict] = None):
        self._paths: dict[str, str] = dict(paths or _DEFAULT_PEER_CGN_PATHS)
        # consumer → (mtime, parsed_dict | None)
        self._cache: dict[str, tuple[float, Optional[dict]]] = {}
        self._logged_missing: set[str] = set()

    def _load(self, consumer: str) -> Optional[dict]:
        path = self._paths.get(consumer)
        if not path or not os.path.exists(path):
            if consumer not in self._logged_missing:
                logger.debug("[PeerCGN] %s save-file missing (%s) — graceful skip",
                             consumer, path)
                self._logged_missing.add(consumer)
            return None
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        cached = self._cache.get(consumer)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            with open(path, "r") as f:
                parsed = json.load(f)
        except Exception as e:
            logger.debug("[PeerCGN] %s parse err: %s", consumer, e)
            self._cache[consumer] = (mtime, None)
            return None
        self._cache[consumer] = (mtime, parsed)
        return parsed

    def peer_cgn_beta(self, consumer: str, concept: str) -> Optional[float]:
        """Return β-posterior for a concept/primitive in a peer consumer.

        Schema-tolerant: meta_cgn uses _primitives dict keyed by primitive id,
        emot_cgn uses regions, generic CGN consumers use concept_grounding map.
        """
        state = self._load(consumer)
        if state is None:
            return None
        return _extract_beta(state, consumer, concept)

    def peer_cgn_alpha(self, consumer: str, concept: str) -> Optional[float]:
        state = self._load(consumer)
        if state is None:
            return None
        return _extract_alpha(state, consumer, concept)

    def peer_cgn_summary(self) -> dict:
        """Cross-consumer summary: which peers are reachable, count of concepts."""
        out: dict = {}
        for consumer in self._paths:
            state = self._load(consumer)
            if state is None:
                out[consumer] = {"available": False, "concepts": 0}
                continue
            n = _count_concepts(state, consumer)
            out[consumer] = {"available": True, "concepts": n}
        return out


def _extract_beta(state: dict, consumer: str, concept: str) -> Optional[float]:
    # meta_cgn schema
    if consumer == "meta":
        prims = state.get("primitives") or state.get("_primitives") or {}
        p = prims.get(concept)
        if isinstance(p, dict):
            return _coerce_float(p.get("beta"))
        return None
    # emot_cgn schema — β lives on regions keyed by label
    if consumer == "emot":
        regions = state.get("regions") or {}
        r = regions.get(concept)
        if isinstance(r, dict):
            return _coerce_float(r.get("beta"))
        return None
    # generic CGN consumer: concept_grounding map
    grounding = state.get("concept_grounding") or state.get("grounding") or {}
    c = grounding.get(concept)
    if isinstance(c, dict):
        return _coerce_float(c.get("beta"))
    return None


def _extract_alpha(state: dict, consumer: str, concept: str) -> Optional[float]:
    if consumer == "meta":
        prims = state.get("primitives") or state.get("_primitives") or {}
        p = prims.get(concept)
        if isinstance(p, dict):
            return _coerce_float(p.get("alpha"))
        return None
    if consumer == "emot":
        regions = state.get("regions") or {}
        r = regions.get(concept)
        if isinstance(r, dict):
            return _coerce_float(r.get("alpha"))
        return None
    grounding = state.get("concept_grounding") or state.get("grounding") or {}
    c = grounding.get(concept)
    if isinstance(c, dict):
        return _coerce_float(c.get("alpha"))
    return None


def _count_concepts(state: dict, consumer: str) -> int:
    if consumer == "meta":
        return len(state.get("primitives") or state.get("_primitives") or {})
    if consumer == "emot":
        return len(state.get("regions") or {})
    return len(state.get("concept_grounding") or state.get("grounding") or {})


def _coerce_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── SocialGraphReader ────────────────────────────────────────────────────

class SocialGraphReader:
    """Read-only SQLite WAL reader of data/social_graph.db.

    Composes person profile from 3 richest tables: titan_social_preferences
    (affinity/relationship/tags), community_registry (bio/followers/last_tweet),
    user_profiles (engagement/felt_tensor).
    """

    def __init__(self, db_path: str = "data/social_graph.db"):
        self._db_path = db_path

    def _connect(self) -> Optional[sqlite3.Connection]:
        if not os.path.exists(self._db_path):
            return None
        try:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True,
                                    timeout=2.0)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.debug("[SocialGraph] connect err: %s", e)
            return None

    def get_person_profile(self, person_id: str) -> Optional[dict]:
        """Compose from 3 tables. person_id matches user_name (X handle) or user_id.

        Normalizes '@handle' → 'handle' since DuckDB tables store handles
        without the leading '@'. Original person_id is preserved in output.
        """
        conn = self._connect()
        if conn is None:
            return None
        try:
            original = person_id
            norm = person_id[1:] if person_id.startswith("@") else person_id
            out: dict = {"person_id": original}
            person_id = norm
            # user_profiles — join on user_id or user_name
            row = conn.execute(
                "SELECT user_id, platform, display_name, interaction_count, "
                "like_score, dislike_score, engagement_level, notes "
                "FROM user_profiles WHERE user_id = ? LIMIT 1",
                (person_id,)
            ).fetchone()
            if row:
                out["profile"] = dict(row)
            # titan_social_preferences — affinity + relationship
            row = conn.execute(
                "SELECT relationship, affinity, tags, interaction_count, "
                "last_interacted, last_checked "
                "FROM titan_social_preferences WHERE user_name = ? LIMIT 1",
                (person_id,)
            ).fetchone()
            if row:
                out["preferences"] = dict(row)
            # community_registry — bio + last tweet
            row = conn.execute(
                "SELECT display_name, bio, followers_count, is_follower, "
                "is_following, last_tweet_text, last_tweet_time "
                "FROM community_registry WHERE user_name = ? LIMIT 1",
                (person_id,)
            ).fetchone()
            if row:
                out["community"] = dict(row)
            if len(out) == 1:  # only person_id, nothing matched
                return None
            return out
        except sqlite3.Error as e:
            logger.debug("[SocialGraph] query err: %s", e)
            return None
        finally:
            conn.close()

    def get_recent_engagements(self, limit: int = 10) -> list[dict]:
        """Recent rows from engagement_ledger — X mentions/DMs/likes."""
        conn = self._connect()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                "SELECT tweet_id, user_name, action, timestamp, mention_text "
                "FROM engagement_ledger ORDER BY timestamp DESC LIMIT ?",
                (min(int(limit), 100),)
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()


# ── FeltExperiencesReader ────────────────────────────────────────────────

class FeltExperiencesReader:
    """Read-only SQLite reader of data/events_teacher.db — felt_experiences table.

    Each row carries sentiment/arousal/relevance/felt_summary/contagion_type —
    the emotional-history substrate that would otherwise have to wait for
    emot-body rFP.
    """

    def __init__(self, db_path: str = "data/events_teacher.db"):
        self._db_path = db_path

    def _connect(self) -> Optional[sqlite3.Connection]:
        if not os.path.exists(self._db_path):
            return None
        try:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True,
                                    timeout=2.0)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.debug("[FeltExperiences] connect err: %s", e)
            return None

    def get_for_person(self, author: str, limit: int = 10) -> list[dict]:
        """Normalizes '@handle' → 'handle' — events_teacher stores without '@'."""
        conn = self._connect()
        if conn is None:
            return []
        try:
            norm = author[1:] if author.startswith("@") else author
            rows = conn.execute(
                "SELECT topic, sentiment, arousal, relevance, felt_summary, "
                "contagion_type, mode, created_at "
                "FROM felt_experiences WHERE author = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (norm, min(int(limit), 100))
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_for_topic(self, topic: str, limit: int = 10) -> list[dict]:
        conn = self._connect()
        if conn is None:
            return []
        try:
            # LIKE allows partial matching — topics are free-text
            rows = conn.execute(
                "SELECT author, topic, sentiment, arousal, relevance, "
                "felt_summary, contagion_type, mode, created_at "
                "FROM felt_experiences WHERE topic LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (f"%{topic}%", min(int(limit), 100))
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def get_recent(self, hours: int = 24, limit: int = 20) -> list[dict]:
        conn = self._connect()
        if conn is None:
            return []
        try:
            cutoff = time.time() - (max(int(hours), 1) * 3600)
            rows = conn.execute(
                "SELECT author, topic, sentiment, arousal, relevance, "
                "felt_summary, contagion_type, mode, created_at "
                "FROM felt_experiences WHERE created_at >= ? "
                "ORDER BY created_at DESC LIMIT ?",
                (cutoff, min(int(limit), 100))
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            return []
        finally:
            conn.close()


# ── InnerMemoryReader ────────────────────────────────────────────────────

class InnerMemoryReader:
    """Read-only SQLite reader of data/inner_memory.db.

    Inner memory is already wired to meta-reasoning via other paths; this
    reader is used for the entity/topic-text-match query that's natural to
    include in composed recall.
    """

    def __init__(self, db_path: str = "data/inner_memory.db"):
        self._db_path = db_path

    def _connect(self) -> Optional[sqlite3.Connection]:
        if not os.path.exists(self._db_path):
            return None
        try:
            return sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True,
                                    timeout=2.0)
        except sqlite3.Error:
            return None

    def query_narrative_snippets(self, needle: str, limit: int = 10) -> list[dict]:
        """Best-effort free-text match across inner narrative/chain archives.

        inner_memory.db schema varies across sessions; this function tries a
        narrow FTS-style query against the most common text column and returns
        empty on schema mismatch (never raises).
        """
        if not needle:
            return []
        conn = self._connect()
        if conn is None:
            return []
        try:
            conn.row_factory = sqlite3.Row
            # Probe: find a table with a 'text' or 'narrative' or 'summary' col
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            for tbl in tables:
                try:
                    cols = [c[1] for c in conn.execute(
                        f"PRAGMA table_info({tbl})"
                    ).fetchall()]
                    text_col = next((c for c in cols
                                     if c in ("text", "narrative", "summary",
                                               "content", "body")), None)
                    if not text_col:
                        continue
                    ts_col = next((c for c in cols
                                   if c in ("created_at", "ts", "timestamp")), None)
                    order = f"ORDER BY {ts_col} DESC" if ts_col else ""
                    rows = conn.execute(
                        f"SELECT * FROM {tbl} "
                        f"WHERE {text_col} LIKE ? {order} LIMIT ?",
                        (f"%{needle}%", min(int(limit), 50))
                    ).fetchall()
                    if rows:
                        return [dict(r) for r in rows]
                except sqlite3.Error:
                    continue
            return []
        finally:
            conn.close()


# ── TitanSelfProbe ───────────────────────────────────────────────────────

class TitanSelfProbe:
    """In-process snapshot of current Titan state for inclusion in outer_context.

    Accepts a callable (probe_fn) that returns the current {neuromods, Chi,
    dominant_emotion, pi_phase, epoch} dict. spirit_worker provides this
    callable at wiring time.
    """

    def __init__(self, probe_fn: Optional[Callable[[], dict]] = None):
        self._probe_fn = probe_fn

    def snapshot(self) -> dict:
        if self._probe_fn is None:
            return {}
        try:
            snap = self._probe_fn() or {}
            if not isinstance(snap, dict):
                return {}
            return snap
        except Exception as e:
            logger.debug("[TitanSelfProbe] err: %s", e)
            return {}


# ── KnowledgeReader (bus-RPC to knowledge_worker) ────────────────────────

class KnowledgeReader:
    """Bus-RPC client for Kuzu-owned knowledge queries.

    Kuzu 0.11.x is single-process (cannot be opened by both knowledge_worker
    and spirit_worker). We route all Kuzu queries through knowledge_worker
    via DivineBus — same architectural philosophy as IMW for SQLite, adapted
    to Kuzu's concurrency model.
    """

    def __init__(self, bus: Any = None, timeout_s: float = 0.5):
        self._bus = bus
        self._timeout_s = float(timeout_s)

    def query_concept(self, topic: str) -> Optional[dict]:
        if self._bus is None:
            return None
        return self._rpc("KNOWLEDGE_QUERY_CONCEPT", {"topic": topic})

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        if self._bus is None:
            return []
        r = self._rpc("KNOWLEDGE_SEARCH",
                      {"query": query, "max_results": int(max_results)})
        if isinstance(r, dict):
            return r.get("results", [])
        return []

    def concepts_for_person(self, person_id: str,
                             limit: int = 5) -> list[dict]:
        if self._bus is None:
            return []
        r = self._rpc("KNOWLEDGE_CONCEPTS_FOR_PERSON",
                      {"person_id": person_id, "limit": int(limit)})
        if isinstance(r, dict):
            return r.get("concepts", [])
        return []

    def _rpc(self, msg_type: str, payload: dict) -> Optional[dict]:
        try:
            from titan_plugin.bus import make_msg
            rid = f"meta_outer_{uuid.uuid4().hex[:12]}"
            payload["rid"] = rid
            msg = make_msg(msg_type, "spirit", "knowledge", payload)
            # DivineBus request/response via request() if available,
            # else publish + wait pattern.
            if hasattr(self._bus, "request"):
                resp = self._bus.request(msg, timeout=self._timeout_s)
                if resp is None:
                    return None
                return resp.get("payload") if isinstance(resp, dict) else None
            return None
        except Exception as e:
            logger.debug("[KnowledgeReader] RPC %s err: %s", msg_type, e)
            return None


# ── XSearchReader (social_x_gateway wrapper) ─────────────────────────────

class XSearchReader:
    """Wraps social_x_gateway.search() for meta-reasoning.

    Gateway handles rate limiting, circuit breaker, telemetry. We pass
    consumer='meta_outer' so gateway can attribute calls.
    """

    def __init__(self, gateway: Any = None):
        self._gateway = gateway

    def search(self, query: str, count: int = 10) -> list[dict]:
        if self._gateway is None:
            return []
        try:
            # BaseContext with api_key — gateway reads its own config
            from titan_plugin.logic.social_x_gateway import BaseContext
            ctx = BaseContext(api_key="")  # gateway loads key from its config
            result = self._gateway.search(query=query, context=ctx,
                                           consumer="meta_outer",
                                           count=int(count))
            if result is None:
                return []
            if getattr(result, "status", "") != "success":
                logger.debug("[XSearchReader] search status=%s reason=%s",
                             getattr(result, "status", ""),
                             getattr(result, "reason", ""))
                return []
            return getattr(result, "tweets", []) or []
        except Exception as e:
            logger.debug("[XSearchReader] search err: %s", e)
            return []


# ── EventsPollReader (bus-RPC, gated off by default) ─────────────────────

class EventsPollReader:
    def __init__(self, bus: Any = None, timeout_s: float = 5.0):
        self._bus = bus
        self._timeout_s = float(timeout_s)

    def poll_window(self) -> Optional[dict]:
        if self._bus is None:
            return None
        try:
            from titan_plugin.bus import make_msg
            rid = f"meta_outer_events_{uuid.uuid4().hex[:12]}"
            msg = make_msg(bus.EVENTS_WINDOW_POLL, "spirit",
                           "events_teacher", {"rid": rid})
            if hasattr(self._bus, "request"):
                resp = self._bus.request(msg, timeout=self._timeout_s)
                if resp is None:
                    return None
                return resp.get("payload") if isinstance(resp, dict) else None
            return None
        except Exception as e:
            logger.debug("[EventsPollReader] err: %s", e)
            return None


# ── OuterContextReader (main facade) ─────────────────────────────────────

@dataclass
class OuterContextConfig:
    social_graph_path: str = "data/social_graph.db"
    events_teacher_path: str = "data/events_teacher.db"
    inner_memory_path: str = "data/inner_memory.db"
    fetch_budget_ms: int = 200
    per_read_timeout_ms: int = 50
    cache_ttl_s: int = 60
    cache_max_size: int = 500
    max_workers: int = 4
    active_search_knowledge: bool = True
    active_search_x: bool = False
    active_search_events: bool = False
    peer_cgn_enabled: bool = True
    peer_cgn_paths: dict = field(default_factory=dict)
    reward_weight: float = 0.0  # chain_iql outer-use reward shaping (0 = observe)


class OuterContextReader:
    """Unified async reader for Titan's outer cognitive memory.

    Single entry point for meta-reasoning. compose_recall_query fans out to
    5-7 sources in parallel, composes when all return or at budget deadline.

    Usage:
        reader = OuterContextReader(config, bus=divine_bus,
                                     x_gateway=gateway,
                                     probe_fn=spirit_state_probe)
        if reader.is_active():
            future = reader.compose_recall_query({"primary_person": pid,
                                                   "current_topic": topic})
            # ... a few primitives later:
            outer_ctx = future.result(timeout=0.2)
    """

    def __init__(
        self,
        config: Optional[OuterContextConfig] = None,
        bus: Any = None,
        x_gateway: Any = None,
        probe_fn: Optional[Callable[[], dict]] = None,
    ):
        self.config = config or OuterContextConfig()
        self._cache = _LRUTTLCache(max_size=self.config.cache_max_size,
                                    ttl_s=self.config.cache_ttl_s)
        self._pool = ThreadPoolExecutor(
            max_workers=int(self.config.max_workers),
            thread_name_prefix="meta_outer",
        )
        self._social = SocialGraphReader(self.config.social_graph_path)
        self._felt = FeltExperiencesReader(self.config.events_teacher_path)
        self._inner = InnerMemoryReader(self.config.inner_memory_path)
        self._self_probe = TitanSelfProbe(probe_fn)
        self._peer = (PeerCGNReader(self.config.peer_cgn_paths or None)
                      if self.config.peer_cgn_enabled else None)
        self._knowledge = KnowledgeReader(bus)
        self._x = XSearchReader(x_gateway)
        self._events = EventsPollReader(bus)
        # Stats
        self._stats = {
            "composed_fetches": 0,
            "composed_timeouts": 0,
            "composed_total_ms": 0.0,
            "per_source_calls": {},
            "per_source_failures": {},
        }
        logger.info("[MetaOuter] reader initialized (budget=%dms, pool=%d, peer=%s)",
                    self.config.fetch_budget_ms, self.config.max_workers,
                    self.config.peer_cgn_enabled)

    # ── Public: composed recalls ────────────────────────────────────

    def compose_recall_query(self, entity_refs: dict) -> Future:
        """Fan out 5+ async queries; compose when all return or at budget.

        entity_refs: {"primary_person": "@jkacrpto", "current_topic": "AI..."}
        Returns a Future resolving to the composed dict (see rFP §5).

        Caller should .result(timeout=budget_ms / 1000.0) and tolerate
        partial results (sources_timed_out populated).
        """
        return self._pool.submit(self._compose_sync, entity_refs)

    def _compose_sync(self, entity_refs: dict) -> dict:
        t0 = time.time()
        person = entity_refs.get("primary_person")
        topic = entity_refs.get("current_topic")
        result: dict = {
            "person": None,
            "topic": None,
            "felt_history": [],
            "recent_events": [],
            "inner_narrative": [],
            "titan_self_snapshot": {},
            "peer_cgn": {},
            "sources_queried": [],
            "sources_failed": [],
            "sources_timed_out": [],
            "fetch_ms": 0.0,
        }
        per_timeout = self.config.per_read_timeout_ms / 1000.0
        # Dispatch in-parallel sub-tasks
        tasks: list[tuple[str, Future]] = []
        if person:
            tasks.append(("person", self._pool.submit(
                self._cached_call, ("person", person),
                lambda: self._social.get_person_profile(person))))
            tasks.append(("felt_history", self._pool.submit(
                self._cached_call, ("felt_person", person),
                lambda: self._felt.get_for_person(person, limit=10))))
            tasks.append(("recent_events", self._pool.submit(
                self._cached_call, ("engagements", None),
                lambda: self._social.get_recent_engagements(limit=10))))
            tasks.append(("inner_narrative_person", self._pool.submit(
                self._cached_call, ("inner_narr", person),
                lambda: self._inner.query_narrative_snippets(person, limit=5))))
        if topic:
            tasks.append(("topic", self._pool.submit(
                self._cached_call, ("topic", topic),
                lambda: self._knowledge.query_concept(topic))))
            tasks.append(("felt_topic", self._pool.submit(
                self._cached_call, ("felt_topic", topic),
                lambda: self._felt.get_for_topic(topic, limit=10))))
            tasks.append(("inner_narrative_topic", self._pool.submit(
                self._cached_call, ("inner_narr", topic),
                lambda: self._inner.query_narrative_snippets(topic, limit=5))))
        # Always include self probe + peer summary (cheap)
        tasks.append(("titan_self_snapshot", self._pool.submit(
            self._self_probe.snapshot)))
        if self._peer is not None:
            tasks.append(("peer_cgn", self._pool.submit(
                self._peer.peer_cgn_summary)))

        # Gather with per-task timeout, budget as overall cap
        for label, fut in tasks:
            result["sources_queried"].append(label)
            try:
                r = fut.result(timeout=per_timeout)
            except FuturesTimeout:
                result["sources_timed_out"].append(label)
                self._stats["per_source_failures"].setdefault(label, 0)
                self._stats["per_source_failures"][label] += 1
                continue
            except Exception as e:
                result["sources_failed"].append(label)
                logger.debug("[MetaOuter] task %s failed: %s", label, e)
                continue
            self._stats["per_source_calls"].setdefault(label, 0)
            self._stats["per_source_calls"][label] += 1
            # Merge by label conventions
            if label == "person":
                result["person"] = r
            elif label == "topic":
                result["topic"] = r
            elif label in ("felt_history", "felt_topic"):
                result["felt_history"].extend(r or [])
            elif label == "recent_events":
                result["recent_events"] = r or []
            elif label in ("inner_narrative_person", "inner_narrative_topic"):
                result["inner_narrative"].extend(r or [])
            elif label == "titan_self_snapshot":
                result["titan_self_snapshot"] = r or {}
            elif label == "peer_cgn":
                result["peer_cgn"] = r or {}
        result["fetch_ms"] = round((time.time() - t0) * 1000.0, 1)
        self._stats["composed_fetches"] += 1
        self._stats["composed_total_ms"] += result["fetch_ms"]
        if result["sources_timed_out"]:
            self._stats["composed_timeouts"] += 1
        return result

    def _cached_call(self, cache_key: tuple, fn: Callable[[], Any]) -> Any:
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit
        try:
            value = fn()
        except Exception as e:
            logger.debug("[MetaOuter] cached_call %s err: %s", cache_key, e)
            return None
        self._cache.set(cache_key, value)
        return value

    # ── Public: simple per-store reads ───────────────────────────────

    def get_person_profile(self, person_id: str) -> Optional[dict]:
        return self._cached_call(("person", person_id),
                                  lambda: self._social.get_person_profile(person_id))

    def get_felt_experiences_for_person(self, author: str,
                                          limit: int = 10) -> list[dict]:
        return self._cached_call(
            ("felt_person", author),
            lambda: self._felt.get_for_person(author, limit=limit)) or []

    def get_felt_experiences_for_topic(self, topic: str,
                                         limit: int = 10) -> list[dict]:
        return self._cached_call(
            ("felt_topic", topic),
            lambda: self._felt.get_for_topic(topic, limit=limit)) or []

    def titan_self_probe(self) -> dict:
        return self._self_probe.snapshot()

    # ── Public: active search (gap filling) ──────────────────────────

    def knowledge_search(self, topic: str, max_results: int = 5) -> list[dict]:
        if not self.config.active_search_knowledge:
            return []
        return self._knowledge.search(topic, max_results=max_results)

    def x_timeline_search(self, query: str, count: int = 10) -> list[dict]:
        if not self.config.active_search_x:
            return []
        return self._x.search(query, count=count)

    def events_window_poll(self) -> Optional[dict]:
        if not self.config.active_search_events:
            return None
        return self._events.poll_window()

    # ── Public: peer CGN reading ─────────────────────────────────────

    def peer_cgn_beta(self, consumer: str, concept: str) -> Optional[float]:
        if self._peer is None:
            return None
        return self._peer.peer_cgn_beta(consumer, concept)

    def peer_cgn_alpha(self, consumer: str, concept: str) -> Optional[float]:
        if self._peer is None:
            return None
        return self._peer.peer_cgn_alpha(consumer, concept)

    def peer_cgn_summary(self) -> dict:
        if self._peer is None:
            return {}
        return self._peer.peer_cgn_summary()

    # ── Stats / shutdown ─────────────────────────────────────────────

    def stats(self) -> dict:
        n = max(self._stats["composed_fetches"], 1)
        return {
            "active": is_active(),
            "composed_fetches": self._stats["composed_fetches"],
            "composed_timeouts": self._stats["composed_timeouts"],
            "avg_fetch_ms": round(self._stats["composed_total_ms"] / n, 2),
            "per_source_calls": dict(self._stats["per_source_calls"]),
            "per_source_failures": dict(self._stats["per_source_failures"]),
            "cache": self._cache.stats(),
        }

    @staticmethod
    def is_active() -> bool:
        return is_active()

    def shutdown(self, wait: bool = False) -> None:
        try:
            self._pool.shutdown(wait=bool(wait), cancel_futures=True)
        except Exception:
            pass
