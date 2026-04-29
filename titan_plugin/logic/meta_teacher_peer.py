"""
titan_plugin/logic/meta_teacher_peer.py — Phase D.1 + D.2 of rFP_meta_teacher_v2.

Cross-Titan teaching exchange — teachers consult each other on stuck topics
via a stats-only RPC. No stylistic prescriptions cross the wire; all data
flows through OVG-style envelope validation outbound and Guardian-style
inbound validation. Trust boundary is unified with the existing outbound
X / social safety layer.

Transport
---------
DivineBus is in-process only — no built-in TCP. Cross-Titan peer queries
ride plain HTTP via the dashboard endpoints `/v4/meta-teacher/peer/query`
(POST). Bus messages METATEACHER_PEER_QUERY / _RESPONSE remain defined
in `bus.py` for INTRA-Titan observability + bus-census visibility, but the
network call is plain `httpx` (lazy import).

Phase D.2 — IQL-Learned Peer-Query Policy
-----------------------------------------
A small per-domain EMA of "did peer-querying this domain → adoption?"
shapes a soft gate on whether to issue a query. Two flags split observation
from learning so D.1 soak collects data safely:

  peer_query_feature_logging   (default true) — every issued peer query is
      logged + a marker is stored to credit/debit the policy on the
      subsequent same-domain critique.

  peer_query_reward_learning_enabled (default false) — when ON, the
      should_peer_query() gate consults the EMA. When OFF, the gate just
      returns True (per D.1 still_needs_push trigger semantics).

Policy state persists at data/reasoning/peer_query_policy.json — sidecar
to chain_iql.json so it lives near other learned-policy state. It does
NOT couple to ChainIQL's template Q-net; the policy is logically separate.

Question-type whitelist (rFP §2 Phase D.1):
    quality_trajectory
    adoption_rate
    still_needs_push_similar_topics
    voice_summary

Inbound responses on these question_types are guaranteed stats-only by
the producer side: `build_response_for_query()` reads from on-disk state
(teaching_journal.jsonl, voice_state.json) and returns numeric/structured
fields with no free-text from the answering teacher.

See rFP_meta_teacher_v2_content_awareness_memory.md §2 Phase D.1 + D.2.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from typing import Any, Optional
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger("titan.meta_teacher_peer")


# ── Constants ──────────────────────────────────────────────────────────────

PEER_QUERY_LOG_FILENAME = "peer_query_log.jsonl"
PEER_POLICY_FILENAME = "peer_query_policy.json"
RECENT_OBS_BUFFER_SIZE = 50          # in-memory cache of inbound responses

# Question-type whitelist — rFP §2 Phase D.1.
ALLOWED_QUESTION_TYPES = (
    "quality_trajectory",
    "adoption_rate",
    "still_needs_push_similar_topics",
    "voice_summary",
)

# Maximum free-text fields per response (defense in depth — Guardian
# pre_prompt mirror; we expect responses to be stats-only and reject any
# response payload that carries strings longer than this).
RESPONSE_FREE_TEXT_FIELDS = ("note",)
RESPONSE_FREE_TEXT_MAX_CHARS = 200

# Outbound envelope size cap — ~2KB on the wire.
OUTBOUND_ENVELOPE_MAX_BYTES = 2048
INBOUND_RESPONSE_MAX_BYTES = 8192


def _now_floor_hour(ts: float) -> int:
    return int(ts // 3600)


# ── PeerQueryPolicy (Phase D.2) ────────────────────────────────────────────

class PeerQueryPolicy:
    """Per-domain EMA of (adoption | peer_queried).

    Sidecar persistence at data/reasoning/peer_query_policy.json. Policy
    initializes neutral (EMA=0.5 per domain) and updates ONLY when
    feature_logging is on AND a real outcome arrives.

    Two-flag gating:
      - feature_logging_enabled: whether record_peer_query / record_outcome
        actually mutate the policy. Default True; safe-on observation.
      - reward_learning_enabled: whether should_peer_query consults the
        EMA. Default False; D.1 trigger semantics are unchanged when off.

    Public surface:
      record_peer_query(domain, chain_id, question_type, ts) → marker
      record_outcome(chain_id, domain, adopted, quality_delta) → +0.05/-0.03
      should_peer_query(domain) → (allowed, reason, ema)
      peek_chain_marker(chain_id) → bool (used by worker to credit reward)
      consume_chain_marker(chain_id) → bool (one-shot credit)
      snapshot() → telemetry dict
      save() / load() — JSON sidecar
    """

    DEFAULT_EMA_NEUTRAL = 0.5
    CHAIN_MARKER_TTL_S = 24 * 3600.0   # marker auto-evicts after 24h

    def __init__(self, config: Optional[dict] = None, data_dir: str = "./data"):
        cfg = config or {}
        self._feature_logging = bool(
            cfg.get("peer_query_feature_logging", True))
        self._reward_learning = bool(
            cfg.get("peer_query_reward_learning_enabled", False))
        self._reward_adopted = float(
            cfg.get("peer_query_reward_adopted", 0.05))
        self._reward_unadopted = float(
            cfg.get("peer_query_reward_unadopted", -0.03))
        self._ema_alpha = float(cfg.get("peer_query_ema_alpha", 0.2))
        self._min_ema = float(cfg.get("peer_query_min_ema_to_query", 0.3))

        save_dir = os.path.join(data_dir, "reasoning")
        self._policy_path = os.path.join(save_dir, PEER_POLICY_FILENAME)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            swallow_warn(f'[PeerQueryPolicy] mkdir {save_dir} failed', e,
                         key="logic.meta_teacher_peer.mkdir_failed", throttle=100)

        # Per-domain learned scalar EMA in [0, 1] — interpreted as
        # "P(adoption | peer_queried)" for this domain. Higher = peer
        # queries pay off; lower = avoid.
        self._domain_ema: dict[str, float] = {}
        # chain_id → (domain, marker_ts) — evicted on outcome OR after TTL.
        self._chain_markers: dict[int, tuple[str, float]] = {}
        # Telemetry counters
        self._queries_logged = 0
        self._outcomes_applied = 0
        self._gate_allow = 0
        self._gate_block = 0
        self._loaded = False

    @property
    def feature_logging_enabled(self) -> bool:
        return self._feature_logging

    @property
    def reward_learning_enabled(self) -> bool:
        return self._reward_learning

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(self._policy_path):
            return
        try:
            with open(self._policy_path) as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return
            raw = payload.get("domain_ema", {}) or {}
            self._domain_ema = {
                str(k): float(v) for k, v in raw.items()
                if isinstance(v, (int, float))}
            raw_markers = payload.get("chain_markers", {}) or {}
            now = time.time()
            for cid_str, info in raw_markers.items():
                try:
                    cid = int(cid_str)
                    dom, ts = info
                    if now - float(ts) <= self.CHAIN_MARKER_TTL_S:
                        self._chain_markers[cid] = (str(dom), float(ts))
                except Exception:
                    continue
            self._queries_logged = int(payload.get("queries_logged", 0))
            self._outcomes_applied = int(payload.get("outcomes_applied", 0))
            self._gate_allow = int(payload.get("gate_allow", 0))
            self._gate_block = int(payload.get("gate_block", 0))
            logger.info(
                "[PeerQueryPolicy] Loaded: domains=%d, markers=%d, "
                "queries_logged=%d, outcomes=%d, allow=%d, block=%d",
                len(self._domain_ema), len(self._chain_markers),
                self._queries_logged, self._outcomes_applied,
                self._gate_allow, self._gate_block)
        except Exception as e:
            logger.debug("[PeerQueryPolicy] load failed: %s", e)

    def save(self) -> None:
        tmp = self._policy_path + ".tmp"
        try:
            payload = {
                "domain_ema": dict(self._domain_ema),
                "chain_markers": {
                    str(cid): [dom, ts]
                    for cid, (dom, ts) in self._chain_markers.items()},
                "queries_logged": self._queries_logged,
                "outcomes_applied": self._outcomes_applied,
                "gate_allow": self._gate_allow,
                "gate_block": self._gate_block,
                "feature_logging_enabled": self._feature_logging,
                "reward_learning_enabled": self._reward_learning,
            }
            with open(tmp, "w") as f:
                json.dump(payload, f, sort_keys=True)
            os.replace(tmp, self._policy_path)
        except Exception as e:
            swallow_warn('[PeerQueryPolicy] save failed', e,
                         key="logic.meta_teacher_peer.save_failed", throttle=100)
            try:
                os.remove(tmp)
            except OSError:
                pass

    def record_peer_query(
        self, domain: str, chain_id: int,
        question_type: str = "still_needs_push_similar_topics",
        ts: Optional[float] = None,
    ) -> bool:
        """Mark a chain_id as 'used a peer query.' Caller must follow up
        with record_outcome() once the next same-domain chain resolves
        adoption. Returns True if the marker was recorded.
        """
        if not self._feature_logging:
            return False
        if not self._loaded:
            self.load()
        ts_now = float(ts if ts is not None else time.time())
        self._chain_markers[int(chain_id)] = (str(domain or "general"), ts_now)
        self._queries_logged += 1
        self._evict_stale_markers(ts_now)
        return True

    def peek_chain_marker(self, chain_id: int) -> bool:
        """True if chain_id is currently marked as 'peer-query-used'."""
        if not self._loaded:
            self.load()
        return int(chain_id) in self._chain_markers

    def consume_chain_marker(self, chain_id: int) -> bool:
        """One-shot consume the marker (idempotent). Returns prior True/False.

        Removes the entry on success — adoption-time reward shaping should
        only credit a chain once.
        """
        if not self._loaded:
            self.load()
        cid = int(chain_id)
        if cid in self._chain_markers:
            self._chain_markers.pop(cid, None)
            return True
        return False

    def record_outcome(
        self, chain_id: int, domain: str, adopted: bool,
        quality_delta: float,
    ) -> tuple[bool, float]:
        """Apply +reward_adopted on adopted, +reward_unadopted on
        unadopted+quality_delta≤0. Updates per-domain EMA. Returns
        (applied, new_ema).
        """
        if not self._feature_logging:
            return False, self._domain_ema.get(str(domain), self.DEFAULT_EMA_NEUTRAL)
        if not self._loaded:
            self.load()
        # Reward signal — three buckets per rFP §2 Phase D.2:
        #   adopted             → target 1.0, "peer query paid off"
        #   unadopted + qd ≤ 0  → target 0.0, "peer query did not help"
        #   unadopted + qd > 0  → no update; quality improved despite
        #                          unadoption, attribution unclear.
        # The +0.05/-0.03 reward magnitudes (self._reward_adopted /
        # _unadopted) are kept in config for downstream chain_iql buffer
        # integration (apply_external_reward blend); EMA step is the plain
        # ema_alpha so the gate responds at the configured rate.
        if adopted:
            target = 1.0
        elif float(quality_delta) <= 0.0:
            target = 0.0
        else:
            return False, self._domain_ema.get(str(domain), self.DEFAULT_EMA_NEUTRAL)
        cur = self._domain_ema.get(str(domain), self.DEFAULT_EMA_NEUTRAL)
        nudge = self._ema_alpha
        new_ema = cur * (1.0 - nudge) + target * nudge
        new_ema = max(0.0, min(1.0, new_ema))
        self._domain_ema[str(domain)] = round(new_ema, 4)
        self._outcomes_applied += 1
        # Drop the marker if still present (defensive — caller usually
        # consume_chain_marker first).
        self._chain_markers.pop(int(chain_id), None)
        self.save()
        return True, new_ema

    def should_peer_query(self, domain: str) -> tuple[bool, str, float]:
        """Soft-gate. When reward learning is OFF: always allow.

        When ON: allow iff per-domain EMA ≥ peer_query_min_ema_to_query.
        Returns (allowed, reason, ema_value).
        """
        if not self._loaded:
            self.load()
        ema = self._domain_ema.get(str(domain), self.DEFAULT_EMA_NEUTRAL)
        if not self._reward_learning:
            self._gate_allow += 1
            return True, "reward_learning_disabled (always-allow)", ema
        if ema >= self._min_ema:
            self._gate_allow += 1
            return True, f"ema {ema:.3f} >= {self._min_ema:.3f}", ema
        self._gate_block += 1
        return False, f"ema {ema:.3f} < {self._min_ema:.3f}", ema

    def snapshot(self) -> dict:
        return {
            "feature_logging_enabled": self._feature_logging,
            "reward_learning_enabled": self._reward_learning,
            "reward_adopted": self._reward_adopted,
            "reward_unadopted": self._reward_unadopted,
            "ema_alpha": self._ema_alpha,
            "min_ema_to_query": self._min_ema,
            "domain_ema": dict(self._domain_ema),
            "active_chain_markers": len(self._chain_markers),
            "queries_logged": self._queries_logged,
            "outcomes_applied": self._outcomes_applied,
            "gate_allow": self._gate_allow,
            "gate_block": self._gate_block,
            "policy_path": self._policy_path,
        }

    def _evict_stale_markers(self, now: float) -> None:
        cutoff = now - self.CHAIN_MARKER_TTL_S
        stale = [cid for cid, (_, ts) in self._chain_markers.items()
                 if ts < cutoff]
        for cid in stale:
            self._chain_markers.pop(cid, None)


# ── Envelope validation (OVG / Guardian mirrors) ───────────────────────────

def validate_outbound_envelope(envelope: dict) -> tuple[bool, str]:
    """OVG-side outbound check. Stateless.

    Validates:
      - shape: src_titan + target_titan + question_type + topic_key + rid
      - whitelist match on question_type
      - size cap
      - no PII bleeding into topic_key (no raw email-like, no
        @-handle when handle redaction is required for the question_type;
        for now we only block obvious raw email patterns)
    """
    if not isinstance(envelope, dict):
        return False, "envelope not a dict"
    for k in ("src_titan", "target_titan", "question_type", "rid"):
        if k not in envelope or not str(envelope.get(k) or ""):
            return False, f"envelope missing {k}"
    qt = str(envelope.get("question_type"))
    if qt not in ALLOWED_QUESTION_TYPES:
        return False, f"question_type {qt!r} not in whitelist"
    tk = str(envelope.get("topic_key") or "")
    # Defensive PII guard — block obvious raw email pattern.
    if "@" in tk and "." in tk and " " not in tk and len(tk.split("@")[0]) > 0:
        # Could be either a person handle (@jkacrpto) OR an email
        # (a@b.com). If it has a dot in the post-@ part AND the post-@
        # has no slashes, treat as email and reject.
        try:
            post = tk.split("@", 1)[1]
            if "." in post and "/" not in post and " " not in post:
                # Reject — likely email
                return False, "topic_key contains email-like pattern"
        except Exception:
            pass
    # Size cap
    try:
        size = len(json.dumps(envelope, separators=(",", ":")).encode("utf-8"))
    except Exception:
        return False, "envelope not serializable"
    if size > OUTBOUND_ENVELOPE_MAX_BYTES:
        return False, f"envelope size {size}B exceeds cap {OUTBOUND_ENVELOPE_MAX_BYTES}B"
    return True, ""


def validate_inbound_response(
    envelope: dict, expected_rid: Optional[str] = None,
    expected_question_type: Optional[str] = None,
) -> tuple[bool, str]:
    """Guardian-side inbound check. Stateless.

    Validates:
      - shape: answering_titan + question_type + rid + data
      - rid round-trip if expected_rid is given
      - question_type whitelist + match expected
      - size cap
      - no unsolicited free-text — only `note` field allowed and length
        capped at RESPONSE_FREE_TEXT_MAX_CHARS
    """
    if not isinstance(envelope, dict):
        return False, "envelope not a dict"
    for k in ("answering_titan", "question_type", "rid", "data"):
        if k not in envelope:
            return False, f"envelope missing {k}"
    qt = str(envelope.get("question_type"))
    if qt not in ALLOWED_QUESTION_TYPES:
        return False, f"question_type {qt!r} not in whitelist"
    if expected_question_type and qt != expected_question_type:
        return False, (
            f"question_type {qt!r} does not match expected "
            f"{expected_question_type!r}")
    if expected_rid and str(envelope.get("rid")) != str(expected_rid):
        return False, f"rid mismatch (got {envelope.get('rid')!r})"
    data = envelope.get("data")
    if not isinstance(data, dict):
        return False, "data must be a dict"
    # Free-text guard: ONLY `note` allowed for free text; everything else
    # must be number / bool / list / dict. Strings inside lists OR dict
    # values are allowed if they're under the per-field cap (topic_keys etc.).
    for k, v in data.items():
        if isinstance(v, str) and k not in RESPONSE_FREE_TEXT_FIELDS:
            if len(v) > RESPONSE_FREE_TEXT_MAX_CHARS:
                return False, (
                    f"data.{k} string longer than "
                    f"{RESPONSE_FREE_TEXT_MAX_CHARS} chars")
    note = str(data.get("note") or "")
    if len(note) > RESPONSE_FREE_TEXT_MAX_CHARS:
        return False, "data.note longer than allowed"
    try:
        size = len(json.dumps(envelope, separators=(",", ":")).encode("utf-8"))
    except Exception:
        return False, "envelope not serializable"
    if size > INBOUND_RESPONSE_MAX_BYTES:
        return False, f"response size {size}B exceeds cap {INBOUND_RESPONSE_MAX_BYTES}B"
    return True, ""


# ── Response builders (server-side; reads on-disk state) ───────────────────

def build_response_for_query(
    question_type: str, topic_key: str, data_dir: str,
) -> tuple[bool, dict, str]:
    """Render a stats-only response from local on-disk state.

    Returns (ok, data_dict, reason). The data_dict is the response.data
    portion — caller wraps in the full envelope. On unknown
    question_type, returns (False, {}, reason).
    """
    if question_type not in ALLOWED_QUESTION_TYPES:
        return False, {}, f"unknown question_type {question_type!r}"
    journal_path = os.path.join(
        data_dir, "meta_teacher", "teaching_journal.jsonl")
    voice_path = os.path.join(data_dir, "meta_teacher", "voice_state.json")
    if question_type == "voice_summary":
        if not os.path.exists(voice_path):
            return True, {"applied_count": 0, "domain_biases": {},
                          "topic_suppressions": []}, ""
        try:
            with open(voice_path) as f:
                raw = json.load(f)
            return True, {
                "applied_count": int(raw.get("applied_count", 0)),
                "domain_biases": dict(raw.get("domain_biases", {})),
                "topic_suppressions": list(raw.get("topic_suppressions", [])),
            }, ""
        except Exception as e:
            return False, {}, f"voice read failed: {e}"
    # All other question_types read from teaching_journal.jsonl latest-row map.
    latest: dict[str, dict] = {}
    if os.path.exists(journal_path):
        try:
            with open(journal_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    tk = row.get("topic_key")
                    if isinstance(tk, str):
                        latest[tk] = row
        except Exception as e:
            return False, {}, f"journal read failed: {e}"
    if question_type == "quality_trajectory":
        cold = latest.get(topic_key)
        if cold is None:
            return True, {"observed": False}, ""
        traj = list(cold.get("quality_trajectory") or [])[-20:]
        return True, {
            "observed": True,
            "critique_count": int(cold.get("critique_count", 0)),
            "quality_delta": round(float(cold.get("quality_delta", 0.0)), 4),
            "still_needs_push": bool(cold.get("still_needs_push", False)),
            "trajectory_tail": [
                {"ts": float(p.get("ts", 0.0)),
                 "q": round(float(p.get("chain_quality", 0.0)), 3)}
                for p in traj],
        }, ""
    if question_type == "adoption_rate":
        cold = latest.get(topic_key)
        if cold is None:
            return True, {"observed": False}, ""
        adop = list(cold.get("adoption_trajectory") or [])
        n = len(adop)
        adopted_n = sum(1 for r in adop if r.get("adopted_bool"))
        rate = round((adopted_n / n) if n else 0.0, 3)
        return True, {
            "observed": True,
            "n": n,
            "adopted_n": adopted_n,
            "adoption_rate": rate,
        }, ""
    if question_type == "still_needs_push_similar_topics":
        # Return up to 5 topic_keys whose still_needs_push is True and
        # whose canonical form shares the FIRST WORD of the queried topic.
        first = (topic_key or "").split("|")[0].split("::")[0].strip().lower()
        sim: list[dict] = []
        for tk, row in latest.items():
            if not row.get("still_needs_push"):
                continue
            tk_l = tk.lower()
            if first and first in tk_l:
                sim.append({
                    "topic_key": tk,
                    "critique_count": int(row.get("critique_count", 0)),
                    "quality_delta": round(
                        float(row.get("quality_delta", 0.0)), 4),
                })
        sim.sort(key=lambda r: -int(r.get("critique_count", 0)))
        return True, {
            "matches": sim[:5],
            "match_count": len(sim),
        }, ""
    return False, {}, f"question_type {question_type!r} not handled"


# ── PeerExchangeClient ─────────────────────────────────────────────────────

class PeerExchangeClient:
    """Owns outbound peer query state — rate limits, recent observations,
    HTTP transport, peer query log, and the PeerQueryPolicy gate.

    Public surface (worker):
      - notify_critique()                  → no-op stub for symmetry
      - maybe_issue_query(...)             → async; gate + send + record
      - record_outcome(chain_id, domain, adopted, quality_delta) → policy update
      - format_recent_observation_for_topic(topic_key) → str | None
      - consume_peer_query_marker_for_chain(chain_id) → bool
      - prune_old_logs(now)                → retention sweep
      - snapshot()                         → telemetry dict

    Stateful:
      - per-Titan rolling deque of issued queries (rate limit window)
      - per-topic last_query_ts (cooldown)
      - per-topic recent_response (LRU-style; latest wins per topic)
      - PeerQueryPolicy embedded; persists separately
    """

    DEFAULT_RATE_LIMIT_PER_HOUR = 10
    DEFAULT_TOPIC_COOLDOWN_S = 86400.0
    DEFAULT_HTTP_TIMEOUT_S = 10.0
    DEFAULT_LOG_RETENTION_DAYS = 30

    def __init__(
        self, config: Optional[dict] = None, data_dir: str = "./data",
        my_titan_id: str = "t1",
        peer_endpoints: Optional[dict] = None,
    ):
        cfg = config or {}
        self._enabled = bool(cfg.get("peer_exchange_enabled", False))
        self._rate_limit = int(cfg.get(
            "peer_query_rate_limit_per_hour", self.DEFAULT_RATE_LIMIT_PER_HOUR))
        self._topic_cooldown_s = float(cfg.get(
            "peer_query_topic_cooldown_seconds", self.DEFAULT_TOPIC_COOLDOWN_S))
        self._min_snp_count = int(cfg.get(
            "peer_query_min_still_needs_push_count", 3))
        self._http_timeout = float(cfg.get(
            "peer_query_http_timeout_seconds", self.DEFAULT_HTTP_TIMEOUT_S))
        self._retention_days = int(cfg.get(
            "peer_query_log_retention_days", self.DEFAULT_LOG_RETENTION_DAYS))

        self._my_titan_id = str(my_titan_id or "t1").lower()
        self._data_dir = os.path.join(data_dir, "meta_teacher")
        self._log_path = os.path.join(self._data_dir, PEER_QUERY_LOG_FILENAME)
        try:
            os.makedirs(self._data_dir, exist_ok=True)
        except Exception as e:
            logger.debug("[PeerExchangeClient] mkdir failed: %s", e)

        # Filter self out of peer endpoints
        endpoints = peer_endpoints or {}
        self._peer_endpoints: dict[str, str] = {
            str(k).lower(): str(v) for k, v in endpoints.items()
            if str(k).lower() != self._my_titan_id and str(v)}

        # Rate-limit window (rolling 1h)
        self._issued_ts: deque[float] = deque(maxlen=max(1, self._rate_limit * 4))
        # Per-topic last query
        self._last_query_ts: dict[str, float] = {}
        # Per-topic recent observation (most recent response wins)
        self._recent_obs: dict[str, dict] = {}

        self._policy = PeerQueryPolicy(cfg, data_dir=data_dir)

        self._loaded = False

    # ── Properties ────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def my_titan_id(self) -> str:
        return self._my_titan_id

    @property
    def policy(self) -> PeerQueryPolicy:
        return self._policy

    # ── Boot-time load ────────────────────────────────────────────────
    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._policy.load()
        # Note: rate-limit / cooldown / recent_obs are in-memory only —
        # they re-bootstrap from log on cold start. Acceptable behavior:
        # restart loses recent peer-obs context (low cost).

    # ── Worker hook ───────────────────────────────────────────────────
    def consume_peer_query_marker_for_chain(self, chain_id: int) -> bool:
        """Idempotent one-shot read of the policy chain marker.

        Returns True iff this chain_id was tagged as 'peer-query-used' by
        a prior issue_query() call. Caller (worker) uses this to set
        peer_query_used flag on the pending_suggestions entry.
        """
        if not self._loaded:
            self.load()
        return self._policy.consume_chain_marker(chain_id)

    def record_outcome(
        self, chain_id: int, domain: str, adopted: bool,
        quality_delta: float,
    ) -> tuple[bool, float]:
        return self._policy.record_outcome(
            chain_id, domain, adopted, quality_delta)

    def format_recent_observation_for_topic(
        self, topic_key: str, now: Optional[float] = None,
    ) -> Optional[str]:
        """Render a one-line peer-observation string for the prompt.

        Returns None when no observation exists or the cached entry is
        older than 7 days (peer obs go stale; a future critique on the
        same topic should fetch fresh).
        """
        if not self._loaded:
            self.load()
        rec = self._recent_obs.get(topic_key)
        if not rec:
            return None
        ts_now = float(now if now is not None else time.time())
        if ts_now - float(rec.get("ts", 0.0)) > 7 * 86400.0:
            self._recent_obs.pop(topic_key, None)
            return None
        peer = rec.get("answering_titan", "?")
        qt = rec.get("question_type", "?")
        data = rec.get("data") or {}
        # One-line stats summary tailored to question_type.
        parts: list[str] = [f"[{peer} • {qt}]"]
        if qt == "still_needs_push_similar_topics":
            mc = int(data.get("match_count", 0))
            tops = data.get("matches") or []
            if tops:
                head = tops[0]
                parts.append(
                    f"{mc} similar stuck topic(s); top={head.get('topic_key')!r} "
                    f"n={head.get('critique_count', 0)} "
                    f"Δq={head.get('quality_delta', 0.0):+.2f}")
            else:
                parts.append("0 similar stuck topics")
        elif qt == "quality_trajectory":
            if data.get("observed"):
                parts.append(
                    f"n={data.get('critique_count', 0)} "
                    f"Δq={data.get('quality_delta', 0.0):+.2f} "
                    f"stuck={data.get('still_needs_push', False)}")
            else:
                parts.append("not observed by peer")
        elif qt == "adoption_rate":
            if data.get("observed"):
                parts.append(
                    f"adoption_rate={data.get('adoption_rate', 0.0):.2f} "
                    f"(n={data.get('n', 0)})")
            else:
                parts.append("not observed by peer")
        elif qt == "voice_summary":
            ac = int(data.get("applied_count", 0))
            db = data.get("domain_biases") or {}
            parts.append(f"applied_count={ac}, biased_domains={len(db)}")
        return " ".join(parts)

    # ── Outbound flow ─────────────────────────────────────────────────
    def _rate_limit_ok(self, now: float) -> bool:
        cutoff = now - 3600.0
        # Drop old entries
        while self._issued_ts and self._issued_ts[0] < cutoff:
            self._issued_ts.popleft()
        return len(self._issued_ts) < self._rate_limit

    def _cooldown_ok(self, topic_key: str, now: float) -> bool:
        last = float(self._last_query_ts.get(topic_key, 0.0))
        return (now - last) >= self._topic_cooldown_s

    def _build_envelope(
        self, target_titan: str, question_type: str, topic_key: str,
        chain_id: int, ts: float,
    ) -> dict:
        return {
            "src_titan": self._my_titan_id,
            "target_titan": str(target_titan).lower(),
            "question_type": str(question_type),
            "topic_key": str(topic_key)[:200],
            "chain_id": int(chain_id),
            "ts": float(ts),
            "rid": str(uuid.uuid4()),
            "envelope_version": 1,
        }

    async def maybe_issue_query(
        self,
        topic_key: str, domain: str, cold_entry: dict, chain_id: int,
        now: Optional[float] = None,
    ) -> Optional[dict]:
        """Decide whether to peer-query, and if so, issue + record outcome.

        Returns the response dict on success, None on suppression or error.
        Suppression reasons are logged at DEBUG; failures at DEBUG too.
        """
        if not self._enabled:
            return None
        if not self._loaded:
            self.load()
        if not self._peer_endpoints:
            return None
        ts_now = float(now if now is not None else time.time())
        # Gate: still_needs_push count threshold
        n_crits = int(cold_entry.get("critique_count", 0))
        if n_crits < self._min_snp_count:
            return None
        # Gate: per-Titan rate limit
        if not self._rate_limit_ok(ts_now):
            logger.debug("[PeerExchangeClient] rate limit exceeded")
            return None
        # Gate: per-topic cooldown
        if not self._cooldown_ok(topic_key, ts_now):
            return None
        # Gate: D.2 policy
        allowed, reason, ema = self._policy.should_peer_query(domain)
        if not allowed:
            logger.debug(
                "[PeerExchangeClient] policy gate blocked: %s (ema=%.3f)",
                reason, ema)
            return None
        # Pick the FIRST configured peer (deterministic — could randomize later).
        target = next(iter(self._peer_endpoints.keys()))
        envelope = self._build_envelope(
            target, "still_needs_push_similar_topics",
            topic_key, chain_id, ts_now)
        ok, reason = validate_outbound_envelope(envelope)
        if not ok:
            logger.warning(
                "[PeerExchangeClient] outbound envelope invalid: %s", reason)
            return None
        # Mark BEFORE send so concurrent worker logic credits this chain.
        self._policy.record_peer_query(
            domain, chain_id, envelope["question_type"], ts=ts_now)
        self._issued_ts.append(ts_now)
        self._last_query_ts[topic_key] = ts_now

        url = self._peer_endpoints[target].rstrip("/") + "/v4/meta-teacher/peer/query"
        response: Optional[dict] = None
        latency_ms = 0
        sent_ok = False
        err: str = ""
        try:
            import httpx                                    # lazy
            t0 = time.time()
            async with httpx.AsyncClient(timeout=self._http_timeout) as cx:
                resp = await cx.post(url, json=envelope)
            latency_ms = int((time.time() - t0) * 1000)
            if resp.status_code == 200:
                try:
                    response = resp.json()
                except Exception as e:
                    err = f"json decode: {e}"
            else:
                err = f"HTTP {resp.status_code}"
        except Exception as e:
            err = str(e)
        if response is not None:
            ok, reason = validate_inbound_response(
                response,
                expected_rid=envelope["rid"],
                expected_question_type=envelope["question_type"])
            if ok:
                self._recent_obs[topic_key] = {
                    "ts": ts_now,
                    "answering_titan": response.get("answering_titan"),
                    "question_type": response.get("question_type"),
                    "data": response.get("data") or {},
                }
                sent_ok = True
            else:
                err = f"inbound invalid: {reason}"
        # Log every issued query (success + failure) for soak observability.
        self._append_log({
            "ts": ts_now,
            "src_titan": self._my_titan_id,
            "target_titan": envelope["target_titan"],
            "question_type": envelope["question_type"],
            "topic_key": envelope["topic_key"],
            "chain_id": envelope["chain_id"],
            "rid": envelope["rid"],
            "answered": bool(sent_ok),
            "latency_ms": int(latency_ms),
            "error": err or None,
        })
        return response if sent_ok else None

    # ── Inbound flow ─────────────────────────────────────────────────
    def handle_inbound_query(
        self, envelope: dict, data_dir: str,
    ) -> tuple[bool, dict, str]:
        """Server-side: validate envelope, render response, return tuple.

        Stateless except for logging the inbound. Used by the dashboard
        endpoint when another Titan calls /v4/meta-teacher/peer/query.

        Returns (ok, response_envelope, reason). On invalid envelope,
        response_envelope is empty and ok=False.
        """
        if not isinstance(envelope, dict):
            return False, {}, "envelope not a dict"
        for k in ("src_titan", "target_titan", "question_type", "rid"):
            if k not in envelope:
                return False, {}, f"envelope missing {k}"
        qt = str(envelope.get("question_type"))
        if qt not in ALLOWED_QUESTION_TYPES:
            return False, {}, f"question_type {qt!r} not in whitelist"
        # Verify target_titan matches us
        if str(envelope.get("target_titan", "")).lower() != self._my_titan_id:
            return False, {}, (
                f"target_titan {envelope.get('target_titan')!r} does not "
                f"match self {self._my_titan_id!r}")
        topic_key = str(envelope.get("topic_key") or "")
        ok, data, reason = build_response_for_query(qt, topic_key, data_dir)
        if not ok:
            return False, {}, reason
        response = {
            "answering_titan": self._my_titan_id,
            "question_type": qt,
            "rid": str(envelope.get("rid")),
            "ts": time.time(),
            "data": data,
        }
        ok2, reason2 = validate_inbound_response(
            response, expected_rid=str(envelope.get("rid")),
            expected_question_type=qt)
        if not ok2:
            return False, {}, f"self-validate failed: {reason2}"
        # Log inbound for observability (different log entry kind).
        self._append_log({
            "ts": time.time(),
            "src_titan": str(envelope.get("src_titan") or "?"),
            "target_titan": self._my_titan_id,
            "question_type": qt,
            "topic_key": topic_key,
            "chain_id": int(envelope.get("chain_id") or -1),
            "rid": str(envelope.get("rid")),
            "answered": True,
            "kind": "inbound_handled",
        })
        return True, response, ""

    # ── Persistence ───────────────────────────────────────────────────
    def _append_log(self, entry: dict) -> None:
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug("[PeerExchangeClient] log append failed: %s", e)

    def prune_old_logs(self, now: Optional[float] = None) -> int:
        """Trim peer_query_log.jsonl to retention_days. Returns lines kept."""
        if not os.path.exists(self._log_path):
            return 0
        ts_now = float(now if now is not None else time.time())
        cutoff = ts_now - (self._retention_days * 86400.0)
        kept: list[str] = []
        try:
            with open(self._log_path) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if float(row.get("ts", 0.0)) >= cutoff:
                        kept.append(line)
        except Exception as e:
            logger.debug("[PeerExchangeClient] prune read failed: %s", e)
            return 0
        tmp = self._log_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                for ln in kept:
                    f.write(ln + "\n")
            os.replace(tmp, self._log_path)
        except Exception as e:
            swallow_warn('[PeerExchangeClient] prune write failed', e,
                         key="logic.meta_teacher_peer.prune_write_failed", throttle=100)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return 0
        return len(kept)

    # ── Telemetry ─────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        if not self._loaded:
            self.load()
        return {
            "enabled": self._enabled,
            "my_titan_id": self._my_titan_id,
            "peer_endpoints": dict(self._peer_endpoints),
            "rate_limit_per_hour": self._rate_limit,
            "topic_cooldown_s": self._topic_cooldown_s,
            "min_snp_count": self._min_snp_count,
            "http_timeout_s": self._http_timeout,
            "retention_days": self._retention_days,
            "issued_in_last_hour": len(self._issued_ts),
            "topics_in_cooldown": len(self._last_query_ts),
            "recent_observations": len(self._recent_obs),
            "log_path": self._log_path,
            "policy": self._policy.snapshot(),
        }
