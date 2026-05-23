"""
SocialGraph Proxy — bridge to the supervised social_graph_worker subprocess.

Phase C v1.7.1 (D-SPEC-50) per rFP_titan_hcl_l2_separation_strategy §4.P.
Replaces the legacy `_proxies["social_graph"] = _proxies["mind"]` alias
rot (Maker 2026-05-12: *"no right to be in microkernel Phase C
architecture that must be lean and fast"*) — exposes the full SocialGraph
public surface as async + sync proxy methods (the MindProxy alias only
exposed a sync subset and lacked every `*_async` method, surfacing as
`AttributeError: 'MindProxy' object has no attribute
'record_interaction_async'` on every chat post-hook).

Classification per SPEC Preamble G18-G22:

  • get_stats / get_stats_async        → SHM read of social_graph_state.bin
                                          (G18 — state via SHM, never bus).
                                          Closes G22 violation (formerly
                                          `mind_worker get_social_stats`
                                          orphan-handler "full migration
                                          deferred").

  • get_donation_mood_boost(amount_sol) → pure compute (DONATION_TIERS
                                          lookup in social_graph.py); no
                                          IO, no bus.

  • All writes + parameterized reads    → bus.request_async work-RPC
                                          (G19, timeout ≤5s). Each entry
                                          documented in
                                          phase_c_rpc_exemptions.yaml ::
                                          work_rpc_sites under
                                          social_graph_proxy:. Sync
                                          wrappers via _work_rpc_sync
                                          (asyncio.run if no loop;
                                          bounded sync fallback if in
                                          loop) — mirrors MindProxy
                                          precedent.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import msgpack

from ..bus import DivineBus
from ..core.state_registry import (
    StateRegistryReader,
    ensure_shm_root,
    resolve_titan_id,
)
from ..guardian import Guardian
from ..logic.social_graph_state_specs import (
    SOCIAL_GRAPH_STATE_SLOT,
    SOCIAL_GRAPH_STATE_SPEC,
)

logger = logging.getLogger(__name__)


# DONATION_TIERS mirror titan_hcl/core/social_graph.py so the pure-
# compute path remains inline without any cross-process call. Single
# source of truth for the values lives in social_graph.py; this is a
# small dual-declaration with a one-line cross-reference. If social_graph
# tiers ever change, update both sites in the same commit.
_DONATION_TIERS = [
    (0.10, 0.10, 5.0),   # 0.10+ SOL → +0.10 mood, 5.0x memory weight
    (0.05, 0.05, 3.0),
    (0.01, 0.02, 2.0),
    (0.00, 0.01, 1.5),
]


class _DictProfile:
    """Lightweight UserProfile wrapper — attribute access over a dict
    from bus response. Mirrors MindProxy._DictProfile pattern."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", dict(data) if data else {})

    def __getattr__(self, name):
        if name == "_data":
            raise AttributeError
        return self._data.get(name)

    def __setattr__(self, name, value):
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    @property
    def net_sentiment(self) -> float:
        like = float(self._data.get("like_score", 0.0) or 0.0)
        dislike = float(self._data.get("dislike_score", 0.0) or 0.0)
        total = like + dislike
        if total == 0:
            return 0.0
        return (like - dislike) / total

    @property
    def is_donor(self) -> bool:
        return float(self._data.get("total_donated_sol", 0.0) or 0.0) > 0.0


class SocialGraphProxy:
    """Drop-in proxy for the social_graph_worker subprocess.

    Replaces the MindProxy alias rot per rFP_titan_hcl_l2_separation_strategy
    §4.P + D-SPEC-50. Exposes the full SocialGraph API:
      - Stats read: SHM (G18) via social_graph_state.bin slot.
      - Pure compute: get_donation_mood_boost (no IO).
      - Writes + parameterized reads: bus.request_async work-RPC (G19,
        timeout ≤5s).
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        # Reply queue retained for every work-RPC method (~25 actions).
        self._reply_queue = bus.subscribe(
            "social_graph_proxy", reply_only=True)
        self._started = False

        # SHM-direct reader for stats (G18 path).
        self._titan_id = resolve_titan_id()
        self._shm_root: Path = ensure_shm_root(self._titan_id)
        self._r_state = StateRegistryReader(
            SOCIAL_GRAPH_STATE_SPEC, self._shm_root)

        self._fallback_counts: dict[str, int] = {}

        logger.info(
            "[SocialGraphProxy] initialized SHM-direct reader — "
            "titan_id=%s shm_root=%s (slot=%s per Preamble G18 + "
            "rFP_titan_hcl_l2_separation_strategy §4.P)",
            self._titan_id, self._shm_root, SOCIAL_GRAPH_STATE_SLOT)

    # ── Lifecycle ────────────────────────────────────────────────────

    def _track_fallback(self, slot_name: str, reason: str) -> None:
        prev = self._fallback_counts.get(slot_name, 0)
        self._fallback_counts[slot_name] = prev + 1
        if prev == 0:
            logger.info(
                "[SocialGraphProxy] FIRST FALLBACK slot=%s reason=%s — "
                "consumer uses default until producer first publish",
                slot_name, reason)

    def _ensure_started(self) -> None:
        from ._start_safe import ensure_started_async_safe
        ready = ensure_started_async_safe(
            self._guardian, "social_graph", id(self),
            proxy_label="SocialGraphProxy",
        )
        if ready:
            self._started = True

    # ── Pure compute (no IO, no bus) ─────────────────────────────────

    def get_donation_mood_boost(self, amount_sol: float) -> tuple[float, float]:
        """Calculate mood boost and memory weight for a donation.

        Pure compute — does not require the worker. Inline tier lookup
        mirrors SocialGraph.get_donation_mood_boost (single source of
        truth for tiers stays in social_graph.py).
        """
        try:
            amount = float(amount_sol)
        except (TypeError, ValueError):
            amount = 0.0
        for threshold, mood_delta, weight in _DONATION_TIERS:
            if amount >= threshold:
                return mood_delta, weight
        return 0.01, 1.5

    # ── SHM stats read (G18 path) ────────────────────────────────────

    def _read_state(self) -> Optional[dict]:
        try:
            raw = self._r_state.read_variable()
        except Exception as e:
            self._track_fallback(
                SOCIAL_GRAPH_STATE_SLOT,
                f"read_raised:{type(e).__name__}")
            return None
        if raw is None:
            self._track_fallback(
                SOCIAL_GRAPH_STATE_SLOT, "shm_unavailable")
            return None
        try:
            decoded = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            self._track_fallback(
                SOCIAL_GRAPH_STATE_SLOT,
                f"decode_raised:{type(e).__name__}")
            return None
        return decoded if isinstance(decoded, dict) else None

    def get_stats(self) -> dict:
        """Return social graph stats — SHM-direct read of
        social_graph_state.bin (G18). Defaults to zeros on cold boot.
        """
        decoded = self._read_state()
        if decoded is None:
            return {
                "users": 0, "edges": 0, "donations": 0,
                "total_donated_sol": 0.0, "inspirations": 0,
            }
        return {
            "users": int(decoded.get("users", 0) or 0),
            "edges": int(decoded.get("edges", 0) or 0),
            "donations": int(decoded.get("donations", 0) or 0),
            "total_donated_sol": float(
                decoded.get("total_donated_sol", 0.0) or 0.0),
            "inspirations": int(decoded.get("inspirations", 0) or 0),
        }

    async def get_stats_async(self) -> dict:
        """Async sibling of get_stats — same SHM-direct read, just an
        async signature for callers in an async context."""
        return self.get_stats()

    # ── Work-RPC primitives (G19 ≤5s) ────────────────────────────────

    async def _work_rpc_async(self, action: str, extra: dict | None = None,
                              timeout: float = 5.0) -> dict:
        """Single async work-RPC primitive."""
        self._ensure_started()
        payload = {"action": action}
        if extra:
            payload.update(extra)
        try:
            reply = await self._bus.request_async(
                "social_graph_proxy", "social_graph", payload,
                timeout=timeout, reply_queue=self._reply_queue,
            )
            return reply.get("payload", {}) if reply else {}
        except Exception as e:
            logger.warning(
                "[SocialGraphProxy] %s async work-RPC raised "
                "(timeout=%.1fs): %s", action, timeout, e)
            return {}

    def _work_rpc_sync(self, action: str, extra: dict | None = None,
                       timeout: float = 5.0) -> dict:
        """Sync wrapper. Mirrors MindProxy._work_rpc_sync rationale:
        in-loop callers fall back to legacy sync bus.request with
        bounded timeout (allow-listed in phase_c_rpc_exemptions.yaml).
        """
        self._ensure_started()
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if not in_loop:
            try:
                return asyncio.run(
                    self._work_rpc_async(action, extra, timeout=timeout))
            except Exception as e:
                logger.warning(
                    "[SocialGraphProxy] %s asyncio.run failed: %s — "
                    "falling back to bounded sync bus.request",
                    action, e)

        payload = {"action": action}
        if extra:
            payload.update(extra)
        reply = self._bus.request(
            "social_graph_proxy", "social_graph", payload,
            timeout=timeout, reply_queue=self._reply_queue,
        )
        return reply.get("payload", {}) if reply else {}

    def _publish_fire_and_forget(self, action: str,
                                 extra: dict | None = None) -> None:
        """One-way bus.publish — for write actions where the caller does
        not need a result (e.g., record_interaction in legacy MindProxy).
        Note: most write methods below still go via _work_rpc_*
        because the worker emits SOCIAL_*_RECORDED bus events after the
        write — the proxy doesn't need the bus event return, but the
        round-trip lets us surface DB errors to the caller.
        """
        self._ensure_started()
        import time as _t
        payload = {"action": action}
        if extra:
            payload.update(extra)
        self._bus.publish({
            "type": "QUERY",
            "src": "social_graph_proxy",
            "dst": "social_graph",
            "ts": _t.time(),
            "rid": None,
            "payload": payload,
        })

    # ── Writes / upserts (async + sync) ──────────────────────────────

    async def record_interaction_async(self, user_id: str,
                                       quality: float = 0.5) -> None:
        """Record an interaction quality update (write/upsert)."""
        await self._work_rpc_async("record_interaction", {
            "user_id": user_id, "quality": quality,
        })

    def record_interaction(self, user_id: str, quality: float = 0.5,
                           **_unused) -> None:
        """Sync sibling. Legacy fire-and-forget shape preserved for
        backward compat — MindProxy.record_interaction was fire-and-
        forget; we now round-trip (still bounded by 5s timeout)."""
        self._work_rpc_sync("record_interaction", {
            "user_id": user_id, "quality": quality,
        })

    async def get_or_create_user_async(
        self, user_id: str, platform: str = "unknown",
        display_name: str = "",
    ) -> _DictProfile:
        result = await self._work_rpc_async("get_or_create_user", {
            "user_id": user_id, "platform": platform,
            "display_name": display_name,
        })
        return _DictProfile(result.get("profile", {}) if result else {})

    def get_or_create_user(self, user_id: str,
                           platform: str = "unknown",
                           display_name: str = "") -> _DictProfile:
        result = self._work_rpc_sync("get_or_create_user", {
            "user_id": user_id, "platform": platform,
            "display_name": display_name,
        })
        return _DictProfile(result.get("profile", {}) if result else {})

    async def _save_profile_async(self, profile) -> None:
        data = profile._data if hasattr(profile, "_data") else \
            (profile.to_dict() if hasattr(profile, "to_dict") else dict(profile))
        await self._work_rpc_async("save_profile", {"profile": data})

    def _save_profile(self, profile) -> None:
        data = profile._data if hasattr(profile, "_data") else \
            (profile.to_dict() if hasattr(profile, "to_dict") else dict(profile))
        self._work_rpc_sync("save_profile", {"profile": data})

    async def should_engage_async(self, user_id: str) -> str:
        result = await self._work_rpc_async("should_engage", {
            "user_id": user_id,
        })
        if isinstance(result, dict):
            level = result.get("level")
            if isinstance(level, str):
                return level
        return "minimal"

    def should_engage(self, user_id: str) -> str:
        result = self._work_rpc_sync("should_engage", {"user_id": user_id})
        if isinstance(result, dict):
            level = result.get("level")
            if isinstance(level, str):
                return level
        return "minimal"

    async def record_edge_async(self, user_a: str, user_b: str) -> None:
        await self._work_rpc_async("record_edge", {
            "user_a": user_a, "user_b": user_b,
        })

    def record_edge(self, user_a: str, user_b: str) -> None:
        self._work_rpc_sync("record_edge", {
            "user_a": user_a, "user_b": user_b,
        })

    async def record_donation_async(
        self, tx_signature: str, sender_address: str, amount_sol: float,
        memo: str = "",
    ) -> Optional[_DictProfile]:
        result = await self._work_rpc_async("record_donation", {
            "tx_signature": tx_signature,
            "sender_address": sender_address,
            "amount_sol": amount_sol, "memo": memo,
        })
        matched = result.get("matched_user") if result else None
        return _DictProfile(matched) if matched else None

    def record_donation(
        self, tx_signature: str, sender_address: str, amount_sol: float,
        memo: str = "",
    ) -> Optional[_DictProfile]:
        result = self._work_rpc_sync("record_donation", {
            "tx_signature": tx_signature,
            "sender_address": sender_address,
            "amount_sol": amount_sol, "memo": memo,
        })
        matched = result.get("matched_user") if result else None
        return _DictProfile(matched) if matched else None

    async def record_inspiration_async(
        self, tx_signature: str, sender_address: str, message: str,
        amount_sol: float = 0.0,
    ) -> Optional[_DictProfile]:
        result = await self._work_rpc_async("record_inspiration", {
            "tx_signature": tx_signature,
            "sender_address": sender_address,
            "message": message, "amount_sol": amount_sol,
        })
        matched = result.get("matched_user") if result else None
        return _DictProfile(matched) if matched else None

    def record_inspiration(
        self, tx_signature: str, sender_address: str, message: str,
        amount_sol: float = 0.0,
    ) -> Optional[_DictProfile]:
        result = self._work_rpc_sync("record_inspiration", {
            "tx_signature": tx_signature,
            "sender_address": sender_address,
            "message": message, "amount_sol": amount_sol,
        })
        matched = result.get("matched_user") if result else None
        return _DictProfile(matched) if matched else None

    def link_sol_address(self, user_id: str, sol_address: str) -> None:
        self._work_rpc_sync("link_sol_address", {
            "user_id": user_id, "sol_address": sol_address,
        })

    def mark_inspiration_processed(self, tx_signature: str,
                                   outcome: str) -> None:
        self._work_rpc_sync("mark_inspiration_processed", {
            "tx_signature": tx_signature, "outcome": outcome,
        })

    def set_titan_preference(self, titan_id: str, user_name: str,
                             affinity_delta: float = 0.1,
                             tags: str = "",
                             discovered_via: str = "") -> None:
        self._work_rpc_sync("set_titan_preference", {
            "titan_id": titan_id, "user_name": user_name,
            "affinity_delta": affinity_delta, "tags": tags,
            "discovered_via": discovered_via,
        })

    def sync_community(self, users: list, relationship: str = "follower") -> None:
        self._work_rpc_sync("sync_community", {
            "users": users, "relationship": relationship,
        })

    def mark_checked(self, titan_id: str, user_name: str) -> None:
        self._work_rpc_sync("mark_checked", {
            "titan_id": titan_id, "user_name": user_name,
        })

    def update_last_tweet(self, user_name: str, tweet_text: str) -> None:
        self._work_rpc_sync("update_last_tweet", {
            "user_name": user_name, "tweet_text": tweet_text,
        })

    async def ledger_record_async(self, tweet_id: str, user_name: str,
                                  action: str,
                                  mention_text: str = "") -> None:
        await self._work_rpc_async("ledger_record", {
            "tweet_id": tweet_id, "user_name": user_name,
            "action_kind": action, "mention_text": mention_text,
        })

    def ledger_record(self, tweet_id: str, user_name: str, action: str,
                      mention_text: str = "") -> None:
        self._work_rpc_sync("ledger_record", {
            "tweet_id": tweet_id, "user_name": user_name,
            "action_kind": action, "mention_text": mention_text,
        })

    async def ledger_cleanup_async(self,
                                   max_age_seconds: float = 172800) -> int:
        result = await self._work_rpc_async("ledger_cleanup", {
            "max_age_seconds": max_age_seconds,
        })
        return int(result.get("removed", 0) or 0) if result else 0

    def ledger_cleanup(self, max_age_seconds: float = 172800) -> int:
        result = self._work_rpc_sync("ledger_cleanup", {
            "max_age_seconds": max_age_seconds,
        })
        return int(result.get("removed", 0) or 0) if result else 0

    # ── Parameterized reads (async + sync) ───────────────────────────

    async def get_top_users_async(self, limit: int = 10) -> list[_DictProfile]:
        result = await self._work_rpc_async("get_top_users", {"limit": limit})
        users = result.get("users", []) if result else []
        return [_DictProfile(u) for u in (users or [])]

    def get_top_users(self, limit: int = 10) -> list[_DictProfile]:
        result = self._work_rpc_sync("get_top_users", {"limit": limit})
        users = result.get("users", []) if result else []
        return [_DictProfile(u) for u in (users or [])]

    def get_user_connections(self, user_id: str) -> list:
        result = self._work_rpc_sync("get_user_connections", {
            "user_id": user_id,
        })
        return result.get("connections", []) if result else []

    def get_community(self, relationship: Optional[str] = None) -> list:
        result = self._work_rpc_sync("get_community", {
            "relationship": relationship,
        })
        return result.get("community", []) if result else []

    def get_titan_favorites(self, titan_id: str, limit: int = 10) -> list:
        result = self._work_rpc_sync("get_titan_favorites", {
            "titan_id": titan_id, "limit": limit,
        })
        return result.get("favorites", []) if result else []

    def get_accounts_to_check(self, titan_id: str, limit: int = 3) -> list:
        result = self._work_rpc_sync("get_accounts_to_check", {
            "titan_id": titan_id, "limit": limit,
        })
        return result.get("accounts", []) if result else []

    def get_pending_inspirations(self, limit: int = 10) -> list:
        result = self._work_rpc_sync("get_pending_inspirations", {
            "limit": limit,
        })
        return result.get("inspirations", []) if result else []

    def find_user_by_sol_address(self, sol_address: str) -> Optional[_DictProfile]:
        result = self._work_rpc_sync("find_user_by_sol_address", {
            "sol_address": sol_address,
        })
        profile = result.get("profile") if result else None
        return _DictProfile(profile) if profile else None

    async def ledger_has_tweet_async(
        self, tweet_id: str, action: Optional[str] = None,
    ) -> bool:
        result = await self._work_rpc_async("ledger_has_tweet", {
            "tweet_id": tweet_id, "action_kind": action,
        })
        return bool(result.get("has", False)) if result else False

    def ledger_has_tweet(self, tweet_id: str,
                         action: Optional[str] = None) -> bool:
        result = self._work_rpc_sync("ledger_has_tweet", {
            "tweet_id": tweet_id, "action_kind": action,
        })
        return bool(result.get("has", False)) if result else False

    async def ledger_user_reply_count_async(
        self, user_name: str, window_seconds: float,
    ) -> int:
        result = await self._work_rpc_async("ledger_user_reply_count", {
            "user_name": user_name, "window_seconds": window_seconds,
        })
        return int(result.get("count", 0) or 0) if result else 0

    def ledger_user_reply_count(self, user_name: str,
                                window_seconds: float) -> int:
        result = self._work_rpc_sync("ledger_user_reply_count", {
            "user_name": user_name, "window_seconds": window_seconds,
        })
        return int(result.get("count", 0) or 0) if result else 0

    async def ledger_last_reply_to_user_async(self, user_name: str) -> float:
        result = await self._work_rpc_async("ledger_last_reply_to_user", {
            "user_name": user_name,
        })
        return float(result.get("ts", 0.0) or 0.0) if result else 0.0

    def ledger_last_reply_to_user(self, user_name: str) -> float:
        result = self._work_rpc_sync("ledger_last_reply_to_user", {
            "user_name": user_name,
        })
        return float(result.get("ts", 0.0) or 0.0) if result else 0.0

    async def ledger_total_today_async(
        self, action: Optional[str] = None,
    ) -> int:
        result = await self._work_rpc_async("ledger_total_today", {
            "action_kind": action,
        })
        return int(result.get("count", 0) or 0) if result else 0

    def ledger_total_today(self, action: Optional[str] = None) -> int:
        result = self._work_rpc_sync("ledger_total_today", {
            "action_kind": action,
        })
        return int(result.get("count", 0) or 0) if result else 0
