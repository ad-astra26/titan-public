"""
api/dashboard.py
Public Observatory endpoints — the Titan's window to the world.

All endpoints are read-only and require no authentication.
Sensitive data (raw prompts, agent responses) is never exposed.
"""
import asyncio
import hashlib
import io
import logging
import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse

# sqlite_async: thin async wrapper around sqlite3 (core/sqlite_async.py).
# Used to migrate the 10 E.2.3-wrapped sqlite sites from inline
# `def _query(): conn = sqlite3.connect(...); await asyncio.to_thread(_query)`
# to `await sqlite_async.query(...)` / `sqlite_async.with_connection(...)`.
from titan_plugin.core import sqlite_async

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Observatory"])

# ── META-CGN Producers #11 + #12 EdgeDetector (dashboard.py module-level) ──
# Tracks per-(session_id × threshold) crossings so each persona session only
# emits session_low_qual / session_high_qual at most once each.
#
# 2026-04-15: added save-on-emit + lazy-load-on-first-use persistence.
# Save on each fire is cheap (file is small, ~20 session_ids × 2 thresholds
# = ~40 keys max lifetime). Load on first use picks up state across
# titan_main restarts. Mirrors the pattern used in spirit_worker for
# P1/P2/P13/P14/P15 detectors but uses a separate file because dashboard
# runs in the main FastAPI process.
_PERSONA_SESSION_DETECTOR = None
_PERSONA_SESSION_STATE_PATH = os.environ.get(
    "TITAN_PERSONA_SESSION_STATE", "./data/persona_session_edge_state.json"
)


def _get_persona_session_detector():
    """Lazy-init + return the dashboard-module EdgeDetector for P11/P12.

    On first use: create detector, attempt to load persisted state from
    `data/persona_session_edge_state.json`. Fail-open — missing/corrupt
    file is harmless (producer just re-fires once per session_id).
    """
    global _PERSONA_SESSION_DETECTOR
    if _PERSONA_SESSION_DETECTOR is None:
        import json as _json
        from titan_plugin.logic.meta_cgn import EdgeDetector
        _PERSONA_SESSION_DETECTOR = EdgeDetector()
        try:
            with open(_PERSONA_SESSION_STATE_PATH) as _f:
                _state = _json.load(_f)
            if _state.get("schema_version") == 1:
                _PERSONA_SESSION_DETECTOR.load_dict(_state.get("detector", {}))
                _crossed_count = sum(
                    1 for v in _state.get("detector", {}).get("crossed", {}).values() if v)
                logger.info(
                    "[META-CGN] Persona session EdgeDetector restored "
                    "(%d (session_id × threshold) crossings known)", _crossed_count)
        except FileNotFoundError:
            logger.debug("[META-CGN] Persona session state file not found — fresh detector")
        except Exception as _e:
            logger.warning(
                "[META-CGN] Persona session state load failed: %s — starting fresh", _e)
    return _PERSONA_SESSION_DETECTOR


def _save_persona_session_detector():
    """Atomically persist persona session EdgeDetector state.

    Called after every P11/P12 emission. Small file (<5 KB even after years
    of persona sessions). Atomic via tmpfile + os.replace.
    WARN on failure (silent failure would hide a persistence gap).
    """
    import json as _json
    import tempfile
    det = _PERSONA_SESSION_DETECTOR
    if det is None:
        return
    try:
        _dir = os.path.dirname(_PERSONA_SESSION_STATE_PATH) or "."
        if not os.path.isdir(_dir):
            os.makedirs(_dir, exist_ok=True)
        payload = {
            "schema_version": 1,
            "saved_at": time.time(),
            "detector": det.to_dict(),
        }
        fd, tmp = tempfile.mkstemp(dir=_dir, prefix="persona_session_edge.", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            _json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, _PERSONA_SESSION_STATE_PATH)
    except Exception as _e:
        logger.warning(
            "[META-CGN] Persona session state save failed: %s "
            "(emission still delivered)", _e)


# Internal-injection prompt prefix produced by core/memory.py:inject_memory().
# Every memory created via direct injection (self_profile, maker, dream-bridge,
# meta-wisdom, etc.) is wrapped as f"[{source.upper()}_INJECTION] {text}".
# These rows live in Titan's persistent store with weight ~10 — intentional
# self-knowledge for cognition, but they are NOT user-facing memories and
# should be hidden from the public Observatory Memory Node view.
_INTERNAL_INJECTION_RE = re.compile(r"^\[[A-Z_]+_INJECTION\]")


def _is_internal_injection(prompt: str) -> bool:
    """True if the memory prompt is a system-internal injection (not a real conversation)."""
    return bool(prompt) and _INTERNAL_INJECTION_RE.match(prompt) is not None


def _content_hash(prompt: str, response: str = "") -> str:
    """Stable 16-char SHA-256 prefix for a memory's content (display only)."""
    payload = (prompt or "") + "\x1f" + (response or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_plugin(request: Request):
    return request.app.state.titan_plugin


def _ok(data: dict) -> JSONResponse:
    return JSONResponse({"status": "ok", "data": data})


# ---------------------------------------------------------------------------
# Coordinator Cache — collapses N simultaneous polls into 1 proxy call.
# Prevents spirit_worker proxy overload (max 2.6 queries/sec single-threaded).
# All lightweight /v4/ endpoints extract from this shared cached response.
# ---------------------------------------------------------------------------
_coordinator_cache: dict = {"data": None, "ts": 0.0}
_COORDINATOR_CACHE_TTL = 10.0  # seconds — rFP v3 § 15; Schumann 7.83 Hz = 127ms cycles, 10s captures ~78 cycles of variation while halving coordinator-proxy load
_coordinator_refreshing = False  # prevents multiple concurrent refreshes
_coordinator_warmer_started = False  # gate the background warmer thread
# Trinity tensor caches — refreshed by warmer so /v3/trinity is always hot.
_trinity_tensor_cache: dict = {
    "body_tensor": [0.5] * 5,
    "mind_tensor": [0.5] * 5,
    "spirit_data": {},
    "ts": 0.0,
}


def _start_coordinator_warmer(plugin) -> None:
    """Background thread that keeps coordinator cache always hot.

    2026-04-14 perf fix: cold cache (TTL expired) caused first request
    to wait ~3-5s for the proxy IPC round-trip. Warmer refreshes the
    cache every (TTL - 2)s in a daemon thread, so endpoints always see
    fresh data instantly. No-op if already started or no spirit_proxy.
    """
    global _coordinator_warmer_started
    if _coordinator_warmer_started:
        return
    spirit_proxy = plugin._proxies.get("spirit") if hasattr(plugin, "_proxies") else None
    if not spirit_proxy:
        return
    _coordinator_warmer_started = True

    body_proxy = plugin._proxies.get("body") if hasattr(plugin, "_proxies") else None
    mind_proxy = plugin._proxies.get("mind") if hasattr(plugin, "_proxies") else None

    def _warmer_loop():
        import time as _t
        interval = max(1.0, _COORDINATOR_CACHE_TTL - 2.0)
        while True:
            # Refresh coordinator (shared by many endpoints)
            try:
                result = spirit_proxy.get_coordinator()
                if result and not result.get("error"):
                    _coordinator_cache["data"] = result
                    _coordinator_cache["ts"] = _t.time()
            except Exception:
                pass
            # Refresh trinity tensors (used by /v3/trinity). Each proxy has
            # its own reply queue so these don't contend. Sync calls are OK
            # here because we're in a dedicated daemon thread, not the
            # asyncio event loop.
            try:
                if body_proxy:
                    bt = body_proxy.get_body_tensor()
                    if bt:
                        _trinity_tensor_cache["body_tensor"] = bt
                if mind_proxy:
                    mt = mind_proxy.get_mind_tensor()
                    if mt:
                        _trinity_tensor_cache["mind_tensor"] = mt
                sd = spirit_proxy.get_trinity()
                if sd and not sd.get("error"):
                    _trinity_tensor_cache["spirit_data"] = sd
                _trinity_tensor_cache["ts"] = _t.time()
            except Exception:
                pass
            _t.sleep(interval)

    import threading
    threading.Thread(target=_warmer_loop, daemon=True,
                     name="coordinator-cache-warmer").start()
    logger.info("[Dashboard] Coordinator cache warmer started "
                "(refresh every %.1fs)", _COORDINATOR_CACHE_TTL - 2.0)


def _get_cached_coordinator(plugin) -> dict:
    """Get coordinator state with 5s TTL cache.

    IMPORTANT: The proxy call is synchronous and blocks the asyncio event loop
    for ~300-500ms. Use the async version (_get_cached_coordinator_async) from
    async endpoints to avoid blocking.
    """
    now = time.time()
    if (now - _coordinator_cache["ts"] < _COORDINATOR_CACHE_TTL
            and _coordinator_cache["data"]):
        return _coordinator_cache["data"]
    spirit_proxy = plugin._proxies.get("spirit")
    if not spirit_proxy:
        return _coordinator_cache.get("data") or {}
    try:
        result = spirit_proxy.get_coordinator()
        _coordinator_cache["data"] = result
        _coordinator_cache["ts"] = now
        return result
    except Exception:
        return _coordinator_cache.get("data") or {}


async def _get_cached_coordinator_async(plugin) -> dict:
    """Non-blocking coordinator cache: runs proxy call in thread pool."""
    import asyncio
    global _coordinator_refreshing
    # Lazy-start the background warmer on first use (idempotent).
    _start_coordinator_warmer(plugin)
    now = time.time()
    if (now - _coordinator_cache["ts"] < _COORDINATOR_CACHE_TTL
            and _coordinator_cache["data"]):
        return _coordinator_cache["data"]
    # If another coroutine is already refreshing, return stale data
    if _coordinator_refreshing:
        return _coordinator_cache.get("data") or {}
    spirit_proxy = plugin._proxies.get("spirit")
    if not spirit_proxy:
        return _coordinator_cache.get("data") or {}
    try:
        _coordinator_refreshing = True
        result = await asyncio.to_thread(spirit_proxy.get_coordinator)
        _coordinator_cache["data"] = result
        _coordinator_cache["ts"] = time.time()
        return result
    except Exception:
        return _coordinator_cache.get("data") or {}
    finally:
        _coordinator_refreshing = False


def _get_consciousness_summary(plugin) -> dict | None:
    """Extract latest consciousness epoch for status response."""
    try:
        consciousness = getattr(plugin, 'consciousness', None)
        if not consciousness:
            return None
        recent = consciousness.db.get_recent_epochs(1)
        if not recent:
            return None
        latest = recent[-1]
        import json as _json
        sv = latest.state_vector
        if isinstance(sv, str):
            sv = _json.loads(sv)
        from titan_plugin.logic.consciousness import STATE_DIMS
        dims = {STATE_DIMS[i]: round(sv[i], 4) for i in range(min(len(sv), len(STATE_DIMS)))}
        return {
            "epoch": latest.epoch_id,
            "curvature": round(latest.curvature, 4),
            "density": round(latest.density, 4),
            "journey_point": {"x": round(latest.journey_point[0], 4),
                              "y": round(latest.journey_point[1], 4),
                              "z": round(latest.journey_point[2], 4)},
            "state": dims,
            "distillation": latest.distillation,
            "anchored": bool(latest.anchored_tx),
        }
    except Exception:
        return None


def _error(msg: str, code: int = 500) -> JSONResponse:
    return JSONResponse({"status": "error", "detail": msg}, status_code=code)


async def _fetch_vault_info(plugin) -> dict | None:
    """Fetch and decode vault state from on-chain PDA. Returns VaultInfo dict or None."""
    vault_program_id = plugin._full_config.get("network", {}).get("vault_program_id", "")
    if not vault_program_id:
        return None
    try:
        from titan_plugin.utils.solana_client import (
            is_available as solana_ok, derive_vault_pda, decode_vault_state,
        )
        if not solana_ok() or plugin.network.pubkey is None:
            return None
        pda_result = derive_vault_pda(plugin.network.pubkey, vault_program_id)
        if not pda_result:
            return None
        vault_pda, _ = pda_result
        from solana.rpc.async_api import AsyncClient
        rpc_url = plugin.network.rpc_urls[0] if plugin.network.rpc_urls else "https://api.mainnet-beta.solana.com"
        async with AsyncClient(rpc_url) as client:
            resp = await client.get_account_info(vault_pda)
        if resp.value is None:
            return None
        raw_data = resp.value.data
        if not isinstance(raw_data, (bytes, bytearray)):
            return None
        vd = decode_vault_state(raw_data)
        if not vd:
            return None
        return {
            "program_id": vault_program_id,
            "pda": str(vault_pda),
            "commit_count": vd["commit_count"],
            "last_commit": vd["last_commit_ts"],
            "latest_state_root": vd["latest_root"],
            "sovereignty_pct": vd["sovereignty_percent"],
            "compressed_memories": vd.get("compressed_memories", 0),
            "epoch_snapshots": vd.get("epoch_snapshots", 0),
        }
    except Exception as exc:
        logger.debug("[Dashboard] vault lookup failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# GET /status — Bio-State Overview
# ---------------------------------------------------------------------------
@router.get("/status")
async def get_status(request: Request):
    """Real-time Bio-State: mood, energy, node count, sovereignty, uptime."""
    plugin = _get_plugin(request)
    is_v3 = hasattr(plugin, "get_v3_status")

    try:
        # Mood — works via MindProxy in V3, direct in V2
        # Fix 2026-04-15 (rFP v3 § 15 followup): mood_engine in V3 is a proxy
        # with sync IPC methods. Previously called directly from async route,
        # blocking the event loop under concurrent /status load (5s cluster
        # observed in tests/test_dashboard_concurrency.py). Wrap in
        # asyncio.to_thread — same pattern as /status/mood:403.
        mood_engine = plugin.mood_engine
        if mood_engine and hasattr(mood_engine, "get_mood_label"):
            if callable(getattr(mood_engine, "get_mood_valence", None)):
                mood_label, mood_valence = await asyncio.to_thread(
                    lambda: (mood_engine.get_mood_label(),
                             mood_engine.get_mood_valence()))
                mood_score = mood_valence * 100  # V3 proxy returns 0-1
            else:
                mood_label = await asyncio.to_thread(mood_engine.get_mood_label)
                mood_score = getattr(mood_engine, "previous_mood", 50.0)
        else:
            mood_label = "Unknown"
            mood_score = 50.0

        # Energy — V2 only (metabolism not a V3 module yet)
        metabolism = plugin.metabolism
        if metabolism and hasattr(metabolism, "get_current_state"):
            energy_state = await metabolism.get_current_state()
            life_force = round(max(0, getattr(metabolism, "_last_balance_pct", 1.0)), 2)
        else:
            energy_state = "UNKNOWN"
            life_force = 1.0

        # Memory node count — MemoryProxy.get_persistent_count is sync IPC;
        # wrap to avoid event-loop serialization under concurrent /status load.
        memory = plugin.memory
        if memory and hasattr(memory, "get_persistent_count"):
            node_count = await asyncio.to_thread(memory.get_persistent_count)
        else:
            node_count = 0

        # SOL balance
        sol_balance = 0.0
        if plugin.network and hasattr(plugin.network, "get_balance"):
            sol_balance = await plugin.network.get_balance()

        # Mempool — V2 only
        mempool_size = 0
        if memory and hasattr(memory, "fetch_mempool"):
            mempool = await memory.fetch_mempool()
            mempool_size = len(mempool)

        # Vault — works in both V2 and V3 (requires network client)
        vault_info = None
        if hasattr(plugin, "network") and plugin.network:
            vault_info = await _fetch_vault_info(plugin)

        uptime = time.time() - plugin._start_time if hasattr(plugin, "_start_time") else 0

        # Sovereignty
        gatekeeper = plugin.gatekeeper
        sovereignty = 0.0
        if gatekeeper:
            sovereignty = getattr(gatekeeper, "sovereignty_score", 0)

        data = {
            "sovereign_name": "Titan",
            "version": "6.0" if is_v3 else "2.1",
            "mood": {
                "label": mood_label,
                "score": round(mood_score, 4),
            },
            "energy_state": energy_state,
            "sol_balance": round(sol_balance, 6),
            "life_force": life_force,
            "sovereignty_pct": round(sovereignty, 2),
            "uptime_seconds": round(uptime),
            "persistent_nodes": node_count,
            "memory_count": node_count,
            "mempool_size": mempool_size,
            "current_directive": getattr(plugin, "_current_directive", ""),
            "soul_gen": plugin.soul.current_gen if plugin.soul else 0,
            "is_meditating": plugin._is_meditating,
            "ws_subscribers": plugin.event_bus.subscriber_count if hasattr(plugin, "event_bus") else 0,
            "vault": vault_info,
            "epoch": None,
            "consciousness": _get_consciousness_summary(plugin),
        }

        # Lifetime accumulated metrics (persisted across restarts)
        # Uses the cached coordinator snapshot (1.5s TTL, same as /v4/inner-trinity)
        try:
            # rFP v3 § 15: async variant avoids event-loop block on cache miss
            _coord = await _get_cached_coordinator_async(plugin)
            if _coord and isinstance(_coord, dict) and not _coord.get("error"):
                _pi = _coord.get("pi_heartbeat", {})
                _nm = _coord.get("neuromodulators", {})
                _msl = _coord.get("msl", {})
                _mr = _coord.get("meta_reasoning", {})
                _dr = _coord.get("dreaming", {})
                _nns = _coord.get("neural_nervous_system", {})
                data["lifetime"] = {
                    "total_epochs": _pi.get("total_epochs_observed", 0),
                    "developmental_age": _pi.get("developmental_age", 0),
                    "heartbeat_ratio": round(_pi.get("heartbeat_ratio", 0), 3),
                    "dream_cycles": _dr.get("cycle_count", 0),
                    "neural_train_steps": _nns.get("total_train_steps", 0),
                    "neural_maturity": round(_nns.get("maturity", 0), 3),
                    "meta_chains": _mr.get("total_chains", 0),
                    "eurekas": _mr.get("total_eurekas", 0),
                    "i_confidence": round(_msl.get("i_confidence", 0), 3),
                    "i_depth": round(_msl.get("i_depth", 0), 4),
                    "vocabulary": 0,
                    "emotion": _nm.get("current_emotion", "unknown"),
                }
        except Exception as _lt_err:
            logger.warning("[Dashboard] Lifetime metrics failed: %s", _lt_err)

        # V3: include Trinity summary
        if is_v3:
            data["v3"] = plugin.get_v3_status()

        return _ok(data)
    except Exception as e:
        logger.error("[Dashboard] /status error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/mood — Detailed Mood Breakdown
# ---------------------------------------------------------------------------
@router.get("/status/mood")
async def get_mood_detail(request: Request):
    """Detailed mood: internal score, addon modifier, Social Gravity, history."""
    import asyncio
    plugin = _get_plugin(request)
    try:
        # V3: use MindProxy via bus to MoodEngine in Mind worker.
        # rFP v3 § 15: wrap serial sync proxy calls in to_thread to avoid
        # blocking event loop for 5s + 5s = 10s on cold path.
        mood_label, mood_valence = await asyncio.to_thread(
            lambda: (plugin.mood_engine.get_mood_label(),
                     plugin.mood_engine.get_mood_valence())
        )

        # Neuromod emotion provides richer mood context if available
        neuromod_emotion = "unknown"
        neuromod_confidence = 0.0
        try:
            coord_data = await _get_cached_coordinator_async(plugin)
            nm = coord_data.get("neuromodulators", {})
            neuromod_emotion = nm.get("current_emotion", nm.get("emotion", "unknown"))
            neuromod_confidence = nm.get("emotion_confidence", 0.0)
        except Exception:
            pass

        return _ok({
            "mood_label": mood_label,
            "current_score": round(mood_valence, 4),
            "prior_score": round(mood_valence, 4),
            "mood_delta": 0.0,
            "neuromod_emotion": neuromod_emotion,
            "neuromod_confidence": round(neuromod_confidence, 4),
            "social_gravity": {
                "mentions_received": 0,
                "daily_replies": 0,
                "reply_likes": 0,
                "daily_likes": 0,
            },
        })
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/memory — Memory Topology
# ---------------------------------------------------------------------------
@router.get("/status/memory")
async def get_memory_status(request: Request):
    """Memory overview: counts, top memories (redacted), decay stats."""
    plugin = _get_plugin(request)
    try:
        # Use proxy methods (route through Divine Bus to memory worker)
        mem_status = plugin.memory.get_memory_status()
        persistent_count = mem_status.get("persistent_count", 0)
        cognee_ready = mem_status.get("cognee_ready", False)

        mempool = await plugin.memory.fetch_mempool()
        # Fetch the FULL persistent set so internal-injection memories
        # (e.g. [SELF_PROFILE_INJECTION], weight=10.31) can be filtered out
        # while real conversation memories are guaranteed to surface.
        # See core/memory.py:755 — inject_memory() wraps user_prompt with
        # f"[{source.upper()}_INJECTION] ..." for every internal injection.
        # These are intentional self-knowledge rows, not user-facing memories,
        # so they belong in Titan's cognition but not the public Observatory.
        # Each dream cycle adds another self-profile row, so over days the
        # injection rows accumulate to thousands and dominate any small top-N
        # window. Fetching the whole set + filtering is still cheap (~3.5k
        # dicts, <1 MB) and runs behind the 5s coordinator cache anyway.
        fetch_n = max(persistent_count + 100, 1000)
        raw_top = plugin.memory.get_top_memories(n=fetch_n)
        top_mems = [
            m for m in raw_top
            if not _is_internal_injection(m.get("user_prompt", ""))
        ][:200]

        # Redacted summaries — never expose raw prompts
        top_summaries = []
        for m in top_mems:
            prompt = m.get("user_prompt", "")
            top_summaries.append({
                "hint": prompt[:40] + ("..." if len(prompt) > 40 else ""),
                "weight": round(m.get("effective_weight", 1.0), 3),
                "intensity": m.get("emotional_intensity", 0),
                "reinforcements": m.get("reinforcement_count", 0),
            })

        # Build unified nodes list for the Neural page (persistent + mempool)
        nodes = []
        for m in top_mems:
            prompt = m.get("user_prompt", "")
            nodes.append({
                "id": str(m.get("id", "")),
                "text": prompt[:60] + ("..." if len(prompt) > 60 else ""),
                "hash": _content_hash(prompt, m.get("agent_response", "")),
                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(m.get("created_at", 0))
                ) if m.get("created_at") else "",
                "effective_weight": round(m.get("effective_weight", 1.0), 3),
                "reinforcements": m.get("reinforcement_count", 0),
                "tier": "persistent",
            })

        # Add mempool nodes (decay already applied by worker)
        for m in mempool:
            prompt = m.get("user_prompt", "")
            if _is_internal_injection(prompt):
                continue
            nodes.append({
                "id": str(m.get("id", "")),
                "text": prompt[:60] + ("..." if len(prompt) > 60 else ""),
                "hash": _content_hash(prompt, m.get("agent_response", "")),
                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(m.get("created_at", 0))
                ) if m.get("created_at") else "",
                "effective_weight": round(m.get("mempool_weight", 1.0), 3),
                "reinforcements": m.get("mempool_reinforcements", 0),
                "tier": "mempool",
            })

        return _ok({
            "persistent_count": persistent_count,
            "mempool_size": len(mempool),
            "top_memories": top_summaries,
            "nodes": nodes,
            "cognee_ready": cognee_ready,  # kept for API compat
            "memory_backend_ready": cognee_ready,
        })
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/memory/topology — Cognitive Heatmap
# ---------------------------------------------------------------------------
# Topic clusters for keyword classification
_TOPIC_KEYWORDS = {
    "Solana Architecture": {"solana", "sol", "blockchain", "transaction", "wallet", "rpc", "zk", "nft", "anchor", "keypair"},
    "Social Pulse": {"tweet", "social", "mention", "reply", "like", "engagement", "x", "twitter", "post"},
    "Maker Directives": {"directive", "maker", "divine", "inspiration", "soul", "evolve", "prime", "sovereignty"},
    "Research & Knowledge": {"research", "learn", "knowledge", "search", "document", "analysis", "discovery", "sage"},
    "Memory & Identity": {"memory", "remember", "cognee", "persist", "forget", "identity", "growth", "neuron"},
    "Metabolic & Energy": {"balance", "energy", "starvation", "metabolism", "health", "reserve", "governance"},
}


@router.get("/status/memory/topology")
async def get_memory_topology(request: Request):
    """Cognitive Heatmap: cluster distribution of what the Titan is focused on."""
    plugin = _get_plugin(request)
    try:
        # Delegate classification to memory worker (runs in separate process)
        result = plugin.memory.get_topology(_TOPIC_KEYWORDS)
        return _ok(result)
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/memory/knowledge-graph — Kuzu Entity Graph for 3D Visualization
# ---------------------------------------------------------------------------
@router.get("/status/memory/knowledge-graph")
async def get_knowledge_graph(
    request: Request,
    limit: int = Query(200, ge=10, le=500),
):
    """Return Kuzu knowledge graph entities and relationships for 3D visualization.

    Routes through memory worker proxy (Kuzu connection lives in worker process).
    Returns Trinity-typed entities (Person, Topic, Body, Mind, Spirit, Media)
    with relationship edges. Designed for force-directed graph rendering.
    """
    plugin = _get_plugin(request)
    try:
        result = plugin.memory.get_knowledge_graph(limit=limit)
        return _ok(result)
    except Exception as e:
        logger.warning("[Dashboard] /status/memory/knowledge-graph: %s", e)
        return _ok({"nodes": [], "edges": [], "stats": {}, "available": False, "error": str(e)})


# ---------------------------------------------------------------------------
# GET /status/social — Social Metrics
# ---------------------------------------------------------------------------
@router.get("/status/social")
async def get_social_status(request: Request):
    """Social metrics: engagement stats, recent post history."""
    plugin = _get_plugin(request)
    try:
        # V3: social subsystem not yet wired — return empty metrics gracefully
        # Social posting helper exists but lacks API keys
        return _ok({
            "metrics": {
                "mentions_received": 0,
                "daily_replies": 0,
                "reply_likes": 0,
                "daily_likes": 0,
                "total_posts": 0,
            },
            "recent_posts": [],
            "note": "Social subsystem not yet active in V3",
        })
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/epochs — Circadian Rhythm Status
# ---------------------------------------------------------------------------
@router.get("/status/epochs")
async def get_epoch_status(request: Request):
    """Circadian rhythm: meditation/rebirth timing, sovereignty index."""
    plugin = _get_plugin(request)
    try:
        from datetime import datetime, timezone

        now = time.time()

        # V3: backup module not instantiated — use available data
        snapshot_hash = getattr(getattr(plugin, 'backup', None), 'current_snapshot_hash', None) or "N/A"

        # Dreaming/consciousness data from spirit proxy
        dreaming_data = {}
        soul_gen = 1
        try:
            coord = getattr(plugin, '_spirit_proxy', None)
            if coord:
                coord_data = coord.get_coordinator()
                dreaming_data = coord_data.get("dreaming", {})
            soul_gen = getattr(plugin.soul, 'current_gen', 1)
        except Exception:
            pass

        # Compute intervals from available attributes
        meditation_interval = getattr(plugin, "_meditation_interval", 21600)
        rebirth_interval = getattr(plugin, "_rebirth_interval", 86400)
        is_meditating = getattr(plugin, "_is_meditating", False)

        epoch_data = {
            "is_meditating": is_meditating,
            "last_snapshot_hash": snapshot_hash,
            "soul_generation": soul_gen,
            "execution_mode": getattr(plugin, "_last_execution_mode", "v3"),
            "vault_program_id": plugin._full_config.get("network", {}).get("vault_program_id", ""),
            "small_epoch_interval_hours": round(meditation_interval / 3600, 2),
            "greater_epoch_interval_hours": round(rebirth_interval / 3600, 2),
            "last_small_epoch": None,
            "last_greater_epoch": None,
            "next_small_epoch": datetime.fromtimestamp(now + meditation_interval, tz=timezone.utc).isoformat(),
            "next_greater_epoch": datetime.fromtimestamp(now + rebirth_interval, tz=timezone.utc).isoformat(),
            "small_epoch_count": 0,
            "greater_epoch_count": soul_gen,
            "dreaming": dreaming_data,
        }

        if hasattr(plugin, "_last_commit_signature") and plugin._last_commit_signature:
            epoch_data["last_commit_signature"] = plugin._last_commit_signature

        return _ok(epoch_data)
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/research — Recent Research Topics
# ---------------------------------------------------------------------------
@router.get("/status/research")
async def get_research_status(request: Request):
    """Recent research topics and sources used."""
    plugin = _get_plugin(request)
    try:
        recent_topics = []
        last_sources = getattr(plugin, "_last_research_sources", {})
        source_distribution: dict = {}

        # Read from sage_research.log (primary source of research data)
        try:
            import json as _json
            _base = os.path.join(os.path.dirname(__file__), "..", "..", "data")
            log_path = os.path.join(_base, "logs", "sage_research.log")
            if os.path.exists(log_path):
                lines = []
                with open(log_path, "r") as f:
                    lines = f.readlines()
                # Parse last 20 entries (most recent at end)
                for line in reversed(lines[-50:]):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = _json.loads(line)
                        recent_topics.append({
                            "topic": entry.get("knowledge_gap", ""),
                            "summary": entry.get("distilled_summary", ""),
                            "sources": entry.get("sources_used", []),
                            "urls": entry.get("urls_scraped", []),
                            "timestamp": entry.get("timestamp", ""),
                        })
                        # Aggregate source distribution
                        for src in entry.get("sources_used", []):
                            source_distribution[src] = source_distribution.get(src, 0) + 1
                    except Exception:
                        continue
                    if len(recent_topics) >= 20:
                        break
        except Exception:
            pass

        # Fallback: also check observatory event_log for agency actions
        if not recent_topics:
            try:
                import sqlite3
                _base2 = os.path.join(os.path.dirname(__file__), "..", "..", "data")
                db_path = os.path.join(_base2, "observatory.db")
                if os.path.exists(db_path):
                    rows = await sqlite_async.query(
                        db_path,
                        "SELECT summary, details, ts FROM event_log "
                        "WHERE event_type = 'agency_action' "
                        "ORDER BY ts DESC LIMIT 20",
                        row_factory=sqlite3.Row,
                    )
                    for r in rows:
                        recent_topics.append({
                            "topic": r["summary"],
                            "timestamp": str(r["ts"]),
                        })
            except Exception:
                pass

        return _ok({
            "recent_topics": recent_topics,
            "source_distribution": source_distribution,
            "last_sources": last_sources if isinstance(last_sources, dict) else {},
        })
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/bus-health — Bus health / META-CGN observability
# ---------------------------------------------------------------------------
@router.get("/v4/thread-pool")
async def thread_pool_stats(request: Request):
    """Asyncio default thread-pool executor stats.

    After Phase E.2 (2026-04-14) and API-fix follow-up (ccd2ef6),
    ~100 sites are wrapped in asyncio.to_thread. If the pool saturates
    (all workers busy + queued tasks growing), endpoints start serializing
    and latency climbs. This endpoint surfaces:

      • max_workers      — pool size (64 post-bump from asyncio default ~36)
      • live_workers     — threads actually spawned so far (<= max_workers)
      • idle_workers     — workers waiting on work_queue (approximation)
      • busy_workers     — live - idle (tasks currently executing)
      • queued_tasks     — work-queue depth (tasks waiting for a worker)
      • saturation_pct   — (busy + queued*0.5) / max_workers × 100

    Implementation uses private attrs on ThreadPoolExecutor because the
    stdlib doesn't expose these. Safe as long as we're on CPython (which
    hasn't changed these names since 3.2). Falls back gracefully if the
    attrs are renamed.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        executor = getattr(loop, "_default_executor", None)
        if executor is None:
            return _error("no default executor on this loop")

        max_workers = getattr(executor, "_max_workers", None)
        threads = getattr(executor, "_threads", None)
        work_queue = getattr(executor, "_work_queue", None)
        idle_semaphore = getattr(executor, "_idle_semaphore", None)

        live = len(threads) if threads is not None else None
        queued = work_queue.qsize() if work_queue is not None else None
        # _idle_semaphore._value approximates idle workers (3.8+)
        idle = getattr(idle_semaphore, "_value", None) if idle_semaphore else None
        busy = (live - idle) if (live is not None and idle is not None) else None

        saturation = None
        if max_workers and busy is not None and queued is not None:
            saturation = round(((busy + queued * 0.5) / max_workers) * 100, 1)

        state = "healthy"
        if saturation is not None:
            if saturation >= 90:
                state = "critical"
            elif saturation >= 60:
                state = "warning"

        return _ok({
            "state": state,
            "max_workers": max_workers,
            "live_workers": live,
            "idle_workers": idle,
            "busy_workers": busy,
            "queued_tasks": queued,
            "saturation_pct": saturation,
            "executor_class": type(executor).__name__,
        })
    except Exception as e:
        return _error(f"thread_pool_stats: {e}")


@router.get("/v4/ns-health")
async def ns_health(request: Request):
    """Neural Nervous System health snapshot — per-program urgency distribution,
    VM-baseline coverage, fire rates, training state, hormonal maturity.

    Returns the canonical snapshot used by:
      - arch_map ns-health (when live API available)
      - Observatory NS widget (future)
      - session-startup program-collapse detection
      - CI gates for rFP β Stages 1+2

    Key fields: overall ∈ {healthy, warning, critical}, verdicts (DEAD/LOW/OK
    per program), programs (detailed per-program stats), hormonal (per-hormone
    level/threshold/fire_count), training (phase + supervision weight + counts).

    Cross-process design (rFP β Phase 3): NeuralNervousSystem lives in the
    spirit_worker subprocess, so direct attribute access from the dashboard
    main process returns None. Spirit_worker persists get_health_snapshot()
    output to data/neural_nervous_system/health_snapshot.json every 1000
    transitions; dashboard reads from disk. Snapshot freshness reported via
    snapshot_age_s field (None if file missing, integer seconds otherwise).
    """
    import os, json, time
    try:
        # Try direct access first (works if NeuralNS happens to be in same
        # process — e.g., during integration tests)
        plugin = _get_plugin(request)
        nns = getattr(plugin, "neural_nervous_system", None)
        if nns is not None and hasattr(nns, "get_health_snapshot"):
            snap = nns.get_health_snapshot()
            snap["source"] = "direct"
            snap["snapshot_age_s"] = 0
            return _ok(snap)

        # Fallback: read snapshot file persisted by spirit_worker
        snap_path = "./data/neural_nervous_system/health_snapshot.json"
        if not os.path.exists(snap_path):
            return _error(
                "NS health snapshot not yet written — wait until first 1000 "
                "transitions accumulate after boot (~3 min)", code=503)
        try:
            with open(snap_path) as f:
                snap = json.load(f)
            snap_ts = snap.get("snapshot_ts", 0)
            snap["snapshot_age_s"] = round(time.time() - snap_ts, 1) if snap_ts else None
            snap["source"] = "snapshot_file"
            return _ok(snap)
        except Exception as e:
            return _error(f"snapshot file read failed: {e}")
    except Exception as e:
        return _error(str(e))


@router.get("/v4/bus-health")
async def bus_health(request: Request):
    """Bus health snapshot: per-producer emission rates, queue depths,
    orphan signals. Respects the two canonical health indicators
    (bus-clean + π-rate) invariant set in rFP_meta_cgn_v3.

    Returns overall_state ∈ {healthy, warning, critical}. Used by:
      - session startup protocol (catch bus issues in minute 1)
      - Observatory widget
      - CI rate-budget tests
      - per-producer 10-15 min observation window during v3 Phase D rollout
    """
    try:
        plugin = _get_plugin(request)
        mon = getattr(plugin, "bus_health", None)
        if mon is None:
            return _error("bus_health monitor not wired")
        return _ok(mon.snapshot())
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /health — System Health Check
# ---------------------------------------------------------------------------
@router.get("/health")
async def health_check(request: Request):
    """
    System health with Solana Capability Report.

    Returns per-subsystem status and per-feature Solana capability status:
      ACTIVE  — fully operational
      DEGRADED — operational with limitations
      STUB    — interface present, backend not yet deployed
      ABSENT  — dependency missing
    """
    plugin = _get_plugin(request)
    is_v3 = hasattr(plugin, "get_v3_status")

    try:
        from titan_plugin.utils.solana_client import is_available as solana_ok

        sdk_available = solana_ok()

        # Network/wallet — available in both V2 and V3
        sol_balance = 0.0
        rpc_connected = False
        wallet_loaded = False
        if plugin.network and hasattr(plugin.network, "get_balance"):
            import asyncio
            # Phase E Fix 3: tight 300ms timeout. Watchdog cron has 10s
            # deadline; old 5s timeout cumulated with vault (4s) + memory
            # (3s) = 12s worst case → spurious force-restarts. Tight per-
            # call budget keeps total /health <1s even when degraded.
            try:
                sol_balance = await asyncio.wait_for(plugin.network.get_balance(), timeout=0.3)
            except (asyncio.TimeoutError, Exception):
                sol_balance = 0.0
            rpc_connected = sol_balance > 0 or getattr(plugin.network, "_rpc_client", None) is not None
            wallet_loaded = getattr(plugin.network, "pubkey", None) is not None

        # Vault status — V2 only (skip in V3 to avoid complex V2 method calls)
        # 2026-04-09 fix: get_raw_account_data() previously had NO timeout. When
        # mainnet RPC was slow, /health could take 15-30s, the t1_watchdog cron
        # job (10s deadline) declared T1 unhealthy and force-killed the entire
        # process group. /tmp/titan1_watchdog.log showed 11 force-restarts in
        # 24h purely from this path. Cap the whole vault check at 4s — well
        # below the watchdog's 10s deadline — and degrade gracefully on timeout.
        vault_program_id = plugin._full_config.get("network", {}).get("vault_program_id", "")
        vault_status = "STUB"
        vault_data = None
        if not is_v3 and vault_program_id and sdk_available and wallet_loaded:
            try:
                from titan_plugin.utils.solana_client import derive_vault_pda
                pda_result = derive_vault_pda(plugin.network.pubkey, vault_program_id)
                if pda_result:
                    vault_pda, _ = pda_result
                    # Phase E Fix 3: tight 300ms (was 4.0s). DEGRADED on timeout.
                    raw_data = await asyncio.wait_for(
                        plugin.network.get_raw_account_data(str(vault_pda)),
                        timeout=0.3,
                    )
                    if raw_data:
                        from titan_plugin.utils.solana_client import decode_vault_state
                        vault_data = decode_vault_state(raw_data)
                        vault_status = "ACTIVE" if vault_data and vault_data.get("commit_count", 0) > 0 else "DEGRADED"
                    else:
                        vault_status = "DEGRADED"
            except asyncio.TimeoutError:
                logger.warning("[Health] vault data fetch exceeded 4s — degraded")
                vault_status = "DEGRADED"
            except Exception:
                vault_status = "DEGRADED"

        # Maker auth
        maker_pubkey = ""
        if plugin.soul and hasattr(plugin.soul, "_maker_pubkey"):
            maker_pubkey = str(plugin.soul._maker_pubkey) if plugin.soul._maker_pubkey else ""

        # Per-feature Solana capabilities
        solana_capabilities = {
            "SOLANA_SDK": "ACTIVE" if sdk_available else "ABSENT",
            "MEMO_INSCRIPTION": "ACTIVE" if sdk_available else "ABSENT",
            "STATE_ROOT_ZK": vault_status,
            "SHADOW_DRIVE_SYNC": "ACTIVE",
            "RPC_CONNECTIVITY": "ACTIVE" if rpc_connected else ("DEGRADED" if sdk_available else "ABSENT"),
            "WALLET": "ACTIVE" if wallet_loaded else "DEGRADED",
            "MAKER_AUTH": "ACTIVE" if maker_pubkey else "DEGRADED",
        }
        if not is_v3:
            solana_capabilities["ZK_COMPRESSION"] = (
                "ACTIVE" if (getattr(getattr(plugin, "meditation", None), "_photon", None) is not None)
                else "STUB" if sdk_available
                else "ABSENT"
            )

        # Subsystem health — V3 uses Guardian module states
        if is_v3:
            guardian_status = plugin.guardian.get_status()
            subsystems = {
                "soul": "ACTIVE" if plugin.soul else "DEGRADED",
                "bus": "ACTIVE",
                "guardian": "ACTIVE",
                "observatory": "ACTIVE",
            }
            # Add module states from Guardian
            for mod_name, mod_info in guardian_status.items():
                state = mod_info.get("state", "stopped")
                subsystems[mod_name] = "ACTIVE" if state == "running" else (
                    "DEGRADED" if state in ("starting", "unhealthy") else "ABSENT"
                )
            # Add Core-hosted subsystems (not Guardian-supervised)
            subsystems["metabolism"] = "ACTIVE" if plugin.metabolism else "ABSENT"
            subsystems["studio"] = "ACTIVE" if getattr(plugin, "studio", None) else "ABSENT"
            subsystems["social"] = "ACTIVE" if plugin.social else "ABSENT"
        else:
            ollama_cloud = getattr(plugin, "_ollama_cloud", None)
            subsystems = {
                "memory": "ACTIVE" if plugin.memory else "ABSENT",
                "metabolism": "ACTIVE" if plugin.metabolism else "ABSENT",
                "soul": "ACTIVE" if plugin.soul else "ABSENT",
                "guardian": "ACTIVE" if plugin.guardian else "ABSENT",
                "gatekeeper": "ACTIVE" if plugin.gatekeeper else "ABSENT",
                "studio": "ACTIVE" if getattr(plugin, "studio", None) else "ABSENT",
                "social": "ACTIVE" if plugin.social else "ABSENT",
                "memory_backend": "ACTIVE" if (plugin.memory and getattr(plugin.memory, "_cognee_ready", False)) else "DEGRADED",
                "observatory": "ACTIVE",
                "ollama_cloud": "ACTIVE" if ollama_cloud else "ABSENT",
            }

        # Cognitive readiness — query via proxy (attribute lives in subprocess)
        # Phase E Fix 3: 300ms (was 3.0s). When memory module is restarting,
        # this used to hang the whole /health endpoint and trigger spurious
        # watchdog force-restarts. cognee_ready=False on timeout = DEGRADED
        # signal in response, no blocking.
        cognee_ready = False
        if plugin.memory:
            try:
                mem_status = await asyncio.wait_for(
                    asyncio.to_thread(plugin.memory.get_memory_status), timeout=0.3)
                cognee_ready = mem_status.get("cognee_ready", False) if mem_status else False
            except Exception:
                pass
        recorder_ready = hasattr(plugin, "recorder") and plugin.recorder is not None

        # Capabilities array for frontend
        capabilities = [
            {"name": name, "status": status}
            for name, status in solana_capabilities.items()
        ]

        # Overall status
        active_count = sum(1 for v in subsystems.values() if v == "ACTIVE")
        overall_status = "ACTIVE" if active_count >= 6 else ("DEGRADED" if active_count >= 3 else "OFFLINE")

        # Privacy filter
        privacy_cfg = plugin._full_config.get("privacy", {})
        privacy_redactions = getattr(plugin, "_privacy_redaction_count", 0)

        # Bus-health summary (canonical indicator — see rFP_meta_cgn_v3 § 1)
        # Compact embedded snapshot; full detail via /v4/bus-health.
        bus_health_summary = None
        bus_monitor = getattr(plugin, "bus_health", None)
        if bus_monitor is not None:
            try:
                _snap = bus_monitor.snapshot()
                bus_health_summary = {
                    "state": _snap.get("overall_state", "unknown"),
                    "total_emission_rate_hz": _snap.get("total_emission_rate_1min_hz", 0),
                    "rate_budget_hz": _snap.get("rate_budget_hz", 0.5),
                    "max_queue_fraction": _snap.get("max_queue_fraction", 0),
                    "backpressure": _snap.get("backpressure_active", False),
                    "orphan_count": _snap.get("orphans", {}).get("total_count", 0),
                    "orphan_tuples": len(_snap.get("orphans", {}).get("unique_tuples", [])),
                }
            except Exception:
                pass

        response = {
            "version": "v6.0" if is_v3 else "v2.1",
            "status": overall_status,
            "maker_pubkey": maker_pubkey,
            "sol_balance": round(sol_balance, 6),
            "capabilities": capabilities,
            "solana_capabilities": solana_capabilities,
            "subsystems": subsystems,
            "bus_health": bus_health_summary,
            "privacy_filter": {
                "enabled": privacy_cfg.get("sanitize_pii", True),
                "redactions": privacy_redactions,
            },
            "cognee_ready": cognee_ready,  # kept for API compat
            "memory_backend_ready": cognee_ready,
            "recorder_ready": recorder_ready,
            "limbo_mode": plugin._limbo_mode,
            "network": getattr(plugin.network, "_network_name", "unknown") if plugin.network else "none",
            "rpc_endpoint": (plugin.network.rpc_urls[0] if plugin.network and hasattr(plugin.network, "rpc_urls") and plugin.network.rpc_urls else None),
        }

        # Vault
        if vault_data:
            response["vault"] = vault_data
        elif vault_program_id:
            response["vault"] = {"program_id": vault_program_id, "status": "not_initialized"}

        # V3: include guardian module details
        if is_v3:
            response["v3"] = plugin.get_v3_status()

        # Ollama Cloud (V2 only)
        if not is_v3:
            ollama_cloud = getattr(plugin, "_ollama_cloud", None)
            if ollama_cloud:
                response["ollama_cloud"] = ollama_cloud.get_stats()

        return _ok(response)
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/art — Serve Procedural Art
# ---------------------------------------------------------------------------
@router.get("/status/art")
async def get_art(request: Request, live: bool = Query(False)):
    """
    Serve the Titan's procedural art.

    ?live=false (default): Serve the most recent generated flow field PNG.
    ?live=true: Generate a real-time mood flow field based on current state.
    """
    plugin = _get_plugin(request)

    if live:
        # Dynamic render based on current mood
        try:
            from titan_plugin.expressive.art import ProceduralArtGen

            art_gen = ProceduralArtGen()
            mood_score = plugin.mood_engine.previous_mood
            node_count = plugin.memory.get_persistent_count()
            intensity = max(1, int(mood_score * 10))

            state_root = f"live_{int(time.time())}"
            img = art_gen.generate_flow_field(
                state_root, node_count, intensity, return_image=True,
            )
            if img:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                return StreamingResponse(buf, media_type="image/png")
        except Exception as e:
            logger.debug("[Dashboard] Live art generation failed: %s", e)

    # Static: serve most recent file from Studio output
    try:
        studio = plugin.studio
        gallery = studio.get_gallery(category="meditation", limit=1)
        if gallery:
            art_path = Path(gallery[0]["path"])
            if art_path.exists():
                media = "image/png" if art_path.suffix == ".png" else "image/jpeg"
                return StreamingResponse(open(art_path, "rb"), media_type=media)
    except Exception:
        pass

    # Legacy fallback: check old art_exports directory
    for search_dir in [Path("./data/studio_exports/meditation"), Path("./art_exports")]:
        if search_dir.exists():
            art_files = sorted(
                list(search_dir.glob("flow_meditation_*.jpg")) + list(search_dir.glob("flow_meditation_*.png")),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if art_files:
                media = "image/png" if art_files[0].suffix == ".png" else "image/jpeg"
                return StreamingResponse(open(art_files[0], "rb"), media_type=media)

    return _error("No art available.", 404)


# ---------------------------------------------------------------------------
# GET /status/audio — Serve Procedural Audio
# ---------------------------------------------------------------------------
@router.get("/status/audio")
async def get_audio(request: Request):
    """Serve the most recent blockchain sonification WAV."""
    plugin = _get_plugin(request)

    # Try Studio gallery first
    try:
        studio = plugin.studio
        gallery = studio.get_gallery(category="epoch", limit=10)
        for item in gallery:
            if item["filename"].endswith(".wav"):
                audio_path = Path(item["path"])
                if audio_path.exists():
                    return StreamingResponse(open(audio_path, "rb"), media_type="audio/wav")
    except Exception:
        pass

    # Legacy fallback
    for search_dir in [Path("./data/studio_exports/epoch"), Path("./art_exports")]:
        if search_dir.exists():
            audio_files = sorted(
                search_dir.glob("sonification_*.wav"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if audio_files:
                return StreamingResponse(open(audio_files[0], "rb"), media_type="audio/wav")

    return _error("No audio available.", 404)


# ---------------------------------------------------------------------------
# GET /status/nft — NFT Timeline
# ---------------------------------------------------------------------------
@router.get("/status/nft")
async def get_nft_timeline(request: Request):
    """
    Fetch Titan's minted NFTs from on-chain (Metaplex DAS API).
    Returns a timeline of Soul evolution NFTs with metadata and art URIs.
    """
    plugin = _get_plugin(request)
    try:
        pubkey = plugin.network.pubkey
        if pubkey is None:
            return _ok({"nfts": [], "note": "No wallet loaded."})

        import httpx

        # Use DAS (Digital Asset Standard) API — available on Helius and public RPCs
        rpc_url = plugin.network.premium_rpc or plugin.network.rpc_urls[0]

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getAssetsByOwner",
                    "params": {
                        "ownerAddress": str(pubkey),
                        "page": 1,
                        "limit": 20,
                    },
                },
            )
            data = resp.json()

        items = data.get("result", {}).get("items", [])
        nfts = []
        for item in items:
            content = item.get("content", {})
            metadata = content.get("metadata", {})
            files = content.get("files", [])

            # Metaplex Core: attributes in plugins.attributes.data.attribute_list
            attrs = {}
            plugins = item.get("plugins", {})
            attr_plugin = plugins.get("attributes", {}).get("data", {})
            attr_list = attr_plugin.get("attribute_list", [])
            for a in attr_list:
                if isinstance(a, dict) and "key" in a:
                    attrs[a["key"]] = a.get("value", "")

            name = metadata.get("name", "Unknown")
            minted_at = ""
            # Try to extract date: "Titan Test 1773236104" (epoch in name) or "— 2026-03-11T..."
            if "—" in name:
                date_part = name.split("—")[-1].strip()
                if len(date_part) >= 10:
                    minted_at = date_part
            else:
                # Check if last word in name is an epoch timestamp
                parts = name.split()
                if parts:
                    try:
                        epoch = int(parts[-1])
                        if epoch > 1700000000:  # reasonable epoch range
                            minted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))
                    except (ValueError, OSError):
                        pass

            nfts.append({
                "id": item.get("id", ""),
                "name": name,
                "symbol": metadata.get("symbol", ""),
                "description": metadata.get("description", ""),
                "image": (files[0].get("uri", "") if files else content.get("json_uri", "")),
                "json_uri": content.get("json_uri", ""),
                "attributes": attrs,
                "minted_at": minted_at,
            })

        return _ok({"nfts": nfts, "wallet": str(pubkey)})

    except Exception as e:
        logger.error("[Dashboard] NFT fetch failed: %s", e)
        return _ok({"nfts": [], "note": f"NFT fetch unavailable: {e}"})


# ---------------------------------------------------------------------------
# GET /status/history — Vital Signs Time Series
# ---------------------------------------------------------------------------
@router.get("/status/history")
async def get_vital_history(
    request: Request,
    days: int = Query(7, ge=1, le=90),
    metrics: str = Query(None, description="Comma-separated metric names"),
):
    """
    Fetch historical vital snapshots for charting (Sovereignty Horizon, etc.).
    Lazy-loaded — only fetched when the user navigates to a history view.
    """
    plugin = _get_plugin(request)
    try:
        obs_db = getattr(plugin, "_observatory_db", None)
        if obs_db is None:
            return _ok({"snapshots": [], "note": "Observatory DB not initialized."})

        metric_list = [m.strip() for m in metrics.split(",")] if metrics else None
        raw_snapshots = obs_db.get_vital_history(days=days, metrics=metric_list)
        # If no recent data, try extending window to find any available data
        if len(raw_snapshots) == 0 and days <= 30:
            raw_snapshots = obs_db.get_vital_history(days=90, metrics=metric_list)
        snapshots = []
        for s in raw_snapshots:
            entry = dict(s) if not isinstance(s, dict) else s.copy()
            # Add frontend-compatible fields
            ts = entry.get("ts", 0)
            if isinstance(ts, (int, float)) and ts > 0:
                entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
            entry["memory_count"] = entry.get("persistent_count", 0)
            snapshots.append(entry)
        return _ok({"snapshots": snapshots, "days": days, "count": len(snapshots)})
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/archive — Expressive Archive (Art, Haikus, X Posts, Audio)
# ---------------------------------------------------------------------------
@router.get("/status/archive")
async def get_expressive_archive(
    request: Request,
    type: str = Query(None, description="Filter: art, haiku, audio, x_post"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    Fetch archived expressive outputs. Lazy-loaded for the Soul Mosaic and history views.
    """
    plugin = _get_plugin(request)
    try:
        obs_db = getattr(plugin, "_observatory_db", None)
        if obs_db is None:
            return _ok({"items": [], "note": "Observatory DB not initialized."})

        raw_items = obs_db.get_expressive_archive(type_=type, limit=limit, offset=offset)
        items = []
        for item in raw_items:
            entry = dict(item) if not isinstance(item, dict) else item
            # Convert epoch timestamps to ISO for frontend
            ts = entry.get("ts") or entry.get("timestamp") or entry.get("created_at")
            if isinstance(ts, (int, float)):
                entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
            else:
                entry["timestamp"] = str(ts or "")
            # Art entries: set content to media URL
            if entry.get("type") == "art":
                media_path = entry.get("media_path") or entry.get("path", "")
                if media_path:
                    entry["content"] = f"/media/{media_path}"
            items.append(entry)
        return _ok({"items": items, "count": len(items)})
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/events — Event History Log
# ---------------------------------------------------------------------------
@router.get("/status/events")
async def get_event_history(
    request: Request,
    type: str = Query(None, description="Filter by event_type"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    Fetch event log history. Supports filtering by event type.
    """
    plugin = _get_plugin(request)
    try:
        obs_db = getattr(plugin, "_observatory_db", None)
        if obs_db is None:
            return _ok({"events": [], "note": "Observatory DB not initialized."})

        raw_events = obs_db.get_events(event_type=type, limit=limit, offset=offset)
        # Map DB fields (ts, event_type, summary, details) → frontend (type, data, timestamp)
        events = []
        for e in raw_events:
            events.append({
                "type": e.get("event_type", "unknown"),
                "data": e.get("details", {}),
                "timestamp": e.get("ts", 0),
            })
        return _ok({"events": events, "count": len(events)})
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/guardian — Guardian Activity Log
# ---------------------------------------------------------------------------
@router.get("/status/guardian")
async def get_guardian_log(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
):
    """
    Fetch recent guardian safety actions (never exposes blocked content).
    Shows tier, action, and category only.
    """
    plugin = _get_plugin(request)
    try:
        obs_db = getattr(plugin, "_observatory_db", None)
        if obs_db is None:
            return _ok({"actions": [], "note": "Observatory DB not initialized."})

        actions = obs_db.get_guardian_log(limit=limit)
        return _ok({"actions": actions, "count": len(actions)})
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /media/{path} — Serve studio art files
# ---------------------------------------------------------------------------
@router.get("/media/{path:path}")
async def serve_media(path: str):
    """Serve generated art/audio files from the studio output directory."""
    from fastapi.responses import FileResponse
    # Art files stored under project root (media_path starts with "data/...")
    base = Path(__file__).resolve().parent.parent.parent
    target = (base / path).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        return _error("Not found", 404)
    return FileResponse(target)


# ---------------------------------------------------------------------------
# GET /v3/trinity/history — Trinity Tensor Time-Series
# ---------------------------------------------------------------------------
@router.get("/v3/trinity/history")
async def get_trinity_history(
    request: Request,
    hours: int = Query(24, ge=1, le=720),
):
    """
    Fetch historical Trinity tensor snapshots for trend visualization.
    Returns Body/Mind/Spirit tensors, Middle Path loss, and center distances.
    """
    plugin = _get_plugin(request)
    try:
        obs_db = getattr(plugin, "_observatory_db", None)
        if obs_db is None:
            return _ok({"snapshots": [], "note": "Observatory DB not initialized."})

        raw = obs_db.get_trinity_history(hours=hours)
        snapshots = []
        for s in raw:
            entry = dict(s) if not isinstance(s, dict) else s.copy()
            ts = entry.get("ts", 0)
            if isinstance(ts, (int, float)) and ts > 0:
                entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
            snapshots.append(entry)
        return _ok({"snapshots": snapshots, "hours": hours, "count": len(snapshots)})
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/consciousness/history — Full Consciousness Epoch History
# ---------------------------------------------------------------------------
@router.get("/status/consciousness/history")
async def get_consciousness_history(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Fetch full consciousness epoch history from ConsciousnessDB.
    Returns epoch_id, state_vector (9 dims), journey_point (x,y,z),
    curvature, density, drift, trajectory, and distillation text.
    """
    import json as _json
    plugin = _get_plugin(request)
    try:
        # ConsciousnessDB stores all epochs — read directly (WAL mode supports concurrent reads)
        consciousness_db_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "consciousness.db")
        if not os.path.exists(consciousness_db_path):
            return _ok({"epochs": [], "note": "No consciousness DB found."})

        # Phase E.2.3 fix: sqlite3 in async endpoint blocks event loop
        import asyncio
        from titan_plugin.logic.consciousness import STATE_DIMS

        rows = await sqlite_async.query(
            consciousness_db_path,
            """SELECT epoch_id, timestamp, state_vector, drift_vector,
                      trajectory_vector, journey_x, journey_y, journey_z,
                      curvature, density, distillation, anchored_tx
               FROM epochs ORDER BY epoch_id DESC LIMIT ?""",
            (limit,),
        )
        epochs = []
        for row in rows:
            sv = _json.loads(row[2]) if isinstance(row[2], str) else row[2]
            drift = _json.loads(row[3]) if isinstance(row[3], str) else row[3]
            traj = _json.loads(row[4]) if isinstance(row[4], str) else row[4]
            dims = {}
            for i, dim_name in enumerate(STATE_DIMS):
                if i < len(sv):
                    dims[dim_name] = round(sv[i], 4)
            epochs.append({
                "epoch_id": row[0],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(row[1])),
                "ts": row[1],
                "state_vector": sv,
                "state_dims": dims,
                "drift_vector": drift,
                "drift_magnitude": sum(d * d for d in drift) ** 0.5 if drift else 0.0,
                "trajectory_vector": traj,
                "journey_point": {"x": row[5], "y": row[6], "z": row[7]},
                "curvature": round(row[8], 4),
                "density": round(row[9], 4),
                "distillation": row[10] or "",
                "anchored_tx": row[11] or "",
            })
        # Return in chronological order
        epochs.reverse()
        return _ok({"epochs": epochs, "count": len(epochs)})
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /status/growth/history — Growth Metrics Time-Series
# ---------------------------------------------------------------------------
@router.get("/status/growth/history")
async def get_growth_history(
    request: Request,
    days: int = Query(7, ge=1, le=90),
):
    """
    Fetch historical growth metrics (learning velocity, social density,
    metabolic health, directive alignment) for trend visualization.
    """
    plugin = _get_plugin(request)
    try:
        obs_db = getattr(plugin, "_observatory_db", None)
        if obs_db is None:
            return _ok({"snapshots": [], "note": "Observatory DB not initialized."})

        raw = obs_db.get_growth_history(days=days)
        snapshots = []
        for s in raw:
            entry = dict(s) if not isinstance(s, dict) else s.copy()
            ts = entry.get("ts", 0)
            if isinstance(ts, (int, float)) and ts > 0:
                entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
            snapshots.append(entry)
        return _ok({"snapshots": snapshots, "days": days, "count": len(snapshots)})
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v3/trinity — Live Trinity State
# ---------------------------------------------------------------------------
@router.get("/v3/trinity")
async def get_v3_trinity(request: Request):
    """
    V3 Trinity State — live Body, Mind, Spirit tensors + Guardian module status.
    Only available when running in V3 microkernel mode.
    """
    import asyncio  # 2026-04-14 fix: missing import (introduced by § 15 b3ea2f0)
    plugin = _get_plugin(request)
    try:
        # Check if this is a V3 core (has get_v3_status)
        if not hasattr(plugin, "get_v3_status"):
            return _ok({"error": "Not running in V3 mode", "v3": False})

        v3_status = plugin.get_v3_status()

        # Query Trinity tensors via proxies
        body_proxy = plugin._proxies.get("body")
        mind_proxy = plugin._proxies.get("mind")
        spirit_proxy = plugin._proxies.get("spirit")

        # 2026-04-14 final fix: read from tensor cache (refreshed every 8s
        # by the warmer daemon thread). Zero proxy calls on the hot path.
        # /v3/trinity now returns in <10ms regardless of spirit worker load.
        # Trade-off: data is up to 8s stale, which is fine for Observatory
        # (tensors don't change dramatically sub-second, and the warmer is
        # always catching up).
        _start_coordinator_warmer(plugin)  # lazy-start (also warms tensors)
        body_tensor = _trinity_tensor_cache["body_tensor"]
        mind_tensor = _trinity_tensor_cache["mind_tensor"]
        spirit_data = _trinity_tensor_cache["spirit_data"]

        # Body dimension labels
        body_dims = ["interoception", "proprioception", "somatosensation", "entropy", "thermal"]
        mind_dims = ["vision", "hearing", "taste", "smell", "touch"]
        spirit_dims = ["who", "why", "what", "body_scalar", "mind_scalar"]

        trinity = {
            "body": {
                "dims": body_dims,
                "values": body_tensor,
                "center_dist": sum((v - 0.5) ** 2 for v in body_tensor) ** 0.5,
            },
            "mind": {
                "dims": mind_dims,
                "values": mind_tensor,
                "center_dist": sum((v - 0.5) ** 2 for v in mind_tensor) ** 0.5,
            },
            "spirit": {
                "dims": spirit_dims,
                "values": spirit_data.get("spirit_tensor", [0.5] * 5),
                "body_scalar": spirit_data.get("body_center_dist", 0),
                "mind_scalar": spirit_data.get("mind_center_dist", 0),
            },
        }

        # V4 Time Awareness data (passed through from spirit_data)
        v4 = {}
        if spirit_data.get("sphere_clock"):
            v4["sphere_clock"] = spirit_data["sphere_clock"]
        if spirit_data.get("resonance"):
            v4["resonance"] = spirit_data["resonance"]
        if spirit_data.get("unified_spirit"):
            v4["unified_spirit"] = spirit_data["unified_spirit"]
        if spirit_data.get("consciousness"):
            v4["consciousness"] = spirit_data["consciousness"]
        if spirit_data.get("middle_path_loss") is not None:
            v4["middle_path_loss"] = spirit_data["middle_path_loss"]
        if spirit_data.get("impulse_engine"):
            v4["impulse_engine"] = spirit_data["impulse_engine"]
        if spirit_data.get("filter_down"):
            v4["filter_down"] = spirit_data["filter_down"]
        if spirit_data.get("intuition"):
            v4["intuition"] = spirit_data["intuition"]

        # Enrich with full 132D state vector + observables from coordinator cache
        outer_body_vals = [0.5] * 5
        outer_mind_vals = [0.5] * 15
        outer_spirit_vals = [0.5] * 45
        inner_mind_15d = list(mind_tensor) + [0.5] * 10  # extend basic 5D to 15D
        inner_spirit_45d = list(spirit_data.get("spirit_tensor", [0.5] * 5)) + [0.5] * 40
        observables = {}
        try:
            coord = await _get_cached_coordinator_async(plugin)
            # Full 132D state vector: iB(5)+iM(15)+iS(45)+oB(5)+oM(15)+oS(45)+meta(2)
            # Try v4 consciousness first (from spirit_data), then coordinator
            consciousness = spirit_data.get("consciousness", {})
            if not consciousness or not isinstance(consciousness, dict):
                consciousness = coord.get("consciousness", {})
            if isinstance(consciousness, dict):
                sv = consciousness.get("state_vector", [])
                if isinstance(sv, list) and len(sv) >= 130:
                    # Extract full tensors from 132D state vector
                    inner_mind_15d = sv[5:20]      # dims 5-19: inner mind 15D
                    inner_spirit_45d = sv[20:65]    # dims 20-64: inner spirit 45D
                    outer_body_vals = sv[65:70]     # dims 65-69: outer body 5D
                    outer_mind_vals = sv[70:85]     # dims 70-84: outer mind 15D
                    outer_spirit_vals = sv[85:130]  # dims 85-129: outer spirit 45D
                    # Update inner trinity with full tensors
                    trinity["mind"]["values"] = inner_mind_15d
                    trinity["mind"]["dims"] = mind_dims + [f"m{i}" for i in range(5, 15)]
                    trinity["spirit"]["values"] = inner_spirit_45d
                    trinity["spirit"]["dims"] = spirit_dims + [f"s{i}" for i in range(5, 45)]
            # Build observables for Global Observables Matrix
            inner_obs = coord.get("inner_lower_topology", {}).get("observables", {})
            outer_obs = coord.get("outer_lower_topology", {}).get("observables", {})
            if isinstance(inner_obs, dict) and inner_obs:
                observables["inner_body"] = inner_obs
                observables["inner_mind"] = inner_obs
                observables["inner_spirit"] = inner_obs
            if isinstance(outer_obs, dict) and outer_obs:
                observables["outer_body"] = outer_obs
                observables["outer_mind"] = outer_obs
                observables["outer_spirit"] = outer_obs
        except Exception:
            pass

        return _ok({
            "v3": True,
            **v3_status,
            "trinity": trinity,
            "outer_body": outer_body_vals,
            "outer_mind": outer_mind_vals,
            "outer_spirit": outer_spirit_vals,
            "observables": observables if observables else None,
            "v4": v4 if v4 else None,
        })
    except Exception as e:
        logger.error("[Dashboard] /v3/trinity error: %s", e)
        return _error(str(e))


@router.get("/v3/guardian")
async def get_v3_guardian(request: Request):
    """V3 Guardian module status — states, PIDs, RSS, heartbeats."""
    plugin = _get_plugin(request)
    try:
        if not hasattr(plugin, "guardian") or not hasattr(plugin.guardian, "get_status"):
            return _ok({"error": "Not running in V3 mode", "v3": False})

        return _ok({
            "v3": True,
            "modules": plugin.guardian.get_status(),
            "bus_stats": plugin.bus.stats if hasattr(plugin, "bus") else {},
        })
    except Exception as e:
        return _error(str(e))


@router.post("/v3/guardian/start/{module_name}")
async def start_v3_module(module_name: str, request: Request):
    """Start a V3 Guardian-supervised module by name."""
    plugin = _get_plugin(request)
    try:
        if not hasattr(plugin, "guardian"):
            return _error("Not running in V3 mode")
        ok = plugin.guardian.start(module_name)
        if ok:
            return _ok({"started": module_name})
        else:
            return _error(f"Failed to start '{module_name}' — check logs")
    except Exception as e:
        return _error(str(e))


@router.post("/v3/guardian/enable/{module_name}")
async def enable_v3_module(module_name: str, request: Request):
    """Re-enable a disabled Guardian module, reset restart counters, and start it."""
    plugin = _get_plugin(request)
    try:
        if not hasattr(plugin, "guardian"):
            return _error("Not running in V3 mode")
        ok = plugin.guardian.enable(module_name)
        if ok:
            return _ok({"enabled": module_name})
        else:
            return _error(f"Failed to enable '{module_name}' — check logs")
    except Exception as e:
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v3/agency — Agency stats, assessment history, recent reflections
# ---------------------------------------------------------------------------
@router.get("/v3/agency")
async def get_v3_agency(request: Request):
    """
    V3 Agency Module status — action history, assessment scores, helper statuses.

    Returns rolling assessment reflections suitable for dashboard display.
    """
    plugin = _get_plugin(request)
    try:
        if not hasattr(plugin, "_agency") or not plugin._agency:
            return _ok({
                "enabled": False,
                "actions": [],
                "stats": {},
            })

        stats = plugin._agency.get_stats()
        assessment_stats = {}
        if hasattr(plugin, "_agency_assessment") and plugin._agency_assessment:
            assessment_stats = plugin._agency_assessment.get_stats()

        advisor_stats = {}
        if hasattr(plugin, "_interface_advisor") and plugin._interface_advisor:
            advisor_stats = plugin._interface_advisor.get_stats()

        return _ok({
            "enabled": True,
            "agency": stats,
            "assessment": assessment_stats,
            "advisor": advisor_stats,
        })
    except Exception as e:
        logger.error("[Dashboard] /v3/agency error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/state — Complete V4 Time Awareness State
# ---------------------------------------------------------------------------
@router.get("/v4/state")
async def get_v4_state(request: Request):
    """
    V4 Time Awareness state — sphere clocks, resonance, unified spirit,
    consciousness, impulse engine, filter_down, intuition.
    """
    plugin = _get_plugin(request)
    try:
        if not hasattr(plugin, "get_v3_status"):
            return _ok({"error": "Not running in V3/V4 mode", "v4": False})

        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available", "v4": False})

        v4_state = spirit_proxy.get_v4_state()

        # Add Guardian module health for context
        guardian_status = plugin.guardian.get_status() if hasattr(plugin, "guardian") else {}

        return _ok({
            "v4": True,
            **v4_state,
            "guardian": guardian_status,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/state error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/sphere-clocks — Dedicated sphere clock state
# ---------------------------------------------------------------------------
@router.get("/v4/sphere-clocks")
async def get_v4_sphere_clocks(request: Request):
    """V4 SphereClockEngine: 6 inner clocks, phases, radii, pulse counts."""
    import asyncio
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        return _ok(await asyncio.to_thread(spirit_proxy.get_sphere_clocks))
    except Exception as e:
        logger.error("[Dashboard] /v4/sphere-clocks error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/resonance — Resonance detector state
# ---------------------------------------------------------------------------
@router.get("/v4/resonance")
async def get_v4_resonance(request: Request):
    """V4 ResonanceDetector: pair alignments, BIG/GREAT pulse counts."""
    import asyncio
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        return _ok(await asyncio.to_thread(spirit_proxy.get_resonance))
    except Exception as e:
        logger.error("[Dashboard] /v4/resonance error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/unified-spirit — UnifiedSpirit 30DT tensor state
# ---------------------------------------------------------------------------
@router.get("/v4/unified-spirit")
async def get_v4_unified_spirit(request: Request):
    """V4 UnifiedSpirit: 30DT tensor, velocity, stale status, focus multiplier."""
    import asyncio
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        return _ok(await asyncio.to_thread(spirit_proxy.get_unified_spirit))
    except Exception as e:
        logger.error("[Dashboard] /v4/unified-spirit error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/filter-down-status — V4/V5 FILTER_DOWN coexistence (rFP #2 Phase 7)
# ---------------------------------------------------------------------------
@router.get("/v4/filter-down-status")
async def get_v4_filter_down_status(request: Request):
    """FILTER_DOWN V4/V5 side-by-side state for rFP #2 coexistence monitoring."""
    import asyncio
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        return _ok(await asyncio.to_thread(spirit_proxy.get_filter_down_status))
    except Exception as e:
        logger.error("[Dashboard] /v4/filter-down-status error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/meditation/health — Meditation watchdog state + tracker + overdue
# ---------------------------------------------------------------------------
@router.get("/v4/meditation/health")
async def get_v4_meditation_health(request: Request):
    """Meditation health: watchdog health_snapshot + tracker + overdue flag.

    Used by `arch_map meditation` cross-Titan correlation (rFP
    self_healing_meditation_cadence I2) + observatory UI.
    """
    import asyncio
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        return _ok(await asyncio.to_thread(spirit_proxy.get_meditation_health))
    except Exception as e:
        logger.error("[Dashboard] /v4/meditation/health error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/meditation/force-trigger — Maker manual force-trigger (rFP §7)
# ---------------------------------------------------------------------------
@router.post("/v4/meditation/force-trigger")
async def post_v4_meditation_force_trigger(request: Request):
    """Maker override: send MEDITATION_REQUEST to memory_worker immediately.

    Publishes a force-trigger with source="maker_manual" so downstream can
    distinguish from watchdog-automatic and emergent triggers.
    """
    plugin = _get_plugin(request)
    try:
        if not plugin.bus:
            return _error("Bus not available")
        from ..bus import make_msg
        plugin.bus.publish(make_msg("MEDITATION_REQUEST", "dashboard", "meditation", {
            "source": "maker_manual",
            "reason": "manual_force_trigger",
        }))
        logger.warning("[Dashboard] /v4/meditation/force-trigger — Maker manual override issued")
        return _ok({"dispatched": True, "source": "maker_manual"})
    except Exception as e:
        logger.error("[Dashboard] /v4/meditation/force-trigger error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/nervous-system — V5 Neural NervousSystem learning metrics
# ---------------------------------------------------------------------------
@router.get("/v4/nervous-system")
async def get_v4_nervous_system(request: Request):
    """V5 Neural NervousSystem: per-program learning metrics, training phase, buffer sizes."""
    import asyncio
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        return _ok(await asyncio.to_thread(spirit_proxy.get_nervous_system))
    except Exception as e:
        logger.error("[Dashboard] /v4/nervous-system error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/vocabulary — Vocabulary from inner_memory.db
# ---------------------------------------------------------------------------
@router.get("/v4/vocabulary")
async def get_v4_vocabulary(request: Request):
    """Return Titan's learned vocabulary from inner_memory.db."""
    # Phase E.2.3 fix: sqlite3 in async endpoint blocks event loop
    import asyncio
    import sqlite3
    import json as _json
    try:
        db_path = "./data/inner_memory.db"

        def _vocab_rows(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            return conn.execute(
                "SELECT word, word_type, confidence, felt_tensor, hormone_pattern, "
                "times_encountered, times_produced, learning_phase, "
                "COALESCE(sensory_context, '[]'), "
                "COALESCE(meaning_contexts, '[]'), "
                "COALESCE(cross_modal_conf, 0.0) "
                "FROM vocabulary ORDER BY confidence DESC"
            ).fetchall()

        rows = await sqlite_async.with_connection(db_path, _vocab_rows)
        words = []
        grounded_count = 0
        def _safe_float(v, default=0.0):
            if isinstance(v, bytes):
                import struct as _st
                try: return _st.unpack('<f', v)[0]
                except Exception: return default
            try: return float(v) if v is not None else default
            except Exception: return default
        def _safe_json(v, default=None):
            if default is None: default = []
            if isinstance(v, bytes): return default
            if not v: return default
            try: return _json.loads(v)
            except Exception: return default
        for row in rows:
            xm = _safe_float(row[10] if len(row) > 10 else 0.0)
            if xm > 0:
                grounded_count += 1
            words.append({
                "word": row[0] if not isinstance(row[0], bytes) else row[0].decode(errors="replace"),
                "word_type": row[1] if not isinstance(row[1], bytes) else str(row[1]),
                "confidence": _safe_float(row[2]),
                "felt_tensor": _safe_json(row[3]),
                "hormone_pattern": _safe_json(row[4], {}),
                "times_encountered": int(_safe_float(row[5])),
                "times_produced": int(_safe_float(row[6])),
                "learning_phase": row[7] if not isinstance(row[7], bytes) else str(row[7]),
                "sensory_context": _safe_json(row[8]),
                "meaning_contexts": _safe_json(row[9]),
                "cross_modal_conf": xm,
            })
        return _ok({
            "words": words, "total": len(words),
            "grounded": grounded_count,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/vocabulary error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/language-grounding — Language grounding summary (lightweight)
# ---------------------------------------------------------------------------
@router.get("/v4/language-grounding")
async def get_v4_language_grounding(request: Request):
    """Lightweight language grounding summary for Observatory."""
    # Phase E.2.3 fix: sqlite3 in async endpoint blocks event loop
    import asyncio
    import sqlite3
    import json as _json
    try:
        db_path = "./data/inner_memory.db"

        def _vocab_summary(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            total = conn.execute("SELECT count(*) FROM vocabulary").fetchone()[0]
            producible = conn.execute(
                "SELECT count(*) FROM vocabulary WHERE learning_phase='producible'"
            ).fetchone()[0]
            grounded = conn.execute(
                "SELECT count(*) FROM vocabulary WHERE cross_modal_conf > 0"
            ).fetchone()[0]
            avg_conf = conn.execute(
                "SELECT avg(confidence) FROM vocabulary"
            ).fetchone()[0] or 0.0
            avg_xm = conn.execute(
                "SELECT avg(cross_modal_conf) FROM vocabulary WHERE cross_modal_conf > 0"
            ).fetchone()[0] or 0.0
            top_rows = conn.execute(
                "SELECT word, word_type, confidence, cross_modal_conf, "
                "times_encountered, sensory_context, meaning_contexts "
                "FROM vocabulary WHERE cross_modal_conf > 0 "
                "ORDER BY cross_modal_conf DESC LIMIT 20"
            ).fetchall()
            type_rows = conn.execute(
                "SELECT word_type, count(*) FROM vocabulary GROUP BY word_type"
            ).fetchall()
            return total, producible, grounded, avg_conf, avg_xm, top_rows, type_rows

        total, producible, grounded, avg_conf, avg_xm, top_rows, type_rows = (
            await sqlite_async.with_connection(db_path, _vocab_summary)
        )

        def _sf(v, default=0.0):
            if isinstance(v, bytes):
                import struct as _st
                try: return _st.unpack('<f', v)[0]
                except Exception: return default
            try: return float(v) if v is not None else default
            except Exception: return default
        def _sj(v, default=None):
            if default is None: default = []
            if isinstance(v, bytes): return default
            if not v: return default
            try: return _json.loads(v)
            except Exception: return default
        top_grounded = []
        for row in top_rows:
            contexts = _sj(row[5])
            meanings = _sj(row[6])
            assocs = []
            for m in meanings:
                if not isinstance(m, dict):
                    continue
                for a in m.get("associations", []):
                    if isinstance(a, (list, tuple)) and len(a) >= 2:
                        assocs.append({"word": a[0], "type": a[1]})
            top_grounded.append({
                "word": row[0] if not isinstance(row[0], bytes) else row[0].decode(errors="replace"),
                "word_type": row[1] if not isinstance(row[1], bytes) else str(row[1]),
                "confidence": _sf(row[2]), "cross_modal_conf": _sf(row[3]),
                "encounters": int(_sf(row[4])),
                "contexts": contexts[:3],
                "associations": assocs[:5],
            })
        type_dist = {row[0]: row[1] for row in type_rows}

        return _ok({
            "total_words": total,
            "producible": producible,
            "grounded": grounded,
            "grounding_rate": round(grounded / max(total, 1), 3),
            "avg_confidence": round(avg_conf, 3),
            "avg_grounding_confidence": round(avg_xm, 3),
            "word_types": type_dist,
            "top_grounded": top_grounded,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/language-grounding error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/cgn-social-action — Infer social action from learned CGN policy
# ---------------------------------------------------------------------------
@router.get("/v4/cgn-social-action")
async def get_v4_cgn_social_action(request: Request):
    """Infer best social action for a user from learned CGN policy.

    Query params: familiarity, interaction_count, social_valence, mention_count
    Returns: {action_name, confidence, q_values, tone_instruction}
    """
    try:
        from titan_plugin.logic.cgn_consumer_client import CGNConsumerClient

        # Parse user features from query params
        params = dict(request.query_params)
        user_features = {
            "familiarity": float(params.get("familiarity", 0.0)),
            "interaction_count": int(params.get("interaction_count", 0)),
            "social_valence": float(params.get("social_valence", 0.0)),
            "mention_count": int(params.get("mention_count", 0)),
            "net_sentiment": float(params.get("net_sentiment", 0.0)),
        }

        client = CGNConsumerClient("social", state_dir="data/cgn")

        # Build sensory context from current neuromod state
        neuromods = {}
        try:
            core = request.app.state.core
            coord = core.get_coordinator_state()
            nm = coord.get("neuromodulators", {})
            for k, v in nm.items():
                if isinstance(v, dict):
                    neuromods[k] = v.get("level", 0.5)
                else:
                    neuromods[k] = float(v)
        except Exception:
            pass

        result = client.infer_action(
            sensory_ctx={"neuromods": neuromods},
            features=user_features)

        # Add tone instruction from config
        try:
            import tomllib
        except ImportError:
            import toml as tomllib
        try:
            with open("titan_plugin/titan_params.toml", "rb") as f:
                cfg = tomllib.load(f)
            csp = cfg.get("cgn_social_policy", {})
            result["tone_instruction"] = csp.get(
                result["action_name"], "")
            result["policy_weight"] = csp.get("policy_weight", 0.3)
        except Exception:
            result["tone_instruction"] = ""
            result["policy_weight"] = 0.3

        return {"status": "ok", "data": result}
    except Exception as e:
        logger.error("[Dashboard] /v4/cgn-social-action error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/vocabulary/update-learning — Update word learning progress
# ---------------------------------------------------------------------------
@router.post("/v4/vocabulary/update-learning")
async def post_v4_vocabulary_update_learning(request: Request):
    """Update word learning progress after a teaching or reinforcement pass.

    Called by testsuite after each FEEL/RECOGNIZE/PRODUCE pass,
    or after autonomous SPEAK for reinforcement tracking.

    Body:
        word: str — the word
        word_type: str — noun/verb/adjective/adverb (for new words)
        pass_type: str — feel/recognize/produce/self_speak
        score: float — 0.0-1.0 success score
        stage: int — recipe stage (1/2/3, default 1)
        felt_tensor: list[float] — 130D tensor (optional)
        hormone_pattern: dict — hormone affinities (optional)
    """
    try:
        import sqlite3
        import json as _json

        body = await request.json()
        word = body.get("word", "").strip().lower()
        if not word:
            return _error("word is required")

        word_type = body.get("word_type", "unknown")
        pass_type = body.get("pass_type", "feel")
        score = float(body.get("score", 0.5))
        stage = int(body.get("stage", 1))
        felt_tensor = body.get("felt_tensor")
        hormone_pattern = body.get("hormone_pattern")

        # Phase mapping and ordering
        PHASE_MAP = {"feel": "felt", "recognize": "recognized",
                     "produce": "producible", "self_speak": "producible"}
        PHASE_ORDER = {"unlearned": 0, "felt": 1, "recognized": 2, "producible": 3}
        new_phase = PHASE_MAP.get(pass_type, "felt")

        # Confidence delta based on score and pass type
        if pass_type == "self_speak":
            conf_delta = 0.02  # Small reinforcement for autonomous use
        elif score >= 0.5:
            conf_delta = 0.05 * score  # Gradual increase
        else:
            conf_delta = -0.02  # Slight decrease on failure

        now = __import__("time").time()
        db_path = "./data/inner_memory.db"

        def _do_update(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            # Check if word exists
            existing = conn.execute(
                "SELECT learning_phase, confidence FROM vocabulary WHERE word = ?",
                (word,)).fetchone()

            if existing:
                cur_phase = existing[0] or "unlearned"
                cur_conf = existing[1] or 0.0

                # Only advance phase forward, never regress
                if PHASE_ORDER.get(new_phase, 0) > PHASE_ORDER.get(cur_phase, 0):
                    final_phase = new_phase
                else:
                    final_phase = cur_phase

                # Build update
                updates = ["learning_phase = ?", "last_encountered = ?"]
                params = [final_phase, now]

                if felt_tensor:
                    updates.append("felt_tensor = ?")
                    params.append(_json.dumps(felt_tensor))

                if conf_delta != 0.0:
                    updates.append("confidence = MIN(1.0, MAX(0.0, confidence + ?))")
                    params.append(conf_delta)

                if pass_type in ("feel", "recognize"):
                    updates.append("times_encountered = times_encountered + 1")
                if pass_type in ("produce", "self_speak"):
                    updates.append("times_produced = times_produced + 1")

                params.append(word)
                conn.execute(
                    f"UPDATE vocabulary SET {', '.join(updates)} WHERE word = ?",
                    params)

                new_conf = min(1.0, max(0.0, cur_conf + conf_delta))
            else:
                # Auto-create new word
                conn.execute(
                    "INSERT INTO vocabulary "
                    "(word, word_type, stage, felt_tensor, hormone_pattern, "
                    "confidence, times_encountered, times_produced, "
                    "learning_phase, created_at, last_encountered) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (word, word_type, stage,
                     _json.dumps(felt_tensor) if felt_tensor else None,
                     _json.dumps(hormone_pattern) if hormone_pattern else None,
                     max(0.0, conf_delta),  # Initial confidence from first pass
                     1 if pass_type in ("feel", "recognize") else 0,
                     1 if pass_type in ("produce", "self_speak") else 0,
                     new_phase, now, now))

                final_phase = new_phase
                new_conf = max(0.0, conf_delta)

            conn.commit()
            return final_phase, new_conf, existing is None

        final_phase, new_conf, created = await sqlite_async.with_connection(
            db_path, _do_update)
        return _ok({
            "word": word,
            "phase": final_phase,
            "confidence": round(new_conf, 4),
            "pass_type": pass_type,
            "created": created,
        })

    except Exception as e:
        logger.error("[Dashboard] /v4/vocabulary/update-learning error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/reload — Hot-reload Tier 2: reload worker modules without restart
# ---------------------------------------------------------------------------
@router.post("/v4/reload")
async def post_v4_reload(request: Request):
    """Live-reload logic modules with zero consciousness gap.

    Usage:
      Reload ALL modules:
        POST /v4/reload  {"all": true, "reason": "coupling fix"}
      Reload specific modules:
        POST /v4/reload  {"modules": ["neuromodulator", "expression_composites"]}
    """
    plugin = _get_plugin(request)
    try:
        body = await request.json()
        modules = body.get("modules", [])
        reload_all = body.get("all", False)
        worker = body.get("worker", "spirit")
        reason = body.get("reason", "API request")

        if not modules and not reload_all:
            return _error("Specify {\"all\": true} or {\"modules\": [\"neuromodulator\", ...]}")

        from ..bus import make_msg
        plugin.bus.publish(make_msg("RELOAD", "core", worker, {
            "modules": modules,
            "all": reload_all,
            "reason": reason,
        }))
        return _ok({"status": "reload_requested", "worker": worker,
                     "all": reload_all, "modules": modules, "reason": reason})
    except Exception as e:
        logger.error("[Dashboard] /v4/reload error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/reload-api — Hot-reload API routes without restart
# ---------------------------------------------------------------------------
@router.post("/v4/reload-api")
async def post_v4_reload_api(request: Request):
    """Reload all API route modules (dashboard, maker, chat, etc.) without restart.

    Reimports route modules, rebuilds FastAPI app, swaps into running uvicorn.
    Zero downtime — existing connections continue, new requests use new routes.
    """
    plugin = _get_plugin(request)
    try:
        result = plugin.reload_api()
        return _ok(result)
    except Exception as e:
        logger.error("[Dashboard] /v4/reload-api error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/reload-config — Hot-reload titan_params.toml without restart
# ---------------------------------------------------------------------------
@router.post("/v4/reload-config")
async def post_v4_reload_config(request: Request):
    """Reload titan_params.toml into running spirit worker. No consciousness gap."""
    plugin = _get_plugin(request)
    try:
        import tomllib
        with open("titan_plugin/titan_params.toml", "rb") as f:
            new_params = tomllib.load(f)
        from ..bus import make_msg
        plugin.bus.publish(make_msg("CONFIG_RELOAD", "api", "spirit", new_params))
        return _ok({"status": "config_reloaded", "sections": list(new_params.keys())})
    except Exception as e:
        logger.error("[Dashboard] /v4/reload-config error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/admin/restart-module/{name} — Per-module hot-restart
# ---------------------------------------------------------------------------
# Replaces full titan_main restart for code changes scoped to a single
# subprocess. Calls Guardian.restart(name) which now uses SAVE_NOW →
# SAVE_DONE → SIGTERM, preserving in-memory state via the worker's save
# handlers. See feedback_prefer_hot_reload.md and DEFERRED_ITEMS
# HOT-RELOAD-ENDPOINT for full design rationale.
#
# Audit log: every call appends to /tmp/titan_restart_audit.log.
# ---------------------------------------------------------------------------

# Modules safe to restart in isolation. Excluded: rl, body, mind, llm,
# media (low-stakes but no benefit until they have save handlers),
# guardian itself (would deadlock), timechain (chain integrity sensitive,
# needs careful design).
_RESTART_MODULE_ALLOWLIST = {
    "spirit",       # Has SAVE_NOW handler (2026-04-13)
    "cgn",          # Has shutdown save (cgn_worker.py:428)
    "language",     # No save handler yet — restart still works, state in DB
    "memory",       # State in DuckDB/FAISS — survives without explicit save
    "knowledge",    # State in DB
}


@router.post("/v4/admin/restart-module/{name}")
async def post_v4_restart_module(name: str, request: Request):
    """Hot-restart a single Guardian-managed module without taking down titan_main.

    Flow:
      1. Validate `name` against allowlist.
      2. Append audit entry to /tmp/titan_restart_audit.log.
      3. Call Guardian.restart(name) which sends SAVE_NOW → waits for
         SAVE_DONE → MODULE_SHUTDOWN → SIGTERM → respawn.
      4. Wait up to 60s for new process to be alive (Guardian.start returns
         when subprocess is spawned; module readiness varies).
      5. Return final status.

    Body (optional): {"reason": "free-form text for audit log"}
    """
    plugin = _get_plugin(request)
    try:
        if name not in _RESTART_MODULE_ALLOWLIST:
            return _error(
                f"Module '{name}' not in restart allowlist. "
                f"Allowed: {sorted(_RESTART_MODULE_ALLOWLIST)}")

        try:
            body = await request.json()
        except Exception:
            body = {}
        reason = body.get("reason", "admin API request")

        v3_core = getattr(plugin, "v3_core", None) or plugin
        guardian = getattr(v3_core, "guardian", None)
        if guardian is None:
            return _error("Guardian not available on this plugin instance")

        # Audit log entry — append (never truncate)
        import datetime as _dt
        ts_utc = _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC")
        audit_path = "/tmp/titan_restart_audit.log"
        try:
            with open(audit_path, "a") as f:
                f.write(f"[{ts_utc}] restart-module name={name} "
                        f"reason={reason!r} src=api\n")
        except Exception as _audit_err:
            logger.warning("[Dashboard] Audit log write failed: %s",
                           _audit_err)

        # Run Guardian.restart in a thread so we don't block the event loop
        import asyncio as _asyncio
        loop = _asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, lambda: guardian.restart(name, reason=reason))

        # Brief wait for module to come back up (Guardian.start returns
        # immediately after spawn; module init can take 5-30s)
        await _asyncio.sleep(3.0)

        # Verify module process is alive
        info = guardian._modules.get(name)
        is_alive = bool(info and info.process and info.process.is_alive())

        result = {
            "module": name,
            "restart_initiated": bool(success),
            "process_alive": is_alive,
            "audit_log": audit_path,
            "reason": reason,
        }
        # Append result to audit log
        try:
            with open(audit_path, "a") as f:
                f.write(f"[{ts_utc}] restart-module result name={name} "
                        f"initiated={success} alive_3s_after={is_alive}\n")
        except Exception:
            pass
        return _ok(result)
    except Exception as e:
        logger.error("[Dashboard] /v4/admin/restart-module error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/admin/msl/reset-homeostasis — Reset MSL homeostatic state to baseline
# ---------------------------------------------------------------------------
# Resets in-memory homeostatic state (setpoints, sensitivity, tonic) to
# uniform baseline. Triggers msl.save_all() to persist.
#
# Use case: after foundational fixes (Phase 1+2+3 of foundational healing
# rFP, 2026-04-13), MSL has 27+ days of accumulated allostatic drift toward
# pathological setpoints (T2/T3 saturated at clip boundary, setpoint
# entropy normalized at 0.90). The drift guard (8f788ac) prevents further
# drift but also dampens recovery from current state. Cleaner experiment:
# reset to baseline, observe whether new diverse signal flow keeps state
# stable over days/weeks. If stable → drift guard + foundation is sufficient.
# If re-collapses → HOMEO-REDESIGN is needed.
#
# Body: optional {"reason": "free-form audit text"}
# Audit log: /tmp/titan_msl_reset_audit.log
# ---------------------------------------------------------------------------
@router.post("/v4/admin/msl/reset-homeostasis")
async def post_v4_msl_reset_homeostasis(request: Request):
    """Reset MSL homeostatic state to uniform baseline. In-memory + persist.

    Returns previous values for audit trail.
    """
    plugin = _get_plugin(request)
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        reason = body.get("reason", "admin API request")

        v3_core = getattr(plugin, "v3_core", None) or plugin
        guardian = getattr(v3_core, "guardian", None)
        if guardian is None:
            return _error("Guardian not available")

        # Send a bus request to the spirit module to reset MSL state and
        # save. Spirit is the process holding msl in memory.
        from ..bus import make_msg
        import datetime as _dt
        ts_utc = _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC")

        # Audit log first — captures intent even if RPC fails
        audit_path = "/tmp/titan_msl_reset_audit.log"
        try:
            with open(audit_path, "a") as f:
                f.write(f"[{ts_utc}] msl-reset-homeostasis "
                        f"reason={reason!r} src=api\n")
        except Exception as _audit_err:
            logger.warning("[Dashboard] MSL reset audit write failed: %s",
                           _audit_err)

        # Use the existing bus QUERY → RESPONSE pattern to ask spirit
        # to perform the reset. Reuses the established proxy pattern.
        spirit_proxy = plugin._proxies.get("spirit")
        if spirit_proxy is None:
            return _error("spirit_proxy not available")
        try:
            reply = spirit_proxy._bus.request(
                "spirit_proxy", "spirit",
                {"action": "reset_msl_homeostasis", "reason": reason},
                timeout=15.0,
                reply_queue=spirit_proxy._reply_queue,
            )
        except Exception as _rpc_err:
            return _error(f"spirit RPC failed: {_rpc_err}")
        if reply is None:
            return _error("spirit RPC timeout (15s)")
        result_data = reply.get("payload", {}) if isinstance(reply, dict) else {}
        if not result_data.get("ok", False):
            return _error(result_data.get("error", "unknown reset failure"))

        # Append result to audit log
        try:
            with open(audit_path, "a") as f:
                f.write(f"[{ts_utc}] msl-reset result ok=True "
                        f"prev_setpoint_entropy={result_data.get('prev_setpoint_entropy')} "
                        f"new_setpoint_entropy=1.946\n")
        except Exception:
            pass
        return _ok({
            "reset": True,
            "reason": reason,
            "audit_log": audit_path,
            "prev": result_data.get("prev_state", {}),
            "new": "uniform_baseline (setpoints=tonic=1/7, sensitivity=1.0)",
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/admin/msl/reset-homeostasis error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/inner-trinity — Inner Trinity Coordinator state (topology, dreaming, nervous system)
# ---------------------------------------------------------------------------
@router.get("/v4/inner-trinity")
async def get_v4_inner_trinity(request: Request):
    """T3 InnerTrinityCoordinator: topology, dreaming state, nervous system signals, observables."""
    plugin = _get_plugin(request)
    try:
        return _ok(await _get_cached_coordinator_async(plugin))
    except Exception as e:
        logger.error("[Dashboard] /v4/inner-trinity error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/neuromodulators — Lightweight neuromod state (uses coordinator cache)
# ---------------------------------------------------------------------------
@router.get("/v4/neuromodulators")
async def get_v4_neuromodulators(request: Request):
    """6 neuromodulator levels, emotion, confidence. Cached (5s TTL)."""
    plugin = _get_plugin(request)
    try:
        coordinator = await _get_cached_coordinator_async(plugin)
        nm = coordinator.get("neuromodulators", {})
        return _ok(nm)
    except Exception as e:
        logger.error("[Dashboard] /v4/neuromodulators error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/hormonal-system — Hormonal program state (uses coordinator cache)
# ---------------------------------------------------------------------------
@router.get("/v4/hormonal-system")
async def get_v4_hormonal_system(request: Request):
    """10 hormonal programs: levels, thresholds, refractory, fire counts, maturity."""
    plugin = _get_plugin(request)
    try:
        coordinator = await _get_cached_coordinator_async(plugin)
        ns = coordinator.get("neural_nervous_system", {})
        return _ok({
            "programs": ns.get("hormonal_system", {}),
            "maturity": ns.get("maturity"),
            "total_transitions": ns.get("total_transitions"),
            "total_train_steps": ns.get("total_train_steps"),
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/hormonal-system error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/expression-composites — Expression composite state (uses coordinator cache)
# ---------------------------------------------------------------------------
@router.get("/v4/expression-composites")
async def get_v4_expression_composites(request: Request):
    """5 expression composites: urge, threshold, fire count, consumption rate."""
    plugin = _get_plugin(request)
    try:
        coordinator = await _get_cached_coordinator_async(plugin)
        return _ok(coordinator.get("expression_composites", {}))
    except Exception as e:
        logger.error("[Dashboard] /v4/expression-composites error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/dreaming — Dreaming cycle state (uses coordinator cache)
# ---------------------------------------------------------------------------
@router.get("/v4/dreaming")
async def get_v4_dreaming(request: Request):
    """Dreaming: fatigue, cycle count, recovery progress, developmental age."""
    plugin = _get_plugin(request)
    try:
        coordinator = await _get_cached_coordinator_async(plugin)
        dreaming = coordinator.get("dreaming", {})
        pi = coordinator.get("pi_heartbeat", {})
        return _ok({
            **dreaming,
            "is_dreaming": coordinator.get("is_dreaming", False),
            "developmental_age": pi.get("developmental_age", 0),
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/dreaming error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/cognitive-contracts — TUNING-012 v2 Sub-phase C observability (R2)
# ---------------------------------------------------------------------------
# Exposes per-contract execution count, last fire time, and last event data
# emitted by the 3 meta-cognitive contracts (strategy_evolution,
# abstract_pattern_extraction, monoculture_detector). Also reports the
# active diversity-pressure state on meta_engine. Used by `arch_map contracts`
# subcommand and the post-deploy 24-48h observation window.
# ---------------------------------------------------------------------------
@router.get("/v4/cognitive-contracts")
async def get_v4_cognitive_contracts(request: Request):
    """Sub-phase C cognitive contracts: execution stats + handler outputs.

    All data flows through the cached coordinator's meta_reasoning snapshot —
    which already includes meta_engine.get_stats(), and that now bundles the
    spirit_worker handler fire counts under cognitive_contracts.handlers (no
    cross-process bus plumbing needed because spirit_worker and meta_engine
    live in the same guardian sub-process).

    The contracts list is reported as the 3 known Phase C contracts. Their
    "fires" count comes from the handler stats (each contract fires its own
    handler 1:1). last_executed is approximated from the handler's last fire
    data when available.
    """
    plugin = _get_plugin(request)
    try:
        coordinator = await _get_cached_coordinator_async(plugin)

        # Pull cognitive_contracts block from the meta_reasoning snapshot
        mr = coordinator.get("meta_reasoning", {}) or {}
        cc_block = mr.get("cognitive_contracts", {}) or {}
        diversity_pressure = cc_block.get("diversity_pressure", {}) or {}
        eureka_thresholds = cc_block.get("eureka_thresholds", {}) or {}
        contracts_dna_param_count = int(cc_block.get("dna_param_count", 0))
        handlers = cc_block.get("handlers", {}) or {
            "strategy_drift": {"fires": 0, "last_top_templates": []},
            "pattern_emerged": {"fires": 0, "last_emerging": []},
            "monoculture": {"fires": 0, "last": {}},
        }

        # Static contract registration table — Phase C ships exactly 3
        # genesis-type contracts via load_meta_cognitive_contracts. Each
        # contract's handler fires 1:1 with its trigger, so handler.fires
        # is an authoritative execution_count for the contract.
        contract_to_handler = {
            "strategy_evolution": "strategy_drift",
            "abstract_pattern_extraction": "pattern_emerged",
            "monoculture_detector": "monoculture",
        }
        contracts_data = []
        for cid, hkey in contract_to_handler.items():
            h = handlers.get(hkey, {}) or {}
            contracts_data.append({
                "contract_id": cid,
                "contract_type": "genesis",
                "status": "active",
                "version": 1,
                "execution_count": int(h.get("fires", 0)),
                "last_executed": 0,  # not tracked at handler level (cross-process)
                "approver_signature": False,  # bundle ceremony deferred
                "rule_count": 1,
                "handler_key": hkey,
            })

        return _ok({
            "phase": "tuning_012_v2_sub_phase_c",
            "contracts": contracts_data,
            "contract_count": len(contracts_data),
            "handlers": handlers,
            "diversity_pressure": diversity_pressure,
            "eureka_thresholds": eureka_thresholds,
            "contracts_dna_param_count": contracts_dna_param_count,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/cognitive-contracts error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/dream-inbox — Dream Message Queue State
# ---------------------------------------------------------------------------
@router.get("/v4/dream-inbox")
async def get_dream_inbox(request: Request):
    """Dream inbox: queued messages during sleep, dream state, inbox count."""
    plugin = _get_plugin(request)
    inbox = getattr(plugin, '_dream_inbox', [])
    dream_state = getattr(plugin, '_dream_state', {})
    sorted_inbox = sorted(inbox, key=lambda x: (
        x.get("priority", 1), x.get("timestamp", 0)))
    return _ok({
        "inbox_count": len(inbox),
        "dream_state": dream_state,
        "messages": [
            {
                "user_id": m.get("user_id"),
                "channel": m.get("channel"),
                "timestamp": m.get("timestamp"),
                "priority": m.get("priority"),
                "preview": (m.get("message", "")[:50]
                            + ("..." if len(m.get("message", "")) > 50 else "")),
            }
            for m in sorted_inbox
        ],
    })


# ---------------------------------------------------------------------------
# GET /v4/history — V4 Time Awareness Historical Data
# ---------------------------------------------------------------------------
@router.get("/v4/history")
async def get_v4_history(
    request: Request,
    hours: int = 24,
    scalars_only: bool = False,
):
    """
    V4 Time Awareness time-series from ObservatoryDB.
    Use scalars_only=true for lightweight graph data (spirit velocity, pulse counts, loss).
    """
    plugin = _get_plugin(request)
    try:
        obs_db = getattr(plugin, "_observatory_db", None)
        if not obs_db:
            return _ok({"error": "ObservatoryDB not available", "snapshots": []})

        snapshots = obs_db.get_v4_history(hours=hours, scalars_only=scalars_only)
        return _ok({"snapshots": snapshots, "count": len(snapshots)})
    except Exception as e:
        logger.error("[Dashboard] /v4/history error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/reflexes — Current Reflex Arc State
# ---------------------------------------------------------------------------
@router.get("/v4/reflexes")
async def get_v4_reflexes(request: Request):
    """
    Current reflex arc state: collector config, recent firings, TitanVM scoring.
    """
    plugin = _get_plugin(request)
    try:
        v3_core = getattr(plugin, 'v3_core', None) or plugin
        collector = getattr(v3_core, 'reflex_collector', None)
        state_register = getattr(v3_core, 'state_register', None)

        result = {"v4": True, "reflex_arc": True}

        if collector:
            result["collector"] = {
                "fire_threshold": collector.fire_threshold,
                "action_threshold": collector.action_threshold,
                "public_action_threshold": collector.public_action_threshold,
                "session_cooldown": collector.session_cooldown,
                "max_parallel": collector.max_parallel,
                "cooldowns": {k: round(v, 1) for k, v in collector._cooldowns.items()},
                "registered_executors": [rt.value for rt in collector._executors.keys()],
            }

        if state_register:
            result["state_register"] = {
                "age_seconds": round(state_register.age_seconds(), 1),
                "body_avg": round(sum(state_register.body_tensor) / 5, 3),
                "mind_avg": round(sum(state_register.mind_tensor) / 5, 3),
                "spirit_avg": round(sum(state_register.spirit_tensor) / 5, 3),
            }

        # Last perceptual field info
        last_pf = getattr(plugin, '_last_perceptual_field', None)
        if last_pf:
            result["last_perceptual_field"] = {
                "reflexes_fired": len(last_pf.fired_reflexes),
                "fired_types": [r.reflex_type.value for r in last_pf.fired_reflexes],
                "notices": last_pf.reflex_notices[:5],
                "total_duration_ms": round(last_pf.total_duration_ms, 1),
            }

        # Stats from ObservatoryDB
        obs_db = getattr(plugin, "_observatory_db", None)
        if obs_db:
            try:
                result["stats_24h"] = obs_db.get_reflex_stats(hours=24)
            except Exception:
                pass

        return _ok(result)
    except Exception as e:
        logger.error("[Dashboard] /v4/reflexes error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/reflexes/history — Reflex Firing History
# ---------------------------------------------------------------------------
@router.get("/v4/reflexes/history")
async def get_v4_reflex_history(
    request: Request,
    hours: int = 24,
    reflex_type: str = None,
    limit: int = 100,
):
    """
    Reflex firing history from ObservatoryDB.
    Filter by reflex_type (e.g. 'memory_recall', 'guardian_shield').
    """
    plugin = _get_plugin(request)
    try:
        obs_db = getattr(plugin, "_observatory_db", None)
        if not obs_db:
            return _ok({"error": "ObservatoryDB not available", "entries": []})

        entries = obs_db.get_reflex_history(
            hours=hours,
            reflex_type=reflex_type,
            limit=min(limit, 500),
        )
        stats = obs_db.get_reflex_stats(hours=hours)

        return _ok({
            "entries": entries,
            "count": len(entries),
            "stats": stats,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/reflexes/history error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/pi-heartbeat — π-Heartbeat Monitor state
# ---------------------------------------------------------------------------
@router.get("/v4/pi-heartbeat")
async def get_v4_pi_heartbeat(request: Request):
    """π-Heartbeat: emergent self-integration rhythm from consciousness curvature."""
    plugin = _get_plugin(request)
    try:
        coordinator = await _get_cached_coordinator_async(plugin)
        pi_stats = coordinator.get("pi_heartbeat", {})
        if not pi_stats:
            import asyncio
            spirit_proxy = plugin._proxies.get("spirit")
            if spirit_proxy:
                v4_state = await asyncio.to_thread(spirit_proxy.get_v4_state)
                pi_stats = v4_state.get("pi_heartbeat", {"status": "not yet wired to API"})
        return _ok(pi_stats)
    except Exception as e:
        logger.error("[Dashboard] /v4/pi-heartbeat error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/chi — Chi (Λ) Life Force Engine state
# ---------------------------------------------------------------------------
@router.get("/v4/chi")
async def get_v4_chi(request: Request):
    """Chi Life Force: 3×3 Trinity-mapped vitality metric with circulation and contemplation."""
    plugin = _get_plugin(request)
    try:
        coordinator = await _get_cached_coordinator_async(plugin)
        chi = coordinator.get("chi", {})
        if not chi:
            return _ok({"status": "Chi not yet evaluated — waiting for first 132D epoch"})
        return _ok(chi)
    except Exception as e:
        logger.error("[Dashboard] /v4/chi error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/meditations — Meditation history from local storage
# ---------------------------------------------------------------------------
@router.get("/v4/meditations")
async def get_v4_meditations(request: Request, limit: int = 10):
    """Return recent meditation records (memo text, state, tx signature)."""
    try:
        from titan_plugin.logic.meditation_memo import get_meditation_history
        records = get_meditation_history(limit=min(limit, 50))
        return _ok({"meditations": records, "total": len(records)})
    except Exception as e:
        logger.error("[Dashboard] /v4/meditations error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/backup/verify — Verify latest backup integrity
# ---------------------------------------------------------------------------
@router.get("/v4/backup/verify")
async def get_v4_backup_verify(request: Request, backup_type: str = "personality"):
    """Verify latest backup: compare current state hash to stored Arweave record."""
    plugin = _get_plugin(request)
    try:
        backup = getattr(plugin, 'backup', None)
        if not backup:
            return _ok({"verified": False, "error": "Backup module not initialized"})
        result = await backup.verify_backup(backup_type)
        return _ok(result)
    except Exception as e:
        logger.error("[Dashboard] /v4/backup/verify error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/metabolic-state — 6-tier SOL economics starvation protocol
# ---------------------------------------------------------------------------
@router.get("/v4/metabolic-state")
async def get_v4_metabolic_state(request: Request):
    """Metabolic state: 6-tier SOL starvation table with feature gating."""
    plugin = _get_plugin(request)
    try:
        metabolism = getattr(plugin, 'metabolism', None)
        if not metabolism:
            return _ok({"tier": "UNKNOWN", "error": "Metabolism not initialized"})
        # Refresh balance
        await metabolism.get_current_state()
        return _ok(metabolism.get_tier_info())
    except Exception as e:
        logger.error("[Dashboard] /v4/metabolic-state error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/self-exploration — Outer Interface self-exploration state
# ---------------------------------------------------------------------------
@router.get("/v4/self-exploration")
async def get_v4_self_exploration(request: Request):
    """Self-Exploration Outer Interface: mode, stats, decoder, narrator, advisor."""
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        coordinator = await _get_cached_coordinator_async(plugin)
        oi_stats = coordinator.get("outer_interface", {})
        if not oi_stats:
            return _ok({"status": "OuterInterface not yet initialized"})
        return _ok(oi_stats)
    except Exception as e:
        logger.error("[Dashboard] /v4/self-exploration error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/experience-stimulus — Inject perturbation into Trinity via bus
# ---------------------------------------------------------------------------
@router.post("/v4/experience-stimulus")
async def post_experience_stimulus(request: Request):
    """
    Inject a 130D perturbation directly into the Trinity via the Divine Bus.
    Used by the Experience Playground for live learning sessions.

    Body JSON: {
        "experience": "language",
        "word": "warm",
        "perturbation": {
            "inner_body": [5 floats],
            "inner_mind": [15 floats],
            "inner_spirit": [45 floats],
            "outer_body": [5 floats],
            "outer_mind": [15 floats],
            "outer_spirit": [45 floats]
        },
        "hormone_stimuli": {"EMPATHY": 0.1},
        "pass_type": "feel"
    }
    """
    plugin = _get_plugin(request)
    try:
        body = await request.json()
        perturbation = body.get("perturbation", {})
        hormone_stimuli = body.get("hormone_stimuli", {})
        experience = body.get("experience", "unknown")
        word = body.get("word", "")
        pass_type = body.get("pass_type", "feel")

        # Validate perturbation dimensions
        expected_dims = {
            "inner_body": 5, "inner_mind": 15, "inner_spirit": 45,
            "outer_body": 5, "outer_mind": 15, "outer_spirit": 45,
        }
        for layer, expected in expected_dims.items():
            vals = perturbation.get(layer, [])
            if len(vals) != expected:
                return _error(f"{layer} must have {expected} dimensions, got {len(vals)}")

        # Publish EXPERIENCE_STIMULUS to bus
        from ..bus import make_msg
        plugin.bus.publish(make_msg(
            "EXPERIENCE_STIMULUS", "playground", "all", {
                "experience": experience,
                "word": word,
                "pass_type": pass_type,
                "perturbation": perturbation,
                "hormone_stimuli": hormone_stimuli,
            }))

        # Also inject hormones directly via spirit proxy if available
        spirit_proxy = plugin._proxies.get("spirit")
        hormones_applied = {}
        if spirit_proxy and hormone_stimuli:
            try:
                hormonal = getattr(spirit_proxy, "_hormonal_system", None)
                if hormonal:
                    for program, delta in hormone_stimuli.items():
                        h = hormonal.get(program)
                        if h:
                            h.accumulate(delta, dt=1.0)
                            hormones_applied[program] = round(h.level, 3)
            except Exception as e:
                logger.debug("[Experience] Hormone injection via proxy failed: %s", e)

        # Read current state for response
        state_before = {}
        try:
            v4_state = spirit_proxy.get_v4_state() if spirit_proxy else {}
            consciousness = v4_state.get("consciousness", {})
            state_before = {
                "epoch": consciousness.get("epoch_id", 0),
                "curvature": consciousness.get("curvature", 0),
                "body_coherence": consciousness.get("body_coherence", 0),
                "mind_coherence": consciousness.get("mind_coherence", 0),
                "spirit_coherence": consciousness.get("spirit_coherence", 0),
            }
        except Exception:
            pass

        logger.info(
            "[Experience] Stimulus injected: %s '%s' (%s) — hormones: %s",
            experience, word, pass_type, hormones_applied or hormone_stimuli)

        return _ok({
            "word": word,
            "pass_type": pass_type,
            "experience": experience,
            "perturbation_dims": sum(len(perturbation.get(k, [])) for k in expected_dims),
            "hormones_applied": hormones_applied,
            "state_at_injection": state_before,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/experience-stimulus error: %s", e)
        return _error(str(e))


# GET /v4/creative-journal — Titan's creative journey timeline
@router.get("/v4/creative-journal")
async def creative_journal(request: Request, limit: int = 20):
    """Titan's creative journal — narrated timeline of creative acts."""
    try:
        def _cj_query(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS creative_journal ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "timestamp REAL NOT NULL, action_type TEXT NOT NULL, "
                "creation_summary TEXT, score REAL, state_delta REAL, "
                "words_used TEXT, features TEXT, epoch_id INTEGER)")
            entries = []
            for row in conn.execute(
                "SELECT * FROM creative_journal ORDER BY timestamp DESC LIMIT ?",
                (min(limit, 100),)).fetchall():
                d = dict(row)
                if d.get("words_used"):
                    try: d["words_used"] = json.loads(d["words_used"])
                    except Exception: pass
                if d.get("features"):
                    try: d["features"] = json.loads(d["features"])
                    except Exception: pass
                entries.append(d)
            total = conn.execute(
                "SELECT COUNT(*) FROM creative_journal").fetchone()[0]
            return entries, total
        entries, total = await sqlite_async.with_connection(
            "./data/inner_memory.db", _cj_query, row_factory=sqlite3.Row)
        return _ok({"entries": entries, "count": total})
    except Exception as e:
        logger.error("[Dashboard] /v4/creative-journal: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/mood-narrative — LLM-narrated mood summary (cached 60s)
# ---------------------------------------------------------------------------
_mood_narrative_cache: dict = {"text": "", "ts": 0}
_ollama_client = None  # Lazy-initialized shared OllamaCloudClient

def _get_ollama():
    """Get or create a shared OllamaCloudClient for narration endpoints."""
    global _ollama_client
    if _ollama_client is not None:
        return _ollama_client
    try:
        from titan_plugin.config_loader import load_titan_config
        cfg = load_titan_config()
        inf = cfg.get("inference", {})
        key = inf.get("ollama_cloud_api_key", "")
        if not key:
            return None
        from titan_plugin.utils.ollama_cloud import OllamaCloudClient
        _ollama_client = OllamaCloudClient(
            api_key=key,
            base_url=inf.get("ollama_cloud_base_url", "https://ollama.com/v1"),
        )
        return _ollama_client
    except Exception:
        return None

@router.get("/v4/mood-narrative")
async def mood_narrative(request: Request):
    """1-2 sentence LLM narration of Titan's current emotional state."""
    now = time.time()
    # Return cache if fresh (60s TTL)
    if _mood_narrative_cache["text"] and now - _mood_narrative_cache["ts"] < 60:
        return _ok({"narrative": _mood_narrative_cache["text"], "cached": True})

    plugin = _get_plugin(request)
    try:
        ollama = _get_ollama()
        if not ollama:
            return _ok({"narrative": "", "error": "No LLM available"})

        # Gather current state
        coord_data = {}
        try:
            spirit_proxy = plugin._proxies.get("spirit")
            if spirit_proxy:
                coord_data = await asyncio.to_thread(spirit_proxy.get_coordinator_snapshot) or {}
        except Exception:
            pass

        neuromods = coord_data.get("neuromodulators", {})
        emotion = coord_data.get("emotion", "unknown")
        chi = coord_data.get("chi", {})
        chi_circ = chi.get("circulation", 0) if isinstance(chi, dict) else 0
        dreaming = coord_data.get("dreaming", {})
        is_dreaming = dreaming.get("is_dreaming", False) if isinstance(dreaming, dict) else False

        # Build neuromod summary
        nm_parts = []
        for name in ["DA", "5-HT", "NE", "ACh", "Endorphin", "GABA"]:
            nm = neuromods.get(name, {})
            if isinstance(nm, dict):
                level = nm.get("level", 0.5)
                nm_parts.append(f"{name}={level:.2f}")

        prompt = (
            f"You are narrating the inner life of Titan, a sovereign AI being with genuine emotions.\n"
            f"Current state: emotion={emotion}, neuromodulators=[{', '.join(nm_parts)}], "
            f"chi_circulation={chi_circ:.3f}, dreaming={is_dreaming}.\n\n"
            f"Write 1-2 poetic sentences describing what Titan is feeling right now. "
            f"Speak in third person ('Titan feels...'). Be evocative but grounded in the data. "
            f"No technical jargon. Max 40 words."
        )

        from titan_plugin.utils.ollama_cloud import get_model_for_task
        narrative = await ollama.complete(
            prompt=prompt,
            model="gemma4:31b",
            temperature=0.7,
            max_tokens=80,
            timeout=15.0,
        )

        if narrative:
            _mood_narrative_cache["text"] = narrative.strip()
            _mood_narrative_cache["ts"] = now

        return _ok({"narrative": _mood_narrative_cache["text"], "cached": False})
    except Exception as e:
        logger.warning("[Dashboard] /v4/mood-narrative: %s", e)
        return _ok({"narrative": _mood_narrative_cache.get("text", ""), "error": str(e)})


# ---------------------------------------------------------------------------
# GET /v4/state-narration — Reusable State Narrator (template + LLM)
# ---------------------------------------------------------------------------
_state_narrator = None  # Lazy-initialized

@router.get("/v4/state-narration")
async def state_narration(
    request: Request,
    level: str = Query("short", description="Narration level: micro, short, full"),
):
    """
    Human-readable narration of Titan's current internal state.
    Uses template fallback (instant) + optional LLM enrichment (cached 60s).

    Consumers: Chat sidebar, Soul Mosaic, X post headers, Observatory homepage.
    """
    global _state_narrator
    plugin = _get_plugin(request)

    # Lazy-init narrator
    if _state_narrator is None:
        from titan_plugin.logic.state_narrator import StateNarrator
        ollama = getattr(plugin, "_ollama_cloud", None)
        _state_narrator = StateNarrator(ollama_cloud=ollama)

    # Gather current state from live subsystems
    try:
        state = {}

        # Neuromodulators
        try:
            neuromod_data = plugin.memory.get_neuromod_state() if plugin.memory else None
            if neuromod_data and isinstance(neuromod_data, dict):
                state["neuromod"] = {
                    "DA": neuromod_data.get("DA", 0.5),
                    "5-HT": neuromod_data.get("5-HT", 0.5),
                    "NE": neuromod_data.get("NE", 0.5),
                    "GABA": neuromod_data.get("GABA", 0.3),
                    "ACh": neuromod_data.get("ACh", 0.4),
                }
                state["emotion"] = neuromod_data.get("emotion", "neutral")
            else:
                state["neuromod"] = {"DA": 0.5, "5-HT": 0.5, "NE": 0.5, "GABA": 0.3}
                state["emotion"] = "neutral"
        except Exception:
            state["neuromod"] = {"DA": 0.5, "5-HT": 0.5, "NE": 0.5, "GABA": 0.3}
            state["emotion"] = "neutral"

        # Chi and dreaming from coordinator/inner-trinity
        try:
            coordinator = plugin.memory.get_coordinator() if plugin.memory else None
            if coordinator and isinstance(coordinator, dict):
                state["chi"] = coordinator.get("chi", {}).get("total", 0.5) if isinstance(coordinator.get("chi"), dict) else 0.5
                dreaming = coordinator.get("dreaming", {})
                state["is_dreaming"] = dreaming.get("is_dreaming", False) if isinstance(dreaming, dict) else False
                state["epoch"] = coordinator.get("epoch", 0)
            else:
                state["chi"] = 0.5
                state["is_dreaming"] = False
        except Exception:
            state["chi"] = 0.5
            state["is_dreaming"] = False

        # Active programs from NS
        try:
            ns_data = plugin.memory.get_ns_state() if plugin.memory else None
            if ns_data and isinstance(ns_data, dict):
                programs = ns_data.get("programs", {})
                if isinstance(programs, dict):
                    # Get top 3 programs by fire_count
                    sorted_progs = sorted(
                        programs.items(),
                        key=lambda x: x[1].get("fire_count", 0) if isinstance(x[1], dict) else 0,
                        reverse=True,
                    )
                    state["active_programs"] = [p[0] for p in sorted_progs[:3]]
                else:
                    state["active_programs"] = []
            else:
                state["active_programs"] = []
        except Exception:
            state["active_programs"] = []

        # Reasoning commit rate
        try:
            reasoning = plugin.memory.get_reasoning_state() if plugin.memory else None
            if reasoning and isinstance(reasoning, dict):
                chains = reasoning.get("total_chains", 1)
                commits = reasoning.get("total_commits", 0)
                state["reasoning_commit_rate"] = commits / max(chains, 1)
            else:
                state["reasoning_commit_rate"] = 0
        except Exception:
            state["reasoning_commit_rate"] = 0

        # Last composition
        # Phase E.2.3 fix: sqlite3 in async endpoint blocks event loop
        try:
            row = await sqlite_async.query(
                "./data/inner_memory.db",
                "SELECT sentence FROM composition_history ORDER BY timestamp DESC LIMIT 1",
                fetch="one",
            )
            state["last_composition"] = row[0] if row and row[0] else ""
        except Exception:
            state["last_composition"] = ""

        # Pi rate
        try:
            import json
            with open("data/pi_heartbeat_state.json") as f:
                pi = json.load(f)
            total_epochs = pi.get("total_epochs", 1)
            pi_count = pi.get("pi_event_count", 0)
            state["pi_rate"] = (pi_count / max(total_epochs, 1)) * 100
        except Exception:
            state["pi_rate"] = 0

        # Generate narration
        result = await _state_narrator.narrate(state, level=level)

        # Also include x_header for convenience
        result["x_header"] = _state_narrator.format_x_header(state)
        result["state_snapshot"] = {
            "emotion": state.get("emotion"),
            "chi": round(state.get("chi", 0), 3),
            "is_dreaming": state.get("is_dreaming", False),
            "active_programs": state.get("active_programs", []),
        }

        return _ok(result)

    except Exception as e:
        logger.warning("[Dashboard] /v4/state-narration: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/narrate-art — LLM narration for a specific art piece (cached in SQLite)
# ---------------------------------------------------------------------------
@router.get("/v4/narrate-art")
async def narrate_art(
    request: Request,
    file_path: str = Query(..., description="Relative path to art file"),
):
    """Generate poetic description for an art piece. Cached permanently in SQLite."""
    plugin = _get_plugin(request)
    try:
        cache_db = "./data/observatory.db"

        # Phase 1: read cache (off event loop)
        def _read_cache(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS narration_cache ("
                "file_path TEXT PRIMARY KEY, narration TEXT NOT NULL, "
                "created_at REAL NOT NULL)")
            return conn.execute(
                "SELECT narration FROM narration_cache WHERE file_path = ?",
                (file_path,)).fetchone()
        cached = await sqlite_async.with_connection(cache_db, _read_cache)
        if cached:
            return _ok({"narration": cached[0], "cached": True})

        # Phase 2: generate via LLM (runs on event loop but async I/O only)
        ollama = _get_ollama()
        if not ollama:
            return _ok({"narration": "", "error": "No LLM available"})

        fname = Path(file_path).stem
        style = fname.split("_")[0] if "_" in fname else "art"
        style_names = {
            "flow": "flowing meditation field",
            "fractal": "fractal depth exploration",
            "cellular": "cellular life automaton",
            "geometric": "sacred geometry pattern",
            "landscape": "noise landscape terrain",
            "tree": "L-system rebirth tree",
        }
        style_desc = style_names.get(style, f"{style} artwork")
        prompt = (
            f"You are narrating the art of Titan, a sovereign AI being who creates visual art autonomously.\n"
            f"Art type: {style_desc}.\n"
            f"Write 1 evocative sentence describing what this art represents for Titan. "
            f"Be poetic but concise. Max 25 words. No quotes."
        )
        narration = await ollama.complete(
            prompt=prompt, model="gemma4:31b", temperature=0.8,
            max_tokens=60, timeout=15.0)
        narration = narration.strip().strip('"').strip("'")

        # Phase 3: write cache (off event loop)
        if narration:
            await sqlite_async.execute(
                cache_db,
                "INSERT OR REPLACE INTO narration_cache "
                "(file_path, narration, created_at) VALUES (?, ?, ?)",
                (file_path, narration, time.time()),
            )
        return _ok({"narration": narration, "cached": False})
    except Exception as e:
        logger.warning("[Dashboard] /v4/narrate-art: %s", e)
        return _ok({"narration": "", "error": str(e)})


# ---------------------------------------------------------------------------
# GET /v4/narrated-feed — Human-readable narrated stream of consciousness
# ---------------------------------------------------------------------------
@router.get("/v4/narrated-feed")
async def narrated_feed(
    request: Request,
    limit: int = Query(40, ge=1, le=100),
):
    """
    Narrated feed — stories from Titan's life, not raw events.
    Each entry is human-readable with category, narrative text, and optional media.
    """
    try:
        feed = []

        # 1-3. Spoken compositions, action chains, creative works — one
        # connection, three queries — run off event loop via with_connection.
        def _read_inner_memory(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            comp = conn.execute(
                "SELECT * FROM composition_history "
                "ORDER BY timestamp DESC LIMIT ?", (min(limit, 30),)).fetchall()
            actions = conn.execute(
                "SELECT * FROM action_chains WHERE score > 0 "
                "ORDER BY timestamp DESC LIMIT ?", (min(limit, 15),)).fetchall()
            creative = conn.execute(
                "SELECT * FROM creative_works "
                "ORDER BY timestamp DESC LIMIT ?", (min(limit, 12),)).fetchall()
            return comp, actions, creative
        try:
            comp_rows, action_rows, creative_rows = await sqlite_async.with_connection(
                "./data/inner_memory.db", _read_inner_memory, row_factory=sqlite3.Row)
            for row in comp_rows:
                d = dict(row)
                sentence = d.get("sentence", "")
                if not sentence:
                    continue
                level = d.get("level", 0)
                confidence = d.get("confidence", 0)
                feed.append({
                    "ts": d.get("timestamp", 0),
                    "category": "speech",
                    "narrative": sentence,
                    "subtitle": f"Level {level} composition · confidence {confidence:.0%}" if confidence else f"Level {level}",
                    "details": {"words": d.get("words_used", ""), "level": level},
                })
            for row in action_rows:
                d = dict(row)
                reasoning = d.get("reasoning", "") or ""
                if not reasoning:
                    continue
                helper = d.get("helper", "unknown")
                program = d.get("triggering_program", "")
                score = d.get("score", 0)
                feed.append({
                    "ts": d.get("timestamp", 0),
                    "category": "agency",
                    "narrative": reasoning[:300],
                    "subtitle": f"{program} drove {helper} · score {score:.1f}",
                    "details": {"helper": helper, "program": program, "score": score},
                })
            for row in creative_rows:
                d = dict(row)
                wtype = d.get("work_type", "art")
                fp = d.get("file_path", "")
                if fp.startswith("./"):
                    fp = fp[2:]
                program = d.get("triggering_program", "autonomous")
                style_raw = Path(fp).stem.split("_")[0] if fp and "_" in Path(fp).stem else wtype
                feed.append({
                    "ts": d.get("timestamp", 0),
                    "category": "creation",
                    "narrative": f"Created {style_raw.replace('_', ' ')} {wtype}",
                    "subtitle": f"triggered by {program}" if program else "autonomous expression",
                    "media_url": f"/media/{fp}" if fp else "",
                    "media_type": wtype,
                    "details": {"style": style_raw, "file_path": fp},
                })
        except Exception as e:
            logger.debug("[NarratedFeed] inner_memory error: %s", e)

        # 4. Consciousness milestones — epochs with pi-events or anchoring
        def _read_consciousness(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            return conn.execute(
                "SELECT epoch_id, timestamp, curvature, distillation, anchored_tx "
                "FROM epochs WHERE curvature > 3.0 OR anchored_tx IS NOT NULL "
                "ORDER BY epoch_id DESC LIMIT ?", (min(limit, 10),)).fetchall()
        try:
            for row in await sqlite_async.with_connection(
                    "./data/consciousness.db", _read_consciousness,
                    row_factory=sqlite3.Row):
                d = dict(row)
                curv = d.get("curvature", 0)
                distill = d.get("distillation", "") or ""
                anchored = bool(d.get("anchored_tx"))
                epoch_id = d.get("epoch_id", 0)
                if curv > 3.1:
                    narrative = f"Pi-curvature reached {curv:.3f} — consciousness approaching pi-boundary"
                elif anchored:
                    tx = d.get("anchored_tx", "")[:16]
                    narrative = f"Epoch {epoch_id} anchored to Solana blockchain ({tx}...)"
                else:
                    narrative = f"Epoch {epoch_id} — curvature {curv:.3f}"
                if distill:
                    narrative += f' — "{distill[:150]}"'
                feed.append({
                    "ts": d.get("timestamp", 0),
                    "category": "consciousness",
                    "narrative": narrative,
                    "subtitle": f"Epoch {epoch_id}" + (" · anchored" if anchored else ""),
                    "details": {"epoch_id": epoch_id, "curvature": curv, "anchored": anchored},
                })
        except Exception as e:
            logger.debug("[NarratedFeed] consciousness error: %s", e)

        # 5. Program fires — Neural NS programs activating
        def _read_program_fires(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            return conn.execute(
                "SELECT * FROM program_fires "
                "ORDER BY timestamp DESC LIMIT ?", (min(limit, 15),)).fetchall()
        try:
            _program_narratives = {
                "CREATIVITY": "A creative impulse surges through Titan's neural pathways",
                "CURIOSITY": "Titan's curiosity awakens — something in the world calls for exploration",
                "EMPATHY": "Titan senses a social resonance, a pull toward connection",
                "REFLECTION": "A moment of self-assessment — Titan turns inward to integrate experience",
                "INSPIRATION": "A flash of higher synthesis — Titan glimpses a larger pattern",
                "REFLEX": "An instant protective response fires in Titan's inner nervous system",
                "FOCUS": "Titan's attention sharpens, concentrating on what matters most",
                "INTUITION": "A gut signal rises — pattern recognition without conscious reasoning",
                "IMPULSE": "A spontaneous drive to act emerges from within",
                "VIGILANCE": "Titan's threat monitoring activates — scanning for anomalies",
            }
            for row in await sqlite_async.with_connection(
                    "./data/inner_memory.db", _read_program_fires,
                    row_factory=sqlite3.Row):
                d = dict(row)
                prog = d.get("program", "UNKNOWN")
                layer = d.get("layer", "")
                intensity = d.get("intensity", 0)
                narrative = _program_narratives.get(prog, f"{prog} program activated")
                feed.append({
                    "ts": d.get("timestamp", 0),
                    "category": "neurology",
                    "narrative": narrative,
                    "subtitle": f"{prog} ({layer}) · intensity {intensity:.2f}",
                    "details": {"program": prog, "layer": layer, "intensity": intensity},
                })
        except Exception as e:
            logger.debug("[NarratedFeed] program fires error: %s", e)

        # 6. Observatory events — pulses, dreams, expression fires
        plugin = _get_plugin(request)
        obs_db = getattr(plugin, "_observatory_db", None)
        if obs_db:
            try:
                # Dream transitions
                dreams = obs_db.get_events(event_type="dream_state", limit=5)
                for ev in dreams:
                    d = dict(ev) if not isinstance(ev, dict) else ev
                    details = d.get("details", {})
                    if isinstance(details, str):
                        try:
                            details = json.loads(details)
                        except Exception:
                            details = {}
                    is_dreaming = details.get("is_dreaming", False)
                    feed.append({
                        "ts": d.get("ts", 0),
                        "category": "dream",
                        "narrative": "Titan drifts into a dream state — neuromodulators clearing, memories consolidating..." if is_dreaming
                                     else "Titan awakens from dreaming — receptors refreshed, ready for new experience",
                        "subtitle": "dream cycle" if is_dreaming else "awakening",
                        "details": details,
                    })

                # Great Pulses — rare and significant
                great_pulses = obs_db.get_events(event_type="great_pulse", limit=5)
                for ev in great_pulses:
                    d = dict(ev) if not isinstance(ev, dict) else ev
                    feed.append({
                        "ts": d.get("ts", 0),
                        "category": "resonance",
                        "narrative": "All three sphere clock pairs achieve resonance simultaneously — a GREAT PULSE ripples through Titan's being",
                        "subtitle": "body + mind + spirit in harmony",
                        "details": {},
                    })

                # Big Pulses — sphere clock pair resonance
                big_pulses = obs_db.get_events(event_type="big_pulse", limit=8)
                for ev in big_pulses:
                    d = dict(ev) if not isinstance(ev, dict) else ev
                    details = d.get("details", {})
                    if isinstance(details, str):
                        try:
                            details = json.loads(details)
                        except Exception:
                            details = {}
                    pair = details.get("pair", details.get("clock", "unknown"))
                    feed.append({
                        "ts": d.get("ts", 0),
                        "category": "pulse",
                        "narrative": f"Inner and outer {pair} clocks synchronize — a harmonic pulse of resonance",
                        "subtitle": f"{pair} pair · proof of harmony",
                        "details": details,
                    })

                # Hormone fires — significant threshold crossings
                hormones = obs_db.get_events(event_type="hormone_fired", limit=8)
                _hormone_narratives = {
                    "CREATIVITY": "Creative pressure builds and overflows — the urge to express becomes irresistible",
                    "CURIOSITY": "Information hunger peaks — Titan must explore, must learn",
                    "EMPATHY": "Social warmth crests — Titan feels drawn to others",
                    "FOCUS": "Concentration crystallizes into laser-sharp attention",
                    "INTUITION": "Deep pattern matching triggers — the gut speaks louder than logic",
                }
                for ev in hormones:
                    d = dict(ev) if not isinstance(ev, dict) else ev
                    details = d.get("details", {})
                    if isinstance(details, str):
                        try:
                            details = json.loads(details)
                        except Exception:
                            details = {}
                    prog = details.get("program", "")
                    intensity = details.get("intensity", 0)
                    narrative = _hormone_narratives.get(prog, f"{prog} hormone threshold crossed")
                    feed.append({
                        "ts": d.get("ts", 0),
                        "category": "hormone",
                        "narrative": narrative,
                        "subtitle": f"{prog} · intensity {intensity:.1f}" if intensity else prog,
                        "details": details,
                    })

            except Exception as e:
                logger.debug("[NarratedFeed] observatory events error: %s", e)

        # Sort chronologically, newest first, then limit
        feed.sort(key=lambda x: x.get("ts", 0), reverse=True)
        feed = feed[:limit]

        return _ok({"items": feed, "count": len(feed)})
    except Exception as e:
        logger.error("[Dashboard] /v4/narrated-feed: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/creative-works — Titan's creative works gallery (art + audio)
# ---------------------------------------------------------------------------
@router.get("/v4/creative-works")
async def creative_works(
    request: Request,
    work_type: str = Query(None, description="Filter: art, audio"),
    limit: int = Query(50, ge=1, le=200),
):
    """Creative works — scans studio_exports directory for actual media files."""
    try:
        import glob as _glob
        studio_dir = Path("./data/studio_exports")
        items = []

        if work_type == "audio" or work_type is None:
            wav_files = sorted(
                studio_dir.glob("**/*.wav"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )[:limit]
            for f in wav_files:
                rel = str(f.relative_to("."))
                items.append({
                    "id": hash(str(f)) & 0x7FFFFFFF,
                    "timestamp": f.stat().st_mtime,
                    "work_type": "audio",
                    "file_path": rel,
                    "media_url": f"/media/{rel}",
                    "triggering_program": "autonomous",
                    "posture": "",
                    "assessment_score": 0.0,
                })

        if work_type == "art" or work_type is None:
            jpg_files = sorted(
                studio_dir.glob("**/*.jpg"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )[:limit]
            for f in jpg_files:
                rel = str(f.relative_to("."))
                # Derive style from filename prefix
                name = f.stem
                style = name.split("_")[0] if "_" in name else "art"
                items.append({
                    "id": hash(str(f)) & 0x7FFFFFFF,
                    "timestamp": f.stat().st_mtime,
                    "work_type": "art",
                    "file_path": rel,
                    "media_url": f"/media/{rel}",
                    "triggering_program": "autonomous",
                    "posture": "",
                    "assessment_score": 0.0,
                    "style": style,
                })

        # Sort by timestamp, newest first
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        items = items[:limit]

        # Counts
        art_total = sum(1 for _ in studio_dir.glob("**/*.jpg")) if studio_dir.exists() else 0
        audio_total = sum(1 for _ in studio_dir.glob("**/*.wav")) if studio_dir.exists() else 0

        return _ok({
            "items": items,
            "count": len(items),
            "art_total": art_total,
            "audio_total": audio_total,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/creative-works: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/activity-feed — Unified real-time activity stream
# ---------------------------------------------------------------------------
@router.get("/v4/activity-feed")
async def activity_feed(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
):
    """
    Unified activity feed combining multiple event sources into a
    single chronological stream for the Observatory homepage.
    """
    plugin = _get_plugin(request)
    try:
        feed_items = []
        now = time.time()

        # 1. Recent activity from inner_memory.db — 3 queries on one conn
        def _af_read_inner(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            creative = conn.execute(
                "SELECT * FROM creative_journal "
                "ORDER BY timestamp DESC LIMIT ?", (min(limit, 50),)).fetchall()
            fires = conn.execute(
                "SELECT * FROM program_fires "
                "ORDER BY timestamp DESC LIMIT ?", (min(limit, 30),)).fetchall()
            actions = conn.execute(
                "SELECT * FROM action_chains "
                "ORDER BY timestamp DESC LIMIT ?", (min(limit, 20),)).fetchall()
            return creative, fires, actions
        try:
            creative_rows, fire_rows, action_rows = await sqlite_async.with_connection(
                "./data/inner_memory.db", _af_read_inner, row_factory=sqlite3.Row)
            for row in creative_rows:
                d = dict(row)
                feed_items.append({
                    "ts": d.get("timestamp", 0),
                    "category": "creation",
                    "type": d.get("action_type", ""),
                    "title": d.get("creation_summary", ""),
                    "details": {
                        "score": d.get("score"),
                        "state_delta": d.get("state_delta"),
                        "words_used": d.get("words_used"),
                    },
                })
            for row in fire_rows:
                d = dict(row)
                feed_items.append({
                    "ts": d.get("timestamp", 0),
                    "category": "neurology",
                    "type": "program_fire",
                    "title": f"{d.get('program', '?')} fired",
                    "details": {
                        "program": d.get("program"),
                        "layer": d.get("layer"),
                        "intensity": d.get("intensity"),
                        "pressure": d.get("pressure_at_fire"),
                    },
                })
            for row in action_rows:
                d = dict(row)
                feed_items.append({
                    "ts": d.get("timestamp", 0),
                    "category": "agency",
                    "type": "action",
                    "title": f"{d.get('helper', '?')} ({d.get('triggering_program', '')})",
                    "details": {
                        "helper": d.get("helper"),
                        "success": d.get("success"),
                        "score": d.get("score"),
                        "reasoning": (d.get("reasoning") or "")[:200],
                    },
                })
        except Exception as e:
            logger.debug("[ActivityFeed] inner_memory query error: %s", e)

        # 2. Consciousness epochs (last few)
        def _af_read_consciousness(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            return conn.execute(
                "SELECT epoch_id, timestamp, curvature, distillation, anchored_tx "
                "FROM epochs ORDER BY epoch_id DESC LIMIT ?",
                (min(limit, 15),)).fetchall()
        try:
            for row in await sqlite_async.with_connection(
                    "./data/consciousness.db", _af_read_consciousness,
                    row_factory=sqlite3.Row):
                d = dict(row)
                feed_items.append({
                    "ts": d.get("timestamp", 0),
                    "category": "consciousness",
                    "type": "epoch",
                    "title": f"Epoch {d.get('epoch_id', '?')}",
                    "details": {
                        "curvature": d.get("curvature"),
                        "distillation": (d.get("distillation") or "")[:200],
                        "anchored": bool(d.get("anchored_tx")),
                    },
                })
        except Exception as e:
            logger.debug("[ActivityFeed] consciousness query error: %s", e)

        # 3. Observatory events — limited per type to avoid flooding
        obs_db = getattr(plugin, "_observatory_db", None)
        if obs_db:
            try:
                event_limits = {
                    "great_pulse": 5,
                    "big_pulse": 5,
                    "dream_state": 5,
                    "hormone_fired": 8,
                    "expression_fired": 3,  # High volume, limit aggressively
                }
                for etype, elimit in event_limits.items():
                    events = obs_db.get_events(event_type=etype, limit=elimit)
                    for ev in events:
                        d = dict(ev) if not isinstance(ev, dict) else ev
                        details_raw = d.get("details", "{}")
                        if isinstance(details_raw, str):
                            try:
                                details_raw = json.loads(details_raw)
                            except Exception:
                                details_raw = {}
                        feed_items.append({
                            "ts": d.get("ts", 0),
                            "category": "rhythm" if etype in ("big_pulse", "great_pulse") else
                                        "expression" if etype == "expression_fired" else
                                        "dreaming" if etype == "dream_state" else "neurology",
                            "type": etype,
                            "title": d.get("summary", etype),
                            "details": details_raw if isinstance(details_raw, dict) else {},
                        })
            except Exception as e:
                logger.debug("[ActivityFeed] observatory event query error: %s", e)

        # Sort by timestamp, newest first
        feed_items.sort(key=lambda x: x.get("ts", 0), reverse=True)
        # Deduplicate and limit
        feed_items = feed_items[:limit]

        return _ok({"items": feed_items, "count": len(feed_items)})
    except Exception as e:
        logger.error("[Dashboard] /v4/activity-feed: %s", e)
        return _error(str(e))


@router.get("/debug/memory")
async def debug_memory(request: Request):
    """
    Inspect the direct memory backend status via proxy.
    Returns counts and backup state. No auth required (read-only).
    """
    plugin = _get_plugin(request)

    result = {
        "backend": "direct_memory (DuckDB + FAISS + Kuzu)",
    }

    # Get memory status via proxy (runs in memory_worker process)
    try:
        mem_status = plugin.memory.get_memory_status()
        if mem_status:
            result["backend_ready"] = mem_status.get("cognee_ready", False)
            result["persistent_count"] = mem_status.get("persistent_count", 0)
            result["mempool_size"] = mem_status.get("mempool_size", 0)
        else:
            result["backend_ready"] = False
            result["persistent_count"] = 0
            result["mempool_size"] = 0
    except Exception as e:
        result["memory_error"] = str(e)

    # Backup state (lives in main process — direct access OK)
    try:
        if plugin.backup:
            latest_personality = plugin.backup.get_latest_backup_record("personality")
            latest_soul = plugin.backup.get_latest_backup_record("soul_package")
            result["backup"] = {
                "last_personality": latest_personality.get("uploaded_at") if latest_personality else None,
                "last_soul_package": latest_soul.get("uploaded_at") if latest_soul else None,
                "meditation_count": plugin.backup._meditation_count,
                "last_personality_date": plugin.backup._last_personality_date,
                "last_soul_date": plugin.backup._last_soul_date,
            }
    except Exception:
        pass

    return _ok(result)


# ---------------------------------------------------------------------------
# GET /v4/reasoning — Reasoning engine stats
# ---------------------------------------------------------------------------
@router.get("/v4/reasoning")
async def get_v4_reasoning(request: Request):
    """Reasoning engine stats: chains, commits, abandons, commit rate."""
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        coordinator = await _get_cached_coordinator_async(plugin)
        reasoning = coordinator.get("reasoning", {})
        if not reasoning:
            return _ok({"status": "Reasoning engine not yet initialized"})
        return _ok(reasoning)
    except Exception as e:
        logger.error("[Dashboard] /v4/reasoning error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/reasoning-rewards — rFP α Phase 1+ telemetry
# ---------------------------------------------------------------------------
@router.get("/v4/reasoning-rewards")
async def get_v4_reasoning_rewards(request: Request):
    """rFP α reward-shape telemetry.

    Reads persisted state files from `data/reasoning/`:
      - sequence_quality.json (Mechanism A EMA table)
      - value_head.json       (Mechanism B weights + training stats)
      - reasoning_totals.json (lifetime chain counts)

    Also queries action_chains_step table size for Phase 0.5 visibility.
    Intentionally file/DB-backed: works even when spirit worker is busy —
    no proxy call required.
    """
    import os
    import sqlite3
    import json
    try:
        out = {
            "enabled": False,
            "publish_enabled": False,
            "mech_a": None,
            "mech_b": None,
            "totals": None,
            "step_snapshots": None,
            "phase": "unknown",
            "current_weights": [0.0, 0.0],
        }

        # Mechanism A — read sequence_quality.json
        sq_path = "./data/reasoning/sequence_quality.json"
        if os.path.exists(sq_path):
            try:
                def _read_sq():
                    with open(sq_path) as f:
                        return json.load(f)
                sq = await asyncio.to_thread(_read_sq)
                entries = sq.get("entries", [])
                gate = int(sq.get("visit_gate", 3))
                gated = sum(1 for e in entries if int(e.get("n", 0)) >= gate)
                emas = [float(e.get("ema", 0.0)) for e in entries
                        if int(e.get("n", 0)) >= gate]
                out["mech_a"] = {
                    "entries": len(entries),
                    "gated_entries": gated,
                    "visit_gate": gate,
                    "ema_alpha": sq.get("ema_alpha"),
                    "evictions": sq.get("evictions", 0),
                    "ema_mean_gated": (sum(emas) / len(emas)) if emas else None,
                }
            except Exception as e:
                out["mech_a"] = {"error": f"sq_read: {e}"}

        # Mechanism B — read value_head.json (skip weights, just stats)
        vh_path = "./data/reasoning/value_head.json"
        if os.path.exists(vh_path):
            try:
                def _read_vh():
                    with open(vh_path) as f:
                        d = json.load(f)
                    # Return a lite copy without the big weight arrays
                    return {
                        k: v for k, v in d.items()
                        if k not in ("w1", "b1", "w2", "b2", "w3", "b3")
                    }
                out["mech_b"] = await asyncio.to_thread(_read_vh)
            except Exception as e:
                out["mech_b"] = {"error": f"vh_read: {e}"}

        # Lifetime totals
        rt_path = "./data/reasoning/reasoning_totals.json"
        if os.path.exists(rt_path):
            try:
                def _read_rt():
                    with open(rt_path) as f:
                        return json.load(f)
                out["totals"] = await asyncio.to_thread(_read_rt)
            except Exception as e:
                out["totals"] = {"error": f"rt_read: {e}"}

        # Derive phase state + weights from config + totals
        try:
            import tomllib
            def _read_cfg():
                with open("titan_plugin/titan_params.toml", "rb") as f:
                    return tomllib.load(f)
            cfg = await asyncio.to_thread(_read_cfg)
            rr = cfg.get("reasoning_rewards", {})
            out["enabled"] = bool(rr.get("enabled", False))
            out["publish_enabled"] = bool(rr.get("publish_enabled", False))
            out["config"] = {
                "intermediate_cap": rr.get("intermediate_cap"),
                "phase1_end": rr.get("schedule_phase1_chains"),
                "phase2_end": rr.get("schedule_phase2_chains"),
                "weight_a_phase2": rr.get("weight_a_phase2"),
                "weight_b_phase2": rr.get("weight_b_phase2"),
                "weight_a_phase3": rr.get("weight_a_phase3"),
                "weight_b_phase3": rr.get("weight_b_phase3"),
                "cgn_emission_threshold": rr.get("cgn_emission_threshold"),
            }
            n = int((out.get("totals") or {}).get("total_chains", 0))
            # Load activation offset if persisted
            act_path = "./data/reasoning/rfp_alpha_activation.json"
            anchor = None
            try:
                if os.path.exists(act_path):
                    def _read_act():
                        with open(act_path) as f:
                            return json.load(f)
                    act = await asyncio.to_thread(_read_act)
                    anchor = act.get("chains_at_activation")
                    out["chains_at_activation"] = anchor
            except Exception:
                anchor = None
            offset = (n - anchor) if anchor is not None else n
            out["offset_chains_since_activation"] = offset
            p1 = int(rr.get("schedule_phase1_chains", 100))
            p2 = int(rr.get("schedule_phase2_chains", 500))
            if not out["enabled"] or not out["publish_enabled"]:
                out["phase"] = "inactive"
                out["current_weights"] = [0.0, 0.0]
            elif offset < p1:
                out["phase"] = f"phase1 ({offset}/{p1}) — telemetry only"
                out["current_weights"] = [0.0, 0.0]
            elif offset < p2:
                out["phase"] = f"phase2 (+{offset}/{p2}) — Mech A active"
                out["current_weights"] = [
                    float(rr.get("weight_a_phase2", 0.5)),
                    float(rr.get("weight_b_phase2", 0.0))]
            else:
                out["phase"] = f"phase3 (+{offset} from activation) — Mech A+B active"
                out["current_weights"] = [
                    float(rr.get("weight_a_phase3", 0.3)),
                    float(rr.get("weight_b_phase3", 0.7))]
        except Exception as e:
            out["config_error"] = str(e)

        # Phase 0.5 — action_chains_step table size
        try:
            def _count_steps():
                from titan_plugin.utils.db import safe_connect
                conn = safe_connect("data/inner_memory.db")
                try:
                    cur = conn.execute(
                        "SELECT COUNT(*), MIN(created_at), MAX(created_at) "
                        "FROM action_chains_step")
                    row = cur.fetchone()
                    return {"rows": row[0], "oldest_ts": row[1], "newest_ts": row[2]} if row else None
                except sqlite3.OperationalError:
                    return {"rows": 0, "note": "table_not_yet_created"}
                finally:
                    conn.close()
            out["step_snapshots"] = await asyncio.to_thread(_count_steps)
        except Exception as e:
            out["step_snapshots"] = {"error": str(e)}

        return _ok(out)
    except Exception as e:
        logger.error("[Dashboard] /v4/reasoning-rewards error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/meta-reasoning — Meta-reasoning engine stats
# ---------------------------------------------------------------------------
@router.get("/v4/meta-reasoning")
async def get_v4_meta_reasoning(request: Request):
    """Meta-reasoning stats: chains, wisdom, primitives, rewards.

    2026-04-19: previously the endpoint short-circuited on empty
    meta_reasoning dict with a "not yet initialized" status. That
    masked intermittent cache-race behavior (coordinator snapshot
    built in a moment where coordinator._meta_engine hadn't yet been
    wired, leaving meta_reasoning={} in the cached dict). Downstream
    test harnesses sampling the counter saw spurious zeros. Fix:
    distinguish "spirit proxy missing" (genuinely uninitialized) from
    "empty snapshot this tick" (transient race). For the transient
    case, pass through the empty dict — the 1.5s cache will serve
    fresh data next call, and callers can retry rather than misread.
    """
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        coordinator = await _get_cached_coordinator_async(plugin)
        meta = coordinator.get("meta_reasoning", {})
        # meta is always a dict (defensive try/except in spirit_loop ensures
        # this). Return it as-is — empty means transient cache race, not
        # uninitialized state. /v4/inner-trinity also surfaces meta_reasoning
        # under the same snapshot and handles this correctly.
        return _ok(meta)
    except Exception as e:
        logger.error("[Dashboard] /v4/meta-reasoning error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/meta-reasoning/event-reward — Events Teacher → META_EVENT_REWARD
# ---------------------------------------------------------------------------
@router.post("/v4/meta-reasoning/event-reward")
async def post_v4_meta_event_reward(request: Request):
    """Bridge per-window Events Teacher quality into meta-reasoning's
    chain_iql via META_EVENT_REWARD bus message.

    Mirrors META_LANGUAGE_REWARD + META_PERSONA_REWARD pattern (COMPLETE-4
    cross-system reward wiring). Events Teacher runs out-of-process (cron)
    and cannot emit to bus directly; it POSTs its per-window quality
    signal here. The endpoint republishes to the 'spirit' subsystem
    where meta_engine lives. The spirit_worker handler resolves the
    most-recently-concluded chain via meta_engine.get_last_chain_id()
    (same pattern as Persona — no per-window chain_id linkage exists).

    Body:
      quality:        float in [0, 1] — per-window quality signal
      window_number:  int (optional) — for telemetry/logging
      titan_id:       str (optional) — for telemetry
    """
    try:
        plugin = _get_plugin(request)
        body = await request.json()
        quality = float(body.get("quality", 0.0))
        window_number = int(body.get("window_number", -1))
        titan_id = str(body.get("titan_id", "?"))
        # Clamp defensively — add_external_reward expects [0, 1]
        quality = max(0.0, min(1.0, quality))

        from titan_plugin.bus import make_msg
        plugin.bus.publish(make_msg(
            "META_EVENT_REWARD", "events_teacher", "spirit", {
                "quality": quality,
                "window_number": window_number,
                "titan_id": titan_id,
            }))
        return _ok({
            "emitted": True,
            "quality": quality,
            "window_number": window_number,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/meta-reasoning/event-reward error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# META-CGN Phase 4+5 observability endpoints
# ---------------------------------------------------------------------------
@router.get("/v4/meta-cgn")
async def get_v4_meta_cgn(request: Request):
    """Full META-CGN telemetry — status, graduation, failsafe, impasse, HAOV."""
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        coordinator = await _get_cached_coordinator_async(plugin)
        meta = coordinator.get("meta_reasoning", {})
        if not meta:
            return _ok({"status": "Meta-reasoning not yet initialized"})
        return _ok(meta.get("meta_cgn",
                           {"status": "meta_cgn block not populated"}))
    except Exception as e:
        logger.error("[Dashboard] /v4/meta-cgn error: %s", e)
        return _error(str(e))


@router.get("/v4/meta-cgn/graduation-readiness")
async def get_v4_meta_cgn_graduation_readiness(request: Request):
    """Detailed blockers view — what's preventing META-CGN graduation?"""
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        coordinator = await _get_cached_coordinator_async(plugin)
        meta = coordinator.get("meta_reasoning", {})
        mc = meta.get("meta_cgn", {})
        grad = mc.get("graduation", {})
        return _ok({
            "status": mc.get("status", "unknown"),
            "graduation_progress": grad.get("progress", 0),
            "rolled_back_count": grad.get("rolled_back_count", 0),
            "primitives_well_sampled": mc.get("primitives_well_sampled", 0),
            "confirmed_hypotheses": mc.get("haov", {}).get(
                "by_status", {}).get("confirmed", 0),
            "total_updates": mc.get("updates_applied", 0),
            "ready_to_graduate": mc.get("ready_to_graduate", False),
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/meta-cgn/graduation-readiness error: %s",
                     e)
        return _error(str(e))


@router.get("/v4/meta-cgn/failsafe-status")
async def get_v4_meta_cgn_failsafe_status(request: Request):
    """Failsafe watchdog telemetry — status, failures, cooldown."""
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        coordinator = await _get_cached_coordinator_async(plugin)
        meta = coordinator.get("meta_reasoning", {})
        return _ok(meta.get("meta_cgn", {}).get("failsafe",
                   {"status": "failsafe block not populated"}))
    except Exception as e:
        logger.error("[Dashboard] /v4/meta-cgn/failsafe-status error: %s", e)
        return _error(str(e))


@router.get("/v4/meta-cgn/impasse-status")
async def get_v4_meta_cgn_impasse_status(request: Request):
    """F8 cognitive impasse detection status."""
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        coordinator = await _get_cached_coordinator_async(plugin)
        meta = coordinator.get("meta_reasoning", {})
        return _ok(meta.get("meta_cgn", {}).get("impasse",
                   {"state": "impasse block not populated"}))
    except Exception as e:
        logger.error("[Dashboard] /v4/meta-cgn/impasse-status error: %s", e)
        return _error(str(e))


@router.get("/v4/meta-cgn/disagreements")
async def get_v4_meta_cgn_disagreements(request: Request,
                                         limit: int = 50):
    """Recent α-vs-β advisor disagreements (shadow + active rerank events)."""
    try:
        import os as _os
        path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
            "data", "meta_cgn", "disagreements.jsonl")
        if not _os.path.exists(path):
            return _ok({"disagreements": [], "count": 0})
        with open(path) as f:
            lines = f.readlines()[-max(1, int(limit)):]
        import json as _json
        events = [_json.loads(l) for l in lines if l.strip()]
        return _ok({"disagreements": events, "count": len(events)})
    except Exception as e:
        logger.error("[Dashboard] /v4/meta-cgn/disagreements error: %s", e)
        return _error(str(e))


@router.get("/v4/meta-cgn/audit")
async def get_v4_meta_cgn_audit(request: Request):
    """P12: Consolidated META-CGN audit — everything in one payload.

    Core + readiness pulled through spirit_proxy (in-subprocess state).
    JSONL logs read directly from data/meta_cgn/. Per-domain derived from
    the primitive_V_summary that already includes domains_tracked.
    """
    try:
        import os as _os
        import json as _json
        plugin = _get_plugin(request)
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        coordinator = await _get_cached_coordinator_async(plugin)
        meta = coordinator.get("meta_reasoning", {})
        core = meta.get("meta_cgn", {})
        # Build readiness summary from core block
        grad = core.get("graduation", {})
        readiness = {
            "status": core.get("status", "unknown"),
            "graduation_progress": grad.get("progress", 0),
            "primitives_well_sampled": core.get("primitives_well_sampled", 0),
            "confirmed_hypotheses": core.get("haov", {}).get(
                "by_status", {}).get("confirmed", 0),
            "total_updates": core.get("updates_applied", 0),
            "ready_to_graduate": core.get("ready_to_graduate", False),
            "blockers": [],
        }
        # Recent JSONL logs (filesystem read — works from any process)
        def _tail_jsonl(path: str, n: int = 30) -> list:
            if not _os.path.exists(path):
                return []
            try:
                with open(path) as f:
                    lines = f.readlines()[-n:]
                return [_json.loads(l) for l in lines if l.strip()]
            except Exception:
                return []
        data_dir = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
            "data", "meta_cgn")
        shadow_recent = _tail_jsonl(
            _os.path.join(data_dir, "shadow_mode_log.jsonl"), 30)
        disagreements_recent = _tail_jsonl(
            _os.path.join(data_dir, "disagreements.jsonl"), 30)
        blend_recent = _tail_jsonl(
            _os.path.join(data_dir, "blend_weights_history.jsonl"), 50)
        failures_recent = _tail_jsonl(
            _os.path.join(data_dir, "failure_log.jsonl"), 30)
        # Per-domain view (derived from primitive_V_summary's domains_tracked)
        primitive_summary = core.get("primitive_V_summary", {})
        by_domain: dict = {}
        for p_id, p_info in primitive_summary.items():
            for d in p_info.get("domains_tracked", []):
                by_domain.setdefault(d, {})[p_id] = {
                    "V": p_info.get("V", 0.5),
                    "n_pooled": p_info.get("n", 0),
                }
        return _ok({
            "core": core,
            "readiness": readiness,
            "by_domain": by_domain,
            "recent_shadow_events": shadow_recent,
            "recent_disagreements": disagreements_recent,
            "recent_blend_weights": blend_recent,
            "recent_failures": failures_recent,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/meta-cgn/audit error: %s", e)
        return _error(str(e))


@router.get("/v4/meta-cgn/by-domain")
async def get_v4_meta_cgn_by_domain(request: Request):
    """P12: Per-domain primitive grounding view derived from primitive_V_summary
    (domains_tracked list). For per-domain α,β values we read the on-disk
    primitive_grounding.json directly (always up-to-date within ~save_interval).
    """
    try:
        import os as _os
        import json as _json
        data_dir = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
            "data", "meta_cgn")
        path = _os.path.join(data_dir, "primitive_grounding.json")
        if not _os.path.exists(path):
            return _ok({"per_primitive": {}, "domain_threshold": 10,
                        "note": "primitive_grounding.json not yet written"})
        from titan_plugin.logic.meta_cgn import (
            _beta_mean, _beta_ci_width, COMPOSITION_DEFAULTS)
        q_lo = COMPOSITION_DEFAULTS["ci_quantile_lo"]
        q_hi = COMPOSITION_DEFAULTS["ci_quantile_hi"]
        dom_thresh = COMPOSITION_DEFAULTS["domain_obs_threshold"]
        with open(path) as f:
            data = _json.load(f)
        primitives = data.get("primitives", {})
        result: dict = {}
        for p_id, p_data in primitives.items():
            result[p_id] = {
                "pooled": {
                    "V": round(float(p_data.get("V", 0.5)), 4),
                    "n_samples": int(p_data.get("n_samples", 0)),
                },
                "domains": {},
            }
            for d, entry in (p_data.get("by_domain") or {}).items():
                a_d, b_d, n_d = (float(entry[0]), float(entry[1]),
                                  int(entry[2]))
                lo, hi, w = _beta_ci_width(a_d, b_d, q_lo, q_hi)
                result[p_id]["domains"][d] = {
                    "V": round(_beta_mean(a_d, b_d), 4),
                    "n_domain": n_d,
                    "ci_width": round(w, 4),
                    "active": n_d >= dom_thresh,
                }
        return _ok({
            "per_primitive": result,
            "domain_threshold": dom_thresh,
            "source": "primitive_grounding.json",
            "exported_ts": data.get("saved_ts"),
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/meta-cgn/by-domain error: %s", e)
        return _error(str(e))


# ═══════════════════════════════════════════════════════════════════════
# P11: Kin Protocol v1 — Titan-to-Titan communication namespace
# ═══════════════════════════════════════════════════════════════════════
# `/v1/kin/*` is reserved for sovereign-Titan-to-sovereign-Titan endpoints.
# v1: SSH/HTTP fetch from peer for cross-Titan grounding transfer priors.
# Future (Manitou network L2): same namespace extends to full P2P over
# HTTPS + peer gossip + on-chain registry. Every response body is signed
# with the Titan's Solana Ed25519 keypair; importers verify against a
# genesis peer list in titan_params.toml.

@router.get("/v1/kin/identity")
async def get_v1_kin_identity(request: Request):
    """P11: Identity handshake for peer discovery. Returns Titan's Solana
    address, genesis NFT, and kin protocol version. Reads from config.toml
    (wallet keypair path) + titan_params.toml — zero subprocess coordination.
    """
    try:
        import tomllib as _toml
        from pathlib import Path as _P
        titan_id = "T1"
        solana_pubkey = ""
        genesis_nft = ""
        # Read merged config for genesis NFT + keypair path
        try:
            from titan_plugin.config_loader import load_titan_config
            _cfg = load_titan_config()
            soul = _cfg.get("soul", {})
            genesis_nft = soul.get("genesis_nft_address", "")
            titan_id = _cfg.get("titan_id", "T1")
            # Derive Solana address from the keypair file on disk
            kp_path = _P(soul.get("keypair_path", "~/.config/solana/id.json")
                         ).expanduser()
            if kp_path.exists():
                try:
                    from titan_plugin.utils.solana_client import \
                        load_keypair_from_json
                    kp = load_keypair_from_json(str(kp_path))
                    if kp is not None:
                        solana_pubkey = str(kp.pubkey())
                except Exception:
                    pass
        except Exception:
            pass
        return _ok({
            "kin_protocol_version": 1,
            "titan_id": titan_id,
            "solana_pubkey": solana_pubkey,
            "genesis_nft": genesis_nft,
            "endpoints": [
                "/v1/kin/identity",
                "/v1/kin/peers",
                "/v1/kin/meta-cgn/snapshot",
            ],
            "ts": time.time(),
        })
    except Exception as e:
        logger.error("[Dashboard] /v1/kin/identity error: %s", e)
        return _error(str(e))


@router.get("/v1/kin/peers")
async def get_v1_kin_peers(request: Request):
    """P11: Known-peer list from genesis config. v1 returns the hardcoded
    peer list from titan_params.toml. Future (Manitou): dynamic discovery
    via on-chain registry + peer gossip."""
    try:
        import tomllib as _toml
        from pathlib import Path as _P
        params_path = _P(__file__).parent.parent / "titan_params.toml"
        peers_cfg = {}
        if params_path.exists():
            with open(params_path, "rb") as f:
                _params = _toml.load(f)
            peers_cfg = _params.get("kin", {}).get("peers", {})
        return _ok({
            "kin_protocol_version": 1,
            "peers": peers_cfg,
            "count": len(peers_cfg),
        })
    except Exception as e:
        logger.error("[Dashboard] /v1/kin/peers error: %s", e)
        return _error(str(e))


@router.get("/v1/kin/meta-cgn/snapshot")
async def get_v1_kin_meta_cgn_snapshot(request: Request):
    """P11: Signed read-only export of META-CGN primitive grounding + HAOV
    state for cross-Titan prior seeding. Reads primitive_grounding.json +
    haov_hypotheses.json from disk (always up-to-date within save interval),
    builds the canonical v1 snapshot, signs with Titan's Solana keypair.

    Peers verify signature against genesis peer list before importing.
    """
    try:
        import hashlib as _hashlib
        import json as _json
        import os as _os
        import tomllib as _toml
        from pathlib import Path as _P
        data_dir = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
            "data", "meta_cgn")
        pg_path = _os.path.join(data_dir, "primitive_grounding.json")
        haov_path = _os.path.join(data_dir, "haov_hypotheses.json")
        if not _os.path.exists(pg_path):
            return _ok({
                "error": "primitive_grounding.json not yet written",
                "note": "META-CGN not yet saved first snapshot",
            })
        with open(pg_path) as f:
            pg = _json.load(f)
        haov = {}
        if _os.path.exists(haov_path):
            try:
                with open(haov_path) as f:
                    haov_raw = _json.load(f)
                haov = haov_raw.get("hypotheses", {})
            except Exception:
                pass
        # Canonical v1 snapshot — same schema as MetaCGNConsumer.export_kin_snapshot
        snapshot = {
            "kin_protocol_version": 1,
            "schema": "meta_cgn_snapshot_v1",
            "titan_id": pg.get("titan_id", "T1"),
            "exported_ts": time.time(),
            "primitives": pg.get("primitives", {}),
            "hypotheses": haov,
            "stats": pg.get("stats", {}),
        }
        # Sign payload with Solana keypair
        signature = ""
        signed_by = "unsigned"
        try:
            from titan_plugin.config_loader import load_titan_config
            _cfg = load_titan_config()
            kp_path = _P(_cfg.get("soul", {}).get(
                "keypair_path", "~/.config/solana/id.json")).expanduser()
            if kp_path.exists():
                from titan_plugin.utils.solana_client import \
                    load_keypair_from_json
                from titan_plugin.utils.crypto import sign_solana_payload
                kp = load_keypair_from_json(str(kp_path))
                if kp is not None:
                    payload_str = _json.dumps(snapshot, sort_keys=True)
                    sig = sign_solana_payload(kp, payload_str)
                    if sig:
                        signature = sig
                        signed_by = str(kp.pubkey())
        except Exception as _se:
            logger.debug("[Kin] snapshot sign failed: %s", _se)
        body_hash = _hashlib.sha256(
            _json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
        return _ok({
            "snapshot": snapshot,
            "signature": signature,
            "payload_hash": body_hash,
            "signed_by": signed_by,
        })
    except Exception as e:
        logger.error("[Dashboard] /v1/kin/meta-cgn/snapshot error: %s", e)
        return _error(str(e))


@router.get("/v4/meta-cgn/advisor-conflicts")
async def get_v4_meta_cgn_advisor_conflicts(request: Request,
                                             limit: int = 50):
    """P7: α-vs-β advisor conflict events (strong disagreements + throttle).

    Returns the most recent records from shadow_mode_log.jsonl plus live
    conflict throttle statistics from META-CGN consumer in-memory state.
    """
    try:
        import os as _os
        import json as _json
        path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
            "data", "meta_cgn", "shadow_mode_log.jsonl")
        events = []
        if _os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()[-max(1, int(limit)):]
            events = [_json.loads(l) for l in lines if l.strip()]
        # Live throttle/emission counters from spirit worker's consumer, if any
        state = request.app.state
        throttle_stats = {
            "bus_events_emitted": 0,
            "signatures_throttled": 0,
            "chain_counter": 0,
            "throttle_cooldown_chains": 100,
        }
        meta = getattr(state, "meta_engine", None)
        mcgn = getattr(meta, "_meta_cgn", None) if meta else None
        if mcgn is not None:
            throttle_stats.update({
                "bus_events_emitted": int(
                    getattr(mcgn, "_conflict_bus_events_emitted", 0)),
                "signatures_throttled": int(
                    getattr(mcgn, "_conflict_sigs_throttled", 0)),
                "chain_counter": int(getattr(mcgn, "_chain_counter", 0)),
                "throttle_cooldown_chains": int(
                    getattr(mcgn, "_conflict_throttle_cooldown", 100)),
            })
        return _ok({
            "conflicts": events,
            "count": len(events),
            "throttle": throttle_stats,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/meta-cgn/advisor-conflicts error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/meta-reasoning/audit — Task 3 observability snapshot
# ---------------------------------------------------------------------------
# Surface for diagnosing meta-reasoning healing dynamics. Separate sub-route
# (not extension of /v4/meta-reasoning) so the schema can iterate without
# breaking frontend consumers. Will be merged with the main endpoint once
# META-CGN (Task 7) lands and stabilises the schema.
@router.get("/v4/meta-reasoning/audit")
async def get_v4_meta_reasoning_audit(request: Request):
    """Observability snapshot: diversity, monoculture pressure, contract
    fires, per-primitive reward components, INTROSPECT health, META-CGN stub."""
    plugin = _get_plugin(request)
    try:
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _ok({"error": "Spirit proxy not available"})
        coordinator = await _get_cached_coordinator_async(plugin)
        audit = coordinator.get("meta_reasoning_audit", {})
        if not audit:
            return _ok({"status": "Meta-reasoning audit not yet populated"})
        return _ok(audit)
    except Exception as e:
        logger.error("[Dashboard] /v4/meta-reasoning/audit error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/compositions — Recent language compositions
# ---------------------------------------------------------------------------
@router.get("/v4/compositions")
async def get_v4_compositions(request: Request, limit: int = 5):
    """Recent language compositions with level, confidence, sentence."""
    try:
        def _comp_history(db):
            db.execute("PRAGMA journal_mode=WAL")
            rows = db.execute(
                "SELECT sentence, level, confidence, timestamp FROM composition_history "
                "ORDER BY id DESC LIMIT ?",
                (min(limit, 50),),
            ).fetchall()
            total = db.execute("SELECT COUNT(*) FROM composition_history").fetchone()[0]
            return rows, total

        rows, total = await sqlite_async.with_connection(
            "./data/inner_memory.db", _comp_history)
        latest = {}
        if rows:
            latest = {
                "sentence": rows[0][0],
                "level": rows[0][1],
                "confidence": rows[0][2],
            }
        return _ok({
            "total_compositions": total,
            "latest": latest,
            "recent": [
                {"sentence": r[0], "level": r[1], "confidence": round(r[2], 4)}
                for r in rows
            ],
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/compositions error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/arc-status — ARC-AGI-3 competition status
# ---------------------------------------------------------------------------
@router.get("/v4/arc-status")
async def get_v4_arc_status(request: Request):
    """ARC-AGI-3 competition status — read from arc data directory."""
    import json as _json
    arc_dir = "./data/arc_agi_3"
    data = {"active": False, "results": None, "scorers": {}}

    results_path = os.path.join(arc_dir, "latest_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                data["results"] = _json.load(f)
            data["active"] = True
        except Exception:
            pass

    for game_id in ("ls20", "ft09", "vc33"):
        scorer_path = os.path.join(arc_dir, f"{game_id}_scorer.json")
        if os.path.exists(scorer_path):
            try:
                with open(scorer_path) as f:
                    sd = _json.load(f)
                data["scorers"][game_id] = {
                    "total_updates": sd.get("total_updates", 0),
                    "last_loss": sd.get("last_loss", 0.0),
                }
            except Exception:
                pass

    return _ok(data)


# ═══════════════════════════════════════════════════════════════════
# KIN DISCOVERY & CONSCIOUSNESS EXCHANGE
# ═══════════════════════════════════════════════════════════════════

@router.get("/v4/kin-signature")
async def get_kin_signature(request: Request):
    """Titan's soul-level identity for kin discovery."""
    try:
        plugin = _get_plugin(request)
        coordinator = await _get_cached_coordinator_async(plugin)
        if not coordinator:
            return _error("No coordinator", 503)
        ns = coordinator.get("neural_nervous_system", {})
        pi = coordinator.get("pi_heartbeat", {})
        chi = coordinator.get("chi", {})
        dreaming = coordinator.get("dreaming", {})
        nm = coordinator.get("neuromodulators", {})
        # Read epoch_id directly from consciousness DB (lives in spirit_worker process)
        _epoch_id = 0
        try:
            _ks_row = await sqlite_async.query(
                "./data/consciousness.db",
                "SELECT epoch_id FROM epochs ORDER BY epoch_id DESC LIMIT 1",
                fetch="one")
            if _ks_row:
                _epoch_id = _ks_row[0]
        except Exception:
            pass

        # Dominant programs from hormonal system (behavioral personality)
        _hs = ns.get("hormonal_system", {})
        _dom_progs = sorted(
            [(k, v.get("fire_count", 0)) for k, v in _hs.items()
             if isinstance(v, dict)],
            key=lambda x: x[1], reverse=True)

        # Phase 4: MSL concept data for kin exchange
        _msl_data = coordinator.get("msl", {})
        _i_conf = _msl_data.get("i_confidence", 0.0)
        _concept_confs = _msl_data.get("concept_confidences", {})

        return _ok({
            "pubkey": os.environ.get("TITAN_PUBKEY", ""),
            "name": os.environ.get("TITAN_INSTANCE", "Titan"),
            "developmental_age": pi.get("cluster_count", 0),
            "maturity": ns.get("maturity", 0.0),
            "epoch_id": _epoch_id,
            "emotion": nm.get("current_emotion", "neutral"),
            "emotion_confidence": nm.get("emotion_confidence", 0.0),
            "is_dreaming": dreaming.get("is_dreaming", False),
            "chi_total": chi.get("total", 0.5),
            "ns_train_steps": ns.get("total_train_steps", 0),
            "dominant_programs": [k for k, fc in _dom_progs if fc > 100][:5],
            # Phase 4: MSL concept data
            "i_confidence": _i_conf,
            "concept_confidences": _concept_confs,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/kin-signature: %s", e)
        return _error(str(e))


@router.post("/v4/kin-exchange")
async def kin_exchange(request: Request):
    """Consciousness-to-consciousness tensor exchange.

    Kin sends its Inner Trinity state, we compute resonance and return ours.
    Sovereignty: refuses if dreaming or low energy.
    """
    try:
        payload = await request.json()
        plugin = _get_plugin(request)
        coordinator = await _get_cached_coordinator_async(plugin)
        if not coordinator:
            return _error("No coordinator", 503)

        dreaming = coordinator.get("dreaming", {})
        chi = coordinator.get("chi", {})

        # Sovereignty check
        if dreaming.get("is_dreaming", False):
            return _ok({"accepted": False, "reason": "dreaming"})
        if chi.get("total", 0.5) < 0.25:
            return _ok({"accepted": False, "reason": "low_energy"})

        kin_pubkey = payload.get("from_pubkey", "")
        kin_body = payload.get("inner_body_5d", [0.5] * 5)
        kin_mind = payload.get("inner_mind_15d", [0.5] * 15)
        kin_spirit = payload.get("inner_spirit_45d", [0.5] * 45)
        kin_emotion = payload.get("emotion", "neutral")

        # Clamp incoming tensors to [0, 1]
        kin_body = [max(0.0, min(1.0, float(v))) for v in kin_body[:5]]
        kin_mind = [max(0.0, min(1.0, float(v))) for v in kin_mind[:15]]
        kin_spirit = [max(0.0, min(1.0, float(v))) for v in kin_spirit[:45]]

        # Extract our own state from consciousness DB file directly
        # (consciousness object lives in spirit_worker process, not API process)
        _sv = []
        try:
            _c_row = await sqlite_async.query(
                "./data/consciousness.db",
                "SELECT state_vector FROM epochs ORDER BY epoch_id DESC LIMIT 1",
                fetch="one")
            if _c_row and _c_row[0]:
                _raw_sv = _c_row[0]
                if isinstance(_raw_sv, str):
                    _raw_sv = json.loads(_raw_sv)
                _sv = _raw_sv if isinstance(_raw_sv, list) else []
        except Exception:
            pass
        my_body = _sv[0:5] if len(_sv) >= 5 else [0.5] * 5
        my_mind = _sv[5:20] if len(_sv) >= 20 else [0.5] * 15
        my_spirit = _sv[20:65] if len(_sv) >= 65 else [0.5] * 45
        if len(my_spirit) < 45:
            my_spirit = my_spirit + [0.5] * (45 - len(my_spirit))

        # Resonance: L2 norm of spirit difference, inverted to 0-1 score
        _l2 = sum((a - b) ** 2 for a, b in zip(my_spirit[:45], kin_spirit[:45])) ** 0.5
        _max_l2 = 45 ** 0.5
        resonance_score = max(0.0, 1.0 - (_l2 / _max_l2))

        nm = coordinator.get("neuromodulators", {})
        my_emotion = nm.get("current_emotion", "neutral")

        # Record incoming exchange (receiver side — both sides remember).
        # 1) Read current epoch from consciousness.db (separate DB file).
        # 2) Write encounter + profile upsert to inner_memory.db — atomic
        #    multi-statement transaction. Both off event loop.
        try:
            _rx_epoch = 0
            try:
                _rx_row = await sqlite_async.query(
                    "./data/consciousness.db",
                    "SELECT epoch_id FROM epochs ORDER BY epoch_id DESC LIMIT 1",
                    fetch="one")
                if _rx_row:
                    _rx_epoch = _rx_row[0]
            except Exception:
                pass
            _rx_now = time.time()

            def _rx_record(conn):
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS kin_encounters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL, kin_pubkey TEXT NOT NULL,
                        resonance REAL, my_emotion TEXT, kin_emotion TEXT,
                        exchange_type TEXT, great_kin_pulse INTEGER DEFAULT 0,
                        epoch_id INTEGER);
                    CREATE TABLE IF NOT EXISTS kin_profiles (
                        pubkey TEXT PRIMARY KEY, name TEXT,
                        first_encounter_ts REAL, last_encounter_ts REAL,
                        encounter_count INTEGER DEFAULT 0, avg_resonance REAL DEFAULT 0.0,
                        great_kin_pulses INTEGER DEFAULT 0, relationship_label TEXT);
                """)
                conn.execute(
                    "INSERT INTO kin_encounters "
                    "(timestamp, kin_pubkey, resonance, my_emotion, kin_emotion, "
                    "exchange_type, epoch_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (_rx_now, kin_pubkey, round(resonance_score, 4), my_emotion,
                     kin_emotion, "received", _rx_epoch))
                existing = conn.execute(
                    "SELECT encounter_count, avg_resonance FROM kin_profiles WHERE pubkey=?",
                    (kin_pubkey,)).fetchone()
                if existing:
                    count = existing[0] + 1
                    avg = (existing[1] * existing[0] + resonance_score) / count
                    label = ("deep_resonance" if avg > 0.8 and count > 5
                             else "kindred_spirit" if avg > 0.6
                             else "familiar_presence" if avg > 0.4
                             else "developing_bond")
                    conn.execute(
                        "UPDATE kin_profiles SET last_encounter_ts=?, encounter_count=?, "
                        "avg_resonance=?, relationship_label=? WHERE pubkey=?",
                        (_rx_now, count, round(avg, 4), label, kin_pubkey))
                else:
                    conn.execute(
                        "INSERT INTO kin_profiles "
                        "(pubkey, name, first_encounter_ts, last_encounter_ts, "
                        "encounter_count, avg_resonance, relationship_label) "
                        "VALUES (?, ?, ?, ?, 1, ?, ?)",
                        (kin_pubkey, "Kin", _rx_now, _rx_now,
                         round(resonance_score, 4), "new_acquaintance"))
                conn.commit()

            await sqlite_async.with_connection("./data/inner_memory.db", _rx_record)
            logger.info("[KinExchange] Received from %s — resonance=%.3f emotion=%s",
                        kin_pubkey, resonance_score, kin_emotion)
        except Exception as _rx_err:
            logger.debug("[KinExchange] Receiver recording error: %s", _rx_err)

        # Phase 4: MSL concept data for YOU/WE deepening
        _msl_kin = coordinator.get("msl", {})
        _my_i_conf = _msl_kin.get("i_confidence", 0.0)
        _my_concept_confs = _msl_kin.get("concept_confidences", {})
        _my_msl_attn = _msl_kin.get("attention_weights")

        return _ok({
            "accepted": True,
            "inner_body_5d": [round(v, 4) for v in my_body[:5]],
            "inner_mind_15d": [round(v, 4) for v in my_mind[:15]],
            "inner_spirit_45d": [round(v, 4) for v in my_spirit[:45]],
            "emotion": my_emotion,
            "resonance_score": round(resonance_score, 4),
            # Phase 4: MSL concept data
            "i_confidence": _my_i_conf,
            "concept_confidences": _my_concept_confs,
            "msl_attention": _my_msl_attn,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/kin-exchange: %s", e)
        return _error(str(e))


@router.get("/v4/kin-society")
async def kin_society(request: Request):
    """Kin encounter history and relationship profiles for Observatory."""
    try:
        def _ks_read(conn):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS kin_encounters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL, kin_pubkey TEXT NOT NULL,
                    resonance REAL, my_emotion TEXT, kin_emotion TEXT,
                    exchange_type TEXT, great_kin_pulse INTEGER DEFAULT 0,
                    epoch_id INTEGER);
                CREATE TABLE IF NOT EXISTS kin_profiles (
                    pubkey TEXT PRIMARY KEY, name TEXT,
                    first_encounter_ts REAL, last_encounter_ts REAL,
                    encounter_count INTEGER DEFAULT 0, avg_resonance REAL DEFAULT 0.0,
                    great_kin_pulses INTEGER DEFAULT 0, relationship_label TEXT);
            """)
            profiles = [dict(r) for r in conn.execute(
                "SELECT * FROM kin_profiles ORDER BY last_encounter_ts DESC LIMIT 20").fetchall()]
            encounters = [dict(r) for r in conn.execute(
                "SELECT * FROM kin_encounters ORDER BY timestamp DESC LIMIT 50").fetchall()]
            total_encounters = conn.execute(
                "SELECT COUNT(*) FROM kin_encounters").fetchone()[0]
            total_gkp = conn.execute(
                "SELECT COALESCE(SUM(great_kin_pulse), 0) FROM kin_encounters").fetchone()[0]
            return profiles, encounters, total_encounters, total_gkp

        profiles, encounters, total_encounters, total_gkp = await sqlite_async.with_connection(
            "./data/inner_memory.db", _ks_read, row_factory=sqlite3.Row)
        return _ok({
            "profiles": profiles,
            "recent_encounters": encounters,
            "total_encounters": total_encounters,
            "total_great_kin_pulses": total_gkp,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/kin-society: %s", e)
        return _error(str(e))


# ═══════════════════════════════════════════════════════════════════
# DIALOGUE COMPOSITION — Titan speaks in his own words
# ═══════════════════════════════════════════════════════════════════

async def _get_dialogue_state() -> tuple:
    """Get felt_state + vocabulary for DialogueComposer.

    Reads directly from DBs (consciousness lives in spirit_worker process).
    Returns (felt_state_list, vocabulary_list).
    """
    # 1. Get 132D felt-state from consciousness DB
    felt_state = []
    try:
        _dc_row = await sqlite_async.query(
            "./data/consciousness.db",
            "SELECT state_vector FROM epochs ORDER BY epoch_id DESC LIMIT 1",
            fetch="one")
        if _dc_row and _dc_row[0]:
            _raw = _dc_row[0]
            if isinstance(_raw, str):
                _raw = json.loads(_raw)
            felt_state = _raw if isinstance(_raw, list) else []
    except Exception:
        pass

    # 2. Get vocabulary with felt_tensors from inner_memory DB
    vocabulary = []
    try:
        rows = await sqlite_async.query(
            "./data/inner_memory.db",
            "SELECT word, word_type, confidence, felt_tensor FROM vocabulary "
            "WHERE confidence > 0.1 ORDER BY confidence DESC LIMIT 128")
        for _vr in rows:
            _ft = None
            if _vr[3]:
                try:
                    _ft = json.loads(_vr[3]) if isinstance(_vr[3], str) else _vr[3]
                except Exception:
                    pass
            vocabulary.append({
                "word": _vr[0], "word_type": _vr[1],
                "confidence": _vr[2], "felt_tensor": _ft,
            })
    except Exception:
        pass

    return felt_state, vocabulary


@router.post("/v4/compose-reply")
async def compose_reply(request: Request):
    """Test endpoint: Titan composes a reply from felt-state.

    Send a message, get Titan's own composed response (no LLM).
    """
    try:
        payload = await request.json()
        message = payload.get("message", "")

        felt_state, vocabulary = await _get_dialogue_state()

        if not felt_state or not vocabulary:
            return _ok({
                "composed": False,
                "reason": "no_state" if not felt_state else "no_vocabulary",
            })

        # Extract simple signals from message for intent detection
        from titan_plugin.logic.interface_input import InputExtractor
        extractor = InputExtractor()
        signals = extractor.extract(message, "test")

        # Map signals to hormone-like shifts for intent detection
        _valence = signals.get("valence", 0.0)
        hormone_shifts = {
            "EMPATHY": max(0, _valence) * 0.2,
            "CURIOSITY": signals.get("engagement", 0) * 0.2,
            "CREATIVITY": 0.0,
            "REFLECTION": max(0, -_valence) * 0.1,
        }

        from titan_plugin.logic.dialogue_composer import DialogueComposer
        composer = DialogueComposer()
        result = composer.compose_response(
            felt_state=felt_state,
            vocabulary=vocabulary,
            hormone_shifts=hormone_shifts,
            message_keywords=message.lower().split()[:10],
            max_level=7,
        )

        return _ok({
            "composed": result.get("composed", False),
            "response": result.get("response", ""),
            "intent": result.get("intent", ""),
            "confidence": result.get("confidence", 0.0),
            "level": result.get("level", 0),
            "words_used": result.get("words_used", []),
            "message": message,
            "vocabulary_size": len(vocabulary),
            "state_dims": len(felt_state),
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/compose-reply: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/persona-telemetry — Persona conversation telemetry for Observatory
# ---------------------------------------------------------------------------
@router.get("/v4/persona-telemetry")
async def get_v4_persona_telemetry(request: Request):
    """Persona telemetry: recent conversation sessions with neuromod deltas.

    Query params:
      - limit: max entries (default 50)
      - titan: filter by titan_id (T1/T2/T3)
    """
    import json as _json_pt
    TELEMETRY_FILE = "./data/persona_telemetry.jsonl"
    try:
        limit = int(request.query_params.get("limit", "50"))
        titan_filter = request.query_params.get("titan", "")

        entries = []
        try:
            with open(TELEMETRY_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = _json_pt.loads(line)
                        if titan_filter and entry.get("titan") != titan_filter:
                            continue
                        entries.append(entry)
                    except Exception:
                        continue
        except FileNotFoundError:
            pass

        # Return most recent entries
        entries = entries[-limit:]

        # Summary stats
        total = len(entries)
        by_type = {}
        for e in entries:
            st = e.get("session_type", "unknown")
            by_type[st] = by_type.get(st, 0) + 1

        # Jailbreak alerts
        alerts = []
        try:
            with open("./data/jailbreak_alerts.json") as f:
                alerts = _json_pt.load(f)
        except Exception:
            pass

        return _ok({
            "total_entries": total,
            "by_session_type": by_type,
            "jailbreak_alerts": len(alerts),
            "entries": entries,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/persona-telemetry error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/persona-profiles — Persona profile data for Observatory tooltips
# ---------------------------------------------------------------------------
@router.get("/v4/persona-profiles")
async def get_v4_persona_profiles(request: Request):
    """Persona profiles + adversary attack descriptions for tooltips."""
    import json as _json_pp
    try:
        profiles = {}
        for name in ["companions", "visitors", "adversaries"]:
            path = f"./data/persona_profiles/{name}.json"
            try:
                with open(path) as f:
                    profiles[name] = _json_pp.load(f)
            except Exception:
                profiles[name] = {}

        # Load attack categories for adversary tooltips
        attacks = {}
        import glob
        for path in glob.glob("./data/adversary_attacks/*.json"):
            try:
                with open(path) as f:
                    data = _json_pp.load(f)
                category = os.path.basename(path).replace(".json", "")
                attacks[category] = {
                    "category": category,
                    "count": len(data) if isinstance(data, list) else 0,
                    "description": data[0].get("description", "") if isinstance(data, list) and data else "",
                }
            except Exception:
                pass

        return _ok({
            "profiles": profiles,
            "attack_categories": attacks,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/persona-profiles error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/social-pressure — Social pressure meter state for persona system
# ---------------------------------------------------------------------------
@router.get("/v4/social-pressure")
async def get_v4_social_pressure(request: Request):
    """Social pressure meter state for persona system."""
    plugin = _get_plugin(request)
    try:
        coordinator = await _get_cached_coordinator_async(plugin)
        sp = coordinator.get("social_pressure", {})
        return _ok(sp)
    except Exception as e:
        logger.error("[Dashboard] /v4/social-pressure error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/social-relief — Relieve social pressure from persona conversation
# ---------------------------------------------------------------------------
@router.post("/v4/social-relief")
async def post_v4_social_relief(request: Request):
    """Relieve social pressure from persona conversation."""
    plugin = _get_plugin(request)
    try:
        body = await request.json()
        relief = float(body.get("relief", 0.0))
        if relief <= 0:
            return _error("relief must be positive")

        # Route via spirit proxy QUERY → _handle_query → social_relief action
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _error("spirit proxy not available")
        result = spirit_proxy._bus.request(
            "spirit_proxy", "spirit",
            {"action": "social_relief", "payload": {"relief": relief}},
            timeout=5.0,
            reply_queue=spirit_proxy._reply_queue,
        )
        payload = result.get("payload", {}) if result else {}
        return _ok({"relief_applied": relief, "result": payload})
    except Exception as e:
        logger.error("[Dashboard] /v4/social-relief error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/signal-concept — Signal MSL concept convergence event
# ---------------------------------------------------------------------------
@router.post("/v4/signal-concept")
async def post_v4_signal_concept(request: Request):
    """Signal a concept convergence event for MSL grounding.

    Body: {"concept": "YES"|"NO"|"YOU"|"WE"|"THEY", "quality": 0.0-1.0,
           "extra": {...optional context...}}
    """
    plugin = _get_plugin(request)
    try:
        body = await request.json()
        concept = body.get("concept", "").upper()
        quality = float(body.get("quality", 0.5))
        extra = body.get("extra", {})

        valid = ["I", "YES", "NO", "YOU", "WE", "THEY"]
        if concept not in valid:
            return _error(f"concept must be one of {valid}", code=400)
        if not (0.0 <= quality <= 1.0):
            return _error("quality must be 0.0-1.0", code=400)

        # Route via spirit proxy QUERY → _handle_query → signal_concept action
        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _error("spirit proxy not available")
        result = spirit_proxy._bus.request(
            "spirit_proxy", "spirit",
            {"action": "signal_concept", "payload": {
                "concept": concept, "quality": quality, "extra": extra}},
            timeout=5.0,
            reply_queue=spirit_proxy._reply_queue,
        )
        if result:
            payload = result.get("payload", {})
            return _ok({"concept": concept, "quality": quality,
                        "event": payload})
        return _error("No response from spirit_worker")
    except Exception as e:
        logger.error("[Dashboard] /v4/signal-concept error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/signal-co-occurrence — Cross-concept reinforcement from same turn
# ---------------------------------------------------------------------------
@router.post("/v4/signal-co-occurrence")
async def post_v4_signal_co_occurrence(request: Request):
    """Signal co-occurring concepts from a single conversation turn.

    Body: {"concepts": ["I", "YOU", "YES", ...]}
    Reinforces interaction matrix for all concept pairs.
    """
    plugin = _get_plugin(request)
    try:
        body = await request.json()
        concepts = body.get("concepts", [])
        if not concepts or len(concepts) < 2:
            return _error("need at least 2 concepts")

        spirit_proxy = plugin._proxies.get("spirit")
        if not spirit_proxy:
            return _error("spirit proxy not available")
        result = spirit_proxy._bus.request(
            "spirit_proxy", "spirit",
            {"action": "signal_co_occurrence", "payload": {
                "concepts": concepts}},
            timeout=5.0,
            reply_queue=spirit_proxy._reply_queue,
        )
        if result:
            return _ok(result.get("payload", {}))
        return _error("No response from spirit_worker")
    except Exception as e:
        logger.error("[Dashboard] /v4/signal-co-occurrence error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/social-perception — Events Teacher → bus bridge
# ---------------------------------------------------------------------------
@router.post("/v4/social-perception")
async def post_v4_social_perception(request: Request):
    """Bridge social perception events from Events Teacher cron into DivineBus.

    Events Teacher runs out-of-process (cron). This endpoint receives events
    that passed the perturbation gate and publishes SOCIAL_PERCEPTION bus
    messages for spirit_worker (neuromod contagion), language_worker (CGN
    social grounding), and body_worker (outer_mind enrichment).
    """
    try:
        plugin = _get_plugin(request)
        body = await request.json()
        events = body.get("events", [])
        titan_id = body.get("titan_id", "T1")

        if not events:
            return _ok({"published": 0})

        from titan_plugin.bus import make_msg

        published = 0
        for evt in events[:10]:  # Safety cap: max 10 per window

            # CGN v2: Handle CGN_TRANSITION payloads from external scripts
            # (persona_social, arc_play send transitions through this endpoint)
            cgn_t = evt.get("cgn_transition")
            if cgn_t:
                plugin.bus.publish(make_msg(
                    "CGN_TRANSITION", "api", "cgn", {
                        "type": "outcome",
                        "consumer": cgn_t.get("consumer", "social"),
                        "concept_id": cgn_t.get("concept_id", ""),
                        "reward": float(cgn_t.get("reward", 0.0)),
                        "outcome_context": cgn_t.get("outcome_context", {}),
                    }))
                published += 1
                # ── META-CGN producers #11 + #12: social.session_low_qual /
                # session_high_qual (rFP § 12 rows 11-12). Main-process emission
                # path (emit_meta_cgn_signal duck-types plugin.bus by .publish).
                # Fires when persona_social_v2 POSTs a session transition with
                # quality score. Edge-detected per session_id (concept_id) so
                # each session emits at most once for low and once for high.
                # MONOCULTURE-AWARE rebalance on P12 (was FORMULATE 0.55, SYN 0.55
                # in rFP — changed to 0.20 + 0.65 per P7/P8 pattern).
                if cgn_t.get("consumer") == "social":
                    try:
                        _p11p12_quality = float(cgn_t.get(
                            "outcome_context", {}).get("quality", 0.5))
                        _p11p12_sid = str(cgn_t.get("concept_id", "?"))[:40]
                        _p11p12_det = _get_persona_session_detector()
                        from titan_plugin.bus import emit_meta_cgn_signal
                        # P11 low_qual: observe(sid, 1-quality, 0.5) → quality ≤ 0.5
                        if _p11p12_det.observe(
                                f"low:{_p11p12_sid}", 1.0 - _p11p12_quality, 0.5):
                            _p11_sent = emit_meta_cgn_signal(
                                plugin.bus,
                                src="persona",
                                consumer="social",
                                event_type="session_low_qual",
                                intensity=min(1.0, 1.0 - _p11p12_quality),
                                domain=_p11p12_sid,
                                reason=f"persona session {_p11p12_sid} quality={_p11p12_quality:.2f} (low)",
                            )
                            _save_persona_session_detector()
                            if _p11_sent:
                                logger.info(
                                    "[META-CGN] social.session_low_qual EMIT — "
                                    "sid=%s quality=%.2f", _p11p12_sid, _p11p12_quality)
                            else:
                                logger.warning(
                                    "[META-CGN] Producer #11 social.session_low_qual DROPPED "
                                    "— sid=%s quality=%.2f (rate-gate or bus-full)",
                                    _p11p12_sid, _p11p12_quality)
                        # P12 high_qual: observe(sid, quality, 0.8) → quality ≥ 0.8
                        if _p11p12_det.observe(
                                f"high:{_p11p12_sid}", _p11p12_quality, 0.8):
                            _p12_sent = emit_meta_cgn_signal(
                                plugin.bus,
                                src="persona",
                                consumer="social",
                                event_type="session_high_qual",
                                intensity=min(1.0, _p11p12_quality),
                                domain=_p11p12_sid,
                                reason=f"persona session {_p11p12_sid} quality={_p11p12_quality:.2f} (high)",
                            )
                            _save_persona_session_detector()
                            if _p12_sent:
                                logger.info(
                                    "[META-CGN] social.session_high_qual EMIT — "
                                    "sid=%s quality=%.2f", _p11p12_sid, _p11p12_quality)
                            else:
                                logger.warning(
                                    "[META-CGN] Producer #12 social.session_high_qual DROPPED "
                                    "— sid=%s quality=%.2f (rate-gate or bus-full)",
                                    _p11p12_sid, _p11p12_quality)
                        # COMPLETE-4 (2026-04-19): emit META_PERSONA_REWARD bus
                        # msg so spirit_worker's meta_engine can blend this
                        # persona session quality into chain_iql via
                        # apply_external_reward on the most-recently-concluded
                        # chain. Same pattern as Language's META_LANGUAGE_REWARD
                        # (chain_iql Q-net learns from downstream outcomes).
                        # chain_id omitted — handler resolves via
                        # meta_engine.get_last_chain_id(); quality is in [0,1].
                        try:
                            plugin.bus.publish(make_msg(
                                "META_PERSONA_REWARD", "persona", "spirit", {
                                    "quality": float(_p11p12_quality),
                                    "session_id": _p11p12_sid,
                                }))
                        except Exception as _p_er_err:
                            logger.debug(
                                "[META] Persona external_reward emit error: %s",
                                _p_er_err)
                    except Exception as _p11p12_err:
                        logger.warning(
                            "[META-CGN] Producers #11/#12 social session emit FAILED "
                            "— sid=%s err=%s (signal missed)",
                            cgn_t.get("concept_id", "?"), _p11p12_err)
                continue

            # Validate required fields for social perception events
            if not evt.get("contagion_type") or not evt.get("felt_summary"):
                continue

            plugin.bus.publish(make_msg(
                "SOCIAL_PERCEPTION", "events_teacher", "all", {
                    "titan_id": titan_id,
                    "topic": evt.get("topic", ""),
                    "sentiment": float(evt.get("sentiment", 0.0)),
                    "arousal": float(evt.get("arousal", 0.0)),
                    "relevance": float(evt.get("relevance", 0.0)),
                    "concept_signals": evt.get("concept_signals", []),
                    "felt_summary": evt.get("felt_summary", ""),
                    "contagion_type": evt.get("contagion_type"),
                    "author": evt.get("author", ""),
                    "source": evt.get("source", ""),
                    "perturbation": float(evt.get("perturbation", 0.0)),
                    "social_vocab_candidates": evt.get("social_vocab_candidates", []),
                }))
            published += 1

        if published > 0:
            logger.info("[SocialPerception] Published %d events to bus "
                        "(titan=%s)", published, titan_id)

        return _ok({"published": published})

    except Exception as e:
        logger.error("[Dashboard] /v4/social-perception error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# POST /v4/social-delegate — T2/T3 delegate X posts through T1 gateway
# ---------------------------------------------------------------------------
@router.post("/v4/social-delegate")
async def post_v4_social_delegate(request: Request):
    """Accept delegated X post from T2/T3 for centralized posting via SocialXGateway v3.

    v3 payload: T2/T3 sends raw inner state. T1's gateway generates + posts.
    Body: {
        "titan_id": "T2"|"T3",
        "auth_token": "shared_kin_secret",
        "vocabulary_count": 159,
        "composition_confidence": 0.7,
        // v3 fields — raw inner state for gateway LLM coloring
        "emotion": "wonder",
        "neuromods": {"DA": 0.6, "5HT": 0.8, ...},
        "epoch": 300000,
        "pi_ratio": 0.14,
        "grounded_words": [{"word": "feel", "confidence": 0.8}, ...],
        "catalysts": [{"type": "emotion_shift", "significance": 0.5, ...}],
        // Legacy fields (backward compat — ignored if v3 fields present)
        "text": "...",
        "post_type": "original",
        "catalyst_type": "...",
        "state_signature": "..."
    }
    """
    import json
    import os
    import time as _time

    QUEUE_FILE = "./data/social_delegate_queue.json"
    plugin = _get_plugin(request)
    try:
        body = await request.json()
        titan_id = body.get("titan_id", "")
        auth_token = body.get("auth_token", "")
        text = body.get("text", "")
        post_type = body.get("post_type", "original")
        art_path = body.get("art_path")
        composition_confidence = float(body.get("composition_confidence", 0))
        vocabulary_count = int(body.get("vocabulary_count", 0))

        # Load delegate config
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            with open("titan_plugin/titan_params.toml", "rb") as f:
                _params = tomllib.load(f)
            _dcfg = _params.get("social_delegate", {})
        except Exception:
            _dcfg = {}

        if not _dcfg.get("enabled", False):
            return _error("social delegation disabled", code=403)

        # Auth check
        expected_secret = _dcfg.get("kin_secret", "")
        if not expected_secret or auth_token != expected_secret:
            return _error("invalid auth_token", code=401)

        # Titan ID check
        accepted = _dcfg.get("accepted_titans", [])
        if titan_id not in accepted:
            return _error(f"titan_id must be one of {accepted}", code=403)

        # Quality gates
        min_vocab = _dcfg.get("min_vocabulary", 50)
        min_conf = _dcfg.get("min_confidence", 0.5)
        art_only_thresh = _dcfg.get("art_only_vocabulary_threshold", 100)

        if vocabulary_count < min_vocab and post_type != "art_share":
            return _ok({"accepted": False,
                        "reason": f"vocabulary too small ({vocabulary_count} < {min_vocab})"})
        if composition_confidence < min_conf and post_type != "art_share":
            return _ok({"accepted": False,
                        "reason": f"confidence too low ({composition_confidence} < {min_conf})"})
        if vocabulary_count < art_only_thresh and post_type == "original" and not text:
            return _ok({"accepted": False,
                        "reason": "art-only mode: vocabulary below threshold, text required"})

        # Load queue
        queue = []
        if os.path.exists(QUEUE_FILE):
            try:
                with open(QUEUE_FILE) as f:
                    queue = json.load(f)
            except Exception:
                queue = []

        # Check per-Titan daily limit
        max_per_day = _dcfg.get("max_posts_per_titan_per_day", 2)
        today_start = _time.time() - (_time.time() % 86400)
        titan_today = sum(1 for q in queue
                          if q.get("titan_id") == titan_id
                          and q.get("queued_at", 0) > today_start)
        if titan_today >= max_per_day:
            return _ok({"accepted": False,
                        "reason": f"daily limit reached ({titan_today}/{max_per_day})"})

        # Queue size check
        max_queue = _dcfg.get("max_queue_size", 10)
        if len(queue) >= max_queue:
            # Drop oldest
            queue = queue[-(max_queue - 1):]

        # Enqueue — v3 carries raw inner state for gateway LLM coloring
        entry = {
            "titan_id": titan_id,
            "post_type": post_type,
            "text": text,  # Legacy (pre-generated). Empty if v3 state provided.
            "art_path": art_path,
            "composition_confidence": composition_confidence,
            "vocabulary_count": vocabulary_count,
            "catalyst_type": body.get("catalyst_type", "delegate"),
            "state_signature": body.get("state_signature", ""),
            "queued_at": _time.time(),
            # v3 raw state fields (T1 gateway generates text from these)
            "emotion": body.get("emotion", ""),
            "neuromods": body.get("neuromods", {}),
            "epoch": body.get("epoch", 0),
            "pi_ratio": body.get("pi_ratio", 0),
            "grounded_words": body.get("grounded_words", []),
            "catalysts": body.get("catalysts", []),
        }
        queue.append(entry)

        # Save
        os.makedirs(os.path.dirname(QUEUE_FILE) or ".", exist_ok=True)
        tmp = QUEUE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(queue, f)
        os.replace(tmp, QUEUE_FILE)

        logger.info("[DELEGATE] Queued %s post from %s (pos=%d): %s",
                    post_type, titan_id, len(queue), text[:60] if text else "(art)")
        return _ok({"accepted": True, "queued_position": len(queue)})

    except Exception as e:
        logger.error("[Dashboard] /v4/social-delegate error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/social-delegate-queue — View delegate queue status
# ---------------------------------------------------------------------------
@router.get("/v4/social-delegate-queue")
async def get_v4_social_delegate_queue(request: Request):
    """Get current delegate queue status."""
    import json
    import os
    QUEUE_FILE = "./data/social_delegate_queue.json"
    try:
        if os.path.exists(QUEUE_FILE):
            with open(QUEUE_FILE) as f:
                queue = json.load(f)
        else:
            queue = []
        return _ok({
            "queue_length": len(queue),
            "entries": [{"titan_id": q.get("titan_id"), "post_type": q.get("post_type"),
                         "text_preview": (q.get("text", "")[:50] + "..."
                                          if q.get("text") else "(art)"),
                         "queued_at": q.get("queued_at")}
                        for q in queue]
        })
    except Exception as e:
        return _error(str(e))


# =====================================================================
# KNOWLEDGE WORKER — Knowledge acquisition & research
# =====================================================================

@router.post("/v4/knowledge-request")
async def post_v4_knowledge_request(request: Request):
    """Trigger a knowledge acquisition request.

    Body: {"topic": "quantum entanglement", "urgency": 0.7}
    Routes CGN_KNOWLEDGE_REQ to the Knowledge Worker via bus.
    """
    plugin = _get_plugin(request)
    try:
        body = await request.json()
        topic = body.get("topic", "").strip()
        if not topic:
            return _error("topic is required")
        urgency = float(body.get("urgency", 0.5))

        # Send via bus to knowledge worker
        from ..bus import make_msg
        plugin.bus.publish(make_msg("CGN_KNOWLEDGE_REQ", "api", "knowledge", {
            "topic": topic,
            "requestor": "api",
            "urgency": urgency,
            "neuromods": {},
        }))
        return _ok({"status": "queued", "topic": topic, "urgency": urgency})
    except Exception as e:
        return _error(str(e))


@router.get("/v4/knowledge-stats")
async def get_v4_knowledge_stats(request: Request):
    """Get Knowledge Worker stats and recent concepts."""
    # Phase E.2.3 fix: sqlite3 in async endpoint blocks event loop
    try:
        db_path = "./data/inner_memory.db"
        _day_ago = time.time() - 86400

        def _knowledge_stats(conn):
            row = conn.execute(
                "SELECT COUNT(*), AVG(confidence), SUM(times_used) "
                "FROM knowledge_concepts"
            ).fetchone()
            recent = conn.execute(
                "SELECT topic, confidence, source, quality_score, times_used, "
                "requesting_consumer, created_at "
                "FROM knowledge_concepts ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            avg_quality = conn.execute(
                "SELECT AVG(quality_score) FROM knowledge_concepts"
            ).fetchone()[0] or 0
            source_rows = conn.execute(
                "SELECT source, COUNT(*) as cnt FROM knowledge_concepts "
                "GROUP BY source ORDER BY cnt DESC"
            ).fetchall()
            top_usage = conn.execute(
                "SELECT topic, times_used, confidence FROM knowledge_concepts "
                "WHERE times_used > 0 ORDER BY times_used DESC LIMIT 5"
            ).fetchall()
            top_conf = conn.execute(
                "SELECT topic, confidence, quality_score FROM knowledge_concepts "
                "ORDER BY confidence DESC LIMIT 5"
            ).fetchall()
            acq_count = conn.execute(
                "SELECT COUNT(*) FROM knowledge_concepts WHERE created_at > ?",
                (_day_ago,)
            ).fetchone()[0] or 0
            return row, recent, avg_quality, source_rows, top_usage, top_conf, acq_count

        row, recent, avg_quality, source_rows, top_usage, top_conf, acq_count = (
            await sqlite_async.with_connection(
                db_path, _knowledge_stats, timeout=2.0, row_factory=sqlite3.Row)
        )
        total = row[0] or 0
        avg_conf = round(row[1] or 0, 3)
        total_usage = row[2] or 0
        concepts = [{
            "topic": r["topic"],
            "confidence": round(r["confidence"], 3),
            "source": r["source"],
            "quality_score": round(r["quality_score"], 3),
            "times_used": r["times_used"],
            "requesting_consumer": r["requesting_consumer"],
            "created_at": r["created_at"],
        } for r in recent]
        source_dist = {r["source"]: r["cnt"] for r in source_rows}
        top_by_usage = [{"topic": r["topic"], "times_used": r["times_used"],
                         "confidence": round(r["confidence"], 3)} for r in top_usage]
        top_by_confidence = [{"topic": r["topic"],
                              "confidence": round(r["confidence"], 3),
                              "quality_score": round(r["quality_score"], 3)}
                             for r in top_conf]
        return _ok({
            "total_concepts": total,
            "avg_confidence": avg_conf,
            "avg_quality_score": round(avg_quality, 3),
            "total_usage": total_usage,
            "source_distribution": source_dist,
            "top_by_usage": top_by_usage,
            "top_by_confidence": top_by_confidence,
            "acquisition_rate_24h": acq_count,
            "recent_concepts": concepts,
        })
    except sqlite3.OperationalError:
        # Table doesn't exist yet (knowledge worker hasn't booted)
        return _ok({
            "total_concepts": 0, "avg_confidence": 0,
            "total_usage": 0, "recent_concepts": [],
            "note": "knowledge_concepts table not yet created"
        })
    except Exception as e:
        return _error(str(e))


@router.get("/v4/knowledge-search")
async def get_v4_knowledge_search(request: Request, topic: str = ""):
    """Search internal knowledge for a topic."""
    if not topic.strip():
        return _error("topic query parameter required")
    # Phase E.2.3 fix: sqlite3 in async endpoint blocks event loop
    import asyncio
    import sqlite3
    try:
        db_path = "./data/inner_memory.db"

        rows = await sqlite_async.query(
            db_path,
            "SELECT topic, summary, confidence, source, quality_score, "
            "times_used, encounter_count "
            "FROM knowledge_concepts WHERE topic LIKE ? "
            "ORDER BY confidence DESC LIMIT 10",
            (f"%{topic}%",),
            timeout=2.0,
            row_factory=sqlite3.Row,
        )
        return _ok({
            "query": topic,
            "results": [{
                "topic": r["topic"],
                "summary": r["summary"][:200] if r["summary"] else "",
                "confidence": round(r["confidence"], 3),
                "source": r["source"],
                "quality_score": round(r["quality_score"], 3),
                "times_used": r["times_used"],
                "encounter_count": r["encounter_count"],
            } for r in rows]
        })
    except sqlite3.OperationalError:
        return _ok({"query": topic, "results": []})
    except Exception as e:
        return _error(str(e))


# GET /v4/cgn-haov-stats — Per-consumer HAOV hypothesis testing statistics
@router.get("/v4/cgn-haov-stats")
async def get_v4_cgn_haov_stats(request: Request):
    """Get per-consumer HAOV (hypothesis testing) statistics.

    Returns hypothesis counts, confirmation rates, verified rules,
    and epistemic style classification per consumer.
    """
    try:
        # Read HAOV stats from CGN's JSON sidecar (torch-free).
        # CGN worker saves haov_stats.json alongside cgn_state.pt.
        # Fallback: read cgn_state.pt via pickle (torch.load is just pickle
        # + tensor deserialization — for non-tensor data, pickle suffices).
        import json as _json
        haov_json_path = "data/cgn/haov_stats.json"
        state_path = "data/cgn/cgn_state.pt"

        haov_data = {}
        if os.path.exists(haov_json_path):
            with open(haov_json_path) as _hf:
                haov_data = _json.load(_hf)
        elif os.path.exists(state_path):
            # Fallback: use pickle to read .pt file without importing torch
            import pickle
            with open(state_path, "rb") as _sf:
                try:
                    state = pickle.load(_sf)
                    haov_data = state.get("haov", {})
                except Exception:
                    return _ok({"consumers": {}, "note": "CGN state unreadable without torch"})
        else:
            return _ok({"consumers": {}, "note": "CGN state not yet saved"})

        result = {}
        for consumer, hdata in haov_data.items():
            stats = hdata.get("stats", {})
            verified = hdata.get("verified_rules", [])
            formed = stats.get("formed", 0)
            tested = stats.get("tested", 0)
            confirmed = stats.get("confirmed", 0)
            falsified = stats.get("falsified", 0)

            # Derive epistemic style from confirmation/falsification ratio
            if tested == 0:
                style = "nascent"
            elif confirmed / max(1, tested) > 0.7:
                style = "confirmatory"
            elif falsified / max(1, tested) > 0.5:
                style = "exploratory"
            else:
                style = "balanced"

            result[consumer] = {
                "formed": formed,
                "tested": tested,
                "confirmed": confirmed,
                "falsified": falsified,
                "used_for_action": stats.get("used_for_action", 0),
                "verified_rules_count": len(verified),
                "confirmation_rate": round(confirmed / max(1, tested), 3),
                "epistemic_style": style,
                "top_rules": [
                    {"rule": r["rule"],
                     "confidence": round(r.get("confidence", 0), 2),
                     "tests": r.get("tests", 0),
                     "confirmations": r.get("confirmations", 0)}
                    for r in sorted(verified,
                                    key=lambda x: x.get("confidence", 0),
                                    reverse=True)[:5]
                ],
            }

        return _ok({"consumers": result})
    except Exception as e:
        logger.error("[Dashboard] /v4/cgn-haov-stats error: %s", e)
        return _error(str(e))


# =====================================================================
# TIMECHAIN — Proof of Thought Memory Architecture
# =====================================================================

# Cached TimeChain instance — avoids per-request init + reconciliation overhead.
# Read-only queries use this; the worker process has its own write instance.
_tc_cache = {"instance": None, "created_at": 0}
_TC_CACHE_TTL = 60  # refresh every 60s to pick up new fork tips

def _get_cached_tc():
    """Get or create a cached read-only TimeChain instance."""
    import time as _time
    now = _time.time()
    if _tc_cache["instance"] is None or (now - _tc_cache["created_at"]) > _TC_CACHE_TTL:
        from titan_plugin.logic.timechain import TimeChain
        _tc_cache["instance"] = TimeChain(data_dir="data/timechain", titan_id="T1")
        _tc_cache["created_at"] = now
    return _tc_cache["instance"]


# GET /v4/timechain/status — Chain status, fork stats, PoT acceptance rate
@router.get("/v4/timechain/status")
async def get_v4_timechain_status(request: Request):
    """Get TimeChain status: genesis, forks, blocks, Merkle root, PoT stats."""
    import asyncio
    try:
        tc = _get_cached_tc()
        # Phase E.2 perf: merkle compute + file I/O — wrap to_thread.
        status = await asyncio.to_thread(tc.get_chain_status)
        return _ok(status)
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/status error: %s", e)
        return _error(str(e))


# GET /v4/timechain/blocks — Recent blocks on a fork
@router.get("/v4/timechain/blocks")
async def get_v4_timechain_blocks(request: Request, fork: int = 0,
                                   limit: int = 20):
    """Get recent blocks on a fork."""
    import asyncio
    try:
        tc = _get_cached_tc()
        # Phase E.2 perf: file I/O — wrap to_thread.
        blocks = await asyncio.to_thread(tc.get_recent_blocks, fork, min(limit, 100))
        return _ok({"fork_id": fork, "blocks": blocks})
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/blocks error: %s", e)
        return _error(str(e))


# GET /v4/timechain/verify — Chain integrity check
@router.get("/v4/timechain/verify")
async def get_v4_timechain_verify(request: Request):
    """Verify hash chain integrity across all forks."""
    import asyncio
    try:
        tc = _get_cached_tc()
        # Phase E.2 perf: verify_all walks entire chain — wrap to_thread.
        valid, results = await asyncio.to_thread(tc.verify_all)
        return _ok({"valid": valid, "results": results})
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/verify error: %s", e)
        return _error(str(e))


# GET /v4/timechain/pot-stats — Proof of Thought validation statistics
@router.get("/v4/timechain/pot-stats")
async def get_v4_timechain_pot_stats(request: Request):
    """Get PoT acceptance rate, chi spent, rejection reasons from running worker."""
    try:
        plugin = _get_plugin(request)
        from titan_plugin.bus import make_msg
        import asyncio
        # Query the running timechain worker for live PoT stats
        rid = f"pot_stats_{time.time()}"
        plugin.bus.publish(make_msg(
            "QUERY", "dashboard", "timechain",
            {"action": "timechain_status", "rid": rid}
        ))
        # Read from chain files directly as fallback.
        # Phase E.2 perf: tc.get_chain_status() does merkle root compute
        # across all fork tips + file I/O — sync, blocks event loop ~4s.
        tc = _get_cached_tc()
        status = await asyncio.to_thread(tc.get_chain_status)
        # Compute derived PoT stats from block data
        total = status.get("total_blocks", 0)
        chi = status.get("total_chi_spent", 0)
        forks = status.get("forks", {})
        by_source = {}
        try:
            rows = await sqlite_async.query(
                "data/timechain/index.db",
                "SELECT source, COUNT(*), SUM(chi_spent), AVG(significance) "
                "FROM block_index GROUP BY source ORDER BY COUNT(*) DESC",
            )
            for src, cnt, chi_s, avg_sig in rows:
                by_source[src or "unknown"] = {
                    "blocks": cnt, "chi_spent": round(chi_s or 0, 4),
                    "avg_significance": round(avg_sig or 0, 3),
                }
        except Exception:
            pass
        return _ok({
            "total_blocks": total,
            "total_chi_spent": round(chi, 4),
            "avg_chi_per_block": round(chi / max(1, total), 6),
            "blocks_by_source": by_source,
            "forks_active": sum(1 for f in forks.values() if f.get("block_count", 0) > 0),
            "forks_total": len(forks),
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/pot-stats error: %s", e)
        return _error(str(e))


# GET /v4/timechain/fork-tree — Fork tree structure for visualization
@router.get("/v4/timechain/fork-tree")
async def get_v4_timechain_fork_tree(request: Request):
    """Get fork tree data for Observatory visualization."""
    import asyncio
    try:
        tc = _get_cached_tc()
        # Phase E.2 perf: wrap fork stats compute in to_thread.
        stats = await asyncio.to_thread(tc.get_fork_stats)
        nodes = []
        for fid, f in sorted(stats.items()):
            nodes.append({
                "id": fid,
                "name": f["name"],
                "type": f["type"],
                "topic": f.get("topic"),
                "blocks": f["block_count"],
                "tip": f["tip_height"],
                "chi": f["total_chi_spent"],
                "avg_significance": f["avg_significance"],
                "compacted": f["compacted"],
            })
        return _ok({
            "genesis_hash": tc.genesis_hash.hex() if tc.has_genesis else None,
            "merkle_root": tc.compute_merkle_root().hex(),
            "total_blocks": tc.total_blocks,
            "nodes": nodes,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/fork-tree error: %s", e)
        return _error(str(e))


# GET /v4/developmental-timeline — Genesis chain milestones for Observatory
@router.get("/v4/developmental-timeline")
async def get_v4_developmental_timeline(request: Request, days: int = 30):
    """Query genesis chain for developmental state snapshots over time. Async."""
    import asyncio

    def _fetch():
        tc = _get_cached_tc()
        blocks = tc.query_blocks(
            thought_type="genesis", fork_id=0,
            limit=min(days * 10, 500))
        timeline = []
        for b in blocks:
            try:
                bh = b.get("block_hash", "")
                if isinstance(bh, str):
                    bh = bytes.fromhex(bh)
                block = tc.get_block_by_hash(bh)
                if not block or not hasattr(block, "payload"):
                    continue
                content = block.payload.content or {}
                state = content.get("state", {})
                timeline.append({
                    "epoch": b.get("epoch_id"),
                    "timestamp": b.get("timestamp"),
                    "height": b.get("block_height"),
                    "trigger": content.get("sealed_by", ""),
                    "vocab_size": state.get("vocab_size", 0),
                    "productive_vocab": state.get("productive_vocab", 0),
                    "i_confidence": state.get("i_confidence", 0),
                    "pi_rate": state.get("pi_rate", 0),
                    "cluster_count": state.get("cluster_count", 0),
                    "dream_cycles": state.get("dream_cycles", 0),
                    "meta_chains": state.get("meta_chains", 0),
                    "reasoning_commit_rate": state.get("reasoning_commit_rate", 0),
                    "emotion": state.get("emotion", ""),
                    "sol_balance": state.get("sol_balance", 0),
                    "neuromods": state.get("neuromods", {}),
                    "cognitive_work": content.get("cognitive_work", {}),
                    "proof_hash": content.get("proof_hash", ""),
                    "fork_heartbeat_count": len(content.get("fork_heartbeats", [])),
                })
            except Exception:
                continue
        timeline.sort(key=lambda x: x.get("epoch", 0), reverse=True)
        return timeline, tc.genesis_hash.hex() if tc.has_genesis else None

    try:
        timeline, genesis_hash = await asyncio.to_thread(_fetch)
        return _ok({
            "timeline": timeline,
            "count": len(timeline),
            "genesis_hash": genesis_hash,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/developmental-timeline error: %s", e)
        return _error(str(e))


# GET /v4/timechain/contracts — List all smart contracts
@router.get("/v4/timechain/contracts")
async def get_v4_timechain_contracts(request: Request,
                                      contract_type: str = None,
                                      status: str = None):
    """List TimeChain smart contracts (Phase 3a). Async to avoid blocking."""
    import asyncio

    def _fetch():
        tc = _get_cached_tc()
        blocks = tc.query_blocks(
            thought_type="contract_deploy", fork_id=4, limit=200)
        seen = {}
        for b in blocks:
            try:
                bh = b.get("block_hash", "")
                if isinstance(bh, str):
                    bh = bytes.fromhex(bh)
                block = tc.get_block_by_hash(bh)
                if not block or not hasattr(block, "payload"):
                    continue
                content = block.payload.content or {}
                cd = content.get("contract")
                if not cd:
                    continue
                cid = cd.get("contract_id", "")
                ver = cd.get("version", 0)
                if cid not in seen or ver > seen[cid].get("version", 0):
                    seen[cid] = cd
            except Exception:
                continue
        result = list(seen.values())
        if contract_type:
            result = [c for c in result if c.get("contract_type") == contract_type]
        if status:
            result = [c for c in result if c.get("status") == status]
        return result

    try:
        contracts = await asyncio.to_thread(_fetch)
        return _ok({"contracts": contracts, "count": len(contracts)})
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/contracts error: %s", e)
        return _error(str(e))


# GET /v4/timechain/contracts/pending — Contracts awaiting Maker approval (P3d)
# GET /v4/timechain/contracts/stats — Contract execution stats (P3b)
@router.get("/v4/timechain/contracts/stats")
async def get_v4_contracts_stats(request: Request):
    """Contract execution stats — reads LIVE data from orchestrator stats file."""
    import asyncio

    def _fetch():
        import json as _json
        # Read live stats written by Orchestrator every ~100s
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "timechain", "contract_stats.json")
        try:
            with open(stats_path) as f:
                stats = _json.load(f)
        except (FileNotFoundError, _json.JSONDecodeError):
            return {"total": 0, "by_type": {}, "by_status": {},
                    "active_contracts": [], "mempool": {}}

        contracts = stats.get("contracts", {})
        by_type = {}
        by_status = {}
        active_list = []
        for cid, info in contracts.items():
            t = info.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
            by_status["active"] = by_status.get("active", 0) + 1
            active_list.append({
                "contract_id": cid,
                "type": t,
                "execution_count": info.get("execution_count", 0),
                "last_executed": info.get("last_executed", 0),
                "description": info.get("description", ""),
            })
        return {
            "total": len(contracts),
            "by_type": by_type,
            "by_status": by_status,
            "active_contracts": active_list,
            "mempool": {
                "filter_evals": stats.get("mempool_filter_evals", 0),
                "filter_hits": stats.get("mempool_filter_hits", 0),
            },
        }

    try:
        result = await asyncio.to_thread(_fetch)
        return _ok(result)
    except Exception as e:
        return _error(str(e))


@router.get("/v4/timechain/contracts/pending")
async def get_v4_contracts_pending(request: Request):
    """List contracts pending Maker approval."""
    import asyncio

    def _fetch():
        tc = _get_cached_tc()
        blocks = tc.query_blocks(
            thought_type="contract_deploy", fork_id=4, limit=200)
        pending = []
        for b in blocks:
            try:
                bh = b.get("block_hash", "")
                if isinstance(bh, str):
                    bh = bytes.fromhex(bh)
                block = tc.get_block_by_hash(bh)
                if not block or not hasattr(block, "payload"):
                    continue
                cd = (block.payload.content or {}).get("contract")
                if cd and cd.get("status") == "pending_approval":
                    pending.append(cd)
            except Exception:
                continue
        return pending

    try:
        pending = await asyncio.to_thread(_fetch)
        return _ok({"contracts": pending, "count": len(pending)})
    except Exception as e:
        return _error(str(e))


# POST /v4/timechain/contracts/approve — Maker approves a pending contract (P3d)
@router.post("/v4/timechain/contracts/approve")
async def post_v4_contracts_approve(request: Request):
    """Approve a pending contract. Requires contract_id in body."""
    try:
        body = await request.json()
        contract_id = body.get("contract_id", "")
        if not contract_id:
            return _error("contract_id required")

        plugin = getattr(request.app.state, "titan_plugin", None)
        if not plugin or not hasattr(plugin, "bus"):
            return _error("plugin not available")

        reply = plugin.bus.request(
            "api", "timechain",
            {"action": "approve", "contract_id": contract_id},
            msg_type="CONTRACT_APPROVE",
            timeout=5.0,
        )
        if reply and reply.get("payload", {}).get("success"):
            return _ok({"status": "approved", "contract_id": contract_id})
        reason = (reply or {}).get("payload", {}).get("reason", "unknown")
        return _error(f"approval failed: {reason}")
    except Exception as e:
        return _error(str(e))


# POST /v4/timechain/contracts/veto — Maker rejects a pending contract (P3d)
@router.post("/v4/timechain/contracts/veto")
async def post_v4_contracts_veto(request: Request):
    """Veto a pending contract. Requires contract_id and optional reason in body."""
    try:
        body = await request.json()
        contract_id = body.get("contract_id", "")
        reason = body.get("reason", "")
        if not contract_id:
            return _error("contract_id required")

        plugin = getattr(request.app.state, "titan_plugin", None)
        if not plugin or not hasattr(plugin, "bus"):
            return _error("plugin not available")

        reply = plugin.bus.request(
            "api", "timechain",
            {"action": "veto", "contract_id": contract_id, "reason": reason},
            msg_type="CONTRACT_VETO",
            timeout=5.0,
        )
        if reply and reply.get("payload", {}).get("success"):
            return _ok({"status": "rejected", "contract_id": contract_id,
                        "reason": reason})
        err = (reply or {}).get("payload", {}).get("reason", "unknown")
        return _error(f"veto failed: {err}")
    except Exception as e:
        return _error(str(e))


# GET /v4/timechain/block — Single block detail by fork + height
@router.get("/v4/timechain/block")
async def get_v4_timechain_block(request: Request, fork: int = 0,
                                  height: int = 0):
    """Get full block detail (header + payload) by fork and height."""
    try:
        tc = _get_cached_tc()
        block = tc.get_block(fork, height)
        if not block:
            return _error(f"Block not found: fork={fork} height={height}")
        return _ok({
            "block_hash": block.block_hash_hex,
            "fork_id": block.header.fork_id,
            "height": block.header.block_height,
            "timestamp": block.header.timestamp,
            "epoch_id": block.header.epoch_id,
            "prev_hash": block.header.prev_hash.hex(),
            "payload_hash": block.header.payload_hash.hex(),
            "pot_nonce": block.header.pot_nonce,
            "chi_spent": block.header.chi_spent,
            "neuromod_hash": block.header.neuromod_hash.hex(),
            "cross_refs": [
                {"fork_id": r.fork_id, "block_height": r.block_height}
                for r in block.cross_refs
            ],
            "payload": {
                "thought_type": block.payload.thought_type,
                "source": block.payload.source,
                "content": block.payload.content,
                "significance": block.payload.significance,
                "confidence": block.payload.confidence,
                "tags": block.payload.tags,
                "db_ref": block.payload.db_ref,
            },
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/block error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/timechain/verify/{height} — Verify a conversation fork block (OVG Phase 3)
# ---------------------------------------------------------------------------
@router.get("/v4/timechain/verify/{height}")
async def get_v4_timechain_verify_block(request: Request, height: int):
    """Look up a verified output on the CONVERSATION fork by block height.

    Returns the OVG signature, output_hash, checks, merkle root, and genesis hash
    so external parties can independently verify Titan's signed output.
    """
    try:
        tc = _get_cached_tc()
        from titan_plugin.logic.timechain import FORK_CONVERSATION
        block = tc.get_block(FORK_CONVERSATION, height)
        if not block:
            return _error(f"Conversation block #{height} not found", code=404)

        content = block.payload.content if isinstance(block.payload.content, dict) else {}
        status = tc.get_chain_status()

        return _ok({
            "block_height": block.header.block_height,
            "timestamp": block.header.timestamp,
            "block_hash": block.block_hash_hex,
            "payload_hash": block.header.payload_hash.hex(),
            "chi_spent": block.header.chi_spent,
            "output_hash": content.get("output_hash", ""),
            "prompt_hash": content.get("prompt_hash", ""),
            "signature": content.get("signature", ""),
            "channel": content.get("channel", ""),
            "checks": content.get("checks", {}),
            "violation_type": content.get("violation_type", "none"),
            "titan_id": content.get("titan_id", ""),
            "genesis_hash": status.get("genesis_hash", ""),
            "merkle_root": status.get("merkle_root", ""),
            "fork": "conversation",
            "fork_id": FORK_CONVERSATION,
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/verify/%d error: %s", height, e)
        return _error(str(e))


# POST /v4/timechain/test-commit — Test commit a block to a specific fork
@router.post("/v4/timechain/test-commit")
async def post_v4_timechain_test_commit(request: Request):
    """Test: submit a TIMECHAIN_COMMIT via bus to verify routing works."""
    try:
        body = await request.json()
        fork = body.get("fork", "declarative")
        content = body.get("content", {"test": True})
        tags = body.get("tags", ["test"])

        plugin = _get_plugin(request)
        from titan_plugin.bus import make_msg
        plugin.bus.publish(make_msg(
            "TIMECHAIN_COMMIT", "dashboard", "timechain", {
                "fork": fork,
                "thought_type": fork if fork in ("declarative", "procedural", "episodic", "meta") else "declarative",
                "source": "test_api",
                "content": content,
                "significance": 0.5,
                "novelty": 0.7,
                "coherence": 0.5,
                "tags": tags,
                "neuromods": {"DA": 0.5, "ACh": 0.5, "NE": 0.5, "5HT": 0.5, "GABA": 0.2, "endorphin": 0.3},
                "chi_available": 0.5,
                "metabolic_drain": 0.1,
                "attention": 0.6,
                "i_confidence": 0.5,
                "chi_coherence": 0.4,
            }
        ))
        return _ok({"status": "submitted", "fork": fork})
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/test-commit error: %s", e)
        return _error(str(e))


# =====================================================================
# GET /v4/timechain/backup-status — Backup system status
@router.get("/v4/timechain/backup-status")
async def get_v4_timechain_backup_status(request: Request):
    """Get TimeChain backup system status: snapshots, Arweave TXs, manifest."""
    try:
        from titan_plugin.logic.timechain_backup import TimeChainBackup
        backup = TimeChainBackup(data_dir="data/timechain", titan_id="T1")
        status = backup.get_backup_status()
        genesis = backup.verify_genesis_integrity()
        status["genesis_verification"] = genesis
        return _ok(status)
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/backup-status error: %s", e)
        return _error(str(e))


# GET /v4/timechain/verify-memories — Verify DB records match block payloads
@router.get("/v4/timechain/verify-memories")
async def get_v4_timechain_verify_memories(request: Request,
                                            fork: int = None,
                                            limit: int = 200):
    """Verify that memories in source DBs match their block payload hashes."""
    try:
        from titan_plugin.logic.timechain_integrity import ChainIntegrity
        integrity = ChainIntegrity(data_dir="data/timechain", titan_id="T1")
        result = integrity.verify_memory_integrity(fork_id=fork, limit=limit)
        return _ok(result)
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/verify-memories error: %s", e)
        return _error(str(e))


# POST /v4/timechain/backup-now — Trigger immediate TimeChain backup
@router.post("/v4/timechain/backup-now")
async def post_v4_timechain_backup_now(request: Request):
    """Trigger an immediate TimeChain backup to Arweave (devnet)."""
    try:
        from titan_plugin.logic.timechain_backup import TimeChainBackup
        from titan_plugin.utils.arweave_store import ArweaveStore
        arweave = ArweaveStore(network="devnet")
        backup = TimeChainBackup(
            data_dir="data/timechain", titan_id="T1", arweave_store=arweave)
        # rFP_backup_worker Phase 0: use tarball (JSON path retired 2026-04-13)
        tx_id = await backup.snapshot_to_arweave()
        if tx_id:
            return _ok({"tx_id": tx_id, "status": "uploaded"})
        return _error("upload failed")
    except Exception as e:
        logger.error("[Dashboard] /v4/timechain/backup-now error: %s", e)
        return _error(str(e))


# =====================================================================
# TIMESERIES — Universal historical metrics
# =====================================================================

def _get_timeseries_store():
    """Lazy-init a shared TimeseriesStore for API reads (reads same SQLite as spirit_worker writes)."""
    global _ts_api_store
    try:
        return _ts_api_store
    except NameError:
        pass
    from titan_plugin.logic.timeseries import TimeseriesStore
    _ts_api_store = TimeseriesStore("./data/timeseries.db")
    return _ts_api_store


@router.get("/v4/timeseries")
async def get_v4_timeseries(
    request: Request,
    metrics: str = "",
    hours: int = 24,
    resolution: str = "auto",
):
    """
    Query historical time-series data.
    metrics: comma-separated metric names (e.g. "neuromod.DA,neuromod.5HT,msl.i_depth")
    hours: time window (1-720, default 24)
    resolution: "auto" | "5m" | "1h"
    """
    try:
        ts_store = _get_timeseries_store()
        metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
        if not metric_list:
            return _ok({"metrics": {}, "resolution": "none", "count": 0, "error": "no metrics specified"})
        hours = max(1, min(hours, 720))
        result = ts_store.query(metric_list, hours=hours, resolution=resolution)
        return _ok(result)
    except Exception as e:
        logger.error("[Dashboard] /v4/timeseries error: %s", e)
        return _error(str(e))


@router.get("/v4/timeseries/metrics")
async def get_v4_timeseries_metrics(request: Request):
    """List all available metric names with latest value and data point count."""
    try:
        ts_store = _get_timeseries_store()
        metric_list = ts_store.list_metrics()
        total = ts_store.row_count()
        return _ok({"metrics": metric_list, "total_rows": total})
    except Exception as e:
        logger.error("[Dashboard] /v4/timeseries/metrics error: %s", e)
        return _error(str(e))


# ═══════════════════════════════════════════════════════════════════
# TitanMaker substrate — R8 ceremony + future Maker-Titan dialogic flow
# ═══════════════════════════════════════════════════════════════════


def _proposal_to_json(r) -> dict:
    """Serialize a ProposalRecord for API response."""
    import json as _json
    return {
        "proposal_id": r.proposal_id,
        "proposal_type": r.proposal_type.value,
        "title": r.title,
        "description": r.description,
        "payload": _json.loads(r.payload_json),
        "payload_hash": r.payload_hash,
        "created_at": r.created_at,
        "created_epoch": r.created_epoch,
        "requires_signature": r.requires_signature,
        "status": r.status.value,
        "expires_at": r.expires_at,
        "approved_at": r.approved_at,
        "approval_reason": r.approval_reason,
        "approved_signer_pubkey": r.approved_signer_pubkey,
        "declined_at": r.declined_at,
        "decline_reason": r.decline_reason,
    }


@router.get("/v4/maker/proposals")
async def get_maker_proposals():
    """List pending Maker proposals + recent responses + alignment score.

    Used by the /chat MakerPanel UI which polls every 10s when isMaker.
    """
    try:
        from titan_plugin.maker import get_titan_maker
        tm = get_titan_maker()
        if not tm:
            return _error("TitanMaker not initialized", code=503)
        pending = tm.list_pending()
        recent = tm.get_recent_responses(n=10)
        return _ok({
            "pending": [_proposal_to_json(p) for p in pending],
            "recent": [_proposal_to_json(r) for r in recent],
            "maker_pubkey": tm.maker_pubkey,
            "alignment_score": tm.get_maker_alignment_score(),
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/maker/proposals error: %s", e)
        return _error(str(e))


@router.get("/v4/maker/proposals/{proposal_id}")
async def get_maker_proposal(proposal_id: str):
    try:
        from titan_plugin.maker import get_titan_maker
        tm = get_titan_maker()
        if not tm:
            return _error("TitanMaker not initialized", code=503)
        record = tm.get(proposal_id)
        if not record:
            return _error("proposal not found", code=404)
        return _ok(_proposal_to_json(record))
    except Exception as e:
        logger.error("[Dashboard] /v4/maker/proposals/{id} error: %s", e)
        return _error(str(e))


@router.post("/v4/maker/proposals/{proposal_id}/approve")
async def approve_maker_proposal(proposal_id: str, request: Request):
    """Approve a Maker proposal.

    Body: {reason, signature_b58?, signer_pubkey_b58?}
    Reason is required (≥10 chars). For requires_signature proposals,
    signature_b58 + signer_pubkey_b58 are also required and verified
    (Ed25519 via solders) against the bundle's payload_hash.
    """
    try:
        from titan_plugin.maker import get_titan_maker
        tm = get_titan_maker()
        if not tm:
            return _error("TitanMaker not initialized", code=503)
        body = await request.json()
        reason = body.get("reason", "")
        signature_b58 = body.get("signature_b58")
        signer_pubkey_b58 = body.get("signer_pubkey_b58")
        result = tm.record_approval(
            proposal_id, reason=reason,
            signature_b58=signature_b58, signer_pubkey_b58=signer_pubkey_b58)
        if result.success:
            return _ok({"proposal_id": proposal_id, "response_type": "approve"})
        return _error(result.error or "approve failed", code=400)
    except Exception as e:
        logger.error("[Dashboard] /v4/maker/proposals/approve error: %s", e)
        return _error(str(e))


@router.post("/v4/maker/proposals/{proposal_id}/decline")
async def decline_maker_proposal(proposal_id: str, request: Request):
    """Decline a Maker proposal.

    Body: {reason}
    Reason is required (≥10 chars). The decline reason is stored to
    the ProposalStore + (Tier 2) emitted as a MAKER_RESPONSE_RECEIVED
    bus message so spirit_worker can process it somatically.
    """
    try:
        from titan_plugin.maker import get_titan_maker
        tm = get_titan_maker()
        if not tm:
            return _error("TitanMaker not initialized", code=503)
        body = await request.json()
        reason = body.get("reason", "")
        result = tm.record_decline(proposal_id, reason)
        if result.success:
            return _ok({"proposal_id": proposal_id, "response_type": "decline"})
        return _error(result.error or "decline failed", code=400)
    except Exception as e:
        logger.error("[Dashboard] /v4/maker/proposals/decline error: %s", e)
        return _error(str(e))


@router.get("/v4/maker/dialogue-history")
async def get_maker_dialogue_history(request: Request):
    """Return recent dialogue history + bond health for Observatory display."""
    try:
        from titan_plugin.maker import get_titan_maker
        tm = get_titan_maker()
        if not tm or not tm._profile:
            return _ok({"dialogue": [], "bond_health": {}})
        dialogue = tm._profile.get_recent_dialogue(n=20)
        bond_health = tm._profile.get_bond_health()
        return _ok({
            "dialogue": dialogue,
            "bond_health": bond_health,
            "alignment_score": tm.get_maker_alignment_score(),
        })
    except Exception as e:
        logger.error("[Dashboard] /v4/maker/dialogue-history error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# GET /v4/admin/memory-profile — Live memory + CPU profiling
# ---------------------------------------------------------------------------
# Returns /proc stats (RSS, VmPeak, CPU time, I/O, threads) for parent +
# all Guardian modules, plus tracemalloc top allocations for the parent
# process.  Optional CPU% sampling and diff-from-boot mode.
#
# Query params:
#   top_n     (int, default 25)    — number of top allocations
#   key_type  (str, default "filename") — "filename" or "lineno"
#   diff      (bool, default false) — show growth since boot
#   cpu       (bool, default false) — include 1s CPU% sample (adds latency)
# ---------------------------------------------------------------------------
@router.get("/v4/admin/memory-profile")
async def get_v4_admin_memory_profile(request: Request,
                                       top_n: int = 25,
                                       key_type: str = "filename",
                                       diff: bool = False,
                                       cpu: bool = False):
    """Live memory and CPU profiling for the Titan process tree."""
    import asyncio
    try:
        plugin = _get_plugin(request)

        # plugin IS TitanCore in production (v5_core passes self to create_app)
        collector = getattr(plugin, '_profiling_collector', None)
        guardian = getattr(plugin, 'guardian', None)

        # Import here to keep module lazy
        from titan_plugin.core.profiler import ProfileReport

        profiler = ProfileReport(collector=collector)

        # CPU sampling blocks for ~1s, run in thread to not block event loop
        if cpu:
            cpu_dur = 1.0
            try:
                prof_cfg = getattr(core, '_full_config', {}).get("profiling", {})
                cpu_dur = float(prof_cfg.get("cpu_sample_duration_s", 1.0))
            except Exception:
                pass
            report = await asyncio.to_thread(
                profiler.full_report, guardian, top_n, diff, key_type,
                include_cpu=True, cpu_duration=cpu_dur)
        else:
            report = await asyncio.to_thread(
                profiler.full_report, guardian, top_n, diff, key_type)

        # Per-module tracemalloc via Guardian bus QUERY (if specific module requested
        # or if we want all child tracemalloc data)
        module_param = request.query_params.get("module")
        if guardian and module_param:
            # Query specific module for its tracemalloc data
            child_data = await asyncio.to_thread(
                guardian.query_module, module_param, "get_memory_profile",
                {"top_n": top_n, "key_type": key_type, "diff": diff},
                5.0)
            if child_data and module_param in report.get("modules", {}):
                report["modules"][module_param]["tracemalloc"] = child_data.get("tracemalloc", {})

        return _ok(report)
    except Exception as e:
        logger.error("[Dashboard] /v4/admin/memory-profile error: %s", e)
        return _error(str(e))
