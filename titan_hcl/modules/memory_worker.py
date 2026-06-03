"""
Memory Module Worker — runs TieredMemoryGraph in its own supervised process.

Receives commands via recv_queue, executes memory operations,
sends responses via send_queue. This isolates Cognee's ~500MB footprint.

Entry point: memory_worker_main(recv_queue, send_queue, name, config)
"""
import asyncio
import logging
import os
import sys
import threading
import time
from titan_hcl import bus
from titan_hcl.modules._memory_dispatch import (
    ActionRouter,
    InFlightRegistry,
    QueryCache,
    WorkerContext,
    ensure_thread_loop,
)
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger(__name__)

# Phase 11 §11.I.3/§11.I.5 — module-level readiness sentinel; SHM heartbeat
# is suppressed until the worker has finished in-process scaffolding.
_WORKER_READY: bool = False

# RFP_synthesis_spine_reads_real_data Phase B — process-lifetime ThoughtSidecar
# (sqlite-WAL tx_hash→thought content), keyed by data_dir. Constructed lazily on
# the first promotion; one persistent writer per memory_worker process.
_THOUGHT_SIDECAR: dict = {}


def _phase11_hb_loop(send_queue, name: str, stop_event: threading.Event,
                     state_writer) -> None:
    """Phase 11 §11.I.5 — bus + SHM heartbeat (boot-window cover).

    Runs alongside the recv-loop's existing 10s `_send_heartbeat` cadence.
    This loop's sole purpose is to keep the SHM slot's `last_heartbeat` fresh
    during the ~30-60s memory backend init (FAISS+Kuzu+DuckDB) so guardian's
    SHM-staleness detector doesn't kill the worker mid-boot. Once init
    completes and _WORKER_READY flips True, the main loop's heartbeat takes
    over SHM republishing too — but this loop keeps running harmlessly."""
    while not stop_event.is_set():
        try:
            send_queue.put_nowait({
                "type": bus.MODULE_HEARTBEAT, "src": name, "dst": "guardian",
                "payload": {"alive": True, "boot_cover": True},
                "ts": time.time(),
            })
        except Exception:
            pass
        if state_writer is not None and _WORKER_READY:
            try:
                state_writer.heartbeat()
            except Exception:
                pass
        elif state_writer is not None:
            # During boot we still want the SHM slot's last_heartbeat refreshed
            # (state stays "starting"); heartbeat() republishes the current
            # state value which is "starting" here.
            try:
                state_writer.heartbeat()
            except Exception:
                pass
        stop_event.wait(10.0)


@with_error_envelope(module_name="memory", subsystem="entry", severity=_phase11_sev.FATAL)
def memory_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """
    Main loop for the Memory module process.

    Args:
        recv_queue: receives messages from DivineBus (bus→worker)
        send_queue: sends messages back to DivineBus (worker→bus)
        name: module name ("memory")
        config: dict from [memory_and_storage] + [inference] config sections
    """
    from queue import Empty

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("[MemoryWorker] Initializing TieredMemoryGraph...")
    init_start = time.time()

    global _WORKER_READY
    _WORKER_READY = False

    # ── Phase 11 §11.I.5 — SHM state-slot writer (G21 per worker) ──
    # Created BEFORE the slow ~30-60s TieredMemoryGraph init so the slot
    # publishes state="starting" immediately and the boot-cover heartbeat
    # thread below keeps last_heartbeat fresh during init.
    _state_writer = None
    try:
        from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
        _state_writer = ModuleStateWriter(
            module_name="memory",
            layer="L2",
            boot_priority=BootPriority.MANDATORY,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning(
            "[MemoryWorker] Phase 11 ModuleStateWriter init failed: %s",
            _sw_err)

    # Phase 11 boot-cover heartbeat thread (started BEFORE slow init).
    _phase11_hb_stop = threading.Event()
    _phase11_hb_thread = threading.Thread(
        target=_phase11_hb_loop,
        args=(send_queue, name, _phase11_hb_stop, _state_writer),
        daemon=True, name=f"memory-phase11-hb-{name}")
    _phase11_hb_thread.start()

    # Synthesis Engine Phase 1 (D-SPEC-123 / SPEC v1.56.0 §25): inject a
    # bus_emit callable so TieredMemoryGraph._cognee_search can emit
    # MEMORY_RETRIEVAL_USED (use-gated reinforcement, INV-Syn-5). The emit
    # is fire-and-forget; synthesis_worker is the sole consumer. Cheap
    # closure over send_queue (multiprocessing.Queue → put_nowait).
    def _synth_bus_emit(msg_type: str, payload: dict) -> None:
        try:
            from titan_hcl.bus import make_msg
            send_queue.put_nowait(
                make_msg(msg_type, name, "all", payload))
        except Exception as _emit_err:
            logger.debug(
                "[MemoryWorker] synth bus_emit failed for %s: %s",
                msg_type, _emit_err)

    # Retry with backoff — DuckDB lock may be held briefly by a dying sibling process
    memory = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            from titan_hcl.core.memory import TieredMemoryGraph
            memory = TieredMemoryGraph(
                config=dict(config), bus_emit=_synth_bus_emit)
            init_ms = (time.time() - init_start) * 1000
            logger.info("[MemoryWorker] TieredMemoryGraph ready in %.0fms", init_ms)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2s, 4s
                logger.warning("[MemoryWorker] Init attempt %d/%d failed (retrying in %ds): %s",
                               attempt + 1, max_retries, wait, e)
                time.sleep(wait)
            else:
                logger.error("[MemoryWorker] Failed to init TieredMemoryGraph after %d attempts: %s",
                             max_retries, e, exc_info=True)
                sys.exit(1)  # non-zero exit so Guardian knows this was a failure

    # Phase 11 §11.I.2 — MODULE_READY bus-emit deleted per locked D2.
    # The SHM slot state=booted transition below is the contract now.

    # Main message loop with async support
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Initialize direct memory backend (FAISS + DuckDB + Kuzu)
    try:
        backend_ok = loop.run_until_complete(memory._ensure_cognee())
        logger.info("[MemoryWorker] Memory backend initialized: %s", "ready" if backend_ok else "unavailable")
    except Exception as e:
        logger.warning("[MemoryWorker] Memory backend init failed: %s", e)

    # Phase 11 §11.I.2 — slot transition: starting → booted (in-process
    # scaffolding complete, backend init attempted).
    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
            logger.info(
                "[MemoryWorker] Phase 11 §11.I.2 — SHM slot state=booted "
                "(awaiting MODULE_PROBE_REQUEST from titan_hcl)")
        except Exception as _swb_err:  # noqa: BLE001
            logger.warning(
                "[MemoryWorker] Phase 11 write_state(booted) failed: %s",
                _swb_err)

    last_heartbeat = time.time()
    last_status_publish = 0.0
    last_topology_publish = 0.0
    STATUS_PUBLISH_INTERVAL_S = 5.0
    # chunk 8M.8 (2026-05-05): tightened from 30.0 → 10.0 per
    # rFP_phase_c_observatory_data_pipeline.md §3.8 + Gap E (§2.5).
    # Observed cache age 23-27s (acceptance gate target <10s). Topology
    # classification + knowledge-graph stats run in a dedicated thread
    # (see _periodic_publish_loop below — 2026-05-01 rFP_bus_payload_contracts
    # §3.1) so cadence tightening doesn't block the recv loop.
    # Bus events carry only summary metrics (cluster_counts /
    # node_count / edge_count / entity_types); bulk payloads go via
    # RPC. Cost estimate: get_top_memories(n=200) is dict scan + 6
    # keyword-substring matches per item ≈ <50 ms; graph.get_stats() is
    # cached + lazy ≈ <10 ms — total ~60 ms per cycle, well under 10 s
    # cadence headroom.
    TOPOLOGY_PUBLISH_INTERVAL_S = 10.0

    # Topic clusters for the Cognitive Heatmap (mirrors dashboard.py keywords;
    # kept in sync — the worker now owns the classification so the endpoint
    # is a pure cache read).
    _TOPIC_KEYWORDS = {
        "Solana Architecture": {
            "solana", "sol", "blockchain", "transaction", "wallet", "rpc", "zk",
            "nft", "anchor", "keypair",
        },
        "Social Pulse": {
            "tweet", "social", "mention", "reply", "like", "engagement", "x",
            "twitter", "post",
        },
        "Maker Directives": {
            "directive", "maker", "divine", "inspiration", "soul", "evolve",
            "prime", "sovereignty",
        },
        "Research & Knowledge": {
            "research", "learn", "knowledge", "search", "document", "analysis",
            "discovery", "sage",
        },
        "Memory & Identity": {
            "memory", "remember", "cognee", "persist", "forget", "identity",
            "growth", "neuron",
        },
        "Metabolic & Energy": {
            "balance", "energy", "starvation", "metabolism", "health", "reserve",
            "governance",
        },
    }

    import re as _re
    _INTERNAL_INJECTION_RE = _re.compile(r"^\[[A-Z_]+_INJECTION\]")

    def _is_internal_injection(prompt: str) -> bool:
        return bool(prompt) and _INTERNAL_INJECTION_RE.match(prompt) is not None

    def _publish_memory_status() -> None:
        """Periodic publish — populates memory.status / memory.mempool /
        memory.top cache keys consumed by /status/memory + Memory tab.
        Microkernel v2 §A.4 pattern: worker publishes, api_subprocess
        BusSubscriber maps to CachedState, endpoints read in O(1).
        Without this, the API endpoints' `_cache.get(...) or {}` reads
        return empty regardless of how many nodes the worker has loaded.

        rFP_observatory_data_loading_v1 §3.4 fix (2026-04-26):
        get_top_memories returns rows sorted by effective_weight DESC.
        Internal injections (self_profile, dream_bridge, meta_wisdom)
        carry weight ~10 — they dominate every top-N window. Pre-fix the
        worker shipped 250 injection-only rows; the dashboard's
        _is_internal_injection filter then stripped all 250 → Memory tab
        showed only the center node. Fix: filter injections in the
        worker, then fill to 250 user-facing rows by scanning deeper
        into the persistent set.
        """
        try:
            pcount = memory.get_persistent_count()
        except Exception as _pc_err:
            logger.warning("[MemoryWorker] get_persistent_count failed: %s", _pc_err)
            pcount = 0
        # rFP_meditation_worker_latency Fix #E (2026-05-07): use the
        # sync sibling. Calling the async fetch_mempool via
        # loop.run_until_complete from this dedicated publisher thread
        # races the main message-handler thread (also driving the same
        # loop) — produced ~5/min "This event loop is already running"
        # warnings + "coroutine was never awaited" RuntimeWarnings on T1.
        try:
            mempool_items = memory.fetch_mempool_sync() or []
        except Exception as _mp_err:
            logger.warning("[MemoryWorker] fetch_mempool_sync failed: %s", _mp_err)
            mempool_items = []
        try:
            # Fetch enough rows that 250 user-facing memories survive the
            # injection filter even when injections dominate the top.
            raw_top = memory.get_top_memories(n=max(pcount, 1000)) or []
            top_items = [
                m for m in raw_top
                if not _is_internal_injection(m.get("user_prompt", ""))
            ][:250]
        except Exception as _tm_err:
            logger.warning("[MemoryWorker] get_top_memories failed: %s", _tm_err,
                           exc_info=True)
            top_items = []
        try:
            now = time.time()
            # rFP_bus_payload_contracts §3.1 — bus events are NOTIFICATIONS,
            # bulk data via memory_proxy RPC. Pre-rFP: MEMORY_TOP_UPDATED
            # carried 250 items × ~8KB embeddings = 2.1 MB, broker rejected
            # with utf-8 decode error → cache empty → memory tab broken on
            # T2/T3. Post-rFP: events are <8 KB; api endpoints fetch bulk
            # via RPC on demand.
            _send_msg(send_queue, bus.MEMORY_STATUS_UPDATED, name, "all", {
                "persistent_count": pcount,
                "mempool_size": len(mempool_items),
                "cognee_ready": True,
                "memory_backend_ready": True,
                "updated_at": now,
            })
            _send_msg(send_queue, bus.MEMORY_MEMPOOL_UPDATED, name, "all", {
                "updated_at": now,
                "count": len(mempool_items),
            })
            _send_msg(send_queue, bus.MEMORY_TOP_UPDATED, name, "all", {
                "updated_at": now,
                "count": len(top_items),
                "last_id": str(top_items[0].get("id", "")) if top_items else None,
            })
        except Exception as _pub_err:
            logger.warning(
                "[MemoryWorker] memory.status publish failed: %s", _pub_err)

    def _classify_topology(top_items: list) -> dict:
        """Bucket recent persistent memories by topic cluster keyword.
        Returns clusters with counts + sample texts."""
        clusters: dict[str, dict] = {
            name: {"count": 0, "samples": []}
            for name in _TOPIC_KEYWORDS
        }
        clusters["Other"] = {"count": 0, "samples": []}
        for item in top_items[:200]:
            text = (item.get("text") if isinstance(item, dict) else str(item)) or ""
            tokens = set(text.lower().split())
            matched = False
            for cluster_name, keywords in _TOPIC_KEYWORDS.items():
                if tokens & keywords:
                    clusters[cluster_name]["count"] += 1
                    if len(clusters[cluster_name]["samples"]) < 3:
                        clusters[cluster_name]["samples"].append(text[:120])
                    matched = True
                    break
            if not matched:
                clusters["Other"]["count"] += 1
                if len(clusters["Other"]["samples"]) < 3:
                    clusters["Other"]["samples"].append(text[:120])
        return clusters

    def _publish_memory_topology() -> None:
        """Periodic publish — emits NOTIFICATION events for topology + knowledge
        graph state changes. Bulk data fetched on demand via memory_proxy RPC.

        rFP_bus_payload_contracts §3.1 (2026-05-01): bus events are light
        notifications + summary metrics; bulk data (per-cluster sample texts,
        Kuzu node/edge dumps) goes through RPC, not the bus. Pre-rFP these
        carried bulk payloads — `MEMORY_KNOWLEDGE_GRAPH_UPDATED` could
        approach msgpack 2 MB limit on dense graphs.
        """
        # Topology event — cluster counts only
        try:
            top_items = memory.get_top_memories(n=200) or []
            clusters = _classify_topology(top_items)
            cluster_counts = {
                name: int(data.get("count", 0))
                for name, data in clusters.items()
            }
            _send_msg(send_queue, bus.MEMORY_TOPOLOGY_UPDATED, name, "all", {
                "updated_at": time.time(),
                "total_classified": sum(cluster_counts.values()),
                "cluster_counts": cluster_counts,
            })
        except Exception as _topo_err:
            logger.warning(
                "[MemoryWorker] memory.topology publish failed: %s", _topo_err)

        # Knowledge graph event — node/edge counts + entity-type histogram
        try:
            graph = getattr(memory, "_graph", None)
            node_count = 0
            edge_count = 0
            entity_types: dict[str, int] = {}
            if graph is not None:
                try:
                    stats = graph.get_stats() or {}
                    # stats may have shape: {"node_count": N, "edge_count": N,
                    # "by_entity_type": {...}} or similar — defensive lookup.
                    node_count = int(stats.get("node_count", 0) or 0)
                    edge_count = int(stats.get("edge_count", 0) or 0)
                    bt = stats.get("by_entity_type") or stats.get("entity_types") or {}
                    if isinstance(bt, dict):
                        entity_types = {str(k): int(v) for k, v in bt.items()
                                        if isinstance(v, (int, float))}
                except Exception:
                    pass
            _send_msg(send_queue, bus.MEMORY_KNOWLEDGE_GRAPH_UPDATED, name, "all", {
                "updated_at": time.time(),
                "node_count": node_count,
                "edge_count": edge_count,
                "entity_types": entity_types,
            })
        except Exception as _kg_err:
            logger.warning(
                "[MemoryWorker] memory.knowledge_graph publish failed: %s",
                _kg_err)

    # ── Microkernel v2 Phase B.1 §6 — readiness/hibernate reporter ──
    from titan_hcl.core.readiness_reporter import trivial_reporter
    def _b1_save_state():
        return []
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L2", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    # 2026-05-01 (rFP_bus_payload_contracts §3.1) — periodic publishes run
    # in a SEPARATE THREAD so the main recv loop stays responsive to RPC
    # queries. Pre-fix: publishes ran inline in the recv loop, blocking
    # recv_queue.get for several seconds during topology classification
    # (200 items × keyword match + graph stats). RPC requests piled up +
    # timed out (worker eventually processed them but caller already gave
    # up). Mirrors spirit_loop's snapshot-builder thread pattern.
    # NOTE: `threading` is imported at module level (line 13). A redundant
    # function-local `import threading` here used to shadow it, making
    # `threading` a local var for the WHOLE function — so the Phase 11
    # boot-cover heartbeat at line ~110 (added later, BEFORE this point)
    # raised UnboundLocalError "cannot access local variable 'threading'"
    # and crash-looped memory on every boot (live T3 2026-05-28). Use the
    # module-level import; do not re-import locally.
    _periodic_stop = threading.Event()

    # ── Phase A: in-flight registry + write_lock allocated EARLY ───────
    # Created here (not below the periodic-publish thread) so that
    # `_periodic_publish_loop` can reference `in_flight` for the
    # orphan-count instrumentation required by PLAN §7.5 + §11.C
    # ("instrument `len(_inflight_registry._futures)` in periodic-publish
    # thread; assert always ≤2"). Router construction happens later, after
    # the loop is defined + thread started.
    in_flight = InFlightRegistry()
    write_lock = threading.RLock()

    # Phase C Session 2 (rFP_phase_c_async_shm_consumer_migration §4.B.8) —
    # SHM-direct memory_state.bin publisher. Replaces the deadlock-prone
    # sync bus.request(action="growth_metrics" / "status") path that
    # wedged T3 outer-sensor sidecars on 2026-05-07 (py-spy
    # post-Session-1 caught sidecars stuck inside
    # memory_proxy.get_growth_metrics → bus.request after spirit_proxy
    # unblocked). Cadence: 1 Hz (matches periodic loop's natural tick).
    _memory_state_pub = None
    try:
        from titan_hcl.core.state_registry import resolve_titan_id
        from titan_hcl.logic.memory_state_publisher import (
            MemoryStatePublisher)
        _memory_state_pub = MemoryStatePublisher(titan_id=resolve_titan_id())
    except Exception as _pub_init_err:
        # Boot-time failure is non-fatal: existing bus.request RPC paths
        # still serve memory_proxy callers (with deadlock surface intact)
        # while we recover. Log loudly so health checks surface it.
        logger.error(
            "[MemoryWorker] MemoryStatePublisher BOOT FAILED — "
            "consumers fall back to sync bus.request path (deadlock "
            "surface remains until publisher recovers): %s",
            _pub_init_err, exc_info=True)

    # Phase A orphan-count instrumentation (PLAN §7.5 + §11.C).
    # Cadence: 30s — frequent enough to catch leaks within the 24h soak
    # gate window, sparse enough not to flood the log. WARN threshold is
    # >2 (the PLAN's expected upper bound: 1 in-flight per concurrent
    # meditation, typically 0).
    INFLIGHT_LOG_INTERVAL_S = 30.0
    INFLIGHT_ORPHAN_THRESHOLD = 2

    def _periodic_publish_loop():
        last_status = 0.0
        last_topology = 0.0
        last_inflight_log = 0.0
        while not _periodic_stop.is_set():
            try:
                now = time.time()
                if now - last_status > STATUS_PUBLISH_INTERVAL_S:
                    _publish_memory_status()
                    last_status = now
                if now - last_topology > TOPOLOGY_PUBLISH_INTERVAL_S:
                    _publish_memory_topology()
                    last_topology = now
                # Phase A orphan-count instrumentation (PLAN §7.5 + §11.C).
                # Log at INFO under threshold (sample observability);
                # WARN at-or-above threshold (regression signal).
                if now - last_inflight_log > INFLIGHT_LOG_INTERVAL_S:
                    try:
                        orphans = in_flight.in_flight_count()
                        if orphans > INFLIGHT_ORPHAN_THRESHOLD:
                            logger.warning(
                                "[MemoryWorker] in_flight registry orphan count "
                                "%d > threshold %d — bus-bridge Future leak suspected "
                                "(PLAN §11.C)",
                                orphans, INFLIGHT_ORPHAN_THRESHOLD)
                        else:
                            logger.info(
                                "[MemoryWorker] in_flight registry orphan count=%d "
                                "(threshold=%d)",
                                orphans, INFLIGHT_ORPHAN_THRESHOLD)
                    except Exception as _of_err:
                        logger.warning(
                            "[MemoryWorker] orphan-count instrumentation "
                            "raised: %s", _of_err)
                    last_inflight_log = now
                # SHM-direct memory_state.bin — every loop tick (~1 Hz).
                # Failure isolated per-publish; never blocks the legacy
                # bus-publish paths above.
                if _memory_state_pub is not None:
                    try:
                        _memory_state_pub.publish(memory)
                    except Exception as _shm_err:
                        # Already logged with exc_info inside publisher;
                        # this top-level catch only prevents the
                        # periodic loop from dying on a publisher bug.
                        logger.warning(
                            "[MemoryWorker] memory_state SHM publish "
                            "raised at top level: %s",
                            _shm_err, exc_info=True)
            except Exception as _per_err:
                logger.warning(
                    "[MemoryWorker] periodic publish thread error: %s",
                    _per_err)
            # Sleep ~1s but wake early if stop event set (clean shutdown).
            _periodic_stop.wait(1.0)

    _periodic_thread = threading.Thread(
        target=_periodic_publish_loop,
        daemon=True,
        name="memory-periodic-publish",
    )
    _periodic_thread.start()

    # ── Phase A: dispatcher (PLAN §2.1) ────────────────────────────────
    # Read pool (8), writer pool (1, serial), dedicated meditation thread.
    # Main loop becomes a pure router — pulls msgs off recv_queue, gives
    # InFlightRegistry first dibs on rid-matched RESPONSEs (meditation's
    # ANCHOR bus-bridge), then dispatches everything else via router.
    # `in_flight` + `write_lock` are allocated above the periodic-publish
    # loop so the loop can read in_flight.in_flight_count() for the
    # orphan-count instrumentation (PLAN §7.5 + §11.C).
    # Phase B (rFP §3.4.1) §B5 — query LRU+TTL cache (G20 closure for chat
    # hot path). Sized per PLAN: 256 entries × 60s TTL. Tunable via config
    # under [memory_and_storage] for soak observability.
    _cache_cfg = (config or {}).get("memory_and_storage", {}) if config else {}
    query_cache = QueryCache(
        maxsize=int(_cache_cfg.get("query_cache_maxsize", 256)),
        ttl_s=float(_cache_cfg.get("query_cache_ttl_s", 60.0)),
    )
    ctx = WorkerContext(
        memory=memory,
        send_queue=send_queue,
        name=name,
        config=dict(config) if config else {},
        in_flight=in_flight,
        write_lock=write_lock,
        query_cache=query_cache,
    )
    router = ActionRouter(
        ctx,
        handle_query=_handle_query,
        handle_memory_add=_handle_memory_add,
        handle_mempool_add=_handle_mempool_add,
        handle_memory_ingest_request=_handle_memory_ingest_request,
    )
    logger.info(
        "[MemoryWorker] dispatch router ready — read_pool=8 writer_pool=1 "
        "meditation_thread=lazy")

    # RFP_synthesis_spine_reads_real_data §7.D (D4) — one-shot spine backfill of
    # pre-Phase-B persistent nodes (`timechain_tx_hash IS NULL`) into the spine,
    # so consolidation + recall draw on ALL promoted memories, not just
    # post-Phase-B ones. Daemon thread, idempotent, settles before emitting —
    # never blocks boot or the bus loop.
    threading.Thread(
        target=_backfill_thought_sidecar, args=(ctx,),
        daemon=True, name=f"memory-spine-backfill-{name}").start()

    while True:
        # Heartbeat stays on main loop (cheap; needs to fire even during
        # recv_queue idle). Bulk periodic publishes happen in dedicated
        # thread above so RPC response latency stays bounded.
        now = time.time()
        if now - last_heartbeat > 10.0:
            _send_heartbeat(send_queue, name)
            last_heartbeat = now

        try:
            msg = recv_queue.get(timeout=1.0)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            _periodic_stop.set()
            break

        # ── Phase A: in-flight rid registry first dibs ─────────────────
        # Meditation thread's ANCHOR_REQUEST/RESPONSE round-trip routes
        # through here. If `resolve` returns True, the message belonged to
        # a thread waiting on a Future — main loop must not double-dispatch.
        if in_flight.resolve(msg):
            continue

        msg_type = msg.get("type", "")

        # ── Phase 11 §11.I.3 — MODULE_PROBE_REQUEST handler ─────────
        if msg_type == bus.MODULE_PROBE_REQUEST and _state_writer is not None:
            try:
                from titan_hcl.core.probe_dispatcher import (
                    handle_module_probe_request,
                )
                handle_module_probe_request(
                    msg,
                    probe_fn=None,  # trivial pass-through per §11.I.2
                    send_queue=send_queue,
                    module_name=name,
                    state_writer=_state_writer,
                )
            except Exception as _phb_err:  # noqa: BLE001
                logger.warning(
                    "[MemoryWorker] Phase 11 probe handler raised: %s",
                    _phb_err)
            continue

        # ── Microkernel v2 Phase B.1 §6 — shadow swap dispatch ────────
        if _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ─
        from titan_hcl.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[MemoryWorker] Shutdown: %s",
                        msg.get("payload", {}).get("reason"))
            break

        # ── Session 3 (RFP_meta-reasoning_CGN_FIX.md §4.2 rows 6/7) ───
        # memory_worker handles CGN_KNOWLEDGE_REQ from meta_service
        # resolvers for kind ∈ {episodic_memory, semantic_graph} —
        # canonical Phase C-extracted home per SPEC §3.4.1 Phase B
        # SHIPPED 2026-05-13. Defense-in-depth filter: only Session 3
        # envelopes (correlation_id + recognized kind) are handled here;
        # anything else falls through to router.dispatch.
        if msg_type == bus.CGN_KNOWLEDGE_REQ:
            _mwk_p = msg.get("payload", {}) or {}
            _mwk_corr = _mwk_p.get("correlation_id")
            _mwk_kind = _mwk_p.get("kind", "")
            if _mwk_corr and _mwk_kind in ("episodic_memory",
                                            "semantic_graph"):
                _mwk_name = _mwk_p.get("name", "")
                _mwk_src = msg.get("src", "meta_service")
                _mwk_output: dict
                _mwk_failure = None
                try:
                    if _mwk_kind == "episodic_memory":
                        _mwk_output = _build_episodic_memory_response(
                            memory, _mwk_name, _mwk_p)
                    elif _mwk_kind == "semantic_graph":
                        _mwk_output = _build_semantic_graph_response(
                            memory, _mwk_name, _mwk_p)
                    else:  # defense-in-depth
                        _mwk_output = {}
                        _mwk_failure = "unknown_kind"
                except Exception as _mwk_err:
                    logger.warning(
                        "[MemoryWorker] CGN_KNOWLEDGE_REQ kind=%s "
                        "handler error: %s", _mwk_kind, _mwk_err)
                    _mwk_output = {"error": str(_mwk_err)}
                    _mwk_failure = "handler_error"
                _mwk_resp_payload = {
                    "correlation_id": _mwk_corr,
                    "kind": _mwk_kind,
                    "name": _mwk_name,
                    "output": _mwk_output,
                    "ts": time.time(),
                }
                if _mwk_failure:
                    _mwk_resp_payload["failure"] = _mwk_failure
                _send_msg(send_queue, bus.CGN_KNOWLEDGE_RESP, name,
                          _mwk_src, _mwk_resp_payload)
                continue
            # Non-Session-3 CGN_KNOWLEDGE_REQ envelopes drop silently —
            # memory_worker is not the legacy responder for those (that's
            # spirit_worker P8 D8.4).
            continue

        # ── Everything else goes through the dispatcher ───────────────
        # router.dispatch classifies bus.QUERY by action (read/write/
        # meditation) and routes bus.MEMORY_ADD / bus.MEMORY_MEMPOOL_ADD
        # to the writer pool. Unknown msg types are logged + skipped
        # inside the router.
        router.dispatch(msg)

    logger.info("[MemoryWorker] Exiting — router shutdown starting")
    router.shutdown(timeout=5.0)
    _periodic_stop.set()
    _phase11_hb_stop.set()
    loop.close()
    logger.info("[MemoryWorker] Exit complete")


# ──────────────────────────────────────────────────────────────────────
# Session 3 CGN_KNOWLEDGE_REQ response builders
# RFP_meta-reasoning_CGN_FIX.md §4.2 rows 6/7. Produce real output dicts
# from TieredMemoryGraph (FAISS + Kuzu + DuckDB) for meta_service
# resolver dispatch.
# ──────────────────────────────────────────────────────────────────────


def _build_episodic_memory_response(memory, name: str,
                                    payload: dict) -> dict:
    """episodic_memory.search dispatch — queries TieredMemoryGraph for
    episodic-memory matches by topic/snippet. Returns top-K matches.
    """
    if memory is None:
        return {"engine": "unavailable", "name": name,
                "suggested_action": "fallback_static"}
    topic = (payload.get("topic") or
             payload.get("payload_snippet") or "").strip()
    if not topic:
        return {"engine": "episodic_memory", "name": name,
                "match_count": 0, "reason": "empty_topic",
                "suggested_action": "provide_topic_first"}
    try:
        # TieredMemoryGraph has multiple query surfaces; the canonical
        # async query is `query` (Phase B contract). Wrap in sync via
        # asyncio.run since memory_worker's main loop is sync.
        results = []
        if hasattr(memory, "query"):
            try:
                _q = memory.query(query_text=topic, top_k=5)
                if asyncio.iscoroutine(_q):
                    results = asyncio.run(_q)
                else:
                    results = _q
            except Exception as _qe:
                logger.debug(
                    "[MemoryWorker] episodic_memory query raised: %s", _qe)
        elif hasattr(memory, "search"):
            try:
                results = memory.search(topic, top_k=5)
            except Exception as _se:
                logger.debug(
                    "[MemoryWorker] episodic_memory search raised: %s", _se)
    except Exception as e:
        return {"engine": "episodic_memory", "name": name, "error": str(e)}
    if not isinstance(results, (list, tuple)):
        results = []
    # Fix4 — suggested_action from match count
    if len(results) >= 3:
        sugg = "recall_top_match"
    elif len(results) >= 1:
        sugg = "consider_weak_match"
    else:
        sugg = "store_new_episode"
    return {
        "engine": "episodic_memory",
        "name": name,
        "topic": topic,
        "match_count": len(results),
        "top_matches": list(results)[:5],
        "suggested_action": sugg,
    }


def _build_semantic_graph_response(memory, name: str,
                                   payload: dict) -> dict:
    """semantic_graph.neighbors dispatch — queries Kuzu graph neighbors
    for the topic via TieredMemoryGraph's graph surface."""
    if memory is None:
        return {"engine": "unavailable", "name": name,
                "suggested_action": "fallback_static"}
    topic = (payload.get("topic") or
             payload.get("payload_snippet") or "").strip()
    if not topic:
        return {"engine": "semantic_graph", "name": name,
                "neighbor_count": 0, "reason": "empty_topic",
                "suggested_action": "provide_topic_first"}
    try:
        # Try multiple known graph-query method names — TieredMemoryGraph
        # implementation has evolved; tolerate the variation.
        method = (getattr(memory, "graph_neighbors", None)
                  or getattr(memory, "neighbors", None)
                  or getattr(memory, "query_graph_neighbors", None))
        if method is None:
            return {"engine": "semantic_graph", "name": name,
                    "error": "no_graph_method_on_memory"}
        result = method(topic)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
    except Exception as e:
        return {"engine": "semantic_graph", "name": name, "error": str(e)}
    if not isinstance(result, (list, tuple)):
        result = []
    # Fix4 — suggested_action from neighbor density
    if len(result) >= 5:
        sugg = "traverse_rich_neighborhood"
    elif len(result) >= 1:
        sugg = "follow_sparse_link"
    else:
        sugg = "extend_graph_with_topic"
    return {
        "engine": "semantic_graph",
        "name": name,
        "topic": topic,
        "neighbor_count": len(result),
        "neighbors": list(result)[:8],
        "suggested_action": sugg,
    }


# rFP_bus_payload_contracts §3.1 (2026-05-31): every memory work-RPC reply
# stays lean + bounded so it can never exceed the 16 MB bus frame ceiling
# (_frame.py MAX_FRAME_SIZE). _MEMPOOL_WIRE_CAP bounds the otherwise-unbounded
# mempool dumps; truncation is logged (no silent caps — feedback_observation).
_MEMPOOL_WIRE_CAP = 500


def _lean_memory_item(m: dict) -> dict:
    """Whitelist the fields bus consumers actually read off a persistent
    memory node (verified_context_builder / reflex_executor / dashboard).
    Embeddings, neuromod_context, embedding_id + every other internal field
    are dropped so a recall reply round-trips small (§3.1)."""
    return {
        "id": str(m.get("id", "")),
        "user_prompt": str(m.get("user_prompt", "")),
        "agent_response": str(m.get("agent_response", "")),
        "effective_weight": float(m.get("effective_weight", 1.0) or 1.0),
        "emotional_intensity": int(m.get("emotional_intensity", 0) or 0),
        "reinforcement_count": int(m.get("reinforcement_count", 0) or 0),
        "created_at": float(m.get("created_at", 0) or 0),
        "status": str(m.get("status", "")),
    }


def _lean_mempool_item(m: dict) -> dict:
    """Lean wire shape for a mempool node (mempool-specific weight fields)."""
    return {
        "id": str(m.get("id", "")),
        "user_prompt": str(m.get("user_prompt", "")),
        "agent_response": str(m.get("agent_response", "")),
        "mempool_weight": float(m.get("mempool_weight", 1.0) or 1.0),
        "mempool_reinforcements": int(m.get("mempool_reinforcements", 0) or 0),
        "created_at": float(m.get("created_at", 0) or 0),
    }


def _get_thought_sidecar(data_dir: str):
    """Process-lifetime ``ThoughtSidecar`` for ``data_dir`` (one per dir, memoized
    in ``_THOUGHT_SIDECAR``). Soft-fail → ``None`` (a missing sidecar degrades
    promotion to emit-only, it never breaks it)."""
    sc = _THOUGHT_SIDECAR.get(data_dir)
    if sc is None:
        try:
            from titan_hcl.synthesis.thought_sidecar import ThoughtSidecar
            sc = ThoughtSidecar(data_dir)
            _THOUGHT_SIDECAR[data_dir] = sc
        except Exception as _sx:
            logger.warning(
                "[MemoryWorker] ThoughtSidecar init failed "
                "(promotion link degrades to emit-only): %s", _sx)
            sc = None
    return sc


def _anchor_promoted_node(node: dict, *, now: float, sidecar, ctx):
    """Anchor ONE promoted ``memory_node`` to the synthesis spine — the single
    promotion mechanic, shared by the LIVE meditation loop and the Phase-D
    backfill (RFP_synthesis_spine_reads_real_data Phase B/D, §7.D).

    Emits the per-node Timechain pointer on the node's ACT-R ``memory_type`` fork,
    stamps the deterministic per-TX hash on the node (``timechain_tx_hash``), and
    writes the real thought to the lock-free content sidecar keyed by that hash.

    ``now`` is folded into the hash (``promotion_anchor.build_promotion_tx``): the
    LIVE loop passes ``time.time()``; the BACKFILL passes the node's stored
    ``created_at`` so re-runs recompute the SAME hash → idempotent. Returns the
    per-TX hash, or ``None`` on build failure. Soft-fail throughout — an anchor
    error must never break promotion."""
    from titan_hcl.synthesis.promotion_anchor import build_promotion_tx
    node_id = node.get("id")
    try:
        payload, tx_hash = build_promotion_tx(node, now=now)
    except Exception as _bx:
        logger.warning(
            "[MemoryWorker] build_promotion_tx failed node %s: %s", node_id, _bx)
        return None
    _send_msg(ctx.send_queue, bus.TIMECHAIN_COMMIT, ctx.name, "timechain", payload)
    try:
        with ctx.write_lock:
            ctx.memory.set_timechain_tx_hash(node_id, tx_hash)
    except Exception as _ux:
        logger.warning(
            "[MemoryWorker] stamp timechain_tx_hash node %s failed: %s",
            node_id, _ux)
    if sidecar is not None:
        sidecar.put(
            tx_hash=tx_hash, node_id=node_id,
            user_prompt=node.get("user_prompt", "") or "",
            agent_response=node.get("agent_response", "") or "",
            memory_type=payload["thought_type"], fork=payload["fork"], ts=now)
    return tx_hash


def _backfill_thought_sidecar(ctx, *, settle_s: float = 45.0,
                              batch_cap: int = 100000,
                              chunk: int = 1000,
                              chunk_pause_s: float = 1.0) -> None:
    """One-shot backfill: anchor PERSISTENT ``memory_nodes`` that predate Phase B
    (``timechain_tx_hash IS NULL``) into the spine — so Titan recalls + synthesizes
    over ALL its promoted memories, not just post-Phase-B ones (RFP §7.D / D4).

    Reuses the SAME single anchor mechanic (``_anchor_promoted_node``) with
    ``now=node['created_at']`` → deterministic + idempotent (the ``IS NULL`` filter
    self-watermarks; re-runs find nothing). ``batch_cap`` is a runaway backstop
    (high enough to drain a realistic backlog in one pass); the pointer emits go in
    ``chunk``-sized waves with a ``chunk_pause_s`` drain between them so even a large
    backlog never bursts the bus (the T3 2000-at-once backfill sealed cleanly, 0
    drops — chunking keeps that true at any fleet scale). Settles ``settle_s`` first
    so timechain_worker is up. Daemon thread — never blocks boot. Soft-fail."""
    # Settle wait (interruptible-style Event.wait, not a recv-loop sleep) so the
    # bus + timechain_worker are up before we emit the chain pointers.
    threading.Event().wait(settle_s)
    try:
        duckdb = getattr(ctx.memory, "_duckdb", None)
        if duckdb is None:
            logger.info(
                "[MemoryWorker] spine backfill: no _duckdb accessor — skipped")
            return
        with ctx.write_lock:
            rows = duckdb.get_nodes_by_status("persistent")
        pending = [
            r for r in rows
            if not (r.get("timechain_tx_hash") or "")
            and (r.get("user_prompt") or r.get("agent_response"))
        ]
        if not pending:
            logger.info(
                "[MemoryWorker] spine backfill: 0 unlinked persistent nodes "
                "(%d persistent total) — spine already complete", len(rows))
            return
        data_dir = os.environ.get("TITAN_DATA_DIR", "data")
        sidecar = _get_thought_sidecar(data_dir)
        todo = pending[:batch_cap]
        linked = 0
        for start in range(0, len(todo), max(1, chunk)):
            for node in todo[start:start + chunk]:
                now = node.get("created_at") or time.time()
                if _anchor_promoted_node(
                        node, now=float(now), sidecar=sidecar, ctx=ctx) is not None:
                    linked += 1
            # Drain pause between waves (skip after the final wave).
            if start + chunk < len(todo):
                threading.Event().wait(chunk_pause_s)
        logger.info(
            "[MemoryWorker] spine backfill: linked %d/%d unlinked persistent "
            "node(s) into the spine (cap=%d, chunk=%d) — indexer embeds them "
            "next tick", linked, len(pending), batch_cap, chunk)
    except Exception as e:
        logger.warning("[MemoryWorker] spine backfill failed: %s", e)


def _handle_query(msg: dict, ctx: WorkerContext) -> None:
    """Dispatch QUERY to appropriate memory method and send response.

    Phase A (rFP §3.4.1) refactor: runs on a thread-pool worker (read pool
    for READ_ACTIONS, writer pool for WRITE_ACTIONS, dedicated meditation
    thread for `run_meditation`). The asyncio event loop is per-thread via
    `ensure_thread_loop()`; the recv_queue is no longer accessed directly
    — meditation's ANCHOR bus-bridge goes through `ctx.in_flight`.
    """
    memory = ctx.memory
    send_queue = ctx.send_queue
    name = ctx.name
    config = ctx.config
    loop = ensure_thread_loop()

    payload = msg.get("payload", {})
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        if action == "query":
            text = payload.get("text", "")
            top_k = payload.get("top_k", 5)
            cache_key = (text, int(top_k))
            results = None
            if ctx.query_cache is not None:
                results = ctx.query_cache.get(cache_key)
            if results is None:
                results = loop.run_until_complete(
                    asyncio.wait_for(memory.query(text, top_k=top_k), timeout=30.0)
                )
                if ctx.query_cache is not None:
                    ctx.query_cache.set(cache_key, results)
            # §3.1: ship the lean whitelisted shape — query() may still merge
            # raw _node_store dicts (full text + embeddings + internal fields);
            # only the fields consumers read go on the wire.
            lean = [_lean_memory_item(m) for m in (results or [])
                    if isinstance(m, dict)]
            _send_response(send_queue, name, src, {"results": lean}, rid)

        # Phase B (rFP §3.4.1) §B6 — orphan `action == "add"` handler RETIRED.
        # The proxy's add_memory now publishes MEMORY_INGEST_REQUEST one-way
        # (handled by _handle_memory_ingest_request on the writer pool); the
        # 3 spirit_worker producer sites also migrated. No remaining caller —
        # G-RPC-4 enforced.
        elif action == "count":
            count = memory.get_persistent_count()
            _send_response(send_queue, name, src, {"count": count}, rid)

        elif action == "fetch_mempool":
            mempool = loop.run_until_complete(memory.fetch_mempool()) or []
            # Apply decay before sending so weights are current
            for node in mempool:
                if isinstance(node, dict):
                    memory._apply_mempool_decay(node)
            # §3.1: bound + lean shape (was raw dicts, unbounded by count).
            if len(mempool) > _MEMPOOL_WIRE_CAP:
                logger.warning(
                    "[MemoryWorker] fetch_mempool: %d nodes exceeds wire cap "
                    "%d — truncating reply (heaviest by weight kept)",
                    len(mempool), _MEMPOOL_WIRE_CAP)
                mempool = sorted(
                    mempool,
                    key=lambda n: float(n.get("mempool_weight", 0) or 0)
                    if isinstance(n, dict) else 0.0,
                    reverse=True)[:_MEMPOOL_WIRE_CAP]
            lean = [_lean_mempool_item(n) for n in mempool if isinstance(n, dict)]
            _send_response(send_queue, name, src, {"mempool": lean}, rid)

        elif action == "top_memories":
            n = payload.get("n", 5)
            top = memory.get_top_memories(n=n) or []
            # §3.1: lean shape (was raw _node_store dicts). Bounded by n.
            lean = [_lean_memory_item(m) for m in top if isinstance(m, dict)]
            _send_response(send_queue, name, src, {"memories": lean}, rid)

        elif action == "top_memories_observatory":
            # rFP_bus_payload_contracts §3.1 — observatory-safe shape:
            # strip embeddings, embedding_ids, and any binary-typed fields
            # so the response round-trips cleanly through msgpack.
            #
            # Internal-injection filter applied here (NOT at endpoint), matching
            # pre-rFP _publish_memory_status logic at memory_worker.py:147-151:
            #   fetch_n = max(persistent_count + 100, 1000)
            #   raw_top = memory.get_top_memories(n=fetch_n)
            #   filtered = [m for m in raw_top if not is_internal_injection(...)]
            #   return filtered[:n_requested]
            # Without this, top-by-weight is dominated by [SELF_PROFILE_INJECTION]
            # rows (weight ~10) and the endpoint's defense-in-depth filter strips
            # all 200 → empty list. Original pcount-aware fetch matters because
            # injection memories accumulate over days (each dream cycle adds one).
            try:
                import re as _re
                _INJECTION_RE = _re.compile(r"^\[[A-Z_]+_INJECTION\]")

                def _is_injection(prompt: str) -> bool:
                    return bool(prompt) and _INJECTION_RE.match(prompt) is not None

                n = int(payload.get("n", 200))
                _t0 = time.time()
                # Fetch enough rows that n user-facing memories survive the
                # injection filter even when injections dominate the top.
                pcount = memory.get_persistent_count()
                fetch_n = max(pcount + 100, n * 10, 1000)
                raw_top = memory.get_top_memories(n=fetch_n) or []
                _t1 = time.time()
                # Filter internal injections + cap at n
                filtered = [
                    m for m in raw_top
                    if isinstance(m, dict)
                    and not _is_injection(m.get("user_prompt", ""))
                ][:n]
                _t2 = time.time()
                observatory_items = []
                for m in filtered:
                    # Whitelist: only fields the dashboard endpoint actually
                    # consumes. Embeddings + raw vectors + internal counters
                    # explicitly excluded.
                    observatory_items.append({
                        "id": str(m.get("id", "")),
                        "user_prompt": str(m.get("user_prompt", "")),
                        "agent_response": str(m.get("agent_response", "")),
                        "effective_weight": float(m.get("effective_weight", 1.0) or 1.0),
                        "emotional_intensity": int(m.get("emotional_intensity", 0) or 0),
                        "reinforcement_count": int(m.get("reinforcement_count", 0) or 0),
                        "created_at": float(m.get("created_at", 0) or 0),
                    })
                logger.info(
                    "[MemoryWorker] top_memories_observatory: fetched %d in %.0fms, "
                    "filtered_injections→%d in %.0fms, stripped→%d in %.0fms "
                    "(n_request=%d, fetch_n=%d, pcount=%d)",
                    len(raw_top), (_t1 - _t0) * 1000,
                    len(filtered), (_t2 - _t1) * 1000,
                    len(observatory_items), (time.time() - _t2) * 1000,
                    n, fetch_n, pcount)
                _send_response(send_queue, name, src,
                               {"memories": observatory_items,
                                "count": len(observatory_items)}, rid)
            except Exception as e:
                logger.warning(
                    "[MemoryWorker] top_memories_observatory failed: %s", e,
                    exc_info=True)
                _send_response(send_queue, name, src,
                               {"memories": [], "count": 0,
                                "error": str(e)}, rid)

        elif action == "fetch_mempool_observatory":
            # rFP_bus_payload_contracts §3.1 — observatory-safe shape for
            # the mempool list (same stripping rationale as top_memories).
            full = loop.run_until_complete(memory.fetch_mempool()) or []
            for node in full:
                if isinstance(node, dict):
                    memory._apply_mempool_decay(node)
            full = [m for m in full if isinstance(m, dict)]
            # §3.1: bound count (lean shape already) — no silent caps.
            if len(full) > _MEMPOOL_WIRE_CAP:
                logger.warning(
                    "[MemoryWorker] fetch_mempool_observatory: %d nodes exceeds "
                    "wire cap %d — truncating reply (heaviest by weight kept)",
                    len(full), _MEMPOOL_WIRE_CAP)
                full = sorted(
                    full,
                    key=lambda n: float(n.get("mempool_weight", 0) or 0),
                    reverse=True)[:_MEMPOOL_WIRE_CAP]
            observatory_items = [_lean_mempool_item(m) for m in full]
            _send_response(send_queue, name, src,
                           {"mempool": observatory_items,
                            "count": len(observatory_items)}, rid)

        elif action == "status":
            _send_response(send_queue, name, src, {
                "cognee_ready": getattr(memory, "_cognee_ready", False),  # kept for API compat
                "backend_ready": getattr(memory, "_cognee_ready", False),
                "persistent_count": memory.get_persistent_count(),
                "mempool_size": len(getattr(memory, "_mempool", [])),
            }, rid)

        elif action == "topology":
            # Compute topology server-side to avoid shipping _node_store over IPC
            from collections import Counter
            topic_keywords = payload.get("topic_keywords", {})
            cluster_counts = Counter()
            total = 0
            for v in memory._node_store.values():
                if v.get("type") != "MemoryNode" or v.get("status") != "persistent":
                    continue
                content = f"{v.get('user_prompt', '')} {v.get('agent_response', '')}".lower()
                words = set(content.split())
                best_cluster = "Uncategorized"
                best_overlap = 0
                for cname, kws in topic_keywords.items():
                    overlap = len(words & set(kws))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_cluster = cname
                cluster_counts[best_cluster] += 1
                total += 1
            topology = {}
            for cluster, count in cluster_counts.most_common():
                pct = round((count / max(1, total)) * 100, 1)
                topology[cluster] = {"count": count, "percentage": pct}
            _send_response(send_queue, name, src, {
                "total_persistent": total,
                "clusters": topology,
            }, rid)

        elif action == "knowledge_graph":
            # Return Kuzu entity graph data for Observatory visualization
            limit = payload.get("limit", 200)
            try:
                graph = getattr(memory, "_graph", None)
                if not graph:
                    _send_response(send_queue, name, src, {"available": False}, rid)
                else:
                    stats = graph.get_stats()
                    total_entities = sum(stats.values())

                    table_meta = {
                        "Person": {"color": "#F5A623", "group": "universal"},
                        "Topic": {"color": "#4FC3F7", "group": "universal"},
                        "BodyEntity": {"color": "#66BB6A", "group": "body"},
                        "MindEntity": {"color": "#42A5F5", "group": "mind"},
                        "SpiritEntity": {"color": "#FFFFFF", "group": "spirit"},
                        "Media": {"color": "#AB47BC", "group": "universal"},
                    }

                    nodes = []
                    node_set = set()
                    for table, meta in table_meta.items():
                        per_table = max(10, limit // len(table_meta))
                        try:
                            qr = graph._conn.execute(
                                f"MATCH (e:{table}) RETURN e.name LIMIT {per_table}")
                            while qr.has_next():
                                row = qr.get_next()
                                n = str(row[0])
                                if n and n not in node_set:
                                    node_set.add(n)
                                    nodes.append({
                                        "id": n, "label": n, "table": table,
                                        "color": meta["color"], "group": meta["group"],
                                    })
                        except Exception:
                            pass

                    edges = []
                    for src_t in table_meta:
                        for dst_t in table_meta:
                            rel = f"REL_{src_t}_{dst_t}"
                            try:
                                qr = graph._conn.execute(
                                    f"MATCH (a:{src_t})-[r:{rel}]->(b:{dst_t}) "
                                    f"RETURN a.name, r.rel_type, b.name LIMIT 300")
                                while qr.has_next():
                                    row = qr.get_next()
                                    s, rt, d = str(row[0]), str(row[1]), str(row[2])
                                    if s in node_set and d in node_set:
                                        edges.append({"source": s, "target": d, "type": rt})
                            except Exception:
                                pass

                    _send_response(send_queue, name, src, {
                        "nodes": nodes, "edges": edges, "stats": stats,
                        "total_entities": total_entities, "total_edges": len(edges),
                        "available": True,
                    }, rid)
            except Exception as e:
                _send_response(send_queue, name, src, {
                    "available": False, "error": str(e),
                }, rid)

        # Phase C Session 5 (rFP §4.D.4): growth_metrics handler RETIRED
        # — memory_proxy.get_growth_metrics now SHM-direct via
        # memory_state.bin (Session 2 §4.C.13). The publisher
        # pre-computes learning_velocity/directive_alignment at the
        # default node_saturation_24h=30 and exposes raw counts for
        # the rare non-default-saturation caller to recompute locally.
        elif action == "run_meditation":
            # Simplified meditation: classify mempool, score, prune, promote, consolidate
            # rFP_meditation_worker_latency Option 1 instrumentation:
            # per-phase t+X.XXXs logs to pinpoint silent-hang patterns and
            # per-step costs (fetch / score / anchor / migrate / consolidate).
            _t_med_start = time.time()
            logger.info("[MemoryWorker] [LAT] meditation t+0.000s: handler entered, running cycle...")
            # BUG-VAULT-COMMITS-NOT-LANDING fix (2026-04-29): on-chain vault
            # commit is now bus-bridged to the kernel. memory_worker still
            # runs MeditationEpoch with network_client=None (deployer keypair
            # stays in main process), but after promotions are decided we
            # send ANCHOR_REQUEST to kernel which signs + submits the TX
            # and replies with the real tx_signature. See ANCHOR_REQUEST
            # docstring in titan_hcl/bus.py for the wire contract.
            try:
                from titan_hcl.logic.meditation import MeditationEpoch
                # Create a lightweight meditation instance
                med = MeditationEpoch(
                    memory_graph=memory,
                    network_client=None,  # TX-submission delegated via ANCHOR_REQUEST
                    config=config or {},
                )
                # Phase 3 Chunk χ (D-SPEC-88, 2026-05-18) — direct
                # OllamaCloudProvider construction REMOVED. Meditation
                # now reads api_base + internal_key from config at __init__
                # and routes scoring through POST /v4/llm-distill (which
                # publishes LLM_DISTILL_REQUEST internally → llm_worker).
                # All LLM traffic appears in llm_state.bin centrally.

                # Run simplified epoch (no TX submission inline; no studio, no social)
                candidates, fading, dead = loop.run_until_complete(memory.fetch_mempool_classified())
                total = len(candidates) + len(fading) + len(dead)
                logger.info(
                    "[MemoryWorker] [LAT] meditation t+%.3fs: fetch_mempool_classified done "
                    "(candidates=%d fading=%d dead=%d total=%d)",
                    time.time() - _t_med_start, len(candidates), len(fading), len(dead), total,
                )

                # Prune dead — write_lock serializes against writer pool
                # (PLAN §2.3). One lock acquire per node to keep contention
                # low and let writer-pool `add` interleave between prunes.
                _t_prune = time.time()
                pruned = 0
                for node in dead:
                    with ctx.write_lock:
                        loop.run_until_complete(memory.prune_mempool_node(node["id"]))
                    pruned += 1
                if dead:
                    logger.info(
                        "[MemoryWorker] [LAT] meditation t+%.3fs: pruned %d dead nodes in %.3fs",
                        time.time() - _t_med_start, pruned, time.time() - _t_prune,
                    )

                # Score candidates — collect promotion decisions WITHOUT
                # migrating yet, so the on-chain anchor TX (next step) can
                # cover the full set with a single state_root commit.
                _t_score_loop = time.time()
                promoted_nodes: list = []
                for _i, node in enumerate(candidates):
                    _send_heartbeat(send_queue, name)  # keep alive during long scoring loop
                    last_heartbeat = time.time()
                    _t_score_one = time.time()
                    try:
                        score, intensity = loop.run_until_complete(
                            asyncio.wait_for(med.get_hippocampus_score([node]), timeout=30.0)
                        )
                        logger.info(
                            "[MemoryWorker] [LAT] meditation t+%.3fs: scored node #%d in %.3fs "
                            "(score=%.2f, will_promote=%s)",
                            time.time() - _t_med_start, _i,
                            time.time() - _t_score_one, score, score >= 40.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[MemoryWorker] [LAT] meditation t+%.3fs: SCORE TIMEOUT node #%d (30s)",
                            time.time() - _t_med_start, _i,
                        )
                        continue
                    if score >= 40.0:
                        promoted_nodes.append((node, intensity))
                logger.info(
                    "[MemoryWorker] [LAT] meditation t+%.3fs: scoring loop done (%.3fs total, "
                    "promoted=%d/%d)",
                    time.time() - _t_med_start, time.time() - _t_score_loop,
                    len(promoted_nodes), len(candidates),
                )

                # On-chain anchor (BUG-VAULT-COMMITS-NOT-LANDING fix). Build
                # state_root + payload from the promoted set, ask kernel to
                # submit the vault commit TX, then migrate every promoted
                # node with the real tx_signature. Falls back to
                # MEDITATION_LOCAL if the bus-bridge times out / errors so
                # the cycle still completes (graceful degradation).
                tx_signature = None
                anchor_error = None
                if promoted_nodes:
                    try:
                        import json as _json_anchor
                        payload_json = _json_anchor.dumps(
                            [{"id": n["id"], "prompt": n.get("user_prompt", "")}
                             for n, _ in promoted_nodes],
                            default=str,
                        )
                        try:
                            from titan_hcl.utils.solana_client import generate_state_hash
                            state_root = "MERKLE_" + generate_state_hash(payload_json)[:16]
                        except Exception:
                            state_root = f"MEDITATION_{int(time.time())}"

                        logger.info(
                            "[MemoryWorker] [LAT] meditation t+%.3fs: requesting on-chain anchor "
                            "(promoted=%d, root=%s, inner_timeout=120s)",
                            time.time() - _t_med_start,
                            len(promoted_nodes), state_root[:24],
                        )
                        _t_anchor = time.time()
                        # Phase A: bus-bridge via InFlightRegistry (PLAN §2.4)
                        # instead of pulling from recv_queue. Main loop owns
                        # recv_queue; we register a rid and wait on a Future
                        # that the main loop resolves when the RESPONSE arrives.
                        anchor_result = _request_anchor_via_kernel(
                            ctx.in_flight, send_queue, name,
                            state_root, payload_json, len(promoted_nodes),
                            timeout=120.0,
                        )
                        logger.info(
                            "[MemoryWorker] [LAT] meditation t+%.3fs: anchor RPC returned in %.3fs "
                            "(tx_signature=%s, error=%s)",
                            time.time() - _t_med_start, time.time() - _t_anchor,
                            (anchor_result.get("tx_signature") or "")[:16],
                            anchor_result.get("error") or "",
                        )
                        tx_signature = anchor_result.get("tx_signature")
                        anchor_error = anchor_result.get("error")
                    except Exception as _ax_err:
                        anchor_error = f"anchor_local_exception: {_ax_err}"
                        logger.warning(
                            "[MemoryWorker] Anchor request failed locally: %s",
                            _ax_err,
                        )

                if tx_signature:
                    logger.info(
                        "[MemoryWorker] Vault anchor confirmed: sig=%s",
                        tx_signature[:24],
                    )
                elif promoted_nodes:
                    # Either anchor disabled (no recv_queue), the bus-bridge
                    # timed out, or kernel reported an error (limbo,
                    # no_vault_program_id, send_failed, etc). Log + use
                    # local sig so the meditation cycle still persists.
                    logger.warning(
                        "[MemoryWorker] Vault anchor TX not landed (error=%s)"
                        " — promoted nodes migrate with sig=MEDITATION_LOCAL"
                        " for this cycle",
                        anchor_error or "no_recv_queue",
                    )

                # Migrate all promoted nodes with the resolved signature.
                # Per-node write_lock acquire (PLAN §2.3) — lets writer-pool
                # `add` actions interleave between migrations.
                _t_migrate_loop = time.time()
                sig_for_migration = tx_signature or "MEDITATION_LOCAL"
                for _i, (node, intensity) in enumerate(promoted_nodes):
                    _t_mig_one = time.time()
                    with ctx.write_lock:
                        loop.run_until_complete(
                            memory.migrate_to_persistent(
                                node["id"], sig_for_migration, intensity,
                            )
                        )
                    logger.info(
                        "[MemoryWorker] [LAT] meditation t+%.3fs: migrated node #%d in %.3fs "
                        "(FAISS embed + Kuzu cognify)",
                        time.time() - _t_med_start, _i, time.time() - _t_mig_one,
                    )
                promoted = len(promoted_nodes)
                if promoted_nodes:
                    logger.info(
                        "[MemoryWorker] [LAT] meditation t+%.3fs: migration loop done in %.3fs",
                        time.time() - _t_med_start, time.time() - _t_migrate_loop,
                    )

                # Consolidate Cognee (can be very slow — 120s timeout).
                # write_lock around the FAISS save phase (PLAN §2.3).
                _send_heartbeat(send_queue, name)
                _t_consolidate = time.time()
                try:
                    with ctx.write_lock:
                        consolidated = loop.run_until_complete(
                            asyncio.wait_for(memory.consolidate(), timeout=120.0)
                        )
                    logger.info(
                        "[MemoryWorker] [LAT] meditation t+%.3fs: consolidate done in %.3fs "
                        "(now no-op = FAISS save only)",
                        time.time() - _t_med_start, time.time() - _t_consolidate,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[MemoryWorker] Cognee consolidation timed out (120s)")
                    consolidated = False

                # persistent_count threaded into the reply so meditation_worker
                # can issue STUDIO_RENDER_REQUEST(type=meditation) with the
                # age_nodes arg (per v1.9.4 §4.K wire-up — closes the post-§4.D
                # studio art-generation regression).
                try:
                    _persistent_count = memory.get_persistent_count()
                except Exception:
                    _persistent_count = 0
                result = {
                    "success": True,
                    "total_mempool": total,
                    "promoted": promoted,
                    "pruned": pruned,
                    "fading": len(fading),
                    "consolidated": consolidated,
                    "persistent_count": _persistent_count,
                }
                logger.info(
                    "[MemoryWorker] [LAT] meditation t+%.3fs: HANDLER COMPLETE — "
                    "promoted=%d pruned=%d fading=%d consolidated=%s",
                    time.time() - _t_med_start,
                    promoted, pruned, len(fading), consolidated,
                )

                # TimeChain: meditation cycle → episodic fork (lifecycle event)
                if promoted > 0 or pruned > 0:
                    _send_msg(send_queue, bus.TIMECHAIN_COMMIT, name, "timechain", {
                        "fork": "episodic",
                        "thought_type": "episodic",
                        "source": "meditation",
                        "content": {
                            "event": "MEDITATION_COMPLETE",
                            "promoted": promoted,
                            "pruned": pruned,
                            "fading": len(fading),
                            "total_mempool": total,
                            "consolidated": consolidated,
                        },
                        "significance": min(1.0, 0.3 + promoted * 0.1),
                        "novelty": 0.4,
                        "coherence": 0.5,
                        "tags": ["meditation", "memory_promotion"],
                        "neuromods": {},
                        "chi_available": 0.5,
                        "attention": 0.5,
                        "i_confidence": 0.5,
                        "chi_coherence": 0.3,
                    })

                # TimeChain: each PROMOTED node → its ACT-R memory_type fork, with a
                # DURABLE tx_hash link (RFP_synthesis_spine_reads_real_data Phase B).
                # Delegates to the shared `_anchor_promoted_node` helper — the single
                # promotion mechanic, also reused by the Phase-D backfill (§7.D): emit
                # the per-node pointer, stamp the deterministic per-TX hash, write the
                # real thought to the lock-free sidecar. (The migrate loop above
                # iterates the actually-promoted `promoted_nodes`, fixing the prior
                # `candidates[:promoted]` bug.)
                if promoted_nodes:
                    _sidecar = _get_thought_sidecar(
                        os.environ.get("TITAN_DATA_DIR", "data"))
                    for _node, _intensity in promoted_nodes:
                        _anchor_promoted_node(
                            _node, now=time.time(), sidecar=_sidecar, ctx=ctx)

                _send_response(send_queue, name, src, result, rid)
            except Exception as e:
                logger.error("[MemoryWorker] Meditation failed: %s", e, exc_info=True)
                _send_response(send_queue, name, src, {"success": False, "error": str(e)}, rid)

        else:
            logger.warning("[MemoryWorker] Unknown action: %s", action)
            _send_response(send_queue, name, src, {"error": f"unknown action: {action}"}, rid)

    except asyncio.TimeoutError:
        # 2026-04-08 audit fix (I-012): asyncio.TimeoutError has empty str repr,
        # so old generic handler logged "Error handling add: " with no useful info.
        # Now we identify timeouts explicitly with the action name.
        logger.error("[MemoryWorker] Timeout handling %s (operation exceeded its timeout)",
                     action, exc_info=True)
        _send_response(send_queue, name, src,
                       {"error": f"timeout handling {action}"}, rid)
    except Exception as e:
        # Defensive: use type name if str(e) is empty (some exception classes do this)
        err_msg = str(e) or type(e).__name__
        logger.error("[MemoryWorker] Error handling %s: %s", action, err_msg, exc_info=True)
        _send_response(send_queue, name, src, {"error": err_msg}, rid)


def _handle_memory_add(msg: dict, ctx: WorkerContext) -> None:
    """Handle bus.MEMORY_ADD event — one-way write to inject_memory.

    spirit_worker emits MEMORY_ADD on Maker-dialogue profile enrichment
    (and any other text-ingest path that wants to bypass the mempool +
    skip the discovery delay). Payload: {text, source, weight}. No
    response sent (one-way event, no rid match).

    Phase A: runs on writer pool (PLAN §2.1) with write_lock around the
    FAISS embed + Kuzu cognify + DuckDB update (PLAN §2.3). Non-blocking
    on errors so a bad payload can't poison the pool worker.
    """
    payload = msg.get("payload", {}) or {}
    text = payload.get("text", "")
    source = payload.get("source", "bus")
    weight = float(payload.get("weight", 1.0))
    # Phase 10G — Dream Bridge A injections carry a felt-state snapshot
    # (neuromods + emotion at consolidation time). inject_memory stores it as
    # JSON in DuckDB; agno_hooks Bridge B reads it back for recall perturbation
    # (somatic re-experiencing). Pre-10G this handler dropped it, so even when
    # the bridge fired the felt context never reached the graph.
    neuromod_context = payload.get("neuromod_context") or None
    if not text:
        logger.debug("[MemoryWorker] MEMORY_ADD ignored (empty text)")
        return
    try:
        loop = ensure_thread_loop()
        with ctx.write_lock:
            loop.run_until_complete(
                ctx.memory.inject_memory(
                    text=text, source=source, weight=weight,
                    neuromod_context=neuromod_context,
                )
            )
        logger.info(
            "[MemoryWorker] MEMORY_ADD injected (source=%s, weight=%.2f, "
            "text_len=%d, felt=%s)",
            source, weight, len(text), bool(neuromod_context),
        )
    except Exception as e:
        logger.warning("[MemoryWorker] MEMORY_ADD handler error: %s", e,
                       exc_info=True)


def _handle_mempool_add(msg: dict, ctx: WorkerContext) -> None:
    """Handle bus.MEMORY_MEMPOOL_ADD event — one-way fire-and-forget mempool write.

    Shipped 2026-05-12 (commit `72dd0374`) to close the G19 violation that
    serialized PostHook for 4-6s on T3 when add_to_mempool was a work-RPC.
    PostHook just bus.publishes this event and returns — the write still
    lands here, asynchronously after the reply has shipped. Payload:
    {user_prompt, agent_response, user_identifier}. No response sent.

    Phase A: runs on writer pool (PLAN §2.1) with write_lock around the
    mempool list mutation + JSON persistence (PLAN §2.3).
    """
    payload = msg.get("payload", {}) or {}
    user_prompt = payload.get("user_prompt", "")
    agent_response = payload.get("agent_response", "")
    user_identifier = payload.get("user_identifier", "Anonymous")
    if not user_prompt and not agent_response:
        logger.debug("[MemoryWorker] MEMORY_MEMPOOL_ADD ignored (empty)")
        return
    try:
        loop = ensure_thread_loop()
        with ctx.write_lock:
            loop.run_until_complete(
                ctx.memory.add_to_mempool(
                    user_prompt, agent_response,
                    user_identifier=user_identifier,
                )
            )
        logger.info(
            "[MemoryWorker] MEMORY_MEMPOOL_ADD persisted "
            "(user=%s, prompt_len=%d, response_len=%d)",
            user_identifier[:24], len(user_prompt), len(agent_response),
        )
    except Exception as e:
        logger.warning("[MemoryWorker] MEMORY_MEMPOOL_ADD handler error: %s",
                       e, exc_info=True)


def _handle_memory_ingest_request(msg: dict, ctx: WorkerContext) -> None:
    """Handle bus.MEMORY_INGEST_REQUEST event — Phase B (rFP §3.4.1) replacement
    for the work-RPC `add` action. One-way write to inject_memory; on success
    or failure publishes a `bus.MEMORY_INGEST_COMPLETED` broadcast carrying
    the original `request_id` so subscribers (`memory_proxy.add_memory_with_
    completion`) can correlate by request_id and resolve their Future.

    Payload (in): {request_id, text, source, weight, neuromod_context?}
    Payload (out, COMPLETED broadcast): {request_id, success, source,
        node_id?, weight?, status?, cognified?, error?}

    Phase B G19 closure: the producer (proxy / worker) returns immediately
    after publish; the writer pool runs the heavy FAISS+Kuzu+DuckDB work
    in the background; consumers that need the node_id subscribe + filter
    by request_id with their own bounded timeout.
    """
    payload = msg.get("payload", {}) or {}
    request_id = payload.get("request_id", "")
    text = payload.get("text", "")
    source = payload.get("source", "bus")
    weight = float(payload.get("weight", 1.0))
    neuromod_context = payload.get("neuromod_context")

    if not request_id:
        # No correlation id — caller can't filter the COMPLETED broadcast
        # back to anything specific. Honour the write but don't broadcast
        # an unfilterable completion (would noise every subscriber).
        logger.warning(
            "[MemoryWorker] MEMORY_INGEST_REQUEST received without request_id "
            "(source=%s, text_len=%d) — performing write but skipping "
            "COMPLETED broadcast", source, len(text))

    if not text:
        logger.debug("[MemoryWorker] MEMORY_INGEST_REQUEST ignored (empty text)")
        if request_id:
            _send_msg(ctx.send_queue, bus.MEMORY_INGEST_COMPLETED, ctx.name, "all", {
                "request_id": request_id,
                "success": False,
                "source": source,
                "error": "empty text",
            })
        return

    try:
        loop = ensure_thread_loop()
        with ctx.write_lock:
            result = loop.run_until_complete(
                ctx.memory.inject_memory(
                    text=text, source=source, weight=weight,
                    neuromod_context=neuromod_context,
                )
            )
        if not isinstance(result, dict):
            result = {"node_id": None, "weight": weight}
        logger.info(
            "[MemoryWorker] MEMORY_INGEST_REQUEST injected "
            "(request_id=%s, source=%s, weight=%.2f, node_id=%s, text_len=%d)",
            request_id[:8] if request_id else "(none)", source, weight,
            result.get("node_id"), len(text),
        )
        if request_id:
            _send_msg(ctx.send_queue, bus.MEMORY_INGEST_COMPLETED, ctx.name, "all", {
                "request_id": request_id,
                "success": True,
                "source": source,
                "node_id": result.get("node_id"),
                "weight": result.get("weight"),
                "status": result.get("status"),
                "cognified": result.get("cognified"),
            })
    except Exception as e:
        logger.warning(
            "[MemoryWorker] MEMORY_INGEST_REQUEST handler error "
            "(request_id=%s, source=%s): %s",
            request_id[:8] if request_id else "(none)", source, e, exc_info=True)
        if request_id:
            _send_msg(ctx.send_queue, bus.MEMORY_INGEST_COMPLETED, ctx.name, "all", {
                "request_id": request_id,
                "success": False,
                "source": source,
                "error": f"{type(e).__name__}: {e}",
            })


def _send_msg(send_queue, msg_type: str, src: str, dst: str, payload: dict, rid: str = None) -> None:
    """Send a message via the send queue (worker→bus)."""
    try:
        send_queue.put_nowait({
            "type": msg_type,
            "src": src,
            "dst": dst,
            "ts": time.time(),
            "rid": rid,
            "payload": payload,
        })
    except Exception:
        from titan_hcl.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _request_anchor_via_kernel(
    in_flight: InFlightRegistry, send_queue, name: str,
    state_root: str, payload_json: str, promoted_count: int,
    timeout: float = 120.0,
) -> dict:
    """Bus-bridge: ask kernel main process to submit the on-chain vault TX.

    Closes BUG-VAULT-COMMITS-NOT-LANDING. memory_worker subprocess has
    ``network_client=None`` (deployer keypair stays in main process for
    security), so on-chain TX submission is delegated to the kernel via
    ``ANCHOR_REQUEST``. Kernel's ``_anchor_request_loop`` handles the
    request and replies via ``bus.RESPONSE`` matched on ``rid``.

    Phase A (rFP §3.4.1) refactor: this function runs on the dedicated
    `mem-meditation` thread, NOT on the main loop. The main loop is the
    sole `recv_queue` reader, so we cannot drain `recv_queue` here without
    racing it. Instead, we register the rid with `InFlightRegistry`, send
    the request, and block on a `concurrent.futures.Future`. The main loop
    sees the kernel's RESPONSE first, gives the registry first dibs on it,
    and the Future resolves on this thread.

    Default timeout = 120s sized for real Solana TX confirmation budget:
      - _build_commit_instructions: ~5s sync httpx vault read + ~5-10s
        SQLite sovereignty reads → ~15s
      - send_sovereign_transaction: priority-fee + retry-with-backoff +
        confirm at 'Confirmed' commitment → 30-90s under network load
    First-cycle measurement on T1 microkernel-v2 boot 2026-04-29 10:53:
      - request sent: 10:53:33
      - kernel TX landed: 10:55:14 (101s round-trip)
    Pre-fix 30s timeout fired memory_worker fallback before kernel reply,
    so the migrated node's persistent sig was 'MEDITATION_LOCAL' even
    though the on-chain TX did land. 120s gives ~20s margin over the
    measured 101s.

    Returns the ANCHOR response payload dict (with ``tx_signature`` /
    ``error`` keys) or ``{"error": "anchor_request_timeout"}`` on timeout.
    """
    import uuid
    from concurrent.futures import TimeoutError as _FutureTimeoutError

    rid = uuid.uuid4().hex
    fut = in_flight.register(rid)

    _t_emit = time.time()
    _send_msg(send_queue, bus.ANCHOR_REQUEST, name, "kernel", {
        "state_root": state_root,
        "payload": payload_json,
        "promoted_count": promoted_count,
        "ts": _t_emit,
    }, rid=rid)
    logger.info(
        "[MemoryWorker] [LAT] anchor_request emitted: rid=%s ts=%.6f "
        "(waiting up to %.0fs via in-flight registry)",
        rid[:8], _t_emit, timeout,
    )

    # Bounded wait with periodic heartbeats so Guardian doesn't kill us
    # during long TX waits. Loop in ≤3s slices to interleave heartbeats.
    deadline = time.time() + timeout
    last_hb = 0.0
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            in_flight.cancel(rid)
            logger.warning(
                "[MemoryWorker] [LAT] anchor_request timeout: rid=%s "
                "after %.1fs (no kernel RESPONSE)",
                rid[:8], timeout,
            )
            return {"error": "anchor_request_timeout"}
        try:
            reply = fut.result(timeout=min(3.0, remaining))
        except _FutureTimeoutError:
            if time.time() - last_hb > 3.0:
                _send_heartbeat(send_queue, name)
                last_hb = time.time()
            continue
        logger.info(
            "[MemoryWorker] [LAT] anchor_response received: rid=%s "
            "round_trip=%.3fs",
            rid[:8], time.time() - _t_emit,
        )
        return reply.get("payload", {}) or {}


def _send_response(send_queue, src: str, dst: str, payload: dict, rid: str) -> None:
    """Send a RESPONSE message back."""
    _send_msg(send_queue, bus.RESPONSE, src, dst, payload, rid)


# Heartbeat throttle (Phase E Fix 2): 3s min interval per process.
_last_hb_ts: float = 0.0


def _send_heartbeat(send_queue, name: str) -> None:
    """Send heartbeat to Guardian (throttled to ≤1 per 3s)."""
    global _last_hb_ts
    now = time.time()
    if now - _last_hb_ts < 3.0:
        return
    _last_hb_ts = now
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        rss_mb = 0
    _send_msg(send_queue, bus.MODULE_HEARTBEAT, name, "guardian", {"rss_mb": round(rss_mb, 1)})
