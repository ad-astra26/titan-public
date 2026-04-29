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
import time
from titan_plugin import bus

logger = logging.getLogger(__name__)


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

    # Retry with backoff — DuckDB lock may be held briefly by a dying sibling process
    memory = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            from titan_plugin.core.memory import TieredMemoryGraph
            memory = TieredMemoryGraph(config=dict(config))
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

    # Signal ready
    _send_msg(send_queue, bus.MODULE_READY, name, "guardian", {})

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

    last_heartbeat = time.time()
    last_status_publish = 0.0
    last_topology_publish = 0.0
    STATUS_PUBLISH_INTERVAL_S = 5.0
    TOPOLOGY_PUBLISH_INTERVAL_S = 30.0  # heavier query — slower cadence

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
        try:
            mempool_items = loop.run_until_complete(memory.fetch_mempool()) or []
        except Exception as _mp_err:
            logger.warning("[MemoryWorker] fetch_mempool failed: %s", _mp_err)
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
            _send_msg(send_queue, bus.MEMORY_STATUS_UPDATED, name, "all", {
                "persistent_count": pcount,
                "mempool_size": len(mempool_items),
                "cognee_ready": True,
                "memory_backend_ready": True,
            })
            _send_msg(send_queue, bus.MEMORY_MEMPOOL_UPDATED, name, "all",
                      {"items": mempool_items, "count": len(mempool_items)})
            _send_msg(send_queue, bus.MEMORY_TOP_UPDATED, name, "all",
                      {"items": top_items, "count": len(top_items)})
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
        """Periodic publish — populates memory.topology + memory.knowledge_graph.
        Called every TOPOLOGY_PUBLISH_INTERVAL_S (heavier than status because
        of Kuzu queries). rFP_observatory_data_loading_v1 Phase 4."""
        # Topology: topic-cluster heatmap from top memories
        try:
            top_items = memory.get_top_memories(n=200) or []
            clusters = _classify_topology(top_items)
            kg_stats: dict = {}
            try:
                if getattr(memory, "_graph", None) is not None:
                    kg_stats = memory._graph.get_stats() or {}
            except Exception:
                kg_stats = {}
            _send_msg(send_queue, bus.MEMORY_TOPOLOGY_UPDATED, name, "all", {
                "by_topic_cluster": clusters,
                "by_entity_type": kg_stats,
                "total_classified": sum(c["count"] for c in clusters.values()),
            })
        except Exception as _topo_err:
            logger.warning(
                "[MemoryWorker] memory.topology publish failed: %s", _topo_err)
        # Knowledge graph: entity counts + sample edges from Kuzu
        try:
            kg_payload: dict = {"available": False, "nodes": [], "edges": [],
                                 "stats": {}}
            graph = getattr(memory, "_graph", None)
            if graph is not None:
                stats = graph.get_stats() or {}
                kg_payload["stats"] = stats
                # Sample up to 50 nodes per entity type for force-directed viz
                node_tables = ["Person", "Topic", "BodyEntity", "MindEntity",
                               "SpiritEntity", "Media"]
                nodes: list[dict] = []
                for table in node_tables:
                    try:
                        df = graph._conn.execute(
                            f"MATCH (e:{table}) RETURN e.name LIMIT 50"
                        ).get_as_df()
                        for _, row in df.iterrows():
                            nodes.append({
                                "id": f"{table}:{row.iloc[0]}",
                                "label": str(row.iloc[0]),
                                "type": table,
                            })
                    except Exception:
                        continue
                # Sample up to 100 edges for visualization
                edges: list[dict] = []
                edge_quota = 100
                for src_t in node_tables:
                    if edge_quota <= 0:
                        break
                    for dst_t in node_tables:
                        if edge_quota <= 0:
                            break
                        rel = f"REL_{src_t}_{dst_t}"
                        try:
                            df = graph._conn.execute(
                                f"MATCH (a:{src_t})-[r:{rel}]->(b:{dst_t}) "
                                f"RETURN a.name, r.rel_type, b.name LIMIT 20"
                            ).get_as_df()
                            for _, row in df.iterrows():
                                if edge_quota <= 0:
                                    break
                                edges.append({
                                    "source": f"{src_t}:{row.iloc[0]}",
                                    "type": str(row.iloc[1]) if row.iloc[1] else "rel",
                                    "target": f"{dst_t}:{row.iloc[2]}",
                                })
                                edge_quota -= 1
                        except Exception:
                            continue
                kg_payload["nodes"] = nodes
                kg_payload["edges"] = edges
                kg_payload["available"] = True
            _send_msg(send_queue, bus.MEMORY_KNOWLEDGE_GRAPH_UPDATED, name, "all",
                      kg_payload)
        except Exception as _kg_err:
            logger.warning(
                "[MemoryWorker] memory.knowledge_graph publish failed: %s",
                _kg_err)

    # ── Microkernel v2 Phase B.1 §6 — readiness/hibernate reporter ──
    from titan_plugin.core.readiness_reporter import trivial_reporter
    def _b1_save_state():
        return []
    _b1_reporter = trivial_reporter(
        worker_name=name, layer="L2", send_queue=send_queue,
        save_state_cb=_b1_save_state,
    )

    while True:
        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            # Send heartbeat every 10s
            if time.time() - last_heartbeat > 10.0:
                _send_heartbeat(send_queue, name)
                last_heartbeat = time.time()
            # Periodic memory.status publish (5s cadence) — keeps api cache warm.
            if time.time() - last_status_publish > STATUS_PUBLISH_INTERVAL_S:
                _publish_memory_status()
                last_status_publish = time.time()
            # Periodic memory.topology + memory.knowledge_graph (30s cadence;
            # heavier Kuzu queries). rFP_observatory_data_loading_v1 Phase 4.
            if time.time() - last_topology_publish > TOPOLOGY_PUBLISH_INTERVAL_S:
                _publish_memory_topology()
                last_topology_publish = time.time()
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        msg_type = msg.get("type", "")

        # ── Microkernel v2 Phase B.1 §6 — shadow swap dispatch ────
        if _b1_reporter.handles(msg_type):
            _b1_reporter.handle(msg)
            if _b1_reporter.should_exit():
                break
            continue

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
        from titan_plugin.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[MemoryWorker] Shutdown: %s", msg.get("payload", {}).get("reason"))
            break

        if msg_type == bus.QUERY:
            _send_heartbeat(send_queue, name)  # keep alive during long queries
            _handle_query(msg, memory, send_queue, name, loop, config, recv_queue)
            last_heartbeat = time.time()

        elif msg_type == bus.MEMORY_ADD:
            # spirit_worker emits MEMORY_ADD on Maker-dialogue profile enrichment
            # (and any other text-ingest path that wants to bypass the mempool +
            # skip the discovery delay). Payload: {text, source, weight}. Call
            # inject_memory synchronously via the event loop, non-blocking on
            # errors so a bad payload can't starve the main recv loop.
            try:
                payload = msg.get("payload", {}) or {}
                text = payload.get("text", "")
                source = payload.get("source", "bus")
                weight = float(payload.get("weight", 1.0))
                if not text:
                    logger.debug("[MemoryWorker] MEMORY_ADD ignored (empty text)")
                else:
                    loop.run_until_complete(
                        memory.inject_memory(text=text, source=source, weight=weight)
                    )
                    logger.info(
                        "[MemoryWorker] MEMORY_ADD injected (source=%s, weight=%.2f, "
                        "text_len=%d)",
                        source, weight, len(text),
                    )
            except Exception as _ma_err:
                logger.warning("[MemoryWorker] MEMORY_ADD handler error: %s", _ma_err,
                               exc_info=True)
            last_heartbeat = time.time()

    logger.info("[MemoryWorker] Exiting")
    loop.close()


def _handle_query(msg: dict, memory, send_queue, name: str, loop, config: dict = None, recv_queue=None) -> None:
    """Dispatch QUERY to appropriate memory method and send response.

    ``recv_queue`` is required for actions that need to bus-bridge back to
    kernel (currently only ``run_meditation``'s ANCHOR_REQUEST round-trip).
    """
    payload = msg.get("payload", {})
    action = payload.get("action", "")
    rid = msg.get("rid")
    src = msg.get("src", "")

    try:
        if action == "query":
            text = payload.get("text", "")
            top_k = payload.get("top_k", 5)
            results = loop.run_until_complete(
                asyncio.wait_for(memory.query(text, top_k=top_k), timeout=30.0)
            )
            _send_response(send_queue, name, src, {"results": results}, rid)

        elif action == "add":
            text = payload.get("text", "")
            kwargs = {k: v for k, v in payload.items() if k not in ("action", "text")}
            # 2026-04-08 audit fix (I-012): bumped timeout 30s → 90s.
            # inject_memory calls _cognee_ingest which does FAISS embed + Kuzu
            # entity extraction (cognify) + DuckDB updates. Under load on T3,
            # this regularly exceeds 30s, causing recurring TimeoutError that
            # was previously logged with empty message because str(TimeoutError()) == ''.
            node_id = loop.run_until_complete(
                asyncio.wait_for(memory.inject_memory(text, **kwargs), timeout=90.0)
            )
            _send_response(send_queue, name, src, {"node_id": node_id}, rid)

        elif action == "add_to_mempool":
            user_prompt = payload.get("user_prompt", "")
            agent_response = payload.get("agent_response", "")
            user_identifier = payload.get("user_identifier", "Anonymous")
            loop.run_until_complete(
                memory.add_to_mempool(user_prompt, agent_response,
                                      user_identifier=user_identifier))
            _send_response(send_queue, name, src, {"success": True}, rid)

        elif action == "count":
            count = memory.get_persistent_count()
            _send_response(send_queue, name, src, {"count": count}, rid)

        elif action == "fetch_mempool":
            mempool = loop.run_until_complete(memory.fetch_mempool())
            # Apply decay before sending so weights are current
            for node in mempool:
                memory._apply_mempool_decay(node)
            _send_response(send_queue, name, src, {"mempool": mempool}, rid)

        elif action == "top_memories":
            n = payload.get("n", 5)
            top = memory.get_top_memories(n=n)
            _send_response(send_queue, name, src, {"memories": top}, rid)

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

        elif action == "growth_metrics":
            # Compute growth metrics inside worker (has _node_store access)
            import math as _math
            now = time.time()
            cutoff = now - 86400  # 24h

            # Learning velocity: weighted count of recently active persistent nodes
            effective_nodes = 0.0
            total_persistent = 0
            high_quality = 0
            for v in memory._node_store.values():
                if v.get("type") != "MemoryNode" or v.get("status") != "persistent":
                    continue
                total_persistent += 1
                if v.get("effective_weight", 1.0) >= 1.15:
                    high_quality += 1
                if v.get("created_at", 0) >= cutoff or v.get("last_accessed", 0) >= cutoff:
                    effective_nodes += v.get("effective_weight", 1.0)

            node_sat = payload.get("node_saturation_24h", 30)
            learning_vel = min(1.0, _math.log(effective_nodes + 1) / _math.log(node_sat + 1))
            directive_align = min(1.0, (high_quality / max(1, total_persistent)) * 1.2)

            _send_response(send_queue, name, src, {
                "learning_velocity": round(learning_vel, 4),
                "directive_alignment": round(directive_align, 4),
                "effective_nodes_24h": round(effective_nodes, 1),
                "total_persistent": total_persistent,
                "high_quality_count": high_quality,
            }, rid)

        elif action == "run_meditation":
            # Simplified meditation: classify mempool, score, prune, promote, consolidate
            logger.info("[MemoryWorker] Running meditation cycle...")
            # BUG-VAULT-COMMITS-NOT-LANDING fix (2026-04-29): on-chain vault
            # commit is now bus-bridged to the kernel. memory_worker still
            # runs MeditationEpoch with network_client=None (deployer keypair
            # stays in main process), but after promotions are decided we
            # send ANCHOR_REQUEST to kernel which signs + submits the TX
            # and replies with the real tx_signature. See ANCHOR_REQUEST
            # docstring in titan_plugin/bus.py for the wire contract.
            try:
                from titan_plugin.logic.meditation import MeditationEpoch
                # Create a lightweight meditation instance
                med = MeditationEpoch(
                    memory_graph=memory,
                    network_client=None,  # TX-submission delegated via ANCHOR_REQUEST
                    config=config or {},
                )
                # Wire Ollama Cloud if available
                ollama_key = config.get("ollama_cloud_api_key", "")
                ollama_url = config.get("ollama_cloud_base_url", "https://ollama.com/v1")
                if ollama_key:
                    try:
                        from titan_plugin.utils.ollama_cloud import OllamaCloudClient
                        med._ollama_cloud = OllamaCloudClient(api_key=ollama_key, base_url=ollama_url)
                    except Exception:
                        pass

                # Run simplified epoch (no TX submission inline; no studio, no social)
                candidates, fading, dead = loop.run_until_complete(memory.fetch_mempool_classified())
                total = len(candidates) + len(fading) + len(dead)

                # Prune dead
                pruned = 0
                for node in dead:
                    loop.run_until_complete(memory.prune_mempool_node(node["id"]))
                    pruned += 1

                # Score candidates — collect promotion decisions WITHOUT
                # migrating yet, so the on-chain anchor TX (next step) can
                # cover the full set with a single state_root commit.
                promoted_nodes: list = []
                for node in candidates:
                    _send_heartbeat(send_queue, name)  # keep alive during long scoring loop
                    last_heartbeat = time.time()
                    score, intensity = loop.run_until_complete(
                        asyncio.wait_for(med.get_hippocampus_score([node]), timeout=30.0)
                    )
                    if score >= 40.0:
                        promoted_nodes.append((node, intensity))

                # On-chain anchor (BUG-VAULT-COMMITS-NOT-LANDING fix). Build
                # state_root + payload from the promoted set, ask kernel to
                # submit the vault commit TX, then migrate every promoted
                # node with the real tx_signature. Falls back to
                # MEDITATION_LOCAL if the bus-bridge times out / errors so
                # the cycle still completes (graceful degradation).
                tx_signature = None
                anchor_error = None
                if promoted_nodes and recv_queue is not None:
                    try:
                        import json as _json_anchor
                        payload_json = _json_anchor.dumps(
                            [{"id": n["id"], "prompt": n.get("user_prompt", "")}
                             for n, _ in promoted_nodes],
                            default=str,
                        )
                        try:
                            from titan_plugin.utils.solana_client import generate_state_hash
                            state_root = "MERKLE_" + generate_state_hash(payload_json)[:16]
                        except Exception:
                            state_root = f"MEDITATION_{int(time.time())}"

                        logger.info(
                            "[MemoryWorker] Requesting on-chain anchor: "
                            "promoted=%d, root=%s",
                            len(promoted_nodes), state_root[:24],
                        )
                        anchor_result = _request_anchor_via_kernel(
                            send_queue, recv_queue, name,
                            state_root, payload_json, len(promoted_nodes),
                            timeout=120.0,
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
                sig_for_migration = tx_signature or "MEDITATION_LOCAL"
                for node, intensity in promoted_nodes:
                    loop.run_until_complete(
                        memory.migrate_to_persistent(
                            node["id"], sig_for_migration, intensity,
                        )
                    )
                promoted = len(promoted_nodes)

                # Consolidate Cognee (can be very slow — 120s timeout)
                _send_heartbeat(send_queue, name)
                last_heartbeat = time.time()
                try:
                    consolidated = loop.run_until_complete(
                        asyncio.wait_for(memory.consolidate(), timeout=120.0)
                    )
                except asyncio.TimeoutError:
                    logger.warning("[MemoryWorker] Cognee consolidation timed out (120s)")
                    consolidated = False

                result = {
                    "success": True,
                    "total_mempool": total,
                    "promoted": promoted,
                    "pruned": pruned,
                    "fading": len(fading),
                    "consolidated": consolidated,
                }
                logger.info(
                    "[MemoryWorker] Meditation complete: promoted=%d pruned=%d fading=%d consolidated=%s",
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

                # TimeChain: each promoted node → declarative fork (permanent memory)
                for node in candidates[:promoted]:
                    _node_id = node.get("id", "?")
                    _node_prompt = node.get("user_prompt", "")[:100]
                    _node_response = node.get("agent_response", "")[:100]
                    _send_msg(send_queue, bus.TIMECHAIN_COMMIT, name, "timechain", {
                        "fork": "declarative",
                        "thought_type": "declarative",
                        "source": "memory_promotion",
                        "content": {
                            "event": "MEMORY_PROMOTED",
                            "node_id": _node_id,
                            "prompt_hash": __import__("hashlib").sha256(
                                _node_prompt.encode()).hexdigest()[:16],
                            "response_hash": __import__("hashlib").sha256(
                                _node_response.encode()).hexdigest()[:16],
                        },
                        "significance": 0.5,
                        "novelty": 0.6,
                        "coherence": 0.5,
                        "tags": ["memory_node", "promoted", "persistent"],
                        "db_ref": f"memory_nodes:{_node_id}",
                        "neuromods": {},
                        "chi_available": 0.5,
                        "attention": 0.5,
                        "i_confidence": 0.5,
                        "chi_coherence": 0.3,
                    })

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
        from titan_plugin.bus import record_send_drop
        record_send_drop(src, dst, msg_type)


def _request_anchor_via_kernel(
    send_queue, recv_queue, name: str,
    state_root: str, payload_json: str, promoted_count: int,
    timeout: float = 120.0,
) -> dict:
    """Bus-bridge: ask kernel main process to submit the on-chain vault TX.

    Closes BUG-VAULT-COMMITS-NOT-LANDING. memory_worker subprocess has
    ``network_client=None`` (deployer keypair stays in main process for
    security), so on-chain TX submission is delegated to the kernel via
    ``ANCHOR_REQUEST``. Kernel's ``_anchor_request_loop`` handles the
    request and replies via ``bus.RESPONSE`` matched on ``rid``.

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

    During the wait, this function:
      - drains ``recv_queue`` looking for the matching RESPONSE
      - re-injects every non-matching message back at the end of recv_queue
        so the main loop processes it after meditation returns
      - sends heartbeats (throttled to once per 3s by ``_send_heartbeat``)
        so Guardian doesn't kill the worker during long TX waits

    Returns the ANCHOR response payload dict (with ``tx_signature`` /
    ``error`` keys) or ``{"error": "anchor_request_timeout"}`` on timeout.
    """
    import uuid
    from queue import Empty

    rid = uuid.uuid4().hex
    _send_msg(send_queue, bus.ANCHOR_REQUEST, name, "kernel", {
        "state_root": state_root,
        "payload": payload_json,
        "promoted_count": promoted_count,
        "ts": time.time(),
    }, rid=rid)

    deadline = time.time() + timeout
    deferred: list = []
    last_hb = 0.0

    try:
        while time.time() < deadline:
            try:
                # Bounded wait so we can interleave heartbeats.
                msg = recv_queue.get(timeout=min(2.0, max(0.05, deadline - time.time())))
            except Empty:
                if time.time() - last_hb > 3.0:
                    _send_heartbeat(send_queue, name)
                    last_hb = time.time()
                continue

            # Found our response?
            if msg.get("type") == bus.RESPONSE and msg.get("rid") == rid:
                return msg.get("payload", {}) or {}

            # Defer for re-injection — preserves order so the main loop
            # sees these in arrival order after meditation returns.
            deferred.append(msg)
    finally:
        for dmsg in deferred:
            try:
                recv_queue.put_nowait(dmsg)
            except Exception:
                logger.warning(
                    "[MemoryWorker] Could not re-inject deferred msg "
                    "type=%s after anchor wait", dmsg.get("type"),
                )

    return {"error": "anchor_request_timeout"}


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
