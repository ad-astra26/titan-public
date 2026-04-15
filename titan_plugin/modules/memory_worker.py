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
    _send_msg(send_queue, "MODULE_READY", name, "guardian", {})

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

    while True:
        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            # Send heartbeat every 10s
            if time.time() - last_heartbeat > 10.0:
                _send_heartbeat(send_queue, name)
                last_heartbeat = time.time()
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        msg_type = msg.get("type", "")

        if msg_type == "MODULE_SHUTDOWN":
            logger.info("[MemoryWorker] Shutdown: %s", msg.get("payload", {}).get("reason"))
            break

        if msg_type == "QUERY":
            _send_heartbeat(send_queue, name)  # keep alive during long queries
            _handle_query(msg, memory, send_queue, name, loop, config)
            last_heartbeat = time.time()

    logger.info("[MemoryWorker] Exiting")
    loop.close()


def _handle_query(msg: dict, memory, send_queue, name: str, loop, config: dict = None) -> None:
    """Dispatch QUERY to appropriate memory method and send response."""
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
            try:
                from titan_plugin.logic.meditation import MeditationEpoch
                # Create a lightweight meditation instance
                med = MeditationEpoch(
                    memory_graph=memory,
                    network_client=None,  # No TX in simplified mode
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

                # Run simplified epoch (no TX, no studio, no social)
                candidates, fading, dead = loop.run_until_complete(memory.fetch_mempool_classified())
                total = len(candidates) + len(fading) + len(dead)

                # Prune dead
                pruned = 0
                for node in dead:
                    loop.run_until_complete(memory.prune_mempool_node(node["id"]))
                    pruned += 1

                # Score candidates
                promoted = 0
                for node in candidates:
                    _send_heartbeat(send_queue, name)  # keep alive during long scoring loop
                    last_heartbeat = time.time()
                    score, intensity = loop.run_until_complete(
                        asyncio.wait_for(med.get_hippocampus_score([node]), timeout=30.0)
                    )
                    if score >= 40.0:
                        # Full migration: DuckDB + FAISS embed + Kuzu cognify
                        loop.run_until_complete(
                            memory.migrate_to_persistent(
                                node["id"], "MEDITATION_LOCAL", intensity,
                            )
                        )
                        promoted += 1

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
                    _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
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
                    _send_msg(send_queue, "TIMECHAIN_COMMIT", name, "timechain", {
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


def _send_response(send_queue, src: str, dst: str, payload: dict, rid: str) -> None:
    """Send a RESPONSE message back."""
    _send_msg(send_queue, "RESPONSE", src, dst, payload, rid)


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
    _send_msg(send_queue, "MODULE_HEARTBEAT", name, "guardian", {"rss_mb": round(rss_mb, 1)})
