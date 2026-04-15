"""
api/maker.py
Authenticated Maker Console — secured by Ed25519 signature verification.

Only the holder of the Maker keypair can access these endpoints.
"""
import logging

from fastapi import APIRouter, Request, Depends

from .auth import verify_maker_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/maker", tags=["Maker Console"], dependencies=[Depends(verify_maker_auth)])


def _get_plugin(request: Request):
    return request.app.state.titan_plugin


def _ok(data: dict):
    return {"status": "ok", "data": data}


# ---------------------------------------------------------------------------
# POST /maker/directive — Submit New Prime Directive
# ---------------------------------------------------------------------------
@router.post("/directive")
async def submit_directive(request: Request):
    """
    Submit a new Prime Directive update.
    Requires Ed25519 signature. Calls soul.evolve_soul() with the payload.

    Body JSON: {"memo_data": "New directive text", "memo_signature": "base58_sig"}
    """
    plugin = _get_plugin(request)
    body = await request.json()

    memo_data = body.get("memo_data", "")
    memo_signature = body.get("memo_signature", "")

    if not memo_data:
        return {"status": "error", "detail": "memo_data is required."}

    try:
        result = await plugin.soul.evolve_soul(memo_data, memo_signature)

        # Emit event for WebSocket subscribers
        if hasattr(plugin, "event_bus"):
            await plugin.event_bus.emit("directive_update", {
                "memo_data": memo_data[:200],
                "result": result,
                "new_gen": plugin.soul.current_gen,
            })

        logger.info("[Maker] Directive submitted: %s → %s", memo_data[:50], result)
        return _ok({"result": result, "soul_gen": plugin.soul.current_gen})

    except Exception as e:
        logger.error("[Maker] Directive submission failed: %s", e)
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# POST /maker/divine-inspiration — Manual DI Trigger
# ---------------------------------------------------------------------------
@router.post("/divine-inspiration")
async def trigger_divine_inspiration(request: Request):
    """
    Manually trigger Divine Inspiration via API (bypasses on-chain memo).
    Forces the MoodEngine into ZEN state (1.0).
    """
    plugin = _get_plugin(request)
    try:
        plugin.mood_engine.force_zen()

        if hasattr(plugin, "event_bus"):
            await plugin.event_bus.emit("divine_inspiration", {
                "source": "maker_api",
                "mood_before": plugin.mood_engine.previous_mood,
            })

        logger.info("[Maker] Divine Inspiration triggered via API.")
        return _ok({"result": "ZEN state activated.", "mood": 1.0})

    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# POST /maker/inject-memory — Direct Memory Injection
# ---------------------------------------------------------------------------
@router.post("/inject-memory")
async def inject_memory(request: Request):
    """
    Direct Memory Injection — bypass the research/mempool pipeline and place
    a high-weight memory directly into the Titan's persistent Cognee graph.

    Use this when the Maker needs the Titan to know something immediately
    (e.g., deadline changes, critical context, identity corrections).

    Body JSON: {"text": "The project deadline moved to March 15th.", "weight": 5.0}
    - text (required): The memory content to inject.
    - weight (optional, default 5.0): Base weight (organic memories start at 1.0).
    """
    plugin = _get_plugin(request)
    body = await request.json()

    text = body.get("text", "").strip()
    weight = float(body.get("weight", 5.0))

    if not text:
        return {"status": "error", "detail": "text is required."}

    if weight < 1.0 or weight > 10.0:
        return {"status": "error", "detail": "weight must be between 1.0 and 10.0."}

    try:
        result = await plugin.memory.inject_memory(text, source="maker", weight=weight)

        # Emit event for WebSocket subscribers
        if hasattr(plugin, "event_bus"):
            await plugin.event_bus.emit("memory_injection", {
                "source": "maker_api",
                "text_preview": text[:100],
                "node_id": result["node_id"],
                "weight": result["weight"],
            })

        logger.info("[Maker] Memory injected: '%s' (weight=%.1f)", text[:50], weight)
        return _ok(result)

    except Exception as e:
        logger.error("[Maker] Memory injection failed: %s", e)
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# POST /maker/resurrect — Trigger Resurrection from Limbo State
# ---------------------------------------------------------------------------
@router.post("/resurrect")
async def trigger_resurrection(request: Request):
    """
    Accepts a Maker Shard 1 envelope and triggers the full Resurrection Protocol.
    Only functional when the Titan is in Limbo State (no keypair available).

    Body JSON: {
        "shard1_envelope": "hex-encoded Maker envelope",
        "skip_onchain": false  (optional — skip on-chain Shard 3 fetch)
    }
    """
    from titan_plugin import TitanPlugin

    if not TitanPlugin._limbo_mode:
        return {"status": "error", "detail": "Titan is not in Limbo State. Resurrection not needed."}

    plugin = _get_plugin(request)
    body = await request.json()

    shard1_envelope = body.get("shard1_envelope", "").strip()
    skip_onchain = body.get("skip_onchain", False)

    if not shard1_envelope:
        return {"status": "error", "detail": "shard1_envelope is required."}

    try:
        import subprocess
        import sys
        import os

        # Build resurrection command
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "resurrection.py")
        script_path = os.path.normpath(script_path)

        cmd = [
            sys.executable, script_path,
            "--shard1", shard1_envelope,
        ]
        if skip_onchain:
            cmd.append("--skip-onchain")

        logger.info("[Maker] Launching Resurrection Protocol...")

        # Run resurrection as subprocess (it handles its own file I/O).
        # Phase E.2.2 fix: 120s timeout would freeze the event loop for
        # entire duration. Wrap in to_thread so /v4/* + /health stay
        # responsive while resurrection runs.
        import asyncio
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            logger.info("[Maker] Resurrection Protocol completed successfully.")

            # Emit event
            if hasattr(plugin, "event_bus"):
                await plugin.event_bus.emit("resurrection", {
                    "status": "complete",
                    "output": result.stdout[-500:] if result.stdout else "",
                })

            return _ok({
                "result": "Resurrection complete. Restart the Titan to boot with restored identity.",
                "output": result.stdout[-1000:] if result.stdout else "",
            })
        else:
            logger.error("[Maker] Resurrection failed: %s", result.stderr)
            return {
                "status": "error",
                "detail": "Resurrection failed.",
                "stderr": result.stderr[-1000:] if result.stderr else "",
            }

    except subprocess.TimeoutExpired:
        return {"status": "error", "detail": "Resurrection timed out after 120 seconds."}
    except Exception as e:
        logger.error("[Maker] Resurrection error: %s", e)
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# POST /maker/verify-cluster — Knowledge Cluster Priority Boost
# ---------------------------------------------------------------------------
@router.post("/verify-cluster")
async def verify_cluster(request: Request):
    """
    The Maker "gardens" the Titan's mind by verifying a knowledge cluster.
    Sends a priority boost signal to the gatekeeper — memories in the verified
    cluster get a weight boost during the next meditation epoch.

    Body JSON: {
        "cluster_name": "Solana Architecture",
        "priority_boost": 1.5
    }
    """
    plugin = _get_plugin(request)
    body = await request.json()

    cluster_name = body.get("cluster_name", "").strip()
    priority_boost = float(body.get("priority_boost", 1.5))

    if not cluster_name:
        return {"status": "error", "detail": "cluster_name is required."}

    if priority_boost < 1.0 or priority_boost > 3.0:
        return {"status": "error", "detail": "priority_boost must be between 1.0 and 3.0."}

    try:
        # Apply boost to memories matching the cluster
        boosted_count = 0
        if plugin.memory:
            from ..api.dashboard import _TOPIC_KEYWORDS
            keywords = _TOPIC_KEYWORDS.get(cluster_name)
            if keywords is None:
                return {"status": "error", "detail": f"Unknown cluster: {cluster_name}"}

            for node_id, v in plugin.memory._node_store.items():
                if v.get("type") != "MemoryNode" or v.get("status") != "persistent":
                    continue
                content = f"{v.get('user_prompt', '')} {v.get('agent_response', '')}".lower()
                words = set(content.split())
                if len(words & keywords) > 0:
                    current_weight = v.get("effective_weight", 1.0)
                    v["effective_weight"] = min(current_weight * priority_boost, 10.0)
                    boosted_count += 1

        # Emit event for WebSocket subscribers
        if hasattr(plugin, "event_bus"):
            await plugin.event_bus.emit("cluster_verified", {
                "cluster": cluster_name,
                "priority_boost": priority_boost,
                "memories_boosted": boosted_count,
                "source": "maker_api",
            })

        # Log to observatory DB
        obs_db = getattr(plugin, "_observatory_db", None)
        if obs_db:
            obs_db.record_event("cluster_verified", f"Maker verified: {cluster_name}", {
                "cluster": cluster_name,
                "boost": priority_boost,
                "count": boosted_count,
            })

        logger.info(
            "[Maker] Cluster verified: %s (boost=%.1fx, %d memories)",
            cluster_name, priority_boost, boosted_count,
        )
        return _ok({
            "cluster": cluster_name,
            "priority_boost": priority_boost,
            "memories_boosted": boosted_count,
        })

    except Exception as e:
        logger.error("[Maker] Cluster verification failed: %s", e)
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# GET /maker/audit — Full Sovereignty Audit
# ---------------------------------------------------------------------------
@router.get("/audit")
async def get_audit(request: Request):
    """
    Full sovereignty audit: gatekeeper stats, research usage,
    guardian blocks, RL buffer size, memory health.
    """
    plugin = _get_plugin(request)
    try:
        # RL buffer stats
        buffer_size = 0
        try:
            buffer_size = len(plugin.recorder.buffer) if plugin.recorder.buffer else 0
        except Exception:
            pass

        # Memory health
        persistent = plugin.memory.get_persistent_count()
        mempool = await plugin.memory.fetch_mempool()
        metrics = await plugin.memory.fetch_social_metrics()

        # Growth metrics
        learning = await plugin.metabolism.get_learning_velocity()
        social = await plugin.metabolism.get_social_density()
        health = await plugin.metabolism.get_metabolic_health()
        alignment = await plugin.metabolism.get_directive_alignment()

        # Soul state
        directives = await plugin.soul.get_active_directives()

        return _ok({
            "soul": {
                "generation": plugin.soul.current_gen,
                "directives": directives,
                "nft_address": plugin.soul._nft_address,
            },
            "memory": {
                "persistent_nodes": persistent,
                "mempool_pending": len(mempool),
                "cognee_ready": plugin.memory._cognee_ready,
            },
            "rl_buffer": {
                "transitions": buffer_size,
            },
            "growth_metrics": {
                "learning_velocity": round(learning, 4),
                "social_density": round(social, 4),
                "metabolic_health": round(health, 4),
                "directive_alignment": round(alignment, 4),
            },
            "social_metrics": metrics,
            "execution_mode": plugin._last_execution_mode,
            "research_sources": plugin._last_research_sources,
        })

    except Exception as e:
        return {"status": "error", "detail": str(e)}
