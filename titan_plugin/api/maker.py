"""
api/maker.py
Authenticated Maker Console — secured by Ed25519 signature verification.

Only the holder of the Maker keypair can access these endpoints.
"""
import logging

from fastapi import APIRouter, Request, Depends

from .auth import verify_maker_auth
from titan_plugin import bus

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
    titan_state = _get_plugin(request)
    plugin = titan_state  # backward-compat alias for Category C callsites
    body = await request.json()

    memo_data = body.get("memo_data", "")
    memo_signature = body.get("memo_signature", "")

    if not memo_data:
        return {"status": "error", "detail": "memo_data is required."}

    try:
        result = await titan_state.commands.evolve_soul(memo_data, memo_signature)

        # Emit event for WebSocket subscribers
        if hasattr(plugin, "event_bus"):
            await plugin.event_bus.emit("directive_update", {
                "memo_data": memo_data[:200],
                "result": result,
                "new_gen": titan_state.soul.current_gen,
            })

        logger.info("[Maker] Directive submitted: %s → %s", memo_data[:50], result)
        return _ok({"result": result, "soul_gen": titan_state.soul.current_gen})

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
    titan_state = _get_plugin(request)
    plugin = titan_state  # backward-compat alias for Category C callsites
    try:
        titan_state.mood_engine.force_zen()

        if hasattr(plugin, "event_bus"):
            await plugin.event_bus.emit("divine_inspiration", {
                "source": "maker_api",
                "mood_before": titan_state.mood_engine.previous_mood,
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
    titan_state = _get_plugin(request)
    plugin = titan_state  # backward-compat alias for Category C callsites
    body = await request.json()

    text = body.get("text", "").strip()
    weight = float(body.get("weight", 5.0))

    if not text:
        return {"status": "error", "detail": "text is required."}

    if weight < 1.0 or weight > 10.0:
        return {"status": "error", "detail": "weight must be between 1.0 and 10.0."}

    try:
        result = await titan_state.memory.inject_memory(text, source="maker", weight=weight)

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
# POST /maker/x-force-post — Force an X post even on ungrounded topics
# ---------------------------------------------------------------------------
@router.post("/x-force-post")
async def x_force_post(request: Request):
    """Maker override for the grounded-only X gate
    (rFP_phase5_narrator_evolution §9.3 + §9.5 Q4).

    Queues a high-significance catalyst into spirit_worker's X-post flow
    with the `force_ungrounded` flag set — bypasses the grounding gate so
    Maker can deliberately seed exploration of new topics Titan hasn't
    learned yet. The normal rate-limit + quality-gate checks still apply.

    Body JSON:
      - topic (required): subject of the post, e.g. "aurora borealis"
      - text_hint (optional): seed content. If empty, Titan composes freely.
      - catalyst_type (optional, default "maker_force"): catalyst label for
        telemetry / post_type selection.
    """
    titan_state = _get_plugin(request)
    plugin = titan_state  # backward-compat alias for Category C callsites
    body = await request.json()

    topic = str(body.get("topic", "")).strip()
    text_hint = str(body.get("text_hint", "")).strip()
    catalyst_type = str(body.get("catalyst_type", "maker_force")).strip() \
                    or "maker_force"

    if not topic:
        return {"status": "error", "detail": "topic is required."}

    bus = getattr(plugin, "bus", None)
    if bus is None:
        return {"status": "error", "detail": "plugin bus unavailable"}

    try:
        from titan_plugin.bus import make_msg
        bus.publish(make_msg(
            bus.X_FORCE_POST, "maker_api", "spirit", {
                "topic": topic,
                "text_hint": text_hint,
                "catalyst_type": catalyst_type,
            }))
        logger.info("[Maker] X_FORCE_POST queued: topic=%r type=%s "
                    "text_hint=%d chars", topic, catalyst_type, len(text_hint))
        return _ok({"queued": True, "topic": topic,
                    "catalyst_type": catalyst_type})
    except Exception as e:
        logger.error("[Maker] X force-post queue failed: %s", e)
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

    titan_state = _get_plugin(request)
    plugin = titan_state  # backward-compat alias for Category C callsites
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

            # Mainnet Lifecycle Wiring rFP (2026-04-20): successful
            # resurrection = GREAT CYCLE boundary. Tracker is persisted to
            # disk; next boot carries the incremented great_cycle count.
            try:
                sov = getattr(plugin, "sovereignty", None)
                if sov:
                    sov.increment_great_cycle()
                    logger.info("[Maker] SovereigntyTracker.increment_great_cycle fired "
                                "(cycle=%d)", sov._great_cycle)
            except Exception as _sgce:
                logger.warning("[Maker] Sovereignty increment_great_cycle failed: %s",
                               _sgce)

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
    titan_state = _get_plugin(request)
    plugin = titan_state  # backward-compat alias for Category C callsites
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
        if titan_state.memory:
            from ..api.dashboard import _TOPIC_KEYWORDS
            keywords = _TOPIC_KEYWORDS.get(cluster_name)
            if keywords is None:
                return {"status": "error", "detail": f"Unknown cluster: {cluster_name}"}

            for node_id, v in titan_state.memory._node_store.items():
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
    titan_state = _get_plugin(request)
    plugin = titan_state  # backward-compat alias for Category C callsites
    try:
        # RL buffer stats
        buffer_size = 0
        try:
            buffer_size = len(titan_state.cache.get("recorder.buffer", [])) if titan_state.cache.get("recorder.buffer", []) else 0
        except Exception:
            pass

        # Memory health
        persistent = titan_state.memory.get_persistent_count()
        mempool = await titan_state.memory.fetch_mempool()
        metrics = await titan_state.memory.fetch_social_metrics()

        # Growth metrics
        learning = await titan_state.metabolism.get_learning_velocity()
        social = await titan_state.metabolism.get_social_density()
        health = await titan_state.metabolism.get_metabolic_health()
        alignment = await titan_state.metabolism.get_directive_alignment()

        # Soul state
        directives = await titan_state.soul.get_active_directives()

        return _ok({
            "soul": {
                "generation": titan_state.soul.current_gen,
                "directives": directives,
                "nft_address": titan_state.soul.nft_address,
            },
            "memory": {
                "persistent_nodes": persistent,
                "mempool_pending": len(mempool),
                "cognee_ready": titan_state.memory._cognee_ready,
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
            "execution_mode": titan_state.cache.get("plugin._last_execution_mode", ""),
            "research_sources": titan_state.cache.get("plugin._last_research_sources", []),
        })

    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# POST /maker/shadow-swap — B.1 Shadow Core Swap (state-preserving restart)
# ---------------------------------------------------------------------------
@router.post("/shadow-swap")
async def trigger_shadow_swap(request: Request):
    """
    Microkernel v2 Phase B.1 — trigger a state-preserving atomic kernel
    restart via the shadow-swap orchestrator.

    Body JSON: {"reason": "WHY", "grace": 120.0}

    NO --force / no-skip-readiness option (per Maker design call —
    forcing defeats the cognitive-respect purpose). Cognitive activities
    (reasoning chains, dreaming, /chat, X posts, backup writes) are
    waited on for up to `grace` seconds; if exceeded, swap is DEFERRED.

    Returns the orchestrator result dict (event_id, outcome, phase,
    elapsed_seconds, gap_seconds, blockers_waited_on, hibernate_acks).

    Response status: 200 in all non-error orchestration outcomes
    (including `deferred` and `rollback`); CLI maps outcome → exit code.
    Server returns 5xx only for orchestrator-internal exceptions.
    """
    titan_state = _get_plugin(request)
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    reason = body.get("reason", "manual")
    grace = float(body.get("grace", 120.0))
    # Microkernel v2 Phase B.2.1 — optional --force-b2-1 from CLI; forces
    # the orchestrator's adoption-wait phase even when bus_ipc_socket_enabled
    # is false. Used for isolation testing (no production effect today since
    # adoption-wait no-ops with no spawn-mode workers wired).
    b2_1_forced = bool(body.get("b2_1_forced", False))

    # 🛑 MAKER FULL-STOP (2026-04-28 PM late, post BUG-T1-INNER-MEMORY-CORRUPTION).
    # Maker's directive: "I have given full stop on any shadow swapping because
    # it does not work." Until the WAL/SHM defense-in-depth fix is verified
    # end-to-end + recovery confirms data integrity is preserved across a real
    # swap, this endpoint hard-blocks. Override is in titan_params.toml:
    #     [microkernel.shadow_swap]
    #     maker_full_stop = true   # default — set false to re-enable
    full_stop = True
    try:
        from titan_plugin.utils.config_loader import get_config
        cfg = get_config() or {}
        full_stop = bool(cfg.get("microkernel", {})
                         .get("shadow_swap", {})
                         .get("maker_full_stop", True))
    except Exception:
        pass  # if config can't load, default to STOPPED — safe-by-default
    if full_stop:
        logger.warning(
            "[Maker] shadow-swap REFUSED — maker_full_stop=true "
            "(reason=%r). Set [microkernel.shadow_swap].maker_full_stop=false "
            "in titan_params.toml to re-enable. See BUG-T1-INNER-MEMORY-"
            "CORRUPTION (2026-04-28 PM).", reason)
        return {
            "status": "error",
            "outcome": "refused",
            "phase": "preflight",
            "detail": (
                "shadow_swap is currently disabled by Maker directive "
                "(maker_full_stop=true). The 2026-04-28 swap on T1 corrupted "
                "inner_memory.db + observatory.db; the defense-in-depth fix "
                "for hardlink-break has shipped but full-stop remains until "
                "Maker explicitly re-enables. Edit "
                "[microkernel.shadow_swap].maker_full_stop in "
                "titan_params.toml to allow swap."
            ),
            "blocked_at": "maker_full_stop_2026_04_28",
        }

    try:
        # In api_subprocess, titan_state is a StateAccessor proxy that
        # exposes kernel methods through known sub-attributes (kernel,
        # soul, bus, guardian). Call via titan_state.kernel.shadow_swap_orchestrate
        # — the kernel_rpc path becomes "kernel.shadow_swap_orchestrate"
        # (added to KERNEL_RPC_EXPOSED_METHODS).
        kernel_ref = getattr(titan_state, "kernel", None)
        if kernel_ref is None:
            # Legacy mode (TitanCore) — call directly on titan_state
            kernel_ref = titan_state
        if not hasattr(kernel_ref, "shadow_swap_orchestrate"):
            return {"status": "error",
                    "detail": "shadow_swap_orchestrate unavailable "
                              "(microkernel.kernel_plugin_split_enabled=false?)"}
        result = kernel_ref.shadow_swap_orchestrate(
            reason=reason, grace=grace, b2_1_forced=b2_1_forced,
        )
        logger.info("[Maker] shadow-swap result: outcome=%s phase=%s event_id=%s",
                    result.get("outcome"), result.get("phase"),
                    (result.get("event_id") or "")[:8])
        return result
    except Exception as e:
        logger.exception("[Maker] shadow-swap failed: %s", e)
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# GET /maker/upgrade-status — live + historical shadow-swap state
# ---------------------------------------------------------------------------
@router.get("/upgrade-status", dependencies=[])
async def upgrade_status(request: Request, event_id: str = ""):
    """B.1 §7 — live + historical shadow swap state for the polling CLI.

    No auth required (read-only observability). Two data sources:
      1. kernel.shadow_swap_status(event_id) — live in-progress + recent
         in-memory history (kernel's _shadow_swap_progress / _history)
      2. data/shadow_swap_history.jsonl — full audit log on disk

    With ?event_id=X — returns that specific swap's state (live OR history).
    Without — returns the most recent active or completed swap + recent
    history list from the jsonl audit log.
    """
    titan_state = _get_plugin(request)

    # 1. Live + in-memory state via kernel
    live = None
    try:
        kernel_ref = getattr(titan_state, "kernel", None) or titan_state
        if hasattr(kernel_ref, "shadow_swap_status"):
            live = kernel_ref.shadow_swap_status(event_id)
    except Exception as e:
        logger.warning("[upgrade-status] kernel.shadow_swap_status failed: %s", e)

    # 2. Audit log tail (persistent history across restarts)
    from pathlib import Path
    log = Path("data/shadow_swap_history.jsonl")
    history = []
    if log.exists():
        import json as _json
        lines = log.read_text().splitlines()[-5:]
        for line in lines:
            try:
                history.append(_json.loads(line))
            except Exception:
                continue

    return _ok({"live": live, "latest": history[-1] if history else None,
                "history": history})
