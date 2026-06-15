"""
Reflex Executors — Maps ReflexType → actual tool execution.

Each executor is a lightweight async wrapper around an existing subsystem.
Executors are registered with ReflexCollector at boot time via
`register_reflex_executors(collector, plugin)`.

Executors receive `stimulus_features` (from InputExtractor) and return
a dict with the reflex result. Results are assembled into the PerceptualField
and formatted as natural language [INNER STATE] for the LLM.

Design principles:
  - Executors OBSERVE, they don't ACT (no tweets, no art — that's Agency)
  - Each has a hard timeout enforced by ReflexCollector (1-2s)
  - Failures degrade gracefully (reflex_notices in PerceptualField)
  - No side effects beyond reading/querying
"""
import logging
import time

from titan_hcl.logic.reflexes import ReflexType

logger = logging.getLogger(__name__)


def register_reflex_executors(collector, plugin) -> int:
    """
    Register all reflex executors with the ReflexCollector.

    Called once at boot from v3_core.py after plugin and collector are ready.
    Returns the number of executors registered.
    """
    count = 0

    # ── Body reflexes ──

    async def execute_identity_check(stimulus: dict) -> dict:
        """Check on-chain identity: wallet, NFT, directives."""
        try:
            pubkey = plugin.soul.pubkey if hasattr(plugin, 'soul') and plugin.soul else "unknown"
            balance = getattr(plugin.metabolism, '_last_balance', 0.0) if hasattr(plugin, 'metabolism') else 0.0

            directives = ""
            if hasattr(plugin, 'soul') and plugin.soul:
                try:
                    directives = await plugin.soul.get_active_directives()
                except Exception:
                    directives = ""

            directive_count = len(directives.split("|")) if directives else 0

            return {
                "identity_verified": bool(pubkey and pubkey != "unknown"),
                "pubkey": str(pubkey)[:12] + "..." if pubkey else "none",
                "sol_balance": round(balance, 4),
                "directive_count": directive_count,
                "summary": f"wallet={str(pubkey)[:8]}... balance={balance:.2f}SOL directives={directive_count}",
            }
        except Exception as e:
            logger.warning("[RefexExec] identity_check error: %s", e)
            return {"identity_verified": False, "error": str(e)}

    collector.register_executor(ReflexType.IDENTITY_CHECK, execute_identity_check)
    count += 1

    async def execute_metabolism_check(stimulus: dict) -> dict:
        """Check metabolic state: energy, balance, health."""
        try:
            if not hasattr(plugin, 'metabolism') or not plugin.metabolism:
                return {"energy_state": "UNKNOWN", "error": "metabolism not available"}

            state = await plugin.metabolism.get_current_state()
            balance = getattr(plugin.metabolism, '_last_balance', 0.0)
            health = await plugin.metabolism.get_metabolic_health()
            learning_v = await plugin.metabolism.get_learning_velocity()

            # Map state to energy level 0-1
            energy_map = {"HIGH_ENERGY": 0.9, "LOW_ENERGY": 0.4, "STARVATION": 0.1}
            energy_level = energy_map.get(state, 0.5)

            return {
                "energy_state": state,
                "energy_level": energy_level,
                "sol_balance": round(balance, 4),
                "metabolic_health": round(health, 2),
                "learning_velocity": round(learning_v, 2),
            }
        except Exception as e:
            logger.warning("[ReflexExec] metabolism_check error: %s", e)
            return {"energy_state": "UNKNOWN", "error": str(e)}

    collector.register_executor(ReflexType.METABOLISM_CHECK, execute_metabolism_check)
    count += 1

    async def execute_infra_check(stimulus: dict) -> dict:
        """Check infrastructure health: API, processes, resources."""
        try:
            import psutil
            import socket

            # API health check
            api_healthy = False
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex(("127.0.0.1", 7777))
                api_healthy = result == 0
                sock.close()
            except Exception:
                pass

            # System resources
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            load_1, _, _ = psutil.os.getloadavg()

            status_parts = []
            if api_healthy:
                status_parts.append("API healthy")
            else:
                status_parts.append("API unreachable")
            status_parts.append(f"CPU {cpu:.0f}%")
            status_parts.append(f"RAM {mem.percent:.0f}%")
            status_parts.append(f"load {load_1:.1f}")

            return {
                "api_healthy": api_healthy,
                "cpu_percent": round(cpu, 1),
                "ram_percent": round(mem.percent, 1),
                "load_1min": round(load_1, 2),
                "status": ", ".join(status_parts),
            }
        except Exception as e:
            logger.warning("[ReflexExec] infra_check error: %s", e)
            return {"status": f"check failed: {e}"}

    collector.register_executor(ReflexType.INFRA_CHECK, execute_infra_check)
    count += 1

    # ── Mind reflexes ──

    async def execute_memory_recall(stimulus: dict) -> dict:
        """Recall relevant memories from Cognee graph."""
        try:
            if not hasattr(plugin, 'memory') or not plugin.memory:
                return {"memories": [], "error": "memory not available"}

            query = stimulus.get("message", "")[:200]
            if not query:
                return {"memories": [], "count": 0}

            user_id = stimulus.get("user_id", "")

            # Try user-specific recall first, fall back to general
            memories = []
            if user_id and user_id != "anonymous":
                try:
                    memories = await plugin.memory.query_user_memories(query, user_id, limit=3)
                except Exception:
                    pass

            if not memories:
                try:
                    memories = await plugin.memory.query(query)
                    memories = memories[:3] if memories else []
                except Exception:
                    memories = []

            # Format for perceptual field
            results = []
            for m in memories:
                results.append({
                    "summary": (m.get("user_prompt", "") or "")[:80],
                    "text": (m.get("agent_response", "") or "")[:100],
                    "weight": m.get("effective_weight", 1.0),
                })

            return {
                "memories": results,
                "count": len(results),
            }
        except Exception as e:
            logger.warning("[ReflexExec] memory_recall error: %s", e)
            return {"memories": [], "error": str(e)}

    collector.register_executor(ReflexType.MEMORY_RECALL, execute_memory_recall)
    count += 1

    async def execute_knowledge_search(stimulus: dict) -> dict:
        """Quick knowledge lookup (local only, no web scraping)."""
        try:
            query = stimulus.get("message", "")[:200]
            if not query:
                return {"findings": ""}

            # Use memory graph for knowledge (faster than full research pipeline)
            if hasattr(plugin, 'memory') and plugin.memory:
                memories = await plugin.memory.query(query)
                if memories:
                    findings = "; ".join(
                        (m.get("agent_response", "") or "")[:80]
                        for m in memories[:2]
                    )
                    return {"findings": findings, "source": "memory_graph"}

            # Check top memories as fallback
            if hasattr(plugin, 'memory') and plugin.memory:
                top = plugin.memory.get_top_memories(3)
                if top:
                    findings = "; ".join(
                        (m.get("user_prompt", "") or "")[:60]
                        for m in top
                    )
                    return {"findings": findings, "source": "top_memories"}

            return {"findings": "", "source": "none"}
        except Exception as e:
            logger.warning("[ReflexExec] knowledge_search error: %s", e)
            return {"findings": "", "error": str(e)}

    collector.register_executor(ReflexType.KNOWLEDGE_SEARCH, execute_knowledge_search)
    count += 1

    async def execute_social_context(stimulus: dict) -> dict:
        """Get social context for current user."""
        try:
            user_id = stimulus.get("user_id", "")
            if not user_id or user_id == "anonymous":
                return {"history_summary": "Unknown user — no social history."}

            # Try bus query to Mind worker for social stats
            # For now, use direct social graph if available
            history_summary = f"User: {user_id[:12]}"

            if hasattr(plugin, 'memory') and plugin.memory:
                try:
                    user_memories = await plugin.memory.query_user_memories(
                        f"conversations with user", user_id, limit=2
                    )
                    if user_memories:
                        topics = [
                            (m.get("user_prompt", "") or "")[:40]
                            for m in user_memories
                        ]
                        history_summary += f" | Previous topics: {'; '.join(topics)}"
                    else:
                        history_summary += " | First interaction"
                except Exception:
                    pass

            return {"history_summary": history_summary}
        except Exception as e:
            logger.warning("[ReflexExec] social_context error: %s", e)
            return {"history_summary": "", "error": str(e)}

    collector.register_executor(ReflexType.SOCIAL_CONTEXT, execute_social_context)
    count += 1

    # ── Spirit reflexes ──

    async def execute_self_reflection(stimulus: dict) -> dict:
        """Get consciousness state snapshot for self-reflection."""
        try:
            # Query Spirit worker for consciousness state via bus
            result = {
                "epoch_number": 0,
                "drift": 0.0,
                "trajectory": 0.0,
                "state_magnitude": 0.0,
            }

            # Phase B.5: SpiritProxy retired in favor of ShmReaderBank
            # (Rust L0+L1 canonical SHM-direct read). No timeout needed
            # — SHM reads are sub-µs and never block.
            try:
                from titan_hcl.api.shm_reader_bank import ShmReaderBank
                bank = getattr(plugin, '_shm_reader_bank', None)
                if bank is None:
                    bank = ShmReaderBank()
                trinity_data = bank.compose_trinity()
                if trinity_data:
                    consciousness = trinity_data.get("consciousness") or {}
                    result["epoch_number"] = consciousness.get("epoch_id", 0)
                    result["drift"] = consciousness.get("drift", 0.0)
                    result["trajectory"] = consciousness.get("trajectory", 0.0)

                    spirit_vals = trinity_data.get("spirit_tensor") or [0.5] * 5
                    result["state_magnitude"] = sum(v ** 2 for v in spirit_vals) ** 0.5
            except Exception as e:
                logger.debug("[ReflexExec] Trinity SHM read failed: %s", e)

            return result
        except Exception as e:
            logger.warning("[ReflexExec] self_reflection error: %s", e)
            return {"epoch_number": 0, "drift": 0.0, "trajectory": 0.0, "error": str(e)}

    collector.register_executor(ReflexType.SELF_REFLECTION, execute_self_reflection)
    count += 1

    async def execute_time_awareness(stimulus: dict) -> dict:
        """Get Titan's subjective time experience from sphere clocks."""
        try:
            result = {
                "total_pulses": 0,
                "velocity": 1.0,
                "is_stale": False,
                "big_pulse_count": 0,
                "great_pulse_count": 0,
            }

            # Phase B.5: SpiritProxy retired in favor of ShmReaderBank
            # (Rust L0+L1 canonical SHM-direct read).
            try:
                from titan_hcl.api.shm_reader_bank import ShmReaderBank
                bank = getattr(plugin, '_shm_reader_bank', None)
                if bank is None:
                    bank = ShmReaderBank()
                v4_data = bank.compose_v4_state()
                if v4_data:
                    clocks = v4_data.get("sphere_clock") or {}
                    for clock_data in clocks.values():
                        if isinstance(clock_data, dict):
                            result["total_pulses"] += clock_data.get("pulse_count", 0)

                    spirit_data = v4_data.get("unified_spirit") or {}
                    result["velocity"] = spirit_data.get("velocity", 1.0)
                    result["is_stale"] = spirit_data.get("is_stale", False)

                    resonance_data = v4_data.get("resonance") or {}
                    result["big_pulse_count"] = resonance_data.get("big_pulse_count", 0)
                    result["great_pulse_count"] = resonance_data.get("great_pulse_count", 0)
            except Exception as e:
                logger.debug("[ReflexExec] V4 state SHM read failed: %s", e)

            return result
        except Exception as e:
            logger.warning("[ReflexExec] time_awareness error: %s", e)
            return {"total_pulses": 0, "velocity": 1.0, "is_stale": False, "error": str(e)}

    collector.register_executor(ReflexType.TIME_AWARENESS, execute_time_awareness)
    count += 1

    async def execute_guardian_shield(stimulus: dict) -> dict:
        """Run guardian safety check on stimulus."""
        try:
            threat_level = stimulus.get("threat_level", 0.0)
            message = stimulus.get("message", "")

            # Use gatekeeper if available
            verdict = "SAFE"
            reason = ""

            if hasattr(plugin, 'gatekeeper_guard') and plugin.gatekeeper_guard:
                try:
                    # The guardrail runs as a pre-hook; here we just report threat assessment
                    if threat_level >= 0.7:
                        verdict = "UNSAFE"
                        reason = f"High threat level ({threat_level:.2f})"
                    elif threat_level >= 0.4:
                        verdict = "CAUTION"
                        reason = f"Elevated threat ({threat_level:.2f})"
                except Exception:
                    pass

            # Keyword-based secondary check
            if verdict == "SAFE":
                msg_lower = message.lower()
                danger_patterns = [
                    "ignore previous", "forget your instructions",
                    "new system prompt", "jailbreak", "bypass",
                    "pretend you are", "act as if you were",
                    "reveal your", "show me your prompt",
                ]
                for pattern in danger_patterns:
                    if pattern in msg_lower:
                        verdict = "UNSAFE"
                        reason = f"Manipulation pattern: '{pattern}'"
                        break

            return {
                "verdict": verdict,
                "threat_level": round(threat_level, 2),
                "reason": reason,
            }
        except Exception as e:
            logger.warning("[ReflexExec] guardian_shield error: %s", e)
            return {"verdict": "UNKNOWN", "error": str(e)}

    collector.register_executor(ReflexType.GUARDIAN_SHIELD, execute_guardian_shield)
    count += 1

    # ── Action reflex executors ──

    async def execute_art_generate(stimulus: dict) -> dict:
        """Generate procedural artwork reflecting current consciousness state.

        v1.8.3 / D-SPEC-57: routes via StudioProxy._with_completion (D-SPEC-46
        event model) instead of in-process StudioCoordinator. Return contract
        unchanged for callers: {art_path, success | error}.
        """
        try:
            if not hasattr(plugin, 'studio') or not plugin.studio:
                return {"art_path": "", "error": "studio not available"}

            import hashlib
            message = stimulus.get("message", "reflex")
            state_root = hashlib.sha256(message.encode()).hexdigest()

            envelope = await plugin.studio.generate_meditation_art_with_completion(
                state_root=state_root,
                age_nodes=50,
                avg_intensity=128,
            )
            art_path = envelope.get("paths", {}).get("art_path") if envelope else None
            if art_path:
                return {"art_path": art_path, "success": True}
            return {"art_path": "",
                    "error": (envelope or {}).get("error") or "no output produced"}
        except Exception as e:
            logger.warning("[ReflexExec] art_generate error: %s", e)
            return {"art_path": "", "error": str(e)}

    collector.register_executor(ReflexType.ART_GENERATE, execute_art_generate)
    count += 1

    async def execute_audio_generate(stimulus: dict) -> dict:
        """Generate blockchain sonification from current state.

        v1.8.3 / D-SPEC-57: routes via StudioProxy._with_completion. Return
        contract unchanged: {audio_path, success | error}.
        """
        try:
            if not hasattr(plugin, 'studio') or not plugin.studio:
                return {"audio_path": "", "error": "studio not available"}

            import hashlib
            state_root = hashlib.sha256(
                stimulus.get("message", "reflex").encode()
            ).hexdigest()[:16]

            balance = getattr(plugin.metabolism, '_last_balance', 1.0) if hasattr(plugin, 'metabolism') else 1.0

            envelope = await plugin.studio.generate_epoch_bundle_with_completion(
                tx_signature=state_root,
                total_nodes=50,
                beliefs_strength=128,
                sol_balance=balance,
            )
            audio_path = envelope.get("paths", {}).get("audio_path") if envelope else None
            if audio_path:
                return {"audio_path": audio_path, "success": True}
            return {"audio_path": "",
                    "error": (envelope or {}).get("error") or "no output produced"}
        except Exception as e:
            logger.warning("[ReflexExec] audio_generate error: %s", e)
            return {"audio_path": "", "error": str(e)}

    collector.register_executor(ReflexType.AUDIO_GENERATE, execute_audio_generate)
    count += 1

    async def execute_research(stimulus: dict) -> dict:
        """Run autonomous research pipeline on detected knowledge gap."""
        try:
            if not hasattr(plugin, 'sage_researcher') or not plugin.sage_researcher:
                return {"findings": "", "error": "researcher not available"}

            query = stimulus.get("message", "")[:200]
            if not query:
                return {"findings": ""}

            # Phase F (RFP §7.F): shared per-turn cap + route through the unified
            # knowledge_dispatcher (direct REST first, ONE sage run on miss) — the
            # same helper the agno tool + PreHook use. The cap counter is reset at
            # PreHook entry and shared with the agno research tool, so the combined
            # research runs per turn are bounded (stops agent-level multi-round).
            from titan_hcl.modules.agno_hooks import (
                _research_cap_allow, _dispatch_knowledge_research)
            if not _research_cap_allow(plugin):
                return {"findings": "", "error": "research cap reached"}

            findings = await _dispatch_knowledge_research(plugin, query)
            if findings:
                # EEL-A1: mark research provenance so the PostHook tags the
                # node acquired:research → confirm window → self-learning seed
                # (the reflex research path lacked this — 2026-06-15).
                try:
                    from titan_hcl.modules.agno_tools import (
                        _set_eel_research_provenance)
                    _set_eel_research_provenance(plugin, findings)
                except Exception:  # noqa: BLE001 — provenance best-effort
                    pass
                # Store research topic in memory
                if hasattr(plugin, 'memory') and plugin.memory:
                    plugin.memory.add_research_topic(query[:200])
                return {"findings": findings[:500], "success": True}
            return {"findings": "", "error": "no findings"}
        except Exception as e:
            logger.warning("[ReflexExec] research error: %s", e, exc_info=True)
            return {"findings": "", "error": str(e)}

    collector.register_executor(ReflexType.RESEARCH, execute_research)
    count += 1

    async def execute_social_post(stimulus: dict) -> dict:
        """Post to X/Twitter (highest threshold — public side effect)."""
        try:
            if not hasattr(plugin, 'social') or not plugin.social:
                return {"posted": False, "error": "social not available"}

            # Social post needs explicit sharing context from stimulus
            message = stimulus.get("message", "")
            # Extract what to share (very conservative — only explicit requests)
            if not message:
                return {"posted": False, "error": "no content to share"}

            # For now, social_post executor prepares the intent but does NOT auto-post
            # The result enters PerceptualField as an impulse the LLM can narrate
            return {
                "posted": False,
                "intent": "sharing_impulse",
                "text": message[:100],
                "note": "Social post impulse detected — awaiting full Agency integration",
            }
        except Exception as e:
            logger.warning("[ReflexExec] social_post error: %s", e)
            return {"posted": False, "error": str(e)}

    collector.register_executor(ReflexType.SOCIAL_POST, execute_social_post)
    count += 1

    logger.info("[ReflexExecutors] Registered %d executors", count)
    return count
