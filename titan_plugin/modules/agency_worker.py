"""
Agency Worker — L3 Subprocess (rFP_microkernel_phase_a8 §A.8.6).

Owns AgencyModule + SelfAssessment + HelperRegistry + 8 helpers (every
helper currently wired in core/plugin.py:_register_helpers minus
social_post which is intentionally unregistered there). Removes the
LLM-call + helper-execution work from the parent event loop, keeping
the parent responsive during shadow-swap adoption + general bus-traffic
spikes — the architectural unblock that lets the shadow-swap re-test
finally validate L3 separation premise (rFP preamble).

Bus protocol:
  CONSUMES: QUERY(dst="agency_worker", payload.action=
    "handle_intent"|"dispatch_from_nervous_signals"|"assess"|
    "agency_stats"|"assessment_stats")
  EMITS:    RESPONSE(dst=requester, payload=action_result|list|assessment|stats)
            MODULE_READY (dst="guardian") — once on boot
            AGENCY_READY (dst="all") — once on boot, marks proxy bus-routed
            AGENCY_STATS (dst="all") — every 60s, refreshes proxy cache
            ASSESSMENT_STATS (dst="all") — every 60s, refreshes proxy cache
            MODULE_HEARTBEAT (dst="guardian") — every 30s

When `microkernel.a8_agency_subprocess_enabled=false` (default), this
worker is NOT autostarted by Guardian — the parent's local AgencyModule
+ SelfAssessment instances handle all calls (legacy behavior, byte-identical).

When the flag flips, parent's _agency / _agency_assessment become an
AgencyProxy + AssessmentProxy that bus.request(...) into this worker.
Parent's _agency_loop stays serial (one IMPULSE at a time) — preserves
the IMPULSE → ACTION_RESULT ordering invariant per §A.8.6 OBS gate.

See: titan-docs/rFP_microkernel_phase_a8_l2_l3_residency_completion.md §A.8.6
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from queue import Empty
from typing import Any, Optional
from titan_plugin import bus

logger = logging.getLogger("agency_worker")

_HEARTBEAT_INTERVAL_S = 30.0
_STATS_PUBLISH_INTERVAL_S = 60.0


def _build_llm_fn(inference_cfg: dict):
    """Replicates core/plugin.py:_create_agency_llm_fn — instantiated in
    the worker process so LLM calls don't cross the bus boundary.

    Two-tier fallback: OllamaCloud (primary) → Venice (fallback). Same
    model selection rules + max_token caps as the parent's local fn.
    """
    async def agency_llm(prompt: str, task: str = "agency_select") -> str:
        try:
            from titan_plugin.utils.ollama_cloud import OllamaCloudClient, get_model_for_task
            client = OllamaCloudClient(
                api_key=inference_cfg.get("ollama_cloud_api_key", ""),
                base_url=inference_cfg.get("ollama_cloud_base_url",
                                           "https://ollama.com/v1"),
            )
            model = get_model_for_task(task)
            max_tok = 800 if task == "agency_code_gen" else 200
            return await client.complete(prompt, model=model, max_tokens=max_tok)
        except Exception as e:
            logger.warning("[Agency LLM] OllamaCloud failed: %s — trying Venice", e)

        try:
            import httpx
            venice_key = inference_cfg.get("venice_api_key", "")
            if venice_key:
                async with httpx.AsyncClient(timeout=15.0) as http:
                    resp = await http.post(
                        "https://api.venice.ai/api/v1/chat/completions",
                        json={
                            "model": "llama-3.3-70b",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 200,
                        },
                        headers={"Authorization": f"Bearer {venice_key}"},
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("[Agency LLM] Venice failed: %s", e)

        raise RuntimeError("No LLM available for agency")

    return agency_llm


def _register_helpers(registry, full_config: dict) -> int:
    """Register the same 8 helpers parent/legacy registers in
    _register_helpers. Returns the count of successfully registered helpers.

    Mirrors core/plugin.py:1230-1344 — kept in sync deliberately
    (mismatch would be visible on the manifest comparison in §A.8.6
    OBS gate).
    """
    n = 0
    try:
        from titan_plugin.logic.agency.helpers.infra_inspect import InfraInspectHelper
        registry.register(InfraInspectHelper(log_path="/tmp/titan_v3.log"))
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] InfraInspect helper failed: %s", e)

    try:
        from titan_plugin.logic.agency.helpers.web_search import WebSearchHelper
        sage_cfg = full_config.get("stealth_sage", {}) or {}
        searxng_host = sage_cfg.get("searxng_host", "http://localhost:8080")
        firecrawl_key = sage_cfg.get("firecrawl_api_key", "")
        kp_cfg = full_config.get("knowledge_pipeline", {}) or {}
        kp_budgets_mb = kp_cfg.get("budgets", {}) or {}
        kp_budgets_bytes = {
            k: int(v) * 1024 * 1024
            for k, v in kp_budgets_mb.items()
            if isinstance(v, (int, float))
        }
        registry.register(WebSearchHelper(
            searxng_url=searxng_host,
            firecrawl_api_key=firecrawl_key,
            budgets=kp_budgets_bytes,
        ))
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] WebSearch helper failed: %s", e)

    # SocialPostHelper intentionally NOT registered (parent's
    # _register_helpers has the same exclusion — all posting routes
    # through SocialPressureMeter narrator instead).

    try:
        from titan_plugin.logic.agency.helpers.art_generate import ArtGenerateHelper
        exp_cfg = full_config.get("expressive", {}) or {}
        output_dir = exp_cfg.get("output_path", "./data/studio_exports")
        registry.register(ArtGenerateHelper(output_dir=output_dir))
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] ArtGenerate helper failed: %s", e)

    try:
        from titan_plugin.logic.agency.helpers.audio_generate import AudioGenerateHelper
        exp_cfg = full_config.get("expressive", {}) or {}
        audio_cfg = full_config.get("audio", {}) or {}
        output_dir = exp_cfg.get("output_path", "./data/studio_exports")
        max_duration = int(audio_cfg.get("max_duration_seconds", 30))
        sample_rate = int(audio_cfg.get("sample_rate", 44100))
        registry.register(AudioGenerateHelper(
            output_dir=output_dir,
            max_duration=max_duration,
            sample_rate=sample_rate,
        ))
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] AudioGenerate helper failed: %s", e)

    try:
        from titan_plugin.logic.agency.helpers.coding_sandbox import CodingSandboxHelper
        registry.register(CodingSandboxHelper())
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] CodingSandbox helper failed: %s", e)

    try:
        from titan_plugin.logic.agency.helpers.code_knowledge import CodeKnowledgeHelper
        registry.register(CodeKnowledgeHelper())
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] CodeKnowledge helper failed: %s", e)

    try:
        from titan_plugin.logic.agency.helpers.memo_inscribe import MemoInscribeHelper
        # memo_inscribe reads config.toml directly for RPC + keypair.
        # In subprocess we have NO access to parent's metabolism proxy,
        # so the gate is None — the helper degrades to "no metabolism gate"
        # mode (still functional, just no governance reserve guard).
        # Acceptable for A.8.6 ship; full metabolism injection deferred to
        # A.8.7 cleanup or a follow-up bus-routed metabolism query.
        registry.register(MemoInscribeHelper(metabolism=None))
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] MemoInscribe helper failed: %s", e)

    try:
        from titan_plugin.logic.agency.helpers.kin_sense import KinSenseHelper
        try:
            import tomllib as _tomllib_kin
        except ImportError:
            import tomli as _tomllib_kin
        kin_params: dict = {}
        kin_params_path = os.path.join(
            os.path.dirname(__file__), "..", "titan_params.toml")
        if os.path.exists(kin_params_path):
            with open(kin_params_path, "rb") as kf:
                kin_params = _tomllib_kin.load(kf)
        kin_cfg = kin_params.get("kin", {}) or {}
        if kin_cfg.get("enabled", False):
            kin_addrs = kin_cfg.get("addresses", []) or []
            env_addrs = os.environ.get("TITAN_KIN_ADDRESSES", "")
            if env_addrs:
                kin_addrs = [a.strip() for a in env_addrs.split(",") if a.strip()]
            registry.register(KinSenseHelper(
                kin_addresses=kin_addrs,
                exchange_strength=float(kin_cfg.get("exchange_strength", 0.03)),
            ))
            n += 1
            logger.info("[AgencyWorker] KinSense registered: addresses=%s", kin_addrs)
    except Exception as e:
        logger.warning("[AgencyWorker] KinSense helper failed: %s", e)

    return n


def _run_async(coro):
    """Run an async coroutine to completion in the worker's main thread.

    The worker process is not itself async — it loops on a sync queue.
    For each handle_intent / assess / dispatch query we spin up a small
    event loop, run the coroutine, and return the result. This isolates
    LLM async calls + helper.execute() awaits without leaking any state.

    Reusing a single loop with run_until_complete is intentional: avoids
    asyncio.run() teardown cost (~5ms each call) on the hot path. The
    loop is created lazily on first call + reused for the worker's
    lifetime.
    """
    global _LOOP
    try:
        loop = _LOOP  # type: ignore[name-defined]
    except NameError:
        loop = None
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        globals()["_LOOP"] = loop
    return loop.run_until_complete(coro)


def _neutral_assessment(action_result: dict, error: str) -> dict:
    """Used when assess() raises — returns a benign neutral assessment so
    the parent's _handle_impulse can still publish ACTION_RESULT without
    stalling on an exception. score=0.5 = "hold" threshold direction."""
    return {
        "action_id": int(action_result.get("action_id", 0) or 0),
        "impulse_id": int(action_result.get("impulse_id", 0) or 0),
        "score": 0.5,
        "reflection": f"assessment_error: {error}",
        "enrichment": {},
        "mood_delta": 0.0,
        "threshold_direction": "hold",
        "ts": time.time(),
    }


def _action_result_dict(result: Any) -> Optional[dict]:
    """Coerce handle_intent return to a JSON-safe dict (or None when
    Agency skipped). action_result is already a flat dict per
    AgencyModule._build_result, but we belt-and-suspenders via dict()."""
    if result is None:
        return None
    try:
        return dict(result)
    except Exception:
        return None


def agency_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the Agency worker subprocess.

    Args:
        recv_queue: bus → worker
        send_queue: worker → bus
        name: Guardian module name (must equal "agency_worker")
        config: full config dict — same shape as kernel.config; passes
                inference + stealth_sage + expressive + audio + agency
                + knowledge_pipeline + info_banner + memory_and_storage
                sections through to the helpers + LLM fn.
    """
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    full_config = config or {}
    titan_id = (full_config.get("info_banner", {}) or {}).get("titan_id") or "T1"
    inference_cfg = full_config.get("inference", {}) or {}
    agency_cfg = full_config.get("agency", {}) or {}
    budget_per_hour = int(agency_cfg.get("llm_budget_per_hour", 10))

    logger.info("[AgencyWorker] Booting — titan_id=%s, budget_per_hour=%d",
                titan_id, budget_per_hour)

    try:
        from titan_plugin.logic.agency.registry import HelperRegistry
        from titan_plugin.logic.agency.module import AgencyModule
        from titan_plugin.logic.agency.assessment import SelfAssessment
    except Exception as e:
        logger.error("[AgencyWorker] Core module import failed: %s — exiting", e)
        return

    registry = HelperRegistry()
    helper_count = _register_helpers(registry, full_config)
    if helper_count == 0:
        logger.warning("[AgencyWorker] Zero helpers registered — proxy will return "
                       "no_suitable_helper for every intent. Boot continues.")

    llm_fn = _build_llm_fn(inference_cfg)
    agency = AgencyModule(registry=registry, llm_fn=llm_fn,
                          budget_per_hour=budget_per_hour)
    assessment = SelfAssessment(llm_fn=llm_fn)

    logger.info("[AgencyWorker] Booted: %d helpers (%s)",
                helper_count, registry.list_all_names())

    # Boot signals: MODULE_READY → Guardian flips state to RUNNING
    # (without this, /health shows the worker DEGRADED forever — same
    # bug A.8.3 hot-fixed in commit 9406f13f). AGENCY_READY broadcast
    # marks proxy bus-routed for any consumer that wants to confirm.
    try:
        send_queue.put({
            "type": bus.MODULE_READY, "src": name, "dst": "guardian",
            "payload": {"titan_id": titan_id, "helper_count": helper_count,
                        "ts": time.time()},
            "ts": time.time(),
        })
        send_queue.put({
            "type": bus.AGENCY_READY, "src": name, "dst": "all",
            "payload": {"titan_id": titan_id, "helper_count": helper_count,
                        "helpers": registry.list_all_names(),
                        "ts": time.time()},
            "ts": time.time(),
        })
    except Exception:
        pass

    last_heartbeat = 0.0
    last_stats_publish = 0.0
    poll_interval_s = 0.2

    while True:
        now = time.time()

        # Periodic heartbeat (Guardian liveness)
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            try:
                send_queue.put({
                    "type": bus.MODULE_HEARTBEAT, "src": name, "dst": "guardian",
                    "payload": {"alive": True, "ts": now},
                    "ts": now,
                })
            except Exception:
                pass
            last_heartbeat = now

        # Periodic stats publish — refreshes proxy cached attrs in parent
        if now - last_stats_publish >= _STATS_PUBLISH_INTERVAL_S:
            try:
                send_queue.put({
                    "type": bus.AGENCY_STATS, "src": name, "dst": "all",
                    "payload": agency.get_stats(),
                    "ts": now,
                })
                send_queue.put({
                    "type": bus.ASSESSMENT_STATS, "src": name, "dst": "all",
                    "payload": assessment.get_stats(),
                    "ts": now,
                })
            except Exception:
                pass
            last_stats_publish = now

        # Drain bus messages
        try:
            msg = recv_queue.get(timeout=poll_interval_s)
        except Empty:
            continue
        except Exception:
            continue

        msg_type = msg.get("type")

        # ── Microkernel v2 Phase B.2.1 — supervision-transfer dispatch ──
        # Spawn-mode A.8.X workers must outlive kernel swaps via BUS_HANDOFF
        # adoption protocol. Without this dispatch, shadow_swap's
        # _phase_hibernate logs `spawn_handoff_ack_missing` for this worker.
        # Added 2026-04-28 PM during shadow swap E2E test (was missing from
        # this newer worker — original 15-worker wiring predates A.8.6).
        from titan_plugin.core import worker_swap_handler as _swap
        if _swap.maybe_dispatch_swap_msg(msg):
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[AgencyWorker] Shutdown received — exiting")
            return
        if msg_type != bus.QUERY:
            continue

        rid = msg.get("rid")
        src = msg.get("src", "unknown")
        payload = msg.get("payload") or {}
        action = payload.get("action")

        try:
            if action == "handle_intent":
                intent = payload.get("intent") or {}
                result = _run_async(agency.handle_intent(intent))
                response_payload = {"action_result": _action_result_dict(result)}

            elif action == "dispatch_from_nervous_signals":
                outer_signals = payload.get("outer_signals") or []
                trinity_snapshot = payload.get("trinity_snapshot") or {}
                results = _run_async(
                    agency.dispatch_from_nervous_signals(
                        outer_signals=outer_signals,
                        trinity_snapshot=trinity_snapshot,
                    ))
                # results is list[dict] from AgencyModule._build_result
                response_payload = {"action_results": [
                    _action_result_dict(r) for r in (results or [])
                    if r is not None
                ]}

            elif action == "assess":
                action_result = payload.get("action_result") or {}
                try:
                    assessment_dict = _run_async(assessment.assess(action_result))
                except Exception as ae:
                    logger.warning("[AgencyWorker] assess failed: %s — neutral", ae)
                    assessment_dict = _neutral_assessment(action_result, str(ae))
                response_payload = {"assessment": assessment_dict}

            elif action == "agency_stats":
                response_payload = {"stats": agency.get_stats()}

            elif action == "assessment_stats":
                response_payload = {"stats": assessment.get_stats()}

            else:
                response_payload = {"error": f"unknown action: {action}"}

        except Exception as e:
            logger.warning("[AgencyWorker] action=%s failed: %s", action, e)
            response_payload = {"error": str(e)}

        try:
            send_queue.put({
                "type": bus.RESPONSE, "src": name, "dst": src, "rid": rid,
                "payload": response_payload, "ts": time.time(),
            })
        except Exception as e:
            logger.warning("[AgencyWorker] response send failed: %s", e)
