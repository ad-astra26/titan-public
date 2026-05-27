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
from titan_hcl import bus
from titan_hcl.core.module_error_handler import with_error_envelope
from titan_hcl.errors import Severity as _phase11_sev

logger = logging.getLogger("agency_worker")

_HEARTBEAT_INTERVAL_S = 30.0
_STATS_PUBLISH_INTERVAL_S = 60.0


def _build_llm_fn(inference_cfg: dict, *, api_base: str = "http://127.0.0.1:7777",
                  internal_key: str = ""):
    """Build async agency LLM function routing through /v4/llm-distill.

    Phase 3 Chunk χ-bis (D-SPEC-88, 2026-05-18) — replaces the direct
    OllamaCloudClient + Venice fallback path. All LLM traffic now goes
    through the centralized HTTP proxy → LLM_DISTILL_REQUEST → llm_worker
    → provider abstraction (which handles Ollama/Venice/OpenRouter
    failover internally). Observable in llm_state.bin.

    Model selection rules preserved: `agency_code_gen` task gets 800
    max_tokens, others get 200. `get_model_for_task` still resolves
    the appropriate model per task.
    """
    async def agency_llm(prompt: str, task: str = "agency_select") -> str:
        try:
            from titan_hcl.inference import get_model_for_task
            from titan_hcl.logic.llm_distill_client import (
                distill_via_http_async)
            model = get_model_for_task(task)
            max_tok = 800 if task == "agency_code_gen" else 200
            result = await distill_via_http_async(
                text=prompt,
                # Agency calls are zero-instruction by convention — the
                # task-routing model is expected to return the right shape
                # without explicit framing. Empty instruction → llm_worker
                # builds just the prompt as the user message.
                instruction="",
                api_base=api_base,
                internal_key=internal_key,
                model=model,
                max_tokens=max_tok,
                consumer=f"agency.{task}",
                timeout_s=30.0,
            )
            if result:
                return result
        except Exception as e:
            logger.warning("[Agency LLM] /v4/llm-distill failed: %s", e)

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
        from titan_hcl.logic.agency.helpers.infra_inspect import InfraInspectHelper
        registry.register(InfraInspectHelper(log_path="/tmp/titan_v3.log"))
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] InfraInspect helper failed: %s", e)

    try:
        from titan_hcl.logic.agency.helpers.web_search import WebSearchHelper
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
        from titan_hcl.logic.agency.helpers.art_generate import ArtGenerateHelper
        exp_cfg = full_config.get("expressive", {}) or {}
        output_dir = exp_cfg.get("output_path", "./data/studio_exports")
        registry.register(ArtGenerateHelper(output_dir=output_dir))
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] ArtGenerate helper failed: %s", e)

    try:
        from titan_hcl.logic.agency.helpers.audio_generate import AudioGenerateHelper
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
        from titan_hcl.logic.agency.helpers.coding_sandbox import CodingSandboxHelper
        registry.register(CodingSandboxHelper())
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] CodingSandbox helper failed: %s", e)

    try:
        from titan_hcl.logic.agency.helpers.code_knowledge import CodeKnowledgeHelper
        registry.register(CodeKnowledgeHelper())
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] CodeKnowledge helper failed: %s", e)

    try:
        from titan_hcl.logic.agency.helpers.memo_inscribe import MemoInscribeHelper
        # memo_inscribe reads config.toml directly for RPC + keypair.
        # Per rFP_titan_hcl_l2_separation_strategy §4.J + D-SPEC-51 (SPEC
        # v1.7.2, 2026-05-14): in subprocess we can't host MetabolismProxy
        # (needs DivineBus + Guardian — kernel-level instances not
        # accessible here). Instead pass a `MetabolismShmReader` — same
        # duck-typed surface (evaluate_gate sync + can_afford async) but
        # reads tier from metabolism_state.bin SHM (sub-ms, G18). Gate
        # decision is LOCAL (no ring-buffer write, no bus broadcast) —
        # acceptable for agency_worker's autonomous memo path. The
        # authoritative ring-buffer path remains available in the main
        # process via plugin._proxies["metabolism"] = MetabolismProxy
        # for dashboard /v4/metabolism/evaluate-gate.
        try:
            from titan_hcl.proxies.metabolism_proxy import MetabolismShmReader
            metabolism_reader = MetabolismShmReader()
        except Exception as _mb_err:
            logger.warning(
                "[AgencyWorker] MetabolismShmReader init failed (memo "
                "gate degrades to fail-open per legacy behavior): %s",
                _mb_err)
            metabolism_reader = None
        registry.register(MemoInscribeHelper(metabolism=metabolism_reader))
        n += 1
    except Exception as e:
        logger.warning("[AgencyWorker] MemoInscribe helper failed: %s", e)

    try:
        from titan_hcl.logic.agency.helpers.kin_sense import KinSenseHelper
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


def _maybe_emit_onchain_anchor_catalyst(
        send_queue, name: str, action_result: Optional[dict]) -> None:
    """D-SPEC-66 v1.11.0 PLAN §1.1 — D8-3 catalyst-producer site #1
    closure (onchain_anchor).

    Emits SOCIAL_CATALYST(type=onchain_anchor) when a memo_inscribe
    helper invocation reports `success=True`. Replaces the dead spirit
    _worker.py:5320 file-polling loop (heartbeat-stub since fleet-wide
    Phase C 2026-05-14). action_result["enrichment_data"] carries the
    helper's `balance` + `result` plus any post-success fields.
    """
    if not isinstance(action_result, dict):
        return
    if action_result.get("helper") != "memo_inscribe":
        return
    if not action_result.get("success"):
        return
    try:
        _enr = action_result.get("enrichment_data") or {}
        _balance = float(_enr.get("balance", 0.0) or 0.0)
        _result_text = str(action_result.get("result", ""))[:200]
        send_queue.put({
            "type": bus.SOCIAL_CATALYST,
            "src": name,
            "dst": "social",
            "payload": {
                "type": "onchain_anchor",
                "significance": 0.4,
                "content": (
                    f"State anchored: {_result_text}"
                    if _result_text else "State anchored on Solana"),
                "data": {
                    "balance": _balance,
                    "action_id": action_result.get("action_id"),
                    "impulse_id": action_result.get("impulse_id"),
                },
            },
        })
    except Exception as _err:
        logger.warning(
            "[AgencyWorker] onchain_anchor catalyst emit failed: %s",
            _err)


@with_error_envelope(module_name="agency_worker", subsystem="entry", severity=_phase11_sev.FATAL)
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
    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = (
        (full_config.get("info_banner", {}) or {}).get("titan_id")
        or resolve_titan_id()
    )
    inference_cfg = full_config.get("inference", {}) or {}
    agency_cfg = full_config.get("agency", {}) or {}
    budget_per_hour = int(agency_cfg.get("llm_budget_per_hour", 10))
    # Phase 3 Chunk χ-bis (D-SPEC-88, 2026-05-18) — api_base + internal_key
    # for /v4/llm-distill round-trip (replaces direct OllamaCloudClient).
    _api_port = int((full_config.get("api", {}) or {}).get("port", 7777))
    _api_base = f"http://127.0.0.1:{_api_port}"
    _internal_key = (full_config.get("api", {}) or {}).get("internal_key", "") or ""

    logger.info("[AgencyWorker] Booting — titan_id=%s, budget_per_hour=%d",
                titan_id, budget_per_hour)

    try:
        from titan_hcl.logic.agency.registry import HelperRegistry
        from titan_hcl.logic.agency.module import AgencyModule
        from titan_hcl.logic.agency.assessment import SelfAssessment
    except Exception as e:
        logger.error("[AgencyWorker] Core module import failed: %s — exiting", e)
        return

    registry = HelperRegistry()
    helper_count = _register_helpers(registry, full_config)
    if helper_count == 0:
        logger.warning("[AgencyWorker] Zero helpers registered — proxy will return "
                       "no_suitable_helper for every intent. Boot continues.")

    llm_fn = _build_llm_fn(inference_cfg, api_base=_api_base,
                           internal_key=_internal_key)
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

    # Phase C Session 3 (rFP §4.B.2 + §4.B.3) — SHM-direct state publishers
    # for agency_state.bin + assessment_state.bin. Replaces the deadlock-
    # prone sync bus.request(action="get_agency_stats" / "get_assessment_stats")
    # path (Session 4 proxy migration §4.C.5 + §4.C.12 will read these slots).
    # Cadence: 1 Hz. Single thread fans out to both publishers via
    # MultiSlotStatePublisher (G21 single-writer per slot — agency_state
    # written ONLY here, assessment_state written ONLY here).
    try:
        from titan_hcl.core.state_registry import resolve_titan_id
        from titan_hcl.logic.agency_state_publisher import (
            AgencyStatePublisher)
        from titan_hcl.logic.assessment_state_publisher import (
            AssessmentStatePublisher)
        from titan_hcl.logic.worker_publisher_runner import (
            run_multi_slot_worker_publisher)
        _titan_id_resolved = resolve_titan_id()
        _agency_state_publisher = AgencyStatePublisher(
            titan_id=_titan_id_resolved)
        _assessment_state_publisher = AssessmentStatePublisher(
            titan_id=_titan_id_resolved)
        # Each publisher takes a different positional arg — agency takes
        # the AgencyModule, assessment takes the SelfAssessment. Solution:
        # publish_args returns (state,) where state is a 2-tuple, and
        # each publisher's _compute_payload extracts what it needs. But
        # simpler: 2 separate publisher threads, one per publisher.
        from titan_hcl.logic.worker_publisher_runner import (
            run_worker_publisher)
        run_worker_publisher(
            publisher=_agency_state_publisher,
            state_fetcher=lambda: agency,
            worker_name="agency_worker",
            cadence_s=1.0,
        )
        run_worker_publisher(
            publisher=_assessment_state_publisher,
            state_fetcher=lambda: assessment,
            worker_name="agency_worker",
            cadence_s=1.0,
        )
    except Exception as _pub_init_err:
        logger.error(
            "[AgencyWorker] SHM publisher BOOT FAILED — "
            "consumers fall back to sync bus.request path (deadlock "
            "surface remains until publisher recovers): %s",
            _pub_init_err, exc_info=True)

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
        from titan_hcl.core import worker_swap_handler as _swap
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
                _ar_single = _action_result_dict(result)
                response_payload = {"action_result": _ar_single}
                # D-SPEC-66 v1.11.0 PLAN §1.1 site #1 closure.
                _maybe_emit_onchain_anchor_catalyst(
                    send_queue, name, _ar_single)

            elif action == "dispatch_from_nervous_signals":
                outer_signals = payload.get("outer_signals") or []
                trinity_snapshot = payload.get("trinity_snapshot") or {}
                results = _run_async(
                    agency.dispatch_from_nervous_signals(
                        outer_signals=outer_signals,
                        trinity_snapshot=trinity_snapshot,
                    ))
                # results is list[dict] from AgencyModule._build_result
                _ar_list = [
                    _action_result_dict(r) for r in (results or [])
                    if r is not None
                ]
                response_payload = {"action_results": _ar_list}
                # D-SPEC-66 v1.11.0 PLAN §1.1 site #1 closure (one
                # catalyst per memo_inscribe success in batch).
                for _ar in _ar_list:
                    _maybe_emit_onchain_anchor_catalyst(
                        send_queue, name, _ar)

            elif action == "assess":
                action_result = payload.get("action_result") or {}
                try:
                    assessment_dict = _run_async(assessment.assess(action_result))
                except Exception as ae:
                    logger.warning("[AgencyWorker] assess failed: %s — neutral", ae)
                    assessment_dict = _neutral_assessment(action_result, str(ae))
                response_payload = {"assessment": assessment_dict}
                # Record stage of the ExperienceOrchestrator distillation loop
                # (rFP_experience_distillation_phase_c). Emit EXPERIENCE_RECORD
                # with the assessed agency outcome — domain inferred from the
                # helper; cognitive_worker enriches (inner-state + hormones +
                # perception key) and persists. Targeted/non-blocking/event-gated
                # per bus-hygiene §3.1.
                try:
                    from titan_hcl.logic.experience_orchestrator import (
                        infer_domain)
                    _exp_helper = str(
                        action_result.get("helper")
                        or action_result.get("action") or "agency_action")
                    bus.emit_experience_record(
                        send_queue, name,
                        domain=infer_domain(_exp_helper),
                        action_taken=_exp_helper,
                        outcome_score=float(assessment_dict.get("score", 0.5)),
                        context={
                            "helper": _exp_helper,
                            "success": bool(action_result.get("success")),
                            "reflection": str(
                                assessment_dict.get("reflection", ""))[:160],
                        },
                        epoch_id=int(action_result.get("epoch_id", 0) or 0),
                    )
                except Exception as _exp_err:
                    logger.debug(
                        "[AgencyWorker] EXPERIENCE_RECORD emit raised: %s",
                        _exp_err)

            # Phase C Session 5 (rFP §4.D.4): agency_stats + assessment_stats
            # handlers RETIRED — agency_proxy.refresh_stats +
            # assessment_proxy.refresh_stats now SHM-direct via
            # agency_state.bin + assessment_state.bin (Session 3
            # §4.B.2/§4.B.3 publishers).

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
