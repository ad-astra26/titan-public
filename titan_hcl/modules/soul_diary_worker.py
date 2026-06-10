"""titan_hcl/modules/soul_diary_worker.py — the Soul-Diary L2 worker.

`RFP_titan_authored_soul_diary` §1.0 (P1). A lean, self-contained L2 worker that
owns the daily soul-diary mechanic. On the day's last meditation it authors a
grounded, OVG-verified first-person reflection in Titan's own voice and persists
it (the narrative SELF). Pipeline (§1.0 ①②④⑤):

    MEDITATION_COMPLETE → ① latch (once/UTC-day) → ② gather grounded bundle (G18)
    → narrate (OWN LLM call: provider.complete + OutputVerifier.verify_safety —
    grounded prompt + OVG, NOT a bare distill; INV-SD-14) → ④ persist
    (core/soul.append_chronicle_entry, single-writer INV-SD-6) → ⑤ hash-chain
    ledger (core/soul_diary_chain).

Self-contained by design (Maker 2026-06-09 — simpler + debuggable than a
cross-worker compose round-trip): the worker makes its own LLM call rather than
delegating to social_worker's gateway.

P2 (§1.0 ⑥⑦) extends the pipeline past persist+hash:
    ⑥ ENRICH  → MEMORY_MEMPOOL_ADD (dst=memory, one-way) → promoted at the
                dream boundary into DuckDB+FAISS+Kuzu, classified `domain="self"`
                → Titan remembers + recalls his narrative path (INV-SD-15).
    ⑦ ANCHOR  → OuterMemoryWriter.emit(fork="main", cumulative_hash) →
                TIMECHAIN_COMMIT → main/genesis chain (FORK_MAIN=0, the SELF
                journey, NOT ACT-R forks) — a hash pointer (INV-SD-17).
The SELF node (P3) and the public expression pillar (P6-P10) are downstream
phases this same worker will fire. Every step soft-fails independently to a
minimal grounded entry / skipped enrich-anchor — never blocks the meditation
cascade (INV-SD-13).
"""
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from queue import Empty

import titan_hcl.bus as bus
from titan_hcl.modules._heartbeat_grace import (
    boot_deadline_from_now, shm_heartbeat_allowed,
)

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_S = 30.0
_COMPOSE_TIMEOUT_S = 90.0
_COMPOSE_MAX_TOKENS = 600
_COMPOSE_TEMPERATURE = 0.7

_WORKER_READY = False
_BOOT_DEADLINE = 0.0


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _resolve_provider_name(inference_cfg: dict) -> str:
    """The configured inference provider for the worker's own LLM call.

    Canonical config key = ``inference_provider`` (``config.toml [inference]``);
    ``provider`` accepted as a legacy alias; ``"venice"`` last-resort.
    BUGFIX 2026-06-10: previously read only ``provider`` (the WRONG key), so it
    always fell back to ``"venice"`` — which is 402-dead on the fleet — and every
    diary entry soft-fell to the minimal grounded template instead of an authored
    reflection. Reading the real key lets the diary use the live provider
    (e.g. ``ollama_cloud``)."""
    return (inference_cfg.get("inference_provider")
            or inference_cfg.get("provider") or "venice")


async def _compose_diary(provider, verifier, prompts: dict) -> tuple[str, bool]:
    """The worker's OWN grounded-compose + OVG (§1.0 ③, self-contained).

    Grounded: the prompt already carries the real day's facts. OVG: the
    deterministic ``verify_safety`` truth-gate. Returns ``(text, ovg_ok)``;
    an empty/failed compose or a blocked/erroring OVG → ``ovg_ok=False`` so the
    caller soft-fails to the minimal grounded entry (INV-SD-13).
    """
    if provider is None:
        return "", False
    try:
        text = await provider.complete(
            prompt=prompts["user_prompt"], system=prompts["system_prompt"],
            temperature=_COMPOSE_TEMPERATURE, max_tokens=_COMPOSE_MAX_TOKENS,
            timeout=_COMPOSE_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] compose failed: %s", e)
        return "", False
    text = (text or "").strip()
    if not text:
        return "", False
    if verifier is None:
        return text, False  # no OVG available → soft-fail (fail-closed, INV-SD-2)
    try:
        result = verifier.verify_safety(
            text, channel="agent", injected_context=prompts["user_prompt"])
        return text, bool(getattr(result, "passed", False))
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] OVG verify_safety failed: %s", e)
        return text, False


def _gather_bundle(payload: dict, shm_reader, orchestrator, *,
                   titan_id: str = "", repo_root: str = "") -> dict:
    """② GATHER — assemble the grounded bundle from G18 reads (best-effort).

    Each source is independently guarded: a missing source degrades the entry,
    never crashes the cascade. Engram day-window / richer memory-social-onchain
    sources land with the P2 synthesis enrichment. The §7.P5 `infra` slot carries
    read-only self-inspection (journal errors + error→code correlation), so the
    entry is grounded in his real substrate (INV-SD-9).
    """
    sovereignty: dict = {}
    try:
        from titan_hcl.synthesis.sovereignty_readout import read_rolling_sovereignty
        sovereignty = read_rolling_sovereignty() or {}
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] sovereignty read failed: %s", e)

    outcome = {
        "promoted": int(payload.get("promoted", 0) or 0),
        "pruned": int(payload.get("pruned", 0) or 0),
        "epoch": payload.get("epoch"),
    }

    felt: dict = {}
    try:
        nm = shm_reader.read_neuromod() if shm_reader is not None else None
        if isinstance(nm, dict) and nm:
            levels = {k: v for k, v in nm.items() if isinstance(v, (int, float))}
            dominant = max(levels, key=levels.get) if levels else None
            felt = {"dominant": dominant, "valence": nm.get("valence"),
                    "arousal": nm.get("arousal")}
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] neuromod read failed: %s", e)

    # P5 — scaffolding self-inspection (read-only, bounded, soft-fail; INV-SD-9).
    infra: dict = {}
    if titan_id and repo_root:
        try:
            from titan_hcl.core import self_inspect
            obs = self_inspect.gather_self_observations(titan_id, repo_root=repo_root)
            infra = {
                "summary": self_inspect.summarize_observations(obs),
                "structure": obs.get("structure") or {},
                "correlations": obs.get("correlations") or [],
            }
        except Exception as e:  # noqa: BLE001
            logger.info("[soul_diary] self-inspection failed: %s", e)

    return orchestrator.build_bundle(
        sovereignty=sovereignty, outcome=outcome, felt=felt,
        engrams_today=[], memory={}, social={}, onchain={}, infra=infra)


def _enrich_synthesis(send_queue, src: str, today: str, entry: str) -> None:
    """⑥ ENRICH (INV-SD-15) — publish the diary as a self-domain thought to the
    mempool (one-way, no-RPC; G19). ``memory_worker`` promotes it at the next
    meditation/dream boundary into DuckDB+FAISS+Kuzu so Titan *recalls* his
    narrative path; consolidation classifies it into the (now clean) ``"self"``
    domain (consolidation_defaults split, P2). Soft-fail — never blocks the
    cascade (INV-SD-13); the private floor (persist+hash) is already committed."""
    try:
        send_queue.put({
            "type": bus.MEMORY_MEMPOOL_ADD,
            "src": src, "dst": "memory",
            "ts": time.time(),
            "payload": {
                "user_prompt": f"Soul-diary reflection — {today}",
                "agent_response": entry,
                "user_identifier": "Titan",
                "source": "soul_diary",
                "tags": ["soul_diary", "domain:self"],
            },
        })
        logger.info("[soul_diary] ⑥ enriched synthesis (mempool add → self domain) "
                    "for %s", today)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] enrich (mempool add) failed: %s", e)


def _enrich_self_inspection(send_queue, src: str, today: str, infra: dict) -> None:
    """P5 ENRICH (INV-SD-9) — publish the self-inspection observation as its OWN
    `domain="self"` thought (`source="self_inspect"`), so his code/error/substrate
    observations become recallable self-Engrams (auto-linked to the `Self` hub via
    the P3a `SELF_HAS_ENGRAM` hook), distinct from the narrative diary entry. This
    is the BRAIN_DOMAIN_SELF seed. Soft-fail — never blocks the cascade."""
    summary = ((infra or {}).get("summary") or "").strip()
    if not summary:
        return
    try:
        send_queue.put({
            "type": bus.MEMORY_MEMPOOL_ADD,
            "src": src, "dst": "memory",
            "ts": time.time(),
            "payload": {
                "user_prompt": f"What I observed about my own substrate — {today}",
                "agent_response": summary,
                "user_identifier": "Titan",
                "source": "self_inspect",
                "tags": ["soul_diary", "self_inspect", "domain:self"],
            },
        })
        logger.info("[soul_diary] P5 self-inspection enriched "
                    "(mempool add, source=self_inspect) for %s", today)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] self-inspection enrich failed: %s", e)


def _anchor_main_chain(send_queue, src: str, today: str, row: dict) -> None:
    """⑦ ANCHOR (INV-SD-17) — seal the diary's cumulative-hash head on the
    main/genesis chain (``fork="main"`` → ``FORK_MAIN=0``, the SELF journey — NOT
    the ACT-R declarative/procedural/episodic forks). A hash POINTER; the entry
    text stays in outer memory. ``significance=0.85`` (a once-daily SELF-journey
    milestone) clears the main-fork PoT threshold (0.20); empty ``neuromods``
    take the tonic 0.5 baseline (create_pot). Soft-fail (INV-SD-13)."""
    cumulative_hash = (row or {}).get("cumulative_hash", "")
    entry_hash = (row or {}).get("entry_hash", "")
    if not cumulative_hash:
        logger.warning("[soul_diary] anchor skipped — no cumulative_hash in ledger row")
        return
    try:
        from titan_hcl.synthesis.outer_memory_writer import (
            OuterMemoryEvent, OuterMemoryWriter,
        )
        writer = OuterMemoryWriter(send_queue, src=src)
        writer.emit(OuterMemoryEvent(
            fork="main",
            thought_type="dailyDiary",
            source="soul_diary",
            content={"date": today, "entry_hash": entry_hash,
                     "cumulative_hash": cumulative_hash},
            tags=["soul_diary", "dailyDiary", f"date:{today}"],
            significance=0.85,   # high — clears main PoT (0.20); SELF-journey milestone
            novelty=0.5,
            coherence=0.8,
        ))
        logger.info("[soul_diary] ⑦ anchored cumulative_hash=%s on main chain "
                    "(fork=main) for %s", cumulative_hash[:12], today)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] anchor (main-chain emit) failed: %s", e)


def _author_daily_entry(payload: dict, *, orchestrator, provider, verifier,
                        shm_reader, send_queue, src: str,
                        titan_id: str = "", repo_root: str = "") -> bool:
    """Run the full pipeline for one MEDITATION_COMPLETE. Returns True if a
    diary entry was authored (or correctly skipped), False on hard error."""
    today = _utc_today()
    if not orchestrator.should_author(today):
        return True  # already wrote today — latch closed (INV-SD-5)

    bundle = _gather_bundle(payload, shm_reader, orchestrator,
                            titan_id=titan_id, repo_root=repo_root)
    if not orchestrator.has_activity(bundle):
        logger.info("[soul_diary] no-op day (no activity) — latching without entry")
        orchestrator.mark_authored(today)
        return True

    prompts = orchestrator.build_compose_prompts(bundle)
    text, ovg_ok = asyncio.run(_compose_diary(provider, verifier, prompts))
    entry = text if (ovg_ok and text) else orchestrator.minimal_entry(bundle)
    if not (ovg_ok and text):
        logger.warning("[soul_diary] authoring soft-fell to minimal grounded entry "
                       "(ovg_ok=%s, text=%d chars)", ovg_ok, len(text))

    try:
        orchestrator.persist(entry)                 # ④ titan_chronicles.md → titan.md
        row = orchestrator.record_hash(today, entry)  # ⑤ hash-chain ledger (row = the hashes)
        orchestrator.mark_authored(today)           # ① latch
    except Exception as e:  # noqa: BLE001
        logger.error("[soul_diary] persist/hash failed: %s", e, exc_info=True)
        return False

    # P2 — the committed entry now ENRICHES synthesis (he remembers + recalls his
    # narrative path) and ANCHORS the SELF journey on the main chain. Each
    # soft-fails independently (INV-SD-13); the private floor above is durable.
    _enrich_synthesis(send_queue, src, today, entry)              # ⑥ INV-SD-15
    _enrich_self_inspection(send_queue, src, today,
                            bundle.get("infra") or {})            # P5 INV-SD-9
    _anchor_main_chain(send_queue, src, today, row)              # ⑦ INV-SD-17

    logger.info("[soul_diary] authored daily entry for %s (%d chars, authored=%s)",
                today, len(entry), ovg_ok and bool(text))
    return True


def soul_diary_worker_main(recv_queue, send_queue, name: str,
                           config: dict) -> None:
    """Main loop for the soul-diary worker subprocess (§1.0 P1)."""
    global _WORKER_READY, _BOOT_DEADLINE
    _WORKER_READY = False
    _BOOT_DEADLINE = boot_deadline_from_now()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    full_config = config or {}

    _state_writer = None
    try:
        from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
        _state_writer = ModuleStateWriter(
            module_name=name, layer="L2",
            boot_priority=BootPriority.OPTIONAL_POST_BOOT)
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        logger.warning("[soul_diary] ModuleStateWriter init failed: %s", _sw_err)

    from titan_hcl.core.state_registry import resolve_titan_id
    titan_id = ((full_config.get("info_banner", {}) or {}).get("titan_id")
                or resolve_titan_id())
    logger.info("[soul_diary] Booting — titan_id=%s", titan_id)

    from titan_hcl.core.soul_diary import SoulDiaryOrchestrator
    orchestrator = SoulDiaryOrchestrator()

    # Own LLM call (self-contained narration; INV-SD-14). Missing key/import →
    # provider None → authoring soft-fails to the minimal grounded entry.
    provider = None
    try:
        inference_cfg = full_config.get("inference", {}) or {}
        provider_name = _resolve_provider_name(inference_cfg)
        from titan_hcl.inference import get_provider
        provider = get_provider(provider_name, inference_cfg)
        logger.info("[soul_diary] inference provider=%s", provider_name)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] inference provider unavailable "
                       "(diary will use minimal grounded entries): %s", e)

    # Own OVG truth-gate (lightweight, regex-based).
    verifier = None
    try:
        from titan_hcl.logic.output_verifier import OutputVerifier
        verifier = OutputVerifier(titan_id=titan_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("[soul_diary] OutputVerifier init failed: %s", e)

    # Felt/neuromod SHM reads (G18).
    shm_reader = None
    try:
        from titan_hcl.api.shm_reader_bank import ShmReaderBank
        shm_reader = ShmReaderBank()
    except Exception as e:  # noqa: BLE001
        logger.info("[soul_diary] ShmReaderBank unavailable: %s", e)

    _WORKER_READY = True
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception:  # noqa: BLE001
            pass
    logger.info("[soul_diary] Ready — listening for MEDITATION_COMPLETE")

    last_heartbeat = 0.0
    processed = 0
    errors = 0

    while True:
        now = time.time()
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
            try:
                send_queue.put({
                    "type": bus.MODULE_HEARTBEAT, "src": name, "dst": "guardian",
                    "payload": {"alive": True, "ts": now, "processed": processed,
                                "errors": errors}, "ts": now})
            except Exception:  # noqa: BLE001
                pass
            if _state_writer is not None and shm_heartbeat_allowed(_WORKER_READY, _BOOT_DEADLINE):
                try:
                    _state_writer.heartbeat()
                except Exception:  # noqa: BLE001
                    pass
            last_heartbeat = now

        try:
            msg = recv_queue.get(timeout=0.5)
        except Empty:
            continue
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(msg, dict):
            continue

        msg_type = msg.get("type")

        try:
            from titan_hcl.core import worker_swap_handler as _swap
            if _swap.maybe_dispatch_swap_msg(msg):
                continue
        except Exception:  # noqa: BLE001
            pass

        if msg_type == bus.MODULE_PROBE_REQUEST:
            try:
                from titan_hcl.core.probe_dispatcher import handle_module_probe_request
                handle_module_probe_request(
                    msg, send_queue=send_queue, module_name=name,
                    ready=_WORKER_READY)
            except Exception:  # noqa: BLE001
                pass
            continue

        if msg_type == bus.MODULE_SHUTDOWN:
            logger.info("[soul_diary] MODULE_SHUTDOWN — exiting")
            if _state_writer is not None:
                try:
                    _state_writer.write_state("stopping")
                except Exception:  # noqa: BLE001
                    pass
            break

        if msg_type == bus.MEDITATION_COMPLETE:
            payload = msg.get("payload") or {}
            try:
                if _author_daily_entry(payload, orchestrator=orchestrator,
                                       provider=provider, verifier=verifier,
                                       shm_reader=shm_reader,
                                       send_queue=send_queue, src=name,
                                       titan_id=titan_id,
                                       repo_root=project_root):
                    processed += 1
                else:
                    errors += 1
            except Exception as e:  # noqa: BLE001
                errors += 1
                logger.error("[soul_diary] MEDITATION_COMPLETE handling failed: %s",
                             e, exc_info=True)
            continue
